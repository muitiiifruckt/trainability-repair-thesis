"""Bounded process waves for independent source histories and repair repeats.

Children write isolated scratch campaigns. Only the parent imports records and
updates its manifest. A branch worker runs the full action menu sequentially, so
its single branch-resume slot is never used concurrently. Package transitions
and reports happen between waves in the parent, never in child workers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

from experiments.rl_runner import Campaign, ROOT, append, dump, file_hash, load_state, rows


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _child_updates(manifest):
    return int(sum(package.get("updates", 0) for package in manifest.get("packages", [])) +
               manifest.get("active_package", {}).get("updates", 0))


def _manifest(path):
    path = Path(path) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _dump_retry(path, value):
    # On Windows a short-lived reader handle can block atomic replacement.
    # Retry the same immutable JSON payload; this does not create a second writer.
    for attempt in range(10):
        try:
            dump(path, value)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.01 * (attempt + 1), 0.1))


def _pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, int(pid))  # query only
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _worker_lease(root, task_id):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("Worker scratch must stay in the research workspace")
    root.mkdir(parents=True, exist_ok=True)
    lease = root / "worker_lease.json"
    if lease.exists():
        previous = json.loads(lease.read_text(encoding="utf-8"))
        if _pid_alive(previous["pid"]):
            raise RuntimeError(f"Worker scratch is already owned by PID {previous['pid']}: {root}")
        lease.unlink()  # checked, exact scratch file; no recursive cleanup
    fd = os.open(str(lease), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "task_id": task_id, "started_at": time.time()}, stream)
        yield
    finally:
        lease.unlink(missing_ok=True)


class _WorkerCampaign(Campaign):
    def persist(self):
        self.manifest["active_package"] = dict(self.package, wall_time_sec=time.monotonic()-self.package_started)
        self.manifest["updated_at"] = time.time()
        self.manifest["worker_pid"] = os.getpid()
        _dump_retry(self.root / "manifest.json", self.manifest)

    def report(self, *args, **kwargs):
        # The base resource guard can rotate a resumed old wall-clock package.
        # No child publishes reports or creates Git commits in that path.
        return {"worker_reporting_disabled": True}

    def progress(self, force=False, **values):
        self.current.update(values)
        now = time.monotonic()
        if force or now - self.last_progress >= self.config["resources"]["progress_every_seconds"]:
            _dump_retry(self.root / "progress.json", dict(self.current, time=time.time(),
                        package_updates=self.package["updates"], package_wall_sec=now-self.package_started))
            self.last_progress = now
            self.persist()

    def tick(self, state):
        self.reserve(math.ceil(state.config.replay_ratio))
        before = state.gradient_updates
        try:
            return state.step()
        finally:
            # Account a performed optimizer step even if its finite check raises.
            self.package["updates"] += state.gradient_updates - before


def _worker_init():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_worker_task(task):
    """Top-level spawn-safe callable; the result contains no tensor payloads."""
    worker_root = Path(task["worker_root"])
    campaign, error = None, None
    with _worker_lease(worker_root, task["task_id"]):
        try:
            campaign = _WorkerCampaign(Path(task["config_path"]), resume=(worker_root / "manifest.json").exists(), workers=1)
            campaign.manifest["scheduler_task_id"] = task["task_id"]
            campaign.manifest["scheduler_task_status"] = "running"
            campaign.persist()
            if task["kind"] == "source":
                checkpoints = campaign.source(task["game"], task["seed"], task["ages"],
                                              source_mode=task["mode"], overrides=task["overrides"])
                diagnostics = {row["checkpoint_id"]: row for row in rows(worker_root / "diagnostics.jsonl")}
                # Recover a crash between checkpoint append and diagnostic append.
                for checkpoint in checkpoints:
                    identifier = checkpoint["checkpoint_id"]
                    if identifier not in diagnostics:
                        artifact = worker_root / checkpoint["artifact_path"]
                        if file_hash(artifact) != checkpoint["sha256"]:
                            raise ValueError(f"Checkpoint checksum mismatch: {identifier}")
                        from experiments.rl_core import TrainingState
                        from experiments.rl_runner import seed_for
                        state = TrainingState.restore(load_state(artifact))
                        measured = state.diagnostics(batch_size=1024, seed=seed_for(identifier, "diagnostics"))
                        diagnostic = {"checkpoint_id": identifier, "diagnostic_version": "rl-pre-v1",
                                      "features": measured.get("features", measured), "details": measured,
                                      "measured_before_intervention": True, "source_mode": task["mode"]}
                        append(worker_root / "diagnostics.jsonl", diagnostic)
                        diagnostics[identifier] = diagnostic
                        del state
                    if not campaign.completed(f"source_checkpoint:{identifier}"):
                        campaign.finish(f"source_checkpoint:{identifier}", **checkpoint)
            else:
                for action in task["actions"]:
                    campaign.branch(task["checkpoint"], action, task["repeat"], horizon=task["horizon"],
                                    source_mode=task["mode"], phase=task["phase"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if campaign is not None:
                campaign.manifest["scheduler_task_status"] = "failed" if error else "complete"
                campaign.manifest["scheduler_task_error"] = error
                campaign.persist()
    return {"task_id": task["task_id"], "worker_root": str(worker_root), "error": error,
            "cumulative_updates": _child_updates(_manifest(worker_root))}


@contextmanager
def _single_thread_environment():
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_key(campaign, worker_root):
    path = Path(worker_root).resolve()
    if not path.is_relative_to(campaign.root.resolve()):
        raise ValueError("Worker output is outside the parent campaign")
    return path.relative_to(campaign.root.resolve()).as_posix()


def _charge_worker(campaign, worker_root, cumulative_updates, allow_reserve=True):
    """Persist the high-water mark BEFORE importing rows, so resume is idempotent."""
    scheduler = campaign.manifest.setdefault("scheduler", {"workers": {}, "waves": []})
    key = _worker_key(campaign, worker_root)
    entry = scheduler["workers"].setdefault(key, {"accounted_updates": 0})
    previous = int(entry["accounted_updates"])
    cumulative_updates = int(cumulative_updates)
    if cumulative_updates < previous:
        raise ValueError("Child cumulative update ledger moved backwards")
    remaining = cumulative_updates - previous
    ceiling = int(campaign.config["resources"]["updates_per_package"])
    while remaining:
        chunk = min(remaining, ceiling)
        if allow_reserve:
            campaign.reserve(chunk)
        elif campaign.package["updates"] + chunk > ceiling:
            raise RuntimeError("Actual worker updates exceeded the reserved wave upper bound")
        campaign.package["updates"] += chunk
        entry["accounted_updates"] += chunk
        remaining -= chunk
        entry["last_accounted_at"] = time.time()
        campaign.persist()
    return cumulative_updates - previous


def _append_unique(path, incoming, key, protected=None):
    existing = {row[key]: row for row in rows(path)}
    added = []
    for row in incoming:
        identifier = row[key]
        if identifier in existing:
            first = existing[identifier]
            fields = protected if protected is not None else set(first) | set(row)
            if any(first.get(field) != row.get(field) for field in fields):
                raise ValueError(f"Conflicting imported {key}: {identifier}")
            continue
        append(path, row)
        existing[identifier] = row
        added.append(identifier)
    return added


def _merge_worker(campaign, worker_root):
    worker_root = Path(worker_root)
    checkpoint_rows = rows(worker_root / "checkpoints.jsonl")
    imported = []
    for row in checkpoint_rows:
        row = dict(row)
        artifact = (worker_root / row["artifact_path"]).resolve()
        if not artifact.is_relative_to(worker_root.resolve()) or not artifact.exists():
            raise ValueError("Imported checkpoint artifact is outside its worker or missing")
        if file_hash(artifact) != row["sha256"]:
            raise ValueError(f"Imported checkpoint checksum mismatch: {row['checkpoint_id']}")
        row["artifact_path"] = str(artifact)
        row["worker_root"] = str(worker_root.resolve())
        imported.append(row)
    _append_unique(campaign.root / "checkpoints.jsonl", imported, "checkpoint_id",
                   protected=("sha256", "trajectory_id", "game", "nominal_age", "gradient_updates"))
    diagnostics = rows(worker_root / "diagnostics.jsonl")
    _append_unique(campaign.root / "diagnostics.jsonl", diagnostics, "checkpoint_id")
    outcomes = rows(worker_root / "outcomes.jsonl")
    _append_unique(campaign.root / "outcomes.jsonl", outcomes, "job_id")
    child = _manifest(worker_root)
    for row in imported:
        if not campaign.completed(f"source_checkpoint:{row['checkpoint_id']}"):
            campaign.finish(f"source_checkpoint:{row['checkpoint_id']}", **row)
    for row in outcomes:
        if not campaign.completed(row["job_id"]):
            campaign.finish(row["job_id"], **{key: value for key, value in row.items() if key != "job_id"})
    references = campaign.manifest.setdefault("random_references", {})
    for key, value in child.get("random_references", {}).items():
        if key in references and references[key] != value:
            raise ValueError(f"Worker random reference disagrees: {key}")
        references[key] = value
    campaign.persist()


def _new_task(campaign, kind, identity, **values):
    worker_root = campaign.root / "workers" / kind / f"{kind}_{_digest(identity)[:20]}"
    worker_root.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(campaign.config)
    config["output_root"] = str(worker_root.relative_to(ROOT))
    config["learner"]["threads"] = 1
    config["scheduler_worker"] = True
    config = json.loads(json.dumps(config))  # normalize tuples to their on-disk JSON form
    config_path = worker_root / "worker_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Worker config changed under an existing stable job")
    dump(config_path, config)
    task_id = f"{kind}:{_digest(identity)}"
    return dict(values, kind=kind, task_id=task_id, worker_root=str(worker_root), config_path=str(config_path))


def _source_bound(config, ages):
    ratio = float(config["replay_ratio"])
    period = int(config["target_period"])
    if ratio <= 0:
        raise ValueError("Phase-aware source checkpoints require positive replay_ratio")
    # A full target period is a conservative phase delay; warmup covers very
    # young requested ages. Ignore existing progress rather than under-reserving.
    steps = max(ages) + int(config["warmup"]) + math.ceil(period / ratio) + 1
    return math.ceil(steps * ratio) + 1


def _waves(tasks, workers, ceiling):
    wave, cost = [], 0
    for task in tasks:
        bound = int(task["maximum_updates"])
        if bound > ceiling:
            raise ValueError("A parallel task exceeds one package; split its ages or continuation horizon")
        if wave and (len(wave) >= workers or cost + bound > ceiling):
            yield wave
            wave, cost = [], 0
        wave.append(task)
        cost += bound
    if wave:
        yield wave


def _heartbeat(campaign, wave, pending, stage, wave_id):
    progress = []
    for task in wave:
        path = Path(task["worker_root"]) / "progress.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (PermissionError, FileNotFoundError, json.JSONDecodeError):
            # Progress is advisory. A reader racing atomic replace must not abort
            # a numerical job; the next heartbeat obtains its latest state.
            state = {"progress_temporarily_unavailable": True}
        progress.append({"task_id": task["task_id"], "worker_root": task["worker_root"],
                         "pending": task["task_id"] in pending, "child_progress": state})
    campaign.progress(force=True, stage=stage, parallel_wave_id=wave_id,
                      pending_tasks=sorted(pending), worker_progress=progress)


def _execute_waves(campaign, executor, tasks, stage, workers):
    ceiling = int(campaign.config["resources"]["updates_per_package"])
    for wave in _waves(tasks, workers, ceiling):
        # Reconcile completed child work from a parent crash before reserving any
        # fresh wave. Replayed training then increases the child's actual counter.
        for task in wave:
            _charge_worker(campaign, task["worker_root"], _child_updates(_manifest(task["worker_root"])))
            _merge_worker(campaign, task["worker_root"])
        bound = sum(task["maximum_updates"] for task in wave)
        campaign.reserve(bound)
        scheduler = campaign.manifest.setdefault("scheduler", {"workers": {}, "waves": []})
        wave_id = len(scheduler["waves"])
        record = {"id": wave_id, "stage": stage, "task_ids": [task["task_id"] for task in wave],
                  "maximum_updates": bound, "started_at": time.time(), "status": "running", "actual_new_updates": 0}
        scheduler["waves"].append(record)
        campaign.persist()
        futures = {executor.submit(_run_worker_task, task): task for task in wave}
        remaining = set(futures)
        errors = []
        _heartbeat(campaign, wave, {futures[f]["task_id"] for f in remaining}, stage, wave_id)
        interval = min(20.0, max(0.1, float(campaign.config["resources"]["progress_every_seconds"])))
        while remaining:
            done, remaining = wait(remaining, timeout=interval, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures[future]
                try:
                    result = future.result()
                    cumulative = result["cumulative_updates"]
                    error = result["error"]
                except Exception as exc:
                    cumulative = _child_updates(_manifest(task["worker_root"]))
                    error = f"{type(exc).__name__}: {exc}"
                record["actual_new_updates"] += _charge_worker(campaign, task["worker_root"], cumulative, allow_reserve=False)
                _merge_worker(campaign, task["worker_root"])
                if error:
                    errors.append({"task_id": task["task_id"], "error": error})
            _heartbeat(campaign, wave, {futures[f]["task_id"] for f in remaining}, stage, wave_id)
        record.update(status="failed" if errors else "complete", finished_at=time.time(), errors=errors)
        campaign.persist()
        # Wall-clock rollover and reporting are deferred until no child is active.
        campaign.reserve(0)
        if errors:
            raise RuntimeError(f"Parallel wave failed; completed work is imported and resumable: {errors}")


def run_parallel_screen(campaign, repeats=None, games=None, seeds=None, ages=None,
                        mode="natural", overrides=None, horizon=None):
    """Four CPU processes, source waves first, then checkpoint/repeat waves."""
    spec = campaign.config["screen"]
    games = list(dict.fromkeys(spec["games"] if games is None else games))
    seeds = list(dict.fromkeys(spec["seeds"] if seeds is None else seeds))
    ages = sorted(set(spec["ages"] if ages is None else ages))
    repeats = list(dict.fromkeys(spec["repeats"] if repeats is None else repeats))
    horizon = spec["horizon"] if horizon is None else horizon
    if not ages or any(isinstance(age, bool) or not isinstance(age, int) or age < 1 for age in ages):
        raise ValueError("Source ages must be positive integers")
    if any(isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0 for repeat in repeats):
        raise ValueError("Continuation repeats must be nonnegative integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be nonnegative integers")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    overrides = dict(overrides or {}, threads=1)
    learner = dict(campaign.config["learner"], **overrides)
    period, minimum = int(learner["target_period"]), int(spec["minimum_target_age"])
    ratio = float(learner["replay_ratio"])
    if not math.isfinite(ratio) or ratio <= 0 or not 0 <= minimum < period:
        raise ValueError("Source target phase is unreachable with this replay ratio/target period")
    if ratio.is_integer() and minimum > period - math.gcd(period, int(ratio)):
        raise ValueError("Atomic replay cycles never reach the requested target phase")
    workers = min(4, max(1, int(getattr(campaign, "workers", 4))))
    source_tasks = []
    existing = {row["checkpoint_id"]: row for row in rows(campaign.root / "checkpoints.jsonl")}
    diagnostic_ids = {row["checkpoint_id"] for row in rows(campaign.root / "diagnostics.jsonl")}
    requested_ids = []
    for game in games:
        for seed in seeds:
            trajectory = f"{mode}_{game}_seed{seed}"
            ids = [f"{trajectory}_age{age}" for age in ages]
            requested_ids.extend(ids)
            missing = [age for age, identifier in zip(ages, ids) if identifier not in existing or identifier not in diagnostic_ids]
            if missing:
                task = _new_task(campaign, "source", {"trajectory": trajectory, "learner": learner},
                                 game=game, seed=seed, ages=missing, mode=mode, overrides=overrides,
                                 maximum_updates=_source_bound(learner, missing))
                source_tasks.append(task)
    context = multiprocessing.get_context("spawn")
    with _single_thread_environment(), ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                                          initializer=_worker_init) as executor:
        _execute_waves(campaign, executor, source_tasks, "parallel_source", workers)
        checkpoints = {row["checkpoint_id"]: row for row in rows(campaign.root / "checkpoints.jsonl")}
        if any(identifier not in checkpoints for identifier in requested_ids):
            raise RuntimeError("Source worker completed without all requested checkpoints")
        requested = [checkpoints[identifier] for identifier in requested_ids]
        outcome_ids = {row["job_id"] for row in rows(campaign.root / "outcomes.jsonl")}
        branch_tasks = []
        actions = list(campaign.config["main_menu"])
        for checkpoint in requested:
            for repeat in repeats:
                ids = [f"branch:{mode}:{checkpoint['checkpoint_id']}:{action}:repeat{repeat}:h{horizon}" for action in actions]
                if all(identifier in outcome_ids for identifier in ids):
                    continue
                task = _new_task(campaign, "branch", {"checkpoint_id": checkpoint["checkpoint_id"],
                                 "checkpoint_sha256": checkpoint["sha256"], "mode": mode, "repeat": repeat,
                                 "horizon": horizon, "actions": actions}, checkpoint=checkpoint, repeat=repeat,
                                 mode=mode, horizon=horizon, actions=actions,
                                 phase="screen" if repeat < 2 else "confirm",
                                 maximum_updates=math.ceil(len(actions) * horizon * ratio) + len(actions))
                branch_tasks.append(task)
        _execute_waves(campaign, executor, branch_tasks, "parallel_branches", workers)
    campaign.manifest.setdefault("scheduler", {})["accounting_caveat"] = \
        "Completed invocation counters are exact; hard-killed work after the last child manifest flush is unobservable. Re-execution is charged."
    campaign.persist()
    return requested
