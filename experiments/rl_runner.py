"""Resumable research campaign. All numerical claims are produced from real jobs.

Run with .venv/Scripts/python.exe -m experiments.rl_runner auto --resume.
Large states live in ignored runs/; reports are exported to research/results/.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.rl_core import DQNConfig, TrainingState

ROOT = Path(__file__).resolve().parents[1]
MAIN = ("continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                         encoding="utf-8", newline="\n")
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            # Windows readers/OneDrive can briefly deny replacing an open file.
            time.sleep(min(.02*2**attempt,.5))


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(clean(value), ensure_ascii=False, allow_nan=False)+"\n")
        stream.flush()


def seed_for(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little") % (2**31-1)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_state(path: Path, snapshot: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    torch.save(snapshot, temporary)
    temporary.replace(path)
    return file_hash(path)


def load_state(path: Path) -> dict:
    # These are our own local artifacts, whose hashes are stored in the manifest.
    return torch.load(path, map_location="cpu", weights_only=False)


class Campaign:
    def __init__(self, config_path: Path, resume: bool, workers: int = 1):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.root = (ROOT / self.config["output_root"]).resolve()
        if not self.root.is_relative_to(ROOT):
            raise ValueError("Campaign output must stay in this workspace")
        manifest = self.root / "manifest.json"
        if manifest.exists() and not resume:
            raise ValueError("Existing campaign: use --resume")
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.manifest = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {
            "schema_version": 1, "program_id": self.config["program_id"], "created_at": time.time(),
            "jobs": {}, "packages": [], "decisions": [], "revisions": []}
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        if self.manifest.get("config_hash", config_hash) != config_hash:
            raise ValueError("Changed campaign config: choose a new output_root")
        self.manifest["config_hash"] = config_hash
        dump(self.root / "config.json", self.config)
        revision = {"time": time.time(), "code": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (ROOT / "experiments").glob("*.py")},
            "python": platform.python_version(), "packages": {
                name: importlib.metadata.version(name) for name in ("torch", "numpy", "MinAtar", "scikit-learn",
                    "pandas","pyarrow","matplotlib","scipy","seaborn","pytz")}}
        self.manifest["revisions"].append(revision)
        self.last_progress = 0.0
        active = self.manifest.get("active_package")
        if active:
            self.package = {key: active[key] for key in ("id", "started_at", "updates")}
            self.package_started = time.monotonic()-max(0, time.time()-active["started_at"])
        else:
            self.package_started = time.monotonic()
            self.package = {"id": len(self.manifest["packages"]), "started_at": time.time(), "updates": 0}
        self.current = {}
        self.manifest.pop("worker_error", None)
        self.manifest["worker_status"] = "running"
        validation = ROOT / "research/results/minatar_environment_validation.json"
        if validation.exists():
            self.manifest["environment_validation"] = {"path": str(validation.relative_to(ROOT)),
                                                       "sha256": file_hash(validation)}
        dump(self.root / "dependency_manifest.json", revision)
        if "queue" not in self.manifest:
            self.manifest["queue"] = {"stages":["smoke","synthetic","screen","mechanisms","confirm",
                                      "adaptive","select","report"], "conditional_stages":{
                "source_extension":"baseline_unlearned", "additional_seeds_and_repeats":"uncertain",
                "expanded_parameters":"positive parameter repair signal", "reserved_transfer":"confirmed selection signal + protocol freeze"}}
        self.manifest["parallel_workers"] = workers
        self.persist()

    def persist(self):
        self.manifest["active_package"] = dict(self.package, wall_time_sec=time.monotonic()-self.package_started)
        self.manifest["updated_at"] = time.time()
        self.manifest["worker_pid"] = os.getpid()
        dump(self.root / "manifest.json", self.manifest)

    def progress(self, force=False, **values):
        self.current.update(values)
        now = time.monotonic()
        if force or now-self.last_progress >= self.config["resources"]["progress_every_seconds"]:
            report = dict(self.current, time=time.time(), package_updates=self.package["updates"],
                          package_wall_sec=now-self.package_started)
            dump(self.root / "progress.json", report)
            print(json.dumps(clean(report), ensure_ascii=False), flush=True)
            self.last_progress = now
            self.persist()

    def reserve(self, maximum_updates=1):
        limits = self.config["resources"]
        if maximum_updates > limits["updates_per_package"]:
            raise ValueError("A non-resumable job exceeds the package update limit")
        if (self.package["updates"]+maximum_updates > limits["updates_per_package"] or
                time.monotonic()-self.package_started >= limits["seconds_per_package"]):
            self.manifest["packages"].append(dict(self.package, wall_time_sec=time.monotonic()-self.package_started,
                                                   closed_at=time.time()))
            self.package = {"id": len(self.manifest["packages"]), "started_at": time.time(), "updates": 0}
            self.package_started = time.monotonic()
            self.persist()
            self.progress(force=True, event="new_package")
            self.report(commit=True, series=f"package_{self.package['id']-1}")

    def tick(self, state):
        self.reserve(math.ceil(state.config.replay_ratio))
        before = state.gradient_updates
        try:
            return state.step()
        finally:
            self.package["updates"] += state.gradient_updates-before

    def completed(self, job_id):
        return self.manifest["jobs"].get(job_id, {}).get("status") == "complete"

    def finish(self, job_id, **details):
        self.manifest["jobs"][job_id] = dict(details, status="complete", finished_at=time.time())
        self.persist()

    def decision(self, decision, reason, **details):
        entry = dict(decision=decision, reason=reason, time=time.time(), **details)
        self.manifest["decisions"].append(entry)
        append(self.root / "decisions.jsonl", entry)
        self.persist()

    def learner_config(self, overrides=None):
        return DQNConfig(**dict(self.config["learner"], **(overrides or {})))

    def development_games(self):
        return set(self.config["screen"]["games"]+self.config["adaptive"]["development_extension_games"])-set(self.config["adaptive"]["reserved_games"])

    def source_savepoint(self, trajectory, base_path, state):
        previous = self.manifest.setdefault("source_resumes",{}).get(trajectory,{})
        suffix = ".b.pt" if previous.get("artifact_path", "").endswith(".a.pt") else ".a.pt"
        destination = base_path.with_name(base_path.stem+suffix)
        # Write the inactive slot before committing its pointer. A crash at any
        # point leaves the previous checksum-verified savepoint intact.
        digest = save_state(destination,state.snapshot())
        self.manifest["source_resumes"][trajectory] = {"sha256":digest,
             "artifact_path":str(destination.relative_to(self.root)),
             "environment_steps":state.environment_steps,"gradient_updates":state.gradient_updates}
        self.persist()

    def random_reference(self, game: str, config: DQNConfig):
        key = f"random:{game}:{config.episode_cap}"
        cached = self.manifest.setdefault("random_references", {})
        if key not in cached:
            from minatar import Environment
            returns, lengths = [], []
            for index in range(self.config["screen"]["evaluation_episodes"]):
                env = Environment(game, sticky_action_prob=config.sticky_action_prob,
                                  difficulty_ramping=config.difficulty_ramping)
                env.seed(seed_for("random_environment", game, index)); env.reset()
                rng = np.random.RandomState(seed_for("random_policy", game, index))
                total = 0.0
                for length in range(1, config.episode_cap+1):
                    reward, terminal = env.act(int(rng.randint(6)))
                    total += reward
                    if terminal:
                        break
                returns.append(total); lengths.append(length)
            cached[key] = {"mean_return": float(np.mean(returns)), "returns": returns, "lengths": lengths}
            self.persist()
        return cached[key]

    def source(self, game, seed, ages, source_mode="natural", overrides=None):
        trajectory = f"{source_mode}_{game}_seed{seed}"
        cfg = self.learner_config(overrides)
        minimum_age = self.config["screen"]["minimum_target_age"]
        if cfg.replay_ratio <= 0 or not 0 <= minimum_age < cfg.target_period:
            raise ValueError("Source checkpoint target-age requirement is unreachable")
        if float(cfg.replay_ratio).is_integer():
            if minimum_age > cfg.target_period-math.gcd(int(cfg.replay_ratio),cfg.target_period):
                raise ValueError("Source update ratio cannot reach the requested target age")
        path = self.root / "source_resume" / f"{trajectory}.pt"
        checkpoint_rows = [r for r in rows(self.root / "checkpoints.jsonl") if r["trajectory_id"] == trajectory]
        existing = {row["nominal_age"]: row for row in checkpoint_rows}
        diagnostic_ids = {row["checkpoint_id"] for row in rows(self.root/"diagnostics.jsonl")}
        for row in checkpoint_rows:
            if row["checkpoint_id"] not in diagnostic_ids:
                artifact = self.root/row["artifact_path"]
                if file_hash(artifact) != row["sha256"]:
                    raise ValueError("Checkpoint checksum mismatch during diagnostic recovery")
                recovered = TrainingState.restore(load_state(artifact))
                diagnostics = recovered.diagnostics(batch_size=1024,seed=seed_for(row["checkpoint_id"],"diagnostics"))
                append(self.root/"diagnostics.jsonl",{"checkpoint_id":row["checkpoint_id"],
                       "diagnostic_version":"rl-pre-v1","features":diagnostics.get("features",diagnostics),
                       "details":diagnostics,"measured_before_intervention":True,"source_mode":source_mode})
                del recovered
            if not self.completed(f"source_checkpoint:{row['checkpoint_id']}"):
                self.finish(f"source_checkpoint:{row['checkpoint_id']}",**row)
        if all(age in existing for age in ages):
            return [existing[age] for age in ages]
        resumed = self.manifest.setdefault("source_resumes",{}).get(trajectory)
        base_path = path
        if resumed and resumed.get("artifact_path"):
            path = self.root/resumed["artifact_path"]
        if path.exists() and resumed and file_hash(path) != resumed["sha256"]:
            raise ValueError("Source resume checksum mismatch")
        state = TrainingState.restore(load_state(path)) if path.exists() else TrainingState.create(game, seed, cfg)
        random = self.random_reference(game, cfg)
        needed = [age for age in sorted(ages) if age not in existing]
        self.progress(force=True, stage="source", trajectory_id=trajectory, job_id=trajectory,
                      environment_steps=state.environment_steps, gradient_updates=state.gradient_updates)
        started = time.monotonic()
        while needed:
            self.tick(state)
            age = needed[0]
            target_age = state.gradient_updates-state.last_target_update
            if state.environment_steps >= age and target_age >= self.config["screen"]["minimum_target_age"]:
                identifier = f"{trajectory}_age{age}"
                artifact = self.root / "checkpoints" / f"{identifier}.pt"
                digest = save_state(artifact, state.snapshot())
                evaluation = state.evaluate([seed_for(identifier, "pre", i) for i in
                                             range(self.config["screen"]["evaluation_episodes"])])
                diagnostics = state.diagnostics(batch_size=1024, seed=seed_for(identifier, "diagnostics"))
                features = diagnostics.get("features", diagnostics)
                row = {"checkpoint_id": identifier, "trajectory_id": trajectory, "game": game,
                       "environment_id": game, "environment_family": "MinAtar", "training_seed": seed,
                       "source_mode": source_mode, "nominal_age": age,
                       "environment_steps": state.environment_steps, "gradient_updates": state.gradient_updates,
                       "target_update_age": target_age, "artifact_path": str(artifact.relative_to(self.root)),
                       "sha256": digest, "config_hash": self.manifest["config_hash"],
                       "j_pre": evaluation["mean_return"], "pre_evaluation": evaluation,
                       "random_return": random["mean_return"],
                       "baseline_learned": evaluation["mean_return"] > random["mean_return"]+max(1., abs(random["mean_return"]))*.05}
                append(self.root / "checkpoints.jsonl", row)
                append(self.root / "diagnostics.jsonl", {"checkpoint_id": identifier,
                       "diagnostic_version": "rl-pre-v1", "features": features, "details": diagnostics,
                       "measured_before_intervention": True, "source_mode": source_mode})
                existing[age] = row; needed.pop(0)
                self.source_savepoint(trajectory,base_path,state)
                self.finish(f"source_checkpoint:{identifier}", **row)
                self.progress(force=True, event="checkpoint_created", checkpoint_id=identifier,
                              environment_steps=state.environment_steps, gradient_updates=state.gradient_updates,
                              j_pre=row["j_pre"], random_return=row["random_return"])
            elif state.environment_steps % self.config["resources"]["resume_every_steps"] == 0:
                self.source_savepoint(trajectory,base_path,state)
            self.progress(stage="source", trajectory_id=trajectory, environment_steps=state.environment_steps,
                          gradient_updates=state.gradient_updates,
                          recent_return=float(np.mean(state.episode_returns[-20:])) if state.episode_returns else None,
                          wall_time_sec=time.monotonic()-started)
        del state
        return [existing[age] for age in ages]

    def branch(self, checkpoint_row, action, repeat, horizon=None, source_mode=None,
               overrides=None, phase="screen", total_probe_updates=0, policy_id=None,
               policy_metadata=None, declared_budget=None):
        spec = self.config["screen"]
        horizon = horizon or spec["horizon"]
        mode = source_mode or checkpoint_row["source_mode"]
        identifier = checkpoint_row["checkpoint_id"]
        job_id = f"branch:{mode}:{identifier}:{policy_id or action}:repeat{repeat}:h{horizon}"
        if self.completed(job_id):
            return self.manifest["jobs"][job_id]
        output_file = "transfer_outcomes.jsonl" if policy_id else "outcomes.jsonl"
        existing = next((r for r in rows(self.root / output_file) if r.get("job_id") == job_id), None)
        if existing:
            self.finish(job_id, **{k:v for k,v in existing.items() if k != "job_id"}); return existing
        artifact = self.root / checkpoint_row["artifact_path"]
        if file_hash(artifact) != checkpoint_row["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {identifier}")
        checkpoint = load_state(artifact)
        continuation_seed = seed_for(identifier, mode, "continuation", repeat)
        resume_path = self.root / "branch_resume.pt"
        resume_meta = self.root / "branch_resume.json"
        metadata = json.loads(resume_meta.read_text(encoding="utf-8")) if resume_meta.exists() else {}
        if metadata.get("job_id") == job_id and resume_path.exists() and file_hash(resume_path) == metadata.get("sha256"):
            state = TrainingState.restore(load_state(resume_path))
            completed_steps = metadata["completed_steps"]
            curve = metadata["curve"]
            initial_updates = metadata["initial_updates"]
            elapsed_prior = metadata["elapsed"]
        else:
            evaluation_started_at = time.time()
            state = TrainingState.restore(checkpoint, continuation_seed=continuation_seed)
            if overrides:
                for key, value in overrides.items():
                    setattr(state.config, key, value)
            state.apply_repair(action, repair_seed=seed_for(identifier, mode, "repair", repeat))
            if overrides and "lr" in overrides:
                for group in state.optimizer.param_groups:
                    group["lr"] = overrides["lr"]
            completed_steps = 0
            initial_updates = state.gradient_updates
            elapsed_prior = 0.0
            evaluation_seeds = [seed_for(identifier, mode, repeat, "evaluation", 0, i)
                                for i in range(spec["evaluation_episodes"])]
            immediate = state.evaluate(evaluation_seeds)
            curve = [{"env_steps": 0, "mean_return": immediate["mean_return"],
                      "returns": immediate["returns"], "evaluation_seeds": evaluation_seeds}]
        if metadata.get("job_id") == job_id:
            evaluation_started_at = metadata.get("evaluation_started_at", time.time())
        del checkpoint
        curve_steps = spec["curve_steps"] if horizon == spec["horizon"] else sorted(set([0,5000,10000,horizon]))
        curve_steps = {value for value in curve_steps if 0 <= value <= horizon}
        curve_steps.add(horizon)
        failure = None
        start = time.monotonic()
        self.progress(force=True, stage=phase, job_id=job_id, checkpoint_id=identifier, action=action,
                      repeat=repeat, completed_steps=completed_steps, horizon=horizon)
        try:
            while completed_steps < horizon:
                self.tick(state); completed_steps += 1
                if completed_steps in curve_steps:
                    seeds = [seed_for(identifier, mode, repeat, "evaluation", completed_steps, i)
                             for i in range(spec["evaluation_episodes"])]
                    evaluation = state.evaluate(seeds)
                    curve.append({"env_steps": completed_steps, "mean_return": evaluation["mean_return"],
                                  "returns": evaluation["returns"], "evaluation_seeds": seeds})
                if completed_steps % self.config["resources"]["resume_every_steps"] == 0:
                    digest = save_state(resume_path, state.snapshot())
                    dump(resume_meta, {"job_id": job_id, "sha256": digest, "completed_steps": completed_steps,
                                      "initial_updates": initial_updates, "curve": curve,
                                      "evaluation_started_at": evaluation_started_at,
                                      "elapsed": elapsed_prior+time.monotonic()-start})
                    self.persist()
                self.progress(completed_steps=completed_steps, gradient_updates=state.gradient_updates,
                              recent_return=float(np.mean(state.episode_returns[-20:])) if state.episode_returns else None,
                              wall_time_sec=elapsed_prior+time.monotonic()-start)
        except FloatingPointError as error:
            failure = str(error)
        # Infrastructure errors propagate: root fixes them and resumes the same seed.
        xs = np.array([r["env_steps"] for r in curve], dtype=float)
        ys = np.array([r["mean_return"] for r in curve], dtype=float)
        auc = float(np.trapezoid(ys, xs)/horizon) if len(curve) > 1 else None
        outcome = {"job_id": job_id, "stage": phase, "checkpoint_id": identifier,
                   "trajectory_id": checkpoint_row["trajectory_id"], "game": checkpoint_row["game"],
                   "source_mode": mode, "training_seed": checkpoint_row["training_seed"],
                   "nominal_age": checkpoint_row["nominal_age"], "repair_id": action,
                   "repair_seed": seed_for(identifier, mode, "repair", repeat),
                   "repeat": repeat, "continuation_seed": continuation_seed,
                   "budget_env_steps": declared_budget or horizon, "completed_env_steps": completed_steps,
                   "budget_gradient_updates": state.gradient_updates-initial_updates,
                   "probe_gradient_updates": total_probe_updates,
                   "wall_time_sec": elapsed_prior+time.monotonic()-start,
                   "j_pre": checkpoint_row["j_pre"], "j_immediate": curve[0]["mean_return"],
                   "j_final": curve[-1]["mean_return"] if failure is None else None,
                   "adaptation_auc": auc, "evaluation_curve": curve, "failure_reason": failure,
                   "checkpoint_sha256": checkpoint_row["sha256"],
                   "evaluation_seeds": curve[-1]["evaluation_seeds"]}
        if policy_id:
            outcome.update(policy_id=policy_id, selected_action=action, phase=phase,
                           evaluation_started_at=evaluation_started_at,
                           continuation_env_steps=horizon,
                           total_budget_gradient_updates=declared_budget or horizon+total_probe_updates,
                           continuation_gradient_updates=state.gradient_updates-initial_updates,
                           **(policy_metadata or {}))
        append(self.root / output_file, outcome)
        self.finish(job_id, **{k:v for k,v in outcome.items() if k != "job_id"})
        # Remove only the scratch files produced by this worker, never source artifacts.
        for scratch in (resume_path, resume_meta):
            if scratch.exists() and scratch.resolve().is_relative_to(self.root):
                scratch.unlink()
        self.progress(force=True, event="branch_complete", j_final=outcome["j_final"], failure_reason=failure)
        del state
        return outcome

    def screen(self, repeats=None, games=None, seeds=None, ages=None, mode="natural", overrides=None, horizon=None):
        if self.workers > 1:
            from experiments.rl_scheduler import run_parallel_screen
            return run_parallel_screen(self,repeats=repeats,games=games,seeds=seeds,ages=ages,
                                       mode=mode,overrides=overrides,horizon=horizon)
        spec = self.config["screen"]
        checkpoint_rows = []
        for game in games or spec["games"]:
            for seed in seeds or spec["seeds"]:
                checkpoint_rows.extend(self.source(game, seed, ages or spec["ages"], mode, overrides))
        for checkpoint in checkpoint_rows:
            for repeat in spec["repeats"] if repeats is None else repeats:
                for action in self.config["main_menu"]:
                    self.branch(checkpoint, action, repeat, horizon=horizon, phase="screen" if repeat < 2 else "confirm")
        return checkpoint_rows

    def mechanisms(self):
        from experiments.rl_mechanisms import run_mechanisms
        settings = self.config["mechanisms"]
        budgets = {k:v for k,v in settings.items() if k not in {"seeds", "lr_grid"}}
        for checkpoint in rows(self.root / "checkpoints.jsonl"):
            if checkpoint["source_mode"] != "natural" or checkpoint["game"] not in self.development_games():
                continue
            for repeat in settings["seeds"]:
                job_id = f"mechanisms:{checkpoint['checkpoint_id']}:{repeat}"
                if self.completed(job_id):
                    continue
                self.reserve(30000)
                self.progress(force=True, stage="mechanisms", job_id=job_id)
                result = run_mechanisms(load_state(self.root / checkpoint["artifact_path"]),
                                        seed=seed_for(job_id), budgets=budgets, lr_grid=settings["lr_grid"])
                result.update(checkpoint_id=checkpoint["checkpoint_id"], trajectory_id=checkpoint["trajectory_id"],
                              game=checkpoint["game"], source_mode=checkpoint["source_mode"], repeat=repeat)
                output = self.root / "mechanisms" / f"{checkpoint['checkpoint_id']}_r{repeat}.json"
                dump(output, result)
                updates = int(result.get("total_updates", result.get("costs", {}).get("optimizer_updates", 26000)))
                self.package["updates"] += updates
                self.finish(job_id, artifact_path=str(output.relative_to(self.root)), updates=updates)
                self.progress(force=True, event="mechanisms_complete", updates=updates)

    def report(self, commit=False, series="intermediate"):
        from experiments.rl_analysis import analyze, build_report
        analysis = analyze(self.root, bootstrap_repeats=self.config["resources"]["bootstrap_repeats"])
        dump(self.root / "analysis.json", analysis)
        output = ROOT / "research/results" / self.config["program_id"]
        build_report(self.root, output, bootstrap_repeats=self.config["resources"]["bootstrap_repeats"])
        for name in ("config.json", "checkpoints.jsonl", "diagnostics.jsonl", "outcomes.jsonl", "decisions.jsonl",
                     "transfer_protocol.json", "transfer_outcomes.jsonl", "transfer_summary.json", "transfer_paired_effects.jsonl"):
            source = self.root / name
            if source.exists():
                (output / name).write_bytes(source.read_bytes())
        import shutil
        for name in ("frozen_policies","mechanisms","chooser"):
            folder = self.root / name
            if folder.exists():
                shutil.copytree(folder,output/name,dirs_exist_ok=True)
        dump(output / "manifest.json", self.manifest)
        if (self.root / "mechanisms").exists() or (self.root / "synthetic_lr_sweep.json").exists():
            from experiments.mechanism_analysis import summarize_mechanisms
            summarize_mechanisms(self.root,output)
        self.write_report(output, analysis, series)
        if commit:
            self.commit_report(output, series)
        return analysis

    def write_report(self, output, analysis, series):
        counts = {stage:sum(1 for row in rows(self.root / "outcomes.jsonl") if row.get("stage") == stage)
                  for stage in ("screen","confirm","stress","parameters","lr_control")}
        lines = ["# Результаты автономной программы", "",
                 "Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.",
                 "", f"Обновление: {time.strftime('%Y-%m-%d %H:%M:%S')}; серия: {series}.",
                 f"Checkpoints: {len(rows(self.root / 'checkpoints.jsonl'))}. Завершённые ветки: {counts}.",
                 f"Текущий пакет: {self.package['id']}; optimizer updates: {self.package['updates']:,}.", ""]
        labels = {"heterogeneous":"перспективно: требуется независимое подтверждение пользы selector",
                  "single_repair":"один стабильный выбор: исследовать его границы; преимущество диагностики пока не показано",
                  "baseline_unlearned":"неопределённо: baseline не показал обучения",
                  "uncertain":"неопределённо: разрешения эксперимента недостаточно",
                  "incomplete":"серия ещё не завершена или содержит пропуски/сбои"}
        for item in analysis.get("strata", []):
            lines += [f"## {item['source_mode']}, бюджет {item['budget_env_steps']} transitions", "",
                      f"Статус: **{labels.get(item['primary_gate'], item['primary_gate'])}**.",
                      f"Историй: {item['source_histories']}; полных checkpoint×repeat панелей: {item['complete_repeat_panels']}; минимум repeats: {item['minimum_complete_repeats']}.",
                      f"Numerical failures: {item['failed_rows_by_reason']}; missing/nonfinite finals: {item['nonfinite_or_missing_final_rows']}.", ""]
            for action, stats in item.get("action_statistics", {}).items():
                for game, estimate in stats["paired_effect"].items():
                    lines.append(f"- {game}, {action}: paired effect {estimate['mean']:.3f}; 95% interval {estimate['ci95']}; histories {estimate['source_histories']}.")
            lines += ["", "Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.", ""]
        lines += ["## Интерпретация и следующие проверки", "",
                  "Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.",
                  "", "Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.",
                  "", "Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.", ""]
        (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        append(self.root / "journal.jsonl", {"time":time.time(), "series":series,
               "what_checked":counts, "why":"separate live repair utility, mechanisms and transfer",
               "result":[{"mode":s["source_mode"],"budget":s["budget_env_steps"],"gate":s["primary_gate"]}
                         for s in analysis.get("strata",[])],
               "explanation":"paired effects with history grouping; unfinished series remain incomplete",
               "next_test":"resume pending stage; advance only through declared gates"})
        (output / "journal.jsonl").write_bytes((self.root / "journal.jsonl").read_bytes())

    def commit_report(self, output, series):
        import subprocess
        relative = str(output.relative_to(ROOT))
        subprocess.run(["git", "add", "--", relative], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=ROOT)
        if changed.returncode == 1:
            result = subprocess.run(["git", "commit", "--only", "-m", f"Research: {series}", "--", relative],
                                    cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                self.decision("git_commit_failed", result.stderr.strip(), series=series)
            else:
                self.manifest.setdefault("report_commits", []).append({"series":series,"time":time.time(),
                    "stdout":result.stdout.strip()})
                self.persist()

    def synthetic(self):
        job_id = "synthetic_lr_sweep"
        if self.completed(job_id):
            return
        from experiments.synthetic_lr_sweep import run
        self.reserve(600000)
        self.progress(force=True, stage="synthetic_lr_controls", job_id=job_id)
        settings = self.config["synthetic"]
        result = run(self.root / "synthetic_lr_sweep.json", continuation_lrs=settings["continuation_lrs"],
                     source_lr=settings["source_lr"], horizons=(100,settings["updates"]))
        self.package["updates"] += result["updates_this_invocation"]
        self.finish(job_id, updates=result["actual_optimizer_updates"], branches=len(result["branches"]))
        export = ROOT / "research/results" / self.config["program_id"] / "synthetic_lr_sweep.json"
        dump(export, result)
        self.report(commit=True, series="synthetic_LR_controls")

    def smoke(self):
        import subprocess
        result = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "discover", "-s", "tests", "-v"],
                                cwd=ROOT, text=True, capture_output=True)
        (self.root / "tests.txt").write_text(result.stdout+result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Research invariants failed; see tests.txt")
        cfg = self.learner_config({"warmup": 64, "replay_capacity": 2048, "target_period": 100})
        state = TrainingState.create("breakout", 123, cfg)
        started = time.monotonic()
        for _ in range(2000):
            self.tick(state)
        seconds = time.monotonic()-started
        save_state(self.root / "smoke.pt", state.snapshot())
        self.finish("smoke", environment_steps=2000, gradient_updates=state.gradient_updates,
                    wall_time_sec=seconds, updates_per_second=state.gradient_updates/seconds,
                    tests_passed=True)
        self.progress(force=True, stage="smoke", event="complete", seconds=seconds)

    def adaptive(self, analysis):
        """Development-only exploration; reserved games stay closed at this stage."""
        primary = self.primary(analysis)
        gate = primary.get("primary_gate", "incomplete")
        current = rows(self.root / "checkpoints.jsonl")
        latest = [row for row in current if row["source_mode"] == "natural" and row["game"] in self.config["screen"]["games"]
                  and row["nominal_age"] == max(self.config["screen"]["ages"])]
        learned_fraction = np.mean([row["baseline_learned"] for row in latest]) if latest else 0.0
        self.decision("development_expansion", "Frozen before reserved-game outcomes", gate=gate,
                      baseline_learned_fraction=float(learned_fraction))
        if learned_fraction < .5:
            self.screen(ages=self.config["adaptive"]["source_extension_ages"][:1], repeats=range(4))
            latest_extended = [r for r in rows(self.root / "checkpoints.jsonl") if r["nominal_age"] == 500000 and r["source_mode"] == "natural"
                               and r["game"] in self.config["screen"]["games"]]
            if np.mean([r["baseline_learned"] for r in latest_extended]) < .5:
                self.screen(ages=self.config["adaptive"]["source_extension_ages"][1:], repeats=range(4))
                for game in self.config["screen"]["games"]:
                    for seed in self.config["adaptive"]["rmsprop_control_seeds"]:
                        self.source(game, seed, [200000], "rmsprop_control", {"optimizer": "rmsprop"})
            analysis = self.report(commit=True, series="extended_baseline_and_learner_control")
            primary = self.primary(analysis)
            gate = primary.get("primary_gate", "uncertain")
            newest = max(r["nominal_age"] for r in rows(self.root / "checkpoints.jsonl") if r["source_mode"] == "natural" and r["game"] in self.config["screen"]["games"])
            latest = [r for r in rows(self.root / "checkpoints.jsonl") if r["source_mode"] == "natural" and r["nominal_age"] == newest and r["game"] in self.config["screen"]["games"]]
            learned_fraction = np.mean([r["baseline_learned"] for r in latest])
        if learned_fraction >= .5 and gate == "uncertain":
            self.decision("increase_resolution", "Uncertain differences after four independent continuations",
                          source_seeds=self.config["confirm"]["additional_seeds"], repeats=self.config["confirm"]["additional_repeats"])
            resolution_ages = sorted({row["nominal_age"] for row in rows(self.root / "checkpoints.jsonl")
                                      if row["source_mode"] == "natural" and row["game"] in self.config["screen"]["games"]})
            self.screen(seeds=self.config["confirm"]["additional_seeds"], repeats=range(4),ages=resolution_ages)
            self.screen(seeds=self.config["screen"]["seeds"]+self.config["confirm"]["additional_seeds"],
                        repeats=self.config["confirm"]["additional_repeats"],ages=resolution_ages)
            self.mechanisms()
            analysis = self.report(commit=True, series="additional_histories_and_repeats")
            primary = self.primary(analysis)
            gate = primary.get("primary_gate", "uncertain")
        if learned_fraction < .5:
            self.decision("stress_deferred", "Unlearned source baseline requires learner assessment first")
            return self.report(commit=True, series="development_baseline_assessment")
        self.lr_controlled_screen()
        if primary.get("parameter_repair_signal", {}).get("positive"):
            self.parameter_repairs(primary)
        if gate == "heterogeneous":
            self.decision("stress_deferred", "Confirmed natural-mode heterogeneity: proceed to diagnostic test first")
            return self.report(commit=True, series="natural_mode_followups")
        # Stressors are an explicitly exploratory development battery, not a test-mode search.
        self.screen(mode="high_replay_ratio", overrides={"replay_ratio": 4},
                    repeats=self.config["adaptive"]["stress_repeats"], horizon=self.config["adaptive"]["stress_horizon"])
        self.report(commit=True, series="high_replay_ratio_screen")
        natural = [r for r in rows(self.root / "checkpoints.jsonl") if r["source_mode"] == "natural"
                   and r["game"] in self.development_games()
                   and r["nominal_age"] in self.config["screen"]["ages"]]
        for checkpoint in natural:
            for repeat in self.config["adaptive"]["stress_repeats"]:
                for action in MAIN:
                    self.branch(checkpoint, action, repeat, source_mode="reward_scale_drop",
                                overrides={"reward_scale": .1}, horizon=self.config["adaptive"]["stress_horizon"], phase="stress")
        return self.report(commit=True, series="reward_scale_shift_screen")

    def primary(self, analysis):
        return next((s for s in analysis.get("strata", []) if s["source_mode"] == "natural"
                     and s["budget_env_steps"] == self.config["screen"]["horizon"]), {})

    def tune_lrs(self, exclude_trajectory=None):
        """Development fitting proxy, not a claim about which live return is best."""
        scores = {action:{lr:[] for lr in self.config["mechanisms"]["lr_grid"]} for action in MAIN}
        for path in (self.root / "mechanisms").glob("*.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("trajectory_id") == exclude_trajectory or result.get("source_mode") != "natural" or result.get("game") not in self.development_games():
                continue
            for branch in result.get("lr_sweep", []):
                value = branch["final"]["heldout"]["old_target"]["huber"]
                if value is not None and branch["status"] == "ok":
                    pre = result["pre_repair"]["heldout"]["old_target"]["huber"]
                    scores[branch["action"]][branch["learning_rate_override"]].append(float(value)/max(float(pre),1e-8))
        return {action:min((lr for lr,values in grid.items() if values),
                    key=lambda lr:np.mean(grid[lr]), default=self.config["learner"]["lr"])
                for action,grid in scores.items()}

    def lr_controlled_screen(self):
        for checkpoint in rows(self.root / "checkpoints.jsonl"):
            if checkpoint["source_mode"] != "natural" or checkpoint["game"] not in self.development_games() or checkpoint["nominal_age"] not in self.config["screen"]["ages"]:
                continue
            chosen = self.tune_lrs(exclude_trajectory=checkpoint["trajectory_id"])
            for repeat in (0,1):
                for action in MAIN:
                    self.branch(checkpoint, action, repeat, source_mode="development_tuned_lr",
                                overrides={"lr":chosen[action]}, phase="lr_control")
        self.report(commit=True, series="development_tuned_LR_live_check")

    def parameter_repairs(self, primary):
        games = {item["game"] for item in primary["parameter_repair_signal"]["positive_contrasts"]}
        for checkpoint in rows(self.root / "checkpoints.jsonl"):
            if checkpoint["source_mode"] != "natural" or checkpoint["game"] not in games:
                continue
            for repeat in self.config["adaptive"]["parameter_repeats"]:
                for action in self.config["adaptive"]["expanded_menu"]:
                    self.branch(checkpoint, action, repeat, phase="parameters")
        self.report(commit=True, series="parameter_intervention_followups")

    def selection(self):
        from experiments.rl_analysis import fit_selectors
        analysis = self.report()
        if self.primary(analysis).get("primary_gate") != "heterogeneous":
            deferred = {"status":"deferred", "reason":"No confirmed natural-mode heterogeneity",
                        "reserved_games_opened":False}
            dump(self.root / "selection.json", deferred)
            self.decision("selection_deferred", deferred["reason"])
            return deferred
        self.screen(games=self.config["adaptive"]["development_extension_games"], repeats=range(4))
        self.report(commit=True, series="SpaceInvaders_development_extension")
        report = fit_selectors(self.root)
        dump(self.root / "selection.json", report)
        primary = next((s for s in report.get("strata",[]) if s["source_mode"] == "natural"
                        and s["budget_env_steps"] == self.config["screen"]["horizon"]), {})
        if primary.get("selector_gate") == "promising_grouped_development_signal":
            self.frozen_transfer()
        else:
            self.decision("transfer_deferred", "No demonstrated diagnostic increment beyond age/return and SBS")
        return report

    def frozen_transfer(self):
        import pickle
        from experiments.rl_analysis import train_selector, choose_action
        from experiments.rl_mechanisms import choose_by_probe
        from experiments.rl_transfer import protocol_hash, validate_frozen_protocol, summarize_transfer
        protocol_path = self.root / "transfer_protocol.json"
        folder = self.root / "frozen_policies"
        folder.mkdir(exist_ok=True)
        policies = ("continue","SBS","age_return_ridge","diagnostic_ridge","diagnostic_tree","chooser")
        model_specs = {"age_return_ridge":("ridge","age_return"),
                       "diagnostic_ridge":("ridge","diagnostic"), "diagnostic_tree":("tree","diagnostic")}
        models = {}
        if not protocol_path.exists():
            artifacts = []
            for name,(family,features) in model_specs.items():
                model = train_selector(self.root, ("natural",self.config["screen"]["horizon"]),
                                       family=family, feature_set=features)
                path = folder / f"{name}.pkl"
                path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
                models[name] = model
                artifacts.append({"path":str(path.relative_to(self.root)),"sha256":file_hash(path)})
                dump(folder / f"{name}_metadata.json",model.metadata)
            dev = self.config["screen"]["games"]+self.config["adaptive"]["development_extension_games"]
            envelope = {"schema_version":1,"frozen_at":time.time(),"protocol":{
                "development_games":dev, "reserved_games":self.config["adaptive"]["reserved_games"],
                "policy_ids":list(policies),"main_menu":list(MAIN),"artifacts":artifacts,
                "software_sha256":self.manifest["revisions"][-1]["code"],
                "development_outcomes_sha256":file_hash(self.root / "outcomes.jsonl"),
                "development_diagnostics_sha256":file_hash(self.root / "diagnostics.jsonl"),
                "single_best_repair":models["diagnostic_ridge"].metadata["single_best_repair"],
                "nominal_learning_rate":self.config["learner"]["lr"],
                "secondary_development_tuned_lrs":self.tune_lrs(),
                "budget_gradient_updates":self.config["screen"]["horizon"],
                "chooser_updates_per_action":self.config["adaptive"]["chooser_updates_per_action"],
                "probe_rule":"old-target heldout Huber; discard probe weights; charge all updates",
                "continuation_repeats":[100,101],"source_seeds":[0,1,2],
                "primary_comparisons":["diagnostic_ridge-SBS","diagnostic_ridge-age_return_ridge"],
                "limitations":["Within MinAtar only", "Three heldout-game source seeds: provisional resolution",
                                "Frozen-policy bootstrap conditional on trained model", "No reserved outcome used to tune policies"]}}
            envelope["protocol_hash"] = protocol_hash(envelope)
            dump(protocol_path,envelope)
            self.decision("protocol_frozen", "Freeze models, preprocessing, primary comparisons and probe costs before reserved outcomes",
                          protocol_hash=envelope["protocol_hash"])
            self.report(commit=True,series="transfer_protocol_frozen")
        else:
            envelope = validate_frozen_protocol(protocol_path)["envelope"]
        validate_frozen_protocol(protocol_path)
        for name in model_specs:
            if name not in models:
                # Own trusted serialized fitted models; SHA verified by protocol validation.
                models[name] = pickle.loads((folder / f"{name}.pkl").read_bytes())
        protocol = envelope["protocol"]
        horizon = protocol["budget_gradient_updates"]
        for game in protocol["reserved_games"]:
            for seed in protocol["source_seeds"]:
                self.source(game,seed,self.config["screen"]["ages"])
        features = {row["checkpoint_id"]:row["features"] for row in rows(self.root / "diagnostics.jsonl")}
        checkpoints = [cp for cp in rows(self.root / "checkpoints.jsonl") if cp["source_mode"] == "natural"
                       and cp["nominal_age"] in self.config["screen"]["ages"]]
        for checkpoint in checkpoints:
            phase = "reserved" if checkpoint["game"] in protocol["reserved_games"] else "development"
            available = dict(features[checkpoint["checkpoint_id"]], nominal_age=checkpoint["nominal_age"],
                             environment_steps=checkpoint["environment_steps"],j_pre=checkpoint["j_pre"])
            for repeat in protocol["continuation_repeats"]:
                chosen = {"continue":"continue","SBS":protocol["single_best_repair"]}
                chosen.update({name:choose_action(model,available) for name,model in models.items()})
                for policy in policies:
                    cost = 0; extra = {"protocol_hash":envelope["protocol_hash"]}
                    if policy == "chooser":
                        chooser_id = f"chooser:{checkpoint['checkpoint_id']}:{repeat}"
                        chooser_path = self.root / "chooser" / f"{checkpoint['checkpoint_id']}_r{repeat}.json"
                        if chooser_path.exists() and self.completed(chooser_id):
                            probe = json.loads(chooser_path.read_text(encoding="utf-8"))
                        else:
                            self.reserve(4*protocol["chooser_updates_per_action"])
                            probe = choose_by_probe(load_state(self.root/checkpoint["artifact_path"]),
                                      updates_per_action=protocol["chooser_updates_per_action"],seed=seed_for(chooser_id))
                            dump(chooser_path,probe)
                            self.package["updates"] += probe["total_updates"]
                            self.finish(chooser_id,updates=probe["total_updates"],status_description=probe["status"])
                        cost = 4*protocol["chooser_updates_per_action"]
                        if probe.get("chosen_action") is None:
                            self.decision("chooser_failed",probe.get("reason",probe["status"]),checkpoint_id=checkpoint["checkpoint_id"],repeat=repeat)
                            failed = dict(checkpoint,job_id=f"failed:{chooser_id}",policy_id="chooser",selected_action="continue",
                                phase=phase,repeat=repeat,j_final=None,failure_reason=probe.get("reason",probe["status"]),
                                total_budget_gradient_updates=horizon,budget_env_steps=horizon,
                                probe_gradient_updates=probe["total_updates"],continuation_gradient_updates=0,
                                wall_time_sec=probe.get("wall_seconds",0),protocol_hash=envelope["protocol_hash"],
                                evaluation_started_at=time.time(),selection_failed=True)
                            if not any(row.get("job_id") == failed["job_id"] for row in rows(self.root/"transfer_outcomes.jsonl")):
                                append(self.root/"transfer_outcomes.jsonl",failed)
                            continue
                        action = probe["chosen_action"]
                        extra.update(probe_forward_examples=probe["forward_examples"],probe_wall_time_sec=probe["wall_seconds"],
                                     probe_artifact_path=str(chooser_path.relative_to(self.root)))
                    else:
                        action = chosen[policy]
                    self.branch(checkpoint,action,repeat,horizon=horizon-cost,phase=phase,policy_id=policy,
                                total_probe_updates=cost,policy_metadata=extra,declared_budget=horizon)
            summarize_transfer(self.root,self.config["resources"]["bootstrap_repeats"])
            self.report(commit=True,series=f"frozen_transfer_{checkpoint['checkpoint_id']}")
        return summarize_transfer(self.root,self.config["resources"]["bootstrap_repeats"])

    def auto(self):
        if not self.completed("smoke"):
            self.smoke()
        self.screen()
        self.report(commit=True,series="first_online_screen_96_branches")
        self.mechanisms()
        self.report(commit=True,series="fixed_TD_mechanisms")
        self.synthetic()
        self.screen(repeats=self.config["confirm"]["repeats"])
        analysis = self.report(commit=True,series="independent_continuation_confirmation")
        self.adaptive(analysis)
        self.selection()
        self.report(commit=True,series="completed_initial_program")
        self.finish("initial_program", note="Declared finite battery complete; reserved games conditional on frozen development signal")
        self.manifest["worker_status"] = "complete"
        self.progress(force=True, stage="complete", event="initial_program_complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "screen", "mechanisms", "confirm", "select", "report", "synthetic", "auto"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_program.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Independent processes; one CPU thread per learner")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    # OS file lock is released automatically on process exit; prevents two
    # coordinators from modifying the same queue/checkpoints concurrently.
    import msvcrt
    lock_config = json.loads(args.config.read_text(encoding="utf-8"))
    lock_root = (ROOT/lock_config["output_root"]).resolve()
    if not lock_root.is_relative_to(ROOT):
        raise ValueError("Lock root must stay in workspace")
    lock_root.mkdir(parents=True,exist_ok=True)
    lock_stream = (lock_root/"worker.lock").open("a+b")
    lock_stream.seek(0); lock_stream.write(b"0"); lock_stream.flush(); lock_stream.seek(0)
    try:
        msvcrt.locking(lock_stream.fileno(),msvcrt.LK_NBLCK,1)
    except OSError as error:
        raise RuntimeError("Another coordinator is active in this campaign") from error
    campaign = Campaign(args.config, args.resume, workers=args.workers)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        if args.stage == "confirm":
            campaign.screen(repeats=campaign.config["confirm"]["repeats"])
        elif args.stage == "select":
            campaign.selection()
        else:
            getattr(campaign, args.stage)()
    except Exception as error:
        campaign.manifest["worker_status"] = "failed"
        campaign.manifest["worker_error"] = {"time": time.time(), "type": type(error).__name__, "message": str(error)}
        campaign.persist()
        raise
    finally:
        if campaign.manifest["worker_status"] == "running":
            campaign.manifest["worker_status"] = "stage_complete"
        campaign.persist()
        lock_stream.seek(0); msvcrt.locking(lock_stream.fileno(),msvcrt.LK_UNLCK,1); lock_stream.close()


if __name__ == "__main__":
    main()
