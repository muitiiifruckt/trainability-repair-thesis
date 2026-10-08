"""Parallel campaign isolation, resume/import accounting and process checks."""
import dataclasses
import json
import os
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from experiments.rl_core import DQNConfig
from experiments.rl_runner import ROOT, append, dump, file_hash, rows
from experiments.rl_scheduler import (
    _append_unique, _charge_worker, _merge_worker, _pid_alive, _waves,
    _heartbeat, _worker_lease, run_parallel_screen,
)


class ParentFixture:
    def __init__(self, root, workers=4):
        self.root, self.workers = root, workers
        learner = DQNConfig(conv_channels=2, hidden_size=8, replay_capacity=32,
                            warmup=4, batch_size=4, target_period=4, episode_cap=6)
        self.config = {
            "program_id": "scheduler_unit_fixture", "kind": "scheduler_test",
            "output_root": str(root.relative_to(ROOT)), "learner": dataclasses.asdict(learner),
            "main_menu": ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"],
            "screen": {"games": ["breakout", "asterix"], "seeds": [0, 1], "ages": [10, 20],
                       "repeats": [0, 1], "horizon": 3, "minimum_target_age": 1,
                       "curve_steps": [0, 3], "evaluation_episodes": 1},
            "resources": {"updates_per_package": 12000000, "seconds_per_package": 86400,
                          "progress_every_seconds": 0.1, "resume_every_steps": 4, "bootstrap_repeats": 0},
        }
        self.manifest = {"jobs": {}, "packages": []}
        self.package = {"id": 0, "started_at": time.time(), "updates": 0}
        self.package_started = time.monotonic()
        self.progress_events, self.reservations, self.reports = [], [], 0

    def persist(self):
        self.manifest["active_package"] = dict(self.package)
        dump(self.root / "manifest.json", self.manifest)

    def reserve(self, maximum_updates=0):
        ceiling = self.config["resources"]["updates_per_package"]
        if maximum_updates > ceiling:
            raise ValueError("Too large")
        self.reservations.append(maximum_updates)
        if self.package["updates"] + maximum_updates > ceiling:
            self.manifest["packages"].append(dict(self.package))
            self.package = {"id": len(self.manifest["packages"]), "started_at": time.time(), "updates": 0}
            self.reports += 1
        self.persist()

    def progress(self, **values):
        self.progress_events.append(values)
        self.persist()

    def completed(self, job_id):
        return self.manifest["jobs"].get(job_id, {}).get("status") == "complete"

    def finish(self, job_id, **metadata):
        self.manifest["jobs"][job_id] = dict(metadata, status="complete", finished_at=time.time())
        self.persist()


class ImmediateExecutor:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, task):
        self.calls.append(task)
        future = Future()
        try:
            future.set_result(fake_worker(task))
        except Exception as exc:
            future.set_exception(exc)
        return future


def fake_worker(task):
    root = Path(task["worker_root"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"jobs": {}, "packages": [], "active_package": {"updates": 0}}
    if task["kind"] == "source":
        checkpoints = {row["checkpoint_id"]: row for row in rows(root / "checkpoints.jsonl")}
        diagnostics = {row["checkpoint_id"] for row in rows(root / "diagnostics.jsonl")}
        previous_age = max((row["nominal_age"] for row in checkpoints.values()), default=0)
        trajectory = f"{task['mode']}_{task['game']}_seed{task['seed']}"
        for age in task["ages"]:
            identifier = f"{trajectory}_age{age}"
            if identifier not in checkpoints:
                artifact = root / "checkpoints" / f"{identifier}.pt"
                artifact.parent.mkdir(exist_ok=True)
                artifact.write_bytes(identifier.encode())
                row = {"checkpoint_id": identifier, "trajectory_id": trajectory, "game": task["game"],
                       "training_seed": task["seed"], "source_mode": task["mode"], "nominal_age": age,
                       "environment_steps": age + 1, "gradient_updates": age,
                       "artifact_path": str(artifact.relative_to(root)), "sha256": file_hash(artifact), "j_pre": 0.0}
                append(root / "checkpoints.jsonl", row)
                checkpoints[identifier] = row
            if identifier not in diagnostics:
                append(root / "diagnostics.jsonl", {"checkpoint_id": identifier, "features": {"age": age}})
        manifest["active_package"]["updates"] += max(0, max(task["ages"]) - previous_age)
    else:
        existing = {row["job_id"] for row in rows(root / "outcomes.jsonl")}
        checkpoint = task["checkpoint"]
        for action in task["actions"]:
            job_id = f"branch:{task['mode']}:{checkpoint['checkpoint_id']}:{action}:repeat{task['repeat']}:h{task['horizon']}"
            if job_id in existing:
                continue
            row = {"job_id": job_id, "checkpoint_id": checkpoint["checkpoint_id"],
                   "trajectory_id": checkpoint["trajectory_id"], "game": checkpoint["game"],
                   "source_mode": task["mode"], "repeat": task["repeat"], "repair_id": action,
                   "budget_env_steps": task["horizon"], "budget_gradient_updates": task["horizon"],
                   "j_final": float(task["actions"].index(action)), "failure_reason": None}
            append(root / "outcomes.jsonl", row)
            manifest["active_package"]["updates"] += task["horizon"]
    dump(manifest_path, manifest)
    return {"task_id": task["task_id"], "worker_root": str(root), "error": None,
            "cumulative_updates": manifest["active_package"]["updates"]}


class ParallelSchedulerTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "runs"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="scheduler_test_", dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(ROOT))
        self.parent = ParentFixture(self.root)
        ImmediateExecutor.calls = []

    def test_waves_have_four_tasks_and_one_package_maximum(self):
        tasks = [{"maximum_updates": 5} for _ in range(9)]
        waves = list(_waves(tasks, workers=4, ceiling=12))
        self.assertEqual([len(wave) for wave in waves], [2, 2, 2, 2, 1])
        self.assertTrue(all(sum(task["maximum_updates"] for task in wave) <= 12 for wave in waves))
        with self.assertRaisesRegex(ValueError, "exceeds one package"):
            list(_waves([{"maximum_updates": 13}], workers=4, ceiling=12))

    def test_parent_merge_once_absolute_artifacts_and_resume_no_execution(self):
        with patch("experiments.rl_scheduler.ProcessPoolExecutor", ImmediateExecutor):
            checkpoints = run_parallel_screen(self.parent)
            self.assertEqual(len(checkpoints), 8)
            self.assertEqual(len(rows(self.root / "diagnostics.jsonl")), 8)
            self.assertEqual(len(rows(self.root / "outcomes.jsonl")), 64)
            self.assertTrue(all(Path(row["artifact_path"]).is_absolute() for row in checkpoints))
            self.assertTrue(all(Path(row["artifact_path"]).exists() for row in checkpoints))
            self.assertEqual(self.parent.package["updates"], 4 * 20 + 64 * 3)
            calls_before = len(ImmediateExecutor.calls)
            updates_before = self.parent.package["updates"]
            finishes = {key: value["finished_at"] for key, value in self.parent.manifest["jobs"].items()}
            resumed = run_parallel_screen(self.parent)
            self.assertEqual(len(resumed), 8)
            self.assertEqual(len(ImmediateExecutor.calls), calls_before)
            self.assertEqual(self.parent.package["updates"], updates_before)
            self.assertEqual({key: value["finished_at"] for key, value in self.parent.manifest["jobs"].items()}, finishes)
        kinds = [task["kind"] for task in ImmediateExecutor.calls]
        self.assertEqual(kinds[:4], ["source"] * 4)
        self.assertTrue(all(kind == "branch" for kind in kinds[4:]))
        self.assertEqual(len({task["worker_root"] for task in ImmediateExecutor.calls}), len(ImmediateExecutor.calls))
        self.assertTrue(any(event.get("stage") == "parallel_source" for event in self.parent.progress_events))

    def test_recover_child_completed_work_after_parent_crash_without_double_cost(self):
        with patch("experiments.rl_scheduler.ProcessPoolExecutor", ImmediateExecutor):
            run_parallel_screen(self.parent, games=["breakout"], seeds=[0], ages=[10], repeats=[0])
            updates = self.parent.package["updates"]
            # Simulate a lost final row after its child work/cost was committed.
            output_path = self.root / "outcomes.jsonl"
            output = rows(output_path)
            lost = output.pop()
            output_path.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
            self.parent.manifest["jobs"].pop(lost["job_id"])
            run_parallel_screen(self.parent, games=["breakout"], seeds=[0], ages=[10], repeats=[0])
            self.assertEqual(len(rows(output_path)), 4)
            self.assertTrue(self.parent.completed(lost["job_id"]))
            self.assertEqual(self.parent.package["updates"], updates)
        worker_root = self.root / "workers" / "ledger_fixture"
        worker_root.mkdir()
        self.assertEqual(_charge_worker(self.parent, worker_root, 5), 5)
        self.assertEqual(_charge_worker(self.parent, worker_root, 5), 0)
        self.assertEqual(_charge_worker(self.parent, worker_root, 8), 3)  # real re-execution
        with self.assertRaisesRegex(ValueError, "backwards"):
            _charge_worker(self.parent, worker_root, 7)

    def test_package_rotations_between_bounded_waves_and_worker_config_limits(self):
        self.parent.config["resources"]["updates_per_package"] = 100
        with patch("experiments.rl_scheduler.ProcessPoolExecutor", ImmediateExecutor):
            run_parallel_screen(self.parent)
        self.assertGreater(self.parent.reports, 0)
        for package in self.parent.manifest["packages"] + [self.parent.package]:
            self.assertLessEqual(package["updates"], 100)
        self.assertTrue(all(len(wave["task_ids"]) <= 4 for wave in self.parent.manifest["scheduler"]["waves"]))
        for task in ImmediateExecutor.calls:
            config = json.loads(Path(task["config_path"]).read_text())
            self.assertEqual(config["learner"]["threads"], 1)
            self.assertEqual(config["resources"]["updates_per_package"], 100)

    def test_conflicting_import_checksum_and_unreachable_phase_stop(self):
        path = self.root / "unique.jsonl"
        _append_unique(path, [{"id": "a", "value": 1}], "id")
        _append_unique(path, [{"id": "a", "value": 1}], "id")
        self.assertEqual(len(rows(path)), 1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            _append_unique(path, [{"id": "a", "value": 2}], "id")
        with self.assertRaisesRegex(ValueError, "target phase"):
            run_parallel_screen(self.parent, overrides={"replay_ratio": 4})  # period4 => phase0 always
        self.parent.config["screen"]["minimum_target_age"] = 4
        with self.assertRaisesRegex(ValueError, "unreachable"):
            run_parallel_screen(self.parent)

    def test_lease_rejects_live_owner_and_recovers_dead_owner(self):
        worker_root = self.root / "workers" / "lease_fixture"
        self.assertTrue(_pid_alive(os.getpid()))
        with _worker_lease(worker_root, "outer"):
            with self.assertRaisesRegex(RuntimeError, "already owned"):
                with _worker_lease(worker_root, "inner"):
                    pass
        self.assertFalse((worker_root / "worker_lease.json").exists())
        dump(worker_root / "worker_lease.json", {"pid": 99999999})
        with patch("experiments.rl_scheduler._pid_alive", return_value=False):
            with _worker_lease(worker_root, "recover"):
                self.assertTrue((worker_root / "worker_lease.json").exists())

    def test_progress_reader_race_is_advisory(self):
        worker_root = self.root / "workers" / "heartbeat_fixture"
        worker_root.mkdir(parents=True)
        dump(worker_root / "progress.json", {"gradient_updates": 2})
        task = {"task_id": "one", "worker_root": str(worker_root)}
        with patch.object(Path, "read_text", side_effect=PermissionError("temporary reader race")):
            _heartbeat(self.parent, [task], {"one"}, "parallel_source", 0)
        progress = self.parent.progress_events[-1]["worker_progress"][0]
        self.assertTrue(progress["child_progress"]["progress_temporarily_unavailable"])

    def test_real_spawn_workers_create_independent_tiny_campaigns(self):
        # One meaningful process integration: own tiny learner, no author train code.
        self.parent.workers = 2
        checkpoints = run_parallel_screen(self.parent, games=["breakout", "asterix"], seeds=[0], ages=[10], repeats=[0], horizon=3)
        self.assertEqual(len(checkpoints), 2)
        outcome = rows(self.root / "outcomes.jsonl")
        self.assertEqual(len(outcome), 8)
        self.assertTrue(all(row["failure_reason"] is None for row in outcome))
        source_updates = sum(row["gradient_updates"] for row in checkpoints)
        continuation_updates = sum(row["budget_gradient_updates"] for row in outcome)
        self.assertEqual(self.parent.package["updates"], source_updates + continuation_updates)
        self.assertTrue(all(wave["status"] == "complete" for wave in self.parent.manifest["scheduler"]["waves"]))
        with patch("experiments.rl_scheduler.ProcessPoolExecutor", ImmediateExecutor):
            run_parallel_screen(self.parent, games=["breakout", "asterix"], seeds=[0], ages=[10], repeats=[0], horizon=3)
        self.assertEqual(ImmediateExecutor.calls, [])


if __name__ == "__main__":
    unittest.main()
