"""Recover this campaign's zero-filled control files without changing its learner.

Completed raw outcomes remain authoritative. An unfinished branch whose resume
metadata was lost restarts from its verified source, preserving IDs and seeds.
Observed previous work is retained in the worker counter. No torch is imported.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resume_recorded_research import read, rows, sha, validate_plan, verify
from scripts.research_supervisor import alive


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_nul(path):
    data = path.read_bytes()
    if not data or data.count(b"\0") != len(data):
        raise ValueError(f"Refusing to replace a file that is not entirely NUL: {path}")
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def reconstruct(config, dependency, checkpoint, progress, outcomes, lease, ledger):
    """Recover only complete jobs and an observed cumulative accounting counter."""
    horizon = config["screen"]["horizon"]
    identifier = checkpoint["checkpoint_id"]
    repeat = progress["repeat"]
    actions = config["main_menu"]
    if actions != ["continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset"]:
        raise ValueError("Unsupported action order for control-file recovery")
    if config["learner"]["replay_ratio"] != 1 or checkpoint["source_mode"] != "natural":
        raise ValueError("Recovery is limited to the recorded natural RR=1 queue")
    unfinished = f"branch:natural:{identifier}:{actions[-1]}:repeat{repeat}:h{horizon}"
    if (progress["job_id"] != unfinished or progress["checkpoint_id"] != identifier
            or progress["horizon"] != horizon or not 0 < progress["completed_steps"] < horizon
            or progress["gradient_updates"] != checkpoint["gradient_updates"] + progress["completed_steps"]):
        raise ValueError("Unfinished progress does not match its source and clocks")
    expected = {f"branch:natural:{identifier}:{action}:repeat{repeat}:h{horizon}" for action in actions[:-1]}
    actual = [r["job_id"] for r in outcomes]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Raw completed jobs are missing, duplicated or from another task")
    for row in outcomes:
        if (row.get("failure_reason") or row["completed_env_steps"] != horizon
                or row["budget_env_steps"] != horizon or row["budget_gradient_updates"] != horizon
                or row.get("probe_gradient_updates", 0) != 0
                or row["trajectory_id"] != checkpoint["trajectory_id"]
                or row["checkpoint_sha256"] != checkpoint["sha256"]
                or row["source_mode"] != "natural" or row["repeat"] != repeat
                or not isinstance(row["j_final"], (int, float)) or not math.isfinite(row["j_final"])):
            raise ValueError("A raw row cannot be restored as a completed numerical job")
    observed = sum(r["budget_gradient_updates"] for r in outcomes) + progress["completed_steps"]
    if progress["package_updates"] != observed or observed < ledger.get("accounted_updates", 0):
        raise ValueError("Observed cumulative work conflicts with the parent ledger")
    return {"schema_version": 1, "program_id": config["program_id"],
            "created_at": dependency["time"], "created_at_reconstruction_source": "dependency_manifest.time",
            "jobs": {r["job_id"]: dict(r, status="complete", recovered_from_raw_outcome=True) for r in outcomes},
            "packages": [], "decisions": [], "revisions": [dependency], "config_hash": sha(config),
            "worker_status": "recovered", "parallel_workers": 1, "worker_pid": lease["pid"],
            "scheduler_task_id": lease["task_id"], "scheduler_task_status": "recovered",
            "active_package": {"id": 0, "started_at": progress["time"] - progress["package_wall_sec"],
                               "updates": observed, "wall_time_sec": progress["package_wall_sec"]},
            "updated_at": progress["time"],
            "recovery_provenance": {"reason": "NUL control files after OS interruption",
                "counter_source": "last parent-recorded child progress", "unfinished_job": unfinished,
                "unfinished_rule": "restart only this branch from the verified source checkpoint",
                "unobserved_work_after_last_counter_possible": True}}


def prepare(folder):
    folder = folder.resolve()
    if not folder.is_relative_to(ROOT) or folder.name != "minatar-repair-20261007":
        raise ValueError("Recovery must remain in the explicitly named campaign")
    manifest, progress = read(folder / "manifest.json"), read(folder / "progress.json")
    config, dependency = read(folder / "config.json"), read(folder / "dependency_manifest.json")
    verify(config, manifest, dependency["code"])
    plan = read(folder / "recorded_resume_plan.json")
    checkpoints = rows(folder / "checkpoints.jsonl")
    validate_plan(plan, config, manifest, checkpoints)
    if alive(manifest.get("worker_pid")):
        raise RuntimeError("A healthy coordinator still owns this queue")
    for lease_path in folder.glob("workers/*/*/worker_lease.json"):
        if alive(read(lease_path).get("pid")):
            raise RuntimeError("A healthy child still owns this queue")
    supervisor_damage = require_nul(folder / "supervisor.json")
    previous = read(ROOT / "research/results/minatar-repair-20261007/monitoring/20261012_sleep_interruption_resumed.json")["supervisor"]
    if alive(previous.get("supervisor_pid")):
        raise RuntimeError("The previous supervisor is still healthy")
    worker_records = []
    for item in progress.get("worker_progress", []):
        if not item.get("pending"):
            continue
        worker = Path(item["worker_root"]).resolve()
        if not worker.is_relative_to(folder / "workers/branch"):
            raise ValueError("Unexpected worker output path")
        damage = {name: require_nul(worker / name) for name in ("manifest.json", "progress.json", "branch_resume.json")}
        child_config = read(worker / "worker_config.json")
        if child_config != read(worker / "config.json") or (ROOT / child_config["output_root"]).resolve() != worker:
            raise ValueError("Changed worker configuration")
        expected_config = json.loads(json.dumps(config))
        expected_config["output_root"] = str(worker.relative_to(ROOT))
        expected_config["learner"]["threads"] = 1
        expected_config["scheduler_worker"] = True
        if child_config != expected_config:
            raise ValueError("Worker configuration differs from the frozen parent")
        child_dependency, lease = read(worker / "dependency_manifest.json"), read(worker / "worker_lease.json")
        if child_dependency["code"] != dependency["code"] or lease["task_id"] != item["task_id"]:
            raise ValueError("Changed scientific code or stable task ID")
        checkpoint = next(r for r in checkpoints if r["checkpoint_id"] == item["child_progress"]["checkpoint_id"])
        identity = {"checkpoint_id": checkpoint["checkpoint_id"], "checkpoint_sha256": checkpoint["sha256"],
                    "mode": "natural", "repeat": item["child_progress"]["repeat"],
                    "horizon": config["screen"]["horizon"], "actions": config["main_menu"]}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if item["task_id"] != f"branch:{digest}" or worker.name != f"branch_{digest[:20]}":
            raise ValueError("Worker folder does not match its frozen task identity")
        source = Path(checkpoint["artifact_path"]).resolve()
        if not source.is_relative_to(folder) or file_sha(source) != checkpoint["sha256"]:
            raise ValueError("Source checksum mismatch")
        outcomes = rows(worker / "outcomes.jsonl")
        key = worker.relative_to(folder).as_posix()
        rebuilt = reconstruct(child_config, child_dependency, checkpoint, item["child_progress"], outcomes,
                              lease, manifest["scheduler"]["workers"].get(key, {}))
        worker_records.append({"worker_root": str(worker), "task_id": item["task_id"], "damaged_files": damage,
            "unverified_resume_pt_sha256": file_sha(worker / "branch_resume.pt"), "rebuilt_manifest": rebuilt,
            "raw_outcomes_sha256": file_sha(worker / "outcomes.jsonl"), "source_sha256_verified": checkpoint["sha256"]})
    if len(worker_records) != 2:
        raise ValueError("Expected exactly the two interrupted branch writers")
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "folder": str(folder),
            "supervisor_damage": supervisor_damage, "supervisor_reconstructed_from": previous,
            "workers": worker_records, "root_outcomes_rows": len(rows(folder / "outcomes.jsonl")),
            "root_outcomes_sha256": file_sha(folder / "outcomes.jsonl"),
            "active_package_before": manifest["active_package"], "scientific_code": dependency["code"],
            "config_hash": manifest["config_hash"], "physical_work_recovered": sum(r["rebuilt_manifest"]["active_package"]["updates"] for r in worker_records),
            "optimizer_updates_in_recovery": 0, "power_settings_changed": False}


def durable_json(path, value):
    temporary = path.with_name(path.name + ".recovery.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def control_lock(path):
    import msvcrt
    with path.open("r+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def apply(folder):
    with ExitStack() as stack:
        for name in ("supervisor.lock", "worker.lock"):
            stack.enter_context(control_lock(folder / name))
        record = prepare(folder)
        backup = folder / "recovery/20261013_nul_controls"
        backup.mkdir(parents=True, exist_ok=False)
        durable_json(backup / "audit.json", record)
        for item in record["workers"]:
            worker = Path(item["worker_root"])
            destination = backup / worker.name
            destination.mkdir()
            for name in ("manifest.json", "progress.json", "branch_resume.json", "branch_resume.pt"):
                original, archived = (worker / name).resolve(), (destination / name).resolve()
                if not original.is_relative_to(folder) or not archived.is_relative_to(backup):
                    raise ValueError("Quarantine paths escaped the campaign")
                original.replace(archived)  # exact files; preserve the original bytes
            durable_json(worker / "manifest.json", item["rebuilt_manifest"])
            durable_json(worker / "progress.json", {"stage": "recovery_pending", "time": time.time(),
                "task_id": item["task_id"], "job_id": item["rebuilt_manifest"]["recovery_provenance"]["unfinished_job"],
                "package_updates": item["rebuilt_manifest"]["active_package"]["updates"]})
            if file_sha(worker / "outcomes.jsonl") != item["raw_outcomes_sha256"]:
                raise ValueError("Completed raw outcomes changed during recovery")
        (folder / "supervisor.json").replace(backup / "supervisor.json")
        restored = dict(record["supervisor_reconstructed_from"], status="recovery_ready", updated_at=time.time(),
                        recovery_source="verified monitoring snapshot; original NUL bytes quarantined")
        durable_json(folder / "supervisor.json", restored)
        if file_sha(folder / "outcomes.jsonl") != record["root_outcomes_sha256"]:
            raise ValueError("Root raw outcomes changed during recovery")
        record["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["quarantine"] = str(backup)
        durable_json(backup / "applied.json", record)
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    campaign_folder = ROOT / "runs/minatar-repair-20261007"
    result = apply(campaign_folder) if args.apply else prepare(campaign_folder)
    print(json.dumps({"applied": args.apply, "root_outcomes_rows": result["root_outcomes_rows"],
                      "worker_count": len(result["workers"]), "physical_work_recovered": result["physical_work_recovered"],
                      "optimizer_updates_in_recovery": 0, "quarantine": result.get("quarantine")}, ensure_ascii=True))
