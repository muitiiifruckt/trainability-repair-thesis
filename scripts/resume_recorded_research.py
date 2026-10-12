"""Возобновляет уже принятое increase_resolution до повторной оценки gate.

Внешняя orchestration: frozen experiments/configs и stable task identities
сохраняются. Новый statistical decision не принимается. Импорт без torch
позволяет дешёвый audit/tests рядом с кампанией.
"""
from __future__ import annotations

import argparse
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


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def make_plan(config, manifest, checkpoints, code):
    decision = next((r for r in manifest.get("decisions", []) if r.get("decision") == "increase_resolution"), None)
    if decision is None:
        return None
    games = list(config["screen"]["games"])
    ages = sorted({r["nominal_age"] for r in checkpoints
                   if r["source_mode"] == "natural" and r["game"] in games})
    if not ages:
        raise ValueError("Recorded expansion has no verified source ages")
    seeds = list(decision["source_seeds"])
    repeats = list(decision["repeats"])
    if seeds != config["confirm"]["additional_seeds"] or repeats != config["confirm"]["additional_repeats"]:
        raise ValueError("Recorded expansion differs from frozen config")
    return {"schema_version": 1, "config_hash": sha(config), "scientific_code": code,
            "decision": decision, "decision_sha256": sha(decision), "created_at": time.time(),
            "games": games, "ages": ages, "menu": list(config["main_menu"]),
            "horizon": config["screen"]["horizon"], "status": "pending", "operations": [
                {"id": "additional_sources", "seeds": seeds, "repeats": [0, 1, 2, 3], "status": "pending"},
                {"id": "additional_repeats", "seeds": list(dict.fromkeys(config["screen"]["seeds"] + seeds)),
                 "repeats": repeats, "status": "pending"}], "orchestration_revisions": []}


def coverage(plan, operation, outcomes):
    expected = {(g, s, a, r, action) for g in plan["games"] for s in operation["seeds"]
                for a in plan["ages"] for r in operation["repeats"] for action in plan["menu"]}
    found = {}
    failures = []
    for row in outcomes:
        key = (row.get("game"), row.get("training_seed"), row.get("nominal_age"), row.get("repeat"), row.get("repair_id"))
        if row.get("source_mode") != "natural" or key not in expected or row.get("budget_env_steps") != plan["horizon"]:
            continue
        if key in found:
            raise ValueError("Duplicate outcome in recorded resume plan")
        if row.get("trajectory_id") != f"natural_{key[0]}_seed{key[1]}":
            raise ValueError("Changed trajectory grouping in recorded resume plan")
        failure = row.get("failure_reason")
        if failure:
            # A finished numerical failure is recorded, never retried as missing.
            if not isinstance(failure, str) or failure.split(":", 1)[0] != "FloatingPointError":
                raise ValueError("Infrastructure failure cannot be a finished numerical outcome")
            failures.append({"job_id": row.get("job_id"), "reason": failure})
        elif (row.get("completed_env_steps") != plan["horizon"]
              or row.get("budget_gradient_updates") != plan["horizon"]
              or row.get("probe_gradient_updates", 0) != 0
              or not isinstance(row.get("j_final"), (int, float)) or not math.isfinite(row["j_final"])):
            raise ValueError("Invalid finished outcome in recorded resume plan")
        found[key] = row
    return {"expected": len(expected), "finished": len(found), "missing": len(expected - found.keys()),
            "numerical_failures": failures, "complete": found.keys() == expected}


def validate_plan(plan, config, manifest, checkpoints):
    if plan["config_hash"] != sha(config) or plan["decision_sha256"] != sha(plan["decision"]):
        raise ValueError("Changed recorded decision/config")
    if not any(sha(r) == plan["decision_sha256"] for r in manifest.get("decisions", [])):
        raise ValueError("Recorded decision is absent from the campaign manifest")
    canonical = make_plan(config, {"decisions": [plan["decision"]]}, checkpoints, plan["scientific_code"])
    for field in ("games", "menu", "horizon"):
        if plan[field] != canonical[field]:
            raise ValueError(f"Changed recorded plan field: {field}")
    if not set(config["screen"]["ages"]).issubset(plan["ages"]):
        raise ValueError("Recorded plan omitted a source age")
    if plan["status"] != "complete" and plan["ages"] != canonical["ages"]:
        raise ValueError("Changed source ages in a pending recorded plan")
    if plan["status"] == "complete" and not set(plan["ages"]).issubset(canonical["ages"]):
        raise ValueError("Recorded source ages are absent from current artifacts")
    recorded = [{k: op[k] for k in ("id", "seeds", "repeats")} for op in plan["operations"]]
    expected = [{k: op[k] for k in ("id", "seeds", "repeats")} for op in canonical["operations"]]
    if recorded != expected:
        raise ValueError("Changed operations in the recorded expansion")


def verify(config, manifest, code):
    if sha(config) != manifest.get("config_hash"):
        raise ValueError("Changed config: recorded resume requires the original campaign")
    for name, digest in code.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed frozen scientific code: {name}")


def ensure_no_live_child_leases(folder):
    from scripts.research_supervisor import alive
    for path in folder.glob("workers/*/*/worker_lease.json"):
        try:
            lease = read(path)
        except FileNotFoundError:
            continue
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Cannot verify existing worker lease") from error
        if alive(lease.get("pid")):
            raise RuntimeError("Existing child workers are healthy; wait before coordinator resume")


def run(config_path, workers):
    config = read(config_path)
    folder = (ROOT / config["output_root"]).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Campaign must stay inside workspace")
    # Own exactly the lock used by the frozen CLI before importing its learner.
    import msvcrt
    lock = (folder / "worker.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    started_lock_wait = time.monotonic()
    while True:
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            from scripts.research_supervisor import alive
            owner = read(folder / "manifest.json").get("worker_pid")
            if alive(owner) and owner != os.getpid():
                lock.close()
                raise RuntimeError("Another coordinator is active; no changes made")
            if time.monotonic() - started_lock_wait >= 10:
                lock.close()
                raise
            lock.seek(0)
            time.sleep(.1)
    campaign = None
    try:
        manifest = read(folder / "manifest.json")
        path = folder / "recorded_resume_plan.json"
        plan = read(path) if path.exists() else make_plan(config, manifest, rows(folder / "checkpoints.jsonl"), read(folder / "dependency_manifest.json")["code"])
        code = plan["scientific_code"] if plan else read(folder / "dependency_manifest.json")["code"]
        verify(config, manifest, code)
        ensure_no_live_child_leases(folder)
        if plan:
            validate_plan(plan, config, manifest, rows(folder / "checkpoints.jsonl"))
            plan["orchestration_revisions"].append({"time": time.time(), "pid": os.getpid(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            write(path, plan)
        # Frozen modules import MinAtar before learner/BLAS work begins.
        from experiments.rl_runner import Campaign, torch
        campaign = Campaign(config_path, resume=True, workers=workers)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if plan:
            campaign.manifest["external_resume_plan"] = {"path": path.name,
                "decision_sha256": plan["decision_sha256"], "script_sha256": plan["orchestration_revisions"][-1]["script_sha256"]}
            campaign.persist()
            for operation in plan["operations"]:
                status = coverage(plan, operation, rows(folder / "outcomes.jsonl"))
                operation["coverage"] = status
                if not status["complete"]:
                    plan["status"] = "running"
                    operation["status"] = "running"
                    write(path, plan)
                    campaign.progress(force=True, stage="resume_recorded_resolution", operation=operation["id"])
                    campaign.screen(seeds=operation["seeds"], repeats=operation["repeats"], ages=plan["ages"])
                    status = coverage(plan, operation, rows(folder / "outcomes.jsonl"))
                    if not status["complete"]:
                        raise RuntimeError("Recorded expansion returned with missing scheduled outcomes")
                operation.update(status="complete", coverage=status)
                write(path, plan)
            if plan["status"] != "complete":
                campaign.mechanisms()
                campaign.report(commit=True, series="restored_recorded_histories_and_repeats")
                plan.update(status="complete", completed_at=time.time())
                write(path, plan)
        campaign.auto()
    except Exception as error:
        if campaign is not None:
            campaign.manifest["worker_status"] = "failed"
            campaign.manifest["worker_error"] = {"time": time.time(), "type": type(error).__name__, "message": str(error)}
            campaign.persist()
        else:
            # Own worker.lock here; surface integrity failures before Campaign
            # instead of leaving an empty supervisor error signature forever.
            manifest = read(folder / "manifest.json")
            manifest.update(worker_status="failed", worker_pid=os.getpid(), worker_error={
                "time": time.time(), "type": type(error).__name__, "message": str(error)})
            write(folder / "manifest.json", manifest)
        raise
    finally:
        if campaign is not None:
            if campaign.manifest["worker_status"] == "running":
                campaign.manifest["worker_status"] = "stage_complete"
            campaign.persist()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_program.json")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        config = read(args.config)
        folder = ROOT / config["output_root"]
        manifest = read(folder / "manifest.json")
        plan = make_plan(config, manifest, rows(folder / "checkpoints.jsonl"), read(folder / "dependency_manifest.json")["code"])
        if plan:
            for operation in plan["operations"]:
                operation["coverage"] = coverage(plan, operation, rows(folder / "outcomes.jsonl"))
        print(json.dumps(plan, ensure_ascii=True))
    else:
        run(args.config.resolve(), args.workers)
