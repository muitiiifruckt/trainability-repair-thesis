"""Read-only audit исторической handoff boundary.

Одноразовый live handoff был выполнен версией из Git 686a7a0. Этот CLI больше
не останавливает и не запускает процессы: advisory JSON не даёт строгого
cooperative acknowledgement. Действующий supervisor уже использует durable
wrapper, поэтому дальнейшие live переключения не требуются.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_supervisor import alive
from resume_recorded_research import read, rows, write


def safe_boundary(config, manifest, progress, checkpoints, active_leases):
    """Только наблюдение; не разрешение на остановку процесса."""
    if active_leases or progress.get("stage") != "mechanisms" or progress.get("event") != "mechanisms_complete":
        return False
    games = set(config["screen"]["games"] + config["adaptive"]["development_extension_games"])
    expected = [f"mechanisms:{c['checkpoint_id']}:{r}" for c in checkpoints
                if c["source_mode"] == "natural" and c["game"] in games for r in config["mechanisms"]["seeds"]]
    jobs = manifest.get("jobs", {})
    return bool(expected) and all(jobs.get(k, {}).get("status") == "complete" for k in expected) and jobs.get(progress.get("job_id"), {}).get("status") == "complete"


def handoff(config_path, coordinator_pid, supervisor_pid, workers):
    """Совместимый audit entry point; все process mutations отключены."""
    config = read(config_path)
    folder = (ROOT / config["output_root"]).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Output outside workspace")
    manifest, progress = read(folder / "manifest.json"), read(folder / "progress.json")
    supervisor = read(folder / "supervisor.json")
    record = {"status": "read_only_audit", "time": time.time(), "process_mutations": "disabled",
              "requested_coordinator_pid": coordinator_pid, "requested_supervisor_pid": supervisor_pid,
              "current_coordinator_pid": manifest.get("worker_pid"),
              "current_supervisor_pid": supervisor.get("supervisor_pid"),
              "coordinator_alive": alive(manifest.get("worker_pid")),
              "supervisor_alive": alive(supervisor.get("supervisor_pid")),
              "reason": "Live handoff retired; normal supervisor resumes the recorded plan."}
    # Preserve the actual executed handoff record instead of overwriting it.
    write(folder / "recorded_handoff_audit.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_program.json")
    parser.add_argument("--coordinator-pid", type=int, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    handoff(args.config.resolve(), args.coordinator_pid, args.supervisor_pid, args.workers)
