"""Export the full DQN outcome panel of minatar-repair-20261007 with code provenance (sprint 03, S1).

Copies the final runs/.../outcomes.jsonl (576 rows) over the 504-row snapshot in research/results/
after checking that:
- the committed snapshot is an exact prefix of the final file;
- every file in the campaign's dependency_manifest has the same sha256 in the git commit that last
  changed experiments/ and in the working tree;
- every outcome was finished after that commit; every row's checkpoint_sha256 is in checkpoints.jsonl;
- the file is the one the T0 report was computed from.

Usage: .venv\\Scripts\\python.exe -m scripts.export_dqn_outcomes
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "minatar-repair-20261007"
OUT = ROOT / "research" / "results" / "minatar-repair-20261007"
T0 = OUT / "monitoring" / "20261022_full_576.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def main():
    final, previous = (RUN / "outcomes.jsonl").read_bytes(), (OUT / "outcomes.jsonl").read_bytes()
    rows = [json.loads(line) for line in final.splitlines()]
    if final.splitlines()[:len(previous.splitlines())] != previous.splitlines():
        raise SystemExit("The committed export is not a prefix of the final outcomes")
    manifest = json.loads((RUN / "dependency_manifest.json").read_text(encoding="utf-8"))
    commit = git("log", "-1", "--format=%H", "--", "experiments/").decode().strip()
    committed_at = int(git("show", "-s", "--format=%ct", commit))
    files = {}
    for name, digest in manifest["code"].items():
        path = name.replace("\\", "/")
        files[path] = {"manifest": digest, "commit": sha(git("show", f"{commit}:{path}")),
                       "working_tree": sha((ROOT / path).read_bytes())}
    if any(len(set(f.values())) != 1 for f in files.values()):
        raise SystemExit(f"Code hashes differ: {files}")
    jobs = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["jobs"]
    finished = [jobs[r["job_id"]]["finished_at"] for r in rows]
    if min(finished) < committed_at:
        raise SystemExit("Some outcomes were finished before the code commit")
    checkpoints = {json.loads(line)["checkpoint_id"]: json.loads(line)["sha256"]
                   for line in (RUN / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()}
    if any(checkpoints.get(r["checkpoint_id"]) != r["checkpoint_sha256"] for r in rows):
        raise SystemExit("A row's checkpoint_sha256 is not in checkpoints.jsonl")
    t0 = json.loads(T0.read_text(encoding="utf-8"))["source_sha256"]
    if sha(final) != t0:
        raise SystemExit("The final outcomes are not the file the T0 report was computed from")
    stamp = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="seconds")
    (OUT / "outcomes.jsonl").write_bytes(final)
    summary = {
        "schema_version": 1, "campaign": "minatar-repair-20261007",
        "outcomes": {"path": "outcomes.jsonl", "rows": len(rows), "sha256": sha(final),
                     "failures": sum(bool(r["failure_reason"]) for r in rows),
                     "finished_utc": [stamp(min(finished)), stamp(max(finished))]},
        "previous_export": {"rows": len(previous.splitlines()), "sha256": sha(previous), "is_prefix": True},
        "t0_report_source_sha256": t0,
        "code": {"git_commit": commit, "committed_utc": stamp(committed_at), "files": files,
                 "python": manifest["python"], "packages": manifest["packages"]},
        "checkpoints": {"rows_matching_checkpoints_jsonl": len(rows),
                        "checkpoints_jsonl_sha256": sha((RUN / "checkpoints.jsonl").read_bytes())},
        "script_sha256": sha(Path(__file__).read_bytes())}
    (OUT / "outcomes_export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(len(rows), "rows", sha(final)[:12], "code", commit[:7])


if __name__ == "__main__":
    main()
