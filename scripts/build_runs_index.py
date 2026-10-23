"""One index over all runs (sprint 02, part 2, T10; docs/observability_spec.md § 4a).

Collects every <folder>/runs_index.jsonl under the root into <root>/index.jsonl, adding the
folder of each run and the artifacts recorded next to it, so cross-run analysis can start
from one file.  Rebuilt from scratch on every call.

Usage: .venv\\Scripts\\python.exe -m scripts.build_runs_index [--root runs/causal-trainability]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_trainability.telemetry import ROOT


def build(root):
    root = Path(root)
    rows = []
    for index in sorted(root.rglob("runs_index.jsonl")):
        folder = index.parent
        artifacts_path = folder / "artifacts.jsonl"
        artifacts = ([json.loads(line) for line in artifacts_path.read_text(encoding="utf-8").splitlines()]
                     if artifacts_path.exists() else [])
        for line in index.read_text(encoding="utf-8").splitlines():
            run = json.loads(line)
            run["folder"] = folder.relative_to(root).as_posix()
            run["artifacts"] = [a for a in artifacts if a["kind"] != "telemetry" or a["path"] == run["telemetry"]]
            rows.append(run)
    (root / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "causal-trainability"))
    args = parser.parse_args()
    rows = build(args.root)
    kinds = {}
    for r in rows:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    print(len(rows), "runs", kinds)


if __name__ == "__main__":
    main()
