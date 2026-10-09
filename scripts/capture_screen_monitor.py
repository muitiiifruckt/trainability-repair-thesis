"""Capture a read-only, descriptive snapshot of the preregistered pilot screen.

No learner imports, optimizer updates, population intervals, or policy selection.
Only natural seed 0--2, age 50k/200k, repeat 0/1, horizon 50k are included.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

from research_supervisor import alive


ACTIONS = ("continue", "optimizer_reset", "head_reset", "head_and_optimizer_reset")
PILOT_PANELS = {(age, repeat) for age in (50000, 200000) for repeat in (0, 1)}


def capture(root, destination):
    raw = (root / "outcomes.jsonl").read_bytes()
    all_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    rows = [row for row in all_rows if row["stage"] == "screen"
            and row["source_mode"] == "natural" and row["game"] in ("breakout", "asterix")
            and row["training_seed"] in (0, 1, 2)
            and (row["nominal_age"], row["repeat"]) in PILOT_PANELS
            and row["budget_env_steps"] == 50000 and row["repair_id"] in ACTIONS]
    assert len({row["job_id"] for row in rows}) == len(rows), "Duplicate pilot job"
    panels = collections.defaultdict(dict)
    for row in rows:
        key = (row["trajectory_id"], row["nominal_age"], row["repeat"])
        assert row["repair_id"] not in panels[key], "Duplicate action in panel"
        panels[key][row["repair_id"]] = row
    complete = {key: panel for key, panel in panels.items()
                if set(panel) == set(ACTIONS)
                and all(not row.get("failure_reason") and math.isfinite(row["j_final"])
                        for row in panel.values())}
    for panel in complete.values():
        assert len({row["continuation_seed"] for row in panel.values()}) == 1
        assert len({json.dumps(row["evaluation_seeds"], sort_keys=True)
                    for row in panel.values()}) == 1
        assert all(row["completed_env_steps"] == row["budget_gradient_updates"] == 50000
                   for row in panel.values()), "Pilot budget mismatch"

    histories = {}
    for history in sorted({key[0] for key in complete}):
        subset = {key: panel for key, panel in complete.items() if key[0] == history}
        values = list(subset.values())
        histories[history] = {
            "game": values[0]["continue"]["game"],
            "complete_pilot_history": {(key[1], key[2]) for key in subset} == PILOT_PANELS,
            "panel_count": len(values),
            "mean_final_effect": {action: statistics.mean(panel[action]["j_final"]
                                  - panel["continue"]["j_final"] for panel in values)
                                  for action in ACTIONS[1:]},
            "mean_interaction": statistics.mean(panel[ACTIONS[3]]["j_final"]
                                - panel[ACTIONS[2]]["j_final"] - panel[ACTIONS[1]]["j_final"]
                                + panel["continue"]["j_final"] for panel in values),
        }
    by_game = {}
    for game in ("breakout", "asterix"):
        eligible = [item for item in histories.values()
                    if item["game"] == game and item["complete_pilot_history"]]
        if not eligible:
            continue
        eligible_names = {name for name, item in histories.items()
                          if item["game"] == game and item["complete_pilot_history"]}
        values = [panel for key, panel in complete.items() if key[0] in eligible_names]
        by_game[game] = {
            "complete_histories": len(eligible),
            "mean_effect": {action: statistics.mean(item["mean_final_effect"][action]
                             for item in eligible) for action in ACTIONS[1:]},
            "mean_interaction": statistics.mean(item["mean_interaction"] for item in eligible),
            "panel_effect_counts": {action: {
                "negative": sum(panel[action]["j_final"] < panel["continue"]["j_final"]
                                for panel in values),
                "positive": sum(panel[action]["j_final"] > panel["continue"]["j_final"]
                                for panel in values),
                "zero": sum(panel[action]["j_final"] == panel["continue"]["j_final"]
                            for panel in values),
                "panels": len(values),
            } for action in ACTIONS[1:]},
            "mean_immediate_effect": {action: statistics.mean(panel[action]["j_immediate"]
                                       - panel["continue"]["j_immediate"] for panel in values)
                                       for action in ACTIONS[1:]},
            "uncertainty": "descriptive only; three source histories per game are not a confirmed population effect",
        }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    supervisor = json.loads((root / "supervisor.json").read_text(encoding="utf-8"))
    dependency = json.loads((root / "dependency_manifest.json").read_text(encoding="utf-8"))
    mismatches = [name for name, expected in dependency["code"].items()
                  if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected]
    assert not mismatches, f"Active scientific code changed: {mismatches}"
    data = {
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kind": "completed_pilot_screen" if len(rows) == 96 else "partial_pilot_screen",
        "source_config_hash": manifest["config_hash"],
        "source_dependency_manifest": dependency,
        "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coordinator_alive": alive(manifest.get("worker_pid", 0)),
        "supervisor_alive": alive(supervisor.get("supervisor_pid", 0)),
        "worker_error": manifest.get("worker_error"),
        "active_package": manifest.get("active_package"),
        "closed_packages": manifest.get("packages", []),
        "supervisor": supervisor,
        "progress": json.loads((root / "progress.json").read_text(encoding="utf-8")),
        "summary": {
            "root_outcomes": len(all_rows),
            "root_outcomes_sha256": hashlib.sha256(raw).hexdigest(),
            "pilot_outcomes": len(rows),
            "complete_pilot_panels": len(complete),
            "by_history": histories,
            "by_game": by_game,
            "numerical_failures": sum(bool(row.get("failure_reason")) for row in rows),
            "nonfinite_final": sum(row["j_final"] is None or not math.isfinite(row["j_final"])
                                   for row in rows),
            "classification": "uncertain; independent confirmation and mechanism/LR controls pending",
        },
        "outcomes": rows,
        "checkpoint_rows": [json.loads(line) for line in (root / "checkpoints.jsonl")
                            .read_text(encoding="utf-8").splitlines() if line.strip()],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    capture(args.root, args.destination)
