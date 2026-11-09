"""Equal-horizon feedback experiment; preregistration: notes/two_state_v2.md.

No outcome-driven horizon search. Each history is independently resumable after
completion, while partial directories are never silently overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from causal_trainability import two_state as ts
from causal_trainability.telemetry import ROOT, Telemetry, TelemetryStore, code_revision, config_hash

CONFIG = dict(ts.CONFIG, histories=8, repeats=4, iterations=30, telemetry_every=30,
              tie_margin=.1, divergence_bar=4, schema_version=1)
MENU = ("continue", "optimizer_reset", "parameter_reset")
MODES = ("live", "frozen")
PREREG = ROOT / "notes" / "two_state_v2.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seeds(history, repeats):
    return {"init": ts.seed_of("two-state-v2", "init", history),
            "short": ts.seed_of("two-state-v2", "short", history),
            "long": ts.seed_of("two-state-v2", "long", history),
            "branches": [ts.seed_of("two-state-v2", "branch", history, r) for r in range(repeats)]}


def trajectory(agent, optimizer, goal, count, seed, cfg, telemetry=None,
               frozen=None, until=None, snapshot_every=None):
    generator = torch.Generator().manual_seed(seed)
    curve = [[0, ts.exact_return(agent, ts.NEW_GOAL, cfg)]]
    if telemetry is not None:
        telemetry.snapshot(0, agent, optimizer)
        telemetry.step(0, force=True, metrics={"new_task_return": curve[0][1]})
    for it in range(1, count + 1):
        batch = ts.sample_batch(agent if frozen is None else frozen, goal, generator, cfg)
        loss = ts.ppo_iteration(agent, optimizer, batch, cfg)
        done = until is not None and until(agent)
        last = done or it == count
        if it % cfg["telemetry_every"] == 0 or last:
            value = ts.exact_return(agent, ts.NEW_GOAL, cfg)
            curve.append([it, value])
            if telemetry is not None:
                probs, _ = ts.policy(agent)
                telemetry.step(it, force=True, metrics={"loss": loss, "new_task_return": value,
                    "switch_prob_s1": float(probs[0, ts.SWITCH]),
                    "switch_prob_s2": float(probs[1, ts.SWITCH])},
                    layer_extra={"body": {"dead_fraction": ts.dead_fraction(agent)}})
        if telemetry is not None and (last or snapshot_every and it % snapshot_every == 0):
            telemetry.snapshot(it, agent, optimizer)
        if done:
            return curve, it
    return curve, count


def repaired(fresh, aged, aged_optimizer, repair, cfg):
    if repair not in MENU:
        raise ValueError(repair)
    params = fresh if repair == "parameter_reset" else aged
    adam = None if repair == "optimizer_reset" else (aged, aged_optimizer)
    return ts.build(params, adam, cfg)


def history_run(history, cfg, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    seed = seeds(history, cfg["repeats"])
    atomic_json(folder / "config.json", dict(config=cfg, seeds=seed, history=history))
    store = TelemetryStore(folder, f"history{history}")
    meta = dict(history=history, config_hash=config_hash(cfg), kind="two_state_v2")

    def telemetry(agent, optimizer, name, **extra):
        return Telemetry(folder, name, agent.layers(), optimizer, store=store,
                         every=cfg["telemetry_every"], meta=dict(meta, **extra))

    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed["init"])
            fresh = ts.Agent(cfg["hidden"])
        fresh_optimizer = ts.make_optimizer(fresh, cfg)
        telemetry(fresh, fresh_optimizer, "theta0", phase="init").snapshot(0, fresh, fresh_optimizer)
        aged, optimizer = ts.build(fresh, None, cfg)
        short_curve, used = trajectory(aged, optimizer, ts.OLD_GOAL, cfg["aging_max_iterations"],
            seed["short"], cfg, telemetry(aged, optimizer, "source_short", phase="source"),
            until=lambda a: float(ts.policy(a)[0][0, ts.SWITCH]) <= cfg["aging_switch_prob"],
            snapshot_every=1000)
        converged = float(ts.policy(aged)[0][0, ts.SWITCH]) <= cfg["aging_switch_prob"]
        long_curve, _ = trajectory(aged, optimizer, ts.OLD_GOAL, cfg["long_aging_iterations"],
            seed["long"], cfg, telemetry(aged, optimizer, "source_long", phase="source"), snapshot_every=1000)
        source_description = ts.describe(aged, optimizer, cfg)
        rows = []
        for repeat in range(cfg["repeats"]):
            for repair in MENU:
                for mode in MODES:
                    agent, opt = repaired(fresh, aged, optimizer, repair, cfg)
                    frozen = copy.deepcopy(agent) if mode == "frozen" else None
                    curve, count = trajectory(agent, opt, ts.NEW_GOAL, cfg["iterations"],
                        seed["branches"][repeat], cfg,
                        telemetry(agent, opt, f"{mode}_{repair}_r{repeat}", phase="branch",
                                  mode=mode, repair=repair, repeat=repeat),
                        frozen=frozen, snapshot_every=10)
                    rows.append(dict(mode=mode, repair=repair, repeat=repeat, curve=curve,
                                     iterations=count, start_return=curve[0][1], final_return=curve[-1][1]))
        result = dict(history=history, config_hash=config_hash(cfg), seeds=seed,
                      source_converged=converged, aging_iterations=used,
                      source_description=source_description, source_curves=[short_curve, long_curve], rows=rows)
    finally:
        store.close()
    atomic_json(folder / "result.json", result)
    atomic_json(folder / "complete.json", dict(result_sha256=sha(folder / "result.json"),
        artifacts_sha256=sha(folder / "artifacts.jsonl"), config_hash=config_hash(cfg)))
    return result


def interval(values):
    # Exactly eight histories are required by the preregistration.
    mean = statistics.mean(values)
    radius = 2.365 * statistics.stdev(values) / len(values) ** .5
    return dict(mean=mean, ci95=[mean - radius, mean + radius], per_history=values)


def scores(rows, mode, repeats):
    return {repair: statistics.mean(x["final_return"] for x in rows if
            x["mode"] == mode and x["repair"] == repair and x["repeat"] in repeats) for repair in MENU}


def winner(score, margin):
    ordered = sorted(MENU, key=lambda r: score[r], reverse=True)
    return ordered[0] if score[ordered[0]] - score[ordered[1]] > margin else "tie"


def summarize(histories, cfg):
    if len(histories) != 8 or sorted(h["history"] for h in histories) != list(range(8)):
        return dict(verdict="incomplete", histories=len(histories))
    if not all(h["source_converged"] for h in histories):
        return dict(verdict="invalid_sources", histories=8)
    if cfg["repeats"] != 4:
        raise ValueError("The confirmatory design requires four repeats")
    differences, regrets, interactions, detail = 0, [], [], []
    for h in sorted(histories, key=lambda h: h["history"]):
        rows = h["rows"]
        expected = {(m, r, rep) for m in MODES for r in MENU for rep in range(4)}
        actual = {(x["mode"], x["repair"], x["repeat"]) for x in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError("Incomplete or duplicate branch panel")
        all_scores = {m: scores(rows, m, range(4)) for m in MODES}
        winners = {m: winner(all_scores[m], cfg["tie_margin"]) for m in MODES}
        different = "tie" not in winners.values() and winners["live"] != winners["frozen"]
        differences += int(different)
        choices = {m: max(MENU, key=scores(rows, m, (0, 1)).get) for m in MODES}
        test = scores(rows, "live", (2, 3))
        regrets.append(test[choices["live"]] - test[choices["frozen"]])
        interactions.append((all_scores["live"]["optimizer_reset"] - all_scores["live"]["continue"])
                            - (all_scores["frozen"]["optimizer_reset"] - all_scores["frozen"]["continue"]))
        detail.append(dict(history=h["history"], scores=all_scores, winners=winners, choices=choices,
                           heldout_live_scores=test, heldout_regret=regrets[-1]))
    regret = interval(regrets)
    success = differences >= cfg["divergence_bar"] and regret["mean"] > .1 and regret["ci95"][0] > 0
    return dict(verdict="feedback_decision_change" if success else "not_demonstrated", histories=8,
                distinct_winners=differences, heldout_regret=regret,
                optimizer_continue_interaction=interval(interactions), per_history=detail)


def load_completed(folder, cfg):
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if complete["config_hash"] != config_hash(cfg):
        raise ValueError("Existing history has a different config")
    for key, name in (("result_sha256", "result.json"), ("artifacts_sha256", "artifacts.jsonl")):
        if complete[key] != sha(folder / name):
            raise ValueError(f"History checksum mismatch: {folder / name}")
    for row in (folder / "artifacts.jsonl").read_text(encoding="utf-8").splitlines():
        artifact = json.loads(row)
        if sha(folder / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Recorded telemetry or snapshot hash mismatch")
    return json.loads((folder / "result.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/causal-trainability/two_state_v2_20261109")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = dict(CONFIG)
    cfg_hash = config_hash(cfg)
    manifest = dict(config=cfg, config_hash=cfg_hash, revision=code_revision(), prereg_sha256=sha(PREREG))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        if original["config_hash"] != cfg_hash or original["revision"]["files"] != manifest["revision"]["files"] or original["prereg_sha256"] != manifest["prereg_sha256"]:
            raise ValueError("Frozen manifest differs; use a new output root")
    else:
        if args.summarize:
            raise ValueError("No manifest to summarize")
        atomic_json(manifest_path, manifest)
    results = []
    started = time.time()
    for history in range(cfg["histories"]):
        folder = args.out / f"history{history}"
        if folder.exists():
            result = load_completed(folder, cfg)
        elif args.summarize:
            continue
        else:
            result = history_run(history, cfg, folder)
            print(f"history {history} complete; elapsed {time.time() - started:.1f}s", flush=True)
        results.append(result)
    summary = dict(config_hash=cfg_hash, manifest_sha256=sha(manifest_path),
                   results_sha256={f"history{h['history']}": sha(args.out / f"history{h['history']}" / "result.json") for h in results},
                   **summarize(results, cfg))
    atomic_json(args.out / "summary.json", summary)
    print(summary["verdict"], flush=True)


if __name__ == "__main__":
    main()
