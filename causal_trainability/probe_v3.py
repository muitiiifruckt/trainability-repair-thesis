"""Long-horizon conditional follow-up; frozen rule: notes/ppo_probe_v3.md.

No new independent source histories: previous probe outcomes informed this design.
Run --queue only after code and protocol have been frozen and reviewed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import torch

from causal_trainability import deficit, probe
from causal_trainability.ppo_minatar import reference_batch, seed_of
from causal_trainability.telemetry import ROOT, Telemetry, code_revision, config_hash, record_artifact

SPEC = dict(probe.PROBE, steps=8000, late_start=4000, telemetry_every=400,
            snapshot_steps=[0, 1000, 4000, 6000, 8000], seed_namespace="probe-v3")
BODIES = ("fresh", "aged_actions_chain", "aged_1M", "aged_actions_chain_scaled", "aged_1M_scaled",
          "fresh_chain_factor", "fresh_1M_factor")
RUNS = tuple((body, mode) for body in BODIES for mode in ("trained", "frozen"))
CONTRASTS = {"chain_vs_fresh": ("aged_actions_chain", "fresh"),
             "chain_vs_1M": ("aged_actions_chain", "aged_1M")}
SCALE_CONTRASTS = {"chain_vs_fresh_same_factor": ("aged_actions_chain_scaled", "fresh_chain_factor"),
                   "1M_vs_fresh_same_factor": ("aged_1M_scaled", "fresh_1M_factor"),
                   "chain_scaled_vs_1M_scaled": ("aged_actions_chain_scaled", "aged_1M_scaled")}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix(f".pending-{os.getpid()}")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def scaled_copy(state, factor):
    return dict(state, **{"fc.weight": state["fc.weight"] * factor, "fc.bias": state["fc.bias"] * factor})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_arm(folder, row, spec, enabled):
    attempt = folder.parent.parent / row["attempt"]
    artifacts_path = attempt / "artifacts.jsonl"
    artifacts = ([json.loads(line) for line in artifacts_path.read_text(encoding="utf-8").splitlines()]
                 if enabled else [])
    seal = {"result_sha256": sha256(folder / "result.json"), "telemetry_enabled": enabled,
            "artifacts_manifest_sha256": sha256(artifacts_path) if enabled else None,
            "artifacts": artifacts}
    atomic_json(folder / "completion.json", seal)
    verify_arm(folder, row, spec, require_telemetry=enabled)


def verify_arm(folder, row, spec, require_telemetry=True):
    seal = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
    if sha256(folder / "result.json") != seal["result_sha256"]:
        raise ValueError("Completed arm result checksum mismatch")
    if require_telemetry and not seal["telemetry_enabled"]:
        raise ValueError("Scientific arm lacks required telemetry")
    if not seal["telemetry_enabled"]:
        return
    root = folder.parent.parent.resolve()
    attempt = (root / row["attempt"]).resolve()
    if root not in attempt.parents:
        raise ValueError("Attempt escapes history directory")
    artifact_manifest = attempt / "artifacts.jsonl"
    if sha256(artifact_manifest) != seal["artifacts_manifest_sha256"]:
        raise ValueError("Artifact manifest checksum mismatch")
    artifacts = [json.loads(line) for line in artifact_manifest.read_text(encoding="utf-8").splitlines()]
    if artifacts != seal["artifacts"]:
        raise ValueError("Completed arm artifact seal mismatch")
    snapshots, telemetry = [], []
    for artifact in artifacts:
        path = (attempt / artifact["path"]).resolve()
        if attempt not in path.parents or sha256(path) != artifact["sha256"]:
            raise ValueError("Artifact missing, outside attempt or checksum mismatch")
        if artifact["kind"] == "snapshot":
            snapshots.append(int(path.stem.rsplit("_u", 1)[1]))
        elif artifact["kind"] == "telemetry":
            telemetry.append(path)
    if sorted(snapshots) != sorted(spec["snapshot_steps"]) or len(telemetry) != 1:
        raise ValueError("Missing or extra snapshots/telemetry")


def prepare_bodies(source, chains, spec):
    cfg, starts, origins = probe.bodies(source, chains)
    starts.pop("aged_channels_chain")
    origins.pop("aged_channels_chain")
    history = deficit.history_of(cfg)
    states = probe.collect_states(starts["aged_1M"], cfg, history, spec)
    with torch.random.fork_rng(devices=[]):
        fresh_norm = float(probe.hidden(probe.ProbeNet(cfg, starts["fresh"], 1), states,
                                        spec["eval_chunk"]).norm(dim=1).mean())
        for body, control in (("aged_actions_chain", "fresh_chain_factor"), ("aged_1M", "fresh_1M_factor")):
            starts[f"{body}_scaled"], factor = probe.rescaled(starts[body], cfg, states, fresh_norm, spec["eval_chunk"])
            starts[control] = scaled_copy(starts["fresh"], factor)
            origins[f"{body}_scaled"] = dict(origins[body], fc_scale=factor)
            origins[control] = dict(origins["fresh"], fc_scale=factor, factor_from=body)
    return cfg, starts, origins, states


def outcomes(curve, late_start):
    points = {row["step"]: row["loss"] for row in curve}
    late = [math.log(row["loss"]) for row in curve if row["step"] > late_start]
    if late_start not in points or not late:
        raise ValueError("Missing preregistered late window")
    return {"late_auc_log": mean(late), "late_gain_log": math.log(points[late_start]) - math.log(curve[-1]["loss"]),
            "final_log": math.log(curve[-1]["loss"])}


def fit(model, optimizer, states, targets, spec, order_seed, log, reference):
    generator = torch.Generator().manual_seed(order_seed)
    start = probe.diagnostics(model, states, spec)
    initial = {name: [p.detach().clone() for p in params] for name, params in model.layers().items()}
    curve = [{"step": 0, "loss": probe.full_loss(model, states, targets, spec["eval_chunk"])}]
    if log.enabled:
        log.step(0, force=True, metrics=curve[-1], layer_extra=probe.layer_metrics(model, reference, states))
        log.snapshot(0, model, optimizer)
    for step in range(1, spec["steps"] + 1):
        idx = torch.randint(len(states), (spec["batch"],), generator=generator)
        loss = ((model(states[idx]) - targets[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % spec["log_every"] == 0:
            curve.append({"step": step, "loss": probe.full_loss(model, states, targets, spec["eval_chunk"])})
        if log.enabled and step % spec["telemetry_every"] == 0:
            log.step(step, force=True, metrics={"minibatch_loss": float(loss.detach())},
                     layer_extra=probe.layer_metrics(model, reference, states))
        if step in spec["snapshot_steps"]:
            log.snapshot(step, model, optimizer)
    moved = {name: float(torch.sqrt(sum(((p.detach() - q).double() ** 2).sum()
                                       for p, q in zip(params, initial[name]))))
             for name, params in model.layers().items()}
    return dict(start=start, curve=curve, distance_moved=moved, **outcomes(curve, spec["late_start"]))


def run_history(source, chains, out, spec=SPEC, telemetry=True, runs=RUNS):
    torch.set_num_threads(1)
    out = Path(out)
    cfg, starts, origins, states = prepare_bodies(source, chains, spec)
    history, revision = deficit.history_of(cfg), code_revision()
    reference = reference_batch(cfg)
    seeds = {"data_env": seed_of("probe-data", history), "data_policy": seed_of("probe-data-policy", history),
             "targets": [seed_of(spec["seed_namespace"], "targets", history, i) for i in range(spec["seeds"])],
             "order": [seed_of(spec["seed_namespace"], "order", history, i) for i in range(spec["seeds"])]}
    manifest = {"schema_version": 1, "history": history, "spec": spec, "config": cfg,
                "protocol_sha256": hashlib.sha256((ROOT / "notes/ppo_probe_v3.md").read_bytes()).hexdigest(),
                "runs_spec": [list(r) for r in runs], "config_hash": config_hash(dict(spec=spec, config=cfg)),
                "origin": origins, "revision": revision, "seeds": seeds,
                "states": {"count": len(states), "sha256": probe.digest(states)},
                "reference_sha256": probe.digest(reference),
                "interpretation": "conditional_followup_on_previously_inspected_histories"}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Refusing resume with changed source, code, protocol, data or seeds")
    else:
        atomic_json(manifest_path, manifest)
    results, targets_sha = {}, []
    for i, target_seed in enumerate(seeds["targets"]):
        generator = torch.Generator().manual_seed(target_seed)
        y = torch.randn(len(states), spec["outputs"], generator=generator)
        y = (y - y.mean(0)) / y.std(0, unbiased=False)
        targets_sha.append(probe.digest(y))
        for body, mode in runs:
            name = probe.run_name(body, mode)
            folder = out / "arms" / f"{name}_t{i}"
            result_path = folder / "result.json"
            if result_path.exists():
                row = json.loads(result_path.read_text(encoding="utf-8"))
                verify_arm(folder, row, spec, require_telemetry=telemetry)
                if row["targets_sha256"] != targets_sha[-1] or row["order_seed"] != seeds["order"][i]:
                    raise ValueError("Corrupt completed arm or changed seeds")
            else:
                # An interrupted attempt is preserved. A fresh attempt is separately indexed.
                attempts = folder / "attempts"
                attempts.mkdir(parents=True, exist_ok=True)
                attempt = attempts / f"{len(list(attempts.iterdir())):03d}"
                with torch.random.fork_rng(devices=[]):
                    model = probe.ProbeNet(cfg, starts[body], spec["outputs"])
                groups = model.layers()
                if mode == "frozen":
                    for p in groups["conv"] + groups["fc"]:
                        p.requires_grad_(False)
                params = groups["head"] if mode == "frozen" else [p for group in groups.values() for p in group]
                optimizer = torch.optim.Adam(params, lr=spec["lr"], eps=spec["adam_eps"])
                log = Telemetry(attempt, f"probe_v3_{history}_{name}_t{i}", groups, optimizer,
                                every=spec["telemetry_every"], enabled=telemetry,
                                meta={"kind": "probe_v3", "history": history, "arm": body, "mode": mode,
                                      "config_hash": manifest["config_hash"], "target_seed": target_seed,
                                      "order_seed": seeds["order"][i]})
                try:
                    row = dict(fit(model, optimizer, states, y, spec, seeds["order"][i], log, reference),
                               seed=i, target_seed=target_seed, order_seed=seeds["order"][i],
                               targets_sha256=targets_sha[-1], attempt=attempt.relative_to(out).as_posix())
                finally:
                    log.close()
                atomic_json(result_path, row)
                seal_arm(folder, row, spec, telemetry)
            results.setdefault(name, []).append(row)
    result = dict(manifest, targets_sha256=targets_sha, runs=results)
    if not (out / "result.json").exists():
        atomic_json(out / "result.json", result)
        record_artifact(out, out / "result.json", "probe_v3_result")
    return result


def compare(results, body, reference, adjusted=True):
    histories = []
    for row in results:
        runs = row["runs"]
        a, b = runs[body], runs[reference]
        if [r["seed"] for r in a] != [r["seed"] for r in b]:
            raise ValueError("Unpaired repeats")
        paired = [x["late_auc_log"] - y["late_auc_log"] for x, y in zip(a, b)]
        if adjusted:
            frozen_a, frozen_b = runs[f"{body}_frozen"], runs[f"{reference}_frozen"]
            paired = [d - (x["late_auc_log"] - y["late_auc_log"])
                      for d, x, y in zip(paired, frozen_a, frozen_b)]
        scale = mean(x["late_auc_log"] - y["late_auc_log"]
                     for x, y in zip(runs["fresh_frozen"], runs["fresh"]))
        if scale <= 0:
            raise ValueError("Non-positive fresh learning scale")
        histories.append({"history": row["history"], "paired": paired, "d": mean(paired),
                          "scale": scale, "r": mean(paired) / scale})
    differences, relative = [h["d"] for h in histories], [h["r"] for h in histories]
    within = math.sqrt(mean(stdev(h["paired"]) ** 2 for h in histories))
    return {"arms": [body, reference], "head_adjusted": adjusted,
            "verdict": probe.verdict(differences, relative, within),
            "mean_d": mean(differences), "mean_r": mean(relative), "ci95_r": probe.interval(relative),
            "positive_histories": sum(d > 0 for d in differences), "within_history_sd": within,
            "per_history": histories}


def summarize(folder):
    paths = sorted(Path(folder).glob("*/result.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = {f"breakout_cnn_s{i}" for i in range(6)}
    if len(rows) != 6 or {r["history"] for r in rows} != expected:
        raise ValueError("Exactly the six preregistered histories are required")
    for key in ("spec", "runs_spec", "revision", "protocol_sha256"):
        if len({json.dumps(r[key], sort_keys=True) for r in rows}) != 1:
            raise ValueError(f"Histories differ in {key}")
    if rows[0]["spec"] != SPEC or rows[0]["runs_spec"] != [list(r) for r in RUNS]:
        raise ValueError("Results do not implement the frozen protocol")
    if not rows[0]["revision"]["git_head"] or rows[0]["revision"]["dirty_causal_trainability"]:
        raise ValueError("Uncommitted or unidentified run code")
    for row in rows:
        history_folder = Path(folder) / row["history"]
        manifest = json.loads((history_folder / "manifest.json").read_text(encoding="utf-8"))
        if any(row.get(key) != value for key, value in manifest.items()):
            raise ValueError("History aggregate differs from frozen manifest")
        if row["protocol_sha256"] != sha256(ROOT / "notes/ppo_probe_v3.md"):
            raise ValueError("Protocol changed since execution")
        names = {probe.run_name(body, mode) for body, mode in RUNS}
        if set(row["runs"]) != names:
            raise ValueError("Missing or extra arm")
        for name, values in row["runs"].items():
            if [r["seed"] for r in values] != list(range(SPEC["seeds"])):
                raise ValueError("Missing or duplicated repeat")
            for i, value in enumerate(values):
                arm_folder = history_folder / "arms" / f"{name}_t{i}"
                verify_arm(arm_folder, value, SPEC)
                if json.loads((arm_folder / "result.json").read_text(encoding="utf-8")) != value:
                    raise ValueError("Aggregate differs from sealed arm result")
                if [point["step"] for point in value["curve"]] != list(range(0, SPEC["steps"] + 1, SPEC["log_every"])):
                    raise ValueError("Incomplete or different training horizon")
                if (value["target_seed"] != row["seeds"]["targets"][i]
                        or value["order_seed"] != row["seeds"]["order"][i]
                        or value["targets_sha256"] != row["targets_sha256"][i]):
                    raise ValueError("Targets/order not paired")
                recomputed = outcomes(value["curve"], SPEC["late_start"])
                if any(value[key] != outcome for key, outcome in recomputed.items()):
                    raise ValueError("Stored outcome differs from curve")
    return {"schema_version": 1, "rule": "notes/ppo_probe_v3.md",
            "interpretation": "conditional_followup_on_previously_inspected_histories",
            "primary": {name: compare(rows, *arms) for name, arms in CONTRASTS.items()},
            "raw_late_auc": {name: compare(rows, *arms, adjusted=False) for name, arms in CONTRASTS.items()},
            "scale_controls": {name: compare(rows, *arms) for name, arms in SCALE_CONTRASTS.items()},
            "secondary_descriptive": {
                name: [{"history": row["history"],
                        **{metric: mean(value[metric] for value in row["runs"][name])
                           for metric in ("late_gain_log", "final_log")}}
                       for row in rows]
                for name in sorted(rows[0]["runs"])},
            "inputs_sha256": {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "run_revision": rows[0]["revision"], "analysis_revision": code_revision()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--chains", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/causal-trainability/probe_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "research/results/ppo-probe-v3/summary.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        atomic_json(args.output, summarize(args.out))
    elif args.queue:
        revision = code_revision()
        if not revision["git_head"] or revision["dirty_causal_trainability"]:
            raise SystemExit("Freeze and commit reviewed code/protocol before queue")
        def job(i):
            history = f"breakout_cnn_s{i}"
            subprocess.run([sys.executable, "-m", "causal_trainability.probe_v3", "--source",
                            str(ROOT / "runs/causal-trainability/ppo_pilot" / history), "--chains",
                            str(ROOT / "runs/causal-trainability/deficit_B" / history), "--out", str(args.out / history)],
                           cwd=ROOT, check=True)
            print(history, "done", flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(job, range(6)))
    else:
        if args.source is None or args.chains is None:
            parser.error("--source and --chains are required for an individual history")
        run_history(args.source, args.chains, args.out)


if __name__ == "__main__":
    main()
