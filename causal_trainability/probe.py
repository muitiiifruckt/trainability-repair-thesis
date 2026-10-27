"""Offline trainability probe (sprint 03, S3; rule in notes/ppo_probe.md, criteria in notes/sprint03_criteria.md).

The body (conv, fc) of a network gets a new head with zero weights and fits random labels -
independent Gaussian targets for every distinct game state, standardised - with a fresh Adam.
There are far more distinct states than features, so the head alone cannot fit the labels and
the body has to change.  Every run starts from loss 1 exactly.  Outcome: the mean log full-data
loss over the logged points after step 0 (lower = faster).

Bodies: `fresh` (theta0), `aged_actions_chain` (end of the stage B chain of action permutations,
where T7 found an adaptation deficit), `aged_1M` (the 1M checkpoint, no deficit shown in T7) and
`aged_channels_chain`.  `*_scaled` bodies have fc (weights and bias) rescaled so that the mean
feature norm on the probe states equals the fresh one (the same function up to the scale of the
features; ReLU is positively homogeneous).  `*_frozen` runs train only the head: the fresh one
sets the scale of every history (frozen body = no body learning at all).

Run:  .venv\\Scripts\\python.exe -m causal_trainability.probe --queue
      .venv\\Scripts\\python.exe -m causal_trainability.probe --summarize --output OUT.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn

from causal_trainability import deficit, hybrids
from causal_trainability.ppo_minatar import (VecMinAtar, effective_rank, reference_batch, representation_metrics,
                                             seed_of)
from causal_trainability.telemetry import ROOT, Telemetry, TelemetryStore, code_revision

PROBE = dict(data_envs=8, states_per_env=2048, random_action_share=0.5, outputs=16, steps=4000, batch=256,
             lr=2.5e-4, adam_eps=1e-8, log_every=100, snapshot_steps=[0, 500, 1000, 2000, 4000], seeds=3,
             eval_chunk=2048, merge_tolerance=1e-4)
BODIES = ("fresh", "aged_actions_chain", "aged_1M", "aged_channels_chain")
SCALED = ("aged_actions_chain", "aged_1M")
RUNS = ([(body, "trained") for body in BODIES] + [(f"{body}_scaled", "trained") for body in SCALED]
        + [(body, "frozen") for body in BODIES])
# Preregistered comparisons (worse arm, reference arm) and their scale-matched sensitivity versions.
COMPARISONS = {"chain_vs_fresh": ("aged_actions_chain", "fresh"), "chain_vs_1M": ("aged_actions_chain", "aged_1M")}
SENSITIVITY = {"chain_vs_fresh": ("aged_actions_chain_scaled", "fresh"),
               "chain_vs_1M": ("aged_actions_chain_scaled", "aged_1M_scaled")}
DESCRIPTIVE = {"1M_vs_fresh": ("aged_1M", "fresh"), "channels_chain_vs_fresh": ("aged_channels_chain", "fresh"),
               "channels_chain_vs_1M": ("aged_channels_chain", "aged_1M"),
               "1M_scaled_vs_fresh": ("aged_1M_scaled", "fresh"),
               **{f"{b}_frozen_vs_fresh_frozen": (f"{b}_frozen", "fresh_frozen") for b in BODIES[1:]}}
HISTORIES = 6
T975_DF5 = 2.570581835636314  # two-sided 95%, 6 histories
RELATIVE = 0.15  # practical size: 15% of the gap between a frozen and a learning fresh body


def run_name(arm, mode):
    return arm if mode == "trained" else f"{arm}_frozen"


def bodies(source, chains):
    """Start parameters of the unscaled bodies and where they come from."""
    aged = hybrids.load(source, "age1.00")
    row = hybrids.checkpoint_row(source, "age1.00")
    states = {"fresh": aged["theta0"], "aged_1M": aged["model"]}
    origin = {"fresh": {"checkpoint": row, "part": "theta0"}, "aged_1M": {"checkpoint": row, "part": "model"}}
    for kind in ("actions", "channels"):
        end, summary = deficit.chain_end(Path(chains) / f"{kind}_chain")
        if any(not torch.equal(end["theta0"][k], aged["theta0"][k]) for k in aged["theta0"]):
            raise ValueError(f"{kind} chain does not belong to this history")
        states[f"aged_{kind}_chain"] = end["model"]
        origin[f"aged_{kind}_chain"] = {"chain_end": summary["end"], "rounds": len(summary["rounds"])}
    return aged["config"], states, origin


def collect_states(policy_state, cfg, history, probe):
    """Distinct native-game states visited by the source agent's sampling policy with a share of
    uniform random actions (the trained policy alone revisits a few hundred states); the same for
    every arm, in a fixed (sorted) order."""
    data_cfg = dict(cfg, num_envs=probe["data_envs"], shift=None)
    model = hybrids.model_from(policy_state, cfg)
    generator = torch.Generator().manual_seed(seed_of("probe-data-policy", history))
    envs = VecMinAtar(data_cfg, seed_of("probe-data", history))
    rows = []
    for _ in range(probe["states_per_env"]):
        obs = torch.tensor(envs.observations())
        rows.append(obs)
        with torch.no_grad():
            logits, _ = model(obs)
        action = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(1)
        explore = torch.rand(len(action), generator=generator) < probe["random_action_share"]
        action[explore] = torch.randint(cfg["num_actions"], (int(explore.sum()),), generator=generator)
        envs.step(action.numpy())
    states = torch.cat(rows)
    return torch.unique(states.flatten(1), dim=0).reshape(-1, *states.shape[1:])


def random_targets(states, probe, key):
    """Independent Gaussian labels for every distinct state, seeded by `key`, standardised per output."""
    generator = torch.Generator().manual_seed(seed_of("probe-target", *key))
    y = torch.randn(len(states), probe["outputs"], generator=generator)
    return (y - y.mean(0)) / y.std(0, unbiased=False)


class ProbeNet(nn.Module):
    """The body's conv and fc with a new zero head; the old actor and critic are not used."""

    def __init__(self, cfg, body, outputs):
        super().__init__()
        self.body = hybrids.model_from(body, cfg)
        self.head = nn.Linear(cfg["hidden"], outputs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, obs):
        return self.body.features(obs)

    def forward(self, obs):
        return self.head(self.features(obs)[0])

    def layers(self):
        groups = self.body.layers()
        return {"conv": groups["conv"], "fc": groups["fc"], "head": list(self.head.parameters())}


def hidden(model, states, chunk):
    with torch.no_grad():
        return torch.cat([model.features(states[i:i + chunk])[0] for i in range(0, len(states), chunk)])


def rescaled(body, cfg, states, target_norm, chunk):
    """The body with fc multiplied by c so that the mean feature norm equals `target_norm`."""
    if cfg["arch"] != "cnn":
        raise ValueError("Rescaling fc preserves the features up to scale only without LayerNorm")
    c = target_norm / float(hidden(ProbeNet(cfg, body, 1), states, chunk).norm(dim=1).mean())
    return dict(body, **{"fc.weight": body["fc.weight"] * c, "fc.bias": body["fc.bias"] * c}), c


def diagnostics(model, states, probe):
    """Start of a run on the probe states: feature scale, dead units, rank, merged states, weight norms."""
    h = hidden(model, states, probe["eval_chunk"]).double()
    norms = h.norm(dim=1)
    close = torch.cdist(h, h) <= probe["merge_tolerance"] * float(norms.mean())
    return {"feature_norm": float(norms.mean()), "merged_state_pairs": int((close.sum() - len(h)) // 2),
            "distinct_feature_vectors": len(torch.unique(close.int().argmax(1))),
            "fc_dead_fraction": float((h <= 0).all(0).double().mean()), "fc_effective_rank": effective_rank(h),
            "weight_norms": {name: float(torch.sqrt(sum((p.detach().double() ** 2).sum() for p in params)))
                             for name, params in model.layers().items()}}


def layer_metrics(model, reference, states):
    out = representation_metrics(model, reference)
    for layer, values in representation_metrics(model, states).items():
        out[layer].update({f"{k}_probe_states": v for k, v in values.items()})
    return out


def full_loss(model, states, y, chunk):
    with torch.no_grad():
        total = sum(float(((model(states[i:i + chunk]) - y[i:i + chunk]) ** 2).sum())
                    for i in range(0, len(states), chunk))
    return total / y.numel()


def fit(model, optimizer, states, y, probe, key, telemetry, reference):
    generator = torch.Generator().manual_seed(seed_of("probe-order", *key))
    start = diagnostics(model, states, probe)
    initial = {name: [p.detach().clone() for p in params] for name, params in model.layers().items()}
    curve = [{"step": 0, "loss": full_loss(model, states, y, probe["eval_chunk"])}]
    telemetry.step(0, force=True, metrics=curve[0], layer_extra=layer_metrics(model, reference, states))
    telemetry.snapshot(0, model, optimizer)
    for step in range(1, probe["steps"] + 1):
        idx = torch.randint(len(states), (probe["batch"],), generator=generator)
        loss = ((model(states[idx]) - y[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % probe["log_every"] == 0:
            curve.append({"step": step, "loss": full_loss(model, states, y, probe["eval_chunk"])})
            telemetry.step(step, force=True, metrics=dict(curve[-1], minibatch_loss=float(loss.detach())),
                           layer_extra=layer_metrics(model, reference, states))
        if step in probe["snapshot_steps"]:
            telemetry.snapshot(step, model, optimizer)
    moved = {name: float(torch.sqrt(sum(((p.detach() - q).double() ** 2).sum() for p, q in zip(params, initial[name]))))
             for name, params in model.layers().items()}
    logs = [math.log(point["loss"]) for point in curve[1:]]
    return {"start": start, "distance_moved": moved, "curve": curve, "auc_log": mean(logs), "final_log": logs[-1]}


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def run_history(source, chains, out, probe=PROBE, runs=RUNS, telemetry=True):
    torch.set_num_threads(1)
    cfg, starts, origin = bodies(source, chains)
    history = deficit.history_of(cfg)
    out = Path(out)
    states = collect_states(starts["aged_1M"], cfg, history, probe)
    reference = reference_batch(cfg)
    fresh_norm = float(hidden(ProbeNet(cfg, starts["fresh"], 1), states, probe["eval_chunk"]).norm(dim=1).mean())
    for body in SCALED:
        starts[f"{body}_scaled"], c = rescaled(starts[body], cfg, states, fresh_norm, probe["eval_chunk"])
        origin[f"{body}_scaled"] = dict(origin[body], fc_scale=c)
    seeds = {"data_env": seed_of("probe-data", history), "data_policy": seed_of("probe-data-policy", history),
             "targets": [seed_of("probe-target", history, s) for s in range(probe["seeds"])],
             "order": [seed_of("probe-order", history, s) for s in range(probe["seeds"])]}
    store = TelemetryStore(out, f"probe_{history}")
    results, targets_sha = {}, []
    for seed in range(probe["seeds"]):
        y = random_targets(states, probe, (history, seed))
        targets_sha.append(digest(y))
        for arm, mode in runs:
            model = ProbeNet(cfg, starts[arm], probe["outputs"])
            groups = model.layers()
            if mode == "frozen":
                for p in groups["conv"] + groups["fc"]:
                    p.requires_grad_(False)
            params = groups["head"] if mode == "frozen" else [p for g in groups.values() for p in g]
            optimizer = torch.optim.Adam(params, lr=probe["lr"], eps=probe["adam_eps"])
            name = run_name(arm, mode)
            log = Telemetry(out, f"probe_{history}_{name}_t{seed}", groups, optimizer, store=store,
                            every=probe["log_every"], enabled=telemetry,
                            meta={"kind": "probe", "history": history, "arm": arm, "mode": mode, "target_seed": seed})
            results.setdefault(name, []).append(
                dict(fit(model, optimizer, states, y, probe, (history, seed), log, reference), seed=seed))
    store.close()
    result = {"history": history, "probe": probe, "runs_spec": [list(r) for r in runs], "origin": origin,
              "config": cfg, "states": {"count": len(states), "sha256": digest(states)},
              "reference_sha256": digest(reference), "seeds": seeds, "targets_sha256": targets_sha,
              "revision": code_revision(), "runs": results}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def interval(values, t=T975_DF5):
    m, half = mean(values), t * stdev(values) / math.sqrt(len(values))
    return [m - half, m + half]


def verdict(d, r, within):
    """d: per-history differences (worse arm - reference arm); r: the same in units of the history's scale."""
    lo, hi = interval(r)
    if lo > 0 and sum(x > 0 for x in d) >= 5 and mean(d) > within:
        return "worse" if mean(r) >= RELATIVE else "worse_but_small"
    if hi < 0 and sum(x < 0 for x in d) >= 5 and -mean(d) > within:
        return "better"
    return "practically_not_worse" if hi < RELATIVE else "undetermined"


def compare(results, worse, reference):
    per_history = []
    for r in results:
        a, b = r["runs"][worse], r["runs"][reference]
        if [x["seed"] for x in a] != [x["seed"] for x in b]:
            raise SystemExit(f"{r['history']}: {worse} and {reference} are not paired")
        paired = [x["auc_log"] - y["auc_log"] for x, y in zip(a, b)]
        scale = mean(x["auc_log"] - y["auc_log"] for x, y in zip(r["runs"]["fresh_frozen"], r["runs"]["fresh"]))
        if scale <= 0:
            raise SystemExit(f"{r['history']}: a frozen body learns as fast as a training one")
        d = mean(paired)
        per_history.append({"history": r["history"], "d": d, "r": d / scale, "scale": scale, "paired": paired,
                            "final_log_difference": mean(x["final_log"] - y["final_log"] for x, y in zip(a, b))})
    d, rel = [h["d"] for h in per_history], [h["r"] for h in per_history]
    within = math.sqrt(mean(stdev(h["paired"]) ** 2 for h in per_history))
    return {"arms": [worse, reference], "verdict": verdict(d, rel, within), "mean_d": mean(d), "mean_r": mean(rel),
            "ci95_r": interval(rel), "ci95_d": interval(d), "positive_histories": sum(x > 0 for x in d),
            "within_history_sd": within, "between_history_sd": stdev(d),
            "mean_final_log_difference": mean(h["final_log_difference"] for h in per_history),
            "per_history": per_history}


def summarize(results):
    """Verdicts of the preregistered comparisons and the reading of notes/ppo_probe.md."""
    if len(results) != HISTORIES:
        raise SystemExit(f"{len(results)} histories, the rule needs exactly {HISTORIES}")
    main = {name: compare(results, *arms) for name, arms in COMPARISONS.items()}
    sensitivity = {name: compare(results, *arms) for name, arms in SENSITIVITY.items()}
    descriptive = {name: compare(results, *arms) for name, arms in DESCRIPTIVE.items()
                   if all(a in results[0]["runs"] for a in arms)}
    c1, c2 = main["chain_vs_fresh"]["verdict"], main["chain_vs_1M"]["verdict"]
    not_worse = ("practically_not_worse", "better")
    if c1 == "worse":
        reading = ("body_slower_specific_to_chain" if c2 == "worse" else
                   "body_slower_not_specific_to_chain" if c2 in not_worse else "body_slower_specificity_undetermined")
    else:
        reading = ("body_slower_but_small" if c1 == "worse_but_small" else
                   "body_not_slower" if c1 in not_worse else "undetermined")
    scale_dependent = [name for name, c in main.items()
                       if c["verdict"] == "worse" and sensitivity[name]["verdict"] != "worse"]
    scale_masks = [name for name, c in main.items()
                   if c["verdict"] != "worse" and sensitivity[name]["verdict"] == "worse"]
    return {"schema_version": 1, "relative_threshold": RELATIVE, "comparisons": main, "sensitivity": sensitivity,
            "descriptive": descriptive, "reading": reading, "scale_dependent": scale_dependent,
            "scale_masks": scale_masks, "rule": "notes/ppo_probe.md"}


def load_results(folder):
    paths = sorted(Path(folder).glob("*/result.json"))
    results = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if len({r["history"] for r in results}) != len(results):
        raise SystemExit("A history is present twice")
    for key in ("probe", "runs_spec"):
        if len({json.dumps(r[key], sort_keys=True) for r in results}) != 1:
            raise SystemExit(f"Histories differ in {key}")
    if len({r["revision"]["git_head"] for r in results}) != 1 or any(r["revision"]["dirty_causal_trainability"]
                                                                     for r in results):
        raise SystemExit("Histories ran on different or uncommitted code")
    return results, hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def queue(sources, deficit_runs, out, workers):
    def job(source):
        folder = Path(out) / Path(source).name
        if (folder / "result.json").exists():
            return f"{folder.name}: done before"
        if folder.exists():
            shutil.rmtree(folder)
        subprocess.run([sys.executable, "-m", "causal_trainability.probe", "--source", str(source),
                        "--chains", str(Path(deficit_runs) / Path(source).name), "--out", str(folder)],
                       cwd=ROOT, check=True)
        return f"{folder.name}: done"

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for line in pool.map(job, sources):
            print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--calibrate", action="store_true", help="fresh body of one history only")
    parser.add_argument("--pilot", default=str(ROOT / "runs" / "causal-trainability" / "ppo_pilot"))
    parser.add_argument("--deficit-runs", default=str(ROOT / "runs" / "causal-trainability" / "deficit_B"))
    parser.add_argument("--histories", nargs="+", default=[f"breakout_cnn_s{s}" for s in range(6)])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source")
    parser.add_argument("--chains")
    parser.add_argument("--out")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "results" / "ppo-probe" / "summary.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    default = "probe_calibration" if args.calibrate else "probe"
    out = Path(args.out or ROOT / "runs" / "causal-trainability" / default)
    if args.queue:
        if code_revision()["dirty_causal_trainability"]:
            raise SystemExit("causal_trainability has uncommitted changes; commit before the queue")
        queue([Path(args.pilot) / h for h in args.histories], args.deficit_runs, out, args.workers)
    elif args.summarize:
        if args.output.exists():
            raise SystemExit("Output exists; analyses are immutable")
        results, digest_all = load_results(out)
        summary = dict(summarize(results), results_sha256=digest_all, git_head=results[0]["revision"]["git_head"],
                       analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name, c in summary["comparisons"].items():
            print(name, c["verdict"], round(c["mean_r"], 3), [round(x, 3) for x in c["ci95_r"]], c["positive_histories"])
        print("reading:", summary["reading"], "scale dependent:", summary["scale_dependent"])
    elif args.calibrate:
        source = Path(args.pilot) / args.histories[0]
        r = run_history(source, Path(args.deficit_runs) / source.name, out, runs=(("fresh", "trained"),))
        for run in r["runs"]["fresh"]:
            print(run["seed"], [(p["step"], round(p["loss"], 3)) for p in run["curve"][:: 5]])
    else:
        run_history(Path(args.source), Path(args.chains), Path(args.out))


if __name__ == "__main__":
    import causal_trainability.probe as module  # pickled classes must not be recorded as __main__.*
    module.main()
