"""Offline trainability probe (sprint 03, S3; notes/ppo_probe.md, criteria in notes/sprint03_criteria.md).

The body (conv, fc) of an aged or a fresh network of one history gets the same new head with
zero weights and fits random targets - the standardised outputs of a fixed random network - on
the same game states with a fresh Adam; the whole network is trained.  Old game knowledge neither
helps nor hurts, and every run starts from loss 1 exactly.  Outcomes: the mean log full-data loss
over the logged points after step 0 (primary) and the final log loss.

Arms: `fresh` (theta0 of the history), `aged_actions_chain` (end of the stage B chain of action
permutations, where T7 found an adaptation deficit; primary), `aged_1M` (the 1M checkpoint) and
`aged_channels_chain` (end of the channel-permutation chain); the last two are exploratory.

Run:  .venv\\Scripts\\python.exe -m causal_trainability.probe --queue
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
from causal_trainability.ppo_minatar import VecMinAtar, reference_batch, representation_metrics, seed_of
from causal_trainability.telemetry import ROOT, Telemetry, TelemetryStore, code_revision

PROBE = dict(data_envs=8, states_per_env=1024, outputs=16, target_hidden=128, steps=4000, batch=256,
             lr=2.5e-4, adam_eps=1e-8, log_every=100, seeds=3, eval_chunk=2048)
ARMS = ("fresh", "aged_actions_chain", "aged_1M", "aged_channels_chain")
PRIMARY = "aged_actions_chain"
HISTORIES = 6
T975_DF5 = 2.570581835636314  # two-sided 95%, 6 histories
PRACTICAL = math.log(1.1)  # "not worse": aged loss at most 10% above fresh on average (log scale)


def bodies(source, chains):
    """Start parameters of every arm and where they come from."""
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
    """Native-game states visited by the source agent's sampling policy; the same for every arm."""
    data_cfg = dict(cfg, num_envs=probe["data_envs"], shift=None)
    model = hybrids.model_from(policy_state, cfg)
    with torch.random.fork_rng(devices=[]):
        envs = VecMinAtar(data_cfg, seed_of("probe-data", history))
        generator = torch.Generator().manual_seed(seed_of("probe-data-policy", history))
        rows = []
        for _ in range(probe["states_per_env"]):
            obs = torch.tensor(envs.observations())
            rows.append(obs)
            with torch.no_grad():
                logits, _ = model(obs)
            envs.step(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(1).numpy())
    return torch.cat(rows)


class TargetNet(nn.Module):
    def __init__(self, cfg, probe):
        super().__init__()
        c = cfg["conv_channels"]
        self.conv = nn.Conv2d(cfg["input_channels"], c, 3)
        self.fc = nn.Linear(c * 8 * 8, probe["target_hidden"])
        self.out = nn.Linear(probe["target_hidden"], probe["outputs"])
        for layer, gain in ((self.conv, math.sqrt(2)), (self.fc, math.sqrt(2)), (self.out, 1.0)):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        x = torch.relu(self.conv(obs.float())).flatten(1)
        return self.out(torch.relu(self.fc(x)))


def random_targets(states, cfg, probe, key):
    """Outputs of a random network seeded by `key`, standardised per output over the states."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_of("probe-target", *key))
        net = TargetNet(cfg, probe)
    with torch.no_grad():
        y = net(states)
    sd = y.std(0, unbiased=False)
    if float(sd.min()) < 1e-6:
        raise ValueError("A target output is constant on the states")
    return (y - y.mean(0)) / sd


class ProbeNet(nn.Module):
    """The arm's conv and fc with a new zero head; the old actor and critic are not used."""

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


def full_loss(model, states, y, chunk):
    with torch.no_grad():
        total = sum(float(((model(states[i:i + chunk]) - y[i:i + chunk]) ** 2).sum())
                    for i in range(0, len(states), chunk))
    return total / y.numel()


def fit(model, optimizer, states, y, probe, key, telemetry, reference):
    generator = torch.Generator().manual_seed(seed_of("probe-order", *key))
    with torch.no_grad():
        hidden, _ = model.features(states[:probe["eval_chunk"]])
    start = {"feature_norm": float(hidden.norm(dim=1).mean()), **representation_metrics(model, reference)}
    curve = [{"step": 0, "loss": full_loss(model, states, y, probe["eval_chunk"])}]
    telemetry.step(0, force=True, metrics=curve[0], layer_extra=representation_metrics(model, reference))
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
                           layer_extra=representation_metrics(model, reference))
    telemetry.snapshot(probe["steps"], model, optimizer)
    logs = [math.log(point["loss"]) for point in curve[1:]]
    return {"start": start, "curve": curve, "auc_log": mean(logs), "final_log": logs[-1]}


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def run_history(source, chains, out, probe=PROBE, arms=ARMS):
    torch.set_num_threads(1)
    cfg, starts, origin = bodies(source, chains)
    history = deficit.history_of(cfg)
    out = Path(out)
    states = collect_states(starts["aged_1M"], cfg, history, probe)
    reference = reference_batch(cfg)
    store = TelemetryStore(out, f"probe_{history}")
    runs, targets_sha = {arm: [] for arm in arms}, []
    for seed in range(probe["seeds"]):
        y = random_targets(states, cfg, probe, (history, seed))
        targets_sha.append(digest(y))
        for arm in arms:
            model = ProbeNet(cfg, starts[arm], probe["outputs"])
            optimizer = torch.optim.Adam([p for group in model.layers().values() for p in group],
                                         lr=probe["lr"], eps=probe["adam_eps"])
            run_id = f"probe_{history}_{arm}_t{seed}"
            telemetry = Telemetry(out, run_id, model.layers(), optimizer, store=store, every=probe["log_every"],
                                  meta={"kind": "probe", "history": history, "arm": arm, "target_seed": seed})
            runs[arm].append(dict(fit(model, optimizer, states, y, probe, (history, seed), telemetry, reference),
                                  seed=seed))
    store.close()
    result = {"history": history, "probe": probe, "arms": list(arms), "origin": origin, "config": cfg,
              "states": {"count": len(states), "sha256": digest(states)}, "targets_sha256": targets_sha,
              "revision": code_revision(), "runs": runs}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def interval(values, t=T975_DF5):
    m, half = mean(values), t * stdev(values) / math.sqrt(len(values))
    return [m - half, m + half]


def summarize(results):
    """Verdict for the primary arm (rule in notes/ppo_probe.md); the other arms are descriptive."""
    if len(results) != HISTORIES:
        raise SystemExit(f"{len(results)} histories, the rule needs exactly {HISTORIES}")
    arms = {}
    for arm in results[0]["arms"]:
        if arm == "fresh":
            continue
        per_history = []
        for r in results:
            paired = {key: [a[key] - f[key] for a, f in zip(r["runs"][arm], r["runs"]["fresh"])]
                      for key in ("auc_log", "final_log")}
            per_history.append({"history": r["history"], **{f"{k}_difference": mean(v) for k, v in paired.items()},
                                "paired": paired})
        diffs = [h["auc_log_difference"] for h in per_history]
        within = math.sqrt(mean(stdev(h["paired"]["auc_log"]) ** 2 for h in per_history))
        ci = interval(diffs)
        positive = sum(d > 0 for d in diffs)
        arms[arm] = {"mean_auc_log_difference": mean(diffs), "ci95": ci, "positive_histories": positive,
                     "within_history_sd": within, "between_history_sd": stdev(diffs),
                     "mean_final_log_difference": mean(h["final_log_difference"] for h in per_history),
                     "ci95_final": interval([h["final_log_difference"] for h in per_history]),
                     "per_history": per_history}
        if arm == PRIMARY:
            arms[arm]["verdict"] = ("worse" if ci[0] > 0 and positive >= 5 and mean(diffs) > within else
                                    "practically_not_worse" if ci[1] < PRACTICAL else "undetermined")
    return {"schema_version": 1, "primary": PRIMARY, "practical_threshold": PRACTICAL, "arms": arms,
            "verdict": arms[PRIMARY]["verdict"], "rule": "notes/ppo_probe.md"}


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
        r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        return f"{folder.name}: " + ", ".join(f"{arm} {mean(x['auc_log'] for x in r['runs'][arm]):.3f}" for arm in r["arms"])

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for line in pool.map(job, sources):
            print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--calibrate", action="store_true", help="fresh arm of one history only")
    parser.add_argument("--pilot", default=str(ROOT / "runs" / "causal-trainability" / "ppo_pilot"))
    parser.add_argument("--deficit-runs", default=str(ROOT / "runs" / "causal-trainability" / "deficit_B"))
    parser.add_argument("--histories", nargs="+", default=[f"breakout_cnn_s{s}" for s in range(6)])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source")
    parser.add_argument("--chains")
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.queue:
        if code_revision()["dirty_causal_trainability"]:
            raise SystemExit("causal_trainability has uncommitted changes; commit before the queue")
        out = Path(args.out or ROOT / "runs" / "causal-trainability" / "probe")
        queue([Path(args.pilot) / h for h in args.histories], args.deficit_runs, out, args.workers)
    elif args.calibrate:
        source = Path(args.pilot) / args.histories[0]
        r = run_history(source, Path(args.deficit_runs) / source.name, Path(args.out), arms=("fresh",))
        for run in r["runs"]["fresh"]:
            print(run["seed"], [(p["step"], round(p["loss"], 3)) for p in run["curve"][:: 5]] + [round(run["curve"][-1]["loss"], 3)])
    else:
        run_history(Path(args.source), Path(args.chains), Path(args.out))


if __name__ == "__main__":
    import causal_trainability.probe as module  # pickled classes must not be recorded as __main__.*
    module.main()
