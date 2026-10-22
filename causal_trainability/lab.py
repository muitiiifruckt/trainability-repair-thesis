"""First lab measurement on PPO (sprint 02, T4; notes/ppo_lab.md).

Trainability of the 2x2 hybrids weights x Adam on a common batch. For one history the
fresh checkpoint (age 0.00) and the aged one (age 1.00) give H(params, adam, data, targets)
= FFFF, AFFF, FAFF, AAFF: data and targets are always fresh, so every hybrid sees the same
batch with the same targets.  Each hybrid takes K Adam steps (one PPO iteration: 4 epochs x
4 minibatches, same minibatch order for all hybrids) on one batch, separately for

* value regression: vf_coef * 0.5 * (V - R)^2 to the fresh-critic GAE returns; recorded as
  the full-batch loss before and after (absolute and relative drop);
* PPO policy loss: clipped surrogate with the ratio anchored to the hybrid's own policy at the
  start and a truncated importance weight min(pi_hybrid / pi_theta0, is_clip) for the theta0
  behaviour (as in T1); recorded as the full-batch surrogate gain.

Gradient clipping is the training one.  Three common batches: the stored fresh rollout and
two more rollouts of theta0 continued from the saved environments and generator.
Telemetry (per layer, every step) is written for every lab run.

Run:  .venv\\Scripts\\python.exe -m causal_trainability.lab
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import tempfile
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn

from causal_trainability import hybrids
from causal_trainability.ppo_minatar import PPO, seed_of
from causal_trainability.telemetry import ROOT, Telemetry, TelemetryStore, code_revision, config_hash

LAB = dict(batches=3, is_clip=1.0, clipfrac_ceiling=0.5, aged_tag="age1.00", fresh_tag="age0.00")
HYBRIDS = ("FFFF", "AFFF", "FAFF", "AAFF")  # params, adam, data, targets
MEASURES = ("value", "policy")


def fresh_batches(fresh, count):
    """The stored fresh rollout and count - 1 more rollouts of theta0 from the saved state."""
    with torch.random.fork_rng(), tempfile.TemporaryDirectory() as scratch:
        ppo = PPO(fresh["config"], scratch)
        ppo.model.load_state_dict(fresh["model"])
        ppo.envs, ppo.obs = copy.deepcopy(fresh["envs"]), fresh["obs"].clone()
        ppo.generator.set_state(fresh["generator"])
        return [fresh["rollout"]] + [ppo.rollout() for _ in range(count - 1)]


def common_batch(rollout, critic, cfg):
    advantages, returns, _ = hybrids.targets(critic, rollout, cfg)
    return {"obs": rollout["obs"], "actions": rollout["actions"], "behaviour_logp": rollout["logprobs"],
            "advantages": advantages, "returns": returns}


def minibatch_order(cfg, size, key):
    generator = torch.Generator().manual_seed(seed_of("lab-order", *key))
    step = size // cfg["num_minibatches"]
    order = []
    for _ in range(cfg["update_epochs"]):
        perm = torch.randperm(size, generator=generator)
        order += [perm[start:start + step] for start in range(0, size, step)]
    return order


def log_prob(model, obs, actions):
    logits, _ = model(obs)
    return torch.log_softmax(logits, -1).gather(1, actions[:, None]).squeeze(1)


def value_loss(model, batch, cfg, idx=None):
    idx = slice(None) if idx is None else idx
    _, values = model(batch["obs"][idx])
    return cfg["vf_coef"] * 0.5 * ((values - batch["returns"][idx]) ** 2).mean()


def surrogate(model, batch, anchor, weight, cfg, idx=None):
    """IS-weighted clipped surrogate (to maximise); advantages normalised within the slice."""
    idx = slice(None) if idx is None else idx
    ratio = torch.exp(log_prob(model, batch["obs"][idx], batch["actions"][idx]) - anchor[idx])
    adv = batch["advantages"][idx]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    clipped = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])
    return (weight[idx] * torch.min(ratio * adv, clipped * adv)).mean()


def measure(model, optimizer, batch, order, kind, cfg, lab, telemetry):
    with torch.no_grad():
        anchor = log_prob(model, batch["obs"], batch["actions"])
        raw = torch.exp(anchor - batch["behaviour_logp"])
        weight = raw.clamp(max=lab["is_clip"])

    def full():
        """Full-batch objective plus its parts: value = bias^2 + variance of V - R; policy = clip fraction."""
        with torch.no_grad():
            if kind == "value":
                _, values = model(batch["obs"])
                residual = values - batch["returns"]
                scale = cfg["vf_coef"] * 0.5
                return {"objective": float(scale * (residual ** 2).mean()),
                        "bias_part": float(scale * residual.mean() ** 2),
                        "variance_part": float(scale * residual.var(unbiased=False))}
            ratio = torch.exp(log_prob(model, batch["obs"], batch["actions"]) - anchor)
            return {"objective": float(surrogate(model, batch, anchor, weight, cfg)),
                    "clipfrac": float(((ratio - 1).abs() > cfg["clip"]).float().mean())}

    curve = [full()]
    telemetry.step(0, force=True, metrics=curve[0])
    for k, idx in enumerate(order, 1):
        loss = value_loss(model, batch, cfg, idx) if kind == "value" else -surrogate(model, batch, anchor, weight, cfg, idx)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"]))
        optimizer.step()
        curve.append(full())
        telemetry.step(k, force=True, metrics=dict(curve[-1], minibatch_loss=float(loss.detach()),
                                                   grad_norm_preclip=grad_norm))
    telemetry.snapshot(len(order), model, optimizer)
    with torch.no_grad():
        moved = log_prob(model, batch["obs"], batch["actions"]) - anchor
    start, after = curve[0], curve[-1]
    row = {"start": start["objective"], "after": after["objective"], "curve": curve,
           "mean_is_weight": float(weight.mean()), "is_clipped_fraction": float((raw > lab["is_clip"]).float().mean()),
           "is_ess_fraction": float(weight.sum() ** 2 / (len(weight) * (weight ** 2).sum())),
           "approx_kl": float(((moved.exp() - 1) - moved).mean())}
    relative = lambda a, b: (a - b) / a if a else None
    if kind == "value":
        row["drop"] = start["objective"] - after["objective"]
        row["relative_drop"] = relative(start["objective"], after["objective"])
        row["relative_drop_variance"] = relative(start["variance_part"], after["variance_part"])
        row["bias_share_start"] = start["bias_part"] / start["objective"] if start["objective"] else None
    else:
        row["gain"] = after["objective"] - start["objective"]
        row["normalized_gain"] = row["gain"] / row["mean_is_weight"]  # self-normalised IS: undo the weight scale
        row["clipfrac_after"] = after["clipfrac"]
    return row


def run_history(folder, out, store, lab):
    fresh, aged = hybrids.load(folder, lab["fresh_tag"]), hybrids.load(folder, lab["aged_tag"])
    cfg = aged["config"]
    history = f"{cfg['game']}_{cfg['arch']}_s{cfg['seed']}"
    critic = hybrids.model_from(fresh["model"], cfg)
    batches = [common_batch(r, critic, cfg) for r in fresh_batches(fresh, lab["batches"])]
    digests = [{k: hashlib.sha256(batch[k].numpy().tobytes()).hexdigest() for k in sorted(batch)} for batch in batches]
    results = {}
    for combo, (b, batch), kind in itertools.product(HYBRIDS, enumerate(batches), MEASURES):
        choice = dict(zip(hybrids.PARTS, ("fresh" if c == "F" else "aged" for c in combo)))
        model, optimizer, _ = hybrids.assemble(fresh, aged, choice)
        order = minibatch_order(cfg, len(batch["actions"]), (history, b))
        run_id = f"lab_{history}_{combo}_b{b}_{kind}"
        telemetry = Telemetry(out, run_id, model.layers(), optimizer, store=store, every=1,
                              meta={"kind": "lab", "history": history, "hybrid": combo, "batch": b,
                                    "measure": kind, "config_hash": config_hash(cfg), "lab": lab})
        results.setdefault(combo, {}).setdefault(kind, []).append(
            measure(model, optimizer, batch, order, kind, cfg, lab, telemetry))
    return history, results, digests


def contrasts(rows, key):
    """Per batch: params (aged - fresh, averaged over Adam) and Adam (aged - fresh, averaged over params)."""
    value = lambda combo, b: rows[combo][b][key]
    n = len(rows["FFFF"])
    params = [(value("AFFF", b) + value("AAFF", b) - value("FFFF", b) - value("FAFF", b)) / 2 for b in range(n)]
    adam = [(value("FAFF", b) + value("AAFF", b) - value("FFFF", b) - value("AFFF", b)) / 2 for b in range(n)]
    return {"params": params, "adam": adam}


METRICS = (("value", "relative_drop"), ("value", "relative_drop_variance"), ("value", "drop"),
           ("policy", "normalized_gain"), ("policy", "gain"))


def same_sign(values):
    return all(values) and len({v > 0 for v in values}) == 1


def summarize(per_history, lab):
    """Tables per history; the reliability rule of notes/ppo_lab.md over all batches of all histories."""
    summary = {}
    for history, results in per_history.items():
        out = {}
        for kind, key in METRICS:
            rows = {combo: results[combo][kind] for combo in HYBRIDS}
            table = {combo: {"mean": mean(r[key] for r in rows[combo]), "sd_over_batches": stdev(r[key] for r in rows[combo]),
                             "per_batch": [r[key] for r in rows[combo]]} for combo in HYBRIDS}
            out[f"{kind}_{key}"] = {"table": table, "contrasts": contrasts(rows, key)}
        out["policy_max_clipfrac_after"] = max(r["clipfrac_after"] for combo in HYBRIDS for r in results[combo]["policy"])
        summary[history] = out
    pooled = lambda metric, part: [v for h in summary.values() for v in h[metric]["contrasts"][part]]
    ceiling = max(h["policy_max_clipfrac_after"] for h in summary.values()) > lab["clipfrac_ceiling"]
    verdict = {}
    for part in ("params", "adam"):
        total, variance = same_sign(pooled("value_relative_drop", part)), same_sign(pooled("value_relative_drop_variance", part))
        verdict[f"value_{part}"] = ("reliable" if total and variance else
                                    "undetermined" if total else "unreliable")
        policy = same_sign(pooled("policy_normalized_gain", part))
        verdict[f"policy_{part}"] = "ceiling" if ceiling else ("reliable" if policy else "unreliable")
    return summary, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", default=str(ROOT / "runs" / "causal-trainability" / "ppo_pilot"))
    parser.add_argument("--histories", nargs="+", default=["breakout_cnn_s0", "breakout_cnn_s1"])
    parser.add_argument("--out", default=str(ROOT / "runs" / "causal-trainability" / "lab_t4"))
    parser.add_argument("--summary", default=str(ROOT / "research" / "results" / "ppo-lab" / "summary.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    revision = code_revision()
    if revision.get("dirty_causal_trainability"):
        raise SystemExit("causal_trainability has uncommitted changes; commit before a lab run")
    missing = [name for name in args.histories if not (Path(args.pilot) / name / "result.json").exists()]
    if missing:
        raise SystemExit(f"Unfinished pilot runs: {missing}")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; lab runs are immutable")
    out.mkdir(parents=True)
    store = TelemetryStore(out, "lab_t4")
    per_history, batch_sha256 = {}, {}
    for name in args.histories:
        history, results, batch_sha256[name] = run_history(Path(args.pilot) / name, out, store, LAB)
        per_history[history] = results
    store.close()
    (out / "per_history.json").write_text(json.dumps(per_history, indent=2), encoding="utf-8")
    summary, verdict = summarize(per_history, LAB)
    sources = {name: [json.loads(line) for line in (Path(args.pilot) / name / "checkpoints.jsonl").read_text(
        encoding="utf-8").splitlines()] for name in args.histories}
    result = {"lab": LAB, "hybrids": HYBRIDS, "sources": sources, "batch_sha256": batch_sha256, "revision": revision,
              "verdict": verdict, "histories": summary,
              "note": "Reliability check of the measurement (sprint 02, T4); not a repair outcome."}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict))
    for history, h in summary.items():
        for metric, m in h.items():
            if not isinstance(m, dict):
                continue
            print(history, metric, {c: round(v["mean"], 4) for c, v in m["table"].items()},
                  {k: [round(x, 4) for x in v] for k, v in m["contrasts"].items()})


if __name__ == "__main__":
    main()
