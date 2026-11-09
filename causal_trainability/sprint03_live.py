"""Small S3/S4 gate; protocol: notes/sprint03_live.md. No learned selector."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean, stdev

import torch

from causal_trainability import deficit, hybrids
from causal_trainability.ppo_minatar import evaluate, reference_batch
from causal_trainability.telemetry import ROOT, code_revision, record_artifact

MAIN_ARMS = ("continue", "adam_reset", "heads_reset")
CONTROL_ARMS = ("fresh", "heads_fresh_adam", "inverse")
SETTINGS = dict(steps=500_000, windows=10, eval_episodes=100, round=5, telemetry_every=16)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def intervene(ppo, initial, arm, old_perm, new_perm):
    """Parameter names determine Adam ownership; permute only action-shaped moments."""
    if arm == "fresh":
        ppo.model.load_state_dict(initial)
        ppo.optimizer.state.clear()
    elif arm == "adam_reset":
        ppo.optimizer.state.clear()
    elif arm in ("heads_reset", "heads_fresh_adam"):
        with torch.no_grad():
            for name, parameter in ppo.model.named_parameters():
                if name.startswith(("actor.", "critic.")):
                    parameter.copy_(initial[name])
                    ppo.optimizer.state.pop(parameter, None)
        if arm == "heads_fresh_adam":
            ppo.optimizer.state.clear()
    elif arm == "inverse":
        order = torch.tensor([old_perm.index(a) for a in new_perm])
        with torch.no_grad():
            for parameter in ppo.model.actor.parameters():
                parameter.copy_(parameter.index_select(0, order))
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = ppo.optimizer.state.get(parameter, {}).get(key)
                    if value is not None:
                        value.copy_(value.index_select(0, order))
    elif arm != "continue":
        raise ValueError(arm)


def start_metrics(ppo, initial):
    with torch.no_grad():
        reference = reference_batch(ppo.cfg)
        logits, values = ppo.model(reference)
        logp = logits.log_softmax(-1)
        fresh_model = copy.deepcopy(ppo.model)
        fresh_model.load_state_dict(initial)
        fresh_logits, fresh_values = fresh_model(reference)
        fresh_logp = fresh_logits.log_softmax(-1)
        return {"entropy": float(-(logp.exp() * logp).sum(-1).mean()),
                "value_rms": float(values.square().mean().sqrt()),
                "value_mean": float(values.mean()),
                "policy_tv_vs_fresh": float((logp.exp() - fresh_logp.exp()).abs().sum(-1).mean() / 2),
                "entropy_delta_vs_fresh": float((-(logp.exp()*logp).sum(-1) + (fresh_logp.exp()*fresh_logp).sum(-1)).mean()),
                "value_difference_rms_vs_fresh": float((values-fresh_values).square().mean().sqrt())}


def branch(source, chain_dir, arm, repeat, out, settings=None):
    settings = dict(SETTINGS if settings is None else settings)
    source, out = Path(source), Path(out)
    fresh, aged = hybrids.load(source, "age0.00"), hybrids.load(source, "age1.00")
    state, chain = deficit.chain_end(chain_dir)
    history = deficit.history_of(aged["config"])
    key = (history, "actions", settings["round"], repeat)
    ppo, cfg, seeds = deficit.prepared(aged["config"], state, state["optimizer"], "actions",
        settings["round"], settings["steps"], key, "deficit", out,
        "sprint03_live", f"s03_{history}_{arm}_rep{repeat}",
        telemetry_every=settings["telemetry_every"],
        checkpoint_fractions=[0.0], snapshot_fractions=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    intervene(ppo, fresh["theta0"], arm, chain["rounds"][-1]["perm"], cfg["shift"]["perm"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"theta0": ppo.theta0}, out / "theta0.pt")
    record_artifact(out, out / "theta0.pt", "theta0")
    eval_seeds = [deficit.seed_of("s03-eval", history, i) for i in range(settings["eval_episodes"])]
    policy_seed = deficit.seed_of("s03-eval-policy", history, repeat)
    policy = lambda: torch.Generator().manual_seed(policy_seed)
    before = start_metrics(ppo, fresh["theta0"])
    before["return"] = evaluate(ppo.model, cfg, eval_seeds, policy())
    seconds = ppo.train()
    area, windows = deficit.auc(ppo.curve, ppo.global_step, settings["windows"])
    if any(w is None for w in windows):
        raise ValueError("Empty outcome window; retain artifacts, do not silently drop it")
    result = {"schema_version": 1, "history": history, "arm": arm, "repeat": repeat,
        "settings": settings, "config": cfg, "revision": code_revision(),
        "source": {tag: hybrids.checkpoint_row(source, tag) for tag in ("age0.00", "age1.00")},
        "chain_sha256": chain["end"]["sha256"],
        "seeds": dict(seeds, evaluation=eval_seeds, evaluation_policy=policy_seed),
        "start": before, "auc": area, "windows": windows, "late_auc": mean(windows[5:]),
        "final_return": evaluate(ppo.model, cfg, eval_seeds, policy()),
        "actual_steps": ppo.global_step, "wall_time_sec": seconds,
        "curve": ppo.curve, "artifacts_sha256": digest(out / "artifacts.jsonl")}
    atomic_json(out / "result.json", result)
    return result


def jobs():
    return [(f"breakout_cnn_s{s}", arm, rep) for s in range(6)
            for arm in MAIN_ARMS + CONTROL_ARMS for rep in range(4 if arm in MAIN_ARMS else 2)]


def identity():
    files = list((ROOT / "causal_trainability").glob("*.py"))
    files += [ROOT / "notes/sprint03_live.md"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def queue(out, workers):
    if workers not in (1, 2):
        raise ValueError("At most two training processes")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    current = {"schema_version": 1, "code": identity(), "settings": SETTINGS, "jobs": jobs()}
    # JSON normalises tuples so a restart compares equivalent values.
    current = json.loads(json.dumps(current))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != current:
            raise ValueError("Frozen manifest mismatch; new code requires new output_root")
    else:
        atomic_json(manifest_path, current)

    failure = threading.Event()
    def one_unchecked(job):
        if failure.is_set() or (out / "STOP").exists():
            return
        if identity() != current["code"]:
            raise ValueError("Code changed during campaign")
        h, arm, rep = job
        cell = out / h / f"{arm}_rep{rep}"
        final = cell / "result.json"
        if final.exists():
            verified_result(final, h, arm, rep, current)
            return
        cell.mkdir(parents=True, exist_ok=True)
        attempts = len(list(cell.glob("attempt_*")))
        attempt = cell / f"attempt_{attempts:03d}"
        command = [sys.executable, "-m", "causal_trainability.sprint03_live", "--history", h,
                   "--arm", arm, "--repeat", str(rep), "--out", str(attempt)]
        print(f"START {h} {arm} {rep}", flush=True)
        with (cell / f"attempt_{attempts:03d}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
        result["attempt"] = attempt.name
        result["attempt_result_sha256"] = digest(attempt / "result.json")
        atomic_json(final, result)
        print(f"DONE {h} {arm} {rep}", flush=True)
    def one(job):
        try:
            return one_unchecked(job)
        except BaseException:
            failure.set()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(one, jobs()):
            pass


def interval(values, t=2.570581835636314):
    m = mean(values)
    half = t * stdev(values) / math.sqrt(len(values))
    return [m - half, m + half]


def verified_result(path, history, arm, repeat, manifest):
    r = json.loads(path.read_text(encoding="utf-8"))
    attempt = path.parent / r["attempt"]
    if attempt.parent.resolve() != path.parent.resolve():
        raise ValueError("Attempt escapes cell")
    if digest(attempt / "result.json") != r["attempt_result_sha256"]:
        raise ValueError("Attempt was changed")
    original = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    if original != {k: v for k, v in r.items() if k not in ("attempt", "attempt_result_sha256")}:
        raise ValueError("Wrapper payload mismatch")
    if (r["history"], r["arm"], r["repeat"], r["settings"]) != (history, arm, repeat, manifest["settings"]):
        raise ValueError("Result identity/settings mismatch")
    if r["revision"]["files"] != {k: v for k, v in manifest["code"].items() if k.startswith("causal_trainability/")}:
        raise ValueError("Revision files differ from frozen manifest")
    if digest(attempt / "artifacts.jsonl") != r["artifacts_sha256"]:
        raise ValueError("Artifact ledger changed")
    for line in (attempt / "artifacts.jsonl").read_text(encoding="utf-8").splitlines():
        artifact = json.loads(line)
        if digest(attempt / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Artifact content changed")
    expected_steps = r["settings"]["steps"] // (r["config"]["num_envs"] * r["config"]["num_steps"]) * (r["config"]["num_envs"] * r["config"]["num_steps"])
    if r["actual_steps"] != expected_steps or r["config"]["total_steps"] != r["settings"]["steps"]:
        raise ValueError("Budget mismatch")
    area, windows = deficit.auc(r["curve"], expected_steps, r["settings"]["windows"])
    if any(v is None or not math.isfinite(v) for v in windows) or windows != r["windows"] or area != r["auc"] or mean(windows[5:]) != r["late_auc"]:
        raise ValueError("Outcomes differ from complete raw curve")
    for tag, checkpoint in r["source"].items():
        source = ROOT / "runs/causal-trainability/ppo_pilot" / history
        if hybrids.checkpoint_row(source, tag)["sha256"] != checkpoint["sha256"]:
            raise ValueError("Source identity changed")
    return r


def summarize(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["code"] != identity() or manifest["settings"] != SETTINGS or manifest["jobs"] != json.loads(json.dumps(jobs())):
        raise ValueError("Frozen manifest mismatch")
    rows = {}
    hashes = {}
    for h, arm, rep in jobs():
        path = out / h / f"{arm}_rep{rep}" / "result.json"
        r = verified_result(path, h, arm, rep, manifest)
        rows[h, arm, rep] = r
        hashes[str(path.relative_to(out))] = digest(path)
    hs = sorted({h for h, _, _ in rows})
    def avg(h, a, reps, metric):
        return mean(rows[h, a, r][metric] for r in reps)
    effects = {}
    for arm in MAIN_ARMS[1:]:
        groups = [[rows[h, arm, r]["auc"] - rows[h, "continue", r]["auc"] for r in range(4)] for h in hs]
        values = [mean(g) for g in groups]
        noise = math.sqrt(mean(stdev(g)**2 for g in groups))
        ci = interval(values, 3.163381449748609)  # two contrasts, Bonferroni 97.5%
        stable = (ci[0] > 0 and all(v > 0 for v in values) or ci[1] < 0 and all(v < 0 for v in values)) and abs(mean(values)) > noise
        effects[arm] = {"mean": mean(values), "ci975": ci, "per_history": values,
                        "paired_within_history_sd": noise, "stable_difference": stable}
    choices = {h: max(MAIN_ARMS, key=lambda a: avg(h, a, (0, 1), "auc")) for h in hs}
    constant = max(MAIN_ARMS, key=lambda a: mean(avg(h, a, (0, 1), "auc") for h in hs))
    gains = [avg(h, choices[h], (2, 3), "auc") - avg(h, constant, (2, 3), "auc") for h in hs]
    contrasts = {}
    for arm in ("heads_reset", "heads_fresh_adam", "inverse"):
        contrasts[arm] = {}
        for metric in ("auc", "late_auc", "final_return"):
            values = [avg(h, arm, (0, 1), metric) - avg(h, "fresh", (0, 1), metric) for h in hs]
            contrasts[arm][metric] = {"mean": mean(values), "ci95": interval(values), "per_history": values}
    comparability = {}
    for arm in ("heads_reset", "heads_fresh_adam"):
        start_diff = [mean(rows[h, arm, r]["start"]["return"] - rows[h, "fresh", r]["start"]["return"] for r in (0, 1)) for h in hs]
        ci90 = interval(start_diff, 2.0150483733330233)
        reference_ok = all(rows[h, arm, r]["start"]["policy_tv_vs_fresh"] <= .02
            and abs(rows[h, arm, r]["start"]["entropy_delta_vs_fresh"]) <= .02
            and rows[h, arm, r]["start"]["value_difference_rms_vs_fresh"] <= .1
            for h in hs for r in (0, 1))
        comparability[arm] = {"return_difference_ci90": ci90, "reference_gate": reference_ok,
                             "comparable": reference_ok and ci90[0] >= -.1 and ci90[1] <= .1}
        groups = [[rows[h, arm, r]["late_auc"]-rows[h, "fresh", r]["late_auc"] for r in (0, 1)] for h in hs]
        values = [mean(g) for g in groups]
        ci = interval(values, 3.163381449748609)
        noise = math.sqrt(mean(stdev(g)**2 for g in groups))
        comparability[arm]["late_effect_ci975"] = ci
        comparability[arm]["late_paired_within_history_sd"] = noise
        comparability[arm]["slower_at_comparable_start"] = comparability[arm]["comparable"] and ci[1] < 0 and sum(v < 0 for v in values) >= 5 and -mean(values) > noise
    summary = {"schema_version": 1, "branches": len(rows), "result_hashes": hashes,
        "s4": {"effects": effects, "checkpoint_choices": choices, "training_selected_constant": constant,
               "heldout_gain": mean(gains), "ci95_heldout_gain": interval(gains), "per_history_gain": gains,
               "selection_supported": interval(gains)[0] > 0 and any(e["stable_difference"] for e in effects.values())},
        "s3": {"contrasts_vs_fresh": contrasts, "comparability": comparability,
               "starts": {h: {a: [rows[h, a, r]["start"] for r in (0, 1)] for a in MAIN_ARMS + CONTROL_ARMS} for h in hs}}}
    atomic_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--history")
    parser.add_argument("--arm", choices=MAIN_ARMS + CONTROL_ARMS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.queue:
        queue(args.out, args.workers)
    elif args.summarize:
        summarize(args.out)
    else:
        source = ROOT / "runs/causal-trainability/ppo_pilot" / args.history
        chain = ROOT / "runs/causal-trainability/deficit_B" / args.history / "actions_chain"
        branch(source, chain, args.arm, args.repeat, args.out)


if __name__ == "__main__":
    main()
