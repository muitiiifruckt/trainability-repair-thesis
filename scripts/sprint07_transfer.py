"""Held-out histories and a sticky-action shift for the actor-reset comparison."""
import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import torch

from causal_trainability import hybrids
from causal_trainability.head_reset_factorial import reset_heads
from causal_trainability.ppo_minatar import CONFIG, PPO, VecMinAtar, evaluate, reference_batch, resolve, seed_of
from causal_trainability.sprint03_live import atomic_json, digest
from causal_trainability.sprint04_pilot import fingerprint
from causal_trainability.telemetry import ROOT, Telemetry, code_revision, config_hash, record_artifact

ROOT_OUT = ROOT / "runs/causal-trainability/sprint07_transfer_v1"
CAL = ROOT_OUT / "calibration"
PROTOCOL = ROOT / "notes/sprint07_transfer_prereg.md"
SEEDS = tuple(range(100, 106))
ARMS = ("keep", "reset")
SOURCE_STEPS = 999424
BRANCH_STEPS = 131072


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity():
    revision = code_revision()
    runtime = {k: revision[k] for k in ("python", "torch", "numpy", "MinAtar")}
    return dict(script=digest(Path(__file__)), protocol=digest(PROTOCOL),
                files=revision["files"], runtime=runtime)


def state(ppo):
    return dict(model=ppo.model.state_dict(), optimizer=ppo.optimizer.state_dict(),
                generator=ppo.generator.get_state())


def replay_last(ppo, checkpoint):
    ppo.restore(checkpoint)
    ppo.learn(ppo.last_rollout)
    return state(ppo)


def archive():
    root = ROOT / "runs/causal-trainability/ppo_pilot/breakout_cnn_s0"
    saved = hybrids.load(root, "age1.00")
    with torch.random.fork_rng(devices=[]):
        ppo = PPO(saved["config"], CAL / "archive")
    model = replay_last(ppo, saved)["model"]
    return dict(model=model, theta0=saved["theta0"]), hybrids.checkpoint_row(root, "age1.00")["sha256"]


def make_learner(out, seed, repeat, arm, steps, source):
    cfg = resolve(dict(CONFIG, seed=seed, total_steps=steps, sticky_action_prob=.10 if arm == "source" else .25,
                       telemetry_every=64, snapshot_fractions=[0., .5, 1.], checkpoint_fractions=[0., 1.],
                       run_name=arm, run_kind="s07_transfer"))
    with torch.random.fork_rng(devices=[]):
        ppo = PPO(cfg, out)
    if arm == "source":
        seeds = {k: seed_of(k, cfg["game"], cfg["arch"], seed) for k in ("init", "ppo")}
        seeds["env"] = seed
    else:
        ppo.model.load_state_dict(source["model"])
        ppo.theta0 = source["theta0"]
        reset_heads(ppo.model, ppo.theta0, "actor" if arm == "reset" else "neither")
        seeds = {k: seed_of(f"s07-transfer-{k}", seed, repeat) for k in ("env", "ppo")}
        ppo.envs = VecMinAtar(cfg, seeds["env"])
        ppo.obs = torch.tensor(ppo.envs.observations())
        ppo.generator.manual_seed(seeds["ppo"])
        for name, tensor in ppo.model.state_dict().items():
            expected = ppo.theta0 if arm == "reset" and name.startswith("actor.") else source["model"]
            assert torch.equal(tensor, expected[name]), name
    assert ppo.envs.cfg["sticky_action_prob"] == cfg["sticky_action_prob"]
    assert not cfg.get("shift") and not ppo.optimizer.state and ppo.global_step == 0
    return ppo, seeds


def score(curve):
    assert [r["env_steps"] for r in curve] == list(range(1024, BRANCH_STEPS + 1, 1024))
    windows = []
    for i in range(4):
        rewards = [v for r in curve if i * 32768 < r["env_steps"] <= (i + 1) * 32768 for v in r["returns"]]
        assert rewards and all(math.isfinite(v) for v in rewards)
        windows.append(mean(rewards))
    return dict(windows=windows, auc=mean(windows))


def run(out, seed, repeat, arm, steps, enabled=True, source=None, origin=None, scientific=False):
    out.mkdir(parents=True, exist_ok=False)
    ppo, seeds = make_learner(out, seed, repeat, arm, steps, source)
    cfg = ppo.cfg
    initial = dict(weights_sha256=fingerprint(state(ppo)), obs_sha256=fingerprint(ppo.obs),
                   non_actor_sha256=fingerprint({k: v for k, v in ppo.model.state_dict().items()
                                                if not k.startswith("actor.")}),
                   rng_sha256=fingerprint(ppo.generator.get_state()))
    rng = torch.get_rng_state().clone()
    torch.save({"theta0": ppo.theta0}, out / "theta0.pt")
    record_artifact(out, out / "theta0.pt", "theta0")
    evaluation = {}
    if scientific and arm != "source":
        seeds["eval_env"] = [seed_of("s07-transfer-eval", seed, i) for i in range(30)]
        seeds["eval_policy"] = seed_of("s07-transfer-eval-policy", seed, repeat)
        evaluation["before"] = evaluate(ppo.model, cfg, seeds["eval_env"],
                                        torch.Generator().manual_seed(seeds["eval_policy"]))
        with torch.no_grad():
            logp = ppo.model(reference_batch(cfg))[0].log_softmax(-1)
            evaluation["initial_entropy"] = float(-(logp.exp() * logp).sum(-1).mean())
    started = perf_counter()
    telemetry = Telemetry(out, "initial", ppo.model.layers(), ppo.optimizer, every=64,
                          meta=dict(kind="s07_initial", config_hash=config_hash(cfg)), enabled=enabled)
    telemetry.step(0, force=True)
    telemetry.close()
    ppo.train(telemetry_enabled=enabled)
    if enabled:
        record_artifact(out, out / "runs_index.jsonl", "runs_index")
    elapsed = perf_counter() - started
    assert ppo.global_step == steps and torch.equal(rng, torch.get_rng_state())
    final = dict(**state(ppo), theta0=ppo.theta0)
    torch.save(final, out / "final.pt")
    record_artifact(out, out / "final.pt", "final_state")
    checkpoint = hybrids.load(out, "age1.00")
    with torch.random.fork_rng(devices=[]):
        resumed = PPO(cfg, out / "resume")
    assert fingerprint(replay_last(resumed, checkpoint)) == fingerprint(state(ppo))
    if scientific and arm != "source":
        evaluation["after"] = evaluate(ppo.model, cfg, seeds["eval_env"],
                                       torch.Generator().manual_seed(seeds["eval_policy"]))
    assert torch.equal(rng, torch.get_rng_state())
    row = dict(identity=identity(), revision=code_revision(), arm=arm, seed=seed, repeat=repeat,
               config=cfg, config_sha256=config_hash(cfg), seeds=seeds, initial=initial, source_sha256=origin,
               steps=steps, resume_bit_equal=True, final_sha256=digest(out / "final.pt"),
               learner_fingerprint=fingerprint(final), rollout_fingerprint=fingerprint(ppo.last_rollout),
               seconds=elapsed, logging=enabled, evaluation=evaluation, curve=ppo.curve,
               artifacts_sha256=digest(out / "artifacts.jsonl"),
               checkpoint_index_sha256=digest(out / "checkpoints.jsonl"))
    if scientific and arm != "source":
        row.update(score(ppo.curve))
    atomic_json(out / "result.json", row)
    return row


def verify(out):
    row = read(out / "result.json")
    assert row["identity"] == identity() and row["resume_bit_equal"]
    assert row["config_sha256"] == config_hash(row["config"])
    assert digest(out / "final.pt") == row["final_sha256"]
    indexes = {}
    for index, key in (("artifacts.jsonl", "artifacts_sha256"), ("checkpoints.jsonl", "checkpoint_index_sha256")):
        path = out / index
        assert digest(path) == row[key]
        indexes[index] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for entry in indexes[index]:
            assert digest(out / entry["path"]) == entry["sha256"]
    artifacts = {item["path"] for item in indexes["artifacts.jsonl"]}
    assert {"theta0.pt", "final.pt"} <= artifacts
    iters = row["steps"] // 1024
    checkpoints = indexes["checkpoints.jsonl"]
    assert [(r["tag"], r["iteration"]) for r in checkpoints] == [("age0.00", 1), ("age1.00", iters)]
    if row["logging"]:
        arm = row["arm"]
        assert {f"snapshots/{arm}_u{i}.pt" for i in (0, round(iters / 2), iters)} <= artifacts
        assert "runs_index.jsonl" in artifacts
        for run_id, updates in (("initial", [0]), (arm, sorted(set(range(64, iters + 1, 64)) | {iters}))):
            filename = f"telemetry/{run_id}.jsonl"
            assert filename in artifacts
            rows = [json.loads(line) for line in (out / filename).read_text(encoding="utf-8").splitlines()]
            assert [r["update"] for r in rows] == updates
            assert all(set(r["layers"]) == {"actor", "critic", "conv", "fc"} for r in rows)
            if run_id != "initial":
                assert all("adam_step" in layer for r in rows for layer in r["layers"].values())
    if "auc" in row:
        assert score(row["curve"]) == dict(windows=row["windows"], auc=row["auc"])
    return row


def calibration():
    CAL.mkdir(parents=True, exist_ok=False)
    saved, origin = archive()
    cells = {}
    for arm in ("source", *ARMS):
        args = dict(source=saved, origin=origin) if arm != "source" else {}
        for log in (False, True):
            p = CAL / arm / f"warm_{log}"
            run(p, 0, 0, arm, 4096, log, **args)
            verify(p)
        pairs = []
        for i in range(3):
            rows = {}
            for log in ((True, False) if i % 2 == 0 else (False, True)):
                p = CAL / arm / f"pair{i}_{log}"
                run(p, 0, 0, arm, BRANCH_STEPS, log, **args)
                rows[log] = verify(p)
            a, b = rows[True], rows[False]
            for key in ("learner_fingerprint", "rollout_fingerprint", "curve", "initial"):
                assert a[key] == b[key]
            pairs.append(dict(on=a["seconds"], off=b["seconds"], overhead=a["seconds"] / b["seconds"] - 1,
                              bit_equal=True))
            print("CAL", arm, i + 1, pairs[-1]["overhead"], flush=True)
        times = [p["overhead"] for p in pairs]
        cells[arm] = dict(pairs=pairs, mean=mean(times), median=median(times))
    passed = all(cell["mean"] <= .05 and cell["median"] <= .05 for cell in cells.values())
    atomic_json(CAL / "summary.json", dict(passed=passed, cells=cells, identity=identity(), revision=code_revision()))
    print("PASS", passed, flush=True)


def gate():
    result = read(CAL / "summary.json")
    assert result["passed"] and result["identity"] == identity()
    assert set(result["cells"]) == {"source", *ARMS}
    for cell in result["cells"].values():
        pairs = cell["pairs"]
        assert len(pairs) == 3 and all(p["bit_equal"] for p in pairs)
        times = [p["on"] / p["off"] - 1 for p in pairs]
        assert cell["mean"] == mean(times) <= .05 and cell["median"] == median(times) <= .05
    return digest(CAL / "summary.json")


def job_folder(seed, arm, repeat):
    folder = ROOT_OUT / "science" / f"s{seed}"
    return folder / ("source" if arm == "source" else f"{arm}_{repeat}")


def sealed(seed, arm, repeat):
    folder = job_folder(seed, arm, repeat)
    seal = read(folder / "done.json")
    out = folder / seal["attempt"]
    assert digest(out / "result.json") == seal["sha256"]
    row = verify(out)
    assert (row["seed"], row["arm"], row["repeat"]) == (seed, arm, repeat)
    assert row["logging"] and row["steps"] == (SOURCE_STEPS if arm == "source" else BRANCH_STEPS)
    return out, row


def worker(seed, arm, repeat):
    gate()
    assert seed in SEEDS
    folder = job_folder(seed, arm, repeat)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "done.json").exists():
        sealed(seed, arm, repeat)
        return
    out = folder / f"attempt_{len(list(folder.glob('attempt_*'))):03}"
    if arm == "source":
        run(out, seed, repeat, arm, SOURCE_STEPS, scientific=True)
    else:
        parent, row = sealed(seed, "source", 0)
        weights = torch.load(parent / "final.pt", weights_only=True, map_location="cpu")
        run(out, seed, repeat, arm, BRANCH_STEPS, source=weights, origin=row["final_sha256"], scientific=True)
    verify(out)
    atomic_json(folder / "done.json", dict(attempt=out.name, sha256=digest(out / "result.json")))
    print("DONE", seed, arm, repeat, flush=True)


def summarize():
    records, effects, sources = [], [], []
    for seed in SEEDS:
        parent, source = sealed(seed, "source", 0)
        sources.append({k: source[k] for k in ("seed", "steps", "config", "seeds", "final_sha256")})
        differences = []
        for repeat in (0, 1):
            rows = {}
            for arm in ARMS:
                path, row = sealed(seed, arm, repeat)
                assert row["source_sha256"] == source["final_sha256"]
                rows[arm] = {k: v for k, v in row.items() if k not in ("curve", "revision", "identity")}
                rows[arm]["result_sha256"] = digest(path / "result.json")
            a, b = rows["reset"], rows["keep"]
            assert a["seeds"] == b["seeds"]
            for key in ("obs_sha256", "non_actor_sha256", "rng_sha256"):
                assert a["initial"][key] == b["initial"][key]
            difference = a["auc"] - b["auc"]
            differences.append(difference)
            records.append(dict(seed=seed, repeat=repeat, difference=difference, **rows))
        effects.append(mean(differences))
    effect = mean(effects)
    delta = 2.570581835636314 * stdev(effects) / math.sqrt(len(SEEDS))
    lower, upper = effect - delta, effect + delta
    verdict = "beneficial" if lower > .1 else ("practical_benefit_not_shown" if upper < .1 else "uncertain")
    result = dict(identity=identity(), revision=code_revision(), gate_sha256=gate(), sources=sources,
                  records=records, primary=dict(mean=effect, ci95=[lower, upper], per_history=effects, verdict=verdict),
                  means={arm: mean(r[arm]["auc"] for r in records) for arm in ARMS})
    atomic_json(ROOT_OUT / "summary.json", result)
    print(json.dumps(result["primary"]), flush=True)


def queue():
    calibration_hash = gate()
    manifest = ROOT_OUT / "dependency_manifest.json"
    expected = dict(identity=identity(), calibration_sha256=calibration_hash, seeds=list(SEEDS), arms=list(ARMS),
                    source_steps=SOURCE_STEPS, branch_steps=BRANCH_STEPS, repeats=[0, 1])
    if manifest.exists():
        assert read(manifest) == expected
    else:
        atomic_json(manifest, expected)

    def launch(spec):
        if (ROOT_OUT / "STOP").exists():
            return
        seed, arm, repeat = spec
        if (job_folder(seed, arm, repeat) / "done.json").exists():
            sealed(seed, arm, repeat)
            return
        print("START", seed, arm, repeat, flush=True)
        subprocess.run([sys.executable, "-m", "scripts.sprint07_transfer", "worker", "--seed", str(seed),
                        "--arm", arm, "--repeat", str(repeat)], cwd=ROOT, check=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(launch, [(seed, "source", 0) for seed in SEEDS]))
        if not (ROOT_OUT / "STOP").exists():
            list(executor.map(launch, [(seed, arm, rep) for seed in SEEDS for rep in (0, 1) for arm in ARMS]))
    if not (ROOT_OUT / "STOP").exists():
        summarize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibrate", "queue", "worker", "summarize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repeat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--arm", choices=("source", *ARMS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert not code_revision()["dirty_causal_trainability"]
    if args.mode == "worker":
        worker(args.seed, args.arm, args.repeat)
    else:
        {"calibrate": calibration, "queue": queue, "summarize": summarize}[args.mode]()


if __name__ == "__main__":
    main()
