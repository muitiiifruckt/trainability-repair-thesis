"""A one-iteration matched-policy pulse, paired with verified Sprint 06 prefixes."""
import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, median, stdev

import torch

from causal_trainability import behavior_match as match
from causal_trainability.ppo_minatar import PPO, reference_batch
from causal_trainability.sprint03_live import atomic_json, digest
from causal_trainability.sprint04_pilot import fingerprint
from causal_trainability.telemetry import ROOT, Telemetry, code_revision, record_artifact

HORIZON = 32768
WINDOW = 8192
CAL = ROOT / "runs/causal-trainability/sprint07_pulse_cal_v1"
LIVE = ROOT / "runs/causal-trainability/sprint07_pulse_live_v1"
EXPORT = ROOT / "research/results/sprint06-match"
PROTOCOL = ROOT / "notes/sprint07_pulse_prereg.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity():
    revision = code_revision()
    runtime = {k: revision[k] for k in ("python", "torch", "numpy", "MinAtar")}
    return dict(files=revision["files"], runtime=runtime, script_sha256=digest(Path(__file__)),
                protocol_sha256=digest(PROTOCOL), old_manifest_sha256=digest(EXPORT / "manifest.json"))


def weights(ppo):
    return dict(model=ppo.model.state_dict(), optimizer=ppo.optimizer.state_dict(),
                generator=ppo.generator.get_state())


def prefix(curve):
    rows = [row for row in curve if row["env_steps"] <= HORIZON]
    assert [r["env_steps"] for r in rows] == list(range(1024, HORIZON + 1, 1024))
    assert all(math.isfinite(v) for row in rows for v in row["returns"])
    windows = [mean(v for row in rows if i * WINDOW < row["env_steps"] <= (i + 1) * WINDOW
                    for v in row["returns"]) for i in range(4)]
    return dict(windows=windows, score=mean(windows[1:]))


def control(history, repeat, arm, hashes):
    folder = match.OUT / history / f"{arm}_{repeat}"
    wrapper = read(folder / "result.json")
    attempt = folder / wrapper["attempt"]
    path = attempt / "result.json"
    assert digest(path) == wrapper["sha256"] == hashes[str(path.relative_to(match.OUT))]
    row = read(path)
    assert (row["history"], row["repeat"], row["arm"]) == (history, repeat, arm)
    current = identity()
    assert row["revision"]["files"] == current["files"]
    assert all(row["revision"][k] == v for k, v in current["runtime"].items())
    artifacts_path = attempt / "artifacts.jsonl"
    assert digest(artifacts_path) == row["artifacts_sha256"]
    artifacts = [json.loads(line) for line in artifacts_path.read_text().splitlines()]
    snapshot, = [a for a in artifacts if a["kind"] == "snapshot" and a["path"].endswith("_u0.pt")]
    snapshot_path = attempt / snapshot["path"]
    assert digest(snapshot_path) == snapshot["sha256"]
    start = torch.load(snapshot_path, weights_only=True, map_location="cpu")
    entry, = [json.loads(line) for line in (attempt / "checkpoints.jsonl").read_text().splitlines()
              if json.loads(line)["tag"] == "age0.00"]
    checkpoint = attempt / entry["path"]
    assert digest(checkpoint) == entry["sha256"]
    batch_state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert batch_state["config"] == row["config"]
    assert batch_state["iteration"] == 1 and batch_state["global_step"] == 1024
    assert batch_state["optimizer"]["state"] == {}
    for key in ("model", "optimizer"):
        assert fingerprint(batch_state[key]) == fingerprint(start[key])
    batch = batch_state["rollout"]
    assert match.behavior_hash(batch) == row["first"]["behavior_sha256"]
    assert fingerprint(batch_state["generator"]) == row["first"]["generator_sha256"]
    assert fingerprint([batch[k] for k in ("values", "advantages", "returns")]) == row["first"]["targets_sha256"]
    origin = dict(result_sha256=wrapper["sha256"], checkpoint_sha256=entry["sha256"],
                  anchored_snapshot_sha256=snapshot["sha256"])
    return row, batch_state, origin


def run(history, repeat, checkpoint, expected_seeds, out, enabled, steps=HORIZON):
    out.mkdir(parents=True, exist_ok=False)
    ppo, cfg, seeds, fresh, source = match.prepared(history, "matched", repeat, out, steps, False)
    assert seeds == {k: expected_seeds[k] for k in seeds}
    excluded = {"total_steps", "snapshot_fractions", "telemetry_every", "run_name"}
    assert {k: v for k, v in cfg.items() if k not in excluded} == {
        k: v for k, v in checkpoint["config"].items() if k not in excluded}
    assert fingerprint(ppo.model.state_dict()) == fingerprint(checkpoint["model"])
    assert fingerprint(ppo.optimizer.state_dict()) == fingerprint(checkpoint["optimizer"])
    cfg.update(snapshot_fractions=[0., .5, 1.], telemetry_every=16, run_name="pulse",
               pulse_switch_iteration=1)
    torch.save({"theta0": ppo.theta0}, out / "theta0.pt")
    record_artifact(out, out / "theta0.pt", "theta0")
    global_rng = torch.get_rng_state().clone()
    anchors = match.anchor_hash(ppo.model)
    reference = reference_batch(cfg)
    before = match.start_metrics(ppo, fresh["theta0"])
    original_rollout, original_learn = ppo.rollout, ppo.learn
    switch = {}

    def rollout():
        assert ppo.model.arm == ("matched" if ppo.iteration == 1 else "neither")
        batch = original_rollout()
        if ppo.iteration == 1:
            assert fingerprint(batch) == fingerprint(checkpoint["rollout"])
            assert torch.equal(ppo.generator.get_state(), checkpoint["generator"])
        return batch

    def learn(batch):
        stats = original_learn(batch)
        if ppo.iteration == 1:
            switch["first_update_sha256"] = fingerprint(weights(ppo))
            with torch.no_grad():
                corrected = ppo.model(reference)[0].softmax(-1)
                ppo.model.arm = "neither"
                cfg["behavior_arm"] = ppo.model.arm
                raw = ppo.model(reference)[0].softmax(-1)
            switch["policy_tv"] = float((corrected - raw).abs().sum(-1).mean() / 2)
        return stats

    ppo.rollout, ppo.learn = rollout, learn
    started = time.perf_counter()
    initial = Telemetry(out, "initial", ppo.model.layers(), ppo.optimizer,
                        meta={"kind": "s07_pulse_initial"}, enabled=enabled)
    initial.step(0, force=True, metrics=before)
    initial.close()
    ppo.train(telemetry_enabled=enabled)
    elapsed = time.perf_counter() - started
    assert ppo.global_step == steps and ppo.model.arm == "neither"
    assert torch.equal(global_rng, torch.get_rng_state())
    assert match.anchor_hash(ppo.model) == anchors
    assert all(p.grad is None and p not in ppo.optimizer.state for p in ppo.model.anchor.parameters())
    final = fingerprint(weights(ppo))
    torch.save(weights(ppo), out / "final.pt")
    record_artifact(out, out / "final.pt", "final_state")
    last = torch.load(out / "checkpoints/age1.00.pt", weights_only=False, map_location="cpu")
    assert last["config"]["behavior_arm"] == "neither" and last["config"]["pulse_switch_iteration"] == 1
    with torch.random.fork_rng(devices=[]):
        replay = PPO(dict(last["config"]), out / "resume")
    match.wrap(replay, fresh["theta0"], last["config"]["behavior_arm"])
    replay.restore(last)
    replay.learn(replay.last_rollout)
    assert fingerprint(weights(replay)) == final
    result = dict(history=history, repeat=repeat, config=cfg, source=source, seeds=seeds,
                  start=before, switch=switch, final_sha256=final, seconds=elapsed, curve=ppo.curve,
                  artifacts_sha256=digest(out / "artifacts.jsonl"))
    atomic_json(out / "result.json", result)
    return result


def verify(out):
    row = read(out / "result.json")
    assert digest(out / "artifacts.jsonl") == row["artifacts_sha256"]
    for index in ("artifacts.jsonl", "checkpoints.jsonl"):
        for line in (out / index).read_text().splitlines():
            item = json.loads(line)
            assert digest(out / item["path"]) == item["sha256"]
    assert fingerprint(torch.load(out / "final.pt", weights_only=False, map_location="cpu")) == row["final_sha256"]
    return row


def calibrate(hashes):
    CAL.mkdir(parents=True, exist_ok=False)
    history = "breakout_cnn_s0"
    row, checkpoint, origin = control(history, 0, "matched", hashes)
    fresh, _, _ = match.inputs(history)
    with torch.random.fork_rng(devices=[]):
        replay = PPO(dict(checkpoint["config"]), CAL / "initial_replay")
    match.wrap(replay, fresh["theta0"], "matched")
    replay.restore(checkpoint)
    replay.learn(replay.last_rollout)
    expected = fingerprint(weights(replay))
    for enabled in (False, True):
        folder = CAL / f"warm_{enabled}"
        run(history, 0, checkpoint, row["seeds"], folder, enabled, 4096)
        assert verify(folder)["switch"]["first_update_sha256"] == expected
    pairs = []
    for i in range(3):
        order = (True, False) if i % 2 == 0 else (False, True)
        runs = {}
        for enabled in order:
            folder = CAL / f"pair{i}_{enabled}"
            run(history, 0, checkpoint, row["seeds"], folder, enabled)
            runs[enabled] = verify(folder)
        on, off = runs[True], runs[False]
        assert on["final_sha256"] == off["final_sha256"]
        assert on["switch"] == off["switch"] and on["switch"]["first_update_sha256"] == expected
        assert on["curve"] == off["curve"]
        pairs.append(dict(on=on["seconds"], off=off["seconds"],
                          overhead=on["seconds"] / off["seconds"] - 1, bit_equal=True))
        print("calibration", i + 1, pairs[-1], flush=True)
    overhead = [r["overhead"] for r in pairs]
    result = dict(identity=identity(), source=origin, revision=code_revision(), pairs=pairs,
                  mean_overhead=mean(overhead), median_overhead=median(overhead),
                  first_update_sha256=expected, passed=mean(overhead) <= .05 and median(overhead) <= .05)
    atomic_json(CAL / "summary.json", result)
    print("PASS", result["passed"], flush=True)


def science(hashes):
    gate = read(CAL / "summary.json")
    assert gate["passed"] and gate["identity"] == identity()
    LIVE.mkdir(parents=True, exist_ok=False)
    records = []
    for h in range(6):
        history = f"breakout_cnn_s{h}"
        for repeat in (0, 1):
            rows, checkpoints, origins = {}, {}, {}
            for arm in match.ARMS:
                rows[arm], checkpoints[arm], origins[arm] = control(history, repeat, arm, hashes)
            assert all(rows[arm]["seeds"] == rows["matched"]["seeds"] for arm in match.ARMS)
            assert all(rows[arm]["source"] == rows["matched"]["source"] for arm in match.ARMS)
            ignored = ("run_name", "behavior_arm")
            configs = [{k: v for k, v in row["config"].items() if k not in ignored} for row in rows.values()]
            assert configs[0] == configs[1] == configs[2]
            match.check_pair(rows, True)
            matched = checkpoints["matched"]
            assert fingerprint(matched["model"]) == fingerprint(checkpoints["neither"]["model"])
            assert fingerprint(matched["rollout"]) == fingerprint(checkpoints["reset"]["rollout"])
            folder = LIVE / history / f"repeat{repeat}"
            run(history, repeat, matched, rows["matched"]["seeds"], folder, True)
            pulse = verify(folder)
            assert pulse["source"] == rows["matched"]["source"]
            assert pulse["curve"][0] == rows["matched"]["curve"][0]
            scores = {arm: prefix(row["curve"]) for arm, row in rows.items()}
            scores["pulse"] = prefix(pulse["curve"])
            records.append(dict(history=history, repeat=repeat, scores=scores, switch=pulse["switch"],
                                control_sources=origins, pulse_sha256=digest(folder / "result.json")))
            print("science", history, repeat, flush=True)
    paired = [[r["scores"]["pulse"]["score"] - r["scores"]["neither"]["score"] for r in records
               if r["history"] == f"breakout_cnn_s{h}"] for h in range(6)]
    values = [mean(g) for g in paired]
    effect = mean(values)
    half = 2.570581835636314 * stdev(values) / math.sqrt(6)
    lower, upper = effect - half, effect + half
    noise = math.sqrt(mean(stdev(g) ** 2 for g in paired))
    signs = sum(v > 0 for v in values)
    verdict = "beneficial" if lower > .1 and signs >= 5 and effect > noise else (
        "practical_benefit_not_shown" if upper < .1 else "uncertain")
    summary = dict(identity=identity(), revision=code_revision(), calibration_sha256=digest(CAL / "summary.json"),
                   primary=dict(mean=effect, ci95=[lower, upper], per_history=values, noise=noise,
                                positive_histories=signs, verdict=verdict),
                   means={arm: mean(r["scores"][arm]["score"] for r in records)
                          for arm in (*match.ARMS, "pulse")}, records=records)
    atomic_json(LIVE / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["calibrate", "science"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert not code_revision()["dirty_causal_trainability"]
    manifest = read(EXPORT / "manifest.json")
    assert digest(EXPORT / "summary.json") == manifest["exported_sha256"]["summary.json"]
    assert digest(match.OUT / "summary.json") == manifest["full_summary_sha256"]
    hashes = read(EXPORT / "summary.json")["inputs_sha256"]
    {"calibrate": calibrate, "science": science}[args.mode](hashes)


if __name__ == "__main__":
    main()
