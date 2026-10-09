"""[B2] Эксперименты по возобновлению/детерминизму на ТОЧНОЙ копии кода кампании (research/debate/tmp/pkg0, SHA rl_core/rl_runner/rl_scheduler
совпадают с experiments/). Ничего в experiments/ runs/ configs/ не пишется: временные кампании создаются в research/debate/tmp/pkg0/runs/b2_*.
Запуск ТОЛЬКО интерпретатором .venv (torch): .venv/Scripts/python.exe scripts/debate_b2_resume_check.py <det|torn|memerr|digest> [...]
  det     побитовая воспроизводимость: прямой прогон = прогон с остановкой, torch.save/torch.load, evaluate в середине; несколько окружений процесса
  torn    F1: падение между save_state(.pt) и dump(.json) savepoint ветки -> перезапуск с шага 0? результат тот же? wall_time_sec занижен?
  memerr  MemoryError в середине третьей ветки внутри _run_worker_task -> импорт родителем -> повторный запуск: потерь/дублей нет? == эталон?
1 поток, RAM < 1 GB."""
import sys, os, json, time, copy, hashlib, shutil, subprocess, tempfile
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve()
PKG = HERE.parents[1] / "research/debate/tmp/pkg0"
sys.path.insert(0, str(PKG))
import numpy as np
import torch
torch.set_num_threads(1)
if "--nodet" not in sys.argv:
    torch.use_deterministic_algorithms(True)
from unittest.mock import patch
from experiments import rl_runner as R, rl_scheduler as S
from experiments.rl_core import DQNConfig, TrainingState
assert Path(R.__file__).resolve().is_relative_to(PKG), R.__file__


def digest(state):
    h = hashlib.sha256()
    for sd in (state.q.state_dict(), state.target.state_dict()):
        for k, v in sd.items(): h.update(v.numpy().tobytes())
    for g in state.optimizer.state_dict()["state"].values():
        for v in g.values(): h.update(v.numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
    h.update(state.replay.obs[:state.replay.size].tobytes()); h.update(state.replay.rewards[:state.replay.size].tobytes())
    h.update(np.asarray(state.exploration_rng.get_state()[1]).tobytes()); h.update(np.asarray(state.replay_rng.get_state()[1]).tobytes())
    h.update(str((state.environment_steps, state.gradient_updates, state.last_target_update, state.episode_id)).encode())
    return h.hexdigest()[:16]


def real_cfg():
    return DQNConfig(replay_capacity=4096, warmup=200, target_period=100, epsilon_decay=2000)   # реальная сеть, float32


def run_digest(n=1200):
    st = TrainingState.create("breakout", 123, real_cfg())
    for _ in range(n): st.step()
    return digest(st)


def test_det():
    out = {}
    n = 1200
    t0 = time.time(); straight = run_digest(n); out["straight_digest"] = straight
    tmp = Path(tempfile.mkdtemp(prefix="b2_det_", dir=PKG / "runs" if (PKG / "runs").exists() else PKG))
    try:
        st = TrainingState.create("breakout", 123, real_cfg())
        for _ in range(600): st.step()
        ev = st.evaluate([11, 12, 13])                      # evaluate в середине (не должен менять траекторию)
        R.save_state(tmp / "mid.pt", st.snapshot()); del st
        st2 = TrainingState.restore(R.load_state(tmp / "mid.pt"))
        for _ in range(600): st2.step()
        out["disk_roundtrip_with_eval_digest"] = digest(st2); out["disk_roundtrip_equal_straight"] = digest(st2) == straight
        # то же без evaluate, но через deepcopy-снимок в памяти
        st3 = TrainingState.create("breakout", 123, real_cfg())
        for _ in range(600): st3.step()
        st4 = TrainingState.restore(copy.deepcopy(st3.snapshot()))
        for _ in range(600): st4.step()
        out["memory_snapshot_equal_straight"] = digest(st4) == straight
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out["sec"] = round(time.time() - t0, 1)
    # другие окружения процесса: тот же прямой прогон в подпроцессах
    py = sys.executable; script = str(HERE)
    variants = {"subprocess OMP/MKL/OPENBLAS=1 (как worker)": dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
                "subprocess без переменных потоков": {}}
    base_env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for name, extra in variants.items():
        r = subprocess.run([py, script, "digest"], env={**base_env, **extra}, capture_output=True, text=True)
        out[name] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        out[name + " equal"] = out[name] == straight
    r = subprocess.run([py, script, "digest", "--nodet"], env=base_env, capture_output=True, text=True)
    out["subprocess без deterministic_algorithms"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    out["subprocess без deterministic_algorithms equal"] = out["subprocess без deterministic_algorithms"] == straight
    return out


def tiny_config(root_rel, resume_every=2, horizon=8):
    config = json.loads((HERE.parents[1] / "configs/research_program.json").read_text(encoding="utf-8"))
    config.update(program_id="b2_resume", output_root=root_rel)
    config["learner"].update(conv_channels=2, hidden_size=8, batch_size=4, replay_capacity=32, warmup=8, target_period=4, episode_cap=7, dtype="float64")
    config["screen"].update(games=["breakout"], seeds=[0], ages=[12, 20], minimum_target_age=2, repeats=[0], horizon=horizon,
                            curve_steps=[0, 3, horizon], evaluation_episodes=1)
    config["resources"].update(updates_per_package=10**8, resume_every_steps=resume_every, bootstrap_repeats=10, progress_every_seconds=3600)
    return config


def new_campaign(name, **kw):
    root_rel = f"runs/{name}"
    folder = PKG / root_rel
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)
    config_path = folder / "cfg.json"
    R.dump(config_path, tiny_config(root_rel, **kw))
    return R.Campaign(config_path, resume=True), config_path, folder


FIELDS = ("j_final", "j_immediate", "evaluation_curve", "continuation_seed", "repair_seed", "budget_gradient_updates", "adaptation_auc", "completed_env_steps")


def test_torn():
    out = {}
    ref_c, _, ref_dir = new_campaign("b2_torn_ref", resume_every=2, horizon=8)
    cp = ref_c.source("breakout", 0, [12])[0]
    ref = ref_c.branch(cp, "optimizer_reset", 0)
    c, cfg_path, folder = new_campaign("b2_torn_run", resume_every=2, horizon=8)
    cp2 = c.source("breakout", 0, [12])[0]
    real_dump = R.dump; real_step = TrainingState.step; calls = {"dump_crash": 0}
    def crashing_dump(path, value):
        if Path(path).name == "branch_resume.json" and value.get("completed_steps") == 4:
            calls["dump_crash"] += 1
            raise RuntimeError("simulated crash between save_state(.pt) and dump(.json)")
        return real_dump(path, value)
    with patch.object(R, "dump", crashing_dump):
        try:
            c.branch(cp2, "optimizer_reset", 0)
        except RuntimeError as e:
            out["crash_raised"] = str(e)
    meta = json.loads((folder / "branch_resume.json").read_text(encoding="utf-8")); out["meta_completed_steps_after_crash"] = meta["completed_steps"]
    out["pt_hash_matches_meta_after_crash"] = R.file_hash(folder / "branch_resume.pt") == meta["sha256"]
    steps = {"n": 0}
    def counting(state):
        steps["n"] += 1; return real_step(state)
    c2 = R.Campaign(cfg_path, resume=True)
    t0 = time.time()
    with patch.object(TrainingState, "step", counting):
        res = c2.branch(cp2, "optimizer_reset", 0)
    out["steps_executed_after_resume"] = steps["n"]; out["horizon"] = 8
    out["restarted_from_zero"] = steps["n"] == 8
    out["fields_equal_reference"] = {f: res[f] == ref[f] for f in FIELDS}
    out["outcome_rows"] = len(R.rows(folder / "outcomes.jsonl"))
    out["wall_time_sec_resumed_vs_ref"] = [round(res["wall_time_sec"], 4), round(ref["wall_time_sec"], 4)]
    shutil.rmtree(folder, ignore_errors=True); shutil.rmtree(ref_dir, ignore_errors=True)
    return out


def test_memerr():
    out = {}
    parent, cfg_path, folder = new_campaign("b2_mem_parent", resume_every=2, horizon=8)
    cp = parent.source("breakout", 0, [12])[0]
    cp = dict(cp, artifact_path=str((folder / cp["artifact_path"]).resolve()))      # как после _merge_worker: абсолютный путь
    actions = list(parent.config["main_menu"])
    identity = {"checkpoint_id": cp["checkpoint_id"], "checkpoint_sha256": cp["sha256"], "mode": "natural", "repeat": 0, "horizon": 8, "actions": actions}
    task = S._new_task(parent, "branch", identity, checkpoint=cp, repeat=0, mode="natural", horizon=8, actions=actions, phase="screen",
                       maximum_updates=4 * 8 + 4)
    # эталон: безаварийный воркер на том же checkpoint в другом корне
    ref_parent, _, ref_dir = new_campaign("b2_mem_ref", resume_every=2, horizon=8)
    ref_cp = copy.deepcopy(cp)
    ref_task = S._new_task(ref_parent, "branch", dict(identity, ref=1), checkpoint=ref_cp, repeat=0, mode="natural", horizon=8, actions=actions, phase="screen",
                           maximum_updates=36)
    r_ref = S._run_worker_task(ref_task)
    ref_rows = {r["repair_id"]: r for r in R.rows(Path(ref_task["worker_root"]) / "outcomes.jsonl")}
    out["reference_error"] = r_ref["error"]; out["reference_arms"] = sorted(ref_rows)
    # аварийный воркер: MemoryError на 3-й ветке (head_reset) после её 5-го шага
    real_step = TrainingState.step; counter = {"n": 0, "armed": False}
    original_branch = R.Campaign.branch
    def tracking_branch(self, checkpoint_row, action, repeat, *a, **k):
        counter["armed"] = (action == "head_reset"); counter["n"] = 0
        return original_branch(self, checkpoint_row, action, repeat, *a, **k)
    def failing(state):
        if counter["armed"]:
            counter["n"] += 1
            if counter["n"] == 6: raise MemoryError()
        return real_step(state)
    with patch.object(R.Campaign, "branch", tracking_branch), patch.object(TrainingState, "step", failing):
        r1 = S._run_worker_task(task)
    out["first_attempt_error"] = r1["error"]
    # родитель импортирует то, что есть
    S._charge_worker(parent, task["worker_root"], r1["cumulative_updates"], allow_reserve=False)
    S._merge_worker(parent, task["worker_root"])
    out["after_first_merge_outcome_rows"] = sorted(r["repair_id"] for r in R.rows(folder / "outcomes.jsonl"))
    wr = Path(task["worker_root"])
    out["worker_resume_slot_exists_after_crash"] = [p.name for p in wr.glob("branch_resume.*")]
    # повторный запуск того же задания (как после рестарта supervisor)
    r2 = S._run_worker_task(task)
    out["second_attempt_error"] = r2["error"]
    S._charge_worker(parent, task["worker_root"], r2["cumulative_updates"], allow_reserve=False)
    S._merge_worker(parent, task["worker_root"])
    rows = R.rows(folder / "outcomes.jsonl")
    ids = [r["job_id"] for r in rows]
    out["final_rows"] = len(rows); out["unique_job_ids"] = len(set(ids)); out["duplicates"] = len(ids) - len(set(ids))
    got = {r["repair_id"]: r for r in rows}
    out["fields_equal_reference_per_arm"] = {a: all(got[a][f] == ref_rows[a][f] for f in FIELDS) for a in actions}
    out["worker_updates_first_second_reference"] = [r1["cumulative_updates"], r2["cumulative_updates"], r_ref["cumulative_updates"]]
    out["parent_package_updates"] = parent.package["updates"]
    out["leftover_worker_scratch"] = [p.name for p in wr.glob("branch_resume.*")]
    shutil.rmtree(folder, ignore_errors=True); shutil.rmtree(ref_dir, ignore_errors=True)
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "digest":
        print(run_digest())
    else:
        fn = {"det": test_det, "torn": test_torn, "memerr": test_memerr}[what]
        res = fn()
        print(json.dumps(res, indent=1, ensure_ascii=False))
        Path(HERE.parents[1] / f"research/debate/b2_resume_{what}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
