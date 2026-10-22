"""Two-state MDP counterexample (sprint 02, T1; docs/topic_card.md).

Can a lab (common-batch) measurement and the live effect of a repair disagree because
an on-policy learner changes its own data?  The design is fixed before the full run
(see notes/two_state_counterexample.md, "Предрегистрация"):

* Same learner everywhere: PPO with `epochs` passes over a batch of episodes; the
  ratio is anchored to the learner's own policy at the start of the iteration, and
  data from another behaviour policy is corrected by a truncated importance weight.
* One outcome on both sides: total effect = new-task return after training minus the
  aged agent's return before the repair (instant change + learning).
* Menu of the card: continue, optimizer_reset, parameter_reset, occupancy_restoration;
  each repair replaces one part of the aged agent, so its lab analogue is the hybrid
  that replaces that part (AAAA, AFAA, FAAF, AAFA).  joint_reset is reported outside it.
* Live control "frozen_behaviour": the same repairs trained for the same budget on
  data from a fixed behaviour policy (the lab's data source) - no feedback loop.
* "targets" in this toy = the critic used as the advantage baseline (MC returns do not
  depend on the critic).

Aged states: A short aging (stop at pi(switch|s1) <= 0.02); B long aging (A plus a fixed
budget on the old task, natural saturation); C = A with 75% of hidden units killed
(constructed); D control = trained on the new task (no deficit; reset should not win).

Run:  .venv\\Scripts\\python.exe -m causal_trainability.two_state [--quick]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from causal_trainability.telemetry import ROOT, Telemetry, TelemetryStore, code_revision, config_hash

CONFIG = dict(gamma=0.9, horizon=10, hidden=16, lr=0.01, adam_eps=1e-8, episodes=8, epochs=4, clip=0.2,
              value_coef=0.5, is_clip=1.0, aging_switch_prob=0.02, aging_max_iterations=5000,
              long_aging_iterations=3000, control_return_fraction=0.8, dead_fraction=0.75,
              lab_batches=16, live_iterations=150, restoration_fraction=0.25, histories=8, live_seeds=4,
              total_threshold=0.33, learning_threshold=0.1, tie_margin=0.1, adam_rule_threshold=0.5,
              divergence_bar=4, telemetry_every=10)
STAY, SWITCH = 0, 1
OLD_GOAL, NEW_GOAL = 0, 1
START = torch.tensor([1.0, 0.0])
NEXT = torch.tensor([[0, 1], [1, 0]])  # NEXT[s, a]
PARTS = ("params", "adam", "data", "targets")
MENU = ("continue", "optimizer_reset", "parameter_reset", "occupancy_restoration")
# Each repair replaces part of the aged agent; parameter_reset also replaces the critic (the baseline).
LAB_OF = {"continue": "AAAA", "optimizer_reset": "AFAA", "parameter_reset": "FAAF", "occupancy_restoration": "AAFA"}
SECONDARY = {"parameter_reset_with_fresh_data": "FAFF", "joint_reset": "FFAF"}
REPAIRS = MENU + ("joint_reset",)
MODES = ("live", "frozen_behaviour")
SCENARIOS = ("A_occupancy", "B_saturated", "C_dead_units", "D_control")


def seed_of(*keys):
    return int(hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:8], 16)


def optimum(cfg):
    return (1 - cfg["gamma"] ** cfg["horizon"]) / (1 - cfg["gamma"])


class Agent(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.body = nn.Linear(2, hidden)
        self.actor = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, states):
        hidden = torch.relu(self.body(nn.functional.one_hot(states, 2).float()))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def layers(self):
        return {"body": list(self.body.parameters()), "actor": list(self.actor.parameters()),
                "critic": list(self.critic.parameters())}


def make_agent(history, cfg):
    torch.manual_seed(seed_of("init", history))
    return Agent(cfg["hidden"])


def make_optimizer(agent, cfg):
    return torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])


def transplant(dst_optimizer, dst_agent, src_optimizer, src_agent):
    for dst, src in zip(dst_agent.parameters(), src_agent.parameters()):
        state = src_optimizer.state.get(src)
        if state:
            dst_optimizer.state[dst] = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}


def build(params_agent, adam_source, cfg):
    """Copy of `params_agent` with an empty Adam, or Adam state transplanted from (agent, optimizer)."""
    agent = copy.deepcopy(params_agent)
    optimizer = make_optimizer(agent, cfg)
    if adam_source is not None:
        transplant(optimizer, agent, adam_source[1], adam_source[0])
    return agent, optimizer


def policy(agent):
    with torch.no_grad():
        logits, _ = agent(torch.tensor([0, 1]))
    return torch.softmax(logits.double(), dim=-1), logits.double()


def exact_return(agent, goal, cfg, start=START):
    probs, _ = policy(agent)
    reward = (NEXT == goal).double()
    value = torch.zeros(2, dtype=torch.float64)
    for _ in range(cfg["horizon"]):
        value = (probs * (reward + cfg["gamma"] * value[NEXT])).sum(1)
    return float((start.double() * value).sum())


def dead_fraction(agent):
    with torch.no_grad():
        pre = agent.body(torch.eye(2))
    return float((pre <= 0).all(0).float().mean())


def adam_stale_fraction(agent, optimizer, layer="actor"):
    """Among the layer's parameters that ever received gradient (v > 0), the share with sqrt(v) < 10 eps.

    Parameters with v == 0 (units that never fired) are excluded: they measure dead units, not Adam.
    """
    eps = optimizer.param_groups[0]["eps"]
    values = [optimizer.state[p]["exp_avg_sq"].flatten() for p in agent.layers()[layer] if p in optimizer.state]
    if not values:
        return 0.0
    v = torch.cat(values)
    v = v[v > 0]
    return float((v.sqrt() < 10 * eps).float().mean()) if v.numel() else 0.0


def describe(agent, optimizer, cfg):
    probs, logits = policy(agent)
    return {"switch_prob_s1": float(probs[0, SWITCH]), "switch_prob_s2": float(probs[1, SWITCH]),
            "logit_gap_s1": float(logits[0, STAY] - logits[0, SWITCH]), "dead_fraction": dead_fraction(agent),
            "new_task_return": exact_return(agent, NEW_GOAL, cfg),
            "adam_stale_fraction_actor": adam_stale_fraction(agent, optimizer),
            "layer_norms": {name: float(torch.sqrt(sum((p.detach() ** 2).sum() for p in params)))
                            for name, params in agent.layers().items()}}


def sample_batch(behaviour, goal, generator, cfg):
    episodes = cfg["episodes"]
    with torch.no_grad():
        state = torch.multinomial(START.expand(episodes, 2), 1, generator=generator).squeeze(1)
        states, actions, rewards, logps = [], [], [], []
        for _ in range(cfg["horizon"]):
            logits, _ = behaviour(state)
            action = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(1)
            nxt = torch.where(action == SWITCH, 1 - state, state)
            states.append(state); actions.append(action)
            rewards.append((nxt == goal).float())
            logps.append(torch.log_softmax(logits, -1).gather(1, action[:, None]).squeeze(1))
            state = nxt
        rewards = torch.stack(rewards, 1)
        returns, running = torch.zeros_like(rewards), torch.zeros(episodes)
        for t in reversed(range(cfg["horizon"])):
            running = rewards[:, t] + cfg["gamma"] * running
            returns[:, t] = running
    return {"states": torch.stack(states, 1).flatten(), "actions": torch.stack(actions, 1).flatten(),
            "returns": returns.flatten(), "behaviour_logp": torch.stack(logps, 1).flatten()}


def ppo_iteration(agent, optimizer, batch, cfg, baseline=None):
    """`epochs` PPO passes; ratio anchored to the agent's policy at the start of the iteration.

    `baseline` is the critic used for advantages (the agent itself unless crossed in the lab).
    """
    with torch.no_grad():
        logits, values = agent(batch["states"])
        anchor = torch.log_softmax(logits, -1).gather(1, batch["actions"][:, None]).squeeze(1)
        weight = torch.exp(anchor - batch["behaviour_logp"]).clamp(max=cfg["is_clip"])
        if baseline is not None:
            _, values = baseline(batch["states"])
        advantage = batch["returns"] - values
    loss_value = 0.0
    for _ in range(cfg["epochs"]):
        logits, values = agent(batch["states"])
        logp = torch.log_softmax(logits, -1).gather(1, batch["actions"][:, None]).squeeze(1)
        ratio = torch.exp(logp - anchor)
        surrogate = torch.min(ratio * advantage, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * advantage)
        loss = -(weight * surrogate).mean() + cfg["value_coef"] * (weight * (values - batch["returns"]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return loss_value


def train(agent, optimizer, goal, iterations, generator, cfg, behaviour_for=None, telemetry=None, until=None):
    """Return the curve [(iteration, new-task return)] and the iterations used.

    behaviour_for(iteration) gives the data-collecting policy; None = the learner itself.
    """
    curve = [(0, exact_return(agent, NEW_GOAL, cfg))]
    for it in range(1, iterations + 1):
        behaviour = agent if behaviour_for is None else behaviour_for(it)
        batch = sample_batch(behaviour, goal, generator, cfg)
        loss = ppo_iteration(agent, optimizer, batch, cfg)
        done = until is not None and until(agent)
        last = done or it == iterations
        if it % cfg["telemetry_every"] == 0 or last:
            new_return = exact_return(agent, NEW_GOAL, cfg)
            curve.append((it, new_return))
            if telemetry is not None:
                probs, _ = policy(agent)
                telemetry.step(it, force=True, layer_extra={"body": {"dead_fraction": dead_fraction(agent)}},
                               metrics={"loss": loss, "new_task_return": new_return,
                                        "old_task_return": exact_return(agent, OLD_GOAL, cfg),
                                        "switch_prob_s1": float(probs[0, SWITCH]),
                                        "switch_prob_s2": float(probs[1, SWITCH])})
        if done:
            return curve, it
    return curve, iterations


def aged_states(history, cfg, out, store):
    fresh = make_agent(history, cfg)
    fresh_optimizer = make_optimizer(fresh, cfg)
    meta = {"history": history, "config_hash": config_hash(cfg)}

    def age(agent, optimizer, goal, iterations, name, until=None):
        telemetry = Telemetry(out, f"aging_{name}_h{history}", agent.layers(), optimizer, store=store,
                              meta=dict(kind="aging", scenario=name, **meta), every=cfg["telemetry_every"])
        telemetry.snapshot(0, agent, optimizer)  # theta0 (or the state aging starts from)
        generator = torch.Generator().manual_seed(seed_of("aging", name, history))
        _, used = train(agent, optimizer, goal, iterations, generator, cfg, telemetry=telemetry, until=until)
        telemetry.snapshot(used, agent, optimizer)
        return used

    short, short_optimizer = build(fresh, None, cfg)
    used_short = age(short, short_optimizer, OLD_GOAL, cfg["aging_max_iterations"], "A_occupancy",
                     until=lambda a: policy(a)[0][0, SWITCH] <= cfg["aging_switch_prob"])
    long, long_optimizer = build(short, (short, short_optimizer), cfg)
    age(long, long_optimizer, OLD_GOAL, cfg["long_aging_iterations"], "B_saturated")
    dead, dead_optimizer = build(short, (short, short_optimizer), cfg)
    with torch.no_grad():
        alive = (dead.body(torch.eye(2)) > 0).any(0).nonzero().flatten()
        order = alive[torch.randperm(len(alive), generator=torch.Generator().manual_seed(seed_of("kill", history)))]
        killed = order[: int(round(cfg["dead_fraction"] * len(alive)))]
        weight = dead.body.weight
        dead.body.bias[killed] = -(weight[killed].max(dim=1).values.clamp_min(0.0)) - 1.0
    control, control_optimizer = build(fresh, None, cfg)
    target = cfg["control_return_fraction"] * optimum(cfg)
    used_control = age(control, control_optimizer, NEW_GOAL, cfg["aging_max_iterations"], "D_control",
                       until=lambda a: exact_return(a, NEW_GOAL, cfg) >= target)
    aged = {"A_occupancy": (short, short_optimizer), "B_saturated": (long, long_optimizer),
            "C_dead_units": (dead, dead_optimizer), "D_control": (control, control_optimizer)}
    used = {"A_occupancy": used_short, "B_saturated": used_short + cfg["long_aging_iterations"],
            "C_dead_units": used_short, "D_control": used_control,
            "converged": {"A_occupancy": used_short < cfg["aging_max_iterations"],
                          "D_control": used_control < cfg["aging_max_iterations"]}}
    return fresh, fresh_optimizer, aged, used


def matched_norm(aged_agent, fresh_agent):
    """Aged parameters rescaled per layer (weights and bias together) to the fresh layer norm."""
    twin = copy.deepcopy(aged_agent)
    with torch.no_grad():
        for params, fresh_params in zip(twin.layers().values(), fresh_agent.layers().values()):
            factor = torch.sqrt(sum((p ** 2).sum() for p in fresh_params)) / torch.sqrt(sum((p ** 2).sum() for p in params))
            for p in params:
                p.mul_(factor)
    return twin


def matched_gap(aged_agent, fresh_agent):
    """Aged actor rescaled so that the |logit gap| at s1 equals the fresh one (entropy matching)."""
    twin = copy.deepcopy(aged_agent)
    aged_gap = abs(float(policy(twin)[1][0, STAY] - policy(twin)[1][0, SWITCH]))
    fresh_gap = abs(float(policy(fresh_agent)[1][0, STAY] - policy(fresh_agent)[1][0, SWITCH]))
    if aged_gap > 0:
        with torch.no_grad():
            for p in twin.actor.parameters():
                p.mul_(fresh_gap / aged_gap)
    return twin


def lab(fresh, fresh_optimizer, aged, aged_optimizer, history, scenario, cfg, out, store):
    """Mean new-task return after one PPO iteration of each hybrid over `lab_batches` common batches."""
    params = {"F": fresh, "A": aged, "Mn": matched_norm(aged, fresh), "Mg": matched_gap(aged, fresh)}
    adam = {"F": None, "A": (aged, aged_optimizer)}
    behaviour = {"F": fresh, "A": aged}
    batches = {level: [sample_batch(behaviour[level], NEW_GOAL, torch.Generator().manual_seed(
        seed_of("lab", history, b, level) if level == "F" else seed_of("lab", scenario, history, b, level)), cfg)
        for b in range(cfg["lab_batches"])] for level in "FA"}
    combos = ["".join(c) for c in itertools.product("FA", repeat=4)]
    combos += ["MnFFF", "MgFFF", "MnAAA", "MgAAA"]
    results = {}
    for combo in combos:
        p, rest = (combo[:2], combo[2:]) if combo.startswith("M") else (combo[0], combo[1:])
        o, d, t = rest
        start_agent, _ = build(params[p], adam[o], cfg)
        start = exact_return(start_agent, NEW_GOAL, cfg)
        after = []
        for b, batch in enumerate(batches[d]):
            agent, optimizer = build(params[p], adam[o], cfg)
            lab_iteration(agent, optimizer, batch, cfg, params[t], out, store,
                          f"lab_{scenario}_h{history}_{combo}_b{b}",
                          {"kind": "lab", "scenario": scenario, "history": history, "hybrid": combo, "batch": b,
                           "config_hash": config_hash(cfg)})
            after.append(exact_return(agent, NEW_GOAL, cfg))
        results[combo] = {"start": start, "after": sum(after) / len(after)}
    return results


def lab_iteration(agent, optimizer, batch, cfg, baseline, out, store, run_id, meta):
    telemetry = Telemetry(out, run_id, agent.layers(), optimizer, store=store, every=1, meta=meta)
    loss = ppo_iteration(agent, optimizer, batch, cfg, baseline=baseline)
    telemetry.step(1, force=True, metrics={"loss": loss, "new_task_return": exact_return(agent, NEW_GOAL, cfg)},
                   layer_extra={"body": {"dead_fraction": dead_fraction(agent)}})


def live(fresh, aged, aged_optimizer, history, scenario, cfg, out, store):
    rows = []
    cutoff = int(cfg["live_iterations"] * cfg["restoration_fraction"])
    aged_return = exact_return(aged, NEW_GOAL, cfg)
    for mode, seed, repair in itertools.product(MODES, range(cfg["live_seeds"]), REPAIRS):
        if repair in ("parameter_reset", "joint_reset"):
            agent, optimizer = build(fresh, (aged, aged_optimizer) if repair == "parameter_reset" else None, cfg)
        else:
            agent, optimizer = build(aged, None if repair == "optimizer_reset" else (aged, aged_optimizer), cfg)
        frozen_aged = copy.deepcopy(aged)
        if mode == "live":
            behaviour_for = ((lambda it, agent=agent: fresh if it <= cutoff else agent)
                             if repair == "occupancy_restoration" else None)
        else:  # the lab's data source, kept fixed (no feedback from the learner to its data)
            frozen_start = copy.deepcopy(agent)
            if repair == "occupancy_restoration":
                behaviour_for = lambda it, frozen=frozen_start: fresh if it <= cutoff else frozen
            else:
                behaviour_for = lambda it, frozen=frozen_aged: frozen
        run_id = f"live_{scenario}_h{history}_{mode}_s{seed}_{repair}"
        telemetry = Telemetry(out, run_id, agent.layers(), optimizer, store=store, every=cfg["telemetry_every"],
                              meta={"kind": "live", "mode": mode, "scenario": scenario, "history": history,
                                    "seed": seed, "repair": repair, "config_hash": config_hash(cfg)})
        start = exact_return(agent, NEW_GOAL, cfg)
        generator = torch.Generator().manual_seed(seed_of("live", scenario, history, seed))
        curve, _ = train(agent, optimizer, NEW_GOAL, cfg["live_iterations"], generator, cfg,
                         behaviour_for=behaviour_for, telemetry=telemetry)
        telemetry.snapshot(cfg["live_iterations"], agent, optimizer)
        returns = [r for _, r in curve]
        rows.append({"history": history, "mode": mode, "seed": seed, "repair": repair,
                     "total_effect": returns[-1] - aged_return, "learning": returns[-1] - start,
                     "instant": start - aged_return, "final_return": returns[-1],
                     "auc": sum(returns) / len(returns), "curve": curve})
    return rows


def mean(values):
    return sum(values) / len(values)


def stats(values):
    return {"mean": mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def decide(lab_result, aged, aged_optimizer, cfg):
    """Pre-registered rules; all choose from MENU.

    Total effects (instant + one iteration) use `total_threshold`; one-iteration learning gains
    use `learning_threshold`.  `instant` ignores learning: it only looks at the starting return.
    """
    total = {repair: lab_result[LAB_OF[repair]]["after"] - lab_result["AAAA"]["start"] for repair in MENU}
    best = max(MENU, key=lambda r: total[r])
    crossed = best if total[best] - total["continue"] > cfg["total_threshold"] else "continue"
    instant_score = {repair: lab_result[LAB_OF[repair]]["start"] - lab_result["AAAA"]["start"] for repair in MENU}
    best_instant = max(MENU, key=lambda r: instant_score[r])
    instant = best_instant if instant_score[best_instant] > cfg["total_threshold"] else "continue"
    learn = lambda c: lab_result[c]["after"] - lab_result[c]["start"]
    common_batch = ("parameter_reset" if learn("FFFF") - learn("AFFF") > cfg["learning_threshold"] else "continue")
    readiness = ("parameter_reset" if learn("FFFF") - learn("AAAA") > cfg["learning_threshold"] else "continue")
    adam_rule = ("optimizer_reset" if adam_stale_fraction(aged, aged_optimizer) > cfg["adam_rule_threshold"]
                 else "continue")
    return {"crossed_lab": crossed, "instant_only": instant, "common_batch_params_only": common_batch,
            "readiness_own_data": readiness, "adam_state": adam_rule}, total


def best_or_tie(scores, margin):
    """The best repair of MENU, or 'tie' if the runner-up is within `margin`."""
    ranked = sorted(MENU, key=lambda r: scores[r], reverse=True)
    return ranked[0] if scores[ranked[0]] - scores[ranked[1]] > margin else "tie"


def summarize(per_history, cfg):
    """per_history: list of dicts with lab, live rows, choices; the history is the unit."""
    out = {"histories": len(per_history)}
    for mode in MODES:
        effect = {r: [mean([x["total_effect"] for x in h["live"] if x["mode"] == mode and x["repair"] == r])
                      for h in per_history] for r in REPAIRS}
        out[f"{mode}_total_effect"] = {r: stats(v) for r, v in effect.items()}
        out[f"{mode}_best_repair_per_history"] = [max(MENU, key=lambda r: effect[r][i]) for i in range(len(per_history))]
        out[f"{mode}_effect_per_history"] = effect
    lab_total = {r: [h["lab_total"][r] for h in per_history] for r in MENU}
    out["lab_total_effect"] = {r: stats(v) for r, v in lab_total.items()}
    out["lab_best_repair_per_history"] = [max(MENU, key=lambda r: h["lab_total"][r]) for h in per_history]
    margin = cfg["tie_margin"]
    lab_best = [best_or_tie(h["lab_total"], margin) for h in per_history]
    live_best = [best_or_tie({r: out["live_effect_per_history"][r][i] for r in MENU}, margin)
                 for i in range(len(per_history))]
    frozen_best = [best_or_tie({r: out["frozen_behaviour_effect_per_history"][r][i] for r in MENU}, margin)
                   for i in range(len(per_history))]
    feedback = [l == f != v and "tie" not in (l, f, v) for l, f, v in zip(lab_best, frozen_best, live_best)]
    other = [l != v and l != f and "tie" not in (l, f, v) for l, f, v in zip(lab_best, frozen_best, live_best)]
    out["divergence"] = {"lab_best": lab_best, "frozen_best": frozen_best, "live_best": live_best,
                         "lab_equals_frozen_not_live": sum(feedback),
                         "lab_differs_from_both": sum(other),
                         "ties": sum("tie" in (l, f, v) for l, f, v in zip(lab_best, frozen_best, live_best)),
                         "bar": cfg["divergence_bar"],
                         "feedback_counterexample": sum(feedback) >= cfg["divergence_bar"]}
    out["lab_secondary"] = {name: stats([h["lab"][combo]["after"] - h["lab"]["AAAA"]["start"] for h in per_history])
                            for name, combo in SECONDARY.items()}
    rules = {}
    for rule in per_history[0]["choices"]:
        chosen = [h["choices"][rule] for h in per_history]
        rule_out = {"choices": chosen}
        for mode in MODES:
            effect = out[f"{mode}_effect_per_history"]
            regret = [max(effect[r][i] for r in MENU) - effect[c][i] for i, c in enumerate(chosen)]
            rule_out[f"{mode}_regret"] = stats(regret)
            rule_out[f"{mode}_agreement"] = sum(c == b for c, b in zip(chosen, out[f"{mode}_best_repair_per_history"]))
        rules[rule] = rule_out
    out["rules"] = rules
    for mode in MODES:
        effect = out[f"{mode}_effect_per_history"]
        constant = {r: stats([max(effect[x][i] for x in MENU) - effect[r][i] for i in range(len(per_history))]) for r in MENU}
        out[f"{mode}_constant_rule_regret"] = constant
    learn = lambda h, c: h["lab"][c]["after"] - h["lab"][c]["start"]
    # Residual: fresh minus matched-aged parameters on fresh data/Adam/targets (>0 = deficit survives matching).
    # Matching gain: matched-aged minus aged parameters with everything else aged.
    out["matching"] = {
        "residual_after_norm_matching": stats([learn(h, "FFFF") - learn(h, "MnFFF") for h in per_history]),
        "residual_after_gap_matching": stats([learn(h, "FFFF") - learn(h, "MgFFF") for h in per_history]),
        "gain_of_norm_matching_on_aged": stats([learn(h, "MnAAA") - learn(h, "AAAA") for h in per_history]),
        "gain_of_gap_matching_on_aged": stats([learn(h, "MgAAA") - learn(h, "AAAA") for h in per_history])}
    out["parameter_deficit_fresh_data"] = stats(
        [(h["lab"]["FFFF"]["after"] - h["lab"]["FFFF"]["start"]) - (h["lab"]["AFFF"]["after"] - h["lab"]["AFFF"]["start"])
         for h in per_history])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = dict(CONFIG)
    if args.quick:
        cfg.update(histories=2, live_seeds=1, lab_batches=2, live_iterations=20, long_aging_iterations=200)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = Path(args.out or ROOT / "runs" / "causal-trainability" / f"two_state_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    store = TelemetryStore(out, "two_state")
    started = time.time()
    per = {s: [] for s in SCENARIOS}
    for history in range(cfg["histories"]):
        fresh, fresh_optimizer, aged, used = aged_states(history, cfg, out, store)
        for scenario, (agent, optimizer) in aged.items():
            lab_result = lab(fresh, fresh_optimizer, agent, optimizer, history, scenario, cfg, out, store)
            choices, total = decide(lab_result, agent, optimizer, cfg)
            per[scenario].append({"history": history, "aging_iterations": used[scenario],
                                  "aging_converged": used["converged"],
                                  "aged": describe(agent, optimizer, cfg),
                                  "fresh": describe(fresh, fresh_optimizer, cfg),
                                  "lab": lab_result, "lab_total": total, "choices": choices,
                                  "live": live(fresh, agent, optimizer, history, scenario, cfg, out, store)})
        print(f"history {history} done in {time.time() - started:.0f}s", flush=True)
    store.close()
    report = {"config": cfg, "config_hash": config_hash(cfg), "revision": code_revision(),
              "output": out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out),
              "wall_time_sec": time.time() - started,
              "scenarios": {s: {"aged_states": [{k: h[k] for k in ("history", "aging_iterations", "aged", "fresh")}
                                                for h in rows],
                                **summarize(rows, cfg)} for s, rows in per.items()}}
    (out / "per_history.json").write_text(json.dumps(per), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for scenario, s in report["scenarios"].items():
        print(scenario, "live best:", s["live_best_repair_per_history"], "frozen best:",
              s["frozen_behaviour_best_repair_per_history"], "lab best:", s["lab_best_repair_per_history"])
        print("   rules:", {r: (v["choices"][0], v["live_agreement"]) for r, v in s["rules"].items()})


if __name__ == "__main__":
    main()
