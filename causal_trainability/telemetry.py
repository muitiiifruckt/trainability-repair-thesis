"""Always-on background telemetry (docs/observability_spec.md, decision 2026-10-22).

Reads parameters and optimizer state only: it never draws random numbers and never
writes to the model or the optimizer, so training is bit-identical with or without it.
Rows are serialised by a background thread into <root>/telemetry/<run_id>.jsonl and
every run is registered in <root>/runs_index.jsonl, one format for all runs.  Closed
telemetry files and L3 snapshots get a sha256 row in <root>/artifacts.jsonl.
"""
from __future__ import annotations

import functools
import importlib.metadata
import hashlib
import json
import platform
import queue
import subprocess
import threading
import time
import copy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_MAX_QUANTILE_ELEMENTS = 1_000_000


@functools.lru_cache(maxsize=1)
def code_revision():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip() or None
    except OSError:
        head = None
    files = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted((ROOT / "causal_trainability").glob("*.py"))}
    versions = {}
    for name in ("numpy", "MinAtar"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "causal_trainability"], cwd=ROOT,
                           capture_output=True, text=True, check=False).stdout.strip() != ""
    return {"git_head": head, "dirty_causal_trainability": dirty, "files": files,
            "python": platform.python_version(), "torch": torch.__version__, **versions}


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def record_artifact(root, path, kind):
    """Append the sha256 of a finished artifact to <root>/artifacts.jsonl."""
    root, path = Path(root), Path(path)
    data = path.read_bytes()
    row = {"path": path.relative_to(root).as_posix(), "kind": kind, "bytes": len(data),
           "sha256": hashlib.sha256(data).hexdigest(), "time": time.time()}
    with (root / "artifacts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _norm(tensors):
    return float(torch.sqrt(sum((t.double() ** 2).sum() for t in tensors))) if tensors else 0.0


def _quantiles(values):
    flat = values.flatten().double()
    if flat.numel() > _MAX_QUANTILE_ELEMENTS:  # deterministic stride, no RNG
        flat = flat[:: flat.numel() // _MAX_QUANTILE_ELEMENTS + 1]
    q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return [float(x) for x in q]


class _Writer(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.items = path, queue.Queue()

    def run(self):
        with self.path.open("a", encoding="utf-8") as stream:
            while True:
                item = self.items.get()
                if item is None:
                    stream.flush()
                    return
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=True) + "\n")


class TelemetryStore:
    """One writer thread and one file shared by many small runs (avoids thousands of files).

    Snapshots of small models are kept in memory and saved as one file on close.
    """

    def __init__(self, root, name):
        self.root, self.name = Path(root), name
        (self.root / "telemetry").mkdir(parents=True, exist_ok=True)
        self.path = f"telemetry/{name}.jsonl"
        self.writer = _Writer(self.root / self.path)
        self.writer.start()
        self.snapshots = {}
        (self.root / "revision.json").write_text(json.dumps(code_revision(), indent=2), encoding="utf-8")

    def snapshot(self, run_id, update, model, optimizer):
        self.snapshots[f"{run_id}_u{update}"] = {"model": copy.deepcopy(model.state_dict()),
                                                 "optimizer": copy.deepcopy(optimizer.state_dict()),
                                                 "update": update}

    def close(self):
        self.writer.items.put(None)
        self.writer.join()
        record_artifact(self.root, self.root / self.path, "telemetry")
        if self.snapshots:
            (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
            path = self.root / "snapshots" / f"{self.name}.pt"
            torch.save(self.snapshots, path)
            record_artifact(self.root, path, "snapshot_bundle")


class Telemetry:
    """Per-layer metrics every `every` updates, plus L3 snapshots on request.

    `layers` maps a layer name to its parameters, e.g. {"body": [...], "actor": [...]}.
    With `store`, rows go to the store's shared file and snapshots into its bundle.
    """

    def __init__(self, root, run_id, layers, optimizer, meta, every=10, enabled=True, store=None):
        self.enabled, self.every, self.run_id = enabled, every, run_id
        self.layers, self.optimizer = layers, optimizer
        self.root, self.store = Path(root), store
        if not enabled:
            return
        (self.root / "telemetry").mkdir(parents=True, exist_ok=True)
        self.theta0 = {name: [p.detach().clone() for p in params] for name, params in layers.items()}
        self.previous = {name: [p.detach().clone() for p in params] for name, params in layers.items()}
        if store is None:
            self.writer = _Writer(self.root / "telemetry" / f"{run_id}.jsonl")
            self.writer.start()
            location = f"telemetry/{run_id}.jsonl"
        else:
            self.writer, location = store.writer, store.path
        index = dict(run_id=run_id, started_at=time.time(), telemetry=location,
                     every=every, layers={name: sum(p.numel() for p in params)
                                          for name, params in layers.items()},
                     revision="revision.json" if store is not None else code_revision(), **meta)
        with (self.root / "runs_index.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(index, ensure_ascii=False) + "\n")

    def _layer_row(self, name, params):
        state = [self.optimizer.state.get(p, {}) for p in params]
        eps = self.optimizer.param_groups[0].get("eps", 0.0)
        row = {"weight_norm": _norm([p.detach() for p in params]),
               "dist_init": _norm([p.detach() - p0 for p, p0 in zip(params, self.theta0[name])]),
               "grad_norm": _norm([p.grad.detach() for p in params if p.grad is not None])}
        window = _norm([p.detach() - q for p, q in zip(params, self.previous[name])])
        row["effective_step"] = window / row["weight_norm"] if row["weight_norm"] else None
        if all("exp_avg" in s for s in state):
            row["adam_m_norm"] = _norm([s["exp_avg"] for s in state])
            sqrt_v = torch.cat([s["exp_avg_sq"].sqrt().flatten() for s in state])
            row["adam_sqrt_v_q01"], row["adam_sqrt_v_q50"], row["adam_sqrt_v_q99"] = _quantiles(sqrt_v)
            row["adam_frac_sqrt_v_below_10eps"] = float((sqrt_v < 10 * eps).double().mean())
            row["adam_step"] = float(state[0]["step"]) if "step" in state[0] else None
        self.previous[name] = [p.detach().clone() for p in params]
        return row

    def step(self, update, metrics=None, layer_extra=None, force=False):
        """Queue one row; call after optimizer.step() and before zero_grad()."""
        if not self.enabled or (update % self.every and not force):
            return
        with torch.no_grad():
            layers = {name: self._layer_row(name, params) for name, params in self.layers.items()}
        for name, extra in (layer_extra or {}).items():
            layers.setdefault(name, {}).update(extra)
        self.writer.items.put({"run_id": self.run_id, "update": update, "time": time.time(),
                               "layers": layers, "metrics": metrics or {}})

    def snapshot(self, update, model, optimizer):
        """L3: weights and optimizer state (no replay), named by run and update."""
        if not self.enabled:
            return None
        if self.store is not None:
            self.store.snapshot(self.run_id, update, model, optimizer)
            return None
        folder = self.root / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.run_id}_u{update}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update}, path)
        record_artifact(self.root, path, "snapshot")
        return path

    def close(self):
        if self.enabled and self.store is None:
            self.writer.items.put(None)
            self.writer.join()
            record_artifact(self.root, self.root / "telemetry" / f"{self.run_id}.jsonl", "telemetry")
