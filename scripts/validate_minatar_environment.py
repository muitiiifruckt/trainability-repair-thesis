"""Headless native MinAtar compatibility and complete-environment clone check."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import io
import json
import platform
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from minatar import Environment

ROOT = Path(__file__).resolve().parents[1]
GAMES = ("breakout", "asterix", "freeway", "seaquest", "space_invaders")
WHEEL_URL = "https://files.pythonhosted.org/packages/c6/6d/aa5ef58c2c45a8de8ac37599179df26b6fd47781d6b2614d2116b8bfca5c/MinAtar-1.0.15-py3-none-any.whl"
WHEEL_SHA = "f40450e133c552b97222702bb7b1c5254b47758390f4bd25e411399af0b21d42"


def validate() -> dict:
    started = time.perf_counter()
    wheel = urllib.request.urlopen(WHEEL_URL, timeout=30).read()
    assert hashlib.sha256(wheel).hexdigest() == WHEEL_SHA
    import minatar
    installed_root = Path(minatar.__file__).parent
    checked_files = []
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in archive.namelist():
            if name.startswith("minatar/") and name.endswith(".py"):
                source = archive.read(name).replace(b"\r\n", b"\n")
                installed = (installed_root.parent / name).read_bytes().replace(b"\r\n", b"\n")
                assert source == installed, name
                checked_files.append(name)
    results = []
    for game in GAMES:
        env = Environment(game, sticky_action_prob=.1, difficulty_ramping=True)
        env.seed(42)
        env.reset()
        assert env.state().dtype == np.bool_
        for index in range(37):
            _, terminal = env.act(index % 6)
            if terminal:
                env.reset()
        clone = copy.deepcopy(env)
        assert clone.random is clone.env.random
        assert clone.random is not env.random
        for index in range(500):
            action = (index * 5 + 2) % 6
            reward, terminal = env.act(action)
            clone_reward, clone_terminal = clone.act(action)
            assert (reward, terminal) == (clone_reward, clone_terminal), (game, index)
            np.testing.assert_array_equal(env.state(), clone.state())
            assert env.last_action == clone.last_action
            if terminal:
                env.reset()
                clone.reset()
        results.append({"game": game, "shape": list(env.state().shape), "clone_trace_steps": 500, "passed": True})
    packages = {name: importlib.metadata.version(name) for name in
                ("MinAtar", "numpy", "torch", "scipy", "matplotlib", "pandas", "seaborn", "scikit-learn", "pytz")}
    return {"kind": "native_environment_semantics_validation_not_learning",
            "date": "2026-10-07", "python": platform.python_version(), "packages": packages,
            "wheel_url": WHEEL_URL, "wheel_sha256": WHEEL_SHA, "installed_files_verified": checked_files,
            "games": results, "wall_time_sec": time.perf_counter()-started,
            "note": "Reserved games were used only for API/clone tests, no learning or policy selection."}


if __name__ == "__main__":
    report = validate()
    output = ROOT / "research/results/minatar_environment_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed_games": len(report["games"]), "packages": report["packages"], "output": str(output)}, ensure_ascii=False))
