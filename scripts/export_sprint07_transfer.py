"""Cross-check the primary contrast from raw curves and publish compact files."""
import json
import math
from statistics import mean, stdev

from scripts.sprint07_transfer import ARMS, CAL, ROOT, ROOT_OUT, SEEDS, digest, read


def main():
    source = ROOT_OUT / "summary.json"
    summary = read(source)
    assert summary["gate_sha256"] == digest(CAL / "summary.json")
    assert len(summary["records"]) == 12 and [r["seed"] for r in summary["sources"]] == list(SEEDS)
    contrasts = []
    for seed in SEEDS:
        repeated = []
        for repeat in (0, 1):
            record, = [r for r in summary["records"] if (r["seed"], r["repeat"]) == (seed, repeat)]
            scores = {}
            for arm in ARMS:
                folder = ROOT_OUT / "science" / f"s{seed}" / f"{arm}_{repeat}"
                seal = read(folder / "done.json")
                path = folder / seal["attempt"] / "result.json"
                assert digest(path) == seal["sha256"] == record[arm]["result_sha256"]
                curve = read(path)["curve"]
                assert len(curve) == 128
                values = [mean(v for step in curve[i:i + 32] for v in step["returns"])
                          for i in range(0, 128, 32)]
                assert values == record[arm]["windows"]
                scores[arm] = mean(values)
            paired = scores["reset"] - scores["keep"]
            assert paired == record["difference"]
            repeated.append(paired)
        contrasts.append(mean(repeated))
    effect = mean(contrasts)
    assert contrasts == summary["primary"]["per_history"] and effect == summary["primary"]["mean"]
    half = 2.570581835636314 * stdev(contrasts) / math.sqrt(6)
    assert [effect - half, effect + half] == summary["primary"]["ci95"]
    result_dir = ROOT / "research/results/sprint07-transfer"
    result_dir.mkdir(parents=True, exist_ok=True)
    files = {"summary.json": source, "calibration.json": CAL / "summary.json",
             "dependency_manifest.json": ROOT_OUT / "dependency_manifest.json"}
    for name, file in files.items():
        (result_dir / name).write_bytes(file.read_bytes())
    manifest = {name: {"source": file.relative_to(ROOT).as_posix(), "sha256": digest(file)}
                for name, file in files.items()}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Verified raw curves and primary effect for 6 histories, 24 branches")
    print(json.dumps({"primary": summary["primary"], "means": summary["means"]}, indent=2))


if __name__ == "__main__":
    main()
