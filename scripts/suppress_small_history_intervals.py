"""Редакционная обработка auto-report без изменения frozen analysis code.

Удаляет отображаемые intervals, если независимых histories меньше пяти.
Не меняет числовые estimates, raw данные, checkpoints или config.
"""
from pathlib import Path
import argparse
import re

ROOT = Path(__file__).resolve().parents[1]
NOTE = ("**Редакционная проверка:** intervals при <5 source histories скрыты согласно AGENTS.md. "
        "Screen и confirmation сравниваются отдельно в [CURRENT_STATUS.md](CURRENT_STATUS.md); "
        "повторы одной source history не являются независимыми histories.\n\n")


def suppress(text):
    result, count = re.subn(r"; 95% interval \[[^\n]*\]; histories ([0-4])\.",
                           r"; histories \1. Интервал скрыт (n<5).", text)
    if count and NOTE not in result:
        first, separator, rest = result.partition("\n\n")
        result = first + separator + NOTE + rest
    return result, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "research/results/minatar-repair-20261007/REPORT.md")
    args = parser.parse_args()
    original = args.report.read_text(encoding="utf-8")
    result, count = suppress(original)
    if result != original:
        args.report.write_text(result, encoding="utf-8")
    print(f"Suppressed {count} intervals; estimates/raw data unchanged.")
