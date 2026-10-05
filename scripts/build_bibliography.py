"""Validate source shards and build a deduplicated catalogue and BibTeX.

Uses only the Python standard library. Run from any directory.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "bibliography"
SHARDS = ("parameters.json", "optimizer_targets.json", "data_selection.json", "additional.json", "core.json", "followup.json")
REQUIRED = {"id", "title", "authors", "year", "venue", "url", "verified_level"}


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.casefold())


def bib_value(value: str) -> str:
    # Preserve author accents in UTF-8; protect BibTeX metacharacters.
    return value.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main() -> None:
    catalogue: dict[str, dict] = {}
    ids: dict[str, str] = {}
    for filename in SHARDS:
        rows = json.loads((BIB / filename).read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError(f"{filename}: expected a list")
        for row in rows:
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"{filename}: missing {sorted(missing)}")
            if not isinstance(row["authors"], list) or not row["authors"]:
                raise ValueError(f"{row['id']}: empty author list")
            if row["verified_level"] not in {"abstract", "full_text"}:
                raise ValueError(f"{row['id']}: unknown verification level")
            if not isinstance(row["year"], int) or not 1900 <= row["year"] <= 2026:
                raise ValueError(f"{row['id']}: invalid year")
            if urlparse(row["url"]).scheme != "https":
                raise ValueError(f"{row['id']}: expected HTTPS primary source")
            key = title_key(row["title"])
            if row["id"] in ids and ids[row["id"]] != key:
                raise ValueError(f"Duplicate ID for distinct works: {row['id']}")
            ids[row["id"]] = key
            if key not in catalogue:
                catalogue[key] = dict(row, source_files=[filename], aliases=[])
            else:
                saved = catalogue[key]
                if saved["year"] != row["year"]:
                    raise ValueError(f"Conflicting publication years: {row['title']}")
                saved["source_files"].append(filename)
                if row["id"] != saved["id"]:
                    saved["aliases"].append(row["id"])
                for field, value in row.items():
                    if field not in saved:
                        saved[field] = value
                if row["verified_level"] == "full_text":
                    saved["verified_level"] = "full_text"
            catalogue[key].setdefault("verified_at", "2026-10-05")
    papers = sorted(catalogue.values(), key=lambda r: (r["year"], r["title"].casefold()))
    (BIB / "papers.json").write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # @misc avoids fabricating booktitle/journal details from shorthand venue strings.
    entries = []
    for paper in papers:
        fields = {
            "title": "{" + bib_value(paper["title"]) + "}",
            "author": " and ".join(bib_value(name) for name in paper["authors"]),
            "year": str(paper["year"]),
            "howpublished": bib_value(paper["venue"]),
            "url": paper["url"],
            "note": "Metadata checked 2026-10-05; " + paper["verified_level"],
        }
        if paper.get("doi"):
            fields["doi"] = paper["doi"]
        lines = [f"@misc{{{paper['id']},"]
        lines.extend(f"  {key} = {{{value}}}," for key, value in fields.items())
        entries.append("\n".join(lines) + "\n}")
    (BIB / "references.bib").write_text("% Working bibliography; verify publisher fields before thesis submission.\n\n" + "\n\n".join(entries) + "\n", encoding="utf-8")
    levels = Counter(p["verified_level"] for p in papers)
    print(f"Built {len(papers)} unique works from {sum(len(json.loads((BIB / f).read_text(encoding='utf-8-sig'))) for f in SHARDS)} records.")
    print("Verification:", dict(levels))


if __name__ == "__main__":
    main()
