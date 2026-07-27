"""Corpus assembly: dedup, subfield classification, temporal windows, stats.

Reads the raw arXiv harvest shards, deduplicates by arXiv id, classifies each
paper into subfields, and provides temporal-window access used everywhere
downstream. The processed corpus is cached as a single JSONL file with the
subfield assignment attached, plus a per-subfield-per-year statistics table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import subfields

RAW_DIR = Path("data/raw/arxiv")
PROCESSED = Path("data/raw/corpus_classified.jsonl")
SUBFIELD_DIR = Path("data/raw/subfields")
STATS_PATH = Path("data/corpus/subfield_year_stats.json")

CUTOFF_YEARS = (2012, 2014, 2016, 2018, 2020)
PAST_WINDOW_YEARS = 8
FUTURE_WINDOW_YEARS = 5

REVIEW_MARKERS = ("review", " survey of", "progress in", "lecture", "annual rev")


def cutoff_date(cutoff_year: int) -> str:
    return f"{cutoff_year}-12-31"


def past_window(cutoff_year: int) -> tuple[str, str]:
    return (f"{cutoff_year - PAST_WINDOW_YEARS + 1}-01-01", f"{cutoff_year}-12-31")


def future_window(cutoff_year: int) -> tuple[str, str]:
    return (f"{cutoff_year + 1}-01-01", f"{cutoff_year + FUTURE_WINDOW_YEARS}-12-31")


def is_reviewish(title: str) -> bool:
    t = title.lower()
    return any(m in t for m in REVIEW_MARKERS)


def build_processed(raw_dir: Path = RAW_DIR, out_path: Path = PROCESSED) -> int:
    """Merge shards, dedup, classify into subfields; returns record count.

    Also writes one JSONL per subfield (papers assigned to it, all years) so
    that per-cell corpus loads don't rescan the full corpus.
    """
    seen: set[str] = set()
    n_out = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SUBFIELD_DIR.mkdir(parents=True, exist_ok=True)
    sub_files = {
        key: (SUBFIELD_DIR / f"{key}.jsonl").open("w", encoding="utf-8")
        for key in subfields.SUBFIELD_KEYS
    }
    shards = sorted(raw_dir.glob("*.jsonl"))
    try:
        with out_path.open("w", encoding="utf-8") as out:
            for shard in shards:
                with shard.open(encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        if rec["id"] in seen or not rec.get("abstract"):
                            continue
                        seen.add(rec["id"])
                        scores = subfields.classify(
                            rec["title"], rec["abstract"], rec.get("categories", [])
                        )
                        rec["subfields"] = scores
                        rec["year"] = int(rec["published"][:4])
                        payload = json.dumps(rec, ensure_ascii=False) + "\n"
                        out.write(payload)
                        for key in scores:
                            sub_files[key].write(payload)
                        n_out += 1
    finally:
        for fh in sub_files.values():
            fh.close()
    return n_out


def iter_processed(path: Path = PROCESSED):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def load_subfield_docs(
    subfield_key: str,
    date_from: str,
    date_to: str,
) -> list[dict]:
    """All docs assigned to the subfield with date_from <= published <= date_to."""
    docs = []
    path = SUBFIELD_DIR / f"{subfield_key}.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if date_from <= rec["published"] <= date_to:
                docs.append(rec)
    docs.sort(key=lambda r: r["published"])
    return docs


def build_stats(path: Path = PROCESSED, out_path: Path = STATS_PATH) -> dict:
    """Per-subfield-per-year publication counts and review counts."""
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    reviews: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[int, int] = defaultdict(int)
    for rec in iter_processed(path):
        year = rec["year"]
        totals[year] += 1
        for key in rec["subfields"]:
            counts[key][year] += 1
            if is_reviewish(rec["title"]):
                reviews[key][year] += 1
    stats = {
        "total_by_year": dict(sorted(totals.items())),
        "subfields": {
            key: {
                "papers_by_year": dict(sorted(counts[key].items())),
                "reviews_by_year": dict(sorted(reviews[key].items())),
            }
            for key in subfields.SUBFIELD_KEYS
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    return stats


def env_features(stats: dict, subfield_key: str, cutoff_year: int) -> dict:
    """Field-environment features known at the cutoff (feature group F)."""
    by_year = {int(y): n for y, n in stats["subfields"][subfield_key]["papers_by_year"].items()}
    rev_by_year = {int(y): n for y, n in stats["subfields"][subfield_key]["reviews_by_year"].items()}
    total_by_year = {int(y): n for y, n in stats["total_by_year"].items()}

    def window_sum(table: dict[int, int], y0: int, y1: int) -> int:
        return sum(table.get(y, 0) for y in range(y0, y1 + 1))

    recent = window_sum(by_year, cutoff_year - 2, cutoff_year)
    earlier = window_sum(by_year, cutoff_year - 5, cutoff_year - 3)
    total_recent = window_sum(total_by_year, cutoff_year - 2, cutoff_year)

    sub = subfields.get_subfield(subfield_key)
    from . import facilities as fac

    gap = fac.next_launch_gap_years(sub.facilities, cutoff_year)
    return {
        "env_pub_volume_3y": recent,
        "env_pub_share_3y": recent / total_recent if total_recent else 0.0,
        "env_growth_ratio": (recent + 1) / (earlier + 1),
        "env_review_count_3y": window_sum(rev_by_year, cutoff_year - 2, cutoff_year),
        # Horizon capped at 10y: "no known upcoming facility" and "a decade
        # away" are equivalent for cutoff-time planning purposes.
        "env_years_to_next_facility": min(gap, 10.0) if gap is not None else 10.0,
        "env_facility_within_3y": 1 if gap is not None and gap <= 3 else 0,
    }
