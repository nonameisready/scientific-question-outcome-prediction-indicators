"""Harvest arXiv astro-ph metadata via the public arXiv API.

Downloads per-category, per-year metadata shards (id, title, abstract,
submission date, categories) for all astro-ph papers between START_YEAR and
END_YEAR. Shards are written as JSONL files and the harvest is resumable at
category-year granularity: a shard is only renamed into place once the full
year has been paginated through.

Usage:
    python -m pipeline.harvest --out data/raw/arxiv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

# astro-ph had no subcategories before 2007; both forms are harvested and
# merged (deduplicated by arXiv id) downstream.
CATEGORIES = [
    "astro-ph",
    "astro-ph.CO",
    "astro-ph.EP",
    "astro-ph.GA",
    "astro-ph.HE",
    "astro-ph.IM",
    "astro-ph.SR",
]

START_YEAR = 2005
END_YEAR = 2025

PAGE_SIZE = 1000
REQUEST_DELAY_S = 3.5
MAX_RETRIES = 6


RATE_LIMIT_BACKOFF_S = 180.0


def _fetch(url: str) -> bytes:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=45) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            last_err = err
            if err.code == 429:
                # arXiv throttling bans last minutes, not seconds: wait it out.
                print(f"[harvest] 429 rate-limited, backing off (attempt {attempt + 1})", flush=True)
                time.sleep(RATE_LIMIT_BACKOFF_S * (attempt + 1))
            else:
                time.sleep(REQUEST_DELAY_S * (attempt + 1))
        except Exception as err:  # noqa: BLE001 - network flakiness is expected
            last_err = err
            time.sleep(REQUEST_DELAY_S * (attempt + 1))
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {url}") from last_err


def _parse_page(payload: bytes) -> tuple[int, list[dict]]:
    root = ET.fromstring(payload)
    total_el = root.find(f"{OPENSEARCH}totalResults")
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    records = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        title = re.sub(r"\s+", " ", entry.findtext(f"{ATOM}title") or "").strip()
        abstract = re.sub(r"\s+", " ", entry.findtext(f"{ATOM}summary") or "").strip()
        published = (entry.findtext(f"{ATOM}published") or "")[:10]
        primary = entry.find(f"{ARXIV}primary_category")
        primary_cat = primary.get("term") if primary is not None else ""
        cats = [c.get("term") for c in entry.findall(f"{ATOM}category")]
        records.append(
            {
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "published": published,
                "primary_category": primary_cat,
                "categories": cats,
            }
        )
    return total, records


# The arXiv API becomes unreliable for offsets past a few thousand, so any
# window bigger than this is recursively split into sub-windows.
PAGINATION_SAFE = 4000

_MONTH_BOUNDS = [f"{m:02d}" for m in range(1, 13)]


def _window_query(category: str, date_from: str, date_to: str) -> str:
    return f"cat:{category} AND submittedDate:[{date_from} TO {date_to}]"


def _probe_total(query: str) -> int:
    params = urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": 1}
    )
    total, _ = _parse_page(_fetch(f"{API_URL}?{params}"))
    return total


def _harvest_window(fh, seen: set[str], query: str, total: int) -> int:
    """Paginate one (safe-sized) query window into the shard file."""
    written = 0
    start = 0
    stall = 0
    while start < total:
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": PAGE_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
        )
        payload = _fetch(f"{API_URL}?{params}")
        _, records = _parse_page(payload)
        if not records:
            if stall < MAX_RETRIES:
                stall += 1
                time.sleep(REQUEST_DELAY_S * (stall + 1))
                continue
            break
        stall = 0
        for rec in records:
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
        start += len(records)
        time.sleep(REQUEST_DELAY_S)
    return written


def harvest_category_year(category: str, year: int, out_dir: Path) -> int:
    """Harvest one category-year shard; returns number of records written."""
    shard = out_dir / f"{category.replace('.', '_')}_{year}.jsonl"
    if shard.exists():
        return sum(1 for _ in shard.open())
    tmp = shard.with_suffix(".jsonl.tmp")
    seen: set[str] = set()
    written = 0
    with tmp.open("w", encoding="utf-8") as fh:
        year_query = _window_query(
            category, f"{year}01010000", f"{year + 1}01010000"
        )
        total = _probe_total(year_query)
        time.sleep(REQUEST_DELAY_S)
        if total <= PAGINATION_SAFE:
            written += _harvest_window(fh, seen, year_query, total)
        else:
            for i, month in enumerate(_MONTH_BOUNDS):
                next_from = (
                    f"{year + 1}01010000"
                    if month == "12"
                    else f"{year}{_MONTH_BOUNDS[i + 1]}010000"
                )
                mq = _window_query(category, f"{year}{month}010000", next_from)
                m_total = _probe_total(mq)
                time.sleep(REQUEST_DELAY_S)
                written += _harvest_window(fh, seen, mq, m_total)
    tmp.rename(shard)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/arxiv"))
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for year in range(args.start_year, args.end_year + 1):
        for category in CATEGORIES:
            # Plain astro-ph stopped accepting new submissions in 2008; skip
            # the empty queries to save API calls.
            if category == "astro-ph" and year >= 2009:
                continue
            if category != "astro-ph" and year < 2007:
                continue
            try:
                n = harvest_category_year(category, year, args.out)
            except Exception as err:  # noqa: BLE001 - keep harvesting other shards
                print(f"[harvest] {category} {year}: FAILED ({err})", flush=True)
                continue
            grand_total += n
            print(f"[harvest] {category} {year}: {n} records", flush=True)
            time.sleep(REQUEST_DELAY_S)
    print(f"[harvest] done, {grand_total} records total (pre-dedup)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
