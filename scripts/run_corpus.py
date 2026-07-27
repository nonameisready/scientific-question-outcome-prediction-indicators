"""Stage 1: process the raw harvest into the classified corpus + statistics.

Produces:
- data/raw/corpus_classified.jsonl   (local, reproducible, not committed)
- data/corpus/subfield_year_stats.json
- data/corpus/corpus_manifest.json   (committed provenance record)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import hashlib
import json
from datetime import date
from pathlib import Path

from pipeline import corpus, subfields


def main() -> None:
    n = corpus.build_processed()
    stats = corpus.build_stats()

    per_subfield_totals = {
        key: sum(stats["subfields"][key]["papers_by_year"].values())
        for key in subfields.SUBFIELD_KEYS
    }
    windows = {}
    for cy in corpus.CUTOFF_YEARS:
        windows[str(cy)] = {
            "cutoff_date": corpus.cutoff_date(cy),
            "past_window": corpus.past_window(cy),
            "future_window": corpus.future_window(cy),
        }

    sha = hashlib.sha256()
    with corpus.PROCESSED.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)

    manifest = {
        "built_on": str(date.today()),
        "source": "arXiv API (export.arxiv.org), categories astro-ph + 6 subcategories",
        "harvest_years": [2005, 2025],
        "n_papers_total": n,
        "corpus_sha256": sha.hexdigest(),
        "subfield_paper_totals": per_subfield_totals,
        "cutoff_windows": windows,
        "classification": {
            "method": "weighted keyword patterns on title+abstract, category bonus",
            "score_threshold": subfields.SCORE_THRESHOLD,
            "multi_membership": True,
        },
    }
    out = Path("data/corpus/corpus_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"n_papers": n, "subfield_totals": per_subfield_totals}, indent=2))


if __name__ == "__main__":
    main()
