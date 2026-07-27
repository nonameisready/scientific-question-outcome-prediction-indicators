"""Stage 2: generate the frozen question dataset for all cells.

40 cells (8 subfields x 5 cutoffs) x quota 25 (10 tension / 5 direct /
5 future-work / 5 controls), plus the 10 frozen Paper 1 evidence-graph
questions imported as an additional small source branch.

Produces data/questions/questions_v1.jsonl (committed, frozen).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import json
from pathlib import Path

from pipeline import corpus, generate, subfields

OUT = Path("data/questions/questions_v1.jsonl")
PAPER1 = Path("data/questions/astronomy_questions_v1.jsonl")


def import_paper1() -> list[dict]:
    if not PAPER1.exists():
        return []
    out = []
    for line in PAPER1.open():
        rec = json.loads(line)
        out.append(
            {
                "question_id": f"paper1-{rec['question_id']}",
                "question": rec["question"],
                "source": "paper1_evidence_graph",
                "control_subtype": None,
                "subfield": "exoplanet_atmospheres",
                "cutoff_year": 2020,
                "cutoff_date": rec.get("cutoff_date", "2020-12-31"),
                "source_material": {
                    "generation_system": rec.get("generation_system"),
                    "source_evidence_ids": rec.get("source_evidence_ids", []),
                },
            }
        )
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_questions: list[dict] = []
    counts: dict[str, dict] = {}
    for cy in corpus.CUTOFF_YEARS:
        date_from, date_to = corpus.past_window(cy)
        for sub in subfields.SUBFIELDS:
            past_docs = corpus.load_subfield_docs(sub.key, date_from, date_to)
            qs = generate.generate_cell(past_docs, sub.key, sub.name, cy)
            all_questions.extend(qs)
            by_src: dict[str, int] = {}
            for q in qs:
                by_src[q["source"]] = by_src.get(q["source"], 0) + 1
            counts[f"{sub.key}@{cy}"] = {"n_past_docs": len(past_docs), **by_src}
            print(
                f"[generate] {sub.key}@{cy}: past={len(past_docs)} questions={len(qs)} {by_src}",
                flush=True,
            )
    all_questions.extend(import_paper1())
    with OUT.open("w", encoding="utf-8") as fh:
        for q in all_questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    summary = Path("data/questions/generation_summary.json")
    summary.write_text(json.dumps({"total": len(all_questions), "cells": counts}, indent=2))
    print(f"[generate] wrote {len(all_questions)} questions -> {OUT}")


if __name__ == "__main__":
    main()
