"""Stage 3: compute the six feature groups for every frozen question.

Produces data/features/features_v1.jsonl (committed), one record per
question_id containing all cutoff-time features (groups A-F).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import json
from collections import defaultdict
from pathlib import Path

from pipeline import corpus, features

QUESTIONS = Path("data/questions/questions_v1.jsonl")
OUT = Path("data/features/features_v1.jsonl")


def main() -> None:
    questions = [json.loads(l) for l in QUESTIONS.open()]
    stats = json.loads(corpus.STATS_PATH.read_text())

    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for q in questions:
        by_cell[(q["subfield"], q["cutoff_year"])].append(q)

    records: dict[str, dict] = {}
    for (subfield_key, cy), qs in sorted(by_cell.items()):
        date_from, date_to = corpus.past_window(cy)
        past_docs = corpus.load_subfield_docs(subfield_key, date_from, date_to)
        feats = features.compute_cell_features(qs, past_docs, stats, subfield_key, cy)
        for q, f in zip(qs, feats):
            records[q["question_id"]] = f
        print(f"[features] {subfield_key}@{cy}: {len(qs)} questions", flush=True)

    # Group D LLM structured coding, batched over all questions at once.
    codes = features.falsifiability_llm_codes([q["question"] for q in questions])
    for q, c in zip(questions, codes):
        records[q["question_id"]].update(c)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(
                json.dumps(
                    {"question_id": q["question_id"], **records[q["question_id"]]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[features] wrote {len(questions)} feature records -> {OUT}")


if __name__ == "__main__":
    main()
