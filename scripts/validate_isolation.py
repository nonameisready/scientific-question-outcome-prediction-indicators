"""Validate temporal isolation of the frozen question dataset.

Checks, for every question record:
1. every piece of source material (mined sentences, evidence ids) predates
   the question's cutoff;
2. label supporting ids, when resolvable to the corpus, postdate the cutoff;
3. every question carries the required frozen metadata.

Exits non-zero on any violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import json
import sys
from pathlib import Path

QUESTIONS = Path("data/questions/questions_v1.jsonl")
LABELS = Path("data/labels/labels_v1.jsonl")
CORPUS = Path("data/raw/corpus_classified.jsonl")


def main() -> int:
    problems: list[str] = []
    questions = [json.loads(l) for l in QUESTIONS.open()]
    for q in questions:
        cutoff = q["cutoff_date"]
        for key in ("question_id", "question", "source", "subfield", "cutoff_year"):
            if not q.get(key):
                problems.append(f"{q.get('question_id', '?')}: missing {key}")
        material = q.get("source_material") or {}
        dated: list[tuple[str, str]] = []
        for item in material.get("cluster", []):
            dated.append((item["arxiv_id"], item["published"]))
        if "statement" in material:
            dated.append((material["statement"]["arxiv_id"], material["statement"]["published"]))
        for arxiv_id, published in dated:
            if published > cutoff:
                problems.append(
                    f"{q['question_id']}: source {arxiv_id} published {published} > cutoff {cutoff}"
                )

    if LABELS.exists() and CORPUS.exists():
        dates = {}
        for line in CORPUS.open():
            rec = json.loads(line)
            dates[rec["id"]] = rec["published"]
        by_id = {q["question_id"]: q for q in questions}
        for line in LABELS.open():
            rec = json.loads(line)
            q = by_id.get(rec["question_id"])
            if not q:
                problems.append(f"label for unknown question {rec['question_id']}")
                continue
            for sid in rec.get("supporting_ids", []):
                pub = dates.get(sid)
                if pub is not None and pub <= q["cutoff_date"]:
                    problems.append(
                        f"{rec['question_id']}: supporting id {sid} ({pub}) not after cutoff"
                    )

    if problems:
        print(f"FAILED: {len(problems)} temporal-isolation problems")
        for p in problems[:40]:
            print(" -", p)
        return 1
    print(f"OK: {len(questions)} questions pass temporal-isolation checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
