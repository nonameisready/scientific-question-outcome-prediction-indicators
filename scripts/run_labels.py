"""Stage 4: tiered outcome labeling against the future corpus.

Produces:
- data/labels/labels_v1.jsonl          (tier 1 + tier 2, committed)
- data/labels/labels_second_judge.jsonl (tier 3 sample, committed)
- data/labels/judge_agreement.json      (reliability report)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline import analysis, corpus, labeling
from pipeline.llm import SECOND_JUDGE_MODEL

QUESTIONS = Path("data/questions/questions_v1.jsonl")
OUT = Path("data/labels/labels_v1.jsonl")
OUT2 = Path("data/labels/labels_second_judge.jsonl")
AGREEMENT = Path("data/labels/judge_agreement.json")

SECOND_JUDGE_FRACTION = 0.2


def main() -> None:
    questions = [json.loads(l) for l in QUESTIONS.open()]
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for q in questions:
        by_cell[(q["subfield"], q["cutoff_year"])].append(q)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []
    ctxs: dict[tuple[str, int], labeling.FutureContext] = {}
    for (subfield_key, cy), qs in sorted(by_cell.items()):
        date_from, date_to = corpus.future_window(cy)
        future_docs = corpus.load_subfield_docs(subfield_key, date_from, date_to)
        ctx = labeling.FutureContext(future_docs, cy)
        ctxs[(subfield_key, cy)] = ctx
        with ThreadPoolExecutor(max_workers=8) as pool:
            labels.extend(pool.map(lambda q: labeling.label_question(q, ctx), qs))
        print(
            f"[labels] {subfield_key}@{cy}: {len(qs)} questions, future corpus {len(future_docs)}",
            flush=True,
        )
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in labels:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[labels] wrote {len(labels)} labels -> {OUT}")

    # Tier 3: independent second judge on a stratified sample: all judged
    # premise_refuted cases + a random slice of everything else.
    rng = random.Random(7)
    by_id = {q["question_id"]: q for q in questions}
    judged = [r for r in labels if r["judge_model"] != "auto_threshold"]
    refuted = [r for r in judged if r["answer_status"] == "premise_refuted"]
    others = [r for r in judged if r["answer_status"] != "premise_refuted"]
    sample = refuted + rng.sample(others, min(len(others), int(len(labels) * SECOND_JUDGE_FRACTION)))
    def second_judge(rec: dict) -> dict:
        q = by_id[rec["question_id"]]
        ctx = ctxs[(q["subfield"], q["cutoff_year"])]
        id_to_doc = {d["id"]: d for d in ctx.future_docs}
        docs = [id_to_doc[i] for i in rec["top_ids"] if i in id_to_doc]
        judged2 = labeling.judge_question(
            q["question"], q["cutoff_date"], docs, model=SECOND_JUDGE_MODEL
        )
        return {"question_id": rec["question_id"], **judged2}

    with ThreadPoolExecutor(max_workers=8) as pool:
        second = list(pool.map(second_judge, sample))
    with OUT2.open("w", encoding="utf-8") as fh:
        for rec in second:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    agreement = analysis.judge_agreement(
        [r for r in labels if r["question_id"] in {s["question_id"] for s in second}],
        second,
    )
    AGREEMENT.write_text(json.dumps(agreement, indent=2))
    print(f"[labels] second judge n={len(second)}, agreement={json.dumps(agreement)[:200]}")


if __name__ == "__main__":
    main()
