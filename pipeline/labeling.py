"""Tiered outcome labeling against the temporally isolated future corpus.

Tier 1 (automatic candidates): for each question, retrieve future papers
published in (cutoff, cutoff + 5y] from the question's subfield corpus and
compute relevance counts, first-follow-up dates, and lead times.

Tier 2 (blinded LLM outcome judge): for questions with any plausible future
evidence, a judge that sees ONLY the retrieved future records (never the
question's generation source) assigns a structured outcome label with
supporting ids, rationale, and confidence.

Tier 3 (independent second judge): a different model re-judges a stratified
sample; agreement and Cohen's kappa are reported as label-reliability
estimates.

Outcome record per question (design section 7):
    addressed, answer_status, premise_status, future_paper_count,
    first_follow_up_date, lead_time_months, review_recognition.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import corpus
from .llm import JUDGE_MODEL, chat_json

REL_THRESHOLD = 0.18  # cosine similarity for "relevant future paper"
WEAK_THRESHOLD = 0.12  # below this for all docs -> auto not_addressed
TOP_K_JUDGE = 8

ANSWER_STATUSES = (
    "answered",
    "partially_addressed",
    "posed_but_open",
    "premise_refuted",
    "not_addressed",
)


class FutureContext:
    """Future-corpus retrieval context for one (subfield, cutoff) cell."""

    def __init__(self, future_docs: list[dict], cutoff_year: int):
        self.future_docs = future_docs
        self.cutoff_year = cutoff_year
        texts = [f"{d['title']}. {d['abstract']}" for d in future_docs] or ["empty"]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True)
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, question: str, k: int = 50) -> tuple[np.ndarray, np.ndarray]:
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.doc_matrix)[0]
        top = np.argsort(-sims)[:k]
        return top, sims


def auto_labels(question: str, ctx: FutureContext) -> dict:
    top, sims = ctx.retrieve(question)
    relevant = [i for i in top if sims[i] >= REL_THRESHOLD]
    weak = [i for i in top if sims[i] >= WEAK_THRESHOLD]
    rel_docs = sorted((ctx.future_docs[i] for i in relevant), key=lambda d: d["published"])
    first_date = rel_docs[0]["published"] if rel_docs else None
    lead_months = None
    if first_date:
        cutoff_y = ctx.cutoff_year
        y, m = int(first_date[:4]), int(first_date[5:7])
        lead_months = (y - cutoff_y - 1) * 12 + m
    return {
        "future_paper_count": len(relevant),
        "future_paper_count_weak": len(weak),
        "first_follow_up_date": first_date,
        "lead_time_months": lead_months,
        "review_recognition": any(corpus.is_reviewish(d["title"]) for d in rel_docs),
        "top_ids": [ctx.future_docs[i]["id"] for i in top[:TOP_K_JUDGE]],
        "top_sims": [float(sims[i]) for i in top[:TOP_K_JUDGE]],
    }


_JUDGE_SYSTEM = (
    "You are a strict outcome judge for a study of scientific questions. You will "
    "see a research question frozen at a historical cutoff date, and candidate "
    "papers published AFTER that cutoff. Judge only from the candidate papers "
    "given: do not use any other knowledge of the literature, and do not guess "
    "who or what generated the question. Be strict: a paper from the same "
    "subfield is usually only topically ADJACENT — it engages the question "
    "DIRECTLY only if it investigates the question's own central objects, "
    "quantities, or claimed relationship, or reports evidence that answers or "
    "undermines the question itself."
)

_JUDGE_TEMPLATE = (
    "Question (frozen at {cutoff}): {question}\n\n"
    "Candidate future papers (published after the cutoff):\n{evidence}\n\n"
    "Step 1 — classify EVERY candidate id as:\n"
    '  "direct"    = investigates this question itself (its specific objects, '
    "quantities, or claimed relationship), or reports evidence bearing on its answer\n"
    '  "adjacent"  = same topic or subfield, but does not investigate this question\n'
    '  "unrelated" = neither\n'
    "Step 2 — outcome, using ONLY the direct candidates:\n"
    '  "not_addressed"       = no direct candidate\n'
    '  "posed_but_open"      = direct candidates pose or flag the question as open, '
    "without substantive new evidence\n"
    '  "partially_addressed" = >=1 direct candidate contributes substantive new '
    "evidence on part of the question, but it is not settled\n"
    '  "answered"            = direct candidates report a resolution that would '
    "reasonably settle the main question\n"
    '  "premise_refuted"     = direct candidates show the question\'s underlying '
    "premise or assumption was wrong\n\n"
    "Output JSON with:\n"
    '- "per_candidate": [{{"id": "...", "relation": "direct|adjacent|unrelated"}}, ...]\n'
    '- "addressed": true only if at least one candidate is direct AND the outcome '
    "is not not_addressed\n"
    '- "answer_status": one of the five outcomes above\n'
    '- "premise_status": one of "intact", "weakened", "refuted", "not_applicable"\n'
    '- "supporting_ids": the direct candidate ids only (may be empty)\n'
    '- "rationale": 1-2 sentences citing those ids\n'
    '- "confidence": 0.0-1.0'
)


def judge_question(
    question: str, cutoff_date: str, evidence_docs: list[dict], model: str = JUDGE_MODEL
) -> dict:
    evidence = "\n".join(
        f"[{d['id']}, {d['published']}] {d['title']} — {d['abstract'][:600]}"
        for d in evidence_docs
    )
    resp = chat_json(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _JUDGE_TEMPLATE.format(
                    cutoff=cutoff_date, question=question, evidence=evidence
                ),
            },
        ],
        model=model,
        max_tokens=700,
    )
    status = str(resp.get("answer_status", "not_addressed"))
    if status not in ANSWER_STATUSES:
        status = "not_addressed"
    per_candidate = resp.get("per_candidate", []) or []
    n_direct = sum(1 for c in per_candidate if str(c.get("relation")) == "direct")
    n_adjacent = sum(1 for c in per_candidate if str(c.get("relation")) == "adjacent")
    # addressed is derived, not taken on faith: it requires at least one
    # direct candidate and a non-null outcome.
    if n_direct == 0 and per_candidate:
        status = "not_addressed"
    addressed = status != "not_addressed" and (n_direct > 0 or not per_candidate)
    return {
        "addressed": addressed,
        "answer_status": status if addressed else "not_addressed",
        "n_direct": n_direct,
        "n_adjacent": n_adjacent,
        "premise_status": str(resp.get("premise_status", "not_applicable")),
        "supporting_ids": [str(x) for x in resp.get("supporting_ids", [])][:TOP_K_JUDGE],
        "rationale": str(resp.get("rationale", ""))[:800],
        "confidence": float(resp.get("confidence", 0.0) or 0.0),
        "judge_model": model,
    }


def label_question(question: dict, ctx: FutureContext) -> dict:
    """Tier 1 + Tier 2 label for one question."""
    auto = auto_labels(question["question"], ctx)
    max_sim = auto["top_sims"][0] if auto["top_sims"] else 0.0
    if max_sim < WEAK_THRESHOLD:
        judged = {
            "addressed": False,
            "answer_status": "not_addressed",
            "n_direct": 0,
            "n_adjacent": 0,
            "premise_status": "not_applicable",
            "supporting_ids": [],
            "rationale": "No future paper reached the minimum relevance threshold.",
            "confidence": 0.9,
            "judge_model": "auto_threshold",
        }
    else:
        id_to_doc = {d["id"]: d for d in ctx.future_docs}
        docs = [id_to_doc[i] for i in auto["top_ids"] if i in id_to_doc]
        judged = judge_question(question["question"], question["cutoff_date"], docs)
    record = {"question_id": question["question_id"], **auto, **judged}
    return record
