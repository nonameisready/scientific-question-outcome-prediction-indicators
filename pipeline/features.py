"""Question feature extraction: six groups, all computable at the cutoff.

A - text and linguistic structure          (rule-based, question text only)
B - evidence context                       (retrieval against pre-cutoff corpus)
C - novelty                                (semantic distance, concept pairs)
D - falsifiability / answerability         (rule-based + LLM structured coding)
E - tractability / resource availability   (facility lexicon, keyword rules)
F - scientific-environment controls        (subfield publication statistics)

Every feature uses only information available on or before the question's
cutoff date. Evidence features are defined identically for all generation
sources via retrieval, so models can compare sources fairly; tension-mined
questions additionally carry native evidence-record features.
"""

from __future__ import annotations

import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import corpus, facilities, mine
from .llm import GENERATION_MODEL, chat_json

# ---------------------------------------------------------------- group A

_CAUSAL = re.compile(
    r"\b(caus\w+|driv\w+|due to|origin|originat\w+|why|mechanis\w+|responsible for|lead(?:s|ing)? to|produce[sd]?)\b",
    re.I,
)
_COMPARISON = re.compile(
    r"\b(than|compar\w+|versus|vs\.?|relative to|higher|lower|larger|smaller|excess|deficit|differ\w+)\b",
    re.I,
)
_MECHANISM = re.compile(r"\b(mechanis\w+|process\w*|physical origin|channel|pathway)\b", re.I)
_MEASURABLE = re.compile(
    r"\b(abundance|mass(?:es)?|radius|radii|rate|ratio|fraction|temperature|luminosit\w+|metallicit\w+|density|flux|amplitude|frequency|period|distance|redshift|spin|magnitude|dispersion measure|equation of state|spectral (?:index|slope)|H_?0|sigma_?8)\b",
    re.I,
)
_ALTERNATIVES = re.compile(
    r"\bwhether\b.{3,80}\bor\b|\b(?:real|genuine|intrinsic)\b.{0,40}\bor\b.{0,60}\b(?:artifact|systematic|bias|spurious)\b|\bor (?:is|are|does|do|can|instead|rather)\b",
    re.I,
)
_FALSIFICATION = re.compile(
    r"\b(rule[sd]? out|distinguish\w*|test(?:able|ed|ing)?|confirm\w*|refute\w*|artifact|systematic error|discriminate|verif\w+)\b",
    re.I,
)
_YESNO = re.compile(r"^(is|are|does|do|can|could|has|have|will|did|was|were|should|must)\b", re.I)
_ENTITY = re.compile(
    r"\b(?:[A-Z]{2,}[- ]?\d[\dA-Za-z.+-]*|[A-Z][a-z]+ \d+[a-z]?\b|NGC ?\d+|HD ?\d+|GJ ?\d+|Kepler-\d+|TRAPPIST-1|WASP-\d+|HAT-P-\d+|GW\d{6}|FRB ?\d{6,8}|SN ?\d{4}\w*|PSR [BJ][\d+-]+)",
)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def text_features(question: str) -> dict:
    words = question.split()
    entities = set(_ENTITY.findall(question))
    n_numbers = len(_NUMBER.findall(question))
    feats = {
        "txt_n_words": len(words),
        "txt_n_entities": len(entities),
        "txt_n_numbers": n_numbers,
        "txt_has_causal": int(bool(_CAUSAL.search(question))),
        "txt_has_comparison": int(bool(_COMPARISON.search(question))),
        "txt_has_mechanism": int(bool(_MECHANISM.search(question))),
        "txt_has_measurable": int(bool(_MEASURABLE.search(question))),
        "txt_has_alternatives": int(bool(_ALTERNATIVES.search(question))),
        "txt_has_falsification": int(bool(_FALSIFICATION.search(question))),
        "txt_is_yesno": int(bool(_YESNO.search(question))),
    }
    feats["txt_specificity"] = (
        feats["txt_n_entities"]
        + feats["txt_has_measurable"]
        + feats["txt_has_falsification"]
        + min(n_numbers, 2)
    )
    return feats


# ---------------------------------------------------------- groups B and C


class CellContext:
    """Pre-cutoff retrieval context shared by all questions of one cell."""

    def __init__(self, past_docs: list[dict], cutoff_year: int):
        self.past_docs = past_docs
        self.cutoff_year = cutoff_year
        texts = [f"{d['title']}. {d['abstract']}" for d in past_docs] or ["empty"]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True)
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self.review_idx = [i for i, d in enumerate(past_docs) if corpus.is_reviewish(d["title"])]
        self.vocab_analyzer = self.vectorizer.build_analyzer()
        self._doc_terms: list[set[str]] | None = None

    def doc_terms(self) -> list[set[str]]:
        if self._doc_terms is None:
            self._doc_terms = [
                set(self.vocab_analyzer(f"{d['title']}. {d['abstract']}"))
                for d in self.past_docs
            ]
        return self._doc_terms

    def retrieve(self, question: str, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.doc_matrix)[0]
        top = np.argsort(-sims)[:k]
        return top, sims


def evidence_features(question: dict, ctx: CellContext) -> dict:
    top, sims = ctx.retrieve(question["question"], k=20)
    top10 = top[:10]
    docs10 = [ctx.past_docs[i] for i in top10]
    ages = [ctx.cutoff_year - d["year"] for d in docs10]
    instr = set()
    tension = []
    for d in docs10:
        text = f"{d['title']} {d['abstract']}"
        instr.update(facilities.find_facilities(text))
        tension.append(mine.tension_marker_count(text))
    feats = {
        "ev_top1_sim": float(sims[top[0]]) if len(top) else 0.0,
        "ev_top5_mean_sim": float(np.mean([sims[i] for i in top[:5]])) if len(top) else 0.0,
        "ev_n_similar": int(np.sum(sims >= 0.15)),
        "ev_n_instruments": len(instr),
        "ev_tension_density": float(np.mean(tension)) if tension else 0.0,
        "ev_age_mean": float(np.mean(ages)) if ages else 0.0,
        "ev_age_std": float(np.std(ages)) if ages else 0.0,
        "ev_recency_share": float(np.mean([a <= 2 for a in ages])) if ages else 0.0,
    }
    cluster = (question.get("source_material") or {}).get("cluster")
    if cluster:
        years = [int(c["published"][:4]) for c in cluster]
        feats["ev_native_n_papers"] = len({c["arxiv_id"] for c in cluster})
        feats["ev_native_year_spread"] = max(years) - min(years)
        feats["ev_has_native_record"] = 1
    else:
        feats["ev_native_n_papers"] = 0
        feats["ev_native_year_spread"] = 0
        feats["ev_has_native_record"] = 0
    return feats


def novelty_features(question: dict, ctx: CellContext) -> dict:
    top, sims = ctx.retrieve(question["question"], k=10)
    nn = float(sims[top[0]]) if len(top) else 0.0
    mean10 = float(np.mean([sims[i] for i in top])) if len(top) else 0.0
    review_sim = (
        float(max(sims[i] for i in ctx.review_idx)) if ctx.review_idx else 0.0
    )
    # Concept-pair novelty: the two most distinctive question terms, counted
    # jointly in the pre-cutoff corpus.
    qv = ctx.vectorizer.transform([question["question"]])
    arr = qv.toarray()[0]
    term_idx = np.argsort(-arr)[:2]
    names = ctx.vectorizer.get_feature_names_out()
    pair = [names[i] for i in term_idx if arr[i] > 0]
    pair_count = 0
    if len(pair) == 2:
        for terms in ctx.doc_terms():
            if pair[0] in terms and pair[1] in terms:
                pair_count += 1
    return {
        "nov_nn_sim": nn,
        "nov_semantic_distance": 1.0 - nn,
        "nov_mean_top10_sim": mean10,
        "nov_pair_doc_count": pair_count,
        "nov_new_pair": int(pair_count == 0 and len(pair) == 2),
        "nov_review_overlap": review_sim,
    }


# ---------------------------------------------------------------- group D


def falsifiability_rule_features(question: str) -> dict:
    return {
        "fal_outcome_variable": int(bool(_MEASURABLE.search(question))),
        "fal_competing_rule": int(bool(_ALTERNATIVES.search(question))),
        "fal_quant_threshold": int(bool(_NUMBER.search(question))),
        "fal_named_dataset": int(bool(facilities.find_facilities(question))),
    }


_CODING_PROMPT = (
    "For each numbered research question below, code four properties:\n"
    "- observable: does the question refer to something measurable with real data? (true/false)\n"
    "- competing_hypotheses: how many distinct competing explanations does the question "
    "explicitly mention? (integer, 0 if none)\n"
    "- quantitative_test: could the question be settled by a quantitative measurement "
    "with a clear threshold? (true/false)\n"
    "- falsification_path: is there a concrete way future evidence could prove the "
    "question's premise or expectation wrong? (true/false)\n\n"
    "{questions}\n\n"
    'Reply as JSON: {{"codes": [{{"observable": true, "competing_hypotheses": 0, '
    '"quantitative_test": false, "falsification_path": false}}, ...]}} with exactly '
    "one object per question, in order."
)


def falsifiability_llm_codes(questions: list[str], batch: int = 10) -> list[dict]:
    codes: list[dict] = []
    for i in range(0, len(questions), batch):
        chunk = questions[i : i + batch]
        listing = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(chunk))
        resp = chat_json(
            [{"role": "user", "content": _CODING_PROMPT.format(questions=listing)}],
            model=GENERATION_MODEL,
        )
        got = resp.get("codes", [])
        for j in range(len(chunk)):
            c = got[j] if j < len(got) else {}
            codes.append(
                {
                    "fal_observable": int(bool(c.get("observable", False))),
                    "fal_competing_llm": int(c.get("competing_hypotheses", 0) or 0),
                    "fal_quant_test": int(bool(c.get("quantitative_test", False))),
                    "fal_falsification_path": int(bool(c.get("falsification_path", False))),
                }
            )
    return codes


# ---------------------------------------------------------------- group E

_ARCHIVAL = re.compile(r"\b(archiv\w+|public(?:ly available)? data|existing (?:data|observations|surveys)|survey data|catalog)\b", re.I)
_LONGITUDINAL = re.compile(r"\b(long[- ]term|monitoring|multi[- ]epoch|decade|follow[- ]up over)\b", re.I)
_THEORY = re.compile(r"\b(simulat\w+|theoretical|models?|modeling|numerical)\b", re.I)
_SAMPLE = re.compile(r"\b(large sample|population|statistical(?:ly)?|ensemble|survey)\b", re.I)


def tractability_features(question: str, cutoff_year: int) -> dict:
    mentioned = facilities.find_facilities(question)
    operational = [f for f in mentioned if facilities.operational_at(f, cutoff_year)]
    future = [f for f in mentioned if f not in operational]
    return {
        "tra_n_facilities_mentioned": len(mentioned),
        "tra_n_facilities_operational": len(operational),
        "tra_mentions_future_facility": int(bool(future)),
        "tra_needs_new_instrument": int(bool(future) and not operational),
        "tra_archival_data": int(bool(_ARCHIVAL.search(question))),
        "tra_longitudinal": int(bool(_LONGITUDINAL.search(question))),
        "tra_needs_theory": int(bool(_THEORY.search(question))),
        "tra_sample_scale": int(bool(_SAMPLE.search(question))),
    }


# ---------------------------------------------------------------- driver


def compute_cell_features(
    questions: list[dict],
    past_docs: list[dict],
    stats: dict,
    subfield_key: str,
    cutoff_year: int,
) -> list[dict]:
    """Full feature records for all questions of one cell (groups A-C, E-F).

    Group D LLM codes are batched globally by the caller for efficiency.
    """
    ctx = CellContext(past_docs, cutoff_year)
    env = corpus.env_features(stats, subfield_key, cutoff_year)
    out = []
    for q in questions:
        feats: dict = {}
        feats.update(text_features(q["question"]))
        feats.update(evidence_features(q, ctx))
        feats.update(novelty_features(q, ctx))
        feats.update(falsifiability_rule_features(q["question"]))
        feats.update(tractability_features(q["question"], cutoff_year))
        feats.update(env)
        out.append(feats)
    return out
