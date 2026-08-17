"""Evidence-graph structure features and the learned structure prior.

Paper 2's outlook posed the question this module answers empirically:
*which structural patterns of pre-cutoff evidence most often lead to
questions the future answers, advances, or refutes?* Every tension-mined
question in the frozen dataset stores its generating evidence cluster
verbatim (sentences, arXiv ids, dates), so the structural signal can be
recomputed offline, transparently, and identically at every cutoff.

All features are derived from the frozen `source_material.cluster` records
only — nothing post-cutoff is touched — with the same transparent keyword
rules used elsewhere in the pipeline (subfields.py, facilities.py).
"""

from __future__ import annotations

import re

from .facilities import find_facilities

# Tension typology, mirroring the evidence-graph signal types of Papers 1-2:
# explicit cross-paper contradiction, unexplained/anomalous result,
# methodological challenge, and detection-vs-limit stance opposition.
_CONTRADICTION = re.compile(
    r"inconsisten|contradict|disagree|discrepan|conflict|at odds|tension|"
    r"incompatible|does not (?:match|agree)|in contrast to",
    re.IGNORECASE,
)
_UNEXPLAINED = re.compile(
    r"unexplained|puzzl|remains? (?:unclear|unknown|uncertain|elusive|"
    r"poorly (?:understood|constrained))|not (?:yet )?(?:understood|known)|"
    r"mysterious|anomal|surpris|unexpected|open question",
    re.IGNORECASE,
)
_METHOD = re.compile(
    r"systematic|artifact|bias|assumption|calibration|degenerac|"
    r"model[- ]dependent|underestimat|overestimat|contaminat|"
    r"limited statistical|caution",
    re.IGNORECASE,
)
_DETECTION = re.compile(
    r"\bdetect(?:ed|ion)?\b|\bconfirm|\bevidence (?:for|of)\b|\bwe find\b|"
    r"\bdiscover|\bmeasured?\b|\bpresence of\b",
    re.IGNORECASE,
)
_NULL = re.compile(
    r"no evidence|non[- ]detection|not detect|upper limit|absence of|"
    r"rule[sd]? out|null result|no significant|fail(?:s|ed)? to",
    re.IGNORECASE,
)
_SIGMA = re.compile(r"(\d+(?:\.\d+)?)\s*[- ]?(?:sigma|σ)", re.IGNORECASE)
_QUANT = re.compile(
    r"\d+(?:\.\d+)?\s*[- ]?(?:sigma|σ|%|per ?cent)|factor of\s*~?\s*\d",
    re.IGNORECASE,
)


def _year(published: str) -> int | None:
    m = re.match(r"(\d{4})", published or "")
    return int(m.group(1)) if m else None


def structure_features(question: dict) -> dict | None:
    """Structural features of one tension question's evidence cluster.

    Returns None for questions without a stored cluster (all non-tension
    sources), so callers can filter with a single truthiness check.
    """
    cluster = (question.get("source_material") or {}).get("cluster")
    if not cluster:
        return None

    sentences = [c.get("sentence", "") for c in cluster]
    years = [y for y in (_year(c.get("published", "")) for c in cluster) if y]
    cutoff = int(question["cutoff_year"])
    joined = " ".join(sentences)

    sigmas = [float(m) for m in _SIGMA.findall(joined)]
    facilities: set[str] = set()
    for s in sentences:
        facilities.update(find_facilities(s))

    has_detection = any(_DETECTION.search(s) for s in sentences)
    has_null = any(_NULL.search(s) for s in sentences)

    return {
        "question_id": question["question_id"],
        # -- graph shape
        "str_n_claims": len(cluster),
        "str_n_papers": len({c.get("arxiv_id") for c in cluster}),
        "str_year_spread": (max(years) - min(years)) if years else 0,
        "str_newest_age": (cutoff - max(years)) if years else 0,
        "str_oldest_age": (cutoff - min(years)) if years else 0,
        # -- tension typology (multi-label)
        "str_type_contradiction": int(bool(_CONTRADICTION.search(joined))),
        "str_type_unexplained": int(bool(_UNEXPLAINED.search(joined))),
        "str_type_method": int(bool(_METHOD.search(joined))),
        "str_type_stance": int(has_detection and has_null),
        # -- quantification of the tension
        "str_quantified": int(bool(_QUANT.search(joined))),
        "str_max_sigma": max(sigmas) if sigmas else 0.0,
        # -- instrument structure of the evidence
        "str_n_facilities": len(facilities),
        "str_cross_facility": int(len(facilities) >= 2),
    }


STRUCTURE_COLUMNS = [
    "str_n_claims",
    "str_n_papers",
    "str_year_spread",
    "str_newest_age",
    "str_oldest_age",
    "str_type_contradiction",
    "str_type_unexplained",
    "str_type_method",
    "str_type_stance",
    "str_quantified",
    "str_max_sigma",
    "str_n_facilities",
    "str_cross_facility",
]

TYPE_COLUMNS = [
    "str_type_contradiction",
    "str_type_unexplained",
    "str_type_method",
    "str_type_stance",
]
