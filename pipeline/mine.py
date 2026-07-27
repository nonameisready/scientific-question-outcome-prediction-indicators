"""Mining of tension statements and open-problem statements from abstracts.

Two sentence-level miners over pre-cutoff abstracts:

- Tension statements: sentences reporting contradiction, discrepancy, or
  unexplained results — the raw material for evidence-tension questions.
- Future-work statements: sentences in which the authors themselves flag an
  open problem ("remains unclear", "further observations are needed") — the
  raw material for paper-derived questions.

Both return (arxiv_id, published, sentence) records and never look past the
cutoff, so all mined material is available at question-generation time.
"""

from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

TENSION_PATTERNS = [
    r"\bin tension with\b",
    r"\binconsistent with\b",
    r"\bin conflict with\b",
    r"\bcontradict\w*\b",
    r"\bdiscrepan\w+\b",
    r"\bat odds with\b",
    r"\bcannot (?:easily )?be explained\b",
    r"\bcannot account for\b",
    r"\bunexpected(?:ly)?\b",
    r"\banomalous(?:ly)?\b",
    r"\bpuzzl\w+\b",
    r"\bchallenges? (?:the|our|current|standard)\b",
    r"\bdisagree\w*\b",
    r"\bdiffers? significantly from\b",
    r"\bin contrast to (?:previous|earlier|prior)\b",
    r"\bfail(?:s|ed)? to reproduce\b",
    r"\btension between\b",
]

FUTURE_WORK_PATTERNS = [
    r"\bremains? (?:unclear|unknown|uncertain|elusive|poorly (?:understood|constrained)|an? open (?:question|problem)|to be (?:determined|seen|understood|established))\b",
    r"\bopen (?:question|problem|issue)s?\b",
    r"\b(?:is|are) (?:still|not yet|yet to be) (?:known|understood|determined|established|clear|resolved|constrained)\b",
    r"\bpoorly (?:understood|constrained|known)\b",
    r"\bfurther (?:observations?|stud(?:y|ies)|work|investigations?|data) (?:is|are|will be|would be) (?:required|needed|necessary)\b",
    r"\bfuture (?:observations?|surveys?|missions?|work|studies) (?:will|should|could|may|are needed)\b",
    r"\bawaits? (?:confirmation|further)\b",
    r"\bunanswered\b",
    r"\bdebated?\b",
    r"\bunder debate\b",
    r"\bnot (?:yet )?(?:well[- ])?(?:understood|constrained|explained|established)\b",
]

_TENSION_RX = [re.compile(p, re.IGNORECASE) for p in TENSION_PATTERNS]
_FUTURE_RX = [re.compile(p, re.IGNORECASE) for p in FUTURE_WORK_PATTERNS]

MIN_SENT_LEN = 60
MAX_SENT_LEN = 420


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _mine(docs: list[dict], patterns: list[re.Pattern]) -> list[dict]:
    out = []
    for doc in docs:
        for sent in split_sentences(doc["abstract"]):
            if not MIN_SENT_LEN <= len(sent) <= MAX_SENT_LEN:
                continue
            matched = [rx.pattern for rx in patterns if rx.search(sent)]
            if matched:
                out.append(
                    {
                        "arxiv_id": doc["id"],
                        "published": doc["published"],
                        "title": doc["title"],
                        "sentence": sent,
                        "n_markers": len(matched),
                    }
                )
    return out


def mine_tension(docs: list[dict]) -> list[dict]:
    return _mine(docs, _TENSION_RX)


def mine_future_work(docs: list[dict]) -> list[dict]:
    return _mine(docs, _FUTURE_RX)


def tension_marker_count(text: str) -> int:
    """Number of tension markers present in a text (used as a feature)."""
    return sum(1 for rx in _TENSION_RX if rx.search(text))
