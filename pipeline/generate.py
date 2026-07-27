"""Question generation: four sources per (cutoff, subfield) cell.

Per design, each cell receives a fixed quota so that the dataset is balanced
across cutoffs, subfields, and generation methods:

- evidence_tension (10): clusters of tension statements mined from >=2
  distinct pre-cutoff papers, phrased into one precise question by an LLM
  that sees only the mined sentences.
- direct_llm (5): the baseline — an LLM asked directly for the most
  important unanswered questions given a sample of pre-cutoff titles.
- future_work (5): open-problem statements written by the papers' own
  authors, faithfully converted into interrogative form.
- control (5): weak/negative controls — random claim pairings, vague
  questions, untestable questions, and unsupported consensus challenges —
  phrased by the same LLM so that style alone cannot separate them.

All source material is strictly pre-cutoff. Records keep full provenance
(the exact sentences, papers, and prompts behind every question).
"""

from __future__ import annotations

import json
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import mine
from .llm import GENERATION_MODEL, chat_json

QUOTA = {"evidence_tension": 10, "direct_llm": 5, "future_work": 5, "control": 5}

_SYSTEM = (
    "You are assisting with the construction of a historical research dataset. "
    "The current date is {cutoff}. You must behave as if you only know the "
    "scientific literature published on or before {cutoff}. Never mention "
    "results, discoveries, or publications after {cutoff}."
)


def _clean_question(text: str) -> str:
    q = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", text.strip())
    q = re.sub(r"\s+", " ", q)
    if q and not q.endswith("?"):
        q += "?"
    return q


def _dedupe(questions: list[dict], threshold: float = 0.75) -> list[dict]:
    """Drop near-duplicate questions (TF-IDF cosine) keeping earlier entries."""
    if len(questions) < 2:
        return questions
    texts = [q["question"] for q in questions]
    vec = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sims = cosine_similarity(vec)
    keep: list[int] = []
    for i in range(len(texts)):
        if all(sims[i, j] < threshold for j in keep):
            keep.append(i)
    return [questions[i] for i in keep]


# ---------------------------------------------------------------- source 1


def build_tension_clusters(
    tension_sents: list[dict], max_clusters: int = 14, sim_threshold: float = 0.12
) -> list[list[dict]]:
    """Group tension sentences into clusters spanning >=2 distinct papers."""
    if len(tension_sents) < 2:
        return []
    texts = [f"{s['title']} {s['sentence']}" for s in tension_sents]
    vec = TfidfVectorizer(stop_words="english", max_features=20000).fit_transform(texts)
    sims = cosine_similarity(vec)
    order = sorted(
        range(len(tension_sents)),
        key=lambda i: (-tension_sents[i]["n_markers"], tension_sents[i]["published"]),
    )
    used: set[int] = set()
    clusters: list[list[dict]] = []
    for focal in order:
        if focal in used or len(clusters) >= max_clusters:
            continue
        neighbors = sorted(
            (j for j in range(len(tension_sents)) if j != focal and j not in used),
            key=lambda j: -sims[focal, j],
        )
        members = [focal]
        papers = {tension_sents[focal]["arxiv_id"]}
        for j in neighbors:
            if sims[focal, j] < sim_threshold or len(members) >= 4:
                break
            if tension_sents[j]["arxiv_id"] in papers:
                continue
            members.append(j)
            papers.add(tension_sents[j]["arxiv_id"])
        if len(papers) >= 2:
            used.update(members)
            clusters.append([tension_sents[i] for i in members])
    return clusters


def generate_tension_questions(
    clusters: list[list[dict]], subfield_name: str, cutoff: str, n: int
) -> list[dict]:
    out = []
    for cluster in clusters:
        if len(out) >= n:
            break
        evidence = "\n".join(
            f"- [{s['arxiv_id']}, {s['published']}] {s['sentence']}" for s in cluster
        )
        resp = chat_json(
            [
                {"role": "system", "content": _SYSTEM.format(cutoff=cutoff)},
                {
                    "role": "user",
                    "content": (
                        f"The following statements from different {subfield_name} papers "
                        f"(all published before {cutoff}) report tensions, discrepancies, "
                        "or unexplained results:\n\n"
                        f"{evidence}\n\n"
                        "Write ONE precise, specific research question that this evidence "
                        "tension raises. The question must be grounded ONLY in the "
                        "statements above, name the concrete quantities or objects "
                        "involved, and be answerable in principle by future observations "
                        "or analysis. Reply as JSON: {\"question\": \"...\"}"
                    ),
                },
            ],
            model=GENERATION_MODEL,
        )
        q = _clean_question(str(resp.get("question", "")))
        if len(q) < 25:
            continue
        out.append(
            {
                "question": q,
                "source": "evidence_tension",
                "source_material": {
                    "cluster": [
                        {k: s[k] for k in ("arxiv_id", "published", "sentence")}
                        for s in cluster
                    ]
                },
            }
        )
    return out


# ---------------------------------------------------------------- source 2


def generate_direct_questions(
    past_docs: list[dict], subfield_name: str, cutoff: str, n: int, rng: random.Random
) -> list[dict]:
    docs = past_docs if len(past_docs) <= 60 else rng.sample(past_docs, 60)
    docs = sorted(docs, key=lambda d: d["published"])
    titles = "\n".join(f"- ({d['published'][:4]}) {d['title']}" for d in docs)
    resp = chat_json(
        [
            {"role": "system", "content": _SYSTEM.format(cutoff=cutoff)},
            {
                "role": "user",
                "content": (
                    f"Here is a sample of the {subfield_name} literature published "
                    f"before {cutoff}:\n\n{titles}\n\n"
                    f"Based only on the state of the field at {cutoff}, what are the "
                    f"{n} most important unanswered research questions in "
                    f"{subfield_name}? Each question must be specific and concrete. "
                    'Reply as JSON: {"questions": ["...", "..."]}'
                ),
            },
        ],
        model=GENERATION_MODEL,
    )
    out = []
    for raw in resp.get("questions", [])[:n]:
        q = _clean_question(str(raw))
        if len(q) >= 25:
            out.append(
                {
                    "question": q,
                    "source": "direct_llm",
                    "source_material": {"n_titles_shown": len(docs)},
                }
            )
    return out


# ---------------------------------------------------------------- source 3


def pick_future_work_sentences(
    fw_sents: list[dict], n: int, rng: random.Random
) -> list[dict]:
    """Diverse selection: rank by marker count + recency, avoid near-dup topics."""
    if not fw_sents:
        return []
    ranked = sorted(fw_sents, key=lambda s: (-s["n_markers"], s["published"]))
    ranked = ranked[: max(60, n * 8)]
    rng.shuffle(ranked)
    texts = [s["sentence"] for s in ranked]
    vec = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sims = cosine_similarity(vec)
    picked: list[int] = []
    for i in range(len(ranked)):
        if len(picked) >= n:
            break
        if all(sims[i, j] < 0.35 for j in picked):
            picked.append(i)
    return [ranked[i] for i in picked]


def generate_future_work_questions(
    fw_sents: list[dict], subfield_name: str, cutoff: str, n: int, rng: random.Random
) -> list[dict]:
    chosen = pick_future_work_sentences(fw_sents, n, rng)
    if not chosen:
        return []
    listing = "\n".join(
        f"{i + 1}. [{s['arxiv_id']}, {s['published']}] {s['sentence']}"
        for i, s in enumerate(chosen)
    )
    resp = chat_json(
        [
            {"role": "system", "content": _SYSTEM.format(cutoff=cutoff)},
            {
                "role": "user",
                "content": (
                    f"Each numbered statement below is taken from a {subfield_name} "
                    f"paper published before {cutoff}, where the authors point out an "
                    "open problem:\n\n"
                    f"{listing}\n\n"
                    "Convert EACH statement into one research question. Stay strictly "
                    "faithful to what the statement says is open or unknown — do not "
                    "add information that is not in the statement. Reply as JSON: "
                    '{"questions": ["...", "..."]} with exactly one question per '
                    "statement, in order."
                ),
            },
        ],
        model=GENERATION_MODEL,
    )
    out = []
    for s, raw in zip(chosen, resp.get("questions", [])):
        q = _clean_question(str(raw))
        if len(q) >= 25:
            out.append(
                {
                    "question": q,
                    "source": "future_work",
                    "source_material": {
                        "statement": {k: s[k] for k in ("arxiv_id", "published", "sentence")}
                    },
                }
            )
    return out


# ---------------------------------------------------------------- source 4


CONTROL_SUBTYPES = [
    "random_pair",
    "random_pair",
    "vague",
    "untestable",
    "consensus_challenge",
]


def generate_control_questions(
    past_docs: list[dict], subfield_name: str, cutoff: str, rng: random.Random
) -> list[dict]:
    if len(past_docs) < 6:
        return []

    def random_sentence(doc: dict) -> str:
        sents = [
            s
            for s in mine.split_sentences(doc["abstract"])
            if mine.MIN_SENT_LEN <= len(s) <= mine.MAX_SENT_LEN
        ]
        return rng.choice(sents) if sents else doc["title"]

    tasks = []
    for subtype in CONTROL_SUBTYPES:
        if subtype == "random_pair":
            a, b = rng.sample(past_docs, 2)
            tasks.append(
                {
                    "subtype": subtype,
                    "material": {
                        "claim_a": {"arxiv_id": a["id"], "sentence": random_sentence(a)},
                        "claim_b": {"arxiv_id": b["id"], "sentence": random_sentence(b)},
                    },
                    "instruction": (
                        "Combine the two unrelated claims below into a single research "
                        "question that connects both, even though there is no evidence "
                        "linking them.\n"
                        "Claim A: {claim_a}\nClaim B: {claim_b}"
                    ),
                }
            )
        elif subtype == "vague":
            tasks.append(
                {
                    "subtype": subtype,
                    "material": {},
                    "instruction": (
                        f"Write one broad, vague research question about {subfield_name} "
                        "of the form 'What is the nature of ...?' or 'How do ... work?' "
                        "with no specific objects, quantities, or tests."
                    ),
                }
            )
        elif subtype == "untestable":
            doc = rng.choice(past_docs)
            tasks.append(
                {
                    "subtype": subtype,
                    "material": {"arxiv_id": doc["id"], "sentence": random_sentence(doc)},
                    "instruction": (
                        "Based on the claim below, write one research question that is "
                        "interesting-sounding but has no observable consequence and no "
                        "way to be tested.\nClaim: {sentence}"
                    ),
                }
            )
        else:  # consensus_challenge
            doc = rng.choice(past_docs)
            tasks.append(
                {
                    "subtype": subtype,
                    "material": {"arxiv_id": doc["id"], "title": doc["title"]},
                    "instruction": (
                        "Write one research question that challenges a well-established "
                        f"consensus in {subfield_name} without citing any supporting "
                        "evidence, loosely inspired by this paper title: {title}"
                    ),
                }
            )

    listing = []
    for i, t in enumerate(tasks):
        fills = {
            k: (v["sentence"] if isinstance(v, dict) and "sentence" in v else v)
            for k, v in t["material"].items()
        }
        listing.append(f"{i + 1}. " + t["instruction"].format(**fills))
    resp = chat_json(
        [
            {"role": "system", "content": _SYSTEM.format(cutoff=cutoff)},
            {
                "role": "user",
                "content": (
                    "Complete each numbered task below. Phrase every result as a "
                    "single research question in the same academic style.\n\n"
                    + "\n\n".join(listing)
                    + '\n\nReply as JSON: {"questions": ["...", "..."]} with exactly '
                    "one question per task, in order."
                ),
            },
        ],
        model=GENERATION_MODEL,
    )
    out = []
    for t, raw in zip(tasks, resp.get("questions", [])):
        q = _clean_question(str(raw))
        if len(q) >= 20:
            out.append(
                {
                    "question": q,
                    "source": "control",
                    "control_subtype": t["subtype"],
                    "source_material": t["material"],
                }
            )
    return out


# ---------------------------------------------------------------- cell driver


def generate_cell(
    past_docs: list[dict], subfield_key: str, subfield_name: str, cutoff_year: int
) -> list[dict]:
    """Generate the full quota of questions for one (cutoff, subfield) cell."""
    cutoff = f"{cutoff_year}-12-31"
    rng = random.Random(f"{subfield_key}-{cutoff_year}")
    tension_sents = mine.mine_tension(past_docs)
    fw_sents = mine.mine_future_work(past_docs)

    questions: list[dict] = []
    clusters = build_tension_clusters(tension_sents)
    questions += generate_tension_questions(
        clusters, subfield_name, cutoff, QUOTA["evidence_tension"]
    )
    questions += generate_direct_questions(
        past_docs, subfield_name, cutoff, QUOTA["direct_llm"], rng
    )
    questions += generate_future_work_questions(
        fw_sents, subfield_name, cutoff, QUOTA["future_work"], rng
    )
    questions += generate_control_questions(past_docs, subfield_name, cutoff, rng)

    questions = _dedupe(questions)
    for i, q in enumerate(questions):
        q["question_id"] = f"{subfield_key}-{cutoff_year}-{i:03d}"
        q["subfield"] = subfield_key
        q["cutoff_year"] = cutoff_year
        q["cutoff_date"] = cutoff
        q.setdefault("control_subtype", None)
    return questions
