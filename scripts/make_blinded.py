"""Generate paper/paper3_blinded.tex from paper/paper3.tex.

Information Processing & Management reviews double-blind, so the manuscript
sent to reviewers must carry no author identifiers. This script derives the
blinded file from the canonical source instead of maintaining two versions
by hand: rerun it after any edit to paper3.tex.

What it removes or rewrites:
- the author block and the series footnote in the title;
- every self-referential mention of the two predecessor papers ("Paper 1",
  "Paper 2", "this series", "ours") -> neutral third-person citations;
- repository URLs that contain account names.

Kept deliberately: the bibliography entries for the two predecessor arXiv
papers, cited in third person. Double-blind practice asks authors to refer
to their own work as they would anyone else's, not to hide verifiable
public references.

The script fails loudly if any identifier survives, and the check runs on
the final text, so new identifying phrasings added to paper3.tex surface
here instead of in a reviewer's PDF.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path("paper/paper3.tex")
DST = Path("paper/paper3_blinded.tex")

# Ordered: longest / most specific first, so broad patterns cannot eat the
# text a specific one needs.
REPLACEMENTS = [
    # --- self-referential framings -> neutral citations ---
    ("Paper 2 of this series~\\cite{paper2} introduced",
     "The backtesting protocol of~\\cite{paper2} introduced"),
    ("Answering the\nstructural-prior question posed by Paper 2's outlook",
     "Answering the\nstructural-prior question posed in~\\cite{paper2}"),
    ("Paper 1~\\cite{paper1} built", "An earlier framework~\\cite{paper1} built"),
    ("Paper\n2~\\cite{paper2} defined", "The\nprotocol paper~\\cite{paper2} defined"),
    ("structural-prior question posed by Paper 2's outlook",
     "structural-prior question posed in~\\cite{paper2}"),
    ("our evidence-tension system", "an evidence-tension mining system"),
    ("The ten frozen Paper 1 evidence-graph questions",
     "Ten frozen evidence-graph questions from~\\cite{paper1}"),
    ("10 Paper 1", "10 from~\\cite{paper1}"),
    ("Paper 1 evidence-graph ($n=10$)", "Evidence-graph~\\cite{paper1} ($n=10$)"),
    ("The ten Paper 1 questions", "The ten evidence-graph questions~\\cite{paper1}"),
    ("the Paper 1\nsubset demonstrates", "the evidence-graph\nsubset demonstrates"),
    ("A cautionary card from the Paper 1 import",
     "A cautionary card from the imported~\\cite{paper1} questions"),
    ("including ours and Paper 1's, needs",
     "including this one and that of~\\cite{paper1}, needs"),
    ("Paper 2 v1.1's prospective\ninstance",
     "the frozen prospective\ninstance of~\\cite{paper2}"),
    # --- remaining possessives and bare mentions ---
    ("Paper 2 v1.1's", "The sealed v1.1 release~\\cite{paper2}"),
    ("Paper 2 v1.1", "the sealed v1.1 release~\\cite{paper2}"),
    ("Paper 2's outlook section poses", "The protocol paper~\\cite{paper2} poses"),
    ("Paper 2's outlook proposes", "\\cite{paper2} proposes"),
    ("Paper 2's", "the protocol paper's"),
    ("Paper 2", "\\cite{paper2}"),
    ("Paper 1's", "that framework's"),
    ("Paper 1", "\\cite{paper1}"),
]


def blind(text: str) -> str:
    # 1. author block and series footnote
    text = re.sub(
        r"from Question Structure and Evidence Context\\thanks\{.*?\}\}",
        "from Question Structure and Evidence Context}",
        text, flags=re.S,
    )
    text = re.sub(r"\\author\{.*?\}\n\\date", "\\\\author{}\n\\\\date", text, flags=re.S)

    # 2. self-references
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    # 3. identifying URLs in the reproducibility statement (plain string
    # match: the href's second argument nests braces, which defeats a
    # bracket-negation regex)
    text = re.sub(
        r"The frozen dataset is additionally published on the Hugging Face Hub at\n"
        r"\\href\{.*?\n"
        r"loadable in one call, with the structure-prior and second-judge subsets as\n"
        r"separate configurations\.",
        "The frozen dataset is additionally published on the Hugging Face Hub,\n"
        "loadable in one call, with the structure-prior and second-judge subsets as\n"
        "separate configurations. Repository and dataset URLs are withheld here for\n"
        "double-blind review and will be supplied on acceptance.",
        text, flags=re.S,
    )
    text = re.sub(
        r"released as a separate repository with the same layout --- code, frozen\n"
        r"questions, features, labels, audit verdicts, and result tables --- at\n"
        r"\\href\{.*?\n"
        r"given its harvest, every replication number is deterministic \(no\n"
        r"language-model calls at any stage\)\.",
        "released as a separate repository with the same layout --- code, frozen\n"
        "questions, features, labels, audit verdicts, and result tables ---\n"
        "(URL likewise withheld for double-blind review); given its harvest, every\n"
        "replication number is deterministic (no language-model calls at any stage).",
        text, flags=re.S,
    )

    # 4. bibliography aliases
    text = text.replace("Referred to as ``Paper 1'' throughout.\n", "")
    text = text.replace("Referred to as ``Paper 2'' throughout.\n", "")
    return text


def residual_identifiers(text: str) -> list[str]:
    body = text.split("\\begin{thebibliography}")[0]
    hits = []
    for pat in (r"Hui Mao", r"alumni\.upenn", r"ORCID", r"nonameisready",
                r"huiluckylucky", r"this series", r"\bours\b", r"Paper [12]"):
        for m in re.finditer(pat, body):
            hits.append(f"{pat!r} at ...{body[max(0, m.start()-40):m.end()+20]!r}")
    return hits


def main() -> None:
    text = blind(SRC.read_text())
    leftovers = residual_identifiers(text)
    if leftovers:
        print("BLINDING FAILED — identifiers remain:")
        for h in leftovers:
            print("  ", h)
        sys.exit(1)
    DST.write_text(text)
    print(f"wrote {DST} ({len(text.splitlines())} lines), no residual identifiers")


if __name__ == "__main__":
    main()
