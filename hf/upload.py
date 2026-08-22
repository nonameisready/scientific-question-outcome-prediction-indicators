"""Publish the frozen dataset to the Hugging Face Hub.

Joins questions + features + labels into one flat table, writes the two
auxiliary configs alongside it, and uploads with the dataset card in
`hf/README.md`.

The card declares `arxiv:` tags, which is what creates the linked paper pages
on the Hub — the reliable route, independent of the curated Daily Papers
submission flow.

    pip install huggingface_hub
    huggingface-cli login
    python hf/upload.py                      # dry run: build files, no upload
    python hf/upload.py --push               # build and upload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ID = "nonameisready/scientific-question-outcomes"

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/questions/questions_v1.jsonl"
FEATURES = ROOT / "data/features/features_v1.jsonl"
LABELS = ROOT / "data/labels/labels_v1.jsonl"
STRUCTURE = ROOT / "data/features/structure_features_v1.jsonl"
SECOND_JUDGE = ROOT / "data/labels/labels_second_judge.jsonl"
CARD = Path(__file__).resolve().parent / "README.md"
BUILD = Path(__file__).resolve().parent / "build"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build() -> dict[str, int]:
    BUILD.mkdir(exist_ok=True)

    feats = {r["question_id"]: r for r in read_jsonl(FEATURES)}
    labels = {r["question_id"]: r for r in read_jsonl(LABELS)}

    merged = []
    for q in read_jsonl(QUESTIONS):
        qid = q["question_id"]
        row = dict(q)
        # source_material is a nested union type across sources; JSON-encode it
        # so the Hub infers one stable string column instead of a ragged struct.
        row["source_material"] = json.dumps(q.get("source_material"), ensure_ascii=False)
        row.update({k: v for k, v in feats[qid].items() if k != "question_id"})
        row.update({k: v for k, v in labels[qid].items() if k != "question_id"})
        merged.append(row)

    write_jsonl(merged, BUILD / "questions_v1.jsonl")
    counts = {"default": len(merged)}

    for src, name in ((STRUCTURE, "structure_features_v1"), (SECOND_JUDGE, "labels_second_judge")):
        if src.exists():
            rows = read_jsonl(src)
            write_jsonl(rows, BUILD / f"{name}.jsonl")
            counts[name] = len(rows)

    (BUILD / "README.md").write_text(CARD.read_text())
    return counts


def push() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(BUILD), repo_id=REPO_ID, repo_type="dataset")
    print(f"pushed https://huggingface.co/datasets/{REPO_ID}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upload after building")
    args = ap.parse_args()

    counts = build()
    print(f"built {BUILD}")
    for name, n in counts.items():
        print(f"  {name}: {n} rows")
    if args.push:
        push()
    else:
        print("\ndry run — inspect build/, then rerun with --push")


if __name__ == "__main__":
    main()
