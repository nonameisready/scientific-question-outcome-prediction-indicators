# Paper 3 data

All committed artifacts are derived, frozen, and versioned; everything can
be rebuilt from scratch with the staged scripts in `scripts/`.

| Path | Contents | Committed |
| --- | --- | --- |
| `raw/arxiv/` | arXiv astro-ph metadata shards, 2005–2025 (per category-year JSONL) | no (rebuild with `python -m pipeline.harvest`) |
| `raw/corpus_classified.jsonl` | deduplicated corpus with subfield assignments | no (rebuild with `run_corpus.py`) |
| `corpus/corpus_manifest.json` | harvest provenance, totals, cutoff windows, corpus SHA-256 | yes |
| `corpus/subfield_year_stats.json` | per-subfield-per-year publication/review counts | yes |
| `questions/questions_v1.jsonl` | frozen question dataset with full provenance | yes |
| `questions/generation_summary.json` | per-cell generation counts | yes |
| `features/features_v1.jsonl` | six-group cutoff-time feature records | yes |
| `labels/labels_v1.jsonl` | tier 1+2 outcome labels | yes |
| `labels/labels_second_judge.jsonl` | tier 3 second-judge sample | yes |
| `labels/judge_agreement.json` | judge agreement + Cohen's κ | yes |
| `cache/llm/` | deterministic LLM response cache | no |

## Question record schema (`questions_v1.jsonl`)

```json
{
  "question_id": "exoplanet_atmospheres-2016-004",
  "question": "…?",
  "source": "evidence_tension | direct_llm | future_work | control | paper1_evidence_graph",
  "control_subtype": "random_pair | vague | untestable | consensus_challenge | null",
  "subfield": "exoplanet_atmospheres",
  "cutoff_year": 2016,
  "cutoff_date": "2016-12-31",
  "source_material": { "cluster | statement | …": "exact pre-cutoff sentences + arXiv ids" }
}
```

## Label record schema (`labels_v1.jsonl`)

```json
{
  "question_id": "…",
  "future_paper_count": 12,
  "future_paper_count_weak": 31,
  "first_follow_up_date": "2017-03-14",
  "lead_time_months": 3,
  "review_recognition": true,
  "top_ids": ["…"], "top_sims": [0.31, "…"],
  "addressed": true,
  "answer_status": "partially_addressed",
  "premise_status": "intact",
  "supporting_ids": ["…"],
  "rationale": "…",
  "confidence": 0.8,
  "judge_model": "gpt-4o-mini | auto_threshold"
}
```

Temporal isolation is enforced by construction (source material is drawn
from the pre-cutoff window only; labels from the post-cutoff window only)
and checked by `scripts/validate_isolation.py`.
