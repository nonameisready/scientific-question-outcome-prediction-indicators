---
license: apache-2.0
language:
  - en
task_categories:
  - text-classification
  - tabular-classification
tags:
  - science-of-science
  - scientometrics
  - research-question-generation
  - llm-as-judge
  - historical-backtesting
  - astronomy
  - arxiv:2608.09968
  - arxiv:2608.16795
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: questions_v1.jsonl
  - config_name: structure
    data_files: structure_features_v1.jsonl
  - config_name: second_judge
    data_files: labels_second_judge.jsonl
---

# Scientific Question Outcomes

**980 astronomy research questions, frozen at five historical cutoffs, each
labelled with what the following five years of literature actually did with
it.**

Systems that propose research questions are usually evaluated by asking a
person or a model how good the questions sound. This dataset supplies the
alternative: questions frozen using only pre-cutoff literature, and outcome
labels drawn from the literature published afterwards. It is, to our
knowledge, the only public dataset that attaches future-outcome labels to
*questions* — rather than to papers — across multiple historical cutoffs and
with controlled generation sources.

## What is in it

Each question was generated from the arXiv astro-ph record available strictly
before its cutoff (2012, 2014, 2016, 2018, or 2020), across eight subfields,
from four distinct sources:

| Source | n | Addressed within 5 years |
| --- | --- | --- |
| Evidence-tension mining | 374 | 0.47 |
| Direct LLM elicitation | 200 | 0.88 |
| Author-stated future work | 196 | 0.51 |
| Weak/negative controls | 200 | 0.32 |
| Prior evidence-graph system | 10 | 0.20 |

The negative controls are deliberately weak questions — random claim pairings,
vague questions, untestable questions, evidence-free contrarian questions —
phrased by the same model so that surface style cannot separate them from real
candidates. **They exist so that any engagement metric computed on this data
has a priced chance level.** Without them, a model can only learn that
question-shaped sentences succeed.

The direct-LLM rate of 0.88 should not be read as generator quality. A model
asked in hindsight for the important open questions of 2016 may simply recall
what 2017–2021 worked on; this is a documented leakage channel, not a result.

## Configurations

**`default`** — 980 rows. Question text, provenance, a cutoff-time feature
record in six groups, and the outcome label.

**`structure`** — 374 rows, the tension-mined subset. Structural properties of
the evidence cluster each question was generated from: cluster breadth, tension
typology, quantification, instrument structure, and cross-cutoff recurrence.

**`second_judge`** — 200 rows. An independent second model's labels on all
premise-refuted cases plus a 20% random sample, for label-reliability work.

## Fields

**Question and provenance** — `question_id`, `question`, `subfield`,
`cutoff_year`, `cutoff_date`, `source`, `control_subtype`, `source_material`
(verbatim generating evidence: sentences, arXiv ids, dates).

**Features**, all computed at cutoff time, by prefix:

| Prefix | Group | Examples |
| --- | --- | --- |
| `txt_` | text structure | entity count, causal/comparative language, specificity |
| `ev_` | evidence context | similarity mass, instrument diversity, tension density |
| `nov_` | novelty | nearest-neighbour distance, concept-pair novelty, review overlap |
| `fal_` | falsifiability | observable, competing hypotheses, quantitative test |
| `tra_` | tractability | operational vs. future facilities, archival data |
| `env_` | field environment | publication volume, growth, next-facility proximity |
| `str_` | evidence structure | cluster breadth, tension type, recurrence (structure config) |

**Labels** — `addressed` (bool), `answer_status` (answered /
partially_addressed / posed_but_open / premise_refuted / not_addressed),
`premise_status`, `future_paper_count`, `first_follow_up_date`,
`lead_time_months`, `review_recognition`, `supporting_ids`, `rationale`,
`confidence`, `judge_model`.

## How the labels were made

Three tiers. First, automatic retrieval of future papers from each question's
subfield over its five-year future window. Second, a strict judge that sees
the question and its top-8 retrieved future records but **never the generation
source**; it classifies every candidate as direct, adjacent, or unrelated
before assigning an outcome, and `addressed` requires at least one direct
candidate. Third, an independent second model re-judges all premise-refuted
cases plus a 20% random sample.

The strictness matters. A first-pass lenient judge called 94% of questions
addressed by accepting topical overlap; the per-candidate rubric brings it to
52.7% and restores variance to every downstream analysis.

## Limitations — please read before using

- **Labels are rater-relative.** Second-judge agreement on `addressed` is
  63.5% (κ = 0.21). The companion protocol paper's seven-rater study found two
  independent *human* annotators agreed at only κ = 0.17 on an analogous
  taxonomy, while frontier models agreed with each other at κ = 0.60 — so
  certifying such a judge by model–model agreement overstates reliability
  roughly threefold. Absolute rates should never be quoted without the rater
  attached; comparisons under a fixed judge are the currency this data
  supports.
- **`not_addressed` is a lower bound.** Retrieval is abstract-level TF-IDF and
  can miss engagement, especially for narrow object-level questions.
- **Parametric leakage is present in one source.** See the direct-LLM note
  above. Source is a covariate in every analysis in the accompanying paper.
- **Rare outcomes are rare.** 8 answered and 4 premise-refuted in total; the
  five-way status taxonomy is underpowered at this n.
- **One domain.** Astronomy only. Cross-subfield transfer is demonstrated;
  cross-science generality is not.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("huiluckylucky/scientific-question-outcomes")
structure = load_dataset("huiluckylucky/scientific-question-outcomes", "structure")
```

## Papers and code

- **Question discovery framework** — [arXiv:2608.09968](https://arxiv.org/abs/2608.09968)
- **Backtesting protocol and benchmark** — [arXiv:2608.16795](https://arxiv.org/abs/2608.16795)
- **This dataset and its predictor analysis** — paper pending; pipeline, judge
  rationales, and all results at
  [github.com/nonameisready/scientific-question-outcome-prediction-indicators](https://github.com/nonameisready/scientific-question-outcome-prediction-indicators)

## Citation

```bibtex
@misc{mao2026questionoutcomes,
  author = {Hui Mao},
  title  = {Scientific Question Outcomes: 980 Research Questions with
            Five-Year Historical Outcome Labels},
  year   = {2026},
  url    = {https://huggingface.co/datasets/huiluckylucky/scientific-question-outcomes}
}
```

License: Apache-2.0.
