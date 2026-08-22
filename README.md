# Paper 3 — What Makes a Scientific Question Succeed?

**Predicting future scientific attention, resolution, and premise revision
from the structure of research questions and their evidence context.**

This repository (Paper 3 of the series) scales the historical-backtesting
idea of Paper 1 ([arXiv:2608.09968](https://arxiv.org/abs/2608.09968),
[code](https://github.com/nonameisready/scientific-question-discovery))
and Paper 2 ([arXiv:2608.16795](https://arxiv.org/abs/2608.16795),
[code](https://github.com/nonameisready/scientific-question-discovery-benchmark))
from a 10-question case study into a supervised-learning dataset of ~1,000
astronomy research questions across **5 historical cutoffs × 8 subfields ×
4 generation sources**, each with a full cutoff-time feature record and a
tiered outcome label derived from the 5 years of literature that actually
followed.

The repository's task (design section 15):

> *Input*: a scientific question + historical evidence context +
> cutoff-time metadata + structured question features.
> *Output*: probabilities of future attention, resolution, remaining open,
> premise revision, and estimated time to first substantive follow-up.

It does **not** score how "interesting" a question feels. It measures which
question properties historically predicted future scientific activity and
knowledge change.

## Dataset design

| Axis | Values |
| --- | --- |
| Cutoffs | 2012, 2014, 2016, 2018, 2020 (Dec 31 each) |
| Past window | 8 years before the cutoff (question generation + features) |
| Future window | 5 years after the cutoff (outcome labels) |
| Subfields | exoplanet atmospheres, protoplanetary disks, stellar activity, fast radio bursts, gravitational waves, galaxy evolution, cosmology tensions, compact objects |
| Sources per cell | 10 evidence-tension, 5 direct-LLM, 5 future-work, 5 weak/negative controls |
| Extra source | the 10 frozen Paper 1 evidence-graph questions (cutoff 2020) |

The underlying corpus is the complete arXiv astro-ph metadata record
(2005–2025, **313,189 papers** after deduplication), harvested via the
public arXiv API and classified into subfields with transparent keyword
rules (`pipeline/subfields.py`).

**Headline results** (details in the paper): 980 frozen questions, 52.7%
substantively addressed within 5 years under a strict blinded judge;
controls 32% vs. mined sources ~50% vs. direct-LLM 88% (leakage-caveated);
temporal-split AUC 0.71 (logistic on structured features), cross-domain
AUC 0.71, leave-one-subfield-out 0.62–0.78; the robust predictors of
future attention are prior review recognition, density of directly
related prior evidence, and explicit competing hypotheses — while five of
six pre-stated mechanism hypotheses (including "tension beats novelty")
are not supported. Within the tension-mined subset, the generating
evidence-graph *structure* is itself predictive (`pipeline/structure.py`,
`results/structure_prior.json`): cluster breadth predicts uptake
(β=+0.45/SD), explicit contradiction predicts engagement, and
quantified, cutoff-recurrent tensions carry every resolution and
premise refutation — a mining-time ranking prior for structure-first
question generation.

## Feature groups (all cutoff-time)

- **A text** — length, entities, causal/comparative/mechanistic language,
  measurable quantities, explicit alternatives, falsification wording.
- **B evidence context** — retrieval against the pre-cutoff subfield
  corpus: similarity mass, instrument diversity, tension-marker density,
  evidence age; native evidence-record features for tension questions.
- **C novelty** — nearest-neighbour semantic distance, concept-pair
  novelty, overlap with pre-cutoff reviews.
- **D falsifiability** — rule-based markers + blinded LLM structured coding
  (observable, competing hypotheses, quantitative test, falsification path).
- **E tractability** — operational vs. future facilities named, archival
  data, longitudinal/theory/sample requirements.
- **F environment (controls)** — subfield publication volume and growth,
  review activity, proximity of the next major facility.

## Outcome labels (tiered)

1. **Automatic candidates** — retrieval of future papers per question:
   relevant-paper counts, first-follow-up date, lead time, review
   recognition.
2. **Blinded LLM outcome judge** — sees only retrieved future records,
   never the question's generation source; outputs
   `addressed`, `answer_status` (answered / partially_addressed /
   posed_but_open / premise_refuted / not_addressed), `premise_status`,
   supporting ids, rationale, confidence.
3. **Independent second judge** — a different model re-judges all
   `premise_refuted` cases plus a 20% random sample; agreement and Cohen's
   κ are reported as label reliability.

## Evaluation protocol

Random train/test splits are forbidden (they leak within eras and topics).
All results use:

- **Temporal split** — train 2012–2016, validate 2018, test 2020.
- **Leave-one-subfield-out** and a fixed **cross-domain split**
  (4 subfields → the other 4).
- **Ablations** by feature group; **univariate + controlled** effects;
  **stability** of predictor signs across cutoffs, subfields, and sources.
- Pre-stated hypotheses **H1–H6** (`pipeline/analysis.py`).

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests -q          # offline tests

# Stage 0 (network, ~1 h): harvest arXiv astro-ph 2005-2025
python -m pipeline.harvest --out data/raw/arxiv

# Stage 1: classify + slice corpus
python scripts/run_corpus.py

# Stage 2 (OPENAI_API_KEY): generate + freeze questions
python scripts/run_generate.py
python scripts/validate_isolation.py

# Stage 3 (OPENAI_API_KEY for group D): features
python scripts/run_features.py

# Stage 4 (OPENAI_API_KEY): tiered outcome labels
python scripts/run_labels.py

# Stage 5: models, analyses, hypotheses, figures
python scripts/run_models.py

# Stage 6 (offline): evidence-graph structure prior
python scripts/run_structure_prior.py
```

All LLM calls are cached under `data/cache/llm/` and pinned to fixed
seeds/temperature 0, so re-runs are deterministic given the cache. The
frozen dataset (`data/questions`, `data/features`, `data/labels`) and all
results are committed; only the raw harvest and cache are not.

## Dataset on the Hub

The frozen dataset is also published at
[huggingface.co/datasets/huiluckylucky/scientific-question-outcomes](https://huggingface.co/datasets/huiluckylucky/scientific-question-outcomes),
with the questions/features/labels joined into one table plus the
structure-prior and second-judge subsets as separate configs:

```python
from datasets import load_dataset

ds = load_dataset("huiluckylucky/scientific-question-outcomes")
```

`hf/upload.py` rebuilds and republishes it from the frozen files here.

## Manuscript

The manuscript lives at [`paper/paper3.md`](paper/paper3.md).

License: Apache-2.0.
