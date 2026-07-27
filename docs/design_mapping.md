# Design → implementation mapping

How each element of the Paper 3 design brief maps onto this codebase.

| Design element | Implementation |
| --- | --- |
| **Task A** — will the question be substantively studied? (binary) | `y_addressed` from tiered labels; primary target of every model (`models.py`) |
| **Task B** — outcome type (5-way) | `answer_status` multiclass (`models.run_multiclass`) |
| **Task C** — degree of impact (counts, lead time) | `future_paper_count`, `lead_time_months`; Cox survival (`models.run_survival`), within-cell ranking concordance |
| Rolling historical backtests, not a single cutoff | 5 cutoffs (2012–2020, step 2), each with independent past/future windows (`corpus.py`) |
| 8–12 subfields, not one narrow topic | 8 subfields with transparent keyword classifiers (`subfields.py`) |
| ≥4 generation sources incl. negative controls | evidence-tension / direct-LLM / future-work / 4-subtype controls (`generate.py`), plus Paper 1's frozen questions |
| Question feature record, six groups | groups A–F in `features.py` (§ feature groups in README) |
| Cross-independent-source tension as the flagship input | tension clusters require ≥2 distinct papers; `ev_tension_density`, `ev_native_n_papers`; H1 |
| Computable novelty, not LLM vibes | nearest-neighbour distance, concept-pair frequency, review overlap (`novelty_features`) |
| Falsifiability structured coding | rule markers + blinded LLM coding `{observable, competing_hypotheses, quantitative_test, falsification_path}` |
| Tractability / observation availability | facility lexicon with operational epochs (`facilities.py`), archival-data markers |
| Environment controls (hot-field confound) | volume, growth, review rate, facility-launch proximity (`corpus.env_features`); controls in every regression |
| Tiered labels: retrieval → blinded judge → verification | `labeling.py`; second independent judge + Cohen's κ (`analysis.judge_agreement`) |
| Multi-dimensional outcome record | addressed, answer_status, premise_status, counts, lead time, review recognition |
| Interpretable models before black boxes | majority/popularity/text baselines → logistic (L2, L1) → RF/GBM (`models.py`) |
| Temporal extrapolation, never random splits | train 2012–2016 / val 2018 / test 2020; random splits not implemented by design |
| Cross-domain + leave-one-subfield-out | `run_cross_domain`, `run_loso` |
| Predictor identification beyond feature importance | univariate scan, controlled logits, group ablations, sign-stability (`analysis.py`) |
| Pre-stated hypotheses H1–H6 | `analysis.hypothesis_tests` (encoded before results were computed) |
| Interpretable prediction card output | `analysis.prediction_cards` |
| ~1,000 questions (8 × 5 × 25) | quota `QUOTA` in `generate.py`; realized counts in `data/questions/generation_summary.json` |

## Deviations from the brief, and why

- **Citation-weighted attention** is not implemented: the arXiv metadata
  corpus carries no citation counts, and adding a second data provider
  (ADS) would make reproduction key-gated. Paper counts, lead time, and
  review recognition are used instead; the manuscript lists this as a
  limitation.
- **Independent research-group counts** are approximated by distinct
  papers/instruments, since author-affiliation clustering is out of scope
  for v1.
- **Human adjudication** is replaced by a second, independent LLM judge on
  a stratified sample (all refuted cases + 20% random). The agreement/κ
  report plays the role of the reliability analysis; the annotation
  protocol is written so that human passes can slot in later.
- **Cutoffs start at 2012** (not 2010) so that even the youngest subfields
  (FRBs) have a non-empty past corpus, while keeping 5 full future windows.
- **Judge escalation.** The first-pass GPT-4o-mini judge accepted topical
  overlap as engagement (94% addressed). The released labels use a strict
  two-stage rubric (per-candidate direct/adjacent/unrelated classification,
  `addressed` derived from direct engagement) with GPT-4o as primary judge
  and GPT-4o-mini as the independent verification judge. Both passes are
  preserved in the LLM cache; the lenient-vs-strict comparison is reported
  in the paper's label-reliability section.
