# What Makes a Scientific Question Succeed? Predicting Future Attention, Resolution, and Premise Revision from Question Structure and Evidence Context

**Hui Mao**
Independent Researcher · `[EMAIL]`

*Third in a series on temporally grounded evaluation of scientific question
discovery; see [arXiv:2608.09968](https://arxiv.org/abs/2608.09968) and
[arXiv:2608.16795](https://arxiv.org/abs/2608.16795).*

## Abstract

Large language models are increasingly asked to propose "important open
questions", and are typically evaluated by asking another model how good
those questions feel. We replace taste with history. We construct a
temporally grounded dataset of **980 astronomy research questions** frozen
at five historical cutoffs (2012–2020), spanning eight subfields and four
distinct generation sources — evidence-tension mining, direct LLM
elicitation, author-stated future work, and weak/negative controls — on
top of the complete arXiv astro-ph metadata record (**313,189 papers,
2005–2025**). Each question carries a six-group cutoff-time feature
record (text structure, evidence context, novelty, falsifiability,
tractability, field environment) and a tiered outcome label derived from
the five years of literature that actually followed. Under a strict,
source-blinded outcome judge, 52.7% of questions were substantively
addressed; negative-control questions were addressed least (32%) and
direct-LLM questions most (88%), the latter carrying a documented
parametric-leakage caveat. Interpretable models trained on 2012–2016
cutoffs predict 2020 outcomes at AUC 0.71 and transfer to entirely
held-out subfields at AUC 0.65–0.71. Across univariate, controlled,
ablation, and stability analyses, the robust predictors of future
attention are **prior community recognition** (overlap with pre-cutoff
review articles, β=+0.34, p<10⁻⁴), **the density of directly related
prior evidence** (β=+0.26), and **explicitly stated competing
hypotheses** (β=+0.17) — while high entity-specificity predicts *less*
and *slower* engagement under literature-bounded labels. Five of six
pre-stated mechanism hypotheses, including "evidence tension beats
novelty", are *not* supported at scale — a result only visible because
the dataset is large, multi-cutoff, and control-laden. Answering the
structural-prior question posed by Paper 2's outlook, we further show
that within the tension-mined subset the generating evidence-graph
structure itself is predictive: breadth of the paper cluster in tension
predicts uptake (β=+0.45/SD), explicit contradiction predicts
engagement, and quantified, cutoff-persistent tensions carry every
resolution and premise refutation — a learnable, mining-time prior for
structure-first question generation. All data, code, labels, and judge
rationales are released.

## 1. Introduction

How scientific attention gets allocated is a central concern of the science
of science, and it has been answered almost entirely at the level of papers
and careers: which combinations of prior work earn citations (Uzzi et al.,
2013), which research strategies scientists actually adopt (Foster et al.,
2015), how impact is distributed across a field and a lifetime (Fortunato
et al., 2018; Wang & Barabási, 2021). The unit that precedes all of these —
the research question itself, the bet placed before any paper exists — has
no comparable empirical treatment. The obstacle is not interest but
measurement: questions leave no bibliographic record, so there is nothing
to attach an outcome to.

This paper constructs those outcomes. We freeze research questions at
historical cutoffs using only the literature available at the time, then
read off what the following five years of literature did with each one. The
resulting dataset makes a question-level analogue of the science-of-science
program possible: rather than asking which papers were cited, we ask which
*questions* the community took up, resolved, or overturned, and which of
their cutoff-time properties predicted that.

The same construction answers a second, narrower question that motivated
the series. Machine-generated research questions are currently evaluated by
LLM or expert panels scoring "importance" and "novelty" at generation time
— a measurement of how a question *sounds*, not of what it *does* (Si et
al., 2024). [Paper 2 of this
series](https://arxiv.org/abs/2608.16795) introduced historical
backtesting as the alternative: freeze
questions using only pre-cutoff literature, then observe what the
scientific community actually did afterwards. Its sealed v1.1 release has
since grown from a ten-question pilot into a benchmark family — a scaled
424-question baseline instance, a four-cutoff temporal stress test (798
judged questions, 2010–2024), and a prospective instance frozen in 2026 —
but its unit of comparison remains the generating *system*: it measures
which generator families produce questions the future engages, not which
measurable properties of an individual question predict that engagement.
That predictor-discovery problem is this paper's subject.

This paper turns the backtesting protocol into a supervised-learning
problem at scale. Three design decisions matter:

1. **The unit of data is (question, cutoff, subfield, source), not the
   question alone.** We generate questions at five rolling cutoffs
   (2012–2020) in eight astronomy subfields, so that one study contains
   what would otherwise be forty historical experiments, and era- or
   field-specific accidents can be detected rather than absorbed.
2. **No single generator.** Predictors learned from one generation system
   risk being that system's stylistic fingerprint. We use four sources —
   our evidence-tension system, a direct-LLM baseline, questions extracted
   from the authors' own future-work statements, and deliberately weak
   controls (random claim pairings, vague, untestable, and
   evidence-free-contrarian questions). Without negative controls a model
   can only learn that "question-shaped sentences succeed".
3. **Prediction is out-of-time or out-of-domain, always.** Random splits
   would place near-duplicate questions from the same era and topic on
   both sides. We train on 2012–2016 cutoffs, validate on 2018, test on
   2020, and additionally hold out entire subfields.

The contribution is deliberately not a leaderboard number. It is a
temporally grounded dataset of scientific questions with subsequent
outcomes, and an analysis of *which properties of a question and its
evidence context* predicted future scientific attention, resolution, and
premise revision — with controls for field popularity, instrument eras,
and generation method, and with stability checks across cutoffs,
subfields, and sources.

## 2. Related work

### 2.1 Problem choice in the science of science

The choice of research problem is treated in this literature as a strategic
act under uncertainty. Foster, Rzhetsky and Evans (2015), analysing
millions of MEDLINE abstracts, showed that scientists overwhelmingly pursue
conservative strategies — deepening established relationships rather than
bridging distant ones — and that this conservatism is individually rational
while collectively suboptimal, later formalised as a question of how a
field should allocate its collective search (Rzhetsky et al., 2015). Uzzi
et al. (2013) reached a compatible conclusion from the citation side: the
highest-impact papers combine a conventional core with a tail of novelty,
so novelty by itself is not the operative virtue. Fortunato et al. (2018)
survey the broader program; Wang and Barabási (2021) give it book-length
treatment.

Our results speak directly to this line. The predictor we find strongest —
prior recognition of a question in review articles — is a question-level
measurement of exactly the conservatism Foster et al. document at the level
of research strategy, and our null result for semantic novelty is what
Uzzi et al.'s conventionality finding would predict. The contribution is
the unit of analysis: this literature observes what scientists *did* and
infers the strategy, whereas we fix the candidate questions in advance and
observe which were taken up. Because the questions are frozen before the
outcome window opens, and because deliberately weak controls are included,
the design separates properties of a question from properties of the field
it sits in.

### 2.2 Generating and evaluating research questions

Machine generation of research questions and hypotheses has a long lineage,
from literature-based discovery (Swanson, 1986) to LLM systems that propose
ideas or run entire studies (Lu et al., 2024; Wang et al., 2024; Baek et
al., 2024), and question-asking has been argued to be a core capability on
the path to more general scientific intelligence (Kitano, 2021). Evaluation
has not kept pace. The most rigorous instance of the dominant paradigm, Si
et al.'s expert study (2024), still measures opinion at generation time
rather than what the ideas turned out to be worth. Outcome-based
alternatives have appeared only recently and independently: HindSight
(Jiang, 2026) matches generated ideas against papers published after a
temporal cutoff and finds LLM-judged novelty correlates *negatively* with
realised impact; the ideation–execution study (Si et al., 2025) had human
researchers execute both LLM and human ideas, finding that the LLM ideas
rated higher before execution scored lower after it. Both converge with our
null results for novelty from different fields and different methods.

[Paper 1](https://arxiv.org/abs/2608.09968) built an evidence-graph
system that generates questions from cross-paper observational
tensions. [Paper 2](https://arxiv.org/abs/2608.16795) defined the
backtesting benchmark and showed, in its pilot, that future literature
substantively
engaged all ten frozen exoplanet-atmosphere questions (one premise — a
strongly subsolar water abundance for HD 209458 b — later refuted by
three independent analyses, the exact convergence the question called
for). Its sealed v1.1 release then scaled the evaluation and sharpened
three findings this paper builds on. First, at n=125 per baseline,
engagement rates discriminate between generator families (random
templates 73% vs. direct-LLM 96%, p<10⁻⁴), and random templates acquire
a measurable 11%-answered floor — engagement metrics have a priced
chance level. Second, a generator decomposition (LLM-only vs.
deterministic evidence structure vs. structure-plus-LLM verbalization)
crossed with a four-cutoff temporal stress test (2010–2024, 798 judged
questions, the last window postdating the model's training) located the
foresight signal in *pre-cutoff evidence structure* rather than model
weights: LLM-only generation shows near-ceiling engagement and the
closest phrasing to future literature at every cutoff, yet an answered
rate indistinguishable from random templates and no premise refutations
outside the deepest-history era. Third, a seven-rater agreement study
(two blinded humans, five judge models, 90 items) found the two humans
agree at only κ=0.17 — indicting the outcome taxonomy, not any
particular judge, and capping what absolute outcome rates can mean.
Both papers evaluate question *generators*; neither learns
question-level predictors, which is the question this paper addresses:
*what, measurably, makes a question one the community will take up?*

### 2.3 Outcome labels and judge reliability

Scoring a strategy on held-out history is standard in quantitative finance,
along with its documented failure mode — overfitting to the backtest itself
(Bailey et al., 2014) — which motivates the freezing rules we inherit;
forecasting benchmarks apply the same logic to events resolving after
training (Zou et al., 2022), and SWE-bench (Jimenez et al., 2024) showed
how frozen data plus a fixed task format can reorganise a field around
measurable progress.

Because our outcome labels are assigned by a language model, their
reliability is itself a measured quantity rather than an assumption. Using
LLMs as judges is now routine (Zheng et al., 2023), and so is certifying
them by agreement with other models — a practice Paper 2's seven-rater
study shows to be badly optimistic: on an analogous taxonomy two
independent human annotators agreed at κ=0.17, below every model–model
pair. We therefore report agreement with the conventional interpretive
benchmarks (Landis & Koch, 1977) attached, treat all absolute rates as
rater-relative, and hold the judge fixed within every comparison.

To our knowledge no prior dataset attaches future-outcome labels to
*questions* frozen at multiple historical cutoffs with controlled
generation sources and negative controls.

## 3. Dataset construction

### 3.1 Corpus

We harvested the complete arXiv astro-ph metadata record — titles,
abstracts, submission dates, and categories — from 2005-01 through
2025-12 via the public arXiv API (**313,189 papers** after deduplication;
`pipeline/harvest.py`). Papers are assigned to eight subfields by
transparent weighted keyword rules over title+abstract (multi-membership
allowed): exoplanet atmospheres (4,354 papers), protoplanetary disks
(10,378), stellar activity (4,347), fast radio bursts (1,966),
gravitational waves (17,989), galaxy evolution (21,390), cosmology
tensions (16,403), and compact objects (34,266). The classifier is
identical at every cutoff, so subfield corpora are time-sliced views of a
fixed rule, not retrofitted topic models.

Each cutoff year c ∈ {2012, 2014, 2016, 2018, 2020} defines a **past
window** [c−7, c] used for generation and features and a **future
window** (c, c+5] used only for labels. The five future windows all close
by 2025-12-31, giving every cutoff the same 5-year outcome horizon.

### 3.2 Question generation (40 cells × quota 25)

For every (cutoff, subfield) cell:

- **Evidence-tension (10).** We mine the cell's past-window abstracts for
  tension statements (contradiction, discrepancy, unexplained-result
  markers), cluster them so that each cluster spans ≥ 2 distinct papers,
  and have an LLM phrase each cluster as one precise question grounded
  only in the mined sentences. Provenance (exact sentences, arXiv ids,
  dates) is stored with the question.
- **Direct LLM (5).** The baseline the field implicitly uses: the model
  sees a year-stratified sample of pre-cutoff titles and is asked for the
  most important unanswered questions, with an explicit
  knowledge-freeze instruction.
- **Future-work (5).** Sentences in which pre-cutoff authors themselves
  flag an open problem ("remains poorly constrained", "further
  observations are needed"), selected for topic diversity and converted
  faithfully into interrogative form. These are real scientists'
  questions.
- **Controls (5).** Two random claim-pairings across unrelated papers,
  one vague question, one untestable question, one evidence-free
  consensus challenge — all phrased by the same LLM so that surface style
  cannot separate controls from real candidates.

Near-duplicates within a cell are removed (TF-IDF cosine ≥ 0.75). The ten
frozen Paper 1 evidence-graph questions join the dataset as a small
additional source (cutoff 2020, exoplanet atmospheres), linking this
study to the original system.

**Realized totals: 980 questions** — 374 evidence-tension, 200
direct-LLM, 196 future-work, 200 controls, 10 Paper 1. Every non-FRB cell
filled its full quota of 25; early fast-radio-burst cells are
data-limited by history itself (the pre-2013 FRB literature contains 21
papers, yielding 11 questions at the 2012 cutoff), which we treat as a
feature of honest backtesting rather than a defect. The dataset is
regenerable bit-identically from the frozen LLM cache; rebuilding the
corpus from scratch and regenerating produced byte-identical questions.

Temporal isolation is enforced by construction — every mined sentence and
every title shown to a generator predates the cutoff — and verified by an
automated validator over the frozen records (980/980 pass).

### 3.3 Feature record (six groups, all cutoff-time)

**A. Text structure** — length, named entities, numbers, causal /
comparative / mechanistic language, measurable quantities, explicit
alternatives ("real or artifact"), falsification wording, yes/no form,
and a composite specificity score.

**B. Evidence context** — retrieval of the top-20 pre-cutoff subfield
papers for each question: similarity mass (top-1, mean top-5, count above
threshold), instrument diversity among retrieved papers, density of
tension markers in the retrieved evidence, and the age profile of that
evidence; plus native evidence-record features (paper count, year spread)
for tension-sourced questions.

**C. Novelty** — nearest-neighbour semantic distance to the pre-cutoff
corpus, mean top-10 similarity, concept-pair novelty (do the question's
two most distinctive terms co-occur in any pre-cutoff paper?), and
overlap with pre-cutoff review articles (was the question already posed
in reviews?).

**D. Falsifiability / answerability** — rule-based markers plus a blinded
LLM structured coding of {observable, competing_hypotheses,
quantitative_test, falsification_path} for every question.

**E. Tractability** — facilities named in the question checked against a
lexicon of ~50 astronomical facilities with operational epochs: counts of
already-operational vs. future instruments, archival-data markers,
longitudinal / theory / large-sample requirements.

**F. Field environment (controls)** — subfield publication volume and
growth at the cutoff, review activity, share of all astro-ph output, and
proximity of the subfield's next major facility launch (capped at 10
years). These are confounders to control for, not virtues of the
question: without them a model mistakes hot fields for good questions.

### 3.4 Tiered outcome labels

**Tier 1 (automatic candidates).** For each question we retrieve future
papers from its subfield's future window: relevant-paper count (cosine ≥
0.18), weak-relevance count, first-follow-up date, lead time in months,
and review recognition.

**Tier 2 (strict, blinded LLM judge).** For every question with any
plausible future evidence (956 of 980; the rest are auto-labeled
not_addressed), a GPT-4o judge sees the question and its top-8 retrieved
future records — never the generation source. The rubric is deliberately
strict and two-stage: the judge first classifies every candidate paper as
**direct** (investigates this question's own objects, quantities, or
claimed relationship), **adjacent** (same topic only), or unrelated, and
only then assigns `addressed`, `answer_status` ∈ {answered,
partially_addressed, posed_but_open, premise_refuted, not_addressed},
`premise_status`, supporting ids, a rationale, and a confidence.
`addressed` is *derived*: it requires at least one direct candidate. This
rubric matters — a first-pass lenient judge (GPT-4o-mini, same evidence)
called 94% of questions addressed by accepting topical overlap; the
strict per-candidate rubric brings the rate to 52.7% and restores
variance to every analysis downstream.

**Tier 3 (independent verification).** A second model (GPT-4o-mini)
re-judges all premise_refuted cases plus a 20% random sample under the
identical rubric (n=200): raw agreement on `addressed` is 63.5%
(κ=0.21), status agreement 58.4% (κ=0.22). The disagreement is
one-directional — the weaker model is systematically more lenient,
almost never the reverse (its "partially_addressed" precision against
the strict judge is 0.99) — so we treat the strict labels as primary and
release both, with all rationales and supporting ids, for audit. These
κ values sit exactly at the measurement ceiling Paper 2 v1.1
established for outcome taxonomies of this kind: in its seven-rater
study, two independent, careful human annotators agreed at only κ=0.17,
and every judge model matched a professional annotator about as well as
the humans matched each other (κ=0.17–0.26). We therefore adopt Paper
2's reporting discipline: absolute rates are rater-relative, and every
comparative claim in this paper is made under a single fixed judge.

## 4. Models and evaluation protocol

Baselines before models, interpretable models before ensembles: majority
class; field-popularity-only; question-text-only (bag of words); then L2
and L1 logistic regression, random forest, and gradient boosting on the
structured features; multinomial GBM for the five-way status; a Cox
proportional-hazards model for time-to-first-follow-up; and within-cell
ranking concordance between model scores and future paper counts.

All headline numbers use the **temporal split** (train 2012–2016,
validate 2018, test 2020). Generalization is probed with
**leave-one-subfield-out** and a fixed **cross-domain split** (train:
exoplanet atmospheres, disks, stellar activity, galaxy evolution → test:
FRBs, gravitational waves, cosmology tensions, compact objects). Random
splits are not reported anywhere in this paper.

Predictor identification goes beyond feature importance: (i) univariate
associations; (ii) controlled logistic effects with subfield, cutoff,
source, and environment covariates; (iii) feature-group ablations; (iv)
sign-stability of controlled effects across cutoffs, subfields, and
generation sources. A predictor is called robust only if it survives all
four.

### Pre-stated hypotheses

- **H1** Cross-source evidence tension predicts future attention better
  than semantic novelty.
- **H2** Observation availability predicts short-term attention but not
  premise refutation.
- **H3** Explicit competing hypotheses raise the probability of being
  answered, given attention.
- **H4** Novelty shows an inverted-U: very high novelty reduces
  short-term attention.
- **H5** Field popularity inflates future paper counts without raising
  answer rates.
- **H6** High-tension × high-tractability questions succeed most.

## 5. Results

### 5.1 What happened to the questions

Of 980 frozen questions, **52.7% were substantively addressed** within
five years of their cutoff: 474 partially addressed, 30 posed-but-open,
8 answered, 4 premise-refuted, 464 not addressed. Addressed questions
drew a mean of 3.8 relevant future papers and were first engaged after a
mean of 16.7 months. Rates are stable across cutoffs (0.45–0.57),
confirming that no single era drives the results.

Generation source separates outcomes exactly as a functioning label
should:

| Source | Addressed |
| --- | --- |
| Direct LLM | 0.88 |
| Author future-work | 0.51 |
| Evidence-tension | 0.47 |
| Weak/negative controls | 0.32 |
| Paper 1 evidence-graph (n=10) | 0.20 |

Controls land lowest — the dataset's sanity check passes. The direct-LLM
rate of 0.88 should be read with care: the generating model may know,
parametrically, which 2016-era questions became hot topics, an
irreducible leakage channel we document rather than claim to eliminate
(section 7). Paper 2 v1.1 measures this channel directly and finds the
same signature at scale: its direct-LLM baseline has the highest
phrasing similarity to future literature of any system, near-ceiling
engagement (96% at n=125), yet an answered rate statistically
indistinguishable from random templates and zero premise refutations
outside the one era its weights could have memorized — "memorized
relevance without specific foresight". The mined sources (tension,
future-work) are built from pre-cutoff sentences and cannot leak this
way; that they land at ~0.5 against a strict judge is the honest base
rate of real open problems. The ten Paper 1 questions — all engaged by
future literature under Paper 2's adjudicated pilot evaluation
(coverage 100%) — score only 2/10 here, a direct measurement of how
much this paper's abstract-only retrieval and strict direct-engagement
bar *under-count* engagement for narrow, object-level questions (they
average 3.4 entities per question vs. 1.5 elsewhere). Paper 2's own
cross-instance anchor shows the same instrument-dependence from the
other side: re-judging its ten questions on a 2.4× larger corpus
flipped four labels in both directions — outcome rates are functions of
the (corpus, retriever, judge) triple, and are comparable only within
one instance.

### 5.2 Predicting future attention (Task A)

Temporal split, test = cutoff 2020 (n=210, base rate 0.53):

| Model | AUC | Notes |
| --- | --- | --- |
| Majority class | 0.500 | |
| Field popularity only | 0.519 | popularity is *not* a question-level predictor |
| Question text only (BoW) | 0.772 | absorbs source style + topic hotness |
| Logistic (structured, L2) | 0.713 | interpretable |
| Logistic (L1) | 0.707 | |
| Random forest | 0.718 | |
| Gradient boosting | 0.688 | |

Three observations. First, everything question-level beats field
popularity by a wide margin: which questions succeed is not just which
fields are hot. Second, raw question text is the strongest single signal
— but it is also the least meaningful, because it can encode the
generator's stylistic fingerprint (direct-LLM questions are both
stylistically distinctive and 88% addressed). Third, the structured
features reach AUC 0.71 while being *auditable*: every unit of signal is
a named, cutoff-time property.

**Out-of-domain.** Trained on four subfields and tested on four entirely
unseen ones (n=470), the structured logistic model holds AUC **0.710**
(GBM 0.647). Leave-one-subfield-out AUCs span 0.62–0.78 with every
subfield above chance (stellar activity 0.78, gravitational waves 0.75,
cosmology 0.71, disks 0.70, exoplanet atmospheres 0.69, galaxy evolution
0.66, FRBs 0.66, compact objects 0.62). Question-level predictors are
not a memorized property of any one community.

**Ablations** (GBM, temporal test): evidence-context is the load-bearing
group — dropping it costs the most (0.688 → 0.665) and it alone nearly
matches the full model (0.685). Text-only structure reaches 0.578,
falsifiability 0.543, tractability 0.540, environment 0.550. Dropping
the novelty group *helps* (0.708), foreshadowing the hypothesis results.

### 5.3 Which properties robustly predict attention

Controlled logistic effects (standardized, with subfield + cutoff +
source + environment covariates), with sign-stability across the 5
cutoffs / 8 subfields / 4 sources:

| Predictor | β | p | Sign-stable (cutoffs/subfields/sources) |
| --- | --- | --- | --- |
| Review overlap (question already visible in pre-cutoff reviews) | **+0.34** | 4×10⁻⁵ | 1.00 / 0.88 / 1.00 |
| N. directly similar prior papers | **+0.26** | 0.005 | 0.80 / 1.00 / 0.75 |
| Instrument diversity of prior evidence | **−0.21** | 0.005 | 1.00 / 0.75 / 1.00 |
| Explicit competing hypotheses (LLM-coded) | **+0.17** | 0.021 | 0.80 / 0.88 / 0.50 |
| Semantic novelty (distance to prior corpus) | +0.17 | 0.030 | 1.00 / 0.75 / 0.50 |
| Entity count | −0.17 | 0.039 | 0.80 / 0.75 / 0.50 |
| Specificity score | −0.16 | 0.041 | 0.80 / 1.00 / 0.50 |
| Evidence tension density | +0.06 | 0.52 | 0.60 / 0.62 / 0.50 |

The picture that survives all four analyses:

1. **The community mostly answers questions it has already begun to
   recognize.** Overlap with pre-cutoff reviews is the strongest, most
   stable predictor — future attention is highly autocorrelated with
   present institutional attention, even after controlling for field
   size and growth.
2. **A dense base of directly related evidence predicts uptake;
   evidence *tension* per se does not.** What matters is that many
   papers already bear on the question, not that they disagree.
3. **Questions whose evidence spans many instruments are *less* likely
   to be engaged** — consistent with multi-instrument questions being
   synthesis-level problems that no single group owns.
4. **Explicitly articulated competing hypotheses help** — the one
   falsifiability property with predictive teeth.
5. **Specificity cuts against measured attention.** Highly
   entity-specific questions are engaged less often and *more slowly*
   (Cox HR 0.73 per SD, p<10⁻⁴) under literature-bounded labels — partly
   a real narrowness effect, partly the retrieval bound (section 7).

**Time-to-engagement (Task C).** In the Cox model (516 events/980),
retrieval-similarity mass dominates speed (HR 3.77 per SD), field growth
accelerates engagement (HR 1.15), an explicit falsification path
accelerates it (HR 1.15), and specificity (HR 0.73) and tension density
(HR 0.88) slow it. Within-cell ranking of 2020 questions by model score
correlates with realized future paper counts at mean Spearman ρ=0.29
across the 8 test cells.

**Outcome type (Task B).** The five-way status classifier reaches macro-F1
0.23 (accuracy 0.56): it separates addressed/partial from not-addressed
but cannot yet learn the rare classes (8 answered, 4 refuted in total) —
exactly the sample-size regime the design anticipated, and the reason
Task A is primary at n≈1,000.

### 5.4 Pre-stated hypotheses: mostly falsified

| Hypothesis | Verdict |
| --- | --- |
| H1 tension > novelty | **No.** Novelty carries a small positive controlled effect (β=0.17, p=0.03); tension density is null (β=0.06, p=0.52). |
| H2 observation availability → attention | **No.** Operational-facility count is null on attention (p=0.91); refutation side untestable (4 cases). |
| H3 competing hypotheses → answered | **Directionally yes, underpowered.** On attention: β=0.17, p=0.02. On answered-given-addressed: β=0.56, p=0.08. |
| H4 novelty inverted-U | **No.** Quadratic term null (p=0.46). |
| H5 popularity → counts, not answers | **Untestable as stated.** Count effect positive but n.s. (p=0.21); answer side degenerate (8 answered). |
| H6 tension × tractability interaction | **Marginal.** β=0.16, p=0.053 — suggestive, unconfirmed. |

We regard this table as a feature of the method, not a failure of the
paper: five of six mechanism intuitions that sound compelling in a
proposal do not survive contact with a thousand historical outcomes and
proper controls.

### 5.5 Interpretable prediction cards

The released pipeline emits, for any frozen question, a card of the form:

```
Question: How does the dark matter distribution in the innermost regions
  of dwarf galaxies compare to the predictions of cold dark matter
  simulations? [evidence_tension, cutoff 2020]
Predicted P(future attention): 0.50 → observed: addressed (partial)
Main positive factors: review overlap, directly similar prior papers,
  explicit competing explanations
Main negative factors: high entity specificity, multi-instrument
  evidence base
```

A cautionary card from the Paper 1 import: *"To what extent have actual
JWST observations validated pre-launch predictions that aerosol-free
TRAPPIST-1 CO₂ features should be detectable…"* (frozen at 2020-12-31,
predicted 0.003, observed not addressed under our retrieval). The
phrase "actual JWST observations" presumes data that did not exist at
the cutoff — an instance of generation-time knowledge contaminating the
*phrasing* of a nominally frozen question. Every LLM-mediated pipeline,
including ours and Paper 1's, needs phrasing audits of exactly this
kind — Paper 2 v1.1's released curation log institutionalizes them,
recording presupposition and phrasing repairs made before freezing —
and the dataset makes such audits possible because provenance is
stored verbatim.

### 5.6 A learned prior over evidence-graph structures

Paper 2's outlook section poses the question its architecture needs
answered next: *which structural patterns of pre-cutoff evidence most
often lead to questions the future answers, advances, or refutes?* Our
dataset can answer it empirically, because every tension-mined question
stores its generating evidence cluster verbatim — sentences, arXiv ids,
dates. We derive a structural feature record for all 374 tension
questions with the same transparent rules used elsewhere in the pipeline
(`pipeline/structure.py`; frozen in
`data/features/structure_features_v1.jsonl`): cluster breadth (papers
and claims in tension), evidence-age profile, a four-way tension
typology (explicit contradiction, unexplained result, methodological
challenge, detection-vs-limit stance opposition), quantification of the
discrepancy (N-sigma / factor-of markers), instrument structure, and
**recurrence** — whether the same tension (shared source papers, same
subfield) was already detectable at an earlier cutoff, a cutoff-time
property since earlier past-windows are subsets of the current one.

| Structure indicator | Addressed rate | Controlled effect on addressed |
| --- | --- | --- |
| 4-paper cluster (vs. 2–3-paper) | 0.54 vs. 0.32 | **β=+0.45/SD, p=0.001** |
| Explicit contradiction markers | 0.49 vs. 0.28 | **β=+0.30, p=0.014** |
| Unexplained-result markers | 0.49 vs. 0.43 | +0.19, p=0.10 |
| Methodological challenge | 0.46 | −0.11, n.s.; 0/61 resolved |
| Quantified tension (N-σ) | 0.48 | null; carries all 3 refutations |
| Recurrent across cutoffs | 0.49 | null; carries all 6 resolutions |

Three structural regularities emerge, one per outcome layer:

1. **Breadth buys attention.** The number of independent papers already
   in tension is the strongest structural predictor of both being
   addressed (0.54 for 4-paper clusters vs. 0.32 for 2–3-paper, Fisher
   p=6×10⁻⁵) and future paper volume (β=+0.12, p=0.03) — the
   structure-level counterpart of the question-level "density of
   directly related prior evidence" predictor of section 5.3. Fresher
   evidence also draws more future papers (evidence age on volume:
   β=−0.17, p=0.002).
2. **Explicit contradiction buys engagement; methodological doubt does
   not resolve.** Clusters whose sentences explicitly disagree are
   addressed at 0.49 vs. 0.28 without such markers (p=0.0095), and five
   of the six resolved cases are contradiction-typed. Clusters built on
   methodological-challenge language engage the community at the average
   rate but resolved nothing (0/61) — consistent with Paper 2's finding
   that questions must aim at a contestable claim, not at a diffuse
   worry about methods.
3. **Quantification and persistence mark the refutation channel.** All
   three premise refutations come from *quantified* clusters (3/63 vs.
   0/311 unquantified, Fisher p=0.005), and all six resolutions come
   from *recurrent* tensions (6/194 vs. 0/180 first-seen, p=0.03) —
   structures the miner found again at two or more successive cutoffs.
   One honesty note bounds this: the three refutations trace to a
   single physical case (the FRB Galactic-latitude detection-rate
   discrepancy, independently re-mined at the 2016, 2018, and 2020
   cutoffs and refuted by the same 2021 study), so they constitute one
   independent refutation event, and we report this layer as a
   case-level observation, not a statistical claim. Its shape is
   nonetheless exactly what Paper 2's protocol predicts: a persistent,
   quantified, explicitly contradictory tension is a published
   conclusion waiting to be overturned.

The prior is learnable and transfers out of time: a logistic model on
structural features alone, trained on the 2012–2018 cutoffs, ranks 2020
tension questions at AUC 0.586 (n=80) — modest in absolute terms, but
notable against the right comparison: restricted to the same
single-source subset, the full 50-feature question-level record manages
only AUC 0.502, because most of its cross-source signal is generator
style and topic hotness that vanish within one source. Within a single
generator, the generating *structure* is the recoverable signal. And
because every structural feature is computable at mining time — before
any question is phrased — the fitted coefficients
(`results/structure_prior.json`) constitute a ranking prior for exactly
the machine-scale structural search Paper 2's outlook proposes:
enumerate candidate tensions, rank by breadth, explicit contradiction,
quantification, and persistence, verbalize the survivors. The
prospective instance (section 8) is the pre-registered test of whether
this prior holds on a future no model has seen.

## 6. Discussion

**What the dataset says about question value.** The strongest honest
summary: five-year scientific attention is predictable to a useful
degree (AUC ~0.7 out-of-time and out-of-domain) from cutoff-time
properties, but the predictive properties are about the question's
*relationship to its community* — already-in-reviews, dense direct
evidence, articulated alternatives — more than about the intrinsic
virtues the literature on "good questions" celebrates (novelty, tension,
specificity). Attention follows preparation. Whether that is how science
*should* allocate attention is precisely the kind of question this
dataset now lets others study: the rare premise-refuted cases, the
slow-burning specific questions, and the unaddressed-but-well-formed
tail are all released with full provenance.

**Specificity and the measurement bound.** The negative specificity
effect is partly real (narrow questions have small addressable
communities) and partly instrumental: abstract-level TF-IDF retrieval
misses engagement with object-level questions, as the Paper 1 subset
demonstrates (2/10 here vs. 10/10 under Paper 2's full-text
evaluation). We report it as "less *measured* attention", never "worse
questions" — and this is the main reason Task A labels should be read
as lower bounds. Paper 2 v1.1 attacks the same confound from the metric
side with a specificity-adjusted foresight score that down-weights
broad questions anything would engage; under that rubric object-anchored
questions are precisely what makes outcomes *resolvable* (its
structure-first pipelines, 95–100% object-anchored, refute premises in
every era; its LLM-only pipeline, 5% anchored, refutes almost none).
The two results agree that raw engagement over-rewards breadth and
under-rewards specificity; they differ only in the remedy — adjust the
metric (Paper 2) versus control and caveat in the regression (here).

**The direct-LLM anomaly is a warning, not a victory.** The 88%
addressed rate for direct-LLM questions is exactly what parametric
leakage would produce: a model asked in hindsight for "the most
important open questions of 2016" can simply recall what 2017–2021
worked on. Because source is a covariate in every regression, the
predictor analysis is shielded from this; but any future benchmark that
scores generators by addressed-rate alone will be gamed by hindsight
knowledge. Mined sources with verbatim pre-cutoff provenance are the
defensible foundation. Paper 2 v1.1's generator decomposition and
temporal stress test now make this point causally rather than
suggestively: over identical pre-cutoff evidence structures, a
weight-free structural generator finds engaged questions in every era,
the LLM adds value only as a separable verbalization layer, and the
LLM-only pipeline's performance never rested on specifics that
memorization could supply — its answered rate is flat across the
model's training boundary because a topic prior needs no memory of the
future. Our regression-level shielding (source as a covariate in every
model) and Paper 2's decomposition are complementary defenses against
the same threat, operating at the analysis and generation levels
respectively.

## 7. Limitations

- **Abstract-level corpus.** Full texts are not used; tension and
  future-work mining see only abstracts, and citation-weighted attention
  is unavailable without a second data provider. Paper counts, lead
  times, and review recognition stand in for citation impact.
- **Retrieval-bounded labels.** A question can be engaged by literature
  our TF-IDF retrieval misses; not_addressed is a lower bound on
  engagement, uniformly across sources but not uniformly across
  specificity (see § 6).
- **LLM judges, not humans.** Label reliability is estimated with an
  independent second model rather than human adjudication (63.5% raw
  agreement, κ=0.21, disagreement one-directionally lenient). Paper 2
  v1.1's seven-rater study reframes what such numbers can mean: on the
  analogous taxonomy, two independent human annotators agreed with each
  other at only κ=0.17, every judge model matched the professional
  annotator as well as or better than the humans matched each other,
  and model–model agreement (κ up to 0.60) would have overstated
  reliability threefold. The weakness lives in outcome taxonomies of
  this kind, not in our judge specifically; accordingly, absolute rates
  here are rater-relative, all comparative results hold the judge
  fixed, and a reliability-gated taxonomy (fewer labels, checkable
  conditions, a measured human–human κ before release) is the shared
  v2 requirement for both papers. The annotation protocol, rationales,
  and supporting ids are released so human passes can replace or audit
  the judges; per-class precision tables are in the released agreement
  report.
- **Parametric-knowledge leakage.** The direct-LLM source (and, weakly,
  LLM phrasing of mined material) can import post-cutoff knowledge
  despite freeze instructions — including into question phrasing, as the
  JWST card shows. Source is controlled for in all analyses; the
  direct-LLM addressed rate should not be read as generator quality.
- **Rare outcomes remain rare.** 8 answered and 4 premise-refuted
  questions cannot support the Task B taxonomy or H2/H5 as stated;
  the design document's projection that these classes need 2,000–5,000
  questions is confirmed empirically.
- **One domain.** Astronomy only; the cross-subfield transfers here are
  necessary but not sufficient evidence of cross-science generality.

## 8. Conclusion

We built the first temporally grounded, multi-cutoff, multi-source
dataset of scientific questions with future-outcome labels, and used it
to replace two common practices: scoring questions by how they sound,
and validating question generators on single-digit case studies. At
n=980 with strict blinded labeling, future scientific attention is
predictable (AUC ≈ 0.71 out-of-time and out-of-domain) — and the robust
predictors are community-relational, not rhetorical. Most pre-registered
mechanism hypotheses failed, which is the point: a dataset an order of
magnitude larger than its predecessor makes intuitions testable, and
most did not survive. The dataset, every prompt, every judge rationale,
and the full pipeline are released for the community to reuse, audit,
and extend to other sciences. A forward-looking test is already armed:
Paper 2 v1.1's prospective instance — 200 questions from four
generators, frozen 2026-08-17 with a pre-registered 2027–2030 scoring
window — will let the predictors learned here be evaluated against a
future no model has seen, converting this paper's retrospective claims
into ones that time itself will grade.

## Reproducibility

All code, the frozen dataset, labels, and results are released; every
pipeline stage is a single script, all LLM calls are cached and pinned to
temperature 0, and the offline test suite plus a temporal-isolation
validator run in CI. Rebuilding the corpus from the public arXiv API and
regenerating the dataset reproduced the frozen questions byte-for-byte.
The frozen dataset is additionally published on the Hugging Face Hub at
[`huiluckylucky/scientific-question-outcomes`](https://huggingface.co/datasets/huiluckylucky/scientific-question-outcomes),
loadable in one call, with the structure-prior and second-judge subsets as
separate configurations.

## References

Baek, J., Jauhar, S. K., Cucerzan, S., & Hwang, S. J. (2024). ResearchAgent:
Iterative Research Idea Generation over Scientific Literature with Large
Language Models. arXiv:2404.07738.

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014).
Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
Overfitting on Out-of-Sample Performance. *Notices of the AMS*, 61(5),
458–471.

Fortunato, S., Bergstrom, C. T., Börner, K., Evans, J. A., Helbing, D.,
Milojević, S., Petersen, A. M., Radicchi, F., Sinatra, R., Uzzi, B.,
Vespignani, A., Waltman, L., Wang, D., & Barabási, A.-L. (2018). Science of
science. *Science*, 359(6379), eaao0185.

Foster, J. G., Rzhetsky, A., & Evans, J. A. (2015). Tradition and Innovation
in Scientists' Research Strategies. *American Sociological Review*, 80(5),
875–908.

Jiang, B. (2026). HindSight: Evaluating LLM-Generated Research Ideas via
Future Impact. arXiv:2603.15164.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., &
Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World
GitHub Issues? *ICLR*.

Kitano, H. (2021). Nobel Turing Challenge: creating the engine for
scientific discovery. *npj Systems Biology and Applications*, 7, 29.

Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement
for Categorical Data. *Biometrics*, 33(1), 159–174.

Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The
AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.
arXiv:2408.06292.

Mao, H. (2026a). Evidence-Based Scientific Question Discovery: A Framework
with Historical Backtesting. arXiv:2608.09968. — "Paper 1".

Mao, H. (2026b). Historical Backtesting for Scientific Question Discovery: A
Protocol and Astronomy Pilot. arXiv:2608.16795 (v1.1, sealed). — "Paper 2".

Rzhetsky, A., Foster, J. G., Foster, I. T., & Evans, J. A. (2015). Choosing
experiments to accelerate collective discovery. *PNAS*, 112(47),
14569–14574.

Si, C., Yang, D., & Hashimoto, T. (2024). Can LLMs Generate Novel Research
Ideas? A Large-Scale Human Study with 100+ NLP Researchers. arXiv:2409.04109.

Si, C., Hashimoto, T., & Yang, D. (2025). The Ideation–Execution Gap:
Execution Outcomes of LLM-Generated versus Human Research Ideas.
arXiv:2506.20803.

Swanson, D. R. (1986). Fish Oil, Raynaud's Syndrome, and Undiscovered Public
Knowledge. *Perspectives in Biology and Medicine*, 30(1), 7–18.

Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. (2013). Atypical
Combinations and Scientific Impact. *Science*, 342(6157), 468–472.

Wang, D., & Barabási, A.-L. (2021). *The Science of Science*. Cambridge
University Press.

Wang, Q., Downey, D., Ji, H., & Hope, T. (2024). SciMON: Scientific
Inspiration Machines Optimized for Novelty. *ACL*.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z.,
Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023).
Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*.

Zou, A., Xiao, T., Jia, R., Kwon, J., Mazeika, M., Li, R., Song, D.,
Steinhardt, J., Evans, O., & Hendrycks, D. (2022). Forecasting Future World
Events with Neural Networks. *NeurIPS*.
