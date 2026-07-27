"""Stage 5: models, evaluation protocols, predictor analyses, hypotheses.

Reads the frozen questions + features + labels, assembles the modeling
table, and writes all results to results/:

- dataset_table.csv           (the assembled modeling table, committed)
- baselines.json              temporal-split baselines
- structured_models.json      logistic / L1 / RF / GBM
- ablations.json              feature-group ablations
- loso.json                   leave-one-subfield-out
- cross_domain.json           train 4 subfields -> test the other 4
- multiclass.json             Task B answer-status
- survival.json               Cox model for time-to-follow-up
- ranking.json                within-cell ranking concordance
- univariate.csv              per-feature associations
- controlled_effects.csv      focal predictors with full controls
- stability.csv               sign-stability across cutoffs/subfields/sources
- hypotheses.json             H1-H6
- prediction_cards.json       interpretable example outputs
- figures/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import analysis, models

QUESTIONS = Path("data/questions/questions_v1.jsonl")
FEATURES = Path("data/features/features_v1.jsonl")
LABELS = Path("data/labels/labels_v1.jsonl")
RESULTS = Path("results")

FOCAL_PREDICTORS = [
    "ev_tension_density",
    "ev_top5_mean_sim",
    "ev_n_similar",
    "ev_n_instruments",
    "ev_has_native_record",
    "nov_semantic_distance",
    "nov_new_pair",
    "nov_review_overlap",
    "fal_observable",
    "fal_competing_llm",
    "fal_quant_test",
    "fal_falsification_path",
    "tra_n_facilities_operational",
    "tra_archival_data",
    "tra_needs_new_instrument",
    "txt_specificity",
    "txt_has_alternatives",
    "txt_n_entities",
]


def assemble() -> pd.DataFrame:
    qs = pd.DataFrame([json.loads(l) for l in QUESTIONS.open()])
    fs = pd.DataFrame([json.loads(l) for l in FEATURES.open()])
    ls = pd.DataFrame([json.loads(l) for l in LABELS.open()])
    df = qs.merge(fs, on="question_id").merge(
        ls.drop(columns=["top_ids", "top_sims", "supporting_ids", "rationale"]),
        on="question_id",
    )
    df["y_addressed"] = df["addressed"].astype(int)
    return df


def save(obj, name: str) -> None:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".json"):
        path.write_text(json.dumps(obj, indent=2, default=float))
    else:
        obj.to_csv(path, index=False)
    print(f"[models] wrote {path}")


def figures(df: pd.DataFrame, ablations: dict, stability: pd.DataFrame) -> None:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Addressed rate by source and cutoff.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rate = df.groupby("source")["y_addressed"].mean().sort_values()
    axes[0].barh(rate.index, rate.values, color="#4878b0")
    axes[0].set_xlabel("addressed rate")
    axes[0].set_title("Future attention by generation source")
    rate2 = df.groupby("cutoff_year")["y_addressed"].mean()
    axes[1].plot(rate2.index, rate2.values, marker="o", color="#b04848")
    axes[1].set_xlabel("cutoff year")
    axes[1].set_ylabel("addressed rate")
    axes[1].set_title("Future attention by cutoff")
    fig.tight_layout()
    fig.savefig(fig_dir / "addressed_rates.png", dpi=150)
    plt.close(fig)

    # Ablation AUCs.
    keys = [k for k in ablations if k.startswith(("all", "without"))]
    aucs = [ablations[k]["auc"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(keys, aucs, color="#559955")
    ax.set_xlabel("AUC (temporal test, cutoff 2020)")
    ax.set_title("Feature-group ablations")
    ax.set_xlim(0.4, max(0.75, max(aucs) + 0.03))
    fig.tight_layout()
    fig.savefig(fig_dir / "ablations.png", dpi=150)
    plt.close(fig)

    # Stability of focal predictors.
    st = stability.dropna(subset=["pooled_beta"]).sort_values("pooled_beta")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4878b0" if c >= 0.75 else "#c0c0c0" for c in st["sign_consistency_cutoffs"].fillna(0)]
    ax.barh(st["feature"], st["pooled_beta"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("standardized controlled log-odds effect on addressed")
    ax.set_title("Controlled predictor effects (blue = sign-stable across cutoffs)")
    fig.tight_layout()
    fig.savefig(fig_dir / "predictor_effects.png", dpi=150)
    plt.close(fig)

    # Answer-status distribution by cutoff.
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["answered", "partially_addressed", "posed_but_open", "premise_refuted", "not_addressed"]
    tab = (
        df.groupby(["cutoff_year", "answer_status"]).size().unstack(fill_value=0)[
            [c for c in order if c in df["answer_status"].unique()]
        ]
    )
    tab_frac = tab.div(tab.sum(axis=1), axis=0)
    bottom = np.zeros(len(tab_frac))
    for col in tab_frac.columns:
        ax.bar(tab_frac.index.astype(str), tab_frac[col].values, bottom=bottom, label=col)
        bottom += tab_frac[col].values
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("fraction")
    ax.set_title("Outcome mix by cutoff")
    fig.tight_layout()
    fig.savefig(fig_dir / "outcome_mix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = assemble()
    RESULTS.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        c
        for c in df.columns
        if c not in ("source_material",)
    ]
    save(df[keep_cols], "dataset_table.csv")

    save(models.run_baselines(df), "baselines.json")
    save(models.run_structured_models(df), "structured_models.json")
    ablations = models.run_ablations(df)
    save(ablations, "ablations.json")
    save(models.run_loso(df), "loso.json")
    save(models.run_cross_domain(df), "cross_domain.json")
    save(models.run_multiclass(df), "multiclass.json")
    save(models.run_survival(df), "survival.json")
    save(models.ranking_concordance(df), "ranking.json")

    uni = analysis.univariate(df)
    save(uni, "univariate.csv")
    controlled = analysis.controlled_scan(df, FOCAL_PREDICTORS)
    save(controlled, "controlled_effects.csv")
    stab = analysis.stability(df, FOCAL_PREDICTORS)
    save(stab, "stability.csv")
    save(analysis.hypothesis_tests(df), "hypotheses.json")
    save(analysis.prediction_cards(df), "prediction_cards.json")

    figures(df, ablations, stab)
    print("[models] all results written")


if __name__ == "__main__":
    main()
