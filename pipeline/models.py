"""Model suite and evaluation protocols for Paper 3.

The scientific goal is predictor discovery, not leaderboard accuracy, so the
suite is ordered from trivial baselines to interpretable models to stronger
ensembles, and every evaluation is out-of-time or out-of-domain:

- temporal split: train on cutoffs 2012-2016, validate on 2018, test on 2020;
- leave-one-subfield-out (LOSO);
- cross-domain: train on four subfields, test on the other four.

Random train/test splits are deliberately not implemented — they leak, because
near-duplicate questions from the same era and topic would land on both sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_GROUPS = {
    "text": "txt_",
    "evidence": "ev_",
    "novelty": "nov_",
    "falsifiability": "fal_",
    "tractability": "tra_",
    "environment": "env_",
}

TRAIN_CUTOFFS = (2012, 2014, 2016)
VAL_CUTOFF = 2018
TEST_CUTOFF = 2020

CROSS_DOMAIN_TRAIN = (
    "exoplanet_atmospheres",
    "protoplanetary_disks",
    "stellar_activity",
    "galaxy_evolution",
)
CROSS_DOMAIN_TEST = (
    "fast_radio_bursts",
    "gravitational_waves",
    "cosmology_tensions",
    "compact_objects",
)


def feature_columns(df: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    prefixes = tuple(FEATURE_GROUPS[g] for g in (groups or FEATURE_GROUPS))
    return [c for c in df.columns if c.startswith(prefixes)]


def make_model(kind: str):
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, C=1.0)),
            ]
        )
    if kind == "logistic_l1":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000, penalty="l1", solver="liblinear", C=0.5
                    ),
                ),
            ]
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, random_state=7, n_jobs=-1
        )
    if kind == "gbm":
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=7
        )
    raise ValueError(kind)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    out = {
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float(np.mean((prob >= 0.5) == y_true)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    else:
        out["auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def eval_model(model, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    return binary_metrics(np.asarray(y_te), prob)


def temporal_split(df: pd.DataFrame):
    tr = df[df.cutoff_year.isin(TRAIN_CUTOFFS)]
    va = df[df.cutoff_year == VAL_CUTOFF]
    te = df[df.cutoff_year == TEST_CUTOFF]
    return tr, va, te


def run_baselines(df: pd.DataFrame, target: str = "y_addressed") -> dict:
    """Baselines on the temporal split (train 2012-2016 -> test 2020)."""
    tr, _, te = temporal_split(df)
    y_tr, y_te = tr[target].values, te[target].values
    results = {}

    # 1. majority class
    p = float(np.mean(y_tr))
    results["majority_class"] = binary_metrics(y_te, np.full(len(y_te), p))

    # 2. subfield popularity only
    env_cols = feature_columns(df, ["environment"])
    results["subfield_popularity_only"] = eval_model(
        make_model("logistic"), tr[env_cols], y_tr, te[env_cols], y_te
    )

    # 3. question text only (bag of words)
    vec = TfidfVectorizer(stop_words="english", max_features=3000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr["question"])
    Xte = vec.transform(te["question"])
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(Xtr, y_tr)
    results["question_text_only"] = binary_metrics(y_te, clf.predict_proba(Xte)[:, 1])

    return results


def run_structured_models(df: pd.DataFrame, target: str = "y_addressed") -> dict:
    tr, va, te = temporal_split(df)
    cols = feature_columns(df)
    results = {}
    for kind in ("logistic", "logistic_l1", "random_forest", "gbm"):
        model = make_model(kind)
        res = eval_model(model, tr[cols], tr[target].values, te[cols], te[target].values)
        # validation-cutoff metrics for the same fitted model
        if len(va):
            prob_va = model.predict_proba(va[cols])[:, 1]
            res["val_auc"] = binary_metrics(va[target].values, prob_va)["auc"]
        results[kind] = res
    return results


def run_ablations(df: pd.DataFrame, target: str = "y_addressed") -> dict:
    tr, _, te = temporal_split(df)
    y_tr, y_te = tr[target].values, te[target].values
    all_groups = list(FEATURE_GROUPS)
    results = {}
    cols = feature_columns(df, all_groups)
    results["all_features"] = eval_model(make_model("gbm"), tr[cols], y_tr, te[cols], y_te)
    for dropped in all_groups:
        keep = [g for g in all_groups if g != dropped]
        cols = feature_columns(df, keep)
        results[f"without_{dropped}"] = eval_model(
            make_model("gbm"), tr[cols], y_tr, te[cols], y_te
        )
    for only in all_groups:
        cols = feature_columns(df, [only])
        results[f"only_{only}"] = eval_model(
            make_model("gbm"), tr[cols], y_tr, te[cols], y_te
        )
    return results


def run_loso(df: pd.DataFrame, target: str = "y_addressed") -> dict:
    cols = feature_columns(df)
    results = {}
    for subfield in sorted(df.subfield.unique()):
        tr = df[df.subfield != subfield]
        te = df[df.subfield == subfield]
        if te[target].nunique() < 2:
            continue
        results[subfield] = eval_model(
            make_model("gbm"), tr[cols], tr[target].values, te[cols], te[target].values
        )
    return results


def run_cross_domain(df: pd.DataFrame, target: str = "y_addressed") -> dict:
    cols = feature_columns(df)
    tr = df[df.subfield.isin(CROSS_DOMAIN_TRAIN)]
    te = df[df.subfield.isin(CROSS_DOMAIN_TEST)]
    out = {
        "train_subfields": list(CROSS_DOMAIN_TRAIN),
        "test_subfields": list(CROSS_DOMAIN_TEST),
        "gbm": eval_model(
            make_model("gbm"), tr[cols], tr[target].values, te[cols], te[target].values
        ),
        "logistic": eval_model(
            make_model("logistic"), tr[cols], tr[target].values, te[cols], te[target].values
        ),
    }
    return out


def run_multiclass(df: pd.DataFrame) -> dict:
    """Task B: answer-status multiclass on the temporal split."""
    tr, _, te = temporal_split(df)
    cols = feature_columns(df)
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=7
    )
    clf.fit(tr[cols], tr["answer_status"])
    pred = clf.predict(te[cols])
    labels = sorted(df["answer_status"].unique())
    per_class = f1_score(te["answer_status"], pred, labels=labels, average=None, zero_division=0)
    return {
        "n_test": int(len(te)),
        "macro_f1": float(
            f1_score(te["answer_status"], pred, average="macro", zero_division=0)
        ),
        "accuracy": float(np.mean(pred == te["answer_status"])),
        "per_class_f1": {l: float(v) for l, v in zip(labels, per_class)},
    }


def run_survival(df: pd.DataFrame) -> dict:
    """Task C: Cox model for time to first substantive follow-up."""
    import statsmodels.api as sm

    d = df.copy()
    d["duration"] = d["lead_time_months"].fillna(60.0).clip(1, 60)
    d["event"] = d["y_addressed"].astype(int)
    cols = [
        "ev_tension_density",
        "ev_top5_mean_sim",
        "nov_semantic_distance",
        "fal_observable",
        "fal_falsification_path",
        "tra_n_facilities_operational",
        "tra_archival_data",
        "env_pub_volume_3y",
        "env_growth_ratio",
        "txt_specificity",
    ]
    cols = [c for c in cols if c in d.columns]
    X = (d[cols] - d[cols].mean()) / d[cols].std().replace(0, 1)
    model = sm.PHReg(d["duration"], X, status=d["event"])
    fit = model.fit()
    summary = {
        c: {
            "log_hr": float(fit.params[i]),
            "hr": float(np.exp(fit.params[i])),
            "p": float(fit.pvalues[i]),
        }
        for i, c in enumerate(cols)
    }
    return {"n": int(len(d)), "events": int(d["event"].sum()), "coefficients": summary}


def ranking_concordance(df: pd.DataFrame, target_count: str = "future_paper_count") -> dict:
    """Within-cell concordance between model score and future paper counts."""
    from scipy.stats import spearmanr

    tr, _, te = temporal_split(df)
    cols = feature_columns(df)
    model = make_model("gbm")
    model.fit(tr[cols], tr["y_addressed"].values)
    te = te.copy()
    te["score"] = model.predict_proba(te[cols])[:, 1]
    rhos = []
    for (_, _), cell in te.groupby(["subfield", "cutoff_year"]):
        if len(cell) >= 8 and cell[target_count].nunique() > 2:
            rho = spearmanr(cell["score"], cell[target_count]).statistic
            if not np.isnan(rho):
                rhos.append(float(rho))
    return {
        "n_cells": len(rhos),
        "mean_spearman": float(np.mean(rhos)) if rhos else float("nan"),
        "per_cell": rhos,
    }
