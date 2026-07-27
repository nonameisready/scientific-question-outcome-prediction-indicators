"""Predictor-identification analyses: univariate, controlled, stability, H1-H6.

Feature importance from a single model is not evidence that a predictor is
real. Each candidate predictor is examined four ways (design section 10):
univariate association, multivariate regression with environment and source
controls, group ablation (in models.py), and stability across cutoffs,
subfields, and generation sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from .models import TRAIN_CUTOFFS, feature_columns

CONTROL_SOURCES = ("evidence_tension", "direct_llm", "future_work", "control")


def _controls_frame(df: pd.DataFrame) -> pd.DataFrame:
    ctrl = pd.get_dummies(df["subfield"], prefix="sub", drop_first=True).astype(float)
    ctrl["cutoff_year_c"] = (df["cutoff_year"] - df["cutoff_year"].mean()).astype(float)
    src = pd.get_dummies(df["source"], prefix="src", drop_first=True).astype(float)
    env = df[["env_pub_volume_3y", "env_growth_ratio"]].astype(float)
    env = (env - env.mean()) / env.std().replace(0, 1)
    out = pd.concat([ctrl, src, env], axis=1)
    # Constant columns (e.g. cutoff year within a single-cutoff subset) make
    # the design singular; drop them so subgroup fits stay estimable.
    return out.loc[:, out.std() > 0]


def univariate(df: pd.DataFrame, target: str = "y_addressed") -> pd.DataFrame:
    """Per-feature AUC and rank correlation with the outcome."""
    rows = []
    y = df[target].values
    for col in feature_columns(df):
        x = df[col].values.astype(float)
        if np.std(x) == 0:
            continue
        try:
            auc = roc_auc_score(y, x) if len(np.unique(y)) > 1 else np.nan
        except ValueError:
            auc = np.nan
        rows.append(
            {
                "feature": col,
                "auc": float(auc),
                "abs_auc_dev": abs(float(auc) - 0.5) if not np.isnan(auc) else np.nan,
                "mean_pos": float(np.mean(x[y == 1])) if (y == 1).any() else np.nan,
                "mean_neg": float(np.mean(x[y == 0])) if (y == 0).any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_auc_dev", ascending=False)


def controlled_effect(
    df: pd.DataFrame, focal: str, target: str = "y_addressed"
) -> dict:
    """Logit of target on standardized focal feature + full controls."""
    x = df[focal].astype(float)
    if x.std() == 0:
        return {"feature": focal, "beta": np.nan, "p": np.nan}
    xz = (x - x.mean()) / x.std()
    X = _controls_frame(df).copy()
    X[focal] = xz.values
    X = sm.add_constant(X.astype(float), has_constant="add")
    try:
        fit = sm.Logit(df[target].astype(float), X).fit(disp=0, maxiter=200)
        return {
            "feature": focal,
            "beta": float(fit.params[focal]),
            "se": float(fit.bse[focal]),
            "p": float(fit.pvalues[focal]),
            "n": int(len(df)),
        }
    except Exception:  # noqa: BLE001 - separation/singularity on rare features
        return {"feature": focal, "beta": np.nan, "p": np.nan, "n": int(len(df))}


def controlled_scan(df: pd.DataFrame, features: list[str], target: str = "y_addressed") -> pd.DataFrame:
    return pd.DataFrame([controlled_effect(df, f, target) for f in features])


def stability(df: pd.DataFrame, features: list[str], target: str = "y_addressed") -> pd.DataFrame:
    """Sign-stability of controlled effects across cutoffs, subfields, sources."""
    rows = []
    for feat in features:
        betas_cutoff, betas_sub, betas_src = [], [], []
        for cy in sorted(df.cutoff_year.unique()):
            d = df[df.cutoff_year == cy]
            if d[target].nunique() > 1 and len(d) >= 60:
                betas_cutoff.append(controlled_effect(d, feat, target).get("beta"))
        for sf in sorted(df.subfield.unique()):
            d = df[df.subfield == sf]
            if d[target].nunique() > 1 and len(d) >= 60:
                betas_sub.append(controlled_effect(d, feat, target).get("beta"))
        for src in CONTROL_SOURCES:
            d = df[df.source == src]
            if d[target].nunique() > 1 and len(d) >= 60:
                betas_src.append(controlled_effect(d, feat, target).get("beta"))

        def sign_consistency(betas: list) -> float:
            vals = [b for b in betas if b is not None and not np.isnan(b)]
            if not vals:
                return np.nan
            pos = sum(1 for b in vals if b > 0)
            return max(pos, len(vals) - pos) / len(vals)

        rows.append(
            {
                "feature": feat,
                "pooled_beta": controlled_effect(df, feat, target).get("beta"),
                "sign_consistency_cutoffs": sign_consistency(betas_cutoff),
                "sign_consistency_subfields": sign_consistency(betas_sub),
                "sign_consistency_sources": sign_consistency(betas_src),
                "n_cutoff_fits": len([b for b in betas_cutoff if b is not None and not np.isnan(b)]),
                "n_subfield_fits": len([b for b in betas_sub if b is not None and not np.isnan(b)]),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- hypotheses


def hypothesis_tests(df: pd.DataFrame) -> dict:
    """Pre-stated hypotheses H1-H6 (design section 11)."""
    out: dict = {}
    d = df.copy()

    # H1: cross-source evidence tension beats semantic novelty for attention.
    h1_t = controlled_effect(d, "ev_tension_density")
    h1_n = controlled_effect(d, "nov_semantic_distance")
    out["H1"] = {
        "statement": "Evidence tension predicts future attention better than semantic novelty.",
        "tension_effect": h1_t,
        "novelty_effect": h1_n,
        "supported": bool(
            not np.isnan(h1_t.get("beta", np.nan))
            and h1_t["beta"] > 0
            and h1_t["p"] < 0.05
            and (np.isnan(h1_n.get("beta", np.nan)) or abs(h1_t["beta"]) > abs(h1_n["beta"]) or h1_n["p"] >= 0.05)
        ),
    }

    # H2: observation availability predicts attention but not premise refutation.
    d["y_refuted"] = (d["answer_status"] == "premise_refuted").astype(int)
    h2_a = controlled_effect(d, "tra_n_facilities_operational", "y_addressed")
    h2_r = (
        controlled_effect(d, "tra_n_facilities_operational", "y_refuted")
        if d["y_refuted"].sum() >= 10
        else {"beta": np.nan, "p": np.nan, "note": "too few refuted cases"}
    )
    out["H2"] = {
        "statement": "Observation availability predicts short-term attention, not premise refutation.",
        "attention_effect": h2_a,
        "refutation_effect": h2_r,
        "supported": bool(
            not np.isnan(h2_a.get("beta", np.nan))
            and h2_a["beta"] > 0
            and h2_a["p"] < 0.05
            and (np.isnan(h2_r.get("p", np.nan)) or h2_r["p"] >= 0.05)
        ),
    }

    # H3: explicit competing hypotheses raise P(answered | addressed).
    da = d[d["y_addressed"] == 1].copy()
    da["y_answered"] = (da["answer_status"] == "answered").astype(int)
    h3 = (
        controlled_effect(da, "fal_competing_llm", "y_answered")
        if da["y_answered"].nunique() > 1 and len(da) >= 100
        else {"beta": np.nan, "p": np.nan, "note": "insufficient addressed sample"}
    )
    out["H3"] = {
        "statement": "Explicit competing hypotheses increase the probability of being answered.",
        "effect": h3,
        "supported": bool(not np.isnan(h3.get("beta", np.nan)) and h3["beta"] > 0 and h3["p"] < 0.05),
    }

    # H4: inverted-U in novelty — quadratic term negative.
    x = d["nov_semantic_distance"].astype(float)
    xz = (x - x.mean()) / x.std()
    X = _controls_frame(d).copy()
    X["nov_z"] = xz.values
    X["nov_z2"] = (xz**2).values
    X = sm.add_constant(X.astype(float), has_constant="add")
    try:
        fit = sm.Logit(d["y_addressed"].astype(float), X).fit(disp=0, maxiter=200)
        h4 = {
            "linear_beta": float(fit.params["nov_z"]),
            "quadratic_beta": float(fit.params["nov_z2"]),
            "quadratic_p": float(fit.pvalues["nov_z2"]),
        }
    except Exception:  # noqa: BLE001
        h4 = {"linear_beta": np.nan, "quadratic_beta": np.nan, "quadratic_p": np.nan}
    out["H4"] = {
        "statement": "Very high novelty reduces short-term attention (inverted U).",
        "effect": h4,
        "supported": bool(
            not np.isnan(h4["quadratic_beta"]) and h4["quadratic_beta"] < 0 and h4["quadratic_p"] < 0.05
        ),
    }

    # H5: field popularity inflates paper counts but not resolution quality.
    dd = d[d["y_addressed"] == 1].copy()
    log_cnt = np.log1p(dd["future_paper_count"].astype(float))
    Xc = _controls_frame(dd).drop(columns=["env_pub_volume_3y"], errors="ignore")
    xv = dd["env_pub_volume_3y"].astype(float)
    Xc["env_vol_z"] = ((xv - xv.mean()) / xv.std()).values
    Xc = sm.add_constant(Xc.astype(float), has_constant="add")
    try:
        ols = sm.OLS(log_cnt.values, Xc).fit()
        h5_cnt = {"beta": float(ols.params["env_vol_z"]), "p": float(ols.pvalues["env_vol_z"])}
    except Exception:  # noqa: BLE001
        h5_cnt = {"beta": np.nan, "p": np.nan}
    dd["y_answered"] = (dd["answer_status"] == "answered").astype(int)
    h5_ans = (
        controlled_effect(dd, "env_pub_volume_3y", "y_answered")
        if dd["y_answered"].nunique() > 1
        else {"beta": np.nan, "p": np.nan}
    )
    out["H5"] = {
        "statement": "Field popularity raises future paper counts but not answer rates.",
        "count_effect": h5_cnt,
        "answer_effect": h5_ans,
        "supported": bool(
            not np.isnan(h5_cnt.get("beta", np.nan))
            and h5_cnt["beta"] > 0
            and h5_cnt["p"] < 0.05
            and (np.isnan(h5_ans.get("p", np.nan)) or h5_ans["p"] >= 0.05)
        ),
    }

    # H6: high uncertainty (tension) x high tractability interaction.
    tz = (d["ev_tension_density"] - d["ev_tension_density"].mean()) / d["ev_tension_density"].std()
    az = (
        d["tra_n_facilities_operational"] - d["tra_n_facilities_operational"].mean()
    ) / d["tra_n_facilities_operational"].std().clip(1e-9)
    X = _controls_frame(d).copy()
    X["tension_z"] = tz.values
    X["avail_z"] = az.values
    X["interaction"] = (tz * az).values
    X = sm.add_constant(X.astype(float), has_constant="add")
    try:
        fit = sm.Logit(d["y_addressed"].astype(float), X).fit(disp=0, maxiter=200)
        h6 = {
            "interaction_beta": float(fit.params["interaction"]),
            "interaction_p": float(fit.pvalues["interaction"]),
        }
    except Exception:  # noqa: BLE001
        h6 = {"interaction_beta": np.nan, "interaction_p": np.nan}
    out["H6"] = {
        "statement": "Questions with high tension AND high tractability succeed most.",
        "effect": h6,
        "supported": bool(
            not np.isnan(h6["interaction_beta"]) and h6["interaction_beta"] > 0 and h6["interaction_p"] < 0.05
        ),
    }
    return out


# ------------------------------------------------------- prediction cards


def prediction_cards(df: pd.DataFrame, n_cards: int = 6) -> list[dict]:
    """Interpretable prediction cards for test-cutoff questions (section 14)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr = df[df.cutoff_year.isin(TRAIN_CUTOFFS)]
    te = df[df.cutoff_year == 2020]
    cols = feature_columns(df)
    scaler = StandardScaler().fit(tr[cols])
    clf = LogisticRegression(max_iter=5000).fit(scaler.transform(tr[cols]), tr["y_addressed"])
    Xte = scaler.transform(te[cols])
    probs = clf.predict_proba(Xte)[:, 1]
    contribs = Xte * clf.coef_[0]

    order = np.argsort(-np.abs(probs - 0.5))
    picked = list(order[: n_cards // 2]) + list(order[-(n_cards - n_cards // 2):])
    cards = []
    for idx in picked:
        row = te.iloc[idx]
        c = contribs[idx]
        top_pos = np.argsort(-c)[:4]
        top_neg = np.argsort(c)[:4]
        cards.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "subfield": row["subfield"],
                "source": row["source"],
                "predicted_p_attention": float(probs[idx]),
                "observed_addressed": bool(row["y_addressed"]),
                "observed_status": row["answer_status"],
                "main_positive_factors": [
                    {"feature": cols[i], "contribution": float(c[i])} for i in top_pos if c[i] > 0
                ],
                "main_negative_factors": [
                    {"feature": cols[i], "contribution": float(c[i])} for i in top_neg if c[i] < 0
                ],
            }
        )
    return cards


def judge_agreement(labels_a: list[dict], labels_b: list[dict]) -> dict:
    """Agreement + Cohen's kappa between two judges on shared question_ids."""
    b_by_id = {r["question_id"]: r for r in labels_b}
    pairs = [(a, b_by_id[a["question_id"]]) for a in labels_a if a["question_id"] in b_by_id]
    if not pairs:
        return {"n": 0}

    def kappa(xs: list, ys: list) -> float:
        cats = sorted(set(xs) | set(ys))
        n = len(xs)
        po = sum(1 for x, y in zip(xs, ys) if x == y) / n
        pe = sum(
            (xs.count(c) / n) * (ys.count(c) / n) for c in cats
        )
        return (po - pe) / (1 - pe) if pe < 1 else 1.0

    a_bin = [int(a["addressed"]) for a, _ in pairs]
    b_bin = [int(b["addressed"]) for _, b in pairs]
    a_st = [a["answer_status"] for a, _ in pairs]
    b_st = [b["answer_status"] for _, b in pairs]
    per_class = {}
    for c in set(a_st):
        tp = sum(1 for x, y in zip(a_st, b_st) if x == c and y == c)
        fp = sum(1 for x, y in zip(a_st, b_st) if x != c and y == c)
        fn = sum(1 for x, y in zip(a_st, b_st) if x == c and y != c)
        per_class[c] = {
            "precision_vs_primary": tp / (tp + fp) if tp + fp else float("nan"),
            "recall_vs_primary": tp / (tp + fn) if tp + fn else float("nan"),
            "n_primary": a_st.count(c),
        }
    return {
        "n": len(pairs),
        "addressed_agreement": sum(1 for x, y in zip(a_bin, b_bin) if x == y) / len(pairs),
        "addressed_kappa": kappa(a_bin, b_bin),
        "status_agreement": sum(1 for x, y in zip(a_st, b_st) if x == y) / len(pairs),
        "status_kappa": kappa(a_st, b_st),
        "per_class": per_class,
    }
