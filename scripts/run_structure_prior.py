"""Stage 6: the learned prior over evidence-graph structures.

Answers the question Paper 2's outlook posed: which structural patterns of
pre-cutoff evidence most often lead to questions the future answers,
advances, or refutes? Runs entirely offline from the frozen dataset.

Writes:
- data/features/structure_features_v1.jsonl  (frozen structural features)
- results/structure_prior.json               (rates, effects, prior model)
- results/structure_prior.csv                (controlled-effect table)
- results/figures/structure_prior.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from pipeline import analysis, models
from pipeline.structure import STRUCTURE_COLUMNS, TYPE_COLUMNS, structure_features

QUESTIONS = Path("data/questions/questions_v1.jsonl")
FEATURES = Path("data/features/features_v1.jsonl")
LABELS = Path("data/labels/labels_v1.jsonl")
STRUCT_OUT = Path("data/features/structure_features_v1.jsonl")
RESULTS = Path("results")

COLUMNS = STRUCTURE_COLUMNS + ["str_recurrent"]

ADVANCED = ("answered", "partially_addressed")
RESOLVED = ("answered", "premise_refuted")


def _recurrence(qs: list[dict]) -> dict[str, int]:
    """str_recurrent: the same tension (>=1 shared source arXiv id, same
    subfield) was already detectable at an earlier cutoff. Cutoff-time
    computable: earlier past-windows are subsets of the current one."""
    ids = {}
    for q in qs:
        cluster = (q.get("source_material") or {}).get("cluster")
        if cluster:
            ids[q["question_id"]] = (
                q["subfield"],
                q["cutoff_year"],
                {c.get("arxiv_id") for c in cluster},
            )
    out = {}
    for qid, (sub, cy, aids) in ids.items():
        out[qid] = int(
            any(
                sub == s2 and c2 < cy and aids & a2
                for q2, (s2, c2, a2) in ids.items()
                if q2 != qid
            )
        )
    return out


def load_table() -> pd.DataFrame:
    qs = [json.loads(l) for l in QUESTIONS.open()]
    feats = {r["question_id"]: r for r in (json.loads(l) for l in FEATURES.open())}
    labels = {r["question_id"]: r for r in (json.loads(l) for l in LABELS.open())}
    recurrent = _recurrence(qs)

    rows, frozen = [], []
    for q in qs:
        s = structure_features(q)
        if s is None:
            continue
        s["str_recurrent"] = recurrent[q["question_id"]]
        frozen.append(s)
        rows.append(
            {
                **s,
                "question": q["question"],
                "subfield": q["subfield"],
                "cutoff_year": q["cutoff_year"],
                "source": q["source"],
                "env_pub_volume_3y": feats[q["question_id"]]["env_pub_volume_3y"],
                "env_growth_ratio": feats[q["question_id"]]["env_growth_ratio"],
                **{
                    k: labels[q["question_id"]][k]
                    for k in (
                        "addressed",
                        "answer_status",
                        "future_paper_count",
                        "lead_time_months",
                    )
                },
            }
        )
    STRUCT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with STRUCT_OUT.open("w") as f:
        for s in frozen:
            f.write(json.dumps(s) + "\n")

    df = pd.DataFrame(rows)
    df["y_addressed"] = df["addressed"].astype(int)
    df["y_advanced"] = df["answer_status"].isin(ADVANCED).astype(int)
    df["y_resolved"] = df["answer_status"].isin(RESOLVED).astype(int)
    return df


def rate_block(df: pd.DataFrame, mask: pd.Series) -> dict:
    d = df[mask]
    return {
        "n": int(len(d)),
        "addressed": round(float(d["y_addressed"].mean()), 3),
        "advanced": round(float(d["y_advanced"].mean()), 3),
        "resolved_n": int(d["y_resolved"].sum()),
        "refuted_n": int((d["answer_status"] == "premise_refuted").sum()),
        "mean_future_papers": round(float(d["future_paper_count"].mean()), 2),
    }


def descriptive_rates(df: pd.DataFrame) -> dict:
    out = {"all_tension": rate_block(df, df.index == df.index)}
    for c in TYPE_COLUMNS:
        out[c] = rate_block(df, df[c] == 1)
        out[c + "_absent"] = rate_block(df, df[c] == 0)
    out["str_quantified"] = rate_block(df, df["str_quantified"] == 1)
    out["str_quantified_absent"] = rate_block(df, df["str_quantified"] == 0)
    out["cross_facility"] = rate_block(df, df["str_cross_facility"] == 1)
    out["single_facility_or_none"] = rate_block(df, df["str_cross_facility"] == 0)
    out["cluster_2"] = rate_block(df, df["str_n_claims"] == 2)
    out["cluster_3"] = rate_block(df, df["str_n_claims"] == 3)
    out["cluster_4"] = rate_block(df, df["str_n_claims"] == 4)
    med = df["str_year_spread"].median()
    out["year_spread_wide"] = rate_block(df, df["str_year_spread"] > med)
    out["year_spread_narrow"] = rate_block(df, df["str_year_spread"] <= med)
    out["recurrent_tension"] = rate_block(df, df["str_recurrent"] == 1)
    out["first_seen_tension"] = rate_block(df, df["str_recurrent"] == 0)
    out["fresh_evidence"] = rate_block(df, df["str_newest_age"] <= 1)
    out["stale_evidence"] = rate_block(df, df["str_newest_age"] > 1)
    return out


def controlled_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in COLUMNS:
        eff_a = analysis.controlled_effect(df, feat, "y_addressed")
        # attention volume: OLS of log1p(future papers) on the same controls
        x = df[feat].astype(float)
        if x.std() > 0:
            X = analysis._controls_frame(df).copy()
            X[feat] = ((x - x.mean()) / x.std()).values
            X = sm.add_constant(X.astype(float), has_constant="add")
            try:
                ols = sm.OLS(np.log1p(df["future_paper_count"].astype(float)), X).fit()
                vol_beta, vol_p = float(ols.params[feat]), float(ols.pvalues[feat])
            except Exception:  # noqa: BLE001
                vol_beta, vol_p = np.nan, np.nan
        else:
            vol_beta, vol_p = np.nan, np.nan
        rows.append(
            {
                "feature": feat,
                "beta_addressed": eff_a.get("beta"),
                "p_addressed": eff_a.get("p"),
                "beta_log_papers": vol_beta,
                "p_log_papers": vol_p,
            }
        )
    return pd.DataFrame(rows)


def prior_model(df: pd.DataFrame) -> dict:
    """Structure-only logistic prior, evaluated out-of-time (train 2012-2016
    cutoffs, test 2020) — the learnable ranking prior Paper 2's outlook asks
    for, restricted to features available before any question is phrased."""
    tr = df[df.cutoff_year.isin(models.TRAIN_CUTOFFS + (models.VAL_CUTOFF,))]
    te = df[df.cutoff_year == models.TEST_CUTOFF]
    scaler = StandardScaler().fit(tr[COLUMNS])
    clf = LogisticRegression(max_iter=5000).fit(
        scaler.transform(tr[COLUMNS]), tr["y_addressed"]
    )
    p = clf.predict_proba(scaler.transform(te[COLUMNS]))[:, 1]
    auc = float(roc_auc_score(te["y_addressed"], p))
    coefs = sorted(
        zip(COLUMNS, clf.coef_[0].round(3).tolist()),
        key=lambda t: -abs(t[1]),
    )
    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_base_rate": round(float(te["y_addressed"].mean()), 3),
        "auc_structure_only": round(auc, 3),
        "coefficients": [{"feature": f, "coef": c} for f, c in coefs],
    }


def resolved_profiles(df: pd.DataFrame) -> list[dict]:
    cols = ["question_id", "answer_status", "subfield", "cutoff_year"] + COLUMNS
    return df[df["y_resolved"] == 1][cols].to_dict(orient="records")


def figure(df: pd.DataFrame, table: pd.DataFrame) -> None:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = {
        "str_type_stance": "stance opposition\n(detection vs. limit)",
        "str_type_contradiction": "explicit contradiction",
        "str_type_method": "methodological challenge",
        "str_type_unexplained": "unexplained result",
        "str_quantified": "quantified tension\n(N-sigma / factor)",
        "str_cross_facility": "cross-instrument evidence",
    }
    keys = list(labels)
    present = [df[df[k] == 1]["y_addressed"].mean() for k in keys]
    absent = [df[df[k] == 0]["y_addressed"].mean() for k in keys]
    ypos = np.arange(len(keys))
    axes[0].barh(ypos + 0.18, present, height=0.36, color="#4878b0", label="structure present")
    axes[0].barh(ypos - 0.18, absent, height=0.36, color="#b8b8b8", label="absent")
    axes[0].set_yticks(ypos, [labels[k] for k in keys], fontsize=8)
    axes[0].axvline(df["y_addressed"].mean(), color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("addressed rate (tension questions)")
    axes[0].set_title("Future attention by evidence structure")
    axes[0].legend(fontsize=8, loc="lower right")

    t = table.dropna(subset=["beta_addressed"]).sort_values("beta_addressed")
    colors = ["#4878b0" if p < 0.05 else "#b8b8b8" for p in t["p_addressed"]]
    axes[1].barh(t["feature"], t["beta_addressed"], color=colors)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].set_xlabel("controlled log-odds effect on addressed")
    axes[1].set_title("Structure effects, controlled (blue: p<0.05)")

    fig.tight_layout()
    fig.savefig(fig_dir / "structure_prior.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_table()
    table = controlled_table(df)
    out = {
        "n_tension_questions": int(len(df)),
        "descriptive_rates": descriptive_rates(df),
        "controlled_effects": table.round(4).to_dict(orient="records"),
        "prior_model": prior_model(df),
        "resolved_question_profiles": resolved_profiles(df),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "structure_prior.json").write_text(json.dumps(out, indent=2))
    table.round(4).to_csv(RESULTS / "structure_prior.csv", index=False)
    figure(df, table)
    print(json.dumps({k: out[k] for k in ("n_tension_questions", "prior_model")}, indent=2))
    print("rates:", json.dumps(out["descriptive_rates"]["all_tension"]))


if __name__ == "__main__":
    main()
