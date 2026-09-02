"""Stage 7: robustness, label-noise ceiling, calibration, and error analysis.

Everything here runs offline from the frozen dataset. The analyses answer
the four questions a reviewer is entitled to ask of the headline results:

1. Do the source orderings depend on the engagement bar (n_direct >= 1) or
   on the retrieval threshold?           -> engagement_bar, retrieval_threshold
2. Given measured label noise (second-judge kappa = 0.21), what is the
   highest AUC any model could reach?    -> noise_ceiling
3. Is the temporal-split model calibrated, or merely discriminative?
                                          -> calibration (+ figure)
4. Where does the model fail, and does the bar-3 crossover between tension
   and control questions mean anything?  -> error_analysis, bar3_anomaly

Writes results/robustness.json and results/figures/calibration.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

from pipeline import models

QUESTIONS = Path("data/questions/questions_v1.jsonl")
LABELS = Path("data/labels/labels_v1.jsonl")
AGREEMENT = Path("data/labels/judge_agreement.json")
TABLE = Path("results/dataset_table.csv")
RESULTS = Path("results")

N_BOOT = 2000
SEED = 0


def load() -> tuple[pd.DataFrame, dict]:
    labs = {r["question_id"]: r for r in map(json.loads, LABELS.open())}
    qs = {r["question_id"]: r for r in map(json.loads, QUESTIONS.open())}
    rows = []
    for qid, lab in labs.items():
        rows.append(
            {
                "question_id": qid,
                "source": qs[qid]["source"],
                "control_subtype": qs[qid].get("control_subtype"),
                "subfield": qs[qid]["subfield"],
                "cutoff_year": qs[qid]["cutoff_year"],
                "addressed": bool(lab["addressed"]),
                "n_direct": lab["n_direct"],
                "n_adjacent": lab["n_adjacent"],
                "confidence": lab["confidence"],
                "future_paper_count": lab["future_paper_count"],
                "top_sims": lab["top_sims"],
            }
        )
    return pd.DataFrame(rows), json.load(AGREEMENT.open())


# ---------------------------------------------------------------- robustness


def engagement_bar(df: pd.DataFrame) -> dict:
    """The addressed rate under stricter direct-engagement bars.

    The released label derives `addressed` from n_direct >= 1; a reviewer may
    reasonably ask whether the source ordering is an artifact of that choice.
    """
    out = {}
    for bar in (1, 2, 3):
        hit = df.n_direct >= bar
        row = {"all": round(hit.mean(), 3)}
        for src in ("control", "evidence_tension", "future_work", "direct_llm"):
            row[src] = round(hit[df.source == src].mean(), 3)
        # the two contrasts that carry the paper's claims
        for name, a, b in (
            ("llm_vs_control", "direct_llm", "control"),
            ("tension_vs_control", "evidence_tension", "control"),
        ):
            ha, hb = hit[df.source == a], hit[df.source == b]
            table = [
                [int(ha.sum()), int((~ha).sum())],
                [int(hb.sum()), int((~hb).sum())],
            ]
            _, p = fisher_exact(table)
            row[name] = {"gap": round(ha.mean() - hb.mean(), 3), "fisher_p": round(p, 4)}
        out[f"n_direct>={bar}"] = row
    return out


def retrieval_threshold(df: pd.DataFrame) -> dict:
    """How fast the stored top-8 evidence pool starves as tau rises."""
    out = {}
    for tau in (0.18, 0.22, 0.26, 0.30):
        counts = df.top_sims.apply(lambda s: sum(1 for x in s if x >= tau))
        out[f"tau={tau}"] = {
            "mean_candidates": round(float(counts.mean()), 2),
            "zero_candidate_share": round(float((counts == 0).mean()), 3),
        }
    return out


def confidence_strata(df: pd.DataFrame) -> dict:
    out = {}
    for c in sorted(df.confidence.unique(), reverse=True):
        sel = df[df.confidence == c]
        out[str(c)] = {
            "n": int(len(sel)),
            "addressed": round(float(sel.addressed.mean()), 3),
            "mean_n_direct": round(float(sel.n_direct.mean()), 2),
        }
    return out


def bootstrap_cis(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    a = df.addressed.to_numpy(float)
    ctrl = (df.source == "control").to_numpy()
    rates, gaps = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(df), len(df))
        rates.append(a[idx].mean())
        m = ctrl[idx]
        if m.any() and (~m).any():
            gaps.append(a[idx][~m].mean() - a[idx][m].mean())

    def ci(xs):
        return [round(float(np.percentile(xs, q)), 3) for q in (2.5, 97.5)]

    return {
        "n_boot": N_BOOT,
        "addressed_rate": {"point": round(float(a.mean()), 3), "ci95": ci(rates)},
        "noncontrol_minus_control": {
            "point": round(float(a[~ctrl].mean() - a[ctrl].mean()), 3),
            "ci95": ci(gaps),
        },
    }


# ------------------------------------------------------------- noise ceiling


def _oracle_auc(e1: float, obs_rate: float) -> float | None:
    """AUC of a perfect oracle for the latent truth, scored against labels
    that flip the truth with probability e1 (class-symmetric noise)."""
    if not 0 <= e1 < 0.5:
        return None
    p = (obs_rate - e1) / (1.0 - 2.0 * e1)
    if not 0 < p < 1:
        return None
    pi1 = (1 - e1) * p / obs_rate
    pi0 = e1 * p / (1 - obs_rate)
    return pi1 * (1 - pi0) + 0.5 * (pi1 * pi0 + (1 - pi1) * (1 - pi0))


def noise_ceiling(df: pd.DataFrame, agreement: dict, observed_auc: float) -> dict:
    """What AUC could a perfect model reach against these labels?

    The two judges agree on `addressed` 63.5% of the time. Model both as
    independent noisy readers of a latent truth: agreement
    a = (1-e1)(1-e2) + e1*e2 fixes one error rate given the other, so we scan
    the split. The symmetric point (e1 = e2) is the headline; the scan is the
    honesty about not knowing which judge is better -- though the agreement
    report's one-directional leniency says the second judge is worse, which
    places the truth between the symmetric point and the upper end.
    """
    a = agreement["addressed_agreement"]
    obs = float(df.addressed.mean())
    e_sym = (1 - math.sqrt(2 * a - 1)) / 2
    scan = []
    for e1 in np.arange(0.02, 0.35, 0.02):
        e2 = (1 - a - e1) / (1 - 2 * e1)
        ceil = _oracle_auc(e1, obs)
        if ceil is None or not 0 < e2 < 0.5:
            continue
        scan.append({"e_primary": round(float(e1), 2), "e_second": round(float(e2), 3),
                     "auc_ceiling": round(float(ceil), 3)})
    sym_ceiling = _oracle_auc(e_sym, obs)
    return {
        "second_judge_agreement": a,
        "symmetric_flip_rate": round(e_sym, 3),
        "auc_ceiling_symmetric": round(float(sym_ceiling), 3),
        "observed_auc_temporal_logistic": observed_auc,
        "skill_fraction_at_symmetric_ceiling": round(
            (observed_auc - 0.5) / (sym_ceiling - 0.5), 3
        ),
        "ceiling_scan": scan,
        "note": "independent class-symmetric noise model; ceiling is for a "
                "perfect oracle of the latent truth scored on the primary labels",
    }


# ----------------------------------------------------- calibration + errors


def fit_temporal_logistic(table: pd.DataFrame):
    tr = table[table.cutoff_year.isin(models.TRAIN_CUTOFFS)]
    te = table[table.cutoff_year == models.TEST_CUTOFF]
    cols = models.feature_columns(table)
    clf = models.make_model("logistic").fit(tr[cols], tr.y_addressed)
    prob = clf.predict_proba(te[cols])[:, 1]
    return te.reset_index(drop=True), prob


def calibration(te: pd.DataFrame, prob: np.ndarray) -> dict:
    y = te.y_addressed.to_numpy(float)
    auc = roc_auc_score(y, prob)
    brier = brier_score_loss(y, prob)
    qs = np.quantile(prob, np.linspace(0, 1, 6))
    qs[0], qs[-1] = 0.0, 1.0
    bins = []
    ece = 0.0
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (prob >= lo) & (prob <= hi if hi == 1.0 else prob < hi)
        if not m.any():
            continue
        bins.append({
            "n": int(m.sum()),
            "mean_predicted": round(float(prob[m].mean()), 3),
            "observed_rate": round(float(y[m].mean()), 3),
        })
        ece += m.mean() * abs(prob[m].mean() - y[m].mean())

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="k")
    ax.plot([b["mean_predicted"] for b in bins], [b["observed_rate"] for b in bins],
            marker="o", color="#4878b0")
    ax.set_xlabel("mean predicted P(addressed)")
    ax.set_ylabel("observed addressed rate")
    ax.set_title("Calibration, temporal test (cutoff 2020)")
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "calibration.png", dpi=150)
    plt.close(fig)

    return {
        "test_auc": round(float(auc), 3),
        "brier": round(float(brier), 3),
        "ece_quintile": round(float(ece), 3),
        "bins": bins,
    }


def error_analysis(te: pd.DataFrame, prob: np.ndarray) -> dict:
    d = te.copy()
    d["prob"] = prob
    fp = d[d.y_addressed == 0].nlargest(5, "prob")
    fn = d[d.y_addressed == 1].nsmallest(5, "prob")

    def rows(sub):
        return [
            {
                "question_id": r.question_id,
                "source": r.source,
                "subfield": r.subfield,
                "predicted": round(float(r.prob), 3),
                "review_overlap": float(r.nov_review_overlap),
                "n_similar": float(r.ev_n_similar),
                "question": r.question[:140],
            }
            for r in sub.itertuples()
        ]

    fp_all = d[(d.y_addressed == 0) & (d.prob >= 0.5)]
    rest0 = d[(d.y_addressed == 0) & (d.prob < 0.5)]
    return {
        "top_false_positives": rows(fp),
        "top_false_negatives": rows(fn),
        "false_positive_profile": {
            "n": int(len(fp_all)),
            "mean_review_overlap": round(float(fp_all.nov_review_overlap.mean()), 3),
            "mean_review_overlap_true_negatives": round(float(rest0.nov_review_overlap.mean()), 3),
            "share_direct_llm": round(float((fp_all.source == "direct_llm").mean()), 3),
        },
    }


# --------------------------------------------------------------- bar-3 story


def bar3_anomaly(df: pd.DataFrame, table: pd.DataFrame) -> dict:
    """At n_direct >= 3 tension questions (0.115) sit below controls (0.140).

    Before narrating, test it; then characterise how thin each source's
    engagement is, and which control subtypes survive strict bars.
    """
    t = df[df.source == "evidence_tension"]
    c = df[df.source == "control"]
    ge3 = lambda s: (s.n_direct >= 3)  # noqa: E731
    _, p = fisher_exact([
        [int(ge3(t).sum()), int((~ge3(t)).sum())],
        [int(ge3(c).sum()), int((~ge3(c)).sum())],
    ])

    engaged = df[df.n_direct >= 1]
    thin = {
        src: {
            "mean_n_direct_given_engaged": round(float(engaged[engaged.source == src].n_direct.mean()), 2),
            "share_thin_1_or_2": round(float((engaged[engaged.source == src].n_direct <= 2).mean()), 3),
        }
        for src in ("evidence_tension", "control", "future_work", "direct_llm")
    }

    surv = df[(df.source == "control") & (df.n_direct >= 3)]
    subtype = surv.control_subtype.value_counts().to_dict()

    m = table.drop(columns=["n_direct"], errors="ignore").merge(
        df[["question_id", "n_direct"]], on="question_id"
    )
    rho_ent, p_ent = spearmanr(m.txt_n_entities, m.n_direct)
    rho_spec, p_spec = spearmanr(m.txt_specificity, m.n_direct)

    return {
        "tension_vs_control_at_bar3": {
            "tension": round(float(ge3(t).mean()), 3),
            "control": round(float(ge3(c).mean()), 3),
            "fisher_p": round(float(p), 3),
        },
        "engagement_thinness": thin,
        "control_subtypes_surviving_bar3": subtype,
        "spearman_entities_vs_n_direct": {"rho": round(float(rho_ent), 3), "p": round(float(p_ent), 5)},
        "spearman_specificity_vs_n_direct": {"rho": round(float(rho_spec), 3), "p": round(float(p_spec), 5)},
    }


def main() -> None:
    df, agreement = load()
    table = pd.read_csv(TABLE)

    te, prob = fit_temporal_logistic(table)
    cal = calibration(te, prob)

    out = {
        "engagement_bar": engagement_bar(df),
        "retrieval_threshold": retrieval_threshold(df),
        "confidence_strata": confidence_strata(df),
        "bootstrap": bootstrap_cis(df),
        "noise_ceiling": noise_ceiling(df, agreement, cal["test_auc"]),
        "calibration": cal,
        "error_analysis": error_analysis(te, prob),
        "bar3_anomaly": bar3_anomaly(df, table),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("noise_ceiling",)}, indent=2)[:1200])
    print("calibration:", cal["test_auc"], "brier", cal["brier"], "ece", cal["ece_quintile"])
    print("bar3:", out["bar3_anomaly"]["tension_vs_control_at_bar3"])


if __name__ == "__main__":
    main()
