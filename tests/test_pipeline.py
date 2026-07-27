"""Offline unit tests for the Paper 3 pipeline (no network, no API keys)."""

import pytest

pytest.importorskip("sklearn", reason="requires the scientific stack")

from pipeline import analysis, facilities, features, generate, mine, subfields  # noqa: E402


def _doc(i, title, abstract, published="2010-05-01", cats=("astro-ph.EP",)):
    return {
        "id": f"1005.{i:05d}",
        "title": title,
        "abstract": abstract,
        "published": published,
        "primary_category": cats[0],
        "categories": list(cats),
        "year": int(published[:4]),
    }


def test_classify_exoplanet_atmospheres():
    scores = subfields.classify(
        "Water in the transmission spectrum of a hot Jupiter",
        "We present the transmission spectrum and atmospheric retrieval of WASP-12b, "
        "finding a low water abundance in the exoplanet atmosphere.",
        ["astro-ph.EP"],
    )
    assert "exoplanet_atmospheres" in scores


def test_classify_rejects_unrelated():
    scores = subfields.classify(
        "A new bound on neutrino masses",
        "We derive a laboratory bound on neutrino mass hierarchies.",
        ["hep-ph"],
    )
    assert scores == {}


def test_mine_tension_and_future_work():
    docs = [
        _doc(
            1,
            "Discrepant winds",
            "We measure a wind speed of 8 km/s. This value is inconsistent with previous "
            "near-infrared measurements that suggested much slower winds in this planet.",
        ),
        _doc(
            2,
            "Open problems",
            "The vertical mixing strength remains poorly constrained and further "
            "observations are needed to establish the dominant chemistry pathways.",
        ),
    ]
    tension = mine.mine_tension(docs)
    fw = mine.mine_future_work(docs)
    assert len(tension) == 1 and tension[0]["arxiv_id"] == "1005.00001"
    assert len(fw) >= 1 and fw[0]["arxiv_id"] == "1005.00002"


def test_tension_clusters_require_two_papers():
    sents = [
        {"arxiv_id": "a", "published": "2010-01-01", "title": "t", "sentence": "water abundance is inconsistent with retrieval models of hot jupiter atmospheres", "n_markers": 1},
        {"arxiv_id": "a", "published": "2010-02-01", "title": "t", "sentence": "water abundance disagrees with hot jupiter atmosphere retrievals again", "n_markers": 1},
    ]
    assert generate.build_tension_clusters(sents) == []
    sents[1]["arxiv_id"] = "b"
    clusters = generate.build_tension_clusters(sents)
    assert len(clusters) == 1 and len(clusters[0]) == 2


def test_text_features_specificity():
    vague = features.text_features("What is the nature of galaxies?")
    specific = features.text_features(
        "Is the low H2O abundance of 1e-5 in WASP-12 b real or a retrieval artifact?"
    )
    assert specific["txt_specificity"] > vague["txt_specificity"]
    assert specific["txt_has_alternatives"] == 1
    assert specific["txt_is_yesno"] == 1


def test_facilities_lexicon():
    found = facilities.find_facilities("We use JWST and ALMA observations of the disk.")
    assert "jwst" in found and "alma" in found
    assert facilities.operational_at("alma", 2012)
    assert not facilities.operational_at("jwst", 2020)
    gap = facilities.next_launch_gap_years(("jwst", "alma"), 2018)
    assert gap == 4


def test_cell_context_retrieval_and_novelty():
    docs = [
        _doc(1, "Water in hot Jupiters", "Transmission spectra show water absorption in hot jupiter atmospheres."),
        _doc(2, "Disk rings", "ALMA imaging reveals dust rings in protoplanetary disks."),
        _doc(3, "A review of exoplanet atmospheres", "We review atmospheric observations of transiting planets."),
    ]
    ctx = features.CellContext(docs, 2012)
    q = {"question": "Is water absorption in hot Jupiter atmospheres ubiquitous?", "source_material": {}}
    ev = features.evidence_features(q, ctx)
    nov = features.novelty_features(q, ctx)
    assert ev["ev_top1_sim"] > 0.1
    assert 0 <= nov["nov_semantic_distance"] <= 1
    assert nov["nov_review_overlap"] >= 0


def test_dedupe_drops_near_duplicates():
    qs = [
        {"question": "Is the water abundance in WASP-12 b subsolar or an artifact?"},
        {"question": "Is the water abundance in WASP-12 b subsolar, or is it just an artifact?"},
        {"question": "What drives quenching in massive galaxies at z=2?"},
    ]
    kept = generate._dedupe(qs)
    assert len(kept) == 2


def test_judge_agreement_kappa():
    a = [
        {"question_id": "q1", "addressed": True, "answer_status": "answered"},
        {"question_id": "q2", "addressed": False, "answer_status": "not_addressed"},
        {"question_id": "q3", "addressed": True, "answer_status": "partially_addressed"},
    ]
    b = [
        {"question_id": "q1", "addressed": True, "answer_status": "answered"},
        {"question_id": "q2", "addressed": False, "answer_status": "not_addressed"},
        {"question_id": "q3", "addressed": True, "answer_status": "answered"},
    ]
    rep = analysis.judge_agreement(a, b)
    assert rep["n"] == 3
    assert rep["addressed_agreement"] == 1.0
    assert 0 < rep["status_agreement"] < 1


def test_env_features_growth():
    stats = {
        "total_by_year": {str(y): 1000 for y in range(2005, 2013)},
        "subfields": {
            "exoplanet_atmospheres": {
                "papers_by_year": {"2007": 10, "2008": 12, "2009": 15, "2010": 30, "2011": 45, "2012": 60},
                "reviews_by_year": {"2011": 2, "2012": 1},
            }
        },
    }
    from pipeline import corpus

    env = corpus.env_features(stats, "exoplanet_atmospheres", 2012)
    assert env["env_pub_volume_3y"] == 135
    assert env["env_growth_ratio"] > 1
    assert env["env_review_count_3y"] == 3
