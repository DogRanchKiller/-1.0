"""真实成绩统计：用能手算的小例子防止混版、错误分母和缺失值冒充零。"""
import json

import pandas as pd
import pytest

from src.evaluate.live_review_metrics import (
    build_live_metrics, calibration_by_group, matched_comparisons, score_summary,
)
from src.review.reconcile import score_metrics


BOOKS = ["平博 Pinnacle", "多家平均"]


def prediction(mid, version="v1", source="模型", **extra):
    p = [0.5, 0.3, 0.2]
    return {"match_id": mid, "model": "gb_l7", "model_version": version,
            "kind": "T-24", "source": source, "archive_status": "on_time",
            "competition": "英超", "confidence": 2, "home_goals": 1, "away_goals": 0,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            **score_metrics(p, None, 1, 0), **extra}


def benchmark(mid, book=BOOKS[0], market="1x2", **extra):
    p = [0.4, 0.4, 0.2]
    return {**prediction(mid), "model": None, "model_version": None,
            "source": "收盘赔率", "kind": f"收盘（{book}）", "benchmark_source": "football-data",
            "bookmaker": book, "market": market, "timing": "close", "line": None,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            **score_metrics(p, None, 1, 0), **extra}


def build(rows, **kwargs):
    return build_live_metrics(pd.DataFrame(rows), benchmark_books=BOOKS,
                             n_bins=5, min_matches=300, **kwargs)


def test_separates_model_version_source_and_late_archive():
    out = build([prediction(1), prediction(1, version="v2", brier_1x2=0.7),
                 prediction(1, source="融合", brier_1x2=0.2),
                 prediction(2, archive_status="late", brier_1x2=0.9)])
    assert len(out["groups"]) == 4
    assert sorted(g["metrics"]["brier_1x2"]["mean"] for g in out["groups"]) == pytest.approx([0.2, 0.38, 0.7, 0.9])
    assert all(g["n_matches"] == 1 for g in out["groups"])
    assert all("样本太少" in g["note"] for g in out["groups"])


def test_two_benchmarks_use_each_versions_own_match_intersection():
    rows = [prediction(1), prediction(2, logloss_1x2=2.0), prediction(2, version="v2"),
            benchmark(1), benchmark(2), benchmark(2, BOOKS[1]), benchmark(999, BOOKS[1])]
    out = build(rows)
    comps = {(r["group"]["model_version"], r["benchmark"]["bookmaker"]): r for r in out["comparisons"]
             if r["benchmark"]["market"] == "1x2"}
    assert len(comps) == 4
    assert comps[("v1", BOOKS[0])]["n_matches"] == 2
    assert comps[("v1", BOOKS[1])]["n_matches"] == 1
    assert comps[("v2", BOOKS[0])]["n_matches"] == 1
    assert comps[("v2", BOOKS[1])]["n_matches"] == 1
    m = comps[("v1", BOOKS[1])]["metrics"]["logloss_1x2"]
    assert m["model_mean"] == 2.0
    assert m["mean_delta"] == pytest.approx(2.0 - m["benchmark_mean"])
    assert comps[("v1", BOOKS[1])]["n_benchmark"] == 1  # 无关比赛 999 不进覆盖分母
    assert out["counts"]["benchmark_rows"] == 3  # 不能因模型有两版扩增标杆
    assert out["counts"]["orphan_benchmark_rows"] == 1


def test_missing_benchmark_is_unavailable_and_not_zero():
    out = build([prediction(1)])
    assert len(out["comparisons"]) == 4  # 两家公司、两个玩法分别说明没有证据
    for comp in out["comparisons"]:
        assert comp["status"] == "unavailable"
        assert comp["n_matches"] == 0
        assert comp["missing_matches"] == 1
        assert comp["note"] == "无匹配证据"
        assert all(v["mean_delta"] is None for v in comp["metrics"].values())


def test_ou_and_1x2_benchmarks_have_independent_denominators():
    rows = [prediction(1, logloss_over25=0.4), prediction(2, logloss_over25=None),
            benchmark(1), benchmark(2), benchmark(1, market="ou", line=2.5,
                      logloss_1x2=None, brier_1x2=None, rps_1x2=None,
                      p_home=None, p_draw=None, p_away=None, logloss_over25=0.6)]
    out = build(rows)
    ou = next(c for c in out["comparisons"] if c["benchmark"]["bookmaker"] == BOOKS[0] and c["benchmark"]["market"] == "ou")
    assert ou["n_matches"] == 1
    assert set(ou["metrics"]) == {"logloss_over25"}
    assert ou["metrics"]["logloss_over25"]["mean_delta"] == pytest.approx(-0.2)
    assert out["groups"][0]["metrics"]["logloss_over25"]["missing_matches"] == 1


def test_confidence_brier_keeps_model_and_fusion_apart_and_unknown_visible():
    out = build([prediction(1), prediction(2, confidence=None),
                 prediction(1, source="融合", brier_1x2=0.1)])
    assert len(out["confidence"]) == 3
    unknown = next(g for g in out["confidence"] if g["group"]["confidence"] == "未知")
    assert unknown["n_matches"] == 1
    assert unknown["metrics"]["brier_1x2"]["mean"] == pytest.approx(0.38)
    assert "calibration" in unknown


def test_calibration_counts_fixtures_for_each_outcome_and_keeps_empty_bins():
    out = build([prediction(1), prediction(2)])
    cal = out["calibration"][0]
    assert cal["n_matches"] == 2
    assert len(cal["outcomes"]) == 3
    for outcome in cal["outcomes"]:
        assert len(outcome["bins"]) == 5
        assert sum(b["n"] for b in outcome["bins"]) == 2
        assert all(b["mean_pred"] is None and b["observed"] is None
                   for b in outcome["bins"] if b["n"] == 0)
    home = cal["outcomes"][0]
    assert home["ece"] == pytest.approx(0.5)
    json.dumps(out, allow_nan=False)


def test_bad_probability_is_counted_as_missing_in_calibration():
    out = build([prediction(1), prediction(2, p_home=None)])
    cal = out["calibration"][0]
    assert cal["n_matches"] == 1
    assert cal["missing_matches"] == 1


def test_missing_tag_is_unknown_not_false_and_after_euro_survives():
    out = build([prediction(1), prediction(2), prediction(3)], labels={
        1: {"德比": "北伦敦德比", "争冠战": False, "保级战": True,
            "home_after_euro": True, "away_after_euro": False},
        2: {"算不出": "没有赛程"},
        3: {"德比": None, "争冠战": False, "保级战": False,
            "home_after_euro": False, "away_after_euro": False},
    })
    derby = [r for r in out["tags"] if r["group"]["tag"] == "德比"]
    assert {r["group"]["tag_value"] for r in derby} == {True, False, "未知"}
    home = [r for r in out["tags"] if r["group"]["tag"] == "home_after_euro"]
    assert len(home) == 3
    assert sum(r["n_matches"] for r in home) == 3


@pytest.mark.parametrize("which", ["prediction", "benchmark"])
def test_duplicate_evidence_fails_explicitly_instead_of_taking_first(which):
    row = prediction(1) if which == "prediction" else benchmark(1)
    with pytest.raises(ValueError, match="重复"):
        build([prediction(2), row, row])


def test_conflicting_result_between_model_and_benchmark_fails():
    with pytest.raises(ValueError, match="比分.*不一致"):
        build([prediction(1), benchmark(1, home_goals=0, away_goals=1)])


def test_legacy_missing_identity_stays_unknown_without_inventing_version():
    row = prediction(1)
    for key in ("model", "model_version", "archive_status", "confidence"):
        row.pop(key)
    out = build([row])
    assert out["groups"][0]["group"]["model_version"] == "未知"
    assert out["groups"][0]["group"]["archive_status"] == "未知"


def test_empty_input_is_json_serializable_and_has_no_fake_percentages():
    out = build([])
    assert out["groups"] == out["comparisons"] == out["calibration"] == []
    assert out["counts"]["reviewed_matches"] == 0
    json.dumps(out, allow_nan=False)


def test_public_helpers_keep_same_contract():
    df = pd.DataFrame([prediction(1), benchmark(1)])
    assert score_summary(df, min_matches=300)[0]["n_matches"] == 1
    assert calibration_by_group(df, n_bins=5)[0]["n_matches"] == 1
    assert matched_comparisons(df, benchmark_books=BOOKS, min_matches=300)[0]["n_matches"] == 1


def test_sample_warning_counts_valid_scores_instead_of_all_review_rows():
    empty = {k: None for k in ("logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score",
                              "logloss_over25", "logloss_btts")}
    rows = [prediction(mid, **empty) for mid in range(300)]
    rows[0]["logloss_1x2"] = 0.7
    group = build(rows)["groups"][0]
    assert group["n_matches"] == 300
    assert group["metrics"]["logloss_1x2"]["n_matches"] == 1
    assert group["note"] == "样本太少，不能下结论"
    assert group["metrics"]["logloss_1x2"]["note"] == "样本太少，不能下结论"
    assert group["metrics"]["brier_1x2"]["note"] == "无匹配证据"


def test_nonfinite_confidence_is_unknown_and_remains_strict_json():
    out = build([prediction(1, confidence=float("inf")), prediction(2, confidence=-float("inf"))])
    assert len(out["confidence"]) == 1
    assert out["confidence"][0]["group"]["confidence"] == "未知"
    json.dumps(out, allow_nan=False)


def test_explicit_unknown_derby_is_not_inferred_true_or_false():
    out = build([prediction(1), prediction(2), prediction(3)], labels={
        1: {"德比": "未知"}, 2: {"德比": float("nan")}, 3: {"德比": None},
    })
    derby = {g["group"]["tag_value"]: g["n_matches"] for g in out["tags"] if g["group"]["tag"] == "德比"}
    assert derby == {"未知": 2, False: 1}
