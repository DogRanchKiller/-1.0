"""逐场解释的网页消费者：只用很小的人工载荷，不拟合模型、不碰真实数据库。"""
import copy
import html
import math
import re
from datetime import datetime, timezone

import pytest

from src.common.config import load_settings
from src.web.explanation import feature_label, render_explanation, validate_explanation
from src.web.preview_page import render as render_preview
from src.web.site.common import TeamFiles
from src.web.site.data import SiteData
from src.web.site.match import explanations_html, render as render_match


NOW = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
IDENTITY = {
    "match_id": 10, "model": "gb_l7", "model_version": "gb_l7-example", "competition": "E0",
    "season": 2026, "home_team": "Liverpool", "away_team": "Arsenal", "cutoff_utc": "2026-11-06 12:00:00",
}


def explanation():
    """两个输出、四个特征；三种加和都成立，用来检查消费者而不是算法。"""
    names = ["rest_days_own", "rest_days_opp", "form5_gf_own", "form5_ga_opp"]
    terms = []
    for side, phis, slope in (("home", [.08, .04, -.03, .01], .3), ("away", [-.04, -.02, .02, -.01], -.2)):
        delta = sum(phis)
        factor = math.expm1(delta) / delta
        for name, phi, value in zip(names, phis, (7.0, 3.0, None, 2.0)):
            terms.append({"side": side, "feature": name, "value": value, "raw_contribution": phi,
                          "goal_contribution": phi * factor, "home_win_contribution": phi * slope})
    return {
        "schema": "gbdt-tree-shap-v1", "status": "available", "reason": None,
        "method": "tree_shap_raw_log_lambda", "conversion": "straight_log_path_v1",
        "model_fingerprint": "a" * 64, "library_version": "4.6.0", "model_trained_at_utc": None,
        "identity": dict(IDENTITY), "context": {"rho": -.1, "max_goals": 10},
        "base": {"home_goals": 1.0, "away_goals": 1.0, "home_win": .4},
        "prediction": {"home_goals": math.exp(.1), "away_goals": math.exp(-.05), "home_win": .44},
        "raw_base": {"home": 0.0, "away": 0.0}, "raw_prediction": {"home": .1, "away": -.05},
        "feature_names": names, "terms": terms,
        "checks": {name: 0.0 for name in ("raw_max_abs_error", "goal_max_abs_error", "home_win_abs_error",
                                         "integration_max_abs_error", "matrix_max_abs_error")},
    }


def archived_row(*, pid=1, kind="T-24", payload=True, version=None, cutoff=None):
    row = {**IDENTITY, "prediction_id": pid, "kind": kind, "archive_status": "on_time",
           "created_at_utc": IDENTITY["cutoff_utc"], "p_home": .44, "p_draw": .3, "p_away": .26,
           "detail": {"融合概率": [.6, .2, .2], "融合": {"状态": "已融合", "w": 1.0, "信心等级": 3,
                                                        "不确定因素": ["首发尚未接入"]}}}
    if version:
        row["model_version"] = version
    if cutoff:
        row["cutoff_utc"] = cutoff
    if payload:
        row["detail"]["模型解释"] = explanation()
        row["detail"]["模型解释"]["identity"] = {key: row[key] for key in IDENTITY}
    return row


def render_payload(payload=None, row=None, **kw):
    row = row or archived_row()
    return render_explanation(explanation() if payload is None else payload, row, uid="match-10-v1",
                              fusion=row["detail"]["融合"], **kw)


def test_available_explanation_keeps_three_units_and_final_fusion_separate():
    page = render_payload()
    assert "SHAP贡献经路径换算" in page
    assert "主胜概率贡献（百分点）" in page and "预期进球贡献（球）" in page
    assert "主队预期进球" in page and "客队预期进球" in page
    assert "40.00%" in page and "44.00%" in page
    assert "+2.40 个百分点" in page and "+1.20 个百分点" in page and "-0.90 个百分点" in page
    assert "其他因素（5 项）" in page and "+1.30 个百分点" in page
    assert "+0.084 球" in page and "原始贡献（对数进球）" in page
    assert "完整条目（8 项）" in page and "输入缺失" in page
    assert "模型未参与该融合概率" in page
    assert "不代表因果" in page and "真实平均进球" in page
    assert "<script" not in page.lower() and "href='#" not in page and 'href="#' not in page
    assert "推荐" not in page


@pytest.mark.parametrize("side,feature,label", [
    ("home", "rest_days_own", "主队休息天数"), ("away", "rest_days_own", "客队休息天数"),
    ("home", "rest_days_opp", "客队休息天数"), ("away", "rest_days_opp", "主队休息天数"),
    ("home", "form5_ga_opp", "客队近5场场均失球"), ("away", "form5_gf_own", "客队近5场场均进球"),
    ("away", "dyn_loglam_against", "主队动态预期进球（对数）"),
])
def test_feature_labels_resolve_own_and_opponent_from_output_side(side, feature, label):
    assert feature_label(side, feature) == label


@pytest.mark.parametrize("path,value", [
    (("base", "home_win"), float("nan")), (("prediction", "away_goals"), float("inf")),
    (("terms", 0, "value"), True), (("terms", 0, "goal_contribution"), ".1"),
    (("terms", 0, "raw_contribution"), .18), (("terms", 0, "goal_contribution"), .4),
    (("terms", 0, "home_win_contribution"), .3), (("terms", 0, "side"), "host"),
    (("terms", 0, "feature"), "missing_feature"), (("raw_prediction", "home"), 710.0),
    (("base", "home_goals"), 2.0), (("feature_names", 1), "rest_days_own"),
    (("checks", "integration_max_abs_error"), .01), (("context", "max_goals"), True),
    (("model_fingerprint",), "not-a-fingerprint"), (("method",), "gain_share"),
    (("conversion",), "unknown"), (("library_version",), None),
])
def test_invalid_explanation_is_isolated_even_when_reported_checks_claim_zero(path, value):
    payload = explanation()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    valid, reason = validate_explanation(payload, archived_row())
    assert valid is None and reason
    page = render_payload(payload)
    assert "模型解释数据校验未通过" in page and "完整条目" not in page


@pytest.mark.parametrize("key", list(IDENTITY))
def test_every_saved_identity_field_must_match_its_prediction(key):
    payload = explanation()
    payload["identity"][key] = "another"
    valid, reason = validate_explanation(payload, archived_row())
    assert valid is None and "身份" in reason
    assert "完整条目" not in render_payload(payload)


def test_missing_required_fields_and_unknown_schema_fail_closed():
    for key in ("identity", "base", "terms", "raw_prediction", "checks", "context"):
        payload = explanation()
        del payload[key]
        valid, reason = validate_explanation(payload, archived_row())
        assert valid is None and reason
    payload = explanation()
    payload["schema"] = "future-v2"
    assert "不支持的模型解释格式" in render_payload(payload)
    for payload in ([], "bad", 1, False):
        valid, reason = validate_explanation(payload, archived_row())
        assert valid is None and reason


@pytest.mark.parametrize("change", ["missing_term", "duplicate_pair", "missing_feature", "missing_side"])
def test_every_feature_has_exactly_one_home_and_one_away_term(change):
    payload = explanation()
    if change == "missing_term":
        payload["terms"].pop()
    elif change == "duplicate_pair":
        payload["terms"][-1] = copy.deepcopy(payload["terms"][-2])
    elif change == "missing_feature":
        payload["feature_names"].pop()
    else:
        del payload["terms"][0]["side"]
    assert validate_explanation(payload, archived_row())[0] is None


def test_prediction_probability_uses_only_the_archived_rounding_tolerance():
    row = archived_row()
    row["p_home"] = .44005
    assert validate_explanation(explanation(), row)[0] is not None
    row["p_home"] = .440052
    assert validate_explanation(explanation(), row)[0] is None
    row["p_home"] = float("nan")
    assert validate_explanation(explanation(), row)[0] is None


@pytest.mark.parametrize("reason,text", [
    ("disabled", "未启用"), ("unsupported_model", "不支持"), ("unsupported_target", "不支持"),
    ("no_prediction", "没有模型预测"), ("explanation_failed", "生成失败"), ("validation_failed", "校验未通过"),
])
def test_unavailable_reasons_do_not_turn_into_zero_contribution(reason, text):
    payload = {"schema": "gbdt-tree-shap-v1", "status": "unavailable", "reason": reason}
    page = render_payload(payload)
    assert text in page and "完整条目" not in page and "0.00 个百分点" not in page


def test_radio_ids_are_unique_and_unknown_feature_names_are_escaped():
    payload = explanation()
    bad = "<img src=x onerror='bad'>&"
    payload["feature_names"][0] = bad
    for term in payload["terms"]:
        if term["feature"] == "rest_days_own":
            term["feature"] = bad
    row = archived_row()
    page = "".join(render_explanation(payload, row, uid=uid, fusion={}) for uid in ("v1", "v2"))
    assert "<img" not in page and "&lt;img" in page
    ids = re.findall(r"\bid=['\"]([^'\"]+)", page)
    assert len(ids) == len(set(ids))
    radios = re.findall(r"<input[^>]*type=['\"]radio['\"][^>]*>", page)
    assert len(radios) == 4
    names = [re.search(r"\bname=['\"]([^'\"]+)", tag).group(1) for tag in radios]
    assert names[0] == names[1] and names[2] == names[3] and names[0] != names[2]
    labels = re.findall(r"<label[^>]*for=['\"]([^'\"]+)", page)
    assert len(labels) == 4 and all(label in ids for label in labels)


def test_fusion_note_uses_saved_status_before_weight():
    row = archived_row()
    row["detail"]["融合"] = {"状态": "无赔率，未融合", "w": 1.0}
    page = render_payload(row=row)
    assert "没有赔率，当前结论使用模型概率" in page and "模型未参与该融合概率" not in page
    row["detail"]["融合"] = {"状态": "已融合", "w": .6}
    page = render_payload(row=row)
    assert "融合概率" in page and "模型未参与该融合概率" not in page


def site_data():
    settings = copy.deepcopy(load_settings())
    settings["explanation"] = {"top_factors": 3, "atol": 1e-6}
    data = SiteData(NOW, settings)
    data.matches[10] = {"match_id": 10, "competition": "E0", "season": 2026, "home_team": "Liverpool",
                        "away_team": "Arsenal", "match_date": "2026-11-07", "kickoff_utc": "2026-11-07 12:00:00",
                        "kickoff_earliest_utc": "2026-11-07 12:00:00", "status": "scheduled"}
    data.team_zh = {"Liverpool": "利物浦", "Arsenal": "阿森纳"}
    first = archived_row()
    last = archived_row(pid=2, kind="T-1", payload=False, version="gb_l7-new", cutoff="2026-11-07 11:00:00")
    data.preds[10] = {(r["model"], r["model_version"], r["kind"]): r for r in (first, last)}
    return data


def test_formal_site_keeps_each_version_and_latest_missing_explanation():
    data = site_data()
    section = explanations_html(data, 10)
    assert "gb_l7-example" in section and "gb_l7-new" in section
    assert "这一版没有保存模型解释" in section and section.count("完整条目（8 项）") == 1
    assert section.index("gb_l7-new") < section.index("gb_l7-example")
    page = render_match(data, 10, TeamFiles(data.team_zh))
    assert "模型依据" in page and "主胜 60%" in html.unescape(page)
    assert "首发尚未接入" in page and "这一版没有保存模型解释" in page
    assert "<script" not in page.lower()


def test_bad_saved_explanation_keeps_prediction_and_inputs_unchanged():
    data = site_data()
    row = next(iter(data.preds[10].values()))
    row["detail"]["模型解释"]["prediction"]["home_win"] = .8
    before = copy.deepcopy(data.preds)
    page = render_match(data, 10, TeamFiles(data.team_zh))
    assert "模型解释数据校验未通过" in page and "主胜 60%" in html.unescape(page)
    assert "完整条目（8 项）" not in page
    assert data.preds == before


def test_current_preview_uses_result_identity_and_its_own_explanation():
    c = {"status": "无赔率，未融合", "p_1x2": [.44, .3, .26], "level_name": "●●○ 中", "w": 1.0,
         "leans": {key: None for key in ("result", "draw", "over25", "btts", "ah")}, "uncertain": []}
    row = {**IDENTITY, "home_zh": "利物浦", "away_zh": "阿森纳", "league": "英超", "status": "scheduled",
           "kickoff_utc": "2026-11-07 12:00:00", "match_date": "2026-11-07", "date_precision": "minute",
           "model_p": [.44, .3, .26], "market": {"p_1x2": None, "n_books_1x2": 0, "ah": None, "ou": None,
                                                  "last_change_utc": None}, "gap": None, "conclusion": c,
           "explanation": explanation()}
    result = {"rows": [row], "model": IDENTITY["model"], "model_version": IDENTITY["model_version"],
              "cutoff_utc": IDENTITY["cutoff_utc"], "gap_threshold": .06, "days_ahead": 7, "fusion_w": 1.0,
              "hist_last_date": "2026-11-05", "hist_n": 100, "lean": {}, "odds_move": {"hours": 24, "min_change": .05}}
    settings = load_settings()
    page = render_preview(result, settings, NOW)
    assert "模型依据" in page and "完整条目（8 项）" in page
    assert "没有赔率，当前结论使用模型概率" in page and "模型未参与该融合概率" not in page
    result["model_version"] = "different-version"
    page = render_preview(result, settings, NOW)
    assert "身份" in page and "完整条目（8 项）" not in page
    result["model_version"] = IDENTITY["model_version"]
    del row["explanation"]
    assert "本次没有提供模型解释" in render_preview(result, settings, NOW)
