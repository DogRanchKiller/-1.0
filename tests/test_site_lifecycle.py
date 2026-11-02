"""逐场验收网站合同：临时数据，页面不把未来/迟到/缺证据写成全链通过。"""
import copy
import json
import sqlite3
import sys
import types
from datetime import datetime, timezone

import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.jobs import build_site
from src.web.site import data as site_data, match
from src.web.site.common import TeamFiles, page


NOW = datetime(2026, 11, 2, 11, 0, tzinfo=timezone.utc)


def report():
    slots = [
        {"kind": "T-24", "due_at_utc": "2026-11-01 17:30:00", "window_end_utc": "2026-11-02 11:30:00",
         "state": "late", "label": "迟到存档", "prediction_id": 218,
         "cutoff_utc": "2026-11-02 01:47:37", "created_at_utc": "2026-11-02 01:48:43",
         "late_minutes": 497.6167, "created_late_minutes": 498.7167, "on_time": False,
         "issues": ["重复存档 <raw>"], "other_prediction_ids": [217], "valid_prediction_ids": [218],
         "archives": [{"prediction_id": 217, "cutoff_utc": "2026-11-02 01:40:00",
                       "created_at_utc": "2026-11-02 01:41:00", "issues": ["比赛身份冲突 <raw>"]}]},
        {"kind": "T-6", "due_at_utc": "2026-11-02 11:30:00", "window_end_utc": "2026-11-02 16:30:00",
         "state": "not_due", "label": "尚未到点", "prediction_id": None, "issues": []},
        {"kind": "T-1", "due_at_utc": "2026-11-02 16:30:00", "window_end_utc": "2026-11-02 17:30:00",
         "state": "not_due", "label": "尚未到点", "prediction_id": None, "issues": []},
    ]
    return {
        "generated_at_utc": "2026-11-02 11:00:05", "evaluation_cutoff_utc": "2026-11-02 11:00:00",
        "scope": {"model": "gb_l7", "model_version": "gb_l7-fixture",
          "leagues": ["I1"], "enabled_since_utc": "2026-11-02 01:47:37", "start_utc": "2026-10-30 11:00:00",
          "end_utc": "2026-11-09 11:00:00", "grace_minutes": 20},
        "summary": {"match_count": 1, "recorded_slot_count": 1, "late_slot_count": 1,
          "pending_slot_count": 2, "overdue_slot_count": 0, "missed_slot_count": 0, "invalid_slot_count": 0,
          "archived_complete_count": 0, "scored_complete_count": 0, "complete_on_time_count": 0,
          "source_missing_count": 0, "closing_complete_count": 0, "due_slot_count": 1},
        "matches": [{"match_id": 146417, "competition": "I1", "season": 2026, "home_team": "Sassuolo",
          "away_team": "Fiorentina", "round": "11", "leg": None, "kickoff_utc": "2026-11-02 17:30:00",
          "status": "scheduled", "source_missing": False, "slots": slots,
          "result": {"state": "not_due", "label": "待比赛结束", "known_at_utc": None,
                     "home_goals": None, "away_goals": None, "source": None},
          "review": {"state": "awaiting_result", "label": "待赛果后评分", "score_complete": False,
                     "review_file_valid": False, "scored_prediction_ids": [], "issues": []},
          "closing": {"state": "not_due", "label": "收盘核验尚未到期", "complete": False,
                      "groups": [{"bookmaker": "Bet365", "market": "ah", "line": -0.25,
                       "state": "not_due", "label": "尚未到期", "qualified": False,
                       "candidate_count": 0, "reasons": []}]},
          "lineup": {"state": "unverified", "label": "首发链路尚未验收"},
          "archives_complete": False, "score_complete": False, "complete_on_time": False}],
        "warnings": ["来源提醒 <script>alert(1)</script>"]}


def site_fixture():
    data = site_data.SiteData(NOW, copy.deepcopy(load_settings()))
    data.lifecycle = report()
    data.comp_names = {"I1": "意甲"}
    data.team_zh = {"Sassuolo": "萨索洛", "Fiorentina": "佛罗伦萨"}
    row = data.lifecycle["matches"][0]
    data.matches[row["match_id"]] = {**row, "match_date": "2026-11-02", "date_precision": "minute",
                                      "kickoff_earliest_utc": row["kickoff_utc"], "site_league": True}
    data.page_ids = {row["match_id"]}
    return data


def test_lifecycle_page_keeps_future_late_and_unverified_evidence_distinct():
    from src.web.site import lifecycle

    data = site_fixture()
    html = lifecycle.render(data)
    assert "逐场验收" in html and "萨索洛 对 佛罗伦萨" in html
    assert "gb_l7-fixture" in html and "10-30 19:00" in html and "11-09 19:00" in html
    assert "尚未到点" in html and "迟到存档" in html and "待赛果后评分" in html
    assert "T-24" in html and "T-6" in html and "T-1" in html
    assert "218" in html and "217" in html and "11-02 09:48:43" in html
    assert "497.6" in html and "498.7" in html
    assert "首发链路尚未验收" in html and "收盘核验尚未到期" in html
    assert "20 分钟" in html and "迟到" in html
    assert "全链验收通过" not in html and "首发验收通过" not in html and "准确率" not in html
    assert "<script>" not in html and "&lt;script&gt;" in html and "&lt;raw&gt;" in html
    assert "id='match-146417'" in html and "href='match/146417.html'" in html


def test_report_generation_time_does_not_hide_historical_evidence_cutoff():
    from src.web.site import lifecycle

    data = site_fixture()
    data.lifecycle["evaluation_cutoff_utc"] = "2026-11-01 01:02:03"
    html = lifecycle.render(data)
    assert "证据截止" in html and "11-01 09:02:03" in html
    assert "报告生成" in html and "11-02 19:00:05" in html
    data.lifecycle.pop("evaluation_cutoff_utc")
    html = lifecycle.render(data)
    assert "证据截止时间未知" in html


@pytest.mark.parametrize("payload", [{}, {"generated_at_utc": "2026-11-02 11:00:00", "scope": {},
                                           "summary": {"match_count": 0}, "matches": [], "warnings": []}])
def test_empty_or_missing_report_is_not_reported_as_pass(payload):
    from src.web.site import lifecycle

    data = site_fixture()
    data.lifecycle = payload
    html = lifecycle.render(data)
    assert "没有可验收比赛" in html or "验收报告暂不可用" in html
    assert "100%" not in html.split("<body>", 1)[1] and "全链验收通过" not in html


def test_navigation_and_match_reference_use_same_report():
    data = site_fixture()
    html = match.render(data, 146417, TeamFiles(["Sassuolo", "Fiorentina"]))
    assert "href='../lifecycle.html#match-146417'" in html
    assert "迟到存档" in html and "尚未到点" in html and "首发链路尚未验收" in html
    assert "href='lifecycle.html'" in page("x", "", "", NOW, 15)
    data.lifecycle = {}
    assert "#match-146417" not in match.render(data, 146417, TeamFiles([]))


def test_site_load_calls_readonly_report_with_original_connection_and_now(tmp_path, monkeypatch):
    db = tmp_path / "site.db"
    writer = connect(db)
    create_tables(writer)
    writer.executemany("insert into teams (team_id, country, name_zh) values (?, '意大利', ?)",
                       [("Sassuolo", "萨索洛"), ("Fiorentina", "佛罗伦萨")])
    writer.commit()
    writer.close()
    con = build_site.open_readonly(db)
    expected = report()
    calls = []
    module = types.ModuleType("src.review.lifecycle")

    def build_report(received_con, settings, now):
        calls.append((received_con, settings, now))
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            received_con.execute("create table forbidden_probe(value integer)")
        return expected

    module.build_report = build_report
    monkeypatch.setitem(sys.modules, "src.review.lifecycle", module)
    s = copy.deepcopy(load_settings())
    s["site"]["review_dir"] = str(tmp_path / "review")
    before = db.read_bytes()
    try:
        data = site_data.load(con, s, NOW)
    finally:
        con.close()
    assert calls == [(con, s, NOW)]
    assert data.lifecycle is expected and db.read_bytes() == before
    assert data.zh("Sassuolo") == "萨索洛" and data.zh("Fiorentina") == "佛罗伦萨"


def test_build_writes_same_report_to_html_and_json(tmp_path):
    data = site_fixture()
    folder = tmp_path / "site"
    counts = build_site.write_site(data, folder, folder)
    assert counts["逐场验收页"] == 1
    assert json.loads((folder / "lifecycle.json").read_text(encoding="utf-8")) == data.lifecycle
    html = (folder / "lifecycle.html").read_text(encoding="utf-8")
    assert "href='lifecycle.json'" in html and "迟到存档" in html
    assert len(list(folder.rglob("*.html"))) == sum(counts.values())


@pytest.mark.parametrize("case", [
    "report_list", "summary_list", "negative_count", "string_count", "boolean_count", "wrong_denominator",
    "false_complete_count", "match_null", "slot_null", "duplicate_slot", "unknown_state", "false_review",
    "groups_string", "lineup_claim",
])
def test_bad_report_contract_cannot_claim_completion_or_break_other_pages(case, tmp_path):
    from src.web.site import lifecycle

    data = site_fixture()
    payload = data.lifecycle
    marker = "不应显示的完成"
    if case == "report_list":
        data.lifecycle = [payload]
    elif case == "summary_list":
        payload["summary"] = [marker]
    elif case in {"negative_count", "string_count", "boolean_count", "wrong_denominator"}:
        payload["summary"]["match_count"] = {"negative_count": -1, "string_count": "1", "boolean_count": True,
                                              "wrong_denominator": 2}[case]
    elif case == "false_complete_count":
        payload["summary"]["complete_on_time_count"] = 1
    elif case == "match_null":
        payload["matches"] = [None]
    elif case == "slot_null":
        payload["matches"][0]["slots"][0] = None
    elif case == "duplicate_slot":
        payload["matches"][0]["slots"][1] = copy.deepcopy(payload["matches"][0]["slots"][0])
    elif case == "unknown_state":
        payload["matches"][0]["slots"][0].update(state="unknown_contract_state", label=marker)
    elif case == "false_review":
        payload["matches"][0]["review"].update(state="complete", label=marker, score_complete=False)
    elif case == "groups_string":
        payload["matches"][0]["closing"]["groups"] = marker
    elif case == "lineup_claim":
        payload["matches"][0]["lineup"] = {"state": "complete", "label": "首发验收通过"}

    html = lifecycle.render(data)
    assert "验收报告暂不可用" in html
    assert marker not in html and "首发验收通过" not in html
    assert "id='match-146417'" not in html
    single = match.render(data, 146417, TeamFiles(["Sassuolo", "Fiorentina"]))
    assert "#match-146417" not in single and marker not in single and "首发验收通过" not in single
    folder = tmp_path / "site"
    build_site.write_site(data, folder, folder)
    assert json.loads((folder / "lifecycle.json").read_text(encoding="utf-8")) == data.lifecycle


@pytest.mark.parametrize("payload", [None, [], {"matches": [None]}])
def test_site_load_keeps_bad_report_diagnostic_without_crashing_team_names(payload, tmp_path, monkeypatch):
    from src.web.site import lifecycle

    db = tmp_path / "site.db"
    writer = connect(db)
    create_tables(writer)
    writer.commit()
    writer.close()
    module = types.ModuleType("src.review.lifecycle")
    module.build_report = lambda *_: payload
    monkeypatch.setitem(sys.modules, "src.review.lifecycle", module)
    settings = copy.deepcopy(load_settings())
    settings["site"]["review_dir"] = str(tmp_path / "review")
    con = build_site.open_readonly(db)
    try:
        data = site_data.load(con, settings, NOW)
    finally:
        con.close()
    assert data.lifecycle is payload
    assert "验收报告暂不可用" in lifecycle.render(data)


def test_lifecycle_footer_and_denominator_do_not_relabel_cutoff_as_generation():
    from src.web.site import lifecycle

    data = site_fixture()
    data.lifecycle["evaluation_cutoff_utc"] = "2026-11-01 01:02:03"
    data.lifecycle["generated_at_utc"] = "2026-11-02 12:34:56"
    html = lifecycle.render(data)
    footer = html.split("<footer", 1)[1]
    assert "2026-11-02 20:34" in footer and "2026-11-02 19:00" not in footer
    assert "比赛分母为本次范围内 1 场" in html
    assert "时点总数 3" in html and "已到期纳入跟踪 1" in html
    assert "范围内的未来比赛也保留在分母中" in html


def test_unknown_state_label_is_not_trusted_as_completed():
    from src.web.site import lifecycle

    html = lifecycle.state_html({"state": "unknown_contract_state", "label": "验收已完成"})
    assert "证据未知" in html and "验收已完成" not in html


@pytest.mark.parametrize("state,label,missing", [
    ("unknown", "来源状态未知", None), ("present", "当前来源可核对", 0),
    ("missing", "来源暂缺，保留最后资料", 1),
])
def test_source_state_keeps_unknown_distinct_from_verified_present(state, label, missing):
    from src.web.site import lifecycle

    data = site_fixture()
    row = data.lifecycle["matches"][0]
    row["source_missing"] = missing
    row["source_state"] = {"state": state, "label": label}
    data.lifecycle["summary"]["source_missing_count"] = int(missing == 1)
    html = lifecycle.render(data)
    assert label in html and "验收报告暂不可用" not in html


def test_completed_report_can_render_without_implying_lineup_acceptance():
    from src.web.site import lifecycle

    data = site_fixture()
    row = data.lifecycle["matches"][0]
    for index, slot in enumerate(row["slots"]):
        slot.update(state="on_time", label="准点存档", prediction_id=index + 1, on_time=True,
                    cutoff_utc=slot["due_at_utc"], created_at_utc=slot["due_at_utc"], issues=[])
    row["review"].update(state="complete", label="三个正式时点的评分与单场复盘已核验",
                         score_complete=True, review_file_valid=True, scored_prediction_ids=[1, 2, 3])
    row["closing"].update(state="complete", label="四组收盘证据齐全", complete=True)
    row["closing"]["groups"] = [
        {"bookmaker": book, "market": market, "line": None if market == "1x2" else 2.5,
         "state": "qualified", "label": "已有合格收盘证据", "qualified": True, "candidate_count": 1, "reasons": []}
        for book in ("Bet365", "威廉希尔") for market in ("1x2", "ou")]
    row.update(archives_complete=True, score_complete=True, complete_on_time=True)
    data.lifecycle["summary"].update(recorded_slot_count=3, late_slot_count=0, pending_slot_count=0,
                                     archived_complete_count=1, scored_complete_count=1,
                                     complete_on_time_count=1, closing_complete_count=1, due_slot_count=3)
    html = lifecycle.render(data)
    assert "验收报告暂不可用" not in html and "三个正式时点的评分与单场复盘已核验" in html
    assert "首发链路尚未验收" in html and "首发验收通过" not in html
