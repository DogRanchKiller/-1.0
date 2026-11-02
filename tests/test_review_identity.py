"""复盘的存档身份和独立标杆回归；只用临时或内存数据，不读取正式数据。"""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.features import FeatureStore
from src.jobs.review import run_review
from src.review import match_review, reconcile, reports
from src.web.site import data as site_data
from src.web.site import match as match_page, pages
from src.web.site.common import TeamFiles


def prediction(pid=1, model="gb", version="v1", kind="T-1", detail=None):
    return pd.Series({
        "prediction_id": pid, "match_id": 1, "competition": "E0", "season": 2026,
        "home_team": "A", "away_team": "B", "match_date": "2026-11-01",
        "kickoff_utc": "2026-11-01 15:00:00", "kind": kind, "model": model,
        "model_version": version, "cutoff_utc": "2026-11-01 14:00:00",
        "created_at_utc": "2026-11-01 14:01:00", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
        "detail": json.dumps(detail if detail is not None else {"计划时点": int(kind[2:]) if kind.startswith("T-") else None, "晚了几分钟": 0,
                                                                  "融合": {"信心等级": 2}}),
    })


@pytest.fixture
def archive():
    con = sqlite3.connect(":memory:")
    create_tables(con)
    for row in (prediction(1), prediction(2, version="v2"), prediction(3, model="other"),
                prediction(4), prediction(5, kind="T-24")):
        row.to_frame().T.to_sql("predictions", con, if_exists="append", index=False)
    yield con
    con.close()


def closing_odds():
    return pd.DataFrame([
        {"bookmaker": book, "source": src, "market": market, "timing": "close", "line": line,
         "home": 2., "draw": 3.5, "away": 4., "over": 1.9, "under": 2.0}
        for book, src, market, line in (("平博 Pinnacle", "football-data", "1x2", None),
                                       ("多家平均", "football-data", "1x2", None),
                                       ("平博 Pinnacle", "football-data", "ou", 2.5),
                                       ("平博 Pinnacle", "other-source", "1x2", None))
    ])


def match_record():
    vs = [reconcile.parse_version(r, .001) for r in (
        prediction(1, kind="T-24"), prediction(2), prediction(3, version="v2", kind="T-24"),
        prediction(4, model="other"))]
    return {"match_id": 1, "competition": "E0", "season": 2026, "season_label": "2026/27",
            "home_team": "A", "away_team": "B", "match_date": "2026-11-01",
            "kickoff_utc": "2026-11-01 15:00:00", "home_goals": 2, "away_goals": 1, "result": "H",
            "xg": None, "red": None, "stats_source": "football-data", "versions": vs,
            "close": {"book": "平博 Pinnacle", "p_1x2": np.array([.5, .3, .2]), "p_over25": None},
            "all_closings": [{"book": book, "bookmaker": book, "benchmark_source": "football-data",
                              "market": "1x2", "timing": "close", "qualified": True,
                              "qualification": "source_declared_close", "p_1x2": np.array(p), "p_over25": None}
                             for book, p in (("平博 Pinnacle", [.5, .3, .2]), ("多家平均", [.4, .3, .3]))]}


def test_latest_identity_is_model_version_and_kind_without_mutating_archive(archive):
    before = archive.execute("select * from predictions order by prediction_id").fetchall()
    rows = reconcile.latest_predictions(archive)
    assert set(rows["prediction_id"]) == {2, 3, 4, 5}
    assert archive.execute("select * from predictions order by prediction_id").fetchall() == before


def test_site_reader_preserves_same_kind_for_every_model_version(archive):
    rows = list(site_data.read_predictions(archive, ["T-1", "T-24"])[1].values())
    assert {r["prediction_id"] for r in rows} == {2, 3, 4, 5}


@pytest.mark.parametrize("kind, detail, expected", [
    ("preview", {}, "preview"),
    ("T-1", {}, "unknown"),
    ("T-1", {"晚了几分钟": 0}, "unknown"),
    ("T-1", {"计划时点": 1}, "unknown"),
    ("T-1", {"计划时点": 1, "晚了几分钟": 0, "改期重出": False}, "on_time"),
    ("T-24", {"计划时点": 24, "晚了几分钟": 9.5}, "late"),
    ("T-1", {"计划时点": 1, "晚了几分钟": 0, "改期重出": True}, "rescheduled"),
])
def test_archive_status_uses_saved_metadata(kind, detail, expected):
    v = reconcile.parse_version(prediction(kind=kind, detail=detail), .001)
    assert v["archive_status"] == expected
    assert v["rescheduled"] is (detail.get("改期重出") is True)


def test_saved_confidence_and_archive_metadata_reach_scores():
    m = match_record()
    rows = reconcile.score_rows([m], load_settings())
    model_rows = rows[rows["source"] == reconcile.SOURCE_MODEL]
    assert set(zip(model_rows["model"], model_rows["model_version"])) == {("gb", "v1"), ("gb", "v2"), ("other", "v1")}
    assert set(model_rows["confidence"]) == {2}
    assert set(model_rows["archive_status"]) == {"on_time"}
    assert set(model_rows["created_at_utc"]) == {"2026-11-01 14:01:00"}
    benchmark = rows[rows["source"] == reconcile.SOURCE_CLOSE]
    assert len(benchmark) == 2
    assert benchmark[["model", "model_version", "prediction_id", "confidence"]].isna().all().all()
    assert set(benchmark["bookmaker"]) == {"平博 Pinnacle", "多家平均"}


def test_all_closings_preserve_source_bookmaker_market_identity():
    closings = reconcile.all_closings(closing_odds(), ["平博 Pinnacle", "多家平均"], "proportional")
    identities = {(c["benchmark_source"], c["bookmaker"], c["market"], c["timing"]) for c in closings}
    assert len(closings) == len(identities) == 4
    assert {c["bookmaker"] for c in closings} == {"平博 Pinnacle", "多家平均"}


def test_json_selects_official_forecast_per_model_version_and_keeps_two_benchmarks():
    m = match_record()
    review = match_review.build_review(m, {}, {}, load_settings(), "2026-11-02 00:00:00")
    officials = review["各模型版本复盘"]
    assert {(r["model"], r["model_version"], r["正式prediction_id"]) for r in officials} == {
        ("gb", "v1", 2), ("gb", "v2", 3), ("other", "v1", 4)}
    assert len(review["收盘标杆"]) == 2
    assert review["收盘赔率"]["公司"] == "平博 Pinnacle"
    assert "单个" in review["兼容说明"]
    assert {(v["model"], v["model_version"]) for v in review["各版本"]} == {("gb", "v1"), ("gb", "v2"), ("other", "v1")}
    assert all(v["confidence"] == 2 for v in review["各版本"])


def test_euro_labels_reach_review(monkeypatch):
    lab = pd.DataFrame([{"match_id": 1, "is_derby": False, "derby_name": "", "is_title_race": False,
                         "is_relegation_race": False, "home_position": 1, "away_position": 2,
                         "home_points": 5, "away_points": 3, "tie_unresolved": False,
                         "home_after_euro": True, "away_after_euro": False}])
    monkeypatch.setattr(match_review, "label_season", lambda *a, **k: lab)
    labels = match_review.attach_labels(object(), [match_record()], load_settings())[1]
    assert labels["home_after_euro"] is True
    assert labels["away_after_euro"] is False


def test_reports_keep_both_versions_and_each_benchmark():
    m = match_record()
    s = load_settings()
    sc = reconcile.score_rows([m], s)
    review = match_review.build_review(m, {}, {}, s, "2026-11-02 00:00:00")
    text = reports.build_report("月", "2026-11", [review], sc, s, pd.Timestamp("2026-11-02", tz="UTC"))
    assert "v1" in text and "v2" in text and "other" in text
    assert "平博 Pinnacle" in text and "多家平均" in text
    assert "on_time" in text or "准点" in text
    off = reports.official_rows(sc, [review])
    assert set(off["prediction_id"].dropna().astype(int)) == {2, 3, 4}


def test_site_renders_model_identity_and_benchmarks_without_discarding_rows(archive, tmp_path):
    settings = load_settings()
    m = match_record()
    data = site_data.SiteData(now=pd.Timestamp("2026-11-02", tz="UTC"), settings=settings)
    data.preds = site_data.read_predictions(archive, ["T-24", "T-1"])
    data.matches = {1: {**m, "kickoff_earliest_utc": m["kickoff_utc"], "site_league": True}}
    data.results = {1: (2, 1)}
    data.scores = reconcile.score_rows([m], settings)
    data.reviews = {1: match_review.build_review(m, {}, {}, settings, "2026-11-02 00:00:00")}
    data.page_ids = {1}
    single = match_page.render(data, 1, TeamFiles(["A", "B"]))
    record = pages.render_record(data, tmp_path)
    for html in (single, record):
        assert "v1" in html and "v2" in html and "other" in html
        assert "准点" in html or "on_time" in html
        assert "平博 Pinnacle" in html and "多家平均" in html
    assert "开球那一刻" not in single + record


def test_live_metrics_consumes_independent_review_scores():
    from src.evaluate.live_review_metrics import build_live_metrics
    result = build_live_metrics(reconcile.score_rows([match_record()], load_settings()),
                                benchmark_books=["平博 Pinnacle", "多家平均"], n_bins=5, min_matches=30)
    assert len(result["groups"]) == 4
    available = [r for r in result["comparisons"] if r["status"] == "available"]
    assert len(available) == 8
    assert all(r["n_matches"] == 1 for r in available)
    assert result["counts"]["benchmark_rows"] == 2


def populate_results(con, fetched="2026-11-01 20:00:00"):
    for team in ("A", "B"):
        con.execute("insert into teams(team_id, country) values (?, 'England')", (team,))
    for mid in (1, 99):
        con.execute("insert into matches(match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status, round) "
                    "values (?, 'E0', 2026, 'A', 'B', '2026-11-01', '2026-11-01 15:00:00', 'finished', ?)", (mid, str(mid)))
        con.execute("insert into match_stats(match_id, source, home_goals, away_goals) values (?, 'football-data', 2, 1)", (mid,))
        con.execute("insert into match_sources values (?, 'football-data', ?, 'data/raw/fd.csv', ?)",
                    (mid, str(mid), fetched if mid == 1 else "2026-11-01 20:00:00"))
        for book in ("平博 Pinnacle", "多家平均"):
            con.execute("insert into odds(match_id, source, bookmaker, market, timing, home, draw, away) "
                        "values (?, 'football-data', ?, '1x2', 'close', 2.0, 3.5, 4.0)", (mid, book))
    con.execute("insert into odds(match_id, source, bookmaker, market, timing, line, over, under) "
                "values (1, 'football-data', '平博 Pinnacle', 'ou', 'close', 2.5, 1.9, 2.0)")
    con.execute("insert into odds(match_id, source, bookmaker, market, timing, line, over, under) "
                "values (1, 'football-data', '多家平均', 'ou', 'close', 2.75, 1.9, 2.0)")


def test_review_rejects_close_downloaded_after_review_cutoff(archive):
    populate_results(archive, fetched="2026-11-03 00:00:00")
    settings = load_settings()
    col = reconcile.collect(archive, FeatureStore(archive, "实时", settings), settings, "2026-11-02 00:00:00")
    assert len(col.matches) == 1
    assert col.matches[0]["all_closings"] == []
    assert col.closing_coverage["requested_match_ids"] == [1]
    assert col.closing_coverage["qualified_match_count"] == 0
    assert any("fetched_after_cutoff" in r["reasons"] for r in col.matches[0]["closing_evidence"])


def test_review_job_writes_exact_coverage_independent_benchmarks_and_live_metrics(archive, tmp_path):
    populate_results(archive)
    settings = load_settings()
    result = run_review(settings, archive, pd.Timestamp("2026-11-02", tz="UTC"), tmp_path)
    saved = pd.read_csv(tmp_path / "scores.csv")
    close = saved[saved["source"] == reconcile.SOURCE_CLOSE]
    assert len(close) == 3 and close["qualified"].eq(True).all()
    assert set(close["market"]) == {"1x2", "ou"}
    coverage = result["收盘证据覆盖"]
    assert coverage["requested_match_ids"] == [1] and coverage["match_count"] == 1
    groups = {(r["bookmaker"], r["market"]): r for r in coverage["by_bookmaker_market"]}
    assert groups[("平博 Pinnacle", "ou")]["qualified_match_count"] == 1
    assert groups[("多家平均", "ou")]["qualified_match_count"] == 0
    assert groups[("多家平均", "ou")]["line"] == 2.5
    metrics = json.loads((tmp_path / "live_metrics.json").read_text(encoding="utf-8"))
    assert metrics["counts"]["benchmark_rows"] == 3
    review = json.loads((tmp_path / "matches" / "1.json").read_text(encoding="utf-8"))
    assert len(review["收盘标杆"]) == 3
    assert len(review["收盘证据"]) == 4
    assert review["收盘覆盖"]["requested_match_ids"] == [1]


def test_collect_never_scores_reused_id_for_different_teams(archive):
    populate_results(archive)
    archive.execute("update matches set home_team='B', away_team='A' where match_id=1")
    settings = load_settings()
    col = reconcile.collect(archive, FeatureStore(archive, "实时", settings), settings, "2026-11-02 00:00:00")
    assert col.matches == []
    assert any("身份" in key for key in col.skipped)


def test_site_never_attaches_old_prediction_to_reused_fixture_id(archive, tmp_path):
    populate_results(archive)
    archive.execute("update matches set home_team='B', away_team='A' where match_id=1")
    assert site_data.read_predictions(archive, ["T-24", "T-1"]) == {}
    settings = load_settings()
    settings["site"]["review_dir"] = str(tmp_path)
    data = site_data.load(archive, settings, pd.Timestamp("2026-11-02", tz="UTC"))
    assert data.latest(1) is None


def test_report_and_record_show_matched_denominators_and_missing_market_reasons(archive, tmp_path):
    populate_results(archive)
    settings = load_settings()
    settings["site"]["review_dir"] = str(tmp_path)
    now = pd.Timestamp("2026-11-02", tz="UTC")
    run_review(settings, archive, now, tmp_path)
    report = next((tmp_path / "reports").glob("月报*.md")).read_text(encoding="utf-8")
    record = pages.render_record(site_data.load(archive, settings, now), tmp_path)
    for text in (report, record):
        assert all(label in text for label in ("模型场数", "标杆覆盖", "缺失场数", "收盘证据覆盖", "大2.5球"))
        assert "not_ou_2_5" in text or "不是2.5球盘口" in text
        assert "按信心等级校准" in text
    confidence_section = report.split("## 按保存的信心等级")[1].split("## 按联赛")[0]
    assert "Brier" in confidence_section


def test_job_error_summary_preserves_every_official_model_identity(archive, tmp_path):
    populate_results(archive)
    result = run_review(load_settings(), archive, pd.Timestamp("2026-11-02", tz="UTC"), tmp_path)
    groups = result["各模型版本误差归类"]
    assert {(row["model"], row["model_version"], row["kind"], row["archive_status"]) for row in groups} == {
        ("gb", "v1", "T-1", "on_time"), ("gb", "v2", "T-1", "on_time"), ("other", "v1", "T-1", "on_time")}


def test_missing_source_note_survives_scores_review_and_all_site_consumers(archive, tmp_path):
    populate_results(archive)
    # 与A的附加状态表一致；在没有该迁移的旧库也不影响其他测试。
    archive.execute("create table if not exists match_source_state(match_id integer primary key, source_missing integer, "
                    "missing_since_utc text, restored_at_utc text, missing_reason text, last_sources_json text)")
    archive.execute("insert into match_source_state values (1, 1, '2026-11-02 00:00:00', null, 'source_removed', '[]')")
    settings = load_settings()
    settings["site"]["review_dir"] = str(tmp_path)
    now = pd.Timestamp("2026-11-02 00:00:00", tz="UTC")
    run_review(settings, archive, now, tmp_path)
    saved = pd.read_csv(tmp_path / "scores.csv")
    assert saved["source_missing"].eq(True).all()
    review = json.loads((tmp_path / "matches" / "1.json").read_text(encoding="utf-8"))
    note = "来源缺失；赛果为最后已知资料"
    assert review["实际"]["来源说明"] == note
    data = site_data.load(archive, settings, now)
    assert data.result_notes[1] == note
    teams = TeamFiles(["A", "B"])
    for html in (match_page.render(data, 1, teams), pages.render_index(data, teams),
                 pages.render_team(data, "A", teams), pages.render_record(data, tmp_path)):
        assert note in html
    assert note in next((tmp_path / "reports").glob("月报*.md")).read_text(encoding="utf-8")


def test_identical_benchmark_duplicates_are_not_repeated():
    odds = closing_odds().iloc[:1]
    closings = reconcile.all_closings(pd.concat([odds, odds]), ["平博 Pinnacle"], "proportional")
    assert len(closings) == 1
