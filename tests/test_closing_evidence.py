"""收盘证据的独立回归：仅用内存库，不下载、不写正式数据、不重考。"""
import sqlite3

import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.features import FeatureStore, SeasonLockedError, UsageNotAllowedError

PIN = "平博 Pinnacle"
AVG = "多家平均"
CUTOFF = "2024-01-12 00:00:00"


@pytest.fixture
def con():
    conn = sqlite3.connect(":memory:")
    create_tables(conn)
    matches = [(i, 2023, "2024-01-10", "2024-01-10 20:00") for i in range(1, 40)]
    matches += [(90, 2024, "2024-08-17", "2024-08-17 14:00"),
                (91, 2026, "2026-08-22", "2026-08-22 14:00")]
    for mid, season, date, kickoff in matches:
        for team in (f"H{mid}", f"A{mid}"):
            conn.execute("insert into teams (team_id, country) values (?, 'England')", (team,))
        conn.execute("insert into matches (match_id, competition, season, home_team, away_team, "
                     "match_date, kickoff_utc, status) values (?, 'E0', ?, ?, ?, ?, ?, 'finished')",
                     (mid, season, f"H{mid}", f"A{mid}", date, kickoff))
    conn.commit()
    yield conn
    conn.close()


def source(con, mid=1, name="football-data", raw="data/raw/fd.csv", fetched="2024-01-11 09:00:00",
           key=None):
    con.execute("insert into match_sources (match_id, source, source_key, raw_file, fetched_at_utc) "
                "values (?, ?, ?, ?, ?)", (mid, name, key or f"{name}:{mid}", raw, fetched))


def odds(con, mid=1, book=PIN, name="football-data", market="1x2", timing="close", line=None,
         home=2.0, draw=3.5, away=4.0, over=None, under=None):
    con.execute("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, "
                "over, under) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, name, book, market, timing, line, home, draw, away, over, under))


def change(con, mid=1, changed="2024-01-10 19:59:00", fetched="2024-01-10 19:59:30", book=PIN):
    con.execute("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, home, "
                "draw, away, changed_at_utc, fetched_at_utc, raw_file) "
                "values (?, 'dongqiudi', ?, 'Pinnacle', '1x2', 2.0, 3.5, 4.0, ?, ?, 'data/raw/dqd.json')",
                (mid, book, changed, fetched))


def gather(con, ids=(1,), cutoff=CUTOFF, mode="开发", **kwargs):
    # 函数内部导入，使新增接口尚未存在时，各条安全边界分别显示为失败。
    from src.review.closing_evidence import collect_closing_evidence
    fs = FeatureStore(con, mode, load_settings())
    return collect_closing_evidence(fs, ids, cutoff, **kwargs)


def summary(df, ids, **kwargs):
    from src.review.closing_evidence import coverage_summary
    return coverage_summary(df, ids, **kwargs)


def test_registered_close_can_be_downloaded_after_kickoff(con):
    source(con)
    odds(con)
    row = gather(con).iloc[0]
    assert bool(row["qualified"])
    assert row["qualification"] == "source_declared_close"
    assert row["reason"] == "registered_source_declared_close"
    assert row["source"] == "football-data" and row["timing"] == "close"
    assert row["fetched_at_utc"] == "2024-01-11 09:00:00"
    assert row["known_at_utc"] == "2024-01-11 09:00:00"
    assert row["close_known_utc"] == "2024-01-10 20:00:00"
    assert row["raw_file"] == "data/raw/fd.csv"
    assert "并非准点现场捕捉" in row["evidence_note"]


def test_review_asof_does_not_use_a_later_download(con):
    source(con, fetched="2024-01-13 09:00:00")
    odds(con)
    df = gather(con)
    assert not df["qualified"].any()
    assert "fetched_after_cutoff" in df.iloc[0]["reasons"]
    assert summary(df, [1])["qualified_match_count"] == 0


@pytest.mark.parametrize("raw,fetched,reason", [
    (None, "2024-01-11 09:00:00", "missing_raw_file"),
    ("", "2024-01-11 09:00:00", "missing_raw_file"),
    ("data/raw/fd.csv", None, "missing_fetched_at_utc"),
    ("data/raw/fd.csv", "bad-time", "invalid_fetched_at_utc"),
])
def test_missing_or_malformed_provenance_is_not_qualified(con, raw, fetched, reason):
    source(con, raw=raw, fetched=fetched)
    odds(con)
    df = gather(con)
    assert not df["qualified"].any()
    assert reason in df.iloc[0]["reasons"]


def test_source_claiming_close_is_not_registered_automatically(con):
    source(con, name="unknown-feed")
    odds(con, name="unknown-feed")
    row = gather(con).iloc[0]
    assert not row["qualified"]
    assert "unregistered_close_source" in row["reasons"]


def test_missing_and_ambiguous_source_links_are_explicit(con):
    odds(con)
    assert "missing_provenance" in gather(con).iloc[0]["reasons"]
    source(con)
    source(con, raw="data/raw/other.csv", key="second-link")
    df = gather(con)
    assert not df["qualified"].any()
    assert all("ambiguous_provenance" in reasons for reasons in df["reasons"])


def test_blank_source_match_key_does_not_establish_complete_provenance(con):
    source(con)
    odds(con)
    con.execute("update match_sources set source_key = ' ' where match_id = 1")
    row = gather(con).iloc[0]
    assert not row["qualified"]
    assert "missing_source_key" in row["reasons"]


def test_dqd_last_observed_pregame_is_never_a_trusted_close(con):
    change(con, changed="2024-01-10 18:00:00", fetched="2024-01-10 18:01:00")
    change(con)
    df = gather(con)
    assert not df["qualified"].any()
    last = df[df["qualification"] == "last_observed_pregame"]
    assert len(last) == 1
    assert last.iloc[0]["changed_at_utc"] == "2024-01-10 19:59:00"
    assert last.iloc[0]["timing"] == "change"
    assert last.iloc[0]["reason"] == "not_verified_close"


def test_historical_pregame_price_was_not_observed_before_kickoff(con):
    change(con, fetched="2024-01-11 09:00:00")
    row = gather(con).iloc[0]
    assert row["qualification"] == "historical_pregame_price"
    assert row["known_at_utc"] == "2024-01-11 09:00:00"
    assert not row["qualified"]
    # 既有 odds_changes 按网站变盘时间可知的规则不能被这次诊断接口改掉。
    fs = FeatureStore(con, "开发", load_settings())
    old = fs.odds_changes(1, "2024-01-10 19:59:30", "B版输入")
    assert old.iloc[0]["known_at_utc"] == "2024-01-10 19:59:00"


@pytest.mark.parametrize("changed", ["2024-01-10 20:00:00", "2024-01-10 20:01:00"])
def test_in_play_change_is_excluded_including_the_kickoff_boundary(con, changed):
    change(con, changed=changed, fetched="2024-01-11 09:00:00")
    row = gather(con).iloc[0]
    assert row["qualification"] == "excluded"
    assert "in_play" in row["reasons"]
    assert not row["qualified"]


def test_future_dqd_download_is_not_last_observed_pregame(con):
    change(con, fetched="2024-01-13 09:00:00")
    row = gather(con).iloc[0]
    assert row["qualification"] == "excluded"
    assert "fetched_after_cutoff" in row["reasons"]


def test_exact_cohort_reports_zero_of_38_despite_unrelated_historical_close(con):
    source(con, mid=39)
    odds(con, mid=39)
    ids = list(range(1, 39))
    df = gather(con, ids, bookmakers=[PIN, AVG], markets=["1x2", "ou"])
    assert set(df["match_id"]) == set(ids)
    assert len(df) == 38 * 2 * 2
    assert set(df["qualification"]) == {"absent"}
    got = summary(df, ids, bookmakers=[PIN, AVG], markets=["1x2", "ou"])
    assert got["match_count"] == 38 and got["qualified_match_count"] == 0
    assert got["coverage"] == 0
    assert len(got["by_bookmaker_market"]) == 4
    assert all(r["match_count"] == 38 and r["qualified_match_count"] == 0
               for r in got["by_bookmaker_market"])


def test_1x2_coverage_does_not_require_ou_and_does_not_count_source_duplicates_twice(con):
    source(con)
    odds(con)
    odds(con, book=AVG)
    df = gather(con, [1, 2], bookmakers=[PIN, AVG], markets=["1x2", "ou"])
    got = summary(df, [1, 2], bookmakers=[PIN, AVG], markets=["1x2", "ou"])
    assert got["match_count"] == 2 and got["qualified_match_count"] == 1
    rows = {(r["bookmaker"], r["market"]): r for r in got["by_bookmaker_market"]}
    assert rows[(PIN, "1x2")]["qualified_match_count"] == 1
    assert rows[(AVG, "1x2")]["qualified_match_count"] == 1
    assert rows[(AVG, "ou")]["qualified_match_count"] == 0
    repeated = pd.concat([df, df], ignore_index=True)
    assert summary(repeated, [1, 2])["qualified_match_count"] == 1


def test_summary_restricts_even_a_supplied_larger_evidence_frame(con):
    source(con)
    odds(con)
    source(con, mid=39)
    odds(con, mid=39)
    df = gather(con, [1, 39])
    got = summary(df, [2, 3], bookmakers=[PIN])
    assert got["match_count"] == 2 and got["qualified_match_count"] == 0
    assert got["by_bookmaker_market"][0]["absence_reasons"] == {"no_record": 2}


@pytest.mark.parametrize("usage", ["A版输入", "B版输入", "融合", "研究实验"])
def test_current_close_usage_guard_applies_even_without_candidates(con, usage):
    with pytest.raises(UsageNotAllowedError):
        gather(con, usage=usage)


@pytest.mark.parametrize("mode", ["开发", "实时"])
def test_exam_mode_guard_applies_even_without_candidates_in_a_mixed_cohort(con, mode):
    with pytest.raises(SeasonLockedError):
        gather(con, [1, 90], cutoff="2026-11-02 00:00", mode=mode)


@pytest.mark.parametrize("table", ["odds", "odds_changes", "match_sources"])
def test_missing_optional_old_database_tables_produce_reasons(con, table):
    if table == "match_sources":
        odds(con)
    con.execute(f"drop table {table}")
    df = gather(con, bookmakers=[PIN])
    assert not df["qualified"].any()
    assert any(f"missing_table:{table}" in reasons for reasons in df["reasons"])


def test_missing_change_table_does_not_invalidate_registered_close(con):
    source(con)
    odds(con)
    con.execute("drop table odds_changes")
    assert gather(con)["qualified"].all()


def test_date_only_match_keeps_conservative_close_known_time(con):
    con.execute("update matches set kickoff_utc = null where match_id = 1")
    source(con, fetched="2024-01-10 22:00:00")
    odds(con)
    early = gather(con, cutoff="2024-01-10 23:59:59").iloc[0]
    assert not early["qualified"] and "not_yet_closed" in early["reasons"]
    late = gather(con, cutoff="2024-01-11 00:00:00").iloc[0]
    assert late["qualified"]
    assert pd.isna(late["kickoff_utc"])
    assert late["close_known_utc"] == "2024-01-11 00:00:00"


def test_timezones_are_normalized_before_comparison(con):
    source(con, fetched="2024-01-11T17:00:00+08:00")
    odds(con)
    row = gather(con, cutoff="2024-01-11T10:00:00+00:00").iloc[0]
    assert row["qualified"] and row["fetched_at_utc"] == "2024-01-11 09:00:00"


@pytest.mark.parametrize("market,line,prices", [
    ("1x2", None, {"home": 1.0}),
    ("1x2", None, {"draw": None}),
    ("1x2", None, {"away": float("inf")}),
    ("ou", None, {"over": 1.9, "under": 2.0}),
    ("ou", 2.5, {"over": None, "under": 2.0}),
    ("ah", None, {"home": 1.9, "away": 2.0}),
])
def test_invalid_prices_or_missing_line_do_not_count_as_evidence(con, market, line, prices):
    source(con)
    odds(con, market=market, line=line, **prices)
    row = gather(con, markets=[market]).iloc[0]
    assert not row["qualified"] and "invalid_odds" in row["reasons"]


def test_empty_cohort_has_no_invented_coverage(con):
    df = gather(con, [])
    got = summary(df, [])
    assert df.empty and got["match_count"] == 0 and got["coverage"] is None


def test_public_store_entry_and_coverage_share_the_exact_requested_cohort(con):
    source(con)
    odds(con)
    fs = FeatureStore(con, "开发", load_settings())
    evidence = fs.closing_evidence([1, 2], CUTOFF, "复盘", bookmaker=PIN, market="1x2")
    from src.review.closing_evidence import closing_coverage
    got = closing_coverage(evidence, [1, 2], bookmakers=[PIN], markets=["1x2"])
    assert got["match_count"] == 2 and got["qualified_match_count"] == 1
    assert got["coverage"] == 0.5
    assert set(evidence["match_id"]) == {1, 2}
    assert evidence.loc[evidence["match_id"] == 2, "reason"].item() == "no_record"


def test_public_store_entry_keeps_prediction_usage_gate_without_records(con):
    fs = FeatureStore(con, "开发", load_settings())
    with pytest.raises(UsageNotAllowedError):
        fs.closing_evidence([1], CUTOFF, "B版输入")


@pytest.mark.parametrize("with_marker", [False, True])
def test_source_missing_metadata_is_optional_and_does_not_erase_known_results(con, with_marker):
    con.execute("drop table if exists match_source_state")
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result) "
                "values (1, 'football-data', 2, 1, 'H')")
    if with_marker:
        # 表由采集/清洗工作实现；这里仅验证取数层的可选透传契约。
        con.execute("create table match_source_state (match_id integer primary key, source_missing integer, "
                    "missing_since_utc text, restored_at_utc text, missing_reason text, last_sources_json text)")
        con.execute("insert into match_source_state values (1, 1, '2024-01-11 09:00:00', null, "
                    "'not_in_current_files', '[\"football-data\"]')")
    fs = FeatureStore(con, "开发", load_settings())
    row = fs.historical_matches(CUTOFF, "复盘").set_index("match_id").loc[1]
    assert row["home_goals"] == 2 and row["status"] == "finished"
    if with_marker:
        assert row["source_missing"] == 1
        assert row["missing_since_utc"] == "2024-01-11 09:00:00"
        assert row["last_sources_json"] == '["football-data"]'
    else:
        assert pd.isna(row["source_missing"]) and pd.isna(row["missing_since_utc"])
        assert pd.isna(row["last_sources_json"])
