"""时间核对（src/evaluate/timing_check.py）的检查。只用编的数据，不碰真实数据库。"""
import copy
import sqlite3

import pandas as pd
import pytest

from src.common.config import load_settings
from src.evaluate import timing_check as tc


def test_collection_back_days():
    # 采集日周二（1）、周五（4）：周五、六、日、一 → 周五；周二、三、四 → 周二
    wd = pd.Series([0, 1, 2, 3, 4, 5, 6])
    assert tc.collection_back_days(wd, [1, 4]).tolist() == [3, 0, 1, 2, 0, 1, 2]


def test_pre_odds_peek_kickoff_and_date_only():
    fs = load_settings()["features"]
    m = pd.DataFrame({
        "match_id": [1, 2],
        # 1：2021-08-14 周六 15:00 英国时间（夏令时）= 14:00 UTC；截止 = 周五 14:00 UTC
        # 2：2010-09-14 周二，只有日期；最早开球 = 当天 00:00 英国时间 = 周一 23:00 UTC，截止 = 周日 23:00 UTC
        "match_date": ["2021-08-14", "2010-09-14"],
        "kickoff_utc": ["2021-08-14 14:00:00", None],
    })
    p = tc.pre_odds_peek(m, fs, [1, 4], [12, 18], 24)
    # 1：周五 12:00 英国 = 11:00 UTC，早 3 小时；18:00 = 17:00 UTC，晚 3 小时
    assert p.loc[0, "peek_12"] == pytest.approx(-3)
    assert p.loc[0, "peek_18"] == pytest.approx(3)
    # 2：周二当天 12:00 英国 = 11:00 UTC，比截止（周日 23:00 UTC）晚 36 小时
    assert p.loc[1, "peek_12"] == pytest.approx(36)
    assert p["era"].tolist() == ["有开球时间", "只有日期"]
    assert p["weekday"].tolist() == ["六", "二"]
    s = tc.summarize_peek(p, [12, 18], "weekday")
    assert s["场数"].sum() == 2


def test_live_odds_fetch_check():
    con = sqlite3.connect(":memory:")
    con.execute("create table predictions (prediction_id, kind, cutoff_utc, created_at_utc, match_id)")
    con.execute("create table odds_changes (match_id, source, bookmaker, market, changed_at_utc, fetched_at_utc)")
    con.execute("insert into predictions values (1, 'T-24', '2026-01-01 12:00:00', '2026-01-01 12:01:00', 7)")
    rows = [
        # A 家：10:00 的价 10:30 存到，11:30 的价 12:30 才存到（出预测后）→ 最新价晚存到
        (7, "dongqiudi", "A", "1x2", "2026-01-01 10:00:00", "2026-01-01 10:30:00"),
        (7, "dongqiudi", "A", "1x2", "2026-01-01 11:30:00", "2026-01-01 12:30:00"),
        # B 家：唯一的价出预测后才存到 → 当时没有这家
        (7, "dongqiudi", "B", "1x2", "2026-01-01 11:50:00", "2026-01-01 13:00:00"),
        # C 家：按时存到；截止后变的价不算
        (7, "dongqiudi", "C", "1x2", "2026-01-01 09:00:00", "2026-01-01 09:10:00"),
        (7, "dongqiudi", "C", "1x2", "2026-01-01 12:10:00", "2026-01-01 12:20:00"),
        # 亚盘不算
        (7, "dongqiudi", "D", "ah", "2026-01-01 11:00:00", "2026-01-01 13:00:00"),
    ]
    con.executemany("insert into odds_changes values (?, ?, ?, ?, ?, ?)", rows)
    f = tc.live_odds_fetch_check(con)
    assert f.loc[0, "公司数"] == 3
    assert f.loc[0, "最新价晚存到"] == 2
    assert f.loc[0, "当时没有这家"] == 1
    lag = tc.collection_lag(con)
    # 这场第一次存快照是 09:10，09:00 的初盘不算；其余 5 行
    assert len(lag) == 5


def test_run_rejects_exam_seasons():
    s = copy.deepcopy(load_settings())
    s["timing_check"]["seasons"] = [2005, s["seasons"]["exam"][0]]
    with pytest.raises(ValueError):
        tc.run(db_path="不存在.db", settings=s)
