"""取数层 src/features 的自动检查：截止时间、用途检查、运行模式、故意偷看未来会不会被拦住。

用内存里的临时数据库（schema.py 建表）造几场比赛，不碰真实数据库。时间都是 UTC。
造的数据（1 月英国时间 = UTC，8 月英国时间 = UTC+1）：
  1 号  E0 2023/24  A 对 B  2024-01-10 20:00 开球  2:1（有射门、角球）
  2 号  E0 2023/24  C 对 D  2024-01-10 只有日期       0:0
  3 号  E0 2023/24  A 对 C  2024-01-14 15:00 开球  1:1  ← 多数测试里的“本场”，有赛前、收盘赔率
  4 号  FAC（足总杯）2023/24 B 对 A  2024-01-17 19:45  杯赛：同一对球队同一赛季第二次碰面
  5 号  E0 2024/25（期末考试赛季）A 对 D  2024-08-17 14:00  3:0
  6 号  E0 2026/27（上线后）A 对 B  2026-08-22 14:00  1:0
"""
import copy
import sqlite3

import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.features import (FeatureAccessError, FeatureStore, LookAheadError, SeasonLockedError,
                          UsageNotAllowedError)
from src.features.known_time import compute_known_times, to_utc_text
from src.features.usage import PAST, THIS, DataUsage

MATCHES = [
    (1, "E0", 2023, "A", "B", "2024-01-10", "2024-01-10 20:00"),
    (2, "E0", 2023, "C", "D", "2024-01-10", None),
    (3, "E0", 2023, "A", "C", "2024-01-14", "2024-01-14 15:00"),
    (4, "FAC", 2023, "B", "A", "2024-01-17", "2024-01-17 19:45"),
    (5, "E0", 2024, "A", "D", "2024-08-17", "2024-08-17 14:00"),
    (6, "E0", 2026, "A", "B", "2026-08-22", "2026-08-22 14:00"),
]
SCORES = {1: (2, 1, "H"), 2: (0, 0, "D"), 3: (1, 1, "D"), 4: (0, 2, "A"), 5: (3, 0, "H"), 6: (1, 0, "H")}


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    create_tables(c)
    c.executemany("insert into teams values (?, null, 'England')", [(t,) for t in "ABCD"])
    c.executemany("insert into matches values (?, ?, ?, ?, ?, ?, ?, 'finished')", MATCHES)
    c.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result, home_shots, away_shots,"
                  " home_corners, away_corners) values (?, 'football-data', ?, ?, ?, 10, 8, 5, 3)",
                  [(m, *s) for m, s in SCORES.items()])
    odds = []
    for mid in (1, 3):
        odds += [(mid, "football-data", "平博 Pinnacle", "1x2", "pre", None, 2.0, 3.4, 3.9),
                 (mid, "football-data", "多家平均", "1x2", "pre", None, 1.95, 3.3, 3.8),
                 (mid, "football-data", "平博 Pinnacle", "1x2", "close", None, 1.8, 3.6, 4.5)]
    odds.append((3, "某个没登记的网站", "平博 Pinnacle", "1x2", "pre", None, 2.1, 3.3, 3.7))
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away)"
                  " values (?, ?, ?, ?, ?, ?, ?, ?, ?)", odds)
    c.commit()
    return c


@pytest.fixture
def dev(con, settings):
    return FeatureStore(con, "开发", settings)


def ids(df: pd.DataFrame) -> list[int]:
    return sorted(df["match_id"].tolist())


# ---------- 截止时间：比分、统计 ----------
def test_scores_only_after_kickoff_plus_buffer(dev):
    """1 号 20:00 开球，缓冲 3 小时 → 23:00 才能知道比分。正好 23:00 能拿到，早 1 秒拿不到。"""
    assert 1 not in ids(dev.historical_matches("2024-01-10 21:30", "A版输入"))      # 比赛还在踢
    assert 1 not in ids(dev.historical_matches("2024-01-10 22:59:59", "A版输入"))   # 边界前 1 秒
    got = dev.historical_matches("2024-01-10 23:00", "A版输入")                      # 正好在边界上
    assert 1 in ids(got)
    assert (got["known_at_utc"] <= "2024-01-10 23:00:00").all()


def test_stats_follow_same_rule(dev):
    df = dev.historical_matches("2024-01-10 23:00", "A版输入", data=["比分", "射门、xG"])
    row = df.set_index("match_id").loc[1]
    assert (row["home_goals"], row["home_shots"]) == (2, 10)
    assert "home_corners" not in df.columns  # 没要角球就不给


def test_date_only_match_known_next_day(dev):
    """2 号只有日期：当作当天 24:00（英国时间）开球，比分要到第二天 03:00 才能用。同一天 20:00 那场 23:00 就能用。"""
    late_evening = ids(dev.historical_matches("2024-01-10 23:59", "A版输入"))
    assert 1 in late_evening and 2 not in late_evening
    assert 2 not in ids(dev.historical_matches("2024-01-11 02:59:59", "A版输入"))
    assert 2 in ids(dev.historical_matches("2024-01-11 03:00", "A版输入"))


def test_date_only_summer_time():
    """英国夏令时：2023-08-12 只有日期 → 当天 00:00 英国时间 = 前一天 23:00 UTC；当天 24:00 = 23:00 UTC。"""
    m = pd.DataFrame({"match_id": [9], "match_date": ["2023-08-12"], "kickoff_utc": [None]})
    kt = compute_known_times(m, load_settings()["features"]).iloc[0]
    assert kt["kickoff_earliest_utc"] == "2023-08-11 23:00:00"
    assert kt["kickoff_latest_utc"] == "2023-08-12 23:00:00"
    assert kt["result_known_utc"] == "2023-08-13 02:00:00"


def test_date_only_match_pre_match_cutoff(dev):
    """只有日期的比赛，T-24 从当天 00:00 往前算；当天早上 8 点再做赛前预测就算开球后（宁早勿晚）。"""
    assert dev.cutoff_before(2, 24, "A版输入") == "2024-01-09 00:00:00"
    with pytest.raises(LookAheadError):
        dev.pre_match_odds(2, "2024-01-10 08:00", "B版输入")


def test_cutoff_with_timezone_is_converted():
    assert to_utc_text("2024-01-14T16:00+01:00") == "2024-01-14 15:00:00"
    assert to_utc_text(pd.Timestamp("2024-01-14 15:00")) == "2024-01-14 15:00:00"  # 没时区当作 UTC
    with pytest.raises(ValueError):
        to_utc_text(None)


# ---------- 截止时间：赔率 ----------
def test_pre_match_odds_known_24h_before(dev):
    """3 号 2024-01-14 15:00 开球：赛前赔率 13 日 15:00（T-24）起能拿到，早 1 秒拿不到。没登记的来源永远拿不到。"""
    t24 = dev.cutoff_before(3, 24, "B版输入")
    assert t24 == "2024-01-13 15:00:00"
    got = dev.pre_match_odds(3, t24, "B版输入")
    assert sorted(got["bookmaker"]) == ["多家平均", "平博 Pinnacle"]
    assert set(got["source"]) == {"football-data"}
    assert dev.pre_match_odds(3, "2024-01-13 14:59:59", "B版输入").empty
    assert len(dev.pre_match_odds(3, dev.cutoff_before(3, 1), "融合", bookmaker="平博 Pinnacle")) == 1


def test_closing_odds_never_for_this_match_before_kickoff(dev):
    with pytest.raises(UsageNotAllowedError):                        # B 版输入：用途不允许
        dev.closing_odds(3, "2024-01-14 14:00", "B版输入")
    with pytest.raises(UsageNotAllowedError):
        dev.closing_odds(3, "2024-01-14 14:00", "A版输入")
    assert dev.closing_odds(3, "2024-01-14 14:59:59", "评估").empty   # 评估也要等开球
    assert len(dev.closing_odds(3, "2024-01-14 15:00", "评估")) == 1  # 开球那一刻收盘


def test_a_version_cannot_read_any_odds(dev):
    with pytest.raises(UsageNotAllowedError, match="A 版不用赔率"):
        dev.pre_match_odds(3, "2024-01-13 20:00", "A版输入")
    for timing in ("pre", "close"):
        with pytest.raises(UsageNotAllowedError):
            dev.historical_odds("2024-01-14 14:00", "A版输入", timing=timing)


def test_historical_closing_odds_for_b_version(dev):
    """以前比赛的收盘赔率可以给 B 版；本场（3 号）的不在里面。"""
    df = dev.historical_odds(dev.cutoff_before(3, 6), "B版输入", timing="close")
    assert ids(df.drop_duplicates("match_id")) == [1]


def test_unknown_or_wrong_usage(dev):
    with pytest.raises(UsageNotAllowedError, match="不认识"):
        dev.historical_matches("2024-01-12", "随便用用")
    with pytest.raises(TypeError):                                    # 不声明用途：连调用都不行
        dev.historical_matches("2024-01-12")
    with pytest.raises(UsageNotAllowedError):                         # 本场比分是答案，不能当输入
        dev.match_result(3, "2024-01-20", "A版输入")
    with pytest.raises(ValueError):                                   # 玩法写错不悄悄返回空表
        dev.pre_match_odds(3, "2024-01-14 14:00", "B版输入", market="1X2")


def test_match_result_only_after_known(dev):
    assert dev.match_result(3, "2024-01-14 17:59:59", "评估").empty
    r = dev.match_result(3, "2024-01-14 18:00", "评估").iloc[0]
    assert (r["home_goals"], r["away_goals"], r["result"]) == (1, 1, "D")


def test_stage_gate(con, settings):
    """角球第 10 阶段、杯赛欧战比分第 8 阶段才能当模型输入；复盘现在就能用。"""
    fs = FeatureStore(con, "开发", settings)
    with pytest.raises(UsageNotAllowedError, match="第 10 阶段"):
        fs.historical_matches("2024-01-12", "A版输入", data=["角球、黄牌"])
    assert len(fs.historical_matches("2024-01-12", "复盘", data=["角球、黄牌"])) == 2
    with pytest.raises(UsageNotAllowedError, match="第 8 阶段"):
        fs.cup_results("2024-01-20", "A版输入")
    later = copy.deepcopy(settings)
    later["features"]["current_stage"] = 10
    fs10 = FeatureStore(con, "开发", later)
    assert "home_corners" in fs10.historical_matches("2024-01-12", "A版输入", data=["角球、黄牌"]).columns
    assert ids(fs10.cup_results("2024-01-20", "A版输入")) == [4]


def test_cup_schedule_only_for_rest_days(dev):
    """杯赛赛程只能算休息天数和欧战后标签：A 版直接要就报错；用正确的用途能拿到，而且不带比分。"""
    with pytest.raises(UsageNotAllowedError, match="休息天数"):
        dev.team_schedule("A", "2024-01-20", "A版输入")
    s = dev.team_schedule("A", "2024-01-20", "休息天数和欧战后标签")
    assert ids(s) == [1, 3, 4] and s.set_index("match_id").loc[4, "is_cup"]
    assert "home_goals" not in s.columns
    assert ids(dev.team_schedule("A", "2024-01-20", "A版输入", include_cups=False)) == [1, 3]
    with pytest.raises(ValueError):                                   # 联赛函数里传杯赛代码 → 拿错了函数
        dev.historical_matches("2024-01-20", "A版输入", competition="FAC")


# ---------- 运行模式 ----------
def test_dev_mode_locks_exam_season(dev):
    assert ids(dev.historical_matches("2024-07-01", "A版输入")) == [1, 2, 3]     # 考试赛季开始前：没问题
    with pytest.raises(SeasonLockedError, match="期末考试"):
        dev.historical_matches("2024-09-01", "A版输入")                        # 会取到 5 号（2024/25）
    with pytest.raises(SeasonLockedError):
        dev.cutoff_before(5, 24, "A版输入")                                     # 把考试赛季的比赛当预测目标
    with pytest.raises(SeasonLockedError):
        dev.prediction_targets("2024-08-01", "2024-09-01", "A版输入")
    with pytest.raises(SeasonLockedError):
        dev.match_result(5, "2024-09-01", "评估")
    with pytest.raises(SeasonLockedError):
        dev.team_recent_matches("A", "2024-09-01", "A版输入", n=3)


def test_exam_mode_can_read_exam_season(con, settings):
    exam = FeatureStore(con, "考试", settings)
    assert 5 in ids(exam.historical_matches("2024-09-01", "A版输入"))
    assert exam.match_result(5, "2024-09-01", "评估")["home_goals"].item() == 3
    assert exam.cutoff_before(5, 24, "A版输入") == "2024-08-16 14:00:00"
    with pytest.raises(SeasonLockedError, match="上线后"):                     # 上线后的赛季考试模式也不读
        exam.historical_matches("2026-09-01", "A版输入")


def test_live_mode_exam_season_only_for_inputs(con, settings):
    """实时模式：考试赛季可以用来估当前实力，但不在上面打分。"""
    live = FeatureStore(con, "实时", settings)
    assert ids(live.historical_matches("2026-09-01", "A版输入")) == [1, 2, 3, 5, 6]
    with pytest.raises(SeasonLockedError, match="打分"):
        live.match_result(5, "2024-09-01", "评估")
    assert live.match_result(6, "2026-09-01", "评估")["home_goals"].item() == 1


# ---------- 故意偷看未来 ----------
def test_deliberate_peeking_is_blocked(con, dev):
    """一个常见的错误写法：“取本场日期当天及以前的比赛”。直接查数据库会把本场的比分也取出来，取数层不会。"""
    target = 3  # 2024-01-14 15:00 开球
    naive = pd.read_sql("select m.match_id, s.home_goals from matches m join match_stats s using (match_id) "
                        "where m.match_date <= '2024-01-14' and m.competition = 'E0'", con)
    assert target in naive["match_id"].tolist()          # 错误写法：偷看到了本场的比分（答案）
    cutoff = dev.cutoff_before(target, 1, "A版输入")       # 正确写法：开球前 1 小时
    assert target not in ids(dev.historical_matches(cutoff, "A版输入"))
    recent = dev.team_recent_matches("A", cutoff, "A版输入", n=10)
    assert target not in ids(recent)
    # 换个花样：截止时间故意设在开球之后，想拿本场赔率做赛前预测 → 直接报错
    with pytest.raises(LookAheadError, match="偷看未来"):
        dev.pre_match_odds(target, "2024-01-14 16:00", "B版输入")
    # 想用收盘赔率（开球时才有）当本场输入 → 报错
    with pytest.raises(FeatureAccessError):
        dev.closing_odds(target, cutoff, "B版输入")


def test_known_at_never_after_cutoff(dev):
    """每个函数返回的 known_at_utc 都不晚于截止时间。"""
    cutoff = "2024-01-14 18:00:00"
    frames = [dev.historical_matches(cutoff, "A版输入"), dev.team_recent_matches("A", cutoff, "A版输入", 5),
              dev.historical_odds(cutoff, "B版输入"), dev.closing_odds(3, cutoff, "评估"),
              dev.pre_match_odds(3, "2024-01-14 14:00", "B版输入"), dev.match_result(3, cutoff, "评估"),
              dev.team_schedule("A", cutoff, "休息天数和欧战后标签")]
    for df in frames:
        assert not df.empty and (df["known_at_utc"] <= cutoff).all()


# ---------- 兼容以后的改动（杯赛的轮次、多个来源） ----------
def test_team_view_and_same_pair_twice(dev):
    """A 和 B 一个赛季碰了两次（联赛 + 足总杯）：按 match_id 区分，联赛函数只拿联赛那场。"""
    r = dev.team_recent_matches("A", "2024-01-25", "A版输入", n=5)
    assert ids(r) == [1, 3]
    first = r.iloc[0]
    assert (first["is_home"], first["opponent"], first["goals_for"], first["goals_against"],
            first["team_result"]) == (True, "B", 2, 1, "W")
    assert dev.team_recent_matches("A", "2024-01-25", "A版输入", n=1)["match_id"].tolist() == [3]
    assert dev.team_recent_matches("B", "2024-01-25", "A版输入", n=5).iloc[0]["team_result"] == "L"


def test_optional_round_column_and_source_priority(con, settings):
    """matches 加了 round 列照样能用并带出来；同一场有两个来源的比分时只出一行，用优先的来源。"""
    con.execute("alter table matches add column round text")
    con.execute("update matches set round = '第 3 轮' where match_id = 4")
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result) "
                "values (1, 'wikipedia', 9, 9, 'D')")
    fs = FeatureStore(con, "开发", settings)
    df = fs.historical_matches("2024-01-20", "A版输入")
    assert "round" in df.columns and df["match_id"].is_unique
    assert df.set_index("match_id").loc[1, "home_goals"] == 2
    assert fs.team_schedule("A", "2024-01-20", "显示").set_index("match_id").loc[4, "round"] == "第 3 轮"


def test_prediction_targets_hide_status(dev):
    t = dev.prediction_targets("2024-01-01", "2024-02-01", "A版输入")
    assert ids(t) == [1, 2, 3] and "status" not in t.columns


# ---------- 数据用途字典本身 ----------
def test_data_usage_dictionary_rules(settings):
    u = DataUsage(settings)
    pre_usages = {k for k in u.usages if u.is_pre_match(k)}
    for name, item in u.data.items():
        if item["什么时候能知道"] in ("赛后", "收盘"):
            # 赛后 / 开球时才知道的数据，本场不能有任何赛前用途
            assert not set(item[THIS]) & pre_usages, name
    for scope in (PAST, THIS):
        assert "A版输入" not in u.data["赛前赔率"][scope]
        assert "A版输入" not in u.data["收盘赔率"][scope]
    assert set(u.data["杯赛欧战赛程"][PAST]) <= {"休息天数和欧战后标签", "显示"}
    assert u.modes["dev"]["exam"] == set()


def test_data_usage_typo_is_caught(tmp_path, settings):
    text = (tmp_path / "x.yaml")
    from src.common.config import resolve_path
    raw = resolve_path(settings["features"]["data_usage"]).read_text(encoding="utf-8")
    text.write_text(raw.replace("本场: [评估, 机器人结算, 复盘]", "本场: [评估, 机器人结算, 复盘盘]"), encoding="utf-8")
    with pytest.raises(ValueError, match="不认识的用途"):
        DataUsage(settings, text)
