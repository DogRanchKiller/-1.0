"""复盘自动标签的自动检查：赛前积分榜（同分排名、截止时间、扣分）、德比、争冠 / 保级的边界。

用内存里的临时数据库（schema.py 建表）造一个 4 队的小联赛，不碰真实数据库。时间都是 UTC。
联赛代码借用 E0（取数层只认 settings 里的联赛代码），赛季 2022（开发赛季），
排名规则用测试自己写的临时 league_rules.yaml，不受真实配置影响。

造的比赛（A = Arsenal, T = Tottenham, C = Chelsea, F = Fulham；英国 8–10 月是夏令时 UTC+1，开球时间直接写 UTC）：
  第 1 轮  2022-08-06 14:00  A 1:1 T     C 2:0 F
  第 2 轮  2022-08-13 14:00  C 1:0 T     F 0:3 A
  第 3 轮  2022-08-20 14:00  T 3:0 F     A 2:0 C
  第 4 轮  2022-10-01 14:00  T 1:0 C     A 0:0 F（只有日期 2022-10-01，没有开球时间）
手算第 3 轮后（截止 2022-08-27）的积分榜：
  A 2胜1平 7 分  进 6 失 1
  T 1胜1平1负 4 分  进 4 失 2  净胜 +2；对 C 0:1 输
  C 2胜1负 6 分 ……
  见下面每个测试里的说明。
"""
import copy
import sqlite3

import pandas as pd
import pytest
import yaml

from src.clean.schema import create_tables
from src.common.config import load_settings, resolve_path
from src.features import FeatureStore, SeasonLockedError
from src.features.standings import (LeagueRules, SeasonTable, compute_table, load_deductions, load_rules,
                                    season_table, standings_at)
from src.review import labels as L

A, T, C, F = "Arsenal", "Tottenham", "Chelsea", "Fulham"
MATCHES = [
    # match_id, 主, 客, 日期, 开球 UTC（None = 只有日期）, 主进, 客进
    (1, A, T, "2022-08-06", "2022-08-06 14:00", 1, 1),
    (2, C, F, "2022-08-06", "2022-08-06 14:00", 2, 0),
    (3, C, T, "2022-08-13", "2022-08-13 14:00", 1, 0),
    (4, F, A, "2022-08-13", "2022-08-13 14:00", 0, 3),
    (5, T, F, "2022-08-20", "2022-08-20 14:00", 3, 0),
    (6, A, C, "2022-08-20", "2022-08-20 14:00", 2, 0),
    (7, T, C, "2022-10-01", "2022-10-01 14:00", 1, 0),
    (8, A, F, "2022-10-01", None, 0, 0),
]


def rules_yaml(tiebreak, needs_all=False, relegation=1, playoff=0) -> dict:
    return {"联赛": {"E0": {"规则": [{"赛季": [1993, 9999], "胜": 3, "平": 1, "同分": tiebreak,
                                      "相互战绩要踢完": "是" if needs_all else "否",
                                      "降级": relegation, "降级附加赛": playoff, "来源": "测试"}]}}}


@pytest.fixture
def settings(tmp_path):
    """真实 settings 的拷贝：排名规则、扣分、标签输出都指到临时文件夹。"""
    s = copy.deepcopy(load_settings())
    (tmp_path / "rules.yaml").write_text(yaml.safe_dump(rules_yaml(["相互战绩积分", "净胜球", "进球"]),
                                                        allow_unicode=True), encoding="utf-8")
    (tmp_path / "ded.yaml").write_text("扣分: []\n", encoding="utf-8")
    s["standings"] = {"rules": str(tmp_path / "rules.yaml"), "deductions": str(tmp_path / "ded.yaml")}
    s["labels"] = {**s["labels"], "output_dir": str(tmp_path / "labels")}
    return s


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    create_tables(c)
    c.executemany("insert into teams (team_id, country) values (?, 'England')", [(t,) for t in (A, T, C, F)])
    c.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                  " status) values (?, 'E0', 2022, ?, ?, ?, ?, 'finished')", [m[:5] for m in MATCHES])
    c.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result) values "
                  "(?, 'football-data', ?, ?, ?)",
                  [(m[0], m[5], m[6], "H" if m[5] > m[6] else "A" if m[5] < m[6] else "D") for m in MATCHES])
    c.commit()
    return c


@pytest.fixture
def dev(con, settings):
    return FeatureStore(con, "开发", settings)


def order(table: pd.DataFrame) -> list[str]:
    return table["team"].tolist()


# ================= 积分榜：手算对照 =================
def test_table_after_round_3_by_hand(dev, settings):
    """第 3 轮后（2022-08-27）：A 7 分；C 6 分；T 4 分；F 0 分。"""
    t = standings_at(dev, "E0", 2022, "2022-08-27 00:00", settings=settings)
    assert order(t) == [A, C, T, F]
    row = t.set_index("team")
    assert row.loc[A, ["played", "won", "drawn", "lost", "goals_for", "goals_against", "points"]].tolist() == [3, 2, 1, 0, 6, 1, 7]
    assert row.loc[C, ["won", "lost", "goals_for", "goals_against", "points"]].tolist() == [2, 1, 3, 2, 6]
    assert row.loc[T, ["points", "goal_diff"]].tolist() == [4, 2]
    assert row.loc[F, ["points", "goal_diff", "remaining"]].tolist() == [0, -8, 3]  # 4 队双循环每队 6 场


def test_tiebreak_head_to_head_vs_goal_difference():
    """T 和 C 同分时：相互战绩 C 赢（1:0），净胜球 T 多。规则不同，排名就不同。"""
    res = pd.DataFrame([(C, T, 1, 0), (T, F, 4, 0), (C, F, 0, 1), (F, A, 0, 1), (A, C, 0, 0), (A, T, 0, 0)],
                       columns=["home_team", "away_team", "home_goals", "away_goals"])
    # T：负 C、胜 F 4:0、平 A → 4 分，净胜 +3；C：胜 T、负 F、平 A → 4 分，净胜 0；A 5 分；F 3 分
    h2h_first = LeagueRules("E0", 2022, tiebreak=["相互战绩积分", "净胜球", "进球"])
    gd_first = LeagueRules("E0", 2022, tiebreak=["净胜球", "进球"])
    teams = [A, T, C, F]
    t1 = compute_table(res, teams, h2h_first)
    t2 = compute_table(res, teams, gd_first)
    assert t1.set_index("team").loc[[T, C], "points"].tolist() == [4, 4]
    assert order(t1).index(C) < order(t1).index(T)   # 西甲 / 意甲：先比相互战绩 → C 在前
    assert order(t2).index(T) < order(t2).index(C)   # 英超：先比净胜球 → T 在前


def test_head_to_head_needs_all_played():
    """“相互战绩要踢完”：两队只踢了一场时跳过相互战绩，改比净胜球。"""
    # C 1:0 T、T 4:0 F → T、C 都 3 分；相互战绩 C 赢，净胜球 T 多（+3 对 +1）。T 和 C 只踢了一场（还差 T 主场）
    res = pd.DataFrame([(C, T, 1, 0), (T, F, 4, 0)],
                       columns=["home_team", "away_team", "home_goals", "away_goals"])
    teams = [A, T, C, F]
    loose = LeagueRules("E0", 2022, tiebreak=["相互战绩积分", "净胜球"])
    strict = LeagueRules("E0", 2022, tiebreak=["相互战绩积分", "净胜球"], h2h_needs_all_played=True)
    assert order(compute_table(res, teams, loose))[:2] == [C, T]
    assert order(compute_table(res, teams, strict))[:2] == [T, C]


def test_three_team_head_to_head_mini_league():
    """三队同分：只看这三队之间的“小联赛”。A 胜 T、T 胜 C、C 胜 A，三队相互战绩积分一样，
    再比相互战绩净胜球：A 3:0 T、T 1:0 C、C 1:0 A → A +2，T −2，C 0 → A、C、T。"""
    res = pd.DataFrame([(A, T, 3, 0), (T, C, 1, 0), (C, A, 1, 0)],
                       columns=["home_team", "away_team", "home_goals", "away_goals"])
    r = LeagueRules("E0", 2022, tiebreak=["相互战绩积分", "相互战绩净胜球", "净胜球"])
    assert order(compute_table(res, [A, T, C, F], r))[:3] == [A, C, T]


def test_unresolved_tie_is_flagged():
    """规则用完还分不开：按队名字母排，并标“没分开”。"""
    res = pd.DataFrame([(A, T, 1, 1)], columns=["home_team", "away_team", "home_goals", "away_goals"])
    t = compute_table(res, [T, A], LeagueRules("E0", 2022, tiebreak=["净胜球", "进球"]))
    assert order(t) == [A, T] and t["tie_unresolved"].all()


def test_two_points_for_a_win():
    """早年胜 2 分。"""
    res = pd.DataFrame([(A, T, 2, 0), (C, F, 1, 1)], columns=["home_team", "away_team", "home_goals", "away_goals"])
    t = compute_table(res, [A, T, C, F], LeagueRules("D1", 1993, win_points=2)).set_index("team")
    assert t.loc[A, "points"] == 2 and t.loc[C, "points"] == 1


# ================= 截止时间：只用赛前知道的比分 =================
def test_cutoff_before_and_after_result_known(dev, settings):
    """第 1 轮 14:00 开球，比分 17:00（开球 + 3 小时）才算知道。16:59 的积分榜里没有这一轮，17:00 就有。"""
    before = standings_at(dev, "E0", 2022, "2022-08-06 16:59", settings=settings)
    after = standings_at(dev, "E0", 2022, "2022-08-06 17:00", settings=settings)
    assert before["played"].sum() == 0 and before["points"].sum() == 0
    assert after["played"].sum() == 4 and order(after)[0] == C


def test_date_only_match_not_seen_same_day(dev, settings):
    """8 号比赛只有日期（2022-10-01）：最晚可能在当天 24:00（英国时间）开球，比分要到第二天凌晨才算知道。
    所以 10 月 1 日 23:00（UTC）时，同一天 14:00 开球的 7 号已经知道，8 号还不知道。"""
    t = standings_at(dev, "E0", 2022, "2022-10-01 23:00", settings=settings).set_index("team")
    assert t.loc[T, "played"] == 4      # 7 号 17:00 已知
    assert t.loc[A, "played"] == 3      # 8 号还不知道


def test_season_table_never_holds_future_scores(dev, settings):
    """SeasonTable 只拿到截止时间之前的比分：截止 2022-08-27 取的表，问 10 月也只有前 3 轮。"""
    st = season_table(dev, "E0", 2022, "2022-08-27 00:00", settings=settings)
    assert st.at("2022-12-31")["played"].sum() == 12


def test_dev_mode_cannot_read_exam_season(con, settings):
    """开发模式不能算期末考试赛季的积分榜（锁住考试赛季，和取数层一样）。"""
    con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                " status) values (99, 'E0', 2024, ?, ?, '2024-08-17', '2024-08-17 14:00', 'finished')", (A, T))
    with pytest.raises(SeasonLockedError):
        standings_at(FeatureStore(con, "开发", settings), "E0", 2024, "2024-09-01", settings=settings)


# ================= 扣分 =================
def write_deductions(settings, rows):
    with open(settings["standings"]["deductions"], "w", encoding="utf-8") as f:
        yaml.safe_dump({"扣分": rows}, f, allow_unicode=True)


def test_deduction_only_after_effective_date(dev, settings):
    """A 被扣 7 分，2022-08-15 宣布 → 从 8 月 16 日 0 点（UTC）起算。
    第 2 轮后：C 6 分、A 4 分、T 1 分、F 0 分。8 月 15 日 A 第 2；16 日 A 变成 −3 分、最后一名。"""
    write_deductions(settings, [{"联赛": "E0", "赛季": 2022, "球队": A, "分数": -7, "生效": "2022-08-15", "原因": "测试"}])
    ded = load_deductions(settings)
    assert ded["effective_utc"].iloc[0] == "2022-08-16 00:00:00"
    st = season_table(dev, "E0", 2022, "2022-12-31", settings=settings, deductions=ded)
    t15 = st.at("2022-08-15 12:00").set_index("team")
    t16 = st.at("2022-08-16 00:00").set_index("team")
    assert t15.loc[A, "points"] == 4 and t15.loc[A, "position"] == 2     # A 平 T、胜 F
    assert t16.loc[A, "points"] == -3 and t16.loc[A, "deduction"] == -7 and t16.loc[A, "position"] == 4


def test_deduction_with_appeal_refund(dev, settings):
    """扣 10 分、上诉退回 4 分：两条记录分别生效，最后合计 −6。"""
    write_deductions(settings, [
        {"联赛": "E0", "赛季": 2022, "球队": C, "分数": -10, "生效": "2022-08-10", "原因": "测试"},
        {"联赛": "E0", "赛季": 2022, "球队": C, "分数": 4, "生效": "2022-08-18", "原因": "上诉退回"}])
    st = season_table(dev, "E0", 2022, "2022-12-31", settings=settings, deductions=load_deductions(settings))
    assert st.at("2022-08-12")["deduction"].min() == -10
    assert st.final().set_index("team").loc[C, "deduction"] == -6


def test_deduction_season_end_only_in_final_table(dev, settings):
    """生效写“赛季结束”：赛季中间的积分榜不扣，最终积分榜扣。"""
    write_deductions(settings, [{"联赛": "E0", "赛季": 2022, "球队": F, "分数": -3, "生效": "赛季结束", "原因": "测试"}])
    st = season_table(dev, "E0", 2022, "2022-12-31", settings=settings, deductions=load_deductions(settings))
    assert st.at("2022-10-05").set_index("team").loc[F, "deduction"] == 0
    assert st.final().set_index("team").loc[F, "deduction"] == -3


def test_deduction_for_unknown_team_is_an_error(dev, settings):
    write_deductions(settings, [{"联赛": "E0", "赛季": 2022, "球队": "Liverpool", "分数": -3, "生效": "2022-09-01"}])
    with pytest.raises(ValueError, match="不在这个联赛"):
        season_table(dev, "E0", 2022, "2022-12-31", settings=settings, deductions=load_deductions(settings))


# ================= 德比 =================
def test_derby_lookup_both_directions(settings, tmp_path):
    p = tmp_path / "derbies.yaml"
    p.write_text(yaml.safe_dump({"德比": [
        {"名字": "北伦敦德比", "英文名": "North London derby", "类型": "同城", "球队": [A, T]},
        {"名字": "测试三队", "英文名": "x", "类型": "地区", "球队": ["Everton", "Liverpool", "No Such Team"]}]},
        allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="No Such Team"):      # 队名不在 teams.yaml
        L.load_derbies(settings, p)
    p.write_text(yaml.safe_dump({"德比": [
        {"名字": "北伦敦德比", "英文名": "North London derby", "类型": "同城", "球队": [A, T]},
        {"名字": "重复", "英文名": "x", "类型": "同城", "球队": [T, A]}]}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError):                         # 同一对写两次
        L.load_derbies(settings, p)
    d = L.load_derbies(settings)                            # 真实清单
    assert d[frozenset((A, T))]["name"] == d[frozenset((T, A))]["name"] == "北伦敦德比"
    assert frozenset((A, C)) not in d
    assert d[frozenset(("Real Madrid", "Barcelona"))]["type"] == "宿敌"


# ================= 争冠 / 保级的边界 =================
def table_of(points, remaining) -> pd.DataFrame:
    """按名次给一张积分榜（只要 race_status 用到的列）。"""
    return pd.DataFrame({"team": [f"t{i + 1}" for i in range(len(points))], "position": range(1, len(points) + 1),
                         "points": points, "remaining": remaining})


RULES = LeagueRules("E0", 2022, relegation=2, relegation_playoff=1)   # 危险区 = 最后 3 名


def test_title_race_gap_boundary():
    """第 2 名落后 6 分（= gap_points）算争冠，落后 7 分不算；第 1 名有人能追就算。"""
    r = L.race_status(table_of([60, 54, 40, 30, 20, 10], [5] * 6), RULES, last_rounds=10, gap_points=6)
    assert r["title_race"].tolist() == [True, True, False, False, False, False]
    r = L.race_status(table_of([60, 53, 40, 30, 20, 10], [5] * 6), RULES, last_rounds=10, gap_points=6)
    assert not r["title_race"].any()


def test_title_race_needs_enough_games_left():
    """落后 6 分但只剩 1 场（最多追 3 分）：数学上追不上，不算。剩 2 场能追 6 分，算。"""
    r = L.race_status(table_of([60, 54, 40, 30, 20, 10], [1, 1, 1, 1, 1, 1]), RULES, 10, 6)
    assert not r["title_race"].any()
    r = L.race_status(table_of([60, 54, 40, 30, 20, 10], [2] * 6), RULES, 10, 6)
    assert r["title_race"].tolist()[:2] == [True, True]


def test_race_only_in_last_rounds():
    """剩余场次（含本场）> last_rounds 时还不算“大战”。剩 10 场算，剩 11 场不算。"""
    pts = [60, 59, 40, 30, 29, 28]
    assert not L.race_status(table_of(pts, [11] * 6), RULES, 10, 6)[["title_race", "relegation_race"]].any().any()
    assert L.race_status(table_of(pts, [10] * 6), RULES, 10, 6)["title_race"].tolist()[:2] == [True, True]


def test_relegation_race_with_playoff_spot():
    """6 队、降 2 + 附加赛 1 → 危险区是第 4–6 名。安全区最后一名（第 3）是 t3。
    t3 40 分；t4 36（差 4，算）；t5 30（差 10，不算）；t6 20（不算）。t3 有人能追 → 算；t2 50 分，没人够得着 → 不算。"""
    r = L.race_status(table_of([60, 50, 40, 36, 30, 20], [4] * 6), RULES, 10, 6)
    assert r["relegation_race"].tolist() == [False, False, True, True, False, False]


def test_relegation_count_changes_danger_zone():
    """同一张表，降级名额不同，保级大战的队也不同（例如英超 1994/95 降 4 队）。"""
    pts, rem = [60, 50, 46, 36, 30, 20], [4] * 6
    one = L.race_status(table_of(pts, rem), LeagueRules("E0", 2022, relegation=1), 10, 6)
    four = L.race_status(table_of(pts, rem), LeagueRules("E0", 1994, relegation=4), 10, 6)
    # 降 1 队：危险区只有 t6（20 分），安全区最后一名 t5 30 分，差 10 → 没有保级大战
    assert not one["relegation_race"].any()
    # 降 4 队：危险区 t3–t6，安全区最后一名 t2 50 分；t3 46 分差 4 → t2、t3 在保级大战里
    assert four["relegation_race"].tolist() == [False, True, True, False, False, False]


# ================= 整季贴标签（经过取数层） =================
def test_label_season_end_to_end(con, settings):
    """last_rounds = 3：只有第 4 轮（每队剩 3 场）算“最后阶段”。第 3 轮后 A 7、C 6、T 4、F 0（降 1 队）。
    争冠：A（有人追得上）、C（差 1）、T（差 3）在；F（差 7）不在。
    保级：F 在降级区，追安全区最后一名 T 差 4 → 在；T 在；C 被 F 追差 6 → 在；A 差 7 → 不在。"""
    settings["labels"]["race"] = {"last_rounds": 3, "gap_points": 6}
    df = L.label_all(settings, ["E0"], [2022], conn=con, verbose=False).set_index("match_id")
    assert len(df) == 8 and (df["season_role"] == "dev").all()
    assert df.loc[1, "derby_name"] == "北伦敦德比" and df.loc[2, "is_derby"]       # C–F 是西伦敦德比
    assert df.index[df["is_derby"]].tolist() == [1, 2]                               # A–T、C–F；T–C、A–F 不是
    assert not df.loc[[1, 2, 3, 4, 5, 6], ["is_title_race", "is_relegation_race"]].any().any()
    assert df.loc[7, ["title_race_teams", "relegation_race_teams"]].tolist() == [2, 2]   # T 对 C
    assert df.loc[8, ["title_race_teams", "relegation_race_teams"]].tolist() == [1, 1]   # A 对 F
    assert df.loc[7, ["home_position", "away_position", "home_points", "away_points"]].tolist() == [3, 2, 4, 6]
    # 8 号只有日期：截止时间 = 最晚可能开球 = 10 月 2 日 0 点（英国夏令时）= 10 月 1 日 23:00 UTC
    assert df.loc[8, "standings_cutoff_utc"] == "2022-10-01 23:00:00"


def test_saved_labels_respect_season_lock(con, settings):
    """标签文件按运行模式检查：开发模式读到期末考试赛季的标签就报错。"""
    df = L.label_all(settings, ["E0"], [2022], conn=con, verbose=False)
    L.save_labels(df, settings)
    assert len(L.load_labels("开发", settings=settings)) == 8
    exam = df.assign(season=2024, season_role="exam", match_id=df["match_id"] + 100)
    L.save_labels(pd.concat([df, exam]), settings)
    with pytest.raises(SeasonLockedError):
        L.load_labels("开发", settings=settings)
    with pytest.raises(SeasonLockedError):          # 实时模式也不在考试赛季上复盘
        L.load_labels("实时", settings=settings)
    assert len(L.load_labels("考试", settings=settings)) == 16
    # 只挑开发赛季读，开发模式就可以；文件里写错的 season_role 不算数（按赛季年份重新判断）
    assert len(L.load_labels("开发", settings=settings, seasons=(1993, 2023))) == 8
    L.save_labels(pd.concat([df, exam.assign(season_role="dev")]), settings)
    with pytest.raises(SeasonLockedError):
        L.load_labels("开发", settings=settings)


# ================= 真实配置文件 =================
def test_real_rules_cover_every_league_and_season():
    """league_rules.yaml 要覆盖每个联赛的每个赛季（1993 起），同分项目都认识，危险区名额合理。"""
    s = load_settings()
    for comp in [c for g in s["leagues"].values() for c in g]:
        for season in range(1993, 2031):
            r = load_rules(comp, season, s)
            assert r.win_points in (2, 3) and 1 <= r.danger_spots <= 6 and r.source


def test_real_deductions_use_known_teams():
    """points_deductions.yaml 里的队名都在 teams.yaml 里，生效时间都能读懂。"""
    s = load_settings()
    ded = load_deductions(s)
    with open(resolve_path(s["paths"]["teams"]), encoding="utf-8") as f:
        known = {t["英文"] for t in yaml.safe_load(f)["球队"]}
    assert set(ded["team"]) <= known
    assert (ded["effective_utc"].str.len() == 19).all()
