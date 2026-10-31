"""src/clean/understat_match.py 的自动检查：射门明细加成每场（乌龙球算给对面、点球不算进 npxG），队名核对，和我们的比赛对上。

用几行造的射门明细，不读真实文件、不碰数据库。
"""
import pandas as pd
import pytest

from src.clean.understat_match import align, check_aliases, decide, match_totals, vote


def shot(mid, h_a, result, xg, situation="OpenPlay", home="Home FC", away="Away FC", hg=2, ag=1,
         date="2020-01-04 19:00:00", comp="E0", season=2019):
    return {"match_id": float(mid), "h_a": h_a, "result": result, "xG": xg, "situation": situation,
            "h_team": home, "a_team": away, "h_goals": float(hg), "a_goals": float(ag), "date": date,
            "competition": comp, "file_season": season}


@pytest.fixture
def shots():
    return pd.DataFrame([
        # 1 号：主队 2:1。主队一个进球 + 一个点球进球 + 一个被扑；客队一个乌龙球（记在主队球员名下，h_a = h）、一个射偏
        shot(1, "h", "Goal", 0.4),
        shot(1, "h", "Goal", 0.76, situation="Penalty"),
        shot(1, "h", "SavedShot", 0.1),
        shot(1, "h", "OwnGoal", 0.0),          # 主队球员踢进自家球门 → 算客队的进球
        shot(1, "a", "MissedShot", 0.05),
        shot(1, "a", "ShotOnPost", 0.2),
        # 2 号：0:0，只有客队射过门
        shot(2, "a", "BlockedShot", 0.03, hg=0, ag=0, date="2020-01-05 23:30:00"),
        # 3 号：文件写 1:0，但射门明细里没有进球（明细不全）
        shot(3, "h", "MissedShot", 0.1, hg=1, ag=0, home="Away FC", away="Home FC", date="2020-02-01 15:00:00"),
    ])


def test_match_totals(shots):
    t = match_totals(shots).set_index("understat_id")
    one = t.loc[1]
    assert (one["home_goals"], one["away_goals"]) == (2, 1)          # 乌龙球算给客队
    assert one["home_xg"] == pytest.approx(1.26) and one["home_npxg"] == pytest.approx(0.5)
    assert (one["home_shots"], one["away_shots"]) == (3, 2)          # 乌龙球不算射门
    assert (one["home_sot"], one["away_sot"]) == (3, 0)              # 门柱不算射正
    assert (one["home_pens"], one["away_pens"]) == (1, 0)
    assert one["shots_add_up"]
    two = t.loc[2]
    assert (two["home_shots"], two["home_xg"], two["away_shots"]) == (0, 0, 1)   # 没射过门的一边是 0
    assert not t.loc[3, "shots_add_up"]


def db_frame(rows):
    df = pd.DataFrame(rows, columns=["match_id", "competition", "season", "match_date", "home_team", "away_team",
                                     "home_goals", "away_goals"])
    df["day"] = pd.to_datetime(df["match_date"])
    return df


def test_vote_and_align(shots):
    us = match_totals(shots)
    db = db_frame([(11, "E0", 2019, "2020-01-04", "Home", "Away", 2, 1),
                   (12, "E0", 2019, "2020-01-06", "Home", "Away", 0, 0),    # Understat 日期晚了 1 天写成 05 号：可以
                   (13, "E0", 2019, "2020-02-01", "Away", "Home", 1, 0)])
    alias = {"Home FC": "Home", "Away FC": "Away"}
    # 一个赛季同一对主客队只踢一次，这里 1 号、2 号是同一对，测试里用不同赛季区分
    us.loc[us["understat_id"] == 2, "season"] = 2018
    db.loc[db["match_id"] == 12, "season"] = 2018
    got = align(us, db, alias).set_index("understat_id")
    assert got.loc[1, "结果"] == "对上" and got.loc[1, "match_id"] == 11
    assert got.loc[2, "结果"] == "对上" and got.loc[2, "date_gap_days"] == 1
    assert got.loc[3, "结果"] == "射门明细加起来和比分不一致"
    decided, _ = decide(vote(us, db))
    assert decided == {("E0", "Home FC"): "Home", ("E0", "Away FC"): "Away"}   # 每个名字 3 票，投出来的就是对的


def test_align_flags_score_and_date(shots):
    us = match_totals(shots)
    us = us[us["understat_id"] == 1]
    alias = {"Home FC": "Home", "Away FC": "Away"}
    wrong_score = db_frame([(11, "E0", 2019, "2020-01-04", "Home", "Away", 3, 0)])  # 例如判罚改了比分
    assert align(us, wrong_score, alias)["结果"].item() == "比分不同"
    far = db_frame([(11, "E0", 2019, "2020-01-15", "Home", "Away", 2, 1)])           # 例如改期
    assert align(us, far, alias)["结果"].item() == "日期差 11 天"
    assert align(us, far.iloc[0:0], alias)["结果"].item() == "数据库里没有这场"


def test_two_understat_matches_cannot_share_one_of_ours(shots):
    us = match_totals(shots)
    one = us[us["understat_id"] == 1]
    twice = pd.concat([one, one.assign(understat_id=99)], ignore_index=True)
    db = db_frame([(11, "E0", 2019, "2020-01-04", "Home", "Away", 2, 1)])
    with pytest.raises(ValueError, match="同一场"):
        align(twice, db, {"Home FC": "Home", "Away FC": "Away"})


def test_check_aliases():
    names = {("E0", "Home FC"), ("E0", "Away FC")}
    check_aliases({"Home FC": "Home", "Away FC": "Away"}, {("E0", "Home FC"): "Home"}, names)
    with pytest.raises(ValueError, match="不在 team_aliases_understat"):
        check_aliases({"Home FC": "Home"}, {}, names)
    with pytest.raises(ValueError, match="不一致"):
        check_aliases({"Home FC": "Away", "Away FC": "Away"}, {("E0", "Home FC"): "Home"}, names)


def test_alias_file_is_valid():
    """真实的对照表：队名都在 teams.yaml 里、没有写两次。"""
    from src.clean.understat_match import load_aliases
    from src.common.config import load_settings
    alias = load_aliases(load_settings())
    assert alias["Manchester United"] == "Man United" and alias["Parma Calcio 1913"] == "Parma"
