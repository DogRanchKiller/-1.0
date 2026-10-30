"""去水方法比较（src/evaluate/devig_compare.py）的自动检查。

1. 不用数据库：造“庄家按幂函数加抽水”的赔率，幂函数法去水应该正好还原真实概率、分数最好。
2. 内存里的临时数据库：检查取数（每组赔率从它的起始赛季开始、缺赔率和没比分的不要、考试赛季拿不到）和输出的表。
"""
import copy
import itertools
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.evaluate import devig_compare as dc
from src.features import FeatureStore


def test_power_margin_is_undone_by_power_devig():
    """真实概率 p，庄家开出的隐含概率 q = p 的 0.9 次方（加起来大于 1，冷门抽得多）。
    幂函数法去水正好还原 p；按实际结果打分，幂函数法的平均 log loss 也比按比例低。"""
    rng = np.random.default_rng(7)
    n = 40000
    p = rng.dirichlet([4, 2.5, 3], size=n)
    odds = 1 / p ** 0.9
    y = np.array([rng.choice(3, p=row) for row in p])
    data = pd.DataFrame({"match_id": np.arange(n), "competition": "E0", "season": 2020,
                         "match_date": pd.date_range("2015-01-01", periods=n, freq="6h").strftime("%Y-%m-%d"),
                         "result": np.array(["H", "D", "A"])[y],
                         "x_home": odds[:, 0], "x_draw": odds[:, 1], "x_away": odds[:, 2]})
    s = dc.score_methods(data, "x", ["proportional", "shin", "power"])
    power = s[s["model"] == "power"]
    np.testing.assert_allclose(power[["p_home", "p_draw", "p_away"]].to_numpy(), p, atol=1e-9)
    mean = s.groupby("model")["log_loss"].mean()
    assert mean["power"] < mean["proportional"]
    assert len(s) == 3 * n and set(s["source"]) == {"x"}


# ---------------- 临时数据库 ----------------
TEAMS = list("ABCDEFGH")


def build_db():
    """E0，2018/19–2023/24 每季 8 个队、56 场（每对主客各一场），每天一场。
    平博收盘：每场都有。多家平均收盘：每场都有（但 2018/19 早于 2019 起始赛季，不能用）。
    每季最后一场没有比分（改期）；另有 1 场 2024/25 考试赛季的比赛。"""
    matches, stats, odds = [], [], []
    rng = np.random.default_rng(1)
    mid = 0
    for season in range(2018, 2024):
        day = pd.Timestamp(f"{season}-08-10")
        pairs = list(itertools.permutations(TEAMS, 2))
        for k, (h, a) in enumerate(pairs):
            mid += 1
            last = k == len(pairs) - 1
            matches.append((mid, "E0", season, h, a, (day + pd.Timedelta(days=k)).strftime("%Y-%m-%d"),
                            "postponed" if last else "finished"))
            if not last:
                r = rng.choice(["H", "D", "A"], p=[0.45, 0.27, 0.28])
                stats.append((mid, {"H": 1, "D": 0, "A": 0}[r], {"H": 0, "D": 0, "A": 1}[r], r))
            odds.append((mid, "平博 Pinnacle", 1.6, 4.0, 6.0))
            odds.append((mid, "多家平均", 1.55, 3.9, 5.8))
    matches.append((9999, "E0", 2024, "A", "B", "2024-08-17", "finished"))
    stats.append((9999, 1, 0, "H"))
    odds.append((9999, "平博 Pinnacle", 2.0, 3.5, 4.0))
    return matches, stats, odds


@pytest.fixture
def setup():
    s = copy.deepcopy(load_settings())
    s["devig_compare"]["leagues"] = ["top"]
    s["compare"]["n_boot"] = 200
    c = sqlite3.connect(":memory:")
    create_tables(c)
    matches, stats, odds = build_db()
    c.executemany("insert into teams (team_id, country) values (?, 'England')", [(t,) for t in TEAMS])
    c.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, status)"
                  " values (?, ?, ?, ?, ?, ?, ?)", matches)
    c.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                  " values (?, 'football-data', ?, ?, ?)", stats)
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away)"
                  " values (?, 'football-data', ?, '1x2', 'close', null, ?, ?, ?)", odds)
    c.commit()
    return FeatureStore(c, "开发", s), s


def test_load_data_respects_sources_and_exam_lock(setup):
    fs, s = setup
    d = dc.load_data(fs, s)
    assert 9999 not in set(d["match_id"])                       # 考试赛季拿不到
    assert d["season"].between(2018, 2023).all()
    assert len(d) == 6 * 55                                     # 每季 56 场，最后一场没比分
    pin = d.dropna(subset=["pinnacle_close_home"])
    avg = d.dropna(subset=["avg_close_home"])
    assert len(pin) == 6 * 55
    assert avg["season"].min() == 2019 and len(avg) == 5 * 55   # 多家平均 2019/20 起


def test_run_outputs(setup):
    fs, s = setup
    res = dc.run(fs, s)
    comp = res["compare"]
    assert len(comp) == 2 * 3 * 3                               # 2 组赔率 × 3 对 × 3 个分数
    assert set(comp["source"]) == {"pinnacle_close", "avg_close"}
    assert set(zip(comp["base"], comp["new"])) == {("proportional@收盘", "shin@收盘"),
                                                    ("proportional@收盘", "power@收盘"), ("shin@收盘", "power@收盘")}
    assert set(comp["verdict"]) <= {"更准", "没差别", "更差"}
    assert res["info"]["每组赔率的场数"] == {"avg_close": 275, "pinnacle_close": 330}
    assert set(res["by_tier"]["tier"]) == {"五大联赛"}
    ls = res["longshot"]
    assert set(ls["source"]) == {"pinnacle_close", "avg_close"} and (ls["n_outcomes"] > 0).all()
    assert dc.dev_cutoff(s) == f"{s['seasons']['dev_last'] + 1}-07-01 00:00:00"
