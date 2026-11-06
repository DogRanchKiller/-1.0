"""历史基线（src/evaluate/baseline.py）的检查，全用编出来的数据，不读数据库。"""
import copy

import numpy as np
import pandas as pd

from src.common.config import load_settings
from src.common.markets import asian_handicap, over_under, poisson_score_matrix
from src.evaluate import baseline as bl
from src.evaluate.ledger import line_score

BOOK_PIN, BOOK_AVG = "平博 Pinnacle", "多家平均"


def _settings():
    s = copy.deepcopy(load_settings())
    s["compare"]["n_boot"] = 200
    return s


def _odds(mid, when, book, market, line=np.nan, home=np.nan, draw=np.nan, away=np.nan, over=np.nan, under=np.nan):
    return {"match_id": mid, "source": "football-data", "bookmaker": book, "market": market, "line": line,
            "home": home, "draw": draw, "away": away, "over": over, "under": under, "when": when}


def test_line_shares_same_as_markets():
    """一批场一起算的胜算，和 markets.asian_handicap / over_under 一场一场算的一样（含四分之一盘、走盘）。"""
    mats = np.stack([poisson_score_matrix(1.6, 1.1), poisson_score_matrix(0.9, 1.4), poisson_score_matrix(2.2, 0.7)])
    n = mats.shape[1] - 1
    i, j = np.indices(mats.shape[1:])
    diff = np.stack([(mats * (i - j == d)).sum(axis=(1, 2)) for d in range(-n, n + 1)], axis=1)
    for line in ([-0.75, 0.0, 0.25], [-1.0, -0.25, 0.5]):
        got = bl.line_shares(diff, np.arange(-n, n + 1), np.array(line), bl._ah_margin)
        want = [asian_handicap(m, ln, "home")["win_share"] for m, ln in zip(mats, line)]
        assert np.allclose(got, want)
    for line in ([2.5, 2.25, 3.0], [2.75, 2.5, 1.5]):
        got = bl.line_shares(bl.total_dist(mats), np.arange(2 * n + 1), np.array(line), bl._ou_margin)
        want = [over_under(m, ln, "over")["win_share"] for m, ln in zip(mats, line)]
        assert np.allclose(got, want)


def test_status_late_none_not_main():
    """比了 / 亚盘线不是主盘口 / 有赛前价、T-24 还没采到 / 没有这家的赛前价，各自认得对。"""
    s = _settings()
    o = pd.DataFrame([
        _odds(1, "当时", BOOK_PIN, "ah", -0.5, home=1.95, away=1.95),     # 主盘口 → 比了
        _odds(2, "当时", BOOK_PIN, "ah", 0.0, home=1.04, away=9.05),      # 赔率不在 1.5–2.7 → 不是主盘口
        _odds(3, "后来", BOOK_PIN, "ah", -0.25, home=1.9, away=2.0),      # 只有后来才有 → T-24 还没采到
    ])
    now = bl.market_tables(o[o["when"] == "当时"], s)[("亚盘让球", BOOK_PIN)]
    later = bl.market_tables(o[o["when"] == "后来"], s)[("亚盘让球", BOOK_PIN)]
    st = bl.status_of(np.array([1, 2, 3, 4]), now, later)
    assert list(st) == [bl.OK, bl.NOT_MAIN, bl.LATE, bl.NONE]
    assert np.isclose(now.set_index("match_id").loc[1, "share"], 0.5)


def _case():
    """三场：1 号平博全有；2 号胜平负只有多家平均、没有亚盘；3 号 T-24 什么都没采到（后来才有）。"""
    mats = np.stack([poisson_score_matrix(1.5, 1.0), poisson_score_matrix(1.2, 1.2), poisson_score_matrix(0.8, 1.7)])
    n = mats.shape[1] - 1
    i, j = np.indices(mats.shape[1:])
    p3 = np.stack([(mats * (i > j)).sum((1, 2)), (mats * (i == j)).sum((1, 2)), (mats * (i < j)).sum((1, 2))], 1)
    pred = pd.DataFrame({"match_id": [1, 2, 3], "competition": ["E0", "D1", "E0"], "season": [2020, 2010, 2021],
                         "match_date": ["2020-09-12", "2010-09-11", "2021-09-14"], "model": "gb_l7",
                         "horizon": "T-24", "result": ["H", "D", "A"], "home_goals": [2, 1, 0],
                         "away_goals": [0, 1, 1], "p_home": p3[:, 0], "p_draw": p3[:, 1], "p_away": p3[:, 2]})
    rows = [_odds(1, "当时", BOOK_PIN, "1x2", home=2.0, draw=3.4, away=4.0),
            _odds(1, "当时", BOOK_AVG, "1x2", home=1.9, draw=3.3, away=3.9),
            _odds(1, "当时", BOOK_AVG, "ou", 2.5, over=1.9, under=1.9),
            _odds(1, "当时", BOOK_PIN, "ou", 2.5, over=1.95, under=1.95),
            _odds(1, "当时", BOOK_PIN, "ah", -0.75, home=2.0, away=1.9),
            _odds(2, "当时", BOOK_AVG, "1x2", home=2.5, draw=3.1, away=2.9),
            _odds(2, "当时", BOOK_AVG, "ou", 2.5, over=2.0, under=1.8),
            _odds(3, "后来", BOOK_PIN, "1x2", home=4.0, draw=3.5, away=1.9)]
    rows += [{**r, "when": "后来"} for r in rows if r["when"] == "当时"]
    return pred, np.array([1, 2, 3]), mats.astype(np.float32), pd.DataFrame(rows), n


def test_build_rows_and_excluded():
    s = _settings()
    pred, ids, mats, odds, n = _case()
    rows, summary, excluded, check = bl.build(pred, ids, mats, odds, s)
    by = rows.groupby(["通道", "市场"])["match_id"].apply(list).to_dict()
    assert by[("胜平负", "平博优先")] == [1, 2] and by[("比分", "平博优先")] == [1, 2]
    assert by[("亚盘让球", BOOK_PIN)] == [1] and ("亚盘让球", BOOK_AVG) not in by
    assert by[("亚盘大小球", BOOK_PIN)] == [1] and by[("亚盘大小球", BOOK_AVG)] == [1, 2]
    # 胜平负：1 号用平博、2 号只有多家平均
    r = rows[rows["通道"] == "胜平负"].set_index("match_id")
    assert list(r["市场价来自"]) == ["平博", "多家平均"]
    pin = np.array([1 / 2.0, 1 / 3.4, 1 / 4.0])
    assert np.isclose(r.loc[1, "市场概率"], (pin / pin.sum())[0])
    assert r.loc[2, "结算"] == "平"
    # 亚盘：−0.75、2-0 主队赢；分数 = −ln(胜算)
    a = rows[(rows["通道"] == "亚盘让球")].iloc[0]
    share = asian_handicap(mats[0].astype(float) / mats[0].sum(), -0.75, "home")["win_share"]
    assert a["结算"] == "win" and np.isclose(a["模型概率"], share, atol=1e-6)
    assert np.isclose(a["市场分数"], line_score((1 / 2.0) / (1 / 2.0 + 1 / 1.9), "win", 1e-15))
    # 大小球：2 球是小，“大”那边输
    assert set(rows.loc[rows["通道"] == "亚盘大小球", "结算"]) == {"loss"}
    # 没比的：3 号后来才有胜平负 → T-24 还没采到；2、3 号没有平博亚盘
    e = excluded.groupby(["通道", "市场", "状态"])["场数"].sum().to_dict()
    assert e[("胜平负", "平博优先", bl.LATE)] == 1
    assert e[("比分", "平博优先", bl.NONE)] == 1          # 3 号后来也只有胜平负、没有大 2.5 球，还原不了市场矩阵
    assert e[("亚盘让球", BOOK_PIN, bl.NONE)] == 2 and e[("亚盘让球", BOOK_AVG, bl.NONE)] == 3
    assert check["胜平负（平博优先）"][bl.OK] == 2 and check["有预测、有比分的场"] == 3
    assert check["市场比分矩阵"]["还原误差（最大）"] < 1e-6
    # 差 = 模型 − 市场
    assert np.allclose(rows["差"], rows["模型分数"] - rows["市场分数"])


def test_summary_verdict_only_main_overall():
    """只有下结论的那一份的“全部”给结论；联赛、两段、多家平均那一份都“只报告”；没有场的那份只写场数 0。"""
    s = _settings()
    pred, ids, mats, odds, n = _case()
    rows, summary, _, _ = bl.build(pred, ids, mats, odds, s)
    main = summary[(summary["分法"] == "全部") & (summary["市场"] != BOOK_AVG) & (summary["场数"] > 0)]
    assert set(main["结论"]) <= {"更准", "没差别", "更差"}
    rest = summary[((summary["分法"] != "全部") | (summary["市场"] == BOOK_AVG)) & (summary["场数"] > 0)]
    assert set(rest["结论"]) == {"只报告"}
    empty = summary[(summary["通道"] == "亚盘让球") & (summary["市场"] == BOOK_AVG)]
    assert list(empty["场数"]) == [0]
    # 每场的差都 > 0 时结论是“更差”
    worse = rows.assign(差=0.1, 模型分数=1.1, 市场分数=1.0)
    w = bl.summarize(worse, s)
    assert w.loc[(w["通道"] == "胜平负") & (w["分法"] == "全部"), "结论"].item() == "更差"
