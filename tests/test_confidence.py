"""定信心等级规则（需求文档第 5 节“信心等级规则怎么定”）。"""
import numpy as np
import pandas as pd

from src.common.config import load_settings
from src.evaluate.confidence_study import group_diff_ci, levels, promoted_flags, run

SETTINGS = load_settings()


def _norm(p):
    p = np.asarray(p, float)
    return p / p.sum()


def _fake(seasons=(2012, 2013, 2014, 2015, 2016), n_teams=20, seed=3):
    """造一次假回测：一个联赛，每季换 1 个队（升班马）。收盘 = 真概率；赛前在前 10 轮被压平（d 大），后面和收盘一样。
    模型 = 赛前，只有每 7 场往主胜挪 0.10（分歧大，但 d 不变）。每 11 场没有平博（只有多家平均赛前），
    有平博的比赛多家平均赛前 = 平博赛前（两家一样准）。第一场没有赔率。"""
    rng = np.random.default_rng(seed)
    rows, odds, mid = [], [], 0
    for si, s in enumerate(seasons):
        teams = [f"T{k}" for k in range(si, si + n_teams)]          # 每季最弱的走、新来一个（升班马）
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        for k, (h, a) in enumerate(fixtures):
            mid += 1
            true = _norm(rng.dirichlet([4, 2.5, 3]))
            res = "HDA"[rng.choice(3, p=true)]
            early = k < n_teams // 2 * 10
            pre = 0.4 * true + 0.6 / 3 if early else true
            model = pre + (np.array([0.10, -0.05, -0.05]) if mid % 7 == 0 else 0)
            model = _norm(np.clip(model, 0.01, None))
            date = pd.Timestamp(f"{s}-08-01") + pd.Timedelta(days=k // 4)
            base = {"match_id": mid, "competition": "E0", "season": s, "match_date": date.strftime("%Y-%m-%d"),
                    "kickoff_utc": (date + pd.Timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "home_team": h, "away_team": a, "result": res}
            rows.append({**base, "model": "gb_l7", "horizon": "T-24", "p_home": model[0], "p_draw": model[1],
                         "p_away": model[2]})
            pin = mid % 11 != 0
            if pin:
                rows.append({**base, "model": "pinnacle_close", "horizon": "收盘", "p_home": true[0], "p_draw": true[1],
                             "p_away": true[2]})
            rows.append({**base, "model": "market_avg", "horizon": "赛前代替", "p_home": pre[0], "p_draw": pre[1],
                         "p_away": pre[2]})
            odds.append({"match_id": mid, "mo_p_home": pre[0], "mo_p_draw": pre[1], "mo_p_away": pre[2],
                         "mo_over25": 0.5, "mo_pinnacle": float(pin)})
    odds[0].update(mo_p_home=np.nan, mo_p_draw=np.nan, mo_p_away=np.nan)
    return pd.DataFrame(rows), pd.DataFrame(odds)


def test_promoted_flags():
    m = pd.DataFrame({"match_id": [1, 2, 3, 4], "competition": ["E0"] * 4, "season": [2012, 2012, 2013, 2013],
                      "home_team": ["A", "B", "A", "C"], "away_team": ["B", "A", "B", "A"]})
    f = promoted_flags(m)
    assert f.tolist() == [False, False, False, True]      # 2012 没有上一季 → 不算；2013 的 C 上季不在


def test_group_diff_ci_sign_and_interval():
    rng = np.random.default_rng(0)
    flag = np.arange(2000) % 4 == 0
    value = rng.normal(0, 0.1, 2000) + 0.05 * flag
    dates = pd.Series(pd.date_range("2015-01-01", periods=2000, freq="D"))
    diff, lo, hi = group_diff_ci(value, flag, dates.dt.strftime("%Y-%W").to_numpy(), SETTINGS)
    assert lo < diff < hi and lo > 0.02 and abs(diff - (value[flag].mean() - value[~flag].mean())) < 1e-12


def test_levels():
    t = pd.DataFrame({"early": [False, True, True, False], "promoted": [False, True, False, False],
                      "has_odds": [True, True, False, False]})
    assert levels(t, ["early", "promoted"]).tolist() == [3, 1, 1, 2]
    assert levels(t, []).tolist() == [3, 3, 2, 2]


def test_run_end_to_end():
    pred, odds = _fake()
    res = run(pred, odds, SETTINGS)
    c, ft, t = res["check"], res["factors"], res["table"]
    n = 5 * 380
    assert c["有预测和结果的比赛"] == n and c["没有赔率（只出模型概率）"] == 1
    assert c["能算 d 的比赛（平博赛前 + 平博收盘）"] == n - n // 11 - 1
    assert c["两家都有赛前的比赛"] == n - n // 11 - 1
    assert c["各因素场数（全部比赛）"]["赛季前几轮"] == 5 * 100
    full = ft[ft["哪一份"] == "全部"].set_index("因素")
    assert full.loc["赛季前几轮", "算数"]                         # 前 10 轮赛前被压平：d 明显大
    assert not full.loc["没有平博", "算数"]                       # 两家一样准：差 0
    assert not full.loc["模型和赔率分歧大", "算数"]               # 分歧只在模型那边，赔率的 d 不变
    assert "赛季前几轮" in c["算数的因素"]
    assert (t.loc[t["early"] & t["has_odds"], "level"] <= 2).all()
    assert t.loc[~t["has_odds"], "level"].iloc[0] <= 2
    assert set(ft["哪一份"]) == {"全部", "2014 之前", "2014 起"}
    assert res["levels"]["场数"].sum() == n
