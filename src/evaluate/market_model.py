"""新模型：只看赛前十几家公司的价格，学出比平博更准的胜平负概率（研究用，只读数据库）。

为什么这样做：旧模型（看球队成绩）比盘口差；最准的信息就在价格里。这个模型学的是
“哪家公司在什么联赛、什么价位更准，几家价格不一样时该信谁”。
输入（都是开赛前几天、同一时刻记下的价格，football-data 的“赛前”列）：
  每家公司胜平负去水后的三个概率和抽水；平博、多家平均的亚盘（线 + 主队那边的胜算）、大小球 2.5 的胜算；
  多家最高价格；联赛。缺的家留空（LightGBM 会自己处理）。
不用：收盘价格（开赛那一刻才有）、比分、任何赛后的东西。

事先定的规则（跑之前写下，跑完不改）：
  - 训练 2012–2019，调参（早停）2020–2021，考试 2022–2025，考试的季只算一次。
  - 准不准：每场 log loss（越小越准），和平博赛前去水（Shin）比；按周自助法 95% 区间，整个 < 0 才算“比平博准”。
  - 赚不赚：EV = 价格 × 模型概率 − 1，每场挑 EV 最大一边，EV ≥ 门槛买 1 份；看 ROI 和 CLV（对平博收盘）。
    买的价格分三种：多家最高、Interwetten、竞彩最终价（竞彩只有 2019 以后五大联赛）。
  - 同样的买法用平博赛前概率代替模型再算一次：模型要比它好，才说明模型有用。

运行：.venv/Scripts/python -m src.evaluate.market_model
"""
from __future__ import annotations

import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.odds import devig

PIN = "平博 Pinnacle"
SIDES = ["home", "draw", "away"]
BOOKS = {PIN: "pin", "Bet365": "b365", "William Hill": "wh", "Interwetten": "iw", "Bet&Win": "bw",
         "VC Bet（后来的 BetVictor）": "vc", "Ladbrokes": "lb", "Betfair 交易所": "bfx", "多家平均": "avg",
         "多家最高": "max"}
TRAIN, VALID, TEST = (2012, 2019), (2020, 2021), (2022, 2025)
THRESHOLDS = [0.0, 0.02, 0.05]
SEED, N_BOOT = 20261107, 1000
PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.02, "num_leaves": 15,
          "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 5.0, "verbose": -1, "seed": SEED}


def _shin(odds: np.ndarray) -> np.ndarray:
    """一批三个价格 → Shin 去水概率（每行一场；价格不全的行是空）。"""
    out = np.full(odds.shape, np.nan)
    ok = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    out[ok] = np.vstack([devig(o, method="shin") for o in odds[ok]])
    return out


def load(db: str) -> pd.DataFrame:
    """一场一行：各家赛前价格算出的输入、平博赛前/收盘 Shin 概率、能买的价格、比分。只要平博赛前有胜平负的场。"""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    o = pd.read_sql("""select match_id, bookmaker, market, timing, line, home, draw, away, over, under from odds
                       where timing in ('pre', 'close')""", con)
    res = pd.read_sql("""select m.match_id, m.competition, m.season, m.match_date, s.home_goals, s.away_goals
                         from matches m join match_stats s on s.match_id = m.match_id
                         where m.status = 'finished' and s.source in ('football-data', 'football-data-new')
                           and s.home_goals is not null""", con).drop_duplicates("match_id")
    lot = pd.read_sql("""select l.match_id, l.home lot_home, l.draw lot_draw, l.away lot_away from lottery_odds l
                         join (select match_id, max(rowid) r from lottery_odds where product = '竞彩胜平负'
                               group by 1) x on l.rowid = x.r where l.handicap = 0""", con)
    con.close()
    d = res.set_index("match_id")
    x12 = o[(o.market == "1x2") & (o.timing == "pre") & o.bookmaker.isin(BOOKS)]
    for book, k in BOOKS.items():
        b = x12[x12.bookmaker == book].drop_duplicates("match_id", keep="last").set_index("match_id")
        price = b[SIDES].where(b[SIDES] > 1)
        inv = 1 / price
        d = d.join(pd.DataFrame({f"{k}_o_{s}": price[s] for s in SIDES}))
        d = d.join(pd.DataFrame({**{f"{k}_p_{s}": inv[s] / inv.sum(axis=1) for s in SIDES},
                                 f"{k}_vig": inv.sum(axis=1) - 1}))
    for book, k in [(PIN, "pin"), ("多家平均", "avg")]:
        for mk, a, b_ in [("ah", "home", "away"), ("ou", "over", "under")]:
            r = o[(o.bookmaker == book) & (o.market == mk) & (o.timing == "pre")]
            r = r.drop_duplicates("match_id", keep="last").set_index("match_id")
            ok = (r[a] > 1) & (r[b_] > 1)
            share = (1 / r[a]) / (1 / r[a] + 1 / r[b_])
            d = d.join(pd.DataFrame({f"{k}_{mk}_line": r.line.where(ok), f"{k}_{mk}_share": share.where(ok)}))
    d = d[d[[f"pin_o_{s}" for s in SIDES]].notna().all(axis=1)].copy()
    pc = o[(o.bookmaker == PIN) & (o.market == "1x2") & (o.timing == "close")].drop_duplicates("match_id", keep="last")
    pc = pc.set_index("match_id")[SIDES].reindex(d.index).to_numpy(float)
    pin_pre = _shin(d[[f"pin_o_{s}" for s in SIDES]].to_numpy(float))
    pin_close = _shin(pc)
    for i, s in enumerate(SIDES):
        d[f"base_{s}"], d[f"close_{s}"] = pin_pre[:, i], pin_close[:, i]
    d = d.join(lot.set_index("match_id"))
    d["y"] = np.sign(d.home_goals - d.away_goals).map({1: 0, 0: 1, -1: 2}).astype(int)
    d["week"] = pd.to_datetime(d.match_date).dt.to_period("W").astype(str)
    d["league"] = d.competition.astype("category")
    return d.reset_index()


def features(d: pd.DataFrame) -> list[str]:
    """模型输入的列：各家去水概率、抽水、亚盘大小球、联赛。不含价格本身、收盘、比分、竞彩。"""
    keep = [c for c in d.columns if any(c.startswith(f"{k}_") for k in BOOKS.values()) and "_o_" not in c]
    return keep + ["league"]


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """每场的 log loss（真实结果那一格概率的 −log）。"""
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def boot_ratio(week: pd.Series, num: np.ndarray, den: np.ndarray, rng) -> tuple[float, float]:
    """按周成块自助法：sum(num) / sum(den) 的 95% 区间。"""
    g = pd.DataFrame({"w": week.to_numpy(), "n": num, "d": den}).groupby("w")[["n", "d"]].sum().to_numpy()
    if len(g) < 10:
        return np.nan, np.nan
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    s = g[idx, 0].sum(axis=1) / g[idx, 1].sum(axis=1)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def train(d: pd.DataFrame, cols: list[str]) -> tuple[lgb.Booster, np.ndarray]:
    """训练季学、调参季早停；返回模型和所有场的预测概率（训练季的是学过的，只看考试季）。"""
    tr = d.season.between(*TRAIN)
    va = d.season.between(*VALID)
    ds = lgb.Dataset(d.loc[tr, cols], d.loc[tr, "y"], categorical_feature=["league"])
    dv = lgb.Dataset(d.loc[va, cols], d.loc[va, "y"], reference=ds)
    booster = lgb.train(PARAMS, ds, num_boost_round=3000, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster, booster.predict(d[cols], num_iteration=booster.best_iteration)


def residual_check(d: pd.DataFrame, cols: list[str], rng) -> pd.DataFrame:
    """第二个版本（第一次考试之后才想到，所以不碰考试季）：从平博赛前概率出发，只学“平博哪里偏了”。

    训练 2012–2017，早停 2018–2019，检验 2020–2021（原来的调参季）。检验季不比平博准，就不再往下做。
    """
    def split(lo, hi):
        m = d.season.between(lo, hi).to_numpy()
        return d.loc[m, cols], d.loc[m, "y"].to_numpy(), np.log(d.loc[m, [f"base_{s}" for s in SIDES]].to_numpy(float)), d.loc[m, "week"]
    (xt, yt, it, _), (xv, yv, iv, _), (xc, yc, ic, wc) = split(2012, 2017), split(2018, 2019), split(*VALID)
    ds = lgb.Dataset(xt, yt, init_score=it, categorical_feature=["league"])
    dv = lgb.Dataset(xv, yv, init_score=iv, reference=ds)
    params = {**PARAMS, "learning_rate": 0.01}
    booster = lgb.train(params, ds, num_boost_round=3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    z = booster.predict(xc, num_iteration=booster.best_iteration, raw_score=True) + ic
    pr = np.exp(z - z.max(axis=1, keepdims=True))
    pr /= pr.sum(axis=1, keepdims=True)
    diff = logloss(pr, yc) - logloss(np.exp(ic), yc)
    lo, hi = boot_ratio(wc, diff, np.ones(len(diff)), rng)
    return pd.DataFrame([{"比的是": "从平博出发的版本 − 平博赛前（2020–2021）", "场数": len(diff),
                          "早停轮数": booster.best_iteration, "差": diff.mean(), "差下限": lo, "差上限": hi,
                          "结论": "比平博准" if hi < 0 else ("比平博差" if lo > 0 else "看不出差别")}])


def betting(t: pd.DataFrame, probs: dict[str, np.ndarray], rng) -> pd.DataFrame:
    """考试季：每种概率 × 每种能买的价格 × 门槛 → 注数、ROI（区间）、CLV（区间）、命中率、平均价格。"""
    sources = {"多家最高": "max_o", "Interwetten": "iw_o", "竞彩最终价": "lot"}
    close = t[[f"close_{s}" for s in SIDES]].to_numpy(float)
    rows = []
    for pname, p in probs.items():
        for sname, pre in sources.items():
            price = t[[f"{pre}_{s}" for s in SIDES]].to_numpy(float)
            ev = np.where(np.isfinite(price), price * p - 1, -np.inf)
            side = ev.argmax(axis=1)
            r = np.arange(len(t))
            best, px = ev[r, side], price[r, side]
            profit = np.where(t.y.to_numpy() == side, px - 1, -1.0)
            clv = px * close[r, side] - 1
            for th in THRESHOLDS:
                m = np.isfinite(best) & (best >= th)
                mc = m & np.isfinite(clv)
                if m.sum() < 30:
                    continue
                roi_lo, roi_hi = boot_ratio(t.week[m], profit[m], np.ones(m.sum()), rng)
                clv_lo, clv_hi = boot_ratio(t.week[mc], clv[mc], np.ones(mc.sum()), rng)
                rows.append({"概率": pname, "价格": sname, "门槛": f"EV ≥ {th:.2f}", "注数": int(m.sum()),
                             "ROI": profit[m].mean(), "ROI下限": roi_lo, "ROI上限": roi_hi,
                             "CLV": clv[mc].mean(), "CLV下限": clv_lo, "CLV上限": clv_hi,
                             "命中率": (profit[m] > 0).mean(), "平均价格": px[m].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    s = load_settings()
    out = resolve_path("data/backtest/market_model")             # 结果不进 Git
    out.mkdir(parents=True, exist_ok=True)
    d = load(str(resolve_path(s["paths"]["database"])))
    cols = features(d)
    booster, p = train(d, cols)
    rng = np.random.default_rng(SEED)
    t_mask = d.season.between(*TEST).to_numpy()
    t, pm = d[t_mask].reset_index(drop=True), p[t_mask]
    pb = t[[f"base_{s}" for s in SIDES]].to_numpy(float)
    pa = t[[f"avg_p_{s}" for s in SIDES]].to_numpy(float)
    ok = np.isfinite(pa).all(axis=1)
    y = t.y.to_numpy()
    print(f"场数：训练 {d.season.between(*TRAIN).sum()}，调参 {d.season.between(*VALID).sum()}，考试 {len(t)}；"
          f"输入 {len(cols)} 列；早停在第 {booster.best_iteration} 轮")

    ll_m, ll_b, ll_a = logloss(pm, y), logloss(pb, y), logloss(np.where(ok[:, None], pa, pb), y)
    rows = []
    for name, ll in [("新模型", ll_m), ("多家平均（按比例去水）", ll_a)]:
        lo, hi = boot_ratio(t.week, ll - ll_b, np.ones(len(t)), rng)
        rows.append({"比的是": f"{name} − 平博赛前", "log loss": ll.mean(), "平博 log loss": ll_b.mean(),
                     "差": (ll - ll_b).mean(), "差下限": lo, "差上限": hi,
                     "结论": "比平博准" if hi < 0 else ("比平博差" if lo > 0 else "看不出差别")})
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "accuracy.csv", index=False, encoding="utf-8-sig")
    print("\n== 准不准（考试季 2022–2025，差 < 0 = 比平博准）==")
    print(acc.round(5).to_string(index=False))
    per = pd.DataFrame({"season": t.season, "差": ll_m - ll_b}).groupby("season")["差"].agg(["mean", "size"])
    print("\n每季 新模型 − 平博：\n" + per.round(5).to_string())

    bet = betting(t, {"新模型": pm, "平博赛前": pb}, rng)
    bet.to_csv(out / "betting.csv", index=False, encoding="utf-8-sig")
    print("\n== 赚不赚（考试季）==")
    print(bet.round(4).to_string(index=False))

    imp = pd.Series(booster.feature_importance("gain"), index=cols).sort_values(ascending=False)
    imp.to_csv(out / "importance.csv", encoding="utf-8-sig")
    print("\n最有用的 10 个输入：\n" + (imp / imp.sum()).head(10).round(3).to_string())
    print(f"\n输出：{out}")


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 200)
    if "--residual" in sys.argv:          # 只跑第二个版本的检验（不碰考试季）
        st = load_settings()
        dd = load(str(resolve_path(st["paths"]["database"])))
        r = residual_check(dd, features(dd), np.random.default_rng(SEED))
        r.to_csv(resolve_path("data/backtest/market_model") / "residual_check.csv", index=False, encoding="utf-8-sig")
        print(r.round(5).to_string(index=False))
    else:
        main()
