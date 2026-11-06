"""价格偏离历史回测（研究用，只读数据库，不改任何表）。

问题：别的公司（或竞彩）给的价格比平博公道价高的时候买，长期赚不赚？
平博是公认最准的公司，它去水后的概率当“公道概率”。EV = 别家价格 × 公道概率 − 1。

两组检验（胜平负，每场每家只挑 EV 最大的一边，EV ≥ 门槛买 1 份，按真实比分结算）：
  A. 各家赛前价格 vs 平博赛前公道价：football-data 的“赛前”价格是同一时刻（比赛前几天）记下的，当时能买到。
     另算 CLV = 买的价格 × 平博收盘公道概率 − 1（收盘价最准，CLV 长期为正是“真的有优势”最可靠的信号）。
  B. 竞彩最终价格 vs 平博收盘公道价：最乐观（平博收盘在开球那一刻，竞彩停售更早），这里都亏 → 竞彩这条路走不通。
区间：按周成块自助法（同一周的注一起抽），95%。

运行：.venv/Scripts/python -m src.evaluate.value_history
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.odds import devig

PIN = "平博 Pinnacle"
SIDES = ["home", "draw", "away"]
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]
SKIP_BOOKS = {PIN, "多家平均"}          # 平博自己不算；多家平均不是一家能买的公司
SEED, N_BOOT = 20261107, 1000


def fair(df: pd.DataFrame, prefix: str, method: str) -> np.ndarray:
    """平博三个价格 → 去水后的三个概率（每行一场）。"""
    odds = df[[f"{prefix}_{s}" for s in SIDES]].to_numpy(float)
    return np.vstack([devig(o, method=method) for o in odds])


def load(db: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(各家胜平负价格长表, 平博赛前+收盘宽表, 比分)。比分用 football-data 和 football-data-new 的。"""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    odds = pd.read_sql("""select match_id, bookmaker, timing, home, draw, away from odds
                          where market = '1x2' and home > 1 and draw > 1 and away > 1""", con)
    res = pd.read_sql("""select m.match_id, m.competition, m.season, m.match_date, s.home_goals, s.away_goals
                         from matches m join match_stats s on s.match_id = m.match_id
                         where m.status = 'finished' and s.source in ('football-data', 'football-data-new')
                           and s.home_goals is not null""", con).drop_duplicates("match_id")
    lot = pd.read_sql("""select l.match_id, l.home, l.draw, l.away from lottery_odds l
                         join (select match_id, max(rowid) r from lottery_odds where product = '竞彩胜平负'
                               group by 1) x on l.rowid = x.r where l.handicap = 0""", con)
    con.close()
    pin = odds[odds.bookmaker == PIN].pivot_table(index="match_id", columns="timing", values=SIDES, aggfunc="last")
    pin.columns = [f"{t}_{s}" for s, t in pin.columns]
    pin = pin.dropna().reset_index()
    lot["bookmaker"], lot["timing"] = "竞彩（最终价）", "lottery"
    return pd.concat([odds[~odds.bookmaker.isin(SKIP_BOOKS)], lot]), pin, res


def bets(odds: pd.DataFrame, pin: pd.DataFrame, res: pd.DataFrame, method: str) -> pd.DataFrame:
    """每场每家每个时间点一行：挑 EV 最大的一边，算 EV、CLV、盈亏（还没按门槛筛）。

    参照：赛前价格 → 平博赛前公道价；收盘价格、竞彩最终价 → 平博收盘公道价。CLV 一律对平博收盘。
    """
    p_pre, p_close = fair(pin, "pre", method), fair(pin, "close", method)
    ref = pd.DataFrame({"match_id": pin.match_id})
    for k, s in enumerate(SIDES):
        ref[f"pre_{s}"], ref[f"close_{s}"] = p_pre[:, k], p_close[:, k]
    d = odds.merge(ref, on="match_id").merge(res, on="match_id")
    price = d[SIDES].to_numpy(float)
    which = np.where(d.timing.eq("pre").to_numpy()[:, None], d[[f"pre_{s}" for s in SIDES]], d[[f"close_{s}" for s in SIDES]])
    ev_all = price * which - 1
    side = ev_all.argmax(axis=1)
    r = np.arange(len(d))
    outcome = np.sign(d.home_goals.to_numpy(float) - d.away_goals.to_numpy(float))   # 1 主胜 / 0 平 / −1 客胜
    won_side = np.where(outcome > 0, 0, np.where(outcome == 0, 1, 2))
    pc = d[[f"close_{s}" for s in SIDES]].to_numpy(float)
    return pd.DataFrame({
        "match_id": d.match_id, "competition": d.competition, "season": d.season,
        "week": pd.to_datetime(d.match_date).dt.to_period("W").astype(str),   # 开球时间很多场没有，按比赛日期分周
        "book": d.bookmaker, "timing": d.timing, "side": np.array(SIDES)[side],
        "price": price[r, side], "ev": ev_all[r, side], "clv": price[r, side] * pc[r, side] - 1,
        "profit": np.where(won_side == side, price[r, side] - 1, -1.0)})


def boot(x: pd.DataFrame, col: str, rng: np.random.Generator) -> tuple[float, float]:
    """按周成块自助法：col 的总和 / 注数 的 95% 区间。"""
    g = x.groupby("week")[col].agg(["sum", "count"]).to_numpy()
    if len(g) < 10:
        return np.nan, np.nan
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    s = g[idx, 0].sum(axis=1) / g[idx, 1].sum(axis=1)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def table(b: pd.DataFrame, keys: list[str], min_n: int = 200) -> pd.DataFrame:
    """按 keys 分组 × 门槛：注数、平均 EV、平均 CLV（区间）、ROI（区间）、命中率、平均价格。"""
    rng = np.random.default_rng(SEED)
    rows = []
    for t in THRESHOLDS:
        for k, g in b[b.ev >= t].groupby(keys):
            if len(g) < min_n:
                continue
            k = k if isinstance(k, tuple) else (k,)
            clv_lo, clv_hi = boot(g, "clv", rng)
            roi_lo, roi_hi = boot(g, "profit", rng)
            rows.append({**dict(zip(keys, k)), "门槛": f"EV ≥ {t:.2f}", "注数": len(g),
                         "平均EV": g.ev.mean(), "平均CLV": g.clv.mean(), "CLV下限": clv_lo, "CLV上限": clv_hi,
                         "ROI": g.profit.mean(), "ROI下限": roi_lo, "ROI上限": roi_hi,
                         "命中率": (g.profit > 0).mean(), "平均价格": g.price.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    s = load_settings()
    out = resolve_path(s["paths"]["data"]) / "backtest" / "value_history" if "data" in s["paths"] else \
        resolve_path("data/backtest/value_history")
    out.mkdir(parents=True, exist_ok=True)
    odds, pin, res = load(str(resolve_path(s["paths"]["database"])))
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_rows", 200)
    for method in ("proportional", "shin"):
        b = bets(odds, pin, res, method)
        b.to_csv(out / f"bets_{method}.csv", index=False, encoding="utf-8-sig")
        t1 = table(b, ["book", "timing"])
        t1.to_csv(out / f"by_book_{method}.csv", index=False, encoding="utf-8-sig")
        print(f"\n===== 去水方法 {method}：各家 × 时间点 =====")
        print(t1.sort_values(["门槛", "ROI"], ascending=[True, False]).round(4).to_string(index=False))
        # 每季分开看（门槛 0.05）：赚钱是不是每年都赚，还是靠一两季
        t2 = table(b, ["book", "timing", "season"], min_n=50)
        t2 = t2[t2["门槛"] == "EV ≥ 0.05"]
        t2.to_csv(out / f"by_season_{method}.csv", index=False, encoding="utf-8-sig")
        t3 = table(b, ["book", "timing", "competition"], min_n=50)
        t3.to_csv(out / f"by_league_{method}.csv", index=False, encoding="utf-8-sig")
    print(f"\n输出：{out}")


if __name__ == "__main__":
    main()
