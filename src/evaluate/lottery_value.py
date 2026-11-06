"""竞彩价格偏离虚拟实盘（需求文档第 14 节“竞彩价格偏离虚拟实盘”，规则是跑之前写下的）。

问题：竞彩的固定价格有没有时候比平博的公道价高出足够多，多到能长期赚钱？
  不靠模型猜比赛。每场在“决定时间”（开球前 6 小时，放进竞彩能买的时间段）看：
  竞彩某一边的价格 × 平博去水后的概率 − 1 = 期望收益（EV）。EV 够大就虚拟买 1 注。
  只是虚拟记录，不连任何真实账户，不是推荐下注。

一、数据（都经过取数层 src/features，按数据字典检查用途）
  - 比赛：五大联赛、上线赛季（实时模式），用途「机器人下注」。
  - 竞彩价格：lottery_odds，决定时间之前已经抓到的最后一组（用途「机器人下注」）。北单是赛后定的浮动奖金，不算。
  - 公道价：odds_changes 里变盘时间、我们存到的时间都不晚于决定时间的价格（用途「机器人下注」）。
    懂球帝快照里平博只有亚盘、大小球，没有胜平负，所以用 平博亚盘主队胜算 + 平博大小球胜算 + 各家胜平负去水后的平局概率
    三个数反推一个比分矩阵（DC），胜平负、让球胜平负的概率都从这个矩阵加出来。
  - 收盘价（算 CLV）：同样的方法，截止时间 = 开球（用途「评估」，赛后才算）。
  - 比分（结算）：用途「机器人结算」。

二、规则：每场 6 个选项（胜平负 3 个 + 让球 3 个），只取 EV 最大的那个，≥ ev_threshold 就下 1 注（平注）。
  CLV = 买入价格 × 收盘公道概率 − 1（主要指标）；ROI = 盈亏 ÷ 注数（次要指标）。区间按周成块 bootstrap。
  只有决定时间在 count_from_utc 之后的注“正式计数”；之前的单独列成“写规则前”，只看看。

三、结果（data/backtest/lottery_value/，不进 Git）
  candidates.csv     每场每个选项一行：价格、公道概率、EV、收盘概率、CLV、结算
  bets.csv           每场 EV 最大的那个选项（有没有达到门槛看 ev 列；主规则那批是 bet = True）
  skipped.csv        过了决定时间但不算的场，带原因（没有竞彩价格、没有平博胜平负……）
  summary.csv        每个时段 × 每个门槛：注数、平均 EV、CLV、ROI（带区间）、结论（只有主门槛下结论）
  decisions_log.csv  每场第一次算出的决定，只追加、不改；以后重算不一样就数出来

只读：数据库只读打开，不写任何表。
用法：
  .venv\\Scripts\\python -m src.evaluate.lottery_value
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.common.config import load_settings, resolve_path
from src.common.markets import asian_handicap, dc_matrices, dc_rho_bounds, over_under, result_probs
from src.common.odds import devig
from src.evaluate.compare import block_ids, bootstrap_mean_diff
from src.evaluate.picks_filter_study import handicap_probs, settle_outcome
from src.features import FeatureStore, open_readonly
from src.features.known_time import to_utc_text
from src.predict.preview import valid_1x2

BET_USAGE = "机器人下注"       # 赛前决定（只能看决定时间以前的东西）
CLOSE_USAGE = "评估"           # 赛后拿收盘价算 CLV
SETTLE_USAGE = "机器人结算"    # 赛后拿比分结算
SIDES = ("home", "draw", "away")
SIDE_TEXT = {"result": ("主胜", "平", "客胜"), "handicap": ("让球主胜", "让球平", "让球客胜")}
BJ = "Asia/Shanghai"           # 显示、竞彩售卖时间用北京时间；内部一律 UTC
PERIOD_COUNTED, PERIOD_BEFORE = "正式计数", "写规则前（只看看）"
LOG_COLS = ["match_id", "decision_utc", "bet", "product", "handicap", "side", "price", "p_fair", "ev", "logged_at_utc"]


# ======================= 决定时间 =======================
def decision_time(kickoff_utc, hours_before: float, window_bj: list[str]) -> str:
    """决定时间（UTC 文字）= 开球前 hours_before 小时，再放进竞彩能买的时间段（北京时间 window_bj = [开始, 结束]）。

    比开始早（例如北京时间凌晨）→ 改成前一天的结束时间；比结束晚 → 改成当天的结束时间。
    例：开球北京时间 03:00 → 开球前 6 小时是 21:00 → 在时间段里，不改；
        开球北京时间 04:00 → 22:00 → 比 21:30 晚 → 改成 21:30；
        开球北京时间 13:30 → 07:30 → 比 09:00 早 → 改成前一天 21:30。
    """
    t = pd.Timestamp(kickoff_utc, tz="UTC") - pd.Timedelta(hours=float(hours_before))
    local = t.tz_convert(BJ)
    day = local.normalize()                                         # 北京时间当天 00:00
    start, end = (day + pd.Timedelta(f"{x}:00") for x in window_bj)
    if local < start:
        local = end - pd.Timedelta(days=1)
    elif local > end:
        local = end
    return to_utc_text(local)


def week_of(kickoff_utc: pd.Series) -> np.ndarray:
    """每注属于哪一周：北京时间开球日期所在的那一周（周一开始），按周成块 bootstrap 用。"""
    d = pd.to_datetime(kickoff_utc, utc=True).dt.tz_convert(BJ).dt.tz_localize(None).dt.normalize()
    return block_ids(d, "week")


# ======================= 取价格 =======================
def lottery_at(fs: FeatureStore, match_id: int, cutoff: str, products: dict) -> dict:
    """决定时间那一刻已经抓到的竞彩价格：每种玩法取抓到时间最晚的那一组（让球数就是这一组的）。

    返回 {'result': {'handicap', 'prices': [主, 平, 客], 'fetched_at_utc'}, 'handicap': {...}}，没有的玩法不出现。
    """
    o = fs.lottery_odds(match_id, cutoff, BET_USAGE, product=list(products.values()))
    out = {}
    for key, name in products.items():
        g = o[o["product"] == name]
        g = g.dropna(subset=list(SIDES))
        g = g[(g[list(SIDES)] > 1).all(axis=1)]
        if g.empty:
            continue
        r = g.sort_values("fetched_at_utc", kind="stable").iloc[-1]    # 同一时间两条时取后导入的
        out[key] = {"handicap": int(r["handicap"]), "prices": [float(r[s]) for s in SIDES],
                    "fetched_at_utc": r["fetched_at_utc"]}
    return out
def _fit_matrix(ah_line: float, ah_share: float, ou_line: float, ou_share: float, p_draw: float,
                n: int) -> tuple[np.ndarray, float]:
    """找 (λ主, λ客, rho)，让 DC 比分矩阵的 亚盘主队胜算、大小球“大”的胜算、平局概率 和三个目标一样。

    三个数解三个数（最小二乘）。rho 截在 dc_rho_bounds 里（矩阵格子不能是负数）。
    返回 (比分矩阵, 还原误差 = 三个数差得最多的那个)。
    """
    def matrix(x):
        lh, la = np.exp(x[0]), np.exp(x[1])
        lo, hi = dc_rho_bounds(lh, la)
        return dc_matrices(lh, la, float(np.clip(x[2], lo[0], hi[0])), n)[0]

    def resid(x):
        m = matrix(x)
        return [asian_handicap(m, ah_line, "home")["win_share"] - ah_share,
                over_under(m, ou_line, "over")["win_share"] - ou_share, result_probs(m)[1] - p_draw]

    fit = least_squares(resid, x0=[np.log(1.45), np.log(1.15), -0.05], method="lm", xtol=1e-12, ftol=1e-12)
    m = matrix(fit.x)
    return m, float(np.abs(resid(fit.x)).max())


def _two_way(row, a: str, b: str) -> tuple[float, float] | None:
    """平博亚盘 / 大小球的一组价格 → (线, a 那边去水后的胜算)；价格不对返回 None。"""
    if pd.isna(row["line"]) or pd.isna(row[a]) or pd.isna(row[b]) or min(row[a], row[b]) <= 1:
        return None
    return float(row["line"]), float(devig([row[a], row[b]], method="proportional")[0])


def pinnacle_fair(fs: FeatureStore, match_id: int, cutoff: str, usage: str, settings: dict) -> dict | None:
    """截止时间那一刻的公道价 → 比分矩阵（胜平负、让球胜平负都从它加出来）。

    懂球帝快照里平博只有亚盘、大小球，没有胜平负（2026-11-07 查过），所以：
      平博亚盘主队胜算 + 平博大小球“大”的胜算 定下两队进球多少；平局概率用各家胜平负去水再平均（平博没有胜平负）。
    只用变盘时间、我们存到的时间都不晚于截止时间的价格（和四本账同一条规则：当时还没存到的不能用）；
    每家每种玩法取最后变的那一组（同一时间两行取后导入的）。
    返回 {'matrix', 'fit_error', 'ah_line', 'ou_line', 'p_draw_books', 'n_books_1x2'}；缺平博亚盘或大小球、
    缺胜平负、还原误差 > max_fit_error 时返回 {'skip': 原因}。
    """
    st = settings["lottery_value"]
    df = fs.odds_changes(match_id, cutoff, usage, market=["1x2", "ah", "ou"])
    df = df[pd.to_datetime(df["fetched_at_utc"]) <= pd.Timestamp(cutoff)]
    now = df.groupby(["bookmaker", "market"], sort=True).tail(1)
    pin = now[now["bookmaker"] == st["pinnacle"]]
    ah = pin[pin["market"] == "ah"]
    ou = pin[pin["market"] == "ou"]
    ah = _two_way(ah.iloc[-1], "home", "away") if len(ah) else None
    ou = _two_way(ou.iloc[-1], "over", "under") if len(ou) else None
    if ah is None or ou is None:
        return {"skip": "没有平博亚盘或大小球"}
    one = valid_1x2(now[now["market"] == "1x2"])
    if one.empty:
        return {"skip": "没有胜平负赔率（算平局概率）"}
    p_draw = float(devig(one.to_numpy(float), method="proportional")[:, 1].mean())
    m, err = _fit_matrix(*ah, *ou, p_draw, int(settings["market_odds"]["max_goals"]))
    if err > float(st["max_fit_error"]):
        return {"skip": f"反推比分矩阵误差 {err:.3f} 太大"}
    return {"matrix": m, "fit_error": err, "ah_line": ah[0], "ou_line": ou[0], "p_draw_books": p_draw,
            "n_books_1x2": int(len(one))}


def side_probs(fair: dict | None, key: str, handicap: int) -> list[float] | None:
    """公道价 → 这种玩法 [主, 平, 客] 三个概率（胜平负就是让球数 0）。没有公道价返回 None。"""
    if fair is None or "matrix" not in fair:
        return None
    return [float(x) for x in handicap_probs(fair["matrix"][None], 0 if key == "result" else handicap)[0]]


# ======================= 每场每个选项 =======================
def build(fs: FeatureStore, settings: dict, now=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """已经过了决定时间的比赛 → (candidates 每场每个选项一行, skipped 跳过的场和原因)。

    now：现在的时间（UTC，测试时可以给）；只算决定时间 ≤ now 的场（还没到决定时间的，现在还不该决定）。
    收盘价只在开球 ≤ now 时算；比分只在取数层拿得到时结算。
    """
    st = settings["lottery_value"]
    now = to_utc_text(now or datetime.now(timezone.utc))
    leagues = list(settings["leagues"][st["leagues"]])
    end = to_utc_text(pd.Timestamp(now) + pd.Timedelta(days=2))   # 决定时间最多比开球早一天多
    targets = fs.prediction_targets(st["first_kickoff_utc"], end, BET_USAGE, competition=leagues)
    rows, skipped = [], []
    for t in targets.itertuples(index=False):
        mid, kickoff = int(t.match_id), to_utc_text(t.kickoff_earliest_utc)
        dec = decision_time(kickoff, st["hours_before"], st["sale_window_bj"])
        if dec > now:
            continue
        info = {"match_id": mid, "competition": t.competition, "home_team": t.home_team, "away_team": t.away_team,
                "kickoff_utc": kickoff, "decision_utc": dec}
        lot = lottery_at(fs, mid, dec, st["products"])
        if not lot:
            skipped.append({**info, "原因": "决定时间之前没有竞彩价格"})
            continue
        fair = pinnacle_fair(fs, mid, dec, BET_USAGE, settings)
        if "skip" in fair:
            skipped.append({**info, "原因": f"决定时间：{fair['skip']}"})
            continue
        close = pinnacle_fair(fs, mid, kickoff, CLOSE_USAGE, settings) if kickoff <= now else None
        result = _result(fs, mid, now)
        for key, q in lot.items():
            hcap = 0 if key == "result" else q["handicap"]
            p = side_probs(fair, key, hcap)
            pc = side_probs(close, key, hcap)            # 收盘价算不了（还没开球、缺价格）时是空值
            outcome = None if result is None else int(settle_outcome([result[0]], [result[1]], hcap)[0])
            for i, side in enumerate(SIDES):
                price = q["prices"][i]
                row = {**info, "product": st["products"][key], "market": key, "handicap": hcap, "side": side,
                       "side_text": SIDE_TEXT[key][i], "price": price, "lottery_fetched_utc": q["fetched_at_utc"],
                       "p_fair": p[i], "ev": price * p[i] - 1,
                       "p_close": np.nan if pc is None else pc[i],
                       "clv": np.nan if pc is None else price * pc[i] - 1,
                       "fit_error": fair["fit_error"], "settled": outcome is not None,
                       "home_goals": np.nan if result is None else result[0],
                       "away_goals": np.nan if result is None else result[1]}
                row["hit"] = np.nan if outcome is None else float(outcome == i)
                row["profit"] = np.nan if outcome is None else price * row["hit"] - 1
                rows.append(row)
    cand = pd.DataFrame(rows)
    if len(cand):
        counted = pd.to_datetime(cand["decision_utc"]) >= pd.Timestamp(st["count_from_utc"])
        cand.insert(0, "period", np.where(counted, PERIOD_COUNTED, PERIOD_BEFORE))
        cand["week"] = week_of(cand["kickoff_utc"])
    return cand, pd.DataFrame(skipped)


def _result(fs: FeatureStore, match_id: int, now: str) -> tuple[int, int] | None:
    """现在已经能知道的 90 分钟比分 (主, 客)；还没踢完、改期、取消的返回 None（不结算）。"""
    r = fs.match_result(match_id, now, SETTLE_USAGE)
    if r.empty or pd.isna(r.iloc[0]["home_goals"]) or pd.isna(r.iloc[0]["away_goals"]):
        return None
    return int(r.iloc[0]["home_goals"]), int(r.iloc[0]["away_goals"])


def best_per_match(cand: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """每场 EV 最大的那个选项（一场最多 1 注），bet = EV ≥ 门槛。一样大时取排在前面的（胜平负在前、主胜在前）。"""
    if cand.empty:
        return cand.assign(bet=pd.Series(dtype=bool))
    b = cand.loc[cand.groupby("match_id", sort=False)["ev"].idxmax()].copy()
    b["bet"] = b["ev"] >= threshold
    return b.sort_values(["decision_utc", "match_id"]).reset_index(drop=True)
# ======================= 汇总和结论 =======================
def _ci(values: pd.Series, weeks: pd.Series, settings: dict) -> tuple[float, float, float]:
    """平均值和按周成块 bootstrap 区间；少于 2 周或没有数时区间是空值。"""
    ok = values.notna()
    v, w = values[ok].to_numpy(float), weeks[ok].to_numpy()
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(set(w)) < 2:
        return float(v.mean()), float("nan"), float("nan")
    c = settings["compare"]
    return bootstrap_mean_diff(v, w, int(c["n_boot"]), float(c["confidence"]), int(c["seed"]))


def verdict(n_clv: int, clv_lo: float, clv_hi: float, roi: float, settings: dict) -> str:
    """主规则那一行的结论（第 14 节“竞彩价格偏离虚拟实盘”第 12 条）。n_clv = 有 CLV 的注数。"""
    st = settings["lottery_value"]
    if n_clv < int(st["min_bets_report"]):
        return f"注数不够（{n_clv} / {st['min_bets_report']}），只列数"
    if clv_hi < 0:
        return "没看到：CLV 区间整个 < 0，停下"
    if n_clv >= int(st["pass_bets"]) and clv_lo > 0 and roi > 0:
        return "通过：交给你定要不要用小钱实盘试"
    if n_clv >= int(st["give_up_bets"]):
        return f"没看到：{st['give_up_bets']} 注还没通过，停下"
    return "继续攒"


def summarize(cand: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每个时段 × 每个门槛一行，另加每个时段“6 个选项全买”一行（方法检查：平均 EV 应该在 −12% 左右）。"""
    st = settings["lottery_value"]
    out = []
    if cand.empty:
        return pd.DataFrame(out)
    for period in (PERIOD_COUNTED, PERIOD_BEFORE):
        c = cand[cand["period"] == period]
        if c.empty:
            continue
        groups = [(f"EV ≥ {t:g}", float(t)) for t in st["report_thresholds"]]
        n = len(c["match_id"].unique())
        for name, t in [*groups, ("6 个选项全买（方法检查）", None)]:
            if t is None:
                b = c
            else:
                b = best_per_match(c, t)
                b = b[b["bet"]]
            out.append(_summary_row(period, name, t, b, n, settings))
            # 主门槛和全买再按玩法分开列（只看看）：让球的概率是从比分矩阵推出来的，可能有系统偏差，要单独看
            if t is None or abs(t - float(st["ev_threshold"])) < 1e-12:
                for prod in st["products"].values():
                    out.append(_summary_row(period, f"{name} · {prod}", None, b[b["product"] == prod], n, settings))
    return pd.DataFrame(out)


def _summary_row(period: str, name: str, t: float | None, b: pd.DataFrame, n_matches: int, settings: dict) -> dict:
    st = settings["lottery_value"]
    clv, clv_lo, clv_hi = _ci(b["clv"], b["week"], settings) if len(b) else (np.nan,) * 3
    settled = b[b["settled"]] if len(b) else b
    roi, roi_lo, roi_hi = _ci(settled["profit"], settled["week"], settings) if len(settled) else (np.nan,) * 3
    n_clv = int(b["clv"].notna().sum()) if len(b) else 0
    main = t is not None and abs(t - float(st["ev_threshold"])) < 1e-12 and period == PERIOD_COUNTED
    return {"时段": period, "规则": name, "主规则": main, "算了的场": n_matches, "注数": len(b),
            "有CLV的注": n_clv, "已结算": len(settled), "平均EV": b["ev"].mean() if len(b) else np.nan,
            "平均CLV": clv, "CLV下限": clv_lo, "CLV上限": clv_hi,
            "命中率": settled["hit"].mean() if len(settled) else np.nan,
            "平均价格": b["price"].mean() if len(b) else np.nan,
            "盈亏": settled["profit"].sum() if len(settled) else 0.0,
            "ROI": roi, "ROI下限": roi_lo, "ROI上限": roi_hi,
            "结论": verdict(n_clv, clv_lo, clv_hi, roi, settings) if main else "只看看"}
# ======================= 决定记录（只追加） =======================
def update_log(log: pd.DataFrame, best: pd.DataFrame, now: str) -> tuple[pd.DataFrame, int]:
    """把这次算出的每场决定（主规则：下没下、下哪一注）追加进记录；已经记过的场不改。

    返回 (新的记录, 和以前记下的不一样的场数)。不一样 = 下没下、玩法、让球数、哪一边、价格、概率有一个变了
    （只用决定时间以前的数据，正常重算应该完全一样；不一样说明数据被重导或改过，要查）。
    """
    new = best.assign(logged_at_utc=now)[LOG_COLS] if len(best) else pd.DataFrame(columns=LOG_COLS)
    if log.empty:
        return new.reset_index(drop=True), 0
    old = log.drop_duplicates("match_id", keep="first").set_index("match_id")
    cur = new.set_index("match_id")
    both = old.index.intersection(cur.index)
    o, c = old.loc[both], cur.loc[both]
    same = ((o["bet"].astype(str) == c["bet"].astype(str)) & (o["product"] == c["product"])
            & (o["handicap"].astype(int) == c["handicap"].astype(int)) & (o["side"] == c["side"])
            & np.isclose(o["price"].astype(float), c["price"].astype(float), atol=1e-9)
            & np.isclose(o["p_fair"].astype(float), c["p_fair"].astype(float), atol=1e-9))
    added = new[~new["match_id"].isin(old.index)]
    return pd.concat([log, added], ignore_index=True), int((~same).sum())


def write(cand: pd.DataFrame, skipped: pd.DataFrame, settings: dict, now=None) -> tuple[dict, pd.DataFrame]:
    """写结果文件、追加决定记录。返回 (几个数字, summary)。"""
    st = settings["lottery_value"]
    now = to_utc_text(now or datetime.now(timezone.utc))
    out = resolve_path(st["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    best = best_per_match(cand, float(st["ev_threshold"]))
    summary = summarize(cand, settings)
    log_path = out / "decisions_log.csv"
    log = pd.read_csv(log_path, encoding="utf-8-sig") if log_path.exists() else pd.DataFrame(columns=LOG_COLS)
    log, mismatched = update_log(log, best, now)
    cand.to_csv(out / "candidates.csv", index=False, encoding="utf-8-sig")
    best.to_csv(out / "bets.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(out / "skipped.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    stats = {"决定过的场": int(best["match_id"].nunique()) if len(best) else 0,
             "虚拟注": int(best["bet"].sum()) if len(best) else 0, "跳过的场": len(skipped),
             "和记录不一样的场": mismatched, "输出": str(out)}
    return stats, summary


def short_text(summary: pd.DataFrame) -> str:
    """心跳摘要用的一句话（40 个字以内）：主规则那一行的注数、平均 CLV、结论。"""
    m = summary[summary["主规则"]] if len(summary) else summary
    if m.empty:
        return "正式计数 0 注"
    r = m.iloc[0]
    clv = f"，CLV {r['平均CLV']:+.3f}" if pd.notna(r["平均CLV"]) else ""
    return f"{int(r['注数'])} 注{clv}，{str(r['结论'])[:12]}"[:40]


def pending(cand: pd.DataFrame, settings: dict, now: str) -> pd.DataFrame:
    """已经过了决定时间、还没开球的虚拟注（主规则）。只是虚拟记录，不是推荐下注。"""
    best = best_per_match(cand, float(settings["lottery_value"]["ev_threshold"]))
    if best.empty:
        return best
    b = best[best["bet"] & (best["kickoff_utc"] > now)].copy()
    b["开球（北京时间）"] = pd.to_datetime(b["kickoff_utc"], utc=True).dt.tz_convert(BJ).dt.strftime("%m-%d %H:%M")
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description="竞彩价格偏离虚拟实盘（只读，虚拟注）")
    ap.add_argument("--db", help="数据库文件（不写 = settings.yaml 的 paths.database）")
    a = ap.parse_args()
    settings = load_settings()
    now = to_utc_text(datetime.now(timezone.utc))
    con = open_readonly(a.db)
    try:
        fs = FeatureStore(con, "实时", settings)          # 上线后的赛季（2026/27 起），用实时模式
        cand, skipped = build(fs, settings, now)
    finally:
        con.close()
    stats, summary = write(cand, skipped, settings, now)
    for k, v in stats.items():
        print(f"{k}：{v}")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        if len(summary):
            print(summary[["时段", "规则", "注数", "有CLV的注", "已结算", "平均EV", "平均CLV", "CLV下限", "CLV上限",
                           "命中率", "ROI", "结论"]].to_string(index=False))
        p = pending(cand, settings, now)
        if len(p):
            print("\n已经决定、还没开球的虚拟注（只是虚拟记录，不是推荐下注）：")
            print(p[["开球（北京时间）", "home_team", "away_team", "product", "handicap", "side_text", "price",
                     "p_fair", "ev"]].to_string(index=False))


if __name__ == "__main__":
    main()
