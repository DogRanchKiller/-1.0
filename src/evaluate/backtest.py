"""按时间滚动回测（需求文档第 5 节“按时间滚动回测”“分赛季看成绩”“回测的计算量”，第 16 节“不偷看未来”）。

做法：假装回到过去，按时间顺序一场一场往前走。
  对每场要预测的比赛、每个预测时点（T-24 / T-6 / T-1 = 开球前 24 / 6 / 1 小时）：
    1. 截止时间 = 开球前这么多小时（取数层 FeatureStore.cutoff_before 算，只有日期的比赛从当天 00:00 英国时间往前算）；
    2. 通过取数层拿到截止时间之前已经能知道的比赛结果（增量拿：只拿上次以来新知道的，用 since 参数）；
    3. 新结果交给模型 update()（Elo 每场赛后加减分就在这里）；离上次重新拟合满 refit_days 天（默认 7）就 fit() 一次；
    4. 模型给这场出预测（比分矩阵或胜平负三个概率）；
  全部跑完，再站在“开球后 72 小时”取本场比分打分（用途“评估”），顺便取两个参考标杆（第 5 节“两个标杆”）：
    标杆 1 pinnacle_close：平博收盘赔率去水（2012/13 起）；
    标杆 2 market_avg：多家公司平均的收盘赔率去水（2019/20 起）。没有收盘平均的比赛，用多家平均的“赛前”赔率代替，
      时点写“赛前代替”，和真正的收盘分开统计（赛前赔率离开球还有 1–3 天，消息少一些，按理会比收盘差一点）。
  各年份能用的标杆不一样，所以每场比赛标一个“标杆分段”（benchmark_period），成绩按分段分开统计。

为什么不能随机分训练集、测试集：随机分的话，2015 年的比赛可能用 2018 年的比赛训练出来的模型预测，
等于知道了“以后谁变强”。足球每年都在变（转会、换帅、规则），只有“用过去预测未来”的成绩才代表上线后的成绩。

不偷看未来，三道关：
1. 取数层：只返回截止时间之前已经能知道的比赛；用途、运行模式不允许就报错（开发模式碰到期末考试赛季直接停）。
2. 本文件：模型拿不到取数层，只拿到取好的表；每次取到的历史再核对一遍 known_at_utc ≤ 截止时间，
   而且要预测的比赛绝不会出现在历史里，不对就报错。
3. 本场的比分只在全部预测做完以后、用“评估”用途取，用来打分。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.backtest --leagues E0 --first 2005 --last 2023 --models frequency elo
  只跑一个赛季量时间：--leagues E0 --first 2023 --last 2023
结果在 data/backtest/<运行名>/（data/ 不进 Git）：
  predictions.csv            每场 × 时点 × 模型一行：概率、实际结果、每场分数
  summary.csv                按 模型 × 时点 × 联赛 × 赛季 汇总的 log loss、Brier、RPS、命中率（另有按联赛、全部的汇总行）
  summary_vs_benchmark.csv   只看有平博收盘赔率的比赛，模型和平博放在同一批比赛上比
  summary_by_benchmark_period.csv  按“标杆分段”汇总：每一段里模型和这一段能用的标杆在同一批比赛上比
  calibration.csv            校准表（模型说 40% 的事，实际是不是四成发生）
  run_info.json              这次用的设置、模型参数、耗时
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import markets
from src.common.config import resolve_path, season_role
from src.common.odds import devig
from src.evaluate import metrics
from src.features import FeatureStore, LookAheadError, open_readonly
from src.features.known_time import compute_known_times, to_utc_text
from src.models.base import Model, check_result_probs

BENCHMARK = "pinnacle_close"   # 标杆 1 在结果表里的模型名：平博收盘
BENCHMARK_AVG = "market_avg"   # 标杆 2 在结果表里的模型名：多家平均（收盘；没有收盘时用赛前代替）
BENCHMARKS = (BENCHMARK, BENCHMARK_AVG)
CLOSE = "收盘"                 # 标杆的“时点”：收盘
PRE_SUBSTITUTE = "赛前代替"     # 标杆 2 没有收盘平均、用赛前平均代替时的“时点”
RESULT_COLS = ["p_home", "p_draw", "p_away"]

# 每场比赛的“标杆分段”：这场能用哪几个标杆（第 5 节“各年份能用的标杆不一样，成绩分段统计”）。
# 按每场实际有的赔率分，不按年份写死：大致 2012/13–2018/19 是“平博收盘+平均赛前”，2019/20–2024/25 是“两个收盘”，
# 2025/26 下半季起是“只有平均收盘”，2005/06–2011/12 是“只有平均赛前”。
PERIOD_BOTH = "平博收盘+平均收盘"
PERIOD_PIN_PRE = "平博收盘+平均赛前"
PERIOD_PIN = "只有平博收盘"
PERIOD_AVG = "只有平均收盘"
PERIOD_PRE = "只有平均赛前"
PERIOD_NONE = "没有标杆"


def benchmark_period(has_pin: bool, avg_horizon: str | None) -> str:
    """有没有平博收盘、标杆 2 是收盘还是赛前代替（None = 都没有）→ 分段名。"""
    if has_pin:
        return {CLOSE: PERIOD_BOTH, PRE_SUBSTITUTE: PERIOD_PIN_PRE}.get(avg_horizon, PERIOD_PIN)
    return {CLOSE: PERIOD_AVG, PRE_SUBSTITUTE: PERIOD_PRE}.get(avg_horizon, PERIOD_NONE)


def horizon_label(hours: float) -> str:
    """24 → 'T-24'。"""
    return f"T-{int(hours) if float(hours).is_integer() else hours}"


@dataclass
class BacktestResult:
    predictions: pd.DataFrame                 # 每场 × 时点 × 模型一行
    summary: pd.DataFrame                     # 分组汇总
    summary_vs_benchmark: pd.DataFrame        # 和平博收盘同一批比赛
    calibration: pd.DataFrame                 # 校准表
    summary_by_period: pd.DataFrame = field(default_factory=pd.DataFrame)  # 按标杆分段汇总
    info: dict = field(default_factory=dict)  # 设置、参数、耗时
    models: list = field(default_factory=list)


# ======================= 第 1 步：要预测哪些比赛 =======================
def season_targets(fs: FeatureStore, competitions: list[str], first: int, last: int) -> pd.DataFrame:
    """first–last 赛季（含两头）这些联赛的全部比赛，当作预测目标。

    取数层按开球时间段取目标，这里先取一个盖住这些赛季的时间段，再按 season 列筛（2019/20 疫情延期到 8 月也不会漏）。
    时间段的终点：下一个赛季也是开发赛季，就放宽到再下一年（反正按 season 筛掉）；
    下一个赛季不是开发赛季（期末考试），就停在 window_end_md（7 月 1 日），不碰考试赛季。
    开发模式下赛季范围里有考试赛季，取数层会直接报错（SeasonLockedError）。
    """
    b = fs.settings["backtest"]
    end_year = last + 2 if season_role(last + 1, fs.settings) == "dev" else last + 1
    start, end = f"{first}-{b['window_start_md']}", f"{end_year}-{b['window_end_md']}"
    t = fs.prediction_targets(start, end, usage="A版输入", competition=competitions)
    t = t[(t["season"] >= first) & (t["season"] <= last)]
    missing = sorted(set(range(first, last + 1)) - set(t["season"].unique()))
    if missing:
        print(f"注意：这些赛季没有比赛：{missing}")
    return t.reset_index(drop=True)


# ======================= 第 2 步：滚动预测 =======================
def _derived(matrix_cache: dict, pred) -> tuple[np.ndarray, float, float]:
    """模型的一个预测 → (胜平负三个概率, 大 2.5 球概率, 双方进球概率)。出不了比分矩阵的，后两个是 nan。

    同一个矩阵对象（例如频率泊松模型同一联赛每场都一样）只推一次：按对象缓存，缓存里留着对象本身，避免内存地址被重用后认错。
    """
    a = np.asarray(pred, dtype=float)
    if a.ndim == 1:
        return check_result_probs(a), np.nan, np.nan
    key = id(pred)
    hit = matrix_cache.get(key)
    if hit is not None and hit[0] is pred:
        return hit[1]
    out = (markets.result_probs(a), markets.over_under(a, 2.5, "over")["win"], markets.btts(a)["yes"])
    matrix_cache[key] = (pred, out)
    return out


def _rolling_predictions(fs: FeatureStore, targets: pd.DataFrame, models: list[Model], horizons: list[float],
                         timing: dict) -> tuple[pd.DataFrame, dict]:
    settings = fs.settings
    refit = pd.Timedelta(days=float(settings["backtest"]["refit_days"]))
    usages = {m.usage for m in models}
    if len(usages) != 1:
        raise ValueError(f"一次回测里的模型要用同一种用途（现在是 {usages}），分开跑")
    usage = usages.pop()
    data = list(dict.fromkeys(d for m in models for d in m.data))
    comps = sorted(targets["competition"].unique())

    # 每场 × 每个时点的截止时间（取数层算，规则只在一处）
    t0 = time.perf_counter()
    rows = []
    for h in horizons:
        for mid in targets["match_id"]:
            rows.append((int(mid), horizon_label(h), fs.cutoff_before(int(mid), h, usage)))
    plan = pd.DataFrame(rows, columns=["match_id", "horizon", "cutoff_utc"])
    plan = plan.merge(targets, on="match_id").sort_values(["cutoff_utc", "match_id"], ignore_index=True)
    timing["截止时间计算秒"] = round(time.perf_counter() - t0, 2)

    # 每场目标比赛的比分“什么时候能知道”：截止时间往前走，中间没有新结果能知道，就不用再去取数（省时间）
    known = np.sort(compute_known_times(targets, fs.fs)["result_known_utc"].to_numpy())
    margin = pd.Timedelta(hours=float(settings["features"]["result_available_after_kickoff_hours"]) + 48)

    hist_parts: list[pd.DataFrame] = []
    hist: pd.DataFrame | None = None
    seen: set[int] = set()
    last_fetch: str | None = None
    last_fit: pd.Timestamp | None = None
    fits = {m.name: [] for m in models}
    out_rows: list[tuple] = []
    matrix_cache: dict = {}
    n_fetch = 0
    season_secs: dict[int, float] = {}

    for cutoff, group in plan.groupby("cutoff_utc", sort=True):
        t_start = time.perf_counter()
        cut_ts = pd.Timestamp(cutoff)
        # ---- 取新结果（增量） ----
        need = last_fetch is None
        if not need:
            new_known = np.searchsorted(known, cutoff, "right") > np.searchsorted(known, last_fetch, "right")
            need = new_known or cut_ts - pd.Timestamp(last_fetch) >= refit
        if need:
            since = None if last_fetch is None else to_utc_text(pd.Timestamp(last_fetch) - margin)
            new = fs.historical_matches(cutoff, usage, competition=comps, data=data, since=since)
            n_fetch += 1
            new = new[~new["match_id"].isin(seen)]
            if (new["known_at_utc"] > cutoff).any():
                raise LookAheadError(f"取数层返回了截止时间 {cutoff} 之后才能知道的比赛，回测停止")
            if not new.empty:
                seen.update(int(x) for x in new["match_id"])
                hist_parts.append(new)
                if last_fetch is not None:
                    for m in models:
                        m.update(new, cutoff)
            last_fetch = cutoff
        if seen.intersection(int(x) for x in group["match_id"]):
            raise LookAheadError(f"要预测的比赛出现在了截止时间 {cutoff} 的历史结果里，回测停止")
        # ---- 每 refit_days 天重新拟合一次 ----
        if last_fit is None or cut_ts - last_fit >= refit:
            if hist_parts:
                hist = pd.concat([hist] + hist_parts if hist is not None else hist_parts, ignore_index=True)
                hist_parts = []
            for m in models:
                m.fit(hist if hist is not None else pd.DataFrame(), cutoff)
                fits[m.name].append(cutoff)
            last_fit = cut_ts
            matrix_cache.clear()
        # ---- 出预测 ----
        info_cols = group.drop(columns=["horizon", "cutoff_utc"])
        for m in models:
            preds = m.predict(info_cols, cutoff)
            if len(preds) != len(group):
                raise ValueError(f"模型 {m.name} 返回了 {len(preds)} 个预测，要的是 {len(group)} 个")
            for (mid, hz), p in zip(group[["match_id", "horizon"]].itertuples(index=False, name=None), preds):
                if p is None:
                    out_rows.append((mid, hz, cutoff, m.name, fits[m.name][-1], np.nan, np.nan, np.nan, np.nan, np.nan))
                    continue
                r, over, yes = _derived(matrix_cache, p)
                out_rows.append((mid, hz, cutoff, m.name, fits[m.name][-1], *r, over, yes))
        s = int(group["season"].max())
        season_secs[s] = season_secs.get(s, 0.0) + time.perf_counter() - t_start

    preds = pd.DataFrame(out_rows, columns=["match_id", "horizon", "cutoff_utc", "model", "fit_cutoff_utc",
                                            *RESULT_COLS, "p_over25", "p_btts"])
    timing["取数次数"] = n_fetch
    timing["每个赛季滚动预测秒"] = {k: round(v, 2) for k, v in sorted(season_secs.items())}
    timing["每个模型重新拟合次数"] = {k: len(v) for k, v in fits.items()}
    return preds, fits


# ======================= 第 3 步：取答案、打分 =======================
def _odds_row(df: pd.DataFrame, bookmaker: str) -> np.ndarray | None:
    """取到的赔率表里这家公司的胜平负三个赔率；没有、或者缺一个都当作没有。"""
    o = df[df["bookmaker"] == bookmaker].dropna(subset=["home", "draw", "away"])
    return None if o.empty else o.iloc[0][["home", "draw", "away"]].to_numpy(dtype=float)


def _answers_and_benchmark(fs: FeatureStore, targets: pd.DataFrame, benchmark: bool
                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """全部预测做完以后，站在开球后 eval_after_kickoff_hours 小时，取本场比分（用途“评估”）和两个标杆的赔率。

    标杆 1：平博收盘。标杆 2：多家平均收盘；这场没有收盘平均时，用多家平均的赛前赔率代替（时点写“赛前代替”）。
    两个标杆都用 backtest.benchmark_devig_method 去水（默认按比例）。
    返回 (答案, 标杆预测, 每场的标杆分段)。
    """
    b = fs.settings["backtest"]
    after = pd.Timedelta(hours=float(b["eval_after_kickoff_hours"]))
    pin, avg = b["benchmark_bookmaker"], b["benchmark_avg_bookmaker"]
    method = b["benchmark_devig_method"]
    ans, bench, periods = [], [], []
    for mid, ko in targets[["match_id", "kickoff_earliest_utc"]].itertuples(index=False, name=None):
        when = to_utc_text(pd.Timestamp(ko) + after)
        r = fs.match_result(int(mid), when, "评估")
        if not r.empty and pd.notna(r.iloc[0]["result"]):
            row = r.iloc[0]
            ans.append((int(mid), row["result"], int(row["home_goals"]), int(row["away_goals"])))
        if not benchmark:
            continue
        close = fs.closing_odds(int(mid), when, "评估", bookmaker=[pin, avg], market="1x2")
        o_pin, o_avg, avg_hz = _odds_row(close, pin), _odds_row(close, avg), CLOSE
        if o_avg is None:     # 没有收盘平均 → 用赛前平均代替
            o_avg = _odds_row(fs.pre_match_odds(int(mid), when, "评估", bookmaker=avg, market="1x2"), avg)
            avg_hz = PRE_SUBSTITUTE if o_avg is not None else None
        for name, hz, o in [(BENCHMARK, CLOSE, o_pin), (BENCHMARK_AVG, avg_hz, o_avg)]:
            if o is not None:
                bench.append((int(mid), hz, when, name, None, *devig(o, method), np.nan, np.nan))
        periods.append((int(mid), benchmark_period(o_pin is not None, avg_hz)))
    answers = pd.DataFrame(ans, columns=["match_id", "result", "home_goals", "away_goals"])
    bench_df = pd.DataFrame(bench, columns=["match_id", "horizon", "cutoff_utc", "model", "fit_cutoff_utc",
                                            *RESULT_COLS, "p_over25", "p_btts"])
    period_df = pd.DataFrame(periods, columns=["match_id", "benchmark_period"])
    return answers, bench_df, period_df


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    """给每行预测算每场分数：log loss、Brier、RPS、命中（概率最大的那个结果是不是真的发生了），
    能出比分矩阵的再算大 2.5 球、双方进球的 Brier。没有预测或没有结果的行，分数是空的。"""
    df = predictions.copy()
    for c in ["log_loss", "brier", "rps", "hit", "brier_over25", "brier_btts"]:
        df[c] = np.nan
    ok = df[RESULT_COLS].notna().all(axis=1) & df["result"].notna()
    if ok.any():
        p = df.loc[ok, RESULT_COLS].to_numpy(dtype=float)
        y = metrics.encode_results(df.loc[ok, "result"])
        df.loc[ok, "log_loss"] = metrics.log_loss_per_match(p, y)
        df.loc[ok, "brier"] = metrics.brier_per_match(p, y)
        df.loc[ok, "rps"] = metrics.rps_per_match(p, y)
        df.loc[ok, "hit"] = (p.argmax(axis=1) == y).astype(float)
    goals = df["home_goals"] + df["away_goals"]
    for col, happened, out in [("p_over25", goals > 2.5, "brier_over25"),
                               ("p_btts", (df["home_goals"] > 0) & (df["away_goals"] > 0), "brier_btts")]:
        m = df[col].notna() & df["result"].notna()
        if m.any():
            df.loc[m, out] = metrics.brier_per_match(df.loc[m, col].to_numpy(dtype=float),
                                                     happened[m].astype(int).to_numpy())
    return df


# ======================= 第 4 步：汇总、校准 =======================
def common_rows(scored: pd.DataFrame) -> pd.DataFrame:
    """同一个时点上，只留“这次回测的每个模型都有预测、也有结果”的比赛，让各模型在同一批比赛上比（参考标杆另算）。"""
    df = scored[scored["log_loss"].notna()]
    is_bench = df["model"].isin(BENCHMARKS)
    mods = df[~is_bench]
    n_models = mods.groupby("horizon")["model"].nunique()
    per = mods.groupby(["horizon", "match_id"])["model"].nunique().reset_index(name="k")
    per = per[per["k"] == per["horizon"].map(n_models)]
    keep = mods.merge(per[["horizon", "match_id"]], on=["horizon", "match_id"])
    return pd.concat([keep, df[is_bench]], ignore_index=True)


def _agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    d = df.assign(is_draw=(df["result"] == "D").astype(float))
    out = d.groupby(keys, sort=True).agg(
        n=("log_loss", "size"), log_loss=("log_loss", "mean"), brier=("brier", "mean"), rps=("rps", "mean"),
        hit=("hit", "mean"), draw_pred=("p_draw", "mean"), draw_obs=("is_draw", "mean"),
        brier_over25=("brier_over25", "mean"), brier_btts=("brier_btts", "mean")).reset_index()
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """三层汇总：每个 联赛 × 赛季；每个联赛（所有赛季）；全部。分数都是每场分数的平均，越小越好（hit 命中率越大越好）。
    draw_pred 平均预测的平局概率，draw_obs 实际平局比例：两个差得多，说明平局估偏了。"""
    if rows.empty:
        return pd.DataFrame()
    parts = []
    for level, keys in [("联赛×赛季", ["competition", "season"]), ("联赛", ["competition"]), ("全部", [])]:
        t = _agg(rows, ["model", "horizon", *keys])
        t.insert(0, "level", level)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["competition"] = out["competition"].fillna("全部")
    out["season"] = out["season"].map(lambda v: "全部" if pd.isna(v) else str(int(v)))
    cols = ["level", "model", "horizon", "competition", "season", "n", "log_loss", "brier", "rps", "hit",
            "draw_pred", "draw_obs", "brier_over25", "brier_btts"]
    return out[cols]


def vs_benchmark_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """只留有平博收盘赔率的比赛（模型和标杆在同一批比赛上比）。"""
    has = set(rows.loc[rows["model"] == BENCHMARK, "match_id"])
    return rows[rows["match_id"].isin(has)]


def summarize_by_period(rows: pd.DataFrame) -> pd.DataFrame:
    """按标杆分段汇总（第 5 节“各年份能用的标杆不一样，成绩分段统计”）。

    每个分段里，只留“这一段的每个标杆都有分数”的比赛，模型和标杆在同一批比赛上比。
    例如“平博收盘+平均赛前”这一段，模型、pinnacle_close、market_avg（赛前代替）三个的 n 一样。
    汇总层次和 summarize 一样（联赛×赛季、联赛、全部），前面多一列 benchmark_period。
    """
    if rows.empty or "benchmark_period" not in rows:
        return pd.DataFrame()
    parts = []
    for period, part in rows.groupby("benchmark_period", sort=False):
        bench = part[part["model"].isin(BENCHMARKS)]
        if not bench.empty:
            need = bench["model"].nunique()
            ok = bench.groupby("match_id")["model"].nunique()
            part = part[part["match_id"].isin(ok[ok == need].index)]
        s = summarize(part)
        if not s.empty:
            s.insert(0, "benchmark_period", period)
            parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def calibration(rows: pd.DataFrame, n_bins: int | None = None) -> pd.DataFrame:
    """每个 模型 × 时点：主胜、平、客胜各一张校准表，再把三个结果合在一起一张（每场贡献 3 个点）。"""
    parts = []
    for (model, hz), part in rows.groupby(["model", "horizon"]):
        p = part[RESULT_COLS].to_numpy(dtype=float)
        y = metrics.one_hot(metrics.encode_results(part["result"]), 3)
        for name, pp, yy in [("主胜", p[:, 0], y[:, 0]), ("平", p[:, 1], y[:, 1]), ("客胜", p[:, 2], y[:, 2]),
                             ("三个合并", p.ravel(), y.ravel())]:
            t = metrics.calibration_table(pp, yy, n_bins)
            t.insert(0, "outcome", name)
            t.insert(0, "horizon", hz)
            t.insert(0, "model", model)
            t["ece"] = metrics.ece(pp, yy, n_bins)
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ======================= 总入口 =======================
def run_backtest(fs: FeatureStore, competitions, first: int, last: int, models: list[Model],
                 horizons: list[float] | None = None, benchmark: bool = True) -> BacktestResult:
    """回测入口。fs：取数层（开发模式只能回测开发赛季）；competitions：联赛代码；first、last：赛季（含两头）；
    models：模型对象（用 src.models.registry.build_model 按名字造，每次回测都新造）；horizons：预测时点（小时）。"""
    if first > last:
        raise ValueError("first 不能大于 last")
    competitions = [competitions] if isinstance(competitions, str) else list(competitions)
    names = [m.name for m in models]
    if len(set(names)) != len(names) or set(BENCHMARKS) & set(names):
        raise ValueError(f"模型名字不能重复，也不能叫 {' / '.join(BENCHMARKS)}：{names}")
    horizons = list(horizons or fs.settings["backtest"]["horizons"])
    timing: dict = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    targets = season_targets(fs, competitions, first, last)
    timing["取预测目标秒"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    preds, _ = _rolling_predictions(fs, targets, models, horizons, timing)
    timing["滚动预测秒"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    answers, bench, periods = _answers_and_benchmark(fs, targets, benchmark)
    timing["取答案和标杆秒"] = round(time.perf_counter() - t0, 2)

    info_cols = ["match_id", "competition", "season", "match_date", "kickoff_utc", "home_team", "away_team"]
    allp = pd.concat([preds, bench], ignore_index=True) if not bench.empty else preds
    allp = allp.merge(targets[info_cols], on="match_id").merge(answers, on="match_id", how="left")
    if benchmark:
        allp = allp.merge(periods, on="match_id", how="left")
    scored = score(allp)
    rows = common_rows(scored)
    summary = summarize(rows)
    has_pin = not bench.empty and (bench["model"] == BENCHMARK).any()
    vs = summarize(vs_benchmark_rows(rows)) if has_pin else pd.DataFrame()
    by_period = summarize_by_period(rows) if benchmark else pd.DataFrame()
    calib = calibration(rows)
    timing["总秒数"] = round(time.perf_counter() - t_all, 2)

    avg_rows = bench[bench["model"] == BENCHMARK_AVG] if not bench.empty else bench
    model_rows = scored[~scored["model"].isin(BENCHMARKS)]
    missing = model_rows["p_home"].isna().groupby([model_rows["model"], model_rows["horizon"]]).sum()
    info = {
        "联赛": competitions, "赛季": [first, last], "时点": [horizon_label(h) for h in horizons],
        "运行模式": fs.mode, "模型": names, "目标比赛数": int(len(targets)), "有结果的比赛数": int(len(answers)),
        "有平博收盘的比赛数": int((bench["model"] == BENCHMARK).sum()) if not bench.empty else 0,
        "多家平均用收盘的比赛数": int((avg_rows["horizon"] == CLOSE).sum()),
        "多家平均用赛前代替的比赛数": int((avg_rows["horizon"] == PRE_SUBSTITUTE).sum()),
        "每个标杆分段的比赛数": periods["benchmark_period"].value_counts().to_dict() if benchmark else {},
        "标杆去水方法": fs.settings["backtest"]["benchmark_devig_method"] if benchmark else None,
        "没出预测的场数": {f"{m}|{h}": int(v) for (m, h), v in missing.items()},
        "耗时": timing,
        "设置": {k: fs.settings[k] for k in ("backtest", "models", "features")},
        "模型参数（最后一次拟合）": {m.name: m.params() for m in models},
    }
    order = ["match_id", "competition", "season", "match_date", "kickoff_utc", "home_team", "away_team", "horizon",
             "cutoff_utc", "model", "fit_cutoff_utc", *RESULT_COLS, "p_over25", "p_btts", "result", "home_goals",
             "away_goals", "log_loss", "brier", "rps", "hit", "brier_over25", "brier_btts"]
    if benchmark:
        order.insert(order.index("horizon"), "benchmark_period")
    scored = scored[order].sort_values(["match_date", "match_id", "model", "horizon"], ignore_index=True)
    return BacktestResult(scored, summary, vs, calib, info=info, models=models, summary_by_period=by_period)


def save(result: BacktestResult, name: str, settings: dict) -> Path:
    """存到 data/backtest/<name>/。CSV 用带 BOM 的 UTF-8，Excel 直接打开中文不乱码。"""
    out = resolve_path(settings["backtest"]["output_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    kw = {"index": False, "encoding": "utf-8-sig"}
    result.predictions.to_csv(out / "predictions.csv", float_format="%.6f", **kw)
    result.summary.to_csv(out / "summary.csv", float_format="%.5f", **kw)
    if not result.summary_vs_benchmark.empty:
        result.summary_vs_benchmark.to_csv(out / "summary_vs_benchmark.csv", float_format="%.5f", **kw)
    if not result.summary_by_period.empty:
        result.summary_by_period.to_csv(out / "summary_by_benchmark_period.csv", float_format="%.5f", **kw)
    result.calibration.to_csv(out / "calibration.csv", float_format="%.4f", **kw)
    for m in result.models:
        if hasattr(m, "ratings_table"):
            m.ratings_table().to_csv(out / f"ratings_{m.name}.csv", **kw)
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(result.info, f, ensure_ascii=False, indent=2, default=str)
    return out


def main() -> None:
    from src.common.config import load_settings
    from src.models.registry import build_model

    settings = load_settings()
    ap = argparse.ArgumentParser(description="按时间滚动回测（开发模式，只能回测开发赛季）")
    ap.add_argument("--leagues", nargs="+", default=list(settings["leagues"]["top"]), help="联赛代码，默认五大联赛")
    ap.add_argument("--first", type=int, required=True, help="第一个赛季（开始年份，2005 = 2005/06）")
    ap.add_argument("--last", type=int, default=settings["seasons"]["dev_last"], help="最后一个赛季")
    ap.add_argument("--models", nargs="+", default=["frequency", "frequency_poisson", "elo"])
    ap.add_argument("--horizons", nargs="+", type=float, default=None, help="开球前几小时，默认 settings 里的 24 6 1")
    ap.add_argument("--name", default=None, help="结果文件夹名，默认按联赛和赛季起")
    ap.add_argument("--no-benchmark", action="store_true", help="不算两个标杆（平博收盘、多家平均）")
    ap.add_argument("--db", default=None, help="数据库文件，默认 settings 里的")
    a = ap.parse_args()

    fs = FeatureStore(open_readonly(a.db), "开发", settings)
    models = [build_model(n, settings) for n in a.models]
    res = run_backtest(fs, a.leagues, a.first, a.last, models, a.horizons, benchmark=not a.no_benchmark)
    out = save(res, a.name or f"{'-'.join(a.leagues)}_{a.first}-{a.last}", settings)
    pd.set_option("display.width", 220)
    fmt = {"float_format": lambda v: f"{v:.4f}", "index": False}
    s = res.summary
    print(s[s["level"] != "联赛×赛季"].to_string(**fmt))
    if not res.summary_vs_benchmark.empty:
        v = res.summary_vs_benchmark
        print("\n和平博收盘同一批比赛：")
        print(v[v["level"] == "全部"].to_string(**fmt))
    if not res.summary_by_period.empty:
        v = res.summary_by_period
        print("\n按标杆分段（每段里模型和标杆同一批比赛）：")
        print(v[v["level"] == "全部"].drop(columns=["level", "competition", "season"]).to_string(**fmt))
    print("\n耗时：", json.dumps(res.info["耗时"], ensure_ascii=False))
    print(f"结果存在 {out}")


if __name__ == "__main__":
    main()
