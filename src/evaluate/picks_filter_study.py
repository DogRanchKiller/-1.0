"""竞彩二次筛选研究（工作树 P1，需求文档第 14 节“竞彩二次筛选研究怎么做”，规则是跑之前写下的）。

问题：第一层模型（gb_l7 的比分矩阵）每场给一个方向（胜平负 / 让球胜平负里概率最大的那个），
再加一个“第二层”给每场打一个信心分，每周只挑信心分最高的 k 场（主看 k = 5），
挑出来的这些场命中率是不是更高？按竞彩赔率虚拟投注长期是赚还是亏？盘口水位变化有没有用？

一条命令重跑：  .venv/Scripts/python -m src.evaluate.picks_filter_study
（要先有基础回测 data/backtest/pf_base，含两个模型的比分矩阵；第 14 节写了怎么跑）

数据怎么走（都经过取数层 src/features，按数据字典检查用途）：
  - 第一层的比分矩阵：已经跑好的滚动回测（每场在自己的 T-24 截止时间预测的）。
  - 特征：每场在自己的截止时间用「研究实验」取赛前赔率、变盘记录、赛程（休息天数）。
  - 标签（第一层方向对没对）：比分，用「研究实验」取以前比赛的比分；每个 28 天的块只用块开始前已经能知道的。
    结算、打分用的比分另外用「评估」取一次（和回测打分同一个用途），两次取到的必须一样。
  - 竞彩赔率：只用「研究评估」取，只拿来给虚拟注结算（赛后用），不进特征、不用来挑比赛。
    唯一例外是方法 D（按竞彩赔率算期望收益挑），只作参考，事先写明了。
    这些价格是 2026-10-30 到 11-02 回补抓的（事后抓的），这点在报告里写明。

方法：A 全买；B 按第一层概率每周前 k；C 二次筛选模型（LightGBM 估“这场方向会不会中”）每周前 k；
C− 同上但去掉盘口水位变化那组特征；D1 / D2 期望收益 > 0 的都买（参考）；E 每周随机 k 场；M 市场参考。
"""
from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.markets import dc_matrices, fit_market_dc
from src.common.odds import devig
from src.evaluate.compare import block_ids, bootstrap_mean_diff, load_predictions
from src.features import FeatureStore, open_readonly
from src.features.errors import LookAheadError
from src.features.known_time import to_utc_text
from src.features.market_odds import _check_known, odds_features
from src.features.rest_days import RestDays, other_season_system

MARKETS = ("result", "handicap")                       # 胜平负、让球胜平负（分开跑、分开下结论）
MARKET_NAMES = {"result": "竞彩胜平负", "handicap": "竞彩让球"}
SIDES = ("home", "draw", "away")                       # 方向 0 / 1 / 2 = 主胜 / 平 / 客胜（让球的是让球后的）
SIDE_TEXT = {"result": ["主胜", "平", "客胜"], "handicap": ["让球主胜", "让球平", "让球客胜"]}
RES = ["home", "draw", "away"]

# ---- 二次筛选模型的特征（第 14 节 G1–G5；G4 是盘口水位变化，C− 去掉的就是这一组） ----
G1 = ["p_pick", "p_second", "p_gap", "p_entropy", "pick_side"]          # 第一层自己的把握
G1_HANDICAP = ["exp_goal_diff"]                                         # 让球再加：第一层预期净胜球
G2 = ["dyn_max_diff", "dyn_agree", "dyn_p_pick"]                        # 两个第一层模型差多少
G3 = ["mk_p_pick", "mk_agree", "mk_gap_pick", "mk_overround", "mk_pinnacle"]   # 市场（赛前赔率）怎么看
G4 = ["mv_d_home", "mv_d_draw", "mv_d_away", "mv_d_pick", "mv_ah_line_d", "mv_in_d_pick",
      "mv_open_age_days", "mv_has_dqd"]                                 # 盘口水位变化（只用截止时间前能知道的价格）
G5 = ["home_rest_days", "away_rest_days", "home_after_euro", "away_after_euro",
      "home_games", "away_games", "league"]                             # 赛程、联赛
CATEGORICAL = ["league"]


def feature_columns(market: str, with_move: bool = True) -> list[str]:
    """这个玩法的二次筛选模型用哪些特征。with_move = False 就是 C−（去掉 G4）。竞彩赔率一列都不在里面。"""
    cols = [*G1, *(G1_HANDICAP if market == "handicap" else []), *G2, *G3, *(G4 if with_move else []), *G5]
    return cols


# ======================= 比分矩阵 → 胜平负 / 让球胜平负 =======================
def handicap_probs(mats: np.ndarray, handicap) -> np.ndarray:
    """一批比分矩阵（场数 × (n+1) × (n+1)，[主队进球, 客队进球]）+ 每场的让球数 → 每场 [让球主胜, 让球平, 让球客胜] 的概率。

    竞彩让球的写法：-1 = 主队让 1 球。主队进球 − 客队进球 + 让球数 > 0 算让球主胜，= 0 让球平，< 0 让球客胜。
    让球数 = 0 就是普通的胜平负。
    """
    mats = np.asarray(mats, dtype=float)
    h = np.broadcast_to(np.asarray(handicap, dtype=float), (len(mats),))
    i, j = np.indices(mats.shape[1:])
    adj = (i - j)[None, :, :] + h[:, None, None]          # 每一格让球之后的净胜球
    out = np.stack([(mats * (adj > 0)).sum(axis=(1, 2)), (mats * (adj == 0)).sum(axis=(1, 2)),
                    (mats * (adj < 0)).sum(axis=(1, 2))], axis=1)
    return out / out.sum(axis=1, keepdims=True)


def settle_outcome(home_goals, away_goals, handicap) -> np.ndarray:
    """实际结果 → 0 / 1 / 2（让球后主胜 / 平 / 客胜）；没有比分（改期、取消、没踢）的是 -1。"""
    hg = pd.to_numeric(pd.Series(home_goals), errors="coerce").to_numpy(float)
    ag = pd.to_numeric(pd.Series(away_goals), errors="coerce").to_numpy(float)
    m = hg - ag + np.broadcast_to(np.asarray(handicap, dtype=float), hg.shape)
    out = np.where(m > 0, 0, np.where(m == 0, 1, 2))
    return np.where(np.isnan(m), -1, out).astype(int)


def expected_goal_diff(mats: np.ndarray) -> np.ndarray:
    """每场的预期净胜球（主 − 客）。"""
    i, j = np.indices(mats.shape[1:])
    return (np.asarray(mats, float) * (i - j)[None]).sum(axis=(1, 2))


def entropy(p: np.ndarray) -> np.ndarray:
    """三个概率的熵：越大越“说不准”。"""
    q = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1)


# ======================= 第 1 步：读基础回测（第一层的比分矩阵） =======================
def load_base(settings: dict) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """读基础回测：每场一行（match_id、联赛、赛季、日期、截止时间）+ 两个模型的比分矩阵。

    返回的矩阵：{'main_ids': gb_l7 有矩阵的 match_id（从小到大）, 'main': 对应的矩阵, 'dyn': 同样的顺序的 dyn 矩阵}；
    dyn 缺矩阵的那几场是空值（dyn 那组特征是空值）。只留 T-24、主模型有矩阵的比赛。
    这里不用 predictions.csv 里的比分（比分另外经取数层取）。
    """
    st = settings["picks_filter_study"]
    run_dir = resolve_path(st["output_dir"]).parent / st["base_run"]
    pred = load_predictions(run_dir)
    pred = pred[(pred["horizon"] == st["horizon"]) & (pred["model"] == st["model"])]
    info = pred[["match_id", "competition", "season", "match_date", "cutoff_utc"]].drop_duplicates("match_id")
    loaded = {}
    for name in (st["model"], st["dyn_model"]):
        npz = np.load(run_dir / f"matrices_{name}.npz")
        m = npz["matrix"].astype(float)
        m /= m.sum(axis=(1, 2), keepdims=True)
        loaded[name] = (npz["match_id"].astype(int), m)
    ids, main = loaded[st["model"]]
    dids, dm = loaded[st["dyn_model"]]
    dyn = np.full_like(main, np.nan)
    dpos = pd.Series(np.arange(len(dids)), index=dids)
    have = np.isin(ids, dids)
    dyn[have] = dm[dpos.loc[ids[have]].to_numpy()]
    info = info[info["match_id"].isin(ids)].sort_values("match_id", ignore_index=True)
    info["match_id"] = info["match_id"].astype(int)
    return info, {"main_ids": ids, "main": main, "dyn": dyn}


# ======================= 第 2 步：经取数层取数 =======================
def match_info(fs: FeatureStore, settings: dict) -> pd.DataFrame:
    """要研究的这些联赛、赛季的比赛信息（主客队、最早开球时间），取数层按“本场”给（比赛信息，研究实验）。"""
    st, b = settings["picks_filter_study"], settings["backtest"]
    first = int(st["train_first_result"])
    start, end = f"{first}-{b['window_start_md']}", f"{int(st['eval_last']) + 1}-{b['window_end_md']}"
    t = fs.prediction_targets(start, end, st["usage"], competition=st["leagues"])
    t = t[(t["season"] >= first) & (t["season"] <= int(st["eval_last"]))]
    return t[["match_id", "home_team", "away_team", "kickoff_earliest_utc"]].reset_index(drop=True)


def fetch_results(fs: FeatureStore, settings: dict, usage: str) -> pd.DataFrame:
    """以前比赛的比分（截止时间 = 最后一个研究赛季的下一年 7 月 1 日，不碰考试赛季），每场一行 + 能知道的时间。"""
    st, b = settings["picks_filter_study"], settings["backtest"]
    cutoff = f"{int(st['eval_last']) + 1}-{b['window_end_md']} 00:00:00"
    r = fs.historical_matches(cutoff, usage, competition=st["leagues"])
    r = r[r["season"] <= int(st["eval_last"])]
    return r[["match_id", "home_goals", "away_goals", "known_at_utc"]].rename(columns={"known_at_utc": "result_known_utc"})
def fetch_lottery(fs: FeatureStore, ids, settings: dict) -> pd.DataFrame:
    """竞彩赔率（只用来给虚拟注结算），每场每种玩法：初始价格（第一次出的）、最终价格（最后的）、让球数。

    用途「研究评估」（赛后用，不是赛前用途）；截止时间 = settings 的 lottery_cutoff_utc（回补抓完之后）。
    同一场、同一种玩法只有一条价格记录 = 价格没变过（初始 = 最终）。
    返回列：match_id、product、handicap、lot_open_home/draw/away、lot_final_home/draw/away。
    """
    st = settings["picks_filter_study"]
    products = list(st["products"].values())
    rows = []
    for mid in ids:
        o = fs.lottery_odds(int(mid), st["lottery_cutoff_utc"], st["eval_usage"], product=products)
        if o.empty:
            continue
        for (prod, hcap), g in o.groupby(["product", "handicap"], sort=True):
            first, last = g.iloc[0], g.iloc[-1]          # 取数层按抓到的时间、导入顺序排好了
            rows.append((int(mid), prod, int(hcap), *first[RES].to_numpy(float), *last[RES].to_numpy(float)))
    cols = ["match_id", "product", "handicap", *[f"lot_open_{s}" for s in SIDES], *[f"lot_final_{s}" for s in SIDES]]
    out = pd.DataFrame(rows, columns=cols)
    # 一场一种玩法只该有一个让球数；万一有两个，用最后出的那个（数一下，写进报告）
    out.attrs["多个让球数的场次"] = int(out.duplicated(["match_id", "product"], keep=False).sum())
    return out.drop_duplicates(["match_id", "product"], keep="last").reset_index(drop=True)


def _devig_rows(df: pd.DataFrame) -> np.ndarray:
    """一张有 home、draw、away 赔率的表 → 按比例去水的三个概率（赔率不全或 ≤ 1 的行是空值）。"""
    x = df[RES].to_numpy(float)
    ok = np.isfinite(x).all(axis=1) & (x > 1).all(axis=1)
    out = np.full(x.shape, np.nan)
    if ok.any():
        out[ok] = devig(x[ok], "proportional")
    return out


def fetch_odds(fs: FeatureStore, base: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每场在自己的截止时间能看到的：赛前赔率特征（G3 用）和盘口水位变化特征（G4）。每场一行。

    赛前赔率：football-data 的赛前价格（按开球前 24 小时能知道算），和步骤 8 同一条规则（odds_features）。
    盘口水位变化：懂球帝的变盘记录只有“初盘”和“最新”两组价格，取截止时间之前已经变过的那些（取数层只给 changed_at ≤ 截止时间）；
      mv_d_*：同一家公司（Bet365、平博）懂球帝初盘 → football-data 赛前价格，去水概率变了多少（几家取平均）；
      mv_ah_line_d：亚盘主队盘口线从初盘到 football-data 赛前变了多少（负数 = 主队让得更多）；
      mv_in_d_pick：懂球帝自己截止时间前“最后一组 − 初盘”在方向那边的去水概率变化（T-24 时多半还没变，是 0）；
      mv_open_age_days：初盘离开球多少天；mv_has_dqd：有没有懂球帝变盘记录。
    每场都检查取到的“能知道的时间”不晚于截止时间，晚了就报错停下。
    """
    st, cfg = settings["picks_filter_study"], settings["market_odds"]
    books = list(dict.fromkeys([*cfg["result_books"], cfg["ou_book"], *st["move_books"]]))
    pre_parts, mv_parts = [], []
    for mid, cutoff in base[["match_id", "cutoff_utc"]].itertuples(index=False):
        o = fs.pre_match_odds(int(mid), cutoff, st["usage"], bookmaker=books, market=["1x2", "ou", "ah"])
        _check_known(o, cutoff, "赛前赔率")
        c = fs.odds_changes(int(mid), cutoff, st["usage"], bookmaker=st["move_books"], market=["1x2", "ah"])
        _check_known(c, cutoff, "变盘记录")
        pre_parts.append(o)
        mv_parts.append(c)
    pre = pd.concat(pre_parts, ignore_index=True) if pre_parts else pd.DataFrame()
    mv = pd.concat(mv_parts, ignore_index=True) if mv_parts else pd.DataFrame()
    out = base[["match_id"]].copy()
    out = out.merge(odds_features(pre, cfg), on="match_id", how="left")
    out = out.merge(_overround(pre, cfg), on="match_id", how="left")
    out = out.merge(_move_features(pre, mv, base, cfg, st), on="match_id", how="left")
    out["mv_has_dqd"] = out["mv_has_dqd"].fillna(0.0)
    return out


def _overround(pre: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """胜平负用的那家（平博优先，没有用多家平均）的抽水：1/主 + 1/平 + 1/客 − 1。"""
    if pre.empty:
        return pd.DataFrame({"match_id": pd.Series(dtype=int), "mk_overround": pd.Series(dtype=float)})
    o = pre[(pre["source"] == cfg["source"]) & (pre["market"] == "1x2")].dropna(subset=RES)
    o = o[(o[RES] > 1).all(axis=1)]
    order = {b: i for i, b in enumerate(cfg["result_books"])}
    o = o[o["bookmaker"].isin(order)].assign(_o=lambda d: d["bookmaker"].map(order))
    o = o.sort_values(["match_id", "_o"]).drop_duplicates("match_id")
    return pd.DataFrame({"match_id": o["match_id"].astype(int).to_numpy(),
                         "mk_overround": (1.0 / o[RES].to_numpy(float)).sum(axis=1) - 1.0})
def _main_line(df: pd.DataFrame) -> pd.DataFrame:
    """亚盘同一时间开了几条线时，取两边赔率最接近的那条（主盘口）。每场每家一行。"""
    d = df.dropna(subset=["line", "home", "away"]).copy()
    d["_gap"] = (d["home"] - d["away"]).abs()
    return d.sort_values(["match_id", "bookmaker", "_gap"]).drop_duplicates(["match_id", "bookmaker"])


def _move_features(pre: pd.DataFrame, mv: pd.DataFrame, base: pd.DataFrame, cfg: dict, st: dict) -> pd.DataFrame:
    """盘口水位变化特征（见 fetch_odds 的说明），每场一行（没有懂球帝记录的场是空值，mv_has_dqd = 0）。"""
    cols = ["mv_d_home", "mv_d_draw", "mv_d_away", "mv_in_d_home", "mv_in_d_draw", "mv_in_d_away",
            "mv_ah_line_d", "mv_open_age_days", "mv_has_dqd"]
    if mv is None or mv.empty:
        return pd.DataFrame({"match_id": pd.Series(dtype=int), **{c: pd.Series(dtype=float) for c in cols}})
    mv = mv.copy()
    mv["match_id"] = mv["match_id"].astype(int)
    out = pd.DataFrame({"match_id": np.sort(mv["match_id"].unique())})
    out["mv_has_dqd"] = 1.0
    # ---- 胜平负：初盘（最早一组）、截止时间前最后一组 ----
    r = mv[mv["market"] == "1x2"].dropna(subset=RES)
    r = r[(r[RES] > 1).all(axis=1)]
    first = r.groupby(["match_id", "bookmaker"], sort=True).head(1)
    last = r.groupby(["match_id", "bookmaker"], sort=True).tail(1)
    p_open = pd.DataFrame(_devig_rows(first), columns=SIDES, index=pd.MultiIndex.from_frame(first[["match_id", "bookmaker"]]))
    p_last = pd.DataFrame(_devig_rows(last), columns=SIDES, index=pd.MultiIndex.from_frame(last[["match_id", "bookmaker"]]))
    inner = (p_last - p_open).groupby(level="match_id").mean()
    out = out.merge(inner.add_prefix("mv_in_d_"), left_on="match_id", right_index=True, how="left")
    # 同一家公司：football-data 赛前价格（按 T-24 能知道）− 懂球帝初盘
    if pre is not None and len(pre):
        f = pre[(pre["source"] == cfg["source"]) & (pre["market"] == "1x2") & pre["bookmaker"].isin(st["move_books"])]
        f = f.dropna(subset=RES)
        f = f[(f[RES] > 1).all(axis=1)].drop_duplicates(["match_id", "bookmaker"])
        p_fd = pd.DataFrame(_devig_rows(f), columns=SIDES,
                            index=pd.MultiIndex.from_frame(f[["match_id", "bookmaker"]].astype({"match_id": int})))
        d = (p_fd - p_open).dropna().groupby(level="match_id").mean()
        out = out.merge(d.add_prefix("mv_d_"), left_on="match_id", right_index=True, how="left")
        # 亚盘主盘口线：football-data 赛前 − 懂球帝初盘（几家取平均）
        a_fd = _main_line(pre[(pre["source"] == cfg["source"]) & (pre["market"] == "ah")
                              & pre["bookmaker"].isin(st["move_books"])].astype({"match_id": int}))
        a = mv[mv["market"] == "ah"]
        if len(a):
            t0 = a.groupby(["match_id", "bookmaker"])["changed_at_utc"].transform("min")
            a_open = _main_line(a[a["changed_at_utc"] == t0])
            j = a_fd.merge(a_open, on=["match_id", "bookmaker"], suffixes=("_fd", "_op"))
            if len(j):
                ld = (j["line_fd"] - j["line_op"]).groupby(j["match_id"]).mean().rename("mv_ah_line_d")
                out = out.merge(ld, left_on="match_id", right_index=True, how="left")
    for c in cols:
        if c not in out:
            out[c] = np.nan
    # 初盘离开球多少天（最早一组变盘时间到最早可能的开球时间）
    t_open = pd.to_datetime(mv.groupby("match_id")["changed_at_utc"].min())
    ko = pd.to_datetime(base.set_index("match_id")["kickoff_earliest_utc"]).reindex(t_open.index)
    age = (ko - t_open).dt.total_seconds() / 86400.0
    out["mv_open_age_days"] = out["match_id"].map(age).astype(float)
    return out[["match_id", *cols]]


def fetch_context(fs: FeatureStore, base: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """G5：每场在自己的截止时间能知道的赛程信息 → 两队休息天数、上一场是不是欧战、本赛季已经踢了几场联赛。

    赛程用「休息天数和欧战后标签」取（和回测同一个用途、同一套 RestDays），按截止时间从早到晚增量取。
    base 要有 match_id、cutoff_utc、season、match_date、home_team、away_team。
    """
    rest = RestDays.from_settings(settings)
    st = settings["picks_filter_study"]
    other = other_season_system(settings, st["leagues"])
    margin = pd.Timedelta(days=float(settings["features"]["round_date_window_days"]) + 2)
    league_games: dict[tuple[str, int], list[int]] = {}
    seen: set[int] = set()
    last: str | None = None
    parts = []
    for cutoff, g in base.sort_values(["cutoff_utc", "match_id"]).groupby("cutoff_utc", sort=True):
        since = None if last is None else to_utc_text(pd.Timestamp(last) - margin)
        sched = fs.schedule(cutoff, "休息天数和欧战后标签", since=since, exclude_competitions=other)
        if len(sched) and (sched["known_at_utc"] > cutoff).any():
            raise LookAheadError(f"取数层返回了截止时间 {cutoff} 之后才开球的赛程，研究停止")
        last = cutoff
        rest.add(sched)
        for mid, season, d, home, away, cup in sched[["match_id", "season", "match_date", "home_team", "away_team",
                                                      "is_cup"]].itertuples(index=False, name=None):
            if int(mid) in seen or cup:
                continue
            seen.add(int(mid))
            day = RestDays._day(d)
            for team in (home, away):
                bisect.insort(league_games.setdefault((team, int(season)), []), day)
        a = rest.annotate(g[["match_id", "match_date", "home_team", "away_team", "season"]])
        for side in ("home", "away"):
            a[f"{side}_games"] = [float(bisect.bisect_left(league_games.get((t, int(s)), []), RestDays._day(d)))
                                  for t, s, d in a[[f"{side}_team", "season", "match_date"]].itertuples(index=False)]
        parts.append(a.drop(columns=["match_date", "home_team", "away_team", "season"]))
    return pd.concat(parts, ignore_index=True)


# ======================= 第 3 步：每个玩法一张表（一场一行：方向、特征、标签、竞彩结算价格） =======================
def _top2(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三个概率 → (最大的是哪一边, 最大的概率, 第二大的概率)。概率一样时取靠前的（主 > 平 > 客）。"""
    pick = np.argmax(p, axis=1)
    s = np.sort(p, axis=1)
    return pick, s[:, -1], s[:, -2]


def market_table(base: pd.DataFrame, mats: dict[str, np.ndarray], lot: pd.DataFrame, results: pd.DataFrame,
                 market: str, settings: dict) -> pd.DataFrame:
    """一个玩法（result 胜平负 / handicap 让球）的研究表，一场一行。

    base：每场一行（load_base 的信息 + 主客队、开球时间 + fetch_odds + fetch_context 的特征），row 列 = 在 mats 里第几行。
    mats：{'main': gb_l7 的比分矩阵, 'dyn': dyn 的比分矩阵}（和 base 的 row 对应）。
    lot：fetch_lottery 的结果（只用来结算，列名都以 lot_ 开头）；results：fetch_results 的比分和能知道的时间。
    让球玩法只留有竞彩让球的比赛（没有让球数就不知道方向和标签）；胜平负留全部（竞彩胜平负没有的，结算价格是空值）。
    """
    st = settings["picks_filter_study"]
    prod = st["products"][market]
    t = base.merge(lot[lot["product"] == prod].drop(columns="product"), on="match_id", how="left")
    if market == "handicap":
        t = t[t["handicap"].notna() & (t["season"] >= int(st["train_first_handicap"]))]
    else:
        t["handicap"] = 0.0
    t = t.sort_values("match_id", ignore_index=True)
    rows = t["row"].to_numpy(int)
    h = t["handicap"].to_numpy(float)
    main, dyn = mats["main"][rows], mats["dyn"][rows]
    p = handicap_probs(main, h)
    pick, p1, p2 = _top2(p)
    t["pick"] = pick
    t["pick_side"] = pick.astype(float)
    t["p_pick"], t["p_second"], t["p_gap"], t["p_entropy"] = p1, p2, p1 - p2, entropy(p)
    for k, s in enumerate(SIDES):
        t[f"p1_{s}"] = p[:, k]
    if market == "handicap":
        t["exp_goal_diff"] = expected_goal_diff(main)
    # G2：两个第一层模型差多少（dyn 缺矩阵的场是空值）
    has_dyn = np.isfinite(dyn).all(axis=(1, 2))
    pd_ = np.full_like(p, np.nan)
    if has_dyn.any():
        pd_[has_dyn] = handicap_probs(dyn[has_dyn], h[has_dyn])
    idx = np.arange(len(t))
    t["dyn_max_diff"] = np.abs(p - pd_).max(axis=1)
    t["dyn_agree"] = np.where(has_dyn, (np.nan_to_num(pd_, nan=-1).argmax(axis=1) == pick).astype(float), np.nan)
    t["dyn_p_pick"] = pd_[idx, pick]
    # G3：市场（赛前赔率，按 T-24 能知道）。胜平负直接用去水概率；让球用市场比分矩阵算让球三个结果
    four = ["mo_p_home", "mo_p_draw", "mo_p_away", "mo_over25"]
    pm = np.full_like(p, np.nan)
    if market == "result":
        ok = t[four[:3]].notna().all(axis=1).to_numpy()
        pm[ok] = t.loc[ok, four[:3]].to_numpy(float)
    else:
        ok = t[four].notna().all(axis=1).to_numpy()
        if ok.any():
            n = main.shape[1] - 1
            lh, la, rho = fit_market_dc(t.loc[ok, four].to_numpy(float), n=n)
            pm[ok] = handicap_probs(dc_matrices(lh, la, rho, n=n), h[ok])
    has_mk = np.isfinite(pm).all(axis=1)
    mk_pick = np.where(has_mk, np.nan_to_num(pm, nan=-1).argmax(axis=1), -1)
    t["mk_pick"] = mk_pick
    t["mk_pmax"] = np.where(has_mk, np.nanmax(np.where(has_mk[:, None], pm, 0.0), axis=1), np.nan)
    t["mk_p_pick"] = pm[idx, pick]
    t["mk_agree"] = np.where(has_mk, (mk_pick == pick).astype(float), np.nan)
    t["mk_gap_pick"] = t["p_pick"] - t["mk_p_pick"]
    t["mk_pinnacle"] = t["mo_pinnacle"]
    for k, s in enumerate(SIDES):
        t[f"mk_{s}"] = pm[:, k]
    # G4：方向那一边的水位变化（让球玩法用对应的主 / 平 / 客那一边，近似）
    for pre in ("mv_d_", "mv_in_d_"):
        arr = t[[f"{pre}{s}" for s in SIDES]].to_numpy(float)
        t[f"{pre}pick"] = arr[idx, pick]
    t["league"] = t["competition"].map({c: i for i, c in enumerate(st["leagues"])}).astype(float)
    # 标签和结算：比分经取数层（研究实验 / 研究评估分别取，见 study()）
    t = t.merge(results, on="match_id", how="left")
    t["outcome"] = settle_outcome(t["home_goals"], t["away_goals"], h)
    t["finished"] = t["outcome"] >= 0
    t["label"] = np.where(t["finished"], (t["outcome"] == t["pick"]).astype(float), np.nan)
    # 能买 = 打分赛季 + 有这个玩法的竞彩最终价格；训练池 = 胜平负 2012/13 起全部，让球 2017/18 起有让球数的
    fin = t[[f"lot_final_{s}" for s in SIDES]].to_numpy(float)
    t["eligible"] = (t["season"] >= int(st["eval_first"])) & (t["season"] <= int(st["eval_last"])) \
        & np.isfinite(fin).all(axis=1) & (fin > 1).all(axis=1)
    t["week"] = block_ids(t["match_date"], "week")
    # D 的期望收益（只作参考）：竞彩最终价格 × 去水概率 − 1，每场取最大的那一边
    t["ev1"], t["ev1_side"] = _best_ev(pm, fin)
    t["ev2"], t["ev2_side"] = _best_ev(p, fin)
    return t


def _best_ev(prob: np.ndarray, odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每场三边里期望收益最大的那一边：(期望收益, 哪一边)。概率或价格缺的场是 (空值, -1)。"""
    ev = prob * odds - 1.0
    ok = np.isfinite(ev).all(axis=1)
    side = np.where(ok, np.nan_to_num(ev, nan=-9).argmax(axis=1), -1)
    best = np.where(ok, np.nan_to_num(ev, nan=-9).max(axis=1), np.nan)
    return best, side


# ======================= 第 4 步：二次筛选模型，按 28 天一块滚动训练 =======================
def lgb_params(settings: dict) -> dict:
    """事先定好的树设置（第 14 节），不调。deterministic + 固定线程：同样的数据每次结果一样。"""
    g = settings["picks_filter_study"]["lgbm"]
    return {"objective": "binary", "num_leaves": int(g["num_leaves"]), "min_data_in_leaf": int(g["min_data_in_leaf"]),
            "learning_rate": float(g["learning_rate"]), "feature_fraction": float(g["feature_fraction"]),
            "bagging_fraction": float(g["bagging_fraction"]), "bagging_freq": int(g["bagging_freq"]),
            "lambda_l2": float(g["lambda_l2"]), "seed": int(g["seed"]), "deterministic": True,
            "force_row_wise": True, "num_threads": 4, "verbose": -1}


def block_index(weeks, block_days: int) -> np.ndarray:
    """每周（周一的日期）属于第几块：从最早那周的周一起，每 block_days 天一块。"""
    w = pd.to_datetime(pd.Series(weeks))
    return ((w - w.min()).dt.days // int(block_days)).to_numpy(int)


def walk_forward(t: pd.DataFrame, market: str, settings: dict) -> pd.DataFrame:
    """按块滚动训练二次筛选模型（C 全部特征、C− 去掉盘口水位变化），给每块里能买的比赛打分。

    每块：截止时间 = 块里最早一场能买的比赛的 T-24 截止时间；训练行 = 比分在这个时间之前已经能知道的比赛（标签 = 第一层方向中没中）。
    训练行少于 min_train_rows 的块不打分。返回：match_id、c_full、c_nomove、block、block_cutoff、n_train。
    """
    import lightgbm as lgb
    st = settings["picks_filter_study"]
    params = lgb_params(settings)
    ev = t[t["eligible"]].copy()
    ev["block"] = block_index(ev["week"], st["block_days"])
    labelled = t[t["label"].notna()]
    out = []
    for b, g in ev.groupby("block", sort=True):
        cut = str(g["cutoff_utc"].min())
        tr = labelled[labelled["result_known_utc"].astype(str) <= cut]
        row = pd.DataFrame({"match_id": g["match_id"].to_numpy(), "block": b, "block_cutoff": cut,
                            "n_train": len(tr), "c_full": np.nan, "c_nomove": np.nan})
        if len(tr) >= int(st["min_train_rows"]):
            for name, move in (("c_full", True), ("c_nomove", False)):
                cols = feature_columns(market, move)
                ds = lgb.Dataset(tr[cols].astype(float), tr["label"].to_numpy(float), feature_name=cols,
                                 categorical_feature=[c for c in CATEGORICAL if c in cols], free_raw_data=False)
                booster = lgb.train(params, ds, num_boost_round=int(st["lgbm"]["n_rounds"]))
                row[name] = booster.predict(g[cols].astype(float))
        out.append(row)
    cols = ["match_id", "block", "block_cutoff", "n_train", "c_full", "c_nomove"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


# ======================= 第 5 步：每周挑、结算 =======================
def select_top_k(df: pd.DataFrame, score: str, k: int) -> np.ndarray:
    """每周按 score 从高到低挑前 k 场（一样高的按 match_id 小的先挑；score 是空值的不挑；不到 k 场就全挑）。
    返回和 df 一样长的 True / False。"""
    d = df[["week", "match_id", score]].reset_index(drop=True)
    d = d[d[score].notna()].sort_values(["week", score, "match_id"], ascending=[True, False, True])
    chosen = d.groupby("week", sort=False).head(int(k)).index
    mask = np.zeros(len(df), dtype=bool)
    mask[chosen.to_numpy()] = True
    return mask


def settle(df: pd.DataFrame, mask, side) -> pd.DataFrame:
    """挑中的场（mask）按 side 那一边各买 1 注，用竞彩最终 / 初始价格结算。

    返回每注一行：week、season、match_id、side、stake（改期、取消的退款 = 0）、hit、odds_final、profit_final、
    odds_open、profit_open。中了赢“赔率 − 1”，不中输 1。
    """
    mask = np.asarray(mask, dtype=bool)
    side = np.broadcast_to(np.asarray(side), (len(df),))[mask].astype(int)
    d = df.loc[mask].reset_index(drop=True)
    idx = np.arange(len(d))
    fin = d[[f"lot_final_{s}" for s in SIDES]].to_numpy(float)[idx, side]
    opn = d[[f"lot_open_{s}" for s in SIDES]].to_numpy(float)[idx, side]
    stake = d["finished"].to_numpy(bool).astype(float)
    hit = ((d["outcome"].to_numpy(int) == side) & (stake > 0)).astype(float)
    return pd.DataFrame({"week": d["week"].to_numpy(), "season": d["season"].to_numpy(int),
                         "match_id": d["match_id"].to_numpy(int), "side": side, "stake": stake, "hit": hit,
                         "odds_final": fin, "profit_final": stake * (hit * fin - 1.0),
                         "odds_open": opn, "profit_open": stake * (hit * opn - 1.0)})


SUMS = ["stake", "hit", "profit_final", "profit_open"]


def weekly(bets: pd.DataFrame, weeks: np.ndarray) -> np.ndarray:
    """每周的 注数、命中数、按最终价格的盈亏、按初始价格的盈亏 → (周数 × 4)，周的顺序按 weeks（没下注的周是 0）。"""
    s = bets.groupby("week")[SUMS].sum().reindex(weeks).fillna(0.0)
    return s.to_numpy(float)


# ======================= 第 6 步：按周成块 bootstrap =======================
def boot_counts(n_weeks: int, n_boot: int, seed: int) -> np.ndarray:
    """按周成块 bootstrap 的抽样：每次有放回地抽 n_weeks 周，返回 (n_boot × n_weeks)，每格 = 这一周这次被抽中几次。
    同一周的全部注一起进出；所有方法都用同一张表 = 配对比较（同一次抽样用同样的周）。"""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_weeks, np.full(n_weeks, 1.0 / n_weeks), size=n_boot).astype(float)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _q(b: np.ndarray, conf: float) -> tuple[float, float]:
    """bootstrap 结果的分位数区间；一注都没有（全是空值）时返回空值。"""
    if not np.isfinite(b).any():
        return float("nan"), float("nan")
    a = (1 - conf) / 2
    return float(np.nanquantile(b, a)), float(np.nanquantile(b, 1 - a))


def ratio_ci(num_w, den_w, counts: np.ndarray, conf: float) -> tuple[float, float, float]:
    """比值（命中数 ÷ 注数、盈亏 ÷ 注数）的点估计和成块 bootstrap 区间：每次抽完重新加总再相除。"""
    num_w, den_w = np.asarray(num_w, float), np.asarray(den_w, float)
    est = float(_ratio(np.array([num_w.sum()]), np.array([den_w.sum()]))[0])
    return (est, *_q(_ratio(counts @ num_w, counts @ den_w), conf))


def ratio_diff_ci(num1, den1, num2, den2, counts: np.ndarray, conf: float) -> tuple[float, float, float]:
    """两个方法比值的差（1 − 2），配对：同一次抽样两边用同样的周。"""
    n1, d1, n2, d2 = (np.asarray(x, float) for x in (num1, den1, num2, den2))
    est = float((_ratio(np.array([n1.sum()]), np.array([d1.sum()])) - _ratio(np.array([n2.sum()]), np.array([d2.sum()])))[0])
    return (est, *_q(_ratio(counts @ n1, counts @ d1) - _ratio(counts @ n2, counts @ d2), conf))


class CICounter:
    """数一共算了多少个区间（报告里写出来：每 20 个区间约有 1 个碰巧不含 0）。"""

    def __init__(self):
        self.n = 0

    def add(self, k: int = 1):
        self.n += k


def method_row(name: str, w: np.ndarray, counts: np.ndarray, conf: float, ci: CICounter) -> dict:
    """一个方法的一行：注数、命中率、两种价格的 ROI，各带区间（w = weekly() 的结果或随机挑的期望）。"""
    st, hit, pf, po = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
    h = ratio_ci(hit, st, counts, conf)
    rf = ratio_ci(pf, st, counts, conf)
    ro = ratio_ci(po, st, counts, conf)
    ci.add(3)
    return {"方法": name, "注数": float(st.sum()), "命中率": h[0], "命中率下限": h[1], "命中率上限": h[2],
            "ROI": rf[0], "ROI下限": rf[1], "ROI上限": rf[2], "ROI_初始价": ro[0], "ROI_初始价下限": ro[1],
            "ROI_初始价上限": ro[2]}


def diff_row(name: str, w1: np.ndarray, w2: np.ndarray, counts: np.ndarray, conf: float, ci: CICounter) -> dict:
    """两个方法的命中率差和 ROI 差（1 − 2），配对成块 bootstrap。"""
    h = ratio_diff_ci(w1[:, 1], w1[:, 0], w2[:, 1], w2[:, 0], counts, conf)
    r = ratio_diff_ci(w1[:, 2], w1[:, 0], w2[:, 2], w2[:, 0], counts, conf)
    ci.add(2)
    return {"比较": name, "命中率差": h[0], "命中率差下限": h[1], "命中率差上限": h[2],
            "ROI差": r[0], "ROI差下限": r[1], "ROI差上限": r[2]}


def random_expectation(universe: pd.DataFrame, k: int, weeks: np.ndarray) -> np.ndarray:
    """方法 E 的期望（不带抽样误差）：每周随机挑 k 场的期望 = 这周全部能买的比赛的合计 × min(k, 场数) ÷ 场数。"""
    a = settle(universe, np.ones(len(universe), bool), universe["pick"].to_numpy())
    n = universe.groupby("week").size().reindex(weeks).fillna(0).to_numpy(float)
    f = np.where(n > 0, np.minimum(k, n) / np.where(n > 0, n, 1), 0.0)
    return weekly(a, weeks) * f[:, None]


def random_simulation(universe: pd.DataFrame, k: int, n_rep: int, seed: int) -> pd.DataFrame:
    """方法 E 的分布：每周从能买的比赛里随机挑 k 场（不放回），重复 n_rep 次，每次算命中率和两种 ROI。"""
    a = settle(universe, np.ones(len(universe), bool), universe["pick"].to_numpy())
    rng = np.random.default_rng(seed)
    key = rng.random((n_rep, len(a)))
    wk = pd.factorize(a["week"])[0]
    # 排序键 = 周的编号 + 一个 [0, 1) 的随机数：排完以后同一周的比赛挨在一起、周内顺序是随机的；
    # 每周排在前 min(k, 场数) 个的被挑中（同一次重复里每周正好挑这么多场）
    order = np.argsort(wk[None, :] + key, axis=1)
    sorted_wk = np.sort(wk)
    starts = np.r_[0, np.cumsum(np.bincount(wk))[:-1]]
    first_k = (np.arange(len(a)) - starts[sorted_wk]) < k        # 排好以后的第几个位置是不是本周前 k 个
    chosen = np.zeros(key.shape)
    np.put_along_axis(chosen, order, np.broadcast_to(first_k, order.shape).astype(float), axis=1)
    stake = chosen @ a["stake"].to_numpy()
    return pd.DataFrame({"命中率": chosen @ a["hit"].to_numpy() / stake,
                         "ROI": chosen @ a["profit_final"].to_numpy() / stake,
                         "ROI_初始价": chosen @ a["profit_open"].to_numpy() / stake})


def per_season_hit_diff(b1: pd.DataFrame, b2: pd.DataFrame) -> pd.Series:
    """每个赛季：方法 1 的命中率 − 方法 2 的命中率。"""
    def rate(b):
        s = b.groupby("season")[["hit", "stake"]].sum()
        return s["hit"] / s["stake"].where(s["stake"] > 0)
    return (rate(b1) - rate(b2)).dropna()


def log_loss_binary(p, y) -> np.ndarray:
    """每场的 log loss（二分类：中 = 1、不中 = 0）。"""
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(score, y) -> float:
    """AUC（按排名算，一样的分数取平均排名）：随便拿一场中的、一场没中的，中的那场分数更高的概率。"""
    s, y = np.asarray(score, float), np.asarray(y, float)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roi_verdict(lo: float, hi: float) -> str:
    """回报的写法（第 14 节判断标准）。"""
    if hi < 0:
        return "长期是亏的"
    if lo > 0:
        return "这 5 个赛季是赚的（事后价格的回测）"
    return "看不出赚还是亏"


def methods_for(u: pd.DataFrame, k: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每个方法挑哪些场、买哪一边：{名字: (mask, side)}。u = 能买的比赛（C 有结果的那些周）。"""
    pick = u["pick"].to_numpy()
    return {"A 全买": (np.ones(len(u), bool), pick),
            "B 第一层概率前k": (select_top_k(u, "p_pick", k), pick),
            "C 二次筛选前k": (select_top_k(u, "c_full", k), pick),
            "C− 去掉水位变化前k": (select_top_k(u, "c_nomove", k), pick),
            "M 市场前k（参考）": (select_top_k(u, "mk_pmax", k), u["mk_pick"].to_numpy()),
            "D1 平博EV>0（参考）": ((u["ev1"] > 0).to_numpy(), u["ev1_side"].to_numpy()),
            "D2 第一层EV>0（参考）": ((u["ev2"] > 0).to_numpy(), u["ev2_side"].to_numpy())}


def run_market(t: pd.DataFrame, market: str, settings: dict, ci: CICounter) -> dict:
    """一个玩法从滚动训练到全部结果（表格都是 DataFrame，写报告用）。"""
    st, cmp_ = settings["picks_filter_study"], settings["compare"]
    n_boot, conf, seed = int(cmp_["n_boot"]), float(cmp_["confidence"]), int(cmp_["seed"])
    k = int(st["k"])
    wf = walk_forward(t, market, settings)
    u = t[t["eligible"]].merge(wf, on="match_id", how="left")
    n_eligible = len(u)
    u = u[u["c_full"].notna()].sort_values(["week", "match_id"], ignore_index=True)
    weeks = np.array(sorted(u["week"].unique()))
    counts = boot_counts(len(weeks), n_boot, seed)
    res: dict = {"能买的场数": n_eligible, "C 有结果的场数": len(u), "周数": len(weeks),
                 "块数": int(wf["block"].nunique()), "没有C的块": int(wf.groupby("block")["c_full"].apply(
                     lambda s: s.isna().all()).sum()),
                 "训练行（第一块 / 最后一块）": [int(wf["n_train"].min()), int(wf["n_train"].max())],
                 "赛季": sorted(int(s) for s in u["season"].unique())}
    # 核对：返奖率（竞彩价格倒数之和的倒数）
    for kind in ("final", "open"):
        x = u[[f"lot_{kind}_{s}" for s in SIDES]].to_numpy(float)
        res[f"返奖率_{kind}"] = float(np.nanmean(1.0 / (1.0 / x).sum(axis=1)))
    # ---- 主结果 k = 5 ----
    bets, w = {}, {}
    for name, (mask, side) in methods_for(u, k).items():
        bets[name] = settle(u, mask, side)
        w[name] = weekly(bets[name], weeks)
    w["E 随机k（期望）"] = random_expectation(u, k, weeks)
    rows = [method_row(n, w[n], counts, conf, ci) for n in w]
    table = pd.DataFrame(rows)
    table["回报结论"] = [roi_verdict(lo, hi) for lo, hi in zip(table["ROI下限"], table["ROI上限"])]
    res["主表"] = table
    sim = random_simulation(u, k, int(st["n_random"]), seed)
    res["E 随机分布"] = {c: {"平均": float(sim[c].mean()), "2.5%": float(sim[c].quantile(0.025)),
                           "97.5%": float(sim[c].quantile(0.975))} for c in sim}
    E = "E 随机k（期望）"
    pairs = [("B − E（J1）", "B 第一层概率前k", E), ("C − B（J2）", "C 二次筛选前k", "B 第一层概率前k"),
             ("C − C−（看看）", "C 二次筛选前k", "C− 去掉水位变化前k"), ("C − A（影子条件）", "C 二次筛选前k", "A 全买"),
             ("C − E（看看）", "C 二次筛选前k", E), ("B − A（看看）", "B 第一层概率前k", "A 全买"),
             ("C− − B（看看）", "C− 去掉水位变化前k", "B 第一层概率前k")]
    diffs = pd.DataFrame([diff_row(n, w[a], w[b], counts, conf, ci) for n, a, b in pairs])
    res["比较"] = diffs
    # ---- 判断 J1、J2、J3 ----
    j1 = diffs.iloc[0]
    res["J1"] = {"命中率差": j1["命中率差"], "下限": j1["命中率差下限"], "上限": j1["命中率差上限"],
                 "结论": "第一层排序有用" if j1["命中率差下限"] > 0 else "没看出第一层排序有用"}
    j2 = diffs.iloc[1]
    seas = per_season_hit_diff(bets["C 二次筛选前k"], bets["B 第一层概率前k"])
    n_pos = int((seas > 0).sum())
    ok2 = bool(j2["命中率差下限"] > 0 and j2["命中率差"] >= float(st["min_hit_diff"])
               and n_pos >= int(st["min_same_sign_seasons"]))
    res["J2"] = {"命中率差": j2["命中率差"], "下限": j2["命中率差下限"], "上限": j2["命中率差上限"],
                 "每季差": {int(s): float(v) for s, v in seas.items()}, "差为正的赛季数": n_pos,
                 "结论": "C 比 B 好（有用）" if ok2 else "没看出 C 比 B 好"}
    fin = u[u["finished"]]
    d_ll = log_loss_binary(fin["c_full"], fin["label"]) - log_loss_binary(fin["c_nomove"], fin["label"])
    m3, lo3, hi3 = bootstrap_mean_diff(d_ll, fin["week"].to_numpy(), n_boot, conf, seed)
    ci.add(1)
    s3 = pd.Series(d_ll).groupby(fin["season"].to_numpy()).mean()
    n_neg = int((s3 < 0).sum())
    ok3 = bool(hi3 < 0 and n_neg >= int(st["min_same_sign_seasons"]))
    res["J3"] = {"log loss 差（C − C−，每场平均）": m3, "下限": lo3, "上限": hi3,
                 "每季差": {int(s): float(v) for s, v in s3.items()}, "差为负的赛季数": n_neg,
                 "k=5 命中率差 C − C−": float(diffs.iloc[2]["命中率差"]),
                 "结论": "盘口水位变化有帮助" if ok3 else "没看出盘口水位变化有帮助"}
    ca = diffs.iloc[3]
    res["影子"] = {"J2 有用": ok2, "C − A 的 ROI 差": ca["ROI差"], "下限": ca["ROI差下限"], "上限": ca["ROI差上限"],
                   "满足": bool(ok2 and ca["ROI差下限"] > 0)}
    res.update(_extra_tables(u, bets, weeks, counts, conf, ci, settings))
    res["每周名单"] = _weekly_list(u, market)
    res["每场"] = u
    return res


def _extra_tables(u: pd.DataFrame, bets: dict, weeks: np.ndarray, counts: np.ndarray, conf: float, ci: CICounter,
                  settings: dict) -> dict:
    """只看看的表：k = 3、10；排序好不好（log loss、AUC）；按 C 分数分 5 档；每季；被挑中那边“初始 → 最终”价格变化。"""
    st = settings["picks_filter_study"]
    out = {}
    rows = []
    for k in st["k_extra"]:
        k = int(k)
        ms = methods_for(u, k)
        for name in ("B 第一层概率前k", "C 二次筛选前k", "C− 去掉水位变化前k", "M 市场前k（参考）"):
            mask, side = ms[name]
            rows.append({"k": k, **method_row(name, weekly(settle(u, mask, side), weeks), counts, conf, ci)})
        rows.append({"k": k, **method_row("E 随机k（期望）", random_expectation(u, k, weeks), counts, conf, ci)})
    out["其他k"] = pd.DataFrame(rows)
    fin = u[u["finished"]]
    y = fin["label"].to_numpy(float)
    out["排序"] = pd.DataFrame([{"分数": name, "log loss": float(log_loss_binary(fin[c], y).mean()),
                               "AUC": auc(fin[c], y), "平均分数": float(fin[c].mean()), "实际命中率": float(y.mean())}
                              for name, c in (("C 全部特征", "c_full"), ("C− 去掉水位变化", "c_nomove"),
                                              ("第一层方向概率", "p_pick"))])
    a = settle(u, np.ones(len(u), bool), u["pick"].to_numpy()).assign(c=u["c_full"].to_numpy())
    a["档"] = pd.qcut(a["c"], int(st["score_bins"]), labels=False) + 1
    g = a.groupby("档")
    out["分档"] = pd.DataFrame({"场数": g["stake"].sum(), "C 平均分数": g["c"].mean(),
                              "命中率": g["hit"].sum() / g["stake"].sum(),
                              "ROI": g["profit_final"].sum() / g["stake"].sum(),
                              "平均最终价格": g["odds_final"].mean()}).reset_index()
    srows = []
    for name in ("A 全买", "B 第一层概率前k", "C 二次筛选前k", "C− 去掉水位变化前k"):
        s = bets[name].groupby("season")[["stake", "hit", "profit_final"]].sum()
        for season, r in s.iterrows():
            srows.append({"方法": name, "赛季": int(season), "注数": r["stake"], "命中率": r["hit"] / r["stake"],
                          "ROI": r["profit_final"] / r["stake"]})
    out["每季"] = pd.DataFrame(srows)
    drows = []
    for name, b in bets.items():
        b = b[b["stake"] > 0]
        drows.append({"方法": name, "注数": len(b), "平均初始价格": float(b["odds_open"].mean()),
                      "平均最终价格": float(b["odds_final"].mean()),
                      "最终比初始高的比例": float((b["odds_final"] > b["odds_open"]).mean()),
                      "最终比初始低的比例": float((b["odds_final"] < b["odds_open"]).mean()),
                      "ROI（最终）": float(b["profit_final"].sum() / max(len(b), 1)),
                      "ROI（初始）": float(b["profit_open"].sum() / max(len(b), 1))})
    out["价格变化"] = pd.DataFrame(drows)
    return out


def _weekly_list(u: pd.DataFrame, market: str) -> pd.DataFrame:
    """每周 C 挑的前 5 场（虚拟注名单，只是研究记录）。"""
    m = select_top_k(u, "c_full", 5)
    d = u.loc[m].copy()
    idx = np.arange(len(d))
    pick = d["pick"].to_numpy(int)
    d["方向"] = [SIDE_TEXT[market][p] for p in pick]
    d["最终价格"] = d[[f"lot_final_{s}" for s in SIDES]].to_numpy(float)[idx, pick]
    d["结果"] = np.where(~d["finished"], "退款", np.where(d["label"] == 1, "中", "没中"))
    d = d.sort_values(["week", "c_full"], ascending=[True, False])
    cols = ["week", "season", "match_date", "competition", "home_team", "away_team", "handicap", "方向", "c_full",
            "p_pick", "最终价格", "home_goals", "away_goals", "结果"]
    return d[cols].rename(columns={"week": "周一", "c_full": "C 分数", "p_pick": "第一层概率"})


# ======================= 第 7 步：把数据接起来、跑两个玩法 =======================
def build_tables(fs: FeatureStore, settings: dict, log=print) -> tuple[dict[str, pd.DataFrame], dict]:
    """读基础回测 + 经取数层取全部数据 → {玩法: 研究表}，以及数据检查信息。"""
    st = settings["picks_filter_study"]
    t0 = time.perf_counter()
    info, mats = load_base(settings)
    info = info[(info["season"] >= int(st["train_first_result"])) & (info["season"] <= int(st["eval_last"]))
                & info["competition"].isin(st["leagues"])].reset_index(drop=True)
    info = info.merge(match_info(fs, settings), on="match_id", how="inner")
    keep = info["match_id"].to_numpy()
    pos = pd.Series(np.arange(len(mats["main_ids"])), index=mats["main_ids"])
    info["row"] = pos.loc[keep].to_numpy()
    log(f"基础回测：{len(info)} 场（{time.perf_counter() - t0:.0f} 秒）")
    odds = fetch_odds(fs, info, settings)
    log(f"赛前赔率、变盘记录取完（{time.perf_counter() - t0:.0f} 秒）")
    ctx = fetch_context(fs, info, settings)
    log(f"赛程取完（{time.perf_counter() - t0:.0f} 秒）")
    base = info.merge(odds, on="match_id", how="left").merge(ctx, on="match_id", how="left")
    lot_ids = info.loc[info["season"] >= min(int(st["train_first_handicap"]), int(st["eval_first"])), "match_id"]
    lot = fetch_lottery(fs, lot_ids, settings)
    log(f"竞彩赔率取完：{lot['match_id'].nunique()} 场（{time.perf_counter() - t0:.0f} 秒）")
    labels = fetch_results(fs, settings, st["usage"])          # 训练标签用（研究实验）
    answers = fetch_results(fs, settings, "评估")              # 结算、打分用（「评估」：以前比赛的比分，和回测打分一样）
    if not labels.set_index("match_id").sort_index().equals(answers.set_index("match_id").sort_index()):
        raise ValueError("两种用途取到的比分不一样，停下检查")
    check = {"多个让球数的场次": lot.attrs.get("多个让球数的场次", 0),
             "dyn 缺矩阵的场数": int((~np.isfinite(mats["dyn"][info["row"].to_numpy()]).all(axis=(1, 2))).sum()),
             "有懂球帝变盘记录的场数": int(base["mv_has_dqd"].sum()),
             "有初盘→赛前变化的场数": int(base["mv_d_home"].notna().sum())}
    tables = {m: market_table(base, mats, lot, answers, m, settings) for m in MARKETS}
    return tables, check


def study(fs: FeatureStore, settings: dict, log=print, reuse: bool = False) -> dict:
    """整个研究：两个玩法分别跑，返回全部结果。取好的研究表存一份 tables.pkl（data/ 下，不进 Git），reuse 时直接读。"""
    cache = resolve_path(settings["picks_filter_study"]["output_dir"]) / "tables.pkl"
    if reuse and cache.exists():
        tables, check = pd.read_pickle(cache)
        log(f"用上次取好的研究表：{cache}")
    else:
        tables, check = build_tables(fs, settings, log)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((tables, check), cache)
    ci = CICounter()
    out = {"检查": check, "玩法": {}}
    for m in MARKETS:
        t0 = time.perf_counter()
        out["玩法"][m] = run_market(tables[m], m, settings, ci)
        log(f"{MARKET_NAMES[m]} 跑完（{time.perf_counter() - t0:.0f} 秒）")
    out["区间个数"] = ci.n
    return out


# ======================= 第 8 步：写结果和报告 =======================
def _fmt(x, pct: bool = False) -> str:
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        return f"{x * 100:.1f}%" if pct else f"{x:.4g}"
    return str(x)


PCT_COLS = {"命中率", "命中率下限", "命中率上限", "ROI", "ROI下限", "ROI上限", "ROI_初始价", "ROI_初始价下限",
            "ROI_初始价上限", "命中率差", "命中率差下限", "命中率差上限", "ROI差", "ROI差下限", "ROI差上限",
            "实际命中率", "平均分数", "C 平均分数", "最终比初始高的比例", "最终比初始低的比例", "ROI（最终）", "ROI（初始）"}


def md_table(df: pd.DataFrame) -> str:
    """DataFrame → markdown 表格（百分比的列写成 %）。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, c in PCT_COLS) for c, v in zip(cols, r)) + " |")
    return "\n".join(lines)


def _ci(est, lo, hi) -> str:
    return f"{est * 100:+.1f}%（{lo * 100:+.1f}% 到 {hi * 100:+.1f}%）"


def write_report(res: dict, settings: dict, out_dir: Path) -> Path:
    """写 report.md、每个玩法的每场表和每周名单、summary.json。"""
    st = settings["picks_filter_study"]
    out_dir.mkdir(parents=True, exist_ok=True)
    L = ["# 竞彩二次筛选研究（工作树 P1）", "",
         "规则见需求文档第 14 节“竞彩二次筛选研究怎么做”（跑之前写下）。这是研究，下的都是虚拟注，不连任何真实账户。",
         "竞彩赔率是 2026-10-30 到 11-02 事后回补抓的：只用来结算，不当特征、不用来挑比赛（方法 D 例外，只作参考）；"
         "“最终”“初始”价格都不一定是当时下注能买到的价格。", "",
         f"程序一共算了 {res['区间个数']} 个区间：每 20 个约有 1 个碰巧不含 0，只有 J1–J3 六个主要结论按事先的标准算数。", "",
         "数据检查：" + json.dumps(res["检查"], ensure_ascii=False), ""]
    summary = {"区间个数": res["区间个数"], "检查": res["检查"], "玩法": {}}
    for m, r in res["玩法"].items():
        name = MARKET_NAMES[m]
        L += [f"## {name}", "",
              f"能买的比赛 {r['能买的场数']} 场，其中 C 有结果的 {r['C 有结果的场数']} 场、{r['周数']} 周（赛季 {r['赛季']}）；"
              f"{r['块数']} 块，没有 C 的块 {r['没有C的块']} 个；训练行最少 / 最多 {r['训练行（第一块 / 最后一块）']}。",
              f"返奖率（1 ÷ 三个价格倒数之和，平均）：最终 {r['返奖率_final'] * 100:.1f}%，初始 {r['返奖率_open'] * 100:.1f}%"
              f"（应该约 {float(st['expected_payout']) * 100:.0f}%）。", "",
              f"### 主结果（每周 k = {st['k']} 场，区间 = 按周成块 bootstrap 95%）", "", md_table(r["主表"]), "",
              "E 随机挑 2,000 次的分布（平均、2.5%、97.5% 分位）："
              + json.dumps(r["E 随机分布"], ensure_ascii=False), "", "### 两两比较（配对）", "", md_table(r["比较"]), ""]
        j1, j2, j3, sh = r["J1"], r["J2"], r["J3"], r["影子"]
        L += ["### 主要结论", "",
              f"- J1 B − E 命中率差 {_ci(j1['命中率差'], j1['下限'], j1['上限'])} → **{j1['结论']}**",
              f"- J2 C − B 命中率差 {_ci(j2['命中率差'], j2['下限'], j2['上限'])}，差为正的赛季 {j2['差为正的赛季数']} / "
              f"{len(j2['每季差'])}（每季 {json.dumps({k: round(v * 100, 1) for k, v in j2['每季差'].items()})}）"
              f" → **{j2['结论']}**",
              f"- J3 每场 log loss 差 C − C− {j3['log loss 差（C − C−，每场平均）']:+.5f}（{j3['下限']:+.5f} 到 {j3['上限']:+.5f}），"
              f"差为负的赛季 {j3['差为负的赛季数']} / {len(j3['每季差'])}；k = 5 命中率差 C − C− "
              f"{j3['k=5 命中率差 C − C−'] * 100:+.1f}% → **{j3['结论']}**",
              f"- 影子运行条件：J2 有用 = {sh['J2 有用']}；C − A 的 ROI 差 {_ci(sh['C − A 的 ROI 差'], sh['下限'], sh['上限'])}"
              f" → **{'满足' if sh['满足'] else '不满足'}**", "",
              "### 只看看", "", "其他 k：", "", md_table(r["其他k"]), "", "排序好不好（全部能买、已踢完的比赛）：", "",
              md_table(r["排序"]), "", "按 C 分数分 5 档（全买）：", "", md_table(r["分档"]), "", "每季：", "",
              md_table(r["每季"]), "", "被挑中那一边竞彩价格“初始 → 最终”：", "", md_table(r["价格变化"]), ""]
        r["每场"].to_csv(out_dir / f"matches_{m}.csv", index=False, encoding="utf-8-sig")
        r["每周名单"].to_csv(out_dir / f"weekly_top5_{m}.csv", index=False, encoding="utf-8-sig")
        summary["玩法"][m] = {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v)
                            for k, v in r.items() if k not in ("每场", "每周名单")}
    L += ["## 已知的限制", "",
          "- 竞彩赔率是事后抓的；让球数假设开球前一天已经公布（【待核实】）。",
          "- 同一周里截止时间不一样：周日那场的特征可能带进了同一周更早几场的结果，只影响这周挑哪几场。",
          "- 懂球帝回补的变盘只有初盘和最新两组价格，“水位变化”只能用 初盘 → football-data 赛前价格 近似。",
          "- 让球玩法的水位变化用胜平负对应那一边的变化近似。"]
    path = out_dir / "report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="竞彩二次筛选研究（需求文档第 14 节）")
    ap.add_argument("--db", default=None, help="数据库文件（默认 settings.yaml 的 paths.database）")
    ap.add_argument("--reuse", action="store_true",
                    help="用上次取好的研究表（tables.pkl，取数约 10 分钟），只重跑训练、挑选和报告")
    args = ap.parse_args()
    settings = load_settings()
    conn = open_readonly(args.db)
    fs = FeatureStore(conn, "开发", settings)
    res = study(fs, settings, reuse=args.reuse)
    path = write_report(res, settings, resolve_path(settings["picks_filter_study"]["output_dir"]))
    print(f"报告：{path}")
    for m, r in res["玩法"].items():
        print(f"\n== {MARKET_NAMES[m]} ==")
        print(r["主表"][["方法", "注数", "命中率", "命中率下限", "命中率上限", "ROI", "ROI下限", "ROI上限"]].round(4)
              .to_string(index=False))
        print("J1:", r["J1"]["结论"], "| J2:", r["J2"]["结论"], "| J3:", r["J3"]["结论"], "| 影子:", r["影子"]["满足"])


if __name__ == "__main__":
    main()
