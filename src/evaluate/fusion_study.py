"""用 A 版重选融合权重 w 和倾向门槛（需求文档第 5 节“用 A 版重选融合权重和倾向门槛怎么做”，跑之前写下的规则）。

一、要什么
  先跑回测，把 gb_l7 每场 T-24 的整张比分矩阵存下来：
    .venv\\Scripts\\python -m src.models.goals_backtest --name w9_gb_l7 --first 2005 --last 2023 --models gb_l7 --save-matrices
  这个程序读这次回测的 predictions.csv 和 matrices_gb_l7.npz，再经取数层按“本场”、用途「融合」取每场 T-24 能看到的赛前赔率。

二、怎么算
  1. 赔率 → 市场比分矩阵：胜平负（平博赛前，没有用多家平均赛前）、大 2.5 球（多家平均赛前）按比例去水，
     每场找 λ主、λ客、rho 让矩阵的四个数和赔率一样（src/common/markets.py 的 fit_market_dc）。两样缺一样的比赛不融合，单独数。
  2. 融合矩阵 = w × 市场矩阵 + (1 − w) × 模型矩阵。各玩法的概率都是矩阵格子加起来，所以对 w 是直线关系：
     先把两个矩阵各自的胜平负、大 2.5 球、双方进球、实际比分那一格、净胜球分布算好，每个 w 直接按比例混，21 档很快。
  3. 选 w：全部融合场次胜平负 log loss 平均最小的那档。另外每档报告其他分数，按联赛、按两段各自最好的 w。
  4. 滚动选 w：每个赛季只用它之前的赛季选 w（2006/07 起），和纯赔率（w = 1）按周成块 bootstrap 比；再各自和平博收盘比。
     滚动融合胜平负 log loss “更准”才上线用选出的 w，否则 w = 1。
  5. 倾向门槛：用最后定下的 w 的融合概率，每个玩法每场算“偏向多大”和“偏向那边实际怎么样”，按偏向分档，
     每档算平均和 95% 区间；门槛 = 从它往上每一档都“区间下限 > 0、平均不小于最低差距”的最低一档的下沿（没有就是“无明显倾向”）。

三、结果（都在 data/backtest/<回测名>/ 下面）
  fusion_odds.csv        每场取到的赔率（下次直接读，不再查数据库；取赔率的设置变了或加 --refresh-odds 会重新取，
                         设置存在 fusion_odds_settings.json）
  fusion_check.json      场数（融合 / 未融合）、市场矩阵还原误差、rho 取边界的场数、模型矩阵和 predictions.csv 对不对得上、
                         两种融合算法在胜平负上差多少、选出的 w、最后定的 w
  fusion_grid.csv        每档 w 的各项分数（全部、每个联赛、两段），“最好”标出每一份里胜平负 log loss 最小的那档
  fusion_rolling_w.csv   滚动选 w：每个赛季选出的 w
  fusion_compare.csv     滚动融合 对 纯赔率、两者各自 对 平博收盘（全部、每个联赛、两段）
  lean_bands.csv         每个玩法每一档的场数、“实际”的平均、95% 区间、满不满足
  lean_thresholds.csv    每个玩法的门槛、有倾向的场次比例、有倾向的场次里偏向那边实际发生的比例

用法：
  .venv\\Scripts\\python -m src.evaluate.fusion_study --run w9_gb_l7 --db data/snapshot.db
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.markets import dc_matrices, dc_rho_bounds, fit_market_dc
from src.common.settlement import settle_margin
from src.evaluate import metrics
from src.evaluate.compare import BETTER, block_ids, bootstrap_mean_diff, load_predictions, verdict
from src.features import FeatureStore, open_readonly
from src.features.market_odds import _check_known, odds_features

FOUR = ["mo_p_home", "mo_p_draw", "mo_p_away", "mo_over25"]      # 去水后的主胜、平、客胜、大 2.5 球
P3 = ["p_home", "p_draw", "p_away"]
SCORES = ["log_loss", "brier", "rps", "brier_over25", "brier_btts", "score_logloss"]
FUSED, MARKET, MODEL, PIN = "fused_rolling", "market_w1", "model_w0", "pinnacle_close"
LEAN_MARKETS = ["result", "draw", "over25", "btts", "ah"]


# ======================= 取赔率（经取数层） =======================
def fetch_odds(fs, targets: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每场（targets 一行一场：match_id、cutoff_utc）在自己的截止时间能看到的赛前赔率 → 每场一行。

    胜平负、大 2.5 球和步骤 8 的赔率特征同一条规则（odds_features）；亚盘只要 ah_book 那家的赛前亚盘线（主队的线，负数 = 主队让球）。
    每场都检查取到的“能知道的时间”不晚于截止时间。没有的是空值。
    """
    cfg, st = settings["market_odds"], settings["fusion_study"]
    books = list(dict.fromkeys([*cfg["result_books"], cfg["ou_book"], st["ah_book"]]))
    parts = []
    for mid, cutoff in targets[["match_id", "cutoff_utc"]].itertuples(index=False):
        o = fs.pre_match_odds(int(mid), cutoff, st["usage"], bookmaker=books, market=["1x2", "ou", "ah"])
        _check_known(o, cutoff, "本场赔率")
        parts.append(o)
    o = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    feats = odds_features(o, cfg)[["match_id", *FOUR, "mo_pinnacle"]]
    if len(o):
        ah = o[(o["source"] == cfg["source"]) & (o["bookmaker"] == st["ah_book"]) & (o["market"] == "ah")]
        ah = ah.dropna(subset=["line"]).drop_duplicates("match_id")[["match_id", "line", "home", "away"]]
    else:
        ah = pd.DataFrame({c: pd.Series(dtype=float) for c in ["match_id", "line", "home", "away"]})
    out = targets[["match_id"]].drop_duplicates().merge(feats, on="match_id", how="left")
    ah = ah.rename(columns={"line": "ah_line", "home": "ah_home_odds", "away": "ah_away_odds"})
    return out.merge(ah, on="match_id", how="left")


# ======================= 比分矩阵 → 各玩法（对 w 是直线关系） =======================
def derive(mats: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> dict:
    """一批比分矩阵（场数 × (n+1) × (n+1)）→ 每场：胜平负三个概率、大 2.5 球、双方进球、实际比分那一格的概率、
    净胜球分布（主 − 客，从 −n 到 +n）。这些都是格子加起来，所以 w × 市场 + (1 − w) × 模型 可以分别混。"""
    n = mats.shape[1] - 1
    i, j = np.indices(mats.shape[1:])
    diff = np.stack([(mats * (i - j == d)).sum(axis=(1, 2)) for d in range(-n, n + 1)], axis=1)
    k = np.arange(len(mats))
    return {"p3": np.stack([diff[:, n + 1:].sum(1), diff[:, n], diff[:, :n].sum(1)], axis=1),
            "over25": (mats * (i + j > 2)).sum(axis=(1, 2)),
            "btts": mats[:, 1:, 1:].sum(axis=(1, 2)),
            "cell": mats[k, np.minimum(hg, n), np.minimum(ag, n)],
            "diff": diff}


def mix(model: dict, market: dict, w) -> dict:
    """融合 = w × 市场 + (1 − w) × 模型（每样都一样混）。w 可以是一个数，也可以每场一个（滚动选 w）。"""
    w = np.asarray(w, float)
    out = {}
    for k in model:
        wk = w.reshape(w.shape + (1,) * (model[k].ndim - w.ndim)) if w.ndim else w
        out[k] = wk * market[k] + (1 - wk) * model[k]
    return out


def per_match_scores(d: dict, y: np.ndarray, over: np.ndarray, btts: np.ndarray) -> dict:
    """每场的分数：胜平负 log loss、Brier、RPS，大 2.5 球 Brier、双方进球 Brier，比分 log loss（越小越好）。"""
    p3 = d["p3"] / d["p3"].sum(axis=1, keepdims=True)          # float 误差补成正好 1
    return {"log_loss": metrics.log_loss_per_match(p3, y), "brier": metrics.brier_per_match(p3, y),
            "rps": metrics.rps_per_match(p3, y),
            "brier_over25": (d["over25"] - over) ** 2, "brier_btts": (d["btts"] - btts) ** 2,
            "score_logloss": -np.log(np.maximum(d["cell"], 1e-15))}


# ======================= 对齐：比赛、模型矩阵、赔率、市场矩阵 =======================
def build(pred: pd.DataFrame, ids: np.ndarray, mats: np.ndarray, odds: pd.DataFrame, settings: dict
          ) -> tuple[pd.DataFrame, dict, dict, dict]:
    """返回 (每场一行的比赛表, 模型各玩法, 市场各玩法, 检查信息)。只留“有比分、有模型矩阵、胜平负和大 2.5 球赔率都有”的比赛。"""
    st = settings["fusion_study"]
    t = pred[(pred["model"] == st["model"]) & (pred["horizon"] == st["horizon"])].dropna(subset=["result", *P3])
    t = t.drop_duplicates("match_id").merge(odds, on="match_id", how="left")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    has_mat = t["match_id"].isin(pos.index)
    ok = has_mat & t[FOUR].notna().all(axis=1)
    check = {"有预测的比赛": int(len(t)), "没有模型矩阵": int((~has_mat).sum()), "融合": int(ok.sum()),
             "未融合（胜平负或大 2.5 球赔率缺）": int((has_mat & ~ok).sum()),
             "未融合按联赛": t[has_mat & ~ok]["competition"].value_counts().to_dict(),
             "胜平负用平博的比赛": int((t.loc[ok, "mo_pinnacle"] == 1).sum())}
    t = t[ok].sort_values("match_id", ignore_index=True)
    m = mats[pos.loc[t["match_id"]].to_numpy()].astype(float)
    m /= m.sum(axis=(1, 2), keepdims=True)
    hg, ag = t["home_goals"].to_numpy(int), t["away_goals"].to_numpy(int)
    model = derive(m, hg, ag)
    # 模型矩阵算出来的胜平负要和 predictions.csv 一样（矩阵存的是同一次预测；float32 存的，差一点点）
    gap = float(np.abs(model["p3"] - t[P3].to_numpy(float)).max())
    if gap > 1e-4:
        raise ValueError(f"模型矩阵算出的胜平负和 predictions.csv 差了 {gap:.2e}，矩阵和预测不是同一次回测？")
    target = t[FOUR].to_numpy(float)
    lh, la, rho = fit_market_dc(target, n=m.shape[1] - 1)
    mk = dc_matrices(lh, la, rho, n=m.shape[1] - 1)
    market = derive(mk, hg, ag)
    got = np.column_stack([market["p3"], market["over25"]])
    lo, hi = dc_rho_bounds(lh, la)
    err = np.abs(got - target)
    check.update({"模型矩阵和 predictions.csv 胜平负最大差": gap,
                  "市场矩阵还原误差（最大）": float(err.max()), "市场矩阵还原误差（平均）": float(err.mean()),
                  "还原误差超过 0.001 的场数": int((err.max(axis=1) > 1e-3).sum()),
                  "rho 取了边界的场数": int(((rho <= lo + 1e-12) | (rho >= hi - 1e-12)).sum()),
                  "rho 中位数": float(np.median(rho)), "市场预期进球中位数（主 / 客）": [float(np.median(lh)),
                                                                                  float(np.median(la))]})
    t["y"] = metrics.encode_results(t["result"])
    t["over"] = (hg + ag > 2).astype(float)
    t["btts_yes"] = ((hg > 0) & (ag > 0)).astype(float)
    # 亚盘线只用“主盘口”：两边赔率都在 ah_main_odds 范围里（否则这场不算亚盘倾向，数一下有几场）
    lo_o, hi_o = st["ah_main_odds"]
    main = t[["ah_home_odds", "ah_away_odds"]].apply(lambda c: c.between(lo_o, hi_o)).all(axis=1)
    check["亚盘线不是主盘口、不用的场数"] = int((t["ah_line"].notna() & ~main).sum())
    check["没有亚盘线的场数"] = int(t["ah_line"].isna().sum())
    t["ah_line"] = t["ah_line"].where(main)
    t["period"] = np.where(t["season"] < st["period_split"], f"{st['period_split']} 之前", f"{st['period_split']} 起")
    return t, model, market, check


def w_grid(step: float) -> np.ndarray:
    return np.round(np.arange(0, 1 + step / 2, step), 10)


def segments(t: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """分几份报告：全部、每个联赛、两段。返回 (分法, 哪一份, 这一份的行在 t 里是不是)。"""
    out = [("全部", "全部", np.ones(len(t), bool))]
    out += [("联赛", c, (t["competition"] == c).to_numpy()) for c in sorted(t["competition"].unique())]
    out += [("两段", p, (t["period"] == p).to_numpy()) for p in sorted(t["period"].unique())]
    return out


# ======================= 选 w、滚动选 w =======================
def grid_table(t: pd.DataFrame, model: dict, market: dict, ws: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """每档 w × 每一份的各项分数平均。返回 (表, 每场每档的胜平负 log loss：场数 × 档数，滚动选 w 用)。"""
    y, over, btts = t["y"].to_numpy(), t["over"].to_numpy(), t["btts_yes"].to_numpy()
    ll = np.empty((len(t), len(ws)))
    rows = []
    segs = segments(t)
    for k, w in enumerate(ws):
        s = per_match_scores(mix(model, market, w), y, over, btts)
        ll[:, k] = s["log_loss"]
        for kind, name, mask in segs:
            rows.append({"分法": kind, "哪一份": name, "w": w, "n": int(mask.sum()),
                         **{m: float(s[m][mask].mean()) for m in SCORES}})
    g = pd.DataFrame(rows)
    best = g.groupby(["分法", "哪一份"])["log_loss"].transform("min")
    g["最好"] = np.isclose(g["log_loss"], best, rtol=0, atol=1e-15)
    return g, ll


def best_w(ll: np.ndarray, ws: np.ndarray) -> float:
    """胜平负 log loss 平均最小的那档 w（一样小时取小的 w）。"""
    return float(ws[int(np.argmin(ll.mean(axis=0)))])


def rolling_w(seasons: np.ndarray, ll: np.ndarray, ws: np.ndarray, first: int) -> dict[int, float]:
    """每个赛季（first 起）只用它之前的赛季选 w。之前一场都没有的赛季不选。"""
    out = {}
    for s in sorted(set(int(x) for x in seasons)):
        prior = seasons < s
        if s >= first and prior.any():
            out[s] = best_w(ll[prior], ws)
    return out


def compare_scores(sc: pd.DataFrame, base: str, new: str, mets, settings: dict) -> list[dict]:
    """sc：每场 × 每个模型一行（match_id、match_date、model、各项分数）。new 比 base：每场差（新 − 旧）按周成块 bootstrap。"""
    c = settings["compare"]
    rows = []
    for m in mets:
        a = sc[sc["model"] == base][["match_id", "match_date", m]].dropna()
        b = sc[sc["model"] == new][["match_id", m]].dropna()
        p = a.merge(b, on="match_id", suffixes=("_base", "_new"))
        if p.empty:
            continue
        blocks = block_ids(p["match_date"], c["block"])
        mean, lo, hi = bootstrap_mean_diff(p[f"{m}_new"] - p[f"{m}_base"], blocks, c["n_boot"], c["confidence"],
                                           c["seed"])
        rows.append({"base": base, "new": new, "metric": m, "n": len(p), "blocks": len(np.unique(blocks)),
                     "base_mean": p[f"{m}_base"].mean(), "new_mean": p[f"{m}_new"].mean(), "diff": mean,
                     "ci_low": lo, "ci_high": hi, "verdict": verdict(lo, hi), "primary": m == c["primary_metric"]})
    return rows


def rolling_compare(t: pd.DataFrame, model: dict, market: dict, rw: dict[int, float], pin: pd.DataFrame,
                    settings: dict) -> pd.DataFrame:
    """滚动融合 对 纯赔率（全部分数），两者各自 对 平博收盘（胜平负三个分数，只用三边都有的比赛）。每份都比一次。"""
    keep = t["season"].isin(list(rw)).to_numpy()
    tt = t[keep].reset_index(drop=True)
    sub = lambda d: {k: v[keep] for k, v in d.items()}
    mo, mk = sub(model), sub(market)
    fused = mix(mo, mk, tt["season"].map(rw).to_numpy(float))          # 每场用自己赛季滚动选出的 w
    y, over, btts = tt["y"].to_numpy(), tt["over"].to_numpy(), tt["btts_yes"].to_numpy()
    base_cols = tt[["match_id", "match_date", "competition", "period"]]
    sc = pd.concat([base_cols.assign(model=FUSED, **per_match_scores(fused, y, over, btts)),
                    base_cols.assign(model=MARKET, **per_match_scores(mk, y, over, btts))], ignore_index=True)
    p = pin[pin["match_id"].isin(tt["match_id"])].dropna(subset=["log_loss", "brier", "rps"]).drop_duplicates("match_id")
    p = base_cols.merge(p[["match_id", "log_loss", "brier", "rps"]], on="match_id").assign(model=PIN)
    rows = []
    for kind, name, mask in segments(tt):
        ids = set(tt.loc[mask, "match_id"])
        s = sc[sc["match_id"].isin(ids)]
        rows += [dict(r, 分法=kind, 哪一份=name) for r in compare_scores(s, MARKET, FUSED, SCORES, settings)]
        s2 = pd.concat([s[s["match_id"].isin(set(p["match_id"]))], p[p["match_id"].isin(ids)]], ignore_index=True)
        for m in (FUSED, MARKET):
            rows += [dict(r, 分法=kind, 哪一份=name) for r in compare_scores(s2, m, PIN, ["log_loss", "brier", "rps"],
                                                                              settings)]
    out = pd.DataFrame(rows)
    first = ["分法", "哪一份"]
    return out[[*first, *[c for c in out.columns if c not in first]]]


# ======================= 倾向门槛 =======================
_VALUE = {"win": 1.0, "half_win": 0.5, "push": 0.0, "half_loss": -0.5, "loss": -1.0}


def ah_lean(diff: np.ndarray, line: np.ndarray, gd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """亚盘：diff 是融合矩阵的净胜球分布（场数 × 2n+1，从 −n 到 +n），line 主队的盘口，gd 实际净胜球（主 − 客）。

    主队这边每个净胜球按这条线结算成 +1 / +0.5 / 0 / −0.5 / −1；胜算 = 赢的部分 ÷ (赢的部分 + 输的部分)（走盘不计，第 3 节）。
    客队这边正好反过来（同一条线，主队赢多少客队就输多少），所以客队胜算 = 1 − 主队胜算。
    返回 (偏向 = 两边较大的胜算 − 0.5, 偏向那边的结算结果)。没有线、全是走盘的场是空值。
    """
    n = (diff.shape[1] - 1) // 2
    ds = np.arange(-n, n + 1)
    strength, value = np.full(len(diff), np.nan), np.full(len(diff), np.nan)
    for ln in np.unique(line[~np.isnan(line)]):
        k = line == ln
        v = np.array([_VALUE[settle_margin(d, ln)] for d in ds])          # 主队这边每个净胜球的结算
        up, down = diff[k] @ np.clip(v, 0, None), diff[k] @ np.clip(-v, 0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = up / (up + down)
        home_side = share >= 0.5
        strength[k] = np.abs(share - 0.5)
        real = np.array([_VALUE[settle_margin(int(g), ln)] for g in gd[k]])
        value[k] = np.where(home_side, real, -real)
    return strength, value


def lean_values(t: pd.DataFrame, d: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每个玩法每场的 (偏向多大, 偏向那边实际怎么样)，规则见需求文档（跑之前写下的）。"""
    p3 = d["p3"] / d["p3"].sum(axis=1, keepdims=True)
    y = t["y"].to_numpy()
    order = np.argsort(-p3, axis=1, kind="stable")
    top, second = order[:, 0], order[:, 1]
    srt = np.take_along_axis(p3, order, axis=1)
    drew = (y == 1).astype(float)
    out = {"result": (srt[:, 0] - srt[:, 1], np.where(y == top, 1.0, np.where(y == second, -1.0, 0.0))),
           "draw": (p3[:, 1], drew - drew.mean())}
    for name, p, happened in [("over25", d["over25"], t["over"].to_numpy()), ("btts", d["btts"], t["btts_yes"].to_numpy())]:
        yes_side = p >= 0.5
        out[name] = (np.abs(p - 0.5), np.where(yes_side == (happened == 1), 1.0, -1.0))
    gd = (t["home_goals"] - t["away_goals"]).to_numpy(int)
    out["ah"] = ah_lean(d["diff"], t["ah_line"].to_numpy(float), gd)
    return out


def lean_bands(strength: np.ndarray, value: np.ndarray, dates: pd.Series, edges, min_gap: float,
               settings: dict) -> pd.DataFrame:
    """按“偏向多大”分档（edges[i] ≤ 偏向 < edges[i+1]，最后一档是 edges[-1] 以上），每档“实际”的平均和 95% 区间（按周成块）。
    比 edges[0] 小的场次不分档（平局风险从 0.26 起才看）。"""
    c = settings["compare"]
    ok = ~np.isnan(strength) & ~np.isnan(value)
    s, v, dt = strength[ok], value[ok], pd.Series(pd.to_datetime(dates)[ok].to_numpy())
    rows = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else np.inf
        k = (s >= lo - 1e-12) & (s < hi - 1e-12)
        row = {"下沿": lo, "上沿": hi, "n": int(k.sum())}
        if k.any():
            mean, cl, ch = bootstrap_mean_diff(v[k], block_ids(dt[k], c["block"]), c["n_boot"], c["confidence"], c["seed"])
            row.update({"平均": mean, "区间下限": cl, "区间上限": ch, "满足": bool(cl > 0 and mean >= min_gap)})
        else:
            row.update({"平均": np.nan, "区间下限": np.nan, "区间上限": np.nan, "满足": None})
        rows.append(row)
    return pd.DataFrame(rows)


def pick_threshold(bands: pd.DataFrame) -> float | None:
    """门槛 = 最低的那一档的下沿：从它往上每一档都满足。最高一档就不满足 → None（一律“无明显倾向”）。
    没有比赛的档没法判断，跳过（不算满足也不算不满足）；门槛那一档自己一定有比赛。"""
    thr = None
    for _, r in bands.iloc[::-1].iterrows():
        if r["n"] == 0:
            continue
        if not r["满足"]:
            break
        thr = float(r["下沿"])
    return thr


def lean_study(t: pd.DataFrame, d: dict, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个玩法：分档表、门槛、有倾向的场次比例、有倾向的场次里偏向那边实际发生的比例。"""
    st = settings["fusion_study"]
    vals = lean_values(t, d)
    band_rows, thr_rows = [], []
    for name in LEAN_MARKETS:
        s, v = vals[name]
        b = lean_bands(s, v, t["match_date"], st["lean_bands"][name], st["lean_min_gap"][name], settings)
        band_rows.append(b.assign(玩法=name))
        thr = pick_threshold(b)
        ok = ~np.isnan(s) & ~np.isnan(v)
        row = {"玩法": name, "门槛": thr, "最低差距": st["lean_min_gap"][name], "可算的场数": int(ok.sum())}
        if thr is not None:
            k = ok & (s >= thr - 1e-12)
            vk = v[k]
            if name == "draw":
                hit = float((vk > 0).mean())          # 有倾向（平局风险高）的场次里真的平了的比例
            elif name == "ah":
                hit = float(np.clip(vk, 0, None).sum() / np.abs(vk).sum())   # 胜算算法：赢半算半个，走盘不计
            else:
                hit = float((vk > 0).mean())          # 偏向那边真的发生的比例（胜平负：最高的那个发生）
            row.update({"有倾向的场数": int(k.sum()), "有倾向的比例": float(k.sum() / ok.sum()), "偏向那边发生的比例": hit,
                        "实际的平均": float(vk.mean())})
        thr_rows.append(row)
    bands = pd.concat(band_rows, ignore_index=True)
    return bands[["玩法", *[c for c in bands.columns if c != "玩法"]]], pd.DataFrame(thr_rows)


def per_market_gap(t: pd.DataFrame, model: dict, market: dict, w: float) -> dict:
    """两种融合算法差多少：比分矩阵上融合 对 每个玩法各自融合（w × 去水赔率 + (1 − w) × 模型概率），看胜平负和大 2.5 球。

    两边都是直线关系，所以差 = w × 市场矩阵的还原误差：市场矩阵还原得一样，两种算法就一样。返回最大差和 log loss 平均差。
    """
    y = t["y"].to_numpy()
    fused = mix(model, market, w)
    mat = fused["p3"] / fused["p3"].sum(axis=1, keepdims=True)
    per = w * t[["mo_p_home", "mo_p_draw", "mo_p_away"]].to_numpy(float) + (1 - w) * model["p3"]
    per = per / per.sum(axis=1, keepdims=True)
    per_over = w * t["mo_over25"].to_numpy(float) + (1 - w) * model["over25"]
    return {"w": w, "胜平负概率最大差": float(np.abs(mat - per).max()),
            "大 2.5 球概率最大差": float(np.abs(fused["over25"] - per_over).max()),
            "胜平负 log loss 平均差（矩阵 − 各自）": float(metrics.log_loss_per_match(mat, y).mean()
                                                  - metrics.log_loss_per_match(per, y).mean())}


# ======================= 主程序 =======================
def run(pred: pd.DataFrame, ids: np.ndarray, mats: np.ndarray, odds: pd.DataFrame, settings: dict) -> dict:
    """全部算完，返回各张结果表和检查信息（main 负责存文件）。"""
    st = settings["fusion_study"]
    t, model, market, check = build(pred, ids, mats, odds, settings)
    ws = w_grid(st["w_step"])
    grid, ll = grid_table(t, model, market, ws)
    w_all = best_w(ll, ws)
    rw = rolling_w(t["season"].to_numpy(), ll, ws, st["rolling_first"])
    pin = pred[(pred["model"] == PIN) & (pred["horizon"] == "收盘")]
    comp = rolling_compare(t, model, market, rw, pin, settings)
    main_row = comp[(comp["分法"] == "全部") & (comp["base"] == MARKET) & (comp["new"] == FUSED)
                    & (comp["metric"] == "log_loss")]
    better = bool(len(main_row)) and main_row["verdict"].iloc[0] == BETTER
    w_final = w_all if better else 1.0
    fused = mix(model, market, w_final)
    y = t["y"].to_numpy()
    check.update({"全部融合场次选出的 w": w_all, "滚动融合比纯赔率更准": better, "最后定的 w": w_final,
                  # 差 = w × 市场矩阵还原误差（见 per_market_gap），w 取 0.5 看一半模型一半赔率时两种算法差多少
                  "两种融合算法差多少（w = 0.5）": per_market_gap(t, model, market, 0.5),
                  "全部场次平局比例": float((y == 1).mean())})
    bands, thr = lean_study(t, fused, settings)
    rolling = pd.DataFrame({"season": list(rw), "w": list(rw.values())})
    return {"check": check, "grid": grid, "rolling": rolling, "compare": comp, "bands": bands, "thresholds": thr}


def main() -> None:
    settings = load_settings()
    st = settings["fusion_study"]
    ap = argparse.ArgumentParser(description="用 A 版重选融合权重 w 和倾向门槛")
    ap.add_argument("--run", required=True, help="回测结果文件夹名（data/backtest/ 下面，要有 matrices_<模型>.npz）")
    ap.add_argument("--db", default=None)
    ap.add_argument("--refresh-odds", action="store_true", help="不用上次存的 fusion_odds.csv，重新取赔率")
    a = ap.parse_args()
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / a.run
    pred = load_predictions(run_dir)
    npz = np.load(run_dir / f"matrices_{st['model']}.npz")
    ids, mats = npz["match_id"], npz["matrix"]
    cache, key_file = run_dir / "fusion_odds.csv", run_dir / "fusion_odds_settings.json"
    # 取赔率用到的设置变了（换公司、换线、换用途）就重新取，不用旧的
    key = json.dumps({"market_odds": settings["market_odds"], "usage": st["usage"], "ah_book": st["ah_book"],
                      "model": st["model"], "horizon": st["horizon"]}, ensure_ascii=False, sort_keys=True)
    if cache.exists() and key_file.exists() and key_file.read_text(encoding="utf-8") == key and not a.refresh_odds:
        odds = pd.read_csv(cache)
    else:
        targets = pred[(pred["model"] == st["model"]) & (pred["horizon"] == st["horizon"])][["match_id", "cutoff_utc"]]
        fs = FeatureStore(open_readonly(a.db), "开发", settings)
        odds = fetch_odds(fs, targets.drop_duplicates("match_id"), settings)
        odds.to_csv(cache, index=False, encoding="utf-8-sig")
        key_file.write_text(key, encoding="utf-8")
    res = run(pred, ids, mats, odds, settings)
    for key, name in [("grid", "fusion_grid"), ("rolling", "fusion_rolling_w"), ("compare", "fusion_compare"),
                      ("bands", "lean_bands"), ("thresholds", "lean_thresholds")]:
        res[key].to_csv(run_dir / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    (run_dir / "fusion_check.json").write_text(json.dumps(res["check"], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(json.dumps(res["check"], ensure_ascii=False, indent=2))
    g = res["grid"]
    print(g[g["最好"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    c = res["compare"]
    print(c[c["分法"] == "全部"].drop(columns=["primary"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(res["thresholds"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"结果存在 {run_dir}")


if __name__ == "__main__":
    main()
