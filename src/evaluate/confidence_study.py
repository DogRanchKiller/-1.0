"""定信心等级规则（需求文档第 5 节“信心等级规则怎么定”，跑之前写下的规则）。

一、信心等级是什么
  “这场给出的概率有多可信”，不是“偏向多大”。融合权重 w = 1，融合概率就是 T-24 的赛前赔率，
  所以问题变成：哪类比赛 T-24 的赛前赔率不太可信？

二、怎么量“可信”
  收盘赔率更准（离开球最近，知道的最多）。每场算 d = 赛前胜平负 log loss − 平博收盘胜平负 log loss。
  某类比赛 d 平均更大 = 这类比赛 T-24 时还有更多没定下来的信息 = T-24 的概率没那么可信。
  只用胜平负用平博赛前、又有平博收盘的比赛（同一家公司从赛前到收盘的变化）。

三、候选的降级因素（每场开球前就知道）
  分歧大：gb_l7 和赛前赔率在主胜、平、客胜上差得最多的那个 ≥ 6%
  前几轮：这个联赛这个赛季的前 10 轮（和 early_rounds.py 同一个数法）
  升班马：两队有一队上赛季不在这个联赛
  没有平博：胜平负只能用多家平均。这些比赛没有平博收盘，没法用 d，
           改在两家都有赛前赔率的比赛上比 多家平均赛前 − 平博赛前 的 log loss。
  判定：有这个因素的 d 平均 − 没有的 d 平均（没有平博那条是两家的平均差），按周成块 bootstrap，
        95% 区间下限 > 0 且差 ≥ min_diff 才“算数”。

四、等级
  从 3（●●● 高）开始，每碰上一个“算数”的因素降一级，最低 1（●○○ 低）。
  一家赔率都没有（只出模型概率）：分歧、没有平博两条没法算，不算；另外再降一级。

五、结果（data/backtest/<回测名>/ 下面）
  confidence_factors.csv   每个因素：两组场数、d 平均、差和 95% 区间、算不算数（全部和两段）
  confidence_levels.csv    开发赛季每个等级的场数、比例、d 平均、胜平负 log loss、Brier
  confidence_check.json    场数核对、算数的因素

用法（先跑 fusion_study，要用它存下的 fusion_odds.csv）：
  .venv\\Scripts\\python -m src.evaluate.confidence_study --run w9_gb_l7
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate import metrics
from src.evaluate.compare import block_ids, bootstrap_mean_diff, load_predictions
from src.evaluate.early_rounds import early_flags

P3 = ["p_home", "p_draw", "p_away"]
MO3 = ["mo_p_home", "mo_p_draw", "mo_p_away"]
FACTORS = ["gap", "early", "promoted", "no_pinnacle"]
FACTOR_NAMES = {"gap": "模型和赔率分歧大", "early": "赛季前几轮", "promoted": "有升班马", "no_pinnacle": "没有平博"}
LEVEL_NAMES = {3: "●●● 高", 2: "●●○ 中", 1: "●○○ 低"}


# ======================= 每场一行：概率、因素、d =======================
def promoted_flags(m: pd.DataFrame) -> pd.Series:
    """每场（按 match_id）有没有升班马：两队有一队上赛季不在这个联赛。

    “上赛季在不在”看上赛季这个联赛的比赛里有没有这队（主队、客队都算）。最早那个赛季没有上一季，当作不知道 → False。
    """
    teams = (m.melt(id_vars=["competition", "season"], value_vars=["home_team", "away_team"])
             .groupby(["competition", "season"])["value"].apply(set).to_dict())
    out = []
    for comp, season, home, away in m[["competition", "season", "home_team", "away_team"]].itertuples(index=False):
        last = teams.get((comp, season - 1))
        out.append(False if last is None else (home not in last or away not in last))
    return pd.Series(out, index=m["match_id"].to_numpy())


def build(pred: pd.DataFrame, odds: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每场一行：模型概率、赛前赔率概率（胜平负、用不用平博）、平博收盘、多家平均赛前、四个因素、d。"""
    st = settings["confidence_study"]
    m = pred[(pred["model"] == st["model"]) & (pred["horizon"] == st["horizon"])].drop_duplicates("match_id")
    m = m.dropna(subset=["result"]).copy()
    m["early"] = m["match_id"].map(early_flags(m, st["early_rounds"])).astype(bool)
    m["promoted"] = m["match_id"].map(promoted_flags(m)).astype(bool)
    m = m.merge(odds[["match_id", *MO3, "mo_pinnacle"]], on="match_id", how="left")
    close = pred[(pred["model"] == "pinnacle_close") & (pred["horizon"] == "收盘")].drop_duplicates("match_id")
    avg = pred[(pred["model"] == "market_avg") & (pred["horizon"] == "赛前代替")].drop_duplicates("match_id")
    m = m.merge(close[["match_id", *P3]].rename(columns=dict(zip(P3, ["c_home", "c_draw", "c_away"]))),
                on="match_id", how="left")
    m = m.merge(avg[["match_id", *P3]].rename(columns=dict(zip(P3, ["a_home", "a_draw", "a_away"]))),
                on="match_id", how="left")
    # predictions.csv 存到小数点后 6 位，三个概率加起来可能差 1e-6：重新归一
    for cols in (P3, MO3, ["c_home", "c_draw", "c_away"], ["a_home", "a_draw", "a_away"]):
        m[cols] = m[cols].div(m[cols].sum(axis=1), axis=0)
    y = metrics.encode_results(m["result"])
    m["y"] = y
    m["has_odds"] = m[MO3].notna().all(axis=1)
    pm = m[P3].to_numpy(float)
    pq = m[MO3].to_numpy(float)
    m["gap"] = m["has_odds"] & (np.abs(pm - pq).max(axis=1) >= st["gap"])        # 没有赔率时 NaN ≥ 0.06 是 False
    m["no_pinnacle"] = m["has_odds"] & (m["mo_pinnacle"] != 1)
    m["period"] = np.where(m["season"] < st["period_split"], f"{st['period_split']} 之前", f"{st['period_split']} 起")
    # 融合概率（w = 1 → 赔率；没有赔率 → 模型）
    fused = np.where(m["has_odds"].to_numpy()[:, None], pq, pm)
    m["ll_pre"] = metrics.log_loss_per_match(fused, y)
    m["brier_pre"] = metrics.brier_per_match(fused, y)
    # d：只用平博赛前 + 平博收盘的比赛
    m["d_ok"] = m["has_odds"] & (m["mo_pinnacle"] == 1) & m[["c_home", "c_draw", "c_away"]].notna().all(axis=1)
    m["ll_close"] = np.nan
    m["d"] = np.nan
    ok = m["d_ok"].to_numpy()
    if ok.any():
        m.loc[ok, "ll_close"] = metrics.log_loss_per_match(m.loc[ok, ["c_home", "c_draw", "c_away"]].to_numpy(float),
                                                           y[ok])
        m.loc[ok, "d"] = m.loc[ok, "ll_pre"] - m.loc[ok, "ll_close"]
    # 没有平博那条：两家都有赛前的比赛，多家平均赛前 − 平博赛前
    m["pair_ok"] = m["has_odds"] & (m["mo_pinnacle"] == 1) & m[["a_home", "a_draw", "a_away"]].notna().all(axis=1)
    m["pair_diff"] = np.nan
    ok = m["pair_ok"].to_numpy()
    if ok.any():
        m.loc[ok, "pair_diff"] = (metrics.log_loss_per_match(m.loc[ok, ["a_home", "a_draw", "a_away"]].to_numpy(float),
                                                             y[ok]) - m.loc[ok, "ll_pre"])
    return m


# ======================= 判定因素 =======================
def group_diff_ci(value, flag, blocks, settings: dict) -> tuple[float, float, float]:
    """有因素那组的平均 − 没有那组的平均，和 95% 区间（按块 bootstrap：抽块，两组在抽到的块里各自求平均再相减）。"""
    c = settings["compare"]
    value, flag = np.asarray(value, float), np.asarray(flag, bool)
    if flag.all() or not flag.any():
        raise ValueError("两组都要有比赛")
    _, inv = np.unique(np.asarray(blocks), return_inverse=True)
    k = inv.max() + 1
    s1 = np.bincount(inv, weights=value * flag, minlength=k)
    n1 = np.bincount(inv, weights=flag.astype(float), minlength=k)
    s0 = np.bincount(inv, weights=value * ~flag, minlength=k)
    n0 = np.bincount(inv, weights=(~flag).astype(float), minlength=k)
    rng = np.random.default_rng(c["seed"])
    r = np.empty(c["n_boot"])
    step = max(1, 2_000_000 // k)
    for i in range(0, c["n_boot"], step):
        idx = rng.integers(0, k, size=(min(step, c["n_boot"] - i), k))
        r[i:i + len(idx)] = s1[idx].sum(1) / n1[idx].sum(1) - s0[idx].sum(1) / n0[idx].sum(1)
    alpha = (1 - c["confidence"]) / 2
    lo, hi = np.quantile(r, [alpha, 1 - alpha])
    return float(value[flag].mean() - value[~flag].mean()), float(lo), float(hi)


def counts(lo: float, diff: float, min_diff: float) -> bool:
    """算不算数：区间下限 > 0，而且差不小于 min_diff（没法比的 NaN 不算数）。"""
    return bool(lo > 0 and diff >= min_diff)


def factor_table(t: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每个因素（全部、两段）：两组场数、平均、差和 95% 区间、算不算数。“全部”那一行决定算不算数，两段只报告。"""
    st = settings["confidence_study"]
    block = settings["compare"]["block"]
    rows = []
    for part, sub in [("全部", t), *sorted(t.groupby("period"), key=lambda kv: kv[0].endswith("起"))]:
        dsub = sub[sub["d_ok"]]
        for f in ["gap", "early", "promoted"]:
            flag = dsub[f].to_numpy(bool)
            if flag.any() and not flag.all():
                diff, lo, hi = group_diff_ci(dsub["d"], flag, block_ids(dsub["match_date"], block), settings)
            else:                                    # 一组没有比赛：没法比，不算数
                diff = lo = hi = np.nan
            rows.append({"因素": FACTOR_NAMES[f], "哪一份": part, "量的是": "d = 赛前 − 平博收盘",
                         "有的场数": int(flag.sum()), "没有的场数": int((~flag).sum()),
                         "有的平均": float(dsub.loc[flag, "d"].mean()), "没有的平均": float(dsub.loc[~flag, "d"].mean()),
                         "有的赛前 log loss": float(dsub.loc[flag, "ll_pre"].mean()),
                         "有的收盘 log loss": float(dsub.loc[flag, "ll_close"].mean()),
                         "差": diff, "区间下限": lo, "区间上限": hi})
        psub = sub[sub["pair_ok"]]
        if len(psub):
            diff, lo, hi = bootstrap_mean_diff(psub["pair_diff"], block_ids(psub["match_date"], block),
                                               settings["compare"]["n_boot"], settings["compare"]["confidence"],
                                               settings["compare"]["seed"])
        else:
            diff = lo = hi = np.nan
        rows.append({"因素": FACTOR_NAMES["no_pinnacle"], "哪一份": part, "量的是": "多家平均赛前 − 平博赛前",
                     "有的场数": int(len(psub)), "没有的场数": 0, "有的平均": np.nan, "没有的平均": np.nan,
                     "有的赛前 log loss": float(psub["ll_pre"].mean() + psub["pair_diff"].mean()),
                     "有的收盘 log loss": np.nan, "差": diff, "区间下限": lo, "区间上限": hi})
    out = pd.DataFrame(rows)
    out["算数"] = [counts(lo, d, st["min_diff"]) and part == "全部"
                 for lo, d, part in zip(out["区间下限"], out["差"], out["哪一份"])]
    return out


def levels(t: pd.DataFrame, counted: list[str], no_odds_down: int = 1) -> np.ndarray:
    """每场的等级：3 − 碰上的“算数”因素个数；一家赔率都没有再减 no_odds_down（settings 的 confidence.no_odds_downgrade）；最低 1。

    没有赔率时“分歧”“没有平博”两条没法算（build 里已经是 False），只看前几轮、升班马。
    """
    lv = np.full(len(t), 3)
    for f in counted:
        lv -= t[f].to_numpy(bool).astype(int)
    lv -= (~t["has_odds"].to_numpy(bool)).astype(int) * no_odds_down
    return np.maximum(lv, 1)


def level_table(t: pd.DataFrame) -> pd.DataFrame:
    """每个等级：场数、比例、有 d 的场数和 d 平均、胜平负 log loss、Brier（只报告，等级之间不直接比 log loss）。"""
    rows = []
    for lv in [3, 2, 1]:
        g = t[t["level"] == lv]
        dg = g[g["d_ok"]]
        rows.append({"等级": LEVEL_NAMES[lv], "场数": len(g), "比例": len(g) / len(t),
                     "有 d 的场数": len(dg), "d 平均": float(dg["d"].mean()) if len(dg) else np.nan,
                     "胜平负 log loss": float(g["ll_pre"].mean()) if len(g) else np.nan,
                     "Brier": float(g["brier_pre"].mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def run(pred: pd.DataFrame, odds: pd.DataFrame, settings: dict) -> dict:
    t = build(pred, odds, settings)
    ft = factor_table(t, settings)
    key = {v: k for k, v in FACTOR_NAMES.items()}
    counted = [key[n] for n in ft.loc[ft["算数"], "因素"]]
    t["level"] = levels(t, counted, settings["confidence"]["no_odds_downgrade"])
    lt = level_table(t)
    check = {"有预测和结果的比赛": int(len(t)), "有赛前赔率": int(t["has_odds"].sum()),
             "没有赔率（只出模型概率）": int((~t["has_odds"]).sum()),
             "胜平负用平博": int((t["mo_pinnacle"] == 1).sum()), "能算 d 的比赛（平博赛前 + 平博收盘）": int(t["d_ok"].sum()),
             "d 平均": float(t["d"].mean()), "两家都有赛前的比赛": int(t["pair_ok"].sum()),
             "各因素场数（全部比赛）": {FACTOR_NAMES[f]: int(t[f].sum()) for f in FACTORS},
             "算数的因素": [FACTOR_NAMES[f] for f in counted]}
    return {"table": t, "factors": ft, "levels": lt, "check": check, "counted": counted}


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="定信心等级规则")
    ap.add_argument("--run", required=True, help="回测结果文件夹名（data/backtest/ 下面，要有 fusion_odds.csv）")
    a = ap.parse_args()
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / a.run
    pred = load_predictions(run_dir)
    odds = pd.read_csv(run_dir / "fusion_odds.csv", encoding="utf-8-sig")
    res = run(pred, odds, settings)
    res["factors"].to_csv(run_dir / "confidence_factors.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    res["levels"].to_csv(run_dir / "confidence_levels.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    (run_dir / "confidence_check.json").write_text(json.dumps(res["check"], ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(json.dumps(res["check"], ensure_ascii=False, indent=2))
    print(res["factors"].to_string(index=False))
    print(res["levels"].to_string(index=False))
    print(f"结果存在 {run_dir}")


if __name__ == "__main__":
    main()
