"""第 5 节“模型和赔率怎么一起用”的三个检验，和市场比分矩阵（需求文档第 5 节“三个检验和市场比分矩阵怎么做”）。

一、拿什么比
  模型：一次已经跑完的回测（goals_backtest 的结果文件夹）里某个模型的 T-24 预测（predictions.csv、scores_matrix.csv）。
  赔率：通过取数层（开发模式，用途“评估”）拿以前比赛的赛前、收盘胜平负赔率，按比例去水（和回测的两个标杆一样）。
    选哪家：settings.yaml market_checks.sources 按顺序，前面的有就用前面的（平博 → 多家平均，和“融合用哪家赔率”同一条规则）。
    football-data 的赛前赔率离开球约 1–3 天，当作 T-24 的赔率。截止时间是最后一个开发赛季之后的 7 月 1 日，考试赛季一场都拿不到。

二、三个检验
  检验一：整体谁更准。每场 log loss 差（模型 − 赛前赔率），按周成块 bootstrap 算 95% 区间（和 compare.py 一样）。
  检验二：分歧时谁更准。分歧 = 主胜、平、客胜三个概率里，模型和赔率差得最多的那个（绝对值）。
    按分歧大小分三组（3% 以内 / 3%–6% / 6% 以上），再分联赛、强弱队、模型更看好哪边。
  检验三：赔率会不会往模型那边走。只用同一家公司赛前、收盘都有的比赛。
    d = 模型 − 赛前赔率，m = 收盘 − 赛前（都是三个概率的差）。
    跟随比例 β = Σ(d·m) ÷ Σ(d·d)：0 = 赔率的变化和模型无关，1 = 收盘正好走到模型的位置。
    例：模型主胜 50%、赛前赔率 44%（d = +6%），收盘变成 45.2%（m = +1.2%）→ 这场走了差距的五分之一。

三、市场比分矩阵
  每场找两队预期进球 λ主、λ客，让独立泊松比分矩阵的“主胜、平、客胜、大 2.5 球”四个概率和去水后的赔率最接近
  （差的平方和最小）。用的是多家平均的赛前胜平负和大小球 2.5。
  怎么找：对每场同时做“高斯-牛顿法”（每一步按斜率解一个 2×2 方程，往误差变小的方向走），几万场一起算，几秒钟。
  然后看：还原误差有多大；这个矩阵的比分 log loss、大 2.5 球、双方进球和模型比（只报告）。

用法（先跑回测，再跑这个）：
  .venv\\Scripts\\python -m src.models.goals_backtest --name t3_dc --first 2005 --last 2023 --models dc --horizons 24
  .venv\\Scripts\\python -m src.evaluate.market_checks --run t3_dc --model dc
结果在 data/backtest/<run>/market_checks/：
  check1.csv  check2.csv  check3.csv  market_matrix.csv（汇总）  market_matrix_matches.csv（每场 λ）  run_info.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.markets import fit_market_lambdas, matrix_probs  # noqa: F401（别处按 mc.xxx 用）
from src.common.odds import devig
from src.evaluate import metrics
from src.evaluate.compare import block_ids, bootstrap_mean_diff, verdict
from src.evaluate.devig_compare import dev_cutoff
from src.features import FeatureStore, open_readonly
from src.models.goals import score_matrices

P = ["p_home", "p_draw", "p_away"]
Q = ["q_home", "q_draw", "q_away"]          # 赛前赔率去水后的概率
C = ["c_home", "c_draw", "c_away"]          # 收盘赔率去水后的概率
ODDS = ["home", "draw", "away"]
HORIZON = "T-24"


# ======================= 取赔率 =======================
def _odds_1x2(fs: FeatureStore, cutoff: str, comps: list[str], timing: str, book: str, method: str) -> pd.DataFrame:
    """这家公司每场的胜平负赔率去水后的三个概率。赔率缺一个、或不大于 1 的比赛不要。"""
    o = fs.historical_odds(cutoff, "评估", timing=timing, competition=comps, bookmaker=book, market="1x2")
    o = o.dropna(subset=ODDS)
    o = o[(o[ODDS] > 1).all(axis=1)].drop_duplicates("match_id")
    p = devig(o[ODDS].to_numpy(dtype=float), method)
    return pd.DataFrame({"match_id": o["match_id"].to_numpy(), "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2]})


def _odds_ou(fs: FeatureStore, cutoff: str, comps: list[str], book: str, method: str) -> pd.DataFrame:
    """这家公司每场赛前大小球 2.5 去水后的“大 2.5 球”概率。"""
    o = fs.historical_odds(cutoff, "评估", timing="pre", competition=comps, bookmaker=book, market="ou")
    o = o[np.isclose(o["line"], 2.5)].dropna(subset=["over", "under"])
    o = o[(o[["over", "under"]] > 1).all(axis=1)].drop_duplicates("match_id")
    p = devig(o[["over", "under"]].to_numpy(dtype=float), method)
    return pd.DataFrame({"match_id": o["match_id"].to_numpy(), "q_over25": p[:, 0]})


def load_market(fs: FeatureStore, settings: dict, comps: list[str]) -> pd.DataFrame:
    """每场一行：选出的赛前赔率（q_*、q_source）、同一家的收盘（c_*，没有就空），多家平均赛前的胜平负和大 2.5 球（m_*）。"""
    c, method = settings["market_checks"], settings["backtest"]["benchmark_devig_method"]
    cutoff = dev_cutoff(settings)
    pre = {b: _odds_1x2(fs, cutoff, comps, "pre", b, method) for b in c["sources"]}
    close = {b: _odds_1x2(fs, cutoff, comps, "close", b, method) for b in c["sources"]}
    parts, seen = [], set()
    for b in c["sources"]:                                   # 按顺序：前面的公司有赛前赔率就用前面的
        q = pre[b][~pre[b]["match_id"].isin(seen)]
        seen |= set(q["match_id"])
        q = q.rename(columns=dict(zip(["p0", "p1", "p2"], Q))).assign(q_source=b)
        q = q.merge(close[b].rename(columns=dict(zip(["p0", "p1", "p2"], C))), on="match_id", how="left")
        parts.append(q)
    out = pd.concat(parts, ignore_index=True)
    mb = c["matrix_source"]
    m = pre[mb].rename(columns={"p0": "m_home", "p1": "m_draw", "p2": "m_away"})
    m = m.merge(_odds_ou(fs, cutoff, comps, mb, method).rename(columns={"q_over25": "m_over25"}), on="match_id")
    return out.merge(m, on="match_id", how="left")

# ======================= 小工具 =======================
def _boot(diff, dates, settings: dict) -> tuple[float, float, float]:
    """每场差值的平均和 95% 区间（按周成块 bootstrap，设置和 compare.py 一样）。"""
    c = settings["compare"]
    return bootstrap_mean_diff(np.asarray(diff, float), block_ids(dates, c["block"]), c["n_boot"], c["confidence"],
                               c["seed"])


def ratio_ci(num, den, dates, settings: dict) -> tuple[float, float, float]:
    """比例 Σnum ÷ Σden 和它的 95% 区间（按周成块 bootstrap：抽周，再把抽到的周的 num、den 各自加起来相除）。"""
    c = settings["compare"]
    _, inv = np.unique(block_ids(dates, c["block"]), return_inverse=True)
    sn = np.bincount(inv, weights=np.asarray(num, float))
    sd = np.bincount(inv, weights=np.asarray(den, float))
    k = len(sn)
    rng = np.random.default_rng(c["seed"])
    r = np.empty(c["n_boot"])
    step = max(1, 2_000_000 // k)
    for i in range(0, c["n_boot"], step):
        idx = rng.integers(0, k, size=(min(step, c["n_boot"] - i), k))
        r[i:i + len(idx)] = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    alpha = (1 - c["confidence"]) / 2
    lo, hi = np.quantile(r, [alpha, 1 - alpha])
    return float(sn.sum() / sd.sum()), float(lo), float(hi)


def add_groups(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """加几列：每场分数（模型、赛前、收盘）、分歧大小和分组、强弱队、模型更看好哪边。"""
    c = settings["market_checks"]
    df = df.copy()
    y = metrics.encode_results(df["result"])
    pm, pq = df[P].to_numpy(float), df[Q].to_numpy(float)
    df["ll_q"] = metrics.log_loss_per_match(pq, y)
    df["brier_q"] = metrics.brier_per_match(pq, y)
    df["rps_q"] = metrics.rps_per_match(pq, y)
    has_c = df[C].notna().all(axis=1).to_numpy()
    for col, f in [("ll_c", metrics.log_loss_per_match), ("brier_c", metrics.brier_per_match),
                   ("rps_c", metrics.rps_per_match)]:
        df[col] = np.nan
        if has_c.any():
            df.loc[has_c, col] = f(df.loc[has_c, C].to_numpy(float), y[has_c])
    d = pm - pq
    df["gap"] = np.abs(d).max(axis=1)
    lo, hi = c["gap_bins"]
    labels = [f"{lo:.0%} 以内", f"{lo:.0%}–{hi:.0%}", f"{hi:.0%} 以上"]
    df["gap_group"] = pd.Categorical(np.select([df["gap"] < lo, df["gap"] < hi], labels[:2], labels[2]),
                                     categories=labels, ordered=True)
    df["strength"] = np.where(pq.max(axis=1) >= c["lopsided"], "强弱分明", "势均力敌")
    df["side"] = np.where(d[:, 0] - d[:, 2] > 0, "模型更看好主队", "模型更看好客队")
    return df


def _splits(df: pd.DataFrame, keys: list[str]):
    """全部 + 按每个 key 分组：给出 (分组方式, 组名, 这组的比赛)。"""
    yield "全部", "全部", df
    for k in keys:
        for v, g in df.groupby(k, sort=True, observed=True):
            yield k, v, g


# ======================= 三个检验 =======================
def check1(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """检验一：模型 − 赔率（赛前 / 收盘）的每场分数差。差 < 0 = 模型更准。"""
    rows = []
    for split, name, g in _splits(df, ["q_source", "competition"]):
        for against, suffix in [("赛前", "q"), ("收盘", "c")]:
            for metric, mcol in [("log_loss", "log_loss"), ("brier", "brier"), ("rps", "rps")]:
                ocol = {"log_loss": "ll", "brier": "brier", "rps": "rps"}[metric] + "_" + suffix
                gg = g.dropna(subset=[ocol, mcol])
                if gg.empty:
                    continue
                mean, lo, hi = _boot(gg[mcol] - gg[ocol], gg["match_date"], settings)
                rows.append({"分组方式": split, "组": name, "和谁比": against, "metric": metric, "n": len(gg),
                             "模型": gg[mcol].mean(), "赔率": gg[ocol].mean(), "差(模型-赔率)": mean,
                             "ci_low": lo, "ci_high": hi, "结论(模型)": verdict(lo, hi)})
    return pd.DataFrame(rows)


def check2(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """检验二：按分歧大小分组（再分联赛、强弱队、模型更看好哪边），模型和赛前赔率谁的 log loss 更小。"""
    rows = []
    for split, name, g in _splits(df, ["competition", "strength", "side"]):
        for gap, gg in g.groupby("gap_group", observed=True):
            mean, lo, hi = _boot(gg["log_loss"] - gg["ll_q"], gg["match_date"], settings)
            rows.append({"分组方式": split, "组": name, "分歧": gap, "n": len(gg),
                         "平均分歧": gg["gap"].mean(), "模型log_loss": gg["log_loss"].mean(),
                         "赔率log_loss": gg["ll_q"].mean(), "差(模型-赔率)": mean, "ci_low": lo, "ci_high": hi,
                         "结论(模型)": verdict(lo, hi), "模型更准的场数比例": (gg["log_loss"] < gg["ll_q"]).mean()})
    return pd.DataFrame(rows)


def check3(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """检验三：同一家公司从赛前到收盘，赔率往模型那边走了多少（跟随比例 β），和往模型方向走的比赛比例。"""
    c = settings["market_checks"]
    g0 = df.dropna(subset=C).copy()
    d = g0[P].to_numpy(float) - g0[Q].to_numpy(float)
    m = g0[C].to_numpy(float) - g0[Q].to_numpy(float)
    g0["dm"], g0["dd"] = (d * m).sum(axis=1), (d * d).sum(axis=1)
    g0["moved"] = np.abs(m).sum(axis=1) > 1e-9
    g0["d_abs"], g0["m_abs"] = np.abs(d).sum(axis=1) / 2, np.abs(m).sum(axis=1) / 2
    rows = []
    for split, name, g in _splits(g0, ["q_source", "competition", "gap_group"]):
        beta, lo, hi = ratio_ci(g["dm"], g["dd"], g["match_date"], settings)
        row = {"分组方式": split, "组": name, "n": len(g), "跟随比例β": beta, "ci_low": lo, "ci_high": hi,
               "结论": "赔率往模型那边走" if lo > 0 else ("赔率往反方向走" if hi < 0 else "看不出"),
               "平均差距(模型-赛前)": g["d_abs"].mean(), "平均变化(收盘-赛前)": g["m_abs"].mean(),
               "赛前log_loss": g["ll_q"].mean(), "收盘log_loss": g["ll_c"].mean()}
        big = g[(g["gap"] >= c["follow_min_gap"]) & g["moved"]]
        if len(big):
            share, slo, shi = ratio_ci((big["dm"] > 0).astype(float), np.ones(len(big)), big["match_date"], settings)
            row.update({f"分歧≥{c['follow_min_gap']:.0%}且赔率变了的场数": len(big), "往模型方向走的比例": share,
                        "比例ci_low": slo, "比例ci_high": shi})
        rows.append(row)
    return pd.DataFrame(rows)


# ======================= 市场比分矩阵 =======================
# matrix_probs、fit_market_lambdas 搬到了 src/common/markets.py（步骤 8 的赔率特征在取数层也要用，取数层不能引用模型），
# 这里照旧能用 mc.matrix_probs、mc.fit_market_lambdas。


def market_matrix(df: pd.DataFrame, scores: pd.DataFrame, settings: dict
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """反推每场的市场矩阵，算还原误差和分数，和模型比。返回 (汇总, 还原误差, 每场)。"""
    n = int(settings["market_checks"]["max_goals"])
    g = df.dropna(subset=["m_home", "m_draw", "m_away", "m_over25"]).copy()
    target = g[["m_home", "m_draw", "m_away", "m_over25"]].to_numpy(float)
    lh, la = fit_market_lambdas(target, n)
    fitted = matrix_probs(lh, la, n)
    g["lam_home"], g["lam_away"] = lh, la
    err = fitted - target
    names = ["主胜", "平", "客胜", "大2.5球"]
    recon = pd.DataFrame({"概率": names, "平均绝对差": np.abs(err).mean(axis=0), "平均差(矩阵-赔率)": err.mean(axis=0),
                          "99%的场差在这以内": np.quantile(np.abs(err), 0.99, axis=0), "最大差": np.abs(err).max(axis=0)})
    mats = score_matrices(lh, la, 0.0, 0.0, n)
    hg, ag = np.minimum(g["home_goals"].to_numpy(int), n), np.minimum(g["away_goals"].to_numpy(int), n)
    k = np.arange(len(g))
    g["mk_score_logloss"] = -np.log(np.maximum(mats[k, hg, ag], 1e-15))
    i, j = np.indices(mats.shape[1:])
    over = (g["home_goals"] + g["away_goals"] > 2.5).to_numpy(float)
    btts = ((g["home_goals"] > 0) & (g["away_goals"] > 0)).to_numpy(float)
    g["mk_brier_over25"] = (fitted[:, 3] - over) ** 2
    g["mk_brier_btts"] = ((mats * ((i > 0) & (j > 0))).sum(axis=(1, 2)) - btts) ** 2
    y = metrics.encode_results(g["result"])
    g["mk_log_loss"] = metrics.log_loss_per_match(fitted[:, :3] / fitted[:, :3].sum(axis=1, keepdims=True), y)
    g["m_log_loss"] = metrics.log_loss_per_match(g[["m_home", "m_draw", "m_away"]].to_numpy(float), y)
    g = g.merge(scores[["match_id", "score_logloss"]], on="match_id", how="left")
    rows = []
    for metric, a, b in [("比分 log loss", "score_logloss", "mk_score_logloss"),
                         ("大 2.5 球 Brier", "brier_over25", "mk_brier_over25"),
                         ("双方进球 Brier", "brier_btts", "mk_brier_btts"),
                         ("胜平负 log loss（模型 对 市场矩阵）", "log_loss", "mk_log_loss"),
                         ("胜平负 log loss（去水赔率 对 市场矩阵）", "m_log_loss", "mk_log_loss")]:
        gg = g.dropna(subset=[a, b])
        mean, lo, hi = _boot(gg[b] - gg[a], gg["match_date"], settings)
        rows.append({"metric": metric, "n": len(gg), "左边": gg[a].mean(), "市场矩阵": gg[b].mean(),
                     "差(市场矩阵-左边)": mean, "ci_low": lo, "ci_high": hi, "结论(市场矩阵)": verdict(lo, hi)})
    per = g[["match_id", "competition", "season", "match_date", "m_home", "m_draw", "m_away", "m_over25",
             "lam_home", "lam_away", "mk_score_logloss"]]
    return pd.DataFrame(rows), recon, per


# ======================= 总入口 =======================
def run(fs: FeatureStore, settings: dict, run_dir, model: str) -> dict:
    pred = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig", low_memory=False)
    pred = pred[(pred["model"] == model) & (pred["horizon"] == HORIZON)].dropna(subset=[*P, "result", "log_loss"])
    pred = pred[pred["season"] <= settings["seasons"]["dev_last"]]
    if pred.empty:
        raise ValueError(f"{run_dir} 里没有 {model} 的 {HORIZON} 预测")
    scores = pd.read_csv(run_dir / "scores_matrix.csv", encoding="utf-8-sig")
    scores = scores[scores["model"] == model]
    keep = ["match_id", "competition", "season", "match_date", "result", "home_goals", "away_goals", *P,
            "log_loss", "brier", "rps", "brier_over25", "brier_btts"]
    market = load_market(fs, settings, sorted(pred["competition"].unique()))
    df = pred[keep].merge(market, on="match_id", how="inner").dropna(subset=Q)
    df = add_groups(df, settings)
    mm, recon, per = market_matrix(df, scores, settings)
    info = {"回测": str(run_dir.name), "模型": model, "时点": HORIZON, "预测场数": int(len(pred)),
            "有赛前赔率的场数": int(len(df)), "赛前赔率来源": df["q_source"].value_counts().to_dict(),
            "有同一家收盘的场数": int(df[C].notna().all(axis=1).sum()), "市场矩阵场数": int(len(per)),
            "设置": settings["market_checks"], "去水": settings["backtest"]["benchmark_devig_method"]}
    return {"check1": check1(df, settings), "check2": check2(df, settings), "check3": check3(df, settings),
            "market_matrix": mm, "market_matrix_recon": recon, "market_matrix_matches": per, "info": info}


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="三个检验和市场比分矩阵（需求文档第 5 节）")
    ap.add_argument("--run", required=True, help="回测结果文件夹名（data/backtest/ 下面），要有 predictions.csv、scores_matrix.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / a.run
    fs = FeatureStore(open_readonly(a.db), "开发", settings)
    res = run(fs, settings, run_dir, a.model)
    out = run_dir / "market_checks"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(res["info"], f, ensure_ascii=False, indent=2, default=str)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    fmt = {"float_format": lambda v: f"{v:.4f}", "index": False}
    c1 = res["check1"]
    print("检验一（全部、按赔率来源）：")
    print(c1[(c1["分组方式"] != "competition") & (c1["metric"] == "log_loss")].to_string(**fmt))
    c2 = res["check2"]
    print("\n检验二（全部、强弱队、模型更看好哪边）：")
    print(c2[c2["分组方式"] != "competition"].to_string(**fmt))
    print("\n检验三：")
    print(res["check3"].to_string(**fmt))
    print("\n市场比分矩阵 还原误差：")
    print(res["market_matrix_recon"].to_string(**fmt))
    print(res["market_matrix"].to_string(**fmt))
    print(json.dumps({k: v for k, v in res["info"].items() if k != "设置"}, ensure_ascii=False))
    print(f"结果存在 {out}")


if __name__ == "__main__":
    main()
