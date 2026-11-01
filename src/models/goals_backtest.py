"""步骤 1–3 的回测和比较（需求文档第 4 节步骤 1–3、第 5 节“怎么算真的更准”）。

为什么另写一个入口（不直接用 src.evaluate.backtest）：
  回测框架只把比分矩阵换成胜平负、大 2.5 球、双方进球存下来，整张矩阵不存。
  步骤 3 要回答“大比分是否被低估”，得看整张矩阵：预测“总进球 5 个以上”的概率 和 实际有几成比赛进了 5 个以上。
  所以这里用一个“记录器”把模型包一层：回测照常跑（同一个 run_backtest，规则一点没变），
  记录器顺手把每场比赛 T-24 的比分矩阵留在内存里，回测完再算这几样：
    1. 比分 log loss：真实比分那一格的概率取 −log。它看整张矩阵，大比分估少了、小比分估错了都会扣分。
    2. 进球分布核对：预测的“总进球 0、1、2 … 5 个以上”的比例 对 实际比例；一队进 4 个以上；平均数、方差。
    3. 两个模型的比较：胜平负 log loss / Brier / RPS（回测框架算的），再加 比分 log loss、大 2.5 球 Brier、双方进球 Brier，
       都用 compare.py 的成块 bootstrap（按周成块、95% 区间）。
  只记 T-24：这几个模型每周重新拟合一次，两次拟合之间不更新，T-6、T-1 和 T-24 几乎一样（见第 4 节步骤 1–3 的说明）。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.models.goals_backtest --name step1 --first 2005 --last 2023 \\
      --models frequency_poisson elo_gd poisson poisson_ridge1 poisson_ridge10 \\
      --pairs frequency_poisson:poisson elo_gd:poisson poisson:poisson_ridge1 poisson:poisson_ridge10
结果在 data/backtest/<名字>/：回测框架的那几个文件，另外加
  scores_matrix.csv       每场 × 有矩阵的模型：比分 log loss、预测的大比分概率、实际比分
  goals_calibration.csv   进球分布核对（每个模型 × 联赛，另有“全部”）
  compare_goals.csv       --pairs 的比较结果
  shot_coefficients.csv   步骤 4：用射门推算进球的模型（target = shots）每次拟合每个联赛的 a、b
  matrices_<模型>.npz      加 --save-matrices 时：每场最早时点的整张比分矩阵（match_id + 矩阵），融合研究用（src/evaluate/fusion_study.py）
步骤 4 的两个实验（需求文档第 4 节“步骤 4 事先定的比较”）也用这个入口，命令写在进度记录里。
一次回测里的模型要用同一种用途：实验一的组都是“研究实验”，实验二的组都是“A版输入”，分开跑。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_settings
from src.evaluate.backtest import run_backtest, save
from src.evaluate.compare import BETTER, SAME, WORSE, block_ids, bootstrap_mean_diff, compare, verdict
from src.features import FeatureStore, open_readonly
from src.models.base import Model
from src.models.registry import build_model

TOTAL_BINS = [0, 1, 2, 3, 4, 5]            # 总进球：0、1、2、3、4、5 个以上
EXTRA_METRICS = ("score_logloss", "brier_over25", "brier_btts")


class MatrixRecorder(Model):
    """把一个模型包一层：拟合、更新、预测都交给里面的模型，只是把每场第一次预测（最早的时点）的比分矩阵记下来。"""

    def __init__(self, inner: Model):
        self.__dict__["inner"] = inner
        self.name, self.data, self.usage = inner.name, inner.data, inner.usage
        self.matrices: dict[int, np.ndarray] = {}

    def __getattr__(self, attr):          # 其他属性（例如 ratings_table）都用里面那个模型的
        return getattr(self.__dict__["inner"], attr)

    def fit(self, history, cutoff):
        self.inner.fit(history, cutoff)

    def update(self, new_rows, cutoff):
        self.inner.update(new_rows, cutoff)

    def predict(self, targets, cutoff):
        preds = self.inner.predict(targets, cutoff)
        for mid, p in zip(targets["match_id"], preds):
            if p is not None and np.ndim(p) == 2 and int(mid) not in self.matrices:
                self.matrices[int(mid)] = np.asarray(p, dtype=np.float32)
        return preds

    def params(self):
        return self.inner.params()


def save_matrices(models: list[MatrixRecorder], out) -> None:
    """每个有矩阵的模型存一个 matrices_<模型>.npz：match_id（从小到大）和对应的比分矩阵（float32，压缩）。"""
    for m in models:
        if not m.matrices:
            continue
        ids = np.array(sorted(m.matrices), dtype=np.int64)
        np.savez_compressed(out / f"matrices_{m.name}.npz", match_id=ids, matrix=np.stack([m.matrices[i] for i in ids]))


# ======================= 用整张矩阵算的分数 =======================
def matrix_scores(matrices: dict[int, np.ndarray], answers: pd.DataFrame, eps: float = 1e-15) -> pd.DataFrame:
    """每场：比分 log loss、预测的 总进球 0…5+ 的概率、一队 4 球以上的概率、总进球的平均和平方的平均。"""
    a = answers.set_index("match_id")
    rows = []
    for mid, m in matrices.items():
        if mid not in a.index:
            continue
        hg, ag = int(a.at[mid, "home_goals"]), int(a.at[mid, "away_goals"])
        n = m.shape[0] - 1
        m = m.astype(float) / m.sum()
        i, j = np.indices(m.shape)
        tot = i + j
        row = {"match_id": mid, "home_goals": hg, "away_goals": ag,
               "score_logloss": -np.log(max(m[min(hg, n), min(ag, n)], eps)),
               "pred_mean_total": float((m * tot).sum()), "pred_mean_total_sq": float((m * tot ** 2).sum()),
               "pred_team4plus": float(m[4:, :].sum() + m[:, 4:].sum() - m[4:, 4:].sum())}
        for b in TOTAL_BINS:
            row[f"pred_total_{b}"] = float(m[tot >= b].sum() if b == TOTAL_BINS[-1] else m[tot == b].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def goals_calibration(scores: pd.DataFrame) -> pd.DataFrame:
    """进球分布核对：每个模型 × 联赛（再加“全部”），预测比例 对 实际比例。

    方差：预测的“总进球方差” = 每场 (平方的平均 − 平均²) 的平均 + 各场平均数之间的方差（总方差公式），和实际总进球的方差比。
    """
    s = scores.assign(total=scores["home_goals"] + scores["away_goals"])
    s["team4plus"] = ((s["home_goals"] >= 4) | (s["away_goals"] >= 4)).astype(float)
    parts = []
    for comp_level in ("联赛", "全部"):
        keys = ["model", "competition"] if comp_level == "联赛" else ["model"]
        for key, g in s.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            row = {"model": key[0], "competition": key[1] if len(key) > 1 else "全部", "n": len(g)}
            for b in TOTAL_BINS:
                obs = (g["total"] >= b) if b == TOTAL_BINS[-1] else (g["total"] == b)
                label = f"{b}+" if b == TOTAL_BINS[-1] else str(b)
                row[f"总进球{label}_预测"] = g[f"pred_total_{b}"].mean()
                row[f"总进球{label}_实际"] = obs.mean()
            row["一队4球以上_预测"] = g["pred_team4plus"].mean()
            row["一队4球以上_实际"] = g["team4plus"].mean()
            row["总进球平均_预测"] = g["pred_mean_total"].mean()
            row["总进球平均_实际"] = g["total"].mean()
            within = (g["pred_mean_total_sq"] - g["pred_mean_total"] ** 2).mean()
            row["总进球方差_预测"] = within + g["pred_mean_total"].var(ddof=0)
            row["总进球方差_实际"] = g["total"].var(ddof=0)
            row["比分logloss"] = g["score_logloss"].mean()
            parts.append(row)
    return pd.DataFrame(parts)


def compare_extra(per_match: pd.DataFrame, base: str, new: str, settings: dict) -> pd.DataFrame:
    """比分 log loss、大 2.5 球 Brier、双方进球 Brier 的成块 bootstrap（和 compare.py 同一个办法，同一套设置）。"""
    c = settings["compare"]
    a = per_match[per_match["model"] == base]
    b = per_match[per_match["model"] == new]
    rows = []
    for m in EXTRA_METRICS:
        p = a[["match_id", "match_date", m]].dropna().merge(b[["match_id", m]].dropna(), on="match_id",
                                                             suffixes=("_base", "_new"))
        if p.empty:
            continue
        blocks = block_ids(p["match_date"], c["block"])
        mean, lo, hi = bootstrap_mean_diff(p[f"{m}_new"] - p[f"{m}_base"], blocks, c["n_boot"], c["confidence"],
                                           c["seed"])
        rows.append({"base": f"{base}@T-24", "new": f"{new}@T-24", "metric": m, "n": len(p),
                     "blocks": len(np.unique(blocks)), "block": c["block"], "base_mean": p[f"{m}_base"].mean(),
                     "new_mean": p[f"{m}_new"].mean(), "diff": mean, "ci_low": lo, "ci_high": hi,
                     "verdict": verdict(lo, hi), "primary": False})
    return pd.DataFrame(rows)


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="步骤 1–3：泊松 / DC / DC + 负二项的回测和比较")
    ap.add_argument("--name", required=True, help="结果文件夹名")
    ap.add_argument("--leagues", nargs="+", default=list(settings["leagues"]["top"]))
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, default=settings["seasons"]["dev_last"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="要比较的 旧:新，例如 poisson:dc")
    ap.add_argument("--horizons", nargs="+", type=float, default=[24])
    ap.add_argument("--db", default=None)
    ap.add_argument("--save-matrices", action="store_true", help="把每场的整张比分矩阵存成 matrices_<模型>.npz")
    a = ap.parse_args()

    fs = FeatureStore(open_readonly(a.db), "开发", settings)
    models = [MatrixRecorder(build_model(n, settings)) for n in a.models]
    res = run_backtest(fs, a.leagues, a.first, a.last, models, a.horizons, benchmark=True)
    out = save(res, a.name, settings)

    pred = res.predictions
    first_hz = f"T-{int(max(a.horizons))}"
    answers = pred[pred["result"].notna()].drop_duplicates("match_id")[["match_id", "home_goals", "away_goals"]]
    info = pred[["match_id", "competition", "season", "match_date"]].drop_duplicates("match_id")
    scores = pd.concat([matrix_scores(m.matrices, answers).assign(model=m.name) for m in models if m.matrices],
                       ignore_index=True)
    scores = scores.merge(info, on="match_id")
    # 只留每个有矩阵的模型都有预测的比赛（和回测框架的“同一批比赛”一样）
    k = scores.groupby("match_id")["model"].nunique()
    scores = scores[scores["match_id"].isin(k[k == scores["model"].nunique()].index)]
    if a.save_matrices:
        save_matrices(models, out)
    scores.to_csv(out / "scores_matrix.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    calib = goals_calibration(scores)
    calib.to_csv(out / "goals_calibration.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    # 步骤 4：用射门推算进球的模型，每次拟合每个联赛估出的 a（每个射正）、b（每个射偏）
    coefs = [pd.DataFrame(m.coef_log).assign(model=m.name) for m in models if getattr(m, "coef_log", None)]
    if coefs:
        coef = pd.concat(coefs, ignore_index=True)
        coef.to_csv(out / "shot_coefficients.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
        print("\n射门推算进球的系数（每次拟合的平均，按联赛）：")
        print(coef.groupby(["model", "competition"])[["a_sot", "b_off"]].agg(["mean", "min", "max"]).round(3)
              .to_string())

    per_match = pred[pred["horizon"] == first_hz][["match_id", "match_date", "model", "brier_over25", "brier_btts"]]
    per_match = per_match.merge(scores[["match_id", "model", "score_logloss"]], on=["match_id", "model"], how="left")
    comps = []
    for pair in a.pairs:
        base, new = pair.split(":")
        comps.append(compare(pred, base, new, first_hz, settings=settings))
        comps.append(compare_extra(per_match, base, new, settings))
    pd.set_option("display.width", 250)
    fmt = {"float_format": lambda v: f"{v:.4f}", "index": False}
    s = res.summary
    print(s[s["level"] == "全部"].to_string(**fmt))
    print("\n进球分布核对（全部联赛）：")
    print(calib[calib["competition"] == "全部"].T.to_string())
    if comps:
        cmp = pd.concat(comps, ignore_index=True)
        cmp.to_csv(out / "compare_goals.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
        print("\n比较（差 = 新 - 旧，负数 = 新的更准）：")
        print(cmp[["base", "new", "metric", "n", "base_mean", "new_mean", "diff", "ci_low", "ci_high", "verdict"]]
              .to_string(**fmt))
    print("\n耗时：", json.dumps(res.info["耗时"], ensure_ascii=False))
    print(f"结果存在 {out}")


if __name__ == "__main__":
    main()
