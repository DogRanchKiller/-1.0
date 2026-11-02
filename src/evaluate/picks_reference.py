"""四个玩法预测的历史参考（规则 picks_v2，需求文档第 3 节“四个玩法的预测和命中率”）。

用期末考试已经存下的结果（data/backtest/final_exam/：2024/25–2025/26 赛季、开球前 24 小时的 gb_l7 比分矩阵和赛前赔率），
按网站上同一套规则（src/web/site/picks.py）给每场比赛做四个预测，对比分算命中率，
存到 settings.yaml picks.reference_dir 下面以规则名命名的文件夹（picks_v2/；picks_v1 的结果在 reference_dir 本身，照留）：
  summary.json：四个玩法的汇总和“一直猜主胜”的对照（战绩页“历史参考”那一列读它）；
  picks.csv：每场一行，四个预测和结果（想自己翻着看时用；比分写最可能的几个，用 / 隔开，最可能的在前）。

注意：
- 这只是描述统计，不是第二次期末考试（考试只考一次，已经考过）；看了这里的数也不改任何规则和门槛。
- 和网站上的区别：这里的融合矩阵 = w × 赔率反推的比分矩阵 + (1 − w) × 模型矩阵（w = settings.yaml fusion.w，现在是 1，
  就是赔率反推的矩阵）；胜平负、大小球赔率按考试的取法（有平博用平博），亚盘线用多家平均的主盘口（两边赔率 1.5–2.7），
  和网站一样用“融合矩阵在这条线上胜算较大的一边”。
- 只读期末考试存下的文件，不连数据库、不重跑模型。

运行：.venv\\Scripts\\python -m src.evaluate.picks_reference
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.markets import asian_handicap, dc_matrices, fit_market_dc
from src.evaluate import fusion_study as fu
from src.evaluate.compare import load_predictions
from src.web.site.picks import MARKET_NAMES, MARKETS, OU_LINE, RULE, best_scores, reference_path, settle, tally


def fused_matrices(t: pd.DataFrame, ids: np.ndarray, mats: np.ndarray, w: float) -> np.ndarray:
    """每场的融合比分矩阵（和 fusion_study.build 同一个做法：赔率四个数反推市场矩阵，再按 w 和模型矩阵混）。"""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    m = mats[pos.loc[t["match_id"]].to_numpy()].astype(float)
    m /= m.sum(axis=(1, 2), keepdims=True)
    lh, la, rho = fit_market_dc(t[fu.FOUR].to_numpy(float), n=m.shape[1] - 1)
    mk = dc_matrices(lh, la, rho, n=m.shape[1] - 1)
    f = w * mk + (1 - w) * m
    return f / f.sum(axis=(1, 2), keepdims=True)


def picks_for(m: np.ndarray, ah_line: float) -> dict:
    """一场的四个预测（和 src/web/site/picks.py 的写法一样，只是输入换成矩阵）。"""
    i, j = np.indices(m.shape)
    p3 = np.array([m[i > j].sum(), m[i == j].sum(), m[i < j].sum()])
    over = float(m[i + j > OU_LINE].sum())
    out = {"1x2": {"side": int(np.argmax(p3)), "p": float(p3.max())},
           "ou": {"side": "over" if over >= 0.5 else "under", "line": OU_LINE, "p": max(over, 1 - over)},
           "ah": None}
    if not np.isnan(ah_line):
        share = asian_handicap(m, float(ah_line), "home")["win_share"]
        if not np.isnan(share):
            home = share >= 0.5
            out["ah"] = {"side": "home" if home else "away", "line": float(ah_line), "p": share if home else 1 - share}
    out["score"] = best_scores(m)
    return out


def run(settings: dict) -> tuple[dict, pd.DataFrame]:
    cfg = settings["picks"]
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / cfg["reference_run"]
    first, last = settings["seasons"]["exam"]
    full = load_predictions(run_dir)
    pred = full[(full["season"] >= first) & (full["season"] <= last)].reset_index(drop=True)
    npz = np.load(run_dir / f"matrices_{settings['fusion_study']['model']}.npz")
    odds = pd.read_csv(run_dir / "fusion_odds.csv", encoding="utf-8-sig")
    t, _model, _market, check = fu.build(pred, npz["match_id"], npz["matrix"], odds, settings)
    mats = fused_matrices(t, npz["match_id"], npz["matrix"], float(settings["fusion"]["w"]))

    rows = []
    for k, r in enumerate(t.itertuples(index=False)):
        score = (int(r.home_goals), int(r.away_goals))
        p = picks_for(mats[k], r.ah_line)
        row = {"match_id": int(r.match_id), "season": int(r.season), "competition": r.competition,
               "home_goals": score[0], "away_goals": score[1]}
        for mk in MARKETS:
            pk = p[mk]
            row[f"{mk}_pick"] = (None if pk is None else
                                 "/".join(f"{h}-{a}" for h, a in pk["scores"]) if mk == "score" else
                                 f"{pk['side']} {pk['line']:g}" if mk in ("ah", "ou") else str(pk["side"]))
            row[f"{mk}_p"] = None if pk is None else round(pk["p"], 6)
            row[f"{mk}_outcome"] = settle(mk, pk, score)
        rows.append(row)
    per = pd.DataFrame(rows)
    # 最可能的那一个比分的胜负方向（0 主胜 / 1 平 / 2 客胜）和胜平负预测对不上的场数
    side = per["score_pick"].str.split("/").str[0].str.split("-", expand=True).astype(int)
    score_side = np.where(side[0] > side[1], 0, np.where(side[0] == side[1], 1, 2))
    differs = int((score_side != per["1x2_pick"].astype(int)).sum())

    summary = {
        "rule": RULE, "run": cfg["reference_run"], "seasons": [first, last], "horizon": settings["fusion_study"]["horizon"],
        "fusion_w": float(settings["fusion"]["w"]), "n_matches": int(len(per)),
        "markets": {mk: tally(per[f"{mk}_outcome"].dropna().tolist()) for mk in MARKETS},
        "always_home": tally(np.where(per["home_goals"] > per["away_goals"], "win", "loss").tolist()),
        "score_side_differs": differs,
        "top_score_picks": [[s.replace("-", "–"), int(n)] for s, n in
                            Counter(per["score_pick"].str.split("/").explode()).most_common(5)],   # 4 个里出现最多的比分
        "score_top": len(per["score_pick"].iloc[0].split("/")),
        "score_p_mean": float(per["score_p"].mean()),       # 4 个比分合起来的可能性，平均多少
        "ah_no_pick": int(per["ah_pick"].isna().sum()),
        "data_check": {k: check[k] for k in ("有预测的比赛", "没有模型矩阵", "融合", "亚盘线不是主盘口、不用的场数", "没有亚盘线的场数")},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "note": "描述统计，不是第二次期末考试；看了这里的数也不改任何规则。",
    }
    return summary, per


def main() -> None:
    settings = load_settings()
    summary, per = run(settings)
    out = reference_path(settings).parent
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "picks.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"期末考试 {summary['seasons']}、{summary['horizon']}，共 {summary['n_matches']} 场（规则 {RULE}）")
    for mk in MARKETS:
        t = summary["markets"][mk]
        print(f"  {MARKET_NAMES[mk]}：{t['n']} 场，命中率 {t['rate']:.1%}（中 {t['win']}、赢半 {t['half_win']}、走 {t['push']}、"
              f"输半 {t['half_loss']}、没中 {t['loss']}）")
    print(f"  对照 一直猜主胜：{summary['always_home']['rate']:.1%}")
    print(f"  比分 {summary['score_top']} 个合起来的可能性平均 {summary['score_p_mean']:.1%}")
    print(f"  最可能的那个比分和胜平负预测方向不一样的场数：{summary['score_side_differs']}；"
          f"4 个里出现最多的比分：{summary['top_score_picks']}")
    print(f"结果放在 {out}")


if __name__ == "__main__":
    main()
