'''第 11 阶段：日职、中超的开发回测和平博收盘比（需求文档第 4 节“第 11 阶段”）。

这个程序不重新预测，只读已经跑好的回测结果（data/backtest/<名字>/predictions.csv）：
  回测本身用 src.evaluate.backtest 跑（经过取数层、开发模式，碰不到 2024、2025 考试赛季）。

做什么：
  1. 每个模型和平博收盘去水概率只比“两边都有分数”的比赛（同一批比赛，才公平）；
  2. 每个联赛分赛季、合计各算一次：场数、模型和平博的 log loss、Brier、命中率，差 = 模型 − 平博；
  3. 合计再用成块 bootstrap（按周，src/evaluate/compare.py）算 95% 区间，给“更准 / 没差别 / 更差”；
     log loss、Brier 越小越好；命中率越大越好，所以命中率的结论倒过来看（差 > 0 才算模型更好）；
  4. 和五大联赛的差距（settings 的 asia_study.five_league_reference）并排放。
结果写一份中文 markdown 报告（settings 的 asia_study.output，data/ 不进 Git），表也存 csv。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.asia_study
'''
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate.compare import block_ids, bootstrap_mean_diff, verdict

NAMES = {"log_loss": "log loss", "brier": "Brier", "hit": "命中率"}


def paired_rows(pred: pd.DataFrame, model: str, horizon: str, bench: str) -> pd.DataFrame:
    '''模型（时点 horizon）和标杆（时点“收盘”）都有分数的比赛，每场一行：model_<分数>、bench_<分数>。'''
    cols = ["match_id", "competition", "season", "match_date", "log_loss", "brier", "hit"]
    a = pred[(pred["model"] == model) & (pred["horizon"] == horizon)][cols].dropna(subset=["log_loss"])
    b = pred[(pred["model"] == bench) & (pred["horizon"] == "收盘")][["match_id", "log_loss", "brier", "hit"]]
    b = b.dropna(subset=["log_loss"])
    return a.merge(b, on="match_id", suffixes=("_model", "_bench"))


def summarize(p: pd.DataFrame, metrics: list[str]) -> dict:
    '''一组比赛的场数和每个分数的两边平均、差（模型 − 标杆）。'''
    out = {"n": len(p)}
    for m in metrics:
        out[f"{m}_model"] = p[f"{m}_model"].mean()
        out[f"{m}_bench"] = p[f"{m}_bench"].mean()
        out[f"{m}_diff"] = out[f"{m}_model"] - out[f"{m}_bench"]
    return out


def model_verdict(metric: str, lo: float, hi: float) -> str:
    '''差 = 模型 − 平博。log loss / Brier 小的好，直接用 compare.verdict；命中率大的好，把区间反过来再判断。'''
    if metric == "hit":
        return verdict(-hi, -lo)
    return verdict(lo, hi)


def league_tables(pred: pd.DataFrame, league: str, cfg: dict, cmp_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''一个联赛：分赛季的表、合计的表（合计带 bootstrap 区间和结论）。'''
    metrics = cfg["metrics"]
    bench = cfg["benchmark"]
    models = [m for m in pred["model"].unique() if m not in {"pinnacle_close", "market_avg"}]
    season_rows, total_rows = [], []
    for model in models:
        p = paired_rows(pred[pred["competition"] == league], model, cfg["horizon"], bench)
        if p.empty:
            continue
        for season, g in p.groupby("season"):
            season_rows.append({"league": league, "model": model, "season": int(season), **summarize(g, metrics)})
        row = {"league": league, "model": model, "seasons": f"{int(p['season'].min())}–{int(p['season'].max())}",
               **summarize(p, metrics)}
        blocks = block_ids(p["match_date"], cmp_cfg["block"])
        for m in metrics:
            d = p[f"{m}_model"] - p[f"{m}_bench"]
            _, lo, hi = bootstrap_mean_diff(d, blocks, cmp_cfg["n_boot"], cmp_cfg["confidence"], cmp_cfg["seed"])
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
            row[f"{m}_verdict"] = model_verdict(m, lo, hi)
        total_rows.append(row)
    return pd.DataFrame(season_rows), pd.DataFrame(total_rows)


def load_league(cfg: dict, league: str) -> pd.DataFrame:
    '''把一个联赛的几次回测拼起来。同一个模型在两次回测里都有时，只留第一次的（名单前面的）。'''
    parts, seen = [], set()
    for run in cfg["runs"][league]:
        path = resolve_path(f"data/backtest/{run}") / "predictions.csv"
        pred = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        bench_rows = pred["model"].isin(["pinnacle_close", "market_avg"])
        keep = bench_rows & (not parts)        # 标杆只取第一次回测的（新的一列，不改 bench_rows 本身）
        keep = keep | (~bench_rows & ~pred["model"].isin(seen))
        seen |= set(pred.loc[~bench_rows, "model"].unique())
        parts.append(pred[keep])
    return pd.concat(parts, ignore_index=True)


def _f(v, nd=4, sign=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def report(seasons: pd.DataFrame, totals: pd.DataFrame, cfg: dict, league_names: dict) -> str:
    '''中文 markdown 报告：合计表、和五大联赛并排、分赛季表。'''
    out = ["# 日职、中超开发回测：模型和平博收盘比", "",
           f"时点 {cfg['horizon']}；只比模型和平博收盘去水概率都有的比赛。差 = 模型 − 平博：",
           "log loss、Brier 差 > 0 是模型更差；命中率差 < 0 是模型更差。区间是按周成块 bootstrap 的 95% 区间。", ""]
    out += ["## 合计", ""]
    head = ["联赛", "模型", "赛季", "场数", "模型 log loss", "平博", "差", "95% 区间", "结论",
            "模型 Brier", "平博", "差", "模型命中率", "平博", "差"]
    rows = []
    for _, r in totals.sort_values(["league", "log_loss_diff"]).iterrows():
        rows.append([league_names.get(r["league"], r["league"]), r["model"], r["seasons"], str(int(r["n"])),
                     _f(r["log_loss_model"]), _f(r["log_loss_bench"]), _f(r["log_loss_diff"], sign=True),
                     f"{_f(r['log_loss_lo'], sign=True)} 到 {_f(r['log_loss_hi'], sign=True)}", r["log_loss_verdict"],
                     _f(r["brier_model"]), _f(r["brier_bench"]), _f(r["brier_diff"], sign=True),
                     _f(r["hit_model"], 3), _f(r["hit_bench"], 3), _f(r["hit_diff"], 3, sign=True)])
    out += [md_table(head, rows), ""]
    ref = cfg.get("five_league_reference") or {}
    if ref:
        out += ["## 和五大联赛比（log loss 差 = 模型 − 平博）", ""]
        head = ["模型", "五大联赛 2014–2023", "场数"] + [league_names.get(lg, lg) for lg in cfg["runs"]]
        rows = []
        for model, v in ref.items():
            cells = [model, _f(v["model"] - v["pinnacle"], sign=True), str(v["n"])]
            for lg in cfg["runs"]:
                t = totals[(totals["league"] == lg) & (totals["model"] == model)]
                cells.append(f"{_f(t['log_loss_diff'].iloc[0], sign=True)}（{int(t['n'].iloc[0])} 场）" if len(t) else "")
            rows.append(cells)
        out += [md_table(head, rows), ""]
    out += ["## 分赛季（log loss、Brier、命中率）", ""]
    head = ["联赛", "模型", "赛季", "场数", "模型 log loss", "平博", "差", "Brier 差", "命中率差", "备注"]
    rows = []
    for _, r in seasons.sort_values(["league", "model", "season"]).iterrows():
        note = "样本少" if r["n"] < cfg["min_season_n"] else ""
        rows.append([league_names.get(r["league"], r["league"]), r["model"], str(r["season"]), str(int(r["n"])),
                     _f(r["log_loss_model"]), _f(r["log_loss_bench"]), _f(r["log_loss_diff"], sign=True),
                     _f(r["brier_diff"], sign=True), _f(r["hit_diff"], 3, sign=True), note])
    out += [md_table(head, rows), ""]
    return "\n".join(out)


def run(settings: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    s = settings or load_settings()
    cfg = s["asia_study"]
    names = {c: n for g in s["leagues"].values() for c, n in g.items()}
    seasons, totals = [], []
    for league in cfg["runs"]:
        a, b = league_tables(load_league(cfg, league), league, cfg, s["compare"])
        seasons.append(a)
        totals.append(b)
    seasons, totals = pd.concat(seasons, ignore_index=True), pd.concat(totals, ignore_index=True)
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report(seasons, totals, cfg, names), encoding="utf-8")
    seasons.to_csv(out.with_name(out.stem + "_分赛季.csv"), index=False, encoding="utf-8-sig", float_format="%.6f")
    totals.to_csv(out.with_name(out.stem + "_合计.csv"), index=False, encoding="utf-8-sig", float_format="%.6f")
    return seasons, totals, out


def main() -> None:
    argparse.ArgumentParser(description="日职、中超开发回测和平博收盘比（读已经跑好的回测）").parse_args()
    _, totals, out = run()
    cols = ["league", "model", "n", "log_loss_model", "log_loss_bench", "log_loss_diff", "log_loss_lo",
            "log_loss_hi", "log_loss_verdict", "brier_diff", "hit_diff"]
    pd.set_option("display.width", 220)
    print(totals[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n报告在 {out}")


if __name__ == "__main__":
    main()
