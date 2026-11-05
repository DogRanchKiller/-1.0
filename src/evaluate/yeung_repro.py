"""复现 Yeung 等 2024 的深度学习：回测、和 gb_l7、平博收盘比（需求文档第 5 节“复现 Yeung 等 2024 的深度学习：怎么做”）。

一、两步
  backtest：用回测框架（src/evaluate/backtest.py，开发模式，只能回测开发赛季）跑一个种子的模型，
            五大联赛 2014/15–2023/24，开球前 24 小时。yeung_s0 是主结果，yeung_s1、yeung_s2 只看“换个种子差多少”。
            三个种子互不相干，分三个程序同时跑（--seed 0 / 1 / 2，快三倍），
            结果放 data/backtest/yeung_repro/s0、s1、s2/（和别的回测一样的文件）。
  compare： 把 gb_l7 那次回测（data/backtest/w9_gb_l7/predictions.csv）同一批比赛的预测接进来，
            用 compare.py 同一套办法（每场分数差 → 按周成块 bootstrap → 95% 区间）比：
              - yeung_s0 对 gb_l7（主结论，log loss）；
              - yeung_s0 对平博收盘；gb_l7 对平博收盘（参照）；
              - yeung_s1、yeung_s2 同样比一遍（只看稳不稳）。
            再分联赛、分赛季比 yeung_s0 对 gb_l7；另出一张平均分表（log loss、RPS、Brier、命中率）。

二、为什么要核对两次回测的比赛信息
  gb_l7 是另一天跑的。数据库后来导入过新数据，比赛编号（match_id）理论上不变，
  但为了确定“同一个编号是同一场比赛”，接之前核对每场的联赛、赛季、日期、主客队、比分，有一场不一样就停下报错。

用法：
  .venv\\Scripts\\python -m src.evaluate.yeung_repro backtest --seed 0 [--db 数据库文件]   （1、2 同样各跑一次）
  .venv\\Scripts\\python -m src.evaluate.yeung_repro compare
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from src.common.config import load_settings, resolve_path, season_role
from src.evaluate.backtest import run_backtest, save
from src.evaluate.compare import compare, load_predictions
from src.features import FeatureStore, open_readonly
from src.models.yeung_nn import YeungModel

CHECK_COLS = ["competition", "season", "match_date", "home_team", "away_team", "home_goals", "away_goals"]
SCORE_COLS = ["log_loss", "rps", "brier", "hit"]


def model_names(cfg: dict) -> list[str]:
    return [f"yeung_s{s}" for s in cfg["seeds"]]


def check_seasons(settings: dict) -> tuple[int, int]:
    """评分赛季：eval_first 到 seasons.dev_last，每一季都必须是开发赛季（考试赛季不碰）。"""
    first, last = int(settings["repro_yeung"]["eval_first"]), int(settings["seasons"]["dev_last"])
    bad = [s for s in range(first, last + 1) if season_role(s, settings) != "dev"]
    if bad:
        raise ValueError(f"这些赛季不是开发赛季，不能用：{bad}")
    return first, last


def backtest(settings: dict, seed: int, db=None):
    cfg = settings["repro_yeung"]
    if seed not in cfg["seeds"]:
        raise ValueError(f"种子 {seed} 不在 settings 的 repro_yeung.seeds 里（事先定的是 {cfg['seeds']}）")
    first, last = check_seasons(settings)
    # 网络很小，一批只有 100 场：多线程分工的开销比算的还大，1 个线程最快（只影响快慢，不影响方法）
    torch.set_num_threads(1)
    fs = FeatureStore(open_readonly(db), "开发", settings)
    model = YeungModel(f"yeung_s{seed}", cfg, seed)
    res = run_backtest(fs, list(settings["leagues"]["top"]), first, last, [model],
                       [float(cfg["horizon_hours"])], benchmark=True)
    res.info["训练记录"] = model.train_log
    return save(res, f"{cfg['run_name']}/s{seed}", settings)


def combined(settings: dict) -> pd.DataFrame:
    """yeung 的预测 + 平博收盘 + 同一批比赛的 gb_l7（核对过比赛信息）。"""
    cfg = settings["repro_yeung"]
    out_dir = resolve_path(settings["backtest"]["output_dir"])
    # 三个种子各一个文件夹；平博收盘、多家平均每个文件夹里都有一份，一样的，只留一份
    ours = pd.concat([load_predictions(out_dir / cfg["run_name"] / f"s{s}") for s in cfg["seeds"]], ignore_index=True)
    ours = ours.drop_duplicates(["match_id", "model", "horizon"], ignore_index=True)
    first, last = check_seasons(settings)
    gb = load_predictions(out_dir / cfg["compare_with"])
    gb = gb[(gb["model"] == cfg["compare_model"]) & (gb["horizon"] == "T-24")
            & gb["season"].between(first, last)]
    ref = ours.drop_duplicates("match_id").set_index("match_id")[CHECK_COLS]
    chk = gb.set_index("match_id")[CHECK_COLS]
    common = ref.index.intersection(chk.index)
    a, b = ref.loc[common].astype(str), chk.loc[common].astype(str)
    diff = (a != b).any(axis=1)
    if diff.any():
        raise ValueError(f"两次回测有 {int(diff.sum())} 场比赛信息对不上（例如 match_id {list(common[diff][:5])}），停下")
    missing = set(ref.index) - set(chk.index)
    if missing:
        print(f"注意：{len(missing)} 场比赛 gb_l7 那次回测里没有，这些场只和平博比")
    return pd.concat([ours, gb[ours.columns.intersection(gb.columns)]], ignore_index=True)


def summary_table(pred: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    """几个模型在“它们都有预测的同一批比赛”上的平均分。"""
    parts = [pred[(pred["model"] == m) & (pred["horizon"] == h)].dropna(subset=["log_loss"]) for m, h in models]
    ids = set.intersection(*(set(p["match_id"]) for p in parts))
    rows = []
    for (m, h), p in zip(models, parts):
        p = p[p["match_id"].isin(ids)]
        rows.append({"model": f"{m}@{h}", "n": len(p), **{c: p[c].mean() for c in SCORE_COLS}})
    return pd.DataFrame(rows)


def run_compare(settings: dict) -> dict:
    cfg = settings["repro_yeung"]
    pred = combined(settings)
    names, base = model_names(cfg), cfg["compare_model"]
    main_name = names[0]
    pairs = [(base, n, "T-24", "T-24") for n in names]
    pairs += [(n, "pinnacle_close", "T-24", "收盘") for n in [base, *names]]
    allc = pd.concat([compare(pred, b, n, bh, nh, settings=settings) for b, n, bh, nh in pairs], ignore_index=True)
    parts = [("联赛", c, s) for c, s in pred.groupby("competition")] + [("赛季", str(y), s) for y, s in pred.groupby("season")]
    split = pd.concat([compare(s, base, main_name, settings=settings).assign(分法=k, 哪一份=v) for k, v, s in parts],
                      ignore_index=True)
    tab_all = summary_table(pred, [(base, "T-24")] + [(n, "T-24") for n in names])
    tab_pin = summary_table(pred, [(base, "T-24")] + [(n, "T-24") for n in names] + [("pinnacle_close", "收盘")])
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / cfg["run_name"]
    kw = {"index": False, "encoding": "utf-8-sig", "float_format": "%.6f"}
    allc.to_csv(run_dir / "compare.csv", **kw)
    split.to_csv(run_dir / "compare_by_league_season.csv", **kw)
    tab_all.to_csv(run_dir / "scores_vs_gb.csv", **kw)
    tab_pin.to_csv(run_dir / "scores_vs_pinnacle.csv", **kw)
    return {"compare": allc, "split": split, "tab_all": tab_all, "tab_pin": tab_pin, "dir": run_dir}


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="复现 Yeung 等 2024 的深度学习：回测、比较")
    ap.add_argument("step", choices=["backtest", "compare"])
    ap.add_argument("--seed", type=int, default=None, help="backtest 用：哪个种子（settings 的 repro_yeung.seeds 之一）")
    ap.add_argument("--db", default=None, help="数据库文件（只读打开），默认 settings 里的")
    a = ap.parse_args()
    if a.step == "backtest":
        if a.seed is None:
            ap.error("backtest 要写 --seed")
        out = backtest(settings, a.seed, a.db)
        info = json.load(open(out / "run_info.json", encoding="utf-8"))
        print("耗时：", json.dumps(info["耗时"], ensure_ascii=False))
        print("没出预测的场数：", info["没出预测的场数"])
        print(f"结果存在 {out}")
        return
    r = run_compare(settings)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    fmt = {"index": False, "float_format": lambda v: f"{v:.5f}"}
    print("同一批比赛的平均分（和 gb_l7）：\n" + r["tab_all"].to_string(**fmt))
    print("\n同一批比赛的平均分（有平博收盘的）：\n" + r["tab_pin"].to_string(**fmt))
    c = r["compare"]
    print("\n比较（差 = 新 − 旧，分数越小越好）：\n"
          + c[c["primary"]].drop(columns=["primary", "block"]).to_string(**fmt))
    s = r["split"]
    print("\n分联赛、分赛季（yeung_s0 对 gb_l7，log loss）：\n"
          + s[s["primary"]][["分法", "哪一份", "n", "base_mean", "new_mean", "diff", "ci_low", "ci_high", "verdict"]]
          .to_string(**fmt))
    print(f"\n结果存在 {r['dir']}")


if __name__ == "__main__":
    main()
