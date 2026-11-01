"""期末考试（预测部分）：需求文档第 5 节“期末考试怎么考”（跑之前写下的五条检查）。

一、考什么
  五大联赛 2024/25、2025/26 两季（考试赛季），T-24。设置全部用现在 settings.yaml 里的，考试中和考完都不改：
  模型 gb_l7、融合 fusion.w（= 1）、倾向门槛 lean、信心等级 confidence。
  机器人策略不在这次考，以后用这次存下来的考试预测（predictions.csv、matrices_gb_l7.npz、fusion_odds.csv）再考，预测不重算。

二、只考一次
  考完写 data/backtest/<final_exam.run_name>/exam_done.json；之后要把 settings.yaml 的 final_exam.done 改成 true 提交。
  两样有一样在，考试模式直接报错，不留“强制再考”的开关。
  开考（回测之前）先写 exam_started.json（commit、设置）；程序出错重跑用 --skip-backtest --reason，设置要和开考时一样。
  正式考要给 --reference：热身赛季（打分赛季之前）的预测要和 W9 一模一样，对不上就报错、不打分。
  正式考之前，用 --dry-run 在开发赛季（默认 2023/24，开发模式）试跑一次，和 W9 / W10 同一批比赛核对。

三、怎么跑（和开发赛季的回测一样）
  1. gb_l7 滚动回测（src.evaluate.backtest.run_backtest），从 final_exam.backtest_first（2005/06，和 W9 同一个起点）
     连着跑到最后一个考试赛季，整张比分矩阵存下来；只给打分的赛季打分，前面的只是“热身”。收盘赔率只用来打分。
     为什么从 2005/06 跑：gb_l7 每 91 天重训一次树，从第一次训练那天开始数；起点不同，重训日期就不同，预测也不同
     （试跑时从 2023/24 起跑，和 W9 同一场的胜平负概率最多差 0.07）。起点一样，打分赛季之前的预测应和 W9 一模一样（--reference 核对）。
  2. 每场 T-24 能看到的赛前赔率（fusion_study.fetch_odds，用途「融合」，和 W9 同一条规则）。
  3. 市场比分矩阵（fusion_study.build，和 W9 同一个函数）。
  4. 五条检查（都只算一次，不选任何数）：
     第 1 条 模型本身：gb_l7 对 平博收盘、多家平均收盘（同一批比赛），只报告。
     第 2 条 融合（w = fusion.w）对 只用模型（w = 0）：胜平负 log loss 区间整个在“融合更准”一边 = 通过。
            另外报告 21 档 w、分联赛 / 分赛季 / 胜平负用不用平博、融合对平博收盘。
     第 3 条 倾向：现在的门槛下，每个玩法“实际”的平均和 95% 区间，区间下限 > 0 = 通过；开发赛季的平均落不落在区间里。
     第 4 条 信心等级：每个等级的场数、d 平均；“高”的 d 不大于“中”“低”（多半比不了）；四个候选因素再算一次（只报告）。
     第 5 条 校准：融合后五个概率每 0.1 一档，预测平均 对 实际比例（只报告）。

四、结果（data/backtest/<名字>/ 下面，data/ 不进 Git）
  回测框架的文件（predictions.csv 等）、matrices_gb_l7.npz、fusion_odds.csv，另外：
  exam_model.csv       第 1 条
  exam_fusion.csv      第 2 条（融合对只用模型、融合对平博收盘，按几种分法）
  exam_w_grid.csv      第 2 条另外报告的 21 档 w
  exam_leans.csv       第 3 条（每个玩法一行）
  exam_lean_bands.csv  第 3 条另外报告的分档表
  exam_levels.csv      第 4 条每个等级
  exam_factors.csv     第 4 条四个候选因素（只报告，“算数”那一列不改规则）
  exam_calibration.csv 第 5 条
  exam_summary.json    场数、核对、每条通过没有
  exam_started.json    考试模式才写：开考记录（含每次重跑）
  exam_done.json       考试模式才写：考过了

用法：
  .venv\\Scripts\\python -m src.evaluate.final_exam --dry-run --db data/snapshot.db --reference <W9 的 w9_gb_l7>  # 试跑（开发模式，打分 2023/24）
  .venv\\Scripts\\python -m src.evaluate.final_exam --db data/snapshot.db --reference <W9 的 w9_gb_l7>            # 正式考（只能一次）
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import subprocess
from datetime import datetime, timezone

from src.common.config import PROJECT_ROOT, load_settings, resolve_path
from src.evaluate import confidence_study as cs
from src.evaluate import fusion_study as fu
from src.evaluate.backtest import run_backtest, save
from src.evaluate.compare import BETTER, block_ids, bootstrap_mean_diff, load_predictions
from src.features import FeatureStore, open_readonly
from src.features.usage import season_label
from src.models.goals_backtest import MatrixRecorder, save_matrices
from src.models.registry import build_model

FUSED, MODEL = "fused", "model_w0"          # 第 2 条：融合（w = fusion.w）、只用模型（w = 0）
BENCH = [("pinnacle_close", "收盘"), ("market_avg", "收盘")]   # 第 1 条比的两个标杆（收盘赔率只用来打分）
DONE_FILE = "exam_done.json"


class ExamAlreadyDone(RuntimeError):
    """期末考试已经考过（只考一次）。"""


def check_not_done(settings: dict, run_dir) -> None:
    """考过了就报错：settings 的 final_exam.done 是 true，或者结果文件夹里有 exam_done.json，有一样就算考过。"""
    if settings["final_exam"]["done"]:
        raise ExamAlreadyDone("settings.yaml 里 final_exam.done 是 true：期末考试已经考过，只考一次（需求文档第 5 节）")
    if (run_dir / DONE_FILE).exists():
        raise ExamAlreadyDone(f"{run_dir / DONE_FILE} 已经有了：期末考试已经考过，只考一次（需求文档第 5 节）")


STARTED_FILE = "exam_started.json"
FROZEN = ("fusion", "lean", "confidence", "models", "final_exam", "fusion_study", "confidence_study", "compare",
          "backtest", "features", "rest_days", "seasons", "leagues")


def frozen_settings(settings: dict) -> dict:
    """考试用到的设置（开考时记下来；中途出错重跑时要和开考时一模一样，不能看了分数再改设置）。done 不算（考完才改）。"""
    out = json.loads(json.dumps({k: settings[k] for k in FROZEN}, default=str))
    out["final_exam"].pop("done", None)
    return out


def start_record(settings: dict, run_dir, git: dict, resume: bool, reason: str | None) -> dict:
    """正式考开考前调用（check_not_done 之后）。

    不是重跑（resume = False）：结果文件夹里不能已经有 exam_started.json（开考过一次就不能当新考试再来），写一份开考记录。
    重跑（resume = True，--skip-backtest）：必须已经有开考记录、设置和开考时一样、写了重跑原因；在记录里追加一条（时间、commit、原因）。
    这是需求文档允许的“程序算错可以改程序重算”，每次重跑都留记录，考完一起写进 exam_done.json。
    """
    path = run_dir / STARTED_FILE
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if not resume:
        if path.exists():
            raise ExamAlreadyDone(f"{path} 已经有了：期末考试开考过。程序出错要重算，用 --skip-backtest 加 --reason 说明原因")
        rec = {"开考时间_utc": now, "git": git, "设置": frozen_settings(settings), "重跑": []}
    else:
        if not path.exists():
            raise RuntimeError(f"没有开考记录 {path}：--skip-backtest 只能用在开考过、中途出错的考试上")
        if not reason:
            raise RuntimeError("重跑要用 --reason 写清楚是什么程序错误、改了什么")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["设置"] != frozen_settings(settings):
            raise RuntimeError("设置和开考时不一样：考试中不能改设置（需求文档第 5 节“期末考试怎么考”）")
        rec["重跑"].append({"时间_utc": now, "git": git, "原因": reason})
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def git_state() -> dict:
    """现在代码是哪一次提交、有没有没提交的改动（写进 exam_done.json，以后能查当时用的是哪份代码和设置）。"""
    def run(*args):
        r = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")          # 新建还没加进 Git 的文件也算改动（data/ 等不进 Git 的不算）
    return {"commit": run("rev-parse", "HEAD"), "有没提交的改动": None if status is None else bool(status)}


def season_text(seasons) -> np.ndarray:
    """赛季数字 → '2024/25'（分赛季报告用）。"""
    return np.array([season_label(int(s)) for s in seasons])


# ======================= 第 1 条：模型本身对两个收盘标杆 =======================
def model_check(pred: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """gb_l7（T-24）对 平博收盘、多家平均收盘：同一批比赛（两边都有分数的），log loss、Brier、RPS，全部和分赛季。只报告。"""
    name, hz = settings["fusion"]["model"], settings["fusion_study"]["horizon"]
    cols = ["match_id", "match_date", "season", "log_loss", "brier", "rps"]
    model = pred[(pred["model"] == name) & (pred["horizon"] == hz)][cols].drop_duplicates("match_id")
    model = model.assign(model=name)
    rows = []
    for bname, bhz in BENCH:
        b = pred[(pred["model"] == bname) & (pred["horizon"] == bhz)][cols].drop_duplicates("match_id")
        b = b.assign(model=bname)
        for part in ["全部", *sorted(model["season"].unique())]:
            pick = (lambda d: d) if part == "全部" else (lambda d, s=part: d[d["season"] == s])
            sc = pd.concat([pick(b), pick(model)], ignore_index=True)
            for r in fu.compare_scores(sc, bname, name, ["log_loss", "brier", "rps"], settings):
                rows.append({"哪一份": part if part == "全部" else season_label(int(part)), "标杆时点": bhz, **r})
    return pd.DataFrame(rows)


# ======================= 第 2 条：融合 对 只用模型 =======================
def exam_segments(t: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """全部、每个联赛、每个赛季（t["period"] 已经是赛季）、胜平负用不用平博。"""
    out = [("赛季" if kind == "两段" else kind, name, mask) for kind, name, mask in fu.segments(t)]
    pin = (t["mo_pinnacle"] == 1).to_numpy()
    out += [("胜平负用不用平博", "用平博", pin), ("胜平负用不用平博", "多家平均", ~pin)]
    return [s for s in out if s[2].any()]


def fusion_check(t: pd.DataFrame, model: dict, market: dict, pred: pd.DataFrame, settings: dict
                 ) -> tuple[pd.DataFrame, dict]:
    """融合（w = fusion.w）对 只用模型（w = 0），全部六个分数；融合 对 平博收盘（胜平负三个分数，两边都有的比赛）。

    “新 − 旧”的差：第一种 新 = 融合，旧 = 只用模型（“更准”= 融合更准）；
    第二种和 W9 一样 旧 = 融合，新 = 平博收盘（“更准”= 平博收盘比融合准，收盘赔率本来就更准，只报告）。
    返回 (表, 融合后的各玩法概率)。
    """
    fused = fu.mix(model, market, settings["fusion"]["w"])
    y, over, btts = t["y"].to_numpy(), t["over"].to_numpy(), t["btts_yes"].to_numpy()
    base = t[["match_id", "match_date"]]
    sc = pd.concat([base.assign(model=MODEL, **fu.per_match_scores(model, y, over, btts)),
                    base.assign(model=FUSED, **fu.per_match_scores(fused, y, over, btts))], ignore_index=True)
    pin = pred[(pred["model"] == fu.PIN) & (pred["horizon"] == "收盘")].dropna(subset=["log_loss", "brier", "rps"])
    pin = base.merge(pin.drop_duplicates("match_id")[["match_id", "log_loss", "brier", "rps"]], on="match_id")
    pin = pin.assign(model=fu.PIN)
    rows = []
    for kind, name, mask in exam_segments(t):
        ids = set(t.loc[mask, "match_id"])
        s = sc[sc["match_id"].isin(ids)]
        rows += [dict(r, 分法=kind, 哪一份=name) for r in fu.compare_scores(s, MODEL, FUSED, fu.SCORES, settings)]
        p = pin[pin["match_id"].isin(ids)]
        s2 = pd.concat([s[(s["model"] == FUSED) & s["match_id"].isin(set(p["match_id"]))], p], ignore_index=True)
        rows += [dict(r, 分法=kind, 哪一份=name) for r in fu.compare_scores(s2, FUSED, fu.PIN, ["log_loss", "brier", "rps"],
                                                                          settings)]
    out = pd.DataFrame(rows)
    first = ["分法", "哪一份"]
    return out[[*first, *[c for c in out.columns if c not in first]]], fused


def fusion_passed(table: pd.DataFrame) -> bool:
    """第 2 条通过：全部比赛、融合 对 只用模型、胜平负 log loss 的区间整个在“融合更准”一边（verdict = 更准）。"""
    r = table[(table["分法"] == "全部") & (table["base"] == MODEL) & (table["new"] == FUSED)
              & (table["metric"] == "log_loss")]
    return bool(len(r)) and r["verdict"].iloc[0] == BETTER


# ======================= 第 3 条：倾向（现在的门槛） =======================
def hit_rate(name: str, v: np.ndarray) -> float:
    """偏向那边发生的比例，和 fusion_study.lean_study 同一个算法（亚盘：赢半算半个、走盘不计；平局：真的平了的比例）。"""
    if name == "ah":
        return float(np.clip(v, 0, None).sum() / np.abs(v).sum()) if np.abs(v).sum() > 0 else np.nan
    return float((v > 0).mean())


def lean_check(t: pd.DataFrame, fused: dict, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个玩法：现在的门槛（settings 的 lean）下有倾向的场数、比例、偏向那边发生的比例、“实际”的平均和 95% 区间（按周成块）。

    “实际”和开发赛季同一个定义（fusion_study.lean_values）：平局那条减的是考试这两季自己的平局比例。
    通过 = 区间下限 > 0。另外看开发赛季的平均落不落在区间里，和分档表（只报告）。门槛是 null 的玩法没法考，写“没有门槛”。
    """
    c, st = settings["compare"], settings["fusion_study"]
    ref = settings["final_exam"]["dev_reference"]["lean_actual_mean"]
    vals = fu.lean_values(t, fused)
    dates = pd.Series(pd.to_datetime(t["match_date"]).to_numpy())
    rows, bands = [], []
    for name in fu.LEAN_MARKETS:
        s, v = vals[name]
        ok = ~np.isnan(s) & ~np.isnan(v)
        thr = settings["lean"][name]
        row = {"玩法": name, "门槛": thr, "可算的场数": int(ok.sum()), "开发赛季的平均": ref.get(name)}
        if thr is None:
            row["通过"] = "没有门槛"
        else:
            k = ok & (s >= thr - 1e-12)
            row.update({"有倾向的场数": int(k.sum()), "有倾向的比例": float(k.sum() / ok.sum()) if ok.any() else np.nan})
            if k.any():
                mean, lo, hi = bootstrap_mean_diff(v[k], block_ids(dates[k], c["block"]), c["n_boot"], c["confidence"],
                                                   c["seed"])
                r = ref.get(name)
                row.update({"偏向那边发生的比例": hit_rate(name, v[k]), "实际的平均": mean, "区间下限": lo, "区间上限": hi,
                            "通过": bool(lo > 0),
                            "开发赛季的平均在区间里": None if r is None else bool(lo <= r <= hi)})
            else:
                row["通过"] = "没有有倾向的比赛"
        rows.append(row)
        b = fu.lean_bands(s, v, t["match_date"], st["lean_bands"][name], st["lean_min_gap"][name], settings)
        bands.append(b.assign(玩法=name))
    bands = pd.concat(bands, ignore_index=True)
    return pd.DataFrame(rows), bands[["玩法", *[col for col in bands.columns if col != "玩法"]]]


# ======================= 第 4 条：信心等级 =======================
def confidence_check(pred: pd.DataFrame, odds: pd.DataFrame, promoted: pd.Series, settings: dict
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """现在的规则（confidence.downgrade_factors、no_odds_downgrade）下每个等级的场数、d 平均；四个候选因素再算一次（只报告）。

    d = 融合赛前 log loss − 平博收盘 log loss，只有平博赛前 + 平博收盘的比赛能算（和开发赛季同一个函数 confidence_study.build）。
    build 里的融合概率按 w = 1 算（有赔率 = 赔率概率），所以 fusion.w 不是 1 时这条不对，直接报错。
    规则：“高”的 d 不大于“中”“低”。哪个等级没有能算 d 的比赛，就“比不了”（如实写）。
    """
    if float(settings["fusion"]["w"]) != 1.0:
        raise ValueError("confidence_study.build 按 w = 1 算融合概率，fusion.w 改了要先改 build")
    t = cs.build(pred, odds, settings, promoted=promoted)
    t["period"] = season_text(t["season"])
    cfg = settings["confidence"]
    t["level"] = cs.levels(t, list(cfg["downgrade_factors"]), cfg["no_odds_downgrade"])
    lt = cs.level_table(t)
    ft = cs.factor_table(t, settings).rename(columns={"算数": "算数（只报告，不改规则）"})
    d = dict(zip(lt["等级"], lt["d 平均"]))
    high = d[cs.LEVEL_NAMES[3]]
    others = {k: v for k, v in d.items() if k != cs.LEVEL_NAMES[3] and not np.isnan(v)}
    if np.isnan(high) or not others:
        rule = "比不了"
    else:
        rule = bool(all(high <= v for v in others.values()))
    check = {"有预测和结果的比赛": int(len(t)), "有赛前赔率": int(t["has_odds"].sum()),
             "没有赔率（只出模型概率）": int((~t["has_odds"]).sum()), "胜平负用平博": int((t["mo_pinnacle"] == 1).sum()),
             "能算 d 的比赛（平博赛前 + 平博收盘）": int(t["d_ok"].sum()),
             "d 平均": float(t["d"].mean()) if t["d_ok"].any() else None,
             "两家都有赛前的比赛（“没有平博”那条能不能算）": int(t["pair_ok"].sum()),
             "各因素场数（全部比赛）": {cs.FACTOR_NAMES[f]: int(t[f].sum()) for f in cs.FACTORS},
             "现在的降级因素": list(cfg["downgrade_factors"]), "“高”的 d 不大于“中”“低”": rule}
    return lt, ft, check


# ======================= 第 5 条：校准 =======================
def calibration_check(t: pd.DataFrame, fused: dict, step: float) -> pd.DataFrame:
    """融合后的主胜、平、客胜、大 2.5 球、双方进球，每 step 一档（下沿 ≤ 概率 < 上沿，1.0 算进最后一档）：场数、预测平均、实际比例。"""
    p3 = fused["p3"] / fused["p3"].sum(axis=1, keepdims=True)
    y = t["y"].to_numpy()
    items = [("主胜", p3[:, 0], y == 0), ("平", p3[:, 1], y == 1), ("客胜", p3[:, 2], y == 2),
             ("大 2.5 球", fused["over25"], t["over"].to_numpy() == 1), ("双方进球", fused["btts"], t["btts_yes"].to_numpy() == 1)]
    n_bins = int(round(1 / step))
    rows = []
    for name, p, happened in items:
        b = np.minimum(np.floor(p / step + 1e-9).astype(int), n_bins - 1)
        for i in range(n_bins):
            k = b == i
            if k.any():
                rows.append({"玩法": name, "下沿": round(i * step, 10), "上沿": round((i + 1) * step, 10), "n": int(k.sum()),
                             "预测平均": float(p[k].mean()), "实际比例": float(happened[k].mean()),
                             "实际 − 预测": float(happened[k].mean() - p[k].mean())})
    return pd.DataFrame(rows)


# ======================= 五条一起算 =======================
def promoted_history(fs, first: int, last: int, settings: dict) -> pd.Series:
    """first − 1 到 last 赛季的全部联赛比赛（只要比赛信息），算每场有没有升班马。
    2024/25 的上一季 2023/24 不在考试的预测里，要从这里拿；first − 1 那季自己的结果不用（只用来看“上赛季在不在”）。"""
    b = settings["backtest"]
    m = fs.prediction_targets(f"{first - 1}-{b['window_start_md']}", f"{last + 1}-{b['window_end_md']}", "A版输入",
                              competition=list(settings["leagues"]["top"]))
    m = m[(m["season"] >= first - 1) & (m["season"] <= last)]
    return cs.promoted_flags(m)


def run_exam(pred: pd.DataFrame, ids: np.ndarray, mats: np.ndarray, odds: pd.DataFrame, promoted: pd.Series,
             settings: dict) -> dict:
    """五条检查都算一次，返回各张表和摘要（main 负责存文件）。不选任何数，设置全用 settings 里现在的。"""
    model_tab = model_check(pred, settings)
    t, model, market, check = fu.build(pred, ids, mats, odds, settings)
    t["period"] = season_text(t["season"])
    fusion_tab, fused = fusion_check(t, model, market, pred, settings)
    grid, _ = fu.grid_table(t, model, market, fu.w_grid(settings["fusion_study"]["w_step"]))
    leans, bands = lean_check(t, fused, settings)
    levels, factors, conf = confidence_check(pred, odds, promoted, settings)
    calib = calibration_check(t, fused, settings["final_exam"]["calibration_step"])

    ref = settings["final_exam"]["dev_reference"]
    main_ll = model_tab[(model_tab["哪一份"] == "全部") & (model_tab["metric"] == "log_loss")]
    val = lambda d, col: float(d[col].iloc[0]) if len(d) else None
    pin_row = main_ll[main_ll["base"] == "pinnacle_close"]
    avg_row = main_ll[main_ll["base"] == "market_avg"]
    lean_pass = {r["玩法"]: r["通过"] for _, r in leans.iterrows()}
    summary = {
        "数据": check,
        "第 1 条 模型本身（只报告）": {
            "gb_l7 log loss（和平博收盘同一批）": val(pin_row, "new_mean"), "平博收盘 log loss": val(pin_row, "base_mean"),
            "同一批场数（平博）": val(pin_row, "n"),
            "gb_l7 log loss（和多家平均收盘同一批）": val(avg_row, "new_mean"),
            "多家平均收盘 log loss": val(avg_row, "base_mean"), "同一批场数（多家平均）": val(avg_row, "n"),
            "开发赛季参考（2014–2023）": {"gb_l7": ref["model_log_loss_2014_2023"],
                                     "平博收盘": ref["pinnacle_log_loss_2014_2023"]}},
        "第 2 条 融合 对 只用模型": {"w": float(settings["fusion"]["w"]), "通过": fusion_passed(fusion_tab)},
        "第 3 条 倾向": lean_pass,
        "第 4 条 信心等级": conf,
        "第 5 条 校准（只报告）": f"见 {OUTPUTS['calibration']}，没有事先定的通过标准，不在这里给一个数",
    }
    return {"summary": summary, "model": model_tab, "fusion": fusion_tab, "w_grid": grid, "leans": leans,
            "lean_bands": bands, "levels": levels, "factors": factors, "calibration": calib}


def reference_check(full: pd.DataFrame, ref_dir, model_name: str, first: int) -> dict:
    """打分赛季之前（热身）的模型预测和参考回测（同一个起点、同一份代码规则）比：比赛一样、概率最大差。只报告。"""
    ref = load_predictions(ref_dir)
    pick = lambda d: (d[(d["model"] == model_name) & (d["horizon"] == "T-24") & (d["season"] < first)]
                      .drop_duplicates("match_id").set_index("match_id").sort_index())
    a, b = pick(full), pick(ref)
    both = a.index.intersection(b.index)
    gap = np.abs(a.loc[both, ["p_home", "p_draw", "p_away"]].to_numpy(float)
                 - b.loc[both, ["p_home", "p_draw", "p_away"]].to_numpy(float))
    return {"参考": str(ref_dir), "这次的场数": int(len(a)), "参考的场数": int(len(b)), "两边都有": int(len(both)),
            "胜平负概率最大差": float(np.nanmax(gap)) if len(both) else None,
            "两边都没出预测的不算，一边出一边没出的场数": int(sum(a.loc[both, "p_home"].isna() != b.loc[both, "p_home"].isna()))}


def check_reference(reference: dict | None) -> None:
    """正式考：热身赛季的预测必须和参考回测一模一样（最大差 0、没有一边有一边没有），否则说明代码或数据和开发时不一样，不打分。"""
    if reference is None:
        raise RuntimeError("正式考要给 --reference（W9 的 w9_gb_l7），先核对热身赛季")
    gap, one_side = reference["胜平负概率最大差"], reference["两边都没出预测的不算，一边出一边没出的场数"]
    if gap is None or gap != 0 or one_side != 0 or reference["两边都有"] != reference["参考的场数"]:
        raise RuntimeError(f"热身赛季和参考回测对不上，不打分：{reference}")


def exam_guards(settings: dict, run_dir, git: dict, args) -> dict | None:
    """正式考开考前的检查（试跑不查，返回 None）：
    没考过；代码和设置都提交了；给了 --reference；然后写（或追加）开考记录 exam_started.json。"""
    if args.dry_run:
        return None
    check_not_done(settings, run_dir)
    if git["有没提交的改动"] is not False:        # 考试用的代码和设置要是一次提交（以后能查），有改动或查不到都不考
        raise RuntimeError(f"正式考试前要先提交全部改动（git 状态：{git}）")
    if not args.reference:
        raise RuntimeError("正式考要给 --reference（W9 的 w9_gb_l7），先核对热身赛季")
    return start_record(settings, run_dir, git, args.skip_backtest, args.reason)


# ======================= 主程序 =======================
OUTPUTS = {"model": "exam_model.csv", "fusion": "exam_fusion.csv", "w_grid": "exam_w_grid.csv", "leans": "exam_leans.csv",
           "lean_bands": "exam_lean_bands.csv", "levels": "exam_levels.csv", "factors": "exam_factors.csv",
           "calibration": "exam_calibration.csv"}


def main() -> None:
    settings = load_settings()
    fe = settings["final_exam"]
    ap = argparse.ArgumentParser(description="期末考试（预测部分，只考一次）")
    ap.add_argument("--db", required=True, help="数据库快照（只读打开）")
    ap.add_argument("--dry-run", action="store_true", help="在开发赛季试跑（开发模式，不算考试）")
    ap.add_argument("--dry-season", type=int, default=settings["seasons"]["dev_last"], help="试跑哪个开发赛季")
    ap.add_argument("--reference", default=None,
                    help="参考回测的结果文件夹（例如 W9 的 w9_gb_l7）：打分赛季之前的预测应和它一模一样，核对结果写进摘要")
    ap.add_argument("--skip-backtest", action="store_true",
                    help="回测已经跑完、后面一步出了程序错误时用：直接读上次存的预测、矩阵和赔率"
                         "（正式考要有开考记录、设置没变、还没写 exam_done.json）")
    ap.add_argument("--reason", default=None, help="正式考用 --skip-backtest 重跑时必须写：是什么程序错误、改了什么")
    a = ap.parse_args()

    # first–last：打分的赛季；回测从 backtest_first 连着跑到 last（重训日期和 W9 一样），前面的赛季只当“热身”、不打分
    if a.dry_run:
        mode, first, last, name = "开发", a.dry_season, a.dry_season, fe["dry_run_name"]
    else:
        mode, (first, last), name = "考试", settings["seasons"]["exam"], fe["run_name"]
    bt_first = int(fe["backtest_first"])
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / name
    git = git_state()
    model_name = settings["fusion"]["model"]
    if settings["fusion_study"]["model"] != model_name or settings["confidence_study"]["model"] != model_name:
        raise ValueError("fusion.model、fusion_study.model、confidence_study.model 要是同一个模型")
    print(f"{'试跑' if a.dry_run else '期末考试'}：{mode}模式，回测 {season_label(bt_first)} 起，"
          f"打分 {season_label(first)}–{season_label(last)}，结果放 {run_dir}")
    started = exam_guards(settings, run_dir, git, a)

    fs = FeatureStore(open_readonly(a.db), mode, settings)
    t0 = time.perf_counter()
    if a.skip_backtest:
        out = run_dir
    else:
        models = [MatrixRecorder(build_model(model_name, settings))]
        res = run_backtest(fs, list(settings["leagues"]["top"]), bt_first, last, models, [24], benchmark=True)
        out = save(res, name, settings)
        save_matrices(models, out)
    full = load_predictions(out)
    reference = reference_check(full, a.reference, model_name, first) if a.reference else None
    if not a.dry_run:
        check_reference(reference)
    pred = full[(full["season"] >= first) & (full["season"] <= last)].reset_index(drop=True)
    npz = np.load(out / f"matrices_{model_name}.npz")
    hz = settings["fusion_study"]["horizon"]
    odds_file = out / "fusion_odds.csv"
    if a.skip_backtest and odds_file.exists():
        odds = pd.read_csv(odds_file, encoding="utf-8-sig")
    else:
        targets = pred[(pred["model"] == model_name) & (pred["horizon"] == hz)][["match_id", "cutoff_utc"]]
        odds = fu.fetch_odds(fs, targets.drop_duplicates("match_id"), settings)
        odds.to_csv(odds_file, index=False, encoding="utf-8-sig", float_format="%.6f")
    promoted = promoted_history(fs, first, last, settings)
    res = run_exam(pred, npz["match_id"], npz["matrix"], odds, promoted, settings)

    summary = {"试跑" if a.dry_run else "期末考试": True, "运行模式": mode, "打分的赛季": [first, last],
               "回测起点赛季（之前的只热身）": bt_first, "和参考回测核对（打分赛季之前）": reference, "git": git,
               "设置": {k: settings[k] for k in ("fusion", "lean", "confidence")}, **res["summary"],
               "秒数": round(time.perf_counter() - t0, 1)}
    for key, fname in OUTPUTS.items():
        res[key].to_csv(out / fname, index=False, encoding="utf-8-sig", float_format="%.6f")
    (out / "exam_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")
    if not a.dry_run:
        check_not_done(settings, out)              # 写之前再查一次
        done = {"考完时间_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "赛季": [first, last],
                "git": git, "开考记录": started, "说明": "期末考试（预测部分）已经考过，只考一次。接着把 settings.yaml 的 final_exam.done 改成 true 提交。"}
        (out / DONE_FILE).write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    for key in ("model", "leans", "levels"):
        print(res[key].to_string(index=False))
    f = res["fusion"]
    print(f[(f["metric"] == "log_loss")].to_string(index=False))
    print(f"结果存在 {out}")


if __name__ == "__main__":
    main()
