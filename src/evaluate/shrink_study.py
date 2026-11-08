'''复盘问题回测：模型是不是把强弱差距、进球数估小了（需求文档第 15 节“复盘问题回测”，跑之前写下的规则）。

起因：2026-10-30 到 11-06 那 47 场里，模型自己那列让球大多挑弱队、大小球大多挑小，强队大胜时一起错。
猜想：模型的让球、大小球胜算太靠近 50%。这里用开发赛季的历史比赛查一遍，看是不是一直这样。

不重新预测：直接读亚盘、大小球收盘线研究（src/evaluate/line_clv_study.py）算好的每场表 per_match.csv，
里面每场有：玩法 market（ah 亚盘 / ou 大小球）、主队盘口 line、模型胜算 p、平博赛前去水胜算 m（都是“主队这边”或“大球这边”）、
这一边的实际结果 out_a（win / half_win / push / half_loss / loss）。

“弱队那边”：亚盘受让的一边（平手盘没有弱队，不算）、大小球的小球。
两个主判断（亚盘、大小球分开，只看五大联赛）：
  1. 弱队那边整体：所有有弱队的比赛，实际赢的比例 − 模型说的胜算。< 0：模型一直高估弱队，也就是把差距估小了。
  2. 挑弱队时：只看模型挑了弱队那边的比赛，同样算实际 − 模型。< 0：模型挑弱队时太乐观。
  区间都用按周成块的 bootstrap。
为什么不用“校准斜率 b > 1”当主判断：平博的线本来就把两边定在 50% 左右，模型把差距估小，表现是胜算一直往弱队那边偏，
不是“太靠近 50%”，这时斜率反而 < 1。斜率只列出来看看（b < 1：模型离 50% 偏得太多）。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.shrink_study
结果在 settings 的 shrink_study.output_dir（data/ 不进 Git）：report.md、results.csv。
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.evaluate.compare import block_ids

# 每种结果换成（赢了多少 y, 权重 w）：赢半、输半各算半场，走盘不算。和第 3 节“胜算”的算法一样。
OUTCOME_YW = {"win": (1.0, 1.0), "half_win": (1.0, 0.5), "push": (0.0, 0.0), "half_loss": (0.0, 0.5), "loss": (0.0, 1.0)}
BOTH = "模型把差距估小了，历史上也是这样"
ONLY_ALL = "模型整体偏向弱队，但挑弱队那些场看不出来，不够确定"
ONLY_PICKED = "不是一直偏向弱队；挑弱队时太乐观是模型本身有误差（自己挑的那边都会偏乐观）"
NONE = "历史上没这个毛病，这一周挑弱队、挑小多半是强队刚好赢得多"


def load_rows(path, settings: dict) -> pd.DataFrame:
    '''读每场表，查赛季都是开发赛季（不是就报错），换成“弱队那边”的写法。

    每行加：y、w（主队 / 大球这边的结果）；dog_is_a（弱队是不是主队 / 大球这边）；
    q（模型说弱队那边的胜算）、qm（平博说弱队那边的胜算）、yd（弱队那边实际赢没赢）。
    亚盘平手盘（line = 0）没有弱队，q、qm、yd 是空的。
    '''
    d = pd.read_csv(path, encoding="utf-8-sig")
    bad = sorted({int(s) for s in d["season"].unique() if season_role(int(s), settings) != "dev"})
    if bad:
        raise ValueError(f"每场表里有不是开发赛季的比赛：{bad}，停下")
    yw = d["out_a"].map(OUTCOME_YW)
    if yw.isna().any():
        raise ValueError(f"认不出的结果：{sorted(d.loc[yw.isna(), 'out_a'].unique())}")
    d["y"] = [v[0] for v in yw]
    d["w"] = [v[1] for v in yw]
    is_ah = d["market"] == "ah"
    # 亚盘：主队盘口 > 0 是主队受让（主队是弱队）；< 0 客队是弱队；= 0 没有弱队。大小球：小球那边当“弱队”。
    d["dog_is_a"] = np.where(is_ah, d["line"] > 0, False)
    has_dog = ~is_ah | (d["line"] != 0)
    d["q"] = np.where(has_dog, np.where(d["dog_is_a"], d["p"], 1 - d["p"]), np.nan)
    d["qm"] = np.where(has_dog, np.where(d["dog_is_a"], d["m"], 1 - d["m"]), np.nan)
    d["yd"] = np.where(has_dog, np.where(d["dog_is_a"], d["y"], 1 - d["y"]), np.nan)
    return d


def logit(p, clip):
    p = np.clip(np.asarray(p, dtype=float), *clip)
    return np.log(p / (1 - p))


def fit_slope(x, y, w, iters: int = 50) -> tuple[float, float]:
    '''带权重的 logistic 回归 y ~ a + b·x（牛顿法），返回 (a, b)。'''
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        g = X.T @ (w * (y - mu))
        H = X.T @ (X * (w * mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def weighted_gap(y, q, w) -> float:
    '''实际赢的比例 − 模型说的胜算（都按权重平均）。'''
    return float(np.sum(w * (y - q)) / np.sum(w))


def bootstrap(df: pd.DataFrame, stat, n_boot: int, confidence: float, seed: int) -> tuple[float, float, float]:
    '''按周成块 bootstrap：返回 (原数据上的值, 区间下限, 区间上限)。stat 是“一张表 → 一个数”的函数。'''
    if df.empty:
        return np.nan, np.nan, np.nan
    est = stat(df)
    blocks = block_ids(df["match_date"], "week")
    groups = [np.flatnonzero(blocks == b) for b in pd.unique(blocks)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(df.iloc[np.concatenate([groups[i] for i in pick])]))
    a = (1 - confidence) / 2
    return est, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def verdict(all_dog_hi: float, picked_dog_hi: float) -> str:
    '''两个“实际 − 模型”的区间上限 < 0 才算过（规则见文件开头和需求文档）。
    只有判断 1 能说明“一直偏向弱队”：判断 2 只看模型自己挑的场，模型只要有误差，自己挑的那边就会偏乐观，
    不偏的模型也常常过判断 2（写测试时用假数据看到的），所以只过判断 2 单独写一种结论。'''
    a, b = all_dog_hi < 0, picked_dog_hi < 0
    return BOTH if a and b else ONLY_ALL if a else ONLY_PICKED if b else NONE


def analyze(d: pd.DataFrame, settings: dict) -> pd.DataFrame:
    '''每个组、每个玩法算一遍。每项一行：组、玩法、项目、值、下限、上限、场数。'''
    c, s = settings["compare"], settings["shrink_study"]
    clip = tuple(s["clip"])
    boot = dict(n_boot=c["n_boot"], confidence=c["confidence"], seed=c["seed"])
    out = []

    def add(group, market, item, v, lo=np.nan, hi=np.nan, n=np.nan):
        out.append({"组": group, "玩法": market, "项目": item, "值": v, "下限": lo, "上限": hi, "场数": n})

    for (group, market), g in d.groupby(["group", "market"], sort=False):
        g = g[(g["w"] > 0) & g["p"].notna() & g["m"].notna()].reset_index(drop=True)
        name = "亚盘" if market == "ah" else "大小球"

        def slope_of(col):
            return lambda t: fit_slope(logit(t[col], clip), t["y"].to_numpy(), t["w"].to_numpy())[1]

        dog = g[g["q"].notna()].reset_index(drop=True)
        picks_dog = dog[dog["q"] > 0.5].reset_index(drop=True)
        picks_fav = dog[dog["q"] < 0.5]
        gap = lambda t: weighted_gap(t["yd"].to_numpy(), t["q"].to_numpy(), t["w"].to_numpy())
        gap_m = lambda t: weighted_gap(t["yd"].to_numpy(), t["qm"].to_numpy(), t["w"].to_numpy())
        add(group, name, "弱队那边整体：实际 − 模型", *bootstrap(dog, gap, **boot), len(dog))
        add(group, name, "弱队那边整体：实际 − 平博（参照）", *bootstrap(dog, gap_m, **boot), len(dog))
        add(group, name, "挑弱队时：实际 − 模型", *bootstrap(picks_dog, gap, **boot), len(picks_dog))
        add(group, name, "挑弱队时：模型说的胜算", float(np.average(picks_dog["q"], weights=picks_dog["w"])), n=len(picks_dog))
        add(group, name, "挑弱队时：实际赢的比例", float(np.average(picks_dog["yd"], weights=picks_dog["w"])), n=len(picks_dog))
        # 挑强队时：强队那边的实际 − 模型 = −(弱队那边的实际 − 模型)
        add(group, name, "只看看：挑强队时：实际 − 模型", -gap(picks_fav), n=len(picks_fav))
        add(group, name, "只看看：挑弱队的场数比例", len(picks_dog) / max(len(picks_dog) + len(picks_fav), 1),
            n=len(picks_dog) + len(picks_fav))
        add(group, name, "只看看：模型胜算离 50% 平均多远", float(np.mean(np.abs(g["p"] - 0.5))), n=len(g))
        add(group, name, "只看看：平博胜算离 50% 平均多远", float(np.mean(np.abs(g["m"] - 0.5))), n=len(g))
        add(group, name, "只看看：斜率 b（模型）", *bootstrap(g, slope_of("p"), **boot), len(g))
        add(group, name, "只看看：斜率 b（平博）", *bootstrap(g, slope_of("m"), **boot), len(g))
        for season, gs in g.groupby("season"):
            add(group, name, f"只看看：{season} 赛季斜率 b（模型）", slope_of("p")(gs), n=len(gs))
    return pd.DataFrame(out)


def verdicts(res: pd.DataFrame, main_group: str) -> dict:
    '''主判断那一组，亚盘、大小球各一个结论。'''
    out = {}
    for market in ["亚盘", "大小球"]:
        r = res[(res["组"] == main_group) & (res["玩法"] == market)].set_index("项目")
        if r.empty:
            continue
        out[market] = verdict(r.loc["弱队那边整体：实际 − 模型", "上限"], r.loc["挑弱队时：实际 − 模型", "上限"])
    return out


def fmt(v, lo, hi, item):
    pct = "比例" in item or "胜算" in item or "实际" in item or "多远" in item
    f = (lambda x: f"{x:+.1%}" if "−" in item else f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
    return f(v) + ("" if pd.isna(lo) else f"（{f(lo)} ~ {f(hi)}）")


def report(res: pd.DataFrame, verd: dict, settings: dict) -> str:
    s = settings["shrink_study"]
    model = s.get("model", "gb_l7")
    lines = ["# 复盘问题回测：模型是不是把强弱差距、进球数估小了", "",
             f"{model} T-24，平博赛前亚盘线、大小球 2.5，2019–2023 开发赛季。规则见需求文档第 15 节“复盘问题回测”。", "",
             f"## 结论（只看{s['main_group']}）", ""]
    lines += [f"- {k}：{v}" for k, v in verd.items()]
    live = s["live_dog_share"]
    lines += ["", f"对照起因那 47 场：让球挑弱队 {live['ah'][0]}/{live['ah'][1]}，大小球挑小 {live['ou'][0]}/{live['ou'][1]}。", ""]
    for (group, market), g in res.groupby(["组", "玩法"], sort=False):
        lines += [f"## {group} {market}", "", "| 项目 | 值（95% 区间） | 场数 |", "|---|---|---|"]
        lines += [f"| {r['项目']} | {fmt(r['值'], r['下限'], r['上限'], r['项目'])} | {int(r['场数'])} |" for _, r in g.iterrows()]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="复盘问题回测（需求文档第 15 节）")
    ap.add_argument("--model", default=None, help="覆盖 shrink_study.model（报告抬头，例如 gb_xg）")
    ap.add_argument("--per-match", default=None, help="覆盖 shrink_study.per_match（收盘线研究的每场表）")
    ap.add_argument("--out", default=None, help="覆盖 shrink_study.output_dir")
    a = ap.parse_args(argv)
    settings = load_settings()
    s = settings["shrink_study"]
    if a.model:
        s["model"] = a.model
    if a.per_match:
        s["per_match"] = a.per_match
    if a.out:
        s["output_dir"] = a.out
    d = load_rows(resolve_path(s["per_match"]), settings)
    res = analyze(d, settings)
    verd = verdicts(res, s["main_group"])
    out = resolve_path(s["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    (out / "report.md").write_text(report(res, verd, settings), encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(verd, ensure_ascii=False, indent=1), encoding="utf-8")
    text = report(res, verd, settings)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.replace("−", "-"))


if __name__ == "__main__":
    main()
