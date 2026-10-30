"""去水方法比较（需求文档第 16 节“去水方法”“去水方法比较”）：按比例、Shin 法、幂函数法，哪种去水后的概率最准。

一、在比什么
  同一组收盘赔率，用三种方法去水，得到三组“市场概率”。哪组和实际比分最接近（对数损失、Brier、RPS 越小越好），
  就说明那种方法更会“把抽水从赔率里拿掉”。赔率本身一样，差别只来自去水方法。
  三种方法的区别（详细见 src/common/odds.py）：
  - 按比例：每个结果的隐含概率除以总和，抽水按比例平摊；
  - Shin 法、幂函数法：认为庄家对冷门抽得更多，所以把冷门的概率压得更低、热门的概率抬高一点。
  足球里常说的“冷门偏差”（买冷门长期更亏）如果真的存在，后两种应该更准。

二、比较之前写下的计划（第 5 节“先写下要试哪几种，不临时加”），都在 settings.yaml 的 devig_compare 段：
  - 两组赔率分开比：平博收盘（2012/13 起）、多家平均收盘（2019/20 起）；10 个联赛；只用开发赛季；
  - 每组比 3 对：Shin 对按比例、幂函数对按比例、幂函数对 Shin；
  - 每对用 compare.py 的成块 bootstrap（按周）判断“更准 / 没差别 / 更差”，主要看对数损失。

三、数据从哪来
  通过取数层（开发模式，用途“评估”）拿以前比赛的收盘赔率和比分。截止时间 = 最后一个开发赛季之后的 7 月 1 日，
  所以期末考试赛季（2024/25 起）一场都拿不到；万一拿到，取数层会直接报错。

用法：
  .venv\\Scripts\\python -m src.evaluate.devig_compare
结果在 data/backtest/devig_compare/：
  compare.csv      6 组比较 × 3 个分数：平均分、平均差、95% 区间、结论
  by_season.csv    每组赔率 × 方法 × 赛季的平均分（只看不下结论）
  by_tier.csv      五大联赛、次级联赛分开的平均分（只看不下结论）
  longshot.csv     冷门：各方法给的平均概率和实际发生比例（只看不下结论）
  run_info.json    场数、平均抽水、设置
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.odds import devig, overround
from src.evaluate import metrics
from src.evaluate.compare import METRICS, compare
from src.features import FeatureStore, open_readonly

ODDS_COLS = ["home", "draw", "away"]
HORIZON = "收盘"   # compare() 要一个“时点”列；这里全是收盘赔率


def league_codes(settings: dict) -> list[str]:
    """devig_compare.leagues 里写的组（top / second）→ 联赛代码。"""
    return [code for group in settings["devig_compare"]["leagues"] for code in settings["leagues"][group]]


def dev_cutoff(settings: dict) -> str:
    """截止时间：最后一个开发赛季结束后的 7 月 1 日（backtest.window_end_md）。考试赛季 8 月才开始，一场都拿不到。"""
    return f"{settings['seasons']['dev_last'] + 1}-{settings['backtest']['window_end_md']} 00:00:00"


def load_data(fs: FeatureStore, settings: dict) -> pd.DataFrame:
    """每场比赛一行：联赛、赛季、日期、结果，加上每组赔率（列名 <组名>_home / _draw / _away）。

    只留开发赛季、比分已知、而且这组赔率三个都有的比赛；每组赔率只从它的起始赛季（first）开始用。
    """
    c = settings["devig_compare"]
    cutoff, comps = dev_cutoff(settings), league_codes(settings)
    res = fs.historical_matches(cutoff, "评估", competition=comps, data=("比分",))
    res = res[res["result"].notna() & (res["season"] <= settings["seasons"]["dev_last"])]
    out = res[["match_id", "competition", "season", "match_date", "result"]].copy()
    for name, src in c["sources"].items():
        o = fs.historical_odds(cutoff, "评估", timing="close", competition=comps,
                               bookmaker=src["bookmaker"], market="1x2")
        o = o[o["season"] >= src["first"]].dropna(subset=ODDS_COLS)
        o = o[(o[ODDS_COLS] > 1).all(axis=1)]                 # 赔率必须大于 1（导入时已筛过，这里再保险一次）
        o = o.drop_duplicates("match_id")[["match_id", *ODDS_COLS]]
        out = out.merge(o.rename(columns={k: f"{name}_{k}" for k in ODDS_COLS}), on="match_id", how="left")
    return out.reset_index(drop=True)


def score_methods(data: pd.DataFrame, source: str, methods: list[str]) -> pd.DataFrame:
    """一组赔率 × 每种去水方法 → 每场的概率和分数。返回和回测 predictions.csv 一样的列，可以直接交给 compare()。"""
    cols = [f"{source}_{k}" for k in ODDS_COLS]
    d = data.dropna(subset=cols)
    odds = d[cols].to_numpy(dtype=float)
    y = metrics.encode_results(d["result"])
    rows = []
    for m in methods:
        p = devig(odds, m)
        rows.append(pd.DataFrame({
            "match_id": d["match_id"].to_numpy(), "competition": d["competition"].to_numpy(),
            "season": d["season"].to_numpy(), "match_date": d["match_date"].to_numpy(),
            "model": m, "horizon": HORIZON, "source": source,
            "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "outcome": y,
            "overround": overround(odds),
            "log_loss": metrics.log_loss_per_match(p, y), "brier": metrics.brier_per_match(p, y),
            "rps": metrics.rps_per_match(p, y)}))
    return pd.concat(rows, ignore_index=True)


def longshot_table(scored: pd.DataFrame, below: float) -> pd.DataFrame:
    """冷门：按比例去水后概率低于 below 的每个选项（一场最多 3 个），各方法给它的平均概率 vs 实际发生比例。

    实际发生比例比按比例的平均概率低，说明冷门被高估了（冷门偏差），Shin / 幂函数把冷门压低就是对的方向。
    """
    parts = []
    for source, s in scored.groupby("source"):
        wide = {m: g.sort_values("match_id") for m, g in s.groupby("model")}
        base = wide["proportional"]
        p0 = base[["p_home", "p_draw", "p_away"]].to_numpy()
        happened = metrics.one_hot(base["outcome"].to_numpy(), 3)
        mask = p0 < below
        row = {"source": source, "longshot_below": below, "n_outcomes": int(mask.sum()),
               "observed_rate": float(happened[mask].mean())}
        for m, g in wide.items():
            row[f"mean_p_{m}"] = float(g[["p_home", "p_draw", "p_away"]].to_numpy()[mask].mean())
        parts.append(row)
    return pd.DataFrame(parts)


def run(fs: FeatureStore, settings: dict) -> dict[str, pd.DataFrame]:
    c = settings["devig_compare"]
    data = load_data(fs, settings)
    scored = pd.concat([score_methods(data, s, c["methods"]) for s in c["sources"]], ignore_index=True)
    top = set(settings["leagues"]["top"])
    scored["tier"] = np.where(scored["competition"].isin(top), "五大联赛", "次级联赛")

    rows = []
    for source, s in scored.groupby("source", sort=False):
        for base, new in c["pairs"]:
            r = compare(s, base, new, HORIZON, settings=settings)
            r.insert(0, "source", source)
            rows.append(r)
    comp = pd.concat(rows, ignore_index=True)

    agg = {m: (m, "mean") for m in METRICS}
    by_season = scored.groupby(["source", "model", "season"]).agg(n=("log_loss", "size"), **agg).reset_index()
    by_tier = scored.groupby(["source", "tier", "model"]).agg(n=("log_loss", "size"), **agg).reset_index()
    longshot = longshot_table(scored, float(c["longshot_below"]))
    first = scored[scored["model"] == c["methods"][0]]
    info = {
        "截止时间": dev_cutoff(settings), "联赛": league_codes(settings),
        "每组赔率的场数": first.groupby("source").size().to_dict(),
        "每组赔率的赛季": {s: [int(g["season"].min()), int(g["season"].max())] for s, g in first.groupby("source")},
        "每组赔率的平均赔率倒数之和": first.groupby("source")["overround"].mean().round(4).to_dict(),
        "设置": c, "比较设置": settings["compare"],
    }
    return {"compare": comp, "by_season": by_season, "by_tier": by_tier, "longshot": longshot, "info": info}


def save(result: dict, settings: dict) -> Path:
    out = resolve_path(settings["devig_compare"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    kw = {"index": False, "encoding": "utf-8-sig", "float_format": "%.6f"}
    for name in ("compare", "by_season", "by_tier", "longshot"):
        result[name].to_csv(out / f"{name}.csv", **kw)
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(result["info"], f, ensure_ascii=False, indent=2, default=str)
    return out


def main() -> None:
    settings = load_settings()
    fs = FeatureStore(open_readonly(), "开发", settings)
    res = run(fs, settings)
    out = save(res, settings)
    pd.set_option("display.width", 220)
    fmt = {"float_format": lambda v: f"{v:.5f}", "index": False}
    print(json.dumps({k: v for k, v in res["info"].items() if k not in ("设置", "比较设置")}, ensure_ascii=False, indent=1))
    print(res["compare"].drop(columns=["block", "primary"]).to_string(**fmt))
    print(res["by_tier"].to_string(**fmt))
    print(res["longshot"].to_string(**fmt))
    print(f"\n结果存在 {out}")


if __name__ == "__main__":
    main()
