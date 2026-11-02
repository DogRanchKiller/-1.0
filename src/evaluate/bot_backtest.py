"""虚拟投注机器人的历史回测入口（需求文档第 14 节“历史回测第一版规则 bots_v1”）。

全是虚拟投注，不连任何账户；结果只说“历史上这样下会怎样”，不写“推荐下注”。

做法：
  1. 读回测 bots.run（w9_gb_l7）的概率和比分矩阵，第一步只留开发赛季（src/bots/data.py）；
  2. 取数层用途「机器人下注」，每场取开球前 24 小时能看到的赛前胜平负（平博、多家最高、多家平均），算融合概率；
  3. 取数层用途「机器人结算」，站在开球后 72 小时取比分和平博收盘；
  4. 19 个机器人（settings bots.list）每个跑一遍：选注、注额、结算、CLV、资金模拟（src/bots/betting.py）；
  5. 成绩、排行榜、两段、每个赛季、合理性检查、中文报告（src/bots/stats.py）。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.bot_backtest
  只跑几个赛季看看程序通不通（结果写到另一个文件夹，不当正式结果）：
  .venv\\Scripts\\python -m src.evaluate.bot_backtest --first 2015 --last 2015 --out data/backtest/bots/试跑
结果在 data/backtest/bots/bots_v1/（data/ 不进 Git）：
  bets.csv       每注明细（成绩用的注额和盈亏 + 资金模拟的 sim_stake、sim_status、sim_profit）
  summary.csv    每个机器人一行，就是排行榜（按 CLV 平均从高到低；注数不够的列在下面）
  by_period.csv  两段（2012 之前 / 2012 起）分开的成绩
  by_season.csv  每个赛季分开的成绩
  bankroll.csv   资金模拟：最后资金、最高点、最大回撤、停没停、停在哪天、下了几注、单日上限跳过几注
  check.json     场数、随机机器人的理论值、赛季检查、可疑信号
  report.html    中文报告
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd

from src.bots import betting, data, stats
from src.common.config import load_settings, resolve_path
from src.features import FeatureStore, open_readonly


def run_all(settings: dict, fs: FeatureStore | None = None, run_dir: Path | None = None,
            out_dir: Path | None = None, first: int | None = None, last: int | None = None) -> dict:
    """跑全部机器人，写结果文件，返回 {表名: 表}。

    fs：取数层（不给就只读打开 settings 里的数据库，开发模式）；run_dir：回测结果文件夹（不给 = data/backtest/<bots.run>）；
    out_dir：结果放哪（不给 = bots.output_dir/<bots.version>）；first、last：只跑这几个赛季（试跑用，不能超出 bots 的范围）。
    """
    settings = copy.deepcopy(settings)
    cfg = settings["bots"]
    if first is not None:
        cfg["first_season"] = max(int(first), int(cfg["first_season"]))
    if last is not None:
        cfg["last_season"] = min(int(last), int(cfg["last_season"]))
    fs = fs or FeatureStore(open_readonly(), "开发", settings)
    if fs.mode != "dev":
        raise ValueError(f"机器人历史回测只能用开发模式（现在是 {fs.mode}）")
    t0 = time.perf_counter()
    pred, info_p = data.load_bot_predictions(cfg["run"], settings, run_dir=run_dir)
    matches, odds, fusion = data.load_bot_odds(fs, pred, settings)
    settle = data.load_settlement(fs, matches, settings)
    matches = matches.merge(settle, on="match_id", how="left")
    if not (matches["match_id"].to_numpy() == fusion["match_id"].to_numpy()).all():
        raise ValueError("比赛表和融合概率的行对不上（程序出错）")
    probs = {"model": matches[data.P3].to_numpy(float), "fusion": fusion[data.P3].to_numpy(float)}
    secs_load = round(time.perf_counter() - t0, 1)

    bots = cfg["list"]
    parts, sims = [], []
    for bot in bots:
        b, sim = betting.run_bot(bot, matches, probs, odds, settings)
        parts.append(b)
        sims.append(sim)
    bets = pd.concat(parts, ignore_index=True)
    summary = stats.summarize(bets, bots, settings)
    by_period = stats.by_group(bets, bots, settings, "period")
    by_season = stats.by_group(bets, bots, settings, "season")
    bankroll = pd.DataFrame(sims)
    theory = stats.random_theory(odds, settings)
    check = {
        "版本": cfg["version"], "回测": cfg["run"], "模型": cfg["model"], "时点": cfg["horizon"],
        "赛季": [cfg["first_season"], cfg["last_season"]],
        "场数": {**info_p, **{k: v for k, v in odds_info(odds, matches, fusion).items()},
                 "有比分的比赛": int(matches["result"].notna().sum()),
                 "没有比分、按“走”退回的比赛": int(matches["result"].isna().sum()),
                 "有平博收盘的比赛": int(matches["close_home"].notna().sum())},
        "每个赛季的比赛数": {int(k): int(v) for k, v in matches["season"].value_counts().sort_index().items()},
        "赛季检查": "都是开发赛季" if all(s <= settings["seasons"]["dev_last"] for s in matches["season"]) else "有问题",
        **stats.checks(summary, theory, bots),
        "取数秒": secs_load, "总秒数": round(time.perf_counter() - t0, 1),
    }
    out = Path(out_dir) if out_dir else resolve_path(cfg["output_dir"]) / cfg["version"]
    out.mkdir(parents=True, exist_ok=True)
    kw = {"index": False, "encoding": "utf-8-sig"}
    bets.to_csv(out / "bets.csv", float_format="%.6f", **kw)
    summary.to_csv(out / "summary.csv", float_format="%.5f", **kw)
    by_period.to_csv(out / "by_period.csv", float_format="%.5f", **kw)
    by_season.to_csv(out / "by_season.csv", float_format="%.5f", **kw)
    bankroll.to_csv(out / "bankroll.csv", float_format="%.2f", **kw)
    with open(out / "check.json", "w", encoding="utf-8") as f:
        json.dump(check, f, ensure_ascii=False, indent=2, default=str)
    (out / "report.html").write_text(stats.report_html(summary, by_period, bankroll, check, settings),
                                     encoding="utf-8")
    return {"bets": bets, "summary": summary, "by_period": by_period, "by_season": by_season,
            "bankroll": bankroll, "check": check, "out": out}


def odds_info(odds: dict, matches: pd.DataFrame, fusion: pd.DataFrame) -> dict:
    """每种价格有几场、几场不像样不下，有融合概率的几场。"""
    out = {"取数层认的比赛": int(len(matches))}
    for name, t in odds.items():
        has = t[["home", "draw", "away"]].notna().all(axis=1)
        out[f"{name}：有赛前胜平负的比赛"] = int(has.sum())
        out[f"{name}：价格不像样、不下的比赛"] = int((has & ~t["ok"]).sum())
    out["有融合概率的比赛"] = int(fusion["p_home"].notna().sum())
    return out


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="虚拟投注机器人的历史回测（开发模式，只用开发赛季；全是虚拟投注）")
    ap.add_argument("--db", default=None, help="数据库文件，默认 settings 里的")
    ap.add_argument("--run-dir", default=None, help="回测结果文件夹，默认 data/backtest/<bots.run>")
    ap.add_argument("--out", default=None, help="结果放哪，默认 data/backtest/bots/<bots.version>")
    ap.add_argument("--first", type=int, default=None, help="只跑从这个赛季起（试跑用）")
    ap.add_argument("--last", type=int, default=None, help="只跑到这个赛季（试跑用）")
    a = ap.parse_args()
    fs = FeatureStore(open_readonly(a.db), "开发", settings)
    res = run_all(settings, fs, Path(a.run_dir) if a.run_dir else None, Path(a.out) if a.out else None,
                  a.first, a.last)
    pd.set_option("display.width", 220)
    cols = ["名次", "机器人", "投注场数", "ROI", "ROI下限", "ROI上限", "有CLV的注数", "CLV平均", "CLV下限", "CLV上限",
            "ROI标记", "CLV标记"]
    print(res["summary"][cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps({k: res["check"][k] for k in ("场数", "随机机器人的理论值")}, ensure_ascii=False, indent=1,
                     default=str))
    print(f"结果存在 {res['out']}")


if __name__ == "__main__":
    main()
