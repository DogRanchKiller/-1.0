"""四本账：按四个通道给真实预测对账（需求文档第 16 节“四本账怎么算（第 2 步）”，跑之前写下的规则）。

一、算哪些
  预测存档（predictions 表）里五大联赛、模型 gb_l7 的 T-24 / T-6 / T-1，每场每个时点取最早的那条；
  比赛已经踢完、取数层用途「评估」拿得到比分的才算。下结论只看 T-24，另外两个只报告。

二、四个通道（差 = 模型 − 市场，小于 0 = 模型更准）
  1. 胜平负：模型 = 存档的 p_home / p_draw / p_away；市场 = 存档的“赔率概率”（多家平均去水，当时存的，不重算）。log loss。
  2. 亚盘让球：市场 = 平博亚盘，取这条预测出之前就有的最后一组（变盘时间、我们存到的时间都不晚于 created_at_utc），
     线就是这一组的线，两边赔率按比例去水 → 主队那边的胜算；模型 = 存档的模型比分矩阵在这条线上主队的胜算。
  3. 亚盘大小球：同上，平博大小球，“大”那边的胜算。
  4. 比分：模型 = 存档的模型比分矩阵；市场 = 存档的融合比分矩阵（w = 1 时就是市场矩阵）。−ln(实际比分那一格)。
  亚盘、大小球每场的分数：按实际比分结算这条线，赢 −ln(胜算)，输 −ln(1 − 胜算)，赢半、输半同样算再乘 0.5，走盘 0。

三、区间和结论
  差的平均按周成块 bootstrap（src/evaluate/compare.py 同一个函数、settings.yaml compare 段同样的设置）。
  一个通道少于 ledger.min_matches 场只列数、不算区间、不下结论。

四、结果（data/backtest/ledger/，不进 Git）
  ledger_matches.csv   每场每个时点每个通道一行。“模型概率”“市场概率”：胜平负、比分是给实际结果的概率；
                       亚盘是主队那边的胜算，大小球是“大”那边的胜算
  ledger_summary.csv   每个时点每个通道：场数、两边平均分、差、区间、结论、两边命中率（只显示）、缺市场价的场数、实际开球前几小时

只读：不改存档、模型、冻结的设置、网站、定时任务。
用法：
  .venv\\Scripts\\python -m src.evaluate.ledger
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.markets import asian_handicap, over_under
from src.common.odds import devig
from src.common.settlement import asian_handicap_margin, over_under_margin, settle_margin
from src.evaluate.compare import block_ids, bootstrap_mean_diff, verdict
from src.features import FeatureStore, open_readonly
from src.features.known_time import to_utc_text

USAGE = "评估"
CHANNELS = ("胜平负", "亚盘让球", "亚盘大小球", "比分")
# 亚盘、大小球结算 → (主队 / 大那边算不算赢, 权重)。走盘不算分（权重 0）
RESULT_NAMES = ("主胜", "平", "客胜")
SETTLE = {"win": (1, 1.0), "half_win": (1, 0.5), "push": (None, 0.0), "half_loss": (0, 0.5), "loss": (0, 1.0)}


def line_score(share: float, outcome: str, eps: float) -> float:
    """亚盘、大小球一场的分数（越小越好）：赢 −ln(胜算)，输 −ln(1 − 胜算)；赢半、输半同样算再乘 0.5；走盘 0。

    胜算是“主队那边”（亚盘）或“大那边”（大小球）的胜算，outcome 是这一边按实际比分结算的结果。
    胜算截到 [eps, 1 − eps]，和 log loss 一样，免得 ln 0。
    """
    won, weight = SETTLE[outcome]
    if weight == 0:
        return 0.0
    q = min(max(float(share), eps), 1 - eps)
    return weight * -np.log(q if won else 1 - q)


def line_hit(share: float, outcome: str) -> tuple[float, float]:
    """命中率（只显示）：胜算 > 0.5 算看好这一边，< 0.5 算看好另一边，正好 0.5 不算。

    返回 (命中的份数, 算进去的份数)：赢、输算 1 份，赢半、输半算 0.5 份，走盘和没看好哪边都是 0 份。
    """
    won, weight = SETTLE[outcome]
    if weight == 0 or share == 0.5 or not np.isfinite(share):
        return 0.0, 0.0
    picked_this_side = share > 0.5
    return (weight if picked_this_side == bool(won) else 0.0), weight


def load_archive(con, leagues: list[str], model: str, kinds: list[str]) -> pd.DataFrame:
    """预测存档里要对账的那些：五大联赛、这个模型、这几个时点；同一场同一个时点有几条时取最早出的那条。"""
    q = f"""select prediction_id, match_id, competition, season, home_team, away_team, kickoff_utc, match_date,
                   kind, model, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail
            from predictions
            where model = ? and competition in ({','.join('?' * len(leagues))}) and kind in ({','.join('?' * len(kinds))})
            order by match_id, kind, created_at_utc, prediction_id"""
    df = pd.read_sql(q, con, params=[model, *leagues, *kinds])
    return df.groupby(["match_id", "kind"], sort=False).head(1).reset_index(drop=True)


def pinnacle_line(fs: FeatureStore, match_id: int, created_at: str, market: str, bookmaker: str) -> dict | None:
    """这条预测出之前就有的平博最后一组价格（变盘时间、我们存到的时间都不晚于 created_at）→ 线和去水后的胜算。

    market：'ah' 亚盘（返回主队那边的胜算），'ou' 大小球（返回“大”那边的胜算）。没有、或赔率不对时返回 None。
    """
    df = fs.odds_changes(match_id, created_at, USAGE, bookmaker=bookmaker, market=market)
    if df.empty:
        return None
    df = df[pd.to_datetime(df["fetched_at_utc"]) <= pd.Timestamp(created_at)]
    if df.empty:
        return None
    r = df.iloc[-1]
    a, b = ("home", "away") if market == "ah" else ("over", "under")
    if pd.isna(r["line"]) or pd.isna(r[a]) or pd.isna(r[b]) or min(r[a], r[b]) <= 1:
        return None
    share = float(devig([r[a], r[b]], method="proportional")[0])
    return {"line": float(r["line"]), "share": share, "changed_at_utc": r["changed_at_utc"],
            "fetched_at_utc": r["fetched_at_utc"]}


def _cell(matrix: np.ndarray, hg: int, ag: int) -> float:
    """比分矩阵里实际比分那一格的概率；超过最大进球数的并进最后一行 / 列（矩阵就是这样截断的）。"""
    n_h, n_a = matrix.shape
    return float(matrix[min(hg, n_h - 1), min(ag, n_a - 1)])


def _top_hit(matrix: np.ndarray, hg: int, ag: int) -> float:
    """比分命中（只显示）：概率最大的那一格是不是实际比分（超过最大进球数的并进最后一行 / 列）。"""
    n_h, n_a = matrix.shape
    return float(tuple(np.unravel_index(matrix.argmax(), matrix.shape)) == (min(hg, n_h - 1), min(ag, n_a - 1)))


def match_rows(fs: FeatureStore, p: pd.Series, settings: dict) -> list[dict]:
    """一条预测 → 四个通道各一行（踢完、拿得到比分的才有；没踢完返回空）。"""
    c = settings["ledger"]
    eps = settings["calc"]["logloss_eps"]
    now = to_utc_text(datetime.now(timezone.utc))
    res = fs.match_result(int(p["match_id"]), now, USAGE)
    if res.empty or pd.isna(res.iloc[0]["home_goals"]) or pd.isna(res.iloc[0]["away_goals"]):
        return []
    hg, ag = int(res.iloc[0]["home_goals"]), int(res.iloc[0]["away_goals"])
    d = json.loads(p["detail"]) if p["detail"] else {}
    created = to_utc_text(p["created_at_utc"])
    kickoff = p["kickoff_utc"] if pd.notna(p["kickoff_utc"]) and p["kickoff_utc"] else f"{p['match_date']} 00:00"
    base = {"prediction_id": int(p["prediction_id"]), "match_id": int(p["match_id"]), "联赛": p["competition"],
            "赛季": int(p["season"]), "主队": p["home_team"], "客队": p["away_team"], "比赛日期": p["match_date"],
            "开球_utc": kickoff, "时点": p["kind"], "出预测_utc": created,
            "开球前几小时": round((pd.Timestamp(kickoff) - pd.Timestamp(created)).total_seconds() / 3600, 2),
            "比分": f"{hg}-{ag}"}
    rows = []

    def add(channel, **kw):
        row = {**base, "通道": channel, "线": np.nan, "结算": "", "模型概率": np.nan, "市场概率": np.nan,
               "模型分数": np.nan, "市场分数": np.nan, "差": np.nan, "模型命中": np.nan, "模型份数": np.nan,
               "市场命中": np.nan, "市场份数": np.nan, "市场价来自": "", "缺什么": ""}
        row.update(kw)
        if pd.notna(row["模型分数"]) and pd.notna(row["市场分数"]):
            row["差"] = row["模型分数"] - row["市场分数"]
        rows.append(row)

    # 1. 胜平负：结果 0 主胜、1 平、2 客胜
    y = 0 if hg > ag else (1 if hg == ag else 2)
    model = np.array([p["p_home"], p["p_draw"], p["p_away"]], dtype=float)
    market = d.get("赔率概率")
    if market is None or len(market) != 3:
        add("胜平负", 结算=RESULT_NAMES[y], 模型概率=model[y], 模型分数=-np.log(max(model[y], eps)),
            模型命中=float(model.argmax() == y), 模型份数=1.0, 缺什么="存档里没有赔率概率")
    else:
        market = np.asarray(market, dtype=float)
        add("胜平负", 结算=RESULT_NAMES[y], 模型概率=model[y], 市场概率=market[y],
            模型分数=-np.log(max(model[y], eps)), 市场分数=-np.log(max(market[y], eps)),
            模型命中=float(model.argmax() == y), 模型份数=1.0, 市场命中=float(market.argmax() == y), 市场份数=1.0,
            市场价来自=f"存档的多家平均（{d.get('赔率公司数', '?')} 家）")

    m = np.asarray(d["模型比分矩阵"], dtype=float) if d.get("模型比分矩阵") else None
    if m is not None:
        m = m / m.sum()

    # 2、3. 亚盘让球、亚盘大小球：线和市场胜算都取平博当时的最后一组
    for channel, market_code in (("亚盘让球", "ah"), ("亚盘大小球", "ou")):
        pin = pinnacle_line(fs, int(p["match_id"]), created, market_code, c["pinnacle"])
        if pin is None or m is None:
            add(channel, 缺什么="这条预测出之前没有平博的价格" if pin is None else "存档里没有模型比分矩阵")
            continue
        line = pin["line"]
        if market_code == "ah":
            share = asian_handicap(m, line, "home")["win_share"]
            outcome = settle_margin(*asian_handicap_margin(hg, ag, line, "home"))
        else:
            share = over_under(m, line, "over")["win_share"]
            outcome = settle_margin(*over_under_margin(hg + ag, line, "over"))
        mh, mw = line_hit(share, outcome)
        kh, kw = line_hit(pin["share"], outcome)
        add(channel, 线=line, 结算=outcome, 模型概率=share, 市场概率=pin["share"],
            模型分数=line_score(share, outcome, eps), 市场分数=line_score(pin["share"], outcome, eps),
            模型命中=mh, 模型份数=mw, 市场命中=kh, 市场份数=kw,
            市场价来自=f"平博 {pin['changed_at_utc']} 变的那组（{pin['fetched_at_utc']} 存到）")

    # 4. 比分：融合比分矩阵在 w = 1、已融合时就是市场矩阵；没融合时那份不是市场的，不算
    fused = d.get("融合比分矩阵")
    state = (d.get("融合") or {}).get("状态")
    if m is None:
        add("比分", 缺什么="存档里没有模型比分矩阵")
    elif not fused or state != "已融合":
        add("比分", 模型概率=_cell(m, hg, ag), 模型分数=-np.log(max(_cell(m, hg, ag), eps)),
            模型命中=_top_hit(m, hg, ag), 模型份数=1.0, 缺什么=f"存档的融合状态是「{state}」，没有市场比分矩阵")
    else:
        f = np.asarray(fused, dtype=float)
        f = f / f.sum()
        add("比分", 模型概率=_cell(m, hg, ag), 市场概率=_cell(f, hg, ag),
            模型分数=-np.log(max(_cell(m, hg, ag), eps)), 市场分数=-np.log(max(_cell(f, hg, ag), eps)),
            模型命中=_top_hit(m, hg, ag), 模型份数=1.0, 市场命中=_top_hit(f, hg, ag), 市场份数=1.0,
            市场价来自=f"存档的融合比分矩阵（w = {(d.get('融合') or {}).get('w')}）")
    return rows


def summarize(rows: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每个时点每个通道一行。两边都有分数的场才对账；少于 min_matches 场只列数、不算区间、不下结论。"""
    c, cmp = settings["ledger"], settings["compare"]
    out = []
    for kind in c["kinds"]:
        for channel in CHANNELS:
            g = rows[(rows["时点"] == kind) & (rows["通道"] == channel)] if len(rows) else rows
            ok = g[g["差"].notna()] if len(g) else g
            r = {"时点": kind, "通道": channel, "下结论的时点": kind == c["main_kind"], "踢完的场数": len(g),
                 "对账的场数": len(ok), "缺市场价的场数": len(g) - len(ok)}
            if len(ok):
                r.update({"模型平均分": ok["模型分数"].mean(), "市场平均分": ok["市场分数"].mean(),
                          "差的平均": ok["差"].mean()})
                for side in ("模型", "市场"):
                    w = ok[f"{side}份数"].sum()
                    r[f"{side}命中率"] = ok[f"{side}命中"].sum() / w if w > 0 else np.nan
                h = ok["开球前几小时"]
                r.update({"开球前几小时_中位": h.median(), "开球前几小时_最少": h.min(), "开球前几小时_最多": h.max()})
            if len(ok) >= c["min_matches"]:
                mean, lo, hi = bootstrap_mean_diff(ok["差"], block_ids(ok["比赛日期"], cmp["block"]),
                                                   cmp["n_boot"], cmp["confidence"], cmp["seed"])
                r.update({"区间下限": lo, "区间上限": hi,
                          "结论": verdict(lo, hi) if kind == c["main_kind"] else "只报告"})
            else:
                r.update({"区间下限": np.nan, "区间上限": np.nan,
                          "结论": f"场数不够（{len(ok)} 场，要 {c['min_matches']} 场），只列数" if len(ok) else "还没有能对账的场"})
            out.append(r)
    return pd.DataFrame(out)


def build(fs: FeatureStore, con, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """存档 → 每场每个通道一行 → 汇总。"""
    c = settings["ledger"]
    leagues = c["leagues"]
    if isinstance(leagues, str):                     # 写组名（例如 top）时换成这一组的联赛代码
        leagues = list(settings["leagues"][leagues])
    preds = load_archive(con, leagues, c["model"], c["kinds"])
    rows = [r for _, p in preds.iterrows() for r in match_rows(fs, p, settings)]
    rows = pd.DataFrame(rows)
    return rows, summarize(rows, settings)


def main() -> None:
    ap = argparse.ArgumentParser(description="四本账：按四个通道给真实预测对账（只读）")
    ap.add_argument("--db", help="数据库文件（不写 = settings.yaml 的 paths.database）")
    a = ap.parse_args()
    settings = load_settings()
    con = open_readonly(a.db)
    fs = FeatureStore(con, "实时", settings)          # 存档的比赛是上线后的赛季（2026/27 起），用实时模式
    rows, summary = build(fs, con, settings)
    out = resolve_path(settings["ledger"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "ledger_matches.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "ledger_summary.csv", index=False, encoding="utf-8-sig")
    print(f"踢完、对上账的预测：{rows['prediction_id'].nunique() if len(rows) else 0} 条，写到 {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["时点", "通道", "对账的场数", "缺市场价的场数", "模型平均分", "市场平均分", "差的平均",
                       "模型命中率", "市场命中率", "结论"]].to_string(index=False))


if __name__ == "__main__":
    main()
