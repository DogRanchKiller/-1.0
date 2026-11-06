"""开发顺序第 1 步：时间核对（需求文档第 12 节“2026-11-06 起的开发顺序”，方法和结果见第 16 节“时间核对（第 1 步）”）。

要回答的问题（写程序之前定下，只查不改）：
T-24 用到的每一项信息（比分和射门、赔率、伤停、首发），各自什么时候公开、系统什么时候拿到、当时全不全；
训练用的历史比赛和出预测两边都查。查出问题另写怎么修，这里一行代码都不修。

查的四件事：
一、football-data 赛前赔率（回测里的“T-24 赔率”）实际什么时候采的，和回测的截止时间（最早开球 − 24 小时）比。
    说明文件写“周末比赛周五下午采、周中比赛周二下午采”，没写几点 → 12:00、18:00（英国时间）两头都算。
    采集晚于截止 = 回测里的市场比模型多看了这段时间的消息（伤停、首发传闻），不是偷看比分。
    能知道的时间用和回测同一个函数（src.features.known_time.compute_known_times），保证截止时间算法一样。
二、真实预测：当时训练表里的比赛场数，和“现在的数据库 + 回测的能知道时间规则”重算的比。
    少了 = 比分当时还没进库（数据晚到）；回测假设开球 3 小时后就有比分，真实情况比它慢。
三、真实预测用的懂球帝赔率：取数层按网站显示的变盘时间（changed_at_utc）判断能不能用，
    可我们每小时才存一次快照，真正拿到是快照时间（fetched_at_utc）。查“变盘时间在截止前、拿到在截止后”的有多少。
四、伤停、首发：gb_l7 不用（预测页只给 A 版模型取数，要伤停的模型会直接报错）；这里只写下事实。

只读打开数据库（mode=ro），不写任何表。只读赛程、“有没有赛前赔率”和赔率的时间，不读开发赛季以外的比分；
第二件事要重算训练表，通过取数层 src/features（实时模式、A版输入）拿，和真实预测一样。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.features import FeatureStore, open_readonly
from src.features.known_time import compute_known_times

SECTION = "timing_check"
WEEKDAY_NAMES = "一二三四五六日"   # 周一 0 … 周日 6


def collection_back_days(weekdays: pd.Series, collection_weekdays: list[int]) -> pd.Series:
    """比赛日是周几 → 往前推几天是采集日（当天或之前最近的一个采集日）。

    例如采集日是周二（1）、周五（4）：周六（5）往前 1 天是周五；周一（0）往前 3 天是周五；周二当天采（0 天）。
    """
    def back(wd: int) -> int:
        return min((wd - c) % 7 for c in collection_weekdays)
    return weekdays.map(back)


def pre_odds_peek(matches: pd.DataFrame, feature_settings: dict, collection_weekdays: list[int],
                  hours: list[int], horizon_hours: float) -> pd.DataFrame:
    """每场比赛：赔率采集时间比回测截止时间晚多少小时（正数 = 晚了，市场多看了消息；负数 = 早了）。

    matches 要有 match_id、match_date（英国日期）、kickoff_utc（可以空）。
    截止时间 = 最早开球 − horizon_hours（只有日期的比赛，最早开球 = 当天 00:00 英国时间，和回测一样）。
    采集时间 = 采集日 hours 点（英国时间），每个钟点一列 peek_<钟点>。
    """
    tz = feature_settings["date_only_timezone"]
    matches = matches.reset_index(drop=True)
    kt = compute_known_times(matches, feature_settings)
    earliest = pd.to_datetime(kt["kickoff_earliest_utc"]).dt.tz_localize("UTC")
    cutoff = earliest - pd.Timedelta(hours=float(horizon_hours))
    date = pd.to_datetime(matches["match_date"], format="%Y-%m-%d").reset_index(drop=True)
    wd = date.dt.day_of_week
    coll_day = date - pd.to_timedelta(collection_back_days(wd, collection_weekdays), unit="D")
    out = pd.DataFrame({
        "match_id": matches["match_id"].to_numpy(),
        "weekday": wd.map(dict(enumerate(WEEKDAY_NAMES))),
        "era": matches["kickoff_utc"].notna().map({True: "有开球时间", False: "只有日期"}).to_numpy(),
        "cutoff_utc": cutoff.dt.strftime("%Y-%m-%d %H:%M:%S"),
    })
    for h in hours:
        coll = (coll_day + pd.Timedelta(hours=int(h))).dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        out[f"peek_{h}"] = ((coll.dt.tz_convert("UTC") - cutoff).dt.total_seconds() / 3600).to_numpy()
    return out


def summarize_peek(peek: pd.DataFrame, hours: list[int], by: str | None = None) -> pd.DataFrame:
    """按时期（和周几）汇总：场数；每个钟点假设下，采集晚于截止的比例、这些场平均晚几小时、早于截止的平均早几小时。"""
    keys = ["era"] + ([by] if by else [])
    rows = []
    for k, g in peek.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(keys, k), 场数=len(g))
        for h in hours:
            p = g[f"peek_{h}"]
            late, early = p[p > 0], p[p <= 0]
            r[f"{h}点 晚于截止"] = f"{len(late) / len(g):.1%}"
            r[f"{h}点 晚的平均小时"] = f"{late.mean():.1f}" if len(late) else "-"
            r[f"{h}点 早的平均小时"] = f"{-early.mean():.1f}" if len(early) else "-"
        rows.append(r)
    out = pd.DataFrame(rows)
    if by == "weekday":
        out["_o"] = out["weekday"].map({n: i for i, n in enumerate(WEEKDAY_NAMES)})
        out = out.sort_values(["era", "_o"]).drop(columns="_o")
    return out.rename(columns={"era": "时期", "weekday": "周几"})


def load_fd_matches(con, cfg: dict, leagues: list[str]) -> pd.DataFrame:
    """开发赛季里、已踢完、有 football-data 赛前胜平负赔率的比赛（只要赛程，不要比分）。"""
    q = f"""select m.match_id, m.competition, m.season, m.match_date, m.kickoff_utc from matches m
            where m.competition in ({','.join('?' * len(leagues))}) and m.status = 'finished'
              and m.season between ? and ?
              and exists (select 1 from odds o where o.match_id = m.match_id and o.source = ?
                          and o.timing = 'pre' and o.market = '1x2')
            order by m.match_date, m.match_id"""
    s0, s1 = (int(x) for x in cfg["seasons"])
    return pd.read_sql(q, con, params=[*leagues, s0, s1, cfg["odds_source"]])


def live_training_check(con, settings: dict, leagues: list[str]) -> pd.DataFrame:
    """真实预测每个截止时间：当时训练表的场数 vs 现在按回测规则重算的场数；顺便看赔率最后变动是不是都在截止之前。"""
    pr = pd.read_sql("select prediction_id, kind, cutoff_utc, detail from predictions", con)
    if pr.empty:
        return pd.DataFrame()
    d = pr["detail"].map(json.loads)
    pr["当时场数"] = d.map(lambda x: x.get("拟合用的比赛场数"))
    pr["odds_last"] = d.map(lambda x: x.get("赔率最后变动"))
    fs = FeatureStore(con, "实时", settings)
    rows = []
    for cutoff, g in pr.groupby("cutoff_utc"):
        hist = fs.historical_matches(cutoff, "A版输入", competition=leagues, data=("比分", "射门、xG"))
        recent = hist[pd.to_datetime(hist["match_date"]) >= pd.Timestamp(cutoff) - pd.Timedelta(days=10)]
        then = g["当时场数"].dropna()
        ol = g["odds_last"].dropna()
        rows.append({"截止时间（UTC）": cutoff, "种类": "、".join(sorted(g["kind"].unique())), "预测条数": len(g),
                     "当时场数": int(then.iloc[0]) if len(then) else None, "按规则重算": len(hist),
                     "近 10 天比赛": len(recent), "近 10 天缺射正": int(recent["home_shots_on_target"].isna().sum()),
                     "赔率变动晚于截止": int((ol > cutoff).sum())})
    out = pd.DataFrame(rows)
    out["少了几场"] = out["按规则重算"] - out["当时场数"]
    return out


def live_odds_fetch_check(con) -> pd.DataFrame:
    """真实预测每条：变盘时间在截止前的胜平负价格，每家公司最新那一行，是不是在出预测（created_at_utc）之后才存到的。

    晚存到 = 当时用的是这家更早的价（或者这家当时还没有价），现在重算会拿到更新的价。
    价格在截止前网站上已经公开（按网站显示的变盘时间），所以不是偷看；只是重算和当时不一样。
    """
    q = """select p.prediction_id, p.kind, p.cutoff_utc, p.created_at_utc, o.bookmaker, o.changed_at_utc, o.fetched_at_utc
           from predictions p join odds_changes o on o.match_id = p.match_id
           where o.market = '1x2' and o.changed_at_utc <= p.cutoff_utc"""
    df = pd.read_sql(q, con)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["prediction_id", "bookmaker", "changed_at_utc", "fetched_at_utc"])
    last = df.groupby(["prediction_id", "bookmaker"]).tail(1)
    seen = df[df["fetched_at_utc"] <= df["created_at_utc"]].groupby(["prediction_id", "bookmaker"]).size()
    last = last.assign(late=last["fetched_at_utc"] > last["created_at_utc"],
                       none_then=~last.set_index(["prediction_id", "bookmaker"]).index.isin(seen.index))
    return last.groupby(["prediction_id", "kind"], as_index=False).agg(
        公司数=("bookmaker", "size"), 最新价晚存到=("late", "sum"), 当时没有这家=("none_then", "sum"))


def collection_lag(con) -> pd.Series:
    """懂球帝变盘到我们存到，隔了几小时：只算这场开始存快照以后才变的价（更早的初盘不算）。"""
    df = pd.read_sql("select match_id, changed_at_utc, fetched_at_utc from odds_changes where source = 'dongqiudi'", con)
    if df.empty:
        return pd.Series(dtype=float)
    first = df.groupby("match_id")["fetched_at_utc"].transform("min")
    w = df[df["changed_at_utc"] >= first]
    return (pd.to_datetime(w["fetched_at_utc"]) - pd.to_datetime(w["changed_at_utc"])).dt.total_seconds() / 3600


def run(db_path=None, settings: dict | None = None) -> dict:
    s = settings or load_settings()
    cfg = s[SECTION]
    for y in range(int(cfg["seasons"][0]), int(cfg["seasons"][1]) + 1):
        if season_role(y, s) != "dev":   # 只查开发赛季
            raise ValueError(f"timing_check.seasons 里的 {y} 不是开发赛季")
    leagues = list(s["leagues"][cfg["leagues"]])
    hours = [int(h) for h in cfg["collection_hours_london"]]
    con = open_readonly(db_path)
    try:
        m = load_fd_matches(con, cfg, leagues)
        peek = pre_odds_peek(m, s["features"], [int(x) for x in cfg["collection_weekdays"]], hours,
                             float(cfg["horizon_hours"]))
        return {"settings": s, "hours": hours, "peek": peek, "seasons": cfg["seasons"],
                "training": live_training_check(con, s, leagues), "fetch": live_odds_fetch_check(con),
                "lag": collection_lag(con)}
    finally:
        con.close()


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "（没有）"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def build_report(out: dict) -> str:
    cfg, hours, peek = out["settings"][SECTION], out["hours"], out["peek"]
    h0, h1 = hours[0], hours[-1]
    L = [f"# {cfg['title']}", "",
         "开发顺序第 1 步（需求文档第 12 节）。只查不改：不改模型、设置、数据库、网站、定时任务。", ""]
    # 一、football-data 赛前赔率
    L += ["## 一、football-data 赛前赔率的采集时间 vs 回测的 T-24 截止时间", "",
          f"- 范围：{cfg['leagues']} 联赛，{out['seasons'][0]}–{out['seasons'][1]} 赛季，已踢完、有 {cfg['odds_source']} 赛前胜平负赔率的 {len(peek)} 场。",
          "- 截止时间 = 最早开球 − 24 小时（只有日期的比赛按当天 00:00 英国时间，和回测同一个函数）。",
          f"- 采集时间 = 比赛日当天或之前最近的周二 / 周五，{h0}:00 或 {h1}:00 英国时间（说明文件只写“下午”）。",
          "- “晚于截止”= 回测里的市场比模型多看了这段时间的消息；“早”= 市场看得比 T-24 少，对模型有利。", ""]
    for h in hours:
        p = peek[f"peek_{h}"]
        L.append(f"- 按 {h}:00：晚于截止 {int((p > 0).sum())} 场（{(p > 0).mean():.1%}），早于或等于 {int((p <= 0).sum())} 场。")
    L += ["", md_table(summarize_peek(peek, hours)), "", "按周几：", "", md_table(summarize_peek(peek, hours, "weekday")), ""]
    # 二、真实预测的训练表
    t = out["training"]
    L += ["## 二、真实预测：当时训练表的场数 vs 现在按回测规则重算", ""]
    if t.empty:
        L.append("（还没有真实预测）")
    else:
        L += [f"- {len(t)} 个截止时间；场数一样的 {int((t['少了几场'] == 0).sum())} 个，当时少了比赛的 {int((t['少了几场'] > 0).sum())} 个"
              f"（最多少 {int(t['少了几场'].max())} 场）；当时多了的 {int((t['少了几场'] < 0).sum())} 个。",
              f"- 近 10 天比赛缺射正的合计 {int(t['近 10 天缺射正'].sum())} 场；赔率最后变动晚于截止的预测 {int(t['赔率变动晚于截止'].sum())} 条（应该是 0）。",
              "- 只能比场数：match_stats 没有“哪天进库”的时间，历史比赛的比分什么时候到的查不出来。", "",
              md_table(t), ""]
    # 三、懂球帝赔率：变盘时间 vs 存到的时间
    f, lag = out["fetch"], out["lag"]
    L += ["## 三、真实预测用的懂球帝赔率：网站变盘时间 vs 我们存到的时间", ""]
    if len(lag):
        q = lag.quantile([.5, .9, .99])
        L.append(f"- 开始存快照以后才变的价 {len(lag)} 行：变盘到存到，中位 {q.iloc[0]:.1f} 小时、90% 在 {q.iloc[1]:.1f} 小时内、"
                 f"99% 在 {q.iloc[2]:.1f} 小时内、最长 {lag.max():.1f} 小时。")
    if not f.empty:
        L += [f"- 真实预测 {len(f)} 条：至少一家公司的最新胜平负价（变盘在截止前）是出预测之后才存到的 {int((f['最新价晚存到'] > 0).sum())} 条，"
              f"合计 {int(f['最新价晚存到'].sum())} 家次（全部 {int(f['公司数'].sum())} 家次）；其中出预测时这家还一行都没有的 {int(f['当时没有这家'].sum())} 家次。",
              "- 这些价在截止前网站上已经公开，不是偷看；问题是取数层按变盘时间判断，现在重算同一条预测，拿到的价和当时不一样。", "",
              md_table(f.groupby("kind", as_index=False)[["公司数", "最新价晚存到", "当时没有这家"]].sum()
                       .assign(条数=f.groupby("kind").size().to_numpy()).rename(columns={"kind": "种类"})), ""]
    # 四、伤停、首发
    L += ["## 四、伤停、首发", "",
          "- 上线模型 gb_l7 是 A 版输入：不用赔率、伤停、首发。预测页只给 A 版模型取数，要伤停或赛前赔率的模型会直接报错。",
          "- 首发（懂球帝首发页）只显示、复盘用；伤停研究表（salimt）是事后整理的，只能研究实验、只在开发模式。",
          "- 所以这两项对现在的预测成绩没有时间问题；以后要进模型，得先找有“什么时候公开”时间的来源。", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="开发顺序第 1 步：时间核对（只读）")
    ap.add_argument("--db", default=None, help="数据库文件（不写用 settings 的 paths.database）")
    ap.add_argument("--report", default=None, help="报告放哪（不写用 timing_check.output）")
    a = ap.parse_args()
    out = run(db_path=a.db)
    rep = Path(resolve_path(a.report or out["settings"][SECTION]["output"]))
    rep.parent.mkdir(parents=True, exist_ok=True)
    text = build_report(out)
    rep.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n报告：{rep}")


if __name__ == "__main__":
    main()
