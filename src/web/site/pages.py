"""首页 index.html、球队页 team/<球队>.html、战绩页 record.html（需求文档第 8 节）。单场页在 match.py。"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.features.known_time import to_utc_text
from src.web.site.common import (LEVEL_NAMES, NOT_ADVICE, OUTCOMES, STATUS_TAGS, TeamFiles, bj_text, day_title, details,
                                 e, kickoff_text, level_name, match_href, page, pct0, placeholder, table)
from src.web.site.data import SiteData
from src.web.site.match import conclusion_html, final_probs, hit_html, score_text, side_of, version_name

SOURCES = ("融合", "模型", "收盘赔率")   # 战绩页里几种概率的先后


def match_link(data: SiteData, mid: int, prefix: str) -> str:
    """“主队 对 客队”，点了去单场页；没有单场页的只写名字。"""
    m = data.matches[mid]
    text = f"{data.zh(m['home_team'])} 对 {data.zh(m['away_team'])}"
    if mid not in data.page_ids:
        return e(text)
    return f"<a class='match' href='{match_href(mid, prefix)}'>{e(text)}</a>"


def state_html(data: SiteData, mid: int) -> str:
    s = score_text(data, mid)
    if s:
        return f"<span class='score'>{e(s)}</span>"
    tag = STATUS_TAGS.get(data.matches[mid].get("status"))
    return f"<span class='tag warn'>{e(tag)}</span>" if tag else ""


def latest_cell(data: SiteData, mid: int) -> str:
    lt = data.latest(mid)
    if lt is None:
        return "<span class='muted'>还没有预测</span>"
    kind, row = lt
    return f"{conclusion_html(row)}<br><span class='small muted'>{e(version_name(kind))}，{e(bj_text(row['cutoff_utc']))} 出的</span>"


# ---------- 首页 ----------
def render_index(data: SiteData, teams: TeamFiles) -> str:
    cfg = data.settings["site"]
    now = to_utc_text(data.now)
    hi = to_utc_text(data.now + timedelta(days=cfg["days_ahead"]))
    lo = to_utc_text(data.now - timedelta(days=cfg["days_back"]))
    site = [m for m in data.matches.values() if m["site_league"]]
    # 还没踢的：开球时间在“现在”到未来几天之间，或者已经开球但还没有比分（正在踢）
    playing = to_utc_text(data.now - timedelta(hours=3))   # 开球 3 小时内、还没有比分的，算“正在踢”，还放在上面

    def is_upcoming(m) -> bool:
        if m["match_id"] in data.results:
            return False
        k = m["kickoff_earliest_utc"]
        return now <= k < hi or (playing <= k < now and m.get("status") not in STATUS_TAGS)

    upcoming = sorted([m for m in site if is_upcoming(m)], key=lambda m: (m["kickoff_earliest_utc"], m["match_id"]))
    up_ids = {m["match_id"] for m in upcoming}
    recent = sorted([m for m in site if lo <= m["kickoff_earliest_utc"] < now and m["match_id"] not in up_ids],
                    key=lambda m: (m["kickoff_earliest_utc"], m["match_id"]), reverse=True)
    n_pred = sum(1 for m in upcoming if m["match_id"] in data.preds)
    stats = (f"<div class='stats'><div class='stat hi'><b>{len(upcoming)}</b><span>未来 {cfg['days_ahead']} 天的比赛</span></div>"
             f"<div class='stat'><b>{n_pred}</b><span>已经有预测</span></div>"
             f"<div class='stat'><b>{len(recent)}</b><span>最近 {cfg['days_back']} 天踢完的</span></div></div>")
    groups: dict[str, tuple[str, list[str]]] = {}
    for m in upcoming:
        key, title = day_title(m)
        mid = m["match_id"]
        row = [e(kickoff_text(m, with_date=False) if key != "9999" else kickoff_text(m)), e(data.comp(m["competition"])),
               f"<td class='match'>{match_link(data, mid, '')}</td>", latest_cell(data, mid), state_html(data, mid)]
        groups.setdefault(key, (title, []))[1].append(row)
    if groups:
        blocks = "".join(f"<h2>{e(title)}</h2>" + table(["时间", "联赛", "比赛", "最新一版的结论", ""], rows)
                         for _, (title, rows) in sorted(groups.items()))
    else:
        blocks = "<h2>即将开赛</h2>" + placeholder(f"未来 {cfg['days_ahead']} 天没有{'、'.join(data.settings['leagues'][cfg['leagues']].values())}的比赛")
    rec_rows = []
    for m in recent:
        mid = m["match_id"]
        lt = data.latest(mid)
        rec_rows.append([e(kickoff_text(m)), e(data.comp(m["competition"])), f"<td class='match'>{match_link(data, mid, '')}</td>",
                         state_html(data, mid) or "<span class='muted'>比分还没进来</span>", latest_cell(data, mid) if lt else "—",
                         hit_html(lt[1], data.results.get(mid)) if lt else "—"])
    rec = (table(["开球", "联赛", "比赛", "赛果", "赛前最后一版", "胜平负猜中"], rec_rows) if rec_rows
           else placeholder(f"最近 {cfg['days_back']} 天没有踢完的比赛"))
    notes = "".join(f"<li>{e(n)}</li>" for n in data.notes[:10])
    body = (f"<h1>足球预测 <span class='accent'>即将开赛</span></h1><p class='muted'>"
            f"{e('、'.join(data.settings['leagues'][cfg['leagues']].values()))}，时间都是北京时间。点比赛看各版本对比、比分概率和赔率变动。</p>"
            f"{stats}<div class='note'>{NOT_ADVICE}</div>{blocks}"
            f"<h2>最近踢完的</h2>{rec}"
            + (f"<h2>读数据时的提示</h2><ul class='points small muted'>{notes}</ul>" if notes else ""))
    return page("足球预测 - 即将开赛", body, "", data.now, cfg["refresh_minutes"])


# ---------- 球队页 ----------
def team_record(data: SiteData, team: str) -> list[list[str]]:
    """这队打过的、有预测、有比分的比赛：最后一版的结论和猜没猜中。"""
    rows = []
    mids = [mid for mid, m in data.matches.items() if team in (m["home_team"], m["away_team"])
            and mid in data.preds and mid in data.results]
    for mid in sorted(mids, key=lambda i: data.matches[i]["kickoff_earliest_utc"], reverse=True):
        kind, row = data.latest(mid)
        rows.append([e(kickoff_text(data.matches[mid])), f"<td class='match'>{match_link(data, mid, '../')}</td>",
                     state_html(data, mid), conclusion_html(row), hit_html(row, data.results[mid])])
    return rows


def render_team(data: SiteData, team: str, teams: TeamFiles) -> str:
    cfg = data.settings["site"]
    now = to_utc_text(data.now)
    pre = "../"
    # 已经开球的：赛程（联赛 + 杯赛、欧战，用途「显示」）；比分只有联赛有（见 data.py 第 5 步），杯赛写“—”
    past_rows = []
    sch = data.schedule
    if sch is not None and not sch.empty:
        mine = sch[(sch["home_team"] == team) | (sch["away_team"] == team)].sort_values("kickoff_earliest_utc", ascending=False)
        for r in mine.to_dict("records"):
            mid = int(r["match_id"])
            home = r["home_team"] == team
            opp = r["away_team"] if home else r["home_team"]
            s = data.results.get(mid)
            res = "—"
            if s is not None:
                me, op = (s[0], s[1]) if home else (s[1], s[0])
                res = f"<span class='score'>{s[0]}–{s[1]}</span> " + ("胜" if me > op else "平" if me == op else "负")
            name = f"{'主' if home else '客'} 对 {data.zh(opp)}"
            link = f"<a class='match' href='{match_href(mid, pre)}'>{e(name)}</a>" if mid in data.page_ids else e(name)
            info = {"kickoff_utc": r["kickoff_utc"] if isinstance(r["kickoff_utc"], str) else None,
                    "match_date": r["match_date"], "date_precision": r.get("date_precision") if isinstance(r.get("date_precision"), str) else "day"}
            past_rows.append([e(kickoff_text(info)), e(data.comp(r["competition"])), f"<td class='match'>{link}</td>", res])
    hi = to_utc_text(data.now + timedelta(days=cfg["team_days_ahead"]))
    up = sorted([m for m in data.matches.values() if team in (m["home_team"], m["away_team"])
                 and now <= m["kickoff_earliest_utc"] < hi], key=lambda m: m["kickoff_earliest_utc"])
    up_rows = [[e(kickoff_text(m)), e(data.comp(m["competition"])), f"<td class='match'>{match_link(data, m['match_id'], pre)}</td>",
                latest_cell(data, m["match_id"])] for m in up]
    rec_rows = team_record(data, team)
    hits = sum("class='hit'" in r[-1] for r in rec_rows)
    body = (f"<h1>{e(data.zh(team))} <span class='accent'>{e(team if data.zh(team) != team else '')}</span></h1>"
            f"<h2>接下来 {cfg['team_days_ahead']} 天</h2>"
            + (table(["开球", "赛事", "比赛", "最新一版的结论"], up_rows) if up_rows else placeholder("这段时间没有联赛"))
            + f"<h2>最近 {cfg['team_days_back']} 天踢过的</h2>"
            + (table(["开球", "赛事", "比赛", "比分（主–客）和这队的胜负"], past_rows) if past_rows else placeholder("这段时间没有踢过的比赛"))
            + "<p class='small muted'>杯赛、欧战现在只有赛程，没有比分，写“—”。</p>"
            + f"<h2>预测战绩</h2>"
            + (f"<p>有预测、已踢完的 {len(rec_rows)} 场，胜平负猜中 {hits} 场。</p>"
               + table(["开球", "比赛", "比分", "赛前最后一版", "胜平负猜中"], rec_rows)
               if rec_rows else placeholder("还没有这队的预测战绩（预测过、而且踢完的比赛才算）")))
    return page(f"{data.zh(team)} - 足球预测", body, pre, data.now, cfg["refresh_minutes"])


# ---------- 战绩页 ----------
def _hit_rate(df: pd.DataFrame) -> float:
    """胜平负概率最大的那项和赛果一样的比例。"""
    p = df[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    actual = np.where(df["home_goals"] > df["away_goals"], 0, np.where(df["home_goals"] == df["away_goals"], 1, 2))
    return float((p.argmax(axis=1) == actual).mean())


def _mean(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return f"{x.mean():.3f}" if len(x) else "—"


def render_record(data: SiteData, out_dir: Path) -> str:
    """总战绩，从 data/review/scores.csv 汇总（K2 每次全部重算）。"""
    cfg = data.settings["site"]
    if data.scores is None:
        why = data.scores_note or "还没有复盘数据"
        body = f"<h1>战绩</h1>{placeholder(why)}" + (placeholder("还没有复盘数据") if data.scores_note else "")
        return page("战绩 - 足球预测", body, "", data.now, cfg["refresh_minutes"])
    s = data.scores.dropna(subset=["home_goals", "away_goals", "p_home", "p_draw", "p_away"]).copy()
    s["kind"] = s["kind"].fillna("")          # 收盘赔率那几行不分版本（kind 是空的），分组时不能丢
    order = {k: i for i, k in enumerate(cfg["versions"])}
    srank = {k: i for i, k in enumerate(SOURCES)}
    rows = []
    for (kind, src), g in sorted(s.groupby(["kind", "source"]), key=lambda x: (order.get(x[0][0], 99), srank.get(x[0][1], 99))):
        rows.append([e(version_name(kind) if kind else "不分版本"), e(src), str(g["match_id"].nunique()), e(pct0(_hit_rate(g))),
                     *(e(_mean(g[c])) for c in ("logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts"))])
    summary = (table(["版本", "概率来自", "场数", "胜平负猜中", "胜平负 log loss", "Brier", "RPS", "比分 log loss",
                      "大小 2.5 log loss", "双方进球 log loss"], rows) if rows else placeholder("还没有踢完、对过账的比赛"))
    # 按信心等级：融合后的概率，按出那一版时的信心等级分组
    f = s[s["source"] == "融合"].copy()
    f["level"] = [data.levels.get(int(p)) if pd.notna(p) else None for p in f["prediction_id"]]
    lv_rows = []
    for lv in sorted(LEVEL_NAMES, reverse=True):
        g = f[f["level"] == lv]
        if len(g):
            lv_rows.append([e(level_name(lv)), str(len(g)), e(pct0(_hit_rate(g))), e(_mean(g["logloss_1x2"]))])
    by_level = (table(["信心等级", "几行（场 × 版本）", "胜平负猜中", "胜平负 log loss"], lv_rows) if lv_rows
                else placeholder("还没有带信心等级的融合预测"))
    # 最近踢完的几场
    recent_rows = []
    ids = (s.sort_values("kickoff_utc", ascending=False)["match_id"].drop_duplicates().head(30))
    for mid in ids:
        g = s[s["match_id"] == mid]
        r0 = g.iloc[0]
        name = (match_link(data, int(mid), "") if int(mid) in data.matches else e(str(mid)))
        hits = []
        for kind in cfg["versions"]:
            k = g[(g["kind"] == kind) & (g["source"] == "融合")]
            k = k if len(k) else g[(g["kind"] == kind) & (g["source"] == "模型")]
            if len(k):
                ok = _hit_rate(k.iloc[:1]) == 1.0
                hits.append(f"{e(kind)} <span class='{'hit' if ok else 'miss'}'>{'中' if ok else '没中'}</span>")
        recent_rows.append([e(bj_text(r0["kickoff_utc"])), e(data.comp(str(r0["competition"]))), f"<td class='match'>{name}</td>",
                            f"<span class='score'>{int(r0['home_goals'])}–{int(r0['away_goals'])}</span>", "　".join(hits) or "—"])
    recent = table(["开球", "联赛", "比赛", "比分", "各版本胜平负猜中"], recent_rows) if recent_rows else ""
    # 周报、月报：链接到 data/review/reports/ 里的文件（相对路径，网站文件夹和复盘文件夹的相对位置不变就能点）
    rep = ""
    if data.reports:
        links = "".join(f"<li><a href='{e(quote(Path(os.path.relpath(p, out_dir)).as_posix()))}'>{e(p.name)}</a></li>"
                        for p in data.reports[:20])
        latest = data.reports[0]
        try:
            text = latest.read_text(encoding="utf-8")
        except OSError:
            text = "读不了"
        rep = (f"<h2>周报、月报</h2><ul class='points'>{links}</ul>"
               + details(f"最新一份：{latest.name}", f"<pre class='small' style='white-space:pre-wrap'>{e(text)}</pre>"))
    body = (f"<h1>战绩 <span class='accent'>赛后对账</span></h1>"
            f"<p class='muted'>从 data/review/scores.csv 汇总：已经踢完、有预测的比赛，每一版、每种概率一行。"
            f"log loss、Brier、RPS 都是越小越准；“收盘赔率”是拿开球那一刻的赔率当对照，模型和融合能接近它就不错了。</p>"
            f"<h2>总的</h2>{summary}<h2>按信心等级</h2>{by_level}"
            + (f"<h2>最近踢完的</h2>{recent}" if recent else "") + rep)
    return page("战绩 - 足球预测", body, "", data.now, cfg["refresh_minutes"])
