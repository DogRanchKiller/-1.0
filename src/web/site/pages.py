"""首页 index.html、球队页 team/<球队>.html、战绩页 record.html（需求文档第 8 节）。单场页在 match.py。"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.features.known_time import to_utc_text
from src.review.reports import EXTRA_NAMES, coverage_cells, coverage_for_reviews, extra_text, group_cells, live_metrics
from src.web.site.common import (NOT_ADVICE, STATUS_TAGS, TeamFiles, bj_text, day_title, details,
                                 e, kickoff_text, level_name, match_href, page, placeholder, table)
from src.web.site.data import SiteData
from src.web.site.match import (archive_name, conclusion_html, hit_html, identity_text, known_text,
                                score_text, side_of, version_name)


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
        note = f" <span class='tag warn'>{e(data.result_notes[mid])}</span>" if mid in data.result_notes else ""
        return f"<span class='score'>{e(s)}</span>{note}"
    tag = STATUS_TAGS.get(data.matches[mid].get("status"))
    return f"<span class='tag warn'>{e(tag)}</span>" if tag else ""


def latest_cell(data: SiteData, mid: int) -> str:
    lt = data.latest(mid)
    if lt is None:
        return "<span class='muted'>还没有预测</span>"
    kind, row = lt
    return (f"{conclusion_html(row)}<br><span class='small muted'>{e(identity_text(row))}，{e(version_name(kind))}，"
            f"{e(archive_name(row.get('archive_status')))}，{e(bj_text(row['cutoff_utc']))} 截止</span>")


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
                     state_html(data, mid), latest_cell(data, mid), hit_html(row, data.results[mid])])
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
                if mid in data.result_notes:
                    res += f" <span class='tag warn'>{e(data.result_notes[mid])}</span>"
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
def render_record(data: SiteData, out_dir: Path) -> str:
    """总战绩，从 data/review/scores.csv 汇总（K2 每次全部重算）。"""
    cfg = data.settings["site"]
    if data.scores is None:
        why = data.scores_note or "还没有复盘数据"
        body = f"<h1>战绩</h1>{placeholder(why)}" + (placeholder("还没有复盘数据") if data.scores_note else "")
        return page("战绩 - 足球预测", body, "", data.now, cfg["refresh_minutes"])
    s = data.scores.copy()
    metrics = live_metrics(s, data.settings, {mid: r.get("自动标签") for mid, r in data.reviews.items()})
    columns = ("logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts")
    group_headers = ["模型", "模型版本", "预测时点", "概率来源", "归档状态"]

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    def identity(group):
        cells = group_cells(group)
        cells[-1] = archive_name(cells[-1])
        return [e(x) for x in cells]

    def summary_table(items, extra=()):
        rows = []
        for item in items:
            cells = identity(item["group"])
            for col in extra:
                value = item["group"][col]
                cells.append(e((level_name(value) if value in (1, 2, 3) else "未知") if col == "confidence"
                               else extra_text(col, value)))
            rows.append(cells + [str(item["n_matches"])]
                        + [e(number(item["metrics"][c]["mean"])) for c in columns] + [e(item["note"])])
        return table([*group_headers, *[EXTRA_NAMES[x] for x in extra], "场数", "胜平负 log loss", "Brier", "RPS", "比分 log loss",
                      "大小 2.5 log loss", "双方进球 log loss", "证据说明"], rows) if rows else placeholder("无匹配证据")

    summary = summary_table(metrics["groups"])
    by_level = summary_table(metrics["confidence"], ("confidence",))
    pair_rows = []
    for item in metrics["comparisons"]:
        benchmark = item["benchmark"]
        for name, values in item["metrics"].items():
            pair_rows.append(identity(item["group"]) + [e(benchmark[k]) for k in ("benchmark_source", "bookmaker", "market", "timing")]
                             + [e(name), str(item["n_model"]), str(item["n_benchmark"]),
                                str(item["n_model"] - values["n_matches"]), str(values["n_matches"]), e(number(values["model_mean"])),
                                e(number(values["benchmark_mean"])), e(number(values["mean_delta"])), e(item["note"])])
    pairs = table([*group_headers, "标杆来源", "公司", "玩法", "标杆时点", "指标", "模型场数", "标杆覆盖", "缺失场数", "同场数", "模型均值", "标杆均值", "差值", "证据说明"], pair_rows)
    mids = sorted(s.loc[s["source"].isin(("模型", "融合")), "match_id"].unique())
    coverage = coverage_for_reviews([{**review, "match_id": mid} for mid, review in data.reviews.items()], data.settings, mids)
    coverage_table = table(["公司", "玩法", "复盘场数", "标杆覆盖", "缺失场数", "覆盖率", "缺失原因"],
                           [[e(str(value)) for value in row] for row in coverage_cells(coverage)])
    # 独立标杆保留原样；旧 CSV 没有来源证据时明确未知，不纳入合格收盘的同场比较。
    benchmark_rows = []
    for row in s[s["source"] == "收盘赔率"].to_dict("records"):
        benchmark_rows.append([str(row["match_id"]), "收盘赔率（不分版本）"]
                              + [e(known_text(row.get(c))) for c in ("benchmark_source", "bookmaker", "market", "timing")]
                              + [e(number(row[c])) if pd.notna(row[c]) else "—" for c in columns])
    benchmarks = table(["比赛", "来源", "数据来源", "公司", "玩法", "时点", "胜平负 log loss", "Brier", "RPS",
                         "比分 log loss", "大小 2.5 log loss", "双方进球 log loss"], benchmark_rows) if benchmark_rows else placeholder("没有合格收盘标杆")
    calibration_rows, bin_rows = [], []
    for item in metrics["calibration"]:
        for outcome in item["outcomes"]:
            calibration_rows.append(identity(item["group"]) + [e(outcome["outcome"]), str(item["n_matches"]), e(number(outcome["ece"]))])
            for bin_ in outcome["bins"]:
                bin_rows.append(identity(item["group"]) + [e(outcome["outcome"]), f"{bin_['bin_low']:.0%}–{bin_['bin_high']:.0%}",
                                str(bin_["n"]), e(number(bin_["mean_pred"])), e(number(bin_["observed"]))])
    calibration = table([*group_headers, "结果", "场数", "校准误差 ECE"], calibration_rows)
    calibration += details("按预测概率分箱（空箱保留为空）", table([*group_headers, "结果", "概率区间", "场数", "平均预测概率", "实际发生比例"], bin_rows))
    confidence_rows, confidence_bins = [], []
    for item in metrics["confidence"]:
        value = item["group"]["confidence"]
        label = level_name(value) if value in (1, 2, 3) else "未知"
        for outcome in item["calibration"]["outcomes"]:
            cells = identity(item["group"]) + [e(label), e(outcome["outcome"])]
            confidence_rows.append(cells + [str(item["calibration"]["n_matches"]), e(number(outcome["ece"]))])
            for bin_ in outcome["bins"]:
                confidence_bins.append(cells + [f"{bin_['bin_low']:.0%}–{bin_['bin_high']:.0%}", str(bin_["n"]),
                                                 e(number(bin_["mean_pred"])), e(number(bin_["observed"]))])
    confidence_calibration = table([*group_headers, "保存的信心等级", "结果", "场数", "校准误差 ECE"], confidence_rows)
    confidence_calibration += details("信心等级内的概率分箱", table(
        [*group_headers, "保存的信心等级", "结果", "概率区间", "场数", "平均预测概率", "实际发生比例"], confidence_bins))
    # 最近踢完的几场
    recent_rows = []
    ids = (s.sort_values("kickoff_utc", ascending=False)["match_id"].drop_duplicates().head(30))
    for mid in ids:
        g = s[s["match_id"] == mid]
        r0 = g.iloc[0]
        name = (match_link(data, int(mid), "") if int(mid) in data.matches else e(str(mid)))
        hits = []
        model = g[g["source"].isin(("模型", "融合"))].dropna(subset=["p_home", "p_draw", "p_away", "home_goals", "away_goals"])
        for _, group in model.groupby(["model", "model_version", "kind", "archive_status"], dropna=False, sort=False):
            fused = group[group["source"] == "融合"]
            selected = (fused if len(fused) else group[group["source"] == "模型"])
            for _, row in selected.iterrows():
                ok = int(np.argmax([row["p_home"], row["p_draw"], row["p_away"]])) == side_of((row["home_goals"], row["away_goals"]))
                label = f"{identity_text(row)} · {row['kind']} · {archive_name(row.get('archive_status'))}"
                hits.append(f"{e(label)} <span class='{'hit' if ok else 'miss'}'>{'中' if ok else '没中'}</span>")
        result = (f"<span class='score'>{int(r0['home_goals'])}–{int(r0['away_goals'])}</span>"
                  if pd.notna(r0["home_goals"]) and pd.notna(r0["away_goals"]) else "—")
        if int(mid) in data.result_notes:
            result += f" <span class='tag warn'>{e(data.result_notes[int(mid)])}</span>"
        recent_rows.append([e(bj_text(r0["kickoff_utc"])), e(data.comp(str(r0["competition"]))), f"<td class='match'>{name}</td>",
                            result, "　".join(hits) or "—"])
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
            f"模型版本、预测时点、实际归档状态分别统计。log loss、Brier、RPS 都是越小越准。"
            f"收盘字段来自有证据的来源声明，不表示恰好在开球瞬间采集；最后一次观察到的赛前价格单独保留。零匹配表示无匹配证据。</p>"
            f"<h2>分组成绩</h2>{summary}<h2>独立收盘标杆</h2>{benchmarks}<h2>逐家公司同场比较</h2>{pairs}"
            f"<h2>收盘证据覆盖</h2><p class='small muted'>分母为当前复盘比赛集合，各公司、玩法独立统计；缺失原因可以重叠。</p>{coverage_table}"
            f"<h2>按保存的信心等级</h2>{by_level}<h2>概率校准</h2>{calibration}"
            f"<h2>按信心等级校准</h2>{confidence_calibration}"
            f"<h2>按联赛</h2>{summary_table(metrics['competitions'], ('competition',))}"
            f"<h2>按比赛标签（未知单列）</h2>{summary_table(metrics['tags'], ('tag', 'tag_value'))}"
            + (f"<h2>最近踢完的</h2>{recent}" if recent else "") + rep)
    return page("战绩 - 足球预测", body, "", data.now, cfg["refresh_minutes"])
