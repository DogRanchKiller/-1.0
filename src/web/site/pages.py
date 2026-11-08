"""首页 index.html、球队页 team/<球队>.html、战绩页 record.html（需求文档第 8 节）。单场页在 match.py。"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.common.config import expand_league_groups
from src.features.known_time import to_utc_text
from src.review.reports import EXTRA_NAMES, coverage_cells, coverage_for_reviews, extra_text, group_cells, live_metrics, metric_text
from src.web.site.common import (NOT_ADVICE, STATUS_TAGS, TeamFiles, bar_html, bj_text, day_title, details,
                                 e, kickoff_text, level_name, match_attrs, match_href, page, placeholder, table)
from src.web.site import picks as pk
from src.web.site import team_stats
from src.web.site.data import SiteData
from src.web.site.match import (archive_name, conclusion_html, final_probs, hit_html, identity_text, known_text, level,
                                live_html, score_text, show_level, side_of, version_name)


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
    # 没有正式比分、也没有改期之类的标记时：服务器现读现出的页面上可能有懂球帝的即时比分（静态文件里永远没有）
    return f"<span class='tag warn'>{e(tag)}</span>" if tag else live_html(data, mid)


def row_attrs(data: SiteData, m: dict) -> str:
    """首页一行 <tr> 上的 data-*（给 /static/site.js 筛选、排序、搜索用；看不见，没有脚本时什么都不影响）。
    在 match_attrs（比赛编号、联赛代码、开球时间）之外再加：
    data-league-name = 联赛中文名（筛选框里显示）；data-teams = 两队的原名（英文，搜索时英文名也能搜到）；
    data-pred = 有没有预测（1 / 0）；data-level = 信心等级 1 / 2 / 3（没有就不写；site.show_confidence 是 false 时也不写）；
    data-p = 最新一版最后的概率“主胜,平,客胜”（三位小数，按概率排序用；没有预测就不写）；
    data-tier-1x2 / data-tier-ou / data-tier-score = 最新一版这三个玩法的把握等级 high / mid / low（picks_v3，“只看高把握”筛选用；
    不分档、不是赔率算的就不写）；
    data-disagree = 模型和多家平均至少有一个玩法差 ≥ site.disagree_gap 时写 1（筛选用；没有就不写）。值都转义过。"""
    mid = m["match_id"]
    out = (match_attrs(m) + f" data-league-name='{e(data.comp(m['competition']))}'"
           f" data-teams='{e(m['home_team'] + ' ' + m['away_team'])}'")
    lt = data.latest(mid)
    if lt is None:
        return out + " data-pred='0'"
    row = lt[1]
    p, _ = final_probs(row)
    out += " data-pred='1'"
    lv = level(row)
    if show_level(data.settings) and isinstance(lv, int) and lv in (1, 2, 3):   # 不显示信心等级时也不写，筛选框就没有可选的
        out += f" data-level='{lv}'"
    if all(isinstance(x, (int, float)) and np.isfinite(x) for x in p):
        out += f" data-p='{','.join(f'{float(x):.3f}' for x in p)}'"
    for k, v in pk.tier_codes(pk.match_picks_v3(row, data.settings)).items():
        if v in pk.TIERS:
            out += f" data-tier-{k}='{e(v)}'"
    if pk.disagree_markets(pk.match_disagree(row, data.settings)):
        out += " data-disagree='1'"
    return out


def state_cell(data: SiteData, mid: int, empty: str = "") -> str:
    """首页“状态 / 赛果”那一格。还没有正式比分、也没有改期之类标记的比赛，格子带 data-live-slot（我们的比赛编号）：
    site.js 每 30 秒拿到懂球帝即时比分后只改这一格里的 .live-score（进行中 63'  2–1、红牌），别的不动。
    empty 是没有即时比分时格子里写的字（最近踢完那张表写“比分还没进来”），带 data-live-placeholder：
    有即时比分时它藏起来（hidden），即时比分没了脚本再把它露出来。"""
    html = state_html(data, mid)
    if mid in data.results or STATUS_TAGS.get(data.matches[mid].get("status")):
        filler = f"<span class='muted'>{e(empty)}</span>" if empty else ""
        return f"<td>{html or filler}</td>"
    hold = (f"<span class='muted' data-live-placeholder{' hidden' if html else ''}>{e(empty)}</span>" if empty else "")
    return f"<td data-live-slot='{int(mid)}'>{html}{hold}</td>"


def latest_cell(data: SiteData, mid: int) -> str:
    lt = data.latest(mid)
    if lt is None:
        return "<span class='muted'>还没有预测</span>"
    kind, row = lt
    # 一格里：最上面一句亚盘倾向（主结论，2026-11-05 起）+ 三色小条 + 概率和信心 + 其他玩法的倾向，最下面一行淡字写是哪一版。
    # 亚盘那一句：picks_v2 的让球预测（pk.pick_ah，存档里写的亚盘倾向那一边），例“倾向利物浦让半球赢盘，胜算 53%”；
    #   没到倾向门槛写“……的胜算 51%，接近五五开”（以存档为准，不重算门槛）。
    #   没有合格的主盘口（pick_ah 是 None）时这一句不写，格子和以前一模一样（胜平负概率 + 各玩法倾向），就是退回胜平负的写法。
    # 有亚盘那一句时，下面的倾向行不再重复亚盘（skip_ah），只写胜平负、平局风险、大小球、双方进球。
    # 再下面一行淡字写“把握”（picks_v3）：胜平负、大小球（主盘口）、比分各是哪一档、这一档的历史命中率，让球不分档；
    #   这一版不是赔率算的（没有赔率、接融合以前的旧版）写“不分档”。
    # 模型和多家平均差 ≥ site.disagree_gap 时标「分歧」并写哪几个玩法（需求文档第 1 节，列表页同样标）；差多少去单场页看。
    # 模型版本、归档状态这些技术信息只放在鼠标悬停的提示里（title），点进单场页的“技术细节”能看到全部。
    p, _ = final_probs(row)
    info = data.matches[mid]
    ah = pk.ah_sentence(pk.pick_ah(row), data.zh(info["home_team"]), data.zh(info["away_team"]))
    lead = f"<b class='lead'>{e(ah)}</b>" if ah else ""
    tip = f"{identity_text(row)}，{archive_name(row.get('archive_status'))}，{bj_text(row['cutoff_utc'])} 截止"
    tiers = pk.tier_line(pk.match_picks_v3(row, data.settings), data.settings,
                         data.zh(info["home_team"]), data.zh(info["away_team"])) or "把握：这一版不是赔率算的，不分档"
    flag = pk.index_disagree_html(pk.match_disagree(row, data.settings))
    return (f"<div title='{e(tip)}'>{lead}{bar_html(p)}{conclusion_html(row, skip_ah=bool(ah), show=show_level(data.settings))}"
            f"<span class='small muted tierline'>{e(tiers)}</span>"
            f"{flag}"
            f"<span class='small muted'>{e(version_name(kind))}这一版</span></div>")


# ---------- 首页 ----------
# 首页表格上面的“怎么看”：颜色条、概率、倾向、信心各是什么意思
def index_guide(settings: dict) -> str:
    """首页表格上面的“怎么看”。信心那一条按 settings.yaml 的 confidence 写：现在一个降级因素都不用时，说明几乎都是 ●●● 高；
    site.show_confidence 是 false（不显示信心等级）时换成一条“小标签写概率从哪来”。
    概率那一条按 settings.yaml 的 fusion.w 写：w = 1 时三个数全部来自赔率，模型只作对照（不然读的人会以为是模型算的）。
    “把握”那一条按 settings.yaml 的 picks_v3 写（哪几季、几场、大小球三档差多少、让球的总命中率）。"""
    w = (settings.get("fusion") or {}).get("w")
    source = ("现在这三个数全部来自博彩公司的赔率（去掉抽水后平均），因为回测里赔率比我们的模型准；"
              "模型自己算的概率在单场页里作对照。没有赔率的比赛才用模型的，会标“只有模型”。"
              if w is not None and float(w) == 1.0 else "")
    conf = settings.get("confidence") or {}
    tiers = tier_guide(settings)
    tier_li = f"<li>{e(tiers)}</li>" if tiers else ""
    gap = (settings.get("site") or {}).get("disagree_gap")
    disagree_li = (
        f"<li>“分歧”：模型自己和懂球帝多家平均在同一个玩法上差 {float(gap) * 100:.0f} 个百分点及以上就标出来，"
        "旁边写哪几个玩法；差多少点进单场页看。买不买你自己看。</li>"
        if gap is not None else "")
    if conf.get("downgrade_factors"):
        level = "按规则碰上提醒因素会降级，规则在单场页“结论是怎么得出来的”最后一步。"
    else:
        level = ("现在几乎每场都是 ●●● 高，因为回测里还没找到能区分把握大小的信号，"
                 "只有完全没有赔率时才降级，所以它暂时不能拿来挑比赛。")
    return ("<div class='card pad guide'><b>怎么看这张表</b><ul class='points small'>"
            "<li>彩色条：<span class='legend'><span><i class='s0'></i>主队赢</span><span><i class='s1'></i>平局</span>"
            "<span><i class='s2'></i>客队赢</span></span>，越长越可能。下面三个数就是三种结果的可能性，加起来 100%。"
            f"{e(source)}</li>"
            "<li>每场最上面那句粗字是亚盘（让球）的倾向：在开的公司最多的那条盘口上，哪一队赢盘的胜算大一些，"
            "例如“倾向主队让半球赢盘，胜算 53%”（赢一半算一半，走盘不算）；两边差不多时写“接近五五开”。"
            "没有合适的盘口时不写这一句，看下面的胜平负。</li>"
            f"{tier_li}"
            "<li>“倾向”：最可能的结果比第二可能的高出足够多才写；差不多时写“无明显倾向”。</li>"
            + (f"<li>●●● 高 / ●●○ 中 / ●○○ 低 是信心等级。{e(level)}</li>" if show_level(settings) else
               "<li>三个数后面的小标签写这几个数从哪来，例如“概率来自去水赔率”（博彩公司的赔率去掉抽水）；"
               "没用上赔率的写“只有模型（没和赔率融合）”。</li>")
            + disagree_li
            + "<li>点比赛名字，看“结论是怎么得出来的”和四块分析。</li></ul></div>")


def tier_guide(settings: dict) -> str | None:
    """“怎么看这张表”里“把握”那一条（纯文字，调用的地方转义）。数字都从 settings.yaml 的 picks_v3 来；
    settings 里没有 picks_v3 块（只给了一部分设置）时是 None，不写这一条。"""
    c = settings.get("picks_v3")
    if not c:
        return None
    ou = c["tiers"]["ou"]
    gap = (float(ou["high"]["rate"]) - float(ou["low"]["rate"])) * 100
    return (f"“把握 高 / 中 / 低”：同样这么写的预测，历史上（{pk.test_seasons_text(settings)}、{int(c['test_n']):,} 场）中了几成，"
            "括号里是这一档的历史命中率，不是这一场的可能性。胜平负、比分分得开；"
            f"大小球按主盘口算，三档只差 {gap:.0f} 个百分点；让球分不开，不分档（历史命中 {pk.rate1(c['ah_all']['rate'])}）。"
            "这一版不是赔率算的不分档。把握只说明以前中了几成，不说明赔率划算。")


def index_lists(data: SiteData) -> tuple[list[dict], list[dict]]:
    """首页列哪些比赛：(还没踢的, 最近踢完的)，各是 data.matches 里的那一条，按首页的顺序排好。
    首页（render_index）和服务器的数据接口 /api/matches（src/web/live_site.py）都用它，两边列的比赛一样。"""
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
    return upcoming, recent


def render_index(data: SiteData, teams: TeamFiles) -> str:
    cfg = data.settings["site"]
    league_names = "、".join(expand_league_groups(cfg["leagues"], data.settings, cfg.get("skip_leagues")).values())
    upcoming, recent = index_lists(data)
    n_pred = sum(1 for m in upcoming if m["match_id"] in data.preds)
    stats = (f"<div class='stats'><div class='stat hi'><b>{len(upcoming)}</b><span>未来 {cfg['days_ahead']} 天的比赛</span></div>"
             f"<div class='stat'><b>{n_pred}</b><span>已经有预测</span></div>"
             f"<div class='stat'><b>{len(recent)}</b><span>最近 {cfg['days_back']} 天踢完的</span></div></div>")
    # 每一组：(标题, 各行的格子, 各行 <tr> 上的 data-* 属性)
    groups: dict[str, tuple[str, list[list[str]], list[str]]] = {}
    for m in upcoming:
        key, title = day_title(m)
        mid = m["match_id"]
        row = [e(kickoff_text(m, with_date=False) if key != "9999" else kickoff_text(m)), e(data.comp(m["competition"])),
               f"<td class='match'>{match_link(data, mid, '')}</td>", latest_cell(data, mid), state_cell(data, mid)]
        g = groups.setdefault(key, (title, [], []))
        g[1].append(row)
        g[2].append(row_attrs(data, m))
    if groups:
        blocks = "".join(f"<h2>{e(title)}</h2>" + table(["时间", "联赛", "比赛", "最新一版的结论", ""], rows, "list", attrs)
                         for _, (title, rows, attrs) in sorted(groups.items()))
    else:
        blocks = "<h2>即将开赛</h2>" + placeholder(f"未来 {cfg['days_ahead']} 天没有{league_names}的比赛")
    rec_rows, rec_attrs = [], []
    for m in recent:
        mid = m["match_id"]
        lt = data.latest(mid)
        rec_rows.append([e(kickoff_text(m)), e(data.comp(m["competition"])), f"<td class='match'>{match_link(data, mid, '')}</td>",
                         state_cell(data, mid, "比分还没进来"), latest_cell(data, mid) if lt else "—",
                         hit_html(lt[1], data.results.get(mid)) if lt else "—"])
        rec_attrs.append(row_attrs(data, m))
    rec = (table(["开球", "联赛", "比赛", "赛果", "赛前最后一版", "胜平负猜中"], rec_rows, "list", rec_attrs) if rec_rows
           else placeholder(f"最近 {cfg['days_back']} 天没有踢完的比赛"))
    notes = "".join(f"<li>{e(n)}</li>" for n in data.notes[:10])
    body = (f"<h1>足球预测 <span class='accent'>即将开赛</span></h1><p class='muted'>"
            f"{e(league_names)}，时间都是北京时间。点比赛看各版本对比、比分概率和赔率变动。</p>"
            f"{stats}<div class='note'>{NOT_ADVICE}</div>{index_guide(data.settings)}{blocks}"
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
            + team_stats.render(data, team, teams)
            + f"<h2>最近 {cfg['team_days_back']} 天踢过的</h2>"
            + (table(["开球", "赛事", "比赛", "比分（主–客）和这队的胜负"], past_rows) if past_rows else placeholder("这段时间没有踢过的比赛"))
            + "<p class='small muted'>杯赛、欧战现在只有赛程，没有比分，写“—”。</p>"
            + f"<h2>预测战绩</h2>"
            + (f"<p>有预测、已踢完的 {len(rec_rows)} 场，胜平负猜中 {hits} 场。</p>"
               + table(["开球", "比赛", "比分", "赛前最后一版", "胜平负猜中"], rec_rows)
               if rec_rows else placeholder("还没有这队的预测战绩（预测过、而且踢完的比赛才算）")))
    return page(f"{data.zh(team)} - 足球预测", body, pre, data.now, cfg["refresh_minutes"])


# ---------- 战绩页 ----------
def picks_record(data: SiteData) -> str:
    """战绩页最上面：四个玩法的预测命中率（规则 picks_v2，src/web/site/picks.py），再列每场的四个预测。

    每场用开球前最新的那一版（和单场页一样）；只算有 90 分钟比分的。旁边写历史参考（期末考试那两季按同一规则算的，
    src/evaluate/picks_reference.py）和“一直猜主胜”的对照。不用 scores.csv，所以还没有复盘数据也能算。
    """
    items, skipped = pk.finished(data)
    if not items:
        return "<h2>四个玩法的命中率</h2>" + placeholder("还没有踢完、开球前出过预测的比赛")
    s = pk.summarize(items)
    ref, ref_why = pk.load_reference(data.settings)
    tiles, rows = [], []
    mref = ref.get("model_markets") if ref else None      # “模型自己”那列的历史参考（旧文件没有就不写）
    for k in pk.MARKETS:
        t, mt = s[k], s["model"][k]
        tiles.append(f"<div class='stat hi'><b>{pk.rate_text(t)}</b><span>{e(pk.MARKET_NAMES[k])}（{t['n']} 场）</span></div>")
        r = ref["markets"][k] if ref else None
        mine = (pk.rate_text(mt) + (f" <span class='small muted'>{mt['n']} 场</span>" if mt["n"] != t["n"] else "")
                + (f" <span class='small muted'>历史 {mref[k]['rate']:.0%}</span>" if mref else ""))
        rows.append([e(pk.MARKET_NAMES[k]), str(t["n"]), e(pk.counts_text(t)) if t["n"] else "—", f"<b>{pk.rate_text(t)}</b>",
                     f"{r['rate']:.0%}" if r else "—", mine])
    home = s["always_home"]
    rows.append(["<span class='muted'>对照：一直猜主胜</span>", str(home["n"]), e(pk.counts_text(home)), pk.rate_text(home),
                 f"{ref['always_home']['rate']:.0%}" if ref else "—", "—"])
    # 表头 table() 会转义，这里不再转义
    ref_head = (f"历史参考（{ref['seasons'][0]}/{str(ref['seasons'][0] + 1)[2:]}–{ref['seasons'][1]}/{str(ref['seasons'][1] + 1)[2:]}"
                f" 赛季 {ref['n_matches']} 场）") if ref else "历史参考"
    notes = []
    if not ref:
        notes.append(f"历史参考：{ref_why}。")
    if s["n_matches"] - s["score"]["n"]:
        notes.append(f"有 {s['n_matches'] - s['score']['n']} 场的最新一版没存比分矩阵（定时版本之前出的预览版），比分不算这几场。")
    if s["n_matches"] - s["ah"]["n"]:
        notes.append(f"有 {s['n_matches'] - s['ah']['n']} 场的最新一版没有让球的线或胜算，让球不算这几场。")
    if s["n_matches"] - s["ou"]["n"]:
        notes.append(f"有 {s['n_matches'] - s['ou']['n']} 场的最新一版没有 2.5 球的大小球概率，大小球不算这几场。")
    if s["model_only"]:
        notes.append(f"有 {s['model_only']} 场的那一版没用上赔率（接赔率融合以前出的，或那时还没有赔率），四个预测都是模型自己的数。")
    notes += [f"{why}的 {n} 场不算。" for why, n in skipped.items()]
    guide = ("<p class='small'>命中率 = 中的场数 ÷ 场数；让球、大小球的“赢半”算半场中、“输半”算半场没中，“走盘”（退回）不算。"
             "每个玩法预测的都是更可能的那一边，所以胜平负、让球、大小球的命中率在 50% 上下才正常；比分给 4 个最可能的，实际比分是其中一个就算中，4 个合起来的可能性一般只有四成左右，命中率也就在四成上下。"
             "几十场的命中率主要看运气，要几百场以上才看得出好坏，所以旁边放了历史参考（同一规则、开球前 24 小时、"
             "期末考试那两季，只是描述，不是又考了一次）。命中率高也不代表能赚钱：赔率已经把可能性算进去了。</p>"
             "<p class='small'>“模型自己”：同样的四条规则，只用我们自己的模型、不看赔率。有赔率时，上面的预测用的是赔率算出来的概率"
             "（历史上赔率比我们的模型准，所以网站用赔率；没用上赔率的那几场两列一样），这一列用来看模型自己能猜中多少、和赔率差多少。"
             "下面每场的表里，模型自己的预测和上面不一样时，用淡字另写一行；模型那一版没法预测时也写一行。</p>")
    # 每场的四个预测（最近 record_matches 场）。每行带 data-match-id、data-league、data-league-name，
    # 每个玩法那一格带 data-market（1x2 / ah / ou / score）、data-outcome（网站那一列的结果）、data-model-outcome（模型自己的结果）：
    # win 中 / half_win 赢半 / push 走盘 / half_loss 输半 / loss 没中 / none 没预测。给 /static/site.js 按玩法、联赛、猜没猜中筛选用，看不见
    per, per_attrs = [], []
    for x in items[:data.settings["site"]["record_matches"]]:
        info = data.matches.get(x["mid"]) or {}
        comp = info.get("competition") or ""
        per_attrs.append(f" data-match-id='{int(x['mid'])}' data-league='{e(comp)}' data-league-name='{e(data.comp(comp) if comp else '')}'")
        home_n, away_n = data.zh(info.get("home_team", "")), data.zh(info.get("away_team", ""))
        name = match_link(data, x["mid"], "") if x["mid"] in data.matches else e(str(x["mid"]))
        cells = [e(bj_text(x["kickoff"])), f"<td class='match'>{name}</td>",
                 f"<span class='score'>{x['score'][0]}–{x['score'][1]}</span>"]
        for k in pk.MARKETS:
            p, q = x["picks"][k], x["model_picks"][k]
            cell = "—" if p is None else f"{e(pk.pick_text(k, p, home_n, away_n))} {pk.outcome_html(x['outcomes'][k])}"
            if q is not None and (p is None or pk.pick_text(k, q, home_n, away_n) != pk.pick_text(k, p, home_n, away_n)):
                cell += (f"<br><span class='small muted'>模型：{e(pk.pick_text(k, q, home_n, away_n))} "
                         f"{pk.outcome_html(x['model_outcomes'][k])}</span>")
            elif q is None and p is not None:
                cell += "<br><span class='small muted'>模型：这一版没法预测</span>"
            cells.append(f"<td data-market='{k}' data-outcome='{e(x['outcomes'][k] or 'none')}'"
                         f" data-model-outcome='{e(x['model_outcomes'][k] or 'none')}'>{cell}</td>")
        per.append(cells)
    shown = min(len(items), data.settings["site"]["record_matches"])
    return ("<h2>四个玩法的命中率</h2>"
            f"<p>踢完、开球前出过预测的 {s['n_matches']} 场，每场用开球以前出的最新一版。</p>"
            f"<div class='stats'>{''.join(tiles)}</div>"
            + table(["玩法", "场数", "结果", "命中率", ref_head, "模型自己"], rows) + guide
            + "".join(f"<p class='small muted'>{e(n)}</p>" for n in notes)
            + f"<h2>每场的四个预测（最近 {shown} 场）</h2>"
            + table(["开球", "比赛", "比分", "胜平负", "让球", "大小球 2.5", "比分预测"], per, "picks-record", per_attrs))


def render_record(data: SiteData, out_dir: Path) -> str:
    """总战绩：最上面四个玩法的命中率和每场的四个预测（从预测存档算，picks_record）；
    下面“详细统计”从 data/review/scores.csv 汇总（K2 每次全部重算）。"""
    cfg = data.settings["site"]
    review_notes = "".join(f"<p class='note'>{e(note)}</p>" for note in data.review_notes.values())
    if data.scores is None:
        why = data.scores_note or "还没有复盘数据"
        body = (f"<h1>战绩 <span class='accent'>赛后对账</span></h1>{review_notes}{picks_record(data)}<h2>详细统计</h2>"
                + placeholder(why) + (placeholder("还没有复盘数据") if data.scores_note else ""))
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
                        + [e(metric_text(item["metrics"][c], 3)) for c in columns] + [e(item["note"])])
        return table([*group_headers, *[EXTRA_NAMES[x] for x in extra], "场数", "胜平负 log loss", "Brier", "RPS", "比分 log loss",
                      "大小 2.5 log loss", "双方进球 log loss", "证据说明"], rows) if rows else placeholder("无匹配证据")

    summary = summary_table(metrics["groups"])
    # 网站不显示信心等级时（site.show_confidence: false），下面“按信心等级”的两张表不放进页面（数照样从 scores.csv 算）
    show = show_level(data.settings)
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
    tech = (f"<h2>分组成绩</h2>{summary}<h2>独立收盘标杆</h2>{benchmarks}<h2>逐家公司同场比较</h2>{pairs}"
            f"<h2>收盘证据覆盖</h2><p class='small muted'>分母为当前复盘比赛集合，各公司、玩法独立统计；缺失原因可以重叠。</p>{coverage_table}"
            + (f"<h2>按保存的信心等级</h2>{by_level}" if show else "") + f"<h2>概率校准</h2>{calibration}"
            + (f"<h2>按信心等级校准</h2>{confidence_calibration}" if show else "")
            + f"<h2>按联赛</h2>{summary_table(metrics['competitions'], ('competition',))}"
            f"<h2>按比赛标签（未知单列）</h2>{summary_table(metrics['tags'], ('tag', 'tag_value'))}")
    body = (f"<h1>战绩 <span class='accent'>赛后对账</span></h1>"
            + review_notes + picks_record(data) + rep
            + "<h2>详细统计</h2>"
            + (details("最近踢完的：各版本胜平负猜中（不分玩法，每个版本都列）", recent) if recent else "")
            + details(f"详细统计表（log loss、校准、{'按信心等级、' if show else ''}按联赛、和收盘赔率逐家比较）",
                      f"<p class='muted'>从 data/review/scores.csv 汇总：已经踢完、有预测的比赛，每一版、每种概率一行。"
            f"模型版本、预测时点、实际归档状态分别统计。log loss、Brier、RPS 都是越小越准。"
            f"收盘字段来自有证据的来源声明，不表示恰好在开球瞬间采集；最后一次观察到的赛前价格单独保留。零匹配表示无匹配证据。</p>"
                      + tech))
    return page("战绩 - 足球预测", body, "", data.now, cfg["refresh_minutes"])
