"""把维基百科的杯赛、欧战页面解析成比赛记录，写进数据库（需求文档第 18 节第 5 项，方案 a）。

整体流程（数据只往一个方向流：原始层 → 清洗 → 数据库）：
1. 读 data/raw/wikipedia/<赛事代码>/<赛季>/ 下面保存好的网页（采集程序 src.collect.wikipedia 下载的）。
2. 从网页里找出每一场比赛。维基百科写比赛有三种格式，都认：
   - footballbox：一场比赛一个方框，日期、开球时间、比分、进球时间都有（欧战、决赛常用）；
   - 可折叠的小方框（tmpl-football-box-collapsible）：一行一场，德国杯、法国杯后几轮常用；
   - 表格（wikitable）：英足总杯、联赛杯前几轮常用，一行一场，常常没有每场的日期，只有“这一轮在哪几天踢”。
3. 根据页面标题和小标题判断是哪一轮（round），两回合的比赛分出第 1、2 回合（leg）。
4. 队名换成我们的统一编号：teams.yaml 里的球队用它的“英文”；别的球队（外国球队、低级别球队）
   用 “wiki:维基百科条目名” 当编号，保证不会和我们的球队撞名（例如皇马二队 ≠ 皇马）。
5. 只把“至少有一方是 teams.yaml 里的球队”的比赛写进数据库，其余的留在原始层。

日期规则：维基百科写的是比赛当地的日期。网页上有具体日期的记 date_precision='day'；
只知道“这一轮在哪几天”的，match_date 记这一轮的第一天，date_precision='round'，不编造具体哪天。
开球时间只有写了时区（例如 “20:45 CET”、“UTC+4”）才换算成 UTC 存进 kickoff_utc，否则不存。

比分规则：match_stats 里的 home_goals/away_goals 一律是“90 分钟（含补时）”的比分，和联赛口径一致。
踢了加时的比赛，加时后的比分、点球大战比分另外存在 cup_details 表里。
加时赛的比赛如果算不出 90 分钟比分（网页上没有进球时间），90 分钟比分留空，不猜。

运行：.venv\\Scripts\\python -m src.clean.wikipedia_import
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

from src.clean.football_data_import import load_teams
from src.clean.schema import create_tables
from src.common.config import load_settings, resolve_path
from src.common.db import connect

log = logging.getLogger(__name__)

SOURCE = "wikipedia"

# 英文月份 → 数字
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9

# 网页上常见的时区缩写 → 比 UTC 早几小时。维基百科写的时间就是“当地时间 + 时区”。
TZ_OFFSETS = {
    "UTC": 0, "GMT": 0, "WET": 0, "WEST": 1, "BST": 1, "IST": 1,
    "CET": 1, "CEST": 2, "EET": 2, "EEST": 3, "MSK": 3, "MSD": 4,
    "TRT": 3, "AZT": 4, "AMT": 4, "GET": 4, "SAMT": 4,
}

# 比分里的横线：维基百科用的是 en dash（–），也有人写普通减号或其他横线
DASH = "[–\\-−—]"      # 不含冒号：冒号会把开球时间 “20:30” 误认成比分
SCORE_RE = re.compile(rf"(\d+)\s*{DASH}\s*(\d+)")
# 进球时间，例如 27'  90+4'  104'
MINUTE_RE = re.compile(r"(\d+)(?:\s*\+\s*(\d+))?\s*['′’]")


def clean_text(node) -> str:
    """取出一个网页元素里的文字，把各种空白统一成一个空格，去掉脚注 [1]、[a]。"""
    if node is None:
        return ""
    if isinstance(node, Tag):
        text = " ".join(_visible_strings(node))
    else:
        text = str(node)
    text = unicodedata.normalize("NFKC", text.replace("\xa0", " "))
    text = re.sub(r"\[\s*(?:\d+|[a-z]|note \d+|nb \d+)\s*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_SKIP_CLASSES = {"reference", "sortkey"}


def _is_skipped(tag: Tag) -> bool:
    """这个元素的文字要不要跳过：脚注、样式表、排序用的隐藏文字、display:none 的隐藏内容。"""
    if tag.name in ("style", "script"):
        return True
    if _SKIP_CLASSES.intersection(tag.get("class") or ()):
        return True
    style = tag.get("style")
    return bool(tag.name == "span" and style and "display:none" in style.replace(" ", ""))


def _visible_strings(node: Tag):
    """按顺序取出元素里看得见的文字片段（不复制网页，直接逐层往下走，比重新解析快很多）。"""
    stack = [iter(node.children)]
    while stack:
        child = next(stack[-1], None)
        if child is None:
            stack.pop()
            continue
        if isinstance(child, Tag):
            if not _is_skipped(child):
                stack.append(iter(child.children))
        elif type(child) is NavigableString:   # 普通文字；注释（Comment）等特殊文字不要
            yield str(child)


def _has_class(*names: str):
    """生成一个检查函数：元素的 class 里有 names 中任意一个就算。用于代替较慢的 CSS 选择器。"""
    wanted = set(names)
    return lambda c: c is not None and bool(wanted.intersection(c.split()))


def _find_class(node, *names: str, tag=None):
    """在 node 里找第一个带指定 class 的元素（找不到返回 None）。"""
    return None if node is None else node.find(tag, class_=_has_class(*names))


# ---------------------------------------------------------------- 日期、时间、比分

DATE_DMY = re.compile(r"(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})")
DATE_MDY = re.compile(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})")
DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_date(text: str) -> dt.date | None:
    """从一段文字里认出日期：“19 July 2000”、“July 19, 2000”、“2000-07-19”都行。认不出返回 None。"""
    text = text or ""
    for pattern, order in ((DATE_ISO, "ymd"), (DATE_DMY, "dmy"), (DATE_MDY, "mdy")):
        for m in pattern.finditer(text):
            try:
                if order == "ymd":
                    y, mo, d = int(m[1]), int(m[2]), int(m[3])
                elif order == "dmy":
                    d, mo, y = int(m[1]), MONTHS.get(m[2].lower()), int(m[3])
                else:
                    mo, d, y = MONTHS.get(m[1].lower()), int(m[2]), int(m[3])
                if mo:
                    return dt.date(y, mo, d)
            except (ValueError, TypeError):
                continue
    return None


_TIME = r"(\d{1,2})[:.](\d{2})"
_TZ_AFTER = re.compile(_TIME + r"\s*(?:(?:UTC|GMT)\s*([+\-−–])\s*(\d{1,2})(?::?(\d{2}))?|([A-Z]{2,5})\b)")


def parse_kickoff(text: str, day: dt.date | None) -> tuple[str | None, str | None]:
    """开球时间。返回 (当地时间文字, UTC 时间)。

    只有写了时区的才换算成 UTC：时区写在哪个时间后面，就用哪个时间换算。例如
      “20:45 CET” → 19:45 UTC；“22:45 UTC+4” → 18:45 UTC；
      “20:45 (19:45 UTC+1)” → 括号里的 19:45 是球场当地时间、当地是 UTC+1 → 18:45 UTC。
    没写时区的（只有 “20:45”）返回 (“20:45”, None)：不知道是哪个时区就不猜。
    """
    text = clean_text(text)
    first = re.search(_TIME, text)
    if not first:
        return None, None
    local_text = f"{int(first[1]):02d}:{first[2]}"
    if day is None:
        return local_text, None
    for m in _TZ_AFTER.finditer(text):
        hour, minute = int(m[1]), int(m[2])
        if hour > 23 or minute > 59:
            continue
        if m[3]:
            sign = -1 if m[3] in "-−–" else 1
            offset = sign * (int(m[4]) + int(m[5] or 0) / 60)
        elif m[6] in TZ_OFFSETS:
            offset = TZ_OFFSETS[m[6]]
        else:
            continue
        local = dt.datetime(day.year, day.month, day.day, hour, minute)
        utc = local - dt.timedelta(hours=offset)
        return local_text, utc.strftime("%Y-%m-%d %H:%M")
    return local_text, None


@dataclass
class Score:
    """一个比分格子里读出来的东西。"""
    home: int | None = None          # 比赛最后的比分（踢了加时就是加时后的）
    away: int | None = None
    extra_time: bool = False         # 踢了加时
    pen_home: int | None = None      # 点球大战
    pen_away: int | None = None
    special: str = ""                # walkover（对手弃权，没踢）、awarded（判的比分）、void（作废）、postponed
    golden_goal: bool = False        # 金球：加时赛谁先进球谁赢，比赛马上结束（2004 年以前用过）
    silver_goal: bool = False        # 银球：加时上半场结束时领先就赢（2002–2004 欧战用过）


def parse_score(text: str) -> Score:
    """读比分格子，例如 “4–1”、“1–1 (a.e.t.)”、“1–1 (6–5 p)”、“0–0 3–2 p.”、“w/o”。"""
    t = clean_text(text)
    low = t.lower()
    s = Score()
    if re.search(r"\bw/?o\b|walkover|w\.o\.", low):
        s.special = "walkover"
        return s
    if re.search(r"\b(postponed|cancelled|canceled|abandoned|suspended)\b", low):
        s.special = "postponed"
    if re.search(r"\bawd\b|awarded|\(\s*w\s*\)|forfeit", low):
        s.special = "awarded"
    if re.search(r"\bvoid\b|annulled", low):
        s.special = "void"
    if re.search(r"\ba\s*\.\s*e\s*\.\s*t\b|\baet\b|extra time", low):
        s.extra_time = True
    if re.search(r"g\s*\.\s*g\s*\.|golden goal", low):
        s.extra_time = s.golden_goal = True
    if re.search(r"s\s*\.\s*g\s*\.|silver goal", low):
        s.extra_time = s.silver_goal = True
    # 点球：“(6–5 p)”、“3–2 p.”、“(4–3 pen.)”、“4–3 on penalties”
    pen = re.search(rf"(\d+)\s*{DASH}\s*(\d+)\s*(?:p\b|p\.|pen|pens|on pen|\(?p\))", low)
    if pen:
        s.pen_home, s.pen_away = int(pen[1]), int(pen[2])
        low = low[:pen.start()] + " " + low[pen.end():]
    m = SCORE_RE.search(low)
    if m:
        s.home, s.away = int(m[1]), int(m[2])
    return s


def count_minutes(cell) -> list[int] | None:
    """数一个进球格子里每个进球的分钟数（90+4 记成 90，表示还在 90 分钟内）。

    点球大战的罚球不在这里（它们在单独的“Penalties”一行）。认不出分钟的进球（写成 “?”）返回 None。
    """
    if cell is None:
        return []
    goals = cell.find_all("span", class_=_has_class("fb-goal"))
    minutes: list[int] = []
    for g in goals:
        text = clean_text(g)
        found = MINUTE_RE.findall(text)
        if not found:
            return None
        for base, _added in found:
            minutes.append(int(base))
    return minutes


def regular_time_score(final: Score, home_cell, away_cell) -> tuple[int | None, int | None]:
    """算 90 分钟比分。

    没踢加时：90 分钟比分就是最后比分。
    踢了加时：数进球时间，90 分钟（含补时，写成 90+x）以内的进球才算。
    数出来的总进球数必须和最后比分一致，否则说明网页上进球时间不全，返回 (None, None)，不猜。
    """
    if not final.extra_time:
        return final.home, final.away
    hm, am = count_minutes(home_cell), count_minutes(away_cell)
    if hm is None or am is None or final.home is None:
        return None, None
    if len(hm) != final.home or len(am) != final.away:
        return None, None
    return sum(1 for x in hm if x <= 90), sum(1 for x in am if x <= 90)


# 不是球队链接的东西：国旗、足协、国家、脚注
_NOT_TEAM = re.compile(
    r"football association|football federation|federation of|^file:|^template:|national football|"
    r"^association football|union of|fútbol|^list of|league$|^help:|^wikipedia:", re.I)


@dataclass
class TeamRef:
    """网页上的一支球队：条目名（链接指向的维基百科条目）和显示出来的名字。"""
    title: str        # 链接的条目名，例如 “Real Madrid CF”；没有链接时等于 shown
    shown: str        # 页面上显示的文字，例如 “Real Madrid”
    country: str = "" # 国旗图标上写的国家（英文，例如 Spain）；国内杯赛一般没有国旗


def team_from_cell(cell) -> TeamRef | None:
    """从一个球队格子里取出球队。国旗图标里的链接（指向足协）不算。"""
    if cell is None:
        return None
    country = ""
    img = None
    for flag in cell.find_all(class_=_has_class("flagicon", "flagdeco")):
        img = flag.find("img", alt=True)
        if img is not None:
            break
    if img is not None:
        country = img.get("alt", "").strip()   # 国旗图片的说明文字就是国家名，例如 “England”
    for flag in cell.find_all(lambda t: t.name == "sup" or bool({"flagicon", "flagdeco", "reference"}.intersection(t.get("class") or ()))):
        flag.extract()
    shown = clean_text(cell)
    shown = re.sub(r"\s*\((?:[^()]*\d[^()]*|[A-Z][a-z]{0,4}\.?|H|A|N)\)\s*$", "", shown).strip()  # 去掉 “(2)”、“(H)” 一类标注
    shown = shown.strip(" *†‡§¶#")
    link = None
    for a in cell.find_all("a"):
        title = (a.get("title") or "").strip()
        href = a.get("href") or ""
        if not title or "redlink=1" in href or _NOT_TEAM.search(title):
            continue
        if "/wiki/" not in href and not href.startswith("./"):
            continue
        link = title
        break
    if not shown and not link:
        return None
    return TeamRef(title=link or shown, shown=shown or link, country=country)


@dataclass
class RawMatch:
    """网页上的一场比赛（还没换成我们的球队编号）。"""
    competition: str
    season: int
    page: str                  # 来自哪个网页（文件名，不含 .html.gz）
    raw_file: str              # 原始文件相对路径
    headings: list[str]        # 这场比赛上面的小标题，从大到小
    home: TeamRef
    away: TeamRef
    date: dt.date | None
    date_precision: str        # day / round
    kickoff_local: str | None
    kickoff_utc: str | None
    score: Score
    ft_home: int | None        # 90 分钟比分
    ft_away: int | None
    fmt: str                   # footballbox / collapsible / table / series
    leg_hint: int = 0          # 网页上直接写了第几回合（表格里的 1st leg / 2nd leg 列）
    replay_hint: bool = False  # 小标题或表格里写了 replay（重赛）
    order: int = 0             # 在网页里的先后顺序
    round: str = ""            # 轮次代码，例如 Q2、G、R16、F


def parse_footballbox(box: Tag, season: int | None = None) -> dict | None:
    """读一个 footballbox 方框。返回各项内容，读不出两支球队时返回 None。

    日期的坑：网页作者只写 “28 February”、不写年份时，模板会自己补上“生成网页那一年”的年份
    （2007/08 国王杯半决赛的 time 标签写的是 2026-02-28）。所以看得见的日期没写年份时，
    不信 time 标签里的年份，按赛季推（见 year_from_season）；推不出就当没有日期。"""
    home = team_from_cell(_find_class(box, "fhome"))
    away = team_from_cell(_find_class(box, "faway"))
    if not home or not away:
        return None
    iso = _find_class(box, "bday", "dtstart")
    day = parse_date(iso.get_text()) if iso else None
    tm = box.find("time", datetime=True)
    shown = clean_text(_find_class(box, "fdate"))
    year_missing = bool(shown) and not re.search(r"\b(?:19|20)\d{2}\b", shown)
    if day is None and tm is not None and not year_missing:
        day = parse_date(tm["datetime"])
    if day is None and year_missing:
        day = first_date_in(shown, season)
    if day is None:
        day = parse_date(shown)
    kickoff_local, kickoff_utc = parse_kickoff(clean_text(_find_class(box, "ftime")), day)
    # time 标签里带了完整的 UTC 时间（例如 2009-04-21T18:30:00+00:00）的，优先用它
    if tm is not None and day is not None:
        m = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::\d{2})?(Z|[+\-]\d{2}:?\d{2})$", tm["datetime"].strip())
        if m:
            off = 0 if m[4] == "Z" else int(m[4][:3]) + (1 if m[4][0] == "+" else -1) * int(m[4][-2:]) / 60
            local = dt.datetime.fromisoformat(f"{m[1]}T{m[2]}:{m[3]}")
            kickoff_utc = (local - dt.timedelta(hours=off)).strftime("%Y-%m-%d %H:%M")
    score = parse_score(_find_class(box, "fscore"))
    # 点球大战：有一行标题是 “Penalties”，下面一行中间格子是点球比分
    for th in box.find_all("th"):
        if clean_text(th).lower() in ("penalties", "penalty shoot-out", "penalty shootout"):
            row = th.find_parent("tr")
            nxt = row.find_next_sibling("tr") if row else None
            mid = None
            if nxt is not None:
                cells = nxt.find_all(["th", "td"])
                mid = cells[len(cells) // 2] if cells else None
            pen = SCORE_RE.search(clean_text(mid)) if mid else None
            if pen:
                score.pen_home, score.pen_away = int(pen[1]), int(pen[2])
            break
    ft = regular_time_score(score, _find_class(box, "fhgoal"), _find_class(box, "fagoal"))
    return dict(home=home, away=away, date=day, kickoff_local=kickoff_local,
                kickoff_utc=kickoff_utc, score=score, ft=ft)


def parse_collapsible(table: Tag) -> dict | None:
    """读一个可折叠小方框：第一行 日期 | 主队 | 比分 | 客队 | 球场，第二行 开球时间 | 主队进球 | 报告 | 客队进球。"""
    rows = table.find_all("tr", recursive=False) or table.select("tbody > tr")
    if not rows:
        return None
    cells = rows[0].find_all(["td", "th"], recursive=False)
    if len(cells) < 4:
        return None
    teams = [c for c in cells if "vcard" in (c.get("class") or [])]
    if len(teams) >= 2:
        hi, ai = cells.index(teams[0]), cells.index(teams[1])
    else:
        hi, ai = 1, 3
    day = parse_date(clean_text(cells[0]))
    score = parse_score(cells[hi + 1] if hi + 1 < ai else cells[2])
    home, away = team_from_cell(cells[hi]), team_from_cell(cells[ai])
    if not home or not away:
        return None
    kickoff_local = kickoff_utc = None
    hgoal = agoal = None
    if len(rows) > 1:
        c2 = rows[1].find_all(["td", "th"], recursive=False)
        if c2:
            kickoff_local, kickoff_utc = parse_kickoff(clean_text(c2[0]), day)
        if len(c2) >= 4:
            hgoal, agoal = c2[1], c2[3]
    for r in rows[2:]:
        txt = clean_text(r).lower()
        if "penalties" in txt or "penalty" in txt:
            nxt = r.find_next_sibling("tr")
            for src in (r, nxt):
                if src is None:
                    continue
                pen = re.search(rf"(\d+)\s*{DASH}\s*(\d+)", clean_text(src))
                if pen:
                    score.pen_home, score.pen_away = int(pen[1]), int(pen[2])
                    break
            break
    ft = regular_time_score(score, hgoal, agoal)
    return dict(home=home, away=away, date=day, kickoff_local=kickoff_local,
                kickoff_utc=kickoff_utc, score=score, ft=ft)


# ---------------------------------------------------------------- 表格

def _header_cells(table: Tag) -> list[str]:
    """表格第一行（表头）每一格的文字，小写。"""
    first = table.find("tr")
    if first is None:
        return []
    return [clean_text(c).lower() for c in first.find_all(["th", "td"], recursive=False)]


def match_table_columns(table: Tag) -> dict | None:
    """判断一个表格是不是“一行一场比赛”的表，是的话返回各列的位置。

    认得的表头：Home team / Team 1 | Score | Away team / Team 2，可以带 Date、Tie no、Attendance。
    两回合汇总表（Team 1 | Agg. | Team 2 | 1st leg | 2nd leg）返回 kind='series'。
    """
    head = _header_cells(table)
    if not head:
        return None
    def find(*keys):
        for i, h in enumerate(head):
            if any(re.search(k, h) for k in keys):
                return i
        return None
    home = find(r"^home", r"^team 1", r"^home team")
    away = find(r"^away", r"^team 2", r"^away team")
    if home is None or away is None or away <= home:
        return None
    agg = find(r"^agg")
    if agg is not None and home < agg < away:
        return dict(kind="series", home=home, away=away, agg=agg,
                    leg1=find(r"1st leg", r"first leg"), leg2=find(r"2nd leg", r"second leg"),
                    date=None, n=len(head))
    score = find(r"^score", r"^result")
    if score is None:
        between = [i for i in range(home + 1, away)]
        score = between[0] if len(between) == 1 else None
    if score is None:
        return None
    return dict(kind="table", home=home, away=away, score=score, n=len(head),
                date=find(r"^date"), tie=find(r"^tie", r"^no\.?$", r"^match"),
                score_is_90=bool(re.search(r"score\s*1|\b90\b", head[score])))


def _expand_rows(table: Tag) -> list[list[Tag]]:
    """把表格的每一行展开成格子列表，处理跨行合并（rowspan，例如同一天的几场共用一个日期格子）。"""
    rows: list[list[Tag]] = []
    carry: dict[int, tuple[Tag, int]] = {}   # 列号 → (格子, 还要往下延续几行)
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = tr.find_all(["td", "th"], recursive=False)
        out: list[Tag] = []
        col = 0
        pending = list(cells)
        while pending or col in carry:
            if col in carry:
                cell, left = carry[col]
                out.append(cell)
                if left <= 1:
                    del carry[col]
                else:
                    carry[col] = (cell, left - 1)
                col += 1
                continue
            cell = pending.pop(0)
            span = int(re.sub(r"\D", "", cell.get("colspan", "1")) or 1)
            rspan = int(re.sub(r"\D", "", cell.get("rowspan", "1")) or 1)
            for k in range(span):
                out.append(cell)
                if rspan > 1:
                    carry[col] = (cell, rspan - 1)
                col += 1
        rows.append(out)
    return rows


def parse_note(text: str, prev: dict) -> None:
    """表格里比赛下面的说明行，例如 “Luton Town won 4–2 on penalties”、“Derby County won 3–1 after extra time”，
    写回上一场。说明行里的比分是“赢的一方在前”，要靠 “某某 won” 判断哪边是主队。"""
    low = text.lower()
    score: Score = prev["score"]
    aet = re.search(rf"(\d+)\s*{DASH}\s*(\d+)\s*\(?\s*(?:after extra time|a\.?e\.?t)", low)
    if aet or "extra time" in low or "golden goal" in low:
        score.extra_time = True
    if aet:
        a, b = int(aet[1]), int(aet[2])
        if a == b:
            prev["aet_from_note"] = (a, b)
        else:
            winner = _note_winner(text[:aet.start()] or text, prev)
            if winner == "home":
                prev["aet_from_note"] = (max(a, b), min(a, b))
            elif winner == "away":
                prev["aet_from_note"] = (min(a, b), max(a, b))
    pen = re.search(rf"(\d+)\s*{DASH}\s*(\d+)\s*(?:on penalties|on pens|pen)", low)
    if pen:
        a, b = int(pen[1]), int(pen[2])
        winner = _note_winner(text[:pen.start()], prev)
        hi, lo = max(a, b), min(a, b)
        if winner == "home":
            score.pen_home, score.pen_away = hi, lo
        elif winner == "away":
            score.pen_home, score.pen_away = lo, hi


def _note_winner(text: str, prev: dict) -> str | None:
    """说明行里 “某某 won ...” 的某某是主队还是客队。认不出返回 None。"""
    m = re.search(r"([^–—\-:;,.]+?)\s+won\b", text)
    if not m:
        return None
    who = m[1].strip().lower()
    names = {"home": prev["home"], "away": prev["away"]}
    for side, ref in names.items():
        for n in (ref.shown, ref.title):
            n = n.lower()
            if n and (n in who or who in n):
                return side
    return None


def parse_series_row(cells: list[Tag], cols: dict) -> list[dict]:
    """两回合汇总表的一行：Team 1 | 总比分 | Team 2 | 第 1 回合 | 第 2 回合。

    维基百科的写法：第 1 回合 Team 1 主场，第 2 回合 Team 2 主场；两个回合的比分都按 “Team 1–Team 2” 的顺序写。
    这种表没有日期，只在同一场比赛找不到 footballbox 时才用（日期用这一轮的日期，精度记 round）。
    """
    t1, t2 = team_from_cell(cells[cols["home"]]), team_from_cell(cells[cols["away"]])
    if not t1 or not t2:
        return []
    out = []
    for leg, key in ((1, "leg1"), (2, "leg2")):
        i = cols.get(key)
        if i is None or i >= len(cells):
            continue
        s = parse_score(cells[i])
        if s.home is None and not s.special:
            continue
        if leg == 1:
            home, away, sc = t1, t2, s
        else:   # 第 2 回合主客对调，比分也要对调
            home, away = t2, t1
            sc = Score(home=s.away, away=s.home, extra_time=s.extra_time, pen_home=s.pen_away,
                       pen_away=s.pen_home, special=s.special, golden_goal=s.golden_goal,
                       silver_goal=s.silver_goal)
        ft = (sc.home, sc.away) if not sc.extra_time else (None, None)
        out.append(dict(home=home, away=away, date=None, kickoff_local=None, kickoff_utc=None,
                        score=sc, ft=ft, leg_hint=leg))
    return out


def parse_match_table(table: Tag, cols: dict, dagger_aet: bool, season: int | None = None) -> list[dict]:
    """读“一行一场”的表格（英足总杯、联赛杯、法国杯前几轮）。

    - 只有一两个格子的行：要么是日期（后面几场都是这一天），要么是上一场的说明（加时、点球）。
    - 英足总杯表格里比分后面的 † 表示“加时后”（页面上有 “† – After extra time” 说明时才这样认）。
    - 联赛杯表头写 “Score 1”（脚注：90 分钟比分），加时后的比分在下面的说明行里。
    - 日期行有时不写年份（法国杯 “16 November”）：给了 season 就按赛季推年份（见 year_from_season）。
    """
    out: list[dict] = []
    current_date = None
    rows = _expand_rows(table)
    for cells in rows[1:]:
        uniq = list(dict.fromkeys(cells))
        if len(uniq) <= 2 and len(uniq) < cols["n"] - 1:
            text = clean_text(uniq[-1]) if uniq else ""
            d = parse_date(text)
            if d is None and season is not None and len(text) <= 30:
                d = first_date_in(text, season)     # 只有“16 November”这样一小段字才这样推，长句子不推
            low = text.lower()
            if d and " won " not in f" {low} " and "extra time" not in low and "penalt" not in low:
                current_date = d
            elif out and text:
                parse_note(text, out[-1])
            continue
        if len(cells) <= max(cols["home"], cols["away"], cols["score"]):
            continue
        if clean_text(cells[cols["score"]]).lower().startswith(("score", "result")):
            continue   # 表格中间又重复了一次表头
        home, away = team_from_cell(cells[cols["home"]]), team_from_cell(cells[cols["away"]])
        if not home or not away:
            continue
        raw_score = clean_text(cells[cols["score"]])
        score = parse_score(raw_score)
        if dagger_aet and "†" in raw_score:
            score.extra_time = True
        day = parse_date(clean_text(cells[cols["date"]])) if cols.get("date") is not None else None
        day = day or current_date
        first = clean_text(cells[cols["tie"]] if cols.get("tie") is not None else cells[0]).lower()
        out.append(dict(home=home, away=away, date=day, kickoff_local=None, kickoff_utc=None,
                        score=score, ft=None, replay_hint="replay" in first,
                        score_is_90=cols["score_is_90"]))
    # 90 分钟比分
    for m in out:
        s: Score = m["score"]
        if not s.extra_time:
            m["ft"] = (s.home, s.away)
        elif m.get("score_is_90"):
            # 联赛杯：格子里就是 90 分钟比分；加时后的比分在说明行里（没有就留空）
            m["ft"] = (s.home, s.away)
            aet = m.get("aet_from_note")
            if aet:
                s.home, s.away = aet
            elif s.pen_home is None:
                s.home = s.away = None
        else:
            m["ft"] = (None, None)
    return out


# ---------------------------------------------------------------- 轮次

# 判断轮次的规则：按顺序试，先匹配上的算。“semi-final”里也有“final”，所以 final 放最后；
# “third qualifying round” 要在 “third round” 前面。
ROUND_RULES: list[tuple[str, str]] = [
    (r"extra preliminary round", "EPR"),
    (r"preliminary round|preliminary qualifying|preliminary stage", "PR"),
    (r"(first|1st) qualifying round|qualifying round 1\b", "Q1"),
    (r"(second|2nd) qualifying round|qualifying round 2\b", "Q2"),
    (r"(third|3rd) qualifying round|qualifying round 3\b", "Q3"),
    (r"(fourth|4th) qualifying round|qualifying round 4\b", "Q4"),
    (r"^qualifying round$", "Q1"),     # 2000/01–2003/04 欧洲联盟杯只有一轮资格赛，不带序号
    (r"knockout (round |phase )?play-?offs?|play-?offs? round of the knockout", "KPO"),
    (r"play-?off round|play-?offs?\b", "PO"),
    (r"league phase|league stage", "LP"),
    (r"first group stage", "G1"),
    (r"second group stage", "G2"),
    (r"group stage|group phase|^groups?\b|\bgroup [a-l1-8]$", "G"),
    (r"round of 128", "R128"),
    (r"round of 64", "R64"),
    (r"round of 32|last 32|sixteenth-?finals|1/16", "R32"),
    (r"round of 16|last 16|eighth-?finals|1/8", "R16"),
    (r"quarter-?finals?|sixth round proper", "QF"),
    (r"semi-?finals?", "SF"),
    (r"third[ -]place|3rd place", "3P"),
    (r"(first|1st) round", "R1"),
    (r"(second|2nd) round", "R2"),
    (r"(third|3rd) round", "R3"),
    (r"(fourth|4th) round", "R4"),
    (r"(fifth|5th) round", "R5"),
    (r"(sixth|6th) round", "R6"),
    (r"(seventh|7th) round", "R7"),
    (r"(eighth|8th) round", "R8"),
    (r"(ninth|9th) round", "R9"),
    (r"(tenth|10th) round", "R10"),
    (r"\bfinals?\b", "F"),
]
_ROUND_RULES = [(re.compile(p, re.I), code) for p, code in ROUND_RULES]

# 这些小标题不说明是哪一轮，跳过，再往上一级找
GENERIC_HEADINGS = re.compile(
    r"^(summary|matches|results|fixtures|first leg|second leg|1st leg|2nd leg|replays?|"
    r"seeding|draw|bracket|format|teams|notes|references|schedule|overview|"
    r"champions path|league path|main path|non-champions path|main route|champions route|"
    r"seeded|unseeded|north|south|northern section|southern section|match details|details|"
    r"overseas regions?|tiebreakers|ranking.*|standings|tables?|match|matchday \d+)$", re.I)


def _round_code(text: str, suffix: str, context: str) -> str | None:
    """一段标题文字 → 轮次代码。"""
    low = text.lower()
    # 资格赛页面里的 “First round” 指第一轮资格赛
    if "qualifying" in suffix.lower() and "qualifying" not in low and "play" not in low:
        low = re.sub(r"\b(first|second|third|fourth) round\b", r"\1 qualifying round", low)
    for rx, code in _ROUND_RULES:
        if rx.search(low):
            if code == "G" and "first group stage" in context:
                return "G1"
            if code == "G" and "second group stage" in context:
                return "G2"
            return code
    return None


# 资格赛阶段：它下面的 “Semi-finals”、“Final” 小标题是这个阶段里的小比赛（例如欧冠预选赛的四队小组），
# 不是正赛的半决赛、决赛
QUALIFYING_STAGES = {"EPR", "PR", "Q1", "Q2", "Q3", "Q4", "PO"}


def classify_round(headings: list[str], page_title: str, main_title: str) -> str | None:
    """根据小标题（从最细的往上找）和页面标题，判断是哪一轮。认不出返回 None。"""
    suffix = page_title[len(main_title):].strip(" _").replace("_", " ") if page_title.startswith(main_title) else ""
    candidates = [h for h in reversed(headings) if h and not GENERIC_HEADINGS.match(h.strip())]
    if suffix:
        candidates.append(suffix)
    context = " ".join(headings + [suffix]).lower()
    codes = [c for c in (_round_code(t, suffix, context) for t in candidates) if c]
    if not codes:
        return None
    if codes[0] in ("SF", "F", "3P"):
        outer = next((c for c in codes[1:] if c in QUALIFYING_STAGES), None)
        if outer:
            return outer
    for h in candidates:
        zone = re.fullmatch(r"group (\d{1,2})[a-z]", h.strip(), re.I)
        if zone:
            return f"R{int(zone[1])}"      # 法国杯 2024/25 起写 “Group 7A”= 第 7 轮的 A 区
    if codes[0] == "G":
        # 法国杯、国王杯的某一轮按地区分组抽签：“Seventh round > Main draw > Group F”。
        # 这里的 Group 只是抽签分区，不是小组赛，轮次看外面那层（R7）
        outer = next((c for c in codes[1:] if re.fullmatch(r"R\d+|EPR|PR", c)), None)
        if outer:
            return outer
    return codes[0]


_DAY_LINKS = {",", "&", "–", "-", "and", "to", "or", "of", "between"}   # 连着写的几天之间的连接词：“16, 17 and 18 November”


def year_from_season(month: int, season: int | None) -> int | None:
    """只写了日、月没写年（“played 16, 17 and 18 November”）：按赛季推年份。
    8–12 月 = 赛季开始那年，1–5 月 = 第二年。6、7 月说不清（可能是上赛季末尾，也可能是新赛季资格赛），不推。"""
    if season is None:
        return None
    if 8 <= month <= 12:
        return season
    if 1 <= month <= 5:
        return season + 1
    return None


def first_date_in(text: str, season: int | None = None) -> dt.date | None:
    """从一段话里找最早的日期，能认出 “between 7 and 9 November 2008”、“18–19 & 25–26 February 2020”
    这种只在最后写一次月份和年份的写法（7、9 两天都属于 November 2008），也认得 “April 14, 2002”。

    数字只有紧挨着月份（或者和紧挨着月份的数字用逗号、and、– 连着）才当作日子，
    这样 “The 2 matches were played on 9 March” 里的 2 不会被当成 3 月 2 日。
    没写年份时：给了 season（赛季开始年份）就按 year_from_season 推，否则不算。"""
    tokens = re.findall(r"\b(?:19|20)\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)?\b(?!:)|[A-Za-z]+|[,&–\-]", text)
    found: list[list] = []      # 已经确定月份的日子，每个是 [日, 月, 年]，年等后面出现了再补上
    run: list[list] = []        # 还没看到月份的、连着写的几天（“16, 17 and 18” 等后面的 November）
    # state：上一个有意义的词是什么
    #   ""        其他词（数字不会被当成日子）
    #   "day"     一串还没有月份的日子，后面可以接连接词、更多日子或月份
    #   "dlink"   日子后面的连接词（“16,” “17 and”）
    #   "month"   刚看到月份。如果前面没有日子串，后面紧跟的数字是“月 日”写法（April 14）
    #   "mday"    “April 14” 里的 14，后面可以接 “and 15”
    #   "mlink"   “April 14 and” 里的 and
    state, mdy_month = "", None
    for tok in tokens:
        low = tok.lower()
        if re.fullmatch(r"(?:19|20)\d{2}", tok):
            for f in found:
                if f[2] is None:
                    f[2] = int(tok)
            run, state = [], ""
        elif re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", tok):
            n = int(re.sub(r"\D", "", tok))
            if not 1 <= n <= 31:
                run, state = [], ""
            elif state in ("month", "mlink") and mdy_month:
                found.append([n, mdy_month, None])          # “April 14”、“April 14 and 15”
                state = "mday"
            else:
                if state not in ("day", "dlink"):
                    run = []
                run.append([n, None, None])
                state = "day"
        elif low in MONTHS:
            if run and state in ("day", "dlink"):           # “16, 17 and 18 November”
                for f in run:
                    f[1] = MONTHS[low]
                found.extend(run)
                mdy_month = None                             # 月份已经用在前面的日子上了，后面的数字不再归它
            else:
                mdy_month = MONTHS[low]
            run, state = [], "month"
        elif low in _DAY_LINKS:
            if state in ("day", "dlink"):
                state = "dlink"
            elif state in ("mday", "mlink"):
                state = "mlink"
            elif state == "month" and tok == ",":
                pass                                         # “November, 2001” 这种逗号不打断
            else:
                run, state = [], ""
        else:
            run, state, mdy_month = [], "", None
    dates = []
    for d, mo, y in found:
        y = y or year_from_season(mo, season)
        if mo and y:
            try:
                dates.append(dt.date(y, mo, d))
            except ValueError:
                pass
    return min(dates) if dates else None


_PLAYED = re.compile(r"\b(played|take place|took place|were held|scheduled|weekend|week commencing|"
                     r"week beginning|commencing|kick(ed)? off|staged)\b", re.I)


def prose_round_date(text: str, season: int | None = None) -> dt.date | None:
    """从一段说明文字里找“这一轮比赛在哪天踢”。只说抽签（draw）的句子不算。没写年份的按赛季推（见 year_from_season）。"""
    for sentence in re.split(r"(?<=[a-z0-9\]\)])\.\s+(?=[A-Z])", text):
        if not _PLAYED.search(sentence):
            continue
        # “The draw was held on 26 October ... and the matches were played on ...”：只看抽签后面的部分
        parts = re.split(r"\b(?:and|with)\s+(?:the\s+)?(?:matches|ties|games|fixtures)\b", sentence, flags=re.I)
        part = parts[-1] if len(parts) > 1 else sentence
        if re.search(r"\bdraw\b", part, re.I) and not re.search(r"\bplayed\b", part, re.I):
            continue
        d = first_date_in(part, season)
        if d:
            return d
    return None


# ---------------------------------------------------------------- 读一个网页

# 这些区域里的东西不是比赛（导航框、参考文献、信息框、目录）
SKIP_AREAS = ".navbox, .reflist, .references, .mw-references-wrap, .infobox, #toc, .toc, .sidebar, .vertical-navbox, .hatnote"


@dataclass
class PageResult:
    matches: list[RawMatch] = field(default_factory=list)
    calendar: dict[str, dt.date] = field(default_factory=dict)     # 轮次代码 → 这一轮的日期（来自赛程表）
    problems: list[str] = field(default_factory=list)


def parse_page(html: bytes | str, competition: str, season: int, page: str, raw_file: str,
               main_title: str, single_match: bool = False) -> PageResult:
    """读一个维基百科网页，找出所有比赛。轮次在这里就判断好；日期没有的先空着，后面统一补。

    single_match=True 表示这个赛事只有一场比赛（超级杯），认不出轮次时就当决赛。
    """
    soup = BeautifulSoup(html, "lxml")
    body = _find_class(soup, "mw-parser-output", tag="div") or soup
    page_title = page.replace("_", " ")
    result = PageResult()
    handled: set[int] = {id(x) for x in body.select(SKIP_AREAS)}
    full_text = clean_text(body)
    dagger_aet = bool(re.search(r"†\s*[–—\-:=]?\s*(?:after extra time|a\.e\.t|aet)", full_text, re.I))
    headings: list[str] = []
    section_dates: dict[tuple, dt.date] = {}
    pending: list[tuple[dict, list[str], str]] = []    # (比赛内容, 当时的小标题, 格式)

    for el in body.find_all(["h2", "h3", "h4", "h5", "h6", "p", "div", "table"]):
        if any(id(p) in handled for p in el.parents) or id(el) in handled:
            continue
        classes = el.get("class") or []
        if el.name[0] == "h" and el.name[1:].isdigit():
            level = int(el.name[1:])
            text = clean_text(el)
            prev = headings[level - 2] if len(headings) > level - 2 else ""
            headings = headings[:max(level - 2, 0)]
            while len(headings) < level - 2:
                headings.append("")
            # 同一级里，“Summary”“Matches”这种不说明轮次的小标题跟在 “First qualifying round” 后面时，
            # 其实还是那一轮的内容（网页作者把它们写成了同级标题），所以接在后面，不把轮次名冲掉
            if prev and GENERIC_HEADINGS.match(text.strip()) and not GENERIC_HEADINGS.match(prev.split(" / ")[0]):
                text = f"{prev.split(' / ')[0]} / {text}"
            headings.append(text)
            continue
        if el.name == "p":
            d = prose_round_date(clean_text(el), None if single_match else season)
            if d and tuple(headings) not in section_dates:
                section_dates[tuple(headings)] = d
            continue
        found: list[dict] = []
        fmt = ""
        if el.name == "div" and "footballbox" in classes:
            one = parse_footballbox(el, None if single_match else season)
            found, fmt = ([one] if one else []), "footballbox"
        elif el.name == "table" and "tmpl-football-box-collapsible" in classes:
            one = parse_collapsible(el)
            found, fmt = ([one] if one else []), "collapsible"
        elif el.name == "table" and "wikitable" in classes:
            cols = match_table_columns(el)
            if cols and cols["kind"] == "series":
                fmt = "series"
                for cells in _expand_rows(el)[1:]:
                    if len(cells) > cols["away"]:
                        found.extend(parse_series_row(cells, cols))
            elif cols:
                fmt = "table"
                found = parse_match_table(el, cols, dagger_aet, None if single_match else season)
            else:
                _read_calendar(el, result.calendar)
                continue
        else:
            continue
        handled.add(id(el))
        for m in found:
            pending.append((m, list(headings), fmt))

    for order, (m, heads, fmt) in enumerate(pending):
        rnd = classify_round(heads, page_title, main_title)
        if rnd is None and single_match:
            rnd = "F"
        if rnd is None:
            result.problems.append(f"{page}：认不出轮次（小标题 {' > '.join(h for h in heads if h) or '无'}）"
                                   f" {m['home'].shown} – {m['away'].shown}")
            continue
        leg_hint = m.get("leg_hint", 0)
        if not leg_hint and heads:
            last = heads[-1].lower()
            leg_hint = 1 if re.search(r"first leg|1st leg", last) else 2 if re.search(r"second leg|2nd leg", last) else 0
        replay = m.get("replay_hint", False) or bool(re.search(r"\breplays?\b", heads[-1].lower() if heads else ""))
        day, precision = m["date"], "day"
        if day is None:
            precision = "round"
            for k in range(len(heads), 0, -1):
                if tuple(heads[:k]) in section_dates:
                    day = section_dates[tuple(heads[:k])]
                    break
        s: Score = m["score"]
        ft = m["ft"] or (None, None)
        result.matches.append(RawMatch(
            competition=competition, season=season, page=page, raw_file=raw_file, headings=heads,
            home=m["home"], away=m["away"], date=day, date_precision=precision,
            kickoff_local=m["kickoff_local"], kickoff_utc=m["kickoff_utc"], score=s,
            ft_home=ft[0], ft_away=ft[1], fmt=fmt, leg_hint=leg_hint, replay_hint=replay, order=order,
            round=rnd))
    return result


def _read_calendar(table: Tag, calendar: dict[str, dt.date]) -> None:
    """赛程表（Round | Main date、Date | Event、Round | Draw date | First leg | Second leg）：记下每一轮的第一天。"""
    head = _header_cells(table)
    if not head:
        return
    round_col = next((i for i, h in enumerate(head) if re.search(r"^(round|stage|phase|event)", h)), None)
    date_cols = [i for i, h in enumerate(head) if re.search(r"date|first leg|1st leg|match", h) and "draw" not in h]
    if round_col is None or not date_cols:
        return
    for cells in _expand_rows(table)[1:]:
        if len(cells) <= max([round_col, *date_cols]):
            continue
        code = classify_round([clean_text(cells[round_col])], "", "")
        if not code:
            continue
        dates = [first_date_in(clean_text(cells[i])) for i in date_cols]
        dates = [d for d in dates if d]
        if dates and code not in calendar:
            calendar[code] = min(dates)


# ---------------------------------------------------------------- 读一个赛季的所有网页

def season_pages(folder: Path) -> list[tuple[str, Path]]:
    """一个赛季文件夹里的网页：(页面名, 文件)。

    已完赛赛季每个页面一个 .html.gz；当前赛季每个页面一个子文件夹，里面按日期存快照，取最新一份。
    .suspicious（内容可疑）的不用。
    """
    out = []
    if not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name.endswith(".html.gz") and not p.name.endswith(".suspicious.html.gz"):
            out.append((p.name[:-len(".html.gz")], p))
        elif p.is_dir():
            snaps = sorted(x for x in p.glob("*.html.gz") if not x.name.endswith(".suspicious.html.gz"))
            if snaps:
                out.append((p.name, snaps[-1]))
    return out


def _canonical(html: str) -> str | None:
    m = re.search(r'<link rel="canonical" href="https://en\.wikipedia\.org/wiki/([^"]+)"', html)
    if not m:
        return None
    import urllib.parse
    return urllib.parse.unquote(m.group(1)).replace("_", " ")


def skip_page(stem: str, main_titles: list[str], skip_rules: list[str]) -> bool:
    """这个页面要不要跳过（女子赛事、低级别赛事、法国杯业余轮次等，规则在 settings.yaml）。

    子页面按“主页面标题后面的部分”比对（和下载时一样）；规则也拿整个标题再比一次（import_skip_pages 是按整个标题写的）。
    """
    title = stem.replace("_", " ").lower()
    for main in main_titles:
        if title.startswith(main.lower() + " "):
            rest = title[len(main) + 1:]
            if any(re.search(r, rest) for r in skip_rules):
                return True
    return any(re.search(r, title) for r in skip_rules if " " in r)


def load_season(raw_root: Path, code: str, comp: dict, year: int, main_titles: list[str],
                problems: list[str], skip_rules: list[str] = ()) -> list[RawMatch]:
    """读一个赛事一个赛季的所有网页，返回比赛列表（同一场比赛在几个页面出现时，这里还没去重）。

    main_titles：这个赛季主页面可能的标题（settings 里的模板生成的），用来判断子页面名字里的轮次。
    同一个页面被重定向保存了两份（例如 “knockout phase” 和 “knockout stage”）只读一份。
    """
    folder = raw_root / "wikipedia" / code / str(year)
    stems = {t.replace(" ", "_"): t for t in main_titles}
    pages = season_pages(folder)
    texts: dict[str, tuple[str, Path, str]] = {}   # 标准标题 → (页面名, 文件, html)
    main_title = None
    for stem, path in pages:
        if skip_page(stem, main_titles, skip_rules):
            continue
        html = gzip.decompress(path.read_bytes()).decode("utf-8")
        canon = _canonical(html) or stem.replace("_", " ")
        if stem in stems:
            main_title = canon
        if canon in texts:
            continue
        texts[canon] = (stem, path, html)
    if not texts:
        return []
    if main_title is None:
        main_title = min(texts, key=len)
    out: list[RawMatch] = []
    calendar: dict[str, dt.date] = {}
    results = []
    single = bool(comp.get("单场"))
    for canon, (stem, path, html) in texts.items():
        rel = path.relative_to(raw_root.parent).as_posix() if raw_root.parent in path.parents else path.as_posix()
        res = parse_page(html, code, year, canon.replace(" ", "_"), rel, main_title, single_match=single)
        results.append(res)
        for k, v in res.calendar.items():
            calendar.setdefault(k, v)
        problems.extend(res.problems)
    for res in results:
        for m in res.matches:
            if m.date is None and m.round in calendar:
                m.date = calendar[m.round]       # 这一轮的日期（来自赛程表），精度还是 round
            out.append(m)
    return out



# ---------------------------------------------------------------- 球队对照

FOREIGN_PREFIX = "wiki:"     # 不在 teams.yaml 里的球队：编号 = “wiki:维基百科条目名”，不会和我们的球队撞名

# 国旗上的国家（英文）→ teams.yaml 里的国家写法。只有这几个国家有我们的球队
FLAG_COUNTRY = {"England": "英格兰", "Wales": "英格兰", "Germany": "德国", "Italy": "意大利",
                "Spain": "西班牙", "France": "法国", "Monaco": "法国",
                # 有的网页国旗图标是足协的标志，说明文字是足协名
                "The Football Association": "英格兰", "Football Association of Wales": "英格兰",
                "German Football Association": "德国", "Italian Football Federation": "意大利",
                "Royal Spanish Football Federation": "西班牙", "French Football Federation": "法国"}


@dataclass
class TeamMap:
    """维基百科条目名 → 我们的球队编号。内容来自 config/team_aliases_wikipedia.yaml。"""
    alias: dict[str, str]            # 条目名 → teams.yaml 的“英文”
    not_ours: dict[str, str]         # 看着像、但确认不是同一支球队的条目名 → 说明
    teams: dict[str, dict]           # teams.yaml 的“英文” → 那一条

    def resolve(self, ref: TeamRef, comp_country: str = "") -> str:
        """返回球队编号：我们的球队用 teams.yaml 的“英文”，其他球队用 wiki:条目名。

        还要核对国家，防止同名的外国球队被当成我们的（例如没有链接、只写了 “Arsenal” 的
        可能是俄罗斯的图拉兵工厂）：有国旗就看国旗；没有国旗就看赛事所在国家（欧战不知道，不核对）。
        """
        team = self.alias.get(ref.title)
        if team is None:
            return FOREIGN_PREFIX + ref.title
        want = self.teams[team]["国家"]
        if ref.country:
            got = FLAG_COUNTRY.get(ref.country)
        else:
            got = comp_country if comp_country and comp_country != "欧洲" else want
        return team if got == want else FOREIGN_PREFIX + ref.title

    def is_ours(self, team_id: str) -> bool:
        return not team_id.startswith(FOREIGN_PREFIX)


def load_team_map(settings: dict) -> TeamMap:
    """读 config/team_aliases_wikipedia.yaml，并检查里面的球队都在 teams.yaml 里、同一个条目名没写两次。"""
    teams, _ = load_teams(settings)
    by_name = {t["英文"]: t for t in teams}
    path = resolve_path(settings["paths"]["team_aliases_wikipedia"])
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    alias: dict[str, str] = {}
    for team, titles in (data.get("对照") or {}).items():
        if team not in by_name:
            raise ValueError(f"{path.name}：{team} 不在 teams.yaml 里")
        for t in ([titles] if isinstance(titles, str) else titles):
            if t in alias and alias[t] != team:
                raise ValueError(f"{path.name}：条目名 {t} 同时对应 {alias[t]} 和 {team}")
            alias[t] = team
    not_ours = {k: str(v) for k, v in (data.get("不是我们的球队") or {}).items()}
    both = set(alias) & set(not_ours)
    if both:
        raise ValueError(f"{path.name}：{sorted(both)} 既写在“对照”里又写在“不是我们的球队”里")
    return TeamMap(alias=alias, not_ours=not_ours, teams=by_name)


def normalize_name(name: str) -> str:
    """把队名变成方便比较的样子：去掉重音、标点和 F.C.、AC、SC 这类前后缀。只用来给人工对照找候选，不直接用来对照。"""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"[.'’&\-]", " ", n)
    words = [w for w in n.split() if w not in {
        "fc", "f", "c", "afc", "cf", "ac", "as", "ss", "us", "sc", "sv", "vfb", "vfl", "tsv", "fsv", "1", "club",
        "football", "calcio", "de", "the", "rcd", "cd", "ud", "sd", "rc", "real", "sporting", "olympique", "stade",
        "en", "athletic", "association", "deportivo", "and", "hove", "albion",
        "ssc", "acf", "bsc", "spvgg", "tsg", "sk", "fk", "kv", "us", "f c", "1899", "1846", "04", "05", "09"}]
    return " ".join(words)


# ---------------------------------------------------------------- 去重、分回合

FORMAT_RANK = {"footballbox": 0, "collapsible": 1, "table": 2, "series": 3}   # 同一场比赛几种写法都有时，信哪个
GROUP_ROUNDS = {"G", "G1", "G2", "LP"}   # 小组赛 / 联赛阶段：同一对球队主客各一场，都是单场（leg 0）


def season_by_date(day: dt.date | None, default: int) -> int:
    """超级杯这种一年一场的比赛按日期算赛季：7 月起算新赛季（2021 年 1 月踢的算 2020/21）。"""
    if day is None:
        return default
    return day.year if day.month >= 7 else day.year - 1


def _score_of(m: RawMatch) -> tuple:
    return (m.score.home, m.score.away)


def merge_duplicates(raws: list[RawMatch], ids: dict[int, tuple[str, str]], problems: list[str]) -> list[RawMatch]:
    """同一场比赛在几个页面、几种写法里出现，只留一份。

    ids：每场比赛（按 id(比赛)）对应的 (主队编号, 客队编号)。
    1. 有具体日期的：同一轮、同一天、同一主队 = 同一场，留写法最可靠的那份（footballbox 最好）。
    2. 没有具体日期的（汇总表、表格）：同一轮里已经有同一主队的比赛（比分一样，或者都是/都不是重赛）就算重复。
    """
    groups: dict[tuple, list[RawMatch]] = collections.defaultdict(list)
    for m in raws:
        h, a = ids[id(m)]
        groups[(m.round, frozenset((h, a)))].append(m)
    out: list[RawMatch] = []
    for (_rnd, _pair), ms in groups.items():
        ms.sort(key=lambda m: (FORMAT_RANK[m.fmt], m.order))
        kept: list[RawMatch] = []
        for m in ms:
            h = ids[id(m)][0]
            exact = m.date_precision == "day" and m.date is not None
            same = None
            for k in kept:
                if ids[id(k)][0] != h:
                    continue
                k_exact = k.date_precision == "day" and k.date is not None
                if exact and k_exact:
                    if k.date == m.date:
                        same = k
                elif _score_of(k) == _score_of(m) or k.replay_hint == m.replay_hint:
                    same = k
                if same:
                    break
            if same is None:
                kept.append(m)
                continue
            # 重复的那份：比分不一样要记下来；另一份有而这份没有的信息补上
            if _score_of(same) != _score_of(m) and None not in _score_of(m) and m.fmt != "series":
                problems.append(f"同一场比赛两处比分不一样：{same.page} {_score_of(same)} / {m.page} {_score_of(m)} "
                                f"{m.home.shown}–{m.away.shown} {m.date}")
            if same.kickoff_utc is None and m.kickoff_utc and same.date == m.date:
                same.kickoff_utc = m.kickoff_utc
            if same.ft_home is None and m.ft_home is not None and _score_of(same) == _score_of(m):
                same.ft_home, same.ft_away = m.ft_home, m.ft_away
            if same.score.pen_home is None and m.score.pen_home is not None:
                same.score.pen_home, same.score.pen_away = m.score.pen_home, m.score.pen_away
            if not same.leg_hint and m.leg_hint:
                same.leg_hint = m.leg_hint
            same.replay_hint = same.replay_hint or m.replay_hint
        out.extend(kept)
    return out


def assign_legs(ms: list[RawMatch], ids: dict[int, tuple[str, str]], replay_comp: bool,
                problems: list[str]) -> list[tuple[RawMatch, str, int]]:
    """给每场比赛定 (轮次, 回合)。

    - 小组赛、联赛阶段：都是 leg 0（主客场的两场靠主队不同区分）。
    - 有重赛的赛事（英足总杯）：同一轮同一对球队的第一场 leg 0，后面的是重赛，轮次加 _replay（第二次重赛 _replay2）。
    - 其他淘汰赛：同一轮同一对球队只有一场 → leg 0（单场）；两场 → 按日期先后 leg 1、2（两回合）。
    """
    groups: dict[tuple, list[RawMatch]] = collections.defaultdict(list)
    for m in ms:
        groups[(m.round, frozenset(ids[id(m)]))].append(m)
    out = []
    far = dt.date(2999, 1, 1)
    for (rnd, _pair), g in groups.items():
        g.sort(key=lambda m: (m.date or far, m.replay_hint, m.leg_hint, m.order))
        if rnd in GROUP_ROUNDS:
            homes = collections.Counter(ids[id(m)][0] for m in g)
            if max(homes.values()) > 1:
                problems.append(f"小组赛同一对主客队出现两次：{g[0].page} {g[0].home.shown}–{g[0].away.shown}")
            out += [(m, rnd, 0) for m in g]
        elif replay_comp:
            for i, m in enumerate(g):
                out.append((m, rnd if i == 0 else rnd + ("_replay" if i == 1 else f"_replay{i}"), 0))
        elif len(g) == 1:
            out.append((g[0], rnd, 0))
        else:
            if len(g) == 2 and g[0].date == g[1].date and g[0].leg_hint > g[1].leg_hint:
                g.reverse()
            if len(g) > 2:
                problems.append(f"同一轮同一对球队有 {len(g)} 场：{g[0].page} {g[0].home.shown}–{g[0].away.shown}")
            out += [(m, rnd, i + 1) for i, m in enumerate(g)]
    return out


# ---------------------------------------------------------------- 整理成要写进数据库的行

@dataclass
class Row:
    """要写进数据库的一场比赛。"""
    competition: str
    season: int
    round: str
    leg: int
    home_team: str
    away_team: str
    match_date: str
    date_precision: str
    kickoff_utc: str | None
    status: str
    home_goals: int | None      # 90 分钟比分
    away_goals: int | None
    result: str | None          # H / D / A（90 分钟）
    extra_time: int
    aet_home: int | None
    aet_away: int | None
    pen_home: int | None
    pen_away: int | None
    note: str
    raw_file: str
    page: str
    home_title: str
    away_title: str
    home_country: str
    away_country: str

    @property
    def source_key(self) -> str:
        """这场比赛在维基百科来源里的编号：赛事|赛季|轮次|回合|主队|客队（同一场只有一个）。"""
        return "|".join([self.competition, str(self.season), self.round, str(self.leg), self.home_team, self.away_team])


def regular_time(m: RawMatch, leg: int) -> tuple[int | None, int | None, str | None, str]:
    """90 分钟比分和赛果。返回 (主队进球, 客队进球, 赛果, 说明)。

    - 没踢加时：就是最后比分。
    - 踢了加时、网页上有进球时间：数 90 分钟以内的进球（解析时已经算好，在 ft_home/ft_away）。
    - 金球：加时赛只进了一个球（进了比赛就结束），所以赢的一方减一球就是 90 分钟比分。
    - 加时后 0-0：90 分钟也是 0-0。加时后别的平局（如 2-2）推不出 90 分钟比分。
    - 单场比赛踢了加时 = 90 分钟一定是平局，但具体几比几不知道：比分留空，赛果记 D。
    - 两回合的第 2 回合踢了加时：90 分钟是不是平局要看总比分，这里不推算，比分和赛果都留空。
    """
    s = m.score
    if s.home is None:
        return None, None, None, ""
    if not s.extra_time:
        h, a = s.home, s.away
    elif m.ft_home is not None:
        h, a = m.ft_home, m.ft_away
    elif s.golden_goal and s.home != s.away:
        h, a = (s.home - 1, s.away) if s.home > s.away else (s.home, s.away - 1)
    elif s.home == s.away == 0:
        h, a = 0, 0                 # 加时后 0-0：整场一个球都没进，90 分钟一定也是 0-0
    # 注意：加时后 2-2 不能推出 90 分钟 2-2（可能 1-1 后加时各进一球），只能走下面的“不知道”
    else:
        if leg == 0:
            return None, None, "D", "加时赛：90 分钟是平局，比分不知道"
        return None, None, None, "加时赛：90 分钟比分不知道"
    return h, a, ("H" if h > a else "A" if h < a else "D"), ""


def _same_name(a: str, b: str) -> bool:
    """两个队名是不是同一支球队的不同写法：规范化以后一样，或者一个的词全包含在另一个里（Porto / FC Porto）。"""
    wa, wb = set(normalize_name(a).split()), set(normalize_name(b).split())
    if not wa or not wb:
        return False
    return wa <= wb or wb <= wa


def unify_foreign_ids(matches: list[RawMatch], ids: dict[int, tuple[str, str]], team_map: TeamMap,
                      problems: list[str]) -> dict[int, tuple[str, str]]:
    """同一支外国球队在同一页上用了两个条目名（一个是正式条目，一个是“重定向”的短名字）时，并成一个编号。

    例子：2009/10 欧冠 16 强，汇总表里写 “FC Porto”，第二回合的比赛框里写 “Porto”（链接是重定向）。
    不并的话会变成两支球队，去重和分第一、二回合都会出错。
    只在很有把握时才并，下面几条都要满足：
      1. 同一赛季、同一轮、对手是同一支球队，两个名字一个包含另一个（规范化以后）；
      2. 两边都有国旗时，国旗一样；
      3. 和这个对手、同样的主客方向，两个名字没有在“不同的具体日期”各踢过一场
         （真是两支队的话，例如小组赛里同一个对手分别碰到 “Milan” 和 “Inter Milan”，日期会不一样）。
    并的时候留出现次数多的名字（一样多就留长的，一般是正式条目名）；我们的球队（teams.yaml 里的）永远不被改。
    """
    rivals: dict[tuple, set[str]] = collections.defaultdict(set)   # (赛事, 赛季, 轮次, 对手) → 和它踢过的球队编号
    days: dict[tuple, set] = collections.defaultdict(set)          # (赛事, 赛季, 轮次, 对手, 对手是不是主队, 球队) → 具体日期
    count: collections.Counter = collections.Counter()
    flags: dict[str, set[str]] = collections.defaultdict(set)     # 编号 → 国旗上的国家（两边国旗都有又不一样，就不并）
    for m in matches:
        h, a = ids[id(m)]
        count[(m.season, h)] += 1
        count[(m.season, a)] += 1
        for t, ref in ((h, m.home), (a, m.away)):
            if ref.country:
                flags[t].add(ref.country)
        rivals[(m.competition, m.season, m.round, a)].add(h)
        rivals[(m.competition, m.season, m.round, h)].add(a)
        if m.date is not None and m.date_precision == "day":
            days[(m.competition, m.season, m.round, a, False, h)].add(m.date)
            days[(m.competition, m.season, m.round, h, True, a)].add(m.date)
    rename: dict[tuple, str] = {}    # (赛季, 旧编号) → 新编号
    for (code, season, rnd, opp), group in rivals.items():
        foreign = sorted(t for t in group if not team_map.is_ours(t))
        for x in foreign:
            for y in group:
                if x == y or (season, y) in rename or (season, x) in rename:
                    continue
                if not _same_name(x[len(FOREIGN_PREFIX):], y[len(FOREIGN_PREFIX):] if not team_map.is_ours(y) else y):
                    continue
                if team_map.is_ours(y):
                    continue        # 外国条目名像我们的球队：不自动并（可能是 B 队），留给对照表人工处理
                if flags[x] and flags[y] and not (flags[x] & flags[y]):
                    continue
                if any(days[(code, season, rnd, opp, side, x)] and days[(code, season, rnd, opp, side, y)]
                       and days[(code, season, rnd, opp, side, x)] != days[(code, season, rnd, opp, side, y)]
                       for side in (True, False)):
                    continue
                keep, drop = sorted((x, y), key=lambda t: (count[(season, t)], len(t)), reverse=True)
                rename[(season, drop)] = keep
                problems.append(f"同一支球队两个条目名，已并成一个：{code} {season} {drop} → {keep}")
    if not rename:
        return ids
    out = {}
    seasons = {id(m): m.season for m in matches}
    for k, (h, a) in ids.items():
        s = seasons.get(k)
        out[k] = (rename.get((s, h), h), rename.get((s, a), a))
    return out


def build_rows(matches: list[RawMatch], team_map: TeamMap, comps: dict, today: dt.date,
               problems: list[str], stats: collections.Counter) -> list[Row]:
    """把一个赛事的全部比赛（几个赛季）整理成数据库的行：换球队编号 → 去重 → 分回合 → 只留有我们球队的。"""
    ids = {id(m): (team_map.resolve(m.home, comps[m.competition].get("国家", "")),
                   team_map.resolve(m.away, comps[m.competition].get("国家", ""))) for m in matches}
    ids = unify_foreign_ids(matches, ids, team_map, problems)
    rows: list[Row] = []
    by_season: dict[tuple, list[RawMatch]] = collections.defaultdict(list)
    for m in matches:
        by_season[(m.competition, m.season)].append(m)
    for (code, season), ms in sorted(by_season.items()):
        comp = comps[code]
        ms = merge_duplicates(ms, ids, problems)
        for m, rnd, leg in assign_legs(ms, ids, bool(comp.get("重赛")), problems):
            home, away = ids[id(m)]
            stats[(code, season, "parsed")] += 1
            if not (team_map.is_ours(home) or team_map.is_ours(away)):
                continue
            s = m.score
            if s.special in ("walkover", "void") or (s.special == "postponed" and s.home is None):
                stats[(code, season, "skipped_special")] += 1
                problems.append(f"没踢或作废（不导入）：{m.page} {m.home.shown}–{m.away.shown} {s.special}")
                continue
            if m.date is None:
                stats[(code, season, "no_date")] += 1
                problems.append(f"找不到日期（不导入）：{m.page} {rnd} {m.home.shown}–{m.away.shown}")
                continue
            if m.date_precision == "round" and "_replay" in rnd:
                stats[(code, season, "replay_no_date")] += 1
                problems.append(f"重赛没有具体日期（不导入，不拿这一轮的日期冒充）：{m.page} {m.home.shown}–{m.away.shown}")
                continue
            season_out = season_by_date(m.date, season) if comp.get("单场") else season
            if s.home is None:
                if m.date >= today:
                    status = "scheduled"
                else:
                    stats[(code, season, "no_score")] += 1
                    problems.append(f"已经过了比赛日期但没有比分（不导入）：{m.page} {m.home.shown}–{m.away.shown} {m.date}")
                    continue
            else:
                status = "awarded" if s.special == "awarded" else "finished"
            h, a, res, note = regular_time(m, leg)
            if status == "awarded":
                h = a = res = None
                note = f"判罚比分 {s.home}–{s.away}"
            notes = [x for x in (note, "golden goal 金球" if s.golden_goal else "",
                                 "silver goal 银球" if s.silver_goal else "") if x]
            rows.append(Row(
                competition=code, season=season_out, round=rnd, leg=leg, home_team=home, away_team=away,
                match_date=m.date.isoformat(), date_precision=m.date_precision, kickoff_utc=m.kickoff_utc,
                status=status, home_goals=h, away_goals=a, result=res, extra_time=int(s.extra_time),
                aet_home=s.home if s.extra_time else None, aet_away=s.away if s.extra_time else None,
                pen_home=s.pen_home, pen_away=s.pen_away, note="；".join(notes), raw_file=m.raw_file,
                page=m.page, home_title=m.home.title, away_title=m.away.title,
                home_country=m.home.country, away_country=m.away.country))
            stats[(code, season_out, "stored")] += 1
            stats[(code, season_out, f"stored_{m.date_precision}")] += 1
    # 同一场比赛（唯一键一样）出现两次：说明去重或分回合有问题，留第一场并记下来
    seen: dict[tuple, Row] = {}
    out = []
    for r in rows:
        k = (r.competition, r.season, r.home_team, r.away_team, r.round, r.leg)
        if k in seen:
            problems.append(f"唯一键重复（只留第一场）：{k} {seen[k].match_date} / {r.match_date}")
            continue
        seen[k] = r
        out.append(r)
    return out


# ---------------------------------------------------------------- 写数据库

FLAG_TO_ZH = {"England": "英格兰", "Wales": "威尔士", "Scotland": "苏格兰", "Northern Ireland": "北爱尔兰",
              "Germany": "德国", "Italy": "意大利", "Spain": "西班牙", "France": "法国", "Monaco": "摩纳哥",
              "Netherlands": "荷兰", "Portugal": "葡萄牙", "Belgium": "比利时", "Turkey": "土耳其",
              "Russia": "俄罗斯", "Ukraine": "乌克兰", "Greece": "希腊", "Austria": "奥地利",
              "Switzerland": "瑞士"}


def write_database(con: sqlite3.Connection, rows: list[Row], team_map: TeamMap, comps: dict,
                   fetched: dict[str, str], codes: list[str] | None = None) -> dict:
    """在一个事务里写入维基百科的比赛。中途出错就全部不算，数据库保持导入前的样子。

    codes：这次导入的赛事代码（只导一部分赛事时用）。只有这些赛事的旧行会被删掉重写，其他赛事不动。

    - 比赛表：有就更新（编号不变），没有才新增。联赛的比赛（round=''、leg=0）不会被碰到。
    - 来源、赛果、加时点球表：只删 source='wikipedia' 的行再重写，football-data 的行一行都不动。
    - 最后删掉“任何来源都不再提到”的比赛（和 football-data 导入的规则一样），以及没有比赛用到的外来球队。
    """
    create_tables(con)
    teams, _ = load_teams_for_db(team_map)
    foreign: dict[str, tuple[str, str]] = {}
    for r in rows:
        for tid, title, flag in ((r.home_team, r.home_title, r.home_country), (r.away_team, r.away_title, r.away_country)):
            if not team_map.is_ours(tid) and tid not in foreign:
                # 没有国旗时用赛事所在国家（国内杯赛的对手一般是本国球队）；欧战没有国旗就记“未知”，不记成“欧洲”
                comp_country = comps[r.competition].get("国家", "")
                country = FLAG_TO_ZH.get(flag, flag) or (comp_country if comp_country != "欧洲" else "")
                foreign[tid] = (title, country or "未知")
    with con:
        con.executemany("insert into teams (team_id, name_zh, country) values (?, ?, ?) "
                        "on conflict(team_id) do nothing", teams)
        con.executemany("insert into teams (team_id, name_zh, country) values (?, null, ?) "
                        "on conflict(team_id) do update set country = excluded.country",
                        [(tid, c) for tid, (_t, c) in foreign.items()])
        if codes is None:   # 全部重导：先清空维基百科的名字表；只导一部分时只增改不删
            con.execute("delete from team_names where source = ?", (SOURCE,))
        names = {}
        for r in rows:
            names.setdefault(r.home_title, r.home_team)
            names.setdefault(r.away_title, r.away_team)
        names.update(team_map.alias)
        con.executemany("insert or replace into team_names (source, name, team_id) values (?, ?, ?)",
                        [(SOURCE, n, t) for n, t in names.items() if t in team_map.teams or t in foreign])
        con.executemany(
            "insert into matches (competition, season, home_team, away_team, match_date, kickoff_utc, status, "
            "round, leg, date_precision) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "on conflict(competition, season, home_team, away_team, round, leg) do update set "
            "match_date = excluded.match_date, kickoff_utc = excluded.kickoff_utc, status = excluded.status, "
            "date_precision = excluded.date_precision",
            [(r.competition, r.season, r.home_team, r.away_team, r.match_date, r.kickoff_utc, r.status,
              r.round, r.leg, r.date_precision) for r in rows])
        codes = sorted(codes if codes is not None else comps)
        ph = ",".join("?" * len(codes))
        id_of = {tuple(k): mid for *k, mid in con.execute(
            f"select competition, season, home_team, away_team, round, leg, match_id from matches "
            f"where competition in ({ph}) and round != ''", codes)}
        for table in ("match_sources", "match_stats", "cup_details"):   # 只删这次导入的赛事的维基行
            con.execute(f"delete from {table} where source = ? and match_id in "
                        f"(select match_id from matches where competition in ({ph}))", (SOURCE, *codes))
        src, st, cup = [], [], []
        for r in rows:
            mid = id_of[(r.competition, r.season, r.home_team, r.away_team, r.round, r.leg)]
            src.append((mid, SOURCE, r.source_key, r.raw_file, fetched.get(r.raw_file)))
            if r.status != "scheduled":
                st.append((mid, SOURCE, r.home_goals, r.away_goals, r.result))
                cup.append((mid, SOURCE, r.extra_time, r.aet_home, r.aet_away, r.pen_home, r.pen_away, r.note or None))
        con.executemany("insert into match_sources values (?, ?, ?, ?, ?)", src)
        con.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result) "
                        "values (?, ?, ?, ?, ?)", st)
        con.executemany("insert into cup_details values (?, ?, ?, ?, ?, ?, ?, ?)", cup)
        con.execute("delete from matches where match_id not in (select match_id from match_sources)")
        con.execute(f"delete from teams where team_id like '{FOREIGN_PREFIX}%' and team_id not in "
                    "(select home_team from matches union select away_team from matches)")
        con.execute(f"delete from team_names where team_id like '{FOREIGN_PREFIX}%' and team_id not in "
                    "(select team_id from teams)")
    return {"matches": len(src), "foreign_teams": len(foreign)}


def load_teams_for_db(team_map: TeamMap) -> tuple[list[tuple], None]:
    """teams.yaml 里的球队（数据库里还没有时补上；已经有的不动，football-data 导入时会按 teams.yaml 重写）。"""
    return [(t["英文"], t.get("中文") or None, t["国家"]) for t in team_map.teams.values()], None


# ---------------------------------------------------------------- 全部赛事

def read_all(settings: dict, only: list[str] | None = None) -> tuple[list[RawMatch], list[str]]:
    """读原始层里所有赛事、所有赛季的网页，返回 (比赛列表, 问题清单)。"""
    from src.collect.wikipedia import titles_for   # 页面标题的规则和下载时用同一份
    w = settings["wikipedia"]
    raw_root = resolve_path(settings["paths"]["raw"])
    problems: list[str] = []
    out: list[RawMatch] = []
    for code, comp in w["competitions"].items():
        if only and code not in only:
            continue
        folder = raw_root / "wikipedia" / code
        if not folder.exists():
            continue
        skip = list(w["skip_subpages"]) + list(comp.get("skip_subpages_extra", [])) + list(w.get("import_skip_pages", []))
        for season_dir in sorted(folder.iterdir()):
            if not season_dir.name.isdigit():
                continue
            year = int(season_dir.name)
            out += load_season(raw_root, code, comp, year, titles_for(comp, year), problems, skip)
    return out, problems


def unmapped_candidates(raws: list[RawMatch], team_map: TeamMap, comps: dict, limit: float = 0.72) -> list[dict]:
    """没对上我们球队、但名字很像 teams.yaml 里某支同国球队的条目名（给人工检查，不自动对照）。"""
    import difflib
    seen: collections.Counter = collections.Counter()
    country_of: dict[str, set] = collections.defaultdict(set)
    for m in raws:
        for ref in (m.home, m.away):
            if ref.title in team_map.alias or ref.title in team_map.not_ours:
                continue
            seen[ref.title] += 1
            c = FLAG_COUNTRY.get(ref.country) or comps[m.competition].get("国家")
            if c in {"英格兰", "德国", "意大利", "西班牙", "法国"}:
                country_of[ref.title].add(c)
    by_country: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for t in team_map.teams.values():
        by_country[t["国家"]].append((t["英文"], normalize_name(t["英文"])))
        for a in t.get("别名", []):
            by_country[t["国家"]].append((t["英文"], normalize_name(a)))
    out = []
    for title, n in seen.items():
        norm = normalize_name(title)
        best = []
        for c in country_of.get(title, ()):
            for team, tn in by_country[c]:
                r = difflib.SequenceMatcher(None, norm, tn).ratio()
                if r >= limit or (tn and (tn in norm.split() or norm.startswith(tn + " ") or tn == norm)):
                    best.append((round(r, 2), team))
        if best:
            best.sort(reverse=True)
            out.append({"条目名": title, "出现次数": n, "像": [t for _, t in best[:3]], "相似度": best[0][0]})
    out.sort(key=lambda d: (-d["出现次数"], d["条目名"]))
    return out


def run(settings: dict | None = None, con: sqlite3.Connection | None = None,
        only: list[str] | None = None, today: dt.date | None = None) -> dict:
    """读全部维基百科页面 → 整理 → 写进数据库 → 写报告。返回结果（给报告和检查用）。"""
    settings = settings or load_settings()
    comps = settings["wikipedia"]["competitions"]
    today = today or dt.date.today()
    team_map = load_team_map(settings)
    raws, problems = read_all(settings, only)
    stats: collections.Counter = collections.Counter()
    rows: list[Row] = []
    by_comp: dict[str, list[RawMatch]] = collections.defaultdict(list)
    for m in raws:
        by_comp[m.competition].append(m)
    for code, ms in by_comp.items():
        rows += build_rows(ms, team_map, comps, today, problems, stats)
    con = con or connect()
    create_tables(con)
    fetched = {}
    for path, when in con.execute("select file_path, max(fetched_at_utc) from fetch_log "
                                  "where source = ? and status = 'saved' group by file_path", (SOURCE,)):
        fetched[path] = when
    written = write_database(con, rows, team_map, comps, fetched, codes=list(only) if only else None)
    return {"raws": raws, "rows": rows, "problems": problems, "stats": stats, "written": written,
            "candidates": unmapped_candidates(raws, team_map, comps)}


def problem_kind(p: str) -> str:
    """问题的种类 = 第一个全角冒号前面的文字，例如 “找不到日期（不导入）”。"""
    return p.split("：", 1)[0]


def write_report(result: dict, settings: dict, con: sqlite3.Connection) -> Path:
    """把这次导入的结果写成一份中文报告（data/reports/wikipedia_杯赛导入.md），完整问题清单另存一个文本文件。

    报告从原始层和数据库算出来，能随时重新生成，不进 Git。
    """
    comps = settings["wikipedia"]["competitions"]
    rows: list[Row] = result["rows"]
    stats: collections.Counter = result["stats"]
    out_dir = resolve_path(settings["paths"]["reports"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "wikipedia_杯赛导入.md"
    problems_path = out_dir / "wikipedia_杯赛导入_问题清单.txt"
    problems_path.write_text("\n".join(result["problems"]) + "\n", encoding="utf-8")

    codes = [c for c in comps if any(r.competition == c for r in rows)]
    seasons = sorted({r.season for r in rows})
    stored = collections.Counter((r.competition, r.season) for r in rows)
    parsed = collections.Counter()
    for (code, season, key), n in stats.items():
        if key == "parsed":
            parsed[code] += n
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    L = [f"# 维基百科杯赛、欧战导入报告", "",
         f"生成时间（北京时间）：{now}。这份报告由 `python -m src.clean.wikipedia_import` 生成，能重新生成。", "",
         "## 总数", "",
         f"- 从网页里读出的比赛（去重前）：{len(result['raws'])} 条",
         f"- 去重、分回合以后的比赛：{sum(parsed.values())} 场",
         f"- 写进数据库（至少一方是我们的球队）：{len(rows)} 场；外来球队 {result['written']['foreign_teams']} 支",
         f"- 日期精确到天：{sum(r.date_precision == 'day' for r in rows)} 场；"
         f"只知道是哪一轮（日期是这一轮的日子）：{sum(r.date_precision == 'round' for r in rows)} 场",
         f"- 有开球时间（UTC）：{sum(bool(r.kickoff_utc) for r in rows)} 场",
         f"- 打了加时：{sum(r.extra_time for r in rows)} 场；其中 90 分钟比分不知道："
         f"{sum(r.extra_time and r.home_goals is None and r.status == 'finished' for r in rows)} 场；"
         f"有点球大战：{sum(r.pen_home is not None for r in rows)} 场",
         f"- 还没踢（scheduled）：{sum(r.status == 'scheduled' for r in rows)} 场；"
         f"判罚比分（awarded）：{sum(r.status == 'awarded' for r in rows)} 场",
         f"- 问题：{len(result['problems'])} 条（完整清单见 `{problems_path.name}`）", ""]

    L += ["## 每个赛事、每个赛季写进数据库的场数", "",
          "赛季用开始年份（2008 = 2008/09）。空格表示这一季没有我们的球队的比赛，或者没有网页。", ""]
    L.append("| 赛季 | " + " | ".join(codes) + " |")
    L.append("|---" * (len(codes) + 1) + "|")
    for s in seasons:
        L.append(f"| {s} | " + " | ".join(str(stored[(c, s)] or "") for c in codes) + " |")
    L.append("| 合计 | " + " | ".join(str(sum(stored[(c, s)] for s in seasons)) for c in codes) + " |")
    L.append("")
    L += ["赛事代码：" + "；".join(f"{c} = {comps[c].get('名称', c)}" for c in codes), ""]

    L += ["## 每个赛事的细节", "",
          "| 赛事 | 读出（去重后） | 写进数据库 | 精确到天 | 只知道轮次 | 有开球时间 | 加时 | 点球 |",
          "|---|---|---|---|---|---|---|---|"]
    for c in codes:
        rs = [r for r in rows if r.competition == c]
        L.append(f"| {c} | {parsed[c]} | {len(rs)} | {sum(r.date_precision == 'day' for r in rs)} | "
                 f"{sum(r.date_precision == 'round' for r in rs)} | {sum(bool(r.kickoff_utc) for r in rs)} | "
                 f"{sum(r.extra_time for r in rs)} | {sum(r.pen_home is not None for r in rs)} |")
    L.append("")

    kinds = collections.Counter(problem_kind(p) for p in result["problems"])
    L += ["## 问题（按种类）", "", "| 种类 | 条数 | 例子 |", "|---|---|---|"]
    for k, n in kinds.most_common():
        example = next(p for p in result["problems"] if problem_kind(p) == k)
        L.append(f"| {k} | {n} | {example.split('：', 1)[-1][:120].replace('|', '/')} |")
    L.append("")

    cands = result["candidates"]
    L += ["## 可能是我们的球队、但还没对照的名字", "",
          "下面的条目名和 teams.yaml 里某支同国球队很像，但没写进 `config/team_aliases_wikipedia.yaml`。"
          "程序不猜：确认是同一支球队就加进“对照”，不是就加进“不是我们的球队”。", ""]
    if cands:
        L += ["| 条目名 | 出现次数 | 像哪支 | 相似度 |", "|---|---|---|---|"]
        for d in cands[:80]:
            L.append(f"| {d['条目名']} | {d['出现次数']} | {'、'.join(d['像'])} | {d['相似度']} |")
        if len(cands) > 80:
            L.append(f"| …… 还有 {len(cands) - 80} 个 | | | |")
    else:
        L.append("没有。")
    L.append("")

    # 数据库现状：联赛和杯赛分开数，方便确认联赛的行没被碰到
    db = con.execute("select round = '' and leg = 0, count(*) from matches group by 1").fetchall()
    league_n = next((n for k, n in db if k), 0)
    cup_n = next((n for k, n in db if not k), 0)
    by_src = dict(con.execute("select source, count(*) from match_sources group by source").fetchall())
    L += ["## 数据库现状", "",
          f"- 联赛比赛（round 为空）：{league_n} 场；杯赛、欧战比赛：{cup_n} 场",
          "- 来源表各来源行数：" + "，".join(f"{k} {v}" for k, v in sorted(by_src.items())), ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="把维基百科的杯赛、欧战页面导入数据库")
    parser.add_argument("--only", nargs="*", help="只导入这些赛事代码（例如 UCL UEL），不写就是全部")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    con = connect()
    result = run(settings, con, only=args.only)
    path = write_report(result, settings, con)
    log.info("导入 %s 场比赛（至少一方是我们的球队），外来球队 %s 支。问题 %s 条。报告在 %s",
             result["written"]["matches"], result["written"]["foreign_teams"], len(result["problems"]), path)
    con.close()


if __name__ == "__main__":
    main()
