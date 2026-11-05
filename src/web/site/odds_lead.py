"""盘口联动观察页 odds_lead.html：一场比赛赔率明显变动时，哪几家公司先变、哪几家跟着一起变、哪几家晚变或不跟
（需求文档第 8 节“盘口联动观察页怎么算”，规则在写这个程序之前写下）。

这一页只是观察，不是预测：先变不等于更准，也不能拿来下注。页面不写“推荐”。
数据只用网站已经读进来的那些比赛（data.odds，经过取数层、用途「显示」、只有开球前的价格），不另开数据库、不联网、不改数据库。

怎么算（一句话版，详细的见需求文档）：
1. 只看已经开球、而且开球前 48 小时都在我们自己快照里的比赛（开球晚于 odds_lead.since + 48 小时）。
2. 三种玩法分开算。每家公司的每个价格换成一个数：
   胜平负 = 去水后的主胜概率；让球 = 共同盘口上去水后主队赢盘的机会；大小球 = 共同盘口上去水后大球的机会。
   “共同盘口”= 各家开球前最后一个价格的盘口里最多的那条（让球一样多取离 0 近的，大小球一样多取小的）。
3. 窗口开始（开球前 48 小时）那一刻每家的值 = 那时它最后一个价格；那时还没价格的这家这场不算。
4. 市场线 = 各家当时最新值的中位数。变动 D = 开球前最后的市场线 − 窗口开始的市场线。
   |D| 不到 2 个百分点、或者不到 5 家的不看。
5. 市场走到一半的时间：市场线第一次朝 D 的方向走过 |D| 的一半。
   每家走到一半的时间：这家自己的值第一次从它自己的起点朝 D 的方向走过 |D| 的一半（市场变动的一半）。
6. 先变多少（小时）= 市场走到一半的时间 − 这家走到一半的时间。早 1 小时以上 = 先变，晚 1 小时以上 = 晚变，
   中间 = 一起变；到开球都没走到 = 没跟。

页面上任何一块出了意外的错：整页换成占位页，不让整个网站生成失败（render_checked）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from src.common.config import resolve_path
from src.common.odds import devig
from src.web.site.common import bj_text, details, e, kickoff_text, match_href, page, placeholder, table
from src.web.site.match import line_name

MARKETS = ("1x2", "ah", "ou")
MARKET_NAMES = {"1x2": "胜平负", "ah": "让球", "ou": "大小球"}
KINDS = ("先变", "一起变", "晚变", "没跟")
HEAD = "<h1>盘口联动 <span class='accent'>谁先变</span></h1>"
NOTE = ("<b>只是观察，不是预测。</b>这一页只看“赔率明显变动时哪家公司先变、哪家跟着变”，"
        "先变不等于更准，也不能拿来下注。")
BAD_FORMAT = "盘口联动页出了意外的错"
EPS = 1e-9          # 比较小数时留的一点余量（例如正好走到一半、正好早 1 小时）


def _t(text) -> pd.Timestamp:
    """标准时间（文字或带时区的时间）→ 不带时区的 UTC 时间，方便互相比较、相减。"""
    t = pd.Timestamp(text)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _utc_text(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


# ---------- 每家公司的值 ----------
def common_line(last: pd.DataFrame, market: str) -> float | None:
    """这场的“共同盘口”：各家开球前最后一个价格的盘口里最多的那条。
    一样多时：让球取离 0 近的（再一样取主队受让的那条，也就是大的），大小球取小的。"""
    lines = last["line"].dropna().astype(float)
    if lines.empty:
        return None
    counts = lines.round(4).value_counts()
    best = counts[counts == counts.max()].index
    if market == "ah":
        return float(sorted(best, key=lambda x: (abs(x), -x))[0])
    return float(min(best))


def book_values(odds: pd.DataFrame, market: str) -> tuple[pd.DataFrame, float | None]:
    """一场比赛、一种玩法里，每家公司每个价格换成的那个数。
    返回 (表：bookmaker、t（时间）、v（数）按时间排好，共同盘口)。胜平负的共同盘口是 None。
    同一家同一分钟有几行（初盘和最新时间一样）时，留后面那行（后导入的是“最新”）。"""
    o = odds[odds["market"] == market].copy()
    cols = ["home", "draw", "away"] if market == "1x2" else (["home", "away"] if market == "ah" else ["over", "under"])
    o = o.dropna(subset=cols)
    o = o[(o[cols].astype(float) > 1).all(axis=1)]          # 赔率要大于 1，不然不是正常价格
    empty = pd.DataFrame(columns=["bookmaker", "t", "v"])
    if o.empty:
        return empty, None
    o["t"] = o["changed_at_utc"].map(_t)
    o = o.sort_values("t", kind="stable")                    # 同一时间保持原来的先后（取数层按导入先后排的）
    line = None
    if market != "1x2":
        # 每家开球前最后一个价格（同一家几条盘口同时开时，这就是最后变的那条，一般是主盘口）
        line = common_line(o.groupby("bookmaker").tail(1), market)
        if line is None:
            return empty, None
        o = o[o["line"].astype(float).round(4) == round(line, 4)]
    o = o.groupby(["bookmaker", "t"], sort=False).tail(1)
    if o.empty:
        return empty, line
    o["v"] = devig(o[cols].to_numpy(dtype=float))[:, 0]    # 主胜 / 主队赢盘 / 大球 的机会
    return o[["bookmaker", "t", "v"]].reset_index(drop=True), line


# ---------- 一场比赛的一次变动 ----------
def match_move(vals: pd.DataFrame, kickoff: pd.Timestamp, cfg: dict) -> dict | None:
    """一场比赛一种玩法：有没有明显变动；有的话每家是先变、一起变、晚变还是没跟。没有明显变动（或家数不够）返回 None。"""
    start = kickoff - pd.Timedelta(hours=float(cfg["window_hours"]))
    vals = vals[vals["t"] < kickoff]
    before = vals[vals["t"] <= start]
    base = before.groupby("bookmaker")["v"].last().to_dict()          # 窗口开始那一刻每家的值
    if len(base) < int(cfg["min_books"]):
        return None
    after = vals[(vals["t"] > start) & vals["bookmaker"].isin(base)]
    cur = dict(base)
    path = [(start, float(np.median(list(cur.values()))))]           # 市场线怎么走：[(时间, 中位数), ...]
    for t, g in after.groupby("t", sort=True):
        cur.update(dict(zip(g["bookmaker"], g["v"])))
        path.append((t, float(np.median(list(cur.values())))))
    m0, m1 = path[0][1], path[-1][1]
    d = m1 - m0
    if abs(d) < float(cfg["min_move"]) - EPS:
        return None
    sign, half = (1.0 if d > 0 else -1.0), abs(d) / 2
    market_half = next(t for t, m in path if sign * (m - m0) >= half - EPS)
    tol = float(cfg["tolerance_hours"])
    books = {}
    for book, b0 in base.items():
        own = after[after["bookmaker"] == book]
        hit = own[sign * (own["v"] - b0) >= half - EPS]
        if hit.empty:
            books[book] = {"kind": "没跟", "lead": None, "half": None}
            continue
        t_half = hit["t"].iloc[0]
        lead = (market_half - t_half).total_seconds() / 3600
        kind = "先变" if lead >= tol - EPS else ("晚变" if lead <= -tol + EPS else "一起变")
        books[book] = {"kind": kind, "lead": lead, "half": t_half}
    reached = [b["half"] for b in books.values() if b["half"] is not None]
    first_t = min(reached) if reached else None
    first = sorted(k for k, b in books.items() if first_t is not None and b["half"] == first_t)
    return {"start": m0, "end": m1, "move": d, "n_books": len(base), "market_half": market_half,
            "books": books, "first": first, "first_t": first_t}


def eligible(data, cfg: dict) -> list[int]:
    """要看的比赛：网站读进来了变盘记录、知道开球时间、已经开球、开球前 48 小时都在快照里。"""
    now = _t(data.now)
    earliest = _t(cfg["since"]) + pd.Timedelta(hours=float(cfg["window_hours"]))
    out = []
    for mid, odds in data.odds.items():
        info = data.matches.get(mid) or {}
        if odds is None or odds.empty or not info.get("kickoff_utc") or info.get("date_precision") == "round":
            continue
        k = _t(info["kickoff_utc"])
        if earliest < k <= now:
            out.append(mid)
    return sorted(out)


def eligible_checked(data) -> list[int]:
    """和 eligible 一样，设置坏了返回空的（现读现出算数据版本用：比赛开球后这一页会多一场，
    数据库文件没变时也要让开着的页面知道内容变了）。"""
    try:
        return eligible(data, data.settings["odds_lead"])
    except Exception:   # noqa: BLE001 —— 故意的：这里只是算版本，页面本身的错由 render_checked 处理
        return []


def collect(data, cfg: dict) -> tuple[list[dict], int]:
    """所有要看的比赛、三种玩法里的明显变动：([每次变动一行], 看了几场比赛)。"""
    mids = eligible(data, cfg)
    moves = []
    for mid in mids:
        kickoff = _t(data.matches[mid]["kickoff_utc"])
        for market in MARKETS:
            vals, line = book_values(data.odds[mid], market)
            mv = match_move(vals, kickoff, cfg) if not vals.empty else None
            if mv:
                moves.append({**mv, "match_id": mid, "market": market, "line": line, "kickoff": kickoff})
    return moves, len(mids)


def summarize(moves: list[dict], cfg: dict) -> list[dict]:
    """每家公司一行：算进来几场、先变 / 一起变 / 晚变 / 没跟各几场、先变多少的中位数（小时）、场数够不够。
    排序：场数够的在前（按“先变”占比、场数），不够的在后（只按场数，不排位置）。"""
    per: dict[str, dict] = {}
    for mv in moves:
        for book, b in mv["books"].items():
            r = per.setdefault(book, {"book": book, "n": 0, **{k: 0 for k in KINDS}, "leads": []})
            r["n"] += 1
            r[b["kind"]] += 1
            if b["lead"] is not None:
                r["leads"].append(b["lead"])
    need = int(cfg["min_matches"])
    rows = []
    for r in per.values():
        rows.append({**{k: v for k, v in r.items() if k != "leads"},
                     "median_lead": float(np.median(r["leads"])) if r["leads"] else None,
                     "enough": r["n"] >= need})
    return sorted(rows, key=lambda r: (not r["enough"], -(r["先变"] / r["n"]) if r["enough"] else 0, -r["n"], r["book"]))


# ---------- 网页 ----------
def guessed_names(settings: dict) -> set[str]:
    """公司对照表里写“推测”的那些公司（我们数据库里的写法）。表不在或读不了就当没有，页面上不标。"""
    path = (settings.get("paths") or {}).get("bookmakers_dongqiudi")
    if not path:
        return set()
    try:
        with open(resolve_path(path), encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return set()
    return {str(v.get("名字")) for v in (cfg.get("公司") or {}).values()
            if isinstance(v, dict) and v.get("依据") == "推测" and v.get("名字")}


def _book(name: str, guessed: set[str]) -> str:
    return e(name) + (" <span class='small muted'>（名字是推测的）</span>" if name in guessed else "")


def _share(k: int, n: int) -> str:
    return f"{k}（{k / n * 100:.0f}%）" if n else "—"


def _hours(h) -> str:
    if h is None:
        return "—"
    return f"早 {h:.1f} 小时" if h > 0 else (f"晚 {-h:.1f} 小时" if h < 0 else "同时")


def books_table(rows: list[dict], guessed: set[str], need: int) -> str:
    """每家公司一行的汇总表。场数不够的只写场数。"""
    out, rank = [], 0
    for r in rows:
        if r["enough"]:
            rank += 1
            out.append([str(rank), _book(r["book"], guessed), str(r["n"]), *(_share(r[k], r["n"]) for k in KINDS),
                        e(_hours(r["median_lead"]))])
        else:
            out.append(["<span class='small muted'>不排</span>", _book(r["book"], guessed), str(r["n"]),
                        f"<td colspan='5' class='small muted'>不到 {need} 场，先不下结论</td>"])
    return table(["位置", "公司", "场数", *KINDS, "先变多少（中位数）"], out)


def change_text(mv: dict) -> str:
    """这次变动的方向和大小，例：主胜 42% → 47%（多 5.0 个百分点）。"""
    what = {"1x2": "主胜", "ah": "主队赢盘", "ou": "大球"}[mv["market"]]
    more = "多" if mv["move"] > 0 else "少"
    return (f"{what} {mv['start'] * 100:.0f}% → {mv['end'] * 100:.0f}%"
            f"（{more} {abs(mv['move']) * 100:.1f} 个百分点）")


def recent_table(data, moves: list[dict], guessed: set[str], limit: int) -> str:
    """最近几次明显变动：比赛、玩法、变动、几家、最先走到一半的公司（同一分钟几家都写）、市场走到一半的时间。"""
    rows, attrs = [], []
    for mv in sorted(moves, key=lambda m: (m["kickoff"], m["match_id"], MARKETS.index(m["market"])), reverse=True)[:limit]:
        info = data.matches[mv["match_id"]]
        name = f"{data.zh(info['home_team'])} 对 {data.zh(info['away_team'])}"
        market = MARKET_NAMES[mv["market"]]
        if mv["market"] == "ah":
            market += f" {line_name(mv['line'])}"
        elif mv["market"] == "ou":
            market += f" {mv['line']:g} 球"
        first = "、".join(_book(b, guessed) for b in mv["first"]) or "—"
        when = bj_text(_utc_text(mv["first_t"])) if mv["first_t"] is not None else "—"
        rows.append([e(kickoff_text(info)), f"<a href='{match_href(mv['match_id'], '')}'>{e(name)}</a>", e(market),
                     e(change_text(mv)), str(mv["n_books"]), f"{first} <span class='small muted'>{e(when)}</span>",
                     e(bj_text(_utc_text(mv["market_half"])))])
    return table(["开球", "比赛", "玩法", "变动（去水后）", "几家", "最先走到一半的公司", "市场走到一半"], rows)


def limits_html(cfg: dict) -> str:
    return ("<ul class='small'>"
            "<li>时间是懂球帝显示的变盘时间（到分钟）。但我们离开赛 6 小时以外 3 小时、12 小时才看一次，"
            "两次之间同一家变了几次只能看到最后一次：中间那次的时间看不到，变得勤的公司可能被看成晚变。</li>"
            "<li>公司名有的是推测的（懂球帝把名字打了星号），表里标了“名字是推测的”。</li>"
            f"<li>场数少时结论不稳：不到 {int(cfg['min_matches'])} 场的公司只写场数。"
            "先变的公司也不一定更准，这一页不看准不准。</li></ul>")


def method_html(cfg: dict) -> str:
    return (f"<ul class='small'><li>只看已经开球、开球前 {float(cfg['window_hours']):g} 小时都在我们快照里的比赛"
            "（以前回补的只有初盘和最新两个点，看不出谁先变）。</li>"
            "<li>每家每个价格换成一个数：胜平负是去水后的主胜概率；让球是“共同盘口”（各家最后价格里最多的那条盘口）"
            "上去水后主队赢盘的机会；大小球是共同盘口上去水后大球的机会。别的盘口上的价格不算。</li>"
            f"<li>市场线 = 各家当时最新值的中位数。开球前 {float(cfg['window_hours']):g} 小时到开球，市场线变了 "
            f"{float(cfg['min_move']) * 100:g} 个百分点以上、至少 {int(cfg['min_books'])} 家的才算“明显变动”。</li>"
            "<li>市场走到一半 = 市场线第一次走过变动的一半；每家走到一半 = 这家自己的值第一次走过市场变动的一半。"
            f"比市场早 {float(cfg['tolerance_hours']):g} 小时以上 = 先变，晚 {float(cfg['tolerance_hours']):g} 小时以上 = 晚变，"
            "中间 = 一起变，到开球都没走到 = 没跟。</li></ul>")


def _page(data, body: str) -> str:
    return page("盘口联动 - 足球预测", body, "", data.now, data.settings["site"]["refresh_minutes"])


def _render(data) -> str:
    cfg = data.settings["odds_lead"]
    moves, n_matches = collect(data, cfg)
    head = HEAD + f"<div class='note'>{NOTE}</div>"
    since = bj_text(_utc_text(_t(cfg["since"]) + pd.Timedelta(hours=float(cfg["window_hours"]))), "%Y-%m-%d %H:%M")
    if not moves:
        why = (f"还没有可以看的明显变动：{since}（北京时间）以后开球、网站上有单场页的比赛里，"
               f"看了 {n_matches} 场，没有一场的赔率变得够明显。")
        return _page(data, head + placeholder(why) + details("怎么算的", method_html(cfg)))
    guessed = guessed_names(data.settings)
    need = int(cfg["min_matches"])
    by_market = {mk: [m for m in moves if m["market"] == mk] for mk in MARKETS}
    counts = "，".join(f"{MARKET_NAMES[mk]} {len(v)} 次" for mk, v in by_market.items())
    intro = (f"<p>{e(since)}（北京时间）以后开球、网站上有单场页的比赛看了 {n_matches} 场，"
             f"开球前 {float(cfg['window_hours']):g} 小时里赔率明显变动的：{e(counts)}。</p>")
    parts = [head, intro, "<h2>每家公司：三种玩法合在一起</h2>",
             "<p class='small muted'>场数 = 这家算进来的“一场比赛的一种玩法”有几次。先变多少 = 比市场早几小时走到一半（中位数）。</p>",
             books_table(summarize(moves, cfg), guessed, need),
             "<h2>最近的明显变动</h2>", recent_table(data, moves, guessed, int(cfg["recent_rows"]))]
    for mk, mvs in by_market.items():
        inner = books_table(summarize(mvs, cfg), guessed, need) if mvs else placeholder("这种玩法还没有明显变动")
        parts.append(details(f"只看{MARKET_NAMES[mk]}（{len(mvs)} 次）", inner))
    parts += [details("怎么算的", method_html(cfg)), "<h2>看的时候注意</h2>", limits_html(cfg)]
    return _page(data, "".join(parts))


def render_checked(data) -> tuple[str, str | None]:
    """盘口联动页：返回 (网页, 出错种类)；正常时出错种类是 None。
    出了任何意外的错，整页换成占位页（故意接住所有 Exception，只写错误种类，不写细节，免得把路径写到网页上）。"""
    try:
        return _render(data), None
    except Exception as err:   # noqa: BLE001 —— 故意的：这一页坏了不能拖垮整个网站
        kind = type(err).__name__
        return _page(data, HEAD + f"<div class='note'>{NOTE}</div>"
                     + placeholder(f"{BAD_FORMAT}，这一页先不显示（{kind}）。别的页不受影响。")), kind


def render(data) -> str:
    return render_checked(data)[0]
