r"""把懂球帝的原始文件导入数据库（需求文档 6.2、6.3、6.4、第 17 节）。

运行方法：Claude 运行 `.venv\Scripts\python -m src.clean.dongqiudi_import`（几秒到一两分钟，看原始文件多少）。
可以反复运行：每次都从原始层整批重导懂球帝的数据，结果一样；已经有的比赛编号不变。
第一次运行时加 `--fill-teams-yaml`：把投票定下来的中文队名写进 config/teams.yaml 里还空着的“中文”
（只填空的，不改已经填的）。这样以后重导 football-data 时中文名不会被清空。

做的事：
1. 队名对照：和 src/clean/dongqiudi_match.py 一样“投票”——懂球帝每场比赛在我们数据库同一联赛同一赛季里
   找“日期差 ≤ 1 天、比分一样”的比赛投票，票够多、占大多数的才定下来。
   除了历届赛季页，当前赛季已经踢完的比赛（赛程接口里的 Played）也拿来投票，这样新升级的球队也能对上。
   定下来的写进 team_names（来源 dongqiudi），中文名填进 teams.name_zh（只填空的）。
2. 比赛对照：按（联赛、赛季、主队、客队）找我们数据库里的比赛（联赛里同一主客队一个赛季只踢一次），
   对上的写进 match_sources（来源 dongqiudi，编号是懂球帝的比赛编号）。
   - 以前的赛季：只对照，不新增比赛（以前的比赛以 football-data 为准）。
   - 当前赛季：数据库里没有的比赛（还没踢、football-data 还没更新）新增一行，状态记“未赛”。
     比分、“已赛”状态不从懂球帝来：等 football-data 更新后重导，它会把同一行改成已赛（比赛编号不变）。
     只有懂球帝一个来源的比赛，开球时间按懂球帝最新的赛程更新（改期了也跟着改）。
3. 变盘记录（odds_changes 表）：赔率文件里每家公司的初盘（begin）和最新（now）各一行，带网站显示的变盘时间。
   我们每小时存的快照里，同一个价格会重复出现很多次，只留第一次看到的那行。
   公司名按 config/bookmakers_dongqiudi.yaml 换成我们的写法（例如 Bet3** → Bet365）。
4. 彩票赔率（lottery_odds 表）：竞彩胜平负、竞彩让球、北单，只显示不进模型。没有变盘时间，按“我们第一次看到”的时间算。
5. 报告：data/reports/dongqiudi_导入.md，另有没对上的比赛、队名对照两个 CSV。

只删、只写来源是 dongqiudi 的行（match_sources、odds_changes、lottery_odds、team_names），不碰别的来源。
采集程序（每小时任务、历史回补）可以同时运行：写数据库的那一步放在一个很短的事务里。
"""
import argparse
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.clean.dongqiudi_match import decide, load_db_matches, load_dqd_seasons, vote, write_csv
from src.clean.dongqiudi_reader import SeasonGame, parse_fixtures, parse_odds, read_json, resolve_ts
from src.clean.schema import create_tables
from src.collect.dongqiudi import current_season
from src.common.config import PROJECT_ROOT, load_settings, resolve_path
from src.common.db import connect

SOURCE = "dongqiudi"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"   # 和取数层的截止时间写法一样，文字直接比大小就是比时间先后
log = logging.getLogger(__name__)

# 懂球帝赛程的状态 → 我们 matches.status。踢完了（Played）也记“未赛”：赛果以 football-data 为准，等它更新
STATUS = {"Fixture": "scheduled", "Uncertain": "scheduled", "Playing": "scheduled", "Played": "scheduled",
          "Postponed": "postponed", "Cancelled": "cancelled", "Cancel": "cancelled"}


def utc_text(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime(TIME_FORMAT)


def rel(path: Path) -> str:
    """文件位置写成相对于项目文件夹的样子（和抓取记录 fetch_log 一样），测试里的临时文件夹就写全路径。"""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


# ---------- 读配置、抓取记录 ----------
def load_bookmakers(path: Path) -> tuple[dict, dict]:
    """公司对照表 → ({懂球帝公司名: 我们的公司名}, {懂球帝公司编号: 彩票种类})。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = {k: v["名字"] for k, v in (cfg.get("公司") or {}).items()}
    lottery = {str(k): v for k, v in (cfg.get("彩票") or {}).items()}
    return names, lottery


def load_fetch_times(con) -> dict[str, str]:
    """抓取记录里每个已保存文件第一次保存的时间：{文件位置: 标准时间文字}。"""
    out = {}
    for path, t in con.execute("select file_path, min(fetched_at_utc) from fetch_log "
                               "where source = ? and status = 'saved' and file_path is not null group by file_path",
                               (SOURCE,)):
        out[path.replace("\\", "/")] = utc_text(datetime.fromisoformat(t))
    return out


STAMP_RE = re.compile(r"(\d{8}T\d{4})Z")


def file_time(path: Path, fetched: dict[str, str]) -> str:
    """一个原始文件是什么时候抓到的：先查抓取记录，没有就看文件名里的时间，再没有就用文件的修改时间。"""
    t = fetched.get(rel(path))
    if t:
        return t
    m = STAMP_RE.search(path.name)
    if m:
        return utc_text(datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc))
    return utc_text(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))


# ---------- 赛程 ----------
@dataclass
class Seen:
    """赛程接口里一场比赛最后一次看到的样子。"""
    league: str
    season: int
    fx: object        # dongqiudi_reader.Fixture
    raw_file: str
    fetched_at: str


def load_fixtures(raw: Path, leagues: list[str], fetched: dict) -> dict[int, Seen]:
    """原始层里全部赛程文件 → {懂球帝比赛编号: 最后一次看到的样子}（按抓取时间先后读，后面的覆盖前面的）。"""
    files = []
    for lg in leagues:
        folder = raw / "dongqiudi" / "fixtures" / lg
        for f in folder.glob("*.json.gz") if folder.exists() else []:
            if ".suspicious." not in f.name:
                files.append((file_time(f, fetched), lg, f))
    out = {}
    for t, lg, f in sorted(files):
        for fx in parse_fixtures(read_json(f)):
            # 赛季按开球时间算（7 月起算新赛季）；没有开球时间的按抓取时间算
            when = fx.kickoff_utc or datetime.strptime(t, TIME_FORMAT).replace(tzinfo=timezone.utc)
            out[fx.match_id] = Seen(lg, current_season(when), fx, rel(f), t)
    return out


def played_games(seen: dict[int, Seen]) -> dict[str, dict[int, list[SeasonGame]]]:
    """赛程里已经踢完、有比分的比赛，换成赛季页的格式（日期用北京时间），拿来一起投票。"""
    out = defaultdict(lambda: defaultdict(list))
    for s in seen.values():
        fx = s.fx
        if fx.status == "Played" and fx.home_goals is not None and fx.away_goals is not None and fx.kickoff_utc:
            bj = fx.kickoff_utc + timedelta(hours=8)
            out[s.league][s.season].append(SeasonGame(fx.match_id, None, bj.month, bj.day, fx.home, fx.away,
                                                      fx.home_goals, fx.away_goals))
    return out
# ---------- 队名、比赛对照 ----------
def team_mapping(con, pages: dict, seen: dict, leagues: list[str]) -> tuple[dict, list[dict], dict]:
    """投票定队名（第 1 步）。返回（{中文名: 球队编号}, 报告表格, 每个联赛数据库里的比赛）。"""
    db = {lg: load_db_matches(con, lg) for lg in leagues}
    played = played_games(seen)
    votes = defaultdict(lambda: defaultdict(float))
    for lg in leagues:
        games = {season: list(gs) for season, gs in pages[lg].items()}
        for season, gs in played[lg].items():   # 赛季页里已经有的比赛不重复投票
            have = {g.match_id for g in games.get(season, [])}
            games.setdefault(season, []).extend(g for g in gs if g.match_id not in have)
        vote(games, db[lg], votes)
    mapping, table = decide(votes)
    known = {r[0] for r in con.execute("select team_id from teams")}
    mapping = {zh: en for zh, en in mapping.items() if en in known}   # 防止对到球队表里没有的名字
    return mapping, table, db


@dataclass
class Candidate:
    """懂球帝的一场比赛（来自赛季页或赛程）。"""
    dqd_id: int
    league: str
    season: int
    home_zh: str
    away_zh: str
    home_goals: int | None
    away_goals: int | None
    raw_file: str
    fetched_at: str
    seen: Seen | None = None   # 赛程里看到过就有（当前赛季的比赛）


def candidates(raw: Path, pages: dict, seen: dict, fetched: dict) -> list[Candidate]:
    """全部要对照的比赛：赛程里看到的（有开球时间，优先用）+ 赛季页里的（赛程里没有的）。"""
    out = [Candidate(i, s.league, s.season, s.fx.home, s.fx.away, s.fx.home_goals, s.fx.away_goals,
                     s.raw_file, s.fetched_at, s) for i, s in seen.items()]
    for lg, by_season in pages.items():
        for season, games in by_season.items():
            f = raw / "dongqiudi" / "season" / lg / f"{season}.html.gz"
            t = file_time(f, fetched) if f.exists() else utc_text(datetime.now(timezone.utc))
            out += [Candidate(g.match_id, lg, season, g.home, g.away, g.home_goals, g.away_goals, rel(f), t)
                    for g in games if g.match_id not in seen]
    return out


@dataclass
class Link:
    cand: Candidate
    match_id: int | None   # 对上的我们的比赛编号；None = 要新增的当前赛季比赛
    result: str


def link_matches(cands: list[Candidate], db: dict, mapping: dict, season_now: int) -> list[Link]:
    """第 2 步：按（联赛、赛季、主队、客队）对比赛。每场懂球帝比赛返回一个 Link（结果写在 result 里）。"""
    idx = {(lg, m.season, m.home, m.away): m for lg, by_season in db.items()
           for ms in by_season.values() for m in ms}
    groups, out = defaultdict(list), []
    for c in cands:
        home, away = mapping.get(c.home_zh), mapping.get(c.away_zh)
        if not (home and away):
            out.append(Link(c, None, "队名没对上"))
        else:
            groups[(c.league, c.season, home, away)].append(c)
    for key, cs in groups.items():
        m = idx.get(key)
        if m is None and not (key[1] == season_now and any(c.seen for c in cs)):
            out += [Link(c, None, "数据库里没有这场") for c in cs]
            continue
        # 同一场比赛懂球帝有两个编号时（很少见，例如重赛）：先挑比分一样的，再挑最近看到的，其余不导入
        def rank(c):
            same = m is not None and c.home_goals is not None and (c.home_goals, c.away_goals) == (m.home_goals, m.away_goals)
            return (same, c.seen is not None, c.fetched_at)
        cs = sorted(cs, key=rank, reverse=True)
        best = cs[0]
        if m is None:
            out.append(Link(best, None, "新增（当前赛季）"))
        else:
            differ = (best.home_goals is not None and m.home_goals is not None
                      and (best.home_goals, best.away_goals) != (m.home_goals, m.away_goals))
            out.append(Link(best, m.match_id, "对上（比分不同）" if differ else "对上"))
        out += [Link(c, None, "重复（不导入）") for c in cs[1:]]
    return out


def fixture_fields(fx, back_days: int, tz: ZoneInfo) -> tuple[str, str | None, str, str]:
    """赛程里的一场 → matches 表的（日期, 开球时间, 状态, 日期精度）。

    日期按英国当地日期（和 football-data 一样，取数层对只有日期的比赛也按英国时间算）。
    时间待定（Uncertain）的：开球时间是占位的，不写开球时间，日期往前挪几天当作“这一轮从哪天开始”。
    """
    if fx.status == "Uncertain" or fx.kickoff_utc is None:
        base = fx.kickoff_utc or datetime.now(timezone.utc)
        d = (base.astimezone(tz) - timedelta(days=back_days)).date()
        return d.isoformat(), None, "scheduled", "round"
    return (fx.kickoff_utc.astimezone(tz).date().isoformat(), fx.kickoff_utc.strftime("%Y-%m-%d %H:%M"),
            STATUS.get(fx.status, "scheduled"), "day")
# ---------- 赔率 ----------
@dataclass
class OddsFile:
    path: Path
    dqd_id: int
    kind: str          # live 赛前快照 / final 踢完后的最终赔率
    fetched_at: str


def odds_files(raw: Path, leagues: list[str], fetched: dict) -> list[OddsFile]:
    """原始层里全部赔率文件，按抓取时间从早到晚排（去重时“第一次看到”的留下）。内容可疑的文件跳过。"""
    out = []
    for lg in leagues:
        base = raw / "dongqiudi"
        for f in (base / "odds_live" / lg).glob("*/*.json.gz"):
            if ".suspicious." not in f.name:
                out.append(OddsFile(f, int(f.parent.name), "live", file_time(f, fetched)))
        for f in (base / "odds_final" / lg).glob("*/*.json.gz"):
            if ".suspicious." not in f.name:
                out.append(OddsFile(f, int(f.name.split(".")[0]), "final", file_time(f, fetched)))
    return sorted(out, key=lambda o: (o.fetched_at, str(o.path)))


@dataclass
class Ref:
    """推变盘时间的年份、判断“开球后的价格”要用的开球时间。"""
    kickoff: datetime
    exact: bool        # True = 真的开球时间；False = 只有日期，用当天中午（标准时间）代替


def kickoff_refs(links: list[Link], db: dict, back_days: int, tz: ZoneInfo) -> dict[int, Ref]:
    """每场导入的懂球帝比赛的开球时间：数据库有就用数据库的，没有用懂球帝赛程的，再没有用日期中午。"""
    by_id = {m.match_id: m for by_season in db.values() for ms in by_season.values() for m in ms}
    out = {}
    for ln in links:
        if ln.match_id is None and ln.result != "新增（当前赛季）":
            continue
        c, m = ln.cand, by_id.get(ln.match_id)
        fx = c.seen.fx if c.seen else None
        if m is not None and m.kickoff_utc:
            ko = datetime.strptime(m.kickoff_utc, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            out[c.dqd_id] = Ref(ko, True)
        elif fx is not None and fx.kickoff_utc and fx.status != "Uncertain":
            out[c.dqd_id] = Ref(fx.kickoff_utc, True)
        elif m is not None:
            out[c.dqd_id] = Ref(datetime.combine(m.day, datetime.min.time(), timezone.utc) + timedelta(hours=12), False)
        else:
            d = date.fromisoformat(fixture_fields(fx, back_days, tz)[0])
            out[c.dqd_id] = Ref(datetime.combine(d, datetime.min.time(), timezone.utc) + timedelta(hours=12), False)
    return out


def price_ok(market: str, line, home, draw, away, q: dict) -> str | None:
    """检查一组价格；有问题返回原因，没问题返回 None。"""
    prices = (home, draw, away) if market == "1x2" else (home, away)
    if any(p is None for p in prices):
        return "价格不全"
    if any(p < q["min_odds"] for p in prices):
        return f"赔率低于 {q['min_odds']}"
    if market == "ah" and (line is None or abs(line) > q["max_handicap"]):
        return "让球盘口缺失或超出范围"
    if market == "ou" and (line is None or not 0 < line <= 10):
        return "大小球盘口缺失或超出范围"
    return None


def overround_ok(market: str, home, draw, away, q: dict) -> bool:
    """赔率倒数之和在不在正常范围（只记进报告，不删：交易所等可能本来就偏）。"""
    lo, hi = q["overround"]["1x2" if market == "1x2" else "2way"]
    s = sum(1 / p for p in ((home, draw, away) if market == "1x2" else (home, away)))
    return lo <= s <= hi


def lottery_rows(data: dict, products: dict, q: dict) -> list[tuple]:
    """彩票赔率（只在胜平负表 euro 里认）→ [(彩票种类, 让球, 初盘/最新, 主, 平, 客)]。占位的（0、1.00 这类）不要。"""
    out = []
    for b in data.get("euro") or []:
        product = products.get(str(b.get("cid", "")))
        if not product:
            continue
        try:
            handicap = int(float(b.get("fixed_odds") or 0))
        except (TypeError, ValueError):
            continue
        for timing in ("begin", "now"):
            o = b.get(timing) or {}
            try:
                h, d, a = (float(o.get(k)) for k in ("homeWin", "draw", "awayWin"))
            except (TypeError, ValueError):
                continue
            if min(h, d, a) >= q["min_odds"]:
                out.append((product, handicap, timing, h, d, a))
    return out
def build_odds(files: list[OddsFile], refs: dict[int, Ref], books: dict, products: dict, q: dict):
    """读全部赔率文件 → (变盘记录行, 彩票赔率行, 统计)。行里先用懂球帝比赛编号，写库时再换成我们的编号。

    变盘记录去重：比赛 + 公司 + 玩法 + 盘口 + 变盘时间 + 赔率都一样的只留第一次看到的。
    彩票赔率去重：每个（比赛, 彩票种类, 让球）只留第一次看到的初盘，之后价格变了才多留一行。
    """
    stats = Counter()
    unknown_books = Counter()
    changes, seen_keys = [], set()
    lottery, lot_last, lot_begin = [], {}, set()
    for of in files:
        ref = refs.get(of.dqd_id)
        if ref is None:
            stats["文件：比赛没对上，跳过"] += 1
            continue
        data = read_json(of.path)
        if str(data.get("matchId", of.dqd_id)) != str(of.dqd_id):
            stats["文件：内容是别的比赛，跳过"] += 1
            continue
        stats[f"文件：{'赛前快照' if of.kind == 'live' else '最终赔率'}"] += 1
        raw_file = rel(of.path)
        for r in parse_odds(data):
            if r.cid in products or r.cid == "9998":   # 彩票不是博彩公司，不进变盘记录
                continue
            bad = price_ok(r.market, r.line, r.home, r.draw, r.away, q)
            if bad:
                stats[f"去掉：{bad}"] += 1
                continue
            t = resolve_ts(r.ts_text, ref.kickoff)
            if t is None:
                stats["去掉：没有变盘时间"] += 1
                continue
            name = books.get(r.bookmaker)
            if name is None:
                unknown_books[r.bookmaker] += 1
                name = r.bookmaker
            over, under = (r.home, r.away) if r.market == "ou" else (None, None)
            home, away = (None, None) if r.market == "ou" else (r.home, r.away)
            changed = utc_text(t)
            key = (of.dqd_id, name, r.market, r.line, changed, home, r.draw, away, over, under)
            if key in seen_keys:
                stats["重复（快照里又看到一次）"] += 1
                continue
            seen_keys.add(key)
            if not overround_ok(r.market, r.home, r.draw, r.away, q):
                stats["注意：赔率倒数之和超出正常范围（保留）"] += 1
            if ref.exact and t >= ref.kickoff:
                stats["注意：开球后的价格（保留，取数层不给赛前用）"] += 1
            changes.append((of.dqd_id, SOURCE, name, r.bookmaker, r.market, r.line, home, r.draw, away, over, under,
                            changed, of.fetched_at, raw_file))
        for product, hc, timing, h, d, a in lottery_rows(data, products, q):
            k = (of.dqd_id, product, hc)
            if timing == "begin":
                if k in lot_begin:
                    continue
                lot_begin.add(k)
            elif lot_last.get(k) == (h, d, a):
                continue
            lot_last[k] = (h, d, a)
            lottery.append((of.dqd_id, SOURCE, product, hc, h, d, a, of.fetched_at, raw_file))
    return changes, lottery, stats, unknown_books
# ---------- 写数据库 ----------
def chinese_names(links: list[Link], mapping: dict) -> dict[str, str]:
    """每支球队的中文名：一支球队有几个中文写法（改过名）时，用最近一个赛季的写法。"""
    latest = {}
    for ln in links:
        if ln.match_id is None and ln.result != "新增（当前赛季）":
            continue
        c = ln.cand
        for zh in (c.home_zh, c.away_zh):
            en = mapping[zh]
            if en not in latest or (c.season, c.fetched_at) > latest[en][0]:
                latest[en] = ((c.season, c.fetched_at), zh)
    return {en: zh for en, (_k, zh) in latest.items()}


def write_db(con, links: list[Link], mapping: dict, zh_names: dict, changes: list, lottery: list,
             back_days: int, tz: ZoneInfo) -> dict:
    """在一个事务里写完。返回 {懂球帝比赛编号: 我们的比赛编号} 和几个计数。"""
    others = {r[0] for r in con.execute("select distinct match_id from match_sources where source != ?", (SOURCE,))}
    counts = Counter()
    started = time.perf_counter()
    with con:
        con.execute("begin immediate")   # 一开始就占住写锁，免得中途和采集程序抢
        con.execute("delete from team_names where source = ?", (SOURCE,))
        con.executemany("insert into team_names (source, name, team_id) values (?, ?, ?)",
                        [(SOURCE, zh, en) for zh, en in mapping.items()])
        cur = con.executemany("update teams set name_zh = ? where team_id = ? and (name_zh is null or name_zh = '')",
                              [(zh, en) for en, zh in zh_names.items()])
        counts["填了中文名的球队"] = cur.rowcount
        ids, sources = {}, []
        for ln in links:
            c = ln.cand
            if ln.match_id is None and ln.result != "新增（当前赛季）":
                continue
            mid = ln.match_id
            if c.seen is not None and (mid is None or mid not in others):
                # 只有懂球帝一个来源的比赛（新增的，或者以前新增过的）：按最新赛程写日期、开球时间
                mdate, ko, status, prec = fixture_fields(c.seen.fx, back_days, tz)
                home, away = mapping[c.home_zh], mapping[c.away_zh]
                con.execute(
                    "insert into matches (competition, season, home_team, away_team, match_date, kickoff_utc, status,"
                    " round, leg, date_precision) values (?, ?, ?, ?, ?, ?, ?, '', 0, ?)"
                    " on conflict(competition, season, home_team, away_team, round, leg) do update set"
                    " match_date = excluded.match_date, kickoff_utc = excluded.kickoff_utc,"
                    " status = excluded.status, date_precision = excluded.date_precision",
                    (c.league, c.season, home, away, mdate, ko, status, prec))
                mid = con.execute("select match_id from matches where competition = ? and season = ? and home_team = ?"
                                  " and away_team = ? and round = '' and leg = 0",
                                  (c.league, c.season, home, away)).fetchone()[0]
                counts["只有懂球帝的比赛（新增或更新）"] += 1
            ids[c.dqd_id] = mid
            sources.append((mid, SOURCE, str(c.dqd_id), c.raw_file, c.fetched_at))
        con.execute("delete from match_sources where source = ?", (SOURCE,))
        con.executemany("insert into match_sources (match_id, source, source_key, raw_file, fetched_at_utc)"
                        " values (?, ?, ?, ?, ?)", sources)
        con.execute("delete from odds_changes where source = ?", (SOURCE,))
        con.executemany("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw,"
                        " away, over, under, changed_at_utc, fetched_at_utc, raw_file)"
                        " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(ids[r[0]], *r[1:]) for r in changes if r[0] in ids])
        con.execute("delete from lottery_odds where source = ?", (SOURCE,))
        con.executemany("insert into lottery_odds (match_id, source, product, handicap, home, draw, away,"
                        " fetched_at_utc, raw_file) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(ids[r[0]], *r[1:]) for r in lottery if r[0] in ids])
        # 没有任何来源的比赛（例如以前新增、后来懂球帝改了队名对照的）删掉，和其他导入程序一样
        cur = con.execute("delete from matches where match_id not in (select match_id from match_sources)")
        counts["删掉的没有来源的比赛"] = cur.rowcount
    counts["写数据库用时（秒）"] = round(time.perf_counter() - started, 1)
    return {"ids": ids, "counts": counts}


def fill_teams_yaml(path: Path, zh_names: dict) -> int:
    """把中文名填进 teams.yaml 里还空着的“中文”。只改这一格，注释、顺序都不动。返回填了几个。"""
    text = path.read_text(encoding="utf-8")
    n = 0

    def put(m):
        nonlocal n
        zh = zh_names.get(m.group(2))
        if not zh:
            return m.group(0)
        n += 1
        return f"{m.group(1)}中文: '{zh.replace(chr(39), chr(39) * 2)}'"
    text = re.sub(r'(\{英文: "([^"]*)", )中文: \'\'', put, text)
    path.write_text(text, encoding="utf-8")
    return n
# ---------- 报告 ----------
def season_table(con, leagues: list[str], first_season: int) -> list[str]:
    """每个联赛、赛季：数据库里几场、对上懂球帝几场、其中有变盘记录的几场、变盘记录几行。"""
    ph = ",".join("?" * len(leagues))
    # 三样东西各查一次，在 Python 里按赛季加总（直接在 SQL 里连起来会一场一场地扫，很慢）
    linked = {r[0] for r in con.execute("select match_id from match_sources where source = ?", (SOURCE,))}
    n_odds = dict(con.execute("select match_id, count(*) from odds_changes where source = ? group by match_id", (SOURCE,)))
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for mid, comp, season in con.execute(
            f"select match_id, competition, season from matches where competition in ({ph}) and season >= ? and round = ''",
            [*leagues, first_season]):
        a = agg[(comp, season)]
        a[0] += 1
        a[1] += mid in linked
        a[2] += mid in n_odds
        a[3] += n_odds.get(mid, 0)
    rows = [(c, s, *v) for (c, s), v in sorted(agg.items())]
    out = ["| 联赛 | 赛季 | 数据库场数 | 对上懂球帝 | 有变盘记录的场数 | 变盘记录行数 |", "|---|---|---|---|---|---|"]
    out += [f"| {c} | {s}/{(s + 1) % 100:02d} | {n} | {k} | {o} | {r} |" for c, s, n, k, o, r in rows]
    return out


def write_report(path: Path, links: list[Link], mapping: dict, table: list[dict], stats: Counter,
                 unknown_books: Counter, written: dict, lottery_n: int, changes_n: int, con, leagues, first_season):
    res = Counter(ln.result for ln in links)
    pending = [r for r in table if r["结果"] != "定下"]
    lines = ["# 懂球帝导入数据库", "",
             f"程序：`src/clean/dongqiudi_import.py`。运行时间（标准时间）：{utc_text(datetime.now(timezone.utc))}。", "",
             "## 队名", "",
             f"中文队名 {len(table)} 个，定下来 {len(mapping)} 个，待人工确认 {len(pending)} 个"
             f"（见 dongqiudi_导入_队名对照.csv）。", ""]
    if pending:
        lines += ["待人工确认：" + "、".join(f"{r['中文名']}（最像 {r['英文名']}，{r['得票']} 票）" for r in pending[:30]), ""]
    lines += ["## 比赛", "", "| 结果 | 场数 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in res.most_common()]
    lines += ["", "没对上的比赛见 dongqiudi_导入_没对上.csv。", "",
              "## 写进数据库", "", "| 项目 | 数量 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in written["counts"].items()]
    lines += [f"| 变盘记录（行） | {changes_n} |", f"| 彩票赔率（行） | {lottery_n} |"]
    lines += ["", "## 读赔率文件", "", "| 项目 | 数量 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(stats.items())]
    if unknown_books:
        lines += ["", "没在公司对照表里的公司名（照原样导入）：" +
                  "、".join(f"{k}（{v} 行）" for k, v in unknown_books.most_common())]
    lines += ["", "## 每个赛季", ""] + season_table(con, leagues, first_season)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(con, raw: Path, reports: Path, s: dict, now: datetime | None = None, teams_yaml: Path | None = None) -> dict:
    """导入一次（测试里直接调用这个，给临时数据库和临时文件夹）。teams_yaml 给了就把中文名填进去。"""
    reports.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    dq, q = s["dongqiudi"], s["quality"]
    leagues, back_days = list(dq["league_ids"]), int(dq["uncertain_back_days"])
    tz = ZoneInfo(s["features"]["date_only_timezone"])
    books, products = load_bookmakers(resolve_path(s["paths"]["bookmakers_dongqiudi"]))

    create_tables(con)
    con.commit()
    fetched = load_fetch_times(con)
    pages = {lg: load_dqd_seasons(raw, lg) for lg in leagues}
    seen = load_fixtures(raw, leagues, fetched)
    log.info("赛季页 %d 个赛季，赛程里 %d 场", sum(len(v) for v in pages.values()), len(seen))
    mapping, table, db = team_mapping(con, pages, seen, leagues)
    links = link_matches(candidates(raw, pages, seen, fetched), db, mapping, current_season(now))
    refs = kickoff_refs(links, db, back_days, tz)
    files = odds_files(raw, leagues, fetched)
    log.info("队名定下 %d 个；要导入的比赛 %d 场；赔率文件 %d 个", len(mapping), len(refs), len(files))
    changes, lottery, stats, unknown = build_odds(files, refs, books, products, q)
    zh_names = chinese_names(links, mapping)
    written = write_db(con, links, mapping, zh_names, changes, lottery, back_days, tz)
    n_changes = sum(1 for r in changes if r[0] in written["ids"])
    n_lottery = sum(1 for r in lottery if r[0] in written["ids"])
    log.info("写好了：变盘记录 %d 行，彩票赔率 %d 行，用时 %s 秒", n_changes, n_lottery,
             written["counts"]["写数据库用时（秒）"])
    if teams_yaml is not None:
        n = fill_teams_yaml(teams_yaml, zh_names)
        written["counts"]["teams.yaml 填了中文名"] = n
        log.info("teams.yaml 填了 %d 个中文名", n)

    write_csv(reports / "dongqiudi_导入_队名对照.csv", table)
    write_csv(reports / "dongqiudi_导入_没对上.csv",
              [{"联赛": ln.cand.league, "赛季": ln.cand.season, "懂球帝编号": ln.cand.dqd_id, "主队": ln.cand.home_zh,
                "客队": ln.cand.away_zh, "比分": "" if ln.cand.home_goals is None else f"{ln.cand.home_goals}-{ln.cand.away_goals}",
                "结果": ln.result, "原始文件": ln.cand.raw_file}
               for ln in links if ln.result not in ("对上", "新增（当前赛季）")])
    write_report(reports / "dongqiudi_导入.md", links, mapping, table, stats, unknown, written,
                 n_lottery, n_changes, con, leagues, int(dq["backfill_first_season"]))
    log.info("报告：%s", reports / "dongqiudi_导入.md")
    return {"links": links, "mapping": mapping, "stats": stats, **written}


def main(argv=None):
    ap = argparse.ArgumentParser(description="把懂球帝的原始文件导入数据库")
    ap.add_argument("--fill-teams-yaml", action="store_true", help="把中文队名写进 config/teams.yaml 里空着的“中文”")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    s = load_settings()
    con = connect()
    try:
        run(con, resolve_path(s["paths"]["raw"]), resolve_path(s["paths"]["reports"]), s,
            teams_yaml=resolve_path(s["paths"]["teams"]) if args.fill_teams_yaml else None)
    finally:
        con.close()


if __name__ == "__main__":
    main()
