r"""韩K（KOR）从懂球帝原始文件清洗后导入数据库（需求文档第 4 节“第 11 阶段：韩K 回补和开发回测”、第 17 节）。

运行：.venv\Scripts\python -m src.clean.dongqiudi_kor_import              （几十秒）
      .venv\Scripts\python -m src.clean.dongqiudi_kor_import --db 别的数据库文件
可以反复运行：每次都从原始层整批重导韩K，比赛编号不变；同一批原始文件导两次，结果完全一样。

读的原始文件（下载程序 src/collect/dongqiudi_kor.py 存的）：
  data/raw/dongqiudi/season/KOR/<年>.html.gz           赛季页（页面数据里有轮次、北京时间开球时刻、两队编号、比分）
  data/raw/dongqiudi/odds_final/KOR/<年>/<比赛编号>.json.gz   每场赔率（胜平负、让球、大小球的初盘和“最新”）

做的事：
1. 只要联赛比赛：轮次名字是“<联赛前缀> 第N轮”（settings 的 dongqiudi_kor.league_round_prefixes）；
   non_league_rounds 里的（升降级附加赛、早年的冠军季后赛）不导入，列进报告；两样都不是就报错。
   只要踢完（Played）、有比分的比赛。再查一遍：一季出场次数少于“本季中位数 × league_team_min_share”的队不该有。
2. 球队编号 = 这个懂球帝球队编号最近一季的中文写法（懂球帝没有英文名，不编英文名），国家“韩国”。
   两个懂球帝编号最后的写法一样就报错（不合并）。各年写法 → 球队编号写进 team_names（来源 dongqiudi-kor）。
3. 开球时刻：北京时间 → UTC。match_date = 英国日期（和日职、中超一样）；英国、北京、韩国日期不一样就报错。
   round = 赛季页里第几个轮次（同一对队一季碰 3–4 次，唯一键要靠它分开），leg = 0。
4. 比分进 match_stats。
5. 赔率只导胜平负、timing = close：每家公司的“最新”价格，变盘时间早于开球的才要（开球以后的是滚球价，这家这场不要）。
   竞彩、北单不要。公司名按 config/bookmakers_dongqiudi.yaml 换。
   再加一行“多家平均”：这场合格的公司（价格正常、早于开球、抽水在 quality.overround 的正常范围）
   至少 min_books_average 家时，三个赔率各自算术平均（保留 4 位小数）。
6. 一个事务写进去：只动来源 dongqiudi-kor 的行，别的来源一行都不碰。

导入报告写到 data/reports/韩K_导入报告.md（每季场数、有多家平均 / 平博的比例、不导入的附加赛、去掉的价格）。
"""
import argparse
import logging
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.clean.dongqiudi_import import (TIME_FORMAT, file_time, load_bookmakers, load_fetch_times, overround_ok,
                                        price_ok, rel, result_letter, utc_text)
from src.clean.dongqiudi_reader import parse_odds, parse_season_state, read_json, read_text, resolve_ts, round_kind
from src.clean.schema import create_tables
from src.clean.source_lifecycle import (delete_replaced_results, new_match_ids, reconcile_source_lifecycle,
                                        snapshot_prediction_sources)
from src.common.config import load_settings, resolve_path, season_role
from src.common.db import connect

log = logging.getLogger(__name__)

AVERAGE = "多家平均"          # 和 football-data 的多家平均用同一个公司名（回测标杆 market_avg 按这个名字取）
PINNACLE = "平博 Pinnacle"
LOTTERY_CIDS = {"9998"}       # 北单（对照表“彩票”段之外，dongqiudi_import 也这样排除）


# ---------- 读赛季页 ----------
@dataclass
class Game:
    """一场要导入的联赛比赛（球队编号换好以后）。"""
    dqd_id: int
    season: int
    round_no: int
    round_name: str
    home: str                 # 我们的球队编号（中文名）
    away: str
    match_date: str           # 英国日期
    kickoff_utc: str | None   # 'YYYY-MM-DD HH:MM'（UTC）；赛季页只有日期时是 None
    kickoff_cmp: str | None   # 和赔率变盘时间比的写法（TIME_FORMAT，带秒）
    home_goals: int
    away_goals: int
    raw_file: str
    fetched_at: str


def season_files(raw: Path, league: str, first: int, last: int, settings: dict) -> dict[int, Path]:
    """first–last 每季的赛季页。只读开发赛季：哪一季不是开发赛季就报错（考试赛季锁住）。"""
    out = {}
    for y in range(first, last + 1):
        role = season_role(y, settings)
        if role != "dev":
            raise ValueError(f"{y} 赛季是“{role}”，不是开发赛季：考试赛季锁住，不导入")
        p = raw / "dongqiudi" / "season" / league / f"{y}.html.gz"
        if p.exists():
            out[y] = p
    return out


def classify(files: dict[int, Path], cfg: dict, report: dict) -> dict[int, list]:
    """读每季赛季页 → {赛季: [联赛比赛（踢完、有比分）]}。不导入的比赛记进 report。"""
    prefixes = list(cfg["league_round_prefixes"])
    non_league = list(cfg.get("non_league_rounds") or [])
    out = {}
    for y, p in sorted(files.items()):
        st = parse_season_state(read_text(p))
        if st is None:
            raise ValueError(f"{rel(p)} 里没有赛季数据（window.__INITIAL_STATE__）")
        info, games = st
        if str(info.get("name", y)) != str(y):
            raise ValueError(f"{rel(p)} 是“{info.get('name')}”赛季的页面，不是 {y}")
        report["page_games"][y] = len(games)
        keep = []
        for g in games:
            if round_kind(g.round_name, prefixes, non_league) != "league":
                report["playoffs"].append((y, g.day_bj, g.round_name, g.home, g.home_goals, g.away_goals, g.away))
            elif g.status != "Played" or g.home_goals is None or g.away_goals is None:
                report["not_played"].append((y, g.day_bj, g.round_name, g.home, g.away, g.status))
            else:
                keep.append(g)
        out[y] = keep
    return out


# ---------- 球队编号 ----------
def team_mapping(league_games: dict[int, list], report: dict) -> tuple[dict[str, str], dict[str, str], list[tuple]]:
    """→ ({懂球帝球队编号: 我们的球队编号}, {中文写法: 我们的球队编号}, team_names 要写的行 [(写法, 球队编号)])。

    我们的球队编号 = 这个懂球帝编号最近一季的中文写法（同一季两种写法时取字典序第一个，保证每次一样）。
    """
    latest: dict[str, tuple[int, str]] = {}
    spellings: dict[str, set[str]] = defaultdict(set)      # 写法 → 用过这个写法的懂球帝编号
    no_id = Counter()
    for y in sorted(league_games):
        for g in league_games[y]:
            for name, tid in ((g.home, g.home_id), (g.away, g.away_id)):
                if not tid:
                    no_id[name] += 1
                    continue
                spellings[name].add(tid)
                old = latest.get(tid)
                if old is None or y > old[0] or (y == old[0] and name < old[1]):
                    latest[tid] = (y, name)
    by_id = {tid: name for tid, (_, name) in latest.items()}
    clash = defaultdict(list)
    for tid, name in by_id.items():
        clash[name].append(tid)
    clash = {n: sorted(ids) for n, ids in clash.items() if len(ids) > 1}
    if clash:
        raise ValueError(f"两个懂球帝球队编号最近一季的写法一样（不合并，先查清楚）：{clash}")
    by_name = {}
    for name, ids in sorted(spellings.items()):
        if len(ids) == 1:
            by_name[name] = by_id[next(iter(ids))]
        else:
            report["ambiguous_names"].append((name, sorted(ids)))
    missing = sorted(n for n in no_id if n not in by_name)
    if missing:
        raise ValueError(f"这些队没有懂球帝球队编号，按写法也认不出（不猜）：{missing}")
    report["no_team_id"] = dict(no_id)
    rows = [(f"懂球帝编号 {tid}", team) for tid, team in sorted(by_id.items())]
    rows += sorted(by_name.items())
    report["teams"] = {team: sorted(n for n, ids in spellings.items() if tid in ids)
                       for tid, team in sorted(by_id.items(), key=lambda kv: kv[1])}
    return by_id, by_name, rows


def check_participation(games: list[Game], min_share: float) -> dict[int, int]:
    """每季每队出场次数都要 ≥ 本季中位数 × min_share（附加赛球队只踢 1–2 场）。返回每季球队数；不对就报错。"""
    per = defaultdict(Counter)
    for g in games:
        per[g.season][g.home] += 1
        per[g.season][g.away] += 1
    out = {}
    for y, n in sorted(per.items()):
        med = statistics.median(n.values())
        low = {t: c for t, c in n.items() if c < med * min_share}
        if low:
            raise ValueError(f"{y} 赛季这些队出场太少（中位数 {med}），多半是附加赛没排除干净：{low}")
        out[y] = len(n)
    return out


# ---------- 每场比赛 ----------
def build_games(league_games: dict[int, list], files: dict[int, Path], by_id: dict, by_name: dict, settings: dict,
                fetched: dict, report: dict) -> list[Game]:
    """联赛比赛 → Game（球队编号、UTC 开球时刻、英国日期）。日期对不上、唯一键重复都报错。"""
    cfg = settings["dongqiudi_kor"]
    london = ZoneInfo(settings["features"]["date_only_timezone"])
    local = ZoneInfo(cfg["local_timezone"])
    out, keys = [], {}
    for y in sorted(league_games):
        raw_file, when = rel(files[y]), file_time(files[y], fetched)
        for g in sorted(league_games[y], key=lambda g: (g.round_no, g.match_id)):
            home = by_id[g.home_id] if g.home_id else by_name[g.home]
            away = by_id[g.away_id] if g.away_id else by_name[g.away]
            if g.kickoff_bj is not None:
                ko = g.kickoff_bj.astimezone(timezone.utc)
                days = {ko.astimezone(london).date().isoformat(), g.day_bj, ko.astimezone(local).date().isoformat()}
                if len(days) != 1:
                    raise ValueError(f"比赛 {g.match_id}（{g.day_bj} {g.home} - {g.away}）英国、北京、韩国日期不一样："
                                     f"{sorted(days)}；需求文档说的是一样，先查清楚再导")
                kickoff, cmp_text = ko.strftime("%Y-%m-%d %H:%M"), utc_text(ko)
            else:
                kickoff = cmp_text = None
                report["no_kickoff"][y] += 1
            key = (y, home, away, g.round_no)
            if key in keys:
                raise ValueError(f"{y} 赛季第 {g.round_no} 个轮次 {home} - {away} 出现两次（懂球帝编号 "
                                 f"{keys[key]}、{g.match_id}）")
            keys[key] = g.match_id
            out.append(Game(g.match_id, y, g.round_no, g.round_name, home, away, g.day_bj, kickoff, cmp_text,
                            g.home_goals, g.away_goals, raw_file, when))
    return out


# ---------- 赔率 ----------
def build_odds(games: list[Game], raw: Path, settings: dict, report: dict) -> list[tuple]:
    """每场的胜平负收盘 → odds 表的行 (懂球帝比赛编号, 公司, 主, 平, 客)。

    每家公司 = “最新”价格，要早于开球；多家平均 = 合格公司（价格正常、早于开球、抽水正常）至少
    min_books_average 家时的算术平均。没有开球时刻的比赛不导赔率（没法判断是不是滚球价）。
    """
    cfg = settings["dongqiudi_kor"]
    q = settings["quality"]
    names, products = load_bookmakers(resolve_path(settings["paths"]["bookmakers_dongqiudi"]))
    skip_cids = set(products) | LOTTERY_CIDS
    min_books = int(cfg["min_books_average"])
    st = report["odds"]
    rows = []
    for g in games:
        p = raw / "dongqiudi" / "odds_final" / cfg["league"] / str(g.season) / f"{g.dqd_id}.json.gz"
        if not p.exists():
            st["没有赔率文件"] += 1
            continue
        data = read_json(p)
        if str(data.get("matchId", g.dqd_id)) != str(g.dqd_id):
            st["文件内容是别的比赛（整场不要）"] += 1
            continue
        if g.kickoff_cmp is None:
            st["没有开球时刻（整场不要）"] += 1
            continue
        ko = datetime.strptime(g.kickoff_cmp, TIME_FORMAT).replace(tzinfo=timezone.utc)
        books = {}
        for r in parse_odds(data):
            if r.market != "1x2" or r.timing != "now":
                continue
            if r.cid in skip_cids:
                st["竞彩、北单（不算公司）"] += 1
                continue
            bad = price_ok("1x2", None, r.home, r.draw, r.away, q)
            if bad:
                st[f"去掉：{bad}"] += 1
                continue
            t = resolve_ts(r.ts_text, ko)
            if t is None:
                st["去掉：没有变盘时间"] += 1
                continue
            if t >= ko:
                st["去掉：最新价格在开球以后（滚球价）"] += 1
                continue
            name = names.get(r.bookmaker)
            if name is None:
                report["unknown_books"][r.bookmaker] += 1
                name = r.bookmaker
            if name in books:
                st["去掉：同一场同一家公司出现两次（留第一次）"] += 1
                continue
            books[name] = (r.home, r.draw, r.away)
            report["books"][name] += 1
        good = {b: v for b, v in books.items() if overround_ok("1x2", *v, q)}
        if len(good) < len(books):
            st["抽水超出正常范围（保留这家，不算进多家平均）"] += len(books) - len(good)
        for b, (h, d, a) in sorted(books.items()):
            rows.append((g.dqd_id, b, h, d, a))
        if len(good) >= min_books:
            avg = [round(sum(v[i] for v in good.values()) / len(good), 4) for i in range(3)]
            rows.append((g.dqd_id, AVERAGE, *avg))
            report["books_per_avg"][g.season].append(len(good))
        elif books:
            st[f"合格公司不到 {min_books} 家（这场没有多家平均）"] += 1
    return rows


# ---------- 写数据库 ----------
def check_nothing_lost(con: sqlite3.Connection, source: str, games: list[Game], odds: list[tuple]) -> None:
    """写入前先核对：数据库里已有的韩K 赛季、已有赔率的比赛，这次都还在。

    写入是“先删掉这个来源的全部行再重写”。要是原始层少了某一季的赛季页（比如从工作树复制文件时漏了），
    或者少了某场的赔率文件，重写以后那一季 / 那场的赔率就悄悄没了。所以发现少了就报错，一行都不动。
    """
    old_seasons = {y for (y,) in con.execute(
        "select distinct m.season from matches m join match_sources s on s.match_id = m.match_id "
        "where s.source = ?", (source,))}
    lost_seasons = sorted(old_seasons - {g.season for g in games})
    if lost_seasons:
        raise ValueError(f"数据库里有这些赛季的韩K 比赛，这次原始层里没有（缺赛季页？）：{lost_seasons}。"
                         "不导入（不然会删掉这几季）")
    old_odds = {k for (k,) in con.execute(
        "select distinct s.source_key from odds o join match_sources s "
        "on s.match_id = o.match_id and s.source = o.source where o.source = ?", (source,))}
    lost_odds = sorted(old_odds - {str(r[0]) for r in odds})
    if lost_odds:
        raise ValueError(f"这些比赛数据库里有赔率，这次没有了（缺赔率文件？）：{len(lost_odds)} 场，"
                         f"懂球帝编号 {lost_odds[:10]}…。不导入（不然会删掉这些赔率）")


def write_db(con: sqlite3.Connection, cfg: dict, games: list[Game], name_rows: list[tuple], odds: list[tuple]) -> int:
    """在一个事务里写入：中途出错就全部不算。只动来源 dongqiudi-kor 的行。返回这个来源的比赛数。"""
    source, league, country = cfg["db_source"], cfg["league"], cfg["country"]
    create_tables(con)
    used = sorted({g.home for g in games} | {g.away for g in games})
    if used:
        ph = ",".join("?" * len(used))
        other = con.execute(f"select team_id, country from teams where team_id in ({ph}) and coalesce(country, '') != ?",
                            (*used, country)).fetchall()
        if other:
            raise ValueError(f"这些球队编号数据库里已经有了，国家不是{country}（不合并）：{other}")
    check_nothing_lost(con, source, games, odds)
    with con:
        con.executemany("insert into teams (team_id, name_zh, country) values (?, ?, ?) "
                        "on conflict(team_id) do update set name_zh = excluded.name_zh, country = excluded.country",
                        [(t, t, country) for t in used])
        con.execute("delete from team_names where source = ?", (source,))
        con.executemany("insert into team_names (source, name, team_id) values (?, ?, ?)",
                        [(source, n, t) for n, t in name_rows])
        # 比赛：有就更新（编号不变），没有才新增。round = 第几个轮次，leg = 0
        new_ids = new_match_ids(con)
        con.executemany(
            "insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status, "
            "round, leg, date_precision) values (?, ?, ?, ?, ?, ?, ?, 'finished', ?, 0, 'day') "
            "on conflict(competition, season, home_team, away_team, round, leg) do update set "
            "match_date = excluded.match_date, kickoff_utc = excluded.kickoff_utc, status = excluded.status",
            [(next(new_ids), league, g.season, g.home, g.away, g.match_date, g.kickoff_utc, str(g.round_no))
             for g in games])
        ids = {(s, h, a, r): mid for mid, s, h, a, r in con.execute(
            "select match_id, season, home_team, away_team, round from matches where competition = ?", (league,))}
        dqd_to_id = {g.dqd_id: ids[(g.season, g.home, g.away, str(g.round_no))] for g in games}

        previous = snapshot_prediction_sources(con)
        con.execute("delete from match_sources where source = ?", (source,))
        con.executemany("insert into match_sources values (?, ?, ?, ?, ?)",
                        [(dqd_to_id[g.dqd_id], source, str(g.dqd_id), g.raw_file, g.fetched_at) for g in games])
        delete_replaced_results(con, source)
        con.execute("delete from odds where source = ?", (source,))
        con.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)",
                        [(dqd_to_id[g.dqd_id], source, g.home_goals, g.away_goals,
                          result_letter(g.home_goals, g.away_goals)) for g in games])
        con.executemany("insert into odds (source, match_id, bookmaker, market, timing, line, home, draw, away, over, under) "
                        "values (?, ?, ?, '1x2', 'close', null, ?, ?, ?, null, null)",
                        [(source, dqd_to_id[mid], b, h, d, a) for mid, b, h, d, a in odds])
        reconcile_source_lifecycle(con, previous)
    return con.execute("select count(*) from match_sources where source = ?", (source,)).fetchone()[0]


# ---------- 全部 ----------
def new_report() -> dict:
    return {"page_games": {}, "playoffs": [], "not_played": [], "ambiguous_names": [], "no_team_id": {},
            "teams": {}, "no_kickoff": Counter(), "odds": Counter(), "unknown_books": Counter(), "books": Counter(),
            "books_per_avg": defaultdict(list)}


def run(con: sqlite3.Connection, settings: dict, raw: Path | None = None) -> dict:
    """读原始层 → 写数据库。返回报告用的数（每季场数、赔率覆盖、不导入的比赛……）。"""
    cfg = settings["dongqiudi_kor"]
    league = cfg["league"]
    registered = {c for g in settings["leagues"].values() for c in g}
    if league not in registered:
        raise ValueError(f"{league} 没有登记在 settings.yaml 的 leagues 里")
    raw = raw or resolve_path(settings["paths"]["raw"])
    report = new_report()
    files = season_files(raw, league, int(cfg["first_season"]), int(cfg["last_season"]), settings)
    if not files:
        raise ValueError("原始层里没有韩K 赛季页，先运行 src.collect.dongqiudi_kor 下载")
    create_tables(con)
    fetched = load_fetch_times(con)
    league_games = classify(files, cfg, report)
    by_id, by_name, name_rows = team_mapping(league_games, report)
    games = build_games(league_games, files, by_id, by_name, settings, fetched, report)
    report["teams_per_season"] = check_participation(games, float(cfg["league_team_min_share"]))
    odds = build_odds(games, raw, settings, report)
    report["matches"] = write_db(con, cfg, games, name_rows, odds)
    report["season_table"] = season_table(con, cfg)
    return report


def season_table(con: sqlite3.Connection, cfg: dict) -> list[dict]:
    """从数据库里数：每季场数、有多家平均 / 平博胜平负收盘 / 任意一家的场数。"""
    src, league = cfg["db_source"], cfg["league"]
    rows = []
    for season, n, avg, pin, anyb in con.execute(
            "select m.season, count(*), "
            " sum(exists (select 1 from odds o where o.match_id = m.match_id and o.source = ? and o.bookmaker = ?)), "
            " sum(exists (select 1 from odds o where o.match_id = m.match_id and o.source = ? and o.bookmaker = ?)), "
            " sum(exists (select 1 from odds o where o.match_id = m.match_id and o.source = ?)) "
            "from matches m join match_sources s on s.match_id = m.match_id and s.source = ? "
            "where m.competition = ? group by m.season order by m.season",
            (src, AVERAGE, src, PINNACLE, src, src, league)):
        rows.append({"赛季": season, "场数": n, "有任意一家": anyb, "有多家平均": avg, "多家平均比例": round(avg / n, 3),
                     "有平博胜平负": pin, "平博比例": round(pin / n, 3)})
    return rows


def content_digest(con: sqlite3.Connection, cfg: dict) -> str:
    """韩K 在数据库里的全部内容（比赛表、来源、比分、赔率、队名对照）→ 一串指纹。两次导入指纹一样 = 结果完全一样。"""
    import hashlib
    src, league = cfg["db_source"], cfg["league"]
    h = hashlib.sha256()
    for sql, params in (
            ("select * from matches where competition = ? order by match_id", (league,)),
            ("select * from match_sources where source = ? order by match_id", (src,)),
            ("select * from match_stats where source = ? order by match_id", (src,)),
            ("select * from odds where source = ? order by match_id, bookmaker", (src,)),
            ("select * from team_names where source = ? order by name", (src,)),
            ("select * from teams where team_id in (select team_id from team_names where source = ?) order by team_id",
             (src,))):
        for r in con.execute(sql, params):
            h.update(repr(r).encode("utf-8"))
    return h.hexdigest()


def report_markdown(rep: dict, cfg: dict, digests: list[str]) -> str:
    """导入报告（中文）。"""
    lines = ["# 韩K（KOR）懂球帝导入报告（自动生成）", "",
             f"来源名 {cfg['db_source']}，共 {rep['matches']} 场联赛比赛。只读开发赛季 "
             f"{cfg['first_season']}–{cfg['last_season']}；赔率从 {cfg['odds_first_season']} 起才下。", ""]
    if digests:
        same = "一样" if len(set(digests)) == 1 else "不一样（有问题）"
        lines += [f"重跑核对：同一批原始文件连着导 {len(digests)} 次，内容指纹{same}：{digests[-1][:16]}…", ""]
    lines += ["## 每季场数和赔率覆盖", "",
              "| 赛季 | 赛季页总场数 | 导入（联赛比赛） | 球队数 | 有任意一家胜平负 | 有多家平均 | 比例 | 多家平均的公司数（中位数） "
              "| 有平博胜平负 | 比例 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["season_table"]:
        y = r["赛季"]
        k = rep["books_per_avg"].get(y)
        med = statistics.median(k) if k else ""
        lines.append(f"| {y} | {rep['page_games'].get(y, '')} | {r['场数']} | {rep['teams_per_season'].get(y, '')} "
                     f"| {r['有任意一家']} | {r['有多家平均']} | {r['多家平均比例']} | {med} | {r['有平博胜平负']} | {r['平博比例']} |")
    lines += ["", "## 不导入的比赛（附加赛、季后赛）", "",
              "规则：轮次名字是“<联赛前缀> 第N轮”才是联赛比赛，前缀 = "
              f"{'、'.join(cfg['league_round_prefixes'])}；下面这些轮次名字不是联赛比赛："
              f"{'、'.join(cfg['non_league_rounds'])}。", ""]
    by = Counter((y, rn) for y, _d, rn, *_ in rep["playoffs"])
    lines += [f"- {y} {rn}：{n} 场" for (y, rn), n in sorted(by.items())] or ["（没有）"]
    lines += ["", f"没踢完或没比分的联赛比赛（不导入）：{len(rep['not_played'])} 场"]
    lines += [f"- {y} {d} {rn} {h} - {a}（{s}）" for y, d, rn, h, a, s in rep["not_played"]]
    lines += ["", f"没有开球时刻的比赛（只有日期，不导赔率）：{sum(rep['no_kickoff'].values())} 场", "",
              "## 赔率处理", "",
              "每家公司的“最新”胜平负，变盘时间早于开球的才要；多家平均 = 价格正常、早于开球、抽水在正常范围的公司"
              f"至少 {cfg['min_books_average']} 家时的算术平均。", ""]
    lines += [f"- {k}：{v}" for k, v in sorted(rep["odds"].items())]
    lines += ["", "各公司进库的场数（换过名字的写法）："]
    lines += [f"- {b}：{n}" for b, n in rep["books"].most_common()]
    if rep["unknown_books"]:
        lines += ["", "对照表里没有的公司（照原样用懂球帝的名字）："]
        lines += [f"- {b}：{n} 组" for b, n in rep["unknown_books"].most_common()]
    lines += ["", "## 球队（编号 = 最近一季的中文写法；后面是用过的写法）", ""]
    lines += [f"- {t}：{'、'.join(ns)}" for t, ns in rep["teams"].items()]
    if rep["ambiguous_names"]:
        lines += ["", "同一个写法对应两个懂球帝球队编号（不写进队名对照）："]
        lines += [f"- {n}：{ids}" for n, ids in rep["ambiguous_names"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="韩K 从懂球帝原始文件导入数据库")
    ap.add_argument("--db", default=None, help="数据库文件，默认 settings 里的")
    ap.add_argument("--twice", action="store_true", help="连着导两次，核对两次的内容指纹一样（重跑不变）")
    a = ap.parse_args(argv)
    settings = load_settings()
    cfg = settings["dongqiudi_kor"]
    con = connect(Path(a.db)) if a.db else connect()
    digests = []
    for _ in range(2 if a.twice else 1):
        rep = run(con, settings)
        digests.append(content_digest(con, cfg))
    if len(set(digests)) > 1:
        raise SystemExit(f"两次导入的结果不一样：{digests}")
    out = resolve_path(settings["paths"]["reports"]) / cfg["report_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_markdown(rep, cfg, digests), encoding="utf-8")
    log.info("导入 %d 场；报告：%s", rep["matches"], out)


if __name__ == "__main__":
    main()
