r"""国际赛（国家队）懂球帝赔率导入数据库（需求文档第 4 节“第 11 阶段：国际赛进数据库和开发回测”的“赔率（INB）”、第 17 节）。

比赛和比分是 INA 从 martj42 导进去的（赛事代码 INT，球队编号 = martj42 英文名 + “ NT”，match_date = martj42 的日期，
kickoff_utc 空着，date_precision 'day_world'，round = 'YYYY-MM-DD'，leg 0，比分在 match_stats 来源 martj42）。
这个程序不新建比赛、不改比赛表、不写比分，只把懂球帝的赔率挂到这些比赛上。

运行：.venv\Scripts\python -m src.clean.dongqiudi_intl_import              （要等 INA 合并、INT 比赛进了数据库以后）
      .venv\Scripts\python -m src.clean.dongqiudi_intl_import --db 别的数据库文件 --twice
      .venv\Scripts\python -m src.clean.dongqiudi_intl_import --suggest results.csv   （只列队名对照候选，不写数据库）
可以反复运行：每次都从原始层整批重导来源 dongqiudi-intl 的行；同一批原始文件导两次，结果完全一样。

读的原始文件（下载程序 src/collect/dongqiudi_intl.py 存的）：
  data/raw/dongqiudi/season/<赛事代码>/<赛季>.html.gz                 赛季页（北京时间开球时刻、两队编号、比分）
  data/raw/dongqiudi/odds_final/<赛事代码>/<赛季>/<比赛编号>.json.gz   每场赔率
队名对照 config/teams_intl.yaml（懂球帝球队编号 → 中文名、martj42 英文名），只按它认，不猜。

做的事：
1. 只看踢完、有比分、开球在 kickoff_from 到 kickoff_to（UTC）的比赛；开球在考试赛季（2024-01-01 UTC 起）的不读。
2. 对上 martj42 的比赛：两队一样（换成 “英文名 NT”）、日期差 ≤ 1 天（懂球帝用北京时间的日期，martj42 用当地日期）。
   - 主客对调的（中立场常见）：赔率的主胜、客胜对调，列进报告；
   - 对不上的、对上不止一场的：不导，列进报告；
   - 比分和 martj42 不一样的：不导赔率，列进报告（INC 打分时不用）。
   注意：懂球帝赛季页的比分含加时（2022 世界杯决赛写 3-3），和 martj42 一样，所以“比分不一样”很少，
   靠它找不出加时分胜负的比赛（需求文档以为懂球帝是 90 分钟比分，实际不是）。
3. 赔率只导胜平负、timing = close：每家公司的“最新”价格，变盘时间早于懂球帝开球时刻的才要（开球以后的是滚球价）。
   竞彩、北单不要；公司名按 config/bookmakers_dongqiudi.yaml 换。再加一行“多家平均”：合格公司（价格正常、早于开球、
   抽水在 quality.overround 的正常范围）至少 min_books_average 家时，三个赔率各自算术平均（保留 4 位小数）。和韩K 一样。
   懂球帝只有日期、没有开球时刻的比赛不导赔率（没法判断是不是滚球价）。
4. 一个事务写进去：只动来源 dongqiudi-intl 的 match_sources、odds、team_names；teams.name_zh 只填空着的。
   别的来源、比赛表、比分一行都不碰。写之前先核对：数据库里已有赔率的比赛，这次原始层里都还在，少了就报错、一行不动。

导入报告写到 data/reports/国际赛_赔率导入报告.md（按年、按赛事：场数、对上、对调、比分不一样、多家平均比例、
平博胜平负比例、去掉的开球后价格组数）。
"""
import argparse
import csv
import hashlib
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from src.clean.dongqiudi_import import TIME_FORMAT, file_time, load_bookmakers, load_fetch_times, overround_ok, price_ok, rel
from src.clean.dongqiudi_reader import StateGame, parse_odds, parse_season_state, read_json, read_text, resolve_ts
from src.clean.schema import create_tables
from src.collect.dongqiudi_intl import in_window, kickoff_span, season_folder, window
from src.common.config import load_settings, resolve_path
from src.common.db import connect

log = logging.getLogger(__name__)

AVERAGE = "多家平均"          # 和 football-data、韩K 的多家平均用同一个公司名
PINNACLE = "平博 Pinnacle"
LOTTERY_CIDS = {"9998"}       # 北单（对照表“彩票”段之外，dongqiudi_import、韩K 也这样排除）


# ---------- 读赛季页 ----------
@dataclass
class DqdGame:
    """懂球帝的一场国际赛（下载范围里、踢完、有比分）。"""
    code: str                 # 我们的赛事文件夹代码，例如 INT_WC
    label: str                # 赛季文件夹名，例如 2022、2020-21
    g: StateGame
    year: int                 # 报告按年分：开球那一年（UTC；只有日期的用北京时间的日期）
    raw_file: str
    fetched_at: str


def load_games(raw: Path, cfg: dict, settings: dict, con, report: dict) -> list[DqdGame]:
    """读 settings 里列的每个赛季页 → 下载范围里踢完、有比分的比赛。同一场在两个赛季页里都有就只留第一个（按 settings 顺序）。"""
    start, end = window(cfg, settings)           # 考试赛季锁：范围碰到 2024 年就报错
    fetched = load_fetch_times(con)
    out, seen = [], {}
    for code, comp in cfg["competitions"].items():
        for label, sid in comp["seasons"].items():
            label = season_folder(label)
            p = raw / "dongqiudi" / "season" / code / f"{label}.html.gz"
            if not p.exists():
                report["missing_pages"].append(f"{code} {label}")
                continue
            st = parse_season_state(read_text(p))
            if st is None:
                raise ValueError(f"{rel(p)} 里没有赛季数据（window.__INITIAL_STATE__）")
            info, games = st
            if str(info.get("sid")) != str(sid) or str(info.get("cid")) != str(comp["cid"]):
                raise ValueError(f"{rel(p)} 是赛事 {info.get('cid')}、赛季 {info.get('sid')} 的页面，settings 写的是 "
                                 f"{comp['cid']}、{sid}")
            raw_file, when = rel(p), file_time(p, fetched)
            report["comp_names"][code] = str(info.get("comp") or "")
            for g in games:
                if not in_window(g, start, end):
                    continue                     # 开球不在范围里（含考试赛季）：不读
                if g.status != "Played" or g.home_goals is None or g.away_goals is None:
                    report["not_played"].append((code, label, g.day_bj, g.home, g.away, g.status))
                    continue
                if g.match_id in seen:
                    report["duplicate_pages"].append((g.match_id, seen[g.match_id], f"{code} {label}"))
                    continue
                seen[g.match_id] = f"{code} {label}"
                year = g.kickoff_bj.astimezone(timezone.utc).year if g.kickoff_bj else int(g.day_bj[:4])
                out.append(DqdGame(code, label, g, year, raw_file, when))
    return out


# ---------- 队名对照 ----------
def load_team_map(path: Path) -> dict[str, dict]:
    """config/teams_intl.yaml → {懂球帝球队编号: {"zh": 中文名, "martj42": 英文名}}。缺字段、空名字就报错。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    teams = data.get("teams") or {}
    out = {}
    for tid, v in teams.items():
        if not isinstance(v, dict) or not str(v.get("zh") or "").strip() or not str(v.get("martj42") or "").strip():
            raise ValueError(f"{path.name} 里懂球帝球队编号 {tid} 要有 zh 和 martj42 两个名字：{v!r}")
        out[str(tid)] = {"zh": str(v["zh"]).strip(), "martj42": str(v["martj42"]).strip()}
    clash = defaultdict(list)
    for tid, v in out.items():
        clash[v["martj42"]].append(tid)
    clash = {n: ids for n, ids in clash.items() if len(ids) > 1}
    if clash:
        raise ValueError(f"{path.name} 里两个懂球帝球队编号对到同一个 martj42 名字（不合并，先查清楚）：{clash}")
    return out


# ---------- 对上 martj42 的比赛 ----------
@dataclass
class Link:
    """一场对上了、比分也一样的比赛：懂球帝的比赛 → 数据库的比赛编号。swapped = 懂球帝的主客和数据库相反。"""
    d: DqdGame
    match_id: int
    swapped: bool


def db_matches(con: sqlite3.Connection, cfg: dict) -> dict[frozenset, list[tuple]]:
    """数据库里 INT 的比赛 → {两队编号的集合: [(比赛编号, 日期, 主队, 客队, 主队进球, 客队进球)]}。比分取 martj42 的。"""
    out = defaultdict(list)
    for row in con.execute(
            "select m.match_id, m.match_date, m.home_team, m.away_team, s.home_goals, s.away_goals from matches m "
            "left join match_stats s on s.match_id = m.match_id and s.source = ? where m.competition = ? "
            "order by m.match_id", (cfg["score_source"], cfg["competition"])):
        out[frozenset((row[2], row[3]))].append(row)
    return out


def link_games(games: list[DqdGame], team_map: dict, by_pair: dict, cfg: dict, report: dict) -> list[Link]:
    """两队一样、日期差 ≤ 1 天 = 同一场。对不上、对上不止一场、比分不一样的都不要，记进报告。"""
    suffix = cfg["martj42_suffix"]
    links = []
    for d in games:
        g = d.g
        grp = report["by"][(d.year, d.code)]
        grp["比赛"] += 1
        h, a = team_map.get(g.home_id), team_map.get(g.away_id)
        if h is None or a is None:
            grp["队名对照里没有"] += 1
            for name, tid, v in ((g.home, g.home_id, h), (g.away, g.away_id, a)):
                if v is None:
                    report["unmapped_teams"][(tid, name)] += 1
            continue
        home, away = h["martj42"] + suffix, a["martj42"] + suffix
        day = date.fromisoformat(g.day_bj)
        cands = [m for m in by_pair.get(frozenset((home, away)), [])
                 if abs((date.fromisoformat(m[1]) - day).days) <= 1]
        desc = (d.code, d.label, g.match_id, g.day_bj, g.home, g.home_goals, g.away_goals, g.away)
        if not cands:
            grp["对不上"] += 1
            report["unmatched"].append(desc)
            continue
        if len(cands) > 1:
            grp["对上不止一场"] += 1
            report["multi"].append((*desc, [c[0] for c in cands]))
            continue
        mid, mdate, mh, ma, hg, ag = cands[0]
        swapped = mh != home
        grp["对上"] += 1
        # 懂球帝的比分摆成数据库的主客顺序再比
        dh, da = (g.away_goals, g.home_goals) if swapped else (g.home_goals, g.away_goals)
        if swapped:
            grp["主客对调"] += 1
            report["swapped"].append((*desc, mid, mdate, mh, ma))
        if hg is None or ag is None:
            grp["martj42 没有比分"] += 1
            report["no_score"].append((*desc, mid))
            continue
        if (dh, da) != (hg, ag):
            grp["比分不一样"] += 1
            report["score_diff"].append((*desc, mid, mdate, mh, hg, ag, ma))
            continue
        links.append(Link(d, mid, swapped))
    # 两场懂球帝比赛对到数据库同一场：都不要（不猜哪场对）
    n = Counter(lk.match_id for lk in links)
    for lk in links:
        if n[lk.match_id] > 1:
            report["same_db_match"].append((lk.d.code, lk.d.label, lk.d.g.match_id, lk.match_id))
            report["by"][(lk.d.year, lk.d.code)]["两场对到同一场"] += 1
    links = [lk for lk in links if n[lk.match_id] == 1]
    for lk in links:
        report["by"][(lk.d.year, lk.d.code)]["导入"] += 1
    return links


# ---------- 赔率 ----------
def build_odds(links: list[Link], raw: Path, settings: dict, report: dict) -> list[tuple]:
    """每场的胜平负收盘 → odds 表的行 (数据库比赛编号, 公司, 主, 平, 客)，已经摆成数据库的主客顺序。

    规则和韩K（src/clean/dongqiudi_kor_import.py 的 build_odds）一样：每家公司 = “最新”价格，要早于懂球帝的开球时刻；
    多家平均 = 合格公司（价格正常、早于开球、抽水正常）至少 min_books_average 家时的算术平均。
    """
    cfg = settings["dongqiudi_intl"]
    q = settings["quality"]
    names, products = load_bookmakers(resolve_path(settings["paths"]["bookmakers_dongqiudi"]))
    skip_cids = set(products) | LOTTERY_CIDS
    min_books = int(cfg["min_books_average"])
    st = report["odds"]
    rows = []
    for lk in sorted(links, key=lambda lk: lk.match_id):
        g, grp = lk.d.g, report["by"][(lk.d.year, lk.d.code)]
        p = raw / "dongqiudi" / "odds_final" / lk.d.code / lk.d.label / f"{g.match_id}.json.gz"
        if not p.exists():
            st["没有赔率文件"] += 1
            continue
        data = read_json(p)
        if str(data.get("matchId", g.match_id)) != str(g.match_id):
            st["文件内容是别的比赛（整场不要）"] += 1
            continue
        if g.kickoff_bj is None:
            st["懂球帝没有开球时刻（整场不要）"] += 1
            continue
        ko = g.kickoff_bj.astimezone(timezone.utc)
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
                grp["去掉的开球后价格"] += 1
                continue
            name = names.get(r.bookmaker)
            if name is None:
                report["unknown_books"][r.bookmaker] += 1
                name = r.bookmaker
            if name in books:
                st["去掉：同一场同一家公司出现两次（留第一次）"] += 1
                continue
            # 主客对调的：懂球帝的主胜是数据库客队赢
            books[name] = (r.away, r.draw, r.home) if lk.swapped else (r.home, r.draw, r.away)
            report["books"][name] += 1
        good = {b: v for b, v in books.items() if overround_ok("1x2", *v, q)}
        if len(good) < len(books):
            st["抽水超出正常范围（保留这家，不算进多家平均）"] += len(books) - len(good)
        for b, (h, d, a) in sorted(books.items()):
            rows.append((lk.match_id, b, h, d, a))
        if books:
            grp["有任意一家"] += 1
        if PINNACLE in books:
            grp["有平博胜平负"] += 1
        if len(good) >= min_books:
            avg = [round(sum(v[i] for v in good.values()) / len(good), 4) for i in range(3)]
            rows.append((lk.match_id, AVERAGE, *avg))
            grp["有多家平均"] += 1
            report["books_per_avg"][(lk.d.year, lk.d.code)].append(len(good))
        elif books:
            st[f"合格公司不到 {min_books} 家（这场没有多家平均）"] += 1
    return rows


# ---------- 写数据库 ----------
def check_nothing_lost(con: sqlite3.Connection, source: str, links: list[Link], odds: list[tuple]) -> None:
    """写入前先核对：数据库里已有赔率的比赛，这次都还有赔率。

    写入是“先删掉这个来源的全部行再重写”。要是原始层少了某个赛季页、某场的赔率文件（比如从工作树复制文件时漏了），
    或者队名对照改了以后某场对不上了，重写以后那场的赔率就悄悄没了。所以发现少了就报错，一行都不动。
    """
    old = {(int(mid), key) for mid, key in con.execute(
        "select distinct s.match_id, s.source_key from odds o join match_sources s "
        "on s.match_id = o.match_id and s.source = o.source where o.source = ?", (source,))}
    with_odds = {r[0] for r in odds}
    new = {(lk.match_id, str(lk.d.g.match_id)) for lk in links if lk.match_id in with_odds}
    lost = sorted(old - new)
    if lost:
        raise ValueError(f"这些比赛数据库里有懂球帝国际赛赔率，这次没有了（缺赛季页、缺赔率文件，还是队名对照改了？）："
                         f"{len(lost)} 场，(比赛编号, 懂球帝编号) {lost[:10]}…。不导入（不然会删掉这些赔率）")


def name_rows(con: sqlite3.Connection, team_map: dict, cfg: dict, report: dict) -> tuple[list[tuple], list[tuple]]:
    """→ (team_names 要写的行 [(写法, 球队编号)], 要填中文名的 [(球队编号, 中文名)])。

    只写数据库 teams 表里已经有的球队（INA 建的 “英文名 NT”）；没有的列进报告。
    写法有两种：“懂球帝编号 <编号>”和中文名。同一个中文名对到两个球队就都不写这个中文名（不猜）。
    """
    suffix = cfg["martj42_suffix"]
    have = {t for (t,) in con.execute("select team_id from teams")}
    rows, zh_fill, by_zh = [], [], defaultdict(set)
    for tid, v in sorted(team_map.items()):
        team = v["martj42"] + suffix
        if team not in have:
            report["teams_not_in_db"].append((tid, v["zh"], team))
            continue
        rows.append((f"懂球帝编号 {tid}", team))
        by_zh[v["zh"]].add(team)
        zh_fill.append((team, v["zh"]))
    for zh, teams in sorted(by_zh.items()):
        if len(teams) == 1:
            rows.append((zh, next(iter(teams))))
        else:
            report["ambiguous_zh"].append((zh, sorted(teams)))
    return rows, zh_fill


def write_db(con: sqlite3.Connection, cfg: dict, links: list[Link], names: list[tuple], zh_fill: list[tuple],
             odds: list[tuple]) -> int:
    """在一个事务里写入：中途出错就全部不算。只动来源 dongqiudi-intl 的行；teams.name_zh 只填空着的。
    返回这个来源对上的比赛数。比赛都已经在数据库里（INA 建的），这里不新建、不删比赛，所以不用 source_lifecycle。"""
    source = cfg["db_source"]
    check_nothing_lost(con, source, links, odds)
    with con:
        con.execute("delete from team_names where source = ?", (source,))
        con.executemany("insert into team_names (source, name, team_id) values (?, ?, ?)",
                        [(source, n, t) for n, t in names])
        con.executemany("update teams set name_zh = ? where team_id = ? and coalesce(name_zh, '') = ''",
                        [(zh, t) for t, zh in zh_fill])
        con.execute("delete from odds where source = ?", (source,))
        con.execute("delete from match_sources where source = ?", (source,))
        con.executemany("insert into match_sources (match_id, source, source_key, raw_file, fetched_at_utc) "
                        "values (?, ?, ?, ?, ?)",
                        [(lk.match_id, source, str(lk.d.g.match_id), lk.d.raw_file, lk.d.fetched_at)
                         for lk in sorted(links, key=lambda lk: lk.match_id)])
        con.executemany("insert into odds (source, match_id, bookmaker, market, timing, line, home, draw, away, over, under) "
                        "values (?, ?, ?, '1x2', 'close', null, ?, ?, ?, null, null)",
                        [(source, mid, b, h, d, a) for mid, b, h, d, a in odds])
    return con.execute("select count(*) from match_sources where source = ?", (source,)).fetchone()[0]


# ---------- 全部 ----------
def new_report() -> dict:
    return {"missing_pages": [], "not_played": [], "duplicate_pages": [], "comp_names": {},
            "by": defaultdict(Counter), "unmapped_teams": Counter(), "unmatched": [], "multi": [], "swapped": [],
            "no_score": [], "score_diff": [], "same_db_match": [], "teams_not_in_db": [], "ambiguous_zh": [],
            "odds": Counter(), "unknown_books": Counter(), "books": Counter(), "books_per_avg": defaultdict(list)}


def run(con: sqlite3.Connection, settings: dict, raw: Path | None = None, teams_file: Path | None = None) -> dict:
    """读原始层和队名对照 → 写数据库。返回报告用的数。"""
    cfg = settings["dongqiudi_intl"]
    comp = cfg["competition"]
    registered = {c for g in settings["leagues"].values() for c in g}
    if comp not in registered:
        raise ValueError(f"{comp} 没有登记在 settings.yaml 的 leagues 里（INA 合并以后才有）")
    raw = raw or resolve_path(settings["paths"]["raw"])
    create_tables(con)
    if not con.execute("select 1 from matches where competition = ? limit 1", (comp,)).fetchone():
        raise ValueError(f"数据库里还没有 {comp} 的比赛：先导入 martj42 的比分（INA）")
    report = new_report()
    games = load_games(raw, cfg, settings, con, report)
    if not games:
        raise ValueError("原始层里没有国际赛赛季页，先运行 src.collect.dongqiudi_intl 下载")
    team_map = load_team_map(teams_file or resolve_path(cfg["teams_file"]))
    links = link_games(games, team_map, db_matches(con, cfg), cfg, report)
    odds = build_odds(links, raw, settings, report)
    names, zh_fill = name_rows(con, team_map, cfg, report)
    report["matches"] = write_db(con, cfg, links, names, zh_fill, odds)
    report["odds_rows"] = len(odds)
    return report


def content_digest(con: sqlite3.Connection, cfg: dict) -> str:
    """来源 dongqiudi-intl 在数据库里的全部内容（来源、赔率、队名对照、这些球队的中文名）→ 一串指纹。
    两次导入指纹一样 = 结果完全一样。"""
    src = cfg["db_source"]
    h = hashlib.sha256()
    for sql in ("select * from match_sources where source = ? order by match_id, source_key",
                "select * from odds where source = ? order by match_id, bookmaker",
                "select * from team_names where source = ? order by name",
                "select * from teams where team_id in (select team_id from team_names where source = ?) order by team_id"):
        for r in con.execute(sql, (src,)):
            h.update(repr(r).encode("utf-8"))
    return h.hexdigest()


# ---------- 报告 ----------
COLUMNS = ["比赛", "对上", "主客对调", "比分不一样", "对不上", "对上不止一场", "队名对照里没有", "两场对到同一场",
           "导入", "有多家平均", "有平博胜平负", "去掉的开球后价格"]


def _table(title: str, groups: dict[str, Counter]) -> list[str]:
    head = [title, *COLUMNS[:9], "多家平均比例", "有平博胜平负", "平博比例", "去掉的开球后价格（组）"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, c in groups.items():
        n = c["导入"]
        ratio = (lambda k: f"{c[k] / n:.3f}" if n else "")
        lines.append("| " + " | ".join([str(key), *(str(c[k]) for k in COLUMNS[:9]), ratio("有多家平均"),
                                        str(c["有平博胜平负"]), ratio("有平博胜平负"), str(c["去掉的开球后价格"])]) + " |")
    return lines


def report_markdown(rep: dict, cfg: dict, digests: list[str]) -> str:
    """导入报告（中文）：按年、按赛事。比例的分母是“导入”（对上、比分一样）的场数。"""
    by_year, by_comp, total = defaultdict(Counter), defaultdict(Counter), Counter()
    for (y, code), c in sorted(rep["by"].items()):
        by_year[y].update(c)
        by_comp[f"{code} {rep['comp_names'].get(code, '')}".strip()].update(c)
        total.update(c)
    lines = ["# 国际赛懂球帝赔率导入报告（自动生成）", "",
             f"来源名 {cfg['db_source']}；开球在 {cfg['kickoff_from']} 到 {cfg['kickoff_to']}（UTC）的比赛。"
             f"导入 {rep['matches']} 场，赔率 {rep['odds_rows']} 行（胜平负收盘，含多家平均）。", "",
             "“对上”= 两队一样、日期差 ≤ 1 天的 martj42 比赛正好一场；“导入”= 对上而且比分一样（两场对到同一场的去掉）。"
             "比例的分母是“导入”。懂球帝比分含加时（和 martj42 一样），所以加时分胜负的比赛靠比分找不出来。", ""]
    if digests:
        same = "一样" if len(set(digests)) == 1 else "不一样（有问题）"
        lines += [f"重跑核对：同一批原始文件连着导 {len(digests)} 次，内容指纹{same}：{digests[-1][:16]}…", ""]
    lines += ["## 按年（开球那一年）", "", *_table("年", {**by_year, "合计": total}), "",
              "## 按赛事", "", *_table("赛事", {**by_comp, "合计": total}), ""]
    if rep["missing_pages"]:
        lines += ["原始层里没有的赛季页（这些赛季没读）：" + "、".join(rep["missing_pages"]), ""]

    def section(title, items, fmt):
        out = ["", f"## {title}（{len(items)} 场）", ""]
        return out + ([fmt(x) for x in items] or ["（没有）"])
    desc = (lambda x: f"{x[0]} {x[1]} 懂球帝 {x[2]}：{x[3]} {x[4]} {x[5]}-{x[6]} {x[7]}")
    lines += section("主客对调（赔率主胜、客胜对调了）", rep["swapped"],
                     lambda x: f"- {desc(x)} → 数据库 {x[8]}：{x[9]} {x[10]} - {x[11]}")
    lines += section("比分和 martj42 不一样（不导赔率）", rep["score_diff"],
                     lambda x: f"- {desc(x)} → 数据库 {x[8]}：{x[9]} {x[10]} {x[11]}-{x[12]} {x[13]}")
    lines += section("martj42 没有比分（不导赔率）", rep["no_score"], lambda x: f"- {desc(x)} → 数据库 {x[8]}")
    lines += section("对不上（不导）", rep["unmatched"], lambda x: f"- {desc(x)}")
    lines += section("对上不止一场（不导）", rep["multi"], lambda x: f"- {desc(x)} → 数据库 {x[8]}")
    lines += section("两场懂球帝比赛对到数据库同一场（都不导）", rep["same_db_match"],
                     lambda x: f"- {x[0]} {x[1]} 懂球帝 {x[2]} → 数据库 {x[3]}")
    lines += ["", f"## 队名对照里没有的懂球帝球队（{len(rep['unmapped_teams'])} 个，这些队的比赛不导）", ""]
    lines += [f"- {tid or '（没有编号）'} {name}：{n} 场" for (tid, name), n in rep["unmapped_teams"].most_common()] or ["（没有）"]
    if rep["teams_not_in_db"]:
        lines += ["", "队名对照里有、数据库 teams 表里没有的球队（不写队名对照）："]
        lines += [f"- {tid} {zh} → {team}" for tid, zh, team in rep["teams_not_in_db"]]
    if rep["ambiguous_zh"]:
        lines += ["", "同一个中文名对到两个球队（不写这个中文名）："]
        lines += [f"- {zh}：{teams}" for zh, teams in rep["ambiguous_zh"]]
    if rep["duplicate_pages"]:
        lines += ["", "同一场在两个赛季页里都有（只用第一个）："]
        lines += [f"- 懂球帝 {mid}：{a}、{b}" for mid, a, b in rep["duplicate_pages"]]
    lines += ["", f"下载范围里没踢完或没比分的比赛（不导）：{len(rep['not_played'])} 场"]
    lines += [f"- {c} {lb} {d} {h} - {a}（{s}）" for c, lb, d, h, a, s in rep["not_played"]]
    lines += ["", "## 赔率处理", "",
              "每家公司的“最新”胜平负，变盘时间早于懂球帝开球时刻的才要；多家平均 = 价格正常、早于开球、抽水在正常范围的公司"
              f"至少 {cfg['min_books_average']} 家时的算术平均（和韩K 一样）。", ""]
    lines += [f"- {k}：{v}" for k, v in sorted(rep["odds"].items())]
    lines += ["", "各公司进库的场数（换过名字的写法）："]
    lines += [f"- {b}：{n}" for b, n in rep["books"].most_common()]
    if rep["unknown_books"]:
        lines += ["", "公司对照表里没有的公司（照原样用懂球帝的名字）："]
        lines += [f"- {b}：{n} 组" for b, n in rep["unknown_books"].most_common()]
    return "\n".join(lines) + "\n"


# ---------- 队名对照候选（只列出来给人看，不写文件、不写数据库） ----------
def suggest_teams(games: list[DqdGame], results_csv: Path) -> list[dict]:
    """按“日期差 ≤ 1 天、比分一样”给每个懂球帝球队配 martj42 英文名候选（需求文档：程序配候选，Claude 逐个看过再写）。

    results_csv 是 martj42 的 results.csv（列 date, home_team, away_team, home_score, away_score, ...）。
    每场懂球帝比赛找 martj42 里日期差 ≤ 1 天、比分一样（主客顺着或对调）的比赛；找到 k 场，每场给两队各投 1/k 票。
    第二遍只留和两队第一遍最高票名字都一致的候选，数“确认的场数”（这一场只剩一场候选，而且两队都对得上）。
    """
    by_day = defaultdict(list)
    with open(results_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
                continue
            by_day[r["date"]].append((r["home_team"], r["away_team"], int(float(r["home_score"])),
                                      int(float(r["away_score"]))))

    def cands(g: StateGame) -> list[tuple[str, str]]:
        """→ [(懂球帝主队对应的 martj42 名字, 懂球帝客队对应的)]。"""
        day = date.fromisoformat(g.day_bj)
        out = []
        for k in (-1, 0, 1):
            for h, a, hg, ag in by_day.get(date.fromordinal(day.toordinal() + k).isoformat(), []):
                if (hg, ag) == (g.home_goals, g.away_goals):
                    out.append((h, a))
                if (ag, hg) == (g.home_goals, g.away_goals):
                    out.append((a, h))
        return sorted(set(out))

    votes, names, n_games = defaultdict(Counter), defaultdict(Counter), Counter()
    per_game = []
    for d in games:
        g = d.g
        c = cands(g)
        per_game.append((g, c))
        for tid, zh in ((g.home_id, g.home), (g.away_id, g.away)):
            names[tid][zh] += 1
            n_games[tid] += 1
        for h, a in c:
            votes[g.home_id][h] += 1 / len(c)
            votes[g.away_id][a] += 1 / len(c)
    best = {tid: v.most_common(1)[0][0] for tid, v in votes.items() if v}
    confirmed = Counter()
    for g, c in per_game:
        keep = [(h, a) for h, a in c if best.get(g.home_id) == h and best.get(g.away_id) == a]
        if len(keep) == 1:
            confirmed[g.home_id] += 1
            confirmed[g.away_id] += 1
    out = []
    for tid in sorted(n_games, key=lambda t: (names[t].most_common(1)[0][0], t)):
        top = votes[tid].most_common(2)
        out.append({"dqd_id": tid, "zh": names[tid].most_common(1)[0][0], "zh_all": sorted(names[tid]),
                    "games": n_games[tid], "best": top[0][0] if top else "", "best_votes": round(top[0][1], 2) if top else 0,
                    "second": top[1][0] if len(top) > 1 else "", "second_votes": round(top[1][1], 2) if len(top) > 1 else 0,
                    "confirmed": confirmed[tid]})
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="国际赛懂球帝赔率导入数据库")
    ap.add_argument("--db", default=None, help="数据库文件，默认 settings 里的")
    ap.add_argument("--twice", action="store_true", help="连着导两次，核对两次的内容指纹一样（重跑不变）")
    ap.add_argument("--suggest", metavar="RESULTS_CSV", help="只列队名对照候选（martj42 的 results.csv），不写数据库")
    a = ap.parse_args(argv)
    settings = load_settings()
    cfg = settings["dongqiudi_intl"]
    con = connect(Path(a.db)) if a.db else connect()
    if a.suggest:
        rep = new_report()
        games = load_games(resolve_path(settings["paths"]["raw"]), cfg, settings, con, rep)
        print("懂球帝编号\t中文名\t场数\t最高票\t票数\t第二\t票数\t确认场数\t别的中文写法")
        for r in suggest_teams(games, Path(a.suggest)):
            print(f"{r['dqd_id']}\t{r['zh']}\t{r['games']}\t{r['best']}\t{r['best_votes']}\t{r['second']}\t"
                  f"{r['second_votes']}\t{r['confirmed']}\t{'、'.join(n for n in r['zh_all'] if n != r['zh'])}")
        return
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
