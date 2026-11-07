r"""竞彩联赛（荷甲、葡超、苏超、澳超、巴甲……settings.yaml 的 leagues.jc）从懂球帝原始文件导入数据库
（需求文档 6.2“竞彩联赛”、第 14 节“竞彩价格偏离虚拟实盘”、第 17 节）。

运行：.venv\Scripts\python -m src.clean.dongqiudi_jc_import          （几秒）
每小时任务（src/jobs/refresh_data.py）采集完也会自动跑一次。可以反复运行：每次都从原始层整批重导，比赛编号不变。

为什么单独一个导入程序：五大联赛的懂球帝导入（dongqiudi_import）要先有 football-data 的比赛，再“投票”对上队名；
这几个联赛数据库里本来一场都没有，只能直接按懂球帝赛程建比赛。它们只给“竞彩价格偏离虚拟实盘”用，
不进模型、预测、网页（settings.yaml 的 leagues.jc 不在 top / second 里）。

读的原始文件（每小时任务存的，和五大联赛同一个文件夹，联赛代码不一样）：
  data/raw/dongqiudi/fixtures/<联赛>/*.json.gz            赛程接口（开球时间、两队编号和中文名、状态、比分、第几轮）
  data/raw/dongqiudi/odds_live/<联赛>/<比赛编号>/*.json.gz  赛前赔率快照
  data/raw/dongqiudi/odds_final/<联赛>/<赛季>/<比赛编号>.json.gz  踢完以后的最终赔率

做的事：
1. 赛程：每场比赛取最后一次看到的样子（和五大联赛一样，后抓的覆盖先抓的）。
   赛季按开球时间算：calendar_year 里的联赛（巴甲）= 开球那年；别的 7 月起算新赛季。
   只导上线赛季（season_role = live）。
2. 球队：认懂球帝的球队编号，球队编号（teams.team_id）= 这个懂球帝编号最近一次的中文写法，国家按联赛记。
   - 两个懂球帝编号最近的写法一样 → 报错（不合并）。
   - 这个中文名已经被别的来源用了、或者数据库里有同名但国家不同的队 → 报错（不合并）。
   - 各个写法 → 球队编号写进 team_names（来源 dongqiudi-jc）。
   - 球队改了中文名：已经有的比赛按懂球帝比赛编号找到原来那一行，改主客队，比赛编号不变。
3. 比赛：日期、开球时间、状态和五大联赛的懂球帝新增比赛一样（dongqiudi_import.fixture_fields：英国日期，
   时间待定的不写开球时间、日期往前挪几天）。round = 懂球帝的第几轮（苏超分组以后同一对队一季碰 3–4 次，唯一键要靠它分开），leg = 0。
   同一个（联赛、赛季、主队、客队、轮次）懂球帝有两个编号时，留最后看到的那个，别的不导入、记进报告。
4. 比分：踢完、有比分的写进 match_stats（来源 dongqiudi-jc）。
5. 赔率：变盘记录（odds_changes）、竞彩赔率（lottery_odds）和五大联赛完全一样的读法（dongqiudi_import.build_odds），
   只是来源名换成 dongqiudi-jc。没有 odds 表的行（没有 football-data 那种收盘价）。
6. 一个事务写进去：只动来源 dongqiudi-jc 的行，别的来源一行都不碰。
   写之前核对：数据库里这个来源已经有赔率的比赛，这次还都在（缺了原始文件就报错，一行不动）。

导入报告写到 data/reports/竞彩联赛_导入报告.md（每个联赛几场、踢完几场、有平博亚盘 / 大小球 / 竞彩价格的场数）。
"""
import argparse
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.clean.dongqiudi_import import (TIME_FORMAT, Ref, build_odds, fixture_fields, load_bookmakers, load_fetch_times,
                                        load_fixtures, odds_files, result_letter)
from src.clean.schema import create_tables
from src.clean.source_lifecycle import (delete_replaced_results, new_match_ids, reconcile_source_lifecycle,
                                        snapshot_prediction_sources)
from src.common.config import load_settings, resolve_path, season_role
from src.common.db import connect

log = logging.getLogger(__name__)

PINNACLE = "平博 Pinnacle"


def season_of(league: str, when: datetime, calendar_year: list[str]) -> int:
    """这场比赛属于哪个赛季（开始年份）。一年一个赛季的联赛 = 开球那年；别的 7 月起算新赛季。"""
    if league in calendar_year:
        return when.year
    return when.year if when.month >= 7 else when.year - 1


@dataclass
class Game:
    """一场要导入的比赛（球队编号换好以后）。"""
    dqd_id: int
    league: str
    season: int
    round: str
    home: str
    away: str
    fx: object         # dongqiudi_reader.Fixture（最后一次看到的样子）
    raw_file: str
    fetched_at: str


def load_games(raw: Path, leagues: list[str], fetched: dict, cfg: dict, settings: dict, report: dict) -> list:
    """第 1 步：赛程 → [(联赛, 赛季, Seen)]，只要上线赛季的。"""
    seen = load_fixtures(raw, leagues, fetched)
    out = []
    for s in seen.values():
        when = s.fx.kickoff_utc or datetime.strptime(s.fetched_at, TIME_FORMAT).replace(tzinfo=timezone.utc)
        season = season_of(s.league, when, cfg.get("calendar_year") or [])
        if season_role(season, settings) != "live":
            report["跳过"]["不是上线赛季"] += 1
            continue
        out.append((s.league, season, s))
    return out


def team_ids(rows: list, cfg: dict) -> tuple[dict, list[tuple], dict]:
    """第 2 步：懂球帝球队编号 → 我们的球队编号（最近一次的中文写法）。
    返回（{(联赛, 懂球帝球队编号): 球队编号}, team_names 的行 [(写法, 球队编号)], {球队编号: 国家}）。"""
    latest = {}                       # (联赛, 懂球帝编号) → (抓取时间, 中文名)
    names = defaultdict(set)          # (联赛, 懂球帝编号) → 见过的全部写法
    for lg, _season, s in rows:
        for tid, name in ((s.fx.home_id, s.fx.home), (s.fx.away_id, s.fx.away)):
            if not tid or not name:
                raise ValueError(f"{lg} 懂球帝比赛 {s.fx.match_id} 缺球队编号或队名（{s.raw_file}），不导入")
            key = (lg, tid)
            names[key].add(name)
            if key not in latest or s.fetched_at >= latest[key][0]:
                latest[key] = (s.fetched_at, name)
    by_key = {k: v[1] for k, v in latest.items()}
    country = {}
    owner = {}                        # 球队编号 → (联赛, 懂球帝编号)
    for (lg, tid), team in by_key.items():
        if team in owner and owner[team][1] != tid:
            raise ValueError(f"两个懂球帝球队编号的中文名都是“{team}”（{owner[team]}、{(lg, tid)}），不合并")
        c = cfg["country"][lg]
        if country.get(team, c) != c:
            raise ValueError(f"球队“{team}”在两个国家的联赛里都出现了（{country[team]}、{c}），不合并")
        owner[team], country[team] = (lg, tid), c
    name_rows = {}
    for key, ns in names.items():
        for n in ns:
            if name_rows.get(n, by_key[key]) != by_key[key]:
                raise ValueError(f"写法“{n}”对到两支球队（{name_rows[n]}、{by_key[key]}），不导入")
            name_rows[n] = by_key[key]
    return by_key, sorted(name_rows.items()), country


def build_games(rows: list, ids: dict, report: dict) -> list[Game]:
    """第 3 步：换好球队编号；同一场比赛（联赛、赛季、主客队、轮次）懂球帝有两个编号时留最后看到的。"""
    groups = defaultdict(list)
    for lg, season, s in rows:
        g = Game(s.fx.match_id, lg, season, s.fx.gameweek or "", ids[(lg, s.fx.home_id)], ids[(lg, s.fx.away_id)],
                 s.fx, s.raw_file, s.fetched_at)
        groups[(lg, season, g.home, g.away, g.round)].append(g)
    out = []
    for key, gs in groups.items():
        gs = sorted(gs, key=lambda g: (g.fetched_at, g.dqd_id), reverse=True)
        out.append(gs[0])
        for g in gs[1:]:
            report["跳过"]["同一场比赛懂球帝有两个编号（留最后看到的）"] += 1
            report["重复"].append(f"{key[0]} {key[1]} {key[2]} - {key[3]} 第{key[4]}轮：懂球帝 {g.dqd_id}（留 {gs[0].dqd_id}）")
    return sorted(out, key=lambda g: (g.league, g.season, g.dqd_id))


def kickoff_refs(games: list[Game], back_days: int, tz: ZoneInfo) -> dict[int, Ref]:
    """推变盘时间的年份、判断“开球后的价格”要用的开球时间（和五大联赛新增比赛一样：没有准确时间用日期中午）。"""
    out = {}
    for g in games:
        if g.fx.kickoff_utc and g.fx.status != "Uncertain":
            out[g.dqd_id] = Ref(g.fx.kickoff_utc, True)
        else:
            d = date.fromisoformat(fixture_fields(g.fx, back_days, tz)[0])
            out[g.dqd_id] = Ref(datetime.combine(d, datetime.min.time(), timezone.utc) + timedelta(hours=12), False)
    return out


# ---------- 写数据库 ----------
def check_teams(con, source: str, country: dict) -> None:
    """球队编号已经被别的来源用了（五大联赛、日职……的队），或者同名但国家不同 → 报错，不合并。"""
    used = sorted(country)
    if not used:
        return
    ph = ",".join("?" * len(used))
    other = con.execute(f"select distinct team_id, source from team_names where team_id in ({ph}) and source != ?",
                        (*used, source)).fetchall()
    if other:
        raise ValueError(f"这些球队编号别的来源已经在用（不合并）：{other[:10]}")
    clash = [(t, c) for t, c in con.execute(f"select team_id, country from teams where team_id in ({ph})", used)
             if c != country[t]]
    if clash:
        raise ValueError(f"这些球队编号数据库里已经有了，国家不一样（不合并）：{clash[:10]}")


def check_nothing_lost(con, source: str, games: list[Game]) -> None:
    """写入是“先删掉这个来源的全部行再重写”。数据库里已经有变盘记录的比赛，这次原始层里要是没有了
    （比如复制文件时漏了一个联赛的文件夹），重写以后那几场的赔率就悄悄没了。所以发现少了就报错，一行都不动。"""
    have = {str(g.dqd_id) for g in games}
    old = {k for (k,) in con.execute(
        "select distinct s.source_key from odds_changes o join match_sources s "
        "on s.match_id = o.match_id and s.source = o.source where o.source = ?", (source,))}
    lost = sorted(old - have)
    if lost:
        raise ValueError(f"这些比赛数据库里有赔率，这次原始层里没有了（缺赛程文件？）：{len(lost)} 场，"
                         f"懂球帝编号 {lost[:10]}…。不导入（不然会删掉这些赔率）")


def write_db(con, cfg: dict, games: list[Game], name_rows: list[tuple], country: dict, changes: list, lottery: list,
             back_days: int, tz: ZoneInfo) -> dict:
    """在一个事务里写完：中途出错就全部不算。只动来源 dongqiudi-jc 的行。返回 {懂球帝比赛编号: 我们的比赛编号} 等。"""
    source = cfg["db_source"]
    check_teams(con, source, country)
    check_nothing_lost(con, source, games)
    with con:
        con.execute("begin immediate")   # 一开始就占住写锁，免得中途和采集程序抢
        previous = snapshot_prediction_sources(con)
        con.executemany("insert into teams (team_id, name_zh, country) values (?, ?, ?) "
                        "on conflict(team_id) do update set name_zh = excluded.name_zh, country = excluded.country",
                        [(t, t, c) for t, c in sorted(country.items())])
        con.execute("delete from team_names where source = ?", (source,))
        con.executemany("insert into team_names (source, name, team_id) values (?, ?, ?)",
                        [(source, n, t) for n, t in name_rows])
        # 比赛：这个懂球帝编号以前导过的，按原来的比赛编号更新（球队改了中文名也不换编号）；没导过的新增
        known = {int(k): mid for k, mid in con.execute(
            "select s.source_key, s.match_id from match_sources s join matches m on m.match_id = s.match_id "
            "where s.source = ?", (source,))}
        new_ids = new_match_ids(con)
        ids = {}
        for g in games:
            mdate, ko, status, prec = fixture_fields(g.fx, back_days, tz)
            values = (g.league, g.season, g.home, g.away, mdate, ko, status, g.round, prec)
            if g.dqd_id in known:
                con.execute("update matches set competition = ?, season = ?, home_team = ?, away_team = ?, match_date = ?,"
                            " kickoff_utc = ?, status = ?, round = ?, date_precision = ? where match_id = ?",
                            (*values, known[g.dqd_id]))
                ids[g.dqd_id] = known[g.dqd_id]
                continue
            con.execute(
                "insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                " status, round, leg, date_precision) values (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)"
                " on conflict(competition, season, home_team, away_team, round, leg) do update set"
                " match_date = excluded.match_date, kickoff_utc = excluded.kickoff_utc,"
                " status = excluded.status, date_precision = excluded.date_precision",
                (next(new_ids), *values))
            ids[g.dqd_id] = con.execute(
                "select match_id from matches where competition = ? and season = ? and home_team = ? and away_team = ?"
                " and round = ? and leg = 0", (g.league, g.season, g.home, g.away, g.round)).fetchone()[0]
        con.execute("delete from match_sources where source = ?", (source,))
        con.executemany("insert into match_sources (match_id, source, source_key, raw_file, fetched_at_utc)"
                        " values (?, ?, ?, ?, ?)",
                        [(ids[g.dqd_id], source, str(g.dqd_id), g.raw_file, g.fetched_at) for g in games])
        delete_replaced_results(con, source)
        # 比分：只写踢完、有比分的（和五大联赛新增比赛一样：说踢完了但没比分的当还没踢）
        done = [g for g in games if fixture_fields(g.fx, back_days, tz)[2] == "finished"]
        con.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)",
                        [(ids[g.dqd_id], source, g.fx.home_goals, g.fx.away_goals,
                          result_letter(g.fx.home_goals, g.fx.away_goals)) for g in done])
        # 变盘记录、竞彩赔率：build_odds 的行第 2 列是来源名（五大联赛的 dongqiudi），换成这个来源
        con.execute("delete from odds_changes where source = ?", (source,))
        con.executemany("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw,"
                        " away, over, under, changed_at_utc, fetched_at_utc, raw_file)"
                        " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(ids[r[0]], source, *r[2:]) for r in changes if r[0] in ids])
        con.execute("delete from lottery_odds where source = ?", (source,))
        con.executemany("insert into lottery_odds (match_id, source, product, handicap, home, draw, away,"
                        " fetched_at_utc, raw_file) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(ids[r[0]], source, *r[2:]) for r in lottery if r[0] in ids])
        lifecycle = reconcile_source_lifecycle(con, previous)
    return {"ids": ids, "finished": len(done), "lifecycle": lifecycle}


# ---------- 报告 ----------
def league_table(con, source: str, leagues: dict, products: list[str]) -> list[dict]:
    """每个联赛：几场、踢完几场、有平博亚盘、平博大小球、竞彩价格（products 里的，北单不算）的各几场。"""
    out = []
    ph = ",".join("?" * len(products))
    for code, name in leagues.items():
        n, done, ah, ou, jc = con.execute(
            "select count(*), sum(m.status = 'finished'),"
            " sum(exists (select 1 from odds_changes o where o.match_id = m.match_id and o.source = s.source"
            "             and o.bookmaker = ? and o.market = 'ah')),"
            " sum(exists (select 1 from odds_changes o where o.match_id = m.match_id and o.source = s.source"
            "             and o.bookmaker = ? and o.market = 'ou')),"
            f" sum(exists (select 1 from lottery_odds l where l.match_id = m.match_id and l.source = s.source"
            f"             and l.product in ({ph})))"
            " from matches m join match_sources s on s.match_id = m.match_id and s.source = ? where m.competition = ?",
            (PINNACLE, PINNACLE, *products, source, code)).fetchone()
        out.append({"联赛": f"{name}（{code}）", "比赛": n, "踢完": done or 0, "有平博亚盘": ah or 0,
                    "有平博大小球": ou or 0, "有竞彩价格": jc or 0})
    return out


def report_markdown(table: list[dict], rep: dict, counts: dict, source: str) -> str:
    lines = ["# 竞彩联赛导入报告", "",
             f"来源名 {source}。比赛直接按懂球帝赛程建，只有上线赛季；只给“竞彩价格偏离虚拟实盘”用"
             "（需求文档 6.2“竞彩联赛”、第 14 节）。", "",
             "| 联赛 | 比赛 | 踢完 | 有平博亚盘 | 有平博大小球 | 有竞彩价格 |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['联赛']} | {r['比赛']} | {r['踢完']} | {r['有平博亚盘']} | {r['有平博大小球']} | {r['有竞彩价格']} |"
              for r in table]
    lines += ["", "## 计数", ""] + [f"- {k}：{v}" for k, v in counts.items()]
    if rep["跳过"]:
        lines += ["", "## 没导入的", ""] + [f"- {k}：{v} 场" for k, v in rep["跳过"].items()]
    if rep["重复"]:
        lines += ["", "## 懂球帝有两个编号的比赛", ""] + [f"- {x}" for x in rep["重复"]]
    if rep["没对照的公司"]:
        lines += ["", "## 没在 bookmakers_dongqiudi.yaml 里的公司（原样保留）", ""] + \
                 [f"- {k}：{v} 行" for k, v in rep["没对照的公司"].most_common()]
    return "\n".join(lines) + "\n"


# ---------- 全部 ----------
def run(con, raw: Path, reports: Path, settings: dict) -> dict:
    """导入一次（测试里直接调用这个，给临时数据库和临时文件夹）。返回 {'counts': 计数, 'ids': {懂球帝编号: 比赛编号}}。"""
    started = time.perf_counter()
    cfg, dq, q = settings["dongqiudi_jc"], settings["dongqiudi"], settings["quality"]
    leagues = list(cfg["collect"]["league_ids"])
    back_days = int(dq["uncertain_back_days"])
    tz = ZoneInfo(settings["features"]["date_only_timezone"])
    books, products = load_bookmakers(resolve_path(settings["paths"]["bookmakers_dongqiudi"]))
    create_tables(con)
    con.commit()
    fetched = load_fetch_times(con)
    rep = {"跳过": Counter(), "重复": [], "没对照的公司": Counter()}
    rows = load_games(raw, leagues, fetched, cfg, settings, rep)
    ids, name_rows, country = team_ids(rows, cfg)
    games = build_games(rows, ids, rep)
    refs = kickoff_refs(games, back_days, tz)
    changes, lottery, odds_stats, unknown = build_odds(odds_files(raw, leagues, fetched), refs, books, products, q)
    rep["没对照的公司"] = unknown
    written = write_db(con, cfg, games, name_rows, country, changes, lottery, back_days, tz)
    counts = {"比赛": len(games), "踢完（写了比分）": written["finished"], "球队": len(country),
              "变盘记录（行）": sum(1 for r in changes if r[0] in written["ids"]),
              "竞彩赔率（行）": sum(1 for r in lottery if r[0] in written["ids"]),
              "删掉的没有来源的比赛": written["lifecycle"]["deleted"],
              "有预测而保留的缺源比赛": written["lifecycle"]["missing"],
              **{k: v for k, v in odds_stats.items() if k.startswith(("去掉", "文件：比赛没对上"))},
              "用时（秒）": round(time.perf_counter() - started, 1)}
    names = {c: settings["leagues"]["jc"].get(c, c) for c in leagues}
    reports.mkdir(parents=True, exist_ok=True)
    (reports / cfg["report_file"]).write_text(
        report_markdown(league_table(con, cfg["db_source"], names, list(settings["lottery_value"]["products"].values())),
                        rep, counts, cfg["db_source"]), encoding="utf-8")
    log.info("竞彩联赛导入：%s", counts)
    return {"counts": counts, "ids": written["ids"]}


def main(argv=None):
    argparse.ArgumentParser(description="竞彩联赛（leagues.jc）从懂球帝原始文件导入数据库").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    s = load_settings()
    con = connect()
    try:
        run(con, resolve_path(s["paths"]["raw"]), resolve_path(s["paths"]["reports"]), s)
    finally:
        con.close()


if __name__ == "__main__":
    main()
