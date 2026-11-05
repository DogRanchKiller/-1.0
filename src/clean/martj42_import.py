r"""国家队比赛比分从 martj42 原始文件导入数据库（需求文档第 4 节“第 11 阶段：国际赛进数据库和开发回测”、第 17 节）。

运行：.venv\Scripts\python -m src.clean.martj42_import                （十几秒）
      .venv\Scripts\python -m src.clean.martj42_import --db 别的数据库文件 --twice
可以反复运行：每次都从原始层整批重导国际赛，比赛编号不变；同一批原始文件导两次，结果完全一样（--twice 核对）。

读的原始文件（下载程序 src/collect/martj42.py 存的；默认用最新日期的那个文件夹）：
  data/raw/martj42/<下载日期>/results.csv     每场一行：date, home_team, away_team, home_score, away_score,
                                               tournament, city, country, neutral（TRUE / FALSE）
  data/raw/martj42/<下载日期>/shootouts.csv   踢了点球大战的比赛：date, home_team, away_team, winner（后面还有一列先罚的队，用不到）

做的事：
1. 只导 settings 里 martj42.last_date（2023-12-31）及以前、比分不空的比赛；每个赛季（= 日期那一年）再按 season_role 查一遍，
   不是开发赛季就报错（考试赛季锁住）。
2. 赛事代码 INT（leagues.intl 里登记）；赛季 = 日期那一年；round = 日期 'YYYY-MM-DD'；leg = 0；
   match_date = martj42 的日期（比赛当地的日期）；kickoff_utc 空着；status = finished；date_precision = 'day_world'。
3. 同一天同一对队（主客不管）两场：报错。只有 settings 的 martj42.same_day_pairs 里登记过的（核对过是真的两场）不报错，
   这几对两场都不导入，列进报告（原因见设置里的说明）。
4. 球队编号 = 英文名 + “ NT”，国家写英文名，中文名空着（已经填了的不动）。和已有编号重复（不是 martj42 建的）就报错。
   team_names 写（martj42, 英文名）→ 编号。
5. 比分进 match_stats（来源 martj42）：比分含加时、不含点球；胜平负照比分算。
6. 点球大战进 cup_details（来源 martj42）：这一场比分是平局的，extra_time = 1、加时比分 = 这个比分、点球比分空着、note 写点球胜者。
   比分不是平局的（两回合总比分打平、第二回合踢点球）：martj42 分不出踢没踢加时，不写 cup_details，列进报告。
   对不上 results.csv 的点球大战（比赛不在、或者不导入）：不写，列进报告。
7. 中立场、城市、国家、赛事名进 match_venue（来源 martj42）。
8. 一个事务写进去：只动来源 martj42 的行，别的来源一行都不碰。数据库里已有的国际赛赛季这次原始层里没有了，就报错、一行都不动。

导入报告写到 data/reports/国际赛_比分导入报告.md（每十年场数、友谊赛 / 正式比赛、中立场、点球大战、球队数等）。
"""
import argparse
import hashlib
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.clean.dongqiudi_import import TIME_FORMAT, rel, result_letter
from src.clean.schema import create_tables
from src.clean.source_lifecycle import (delete_replaced_results, new_match_ids, reconcile_source_lifecycle,
                                        snapshot_prediction_sources)
from src.common.config import load_settings, resolve_path, season_role
from src.common.db import connect

log = logging.getLogger(__name__)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RESULT_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"]
SHOOTOUT_COLUMNS = ["date", "home_team", "away_team", "winner"]
FRIENDLY = "Friendly"          # martj42 友谊赛的赛事名
NO_SCORE = {"", "NA"}          # 还没踢的比赛比分是空的（读成文字时可能是空字符串或 NA）


@dataclass
class Game:
    """一场要导入的国家队比赛（martj42 的一行）。"""
    date: str                 # 'YYYY-MM-DD'，比赛当地的日期
    home_name: str            # martj42 的英文名
    away_name: str
    home_goals: int
    away_goals: int
    tournament: str
    city: str | None
    country: str | None
    neutral: int              # 1 = 中立场
    line: int                 # 在 results.csv 里是第几行（报错时好找）

    @property
    def season(self) -> int:
        return int(self.date[:4])

    @property
    def pair(self) -> tuple:
        """同一天同一对队（主客不管）的认法。"""
        return (self.date, *sorted((self.home_name, self.away_name)))


# ---------- 找原始文件 ----------
def latest_folder(raw: Path, cfg: dict) -> Path:
    """data/raw/martj42/ 下面日期最新、两个文件都在的文件夹。一个都没有就报错。"""
    base = raw / cfg["raw_folder"]
    folders = sorted(p for p in base.glob("*") if p.is_dir() and DATE_RE.match(p.name)) if base.exists() else []
    good = [p for p in folders if all((p / f"{n}.csv").exists() for n in cfg["files"])]
    if not good:
        raise ValueError(f"{base} 下面没有 results.csv、shootouts.csv 都在的日期文件夹，先运行 src.collect.martj42 下载")
    return good[-1]


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """按文字读（不让 pandas 猜类型、不把 NA 当空值），检查前几列是不是预期的列名。"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(df.columns[:len(columns)]) != columns:
        raise ValueError(f"{path.name} 的列不对：应该以 {columns} 开头，实际是 {list(df.columns)}")
    for c in columns:
        df[c] = df[c].str.strip()
    return df


# ---------- 读比分 ----------
def read_results(path: Path, cfg: dict, settings: dict, report: dict) -> list[Game]:
    """results.csv → 要导入的比赛。超过 last_date 的、比分空的不要；写法不对的直接报错（不猜）。"""
    df = read_csv(path, RESULT_COLUMNS)
    last = str(cfg["last_date"])
    games, roles = [], {}
    for i, r in enumerate(df.itertuples(index=False), start=2):     # 第 1 行是列名
        if not DATE_RE.match(r.date):
            raise ValueError(f"results.csv 第 {i} 行日期写法不对：{r.date!r}")
        datetime.strptime(r.date, "%Y-%m-%d")                         # 不存在的日子（例如 2 月 30 日）这里报错
        if r.date > last:
            report["after_last_date"] += 1
            continue
        if r.home_score in NO_SCORE or r.away_score in NO_SCORE:
            report["no_score"] += 1
            continue
        if not (r.home_score.isdigit() and r.away_score.isdigit()):
            raise ValueError(f"results.csv 第 {i} 行比分不是整数：{r.home_score!r}-{r.away_score!r}")
        if r.neutral not in ("TRUE", "FALSE"):
            raise ValueError(f"results.csv 第 {i} 行 neutral 不是 TRUE / FALSE：{r.neutral!r}")
        if not (r.home_team and r.away_team and r.tournament):
            raise ValueError(f"results.csv 第 {i} 行球队或赛事是空的")
        if r.home_team == r.away_team:
            raise ValueError(f"results.csv 第 {i} 行主客队是同一支：{r.home_team}")
        g = Game(r.date, r.home_team, r.away_team, int(r.home_score), int(r.away_score), r.tournament,
                 r.city or None, r.country or None, 1 if r.neutral == "TRUE" else 0, i)
        # 开发赛季才导：程序里按 season_role 再查一遍（last_date 设错了也不会把考试赛季导进来）
        role = roles.setdefault(g.season, season_role(g.season, settings))
        if role != "dev":
            raise ValueError(f"{g.season} 赛季是“{role}”，不是开发赛季：考试赛季锁住，不导入（第 {i} 行 {g.date}）")
        games.append(g)
    return games


def drop_same_day_pairs(games: list[Game], cfg: dict, report: dict) -> list[Game]:
    """同一天同一对队（主客不管）出现两场：报错。

    例外：settings 的 martj42.same_day_pairs 里登记过的（Claude 核对过是真的踢了两场，例如 1916 年阿根廷、乌拉圭
    同一天在两个城市各踢一场不同的杯赛）：这一对这一天的几场都不导入，列进报告。
    为什么不导：唯一键靠 round = 日期 分开同一年的几场，同一天的两场分不开；用哪一场都是猜。
    """
    allowed = {(str(d), *sorted((h, a))) for d, h, a in (cfg.get("same_day_pairs") or [])}
    groups: dict[tuple, list[Game]] = {}
    for g in games:
        groups.setdefault(g.pair, []).append(g)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    bad = sorted(set(dup) - allowed)
    if bad:
        lines = "；".join(f"{k[0]} {k[1]} - {k[2]}（第 {'、'.join(str(g.line) for g in dup[k])} 行）" for k in bad[:10])
        raise ValueError(f"同一天同一对队有两场，分不开（不导入）：{len(bad)} 对：{lines}。"
                         "核对过确实是两场的，写进 settings 的 martj42.same_day_pairs")
    report["same_day_skipped"] = [g for k in sorted(dup) for g in dup[k]]
    report["same_day_unused"] = sorted(allowed - set(dup))      # 登记了、这次却没有两场的（列进报告，不报错）
    return [g for g in games if g.pair not in dup]


# ---------- 读点球大战 ----------
def read_shootouts(path: Path, games: list[Game], cfg: dict, report: dict) -> dict[tuple, str]:
    """shootouts.csv → {(日期, 主队, 客队): 点球胜者英文名}，只要能对上这次导入的比赛、而且比分是平局的。

    对不上的（results.csv 里没有这场、或者这场不导入）、比分不是平局的（两回合总比分打平、第二回合踢点球，
    分不出踢没踢加时）都不要，列进报告。同一场写了两次、胜者不是两队之一：报错。
    """
    df = read_csv(path, SHOOTOUT_COLUMNS)
    last = str(cfg["last_date"])
    by_key = {(g.date, g.home_name, g.away_name): g for g in games}
    seen, out = set(), {}
    for i, r in enumerate(df.itertuples(index=False), start=2):
        key = (r.date, r.home_team, r.away_team)
        if key in seen:
            raise ValueError(f"shootouts.csv 第 {i} 行：同一场点球大战写了两次 {key}")
        seen.add(key)
        if r.winner not in (r.home_team, r.away_team):
            raise ValueError(f"shootouts.csv 第 {i} 行：点球胜者 {r.winner!r} 不是两队之一 {key}")
        if r.date > last:
            report["shootouts_after_last_date"] += 1
            continue
        g = by_key.get(key)
        if g is None:
            report["shootouts_unmatched"].append(key)
            continue
        if g.home_goals != g.away_goals:
            report["shootouts_not_draw"].append((*key, g.home_goals, g.away_goals, g.tournament))
            continue
        out[key] = r.winner
    return out


# ---------- 写数据库 ----------
def team_id(name: str, cfg: dict) -> str:
    """球队编号 = 英文名 + 后缀（“ NT”）。"""
    return f"{name}{cfg['team_suffix']}"


def fetched_time(con: sqlite3.Connection, path: Path, source: str) -> str | None:
    """原始文件是什么时候下载的：按文件内容的 sha256 查抓取记录（换了文件夹也认得出）。查不到就空着（不编时间）。"""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = con.execute("select min(fetched_at_utc) from fetch_log where source = ? and sha256 = ? and status = 'saved'",
                      (source, digest)).fetchone()
    if not row or not row[0]:
        return None
    return datetime.fromisoformat(row[0]).astimezone(timezone.utc).strftime(TIME_FORMAT)


def check_teams(con: sqlite3.Connection, cfg: dict, names: list[str]) -> None:
    """球队编号不能和已有的撞：已经有这个编号、又不是 martj42 建的（team_names 里没有 martj42 的写法指向它）就报错。"""
    ids = [team_id(n, cfg) for n in names]
    if len(set(ids)) != len(ids):
        raise ValueError("两个 martj42 英文名加后缀以后编号一样（不合并）")
    ours = {t for (t,) in con.execute("select team_id from team_names where source = ?", (cfg["db_source"],))}
    existing = {t for (t,) in con.execute("select team_id from teams")}
    clash = sorted((set(ids) & existing) - ours)
    if clash:
        raise ValueError(f"这些国家队编号数据库里已经有了，不是 martj42 建的（不合并）：{clash[:20]}")


def check_nothing_lost(con: sqlite3.Connection, cfg: dict, games: list[Game]) -> None:
    """写入前核对：数据库里已有的国际赛赛季，这次原始层里都还在。

    写入是“先删掉这个来源的行再重写”，原始文件要是少了某几年（比如拿错了一份文件），重写以后那几年就悄悄没了。
    所以发现少了就报错，一行都不动（和韩K 一样的保护）。
    """
    old = {y for (y,) in con.execute(
        "select distinct m.season from matches m join match_sources s on s.match_id = m.match_id "
        "where s.source = ? and m.competition = ?", (cfg["db_source"], cfg["competition"]))}
    lost = sorted(old - {g.season for g in games})
    if lost:
        raise ValueError(f"数据库里有这些年份的国际赛，这次原始层里没有：{lost[:20]}{'…' if len(lost) > 20 else ''}"
                         f"（共 {len(lost)} 年）。不导入（不然会删掉这几年）")


def write_db(con: sqlite3.Connection, cfg: dict, games: list[Game], shootouts: dict[tuple, str],
             raw_file: str, fetched_at: str | None) -> int:
    """在一个事务里写入：中途出错就全部不算。只动来源 martj42 的行。返回这个来源的比赛数。"""
    source, comp = cfg["db_source"], cfg["competition"]
    create_tables(con)
    names = sorted({g.home_name for g in games} | {g.away_name for g in games})
    check_teams(con, cfg, names)
    check_nothing_lost(con, cfg, games)
    with con:
        # 球队：国家写英文名；中文名不动（INB 以后按懂球帝对照填，填过的不能被冲掉）
        con.executemany("insert into teams (team_id, name_zh, country) values (?, null, ?) "
                        "on conflict(team_id) do update set country = excluded.country",
                        [(team_id(n, cfg), n) for n in names])
        con.execute("delete from team_names where source = ?", (source,))
        con.executemany("insert into team_names (source, name, team_id) values (?, ?, ?)",
                        [(source, n, team_id(n, cfg)) for n in names])
        # 比赛：有就更新（编号不变），没有才新增。round = 日期，leg = 0
        new_ids = new_match_ids(con)
        con.executemany(
            "insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status, "
            "round, leg, date_precision) values (?, ?, ?, ?, ?, ?, null, 'finished', ?, 0, 'day_world') "
            "on conflict(competition, season, home_team, away_team, round, leg) do update set "
            "match_date = excluded.match_date, kickoff_utc = excluded.kickoff_utc, status = excluded.status, "
            "date_precision = excluded.date_precision",
            [(next(new_ids), comp, g.season, team_id(g.home_name, cfg), team_id(g.away_name, cfg), g.date, g.date)
             for g in games])
        ids = {(h, a, r): mid for mid, h, a, r in con.execute(
            "select match_id, home_team, away_team, round from matches where competition = ?", (comp,))}
        mid = {(g.date, g.home_name, g.away_name): ids[(team_id(g.home_name, cfg), team_id(g.away_name, cfg), g.date)]
               for g in games}

        previous = snapshot_prediction_sources(con)
        con.execute("delete from match_sources where source = ?", (source,))
        con.executemany("insert into match_sources values (?, ?, ?, ?, ?)",
                        [(mid[k], source, "|".join(k), raw_file, fetched_at) for k in mid])
        delete_replaced_results(con, source, tables=("match_stats", "cup_details"))
        con.execute("delete from match_venue where source = ?", (source,))
        con.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)",
                        [(mid[(g.date, g.home_name, g.away_name)], source, g.home_goals, g.away_goals,
                          result_letter(g.home_goals, g.away_goals)) for g in games])
        by_key = {(g.date, g.home_name, g.away_name): g for g in games}
        con.executemany(
            "insert into cup_details (match_id, source, extra_time, aet_home, aet_away, pen_home, pen_away, note) "
            "values (?, ?, 1, ?, ?, null, null, ?)",
            [(mid[k], source, by_key[k].home_goals, by_key[k].away_goals, f"点球大战胜者 {team_id(w, cfg)}")
             for k, w in sorted(shootouts.items())])
        con.executemany("insert into match_venue (match_id, source, neutral, city, country, tournament) "
                        "values (?, ?, ?, ?, ?, ?)",
                        [(mid[(g.date, g.home_name, g.away_name)], source, g.neutral, g.city, g.country, g.tournament)
                         for g in games])
        reconcile_source_lifecycle(con, previous)
        # 公共的清理只管比分、赔率等表，不认识新表 match_venue：比赛被删掉的，场地行也删掉（不留孤儿）
        con.execute("delete from match_venue where match_id not in (select match_id from matches)")
    return con.execute("select count(*) from match_sources where source = ?", (source,)).fetchone()[0]


# ---------- 全部 ----------
def new_report() -> dict:
    return {"after_last_date": 0, "no_score": 0, "same_day_skipped": [], "same_day_unused": [],
            "shootouts_after_last_date": 0, "shootouts_unmatched": [], "shootouts_not_draw": []}


def run(con: sqlite3.Connection, settings: dict, folder: Path | None = None) -> dict:
    """读原始层 → 写数据库。返回报告用的数。folder 不给就用 data/raw/martj42/ 下最新的日期文件夹。"""
    cfg = settings["martj42"]
    comp = cfg["competition"]
    groups = {name for name, codes in settings["leagues"].items() if comp in codes}
    if not groups:
        raise ValueError(f"{comp} 没有登记在 settings.yaml 的 leagues 里")
    if groups & {"top", "second"}:
        raise ValueError(f"{comp} 不能放进 leagues.top / second（现有的下载、预测、网页都按这两组跑）")
    folder = folder or latest_folder(resolve_path(settings["paths"]["raw"]), cfg)
    report = new_report()
    report["folder"] = rel(folder)
    games = read_results(folder / "results.csv", cfg, settings, report)
    if not games:
        raise ValueError(f"{folder / 'results.csv'} 里没有能导入的比赛")
    games = drop_same_day_pairs(games, cfg, report)
    shootouts = read_shootouts(folder / "shootouts.csv", games, cfg, report)
    create_tables(con)
    raw_file = rel(folder / "results.csv")
    fetched = fetched_time(con, folder / "results.csv", cfg["db_source"])
    report["fetched_at"] = fetched
    report["matches"] = write_db(con, cfg, games, shootouts, raw_file, fetched)
    report["games"] = games
    report["shootouts"] = shootouts
    return report


def content_digest(con: sqlite3.Connection, cfg: dict) -> str:
    """国际赛在数据库里的全部内容（比赛、来源、比分、点球大战、场地、队名对照、球队）→ 一串指纹。两次导入指纹一样 = 结果完全一样。"""
    src, comp = cfg["db_source"], cfg["competition"]
    h = hashlib.sha256()
    for sql, params in (
            ("select * from matches where competition = ? order by match_id", (comp,)),
            ("select * from match_sources where source = ? order by match_id", (src,)),
            ("select * from match_stats where source = ? order by match_id", (src,)),
            ("select * from cup_details where source = ? order by match_id", (src,)),
            ("select * from match_venue where source = ? order by match_id", (src,)),
            ("select * from team_names where source = ? order by name", (src,)),
            ("select * from teams where team_id in (select team_id from team_names where source = ?) order by team_id",
             (src,))):
        for r in con.execute(sql, params):
            h.update(repr(r).encode("utf-8"))
    return h.hexdigest()


def summary_tables(con: sqlite3.Connection, cfg: dict) -> dict:
    """从数据库里数（导入以后的样子）：每十年场数、友谊赛 / 正式比赛、中立场、点球大战、球队数、赛事。"""
    src, comp = cfg["db_source"], cfg["competition"]
    base = ("from matches m join match_venue v on v.match_id = m.match_id and v.source = ? "
            "where m.competition = ?")
    p = (src, comp)
    decades = con.execute(
        "select (m.season / 10) * 10, count(*), sum(v.tournament = ?), sum(v.tournament != ?), sum(v.neutral), "
        "sum(exists (select 1 from cup_details c where c.match_id = m.match_id and c.source = ?)), "
        "count(distinct m.home_team) "
        f"{base} group by 1 order by 1", (FRIENDLY, FRIENDLY, src, *p)).fetchall()
    total = con.execute(
        "select count(*), sum(v.tournament = ?), sum(v.neutral), min(m.match_date), max(m.match_date) "
        f"{base}", (FRIENDLY, *p)).fetchone()
    teams = con.execute("select count(*) from team_names where source = ?", (src,)).fetchone()[0]
    playing = con.execute(
        "select count(distinct t) from (select home_team t from matches where competition = ? "
        "union select away_team from matches where competition = ?)", (comp, comp)).fetchone()[0]
    shootouts = con.execute("select count(*) from cup_details where source = ?", (src,)).fetchone()[0]
    tournaments = con.execute(f"select v.tournament, count(*) {base} group by 1 order by 2 desc, 1", p).fetchall()
    results = con.execute(
        "select s.result, count(*) from match_stats s join matches m using (match_id) "
        "where s.source = ? and m.competition = ? group by 1 order by 1", p).fetchall()
    return {"decades": decades, "total": total, "teams": teams, "playing": playing, "shootouts": shootouts,
            "tournaments": tournaments, "results": results}


def report_markdown(rep: dict, cfg: dict, tables: dict, digests: list[str]) -> str:
    """导入报告（中文）。"""
    n, friendly, neutral, first, last = tables["total"]
    lines = ["# 国际赛（INT）martj42 比分导入报告（自动生成）", "",
             f"来源名 {cfg['db_source']}，原始文件 {rep['folder']}（results.csv 下载时间 {rep['fetched_at'] or '抓取记录里没查到'} UTC）。",
             f"共导入 {rep['matches']} 场国家队比赛（{first} 到 {last}；只导 {cfg['last_date']} 及以前、比分不空的开发赛季）。", ""]
    if digests:
        same = "一样" if len(set(digests)) == 1 else "不一样（有问题）"
        lines += [f"重跑核对：同一批原始文件连着导 {len(digests)} 次，内容指纹{same}：{digests[-1][:16]}…", ""]
    lines += ["## 合计", "",
              f"- 比赛：{n} 场；友谊赛 {friendly} 场，正式比赛 {n - friendly} 场",
              f"- 中立场：{neutral} 场（{neutral / n:.1%}）",
              f"- 点球大战（进 cup_details）：{tables['shootouts']} 场",
              f"- 球队：{tables['teams']} 支（英文名 + “{cfg['team_suffix']}”；真的踢过导入比赛的 {tables['playing']} 支）",
              f"- 胜平负：{'，'.join(f'{r} {c}' for r, c in tables['results'])}",
              f"- 不导入：{cfg['last_date']} 以后的 {rep['after_last_date']} 场（考试、上线赛季）；比分空的（还没踢） {rep['no_score']} 场", "",
              "## 每十年", "",
              "| 年代 | 场数 | 友谊赛 | 正式比赛 | 中立场 | 点球大战 | 主队数 |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {d}s | {c} | {f} | {o} | {nt} | {so} | {ht} |" for d, c, f, o, nt, so, ht in tables["decades"]]
    lines += ["", "## 同一天同一对队两场（不导入）", "",
              "这几对核对过确实是同一天踢了两场（settings 的 martj42.same_day_pairs 登记）。唯一键靠 round = 日期 分开，"
              "同一天的分不开，导哪一场都是猜，所以都不导：", ""]
    lines += [f"- {g.date} {g.home_name} {g.home_goals}-{g.away_goals} {g.away_name}（{g.tournament}，{g.city}）"
              for g in rep["same_day_skipped"]] or ["（没有）"]
    if rep["same_day_unused"]:
        lines += ["", "登记了、这次却没有两场的（不影响导入）："] + [f"- {k}" for k in rep["same_day_unused"]]
    lines += ["", "## 点球大战没进 cup_details 的", "",
              f"- {cfg['last_date']} 以后的：{rep['shootouts_after_last_date']} 场",
              f"- 对不上 results.csv（比赛不在、或者不导入）：{len(rep['shootouts_unmatched'])} 场"]
    lines += [f"  - {d} {h} - {a}" for d, h, a in rep["shootouts_unmatched"]]
    lines += [f"- 比分不是平局（两回合总比分打平、第二回合踢点球；martj42 分不出踢没踢加时，所以不写）："
              f"{len(rep['shootouts_not_draw'])} 场。这些比赛的比分照样进 match_stats，胜平负照比分算"]
    lines += [f"  - {d} {h} {hg}-{ag} {a}（{t}）" for d, h, a, hg, ag, t in rep["shootouts_not_draw"]]
    lines += ["", "## 赛事（match_venue.tournament，按场数）", "", "| 赛事 | 场数 |", "|---|---|"]
    lines += [f"| {t} | {c} |" for t, c in tables["tournaments"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="国家队比赛比分从 martj42 原始文件导入数据库")
    ap.add_argument("--db", default=None, help="数据库文件，默认 settings 里的")
    ap.add_argument("--folder", default=None, help="原始文件所在的日期文件夹，默认 data/raw/martj42/ 下最新的")
    ap.add_argument("--twice", action="store_true", help="连着导两次，核对两次的内容指纹一样（重跑不变）")
    a = ap.parse_args(argv)
    settings = load_settings()
    cfg = settings["martj42"]
    con = connect(Path(a.db)) if a.db else connect()
    digests = []
    for _ in range(2 if a.twice else 1):
        rep = run(con, settings, Path(a.folder) if a.folder else None)
        digests.append(content_digest(con, cfg))
    if len(set(digests)) > 1:
        raise SystemExit(f"两次导入的结果不一样：{digests}")
    out = resolve_path(settings["paths"]["reports"]) / cfg["report_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_markdown(rep, cfg, summary_tables(con, cfg), digests), encoding="utf-8")
    con.close()
    log.info("导入 %d 场；报告：%s", rep["matches"], out)


if __name__ == "__main__":
    main()
