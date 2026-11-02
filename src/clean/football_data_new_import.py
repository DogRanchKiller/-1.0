r"""把 football-data “新联赛”文件（日职 JPN、中超 CHN）清洗后导入数据库（需求文档 6.1“新联赛文件”、第 12 节第 11 阶段、第 17 节）。

运行：.venv\Scripts\python -m src.clean.football_data_new_import          （几秒钟）
      .venv\Scripts\python -m src.clean.football_data_new_import --db 别的数据库文件
可以反复运行：每次都从原始层最新的快照整批重导这两个联赛，比赛编号不变。

文件长什么样（2026-11-03 下载的版本）：
  Country, League, Season, Date, Time, Home, Away, HG, AG, Res,
  PSCH/PSCD/PSCA（平博收盘）, MaxCH/D/A（多家最高收盘）, AvgCH/D/A（多家平均收盘）,
  BFECH/D/A（Betfair 交易所收盘）, B365CH/D/A（Bet365 收盘）
  - 一个联赛一个文件，所有赛季都在里面；每次下载存一份带日期的快照，这里用最新的一份。
  - 只有收盘赔率（列名里公司缩写后面带 C），没有赛前赔率。所以这两个联赛不能做“B 版”（要赛前赔率）的模型。
  - 没有半场比分、射门、角球、红黄牌。
  - Season 列：一年一个赛季写 “2014”；日职 2026 年改成秋春制，写 “2026/2027”。都换成开始年份（2026/2027 → 2026）。
  - League 列：日职旧行前面多一个空格（" J1 League"），读的时候去掉。

每个文件做的事：
1. 读最新快照（football_data_reader 处理 BOM、行尾多余逗号、没有列名的空列、值前后的空格）。
   快照必须有下载记录（fetch_log 里 sha256 和文件一致）：没有就不导入，免得导入来路不明的文件。
2. 赛季：Season 列换成开始年份（season_from_text）。
3. 开球时间：Time 是英国当地时间（2026-11-03 核对：按英国时间换算后，日本、北京时间的开球时刻冬夏都一样整齐，
   见需求文档 6.1），按英国夏令时规则换成 UTC。日期 match_date 和老联赛一样用英国日期。
4. 队名：按 config/teams.yaml 换成统一的球队编号（改过名的中超球队用“别名”合并）；表里没有的队名，那场不导入，列进报告。
5. 只留联赛比赛：文件里混着升降级附加赛（J1 2018、2019、2022，中超 2020、2021）。
   每个赛季出场次数少于“本季中位数 × league_team_min_share”的队不算本季联赛球队，有它们的比赛不导入，列进报告。
6. 同一赛季同一对主客队出现两次（日职 2015、2016 冠军赛，中超 2020、2021 第二阶段）：
   数据库的唯一键是 联赛 × 赛季 × 主队 × 客队，和老联赛导入一样只留日期最早的一场，后面的列进报告。
7. 比分、赛果进 match_stats；收盘赔率按数据字典（config/football_data_columns.yaml）归到“公司 × 玩法 × 时点”进 odds。

数据库里的来源名是 football-data-new（settings.yaml 的 football_data_new.db_source），和老联赛的 football-data 分开：
老联赛导入（src/clean/football_data_import.py）整批重导时只删 football-data 的行，不会删掉这两个联赛。
"""
import argparse
import hashlib
import logging
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd

from src.clean.football_data_import import (first_present, kickoff_to_utc, load_teams, odds_slots, parse_dates,
                                            rows_of, to_number)
from src.clean.football_data_reader import read_football_data_csv
from src.clean.schema import create_tables
from src.clean.source_lifecycle import (delete_replaced_results, new_match_ids, reconcile_source_lifecycle,
                                        snapshot_prediction_sources)
from src.collect.football_data import SOURCE as RAW_SOURCE
from src.collect.football_data_new import DATA_TYPE, new_league_url
from src.common.config import PROJECT_ROOT, load_football_data_columns, load_settings, resolve_path
from src.common.db import connect

log = logging.getLogger(__name__)

NEEDED = ["Season", "Date", "Time", "Home", "Away", "HG", "AG", "Res"]   # 少了这些列就不导入
ODDS_COLS = ["match_id", "bookmaker", "market", "timing", "line", "home", "draw", "away", "over", "under"]


# ---------- 小工具 ----------

def season_from_text(text) -> int:
    """Season 列 → 赛季开始年份。'2014' → 2014；'2026/2027' → 2026（秋春制，第二个年份必须是第一个 + 1）。
    认不出的写法直接报错（不猜）：赛季错了会影响“开发 / 考试 / 实时”的划分。"""
    s = str(text).strip()
    if len(s) == 4 and s.isdigit():
        return int(s)
    parts = s.split("/")
    if len(parts) == 2 and all(len(p) == 4 and p.isdigit() for p in parts) and int(parts[1]) == int(parts[0]) + 1:
        return int(parts[0])
    raise ValueError(f"Season 列的写法认不出：{text!r}")


def latest_snapshot(raw_dir: Path, raw_folder: str, code: str) -> Path | None:
    """原始层里这个联赛最新的快照（文件名是下载日期，同一天补下的带序号，按文件名排序最后一个最新）。"""
    folder = raw_dir / RAW_SOURCE / raw_folder / code
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    return files[-1] if files else None


def rel_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def verified_snapshot(con: sqlite3.Connection, path: Path, url: str) -> str | None:
    """快照有没有对得上的下载记录（同一网址、同一文件、内容 sha256 一样）。对得上返回最早的下载时间（UTC），否则 None。"""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = con.execute("select min(fetched_at_utc) from fetch_log where source = ? and data_type = ? and status = 'saved' "
                      "and url = ? and file_path = ? and sha256 = ?",
                      (RAW_SOURCE, DATA_TYPE, url, rel_path(path), digest)).fetchone()
    return row[0] if row and row[0] else None


def league_teams(table: pd.DataFrame, min_share: float) -> dict[int, set[str]]:
    """每个赛季的联赛球队：出场次数 ≥ 本季中位数 × min_share 的队（附加赛球队只踢 1–3 场，达不到）。"""
    out = {}
    for season, part in table.groupby("season"):
        n = pd.concat([part["home_team"], part["away_team"]]).value_counts()
        out[int(season)] = set(n[n >= n.median() * min_share].index)
    return out


# ---------- 一个文件 ----------

def parse_new_file(table: pd.DataFrame, code: str, names: dict[str, str], slots: list[dict], settings: dict,
                   report: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """一个新联赛文件的表格 → (比赛, 比分, 赔率)。三张表用 row 列对应同一场比赛。
    report 里记下：认不出的队名、不导入的附加赛、重复的主客队、日期认不出的行、错误赔率个数。"""
    cfg = settings["football_data_new"]
    missing = [c for c in NEEDED if c not in table.columns]
    if missing:
        raise ValueError(f"{code} 文件少了这些列：{missing}")
    t = table.copy()
    t["row"] = range(len(t))
    t["season"] = t["Season"].map(season_from_text)
    t["home_team"] = t["Home"].map(names)
    t["away_team"] = t["Away"].map(names)
    for col, mapped in (("Home", t["home_team"]), ("Away", t["away_team"])):
        for n in t.loc[mapped.isna() & t[col].notna(), col]:
            report["unknown"][n] += 1
    dates = parse_dates(t["Date"])
    report["bad_dates"] = t.loc[dates.isna(), "Date"].tolist()
    t = t[t["home_team"].notna() & t["away_team"].notna() & dates.notna()].copy()
    dates = dates[t.index]

    # 附加赛：有“本季联赛球队”以外的队参加的比赛
    members = league_teams(t, float(cfg["league_team_min_share"]))
    in_league = [h in members[s] and a in members[s] for s, h, a in zip(t["season"], t["home_team"], t["away_team"])]
    in_league = pd.Series(in_league, index=t.index)
    report["playoffs"] = t.loc[~in_league, ["season", "Date", "Home", "Away", "HG", "AG"]].values.tolist()
    report["teams_per_season"] = {s: len(v) for s, v in sorted(members.items())}
    t, dates = t[in_league], dates[in_league]

    matches = pd.DataFrame({
        "row": t["row"], "competition": code, "season": t["season"],
        "home_team": t["home_team"], "away_team": t["away_team"],
        "match_date": dates.dt.strftime("%Y-%m-%d"),
        "kickoff_utc": kickoff_to_utc(dates, t["Time"], cfg["timezone"]),
        "status": "finished",            # 文件里只有踢完的比赛
        "source_key": code + "|" + t["Season"].str.strip() + "|" + t["Date"] + "|" + t["Home"] + "|" + t["Away"],
    })
    # 同一赛季同一对主客队：按日期排，只留最早的一场（数据库唯一键；和老联赛导入一样）
    matches = matches.sort_values(["match_date", "row"])
    dup = matches.duplicated(["season", "home_team", "away_team"], keep="first")
    report["repeats"] = matches.loc[dup, "source_key"].tolist()
    matches = matches[~dup].sort_values("row")
    keep = set(matches["row"])

    issues: list[str] = []
    stats = pd.DataFrame({"row": t["row"]})
    for field, col in (("home_goals", "HG"), ("away_goals", "AG")):
        num = to_number(t[col], col, issues)
        stats[field] = num.where(num == num.round()).astype("Int64")
    stats["result"] = t["Res"]
    stats = stats[stats["row"].isin(keep)]

    q = settings["quality"]
    frames, dropped = [], 0
    for slot in slots:
        if slot["market"] != "1x2":
            continue                      # 新联赛文件只有胜平负赔率
        if not any(c in t.columns for cs in slot["fields"].values() for c in cs):
            continue
        df = pd.DataFrame({"row": t["row"], "bookmaker": slot["bookmaker"], "market": "1x2", "timing": slot["timing"],
                           "line": None})
        for field, alts in slot["fields"].items():
            price = to_number(first_present(t, alts), "/".join(alts), issues)
            bad = price < q["min_odds"]
            dropped += int(bad.sum())
            df[field] = price.where(~bad)
        has = df[["home", "draw", "away"]].notna().any(axis=1) & df["row"].isin(keep)
        frames.append(df[has])
    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    odds = odds.reindex(columns=["row", "bookmaker", "market", "timing", "line", "home", "draw", "away", "over", "under"])
    report["issues"] = issues
    report["dropped_odds"] = dropped
    return matches, stats, odds


# ---------- 写进数据库 ----------

def write_new_database(con: sqlite3.Connection, source: str, teams: list[dict], names: dict[str, str],
                       matches: pd.DataFrame, stats: pd.DataFrame, odds: pd.DataFrame) -> int:
    """在一个事务里写入：中途出错就全部不算，数据库保持导入前的样子。返回这个来源的比赛数。

    只动来源 = source（football-data-new）的行：老联赛（football-data）、懂球帝、维基百科的数据一行都不碰。
    球队表只写这次用到的球队（日本、中国的队），队名对照只写这个来源用到的写法。
    """
    create_tables(con)
    used = set(matches["home_team"]) | set(matches["away_team"])
    with con:
        con.executemany("insert into teams (team_id, name_zh, country) values (?, ?, ?) "
                        "on conflict(team_id) do update set name_zh = excluded.name_zh, country = excluded.country",
                        [(t["英文"], t.get("中文") or None, t["国家"]) for t in teams if t["英文"] in used])
        con.execute("delete from team_names where source = ?", (source,))
        con.executemany("insert into team_names (source, name, team_id) values (?, ?, ?)",
                        [(source, n, tid) for n, tid in names.items() if tid in used])

        # 比赛：有就更新（编号不变），没有才新增。联赛比赛的 round 固定 ''、leg 固定 0（和老联赛一样）
        new_ids = new_match_ids(con)
        con.executemany(
            "insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status, "
            "round, leg, date_precision) values (?, ?, ?, ?, ?, ?, ?, ?, '', 0, 'day') "
            "on conflict(competition, season, home_team, away_team, round, leg) do update set "
            "match_date = excluded.match_date, kickoff_utc = coalesce(excluded.kickoff_utc, matches.kickoff_utc), "
            "status = excluded.status",
            ((next(new_ids), *r) for r in rows_of(matches,
                ["competition", "season", "home_team", "away_team", "match_date", "kickoff_utc", "status"])))
        ids = pd.read_sql("select match_id, competition, season, home_team, away_team from matches "
                          "where round = '' and leg = 0", con)
        matches = matches.merge(ids, on=["competition", "season", "home_team", "away_team"], how="left")
        row_to_id = matches.set_index("row")["match_id"]

        # 换来源的顺序和老联赛导入一样：先记下有预测的比赛原来的出处，换完再处理“缺源”
        previous = snapshot_prediction_sources(con)
        con.execute("delete from match_sources where source = ?", (source,))
        con.executemany("insert into match_sources values (?, ?, ?, ?, ?)",
                        [(int(r.match_id), source, r.source_key, r.raw_file, r.fetched_at_utc)
                         for r in matches.itertuples()])
        delete_replaced_results(con, source)
        con.execute("delete from odds where source = ?", (source,))
        stats = stats.assign(match_id=stats["row"].map(row_to_id), source=source)
        con.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)",
                        rows_of(stats, ["match_id", "source", "home_goals", "away_goals", "result"]))
        odds = odds.assign(match_id=odds["row"].map(row_to_id))
        con.executemany(f"insert into odds (source, {', '.join(ODDS_COLS)}) values (?{', ?' * len(ODDS_COLS)})",
                        [(source, *r) for r in rows_of(odds, ODDS_COLS)])
        reconcile_source_lifecycle(con, previous)
    return con.execute("select count(*) from match_sources where source = ?", (source,)).fetchone()[0]


# ---------- 全部 ----------

def run(settings: dict | None = None, con: sqlite3.Connection | None = None) -> dict:
    """导入 settings 里 football_data_new.leagues 的每个联赛（最新快照）。返回每个联赛的报告。"""
    settings = settings or load_settings()
    cfg = settings["football_data_new"]
    registered = {c for g in settings["leagues"].values() for c in g}
    wrong = [c for c in cfg["leagues"] if c not in registered]
    if wrong:
        raise ValueError(f"{wrong} 没有登记在 settings.yaml 的 leagues 里")
    raw_dir = resolve_path(settings["paths"]["raw"])
    teams, names = load_teams(settings)
    slots = odds_slots(load_football_data_columns(settings))
    con = con or connect()
    create_tables(con)
    all_m, all_s, all_o, reports = [], [], [], {}
    next_row = 0
    for code in cfg["leagues"]:
        path = latest_snapshot(raw_dir, cfg["raw_folder"], code)
        if path is None:
            raise ValueError(f"{code} 原始层里没有文件，先运行 src.collect.football_data_new 下载")
        fetched = verified_snapshot(con, path, new_league_url(cfg, code))
        if fetched is None:
            raise ValueError(f"{rel_path(path)} 没有对得上的下载记录（fetch_log），不导入")
        r = read_football_data_csv(path)
        rep = {"file": rel_path(path), "fetched_at_utc": fetched, "read_issues": r.issues, "unknown": Counter(),
               "rows": len(r.table)}
        m, s, o = parse_new_file(r.table, code, names, slots, settings, rep)
        if rep["unknown"]:
            raise ValueError(f"{code} 有 teams.yaml 里没有的队名，先加进去：{dict(rep['unknown'])}")
        m = m.assign(raw_file=rep["file"], fetched_at_utc=fetched)
        # 不同联赛的 row 编号错开（每个文件都从 0 编），三张表合在一起还能对上
        for df in (m, s, o):
            df["row"] = df["row"] + next_row
        next_row += rep["rows"]
        all_m.append(m); all_s.append(s); all_o.append(o)
        reports[code] = rep
    n = write_new_database(con, cfg["db_source"], teams, names, pd.concat(all_m, ignore_index=True),
                           pd.concat(all_s, ignore_index=True), pd.concat(all_o, ignore_index=True))
    return {"matches": n, "leagues": reports}


# ---------- 质量检查（从数据库里查这次导入的结果） ----------

def quality_tables(con: sqlite3.Connection, settings: dict) -> dict[str, pd.DataFrame]:
    """导入以后的检查表。赛季直接写开始年份（2014 = 2014 赛季；这两个联赛一年一个赛季，不写成 2014/15）。"""
    from src.common.config import season_role
    cfg = settings["football_data_new"]
    src = cfg["db_source"]
    m = pd.read_sql("select m.match_id, m.competition, m.season, m.home_team, m.away_team, m.match_date, m.kickoff_utc, "
                    "s.home_goals, s.away_goals, s.result from matches m join match_sources ms using (match_id) "
                    "left join match_stats s on s.match_id = m.match_id and s.source = ms.source where ms.source = ?",
                    con, params=[src])
    out = {}
    # 1. 每季场数：n 队双循环应该 n ×（n − 1）场
    rows = []
    for (c, s), p in m.groupby(["competition", "season"]):
        teams = pd.concat([p["home_team"], p["away_team"]])
        n = teams.nunique()
        home, away = p["home_team"].value_counts(), p["away_team"].value_counts()
        uneven = int((home.reindex(teams.unique(), fill_value=0) != away.reindex(teams.unique(), fill_value=0)).sum())
        rows.append({"联赛": c, "赛季": int(s), "用途": season_role(int(s), settings), "球队数": n, "场数": len(p),
                     "应有场数": n * (n - 1), "主客场数不一样的队": uneven,
                     "第一场": p["match_date"].min(), "最后一场": p["match_date"].max()})
    out["每季场数"] = pd.DataFrame(rows)
    # 2. 比分
    res = m["home_goals"].gt(m["away_goals"]).map({True: "H", False: "A"}).where(m["home_goals"].ne(m["away_goals"]), "D")
    out["比分问题"] = m.loc[m["home_goals"].isna() | m["away_goals"].isna() | (m["result"] != res),
                         ["competition", "season", "match_date", "home_team", "away_team", "home_goals", "away_goals", "result"]]
    # 3. 开球时间：换成当地时间后几点开球（整齐 = 时区对）；英国日期和当地日期不一样的场数
    rows = []
    for c, p in m.groupby("competition"):
        tz = cfg["local_timezone"][c]
        utc = pd.to_datetime(p["kickoff_utc"], format="%Y-%m-%d %H:%M", errors="coerce").dt.tz_localize("UTC")
        local = utc.dt.tz_convert(tz)
        hm = local.dt.strftime("%H:%M")
        top = hm.value_counts().head(5)
        rows.append({"联赛": c, "没有开球时间": int(utc.isna().sum()),
                     "当地开球时刻最多的 5 个": "、".join(f"{k}（{v} 场）" for k, v in top.items()),
                     "整点或半点开球的比例": round(float(local.dt.minute.isin([0, 30]).mean()), 4),
                     "英国日期和当地日期不同的场数": int((local.dt.strftime("%Y-%m-%d") != p["match_date"]).sum())})
    out["开球时间"] = pd.DataFrame(rows)
    # 4. 赔率覆盖：每季有几成比赛有这家公司的收盘胜平负
    o = pd.read_sql("select o.match_id, o.bookmaker, o.home, o.draw, o.away from odds o where o.source = ? "
                    "and o.market = '1x2' and o.timing = 'close'", con, params=[src])
    has = o.dropna(subset=["home", "draw", "away"]).merge(m[["match_id", "competition", "season"]], on="match_id")
    cov = has.groupby(["competition", "season", "bookmaker"])["match_id"].nunique().unstack("bookmaker", fill_value=0)
    n = m.groupby(["competition", "season"])["match_id"].nunique()
    out["收盘赔率覆盖"] = cov.div(n, axis=0).round(3).reset_index().rename(columns={"competition": "联赛", "season": "赛季"})
    # 5. 抽水（三个赔率倒数之和）超出 quality.overround 正常范围的组数
    q = settings["quality"]
    r = has.assign(over=1 / has["home"] + 1 / has["draw"] + 1 / has["away"])
    def bounds(b):
        key = "exchange" if b in q["exchanges"] else "best" if b in q["best_of"] else "1x2"
        return q["overround"][key]
    r["bad"] = [not (bounds(b)[0] <= v <= bounds(b)[1]) for b, v in zip(r["bookmaker"], r["over"])]
    out["抽水异常"] = r.groupby(["competition", "bookmaker"]).agg(组数=("bad", "size"), 超出范围=("bad", "sum")).reset_index()
    # 6. 懂球帝有没有同一联赛的比赛可以对照（第 11 阶段只调查，不抓）
    out["懂球帝同联赛比赛"] = pd.read_sql(
        "select m.competition, count(*) as 场数 from matches m join match_sources s using (match_id) "
        f"where s.source = 'dongqiudi' and m.competition in ({','.join('?' * len(cfg['leagues']))}) group by 1",
        con, params=list(cfg["leagues"]))
    return out


def df_markdown(df: pd.DataFrame) -> str:
    """把表格写成 markdown（项目没装 tabulate，自己拼）。"""
    if df.empty:
        return "（没有）"
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def report_markdown(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    lines = ["# football-data 新联赛（日职、中超）导入和质量检查（自动生成）", "",
             f"来源名 football-data-new，共 {result['matches']} 场。", ""]
    for code, rep in result["leagues"].items():
        lines += [f"## {code}", "", f"- 文件：{rep['file']}（下载时间 {rep['fetched_at_utc']} UTC），{rep['rows']} 行",
                  f"- 每季联赛球队数：{rep['teams_per_season']}",
                  f"- 读文件时处理的问题：{len(rep['read_issues'])} 个；日期认不出的行：{len(rep['bad_dates'])}；"
                  f"错误赔率（低于 1.01）：{rep['dropped_odds']} 个",
                  f"- 附加赛（有本季联赛以外的队，不导入）{len(rep['playoffs'])} 场："]
        lines += [f"  - {s} {d} {h} {hg}-{ag} {a}" for s, d, h, a, hg, ag in rep["playoffs"]]
        lines += [f"- 同一赛季同一对主客队第二次出现（只留最早一场）{len(rep['repeats'])} 场："]
        lines += [f"  - {k}" for k in rep["repeats"]]
        lines.append("")
    for name, df in tables.items():
        lines += [f"## {name}", "", df_markdown(df), ""]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="导入 football-data 新联赛文件（日职、中超）")
    ap.add_argument("--db", default=None, help="数据库文件，默认 settings 里的")
    a = ap.parse_args()
    settings = load_settings()
    con = connect(Path(a.db)) if a.db else connect()
    result = run(settings, con)
    tables = quality_tables(con, settings)
    out = resolve_path(settings["paths"]["reports"]) / "football_data_new_导入和质量检查.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_markdown(result, tables), encoding="utf-8")
    for code, rep in result["leagues"].items():
        log.info("%s：附加赛不导入 %s 场，重复主客队不导入 %s 场", code, len(rep["playoffs"]), len(rep["repeats"]))
    log.info("导入 %s 场。报告在 %s", result["matches"], out)
    con.close()


if __name__ == "__main__":
    main()
