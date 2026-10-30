"""把 football-data 的原始 CSV 清洗后导入数据库（需求文档第 17 节“历史数据下载后怎么整理”第 3、4 步）。

运行方法：Claude 运行 `.venv\\Scripts\\python -m src.clean.football_data_import`（约 1 分钟）。
可以反复运行：每次都从原始层整批重导 football-data 的数据，结果一样；比赛编号不变。

每个文件做的事：
1. 读文件（football_data_reader 处理格式问题），去掉联赛代码不对的行。
2. 日期：两位年份（25/08/07）和四位年份（25/08/2007）都认。
3. 开球时间：football-data 写的是英国当地时间，按英国夏令时规则换成标准时间（UTC）。2019/20 以前没有开球时间，留空。
4. 队名：按 config/teams.yaml 换成统一的球队编号；表里没有的队名，那场比赛不导入，列进“待确认”。
5. 数字：比分、射门等换成整数；赔率换成小数，低于 1.01 的（例如 0）当作错误值不要。
6. 赔率：按数据字典把每一列归到“公司 × 玩法 × 时点”，改过名的列（BbAvH → AvgH）合成一组；
   亚盘配上对应的盘口（2003–2004 各家有自己的盘口列，2005–2018 用 BbAHh，2019 起赛前用 AHh、收盘用 AHCh）。
只导入有用途的列（数据字典的“用途”），其余留在原始文件里，以后要用再导。
"""
import logging
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.clean.football_data_inventory import season_files, season_label
from src.clean.football_data_reader import keep_league_rows, read_football_data_csv
from src.clean.schema import create_tables
from src.common.config import load_football_data_columns, load_settings, resolve_path, PROJECT_ROOT
from src.common.db import connect

SOURCE = "football-data"
log = logging.getLogger(__name__)

# 赛果和统计：数据库里的列 ← football-data 的列（数据字典里这些列的用途都不是“只存”，裁判除外：第 17 节说先存）
STAT_COLUMNS = {
    "home_goals": "FTHG", "away_goals": "FTAG", "result": "FTR",
    "ht_home_goals": "HTHG", "ht_away_goals": "HTAG",
    "home_shots": "HS", "away_shots": "AS", "home_shots_on_target": "HST", "away_shots_on_target": "AST",
    "home_corners": "HC", "away_corners": "AC", "home_fouls": "HF", "away_fouls": "AF",
    "home_yellow": "HY", "away_yellow": "AY", "home_red": "HR", "away_red": "AR",
    "home_xg": "HxG", "away_xg": "AxG", "referee": "Referee",
}
TEXT_STATS = {"result", "referee"}
REAL_STATS = {"home_xg", "away_xg"}

# 数据字典里的“部分” → 赔率表的列
PART_TO_FIELD = {"主胜": "home", "平": "draw", "客胜": "away", "大": "over", "小": "under",
                 "主队赔率": "home", "客队赔率": "away"}
TIMING = {"赛前": "pre", "收盘": "close"}
ODDS_USES = {"预测时点的赔率", "收盘赔率"}


# ---------- 读配置 ----------

def load_teams(settings: dict) -> tuple[list[dict], dict[str, str]]:
    """读球队对照表，返回 (球队列表, {football-data 的写法: 统一英文名})。"""
    with open(resolve_path(settings["paths"]["teams"]), encoding="utf-8") as f:
        teams = yaml.safe_load(f)["球队"]
    names: dict[str, str] = {}
    for t in teams:
        for n in [t["英文"], *t.get("别名", [])]:
            if n in names:
                raise ValueError(f"teams.yaml 里队名 {n} 出现了两次")
            names[n] = t["英文"]
    return teams, names


def odds_slots(columns: dict) -> list[dict]:
    """按数据字典把赔率列分组：每组是“一家公司 × 一种玩法 × 一个时点”。

    返回的每一组：{bookmaker, market, timing, fields: {home: [列名, ...], ...}, line: [盘口列名, ...] 或 2.5}
    同一部分有几个列名（改过名）时都列上，读的时候取第一个有值的。
    """
    lines: dict[tuple[str, str], list[str]] = {}   # (公司, 时点) → 亚盘盘口列；公司为空 = 多家通用的盘口
    slots: dict[tuple[str, str, str], dict] = {}
    for col, info in columns.items():
        if info.get("用途") not in ODDS_USES:
            continue
        item, timing = info["数据项"], TIMING[info["时点"]]
        if "亚盘盘口" in item:
            lines.setdefault((info.get("公司", ""), timing), []).append(col)
            continue
        if "胜平负" in item:
            market = "1x2"
        elif "大小球" in item:
            if "2.5" not in item:
                raise ValueError(f"{col}：只认得 2.5 球的大小球")
            market = "ou"
        elif "亚盘" in item:
            market = "ah"
        else:
            raise ValueError(f"{col}：数据项“{item}”认不出是哪种玩法")
        key = (info["公司"], market, timing)
        slot = slots.setdefault(key, {"bookmaker": info["公司"], "market": market, "timing": timing, "fields": {}})
        slot["fields"].setdefault(PART_TO_FIELD[info["部分"]], []).append(col)
    for (bookmaker, market, timing), slot in slots.items():
        if market == "ou":
            slot["line"] = 2.5
        elif market == "ah":
            # 先用这家公司自己的盘口列（2003–2004 才有），没有再用多家通用的
            slot["line"] = lines.get((bookmaker, timing), []) + lines.get(("", timing), [])
        else:
            slot["line"] = None
    return list(slots.values())


# ---------- 换算 ----------

def first_present(table: pd.DataFrame, cols: list[str]) -> pd.Series:
    """几个列名（改过名的同一种数据）里，每行取第一个有值的。都没有这些列时返回全空。"""
    out = pd.Series(None, index=table.index, dtype=object)
    for c in cols:
        if c in table.columns:
            out = out.where(out.notna(), table[c])
    return out


def to_number(values: pd.Series, what: str, issues: list[str]) -> pd.Series:
    """文字 → 小数。认不出的当空值，并记进问题清单。"""
    num = pd.to_numeric(values, errors="coerce")
    bad = values.notna() & num.isna()
    if bad.any():
        issues.append(f"{what}：{int(bad.sum())} 个值不是数字，例如 {sorted(set(values[bad]))[:3]}，当作空")
    return num


def parse_handicap(text):
    """亚盘盘口文字 → 数字。'-0.5' → -0.5；'-0.5,-1'（半一，西乙 2009/10 的写法）→ -0.75。认不出返回 None。"""
    if text is None or pd.isna(text):
        return None
    try:
        parts = [float(p) for p in str(text).split(",")]
    except ValueError:
        return None
    return sum(parts) / len(parts) if len(parts) <= 2 else None


def parse_dates(values: pd.Series) -> pd.Series:
    """日期文字 → 日期。两位年份（25/08/07）和四位年份（25/08/2007）都认；两位年份 69–99 算 19xx，00–68 算 20xx。"""
    text = values.fillna("")
    long_ = pd.to_datetime(text.where(text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}")), format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(text.where(text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}")), format="%d/%m/%y", errors="coerce")
    return long_.fillna(short)


def kickoff_to_utc(dates: pd.Series, times: pd.Series, tz: str) -> pd.Series:
    """英国当地的日期 + 开球时间 → 标准时间文字 'yyyy-mm-dd HH:MM'。没有开球时间的返回空。

    夏令时由时区规则自动处理：例如 8 月的 15:00 是标准时间 14:00，1 月的 15:00 还是 15:00。
    """
    ok = times.fillna("").str.fullmatch(r"\d{1,2}:\d{2}") & dates.notna()
    out = pd.Series(None, index=dates.index, dtype=object)
    if ok.any():
        local = pd.to_datetime(dates[ok].dt.strftime("%Y-%m-%d") + " " + times[ok], format="%Y-%m-%d %H:%M")
        utc = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        out[ok] = utc.dt.strftime("%Y-%m-%d %H:%M").where(utc.notna(), None)
    return out


# ---------- 一个文件 ----------

def parse_file(table: pd.DataFrame, league: str, season: int, names: dict[str, str], slots: list[dict],
               settings: dict, issues: list[str], unknown: Counter) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """一个文件的表格 → (比赛, 赛果统计, 赔率, 不要的赔率个数)。三张表都用 table 的 row 列对应同一场比赛。"""
    q = settings["quality"]
    home = table["HomeTeam"].map(names)
    away = table["AwayTeam"].map(names)
    for col, mapped in (("HomeTeam", home), ("AwayTeam", away)):
        for n in table.loc[mapped.isna() & table[col].notna(), col]:
            unknown[(league, season_label(season), n)] += 1
    dates = parse_dates(table["Date"])
    if dates.isna().any():
        issues.append(f"{int(dates.isna().sum())} 行日期认不出，例如 {table.loc[dates.isna(), 'Date'].head(3).tolist()}，不导入")
    ok = home.notna() & away.notna() & dates.notna()

    times = table["Time"] if "Time" in table.columns else pd.Series(None, index=table.index, dtype=object)
    matches = pd.DataFrame({
        "row": table["row"], "competition": league, "season": season,
        "home_team": home, "away_team": away,
        "match_date": dates.dt.strftime("%Y-%m-%d"),
        "kickoff_utc": kickoff_to_utc(dates, times, settings["football_data"]["timezone"]),
        "status": "finished",   # football-data 只列已经踢完的比赛；没有比分的场次由质量检查列出
        "source_key": table["Div"].fillna(league) + "|" + table["Date"].fillna("") + "|"
                      + table["HomeTeam"].fillna("") + "|" + table["AwayTeam"].fillna(""),
    })[ok]
    dup = matches.duplicated(["home_team", "away_team"], keep="first")
    if dup.any():
        issues.append(f"同一对主客队出现两次 {int(dup.sum())} 场，只留第一场：{matches.loc[dup, 'source_key'].head(3).tolist()}")
        matches = matches[~dup]
    keep = matches.index

    stats = pd.DataFrame({"row": table["row"]})
    for field, col in STAT_COLUMNS.items():
        values = table[col] if col in table.columns else pd.Series(None, index=table.index, dtype=object)
        if field in TEXT_STATS:
            stats[field] = values
        else:
            num = to_number(values, col, issues)
            if field not in REAL_STATS:
                frac = num.notna() & (num != num.round())
                if frac.any():
                    issues.append(f"{col}：{int(frac.sum())} 个值不是整数，当作空")
                num = num.where(~frac).astype("Int64")
            stats[field] = num
    stats = stats.loc[keep]

    frames, dropped = [], 0
    for slot in slots:
        cols = [c for cs in slot["fields"].values() for c in cs]
        if not any(c in table.columns for c in cols):
            continue
        df = pd.DataFrame({"row": table["row"], "bookmaker": slot["bookmaker"],
                           "market": slot["market"], "timing": slot["timing"]})
        for field, alts in slot["fields"].items():
            price = to_number(first_present(table, alts), "/".join(alts), issues)
            bad = price < q["min_odds"]
            dropped += int(bad.sum())
            df[field] = price.where(~bad)
        if isinstance(slot["line"], list):
            raw = first_present(table, slot["line"])
            line = pd.to_numeric(raw.map(parse_handicap), errors="coerce")
            bad = raw.notna() & (line.isna() | (line.abs() > q["max_handicap"]) | ((line * 4) % 1 != 0))
            if bad.any():
                issues.append(f"{slot['bookmaker']} 亚盘盘口：{int(bad.sum())} 个值不对，例如 {sorted(set(raw[bad]))[:3]}，当作空")
            df["line"] = line.where(~bad)
        else:
            df["line"] = slot["line"]
        prices = [f for f in ("home", "draw", "away", "over", "under") if f in df.columns]
        has = df[prices].notna().any(axis=1)
        if slot["market"] == "ah":
            has &= df["line"].notna()            # 亚盘没有盘口，赔率就没有意义
        frames.append(df[has & df.index.isin(keep)])
    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    odds = odds.reindex(columns=["row", "bookmaker", "market", "timing", "line", "home", "draw", "away", "over", "under"])
    return matches, stats, odds, dropped


# ---------- 写进数据库 ----------

def plain(value):
    """pandas / numpy 的值 → SQLite 认得的普通值（空值 → None）。"""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def rows_of(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    return [tuple(plain(v) for v in r) for r in df[cols].itertuples(index=False, name=None)]


ODDS_COLS = ["match_id", "bookmaker", "market", "timing", "line", "home", "draw", "away", "over", "under"]
STAT_FIELDS = list(STAT_COLUMNS)


def write_database(con: sqlite3.Connection, teams: list[dict], names: dict[str, str],
                   matches: pd.DataFrame, stats: pd.DataFrame, odds: pd.DataFrame) -> int:
    """在一个事务里写入：中途出错就全部不算，数据库保持导入前的样子。返回 football-data 的比赛数。"""
    create_tables(con)
    with con:
        con.executemany("insert into teams (team_id, name_zh, country) values (?, ?, ?) "
                        "on conflict(team_id) do update set name_zh = excluded.name_zh, country = excluded.country",
                        [(t["英文"], t.get("中文") or None, t["国家"]) for t in teams])
        con.execute("delete from team_names where source = ?", (SOURCE,))
        con.executemany("insert into team_names (source, name, team_id) values (?, ?, ?)",
                        [(SOURCE, n, tid) for n, tid in names.items()])

        # 比赛：有就更新（编号不变），没有才新增
        con.executemany(
            "insert into matches (competition, season, home_team, away_team, match_date, kickoff_utc, status) "
            "values (?, ?, ?, ?, ?, ?, ?) on conflict(competition, season, home_team, away_team) do update set "
            "match_date = excluded.match_date, kickoff_utc = coalesce(excluded.kickoff_utc, matches.kickoff_utc), "
            "status = excluded.status",
            rows_of(matches, ["competition", "season", "home_team", "away_team", "match_date", "kickoff_utc", "status"]))
        ids = pd.read_sql("select match_id, competition, season, home_team, away_team from matches", con)
        matches = matches.merge(ids, on=["competition", "season", "home_team", "away_team"], how="left")
        row_to_id = matches.set_index("row")["match_id"]

        # 从原始层算出来的表：football-data 的整批删掉重导
        for table in ("match_sources", "match_stats", "odds"):
            con.execute(f"delete from {table} where source = ?", (SOURCE,))
        con.executemany("insert into match_sources values (?, ?, ?, ?, ?)",
                        [(int(r.match_id), SOURCE, r.source_key, r.raw_file, r.fetched_at_utc)
                         for r in matches.itertuples()])
        stats = stats.assign(match_id=stats["row"].map(row_to_id), source=SOURCE)
        con.executemany(f"insert into match_stats (match_id, source, {', '.join(STAT_FIELDS)}) "
                        f"values ({', '.join('?' * (len(STAT_FIELDS) + 2))})",
                        rows_of(stats, ["match_id", "source", *STAT_FIELDS]))
        odds = odds.assign(match_id=odds["row"].map(row_to_id))
        con.executemany(f"insert into odds (source, {', '.join(ODDS_COLS)}) values (?{', ?' * len(ODDS_COLS)})",
                        [(SOURCE, *r) for r in rows_of(odds, ODDS_COLS)])
        # 原始层里已经没有的比赛（任何来源都不再提到它）删掉。以后有了预测存档，要改成只标记不删
        con.execute("delete from matches where match_id not in (select match_id from match_sources)")
    return con.execute("select count(*) from match_sources where source = ?", (SOURCE,)).fetchone()[0]


# ---------- 全部文件 ----------

def run(settings: dict | None = None, con: sqlite3.Connection | None = None) -> dict:
    """导入全部 football-data 文件。返回导入结果（给报告用）。"""
    settings = settings or load_settings()
    raw_dir = resolve_path(settings["paths"]["raw"])
    teams, names = load_teams(settings)
    slots = odds_slots(load_football_data_columns(settings))
    leagues = {**settings["leagues"]["top"], **settings["leagues"]["second"]}
    con = con or connect()
    create_tables(con)
    fetched = dict(con.execute("select file_path, max(fetched_at_utc) from fetch_log "
                               "where status = 'saved' group by file_path").fetchall())

    all_m, all_s, all_o = [], [], []
    issues: list[str] = []
    unknown: Counter = Counter()
    dropped = files = 0
    next_row = 0
    for lg in leagues:
        for season, path in season_files(raw_dir, lg).items():
            r = read_football_data_csv(path)
            file_issues = list(r.issues)
            table = keep_league_rows(r.table, lg, file_issues)
            if len(table):
                table = table.assign(row=range(next_row, next_row + len(table)))
                next_row += len(table)
                m, s, o, d = parse_file(table, lg, season, names, slots, settings, file_issues, unknown)
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                if rel in fetched:
                    m["fetched_at_utc"] = fetched[rel]
                else:   # 抓取记录里没有（例如手工放进来的文件）：用文件的修改时间
                    m["fetched_at_utc"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                m["raw_file"] = rel
                all_m.append(m); all_s.append(s); all_o.append(o)
                dropped += d
                files += 1
            issues += [f"{lg} {season_label(season)}：{i}" for i in file_issues]

    matches = pd.concat(all_m, ignore_index=True)
    n = write_database(con, teams, names, matches,
                       pd.concat(all_s, ignore_index=True), pd.concat(all_o, ignore_index=True))
    odds_rows = con.execute("select count(*) from odds where source = ?", (SOURCE,)).fetchone()[0]
    return {"files": files, "matches": n, "odds_rows": odds_rows, "dropped_odds": dropped,
            "issues": issues, "unknown": unknown}


def main() -> None:
    from src.clean.quality import report_markdown, run_checks
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    start = time.time()
    log.info("开始导入 football-data……")
    con = connect()
    result = run(settings, con)
    log.info("导入了 %s 个文件、%s 场比赛、%s 组赔率；不要的错误赔率 %s 个；没对上的队名 %s 个。用了 %.0f 秒。",
             result["files"], result["matches"], result["odds_rows"], result["dropped_odds"],
             len(result["unknown"]), time.time() - start)
    checks = run_checks(con, settings)
    out = resolve_path(settings["paths"]["reports"]) / "football_data_导入和质量检查.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_markdown(result, checks), encoding="utf-8")
    log.info("质量检查 %s 项，有问题的 %s 项。报告在 %s", len(checks), sum(1 for c in checks if c["rows"]), out)


if __name__ == "__main__":
    main()
