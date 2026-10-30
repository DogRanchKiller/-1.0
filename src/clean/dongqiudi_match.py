r"""把懂球帝的比赛和我们数据库里的比赛对上号（试验，需求文档 6.4“队名对照”）。只出报告，不改数据库。

为什么要对：懂球帝用中文队名和它自己的比赛编号，我们数据库用 football-data 的英文队名和自己的编号。
以后把懂球帝的赔率导进数据库，必须先知道“懂球帝 53615071 = 我们的 11939”。

怎么对（不用人工先填中文队名）：
1. 先“投票”找中文队名对应哪个英文队名：懂球帝赛季页的每场比赛，在我们数据库同一联赛同一赛季里找
   “日期差不超过 1 天、比分完全一样”的比赛。找到 k 场，每场给（中文主队→英文主队）、（中文客队→英文客队）各投 1/k 票。
   一个队一个赛季要踢 38 场，正确的对应会得到几十票，碰巧比分日期一样的错配只有零星几票。
2. 每个中文名取得票最多的英文名；得票要够多、而且占这个中文名全部票数的大多数，才算定下来。
3. 用定下来的队名对照，按（联赛、赛季、主队、客队）找比赛：联赛里同一主客队一个赛季只踢一次。
   再核对日期和比分，不一致的列出来。
4. 对上号的比赛里，把懂球帝的“最新”赔率和 football-data 的收盘赔率比一比，看是不是同一个东西。

懂球帝赛季页的日期是北京时间，没有年份：我们数据库有开球时间（2019/20 起）就换成北京时间的日期再比，
没有的只能用英国当地日期，所以允许差 1 天。

运行方法：Claude 运行 `.venv\Scripts\python -m src.clean.dongqiudi_match`
结果写在 data/reports/：dongqiudi_队名对照.csv、dongqiudi_比赛对照.csv、dongqiudi_对照报告.md
"""
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.clean.dongqiudi_reader import parse_fixtures, parse_odds, parse_season_page, read_json, read_text, resolve_ts
from src.common.config import load_settings, resolve_path
from src.common.db import connect

log = logging.getLogger("dongqiudi_match")

MIN_VOTES = 3        # 一个中文名至少得几票才定下来
MIN_SHARE = 0.6      # 得票最多的英文名要占这个中文名全部票数的多少
# 懂球帝的公司名（打了星号）→ 我们数据库 odds 表里的公司名，用来核对“最新 = 收盘”
BOOKMAKERS = {"Bet3**": "Bet365", "威廉希*": "William Hill", "平*": "平博 Pinnacle"}


@dataclass
class DbMatch:
    match_id: int
    season: int
    day: date            # 北京时间的日期（有开球时间时），否则是当地日期
    exact_day: bool      # day 是不是从开球时间换算来的
    home: str
    away: str
    home_goals: int | None
    away_goals: int | None
    kickoff_utc: str | None = None  # 开球时间（标准时间），2019/20 起才有


def load_db_matches(con, competition: str) -> dict[int, list[DbMatch]]:
    """数据库里一个联赛的全部比赛（带比分），按赛季分组。"""
    out = defaultdict(list)
    rows = con.execute(
        "select m.match_id, m.season, m.match_date, m.kickoff_utc, m.home_team, m.away_team, s.home_goals, s.away_goals"
        " from matches m left join match_stats s on s.match_id = m.match_id and s.source = 'football-data'"
        " where m.competition = ?", (competition,))
    for mid, season, mdate, ko, home, away, hg, ag in rows:
        if ko:
            day = (datetime.strptime(ko, "%Y-%m-%d %H:%M") + timedelta(hours=8)).date()  # 标准时间 → 北京时间
        else:
            day = date.fromisoformat(mdate)
        out[season].append(DbMatch(mid, season, day, bool(ko), home, away, hg, ag, ko))
    return out


def guess_year(month: int, day: int, season: int, db: list[DbMatch]) -> date | None:
    """懂球帝的“月-日”是哪一年：赛季跨两年，选落在这个赛季比赛日期范围里的那一年。"""
    lo = min(m.day for m in db) - timedelta(days=3) if db else date(season, 7, 1)
    hi = max(m.day for m in db) + timedelta(days=3) if db else date(season + 1, 7, 31)
    for y in (season, season + 1):
        try:
            d = date(y, month, day)
        except ValueError:
            continue
        if lo <= d <= hi:
            return d
    return None


def load_dqd_seasons(raw: Path, league: str) -> dict:
    """原始层里一个联赛已存的赛季页 → {赛季: 比赛列表}。"""
    folder = raw / "dongqiudi" / "season" / league
    out = {}
    for f in sorted(folder.glob("*.html.gz")) if folder.exists() else []:
        if ".suspicious." not in f.name:
            out[int(f.name.split(".")[0])] = parse_season_page(read_text(f))
    return out


def vote(dqd: dict, db: dict, votes: dict):
    """第 1 步：按“日期差 ≤ 1 天、比分一样”投票，结果累加到 votes[中文名][英文名]。"""
    for season, games in dqd.items():
        by_score = defaultdict(list)
        for m in db.get(season, []):
            by_score[(m.home_goals, m.away_goals)].append(m)
        for g in games:
            d = guess_year(g.month, g.day, season, db.get(season, []))
            if d is None:
                continue
            cands = [m for m in by_score[(g.home_goals, g.away_goals)] if abs((m.day - d).days) <= 1]
            for m in cands:
                votes[g.home][m.home] += 1 / len(cands)
                votes[g.away][m.away] += 1 / len(cands)


def decide(votes: dict) -> tuple[dict, list[dict]]:
    """第 2 步：每个中文名取得票最多的英文名。返回（定下来的对照, 报告表格）。"""
    mapping, table = {}, []
    for zh, c in sorted(votes.items()):
        total = sum(c.values())
        ranked = sorted(c.items(), key=lambda kv: -kv[1])
        en, top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ("", 0)
        ok = top >= MIN_VOTES and top / total >= MIN_SHARE
        if ok:
            mapping[zh] = en
        table.append({"中文名": zh, "英文名": en, "得票": round(top, 1), "占比": round(top / total, 3),
                      "第二名": second[0], "第二名得票": round(second[1], 1), "结果": "定下" if ok else "待人工确认"})
    return mapping, table


def match_games(league: str, dqd: dict, db: dict, mapping: dict) -> tuple[list[dict], dict]:
    """第 3 步：按（赛季、主队、客队）对比赛，再核对日期和比分。返回（每场一行, 每个赛季数据库里没对上的场数）。"""
    rows, db_left = [], {}
    for season, games in sorted(dqd.items()):
        idx = {(m.home, m.away): m for m in db.get(season, [])}
        used = set()
        for g in games:
            row = {"联赛": league, "赛季": season, "懂球帝编号": g.match_id, "轮次": g.round, "懂球帝日期": f"{g.month:02d}-{g.day:02d}",
                   "主队": g.home, "客队": g.away, "比分": f"{g.home_goals}-{g.away_goals}",
                   "我们的编号": "", "我们的日期": "", "我们的开球": "", "我们的比分": "", "结果": ""}
            home, away = mapping.get(g.home), mapping.get(g.away)
            m = idx.get((home, away)) if home and away else None
            if not (home and away):
                row["结果"] = "队名没对上"
            elif m is None:
                row["结果"] = "数据库里没有这场"
            else:
                used.add(m.match_id)
                d = guess_year(g.month, g.day, season, db[season])
                gap = abs((m.day - d).days) if d else 99
                row.update({"我们的编号": m.match_id, "我们的日期": m.day.isoformat(), "我们的开球": m.kickoff_utc or "",
                            "我们的比分": f"{m.home_goals}-{m.away_goals}"})
                if (m.home_goals, m.away_goals) != (g.home_goals, g.away_goals):
                    row["结果"] = "比分不同"
                elif gap > (0 if m.exact_day else 1):
                    row["结果"] = f"日期差 {gap} 天"
                else:
                    row["结果"] = "对上"
            rows.append(row)
        db_left[season] = len([m for m in db.get(season, []) if m.match_id not in used])
    return rows, db_left


def compare_odds(con, raw: Path, rows: list[dict]) -> dict:
    """第 4 步：懂球帝“最新”赔率 vs football-data 收盘赔率。

    试验发现几件事，所以按“最新”赔率的时间分组数：
    - 2023/24 起，“最新”的时间都在开球前，基本就是收盘。
    - 2020/21–2022/23 很多场的时间是开球后 6~8 小时，但赔率和收盘一样：像是懂球帝后来同步数据时改了时间。
    - 少数是比赛进行中的赔率（例如主胜 1.01、客胜 151），时间在开球后不久，也有更晚的。
    - football-data 的让球、大小球收盘只有一条“主盘口”（大小球只有 2.5 球），懂球帝记的是每家自己的主盘口。
      两边盘口不一样就没法比赔率，只数“盘口不同”。
    返回 {(公司, 玩法, 时间分组): {各种情况: 场数}}。只看 2019/20 起有开球时间的比赛。
    """
    stats = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["结果"] != "对上" or not r["我们的开球"]:
            continue
        path = raw / "dongqiudi" / "odds_final" / r["联赛"] / str(r["赛季"]) / f"{r['懂球帝编号']}.json.gz"
        if not path.exists():
            continue
        kickoff = datetime.strptime(r["我们的开球"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        close = {(b, mk): (line, h, d, a, o, u) for b, mk, line, h, d, a, o, u in con.execute(
            "select bookmaker, market, line, home, draw, away, over, under from odds"
            " where match_id = ? and source = 'football-data' and timing = 'close'", (r["我们的编号"],))}
        for o in parse_odds(read_json(path)):
            en = BOOKMAKERS.get(o.bookmaker)
            if o.timing != "now" or en is None or (en, o.market) not in close:
                continue
            ts = resolve_ts(o.ts_text, kickoff)
            if ts is None:
                group = "没有时间"
            elif ts <= kickoff:
                group = "开球前"
            elif ts <= kickoff + timedelta(hours=2.5):
                group = "开球后 2.5 小时内"
            else:
                group = "开球 2.5 小时以后"
            s = stats[(o.bookmaker, o.market, group)]
            s["场数"] += 1
            ref = close[(en, o.market)]
            theirs = ref[1:4] if o.market != "ou" else (ref[4], None, ref[5])
            if o.market != "1x2" and (ref[0] is None or o.line is None or abs(ref[0] - o.line) > 1e-6):
                s["盘口不同"] += 1
                continue
            gap = max(abs(x - y) for x, y in zip((o.home, o.draw, o.away), theirs) if x is not None and y is not None)
            s["一样（差≤0.01）"] += gap <= 0.011
            s["接近（差≤0.05）"] += 0.011 < gap <= 0.051
            s["差得多"] += gap > 0.051
    return {k: dict(v) for k, v in stats.items()}


def check_kickoffs(con, raw: Path, league: str, mapping: dict) -> list[tuple]:
    """当前赛季赛程接口里的开球时间（start_play，我们当作标准时间）和数据库开球时间比。返回 [(差几分钟, 场数)]。"""
    folder = raw / "dongqiudi" / "fixtures" / league
    seen = {}
    for f in sorted(folder.glob("*.json.gz")) if folder.exists() else []:
        if ".suspicious." not in f.name:
            for fx in parse_fixtures(read_json(f)):
                seen[fx.match_id] = fx
    db = {(h, a): ko for h, a, ko in con.execute(
        "select home_team, away_team, kickoff_utc from matches where competition = ? and kickoff_utc is not null"
        " and match_date >= date('now', '-60 days')", (league,))}
    diffs = defaultdict(int)
    for fx in seen.values():
        ko = db.get((mapping.get(fx.home), mapping.get(fx.away)))
        if ko and fx.kickoff_utc:
            ours = datetime.strptime(ko, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            diffs[round((fx.kickoff_utc - ours).total_seconds() / 60)] += 1
    return sorted(diffs.items())


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # 带 BOM，Excel 打开中文不乱码
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    s = load_settings()
    raw, reports = resolve_path(s["paths"]["raw"]), resolve_path(s["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    con = connect()
    leagues = list(s["dongqiudi"]["league_ids"])
    dqd = {lg: load_dqd_seasons(raw, lg) for lg in leagues}
    db = {lg: load_db_matches(con, lg) for lg in leagues}

    votes = defaultdict(lambda: defaultdict(float))
    for lg in leagues:
        vote(dqd[lg], db[lg], votes)
    mapping, team_table = decide(votes)
    # 两个中文名对到同一个英文名：可能是改过名，也可能是错配，列出来
    by_en = defaultdict(list)
    for zh, en in mapping.items():
        by_en[en].append(zh)
    shared = {en: zhs for en, zhs in by_en.items() if len(zhs) > 1}

    all_rows, lines = [], []
    lines += ["# 懂球帝 ↔ 数据库 比赛对照（试验）", "",
              "程序：`src/clean/dongqiudi_match.py`。只出报告，不改数据库。", "",
              f"中文队名 {len(team_table)} 个，定下来 {len(mapping)} 个，待人工确认 {len(team_table) - len(mapping)} 个。", ""]
    if shared:
        lines += ["几个中文名对到同一个英文名（要看一眼）：", ""]
        lines += [f"- {en} ← {'、'.join(zhs)}" for en, zhs in shared.items()]
        lines += [""]
    lines += ["## 每个赛季的对照结果", "", "| 联赛 | 赛季 | 懂球帝场数 | 对上 | 其他情况 | 数据库没对上 |", "|---|---|---|---|---|---|"]
    for lg in leagues:
        rows, left = match_games(lg, dqd[lg], db[lg], mapping)
        all_rows += rows
        for season in sorted(dqd[lg]):
            rs = [r for r in rows if r["赛季"] == season]
            other = defaultdict(int)
            for r in rs:
                if r["结果"] != "对上":
                    other[r["结果"]] += 1
            lines.append(f"| {lg} | {season}/{(season + 1) % 100:02d} | {len(rs)} | {sum(r['结果'] == '对上' for r in rs)} | "
                         f"{'，'.join(f'{k} {v}' for k, v in other.items()) or '—'} | {left[season]} |")
    cols = ["场数", "盘口不同", "一样（差≤0.01）", "接近（差≤0.05）", "差得多"]
    lines += ["", "## 懂球帝“最新”赔率是不是收盘赔率", "",
              "只比对上号、2019/20 起（有开球时间）、而且原始层里有赔率文件的比赛。和 football-data 的收盘赔率比，看三个价格里差得最多的那个。", "",
              "| 公司 | 玩法 | “最新”的时间 | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    order = {"开球前": 0, "开球后 2.5 小时内": 1, "开球 2.5 小时以后": 2, "没有时间": 3}
    for (bk, mk, g), st in sorted(compare_odds(con, raw, all_rows).items(), key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]])):
        lines.append(f"| {bk} | {mk} | {g} | " + " | ".join(str(st.get(c, 0)) for c in cols) + " |")
    lines += ["", "## 当前赛季开球时间核对（懂球帝 start_play − 数据库开球时间）", ""]
    for lg in leagues:
        diffs = check_kickoffs(con, raw, lg, mapping)
        lines.append(f"- {lg}：" + ("，".join(f"差 {m} 分钟 {n} 场" for m, n in diffs) if diffs else "没有可比的比赛"))
    con.close()

    write_csv(reports / "dongqiudi_队名对照.csv", team_table)
    write_csv(reports / "dongqiudi_比赛对照.csv", all_rows)
    (reports / "dongqiudi_对照报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("写好了：%s", reports / "dongqiudi_对照报告.md")


if __name__ == "__main__":
    main()
