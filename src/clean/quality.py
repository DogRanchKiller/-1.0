"""数据质量检查（需求文档第 16 节“自动检查”：赔率异常、比赛重复、比分缺失、队名没匹配上）。

检查的是数据库里已经清洗好的数据，所以以后国内网站的数据进来也用这一套。
每项检查返回有问题的行；没问题就是空的。结果写进 data/reports/ 的报告，以后显示在数据状态页。
这些问题不会自动修改：先看清楚是原始数据本身就这样，还是清洗程序写错了，再决定怎么办。
"""
import sqlite3
from datetime import date

from src.clean.football_data_inventory import season_label


def current_season(today: date | None = None) -> int:
    """今天属于哪个赛季（开始年份）：7 月起算新赛季。"""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _rows(con: sqlite3.Connection, sql: str, params=()) -> list[tuple]:
    return con.execute(sql, params).fetchall()


def run_checks(con: sqlite3.Connection, settings: dict, today: date | None = None) -> list[dict]:
    """跑全部检查。返回 [{名称, 说明, rows: [每个问题一行文字]}]。"""
    q = settings["quality"]
    now = current_season(today)
    checks = []

    def add(name, explain, rows):
        checks.append({"名称": name, "说明": explain, "rows": rows})

    m = "matches m join match_stats s using (match_id)"
    add("比分缺失", "已经踢完的比赛，没有全场比分",
        [f"{c} {season_label(se)} {d} {h} - {a}" for c, se, d, h, a in _rows(con,
         f"select competition, season, match_date, home_team, away_team from {m} "
         "where status = 'finished' and (home_goals is null or away_goals is null) order by match_date")])

    add("赛果和比分对不上", "FTR（H/D/A）和全场比分算出来的胜负不一样",
        [f"{c} {d} {h} {hg}-{ag} {a}，赛果写的是 {r}" for c, d, h, a, hg, ag, r in _rows(con,
         f"select competition, match_date, home_team, away_team, home_goals, away_goals, result from {m} "
         "where result is not null and home_goals is not null and result != "
         "case when home_goals > away_goals then 'H' when home_goals < away_goals then 'A' else 'D' end")])

    add("主客队是同一支", "主队和客队是同一个球队编号（多半是队名对照表合并错了）",
        [f"{c} {d} {h}" for c, d, h in _rows(con,
         "select competition, match_date, home_team from matches where home_team = away_team")])

    add("同一天同一队踢两场", "一支球队同一天出现在两场比赛里（多半是队名对照表合并错了，或者重复记录）",
        [f"{d} {t}：{n} 场" for d, t, n in _rows(con,
         "select match_date, team, count(*) from (select match_date, home_team team from matches "
         "union all select match_date, away_team from matches) group by match_date, team having count(*) > 1")])

    seasons = _rows(con, "select competition, season, count(*), count(distinct home_team) from matches "
                         "where season < ? group by competition, season", (now,))
    add("赛季场数不对", f"完赛赛季的场数应该是 球队数 ×（球队数 − 1）（每两队主客各一场）；{season_label(now)} 还在进行，不查",
        [f"{c} {season_label(se)}：{teams} 队应该 {teams * (teams - 1)} 场，实际 {n} 场"
         for c, se, n, teams in seasons if n != teams * (teams - 1)])

    add("主客场数不一样", "完赛赛季里，一支球队的主场数和客场数应该相同",
        [f"{c} {season_label(se)} {t}：主场 {h}，客场 {a}" for c, se, t, h, a in _rows(con,
         "select competition, season, team, sum(home), sum(away) from ("
         " select competition, season, home_team team, 1 home, 0 away from matches"
         " union all select competition, season, away_team, 0, 1 from matches) "
         "where season < ? group by competition, season, team having sum(home) != sum(away)", (now,))])

    add("日期不在赛季范围内", "比赛日期应该在 赛季开始年 7 月 1 日 到 次年 8 月 31 日之间",
        [f"{c} {season_label(se)} {d} {h} - {a}" for c, se, d, h, a in _rows(con,
         "select competition, season, match_date, home_team, away_team from matches "
         "where match_date < (season || '-07-01') or match_date > ((season + 1) || '-08-31')")])

    add("开球时间缺失", "football-data 从 2019/20 起有开球时间，这之后的比赛应该都有",
        [f"{c} {d} {h} - {a}" for c, d, h, a in _rows(con,
         "select competition, match_date, home_team, away_team from matches "
         "where season >= 2019 and kickoff_utc is null")])

    # 抽水检查：赔率倒数之和 = 公司给的各结果概率之和，比 1 多出来的部分就是公司的抽水。
    # 三类数据的正常范围不一样（数字在 settings.yaml 的 quality.overround）：
    #   普通公司：一定大于 1；交易所：接近 1；多家最高：每个结果各挑一家最高的，常常低于 1。
    r = q["overround"]
    exchanges, best_of = q["exchanges"], q["best_of"]

    def bounds(normal):
        """拼一段 SQL：按公司类别取正常范围的下限、上限。"""
        ex = ", ".join("?" * len(exchanges))
        be = ", ".join("?" * len(best_of))
        sql = (f"case when bookmaker in ({ex}) then ? when bookmaker in ({be}) then ? else ? end",
               f"case when bookmaker in ({ex}) then ? when bookmaker in ({be}) then ? else ? end")
        lo = [*exchanges, r["exchange"][0], *best_of, r["best"][0], r[normal][0]]
        hi = [*exchanges, r["exchange"][1], *best_of, r["best"][1], r[normal][1]]
        return sql, lo + hi

    rule = (f"普通公司 {r['1x2'][0]}–{r['1x2'][1]}（大小球、亚盘 {r['2way'][0]}–{r['2way'][1]}）；"
            f"交易所 {r['exchange'][0]}–{r['exchange'][1]}；多家最高 {r['best'][0]}–{r['best'][1]}")

    (lo_sql, hi_sql), params = bounds("1x2")
    add("胜平负赔率抽水异常", f"三个赔率倒数之和应该在正常范围里：{rule}。超出的多半是录错。按公司统计",
        [f"{b} {t}：{n} 组，例如 {c} {d} {h} - {a}（{oh} / {od} / {oa}）" for b, t, n, c, d, h, a, oh, od, oa in _rows(con,
         "select bookmaker, timing, count(*), competition, match_date, home_team, away_team, o.home, o.draw, o.away "
         "from odds o join matches using (match_id) where market = '1x2' and o.home and o.draw and o.away "
         f"and (1 / o.home + 1 / o.draw + 1 / o.away not between {lo_sql} and {hi_sql}) group by bookmaker, timing "
         "order by count(*) desc", params)])

    (lo_sql, hi_sql), params = bounds("2way")
    two = "1 / coalesce(o.over, o.home) + 1 / coalesce(o.under, o.away)"
    add("大小球、亚盘赔率抽水异常", f"两个赔率倒数之和应该在正常范围里：{rule}。按公司和玩法统计",
        [f"{b} {mk} {t}：{n} 组，例如 {c} {d} {h} - {a}（{x} / {y}）" for b, mk, t, n, c, d, h, a, x, y in _rows(con,
         "select bookmaker, market, timing, count(*), competition, match_date, home_team, away_team, "
         "coalesce(o.over, o.home), coalesce(o.under, o.away) from odds o join matches using (match_id) "
         "where market in ('ou', 'ah') and coalesce(o.over, o.home) and coalesce(o.under, o.away) "
         f"and ({two} not between {lo_sql} and {hi_sql}) "
         "group by bookmaker, market, timing order by count(*) desc", params)])
    return checks


def report_markdown(result: dict, checks: list[dict], max_rows: int = 30) -> str:
    """导入结果 + 质量检查 → 报告文字。"""
    out = ["# football-data 导入和质量检查（自动生成）", "",
           f"- 导入了 {result['files']} 个文件、{result['matches']} 场比赛、{result['odds_rows']} 组赔率。",
           f"- 低于 1.01 的错误赔率（例如 0）{result['dropped_odds']} 个，没有导入。", ""]
    out += ["## 待确认的队名", "",
            "这些队名 config/teams.yaml 里没有，比赛没导入。确认是哪支球队后加进 teams.yaml，再导一次。", ""]
    out += [f"- {lg} {se}：{n}（{c} 次）" for (lg, se, n), c in sorted(result["unknown"].items())] or ["- 没有"]
    out += ["", "## 导入时处理过的问题", ""] + ([f"- {i}" for i in result["issues"]] or ["- 没有"])
    out += ["", "## 质量检查", ""]
    for c in checks:
        out += [f"### {c['名称']}：{'没问题' if not c['rows'] else f'{len(c['rows'])} 条'}", "", c["说明"], ""]
        out += [f"- {r}" for r in c["rows"][:max_rows]]
        if len(c["rows"]) > max_rows:
            out.append(f"- ……还有 {len(c['rows']) - max_rows} 条")
        out.append("")
    return "\n".join(out)
