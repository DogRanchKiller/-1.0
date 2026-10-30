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


def league_codes(settings: dict) -> list[str]:
    """第一阶段的联赛代码（settings.yaml 的 leagues）。其他赛事代码都是杯赛、欧战。"""
    return [*settings["leagues"]["top"], *settings["leagues"]["second"]]


# 小组赛、联赛阶段（不是淘汰赛，没有“谁晋级”）
GROUP_ROUNDS = ("G", "G1", "G2", "LP")


def _final_score(hg, ag, et, aet_h, aet_a):
    """这场比赛最后的比分：打了加时用加时后的比分，没打用 90 分钟比分。不知道就返回 None。"""
    if et:
        return (aet_h, aet_a) if aet_h is not None else None
    return (hg, ag) if hg is not None else None


def cup_checks(con: sqlite3.Connection, leagues: list[str]) -> list[tuple[str, str, list[str]]]:
    """杯赛、欧战淘汰赛的检查：两回合加起来的总比分、点球、谁晋级是不是对得上。

    数据库只存了“至少一方是我们的球队”的比赛，所以这里查的也只是这些淘汰赛对阵。
    一个对阵 = 同一赛事、同一赛季、同一轮（重赛算同一轮）、同两支球队的全部比赛。
    """
    ph = ",".join("?" * len(leagues))
    rows = _rows(con,
        "select m.match_id, m.competition, m.season, m.round, m.leg, m.match_date, m.home_team, m.away_team, "
        "m.status, s.home_goals, s.away_goals, c.extra_time, c.aet_home, c.aet_away, c.pen_home, c.pen_away "
        "from matches m left join match_stats s using (match_id) "
        "left join cup_details c on c.match_id = m.match_id and c.source = s.source "
        f"where m.competition not in ({ph}) and m.round != '' and m.status in ('finished', 'awarded') "
        "order by m.match_date, m.leg, m.match_id", leagues)
    seen: set[int] = set()
    ties: dict[tuple, list[tuple]] = {}
    appear: dict[tuple, list[tuple]] = {}        # (赛事, 赛季, 球队) → [(日期, 轮次)]
    for r in rows:
        if r[0] in seen:        # 一场比赛有几个来源时只看第一个
            continue
        seen.add(r[0])
        mid, comp, season, rnd, leg, day, home, away = r[:8]
        appear.setdefault((comp, season, home), []).append((day, rnd))
        appear.setdefault((comp, season, away), []).append((day, rnd))
        if rnd in GROUP_ROUNDS:
            continue
        base = rnd.split("_replay")[0]
        ties.setdefault((comp, season, base, frozenset((home, away))), []).append(r)

    wrong_pen, half, unclear, loser_again = [], [], [], []
    for (comp, season, base, _pair), ms in sorted(ties.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        label = f"{comp} {season_label(season)} {base} {ms[0][6]} – {ms[0][7]}"
        if any(m[8] == "awarded" for m in ms):
            continue            # 判罚比分的对阵不查
        legs = sorted(m[4] for m in ms)
        two_leg = any(l in (1, 2) for l in legs)
        if two_leg and legs != [1, 2]:
            half.append(f"{label}：回合是 {legs}")
            continue
        winner = None
        if two_leg:
            first, second = sorted(ms, key=lambda m: m[4])
            f1, f2 = _final_score(*first[9:14]), _final_score(*second[9:14])
            if f1 is None or f2 is None:
                continue
            # 按球队加总（第二回合主客队对调）
            a, b = first[6], first[7]
            goals = {a: f1[0] + (f2[1] if second[7] == a else f2[0]),
                     b: f1[1] + (f2[0] if second[6] == b else f2[1])}
            pens = second[14:16]
            if goals[a] != goals[b]:
                winner = a if goals[a] > goals[b] else b
                if pens[0] is not None:
                    wrong_pen.append(f"{label}：总比分 {goals[a]}–{goals[b]} 不是平局，却记了点球 {pens[0]}–{pens[1]}")
            elif pens[0] is not None:
                winner = second[6] if pens[0] > pens[1] else second[7]
            elif second[7] != a:
                unclear.append(f"{label}：两回合主客队没对调，又是平局，看不出谁晋级")
            else:
                away_a = f2[1]          # a 在第二回合是客队，这一回合的进球就是它的客场进球
                away_b = f1[1]          # b 在第一回合是客队
                if away_a != away_b:
                    winner = a if away_a > away_b else b         # 客场进球多的晋级（2021/22 以前的规则）
                else:
                    unclear.append(f"{label}：总比分 {goals[a]}–{goals[b]}、客场进球一样，又没有点球，看不出谁晋级")
        else:
            last = ms[-1]                     # 有重赛时看最后一场
            fs = _final_score(*last[9:14])
            if fs is None:
                continue
            pens = last[14:16]
            if fs[0] != fs[1]:
                winner = last[6] if fs[0] > fs[1] else last[7]
                if pens[0] is not None:
                    wrong_pen.append(f"{label}：比分 {fs[0]}–{fs[1]} 不是平局，却记了点球 {pens[0]}–{pens[1]}")
            elif pens[0] is not None:
                winner = last[6] if pens[0] > pens[1] else last[7]
            elif base not in ("3P",):
                unclear.append(f"{label}：最后一场 {fs[0]}–{fs[1]} 平局，没有点球也没有重赛（可能重赛没导入）")
        if winner is None or base in ("F", "3P"):
            continue
        loser = next(iter(_pair - {winner}), None)
        last_day = max(m[5] for m in ms)
        later = [(d, r) for d, r in appear.get((comp, season, loser), []) if d > last_day and r != "3P"]
        if later:
            loser_again.append(f"{label}：{loser} 被淘汰了，{later[0][0]} 又在 {later[0][1]} 出场")

    return [
        ("杯赛点球和比分对不上", "只有比分（两回合时是总比分）打平才会踢点球", wrong_pen),
        ("杯赛两回合缺一回合", "两回合的对阵应该正好有第 1、第 2 回合", half),
        ("杯赛看不出谁晋级", "淘汰赛对阵按比分、点球、客场进球都分不出胜负（多半是重赛或某一回合没导入）", unclear),
        ("杯赛被淘汰的队后面又出场", "输掉淘汰赛的球队，同一赛事同一赛季后面不应该再出场（三四名决赛除外）", loser_again),
    ]


def run_checks(con: sqlite3.Connection, settings: dict, today: date | None = None) -> list[dict]:
    """跑全部检查。返回 [{名称, 说明, rows: [每个问题一行文字]}]。"""
    q = settings["quality"]
    now = current_season(today)
    checks = []

    def add(name, explain, rows):
        checks.append({"名称": name, "说明": explain, "rows": rows})

    leagues = league_codes(settings)
    ph = ",".join("?" * len(leagues))
    is_league = f"competition in ({ph})"      # 联赛结构的检查（场数、主客场数等）只查联赛
    m = "matches m join match_stats s using (match_id)"
    # 杯赛打了加时、网页上又没写 90 分钟比分的，90 分钟比分本来就不知道（不是漏了），单独列
    et_unknown = ("exists (select 1 from cup_details c where c.match_id = m.match_id and c.extra_time = 1 "
                  "and c.aet_home is not null)")
    add("比分缺失", "已经踢完的比赛，没有全场比分（杯赛打了加时、90 分钟比分本来不知道的不算，见下一项）",
        [f"{c} {season_label(se)} {d} {h} - {a}" for c, se, d, h, a in _rows(con,
         f"select competition, season, match_date, home_team, away_team from {m} "
         f"where status = 'finished' and (home_goals is null or away_goals is null) and not {et_unknown} "
         "order by match_date")])

    add("加时赛 90 分钟比分不知道", "杯赛打了加时，网页上只有加时后的比分、进球分钟也不全，90 分钟比分算不出来"
        "（不是错误，只是用 90 分钟比分的地方要跳过这些比赛）",
        [f"{c} {season_label(se)} {d} {h} - {a}" for c, se, d, h, a in _rows(con,
         f"select competition, season, match_date, home_team, away_team from {m} "
         f"where status = 'finished' and home_goals is null and {et_unknown} order by match_date")])

    add("赛果和比分对不上", "FTR（H/D/A）和全场比分算出来的胜负不一样",
        [f"{c} {d} {h} {hg}-{ag} {a}，赛果写的是 {r}" for c, d, h, a, hg, ag, r in _rows(con,
         f"select competition, match_date, home_team, away_team, home_goals, away_goals, result from {m} "
         "where result is not null and home_goals is not null and result != "
         "case when home_goals > away_goals then 'H' when home_goals < away_goals then 'A' else 'D' end")])

    add("主客队是同一支", "主队和客队是同一个球队编号（多半是队名对照表合并错了）",
        [f"{c} {d} {h}" for c, d, h in _rows(con,
         "select competition, match_date, home_team from matches where home_team = away_team")])

    add("同一天同一队踢两场", "一支球队同一天出现在两场比赛里（联赛、杯赛一起查；多半是队名对照表合并错了，或者重复记录）。"
        "杯赛里只知道是哪一轮、日期是这一轮的日子的比赛不查",
        [f"{d} {t}：{n} 场" for d, t, n in _rows(con,
         "select match_date, team, count(*) from (select match_date, home_team team, date_precision from matches "
         "union all select match_date, away_team, date_precision from matches) where date_precision = 'day' "
         "group by match_date, team having count(*) > 1")])

    seasons = _rows(con, "select competition, season, count(*), count(distinct home_team) from matches "
                         f"where season < ? and {is_league} group by competition, season", (now, *leagues))
    add("赛季场数不对", f"完赛赛季的场数应该是 球队数 ×（球队数 − 1）（每两队主客各一场）；{season_label(now)} 还在进行，不查",
        [f"{c} {season_label(se)}：{teams} 队应该 {teams * (teams - 1)} 场，实际 {n} 场"
         for c, se, n, teams in seasons if n != teams * (teams - 1)])

    add("主客场数不一样", "完赛赛季里，一支球队的主场数和客场数应该相同",
        [f"{c} {season_label(se)} {t}：主场 {h}，客场 {a}" for c, se, t, h, a in _rows(con,
         "select competition, season, team, sum(home), sum(away) from ("
         " select competition, season, home_team team, 1 home, 0 away from matches"
         " union all select competition, season, away_team, 0, 1 from matches) "
         f"where season < ? and {is_league} group by competition, season, team having sum(home) != sum(away)",
         (now, *leagues))])

    add("日期不在赛季范围内", "联赛比赛日期应该在 赛季开始年 7 月 1 日 到 次年 8 月 31 日之间；"
        "杯赛、欧战的资格赛 6 月就开始，从 6 月 1 日算",
        [f"{c} {season_label(se)} {d} {h} - {a}" for c, se, d, h, a in _rows(con,
         "select competition, season, match_date, home_team, away_team from matches "
         f"where match_date < (season || case when {is_league} then '-07-01' else '-06-01' end) "
         "or match_date > ((season + 1) || '-08-31')", leagues)])

    add("开球时间缺失", "football-data 从 2019/20 起有开球时间，这之后的联赛比赛应该都有（杯赛不查：维基百科常常不写）",
        [f"{c} {d} {h} - {a}" for c, d, h, a in _rows(con,
         "select competition, match_date, home_team, away_team from matches "
         f"where season >= 2019 and kickoff_utc is null and {is_league}", leagues)])

    for name, explain, rows in cup_checks(con, leagues):
        add(name, explain, rows)

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
