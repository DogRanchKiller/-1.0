"""球队页的四块：近期状态、历史交锋、Elo 评分变化、积分榜（需求文档第 8 节“球队数据页怎么算”，工作树 TEAM）。

分两步：
  collect：生成网站读数据时（data.py 的 load 最后）算一次，结果放进 SiteData.team_stats。
    比分全部经过取数层（FeatureStore.historical_matches，截止时间 = 现在，用途「显示」），只用联赛比分。
    SiteData 里只放算好的几块（每队十几行），不放整张比分表：现读现出的网站每次都要把 SiteData 的内容过一遍算数据版本。
  render：画球队页时把这几块排成 HTML（pages.py 的 render_team 调用）。

只是显示：不出预测、不改模型、不写数据库。
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.features.known_time import to_utc_text
from src.features.rest_days import calendar_year_codes
from src.features.standings import season_table, standings_leagues
from src.features.usage import season_label
from src.models.elo import EloModel
from src.models.registry import model_config
from src.web.site.common import bj_text, e, match_href, placeholder, table

USAGE = "显示"
BAD = "球队数据暂时算不出来"
RESULT_NAMES = {"W": "胜", "D": "平", "L": "负"}


# ================= 算 =================
def team_rows(hist: pd.DataFrame, team: str) -> list[dict]:
    """这队踢过的联赛（从早到晚），每场换成“这队视角”的一行。"""
    return [_view(r, team) for r in by_team(hist, {team}).get(team, [])]


def by_team(hist: pd.DataFrame, teams: set[str]) -> dict[str, list[dict]]:
    """把比分表过一遍，分到 teams 每队名下（原样的行，从早到晚）。一队一队用 pandas 筛要几秒，过一遍不到半秒。"""
    out: dict[str, list[dict]] = {t: [] for t in teams}
    for r in hist[hist["home_team"].isin(teams) | hist["away_team"].isin(teams)].to_dict("records"):
        for t in {r["home_team"], r["away_team"]} & teams:
            out[t].append(r)
    return out


def _view(r: dict, team: str) -> dict:
    home = r["home_team"] == team
    gf, ga = (int(r["home_goals"]), int(r["away_goals"])) if home else (int(r["away_goals"]), int(r["home_goals"]))
    return {"match_id": int(r["match_id"]), "competition": r["competition"], "season": int(r["season"]),
            "kickoff_utc": r["kickoff_utc"] if isinstance(r.get("kickoff_utc"), str) else None,
            "match_date": str(r["match_date"])[:10], "home": home, "opponent": r["away_team"] if home else r["home_team"],
            "home_goals": int(r["home_goals"]), "away_goals": int(r["away_goals"]),
            "gf": gf, "ga": ga, "result": "W" if gf > ga else "D" if gf == ga else "L"}


def summary(rows: list[dict]) -> dict:
    """几胜几平几负、拿了几分（胜 3 平 1）、进几球丢几球。"""
    w = sum(r["result"] == "W" for r in rows)
    d = sum(r["result"] == "D" for r in rows)
    return {"n": len(rows), "w": w, "d": d, "l": len(rows) - w - d, "points": 3 * w + d,
            "gf": sum(r["gf"] for r in rows), "ga": sum(r["ga"] for r in rows)}


def elo_tracks(hist: pd.DataFrame, settings: dict, teams: set[str]) -> dict[str, list[dict]]:
    """从第一场联赛一场一场算 Elo，记下 teams 里每队每场的赛前、赛后评分（比本联赛基准高多少分）。

    用的是 gb_l7 里带的那套 Elo（settings 的 models.gbdt.elo），和单场页“实力对比”的 Elo 是同一个数：
    基准 = 这个联赛这个赛季开始时的“上赛季末联赛平均”。
    赛前评分是“换完赛季、拉回平均以后”的：所以“这场变了多少”= 赛后 − 赛前，只算这场比赛本身的变化。
    """
    elo = EloModel(name="elo", **model_config(settings["models"]["gbdt"]["elo"], settings))
    rows = hist[hist["result"].notna() & hist["home_goals"].notna()].sort_values(["kickoff_earliest_utc", "match_id"])
    out: dict[str, list[dict]] = {t: [] for t in teams}
    cols = ["match_id", "competition", "season", "home_team", "away_team", "home_goals", "away_goals", "result",
            "kickoff_earliest_utc"]
    for mid, comp, season, home, away, hg, ag, res, ko in rows[cols].itertuples(index=False, name=None):
        season = int(season)
        mark = home in out or away in out
        if mark:
            elo.rating_diff(comp, season, home, away)        # 先换赛季、给新队初始分（和模型里一样，不改这场的结果）
            before = _relative(elo, comp)
            pre = {t: before(t) for t in (home, away) if t in out}
        elo.process_row(mid, comp, season, home, away, hg, ag, res, ko)
        if mark:
            after = _relative(elo, comp)
            for t, b in pre.items():
                if b is None or after(t) is None:
                    continue
                out[t].append({"match_id": int(mid), "competition": comp, "season": season, "before": round(b, 1),
                               "after": round(after(t), 1)})
    return out


def _relative(elo: EloModel, comp: str):
    """返回一个函数：球队 → 现在的评分比本联赛基准高多少（没有评分是 None）。"""
    base = elo.season_base.get((comp, elo.current_season.get(comp)), elo.initial)
    return lambda t: (elo.ratings[(comp, t)] - base) if (comp, t) in elo.ratings else None


def context_match(data, team: str) -> dict | None:
    """接下来 team_days_ahead 天里这队第一场联赛（data.matches 里只有联赛，见 data.py 第 2 步）。"""
    now = to_utc_text(data.now)
    hi = to_utc_text(data.now + timedelta(days=data.settings["site"]["team_days_ahead"]))
    up = [m for m in data.matches.values() if team in (m["home_team"], m["away_team"]) and now <= m["kickoff_earliest_utc"] < hi
          and m.get("status") not in ("postponed", "cancelled")]
    return min(up, key=lambda m: (m["kickoff_earliest_utc"], m["match_id"])) if up else None


def collect(fs, data, teams: set[str]) -> dict:
    """算 teams 每队的四块，返回 {"teams": {队: {...}}, "tables": {"联赛|赛季": {...}}}。"""
    cfg = data.settings["site"]
    n_form, n_h2h = int(cfg["team_form_matches"]), int(cfg["team_h2h_matches"])
    # 只读网站要的联赛：site.team_skip_league_groups 里的联赛组（国际赛 intl：国家队不在网站上）不读，
    # 否则 Elo 要多算 4 万多场国家队比赛，整站读数据慢约 2.5 秒（2026-11-05 工作树 INA 量过）
    skip = set(cfg.get("team_skip_league_groups") or [])
    comps = [c for g, codes in data.settings["leagues"].items() if g not in skip for c in codes]
    hist = fs.historical_matches(data.now, USAGE, competition=comps, data=("比分",))
    hist = hist.dropna(subset=["home_goals", "away_goals"]).sort_values(["kickoff_earliest_utc", "match_id"])
    tracks = elo_tracks(hist, data.settings, teams)
    can_table = set(standings_leagues(data.settings))
    out_teams, tables = {}, {}
    raw = by_team(hist, teams)
    for team in sorted(teams):
        mine = raw[team]
        rows = [_view(r, team) for r in mine[-n_form:]]          # 近期状态那几场
        nxt = context_match(data, team)
        # 现在所在的联赛赛季：接下来有联赛就用那一场的，否则用最近踢完的那场的
        where = (nxt["competition"], int(nxt["season"])) if nxt else \
            ((rows[-1]["competition"], rows[-1]["season"]) if rows else None)
        opp = None
        if nxt:
            opp = nxt["away_team"] if nxt["home_team"] == team else nxt["home_team"]
        h2h = [_view(r, team) for r in mine if opp in (r["home_team"], r["away_team"])][-n_h2h:] if opp else []
        track = [t for t in tracks.get(team, []) if where and t["competition"] == where[0]]
        form_ids = {r["match_id"] for r in rows}
        info = {"form": rows[::-1], "form_summary": summary(rows),
                "opponent": opp, "h2h": h2h[::-1], "h2h_summary": summary(h2h),
                "elo": track[-int(cfg["team_elo_matches"]):],
                "elo_form": {t["match_id"]: t for t in tracks.get(team, []) if t["match_id"] in form_ids},
                "elo_other_league": bool(where) and any(t["competition"] != where[0] for t in tracks.get(team, [])[-int(cfg["team_elo_matches"]):]),
                "where": list(where) if where else None}
        if where:
            key = f"{where[0]}|{where[1]}"
            if key not in tables:
                tables[key] = _table(fs, data, where[0], where[1], can_table)
        out_teams[team] = info
    return {"teams": out_teams, "tables": tables}


def _table(fs, data, comp: str, season: int, can_table: set[str]) -> dict:
    """一个联赛赛季截止现在的积分榜。没有积分规则的联赛、规则没覆盖到的赛季写原因，不报错。"""
    if comp not in can_table:
        return {"rows": None, "note": "这个联赛还没有积分规则，不算积分榜"}
    try:
        t = season_table(fs, comp, season, data.now, USAGE, data.settings).at(data.now)
    except ValueError as err:
        return {"rows": None, "note": f"积分榜算不出来：{str(err).splitlines()[0]}"}
    return {"rows": t.to_dict("records"), "note": None}


# ================= 画 =================
def render(data, team: str, teams) -> str:
    """球队页的四块 HTML。teams 是 TeamFiles（对手、积分榜里的队链到他们的球队页）。"""
    stats = data.team_stats
    if data.team_stats_error or not stats or team not in stats.get("teams", {}):
        return "<h2>球队数据</h2>" + placeholder(BAD)
    info = stats["teams"][team]
    return (_form_html(data, info) + _h2h_html(data, team, info, teams) + _elo_html(data, team, info)
            + _table_html(data, team, info, stats["tables"], teams))


def _summary_text(s: dict) -> str:
    return f"{s['w']} 胜 {s['d']} 平 {s['l']} 负，拿了 {s['points']} 分，进 {s['gf']} 球、丢 {s['ga']} 球"


def _match_cell(data, r: dict, pre: str = "../") -> str:
    name = f"{'主' if r['home'] else '客'} 对 {data.zh(r['opponent'])}"
    if r["match_id"] in data.page_ids:
        return f"<td class='match'><a class='match' href='{match_href(r['match_id'], pre)}'>{e(name)}</a></td>"
    return f"<td class='match'>{e(name)}</td>"


def _when(r: dict, fmt: str) -> str:
    return bj_text(r["kickoff_utc"], fmt) if r["kickoff_utc"] else e(r["match_date"] if "%Y" in fmt else r["match_date"][5:])


def _score(r: dict) -> str:
    return f"<span class='score'>{r['home_goals']}–{r['away_goals']}</span> {RESULT_NAMES[r['result']]}"


def _season_text(data, comp: str, season: int) -> str:
    return str(season) if comp in calendar_year_codes(data.settings) else season_label(season)


def _form_html(data, info: dict) -> str:
    rows = info["form"]
    if not rows:
        return "<h2>近期状态</h2>" + placeholder("数据库里还没有这队踢完的联赛")
    body = [[_when(r, "%m-%d"), e(data.comp(r["competition"])), _match_cell(data, r), _score(r)] for r in rows]
    return (f"<h2>近期状态（最近 {len(rows)} 场联赛）</h2><p>{e(_summary_text(info['form_summary']))}。</p>"
            + table(["开球", "联赛", "比赛", "比分（主–客）和这队的胜负"], body))


def _h2h_html(data, team: str, info: dict, teams) -> str:
    opp = info["opponent"]
    if opp is None:
        return "<h2>历史交锋</h2>" + placeholder(f"接下来 {data.settings['site']['team_days_ahead']} 天没有联赛，不列交锋")
    head = f"<h2>历史交锋：对 {teams.link(opp, data.zh(opp), '../')}</h2>"
    rows = info["h2h"]
    if not rows:
        return head + placeholder("数据库里没有两队的联赛交锋")
    body = [[_when(r, "%Y-%m-%d"), e(data.comp(r["competition"])), _match_cell(data, r), _score(r)] for r in rows]
    return (head + f"<p>接下来要踢的对手。最近 {len(rows)} 场联赛交锋，{e(data.zh(team))} "
            f"{e(_summary_text(info['h2h_summary']))}。</p>"
            + table(["开球", "联赛", "比赛", "比分（主–客）和这队的胜负"], body))


def _elo_html(data, team: str, info: dict) -> str:
    track = info["elo"]
    if not track:
        return "<h2>Elo 评分变化</h2>" + placeholder("这队在现在这个联赛还没有踢完的比赛，没有评分变化")
    comp = track[-1]["competition"]
    now = track[-1]["after"]
    text = (f"Elo 实力评分，写的是比本联赛基准高多少分（0 ≈ 本联赛平均水平；基准 = 上赛季末这个联赛的平均分），"
            f"和单场页“实力对比”里的 Elo 是同一个数。{data.zh(team)} 现在 {now:+.0f}。"
            f"每场赢了加分、输了减分，对手越强、赢得越多，变得越多；新赛季开始时往平均拉回一些。")
    note = ""
    if info["elo_other_league"]:
        note = ("<p class='small muted'>这段时间里这队换过联赛（升级或降级）。Elo 每个联赛一套，"
                f"图上只画现在这个联赛（{e(data.comp(comp))}）的。</p>")
    body = []
    for r in info["form"]:                                   # 近期状态那几场，从近到远
        t = info["elo_form"].get(r["match_id"])
        if t is None:
            continue
        change = t["after"] - t["before"]
        cls = "hit" if change > 0.05 else "miss" if change < -0.05 else ""
        body.append([_when(r, "%m-%d"), _match_cell(data, r), _score(r), f"{t['before']:+.0f}", f"{t['after']:+.0f}",
                     f"<td class='num {cls}'>{change:+.1f}</td>"])
    return (f"<h2>Elo 评分变化（{e(data.comp(comp))}）</h2><p>{e(text)}</p>"
            + elo_svg(track, lambda s: _season_text(data, comp, s), f"{data.zh(team)} 最近 {len(track)} 场联赛赛后的 Elo")
            + note
            + (table(["开球", "比赛", "比分", "赛前", "赛后", "这场变了"], body) if body else ""))


def elo_svg(track: list[dict], season_name, title: str) -> str:
    """赛后评分的折线（内嵌 SVG，不用脚本）。虚线是 0（本联赛平均水平），竖线是换赛季的地方。"""
    w, h, left, right, top, bottom = 760, 220, 48, 16, 16, 30
    vals = [t["after"] for t in track]
    lo, hi = min(min(vals), 0.0) - 10, max(max(vals), 0.0) + 10
    n = len(track)

    def x(i):
        return left + (i / (n - 1) if n > 1 else 0.5) * (w - left - right)

    def y(v):
        return top + (hi - v) / (hi - lo) * (h - top - bottom)

    parts = [f"<svg class='chart' viewBox='0 0 {w} {h}' role='img' aria-label='{e(title)}'>",
             f"<line x1='{left}' x2='{w - right}' y1='{y(0):.1f}' y2='{y(0):.1f}' stroke='#5f7468' stroke-dasharray='3 3'/>"]
    for v in (lo + 10, hi - 10):
        parts.append(f"<text x='{left - 6}' y='{y(v) + 4:.1f}' text-anchor='end'>{v:+.0f}</text>")
    for i, t in enumerate(track):
        if i == 0 or t["season"] != track[i - 1]["season"]:
            xi = x(i) if i == 0 else (x(i) + x(i - 1)) / 2
            if i:
                parts.append(f"<line class='axis' x1='{xi:.1f}' x2='{xi:.1f}' y1='{top}' y2='{h - bottom}'/>")
            parts.append(f"<text x='{xi + 4:.1f}' y='{h - 10}'>{e(season_name(t['season']))}</text>")
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    parts.append(f"<polyline fill='none' stroke='#34d399' stroke-width='2' points='{pts}'><title>{e(title)}</title></polyline>")
    parts.append("</svg>")
    return "".join(parts)


def _table_html(data, team: str, info: dict, tables: dict, teams) -> str:
    where = info["where"]
    if not where:
        return "<h2>积分榜</h2>" + placeholder("数据库里还没有这队的联赛")
    comp, season = where
    head = f"<h2>积分榜：{e(data.comp(comp))} {e(_season_text(data, comp, season))}</h2>"
    t = tables.get(f"{comp}|{season}") or {"rows": None, "note": BAD}
    if t["rows"] is None:
        return head + placeholder(t["note"])
    body, attrs = [], []
    for r in t["rows"]:
        name = teams.link(r["team"], data.zh(r["team"]), "../") if r["team"] != team else e(data.zh(r["team"]))
        if r["tie_unresolved"]:
            name += " <span class='tag'>同分未分出</span>"
        pts = f"{r['points']}" + (f"（扣 {-r['deduction']} 分）" if r["deduction"] else "")
        body.append([str(r["position"]), f"<td class='match'>{name}</td>", str(r["played"]), str(r["won"]), str(r["drawn"]),
                     str(r["lost"]), f"{r['goals_for']}:{r['goals_against']}", f"{r['goal_diff']:+d}", e(pts)])
        attrs.append(" class='mine'" if r["team"] == team else "")
    return (head + f"<p class='small muted'>截至现在已经知道比分的联赛算进来；同分按这个联赛的规则排"
            f"（规则最后一级没有数据时按队名排，标“同分未分出”）。{e(data.zh(team))} 那一行标出来了。</p>"
            + table(["名次", "球队", "场", "胜", "平", "负", "进:失", "净胜", "积分"], body, row_attrs=attrs))
