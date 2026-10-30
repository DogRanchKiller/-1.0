"""维基百科杯赛、欧战赛程的自动检查：采集、解析、导入数据库、质量检查、交叉核对。

全部用手写的小段假网页（HTML）和内存里的临时数据库，不上网、不碰真实数据库。
网页的写法照着维基百科真实页面的结构简化（footballbox 方框、可折叠小方框、wikitable 表格）。
"""
import collections
import datetime as dt
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from src.clean import wikipedia_crosscheck as X
from src.clean import wikipedia_import as W
from src.clean.quality import cup_checks
from src.clean.schema import create_tables, migrate_matches
from src.collect import wikipedia as C
from src.common.config import load_settings
from src.common.db import FETCH_LOG_SCHEMA

D = dt.date


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def soup_of(html: str):
    return BeautifulSoup(html, "lxml")


def team_map() -> W.TeamMap:
    """测试用的小对照表：两支英格兰球队、一支西班牙球队。"""
    teams = {"Arsenal": {"英文": "Arsenal", "国家": "英格兰"},
             "Chelsea": {"英文": "Chelsea", "国家": "英格兰"},
             "Real Madrid": {"英文": "Real Madrid", "国家": "西班牙"}}
    alias = {"Arsenal F.C.": "Arsenal", "Arsenal": "Arsenal", "Chelsea F.C.": "Chelsea",
             "Real Madrid CF": "Real Madrid"}
    return W.TeamMap(alias=alias, not_ours={"Real Madrid Castilla": "皇马二队"}, teams=teams)


def ref(title, shown=None, country=""):
    return W.TeamRef(title=title, shown=shown or title, country=country)


# ---------------------------------------------------------------- 采集程序（不上网）

class FakeResponse:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}


class FakeSession:
    """假装的网络：按顺序返回事先准备好的回复，并记下请求了哪些网址。"""
    def __init__(self, replies):
        self.replies, self.urls, self.headers = list(replies), [], {}

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self.replies.pop(0)


def wiki_settings():
    return {"wikipedia": {"user_agent": "Test/0.1 (test)", "min_interval_seconds": 3, "retry_waits": [30, 120],
                          "base_url": "https://en.wikipedia.org/wiki/", "skip_subpages": [], "max_subpages": 5}}


PAGE = b'<html><div class="mw-parser-output"><p>ok</p></div></html>'


def downloader(tmp_path, replies):
    con = sqlite3.connect(":memory:")
    con.execute(FETCH_LOG_SCHEMA)
    waits, now = [], [0.0]
    d = C.WikiDownloader(con, tmp_path, wiki_settings(), session=FakeSession(replies),
                         sleep=lambda s: (waits.append(s), now.__setitem__(0, now[0] + s)), clock=lambda: now[0])
    return d, con, waits


def test_season_label_and_titles():
    assert C.season_label(2008) == "2008–09" and C.season_label(1999) == "1999–2000"
    comp = {"titles": [[2000, 2001, "{year} FA Charity Shield"], [2002, 2100, "{year} FA Community Shield"]]}
    assert C.titles_for(comp, 2001) == ["2001 FA Charity Shield"]
    assert C.page_url("https://en.wikipedia.org/wiki/", "2008–09 FA Cup").endswith("/wiki/2008%E2%80%9309_FA_Cup")
    assert C.file_stem("A: b/c") == "A__b_c"


def test_find_subpages_skips_rules_and_redlinks():
    html = ('<a href="/wiki/2010%E2%80%9311_UEFA_Champions_League_group_stage">g</a>'
            '<a href="/wiki/2010%E2%80%9311_UEFA_Champions_League_squads">s</a>'
            '<a href="/w/index.php?title=2010–11_UEFA_Champions_League_knockout&action=edit&redlink=1">r</a>'
            '<a href="/wiki/2010%E2%80%9311_UEFA_Champions_League_group_stage">again</a>')
    assert C.find_subpages(html, "2010–11 UEFA Champions League", ["squads"]) == \
        ["2010–11 UEFA Champions League group stage"]


def test_download_waits_retries_and_never_downloads_twice(tmp_path):
    d, con, waits = downloader(tmp_path, [FakeResponse(503), FakeResponse(429, headers={"Retry-After": "7"}),
                                          FakeResponse(200, PAGE)])
    status, html = d.fetch("2005–06 FA Cup", tmp_path / "FAC" / "2005", snapshot=False)
    assert status == "saved" and "ok" in html
    assert d.session.headers["User-Agent"] == "Test/0.1 (test)"     # 写明身份的 User-Agent
    assert waits[0] == 30 and 7 in waits                             # 5xx 按 retry_waits 等；429 按 Retry-After 等
    # 每两次请求之间至少隔 3 秒（等待里有补足间隔的那几次）
    assert all(w >= 3 for w in waits)
    # 文件已经在：不再请求
    status2, _ = d.fetch("2005–06 FA Cup", tmp_path / "FAC" / "2005", snapshot=False)
    assert status2 == "skipped" and len(d.session.urls) == 3
    logged = con.execute("select source, data_type, status, http_status from fetch_log").fetchall()
    assert logged == [("wikipedia", "wiki_page", "saved", 200)]


def test_download_not_found_is_remembered(tmp_path):
    d, con, _ = downloader(tmp_path, [FakeResponse(404)])
    assert d.fetch("2099–00 FA Cup", tmp_path / "FAC" / "2099", snapshot=False) == ("not_found", None)
    assert d.fetch("2099–00 FA Cup", tmp_path / "FAC" / "2099", snapshot=False) == ("skipped", None)
    assert len(d.session.urls) == 1


def test_download_gives_up_after_retries(tmp_path):
    d, con, _ = downloader(tmp_path, [FakeResponse(500), FakeResponse(502), FakeResponse(503)])
    assert d.fetch("X", tmp_path, snapshot=False) == ("failed", None)
    assert con.execute("select status, note from fetch_log").fetchone() == ("failed", "HTTP 503")


def test_retry_after_formats():
    assert C.retry_after("12", 30) == 12
    assert C.retry_after(None, 30) == 30
    assert C.retry_after("not a date", 30) == 30


# ---------------------------------------------------------------- 比分、日期、时间

def test_parse_score_variants():
    s = W.parse_score("1–1 (a.e.t.)")
    assert (s.home, s.away, s.extra_time, s.pen_home) == (1, 1, True, None)
    s = W.parse_score("1–1 (6–5 p)")
    assert (s.home, s.away, s.pen_home, s.pen_away) == (1, 1, 6, 5)
    s = W.parse_score("0–0 3–2 p.")
    assert (s.home, s.away, s.pen_home, s.pen_away) == (0, 0, 3, 2)
    assert W.parse_score("w/o").special == "walkover"
    assert W.parse_score("3–0 (awd)").special == "awarded"
    s = W.parse_score("2–1 (g.g.)")
    assert s.golden_goal and s.extra_time


def test_parse_date_formats():
    assert W.parse_date("19 July 2000") == D(2000, 7, 19)
    assert W.parse_date("July 19, 2000") == D(2000, 7, 19)
    assert W.parse_date("(2000-07-19)") == D(2000, 7, 19)
    assert W.parse_date("19 July") is None            # 没有年份：这里不猜


def test_parse_kickoff_only_with_timezone():
    day = D(2010, 3, 9)
    assert W.parse_kickoff("20:45 CET", day) == ("20:45", "2010-03-09 19:45")
    assert W.parse_kickoff("22:45 UTC+4", day) == ("22:45", "2010-03-09 18:45")
    assert W.parse_kickoff("20:45", day) == ("20:45", None)         # 不知道时区：不换算
    assert W.parse_kickoff("19:45 BST", None) == ("19:45", None)    # 不知道日期：不换算


def test_clean_text_drops_references_and_hidden_text():
    node = soup_of('<p>Played on 9 March<sup class="reference">[1]</sup> 2010'
                   '<span style="display:none">hidden</span></p>').p
    assert W.clean_text(node) == "Played on 9 March 2010"


def test_year_from_season():
    assert W.year_from_season(11, 2001) == 2001
    assert W.year_from_season(2, 2001) == 2002
    assert W.year_from_season(7, 2001) is None        # 6、7 月说不清，不推
    assert W.year_from_season(11, None) is None


@pytest.mark.parametrize("text, season, want", [
    ("between 7 and 9 November 2008", None, D(2008, 11, 7)),
    ("18–19 & 25–26 February 2020", None, D(2020, 2, 18)),
    ("April 14, 2002", None, D(2002, 4, 14)),
    ("16 November and 2 December 2001", None, D(2001, 11, 16)),
    ("the 2nd of December 2001", None, D(2001, 12, 2)),
    ("The 2 matches were played on 9 March 2002", None, D(2002, 3, 9)),   # “2 matches” 的 2 不是日子
    ("played 16, 17 and 18 November", 2001, D(2001, 11, 16)),             # 没写年份：按赛季推
    ("played 16, 17 and 18 November", None, None),                        # 没给赛季：不推
    ("played on 12 July", 2001, None),                                    # 7 月：不推
    ("Round 3 had 64 teams", 2001, None),
])
def test_first_date_in(text, season, want):
    assert W.first_date_in(text, season) == want


def test_prose_round_date_ignores_draw_sentence():
    text = "The draw was held on 26 October 2008. The matches were played on 8 and 9 November 2008."
    assert W.prose_round_date(text) == D(2008, 11, 8)
    assert W.prose_round_date("The draw was held on 26 October 2008.") is None
    assert W.prose_round_date("The draw was made on 2 May and the matches were played on 9 March", 2001) == D(2002, 3, 9)


# ---------------------------------------------------------------- 三种比赛写法

FOOTBALLBOX = """
<div class="footballbox">
 <div class="fleft"><time><div class="fdate"><span class="bday dtstart">2010-03-09</span></div>
   <div class="ftime">20:45 CET</div></time></div>
 <table class="fevent"><tbody>
  <tr><th class="fhome"><span class="flagicon"><img alt="England"></span>
        <a href="/wiki/Arsenal_F.C." title="Arsenal F.C.">Arsenal</a></th>
      <th class="fscore">2–2 (<abbr>a.e.t.</abbr>)</th>
      <th class="faway"><a href="/wiki/FC_Porto" title="FC Porto">Porto</a>
        <span class="flagicon"><img alt="Portugal"></span></th></tr>
  <tr class="fgoals"><td class="fhgoal"><ul><li><span class="fb-goal">Bendtner 10'</span></li>
        <li><span class="fb-goal">Bendtner 105'</span></li></ul></td><td></td>
      <td class="fagoal"><ul><li><span class="fb-goal">Hulk 90+2'</span></li>
        <li><span class="fb-goal">Hulk 119'</span></li></ul></td></tr>
  <tr><th colspan="3">Penalties</th></tr>
  <tr><td>Fàbregas</td><th>4–3</th><td>Hulk</td></tr>
 </tbody></table>
</div>"""


def test_footballbox_extra_time_and_penalties():
    m = W.parse_footballbox(soup_of(FOOTBALLBOX).find("div", class_="footballbox"))
    assert m["home"].title == "Arsenal F.C." and m["home"].country == "England"
    assert m["away"].title == "FC Porto" and m["away"].shown == "Porto"
    assert m["date"] == D(2010, 3, 9) and m["kickoff_utc"] == "2010-03-09 19:45"
    s = m["score"]
    assert (s.home, s.away, s.extra_time, s.pen_home, s.pen_away) == (2, 2, True, 4, 3)
    assert m["ft"] == (1, 1)              # 90 分钟（含补时 90+2'）是 1–1，加时各进一球


def test_footballbox_missing_minutes_gives_no_90_minute_score():
    html = FOOTBALLBOX.replace("Bendtner 105'", "Bendtner ?")
    m = W.parse_footballbox(soup_of(html).find("div", class_="footballbox"))
    assert m["ft"] == (None, None)        # 进球时间不全：不猜


COLLAPSIBLE = """
<table class="tmpl-football-box-collapsible"><tbody>
 <tr><td>4 March 2014</td><td class="vcard"><a href="/wiki/FC_Bayern_Munich" title="FC Bayern Munich">Bayern Munich</a></td>
     <td>1–1 (a.e.t.)</td><td class="vcard"><a href="/wiki/Hamburger_SV" title="Hamburger SV">Hamburger SV</a></td><td>Allianz Arena</td></tr>
 <tr><td>20:30 CET</td><td><span class="fb-goal">Müller 50'</span></td><td>Report</td><td><span class="fb-goal">Lasogga 80'</span></td></tr>
 <tr><th>Penalties</th></tr>
 <tr><td>5–4</td></tr>
</tbody></table>"""


def test_collapsible_box():
    m = W.parse_collapsible(soup_of(COLLAPSIBLE).find("table"))
    assert m["home"].title == "FC Bayern Munich" and m["away"].title == "Hamburger SV"
    assert m["date"] == D(2014, 3, 4) and m["kickoff_utc"] == "2014-03-04 19:30"
    assert (m["score"].home, m["score"].pen_home, m["score"].pen_away) == (1, 5, 4)
    assert m["ft"] == (1, 1)


def table_of(html):
    t = soup_of(html).find("table")
    return t, W.match_table_columns(t)


FA_TABLE = """
<table class="wikitable"><tbody>
 <tr><th>Tie no</th><th>Home team</th><th>Score</th><th>Away team</th><th>Attendance</th></tr>
 <tr><td colspan="5">5 January 2008</td></tr>
 <tr><td>1</td><td><a href="/wiki/Arsenal_F.C." title="Arsenal F.C.">Arsenal</a> (1)</td><td>2–1†</td>
     <td><a href="/wiki/Burnley_F.C." title="Burnley F.C.">Burnley</a> (2)</td><td>50,000</td></tr>
 <tr><td>2</td><td><a href="/wiki/Chelsea_F.C." title="Chelsea F.C.">Chelsea</a></td><td>0–0</td>
     <td><a href="/wiki/Hull_City_A.F.C." title="Hull City A.F.C.">Hull City</a></td><td>40,000</td></tr>
 <tr><td>Replay</td><td><a href="/wiki/Hull_City_A.F.C." title="Hull City A.F.C.">Hull City</a></td><td>1–1†</td>
     <td><a href="/wiki/Chelsea_F.C." title="Chelsea F.C.">Chelsea</a></td><td>20,000</td></tr>
 <tr><td colspan="5">Chelsea won 4–3 on penalties</td></tr>
</tbody></table>"""


def test_fa_cup_table_dagger_replay_and_penalty_note():
    table, cols = table_of(FA_TABLE)
    assert cols["kind"] == "table" and not cols["score_is_90"]
    ms = W.parse_match_table(table, cols, dagger_aet=True)
    assert [m["home"].title for m in ms] == ["Arsenal F.C.", "Chelsea F.C.", "Hull City A.F.C."]
    assert all(m["date"] == D(2008, 1, 5) for m in ms)
    assert ms[0]["home"].shown == "Arsenal"             # “(1)” 联赛级别标注去掉
    assert ms[0]["score"].extra_time and ms[0]["ft"] == (None, None)   # † = 加时后，90 分钟比分不知道
    assert not ms[1]["score"].extra_time and ms[1]["ft"] == (0, 0)
    assert ms[2]["replay_hint"]
    assert (ms[2]["score"].pen_home, ms[2]["score"].pen_away) == (3, 4)   # 客队切尔西赢：点球按主客方向摆


LC_TABLE = """
<table class="wikitable"><tbody>
 <tr><th>Home team</th><th>Score 1</th><th>Away team</th></tr>
 <tr><td colspan="3">16 November</td></tr>
 <tr><td><a title="Arsenal F.C." href="/wiki/Arsenal_F.C.">Arsenal</a></td><td>1–1</td>
     <td><a title="Leeds United F.C." href="/wiki/Leeds_United_F.C.">Leeds United</a></td></tr>
 <tr><td colspan="3">Arsenal won 3–1 after extra time</td></tr>
</tbody></table>"""


def test_league_cup_table_score_1_and_year_from_season():
    table, cols = table_of(LC_TABLE)
    assert cols["score_is_90"]                     # “Score 1”：格子里是 90 分钟比分
    ms = W.parse_match_table(table, cols, dagger_aet=False, season=2013)
    m = ms[0]
    assert m["date"] == D(2013, 11, 16)           # 日期行没写年份：按 2013/14 赛季推成 2013 年
    assert m["ft"] == (1, 1) and (m["score"].home, m["score"].away) == (3, 1) and m["score"].extra_time
    assert W.parse_match_table(table, cols, dagger_aet=False)[0]["date"] is None   # 不给赛季就不推


SERIES = """
<table class="wikitable"><tbody>
 <tr><th>Team 1</th><th>Agg.</th><th>Team 2</th><th>1st leg</th><th>2nd leg</th></tr>
 <tr><td><a title="FC Porto" href="/wiki/FC_Porto">Porto</a></td><td>2–6</td>
     <td><a title="Arsenal F.C." href="/wiki/Arsenal_F.C.">Arsenal</a></td><td>2–1</td><td>0–5</td></tr>
</tbody></table>"""


def test_series_table_swaps_second_leg():
    table, cols = table_of(SERIES)
    assert cols["kind"] == "series"
    legs = W.parse_series_row(W._expand_rows(table)[1], cols)
    assert [(m["home"].title, m["score"].home, m["score"].away, m["leg_hint"]) for m in legs] == \
        [("FC Porto", 2, 1, 1), ("Arsenal F.C.", 5, 0, 2)]


@pytest.mark.parametrize("heads, page, want", [
    (["First qualifying round"], "2010–11 UEFA Champions League qualifying phase and play-off round", "Q1"),
    (["Qualifying round"], "2001–02 UEFA Cup", "Q1"),
    (["Third round", "Matches"], "2010–11 UEFA Champions League qualifying phase and play-off round", "Q3"),
    (["Round of 16 / Summary"], "2009–10 UEFA Champions League knockout phase", "R16"),
    (["Group A"], "2009–10 UEFA Champions League group stage", "G"),
    (["Preliminary round", "Semi-finals"], "2010–11 UEFA Champions League qualifying phase and play-off round", "PR"),
    (["Semi-finals"], "2009–10 FA Cup", "SF"),
    (["Third round proper", "Replays"], "2009–10 FA Cup", "R3"),
    (["Final"], "2009–10 FA Cup", "F"),
    (["Notes"], "2009–10 FA Cup", None),
])
def test_classify_round(heads, page, want):
    main = page.split(" qualifying")[0].split(" knockout")[0].split(" group")[0]
    assert W.classify_round(heads, page, main) == want


# ---------------------------------------------------------------- 球队对照、去重、分回合

def test_resolve_checks_country():
    tm = team_map()
    assert tm.resolve(ref("Arsenal F.C.", country="England"), "欧洲") == "Arsenal"
    assert tm.resolve(ref("Arsenal", country="Russia"), "欧洲") == "wiki:Arsenal"     # 图拉兵工厂，不是阿森纳
    assert tm.resolve(ref("Arsenal F.C."), "英格兰") == "Arsenal"
    assert tm.resolve(ref("Arsenal F.C."), "西班牙") == "wiki:Arsenal F.C."            # 西班牙杯赛里的“Arsenal”不是阿森纳
    assert tm.resolve(ref("Real Madrid Castilla"), "西班牙") == "wiki:Real Madrid Castilla"   # 皇马二队 ≠ 皇马
    assert not tm.is_ours("wiki:Real Madrid Castilla")


def test_real_alias_file_loads(settings):
    tm = W.load_team_map(settings)
    assert tm.alias["Real Madrid CF"] == "Real Madrid"
    assert "Real Madrid Castilla" not in tm.alias
    assert all(t in tm.teams for t in tm.alias.values())


def raw(home, away, rnd="R16", day=None, fmt="footballbox", score=(1, 0), leg_hint=0, replay=False,
        precision="day", order=0, season=2009, comp="UCL", hc="", ac=""):
    s = W.Score(home=score[0], away=score[1]) if score else W.Score()
    return W.RawMatch(competition=comp, season=season, page="p", raw_file="data/raw/wikipedia/x.html.gz",
                      headings=[], home=ref(home, country=hc), away=ref(away, country=ac), date=day,
                      date_precision=precision if day else "round", kickoff_local=None, kickoff_utc=None,
                      score=s, ft_home=None, ft_away=None, fmt=fmt, leg_hint=leg_hint, replay_hint=replay,
                      order=order, round=rnd)


def ids_of(ms, tm, country="欧洲"):
    return {id(m): (tm.resolve(m.home, country), tm.resolve(m.away, country)) for m in ms}


def test_unify_foreign_ids_merges_redirect_names_only():
    """汇总表写 “FC Porto”，比赛框写 “Porto”：同一轮、对手一样 → 并成一支；名字不相干的不并。"""
    tm = team_map()
    ms = [raw("FC Porto", "Arsenal F.C.", day=D(2010, 2, 17)), raw("Arsenal F.C.", "Porto", day=D(2010, 3, 9)),
          raw("FC Porto", "Arsenal F.C.", fmt="series", leg_hint=1, day=D(2010, 2, 16), precision="round"),
          raw("Chelsea F.C.", "Inter Milan", day=D(2010, 2, 24)), raw("Chelsea F.C.", "Milan", day=D(2010, 3, 1))]
    ids = W.unify_foreign_ids(ms, ids_of(ms, tm), tm, problems := [])
    assert ids[id(ms[1])] == ("Arsenal", "wiki:FC Porto")
    # “Milan” 和 “Inter Milan” 名字一个包含另一个，但都在切尔西主场、不同日期各踢了一场 → 是两支队，不并
    assert ids[id(ms[4])] == ("Chelsea", "wiki:Milan")
    assert ids[id(ms[3])] == ("Chelsea", "wiki:Inter Milan")
    assert len(problems) == 1


def test_unify_foreign_ids_respects_flags():
    tm = team_map()
    ms = [raw("Arsenal F.C.", "Dinamo", ac="Romania", rnd="G", day=D(2009, 9, 1)),
          raw("Arsenal F.C.", "Dinamo Zagreb", ac="Croatia", rnd="G", day=D(2009, 10, 1))]
    ids = W.unify_foreign_ids(ms, ids_of(ms, tm), tm, [])
    assert ids[id(ms[0])][1] != ids[id(ms[1])][1]                    # 国旗不一样：不并


def test_merge_duplicates_keeps_footballbox_over_summary_table():
    tm = team_map()
    ms = [raw("FC Porto", "Arsenal F.C.", fmt="series", leg_hint=1, score=(2, 1), day=D(2010, 2, 16), precision="round"),
          raw("FC Porto", "Arsenal F.C.", score=(2, 1), day=D(2010, 2, 17)),
          raw("Arsenal F.C.", "FC Porto", fmt="series", leg_hint=2, score=(5, 0), day=D(2010, 2, 16), precision="round"),
          raw("Arsenal F.C.", "FC Porto", score=(5, 0), day=D(2010, 3, 9))]
    ids = ids_of(ms, tm)
    kept = W.merge_duplicates(ms, ids, problems := [])
    assert sorted(m.fmt for m in kept) == ["footballbox", "footballbox"] and not problems
    legs = {(m.date, leg) for m, _rnd, leg in W.assign_legs(kept, ids, False, [])}
    assert legs == {(D(2010, 2, 17), 1), (D(2010, 3, 9), 2)}      # 按日期先后分第 1、2 回合


def test_merge_duplicates_reports_score_conflict():
    tm = team_map()
    ms = [raw("Arsenal F.C.", "Chelsea F.C.", rnd="F", score=(1, 0), day=D(2017, 5, 27)),
          raw("Arsenal F.C.", "Chelsea F.C.", rnd="F", fmt="collapsible", score=(2, 0), day=D(2017, 5, 27))]
    kept = W.merge_duplicates(ms, ids_of(ms, tm, "英格兰"), problems := [])
    assert len(kept) == 1 and kept[0].score.home == 1 and "比分不一样" in problems[0]


def test_assign_legs_replays_and_groups():
    tm = team_map()
    fa = [raw("Hull City", "Chelsea F.C.", rnd="R3", day=D(2008, 1, 15), comp="FAC"),
          raw("Chelsea F.C.", "Hull City", rnd="R3", day=D(2008, 1, 5), comp="FAC")]
    out = W.assign_legs(fa, ids_of(fa, tm, "英格兰"), True, [])
    assert sorted((m.date, rnd, leg) for m, rnd, leg in out) == \
        [(D(2008, 1, 5), "R3", 0), (D(2008, 1, 15), "R3_replay", 0)]
    grp = [raw("Arsenal F.C.", "FC Porto", rnd="G", day=D(2009, 9, 1)),
           raw("FC Porto", "Arsenal F.C.", rnd="G", day=D(2009, 11, 1))]
    assert {leg for _m, _r, leg in W.assign_legs(grp, ids_of(grp, tm), False, [])} == {0}


def test_regular_time_rules():
    def m(score, ft=(None, None), **kw):
        x = raw("A", "B", score=None)
        x.score = W.Score(home=score[0], away=score[1], extra_time=True, **kw)
        x.ft_home, x.ft_away = ft
        return x
    assert W.regular_time(m((2, 1), ft=(1, 1)), 0)[:3] == (1, 1, "D")          # 有进球时间
    assert W.regular_time(m((2, 1), golden_goal=True), 0)[:3] == (1, 1, "D")   # 金球：赢的一方减一球
    assert W.regular_time(m((0, 0)), 0)[:3] == (0, 0, "D")                     # 加时没进球
    assert W.regular_time(m((2, 2)), 0)[:3] == (None, None, "D")               # 加时后 2-2：90 分钟可能是 1-1，不猜
    assert W.regular_time(m((2, 1)), 0)[:3] == (None, None, "D")               # 单场加时：一定是平局，比分不知道
    assert W.regular_time(m((2, 1)), 2)[:3] == (None, None, None)              # 第二回合加时：不推算


def test_build_rows_skips_dateless_replays_and_foreign_only(settings):
    tm = team_map()
    comps = {"FAC": {"国家": "英格兰", "重赛": True}}
    ms = [raw("Chelsea F.C.", "Hull City", rnd="R3", day=D(2008, 1, 5), score=(0, 0), comp="FAC", season=2007),
          raw("Hull City", "Chelsea F.C.", rnd="R3", day=D(2008, 1, 5), precision="round", comp="FAC", season=2007,
              replay=True),
          raw("Hull City", "Burnley", rnd="R3", day=D(2008, 1, 5), comp="FAC", season=2007),
          raw("Chelsea F.C.", "Leeds", rnd="R4", day=None, comp="FAC", season=2007)]
    stats, problems = Counter(), []
    rows = W.build_rows(ms, tm, comps, D(2026, 10, 30), problems, stats)
    assert [(r.home_team, r.away_team, r.round) for r in rows] == [("Chelsea", "wiki:Hull City", "R3")]
    text = " ".join(problems)
    assert "重赛没有具体日期" in text and "找不到日期" in text      # 不拿这一轮的日期冒充重赛日期，也不编日期


# ---------------------------------------------------------------- 数据库：升级、写入、两个来源互不影响

def box(day, home, home_title, away, away_title, score):
    return (f'<div class="footballbox"><div class="fdate"><span class="bday">{day}</span></div>'
            f'<table><tr><th class="fhome"><a href="/wiki/x" title="{home_title}">{home}</a></th>'
            f'<th class="fscore">{score}</th>'
            f'<th class="faway"><a href="/wiki/y" title="{away_title}">{away}</a></th></tr></table></div>')


KNOCKOUT_PAGE = f"""<html><body><div class="mw-parser-output">
<h2>Round of 16</h2>
<h3>Summary</h3>
<p>The first legs were played on 16, 17, 23 and 24 February, and the second legs on 9, 10, 16 and 17 March 2010.</p>
{SERIES}
<h3>Matches</h3>
{box("2010-02-17", "Porto", "FC Porto", "Arsenal", "Arsenal F.C.", "2–1")}
{box("2010-03-09", "Arsenal", "Arsenal F.C.", "Porto", "Porto", "5–0")}
<div class="navbox">{box("2010-01-01", "Chelsea", "Chelsea F.C.", "Arsenal", "Arsenal F.C.", "9–9")}</div>
</div></body></html>"""


def test_parse_page_and_build_rows_end_to_end():
    """汇总表 + 两个比赛框（第二回合的客队链接是重定向短名）→ 两行，回合、日期、比分都对；导航框里的不算。"""
    res = W.parse_page(KNOCKOUT_PAGE, "UCL", 2009, "2009–10_UEFA_Champions_League_knockout_phase", "raw/x.html.gz",
                       "2009–10 UEFA Champions League")
    assert len(res.matches) == 4 and {m.round for m in res.matches} == {"R16"}
    series = [m for m in res.matches if m.fmt == "series"]
    assert all(m.date == D(2010, 2, 16) and m.date_precision == "round" for m in series)   # 说明文字里这一轮的第一天
    rows = W.build_rows(res.matches, team_map(), {"UCL": {"国家": "欧洲"}}, D(2026, 10, 30), [], Counter())
    got = sorted((r.match_date, r.leg, r.home_team, r.away_team, r.home_goals, r.away_goals, r.date_precision)
                 for r in rows)
    assert got == [("2010-02-17", 1, "wiki:FC Porto", "Arsenal", 2, 1, "day"),
                   ("2010-03-09", 2, "Arsenal", "wiki:FC Porto", 5, 0, "day")]


OLD_MATCHES = """create table matches (
    match_id integer primary key, competition text not null, season integer not null,
    home_team text not null, away_team text not null, match_date text not null, kickoff_utc text,
    status text not null, unique (competition, season, home_team, away_team))"""


def test_migration_keeps_league_ids_and_is_idempotent():
    con = sqlite3.connect(":memory:")
    con.execute(OLD_MATCHES)
    con.executemany("insert into matches values (?, 'E0', 2024, ?, ?, '2024-08-17', null, 'finished')",
                    [(7, "Arsenal", "Chelsea"), (42, "Chelsea", "Arsenal")])
    create_tables(con)                                   # 自动升级
    rows = con.execute("select match_id, home_team, round, leg, date_precision from matches order by 1").fetchall()
    assert rows == [(7, "Arsenal", "", 0, "day"), (42, "Chelsea", "", 0, "day")]
    assert migrate_matches(con) is False                 # 已经是新版：什么都不做
    # 新的唯一键：同一对球队，杯赛不同轮次可以各有一场
    con.execute("insert into matches (competition, season, home_team, away_team, match_date, status, round, leg) "
                "values ('FAC', 2024, 'Arsenal', 'Chelsea', '2025-01-05', 'finished', 'R3', 0)")
    con.execute("insert into matches (competition, season, home_team, away_team, match_date, status, round, leg) "
                "values ('FAC', 2024, 'Arsenal', 'Chelsea', '2025-03-01', 'finished', 'QF', 0)")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("insert into matches (competition, season, home_team, away_team, match_date, status, round, leg) "
                    "values ('FAC', 2024, 'Arsenal', 'Chelsea', '2025-03-02', 'finished', 'QF', 0)")


def wiki_row(comp, season, rnd, leg, home, away, day, hg=1, ag=0, **kw):
    base = dict(competition=comp, season=season, round=rnd, leg=leg, home_team=home, away_team=away,
                match_date=day, date_precision="day", kickoff_utc=None, status="finished", home_goals=hg,
                away_goals=ag, result="H" if hg > ag else "A" if hg < ag else "D", extra_time=0, aet_home=None,
                aet_away=None, pen_home=None, pen_away=None, note="", raw_file="data/raw/wikipedia/x.html.gz",
                page="x", home_title=home.replace("wiki:", ""), away_title=away.replace("wiki:", ""),
                home_country="", away_country="")
    base.update(kw)
    return W.Row(**base)


def football_data_import(con, settings):
    """用 football-data 导入程序导一场英超（阿森纳 2–1 切尔西），和真实流程一样。"""
    import pandas as pd
    from src.clean.football_data_import import odds_slots, parse_file, write_database
    from src.common.config import load_football_data_columns
    slots = odds_slots(load_football_data_columns(settings))
    t = pd.DataFrame([{"Div": "E0", "Date": "16/08/2025", "Time": "15:00", "HomeTeam": "Arsenal",
                       "AwayTeam": "Chelsea", "FTHG": "2", "FTAG": "1", "FTR": "H", "B365H": "2.0",
                       "B365D": "3.5", "B365A": "4.0"}]).assign(row=[0])
    names = {"Arsenal": "Arsenal", "Chelsea": "Chelsea"}
    m, s, o, _ = parse_file(t, "E0", 2025, names, slots, settings, [], Counter())
    m = m.assign(raw_file="data/raw/test.csv", fetched_at_utc="2026-10-30T00:00:00+00:00")
    write_database(con, [{"英文": n, "国家": "英格兰"} for n in names], names, m, s, o)


COMPS = {"UCL": {"国家": "欧洲"}, "FAC": {"国家": "英格兰", "重赛": True}}


def counts(con):
    return dict(con.execute("select source, count(*) from match_sources group by source").fetchall())


def test_two_sources_never_delete_each_other(settings):
    con = sqlite3.connect(":memory:")
    football_data_import(con, settings)
    league = con.execute("select match_id from matches").fetchall()
    odds = con.execute("select count(*) from odds").fetchone()[0]
    rows = [wiki_row("FAC", 2025, "R3", 0, "Arsenal", "Chelsea", "2026-01-10", 2, 2, extra_time=1, aet_home=3,
                     aet_away=2),
            wiki_row("UCL", 2025, "R16", 1, "wiki:FC Porto", "Arsenal", "2026-02-17", 2, 1),
            wiki_row("UCL", 2025, "R16", 2, "Arsenal", "wiki:FC Porto", "2026-03-10", 5, 0)]
    tm = team_map()
    W.write_database(con, rows, tm, COMPS, {})
    # 同一对球队、同一赛季：联赛一场 + 足总杯一场，是两场不同的比赛
    assert con.execute("select count(*) from matches where home_team = 'Arsenal' and away_team = 'Chelsea'").fetchone()[0] == 2
    assert con.execute("select match_id from matches where round = ''").fetchall() == league   # 联赛编号不变
    assert counts(con) == {"football-data": 1, "wikipedia": 3}
    assert con.execute("select extra_time, aet_home, aet_away from cup_details c join matches m using (match_id) "
                       "where m.round = 'R3'").fetchone() == (1, 3, 2)
    assert con.execute("select country from teams where team_id = 'wiki:FC Porto'").fetchone() == ("未知",)
    wiki_ids = con.execute("select match_id from matches where round != '' order by 1").fetchall()

    # 再导一次 football-data：维基百科的行一行都不少
    football_data_import(con, settings)
    assert counts(con) == {"football-data": 1, "wikipedia": 3}
    assert con.execute("select count(*) from odds").fetchone()[0] == odds
    # 再导一次维基百科：编号不变，football-data 的行和赔率不动
    W.write_database(con, rows, tm, COMPS, {})
    assert con.execute("select match_id from matches where round != '' order by 1").fetchall() == wiki_ids
    assert con.execute("select match_id from matches where round = ''").fetchall() == league
    assert con.execute("select count(*) from odds").fetchone()[0] == odds

    # 只导一部分赛事（--only FAC）：欧冠的行不动；足总杯里不再出现的比赛删掉
    W.write_database(con, [], tm, COMPS, {}, codes=["FAC"])
    assert counts(con) == {"football-data": 1, "wikipedia": 2}
    assert con.execute("select count(*) from matches where competition = 'UCL'").fetchone()[0] == 2
    assert con.execute("select match_id from matches where round = ''").fetchall() == league


# ---------------------------------------------------------------- 质量检查：杯赛

def cup_db(rows):
    """rows：(编号, 赛事, 轮次, 回合, 日期, 主队, 客队, 主队进球, 客队进球, 加时, 加时主, 加时客, 点球主, 点球客)"""
    con = sqlite3.connect(":memory:")
    create_tables(con)
    for mid, comp, rnd, leg, day, h, a, hg, ag, et, ah, aa, ph, pa in rows:
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, status, "
                    "round, leg) values (?, ?, 2009, ?, ?, ?, 'finished', ?, ?)", (mid, comp, h, a, day, rnd, leg))
        con.execute("insert into match_stats (match_id, source, home_goals, away_goals) values (?, 'wikipedia', ?, ?)",
                    (mid, hg, ag))
        con.execute("insert into cup_details values (?, 'wikipedia', ?, ?, ?, ?, ?, null)", (mid, et, ah, aa, ph, pa))
    return con


def test_cup_checks_find_problems():
    con = cup_db([
        # 波尔图 2–1 阿森纳、阿森纳 2–0 波尔图：总比分 2–3（按第一回合主队在前）不是平局，却记了点球 → 要报
        (1, "UCL", "R16", 1, "2010-02-17", "P", "A", 2, 1, 0, None, None, None, None),
        (2, "UCL", "R16", 2, "2010-03-09", "A", "P", 2, 0, 0, None, None, 4, 3),
        # 波尔图被淘汰了，后面又在八强出场 → 要报
        (3, "UCL", "QF", 1, "2010-03-30", "P", "X", 0, 0, 0, None, None, None, None),
        (4, "UCL", "QF", 2, "2010-04-06", "X", "P", 1, 0, 0, None, None, None, None),
        # 只有第 1 回合 → 要报
        (5, "UCL", "R16", 1, "2010-02-24", "I", "C", 2, 1, 0, None, None, None, None),
        # 单场平局，没有点球也没有重赛 → 要报
        (6, "FAC", "R3", 0, "2010-01-02", "A", "B", 1, 1, 0, None, None, None, None),
        # 单场加时 2–1：正常
        (7, "FAC", "R4", 0, "2010-01-23", "A", "D", 1, 1, 1, 2, 1, None, None),
        # 联赛不查
        (8, "E0", "", 0, "2010-01-02", "A", "B", 1, 1, 0, None, None, None, None),
    ])
    found = {name: rows for name, _desc, rows in cup_checks(con, ["E0"])}
    assert len(found["杯赛点球和比分对不上"]) == 1 and "总比分 2–3" in found["杯赛点球和比分对不上"][0]
    assert len(found["杯赛被淘汰的队后面又出场"]) == 1 and "P 被淘汰了" in found["杯赛被淘汰的队后面又出场"][0]
    assert len(found["杯赛两回合缺一回合"]) == 1 and "[1]" in found["杯赛两回合缺一回合"][0]
    assert len(found["杯赛看不出谁晋级"]) == 1 and "FAC" in found["杯赛看不出谁晋级"][0]


def test_quality_league_structure_ignores_cups(settings):
    """“同一对主客队出现两次”这类联赛结构检查只看联赛；杯赛里同一对主客队碰两次是正常的。
    同一天踢两场的检查联赛和杯赛一起看。"""
    from src.clean.quality import run_checks
    con = cup_db([
        (1, "FAC", "R3", 0, "2010-01-02", "Arsenal", "Chelsea", 2, 1, 0, None, None, None, None),
        (2, "FAC", "SF", 0, "2010-04-10", "Arsenal", "Chelsea", 2, 1, 0, None, None, None, None),
    ])
    con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, status) "
                "values (9, 'E0', 2009, 'Arsenal', 'Burnley', '2010-01-02', 'finished')")
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result) "
                "values (9, 'football-data', 1, 0, 'H')")
    found = {c["名称"]: c["rows"] for c in run_checks(con, settings, today=D(2026, 10, 30))}
    for name in ("赛季场数不对", "主客场数不一样", "日期不在赛季范围内", "开球时间缺失"):
        assert not any("FAC" in r for r in found[name]), name
    assert found["同一天同一队踢两场"] == ["2010-01-02 Arsenal：2 场"]   # 阿森纳 1 月 2 日联赛一场、足总杯一场


# ---------------------------------------------------------------- 交叉核对

def test_openfootball_score_formats():
    assert X.parse_openfootball_score("1-2 (1-2)")["ft"] == (1, 2)
    s = X.parse_openfootball_score("2-1 a.e.t. (1-1, 0-0)")
    assert (s["aet"], s["ft"], s["pens"]) == ((2, 1), (1, 1), None)
    s = X.parse_openfootball_score("4-5 pen. 0-0 a.e.t. (0-0)")
    assert (s["pens"], s["aet"], s["ft"], s["et_known"]) == ((4, 5), (0, 0), (0, 0), True)
    s = X.parse_openfootball_score("3-4 pen. (1-1)")
    assert (s["pens"], s["ft"], s["et_known"]) == ((3, 4), (1, 1), False)   # 有没有加时说不清
    assert X.parse_openfootball_score("[cancelled]") is None


OF_EUROPE = """= UEFA Champions League 2018/19

# Teams      32

▪ Group A
  Tue Sep 18 2018
    21:00  AS Monaco (FRA)         v Atlético Madrid (ESP)    1-2 (1-2)
  Wed Oct 3
    21:00  Borussia Dortmund (GER) v AS Monaco (FRA)          3-0 (0-0)  @ Signal Iduna Park
"""

OF_DOMESTIC = """= Copa del Rey 2020/21

▪ Round 1
Wed Nov 11
  18:00  AUGC Deportiva Ceuta     0-3  Racing Murcia FC
         Móstoles CF              4-5 pen. 0-0 a.e.t. (0-0)  CD Marchamalo
         CD Cardassar             -  AD CF Épila
"""


def test_read_openfootball_both_layouts(tmp_path):
    p = tmp_path / "UCL" / "2018" / "cl.txt"
    p.parent.mkdir(parents=True)
    p.write_text(OF_EUROPE, encoding="utf-8")
    ms = X.read_openfootball_file(p, "UCL", 2018)
    assert [(m.date, m.home, m.away, m.home_country, m.ft) for m in ms] == [
        (D(2018, 9, 18), "AS Monaco", "Atlético Madrid", "法国", (1, 2)),
        (D(2018, 10, 3), "Borussia Dortmund", "AS Monaco", "德国", (3, 0))]     # 没写年份：靠星期几推出 2018
    assert ms[0].round_text == "Group A"
    q = tmp_path / "CDR" / "2020" / "cup.txt"
    q.parent.mkdir(parents=True)
    q.write_text(OF_DOMESTIC, encoding="utf-8")
    ms = X.read_openfootball_file(q, "CDR", 2020)
    assert [(m.date, m.home, m.away, m.pens) for m in ms] == [
        (D(2020, 11, 11), "AUGC Deportiva Ceuta", "Racing Murcia FC", None),
        (D(2020, 11, 11), "Móstoles CF", "CD Marchamalo", (4, 5))]            # 没比分的那行不要


def test_read_engsoccerdata_columns(tmp_path):
    (tmp_path / "champs.csv").write_text(
        '"Date","Season","round","leg","home","visitor","FT","HT","aet","pens","hcountry","vcountry"\n'
        '2010-03-09,2009,"R16","2","Arsenal","FC Porto","5-0","2-0",NA,NA,"ENG","POR"\n'
        '2010-03-10,2009,"R16","2","Real Madrid","Lyon","1-1","0-0",NA,"away goals","ESP","FRA"\n', encoding="utf-8")
    (tmp_path / "facup.csv").write_text(
        '"Date","Season","home","visitor","FT","round","tie","aet","pen","pens","hp","vp","nonmatch"\n'
        '2008-01-15,2007,"Hull City","Chelsea","1-1","3","replay","yes","yes",NA,3,4,NA\n'
        '1999-01-15,1998,"Old","Match","1-1","3","initial",NA,NA,NA,NA,NA,NA\n', encoding="utf-8")
    ms = X.read_engsoccerdata(tmp_path, {"UCL": "champs.csv", "FAC": "facup.csv"}, 2000)
    assert len(ms) == 3                                                    # 2000 年以前的不要
    assert (ms[0].ft, ms[0].home_country, ms[0].away_country) == ((5, 0), "英格兰", "POR")   # 只换我们五国，别的代码原样
    assert ms[1].pens is None                                             # “away goals” 不是点球
    assert (ms[2].ft, ms[2].aet, ms[2].pens) == (None, (1, 1), (3, 4))    # aet=yes：FT 是加时后比分


def test_name_key_and_similarity():
    assert X.name_key("FC Bayern München") == X.name_key("Bayern Munich") == ("bayern", "munich")
    assert X.name_key("Manchester Utd") == X.name_key("Manchester United FC")
    assert X.name_similarity(("dortmund",), ("borussia", "dortmund")) == pytest.approx(0.85)
    assert X.name_similarity(("arsenal",), ("chelsea",)) < X.MIN_SIMILARITY


class FakeNames:
    """交叉核对用的假队名表：只认名字完全一样（按 name_key）。"""
    def similarity(self, ref_name, db_team, db_title):
        return X.name_similarity(X.name_key(ref_name), X.name_key(db_title))

    def ours(self, name, country, comp_country):
        return "Arsenal" if X.name_key(name) == ("arsenal",) else None


def db(mid, day, home, away, ft, precision="day", aet=None, pens=None, rnd="R16", leg=1):
    return X.DbMatch(mid, "UCL", 2009, day, precision, rnd, leg, home, away, "finished", ft, aet, pens, aet is not None)


def test_pair_up_and_compare():
    refs = [X.RefMatch("t", "UCL", 2009, D(2010, 3, 9), "Arsenal", "FC Porto", ft=(5, 0)),
            X.RefMatch("t", "UCL", 2009, D(2010, 2, 17), "Arsenal", "FC Porto", ft=(1, 2)),   # 主客写反了
            X.RefMatch("t", "UCL", 2009, D(2010, 4, 1), "Arsenal", "Barcelona", ft=(2, 2))]
    dbs = [db(1, D(2010, 3, 9), "Arsenal", "wiki:FC Porto", (5, 0), leg=2),
           db(2, D(2010, 2, 17), "wiki:FC Porto", "Arsenal", (2, 1)),
           db(3, D(2010, 4, 6), "wiki:Barcelona", "Arsenal", (4, 1), rnd="QF", leg=2)]
    pairs = X.pair_up(refs, dbs, FakeNames())
    got = {(r.date, d.match_id, sw) for r, d, sw in pairs}
    assert got == {(D(2010, 3, 9), 1, False), (D(2010, 2, 17), 2, True)}   # 第三场日期差 5 天，精确到天的不配
    assert X.compare(refs[0], dbs[0], False) == []
    assert X.compare(refs[1], dbs[1], True) == ["主客队反了"]               # 换过方向以后比分一致
    r = X.RefMatch("t", "UCL", 2009, D(2010, 3, 10), "Arsenal", "FC Porto", ft=(1, 1), aet=(2, 1))
    d = db(4, D(2010, 3, 9), "Arsenal", "wiki:FC Porto", (1, 1), aet=(2, 2))
    assert X.compare(r, d, False) == ["日期不一样（数据库 2010-03-09，参照 2010-03-10）",
                                      "加时比分不一样（数据库 2-2，参照 2-1）"]
    # 英联赛杯：参照只有一个比分、说不清含不含加时 → 和 90 分钟或加时后比分任何一个一样都算对
    lc = X.RefMatch("t", "ELC", 2009, D(2010, 3, 9), "Arsenal", "FC Porto", final=(2, 2), et_known=False)
    assert X.compare(lc, d, False) == []
    unknown_90 = db(5, D(2010, 3, 9), "Arsenal", "wiki:FC Porto", None, aet=(4, 3))
    for final, ok in [((3, 3), True), ((4, 3), True), ((4, 4), False)]:   # 3-3 可能是 90 分钟比分；4-4 不可能
        lc = X.RefMatch("t", "ELC", 2009, D(2010, 3, 9), "Arsenal", "FC Porto", final=final, et_known=False)
        assert (X.compare(lc, unknown_90, False) == []) == ok


def test_check_source_counts_missing_and_extra():
    refs = [X.RefMatch("t", "UCL", 2009, D(2010, 3, 9), "Arsenal", "FC Porto", ft=(5, 0)),
            X.RefMatch("t", "UCL", 2009, D(2010, 3, 16), "Arsenal", "Inter", ft=(0, 1))]       # 数据库没有
    dbs = [db(1, D(2010, 3, 9), "Arsenal", "wiki:FC Porto", (5, 0), leg=2),
           db(2, D(2010, 3, 20), "Arsenal", "wiki:Lyon", (1, 0), leg=1),                      # 参照没有（同一轮）
           db(3, D(2010, 9, 1), "Arsenal", "wiki:Braga", (6, 0), rnd="G", leg=0)]            # 参照没覆盖小组赛：不算
    out = X.check_source(refs, dbs, FakeNames())["UCL"]
    assert (out["paired"], out["consistent"], len(out["missing"]), len(out["extra"])) == (1, 1, 1, 1)
    assert out["extra"][0].match_id == 2


def test_footballbox_year_missing_uses_season():
    """看得见的日期没写年份时，time 标签里的年份是模板自己补的（生成网页那年），不能信，按赛季推。"""
    html = ('<div class="footballbox"><div class="fleft"><time datetime="2026-02-28T00:00:00+00:00">'
            '<div class="fdate">28 February</div></time></div><table><tr>'
            '<th class="fhome"><a href="/wiki/x" title="Getafe CF">Getafe</a></th><th class="fscore">3–1</th>'
            '<th class="faway"><a href="/wiki/y" title="Racing de Santander">Racing</a></th></tr></table></div>')
    b = soup_of(html).find("div", class_="footballbox")
    assert W.parse_footballbox(b, 2007)["date"] == D(2008, 2, 28)
    assert W.parse_footballbox(b)["date"] is None                       # 不知道赛季：不猜
    full = html.replace(">28 February<", ">28 February 2008<").replace("2026-02-28", "2008-02-28")
    assert W.parse_footballbox(soup_of(full).find("div", class_="footballbox"), 2007)["date"] == D(2008, 2, 28)


def test_pair_up_second_pass_by_score():
    """另一支队写法差太远（CSKA Moskva / PFC CSKA Moscow）：同一天、一支队像、比分一样 → 还是配上；比分不一样就不配。"""
    refs = [X.RefMatch("t", "UCL", 2011, D(2011, 12, 7), "Arsenal", "Zzzz Moskva", ft=(1, 2)),
            X.RefMatch("t", "UCL", 2011, D(2011, 12, 8), "Arsenal", "Yyyy Kyiv", ft=(0, 0))]
    dbs = [db(1, D(2011, 12, 7), "Arsenal", "wiki:PFC CSKA Moscow", (1, 2), rnd="G", leg=0),
           db(2, D(2011, 12, 8), "Arsenal", "wiki:FC Dynamo Kyiv", (3, 0), rnd="G", leg=0)]
    pairs = X.pair_up(refs, dbs, FakeNames())
    assert [(r.date, d.match_id) for r, d, _ in pairs] == [(D(2011, 12, 7), 1)]


def test_group_inside_numbered_round_is_that_round():
    """法国杯第七轮按地区分组抽签（Seventh round > Main draw > Group F）：是 R7，不是小组赛；欧冠的 Group A 还是小组赛。"""
    assert W.classify_round(["Seventh round", "Main draw", "Group F"], "2021–22_Coupe_de_France",
                            "2021–22_Coupe_de_France") == "R7"
    assert W.classify_round(["Round 7", "Mainland ties", "Group 7A"], "2024–25_Coupe_de_France",
                            "2024–25_Coupe_de_France") == "R7"
    assert W.classify_round(["Group stage", "Group A"], "2021–22_UEFA_Champions_League_group_stage",
                            "2021–22_UEFA_Champions_League") == "G"
