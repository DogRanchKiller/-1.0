"""懂球帝读取和采集程序的自动检查。用假网站代替真网站，不联网。"""
import gzip
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytest

from src.clean.dongqiudi_reader import (asia_line, count_game_rows, detail_columns, next_start, ou_line, parse_fixtures,
                                        parse_history, parse_match_detail, parse_odds, parse_season_page, resolve_ts)
from src.collect.dongqiudi import (DETAIL_URL, FIXTURES_URL, HISTORY_URL, ODDS_URL, SEASON_URL, Blocked, Collector,
                                   check_detail, check_odds, snapshot_gap, spread)
from src.common.db import connect

UTC = timezone.utc

SEASON_HTML = (
    '<div><h2><a href="/zhanbao/lunci/21198-1">第1轮</a></h2><ul class="zbl-games">'
    '<li><span class="d">08-12</span><a href="/zhanbao/101">伯恩利 <b>0-3</b> 曼城<!----></a></li>'
    '<li><span class="d">08-13</span><a href="/zhanbao/102">切尔西 <b>1-1</b> 利物浦<!----></a></li></ul></div>'
    '<div><h2><a href="/zhanbao/lunci/21198-2">第2轮</a></h2><ul class="zbl-games">'
    '<li><span class="d">01-02</span><a href="/zhanbao/103">曼城 <b>10-0</b> 切尔西<!----><!----></a></li></ul></div>'
)
HISTORY_HTML = ('<tr><td><a href="/zhanbao/saiji/24646">2025/26</a></td></tr>'
                '<tr><td><a href="/zhanbao/saiji/21198">2023/24</a></td></tr>'
                '<a href="/zhanbao/qiudui/50000513">阿森纳</a>')


def odds_json(match_id, home="1.38"):
    return {"matchId": match_id,
            "euro": [{"name": "Bet3**", "cid": "281", "begin": {"homeWin": "1.45", "draw": "5.00", "awayWin": "6.00", "ts": "05-13 21:34"},
                      "now": {"homeWin": home, "draw": "5.25", "awayWin": "7.00", "ts": "05-19 22:49"}}],
            "asia": [{"name": "平*", "cid": "47", "begin": {"homeWin": "0.00", "awayWin": "0.00", "draw": "受半球", "draw_value": "-0.50", "ts": ""},
                      "now": {"homeWin": "1.02", "awayWin": "0.91", "draw": "一球半", "draw_value": "1.50", "ts": "05-19 22:55"}}],
            "size": [{"name": "澳*", "cid": "1", "begin": {"homeWin": "0.79", "awayWin": "1.01", "draw": "3.5/4", "ts": "05-13 21:34"},
                      "now": {"homeWin": "0.96", "awayWin": "0.84", "draw": "2.5", "ts": "05-19 22:36"}}],
            "has_odds": True}


# ---------- 读取 ----------
def test_parse_season_page():
    games = parse_season_page(SEASON_HTML)
    assert count_game_rows(SEASON_HTML) == 3 and len(games) == 3
    g = games[0]
    assert (g.match_id, g.round, g.month, g.day, g.home, g.away, g.home_goals, g.away_goals) == \
        (101, 1, 8, 12, "伯恩利", "曼城", 0, 3)
    assert games[2].round == 2 and (games[2].home_goals, games[2].away_goals) == (10, 0) and games[2].away == "切尔西"


def test_parse_history_only_finished_seasons():
    assert parse_history(HISTORY_HTML) == {2025: 24646, 2023: 21198}


def test_odds_conversion():
    rows = {(r.market, r.timing): r for r in parse_odds(odds_json(53615071))}
    assert (rows["1x2", "now"].home, rows["1x2", "now"].draw, rows["1x2", "now"].away) == (1.38, 5.25, 7.0)
    ah = rows["ah", "now"]          # 懂球帝“主队让一球半”(+1.5) → 我们的写法 -1.5；香港盘 1.02 → 小数 2.02
    assert (ah.line, ah.home, ah.away) == (-1.5, 2.02, 1.91)
    assert ("ah", "begin") not in rows  # 初盘赔率是 0：当作没有
    assert rows["ou", "begin"].line == 3.75 and rows["ou", "now"].line == 2.5
    assert rows["ou", "now"].home == 1.96 and rows["ou", "now"].cid == "1"


@pytest.mark.parametrize("text, line", [("2.5", 2.5), ("3.5/4", 3.75), ("2/2.5", 2.25), ("", None), ("大", None)])
def test_ou_line(text, line):
    assert ou_line(text) == line


def test_asia_line_sign():
    assert asia_line("-0.50") == 0.5 and asia_line("0.25") == -0.25 and asia_line("0") == 0.0 and asia_line("") is None


def test_resolve_ts_picks_nearest_year():
    ko = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    assert resolve_ts("12-31 23:00", ko) == datetime(2024, 12, 31, 15, 0, tzinfo=UTC)  # 北京时间 → 标准时间减 8 小时
    assert resolve_ts("01-01 19:00", ko) == datetime(2025, 1, 1, 11, 0, tzinfo=UTC)
    assert resolve_ts("", ko) is None


def test_parse_fixtures_and_next_page():
    data = {"list": [{"match_id": "7", "start_play": "2026-10-31 12:30:00", "status": "Fixture", "team_A_name": "切尔西",
                      "team_B_name": "曼联", "fs_A": "", "fs_B": "", "gameweek": "9"}],
            "nextDate": "2026-11-30 00:00:00next"}
    f = parse_fixtures(data)[0]
    assert f.match_id == 7 and f.kickoff_utc == datetime(2026, 10, 31, 12, 30, tzinfo=UTC) and f.home_goals is None
    assert next_start(data) == "2026-11-30 00:00:00next" and next_start({"nextDate": ""}) is None


def test_check_odds_rejects_other_match():
    assert check_odds(json.dumps(odds_json(53000000)).encode(), 3000000).startswith("返回了另一场比赛")
    assert check_odds(json.dumps(odds_json(5)).encode(), 5) is None
    assert check_odds(b"<html>", 5) == "不是 JSON"


def test_match_by_voting():
    """中文队名靠“日期 + 比分”投票对上英文名，再按主客队对比赛。"""
    from datetime import date
    from src.clean.dongqiudi_match import DbMatch, decide, match_games, vote
    from src.clean.dongqiudi_reader import SeasonGame
    teams = [("曼城", "Man City"), ("切尔西", "Chelsea"), ("利物浦", "Liverpool"), ("阿森纳", "Arsenal")]
    db, dqd, k = {2023: []}, {2023: []}, 0
    for i, (zh_h, en_h) in enumerate(teams):          # 双循环，每场比分不同
        for j, (zh_a, en_a) in enumerate(teams):
            if i == j:
                continue
            k += 1
            d = date(2023, 8, 1) + timedelta(days=7 * k)
            db[2023].append(DbMatch(k, 2023, d, True, en_h, en_a, k % 5, k // 5))
            dqd[2023].append(SeasonGame(1000 + k, k, d.month, d.day, zh_h, zh_a, k % 5, k // 5))
    dqd[2023][0].home_goals = 9   # 懂球帝这场比分写错了：投票时不算，对比赛时报“比分不同”
    votes = defaultdict(lambda: defaultdict(float))
    vote(dqd, db, votes)
    mapping, table = decide(votes)
    assert mapping == dict(teams)
    rows, left = match_games("E0", dqd, db, mapping)
    results = [r["结果"] for r in rows]
    assert results.count("对上") == 11 and results[0] == "比分不同" and left[2023] == 0
    assert rows[1]["我们的编号"] == 2


def test_snapshot_gap_and_spread():
    rules = [[6, 1], [48, 3], [192, 12]]
    assert snapshot_gap(2, rules) == 1 and snapshot_gap(24, rules) == 3 and snapshot_gap(100, rules) == 12
    assert snapshot_gap(300, rules) is None
    assert spread(list(range(10)), 3) == [0, 4, 9] and spread([1, 2], 3) == [1, 2]


# ---------- 采集（假网站） ----------
class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content


class FakeSite:
    """假网站：pages 里有的网址返回 (状态码, 内容)，没有的返回 404。记下每次请求。"""
    def __init__(self):
        self.pages, self.requests, self.headers = {}, [], {}

    def get(self, url, timeout):
        self.requests.append(url)
        code, body = self.pages.get(url, (404, b""))
        return FakeResponse(code, body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode())


class Clock:
    def __init__(self, t):
        self.t = t


CFG = {"min_interval_seconds": 0, "league_ids": {"E0": 4}, "upcoming_lookback_days": 3, "upcoming_ahead_days": 8,
       "max_fixture_pages": 4, "odds_snapshot_rules": [[6, 1], [48, 3], [192, 12]], "final_after_hours": 3,
       "detail_after_hours": 6}


@pytest.fixture
def env(tmp_path):
    con = connect(tmp_path / "test.db")
    site, clock = FakeSite(), Clock(datetime(2026, 10, 30, 8, 0, tzinfo=UTC))
    c = Collector(con, tmp_path / "raw", CFG, session=site, sleep=lambda s: None, now=lambda: clock.t)
    return con, site, clock, c, tmp_path / "raw" / "dongqiudi"


def fixture_item(mid, ko: datetime, status="Fixture"):
    return {"match_id": str(mid), "start_play": ko.strftime("%Y-%m-%d %H:%M:%S"), "status": status,
            "team_A_name": "甲", "team_B_name": "乙", "fs_A": "1" if status == "Played" else "",
            "fs_B": "0" if status == "Played" else "", "gameweek": "9"}


def fixtures_url(day: str):
    return FIXTURES_URL.format(league_id=4, start=day.replace(" ", "%20").replace(":", "%3A"))


def test_odds_final_saved_once_and_mismatch_marked(env):
    con, site, clock, c, out = env
    site.pages[ODDS_URL.format(match_id=5)] = (200, odds_json(5))
    site.pages[ODDS_URL.format(match_id=3000000)] = (200, odds_json(53000000))  # 网站换成了别的比赛
    assert c.odds_final("E0", 2023, 5) == "saved"
    assert c.odds_final("E0", 2023, 5) == "skipped"            # 踢完的比赛只抓一次
    assert c.odds_final("E0", 2023, 3000000) == "suspicious"
    assert c.odds_final("E0", 2023, 3000000) == "skipped"      # 可疑的不反复请求
    assert len(site.requests) == 2
    assert json.loads(gzip.decompress((out / "odds_final/E0/2023/5.json.gz").read_bytes()))["matchId"] == 5
    assert (out / "odds_final/E0/2023/3000000.suspicious.json.gz").exists()
    assert not (out / "odds_final/E0/2023/3000000.json.gz").exists()
    assert con.execute("select count(*) from fetch_log where source='dongqiudi'").fetchone()[0] == 2


def test_blocked_stops(env):
    con, site, clock, c, out = env
    site.pages[HISTORY_URL.format(league_id=4)] = (200, b"<html>please verify</html>")  # 拦截页
    with pytest.raises(Blocked):
        c.history("E0")
    site.pages[ODDS_URL.format(match_id=9)] = (403, b"")
    with pytest.raises(Blocked):
        c.odds_final("E0", 2023, 9)


def test_backfill_finished_season(env):
    con, site, clock, c, out = env
    site.pages[HISTORY_URL.format(league_id=4)] = (200, HISTORY_HTML.encode())
    site.pages[SEASON_URL.format(season_id=21198)] = (200, SEASON_HTML.encode())
    for mid in (101, 102, 103):
        site.pages[ODDS_URL.format(match_id=mid)] = (200, odds_json(mid))
    c.backfill(["E0"], 2023, 2023)
    assert sorted(p.name for p in (out / "odds_final/E0/2023").iterdir()) == ["101.json.gz", "102.json.gz", "103.json.gz"]
    n = len(site.requests)
    c.backfill(["E0"], 2023, 2023)  # 再跑一次：赛季页和赔率都有了，不再请求（历届赛季页当天已抓过）
    assert len(site.requests) == n


def test_upcoming_snapshot_schedule(env):
    con, site, clock, c, out = env
    now = clock.t
    site.pages[fixtures_url("2026-10-27 00:00:00")] = (200, {"list": [
        fixture_item(1, now - timedelta(hours=20), "Played"),   # 昨天踢完 → 最终赔率
        fixture_item(2, now + timedelta(hours=2)),              # 2 小时后开球 → 每小时一次
        fixture_item(3, now + timedelta(hours=30)),             # 30 小时后 → 每 3 小时
        fixture_item(4, now + timedelta(days=12)),              # 太远 → 不抓
        fixture_item(5, now + timedelta(hours=1), "Postponed"),  # 延期 → 不抓
    ], "nextDate": ""})
    for mid in (1, 2, 3, 4, 5):
        site.pages[ODDS_URL.format(match_id=mid)] = (200, odds_json(mid))
    c.upcoming(["E0"])
    odds_requests = lambda: sorted(int(u.split("/index/")[1].split("?")[0]) for u in site.requests if "/odds/" in u)
    assert odds_requests() == [1, 2, 3]
    assert (out / "odds_final/E0/2026/1.json.gz").exists()
    assert len(list((out / "odds_live/E0/2").glob("*.json.gz"))) == 1

    # 1 小时后再跑：2 号要抓（内容没变 → 只记“没变化”），3 号不到 3 小时不抓，1 号已经有最终赔率
    site.requests.clear()
    clock.t = now + timedelta(hours=1)
    c.upcoming(["E0"])
    assert odds_requests() == [2]
    assert len(list((out / "odds_live/E0/2").glob("*.json.gz"))) == 1
    assert con.execute("select status from fetch_log where url=? order by id desc limit 1",
                       (ODDS_URL.format(match_id=2),)).fetchone()[0] == "unchanged"

    # 赔率变了：存一份新快照
    site.requests.clear()
    clock.t = now + timedelta(hours=1, minutes=55)
    site.pages[ODDS_URL.format(match_id=2)] = (200, odds_json(2, home="1.40"))
    c.upcoming(["E0"])
    assert odds_requests() == [2]
    assert len(list((out / "odds_live/E0/2").glob("*.json.gz"))) == 2


def test_requests_spaced_out(tmp_path):
    """两次请求至少隔 min_interval 秒。"""
    con = connect(tmp_path / "t.db")
    site, sleeps, t = FakeSite(), [], [0.0]
    c = Collector(con, tmp_path / "raw", {**CFG, "min_interval_seconds": 5}, session=site,
                  sleep=lambda s: sleeps.append(s), clock=lambda: t[0],
                  now=lambda: datetime(2020, 1, 1, tzinfo=UTC))
    c.odds_live("E0", 1)
    c.odds_live("E0", 2)
    assert sleeps == [5]


def test_backfill_progress_report(env):
    """回补进度：已抓 = 有文件的 + 网站说没有的；不在历届列表里的赛季标“懂球帝没有”；全程不联网。"""
    from src.jobs.dongqiudi_progress import report, run_state, season_rows
    con, site, clock, c, out = env
    site.pages[HISTORY_URL.format(league_id=4)] = (200, HISTORY_HTML.encode())
    site.pages[SEASON_URL.format(season_id=21198)] = (200, SEASON_HTML.encode())
    site.pages[ODDS_URL.format(match_id=101)] = (200, odds_json(101))
    c.season_games("E0", 2023, c.history("E0")[2023])
    c.odds_final("E0", 2023, 101)          # 有赔率
    c.odds_final("E0", 2023, 102)          # 假网站返回 404：网站没有这场
    n = len(site.requests)
    rows = {r["赛季"]: r for r in season_rows(con, out, ["E0"], 2023, clock.t)}
    assert len(site.requests) == n         # 查进度不发请求
    assert (rows["2023/24"]["应有"], rows["2023/24"]["已抓"], rows["2023/24"]["有赔率"]) == (3, 2, 1)
    assert rows["2024/25"]["懂球帝没有"]   # 历届列表里只有 2025/26 和 2023/24
    assert rows["2025/26"]["应有"] is None and not rows["2025/26"]["懂球帝没有"]  # 赛季页还没抓
    assert rows["2026/27"]["进行中"]
    state, _ = run_state(con, clock.t + timedelta(minutes=1))
    assert state == "正在运行"
    assert run_state(con, clock.t + timedelta(hours=1))[0].startswith("没在运行")
    text = report(list(rows.values()), state, None, 5, clock.t)
    assert "还剩 1 场" in text and "懂球帝没有这个赛季" in text and "还没开始" in text


def test_backfill_limit_and_no_half_files(env):
    """小批量：每个赛季最多新抓 limit 场，再跑一次接着抓下一批；保存时不留临时文件。"""
    con, site, clock, c, out = env
    site.pages[HISTORY_URL.format(league_id=4)] = (200, HISTORY_HTML.encode())
    site.pages[SEASON_URL.format(season_id=21198)] = (200, SEASON_HTML.encode())
    for mid in (101, 102, 103):
        site.pages[ODDS_URL.format(match_id=mid)] = (200, odds_json(mid))
    c.backfill(["E0"], 2023, 2023, limit=2)
    assert sorted(p.name for p in (out / "odds_final/E0/2023").iterdir()) == ["101.json.gz", "102.json.gz"]
    c.backfill(["E0"], 2023, 2023, limit=2)
    assert sorted(p.name for p in (out / "odds_final/E0/2023").iterdir()) == ["101.json.gz", "102.json.gz", "103.json.gz"]
    assert not list(out.rglob("*.tmp"))


# ---------- 比赛详情页（xG） ----------
def detail_html(page_id, xg=True, home="桑德兰", away="利兹联", hg="0", ag="0"):
    """假的比赛详情页：和真网页一样，开头有 data-id，数据在 window.__INITIAL_STATE__ 里（JSON 里的编号少 50000000）。"""
    stats = [{"type": "射门", "team_A": {"value": 15}, "team_B": {"value": 11}},
             {"type": "射正", "team_A": {"value": 4}, "team_B": {"value": 4}},
             {"type": "角球", "team_A": {"value": 4}, "team_B": {"value": 5}}]
    if xg:
        stats.append({"type": "预期进球(xG)", "team_A": {"value": 0.83}, "team_B": {"value": 0.76}})
    state = {"matchContentStore": {"matchDetailHeaderData": {
        "match": {"id": str(page_id - 50000000), "status": "Played",
                  "team_A": {"name": home, "fs": hg}, "team_B": {"name": away, "fs": ag}},
        "info": {"statistics": {"list": stats}}}}}
    return (f'<html><div data-id="{page_id}" class="matchd"></div><script>window.__INITIAL_STATE__='
            f'{json.dumps(state, ensure_ascii=False)};(function(){{}})()</script></html>').encode()


def test_parse_match_detail():
    d = parse_match_detail(detail_html(54483744).decode())
    assert (d.page_id, d.status, d.home, d.away, d.home_goals, d.away_goals) == (54483744, "Played", "桑德兰", "利兹联", 0, 0)
    assert detail_columns(d) == {"home_shots": 15, "away_shots": 11, "home_shots_on_target": 4,
                                 "away_shots_on_target": 4, "home_xg": 0.83, "away_xg": 0.76}
    old = detail_columns(parse_match_detail(detail_html(54483744, xg=False).decode()))
    assert old["home_xg"] is None and old["home_shots"] == 15          # 2025 年 12 月以前的比赛没有 xG
    assert parse_match_detail("<html>请验证</html>") is None
    assert check_detail(detail_html(54483744), 54483744) is None
    assert check_detail(detail_html(54483744), 54483745) == "页面不是这一场比赛"
    assert check_detail(b"<html>please verify</html>", 54483744).startswith("页面里没有比赛数据")


def test_upcoming_fetches_detail_after_hours(env):
    """踢完 3 小时抓最终赔率；6 小时后才抓详情页（统计定下来要一点时间），只抓一次。"""
    con, site, clock, c, out = env
    now = clock.t
    site.pages[fixtures_url("2026-10-27 00:00:00")] = (200, {"list": [
        fixture_item(1, now - timedelta(hours=20), "Played"),
        fixture_item(2, now - timedelta(hours=4), "Played"),     # 4 小时前开球：有最终赔率，详情页还不抓
    ], "nextDate": ""})
    for mid in (1, 2):
        site.pages[ODDS_URL.format(match_id=mid)] = (200, odds_json(mid))
        site.pages[DETAIL_URL.format(match_id=mid)] = (200, detail_html(mid))
    c.upcoming(["E0"])
    assert [u for u in site.requests if "matchDetail" in u] == [DETAIL_URL.format(match_id=1)]
    assert (out / "match_detail/E0/2026/1.html.gz").exists()
    site.requests.clear()
    clock.t = now + timedelta(hours=3)
    c.upcoming(["E0"])
    assert [u for u in site.requests if "matchDetail" in u] == [DETAIL_URL.format(match_id=2)]


def test_detail_backfill_since_and_stop(env):
    """补详情页：只要 since 以后踢完的（赛季页的“月-日”按赛季推年份），从近往远；连着 3 场可疑就停。"""
    con, site, clock, c, out = env
    clock.t = datetime(2026, 2, 1, 8, 0, tzinfo=UTC)        # 2025/26 赛季进行中
    site.pages[HISTORY_URL.format(league_id=4)] = (200, HISTORY_HTML.encode())
    site.pages[SEASON_URL.format(season_id=21198)] = (200, SEASON_HTML.encode())  # 2023/24（since 之前，不抓）
    start = clock.t.replace(year=2025, month=7, day=1, hour=0)
    site.pages[fixtures_url(start.strftime("%Y-%m-%d 00:00:00"))] = (200, {"list": [
        fixture_item(11, datetime(2025, 11, 1, 15, tzinfo=UTC), "Played"),   # since 之前
        fixture_item(12, datetime(2025, 12, 6, 15, tzinfo=UTC), "Played"),
        fixture_item(13, datetime(2026, 1, 20, 15, tzinfo=UTC), "Played"),
        fixture_item(14, datetime(2026, 2, 7, 15, tzinfo=UTC)),               # 还没踢
    ], "nextDate": ""})
    for mid in (12, 13):
        site.pages[DETAIL_URL.format(match_id=mid)] = (200, detail_html(mid))
    from datetime import date
    c.detail_backfill(["E0"], date(2025, 11, 15))
    assert [u for u in site.requests if "matchDetail" in u] == [DETAIL_URL.format(match_id=13), DETAIL_URL.format(match_id=12)]
    assert sorted(p.name for p in (out / "match_detail/E0/2025").iterdir()) == ["12.html.gz", "13.html.gz"]

    # 已结束的赛季用赛季页：2023/24 赛季页里 08-12、08-13 是 2023 年，01-02 是 2024 年
    clock.t = datetime(2026, 10, 30, 8, 0, tzinfo=UTC)
    site.pages[fixtures_url("2026-07-01 00:00:00")] = (200, {"list": [], "nextDate": ""})
    for mid in (101, 102, 103):
        site.pages[DETAIL_URL.format(match_id=mid)] = (200, b"<html>please verify</html>")
    site.requests.clear()
    c.detail_backfill(["E0"], date(2023, 8, 13), limit=5)   # 101（2023-08-12）在 since 之前
    assert [u for u in site.requests if "matchDetail" in u] == [DETAIL_URL.format(match_id=103), DETAIL_URL.format(match_id=102)]
    site.requests.clear()
    with pytest.raises(Blocked):                             # 再加上 101：连着 3 场拦截页 → 停
        c.detail_backfill(["E0"], date(2023, 8, 1), retry_bad=True)
    assert [u for u in site.requests if "matchDetail" in u] == [DETAIL_URL.format(match_id=m) for m in (103, 102, 101)]
