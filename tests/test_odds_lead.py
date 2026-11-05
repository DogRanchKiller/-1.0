"""盘口联动观察页 odds_lead.html（src/web/site/odds_lead.py）的自动检查。只用造的变盘记录，不读真实数据库。

造的比赛（开球 K = 2026-11-03 20:00 UTC，窗口 48 小时）：赔率都用“没有抽水”的价格（1/p），去水后正好是 p，好算。
  A 在 K-20h 先从 0.50 变到 0.56；B、C 在 K-10h 一起变；D 在 K-3h 晚变；E 一直不动（没跟）；
  F 在 K-11h 变（正好早 1 小时，算先变）；G 只动了 0.01（自己动完了，但没走到市场变动的一半，算没跟）。
  7 家的中位数：开始 0.50，K-10h 起 0.56，D = +0.06，市场在 K-10h 走到一半。
"""
import copy
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.common.config import load_settings
from src.web.site import odds_lead
from src.web.site.common import page
from src.web.site.data import SiteData

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
K = pd.Timestamp("2026-11-03 20:00:00")
WEIRD = "A<b>&\"x'"


def at(hours_before: float) -> str:
    return (K - pd.Timedelta(hours=hours_before)).strftime("%Y-%m-%d %H:%M:%S")


def row(book, hours_before, market="1x2", p=0.5, line=None, mid=1):
    """一行变盘记录。p = 主胜（胜平负，平局固定 0.25）/ 主队赢盘（让球）/ 大球（大小球）的机会。"""
    r = {"match_id": mid, "source": "dongqiudi", "bookmaker": book, "bookmaker_raw": book, "market": market,
         "line": line, "home": None, "draw": None, "away": None, "over": None, "under": None,
         "changed_at_utc": at(hours_before), "fetched_at_utc": at(hours_before)}
    if market == "1x2":
        r.update(home=1 / p, draw=1 / 0.25, away=1 / (0.75 - p))
    elif market == "ah":
        r.update(home=1 / p, away=1 / (1 - p))
    else:
        r.update(over=1 / p, under=1 / (1 - p))
    return r


def base_rows(market="1x2", line=None, mid=1, books=("A", "B", "C", "D", "E", "F", "G")):
    """每家在 K-60h 开出 0.50（窗口开始之前）。"""
    return [row(b, 60, market, 0.50, line, mid) for b in books]


def story_rows(market="1x2", line=None, mid=1):
    moves = [("A", 20, 0.56), ("B", 10, 0.56), ("C", 10, 0.56), ("D", 3, 0.56), ("F", 11, 0.56), ("G", 15, 0.51)]
    return base_rows(market, line, mid) + [row(b, h, market, p, line, mid) for b, h, p in moves]


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.sort_values("changed_at_utc", kind="stable").reset_index(drop=True)


@pytest.fixture
def cfg():
    return dict(copy.deepcopy(load_settings())["odds_lead"])


def make_data(odds: dict[int, pd.DataFrame], kickoffs: dict[int, str | None] | None = None, min_matches=None) -> SiteData:
    s = copy.deepcopy(load_settings())
    if min_matches is not None:
        s["odds_lead"]["min_matches"] = min_matches
    data = SiteData(NOW, s)
    for mid in odds:
        kick = (kickoffs or {}).get(mid, K.strftime("%Y-%m-%d %H:%M:%S"))
        data.matches[mid] = {"match_id": mid, "competition": "E0", "season": 2026, "home_team": f"Home{mid}",
                             "away_team": f"Away{mid}", "kickoff_utc": kick, "match_date": "2026-11-03",
                             "kickoff_earliest_utc": kick or "2026-11-03", "date_precision": "day", "site_league": True}
    data.odds = odds
    data.team_zh = {"Home1": "主队一", "Away1": "客队一"}
    return data


def check_safe(html: str) -> None:
    assert "推荐" not in html
    assert "<script" not in html.lower()
    assert "只是观察，不是预测" in html and "不能拿来下注" in html


# ---------- 规则 ----------
def test_settings_block(cfg):
    assert cfg == {"since": "2026-10-30 18:12", "window_hours": 48, "min_move": 0.02, "min_books": 5,
                   "tolerance_hours": 1, "min_matches": 10, "recent_rows": 20}


def test_story_kinds(cfg):
    vals, line = odds_lead.book_values(frame(story_rows()), "1x2")
    assert line is None
    mv = odds_lead.match_move(vals, K, cfg)
    assert mv["move"] == pytest.approx(0.06) and mv["n_books"] == 7
    assert mv["market_half"] == K - pd.Timedelta(hours=10)
    kinds = {b: x["kind"] for b, x in mv["books"].items()}
    assert kinds == {"A": "先变", "B": "一起变", "C": "一起变", "D": "晚变", "E": "没跟", "F": "先变", "G": "没跟"}
    assert mv["books"]["A"]["lead"] == pytest.approx(10) and mv["books"]["D"]["lead"] == pytest.approx(-7)
    assert mv["books"]["F"]["lead"] == pytest.approx(1)          # 正好早 1 小时：算先变
    assert mv["first"] == ["A"] and mv["first_t"] == K - pd.Timedelta(hours=20)


def test_tolerance_inside_is_together(cfg):
    rows = story_rows() + [row("H", 60), row("H", 10.5, p=0.56)]  # 只早半小时：一起变
    mv = odds_lead.match_move(odds_lead.book_values(frame(rows), "1x2")[0], K, cfg)
    assert mv["books"]["H"]["kind"] == "一起变"


def test_downward_move(cfg):
    rows = base_rows() + [row(b, h, p=0.44) for b, h in (("A", 20), ("B", 10), ("C", 10), ("D", 10))]
    mv = odds_lead.match_move(odds_lead.book_values(frame(rows), "1x2")[0], K, cfg)
    assert mv["move"] == pytest.approx(-0.06)
    assert mv["books"]["A"]["kind"] == "先变" and mv["books"]["E"]["kind"] == "没跟"


def test_small_move_is_skipped(cfg):
    rows = base_rows() + [row(b, 10, p=0.51) for b in "ABCDEFG"]          # 只变 1 个百分点
    assert odds_lead.match_move(odds_lead.book_values(frame(rows), "1x2")[0], K, cfg) is None


def test_too_few_books_and_late_starters(cfg):
    # 窗口开始时只有 4 家有价格（另外 3 家 K-30h 才开出来，不算）：不够 5 家
    rows = base_rows(books="ABCD") + [row(b, 30, p=0.50) for b in "EFG"] + [row(b, 10, p=0.56) for b in "ABCDEFG"]
    assert odds_lead.match_move(odds_lead.book_values(frame(rows), "1x2")[0], K, cfg) is None
    rows += [row("H", 60)]
    mv = odds_lead.match_move(odds_lead.book_values(frame(rows), "1x2")[0], K, cfg)
    assert set(mv["books"]) == set("ABCDH")


def test_prices_after_kickoff_ignored(cfg):
    rows = base_rows() + [row(b, -1, p=0.60) for b in "ABCDEFG"]          # 开球后的价格（取数层本来就不给）
    assert odds_lead.match_move(odds_lead.book_values(frame(rows), "1x2")[0], K, cfg) is None


def test_same_minute_keeps_last_row(cfg):
    rows = base_rows() + [row("A", 60, p=0.56)]          # A 同一分钟两行：后面那行是“最新”
    vals, _ = odds_lead.book_values(frame(rows), "1x2")
    assert vals[vals["bookmaker"] == "A"]["v"].tolist() == [pytest.approx(0.56)]


def test_common_line_ties():
    ah = pd.DataFrame({"line": [-0.25, 0.25, -0.5, -0.5, 0.25, -0.25]})
    assert odds_lead.common_line(ah, "ah") == 0.25                         # -0.25、0.25、-0.5 各两家：离 0 近的里取大的
    assert odds_lead.common_line(pd.DataFrame({"line": [0.0, -0.25]}), "ah") == 0.0
    assert odds_lead.common_line(pd.DataFrame({"line": [2.5, 2.75, 2.75, 2.5]}), "ou") == 2.5
    assert odds_lead.common_line(pd.DataFrame({"line": [3.0, 2.75, 3.0]}), "ou") == 3.0
    assert odds_lead.common_line(pd.DataFrame({"line": [None]}), "ou") is None


def test_ah_uses_only_common_line(cfg):
    rows = story_rows("ah", -0.5)
    # A 另外开了一条 -0.25 的盘（价格变得很多），但它最后一个价格还在 -0.5 上……
    rows += [row("A", 50, "ah", 0.80, -0.25)]
    rows += [row("A", 1, "ah", 0.56, -0.5)]
    vals, line = odds_lead.book_values(frame(rows), "ah")
    assert line == -0.5
    assert vals["v"].max() == pytest.approx(0.56)                         # -0.25 那条的 0.80 不算
    mv = odds_lead.match_move(vals, K, cfg)
    assert mv["books"]["A"]["kind"] == "先变"
    ou = odds_lead.match_move(odds_lead.book_values(frame(story_rows("ou", 2.5)), "ou")[0], K, cfg)
    assert ou["books"]["D"]["kind"] == "晚变"


def test_eligible_matches(cfg):
    odds = {1: frame(story_rows()), 2: frame(story_rows(mid=2)), 3: frame(story_rows(mid=3)),
            4: frame(story_rows(mid=4)), 5: pd.DataFrame()}
    kicks = {2: None,                               # 不知道开球时间
             3: "2026-11-06 20:00:00",              # 还没开球
             4: "2026-11-01 18:00:00"}              # 开球前 48 小时有一部分不在快照里（早于 since + 48h）
    data = make_data(odds, kicks)
    assert odds_lead.eligible(data, cfg) == [1]
    assert odds_lead.eligible_checked(data) == [1]
    data.matches[1]["date_precision"] = "round"
    assert odds_lead.eligible(data, cfg) == []
    del data.settings["odds_lead"]
    assert odds_lead.eligible_checked(data) == []


def test_summarize_ranks_only_enough(cfg):
    vals = odds_lead.book_values(frame(story_rows()), "1x2")[0]
    mv = odds_lead.match_move(vals, K, cfg)
    small = odds_lead.match_move(odds_lead.book_values(frame(base_rows(books="ABCDE") + [row("A", 20, p=0.56)]
                                                             + [row(b, 10, p=0.56) for b in "BC"]), "1x2")[0], K, cfg)
    rows = odds_lead.summarize([mv, mv, small], {**cfg, "min_matches": 3})
    by = {r["book"]: r for r in rows}
    assert by["A"]["n"] == 3 and by["A"]["先变"] == 3 and by["A"]["median_lead"] == pytest.approx(10)
    assert by["G"]["n"] == 2 and by["G"]["enough"] is False
    enough = [r["book"] for r in rows if r["enough"]]
    assert enough[0] == "A" and set(enough) == set("ABCDE")
    assert [r["book"] for r in rows][-2:] == ["F", "G"]                # 场数不够的排后面


# ---------- 网页 ----------
def test_page_with_moves():
    data = make_data({1: frame(story_rows() + story_rows("ah", -0.5) + story_rows("ou", 2.5))}, min_matches=2)
    html, err = odds_lead.render_checked(data)
    assert err is None
    check_safe(html)
    assert "主队一 对 客队一" in html and "href='match/1.html'" in html
    assert "主胜 50% → 56%（多 6.0 个百分点）" in html
    assert "主让半球" in html and "2.5 球" in html
    assert "胜平负 1 次，让球 1 次，大小球 1 次" in html
    assert "不到 2 场，先不下结论" in html                              # G：只在胜平负没跟……三种玩法都算进来，3 场够了
    assert html.count("<table") == 5                                  # 合在一起、最近的、三种玩法各一张


def test_page_not_enough_matches_and_guessed_names(tmp_path):
    yml = tmp_path / "books.yaml"
    yml.write_text("公司:\n  \"澳*\": {名字: E, 依据: 推测}\n  \"Bet3**\": {名字: A, 依据: 核对过}\n", encoding="utf-8")
    data = make_data({1: frame(story_rows())})
    data.settings["paths"]["bookmakers_dongqiudi"] = str(yml)
    html = odds_lead.render(data)
    check_safe(html)
    assert "不到 10 场，先不下结论" in html and "不排" in html
    assert html.count("（名字是推测的）") >= 1
    assert odds_lead.guessed_names(data.settings) == {"E"}
    data.settings["paths"]["bookmakers_dongqiudi"] = str(tmp_path / "没有这个文件.yaml")
    assert odds_lead.guessed_names(data.settings) == set()


def test_page_escapes_names():
    rows = [{**r, "bookmaker": WEIRD} if r["bookmaker"] == "A" else r for r in story_rows()]
    data = make_data({1: frame(rows)})
    data.team_zh = {"Home1": "<i>主</i>"}
    html = odds_lead.render(data)
    assert WEIRD not in html and "<b>&" not in html and "<i>主</i>" not in html
    assert "A&lt;b&gt;&amp;" in html


def test_page_without_moves():
    html, err = odds_lead.render_checked(make_data({}))
    assert err is None
    check_safe(html)
    assert "还没有可以看的明显变动" in html and "看了 0 场" in html and "<table" not in html


def test_page_fallback_on_error():
    data = make_data({1: frame(story_rows())})
    del data.settings["odds_lead"]["window_hours"]
    html, err = odds_lead.render_checked(data)
    assert err == "KeyError"
    check_safe(html)
    assert "盘口联动页出了意外的错" in html and "<table" not in html


def test_nav_links_to_page():
    assert "href='odds_lead.html'>盘口联动</a>" in page("x", "", "", NOW, 15)
    assert "href='../odds_lead.html'" in page("x", "", "../", NOW, 15)
