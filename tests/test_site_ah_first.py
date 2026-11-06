"""亚盘倾向放到最前面当主结论（工作树 WD5，需求文档 v0.98）：只改显示顺序和措辞的自动检查。全是造的数据。

要检查的：
- 首页每场那一格：有亚盘倾向（picks_v2 的让球预测）时最上面一句写亚盘，下面倾向行不再重复亚盘；
  没有合格的主盘口（pick_ah 是 None）时没有这一句，格子和以前一样写胜平负。
- 单场页：大字是亚盘那句、下面一行写胜平负；“四个玩法的预测”表里亚盘排在胜平负前面；没有亚盘时大字照旧是胜平负。
- 没到倾向门槛（存档 lean 是 False）写“接近五五开”。
- 接口 /api/... 的字段不变：leans 里照样有亚盘那句，picks 的四个玩法还是原来的顺序。
- 全站没有“推荐”。
用 tests/test_site.py 的临时数据库（利物浦 对 阿森纳、利物浦 对 曼城）和 tests/test_explanation_pages.py 的 site_data。
"""
import html
import re

from src.web.live_site import LiveSite
from src.web.site import picks as pk
from src.web.site import story
from src.web.site.common import TeamFiles
from src.web.site.match import conclusion_html, render as render_match
from src.web.site.pages import latest_cell
from tests.test_explanation_pages import site_data
from tests.test_site import all_pages, build, env, read, write_review  # noqa: F401  （env 是 pytest 的 fixture）


def text_of(page: str) -> str:
    """去掉标签、换回转义过的符号，只留文字。"""
    return html.unescape(re.sub(r"<[^>]+>", "", page))


def with_ah(data, ah):
    """site_data 里第 10 场最新一版（T-1）的融合倾向：胜平负倾向主胜 60%，亚盘按给的 ah（None = 没有合格的主盘口）。"""
    row = data.latest(10)[1]
    row["detail"]["融合"]["倾向"] = {"result": {"side": 0, "p": .6, "lean": True}, "draw": {"p": .2, "risk": False},
                                    "over25": None, "btts": None, "ah": ah}
    return row


# ---------- 一句话怎么写 ----------
def test_ah_sentence_wording():
    s = pk.ah_sentence
    home = {"side": "home", "line": -0.5, "p": 0.53, "lean": True}
    assert s(home, "利物浦", "阿森纳") == "倾向利物浦让半球赢盘，胜算 53%"
    # 客队那边：主队 -0.25（主让平半）= 客队 +0.25，客队“受让平半”
    assert s({"side": "away", "line": -0.25, "p": 0.55, "lean": True}, "利物浦", "阿森纳") == "倾向阿森纳受让平半赢盘，胜算 55%"
    assert s({"side": "away", "line": 1.0, "p": 0.54, "lean": True}, "利物浦", "阿森纳") == "倾向阿森纳让一球赢盘，胜算 54%"
    assert s({"side": "home", "line": 0.0, "p": 0.54, "lean": True}, "利物浦", "阿森纳") == "倾向利物浦赢盘（平手盘），胜算 54%"
    # 存档写着没到倾向门槛：接近五五开（不在网页这边重新定门槛）
    assert s({**home, "p": 0.51, "lean": False}, "利物浦", "阿森纳") == "利物浦让半球赢盘的胜算 51%，接近五五开"
    # 接融合以前的旧存档（lean 是 None）：不说“倾向”
    assert s({**home, "lean": None}, "利物浦", "阿森纳") == "利物浦让半球赢盘的胜算 53%（这一版没有存倾向）"
    assert s(None, "利物浦", "阿森纳") is None
    for p in (home, {**home, "lean": False}, {**home, "lean": None}):
        assert "推荐" not in s(p, "利物浦", "阿森纳")


# ---------- 首页那一格 ----------
def test_index_cell_leads_with_ah_and_falls_back_to_1x2():
    data = site_data()
    with_ah(data, {"side": "home", "p": .53, "line": -0.5, "lean": True})
    cell = latest_cell(data, 10)
    t = text_of(cell)
    assert t.startswith("倾向利物浦让半球赢盘，胜算 53%")              # 亚盘那句在最前面（在概率条、胜平负之前）
    assert t.index("赢盘") < t.index("主胜 60%")
    assert "倾向 主胜 60%" in t and "主让半球" not in t                 # 下面倾向行只写胜平负，不重复亚盘
    assert cell.count("<b class='lead'>") == 1
    # 没有合格的主盘口：没有亚盘那句，和以前一样（胜平负概率 + 胜平负倾向）
    row = with_ah(data, None)
    cell = latest_cell(data, 10)
    t = text_of(cell)
    assert "class='lead'" not in cell and "赢盘" not in t and "胜算" not in t
    assert t.startswith("主胜 60% / 平 20% / 客胜 20%") and "倾向 主胜 60%" in t
    assert conclusion_html(row, show=False) in cell                     # 胜平负这一块和单独写的一模一样（不显示信心等级）


def test_index_cell_near_even():
    data = site_data()
    with_ah(data, {"side": "away", "p": .51, "line": -0.25, "lean": False})
    t = text_of(latest_cell(data, 10))
    assert t.startswith("阿森纳受让平半赢盘的胜算 51%，接近五五开")
    assert "亚盘 无明显倾向" not in t                                    # 不重复写一遍


# ---------- 单场页 ----------
def picks_rows(page: str) -> list[str]:
    """“四个玩法的预测”表每行第一格（玩法名），从上到下。"""
    table = page[page.index("<table class='picks'>"):]
    table = table[:table.index("</table>")]
    return re.findall(r"<tr><td>([^<]+)</td>", table)


def test_match_page_ah_before_1x2(env):
    write_review(env[2])
    out, _ = build(env)
    for rel in ("match/1.html", "match/10.html"):
        page = read(out, rel)
        # picks_v3（2026-11-05）起大小球那一行改看主盘口，行名是“大小球（主盘口）”；顺序照旧亚盘第一
        assert picks_rows(page) == ["让球（亚盘）", "胜平负", "大小球（主盘口）", "比分 4 个"], rel
        hero = text_of(page[page.index("<div class='card pad hero'>"):page.index("<div class='version-switch'")])
        # 造的数据：亚盘倾向主队 -0.5 胜算 53%；胜平负利物浦 50%
        assert hero.startswith("倾向利物浦让半球赢盘，胜算 53%"), rel
        assert "胜平负：利物浦赢的可能更大（50%）" in hero, rel
        assert hero.index("赢盘") < hero.index("胜平负：")
    idx = text_of(read(out, "index.html"))
    assert "倾向利物浦让半球赢盘，胜算 53%" in idx and "倾向 主队（主让半球）" not in idx
    assert "每场最上面那句粗字是亚盘（让球）的倾向" in idx


def test_match_page_without_ah_keeps_1x2_headline():
    data = site_data()
    with_ah(data, None)
    page = render_match(data, 10, TeamFiles(data.team_zh))
    hero = text_of(page[page.index("<div class='card pad hero'>"):page.index("<div class='version-switch'")])
    assert hero.startswith("利物浦赢的可能更大（60%）") and "赢盘" not in hero and "<p class='subhead'>" not in page
    rows = picks_rows(page)
    assert rows[:2] == ["让球（亚盘）", "胜平负"]                         # 表的顺序照样，让球那行写“这一版没法预测”
    assert "这一版没法预测" in text_of(page)


def test_match_page_near_even_headline():
    data = site_data()
    row = with_ah(data, {"side": "home", "p": .51, "line": -0.75, "lean": False})
    assert story.hero_html(data, 10, row).startswith("<p class='headline'>利物浦让半一赢盘的胜算 51%，接近五五开</p>"
                                                     "<p class='subhead'>胜平负：利物浦赢的可能更大（60%）</p>")
    page = render_match(data, 10, TeamFiles(data.team_zh))
    assert "接近五五开" in text_of(page) and "推荐" not in page


# ---------- 接口不变 ----------
def test_api_fields_unchanged():
    data = site_data()
    with_ah(data, {"side": "home", "p": .53, "line": -0.5, "lean": True})
    latest = LiveSite._latest(data, 10)
    assert latest["leans"] == ["倾向 主胜 60%", "倾向 主队（主让半球）53%"]   # leans 还是全部玩法、原来的写法
    got = LiveSite._picks(data, 10, data.latest(10)[1])
    assert list(got["pick"]) == ["1x2", "ah", "ou", "score"] == list(pk.MARKETS)
    assert got["pick"]["ah"]["text"] == "利物浦 -0.5" and got["pick"]["ah"]["line"] == -0.5


def test_no_advice_word_anywhere(env):
    write_review(env[2])
    out, _ = build(env)
    for p in all_pages(out):
        assert "推荐" not in p.read_text(encoding="utf-8"), p
