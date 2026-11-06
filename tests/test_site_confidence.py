"""网站不显示信心等级（settings.yaml site.show_confidence: false，2026-11-07 起，需求文档第 8 节）。

实时预测每场都是 ●●● 高，显示出来没有区分作用，所以网站上不写信心，结论那一行的小标签改写“概率从哪来”：
赔率占 100% → “概率来自去水赔率”；赔率占一部分 → “赔率占 x%、模型占 y%”；没用上赔率 → 不写这个标签（另有“只有模型”的提醒）。
存档、接口里的信心等级照样有，只是页面不显示。
"""
import json
import re
from pathlib import Path

from src.web.site.match import level_tag, show_level, source_text
from tests.test_serve import tree  # noqa: F401（world 要用的 fixture）
from tests.test_serve_live import LiveRunning, world  # noqa: F401（world 是 pytest 的 fixture）
from tests.test_site import build, env, read  # noqa: F401（env 是 pytest 的 fixture）
from tests.test_site_story import text_of

JS = (Path(__file__).resolve().parents[1] / "src" / "web" / "static" / "site.js").read_text(encoding="utf-8")


def row_with(fusion: dict) -> dict:
    return {"detail": {"融合": fusion}}


def test_setting_is_off_and_default_is_on():
    from src.common.config import load_settings
    assert show_level(load_settings()) is False                      # 现在的设置：不显示
    assert show_level({}) is True and show_level({"site": {}}) is True   # 没写当 true（以前的样子）


def test_source_text_uses_saved_w():
    assert source_text(row_with({"状态": "已融合", "w": 1.0})) == "概率来自去水赔率"
    assert source_text(row_with({"状态": "已融合", "w": 1})) == "概率来自去水赔率"
    assert source_text(row_with({"状态": "已融合", "w": 0.6})) == "赔率占 60%、模型占 40%"
    assert source_text(row_with({"状态": "赔率不全", "w": 1.0})) == "概率来自去水赔率"   # 赔率不全也是用上了赔率
    assert source_text(row_with({"状态": "无赔率，未融合", "w": 1.0})) is None          # 没用上赔率：不写
    assert source_text(row_with({"状态": "已融合"})) is None                              # 没存 w：不编
    assert source_text(row_with({"状态": "已融合", "w": True})) is None
    assert source_text({"detail": {}}) is None                                           # 接融合以前的旧版


def test_level_tag_switches_between_confidence_and_source():
    r = row_with({"状态": "已融合", "w": 1.0, "信心等级": 3})
    assert "●●● 高" in level_tag(r, True)
    assert level_tag(r, False) == "<span class='tag'>概率来自去水赔率</span>"
    assert level_tag(row_with({"状态": "无赔率，未融合", "信心等级": 2}), False) == ""


def test_static_pages_hide_confidence(env):
    out, _ = build(env)
    page = read(out, "match/1.html")
    t = text_of(page)
    assert "<span class='tag'>赔率占 60%、模型占 40%</span>" in page        # 造的数据 w = 0.6
    assert "信心 " not in t and "●●●" not in t and "●●○" not in t
    assert "<th>信心</th>" not in page                                    # 版本表、简表都去掉了“信心”那一列
    assert "按门槛写倾向" in t and "再定信心" not in t                     # 结论链第 6 步
    index = read(out, "index.html")
    row1 = re.search(r"<tr( data-match-id='1'[^>]*)>", index).group(1)
    assert "data-level=" not in row1                                      # 筛选“信心”没有可选的，脚本会把那一项藏起来
    assert "赔率占 60%、模型占 40%" in index and "●●●" not in index


def test_static_pages_show_confidence_when_turned_on(env):
    env[1]["site"]["show_confidence"] = True
    out, _ = build(env)
    page = read(out, "match/1.html")
    assert "●●● 高" in page and "<th>信心</th>" in page
    assert "data-level='3'" in read(out, "index.html")


def test_script_follows_the_setting():
    assert 'api.show_confidence !== false' in JS                         # 单场页：接口说不显示就不写信心
    assert "v.source_text" in JS
    assert 'hasAttribute("data-level")' in JS and "fields[3].wrap.hidden = true" in JS   # 首页：没有等级就藏起筛选框


def test_api_tells_page_whether_to_show_confidence(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        resp, body = srv.req("GET", "/api/match/1", srv.cookie())
    assert resp.status == 200
    api = json.loads(body.decode("utf-8"))
    assert api["show_confidence"] is False
    for v in api["versions"] + [api["latest"]]:
        assert "source_text" in v and "confidence" in v                   # 等级照样给，页面不显示
        if v["source"] == "model":
            assert v["source_text"] is None
        else:
            assert isinstance(v["source_text"], str) and v["source_text"]
