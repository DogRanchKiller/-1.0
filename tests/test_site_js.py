"""网站前端脚本 /static/site.js、site.css 的自动检查（需求文档第 8 节“前端脚本”、第 16 节“安全”）。

分三块：
  1. 静态检查（用 Python 读文件）：脚本里没有 innerHTML、eval 这类危险写法，没有外部地址，没有“推荐”。
  2. 生成出来的网页：只引用 /static/site.js、没有内联脚本、没有 onclick 这类属性；脚本筛选、排序要用的 data-* 都在；
     live.meta_refresh: false 时没有整页重新加载；静态文件（live.dynamic: false 的那份）照旧没有脚本。
  3. 脚本里的纯计算（即时比分怎么写、命中率、挑哪几版……）：这台电脑上有 node 时用 node 跑一遍，和 Python 那边的结果对比；
     没有 node 就跳过（node 不是项目要装的东西，不为它新装任何依赖）。
全部用临时数据（test_site.py、test_serve_live.py、test_bots_page.py 造的），不碰真实数据库。
"""
import html as html_lib
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.config import load_settings
from src.web import live_site
from src.web.site import bots, match as match_page, picks as pk
from src.web.site.common import LEVEL_NAMES
from tests.test_bots_page import site, write_results  # noqa: F401（site 是 pytest 的 fixture）
from tests.test_serve_live import PLAYING, LiveRunning, world, write_scores  # noqa: F401（world 是 pytest 的 fixture）
from tests.test_serve import tree  # noqa: F401
from tests.test_site import build as build_static, env  # noqa: F401

STATIC = Path(__file__).resolve().parents[1] / "src" / "web" / "static"
JS = (STATIC / "site.js").read_text(encoding="utf-8")
CSS = (STATIC / "site.css").read_text(encoding="utf-8")

# 脚本里不准出现的写法（第 16 节）：把文字当网页代码放进去、把文字当程序跑、写死外部地址
BANNED = [r"\binnerHTML\b", r"\bouterHTML\b", r"\binsertAdjacentHTML\b", r"document\s*\.\s*write", r"\beval\s*\(",
          r"\bnew\s+Function\b", r"\bFunction\s*\(", r"set(Timeout|Interval)\s*\(\s*['\"`]", r"\bcreateContextualFragment\b",
          r"\bimportScripts\b", r"\bsrcdoc\b", r"(?i)https?://", r"(?i)//[a-z0-9.-]+\.(com|net|org|cn|io)\b",
          r"(?i)\bjavascript:\s*[^/]",
          r"\bimport\s*\(", r"\bWebSocket\b", r"\bpostMessage\b", r"\.setAttribute\(\s*['\"]on", r"\blocalStorage\b"]


# ---------- 1. 静态检查 ----------

def test_site_js_has_no_banned_constructs():
    for pat in BANNED:
        # 区分大小写（function 是普通写法，Function( 才是“把文字当程序跑”）；地址那几条自己带了 (?i)，不分大小写。
        # javascript: 只出现在“去掉 javascript: 链接”的那一句正则里，后面紧跟 /，不算
        hit = re.search(pat, JS)
        assert hit is None, f"site.js 里有不准用的写法：{pat} → {hit.group(0) if hit else ''}"
    assert "推荐" not in JS and "推荐" not in CSS
    assert '"use strict"' in JS
    # 连服务器只用本站的相对地址
    urls = re.findall(r"""(?:getJSON|request|fetch)\(\s*["']([^"']+)""", JS)
    assert urls and all(u.startswith("/") and not u.startswith("//") for u in urls), urls
    # 数据接口只读这几个；/api/actions（按钮进度）和 /api/action/（点按钮，只用 POST）是数据状态页上的操作按钮用的（WD4）
    assert set(re.findall(r'"(/api/[a-z]+)', JS)) == {"/api/version", "/api/live", "/api/match", "/api/actions", "/api/action"}
    assert re.findall(r'method:\s*"([A-Z]+)"', JS) == ["POST"]                  # 只有点按钮那一处是 POST


def test_site_css_has_no_external_things():
    assert not re.search(r"@import|url\s*\(|https?://|expression\s*\(", CSS, re.I)
    # 只给脚本加的东西写样式（class 名以 js- 开头），不改页面本身的样子
    for sel in re.findall(r"\.([a-zA-Z][\w-]*)", re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)):
        assert sel.startswith("js-") or sel in {"pbar", "chart", "on", "hit", "miss"}, sel


def test_site_js_comments_are_chinese():
    """给不会编程的人看：注释是中文的、不少。"""
    comments = [c for c in re.findall(r"//[^\n]*", JS) if re.search(r"[A-Za-z一-鿿]", c)]   # 不算分隔线、空行
    assert len(comments) > 120
    assert sum(1 for c in comments if re.search(r"[一-鿿]", c)) / len(comments) > 0.9


# ---------- 2. 生成出来的网页 ----------

PAGES = ("/", "/match/1.html", "/match/10.html", "/record.html", "/lifecycle.html", "/bots.html", "/odds_lead.html")


def test_live_pages_only_reference_site_js(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        snap = srv.live.snapshot()
        for path in PAGES + (f"/team/{next(iter(snap.team_by_file))}",):
            html = srv.req("GET", path, h)[1].decode("utf-8")
            scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", html, re.I | re.S)
            assert scripts == [""], path                                         # 只有一个、里面是空的（不是内联脚本）
            assert '<script src="/static/site.js" defer></script>' in html, path
            assert '<link rel="stylesheet" href="/static/site.css">' in html, path
            assert not re.search(r"\son[a-z]+\s*=", html, re.I), path           # 没有 onclick= 这类
            assert "javascript:" not in html.lower(), path
            assert re.search(r'<meta name="site-data-version" content="[0-9a-f]{16}">', html), path
            assert f'<meta name="site-data-version" content="{snap.version}">' in html, path
            assert '<meta name="site-poll-seconds" content="30">' in html, path
            assert "推荐" not in html, path


def test_data_attributes_for_filters_exist(world):
    s, tmp = world
    write_scores(tmp, [PLAYING])
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        index = srv.req("GET", "/", h)[1].decode("utf-8")
        # 首页：联赛、有没有预测、信心、概率、球队原名
        row10 = re.search(r"<tr( data-match-id='10'[^>]*)>", index).group(1)
        for attr in ("data-league='E0'", "data-league-name='英超'", "data-pred='1'", "data-teams=", "data-kickoff="):
            assert attr in row10, attr
        assert re.search(r"data-level='[123]'", row10) and re.search(r"data-p='0\.\d{3},0\.\d{3},0\.\d{3}'", row10)
        assert "data-pred='0'" in re.search(r"<tr( data-match-id='11'[^>]*)>", index).group(1)   # 11 还没有预测
        # 即时比分的格子：比赛 10 在踢（服务器已经放了一份），比赛 1 有正式赛果（没有格子）
        assert re.search(r"<td data-live-slot='10'><span class='live-score'>", index)
        assert "data-live-slot='1'" not in index
        assert "比分还没进来" not in index or "data-live-placeholder" in index
        # 单场页：标题下面的格子、版本切换的空容器
        m10 = srv.req("GET", "/match/10.html", h)[1].decode("utf-8")
        assert "<span data-live-slot='10'><span class='live-score'>" in m10
        assert "<span class='tag' data-live-placeholder hidden>还没踢</span>" in m10
        assert "<div class='version-switch' data-version-switch='10'></div>" in m10
        m1 = srv.req("GET", "/match/1.html", h)[1].decode("utf-8")
        assert "data-version-switch='1'" in m1 and "data-live-slot" not in m1        # 踢完的没有即时比分格子
        # 战绩页：每场的四个预测
        rec = srv.req("GET", "/record.html", h)[1].decode("utf-8")
        assert "<table class='picks-record'>" in rec
        assert re.search(r"<tr data-match-id='1' data-league='E0' data-league-name='英超'>", rec)
        for k in pk.MARKETS:
            assert re.search(rf"<td data-market='{k}' data-outcome='(win|loss|half_win|half_loss|push|none)' "
                             rf"data-model-outcome='(win|loss|half_win|half_loss|push|none)'>", rec), k


def test_bots_board_sort_and_curve_attributes(site):
    data, folder = site
    write_results(folder)
    html = bots.render(data)
    assert "<table class='bots-board'>" in html
    rows = re.findall(r"<tr data-book='([^']+)' data-control='([01])'>", html)
    assert len(rows) == 4 and [c for _b, c in rows].count("1") == 1                # 4 个机器人，1 个对照组
    assert "<td data-sort='9999'>" in html                                        # 没排名次的排最后
    assert "<td data-sort='800'>" in html and "<td data-sort='0.16'>" in html
    raw = re.search(r"data-curve='([^']+)'", html).group(1)
    curve = json.loads(html_lib.unescape(raw))
    assert curve["name"] == "融合价值5_最高"
    assert curve["seasons"] == [["2012/13", 300.0, 300.0], ["2013/14", -900.0, -600.0]]
    assert "<script" not in html.lower() and "推荐" not in html


def test_meta_refresh_false_from_settings_has_no_meta_refresh(world):
    """settings.yaml 现在写的是 live.meta_refresh: false、poll_seconds: 30：现读现出的页面不带整页重新加载。"""
    real = live_site.settings_live(load_settings())
    assert real["meta_refresh"] is False and real["poll_seconds"] == 30 and real["dynamic"] is True
    assert real["live_poller"] is False                                           # 不联网
    s, tmp = world
    s["live"] = {**s["live"], "meta_refresh": real["meta_refresh"], "poll_seconds": real["poll_seconds"]}
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        for path in PAGES:
            html = srv.req("GET", path, h)[1].decode("utf-8")
            assert "http-equiv" not in html.lower(), path
            assert '<script src="/static/site.js" defer></script>' in html, path


def test_poll_seconds_setting_and_floor(world):
    s, tmp = world
    for given, want in ((12, 12), (1, 5)):                                      # 最少 5 秒
        s["live"] = {**s["live"], "poll_seconds": given}
        live = live_site.LiveSite(s, tmp / "site")
        assert live.settings["site"]["refresh_minutes"].poll_seconds == want


def test_static_pages_unchanged_without_script(env):
    """静态文件（“生成网站”那份，live.dynamic: false 时网站发的就是它）：没有脚本、没有 data-site-* meta，照旧整页重新加载。"""
    out, _summary = build_static(env)
    minutes = int(env[1]["site"]["refresh_minutes"])
    for f in out.rglob("*.html"):
        html = f.read_text(encoding="utf-8")
        assert "<script" not in html.lower() and "/static/" not in html, f.name
        assert "site-data-version" not in html and "site-poll-seconds" not in html, f.name
        assert f'<meta http-equiv="refresh" content="{minutes * 60}">' in html, f.name


# ---------- 3. 脚本里的纯计算（有 node 才跑） ----------

NODE = shutil.which("node")


def run_node(code: str):
    """用 node 载入 site.js（只有纯计算那部分会跑，碰网页的部分在 node 里自己跳过），执行 code，把它 print 的 JSON 读回来。"""
    script = f"const S = require({json.dumps(str(STATIC / 'site.js'))});\n{code}\n"
    got = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


def py_live_text(item, stale=False) -> str:
    """服务器 match.live_html 写出来的那一块，去掉网页标签后的字（拿来和脚本的结果比）。"""
    data = SimpleNamespace(results={}, live_scores={1: item}, live_stale=stale)
    h = match_page.live_html(data, 1)
    return html_lib.unescape(re.sub(r"<[^>]+>", "", h))


LIVE_CASES = [
    PLAYING,
    {**PLAYING, "minute": "中场"},
    {**PLAYING, "minute": "", "home_red": 0, "away_red": 0},
    {**PLAYING, "minute": 0},
    {**PLAYING, "minute": "45+2"},
    {**PLAYING, "home_goals": None},
    {**PLAYING, "home_goals": "2"},                     # 文字不算数
    {**PLAYING, "home_goals": 100},                     # 太大不算
    {**PLAYING, "home_red": True, "away_red": 1},       # True 不算数
    {**PLAYING, "state": "Played"},
    {**PLAYING, "state": "Fixture"},
    {"state": "Playing"},
]


@pytest.mark.skipif(NODE is None, reason="这台电脑上没有 node（不是项目依赖），跳过脚本逻辑的检查")
def test_js_live_parts_match_server():
    got = run_node(f"const cases = {json.dumps(LIVE_CASES, ensure_ascii=False)};\n"
                   "console.log(JSON.stringify(cases.map(c => S.liveParts(c, {}).map(p => p.text).join(' '))));")
    for item, text in zip(LIVE_CASES, got):
        assert text == py_live_text(item), item
    flags = run_node(f"const c = {json.dumps(PLAYING)};\n"
                     "console.log(JSON.stringify([S.liveParts(c, {paused: true}).map(p => p.text),"
                     " S.liveParts(c, {stale: true}).map(p => p.text), S.liveParts(null, {})]));")
    assert flags[0][-1] == "实时比分暂停" and flags[1][-1] == "很久没更新" and flags[2] == []
    assert flags[0][:3] == ["进行中 63'", "2–1", "红牌 0–1"]


@pytest.mark.skipif(NODE is None, reason="这台电脑上没有 node（不是项目依赖），跳过脚本逻辑的检查")
def test_js_hit_rate_money_and_filters_match_server():
    cases = [["win", "win", "loss"], ["half_win", "push", "loss"], ["half_loss", "half_win"], ["push"], []]
    counts = [{k: c.count(k) for k in ("win", "half_win", "push", "half_loss", "loss")} for c in cases]
    money_in = [1234.5, -1234.5, 1235.5, -1235.5, 0.5, -0.5, 0, 999.4, -0.4, 1e6, 12.6]
    got = run_node(
        f"const counts = {json.dumps(counts)}; const money = {json.dumps(money_in)};\n"
        "console.log(JSON.stringify({hit: counts.map(S.hitRate), money: money.map(v => S.money(v)),"
        " signed: [S.money(12, true), S.money(-12, true), S.money(0, true)],"
        " groups: ['win','loss','half_win','push','half_loss','none','x'].map(S.outcomeGroup),"
        " bj: [S.bjText('2026-11-07T10:00:00+00:00'), S.bjText('2026-12-31T20:30:00+00:00'), S.bjText('坏的')],"
        " diff: [S.diffText(0.45, 0.42), S.diffText(0.40, 0.42), S.diffText(0.4, 0.4002), S.diffText(0.4, null)],"
        " probs: [S.parseProbs('0.412,0.270,0.318'), S.parseProbs('1,2'), S.parseProbs('a,b,c'), S.parseProbs('')],"
        " key: [S.probKey([0.4,0.3,0.3],'home'), S.probKey([0.2,0.3,0.5],'max'), S.probKey(null,'max')],"
        " sort: [3, null, 1, 2].sort((a, b) => S.compareValues(a, b, true)),"
        " sortAsc: ['b', '', 'a'].sort((a, b) => S.compareValues(a, b, false)),"
        " match: [S.rowMatches({league:'E0',pred:'1',level:'3',text:'利物浦 对 曼城 liverpool man city'}, {league:'E0',pred:'1',level:'3',q:'利物浦  曼城'}),"
        "         S.rowMatches({league:'E0',pred:'1',level:'3',text:'利物浦 liverpool'}, {q:'LIVER'}),"
        "         S.rowMatches({league:'E0',pred:'1',level:'',text:'x'}, {level:'none'}),"
        "         S.rowMatches({league:'E0',pred:'1',level:'3',text:'x'}, {level:'none'}),"
        "         S.rowMatches({league:'SP1',pred:'0',level:'',text:'x'}, {league:'E0'}),"
        "         S.rowMatches({league:'E0',pred:'0',level:'',text:'x'}, {pred:'1'})]}));")
    for c, h in zip(cases, got["hit"]):
        want = pk.hit_rate(c)
        assert (h is None and want is None) or abs(h - want) < 1e-12, c
    assert got["money"] == [bots.money(v) for v in money_in]
    assert got["signed"] == ["+12", "−12", "0"]
    assert got["groups"] == ["hit", "miss", "part", "part", "part", "none", "none"]
    assert got["bj"] == ["11-07 18:00", "01-01 04:30", ""]
    assert got["diff"] == ["↑ 3.0 个百分点", "↓ 2.0 个百分点", "没变", ""]
    assert got["probs"] == [[0.412, 0.27, 0.318], None, None, None]
    assert got["key"] == [0.4, 0.5, None]
    assert got["sort"] == [3, 2, 1, None] and got["sortAsc"] == ["a", "b", ""]      # 空的永远在最后
    assert got["match"] == [True, True, True, False, False, False]
    assert run_node("console.log(JSON.stringify(S.LEVEL_NAMES))") == {str(k): v for k, v in LEVEL_NAMES.items()}
    assert run_node("console.log(JSON.stringify(S.OUTCOME_TEXT))") == pk.OUTCOME_TEXT


@pytest.mark.skipif(NODE is None, reason="这台电脑上没有 node（不是项目依赖），跳过脚本逻辑的检查")
def test_js_pick_versions_from_real_api(world):
    """拿真的 /api/match/1（临时数据库）给脚本挑版本：同一个模型的 T-24、T-6、T-1、预览版都在，最新的是 T-1。"""
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        status, body = srv.req("GET", "/api/match/1", srv.cookie())
    api = json.loads(body.decode("utf-8"))
    got = run_node(f"const api = {json.dumps(api, ensure_ascii=False)};\n"
                   "const r = S.pickVersions(api);\n"
                   "console.log(JSON.stringify({kinds: r.versions.map(v => v.kind), latest: r.latestIndex,"
                   " labels: S.versionLabels(r.versions), none: S.pickVersions({versions: []})}));")
    kinds = [v["kind"] for v in api["versions"]]
    assert got["kinds"] == kinds and kinds[got["latest"]] == api["latest"]["kind"] == "T-1"
    assert got["labels"][:3] == ["开球前 24 小时", "开球前 6 小时", "开球前 1 小时"] and got["none"] is None
    other = {**api, "versions": api["versions"] + [{**api["versions"][0], "model": "另一个模型"}]}
    got = run_node(f"const api = {json.dumps(other, ensure_ascii=False)};\n"
                   "console.log(JSON.stringify(S.pickVersions(api).versions.map(v => v.model)));")
    assert "另一个模型" not in got                                               # 不同模型的版本不混在一起比
