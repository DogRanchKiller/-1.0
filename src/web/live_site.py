"""网站“现读现出”：登录后打开网页时，服务器现场从数据库读数据、用 src/web/site 里同一套生成函数做出网页（需求文档第 8 节）。

和“足球预测_生成网站”（src/jobs/build_site.py，每 15 分钟写一次静态文件）用的是同一批函数，所以页面内容一样，
区别只是“什么时候生成”：静态文件是定时做好放着，这里是有人打开时才做。

怎么做（给不会编程的人看）：
- 读一次数据库要好几秒，所以读出来的数据（SiteData）在内存里留一会儿（live.cache_seconds，默认 60 秒）。
  这段时间里打开网页都用内存里的，不再读数据库；生成一页只要零点几秒。
- 数据库文件、data/review/（复盘）、虚拟投注结果文件夹、比分预测的历史参考文件一改（看文件的“修改时间”和大小），
  不等 cache_seconds 就重新读；但两次读之间至少隔 live.min_reload_seconds 秒（默认 60）：
  比赛期间即时比分程序每两分钟往数据库写一次抓取记录，数据库的修改时间老在变，不能每次都整个重读。
- 几个人同时打开时只读一次（一把锁）：抢到锁的那个去读，别的请求先用内存里旧的那份；
  内存里一份都没有时排队等它读完，最多等 live.wait_seconds 秒（默认 10），等不到就先发静态文件。
- 读失败（数据库被别的程序锁着、文件坏了、任何意外）：继续用内存里上一次读成功的那份，日志里记原因（不记路径和密码）；
  一份都没有时返回 None，服务器改发 data/site/ 里的静态文件（和以前一样）。失败以后隔 live.retry_seconds 秒才再试。
- 数据库只用只读方式打开（mode=ro），这里不写任何东西。

即时比分：scores.json（settings.yaml 的 dongqiudi.live.output，没写时是 data/live/scores.json）是 WD3 的程序写的（懂球帝即时比分），
这里只读、原样转发给 /api/live；页面上显示时，用数据库 match_sources 表（懂球帝导入时记的“懂球帝比赛编号 → 我们的 match_id”）
对上我们的比赛，对不上的不显示。

几个词：
- 指纹（fingerprint）：把一串文件的“修改时间 + 大小”算成一小段文字。哪个文件改了，指纹就变，用来判断“该不该重新读”。
- 数据版本（data_version）：把读出来的内容本身算成一小段文字（内容哈希）。重新读了但内容没变，它就不变，
  前端看它没变就不用刷新（数据库只是多了几行抓取记录时就是这样）。
- API（数据接口）：网址返回的不是网页，是 JSON 格式的数据，给网页里的脚本（/static/site.js）读。
"""
from __future__ import annotations

import dataclasses
import hashlib
import html as _html
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

import pandas as pd

from src.common import markets as site_markets
from src.common.config import resolve_path
from src.features import open_readonly
from src.web.site import bots, lifecycle, match, pages
from src.web.site import data as site_data
from src.web.site import picks as pk
from src.web.site.common import LiveMinutes, TeamFiles, utc_iso
from src.web.site.match import final_probs, leans, level, uncertain, version_matrices, version_name

LOG = logging.getLogger("site_server")
# settings.yaml 里没写 live: 时用的数（写了以 settings.yaml 为准）
DEFAULTS = {"dynamic": True, "meta_refresh": True, "poll_seconds": 30, "cache_seconds": 60, "min_reload_seconds": 60, "wait_seconds": 10,
            "stale_seconds": 600, "retry_seconds": 15, "live_poller": False, "poller_stop_seconds": 5}
# 即时比分文件的位置写在 settings.yaml 的 dongqiudi.live.output（WD3 的程序也读这一项，两边只认这一个地方）；没写时用这个
DEFAULT_SCORES_FILE = "data/live/scores.json"
LIFECYCLE_CLOCK_KEYS = ("generated_at_utc", "evaluation_cutoff_utc")   # 逐场验收报告里“现在几点”的两项，算数据版本时不算
LIFECYCLE_SCOPE_CLOCK_KEYS = ("start_utc", "end_utc")                  # 报告 scope 里按“现在”推出来的窗口，也不算
MAX_LIVE_DEPTH = 8                 # 即时比分文件最多套几层（正常是 3 层：最外面 → matches 列表 → 每场一个）；套得太深当坏文件
# 懂球帝导入时在 match_sources 表里记的来源名（src/clean/dongqiudi_import.py、dongqiudi_kor_import.py），source_key = 懂球帝比赛编号
DQD_SOURCES = ("dongqiudi", "dongqiudi-kor")
MAX_LIVE_BYTES = 2_000_000          # 即时比分文件最大读多少（正常只有几十 KB；太大说明出问题了，不读）
LIVE_STATUSES = ("ok", "paused", "off")   # WD3 约定的状态；别的写法当作 "unknown"
MATCH_PAGE = re.compile(r"/match/(\d{1,12})\.html")
TEAM_PAGE = re.compile(r"/team/([^/]+\.html)")


def settings_live(settings: dict) -> dict:
    """settings.yaml 的 live: 一节，缺的项用 DEFAULTS 补上。"""
    return {**DEFAULTS, **(settings.get("live") or {})}


def scores_path(settings: dict) -> Path:
    """即时比分文件 scores.json 在哪：settings.yaml 的 dongqiudi.live.output（WD3 加的那一项）；没写时 data/live/scores.json。"""
    out = ((settings.get("dongqiudi") or {}).get("live") or {}).get("output")
    return resolve_path(out if isinstance(out, str) and out else DEFAULT_SCORES_FILE)


def _canon(x):
    """把读出来的数据变成固定写法的一串（算数据版本用）：字典按键排好、表格按行列写出来，同样的内容永远写成同样的一串。"""
    if isinstance(x, pd.DataFrame):
        return ["df", [str(c) for c in x.columns], [[_canon(v) for v in row] for row in x.itertuples(index=False)]]
    if isinstance(x, dict):
        return ["dict", sorted(([repr(_canon(k)), _canon(v)] for k, v in x.items()), key=lambda kv: kv[0])]
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return ["set", sorted(repr(_canon(v)) for v in x)]
    if isinstance(x, Path):
        return ["path", x.name]                    # 只用文件名（完整路径和内容无关，换电脑也一样）
    if isinstance(x, float) and not math.isfinite(x):
        return ["float", repr(x)]                  # nan 和 nan 不相等，写成文字才比得了
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return ["obj", type(x).__name__, str(x)]       # 时间、numpy 数字等


def content_version(data: site_data.SiteData, extra) -> str:
    """数据版本：读出来的内容（不算“现在几点”和设置）+ extra（虚拟投注结果、比分参考文件的修改时间，这两样页面上直接用文件）
    算成 16 位文字。内容没变，重新读多少次它都一样。"""
    parts = {f.name: getattr(data, f.name) for f in dataclasses.fields(data)
             if f.name not in ("now", "settings", "live_scores", "live_stale")}
    # 逐场验收报告里记着“什么时候做的”和按“现在”往前往后推的窗口（每次读都不一样），这几项不算内容
    report = {k: v for k, v in (data.lifecycle or {}).items() if k not in LIFECYCLE_CLOCK_KEYS}
    if isinstance(report.get("scope"), dict):
        report["scope"] = {k: v for k, v in report["scope"].items() if k not in LIFECYCLE_SCOPE_CLOCK_KEYS}
    parts["lifecycle"] = report
    text = json.dumps(_canon([parts, extra]), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _too_deep(obj, limit: int) -> bool:
    """JSON 读出来的东西是不是套了超过 limit 层（一层层往下数，不用递归，套得再深也不会出错）。"""
    stack = [(obj, 1)]
    while stack:
        cur, depth = stack.pop()
        if isinstance(cur, (dict, list)):
            if depth > limit:
                return True
            stack.extend((v, depth + 1) for v in (cur.values() if isinstance(cur, dict) else cur))
    return False


def _finite_float(text: str) -> float:
    """读 JSON 里的小数：1e999 这种会变成无穷大，当作文件坏了。"""
    v = float(text)
    if not math.isfinite(v):
        raise ValueError("JSON 里的数太大")
    return v


@dataclasses.dataclass
class Snapshot:
    """内存里的一份数据：一次读数据库得到的全部东西，加上几样提前做好的（每次请求都要用、做一次就够）。"""
    data: site_data.SiteData                 # site_data.load 读出来的
    teams: TeamFiles                         # 每支球队的页面文件名
    team_by_file: dict[str, str]             # 反过来：页面文件名 → 球队
    bots_html: str                           # 虚拟投注页（要读好几个结果文件，提前做好）
    dqd: dict[str, int]                      # 懂球帝比赛编号（文字）→ 我们的 match_id，只留这次页面里有的比赛
    version: str                             # 数据版本（内容哈希，见 content_version），/api/version 返回它
    stamp: str                               # 读之前那一刻的指纹（文件的修改时间和大小），判断“文件变了没有”用
    loaded_at: float                         # 什么时候读的（服务器时钟，秒）
    generated_at: str                        # 什么时候读的（UTC，ISO 写法）


def _clean_error(err: BaseException) -> str:
    """日志里写的错误原因：只写错误种类；数据库的错误加上一小段说明（例如 database is locked）。
    别的错误说明里常带着文件的完整路径（里面有电脑用户名），不写。"""
    kind = type(err).__name__
    if isinstance(err, sqlite3.Error):
        msg = "".join(c if c.isprintable() else "?" for c in str(err))[:80]
        return f"{kind}: {msg}" if msg else kind
    return kind


def _stamp(path: Path) -> tuple | None:
    """一个文件的（修改时间, 大小）；没有这个文件是 None。"""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _folder_stamps(folder: Path | None, label: str) -> list:
    """一个文件夹里（只看这一层）每个文件的（名字, 修改时间, 大小）。文件夹不在就是空的。"""
    if folder is None:
        return []
    out = []
    try:
        with os.scandir(folder) as it:
            for entry in it:
                try:
                    if entry.is_file():
                        st = entry.stat()
                        out.append((label, entry.name, st.st_mtime_ns, st.st_size))
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(out)


def _num(x, digits: int = 4) -> float | None:
    """数字 → 保留几位小数的普通小数（JSON 里用）。空的、不是数字的、无穷大的都是 None。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _no_constant(name: str):
    """读 JSON 时碰到 NaN、Infinity 这种（标准 JSON 里没有）就当作文件坏了。"""
    raise ValueError(f"JSON 里不能有 {name}")


def _matched(snap: Snapshot, info: dict) -> list[tuple[str, int, dict]]:
    """即时比分文件里，哪些比赛对上了我们的比赛：[(懂球帝比赛编号, 我们的 match_id, 文件里那一条), ...]。
    靠数据库 match_sources 表（懂球帝导入时记的编号）对；联赛代码也写了的，还要和我们这场的联赛一样，免得张冠李戴。"""
    out = []
    for item in info.get("matches") or []:
        if not isinstance(item, dict):
            continue
        key = item.get("dqd_id")
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            continue
        mid = snap.dqd.get(str(key))
        if mid is None:
            continue                               # 对不上我们的比赛：不显示
        league = item.get("league")
        if isinstance(league, str) and league and league != snap.data.matches[mid]["competition"]:
            continue
        out.append((str(key), mid, item))
    return out


def load_from_db(settings: dict, db_path: Path, now: datetime) -> tuple[site_data.SiteData, dict[str, int]]:
    """读一次数据库（只读方式）：和生成静态网站完全一样的 site_data.load，另外读“懂球帝比赛编号 → 我们的 match_id”。
    读完马上关掉数据库，后面生成网页只用内存里的数据。"""
    con = open_readonly(db_path)
    try:
        data = site_data.load(con, settings, now)
        try:
            marks = ",".join("?" * len(DQD_SOURCES))
            rows = con.execute(f"select source_key, match_id from match_sources where source in ({marks})",
                               DQD_SOURCES).fetchall()
        except sqlite3.OperationalError:
            rows = []                      # 很老的数据库没有这张表：即时比分对不上，页面照常
    finally:
        con.close()
    dqd = {str(k): int(m) for k, m in rows if int(m) in data.matches}
    return data, dqd


def with_version(html: str, version: str) -> str:
    """在网页 <head> 最后加一行 <meta name="site-data-version" content="数据版本">（和 /api/version 的 data_version 一样）。
    site.js 拿它当“这一页是哪一版数据做的”：之后问 /api/version，版本不一样才去换内容（不会因为刚打开时的先后差一点就白换一次）。
    版本只有 16 位字母数字，也再转义一次；找不到 </head> 就原样返回。"""
    i = html.find("</head>")
    if i < 0:
        return html
    tag = f'<meta name="site-data-version" content="{_html.escape(str(version))}">'
    return html[:i] + tag + html[i:]


class LiveSite:
    """服务器里的“现读现出”。一个服务器一个，几个请求线程一起用（内部有锁）。

    settings：完整的 settings.yaml；site_dir：静态网站文件夹（战绩页算周报链接用，和静态网站一样）。
    clock：服务器时钟（秒），算“过了几秒”用；测试里换成假时钟。now：生成网页时的“现在”（UTC）；测试里给固定时间。
    loader：读数据的函数，默认 load_from_db；测试里换成假的（例如故意出错）。
    """

    def __init__(self, settings: dict, site_dir: Path, clock: Callable[[], float] = time.time,
                 now: Callable[[], datetime] | None = None, loader: Callable | None = None):
        live = settings_live(settings)
        site = settings["site"]
        # 现读现出的页面加载 /static/site.js（src/web/site/common.py 的 LiveMinutes）。live.meta_refresh 是 true 时
        # 另外每 site.refresh_minutes 分钟整页重新加载；false（现在的设置）时不带，靠 site.js 每 live.poll_seconds 秒问一次服务器、
        # 数据变了才换内容
        minutes = int(site["refresh_minutes"]) if live["meta_refresh"] else 0
        poll = max(5, int(live["poll_seconds"]))          # 最少 5 秒，免得写错成 0 时脚本一直问
        self.settings = {**settings, "site": {**site, "refresh_minutes": LiveMinutes(minutes, poll)}}
        self.site_dir = Path(site_dir)
        self.db_path = resolve_path(settings["paths"]["database"])
        self.review_dir = resolve_path(site["review_dir"])
        self.bots_dir = resolve_path(site["bots_dir"]) if site.get("bots_dir") else None
        self.picks_file = pk.reference_path(settings) if settings.get("picks") else None
        self.scores_file = scores_path(settings)
        self.cache_seconds = float(live["cache_seconds"])
        self.min_reload_seconds = float(live["min_reload_seconds"])
        self.wait_seconds = float(live["wait_seconds"])
        self.stale_seconds = float(live["stale_seconds"])
        self.retry_seconds = float(live["retry_seconds"])
        self.clock = clock
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.loader = loader or load_from_db
        self._snap: Snapshot | None = None
        self._reload_lock = threading.Lock()      # 同一时间只有一个请求去读数据库
        self._retry_at = 0.0                      # 上次读失败：到这个时间以前不再试
        self._live_lock = threading.Lock()
        self._live_cache: tuple | None = None     # (文件的修改时间和大小, 读出来的内容)：文件没变就不重新读

    # ----- 指纹：这些文件变了，就该重新读 -----
    def fingerprint(self) -> str:
        """数据库（和它旁边的 -wal 文件）、复盘文件夹、虚拟投注结果、比分预测历史参考的“修改时间 + 大小”算成的 16 位文字。"""
        parts: list = [("db", _stamp(self.db_path)), ("wal", _stamp(Path(f"{self.db_path}-wal"))),
                       ("picks", _stamp(self.picks_file) if self.picks_file else None)]
        parts += _folder_stamps(self.review_dir, "review")
        parts += _folder_stamps(self.review_dir / "matches", "review/matches")
        parts += _folder_stamps(self.review_dir / "reports", "review/reports")
        parts += _folder_stamps(self.bots_dir, "bots")
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]

    # ----- 取内存里的数据 -----
    def _fresh(self, snap: Snapshot) -> bool:
        """内存里这份还能不能直接用：
        读了不到 min_reload_seconds 秒 → 能用（文件变了也先不读，免得数据库一直在写时每次都整个重读）；
        不到 cache_seconds 秒、文件也都没变 → 能用；别的情况该重新读了。"""
        age = self.clock() - snap.loaded_at
        if age < self.min_reload_seconds:
            return True
        return age < self.cache_seconds and self.fingerprint() == snap.stamp

    def snapshot(self) -> Snapshot | None:
        """现在该用的那份数据；一份都拿不到时 None（服务器改发静态文件）。"""
        snap = self._snap
        if snap is not None and (self._fresh(snap) or self.clock() < self._retry_at):
            return snap                            # 还新鲜；或者刚读失败过，先用旧的，过一会儿再试
        if snap is not None:
            # 有旧的：抢得到锁就自己去读，抢不到（别人正在读）就先用旧的，不排队
            if not self._reload_lock.acquire(blocking=False):
                return snap
        else:
            if self.clock() < self._retry_at:
                return None
            # 一份都没有：排队等正在读的那个读完，最多等 wait_seconds 秒；等不到就先发静态文件（接口回 503）
            if not self._reload_lock.acquire(timeout=self.wait_seconds):
                return None
        try:
            cur = self._snap
            if cur is not None and cur is not snap:
                return cur                         # 排队的时候别人已经读好了
            if self.clock() < self._retry_at:
                return cur                         # 排队的时候别人刚读失败：不马上再试
            return self._reload(cur)
        finally:
            self._reload_lock.release()

    def _reload(self, old: Snapshot | None) -> Snapshot | None:
        """真的去读一次。失败时返回旧的那份（可能是 None），并且 retry_seconds 秒内不再试。"""
        stamp = self.fingerprint()                 # 先算指纹再读：读的时候文件又变了，下一次请求还会再读
        started = self.clock()
        try:
            snap = self._build(stamp)
        except Exception as err:   # noqa: BLE001 —— 故意的：任何错误都不能让网站打不开，继续用旧数据或静态文件
            self._retry_at = self.clock() + self.retry_seconds
            LOG.warning("现读现出：读数据失败（%s），%s", _clean_error(err),
                        "继续用上一次读到的数据" if old is not None else "先发静态文件")
            return old
        self._snap, self._retry_at = snap, 0.0
        same = old is not None and old.version == snap.version
        LOG.info("现读现出：重新读了数据（%d 场比赛，用了 %.1f 秒%s）", len(snap.data.page_ids), self.clock() - started,
                 "，内容没变" if same else "")
        return snap

    def _build(self, stamp: str) -> Snapshot:
        now = self.now()
        data, dqd = self.loader(self.settings, self.db_path, now)
        bots_html, bots_error = bots.render_checked(data)
        if bots_error:      # 和生成静态网站一样（src/jobs/build_site.py write_site）：提示排在首页提示的最前面
            data.notes.insert(0, f"虚拟投注页出错（{bots_error}），这一页先换成占位页；别的页照常。")
        teams = TeamFiles({data.matches[mid][k] for mid in data.page_ids for k in ("home_team", "away_team")})
        by_file = {teams.file(t): t for t in teams.names}
        # 虚拟投注页、比分参考是直接读文件做的（不在 data 里），把这两样文件的修改时间也算进数据版本
        extra = [_folder_stamps(self.bots_dir, "bots"), _stamp(self.picks_file) if self.picks_file else None,
                 sorted(dqd.items()), bots_error]
        return Snapshot(data=data, teams=teams, team_by_file=by_file, bots_html=bots_html, dqd=dqd,
                        version=content_version(data, extra), stamp=stamp,
                        loaded_at=self.clock(), generated_at=now.astimezone(timezone.utc).isoformat(timespec="seconds"))

    # ----- 网页 -----
    def page(self, route: str) -> tuple[int, str, str] | None:
        """现场生成 route 这一页：返回 (状态码, 内容, 文件类型)。
        返回 None = 这一页不归这里管（例如 style.css），或者现在拿不到数据、生成出错 → 服务器改发静态文件。
        状态码 404 = 网址格式对、但没有这场比赛 / 这支球队（内容是空的，服务器发“找不到这一页”）。"""
        snap = self.snapshot()
        if snap is None:
            return None
        try:
            got = self._page(snap, route)
        except Exception as err:   # noqa: BLE001 —— 故意的：这一页生成出错，改发静态文件，不让网站打不开
            from src.web.serve import _clean_for_log   # 放在这里才导入：serve.py 也要导入这个文件，免得互相导入
            LOG.warning("现读现出：生成 %s 出错（%s），改发静态文件", _clean_for_log(route), _clean_error(err))
            return None
        if got is not None and got[0] == 200 and got[2].startswith("text/html"):
            got = (got[0], with_version(got[1], snap.version), got[2])
        return got

    def _page(self, snap: Snapshot, route: str) -> tuple[int, str, str] | None:
        html_type = "text/html; charset=utf-8"
        if route in ("/", "/index.html"):
            return 200, pages.render_index(self._with_live(snap), snap.teams), html_type
        if route == "/record.html":
            return 200, pages.render_record(snap.data, self.site_dir), html_type
        if route == "/lifecycle.html":
            return 200, lifecycle.render(snap.data), html_type
        if route == "/lifecycle.json":
            return (200, json.dumps(snap.data.lifecycle, ensure_ascii=False, indent=2, allow_nan=False),
                    "application/json; charset=utf-8")
        if route == "/bots.html":
            return 200, snap.bots_html, html_type
        m = MATCH_PAGE.fullmatch(route)
        if m:
            mid = int(m.group(1))
            if mid not in snap.data.page_ids:
                return 404, "", html_type
            return 200, match.render(self._with_live(snap), mid, snap.teams), html_type
        m = TEAM_PAGE.fullmatch(route)
        if m:
            team = snap.team_by_file.get(unquote(m.group(1)))
            if team is None:
                return 404, "", html_type
            return 200, pages.render_team(self._with_live(snap), team, snap.teams), html_type
        return None

    # ----- 即时比分 -----
    def _with_live(self, snap: Snapshot) -> site_data.SiteData:
        """给这一次请求用的数据：内存里那份的浅拷贝，加上对上了的即时比分（内存里那份本身不改，几个请求同时用不会互相影响）。"""
        info = self.live_info()
        if info.get("status") in (None, "off"):
            return snap.data
        found = {mid: item for _key, mid, item in _matched(snap, info)}
        if not found:
            return snap.data
        return dataclasses.replace(snap.data, live_scores=found,
                                   live_stale=bool(info.get("stale")) or info.get("status") == "paused")

    def live_info(self) -> dict:
        """/api/live 返回的内容。data/live/scores.json 不在：{"matches": [], "status": "off"}。
        在：matches 原样转发，另外加 file_time（文件最后修改的时间，UTC）、age_seconds（多少秒以前）、
        stale（超过 live.stale_seconds 没更新 = true）。文件坏了：status 写 "error"，matches 是空的。"""
        st = _stamp(self.scores_file)
        if st is None:
            return {"matches": [], "status": "off"}
        with self._live_lock:
            if self._live_cache is None or self._live_cache[0] != st:
                try:
                    body = self._read_live(st)
                except Exception as err:   # noqa: BLE001 —— 故意的：比分文件怎么坏都只算“读不了”，页面照样现读现出
                    LOG.warning("即时比分文件读不了（%s）", type(err).__name__)
                    body = {"matches": [], "status": "error", "note": "即时比分文件读不了"}
                self._live_cache = (st, body)
            body = self._live_cache[1]
        age = max(0.0, time.time() - st[0] / 1e9)        # 用电脑真实的时间比，文件的修改时间也是真实时间
        file_time = datetime.fromtimestamp(st[0] / 1e9, timezone.utc).isoformat(timespec="seconds")
        return {**body, "file_time": file_time, "age_seconds": int(age),
                "stale": bool(body.get("status") == "error" or age > self.stale_seconds)}

    def _read_live(self, st: tuple) -> dict:
        """读 scores.json。只取 WD3 约定的几项，matches 原样不改。"""
        def bad(why: str) -> dict:
            LOG.warning("即时比分文件读不了（%s）", why)
            return {"matches": [], "status": "error", "note": "即时比分文件读不了"}

        if st[1] > MAX_LIVE_BYTES:
            return bad("文件太大")
        try:
            raw = json.loads(self.scores_file.read_bytes().decode("utf-8-sig"),
                             parse_constant=_no_constant, parse_float=_finite_float)
        except (Exception, RecursionError) as err:   # noqa: BLE001 —— 读不出来、不是 JSON、套得太深、有 NaN / 无穷大，都当坏文件
            return bad(type(err).__name__)
        if not isinstance(raw, dict):
            return bad("格式不对")
        if _too_deep(raw, MAX_LIVE_DEPTH):
            return bad("套得太深")
        matches = raw.get("matches")
        status = raw.get("status")
        note = raw.get("note")
        return {"matches": matches if isinstance(matches, list) else [],
                "status": status if status in LIVE_STATUSES else "unknown",
                "note": note[:200] if isinstance(note, str) else "",
                "updated_at": raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None}

    def live_api(self) -> dict:
        """/api/live：live_info 再加一项 match_ids = {"懂球帝比赛编号": 我们的 match_id}（只有对上了的，前端按它找页面上的那一行）。"""
        info = self.live_info()
        snap = self.snapshot()                     # 和别的接口一样：内存里的那份（过期了按规则重新读）
        # 文件不在、对不上、还没有数据时都是 {}（match_ids 这一项一直都有）
        ids = {key: mid for key, mid, _item in _matched(snap, info)} if snap is not None else {}
        return {**info, "match_ids": ids}

    # ----- 数据接口（JSON）：只放页面上本来就显示的东西，不放路径、密码 -----
    def version_info(self) -> dict:
        """/api/version：{"data_version": 数据版本, "generated_at": 这份数据什么时候读的（UTC）, "mode": live / stale / static}。
        mode：live = 现读现出；stale = 刚才读失败了、还在用上一次的数据；static = 拿不到数据，网页是静态文件
        （这时 data_version 按静态首页文件的修改时间算，“生成网站”换了新文件它也会变）。"""
        snap = self.snapshot()
        if snap is not None:
            return {"data_version": snap.version, "generated_at": snap.generated_at,
                    "mode": "stale" if self._retry_at > self.clock() else "live"}
        st = _stamp(self.site_dir / "index.html")
        if st is None:
            return {"data_version": "none", "generated_at": None, "mode": "static"}
        return {"data_version": "static-" + hashlib.sha256(repr(st).encode()).hexdigest()[:12],
                "generated_at": datetime.fromtimestamp(st[0] / 1e9, timezone.utc).isoformat(timespec="seconds"),
                "mode": "static"}

    def matches_api(self) -> dict | None:
        """/api/matches：首页上的两张表（还没踢的、最近踢完的），每场一条（字段见 _head 和 _latest）。拿不到数据时 None。"""
        snap = self.snapshot()
        if snap is None:
            return None
        data = snap.data
        upcoming, recent = pages.index_lists(data)
        return {"data_version": snap.version, "generated_at": snap.generated_at,
                "upcoming": [{**self._head(data, m["match_id"]), "latest": self._latest(data, m["match_id"])} for m in upcoming],
                "recent": [{**self._head(data, m["match_id"]), "latest": self._latest(data, m["match_id"])} for m in recent]}

    def match_api(self, mid: int) -> tuple[int, dict]:
        """/api/match/<match_id>：一场比赛单场页上的预测数字。返回 (状态码, 内容)：200；404 = 没有这场的单场页；503 = 拿不到数据。"""
        snap = self.snapshot()
        if snap is None:
            return 503, {"error": "data_unavailable"}
        data = snap.data
        if mid not in data.page_ids:
            return 404, {"error": "not_found"}
        rows = list((data.preds.get(mid) or {}).values())
        order = {k: i for i, k in enumerate(data.settings["site"]["versions"])}
        rows.sort(key=lambda r: (str(r.get("model")), str(r.get("model_version")), order.get(r["kind"], len(order)),
                                 str(r.get("cutoff_utc") or ""), r["prediction_id"]))
        latest = data.latest(mid)
        return 200, {"data_version": snap.version, "generated_at": snap.generated_at, **self._head(data, mid),
                     "latest": self._latest(data, mid),
                     "picks": self._picks(data, mid, latest[1]) if latest else None,
                     "versions": [self._version(data, r) for r in rows]}

    @staticmethod
    def _head(data: site_data.SiteData, mid: int) -> dict:
        """一场比赛的基本信息（首页那一行、单场页最上面写的东西）。"""
        m = data.matches[mid]
        score = data.results.get(mid)
        precise = m.get("date_precision") != "round"
        return {"match_id": int(mid), "league": m["competition"], "league_name": data.comp(m["competition"]),
                "season": int(m["season"]), "home": data.zh(m["home_team"]), "away": data.zh(m["away_team"]),
                "kickoff_utc": utc_iso(m.get("kickoff_utc")) if precise else None,
                "match_date": str(m.get("match_date") or "")[:10] or None,
                "date_precision": m.get("date_precision") or "day",
                "status": m.get("status"),
                "score": [int(score[0]), int(score[1])] if score else None,
                "result_note": data.result_notes.get(mid),
                "page": f"/match/{int(mid)}.html" if mid in data.page_ids else None}

    @staticmethod
    def _latest(data: site_data.SiteData, mid: int) -> dict | None:
        """最新一版的结论（首页“最新一版的结论”那一格）。还没有预测是 None。"""
        lt = data.latest(mid)
        if lt is None:
            return None
        kind, row = lt
        p, src = final_probs(row)
        return {"kind": kind, "kind_name": version_name(kind), "cutoff_utc": utc_iso(row.get("cutoff_utc")),
                "probs": [_num(x) for x in p], "source": "fusion" if src == "融合" else "model",
                "confidence": level(row) if isinstance(level(row), int) else None, "leans": leans(row)}

    @staticmethod
    def _version(data: site_data.SiteData, row: dict) -> dict:
        """一版存档（单场页“各版本对比”的一行 + “比分概率”里那一版最可能的几个比分）。"""
        d = row.get("detail") or {}
        p, src = final_probs(row)
        mats, _why = version_matrices(row, data.settings["site"]["matrix_sum_tol"])
        n = data.settings["site"]["top_scores"]
        late = _num(d.get("晚了几分钟"), 0)
        return {"kind": row["kind"], "kind_name": version_name(row["kind"]),
                "model": row.get("model"), "model_version": row.get("model_version"),
                "archive_status": row.get("archive_status"),
                "cutoff_utc": utc_iso(row.get("cutoff_utc")), "created_at_utc": utc_iso(row.get("created_at_utc")),
                "late_minutes": int(late) if late else 0,
                "model_probs": [_num(row.get(k)) for k in ("p_home", "p_draw", "p_away")],
                "final_probs": [_num(x) for x in p], "source": "fusion" if src == "融合" else "model",
                "confidence": level(row) if isinstance(level(row), int) else None,
                "leans": leans(row), "uncertain": uncertain(row),
                "top_scores": [{"matrix": "fusion" if name == "融合后" else "model",
                                "scores": [[h, a, _num(q)] for h, a, q in site_markets.top_scores(m, n)]}
                               for name, m in mats]}

    @staticmethod
    def _picks(data: site_data.SiteData, mid: int, row: dict) -> dict:
        """单场页“四个玩法的预测”：pick = 预测那一列，model = “模型自己”那一列。各有 1x2 / ah / ou / score 四项，做不出来的是 None。"""
        m = data.matches[mid]
        home, away = data.zh(m["home_team"]), data.zh(m["away_team"])
        score = data.results.get(mid)
        ps, mine = pk.match_picks(row, data.settings), pk.model_picks(row, data.settings)

        def one(market: str, q: dict | None) -> dict | None:
            if q is None:
                return None
            out = {"text": pk.pick_text(market, q, home, away), "p": _num(q.get("p")), "source": q.get("source"),
                   "outcome": pk.settle(market, q, score) if score else None}
            if market == "score":
                out["scores"] = [[int(h), int(a)] for h, a in q["scores"]]
            else:
                out["side"] = int(q["side"]) if market == "1x2" else q["side"]
                if "line" in q:
                    out["line"] = _num(q["line"], 2)
            return out

        return {"pick": {k: one(k, ps[k]) for k in pk.MARKETS}, "model": {k: one(k, mine[k]) for k in pk.MARKETS}}
