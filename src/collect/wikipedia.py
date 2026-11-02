r"""下载维基百科（英文）的杯赛、欧战赛季页面（需求文档 6.1、第 18 节待定问题 5 方案 a）。

为什么用 /wiki/条目名 而不用 API：
  2026-10-30 看了 https://en.wikipedia.org/robots.txt：对所有程序写的是 Disallow: /w/ 和 Disallow: /api/，
  也就是 api.php（Action API）和 REST API 都不让程序访问（只例外放行了 action=mobileview）；
  同一个文件里写着“欢迎低速、友好的程序看条目页面”。维基媒体的《Robot policy》也要求“遵守 robots.txt 的每一条”，
  并说要 HTML 时优先用 /wiki/条目名（有缓存，对网站负担最小）、不要带问号参数。
  所以这里只请求 https://en.wikipedia.org/wiki/条目名，一次一个，间隔至少 3 秒（settings.yaml 的 wikipedia.min_interval_seconds）。
  条目页面是渲染好的 HTML，嵌进来的子页面内容（{{#lst:...}}）已经展开，比赛框（footballbox）可以直接解析。

下载哪些页面：
  1. 每个赛事每个赛季的主页面，例如 “2008–09 UEFA Champions League”（标题规律在 settings.yaml）。
  2. 主页面里链接到“主页面标题 + 空格 + ……”的子页面，例如 “2008–09 UEFA Champions League group stage”，
     子页面里再链接的同类子页面也跟进（例如早年欧冠的第二阶段小组赛）。决赛单场页、名单、统计等不跟进。

保存规则（原始层只增不改，和 football_data.py 一样）：
  - 已经踢完的赛季：页面存一次，data/raw/wikipedia/UCL/2008/2008–09_UEFA_Champions_League.html.gz，以后不再下载。
  - 当前赛季：每次下载存一份带 UTC 时间和唯一标识的快照，同一天的重试也不覆盖旧文件；
    和上一份一样就不存，只记“没变化”。
  - 页面用 gzip 压缩保存（需求文档第 17 节“原始页面压缩保存”），解压后就是网站返回的原样 HTML。
  - 每次请求（包括网站没有这个页面）都记进 fetch_log 表，source = 'wikipedia'。

运行方法：Claude 运行 `.venv\Scripts\python -m src.collect.wikipedia`（全部下载一次约 40 分钟，中途关掉下次会接着下）。
  只下某几个赛事：加 `--only UCL UEL`；只下某几个赛季：加 `--seasons 2008 2009`。
内容许可：维基百科的文字是 CC BY-SA，只在本机个人学习用；以后公开展示要注明出处。
"""
import argparse
import gzip
import hashlib
import logging
import re
import time
import urllib.parse
import uuid
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from src.common.config import PROJECT_ROOT, load_settings, resolve_path
from src.common.db import connect

SOURCE = "wikipedia"
log = logging.getLogger("wikipedia")


# ---------- 标题 ----------

def season_label(year: int) -> str:
    """维基百科的赛季写法：2008 → '2008–09'，1999 → '1999–2000'（中间是 en-dash “–”，不是减号）。"""
    end = year + 1
    return f"{year}–{end % 100:02d}" if end % 100 != 0 else f"{year}–{end}"


def current_season(today: date) -> int:
    """当前赛季的开始年份：7 月起算新赛季。"""
    return today.year if today.month >= 7 else today.year - 1


def titles_for(comp: dict, year: int) -> list[str]:
    """一个赛事某个赛季可能的页面标题（改过名的赛事可能有几种，依次试）。"""
    out = []
    for first, last, pattern in comp["titles"]:
        if first <= year <= last:
            out.append(pattern.format(season=season_label(year), year=year, next_year=year + 1))
    return out


def page_url(base_url: str, title: str) -> str:
    """标题 → 网址。空格换成下划线，其他特殊字符（en-dash、é 等）按网址规则编码。"""
    return base_url + urllib.parse.quote(title.replace(" ", "_"), safe="()',!:")


def file_stem(title: str) -> str:
    """标题 → 文件名（不含扩展名）。Windows 文件名不能有 \\ / : * ? " < > |，换成下划线。"""
    return re.sub(r'[\\/:*?"<>|]', "_", title.replace(" ", "_"))


def raw_path(path: Path) -> str:
    """采集和导入共用项目根路径口径；测试或外置原始层使用绝对路径。"""
    path = path.resolve()
    root = PROJECT_ROOT.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


_LINK = re.compile(r'href="(?:https://en\.wikipedia\.org/wiki/|/wiki/|\./)([^"#?]+)(?:#[^"]*)?"')


def canonical_title(html: str) -> str | None:
    """页面自己声明的标准标题（<link rel="canonical">）。请求的标题被重定向时，以这个为准。"""
    m = re.search(r'<link rel="canonical" href="https://en\.wikipedia\.org/wiki/([^"]+)"', html)
    return urllib.parse.unquote(m.group(1)).replace("_", " ") if m else None


def find_subpages(html: str, main_title: str, skip_rules: list[str]) -> list[str]:
    """页面里链接到“main_title + 空格 + ……”的子页面标题（去重，按出现顺序）。

    不要的：网址里带问号的（例如还没建立的红链 ?action=edit），以及“……”部分符合 skip_rules 的（决赛单场页、名单等）。
    """
    out = []
    prefix = main_title + " "
    for raw in _LINK.findall(html):
        title = urllib.parse.unquote(raw).replace("_", " ")
        if not title.startswith(prefix) or title in out:
            continue
        rest = title[len(prefix):].lower()
        if any(re.search(rule, rest) for rule in skip_rules):
            continue
        out.append(title)
    return out


# ---------- 下载 ----------

class WikiDownloader:
    def __init__(self, con, raw_dir: Path, settings: dict, session=None, sleep=time.sleep, clock=time.monotonic):
        w = settings["wikipedia"]
        self.con, self.raw_dir, self.w = con, raw_dir / SOURCE, w
        self.session = session or requests.Session()
        # 描述性的 User-Agent：写明是谁、联系方式（维基媒体 User-Agent 政策要求）；要求 gzip 压缩传输（Robot policy）
        self.session.headers["User-Agent"] = w["user_agent"]
        self.session.headers["Accept-Encoding"] = "gzip"
        self.sleep, self.clock = sleep, clock
        self._last = None
        self.counts: dict[str, int] = {}
        self.today = date.today()
        self.last_fetch: dict = {}

    def _wait_interval(self):
        """顺序请求：离上一次请求不到 min_interval 秒就先等着。"""
        if self._last is not None:
            wait = self.w["min_interval_seconds"] - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()

    def _get(self, url: str):
        """请求一个网址。连不上或 5xx：按 retry_waits 越等越久再试；429（请求太多）：按网站给的 Retry-After 等。
        返回响应；全部失败返回错误说明（文字）。"""
        waits = list(self.w["retry_waits"])
        error = ""
        for attempt in range(len(waits) + 1):
            self._wait_interval()
            try:
                r = self.session.get(url, timeout=60)
                if r.status_code == 429:
                    error = "HTTP 429"
                    wait = retry_after(r.headers.get("Retry-After"), waits[min(attempt, len(waits) - 1)])
                elif r.status_code >= 500:
                    error = f"HTTP {r.status_code}"
                    wait = waits[attempt] if attempt < len(waits) else 0
                else:
                    return r
            except requests.RequestException as e:
                error = f"{type(e).__name__}: {e}"
                wait = waits[attempt] if attempt < len(waits) else 0
            log.warning("第 %d 次请求失败（%s）：%s", attempt + 1, error, url)
            if attempt < len(waits):
                self.sleep(wait)
        return error

    def _record(self, url, status, http_status=None, path=None, content=None, note=None):
        rel = raw_path(path) if path is not None else None
        when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        digest = hashlib.sha256(content).hexdigest() if content is not None else None
        cur = self.con.execute(
            "insert into fetch_log (url, source, data_type, fetched_at_utc, status, http_status, file_path, bytes, sha256, note)"
            " values (?,?,?,?,?,?,?,?,?,?)",
            (url, SOURCE, "wiki_page", when, status, http_status, rel,
             len(content) if content is not None else None, digest, note))
        self.con.commit()
        self.counts[status] = self.counts.get(status, 0) + 1
        self.last_fetch = {"url": url, "status": status, "http_status": http_status,
                           "raw_file": rel, "sha256": digest, "fetched_at_utc": when,
                           "fetch_log_id": cur.lastrowid, "note": note}

    def _last_status(self, url):
        row = self.con.execute("select status from fetch_log where url = ? order by id desc limit 1", (url,)).fetchone()
        return row[0] if row else None

    def fetch(self, title: str, folder: Path, snapshot: bool) -> tuple[str, str | None]:
        """下载一个页面。返回 (状态, 页面 HTML 文字或 None)。

        snapshot=False（已完赛赛季）：文件已经在就不再请求，直接读本地文件。
        snapshot=True（当前赛季）：每次都请求，内容变了才存新快照。
        """
        url = page_url(self.w["base_url"], title)
        target = folder / f"{file_stem(title)}.html.gz"
        snap_dir = folder / file_stem(title)
        if not snapshot and target.exists():
            return "skipped", read_page(target)
        if not snapshot and self._last_status(url) == "not_found":
            return "skipped", None
        r = self._get(url)
        if isinstance(r, str):
            self._record(url, "failed", note=r)
            return "failed", None
        if r.status_code == 404:
            self._record(url, "not_found", http_status=404)
            return "not_found", None
        if r.status_code != 200:
            self._record(url, "failed", http_status=r.status_code)
            return "failed", None
        content = r.content   # requests 已经把 gzip 传输解开，这里是网站返回的原样 HTML 字节
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex
        if b'class="mw-parser-output"' not in content and b"mw-content-text" not in content:
            target = snap_dir / f"{stamp}.suspicious.html.gz"   # 可疑响应也保留各次原件，但清洗时不用
            status = "suspicious"
        elif snapshot:
            files = sorted(p for p in snap_dir.glob("*.html.gz")
                           if not p.name.endswith(".suspicious.html.gz")) if snap_dir.exists() else []
            if files and hashlib.sha256(gzip.decompress(files[-1].read_bytes())).hexdigest() == hashlib.sha256(content).hexdigest():
                self._record(url, "unchanged", http_status=200, path=files[-1], content=content,
                             note=f"title={title}")
                return "unchanged", content.decode("utf-8")
            target = snap_dir / f"{stamp}.html.gz"
            status = "saved"
        else:
            status = "saved"
        target.parent.mkdir(parents=True, exist_ok=True)
        # mtime=0：同样的内容压缩出来的文件也一样，方便比较
        # x 模式保证任何重试、并发或意外重名都不能覆盖旧证据。
        with target.open("xb") as stream:
            stream.write(gzip.compress(content, mtime=0))
        self._record(url, status, http_status=200, path=target, content=content,
                     note=f"title={title}")
        return status, content.decode("utf-8") if status == "saved" else None

    def collect_scope(self, code: str, comp: dict, year: int, today: date) -> dict:
        """日更专用：只取当前赛季，逐页报告；complete 只表示本轮页面遍历完整。

        不用本地旧子页冒充本轮成功。主页面不存在、任一所需子页失败或达到页面上限，
        都不能授权导入；完整抓取也不保证来源列出了整季赛历。
        """
        if year != current_season(today):
            raise ValueError("杯赛日更范围只能是当前赛季")
        self.today = today
        folder = self.raw_dir / code / str(year)
        report = {"competition": code, "season": year, "status": "failed", "complete": False,
                  "calendar_complete": None, "pages": [], "errors": []}

        def fetch_page(title):
            status, html = self.fetch(title, folder, snapshot=True)
            report["pages"].append({**self.last_fetch, "title": title})
            return status, html

        main, html = None, None
        for title in titles_for(comp, year):
            _, html = fetch_page(title)
            if html is not None:
                main = canonical_title(html) or title
                break
        if main is None:
            report["errors"].append("主页面没有取得有效内容")
            return report
        skip = list(self.w["skip_subpages"]) + list(comp.get("skip_subpages_extra", []))
        queue, seen = find_subpages(html, main, skip), {main, title}
        attempted_subpages = 0
        while queue:
            sub = queue.pop(0)
            if sub in seen:
                continue
            if attempted_subpages >= int(self.w["max_subpages"]):
                report["errors"].append("达到子页面上限，仍有页面未检查")
                break
            seen.add(sub)
            attempted_subpages += 1
            status, sub_html = fetch_page(sub)
            if sub_html is None:
                report["errors"].append(f"子页面未完成：{sub} ({status})")
                continue
            queue.extend(t for t in find_subpages(sub_html, main, skip) if t not in seen)
        report["complete"] = not report["errors"]
        report["status"] = "complete" if report["complete"] else "partial"
        return report

    def run_competition(self, code: str, comp: dict, year: int, this_season: int) -> int:
        """下载一个赛事一个赛季：主页面 + 子页面。返回下载到（或本地已有）的页面数。"""
        folder = self.raw_dir / code / str(year)
        snapshot = year >= this_season
        skip = list(self.w["skip_subpages"]) + list(comp.get("skip_subpages_extra", []))
        for title in titles_for(comp, year):
            status, html = self.fetch(title, folder, snapshot)
            if html is None:
                continue
            main = canonical_title(html) or title
            pages, queue, seen = 1, find_subpages(html, main, skip), {title, main}
            while queue and pages <= self.w["max_subpages"]:
                sub = queue.pop(0)
                if sub in seen:
                    continue
                seen.add(sub)
                _, sub_html = self.fetch(sub, folder, snapshot)
                if sub_html is None:
                    continue
                pages += 1
                queue += [t for t in find_subpages(sub_html, main, skip) if t not in seen]
            return pages
        return 0

    def run(self, settings: dict, today: date, only: list[str] | None = None, seasons: list[int] | None = None):
        self.today = today
        this = current_season(today)
        comps = settings["wikipedia"]["competitions"]
        if only is not None and (not only or any(c not in comps for c in only)):
            raise ValueError("赛事范围不能为空或包含未知赛事")
        if seasons is not None and not seasons:
            raise ValueError("赛季范围不能为空")
        years = seasons if seasons is not None else list(range(settings["wikipedia"]["first_season"], this + 1))
        for code, comp in comps.items():
            if only and code not in only:
                continue
            for y in years:
                if not titles_for(comp, y):
                    continue
                n = self.run_competition(code, comp, y, this)
                log.info("%s %s：%s", code, season_label(y), f"{n} 个页面" if n else "没有页面")
        return self.counts


def retry_after(value, default: float) -> float:
    """429 回复里的 Retry-After：可能是秒数，也可能是一个时间。认不出就用 default。"""
    if not value:
        return default
    try:
        return max(float(value), 1.0)
    except ValueError:
        try:
            return max((parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds(), 1.0)
        except (TypeError, ValueError):
            return default


def read_page(path: Path) -> str:
    """读一个保存的页面（.html.gz）→ HTML 文字。"""
    return gzip.decompress(path.read_bytes()).decode("utf-8")


STATUS_NAMES = {"saved": "已保存", "unchanged": "没变化", "not_found": "网站没有", "suspicious": "内容可疑",
                "failed": "失败", "skipped": "已有，跳过"}


def main():
    parser = argparse.ArgumentParser(description="下载维基百科的杯赛、欧战赛季页面")
    parser.add_argument("--only", nargs="*", help="只下这些赛事代码（例如 UCL UEL），不写就是全部")
    parser.add_argument("--seasons", nargs="*", type=int, help="只下这些赛季（开始年份），不写就是全部")
    args = parser.parse_args()
    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_dir / "wikipedia.log", encoding="utf-8"), logging.StreamHandler()])
    log.info("开始下载维基百科页面，每次请求间隔 %s 秒。中途关掉也没关系，下次运行会接着下载。",
             s["wikipedia"]["min_interval_seconds"])
    con = connect()
    d = WikiDownloader(con, resolve_path(s["paths"]["raw"]), s)
    counts = d.run(s, date.today(), only=args.only, seasons=args.seasons)
    con.close()
    log.info("完成：%s", "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in counts.items()) or "没有新的请求")


if __name__ == "__main__":
    main()
