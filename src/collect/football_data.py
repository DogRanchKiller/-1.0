"""下载 football-data.co.uk 的历史 CSV（需求文档第 6.1、17 节）。

每个联赛每个赛季一个文件，网址格式：https://football-data.co.uk/mmz4281/2425/E0.csv
（2425 表示 2024/25 赛季，E0 是英超）。

保存规则（原始层只增不改）：
- 已经踢完的赛季：文件不会再变，下载一次存为 data/raw/football-data/E0/2425.csv，以后不再下载。
- 当前赛季：网站每周更新，每次下载存一份带日期的快照 data/raw/football-data/E0/2627/2026-10-30.csv；
  和上一份内容一样就不存，只在抓取记录里记“没变化”。
- 每次请求（包括网站没有这个文件）都记进数据库的 fetch_log 表。

运行方法：双击项目文件夹里的“下载历史数据.bat”。
这个网站的 robots.txt 不允许 AI 程序抓取，所以下载由你在自己电脑上运行，不由 Claude 运行。
"""
import argparse
import hashlib
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from src.common.config import PROJECT_ROOT, load_settings, resolve_path
from src.common.db import connect

BASE_URL = "https://football-data.co.uk"
SOURCE = "football-data"
FIRST_SEASON = 1993  # 网站最早的数据是 1993/94 赛季；某个联赛那年没有就记“网站没有”
USER_AGENT = "football-study-project/0.1 (personal, non-commercial learning)"

log = logging.getLogger("football_data")


def season_code(start_year: int) -> str:
    """2024 → '2425'，1999 → '9900'。"""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season(today: date) -> int:
    """当前赛季的开始年份：7 月起算新赛季（五大联赛 8 月开赛）。"""
    return today.year if today.month >= 7 else today.year - 1


def looks_like_csv(content: bytes) -> bool:
    """football-data 的 CSV 第一列是 Div。网站出错时可能返回网页，那就不是 CSV。"""
    return content.lstrip(b"\xef\xbb\xbf").lstrip()[:3] == b"Div"


class Downloader:
    def __init__(self, con, raw_dir: Path, min_interval: float, session=None, sleep=time.sleep, clock=time.monotonic):
        self.con = con
        self.raw_dir = raw_dir
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.sleep, self.clock = sleep, clock
        self._last_request = None
        self.counts = {}
        self.today = date.today()  # 快照文件名用的日期；run() 里会按传入的日期覆盖

    # ---------- 网络 ----------
    def _get(self, url: str):
        """低频请求：两次请求至少隔 min_interval 秒；连不上或服务器出错时等 30 秒重试，最多 3 次。"""
        for attempt in range(3):
            if self._last_request is not None:
                wait = self.min_interval - (self.clock() - self._last_request)
                if wait > 0:
                    self.sleep(wait)
            self._last_request = self.clock()
            try:
                r = self.session.get(url, timeout=60)
                if r.status_code < 500:
                    return r
                error = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                error = f"{type(e).__name__}: {e}"
            log.warning("第 %d 次请求失败（%s）：%s", attempt + 1, error, url)
            if attempt < 2:
                self.sleep(30)
        return error  # 3 次都失败，返回错误说明

    # ---------- 记录 ----------
    def _record(self, url, data_type, status, http_status=None, path=None, content=None, note=None):
        self.con.execute(
            "insert into fetch_log (url, source, data_type, fetched_at_utc, status, http_status, file_path, bytes, sha256, note)"
            " values (?,?,?,?,?,?,?,?,?,?)",
            (url, SOURCE, data_type, datetime.now(timezone.utc).isoformat(timespec="seconds"), status, http_status,
             path.relative_to(PROJECT_ROOT).as_posix() if path and path.is_relative_to(PROJECT_ROOT) else (str(path) if path else None),
             len(content) if content is not None else None,
             hashlib.sha256(content).hexdigest() if content is not None else None, note),
        )
        self.con.commit()
        self.counts[status] = self.counts.get(status, 0) + 1

    def _known_missing(self, url) -> bool:
        row = self.con.execute(
            "select status from fetch_log where url = ? order by id desc limit 1", (url,)).fetchone()
        return row is not None and row[0] == "not_found"

    def _latest_hash(self, folder: Path, suffix: str):
        files = sorted(folder.glob(f"*{suffix}")) if folder.exists() else []
        return hashlib.sha256(files[-1].read_bytes()).hexdigest() if files else None

    # ---------- 下载一个文件 ----------
    def fetch(self, url: str, data_type: str, target: Path, snapshot_dir: Path | None = None, retry_missing=False) -> str:
        """下载一个文件。target 是已完赛赛季的保存位置；snapshot_dir 不为空时按“带日期快照”保存。返回状态。"""
        suffix = Path(url).suffix  # .csv 或 .txt，快照文件沿用网址里的扩展名
        if snapshot_dir is None and target.exists():
            return "skipped"  # 已完赛的赛季已经下载过，不再请求
        if not retry_missing and self._known_missing(url):
            return "skipped"
        r = self._get(url)
        if isinstance(r, str):
            self._record(url, data_type, "failed", note=r)
            return "failed"
        if r.status_code == 404:
            self._record(url, data_type, "not_found", http_status=404)
            return "not_found"
        content = r.content
        if r.status_code != 200:
            self._record(url, data_type, "failed", http_status=r.status_code)
            return "failed"
        if data_type == "league_csv" and not looks_like_csv(content):
            # 内容不像 CSV：网站可能返回了错误页面。照样存下来方便查，但标成“可疑”，清洗时不用
            target = target.with_suffix(".suspicious.html")
            status = "suspicious"
        elif snapshot_dir is not None and self._latest_hash(snapshot_dir, suffix) == hashlib.sha256(content).hexdigest():
            self._record(url, data_type, "unchanged", http_status=200, content=content)
            return "unchanged"
        else:
            status = "saved"
        if snapshot_dir is not None and status == "saved":
            target = snapshot_dir / f"{self.today:%Y-%m-%d}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)  # 原样保存字节，不改编码、不改内容
        self._record(url, data_type, status, http_status=200, path=target, content=content)
        return status

    # ---------- 全部下载 ----------
    def run(self, leagues: list[str], today: date, retry_missing=False):
        self.today = today
        out = self.raw_dir / SOURCE
        self.fetch(f"{BASE_URL}/notes.txt", "notes", out / "notes", snapshot_dir=out / "notes")
        this_season = current_season(today)
        jobs = [(lg, y) for lg in leagues for y in range(FIRST_SEASON, this_season + 1)]
        for i, (lg, y) in enumerate(jobs, 1):
            code = season_code(y)
            url = f"{BASE_URL}/mmz4281/{code}/{lg}.csv"
            if y == this_season:
                status = self.fetch(url, "league_csv", out / lg / code, snapshot_dir=out / lg / code, retry_missing=True)
            else:
                status = self.fetch(url, "league_csv", out / lg / f"{code}.csv", retry_missing=retry_missing)
            if status != "skipped":
                log.info("[%d/%d] %s %s/%s：%s", i, len(jobs), lg, y, y + 1, STATUS_NAMES[status])
        return self.counts


STATUS_NAMES = {"saved": "已保存", "unchanged": "没变化", "not_found": "网站没有", "suspicious": "内容可疑",
                "failed": "失败", "skipped": "已有，跳过"}


def main():
    parser = argparse.ArgumentParser(description="下载 football-data.co.uk 历史数据")
    parser.add_argument("--retry-missing", action="store_true", help="以前“网站没有”的文件也重新试一次")
    args = parser.parse_args()

    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_dir / "football_data.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    leagues = list(s["leagues"]["top"]) + list(s["leagues"]["second"])
    log.info("开始下载 %d 个联赛，每次请求间隔 %s 秒。中途关掉也没关系，下次运行会接着下载。",
             len(leagues), s["scrape"]["min_interval_seconds"])
    con = connect()
    d = Downloader(con, resolve_path(s["paths"]["raw"]), s["scrape"]["min_interval_seconds"])
    counts = d.run(leagues, date.today(), retry_missing=args.retry_missing)
    con.close()
    summary = "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in counts.items()) or "没有新的请求"
    log.info("完成：%s", summary)
    if counts.get("failed") or counts.get("suspicious"):
        log.info("有失败或可疑的文件，告诉 Claude 即可，它会查看日志 logs/football_data.log")


if __name__ == "__main__":
    main()
