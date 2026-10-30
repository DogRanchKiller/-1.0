r"""下载交叉核对用的两个开放数据集（engsoccerdata、openfootball），用来检查维基百科解析得对不对。

这两个数据集只用来核对，不导入数据库：
  - engsoccerdata（GPL 许可）：英足总杯 facup.csv、英联赛杯 leaguecup.csv、欧冠 champs.csv，年代很早，但只更新到 2010 年代末左右。
  - openfootball（CC0 公有领域）：欧冠 2011/12 起、欧联 2020/21 起、欧协联、英格兰 / 德国 / 西班牙 / 意大利杯赛的最近几个赛季。
文件都在 GitHub（raw.githubusercontent.com）上，这个网址没有 robots.txt 限制（2026-10-30 请求 /robots.txt 返回 404）。
要下哪些文件写在 settings.yaml 的 crosscheck 一节。

保存规则（和其他下载程序一样）：
  - data/raw/engsoccerdata/champs.csv、data/raw/openfootball/UCL/2018/cl.txt……
  - 文件已经在就不再下载（这些都是已经踢完的赛季，不会再变）。
  - 每次请求（包括网站没有）都记进 fetch_log 表，source = 'engsoccerdata' 或 'openfootball'。
  - 一次一个请求，间隔至少 3 秒（crosscheck.min_interval_seconds），失败按 retry_waits 越等越久再试。

运行方法：Claude 运行 `.venv\Scripts\python -m src.collect.crosscheck_sources`（约 70 个文件，4 分钟左右）。
"""
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.collect.wikipedia import retry_after
from src.common.config import PROJECT_ROOT, load_settings, resolve_path
from src.common.db import connect

log = logging.getLogger("crosscheck_sources")


def openfootball_folder(year: int) -> str:
    """openfootball 的赛季文件夹写法：2018 → '2018-19'。"""
    return f"{year}-{(year + 1) % 100:02d}"


def file_list(settings: dict) -> list[tuple[str, str, Path]]:
    """按 settings.yaml 列出要下载的文件：[(来源, 网址, 保存位置)]。"""
    c = settings["crosscheck"]
    raw = resolve_path(settings["paths"]["raw"])
    out = []
    e = c["engsoccerdata"]
    for _code, name in e["files"].items():
        out.append(("engsoccerdata", e["base_url"] + name, raw / "engsoccerdata" / name))
    o = c["openfootball"]
    for code, (repo, name, first, last) in o["files"].items():
        for y in range(first, last + 1):
            url = f"{o['base_url']}{repo}/master/{openfootball_folder(y)}/{name}"
            out.append(("openfootball", url, raw / "openfootball" / code / str(y) / name))
    return out


class Fetcher:
    """顺序下载、间隔、重试、记 fetch_log。和 wikipedia.py 的 WikiDownloader 规则一样，只是保存位置不同。"""

    def __init__(self, con, settings: dict, session=None, sleep=time.sleep, clock=time.monotonic):
        self.con, self.c = con, settings["crosscheck"]
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = self.c["user_agent"]
        self.sleep, self.clock = sleep, clock
        self._last = None
        self.counts: dict[str, int] = {}

    def _get(self, url: str):
        waits = list(self.c["retry_waits"])
        error = ""
        for attempt in range(len(waits) + 1):
            if self._last is not None:   # 离上一次请求不到 min_interval 秒就先等着
                wait = self.c["min_interval_seconds"] - (self.clock() - self._last)
                if wait > 0:
                    self.sleep(wait)
            self._last = self.clock()
            try:
                r = self.session.get(url, timeout=60)
                if r.status_code == 429:
                    error, wait = "HTTP 429", retry_after(r.headers.get("Retry-After"), waits[min(attempt, len(waits) - 1)])
                elif r.status_code >= 500:
                    error, wait = f"HTTP {r.status_code}", (waits[attempt] if attempt < len(waits) else 0)
                else:
                    return r
            except requests.RequestException as ex:
                error, wait = f"{type(ex).__name__}: {ex}", (waits[attempt] if attempt < len(waits) else 0)
            log.warning("第 %d 次请求失败（%s）：%s", attempt + 1, error, url)
            if attempt < len(waits):
                self.sleep(wait)
        return error

    def _record(self, source, url, status, http_status=None, path=None, content=None, note=None):
        rel = None
        if path is not None:
            rel = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
        self.con.execute(
            "insert into fetch_log (url, source, data_type, fetched_at_utc, status, http_status, file_path, bytes, sha256, note)"
            " values (?,?,?,?,?,?,?,?,?,?)",
            (url, source, "crosscheck_file", datetime.now(timezone.utc).isoformat(timespec="seconds"), status, http_status,
             rel, len(content) if content is not None else None,
             hashlib.sha256(content).hexdigest() if content is not None else None, note))
        self.con.commit()
        self.counts[status] = self.counts.get(status, 0) + 1

    def fetch(self, source: str, url: str, target: Path) -> str:
        if target.exists():
            return "skipped"
        last = self.con.execute("select status from fetch_log where url = ? order by id desc limit 1", (url,)).fetchone()
        if last and last[0] == "not_found":
            return "skipped"
        r = self._get(url)
        if isinstance(r, str):
            self._record(source, url, "failed", note=r)
            return "failed"
        if r.status_code == 404:
            self._record(source, url, "not_found", http_status=404)
            return "not_found"
        if r.status_code != 200:
            self._record(source, url, "failed", http_status=r.status_code)
            return "failed"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)   # 原样保存
        self._record(source, url, "saved", http_status=200, path=target, content=r.content)
        return "saved"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    con = connect()
    f = Fetcher(con, settings)
    for source, url, target in file_list(settings):
        status = f.fetch(source, url, target)
        if status != "skipped":
            log.info("%s %s", status, url)
    log.info("完成：%s", f.counts)
    con.close()


if __name__ == "__main__":
    main()
