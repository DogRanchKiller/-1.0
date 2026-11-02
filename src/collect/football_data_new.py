r"""下载 football-data.co.uk 的“新联赛”文件（日职 JPN、中超 CHN；需求文档 6.1、第 12 节第 11 阶段）。

和老联赛（五大联赛、次级联赛）有什么不同：
- 老联赛每个赛季一个文件（mmz4281/2425/E0.csv）；新联赛一个联赛只有一个文件，所有赛季都在里面：
  https://www.football-data.co.uk/new/JPN.csv、.../new/CHN.csv（2026-11-03 在网站 japan.php、china.php 页面上确认的地址）。
- 列也不一样：第一列是 Country（国家），不是 Div。列的意思见 src/clean/football_data_new_import.py 开头。
- 这个文件网站会一直更新（当前赛季每周加新比赛），所以每次下载存一份带日期的快照：
  data/raw/football-data/new/JPN/2026-11-03.csv；和上一份一模一样就不存，只在抓取记录里记“没变化”。

规则：
- football-data 的 robots.txt 点名禁止 AI 爬虫；你决定（2026-10-30）由 Claude 在你电脑上替你低频运行，只下个人学习要用的文件。
  这里只下 settings.yaml 里 football_data_new.leagues 列的两个文件和列说明 notes.txt，两次请求至少隔 5 秒。
- 遇到 403（拒绝）、429（太频繁）马上停下，不重试、不绕过。
- 每次请求都记进 fetch_log（来源 football-data，数据类型 new_league_csv，和老联赛的 league_csv 分开，
  数据状态页按 league_csv 检查老联赛，不会把新联赛文件算进去）。

运行：.venv\Scripts\python -m src.collect.football_data_new
"""
import argparse
import hashlib
import logging
from datetime import date
from pathlib import Path

from src.collect import football_data
from src.common.config import load_settings, resolve_path
from src.common.db import connect

DATA_TYPE = "new_league_csv"
NOTES_URL = f"{football_data.BASE_URL}/notes.txt"

log = logging.getLogger("football_data_new")


class Stopped(Exception):
    """网站拒绝（403）或嫌太频繁（429）：整个程序停下，不绕过、不重试。"""


def looks_like_new_csv(content: bytes) -> bool:
    """新联赛文件的第一列是 Country。网站出错时可能返回网页，那就不是这个文件。"""
    return content.lstrip(b"\xef\xbb\xbf").lstrip()[:7] == b"Country"


def new_league_url(cfg: dict, code: str) -> str:
    """一个新联赛文件的网址，格式写在 settings.yaml（football_data_new.url）。"""
    return cfg["url"].format(code=code)


class NewLeagueDownloader(football_data.Downloader):
    """借用老联赛下载程序的低频请求、抓取记录和快照保存，只改：文件内容检查、403 / 429 马上停。"""

    def fetch_new(self, url: str, data_type: str, snapshot_dir: Path, check) -> str:
        """下载一个会更新的文件，按日期存快照。check(content) 为 False 的存成“内容可疑”。返回状态。"""
        suffix = Path(url).suffix
        r = self._get(url)
        if isinstance(r, str):                       # 连不上或服务器出错 3 次
            self._record(url, data_type, "failed", note=r)
            return "failed"
        if r.status_code in (403, 429):
            self._record(url, data_type, "failed", http_status=r.status_code)
            raise Stopped(f"HTTP {r.status_code}：{url}")
        if r.status_code in (300, 404, 410):
            self._record(url, data_type, "not_found", http_status=r.status_code)
            return "not_found"
        if r.status_code != 200:
            self._record(url, data_type, "failed", http_status=r.status_code)
            return "failed"
        content = r.content
        if not check(content):
            # 内容不像这个文件：照样存下来方便查，但标成“可疑”，清洗时不用
            path = self._save_snapshot(snapshot_dir, ".suspicious.html", content)
            self._record(url, data_type, "suspicious", http_status=200, path=path, content=content)
            return "suspicious"
        if self._latest_hash(snapshot_dir, suffix) == hashlib.sha256(content).hexdigest():
            self._record(url, data_type, "unchanged", http_status=200, content=content)
            return "unchanged"
        path = self._save_snapshot(snapshot_dir, suffix, content)
        self._record(url, data_type, "saved", http_status=200, path=path, content=content)
        return "saved"

    def run_new(self, cfg: dict, today: date, with_notes: bool = True) -> dict:
        """下载 notes.txt（列说明）和 cfg["leagues"] 里每个联赛的文件。"""
        self.today = today
        out = self.raw_dir / football_data.SOURCE
        if with_notes:
            status = self.fetch_new(NOTES_URL, "notes", out / "notes", lambda c: b"Notes for Football Data" in c[:200])
            log.info("notes.txt：%s", football_data.STATUS_NAMES[status])
        for code in cfg["leagues"]:
            status = self.fetch_new(new_league_url(cfg, code), DATA_TYPE, out / cfg["raw_folder"] / code,
                                    looks_like_new_csv)
            log.info("%s：%s", code, football_data.STATUS_NAMES[status])
        return self.counts


def main():
    parser = argparse.ArgumentParser(description="下载 football-data 新联赛文件（日职、中超）")
    parser.add_argument("--no-notes", action="store_true", help="不下载 notes.txt")
    args = parser.parse_args()
    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_dir / "football_data.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    con = connect()
    d = NewLeagueDownloader(con, resolve_path(s["paths"]["raw"]), s["scrape"]["min_interval_seconds"])
    try:
        counts = d.run_new(s["football_data_new"], date.today(), with_notes=not args.no_notes)
    except Stopped as err:
        log.error("被网站拒绝，已停下（不重试）：%s", err)
        counts = d.counts
    finally:
        d.session.close()
        con.close()
    log.info("完成：%s", "，".join(f"{football_data.STATUS_NAMES.get(k, k)} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
