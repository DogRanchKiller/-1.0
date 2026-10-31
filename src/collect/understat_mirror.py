r"""下载 GitHub 上别人整理的 Understat 射门明细（需求文档 6.1“Understat 射门明细镜像”）。

这是什么：
  仓库 douglasbc/scraping-understat-dataset（2021 年最后更新）里，五大联赛 2014/15–2021/22 每个赛季一个 CSV，
  一行是一次射门：比赛编号、主客队、日期、比分、这次射门的 xG 等。按“比赛编号 + 主客”加起来就是每场的 xG。
  2021/22 只到 2021-12-12（半季）。

规则（你决定，2026-10-31）：
  - 仓库没有许可证（等于作者保留所有权利），数据是从 Understat 抓的，Understat 的 robots.txt 禁止所有程序。
    你决定下载到本机，只做个人学习和研究实验（例如“xG 比射门更有用吗”），不转发、不上传 GitHub。
  - 只从 raw.githubusercontent.com 下载文件，不访问 understat.com。
  - 网址里写死一个提交编号（settings.yaml 的 understat_mirror.commit），以后仓库改了，下载到的还是同一版。

保存：data/raw/understat_mirror/<联赛>/<赛季开始年份>.csv，原样保存，下载过就不再下载。
每次请求都记进 fetch_log。遇到 403 / 429（被拒绝、太频繁）马上停下，不重试。

运行：.venv\Scripts\python -m src.collect.understat_mirror
"""
import logging
import time
from pathlib import Path

from src.collect import football_data
from src.common.config import load_settings, resolve_path
from src.common.db import connect

SOURCE = "understat-mirror"
DATA_TYPE = "understat_shots"
RAW_BASE = "https://raw.githubusercontent.com"
HEADER = b"id,minute,result,X,Y,xG,"   # 射门明细 CSV 的开头几列；不是这样的就当作“内容可疑”

log = logging.getLogger("understat_mirror")


class Stopped(Exception):
    """网站拒绝（403）或嫌太频繁（429）：整个程序停下，不绕过、不重试。"""


def season_tag(start_year: int) -> str:
    """2014 → '14-15'（仓库文件名里赛季的写法）。"""
    return f"{start_year % 100:02d}-{(start_year + 1) % 100:02d}"


def file_url(cfg: dict, folder: str, start_year: int) -> str:
    return f"{RAW_BASE}/{cfg['repo']}/{cfg['commit']}/datasets/{folder}/shots_{folder}_{season_tag(start_year)}.csv"


class MirrorDownloader(football_data.Downloader):
    """借用 football-data 下载程序的低频请求和记录，只改来源名、内容检查，加上 403 / 429 马上停。"""
    source = SOURCE

    def fetch_shots(self, url: str, target: Path) -> str:
        if target.exists():
            return "skipped"
        r = self._get(url)
        if isinstance(r, str):
            self._record(url, DATA_TYPE, "failed", note=r)
            return "failed"
        if r.status_code in (403, 429):
            # 和懂球帝一样记成 failed + 403/429，数据状态页“有没有被网站拒绝”那一项会报出来
            self._record(url, DATA_TYPE, "failed", http_status=r.status_code)
            raise Stopped(f"HTTP {r.status_code}：{url}")
        if r.status_code == 404:
            self._record(url, DATA_TYPE, "not_found", http_status=404)
            return "not_found"
        if r.status_code != 200:
            self._record(url, DATA_TYPE, "failed", http_status=r.status_code)
            return "failed"
        content = r.content
        status = "saved" if content.lstrip(b"\xef\xbb\xbf").startswith(HEADER) else "suspicious"
        if status == "suspicious":
            target = target.with_suffix(".suspicious.txt")   # 照样存下来方便查，清洗时不用
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self._record(url, DATA_TYPE, status, http_status=200, path=target, content=content)
        return status

    def run_mirror(self, cfg: dict) -> dict:
        out = self.raw_dir / "understat_mirror"
        jobs = [(code, folder, y) for code, folder in cfg["leagues"].items()
                for y in range(int(cfg["first_season"]), int(cfg["last_season"]) + 1)]
        for i, (code, folder, y) in enumerate(jobs, 1):
            status = self.fetch_shots(file_url(cfg, folder, y), out / code / f"{y}.csv")
            if status != "skipped":
                log.info("[%d/%d] %s %s/%s：%s", i, len(jobs), code, y, y + 1, football_data.STATUS_NAMES[status])
        return self.counts


def main():
    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_dir / "understat_mirror.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    con = connect()
    d = MirrorDownloader(con, resolve_path(s["paths"]["raw"]), s["scrape"]["min_interval_seconds"])
    try:
        counts = d.run_mirror(s["understat_mirror"])
    except Stopped as err:
        log.error("被拒绝，已停下（不重试）：%s", err)
        counts = d.counts
    finally:
        con.close()
    log.info("完成：%s", "，".join(f"{football_data.STATUS_NAMES.get(k, k)} {v}" for k, v in counts.items())
             or "没有新的请求（都下载过了）")


if __name__ == "__main__":
    main()
