r"""下载 GitHub 上别人整理的 Transfermarkt 球员数据（需求文档 6.1“Transfermarkt 球员数据镜像（salimt）”）。

这是什么：
  仓库 salimt/football-datasets 里有一批从 Transfermarkt 抓下来的 CSV。我们只要研究“伤停有没有用”需要的几份：
    - 伤病记录：哪个球员、从哪天伤到哪天、什么伤
    - 身价记录：每个球员每次被估价的日期和身价（用来估“伤的这个人有多重要”）
    - 转会记录：每次转会的日期、从哪个队到哪个队（用来知道某一天这个球员在哪个队）
    - 球队每个赛季的联赛成绩：用来把 Transfermarkt 的球队编号对上我们的球队
    - 球队资料：球队编号、名字、国家
  转会记录太大（81MB），仓库用 Git LFS（大文件另外存）放，raw.githubusercontent.com 上只有一个指向它的小文件，
  所以这一个从 media.githubusercontent.com（GitHub 存大文件的地方）下载。两个地址的 robots.txt 都不存在（404，没有限制）。

规则（你决定，2026-10-31）：
  - 仓库没有许可证（等于作者保留所有权利），数据是从 Transfermarkt 抓的。
    你决定下载到本机，只做个人学习和研究实验（“伤停信息能不能让预测更准”），不转发、不上传 GitHub、不进数据库。
  - 只访问 raw.githubusercontent.com 和 media.githubusercontent.com，不访问 transfermarkt.com。
  - 网址里写死一个提交编号（settings.yaml 的 salimt_mirror.commit），以后仓库改了，下载到的还是同一版。

保存：data/raw/salimt/<文件名>.csv，原样保存，下载过就不再下载。
每次请求都记进 fetch_log。遇到 403 / 429（被拒绝、太频繁）马上停下，不重试。

运行：.venv\Scripts\python -m src.collect.salimt_mirror
（在工作树里运行、但要存进主项目文件夹时，用 --raw 和 --db 指定主项目的 data/raw 和数据库）
"""
import argparse
import logging
from pathlib import Path

from src.collect import football_data
from src.common.config import load_settings, resolve_path
from src.common.db import connect

SOURCE = "salimt-mirror"
DATA_TYPE = "transfermarkt_csv"
RAW_BASE = "https://raw.githubusercontent.com"
LFS_BASE = "https://media.githubusercontent.com/media"   # Git LFS 大文件的下载地址

log = logging.getLogger("salimt_mirror")


class Stopped(Exception):
    """网站拒绝（403）或嫌太频繁（429）：整个程序停下，不绕过、不重试。"""


def file_url(cfg: dict, item: dict) -> str:
    """一个文件的下载地址。lfs: true 的走大文件地址，其余走普通地址；都写死提交编号。"""
    base = LFS_BASE if item.get("lfs") else RAW_BASE
    return f"{base}/{cfg['repo']}/{cfg['commit']}/{item['path']}"


class MirrorDownloader(football_data.Downloader):
    """借用 football-data 下载程序的低频请求和记录，只改来源名、内容检查，加上 403 / 429 马上停。"""
    source = SOURCE

    def fetch_file(self, url: str, target: Path, header: str) -> str:
        if target.exists():
            return "skipped"
        r = self._get(url)
        if isinstance(r, str):
            self._record(url, DATA_TYPE, "failed", note=r)
            return "failed"
        if r.status_code in (403, 429):
            self._record(url, DATA_TYPE, "failed", http_status=r.status_code)
            raise Stopped(f"HTTP {r.status_code}：{url}")
        if r.status_code == 404:
            self._record(url, DATA_TYPE, "not_found", http_status=404)
            return "not_found"
        if r.status_code != 200:
            self._record(url, DATA_TYPE, "failed", http_status=r.status_code)
            return "failed"
        content = r.content
        # 开头不是预期的列名（例如拿到的是 LFS 指针小文件、或者错误页）：照样存下来方便查，但不当正式文件
        status = "saved" if content.lstrip(b"\xef\xbb\xbf").startswith(header.encode()) else "suspicious"
        if status == "suspicious":
            target = target.with_suffix(".suspicious.txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self._record(url, DATA_TYPE, status, http_status=200, path=target, content=content)
        return status

    def run_mirror(self, cfg: dict) -> dict:
        out = self.raw_dir / "salimt"
        items = list(cfg["files"].items())
        for i, (name, item) in enumerate(items, 1):
            status = self.fetch_file(file_url(cfg, item), out / f"{name}.csv", item["header"])
            if status != "skipped":
                log.info("[%d/%d] %s：%s", i, len(items), name, football_data.STATUS_NAMES[status])
        return self.counts


def main():
    parser = argparse.ArgumentParser(description="下载 salimt/football-datasets 里的 Transfermarkt 球员数据（只本机研究用）")
    parser.add_argument("--raw", help="原始数据文件夹（默认 settings.yaml 的 paths.raw）")
    parser.add_argument("--db", help="记抓取记录的数据库（默认 settings.yaml 的 paths.database）")
    args = parser.parse_args()
    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_dir / "salimt_mirror.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    con = connect(Path(args.db) if args.db else None)
    raw = Path(args.raw) if args.raw else resolve_path(s["paths"]["raw"])
    d = MirrorDownloader(con, raw, s["scrape"]["min_interval_seconds"])
    try:
        counts = d.run_mirror(s["salimt_mirror"])
    except Stopped as err:
        log.error("被拒绝，已停下（不重试）：%s", err)
        counts = d.counts
    finally:
        con.close()
    log.info("完成：%s", "，".join(f"{football_data.STATUS_NAMES.get(k, k)} {v}" for k, v in counts.items())
             or "没有新的请求（都下载过了）")


if __name__ == "__main__":
    main()
