r"""下载国家队比赛比分 martj42/international_results（需求文档第 4 节“第 11 阶段：国际赛进数据库和开发回测”、6.1“国际赛来源调查”）。

这是什么：
  GitHub 上公开的数据集（CC0 公有领域，随便用），男子成年国家队 1872 年起的全部比赛。只下两个文件：
    - results.csv     每场一行：日期、主队、客队、比分（含加时、不含点球）、赛事、城市、国家、是否中立场
    - shootouts.csv   踢了点球大战的比赛：日期、主队、客队、点球赢的队
  进球者文件 goalscorers.csv 用不到，不下。

规则：
  - 只访问 raw.githubusercontent.com。下载前先看一次它的 robots.txt（2026-11-05 是 404：没有 robots.txt = 没有限制）；
    robots.txt 不让下就停下，不下载。
  - 每个文件下一次：data/raw/martj42/<下载日期>/results.csv、shootouts.csv。同一天已经下过的不再下。
  - 两次请求至少隔 settings.yaml 的 scrape.min_interval_seconds 秒（至少 5 秒）。
  - 遇到 403（拒绝）、429（太频繁）马上停下，不重试、不绕过。
  - 每次请求都记进 fetch_log（来源 martj42）。

运行：.venv\Scripts\python -m src.collect.martj42
（在工作树里运行、但要存进主项目文件夹时，用 --raw 指定主项目的 data/raw；--db 指定记抓取记录的数据库）
"""
import argparse
import logging
from datetime import date
from pathlib import Path
from urllib.robotparser import RobotFileParser

from src.collect import football_data
from src.common.config import load_settings, resolve_path
from src.common.db import connect

DATA_TYPE = "results_csv"
ALLOWED_HOST = "raw.githubusercontent.com"   # 只访问这一个网站

log = logging.getLogger("martj42")


class Stopped(Exception):
    """网站拒绝（403）、嫌太频繁（429）或 robots.txt 不让下：整个程序停下，不绕过、不重试。"""


def check_host(url: str) -> None:
    """网址必须在 raw.githubusercontent.com 上（设置写错时不去访问别的网站）。"""
    if url.split("/")[2] != ALLOWED_HOST or not url.startswith("https://"):
        raise ValueError(f"只访问 https://{ALLOWED_HOST}，这个网址不行：{url}")


def file_url(cfg: dict, name: str) -> str:
    """一个文件的下载地址：base_url/<文件名>.csv。"""
    url = f"{cfg['base_url'].rstrip('/')}/{name}.csv"
    check_host(url)
    return url


class Martj42Downloader(football_data.Downloader):
    """借用 football-data 下载程序的低频请求和抓取记录，只改来源名、内容检查，加上 robots.txt、403 / 429 马上停。"""
    source = "martj42"

    def check_robots(self, robots_url: str, urls: list[str]) -> str:
        """先看 robots.txt。404 = 没有 robots.txt，不限制；200 就按里面写的规则查每个网址；不让下就停下。"""
        check_host(robots_url)
        r = self._get(robots_url)
        if isinstance(r, str):
            self._record(robots_url, "robots", "failed", note=r)
            raise Stopped(f"robots.txt 读不到（{r}），不下载")
        if r.status_code in (403, 429):
            self._record(robots_url, "robots", "failed", http_status=r.status_code)
            raise Stopped(f"HTTP {r.status_code}：{robots_url}")
        if r.status_code == 404:
            self._record(robots_url, "robots", "not_found", http_status=404)
            return "没有 robots.txt（404），没有限制"
        if r.status_code != 200:
            self._record(robots_url, "robots", "failed", http_status=r.status_code)
            raise Stopped(f"robots.txt 返回 HTTP {r.status_code}，不确定让不让下，不下载")
        self._record(robots_url, "robots", "saved", http_status=200, content=r.content)
        parser = RobotFileParser()
        parser.parse(r.text.splitlines())
        ua = self.session.headers["User-Agent"]
        blocked = [u for u in urls if not parser.can_fetch(ua, u)]
        if blocked:
            raise Stopped(f"robots.txt 不让下这些网址：{blocked}")
        return "robots.txt 允许"

    def fetch_file(self, url: str, target: Path, header: str) -> str:
        """下载一个文件存到 target。已经有了就跳过；开头不是预期列名的存成 .suspicious.txt（导入时不用）。返回状态。"""
        if target.exists():
            return "skipped"
        r = self._get(url)
        if isinstance(r, str):                       # 连不上或服务器出错 3 次
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
        ok = content.lstrip(b"\xef\xbb\xbf").startswith(header.encode())
        status = "saved" if ok else "suspicious"
        if not ok:
            target = target.with_suffix(".suspicious.txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self._record(url, DATA_TYPE, status, http_status=200, path=target, content=content)
        return status

    def run_martj42(self, cfg: dict, today: date) -> dict:
        """先看 robots.txt，再下 cfg["files"] 里的每个文件，存到 raw/martj42/<今天>/。"""
        out = self.raw_dir / cfg["raw_folder"] / f"{today:%Y-%m-%d}"
        jobs = [(name, file_url(cfg, name), header) for name, header in cfg["files"].items()]
        todo = [j for j in jobs if not (out / f"{j[0]}.csv").exists()]
        if not todo:
            log.info("今天（%s）两个文件都已经下过，不再请求", today)
            return self.counts
        log.info("robots.txt：%s", self.check_robots(cfg["robots_url"], [u for _n, u, _h in todo]))
        for name, url, header in todo:
            status = self.fetch_file(url, out / f"{name}.csv", header)
            log.info("%s：%s", name, football_data.STATUS_NAMES.get(status, status))
        return self.counts


def main():
    parser = argparse.ArgumentParser(description="下载 martj42/international_results 的比分和点球大战文件")
    parser.add_argument("--raw", help="原始数据文件夹（默认 settings.yaml 的 paths.raw）")
    parser.add_argument("--db", help="记抓取记录的数据库（默认 settings.yaml 的 paths.database）")
    args = parser.parse_args()
    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_dir / "martj42.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    con = connect(Path(args.db) if args.db else None)
    raw = Path(args.raw) if args.raw else resolve_path(s["paths"]["raw"])
    interval = max(float(s["scrape"]["min_interval_seconds"]), 5.0)   # 至少 5 秒
    d = Martj42Downloader(con, raw, interval)
    try:
        counts = d.run_martj42(s["martj42"], date.today())
    except Stopped as err:
        log.error("已停下（不重试）：%s", err)
        counts = d.counts
    finally:
        d.session.close()
        con.close()
    log.info("完成：%s", "，".join(f"{football_data.STATUS_NAMES.get(k, k)} {v}" for k, v in counts.items())
             or "没有新的请求")


if __name__ == "__main__":
    main()
