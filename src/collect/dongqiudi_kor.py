r"""韩K（K联赛 1 级）历史回补：只存原始层（需求文档第 4 节“第 11 阶段：韩K 回补和开发回测”、6.1、第 17 节）。

用的是懂球帝采集程序（src/collect/dongqiudi.py）的 Collector，规矩一样：
- 低频：两次请求至少隔 settings.yaml 的 dongqiudi_kor.min_interval_seconds 秒（10 秒）；
- 遇到 403、429 马上停，不重试、不绕过；赛季页、历届赛季页内容不对（可能是拦截页）也马上停；
  赔率接口返回的不是 JSON（多半是拦截页）也马上停；返回了别的比赛（网站的老毛病）连着 3 次才停；
- 每次请求记进 fetch_log（来源 dongqiudi），原始文件放 data/raw/dongqiudi/ 下，和五大联赛的放法一样。

下什么：
1. 历届赛季页 history/KOR/<日期>.html.gz（一天最多一次）；
2. 赛季页 season/KOR/<年>.html.gz，first_season 到 last_season（2010–2023），每季一页。
   页面里藏着的数据（window.__INITIAL_STATE__）有每场的北京时间、两队编号、比分和轮次名字；
3. 每场赔率 odds_final/KOR/<年>/<比赛编号>.json.gz：odds_first_season 到 last_season（2017–2023），
   只下联赛比赛（升降级附加赛不下），从近往远下。
只下开发赛季：哪一季 season_role 不是“开发”就报错，不下（考试赛季 2024、2025 锁住）。

可以中途停、下次接着来（已经有的文件跳过）。一次最多发 max_requests 次请求，到数就停（分段跑）。
运行（在项目文件夹里，Claude 放后台跑）：
    .venv\Scripts\python -m src.collect.dongqiudi_kor               # 接着上次往下抓
    .venv\Scripts\python -m src.collect.dongqiudi_kor --max 50      # 这次最多 50 次请求
日志：logs/dongqiudi_kor.log。
"""
import argparse
import gzip
import logging
import sys

from src.clean.dongqiudi_reader import parse_season_state, round_kind
from src.collect.dongqiudi import (ODDS_URL, SEASON_URL, STATUS_NAMES, Blocked, Collector, check_html, check_odds)
from src.common.config import load_settings, resolve_path, season_role
from src.common.db import connect

log = logging.getLogger("dongqiudi")


class Enough(Exception):
    """这一段的请求数到了上限：正常停下，下次接着来。"""


def check_season(content: bytes) -> str | None:
    """赛季页：要有战报链接，还要读得出页面里的赛季数据。返回问题说明，没问题返回 None。"""
    problem = check_html(content)
    if problem:
        return problem
    try:
        ok = parse_season_state(content.decode("utf-8")) is not None
    except (UnicodeDecodeError, ValueError):
        ok = False
    return None if ok else "页面里没有赛季数据（window.__INITIAL_STATE__），可能是拦截页或网站改版"


class KorCollector(Collector):
    """韩K 回补。cfg 是 settings 的 dongqiudi_kor 段（有 min_interval_seconds、league_ids、calendar_year）。"""

    def __init__(self, con, raw_dir, cfg: dict, settings: dict, max_requests: int | None = None, **kw):
        super().__init__(con, raw_dir, cfg, **kw)
        self.settings = settings
        self.max_requests = max_requests
        self.league = cfg["league"]

    def requests_made(self) -> int:
        """这次运行一共发了几次请求（跳过的不算）。"""
        return sum(v for k, v in self.counts.items() if k != "skipped")

    def _budget(self):
        if self.max_requests is not None and self.requests_made() >= self.max_requests:
            raise Enough()

    def _dev_only(self, season: int):
        role = season_role(season, self.settings)
        if role != "dev":
            raise ValueError(f"{season} 赛季是“{role}”，不是开发赛季：考试赛季锁住，不下载")

    def season_state(self, season: int, season_id: int):
        """赛季页（只存一次）→ (赛季信息, 比赛列表)；没拿到返回 None。"""
        target = self.out / "season" / self.league / f"{season}.html.gz"
        if not target.exists():
            self._budget()
            self.fetch(SEASON_URL.format(season_id=season_id), "season_page", target, check_season, once=True)
        if not target.exists():
            return None
        return parse_season_state(gzip.decompress(target.read_bytes()).decode("utf-8"))

    def odds(self, season: int, match_id: int, bad_in_row: int) -> int:
        """一场的赔率（只存一次）。返回“连着几场返回了别的比赛”。不是 JSON 就马上停（多半是拦截页）。"""
        target = self.out / "odds_final" / self.league / str(season) / f"{match_id}.json.gz"
        if target.exists():
            return 0
        self._budget()
        problems = []

        def check(content):
            p = check_odds(content, match_id)
            if p:
                problems.append(p)
            return p
        status, _ = self.fetch(ODDS_URL.format(match_id=match_id), "odds_final", target, check, once=True)
        if status == "suspicious":
            if problems and problems[-1] in ("不是 JSON", "JSON 里没有赔率"):
                raise Blocked(f"赔率接口返回的不是赔率（{problems[-1]}），可能是拦截页：{ODDS_URL.format(match_id=match_id)}")
            bad_in_row += 1
            if bad_in_row >= 3:
                raise Blocked(f"连着 3 场赔率接口返回了别的比赛（最后一场 {match_id}）")
            return bad_in_row
        return 0

    def run(self, first: int, odds_first: int, last: int):
        """先下全部赛季页（从近往远），再下赔率（从近往远）。"""
        for y in range(first, last + 1):
            self._dev_only(y)
        seasons = self.history(self.league)
        states = {}
        for y in range(last, first - 1, -1):
            if y not in seasons:
                log.info("%s %d：历届赛季页里没有这个赛季，跳过", self.league, y)
                continue
            st = self.season_state(y, seasons[y])
            if st is None:
                log.warning("%s %d：赛季页没拿到（见 fetch_log），这季跳过", self.league, y)
                continue
            info, games = st
            if info.get("cur"):
                log.info("%s %d：网站说这季还在进行，跳过", self.league, y)
                continue
            states[y] = games
            log.info("%s %d：赛季页 %d 场", self.league, y, len(games))
        prefixes = list(self.cfg["league_round_prefixes"])
        non_league = list(self.cfg.get("non_league_rounds") or [])
        for y in range(last, odds_first - 1, -1):
            if y not in states:
                continue
            ids = [g.match_id for g in states[y] if g.status == "Played"
                   and round_kind(g.round_name, prefixes, non_league) == "league"]
            before = dict(self.counts)
            bad = 0
            for mid in ids:
                bad = self.odds(y, mid, bad)
            new = {k: v - before.get(k, 0) for k, v in self.counts.items() if v - before.get(k, 0)}
            log.info("%s %d：联赛比赛 %d 场的赔率；这次 %s", self.league, y, len(ids),
                     "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in new.items()) or "都已有")


def setup_logging(log_dir):
    log_dir.mkdir(exist_ok=True)
    handlers = [logging.FileHandler(log_dir / "dongqiudi_kor.log", encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S", handlers=handlers)


def main():
    ap = argparse.ArgumentParser(description="韩K 历史回补（只存原始层）")
    ap.add_argument("--max", type=int, help="这次最多发几次请求；不写用 settings 的 dongqiudi_kor.max_requests")
    args = ap.parse_args()
    s = load_settings()
    cfg = s["dongqiudi_kor"]
    setup_logging(resolve_path(s["paths"]["logs"]))
    con = connect()
    c = KorCollector(con, resolve_path(s["paths"]["raw"]), cfg, s,
                     max_requests=args.max if args.max is not None else cfg["max_requests"])
    code = 0
    try:
        c.run(int(cfg["first_season"]), int(cfg["odds_first_season"]), int(cfg["last_season"]))
        log.info("全部下完了")
    except Enough:
        log.info("这一段到了 %d 次请求，先停，下次接着来", c.max_requests)
    except Blocked as e:
        log.error("懂球帝拒绝访问或返回了拦截页，程序已停止（不会绕过）：%s", e)
        code = 2
    except Exception:
        log.exception("程序出错停止，下面是出错的位置")
        code = 1
    finally:
        con.close()
        log.info("本次请求：%s", "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in c.counts.items()) or "没有新的请求")
    sys.exit(code)


if __name__ == "__main__":
    main()
