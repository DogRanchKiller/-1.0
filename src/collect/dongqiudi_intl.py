r"""国际赛（国家队）懂球帝赔率回补：只存原始层（需求文档第 4 节“第 11 阶段：国际赛进数据库和开发回测”的“赔率（INB）”、
6.1“国际赛来源调查”、第 17 节）。

用的是懂球帝采集程序（src/collect/dongqiudi.py）的 Collector，规矩和韩K（src/collect/dongqiudi_kor.py）一样：
- 低频：两次请求至少隔 settings.yaml 的 dongqiudi_intl.min_interval_seconds 秒（10 秒）；
- 遇到 403、429 马上停，不重试、不绕过；赛季页内容不对（可能是拦截页）也马上停；
  赔率接口返回的不是 JSON（多半是拦截页）也马上停；返回了别的比赛（网站的老毛病）连着 3 次才停；
- 5xx 和连不上：等 30 秒再试，最多 3 次（Collector 里做的）；
- 每次请求记进 fetch_log（来源 dongqiudi），原始文件放 data/raw/dongqiudi/ 下，和韩K 一样的放法。

下什么（赛事和赛季编号写在 settings 的 dongqiudi_intl.competitions，预选赛没有历届赛季页，所以编号是先查好写进去的）：
1. 赛季页 season/<赛事代码>/<赛季>.html.gz，每季一页。页面里藏着的数据（window.__INITIAL_STATE__）有每场的
   北京时间、两队编号、比分和轮次名字。下载后核对页面里的赛季编号（sid）、赛事编号（cid）和 settings 一样，不一样就停；
2. 每场赔率 odds_final/<赛事代码>/<赛季>/<比赛编号>.json.gz：只下踢完的、开球时间（UTC）在
   kickoff_from 到 kickoff_to 之间（含两头）的比赛，从近往远下。
考试赛季锁住：kickoff_to 那一年不是开发赛季就报错，什么都不下；每场下赔率前再查一次，
开球在考试赛季第一天（2024-01-01 UTC）或以后的，直接报错，不下。
只有日期、没有开球时刻的比赛，按北京时间这一天的头尾（换成 UTC）算，两头都要在范围里才下。

可以中途停、下次接着来（已经有的文件跳过）。一次最多发 max_requests 次请求，到数就停（分段跑）。
运行（在项目文件夹里，Claude 放后台跑）：
    .venv\Scripts\python -m src.collect.dongqiudi_intl                     # 接着上次往下抓
    .venv\Scripts\python -m src.collect.dongqiudi_intl --max 50            # 这次最多 50 次请求
    .venv\Scripts\python -m src.collect.dongqiudi_intl --only INT_WC --limit 5   # 小试：只下世界杯、每季最多 5 场赔率
日志：logs/dongqiudi_intl.log（--log-dir 可以换文件夹）。
退出码：0 全部下完；3 这一段到了请求上限（还没下完，下次接着来）；2 被网站拒绝或拦截页；1 其他错误。
"""
import argparse
import gzip
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.clean.dongqiudi_reader import BEIJING, StateGame, parse_season_state
from src.collect.dongqiudi import ODDS_URL, SEASON_URL, STATUS_NAMES, Blocked, Collector, check_odds
from src.collect.dongqiudi_kor import Enough, check_season
from src.common.config import load_settings, resolve_path, season_role
from src.common.db import connect

log = logging.getLogger("dongqiudi")

EXIT_DONE, EXIT_ERROR, EXIT_BLOCKED, EXIT_ENOUGH = 0, 1, 2, 3


def window(cfg: dict, settings: dict) -> tuple[datetime, datetime]:
    """下赔率的开球时间范围（UTC）：[kickoff_from 当天 00:00, kickoff_to 第二天 00:00)，左闭右开。

    kickoff_to 那一年必须是开发赛季；范围的右端不能超过考试赛季第一年的 1 月 1 日（2024-01-01 UTC）。
    国家队赛季 = 比赛那一年（需求文档第 11 阶段），所以 2024 年 1 月 1 日起都是考试赛季。
    """
    start = datetime.combine(date.fromisoformat(str(cfg["kickoff_from"])), datetime.min.time(), timezone.utc)
    last_day = date.fromisoformat(str(cfg["kickoff_to"]))
    end = datetime.combine(last_day + timedelta(days=1), datetime.min.time(), timezone.utc)
    for y in range(start.year, last_day.year + 1):
        role = season_role(y, settings)
        if role != "dev":
            raise ValueError(f"settings 的 dongqiudi_intl 日期范围包含 {y} 年，这一年是“{role}”，不是开发赛季：考试赛季锁住，不下载")
    exam_first = datetime(min(settings["seasons"]["exam"]), 1, 1, tzinfo=timezone.utc)
    if end > exam_first:
        raise ValueError(f"dongqiudi_intl.kickoff_to 超过了考试赛季的第一天 {exam_first:%Y-%m-%d}：不下载")
    if start >= end:
        raise ValueError("dongqiudi_intl.kickoff_from 晚于 kickoff_to")
    return start, end


def kickoff_span(g: StateGame) -> tuple[datetime, datetime]:
    """一场比赛最早、最晚可能的开球时间（UTC）。有开球时刻就是同一个时刻；只有日期就是北京时间这一天的头和尾。"""
    if g.kickoff_bj is not None:
        t = g.kickoff_bj.astimezone(timezone.utc)
        return t, t
    d = date.fromisoformat(g.day_bj)
    first = datetime.combine(d, datetime.min.time(), BEIJING).astimezone(timezone.utc)
    return first, first + timedelta(days=1) - timedelta(seconds=1)


def in_window(g: StateGame, start: datetime, end: datetime) -> bool:
    lo, hi = kickoff_span(g)
    return lo >= start and hi < end


def season_folder(label: str) -> str:
    """settings 里的赛季名 → 文件夹名（只认数字和“-”，例如 2022、2020-21），别的报错，免得写出奇怪的路径。"""
    label = str(label)
    if not label or not all(ch.isdigit() or ch == "-" for ch in label):
        raise ValueError(f"赛季名“{label}”只能有数字和“-”（例如 2022、2020-21）")
    return label


class IntlCollector(Collector):
    """国际赛赔率回补。cfg 是 settings 的 dongqiudi_intl 段（有 min_interval_seconds、competitions、kickoff_from、kickoff_to）。"""

    def __init__(self, con, raw_dir, cfg: dict, settings: dict, max_requests: int | None = None, **kw):
        super().__init__(con, raw_dir, cfg, **kw)
        self.settings = settings
        self.max_requests = max_requests
        self.start, self.end = window(cfg, settings)   # 这里就查考试赛季锁，日期范围不对什么都不下
        self.exam_first = datetime(min(settings["seasons"]["exam"]), 1, 1, tzinfo=timezone.utc)

    def requests_made(self) -> int:
        """这次运行一共发了几次请求（跳过的不算）。"""
        return sum(v for k, v in self.counts.items() if k != "skipped")

    def _budget(self):
        if self.max_requests is not None and self.requests_made() >= self.max_requests:
            raise Enough()

    def season_state(self, code: str, label: str, cid: int, sid: int):
        """赛季页（只存一次）→ (赛季信息, 比赛列表)；没拿到返回 None。页面里的 sid、cid 和 settings 不一样就停。"""
        target = self.out / "season" / code / f"{season_folder(label)}.html.gz"
        if not target.exists():
            self._budget()
            self.fetch(SEASON_URL.format(season_id=sid), "season_page", target, check_season, once=True)
        if not target.exists():
            return None
        info, games = parse_season_state(gzip.decompress(target.read_bytes()).decode("utf-8"))
        if str(info.get("sid")) != str(sid) or str(info.get("cid")) != str(cid):
            raise ValueError(f"{code} {label}：赛季页里是赛事 {info.get('cid')}（{info.get('comp')}）、赛季 {info.get('sid')}，"
                             f"settings 写的是赛事 {cid}、赛季 {sid}：编号写错了还是网站改了？查清楚再下")
        return info, games

    def odds(self, code: str, label: str, g: StateGame, bad_in_row: int) -> int:
        """一场的赔率（只存一次）。返回“连着几场返回了别的比赛”。不是 JSON 就马上停（多半是拦截页）。"""
        lo, hi = kickoff_span(g)
        if hi >= self.exam_first or not in_window(g, self.start, self.end):
            # 正常走不到这里（run 先按日期挑过）；万一走到了，宁可报错也不下考试赛季
            raise ValueError(f"比赛 {g.match_id} 开球 {lo:%Y-%m-%d %H:%M} UTC 不在下载范围里（考试赛季锁住），不下")
        target = self.out / "odds_final" / code / season_folder(label) / f"{g.match_id}.json.gz"
        if target.exists():
            return 0
        self._budget()
        problems = []

        def check(content):
            p = check_odds(content, g.match_id)
            if p:
                problems.append(p)
            return p
        url = ODDS_URL.format(match_id=g.match_id)
        status, _ = self.fetch(url, "odds_final", target, check, once=True)
        if status == "suspicious":
            if problems and problems[-1] in ("不是 JSON", "JSON 里没有赔率"):
                raise Blocked(f"赔率接口返回的不是赔率（{problems[-1]}），可能是拦截页：{url}")
            bad_in_row += 1
            if bad_in_row >= 3:
                raise Blocked(f"连着 3 场赔率接口返回了别的比赛（最后一场 {g.match_id}）")
            return bad_in_row
        return 0

    def run(self, only: list[str] | None = None, limit: int | None = None):
        """先下全部赛季页，再下赔率（每季从近往远）。only：只下这几个赛事；limit：每季最多新下几场赔率（小试用）。"""
        comps = self.cfg["competitions"]
        unknown = [c for c in (only or []) if c not in comps]
        if unknown:
            raise ValueError(f"settings 的 dongqiudi_intl.competitions 里没有 {unknown}")
        todo = [(code, str(label), int(comps[code]["cid"]), int(sid))
                for code in comps if not only or code in only
                for label, sid in comps[code]["seasons"].items()]
        states = []
        for code, label, cid, sid in todo:
            st = self.season_state(code, label, cid, sid)
            if st is None:
                log.warning("%s %s：赛季页没拿到（见 fetch_log），这季跳过", code, label)
                continue
            info, games = st
            played = [g for g in games if g.status == "Played"]
            chosen = [g for g in played if in_window(g, self.start, self.end)]
            log.info("%s %s（%s）：赛季页 %d 场，踢完 %d 场，开球在 %s 到 %s 的 %d 场", code, label, info.get("comp"),
                     len(games), len(played), self.cfg["kickoff_from"], self.cfg["kickoff_to"], len(chosen))
            states.append((code, label, chosen))
        for code, label, chosen in states:
            chosen = sorted(chosen, key=lambda g: (kickoff_span(g)[0], g.match_id), reverse=True)
            before = dict(self.counts)
            bad, new_files = 0, 0
            for g in chosen:
                if limit is not None and new_files >= limit:
                    break
                had = (self.out / "odds_final" / code / season_folder(label) / f"{g.match_id}.json.gz").exists()
                bad = self.odds(code, label, g, bad)
                if not had:
                    new_files += 1
            new = {k: v - before.get(k, 0) for k, v in self.counts.items() if v - before.get(k, 0)}
            log.info("%s %s：%d 场的赔率；这次 %s", code, label, len(chosen),
                     "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in new.items()) or "都已有")


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_dir / "dongqiudi_intl.log", encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S", handlers=handlers)


def main():
    ap = argparse.ArgumentParser(description="国际赛懂球帝赔率回补（只存原始层）")
    ap.add_argument("--max", type=int, help="这次最多发几次请求；不写用 settings 的 dongqiudi_intl.max_requests")
    ap.add_argument("--only", nargs="*", help="只下这几个赛事代码，例如 INT_WC INT_EURO；不写就是全部")
    ap.add_argument("--limit", type=int, help="小试：每季最多新下几场赔率")
    ap.add_argument("--log-dir", help="日志放哪个文件夹；不写用 settings 的 paths.logs")
    args = ap.parse_args()
    s = load_settings()
    cfg = s["dongqiudi_intl"]
    setup_logging(Path(args.log_dir) if args.log_dir else resolve_path(s["paths"]["logs"]))
    con = connect()
    code = EXIT_ERROR
    c = None
    try:
        c = IntlCollector(con, resolve_path(s["paths"]["raw"]), cfg, s,
                          max_requests=args.max if args.max is not None else cfg["max_requests"])
        c.run(args.only, args.limit)
        log.info("全部下完了")
        code = EXIT_DONE
    except Enough:
        log.info("这一段到了 %d 次请求，先停，下次接着来", c.max_requests)
        code = EXIT_ENOUGH
    except Blocked as e:
        log.error("懂球帝拒绝访问或返回了拦截页，程序已停止（不会绕过）：%s", e)
        code = EXIT_BLOCKED
    except Exception:
        log.exception("程序出错停止，下面是出错的位置")
        code = EXIT_ERROR
    finally:
        con.close()
        if c is not None:
            log.info("本次请求：%s", "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in c.counts.items()) or "没有新的请求")
    sys.exit(code)


if __name__ == "__main__":
    main()
