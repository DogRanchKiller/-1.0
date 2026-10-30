r"""懂球帝采集程序（需求文档 6.2、6.2.1、第 17 节）。只存进原始层 data/raw/dongqiudi/，先不进数据库。

为什么用懂球帝：它的赔率接口不用登录就能看，robots.txt 允许程序访问（2026-10-30 查过：
m.dongqiudi.com 只禁 /_s/，api.dongqiudi.com 全部允许），有 20 家左右公司的胜平负、让球、大小球，
包括竞彩官方，而且历史比赛也有（大约 2017/18 赛季起）。
它只给“初盘”和“最新”两个时间点，没有中间变化记录；中间的变化要我们自己定时去抓（“即将开赛”模式）。

遵守的规矩：
- 低频：两次请求至少隔 5 秒（settings.yaml 的 dongqiudi.min_interval_seconds），出错等 30 秒，最多试 3 次。
- 只看公开页面和接口，不登录、不碰会员内容（例如需要登录的 lottery.dongqiudi.com 接口不用），
  遇到验证码、拦截页面就停下来记“内容可疑”，不绕过。
- 每次请求都记进数据库的 fetch_log 表（来源 dongqiudi）。

三种运行方式（Claude 运行，或者由定时任务运行）：
    .venv\Scripts\python -m src.collect.dongqiudi upcoming     # 即将开赛：抓赛程和赛前赔率快照（每小时运行）
    .venv\Scripts\python -m src.collect.dongqiudi probe        # 试验：每个赛季抽几场，看赔率能往回补到哪年
    .venv\Scripts\python -m src.collect.dongqiudi backfill --leagues E0 --from 2019   # 历史回补（可中途停、下次接着来）

原始文件放在哪（都是 gzip 压缩，内容原样不改）：
    history/<联赛>/<日期>.html.gz                  历届赛季页（变了才存新的一份）
    season/<联赛>/<赛季>.html.gz                   已结束赛季的赛季页（只存一次）
    fixtures/<联赛>/<抓取时间>.json.gz             赛程接口（和上一次同一网址内容一样就不存）
    odds_final/<联赛>/<赛季>/<比赛编号>.json.gz    踢完以后的赔率（初盘 + 收盘，只存一次）
    odds_live/<联赛>/<比赛编号>/<抓取时间>.json.gz  赛前赔率快照（变了才存）
    *.suspicious.*                                  内容可疑的文件（例如返回了别的比赛），照存方便查，清洗时不用
"""
import argparse
import gzip
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests

from src.clean.dongqiudi_reader import parse_fixtures, parse_history, next_start, parse_season_page, read_json
from src.common.config import PROJECT_ROOT, load_settings, resolve_path
from src.common.db import connect

SOURCE = "dongqiudi"
USER_AGENT = "Mozilla/5.0 (Linux; Android 13) football-study-project/0.1 (personal, non-commercial learning)"
HISTORY_URL = "https://m.dongqiudi.com/zhanbao/lishi/{league_id}"
SEASON_URL = "https://m.dongqiudi.com/zhanbao/saiji/{season_id}"
FIXTURES_URL = "https://api.dongqiudi.com/data/tab/league/new/{league_id}?start={start}&version=576&init=1"
ODDS_URL = ("https://sport-data.dongqiudi.com/soccer/biz/dqd/v1/match/odds/index/{match_id}"
            "?app=dqd&lang=zh-cn&cmp_type=&platform=android")

log = logging.getLogger("dongqiudi")


def current_season(now: datetime) -> int:
    """当前赛季的开始年份：7 月起算新赛季。"""
    return now.year if now.month >= 7 else now.year - 1


def stamp(now: datetime) -> str:
    """文件名用的时间（标准时间），例如 20261030T0805Z。"""
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")


# ---------- 内容检查：内容不对就标“可疑”，不当正常数据用 ----------
def check_html(content: bytes) -> str | None:
    """懂球帝的战报页都有 /zhanbao/ 链接；没有说明返回的是拦截页或错误页。返回问题说明，没问题返回 None。"""
    return None if b"/zhanbao/" in content else "页面里没有战报链接，可能是拦截页或错误页"


def check_fixtures(content: bytes) -> str | None:
    try:
        data = json.loads(content)
    except ValueError:
        return "不是 JSON"
    return None if isinstance(data, dict) and isinstance(data.get("list"), list) else "JSON 里没有比赛列表"


def check_odds(content: bytes, match_id: int) -> str | None:
    """返回的 matchId 必须和请求的一样：网站有时把编号换成另一场比赛（例如 3000000 → 53000000）。"""
    try:
        data = json.loads(content)
    except ValueError:
        return "不是 JSON"
    if not isinstance(data, dict) or "euro" not in data:
        return "JSON 里没有赔率"
    if str(data.get("matchId")) != str(match_id):
        return f"返回了另一场比赛 {data.get('matchId')}"
    return None


class Blocked(Exception):
    """网站拒绝访问（403/429）或返回了拦截页：整个程序停下来，不重试、不绕过。"""


def snapshot_gap(hours_to_kickoff: float, rules) -> float | None:
    """离开球还有几小时 → 至少隔几小时抓一次快照；太远了不抓，返回 None。rules 见 settings.yaml。"""
    for limit, gap in rules:
        if hours_to_kickoff <= limit:
            return gap
    return None


def spread(items: list, n: int) -> list:
    """从列表里均匀抽 n 个（第一个、中间、最后一个……），试验模式用。"""
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


class Collector:
    def __init__(self, con, raw_dir: Path, cfg: dict, session=None, sleep=time.sleep, clock=time.monotonic, now=None):
        self.con = con
        self.out = raw_dir / SOURCE
        self.cfg = cfg
        self.min_interval = cfg["min_interval_seconds"]
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.sleep, self.clock = sleep, clock
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._last_request = None
        self.counts = {}

    # ---------- 网络 ----------
    def _shared_wait(self) -> float:
        """别的程序（例如定时任务和回补同时在跑）刚请求过懂球帝，也要等够间隔。用 fetch_log 里最后一条的时间判断。"""
        row = self.con.execute("select max(fetched_at_utc) from fetch_log where source = ?", (SOURCE,)).fetchone()
        if not row or not row[0]:
            return 0
        return self.min_interval - (self.now() - datetime.fromisoformat(row[0])).total_seconds()

    def _get(self, url: str):
        """低频请求：两次请求至少隔 min_interval 秒；连不上或服务器出错时等 30 秒重试，最多 3 次。"""
        for attempt in range(3):
            wait = self._shared_wait()
            if self._last_request is not None:
                wait = max(wait, self.min_interval - (self.clock() - self._last_request))
            if wait > 0:
                self.sleep(min(wait, self.min_interval))
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
            (url, SOURCE, data_type, self.now().isoformat(timespec="seconds"), status, http_status,
             path.relative_to(PROJECT_ROOT).as_posix() if path and path.is_relative_to(PROJECT_ROOT) else (str(path) if path else None),
             len(content) if content is not None else None,
             hashlib.sha256(content).hexdigest() if content is not None else None, note),
        )
        self.con.commit()
        self.counts[status] = self.counts.get(status, 0) + 1

    def _last(self, url, data_type, statuses=None):
        """这个网址（这种用途）最近一次的记录：(状态, 指纹, 时间)。statuses 不为空时只看这些状态。"""
        sql = "select status, sha256, fetched_at_utc from fetch_log where url = ? and data_type = ?"
        args = [url, data_type]
        if statuses:
            sql += f" and status in ({','.join('?' * len(statuses))})"
            args += list(statuses)
        return self.con.execute(sql + " order by id desc limit 1", args).fetchone()

    # ---------- 抓一个网址 ----------
    def fetch(self, url, data_type, target: Path, check, once: bool, retry_bad=False):
        """抓一个网址，压缩保存到 target。返回 (状态, 内容)；内容只在“已保存”“没变化”时有。

        once=True：踢完的东西不会再变，target 已经有了就不再请求；以前“网站没有”“内容可疑”的也跳过（retry_bad 除外）。
        once=False：快照，和这个网址上一次的内容一样就不存，只记“没变化”。
        """
        if once:
            if target.exists():
                return "skipped", None
            last = self._last(url, data_type)
            if not retry_bad and last and last[0] in ("not_found", "suspicious"):
                return "skipped", None
        r = self._get(url)
        if isinstance(r, str):
            self._record(url, data_type, "failed", note=r)
            return "failed", None
        if r.status_code in (403, 429):
            self._record(url, data_type, "failed", http_status=r.status_code, note="网站拒绝访问，程序停止")
            raise Blocked(f"HTTP {r.status_code}：{url}")
        if r.status_code in (404, 410):
            self._record(url, data_type, "not_found", http_status=r.status_code)
            return "not_found", None
        if r.status_code != 200:
            self._record(url, data_type, "failed", http_status=r.status_code)
            return "failed", None
        content = r.content
        problem = check(content)
        if problem:
            target = target.with_name(target.name.split(".")[0] + ".suspicious." + target.name.split(".", 1)[1])
            status = "suspicious"
        elif not once and (last := self._last(url, data_type, ("saved", "unchanged"))) \
                and last[1] == hashlib.sha256(content).hexdigest():
            self._record(url, data_type, "unchanged", http_status=200, content=content)
            return "unchanged", content
        else:
            status = "saved"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(content, mtime=0))  # 原样压缩保存，不改内容
        self._record(url, data_type, status, http_status=200, path=target, content=content, note=problem)
        if status == "suspicious":
            log.warning("内容可疑（%s）：%s", problem, url)
            if data_type in ("history_page", "season_page", "fixtures"):
                raise Blocked(f"{problem}：{url}")  # 页面不对，可能被拦截了，停下来不硬试
            return status, None
        return status, content


    # ---------- 各种页面 ----------
    def history(self, league: str) -> dict[int, int]:
        """历届赛季页 → {赛季开始年份: 懂球帝赛季编号}。一天最多抓一次，其他时候用已存的最新一份。"""
        folder = self.out / "history" / league
        today = self.now().strftime("%Y-%m-%d")
        url = HISTORY_URL.format(league_id=self.cfg["league_ids"][league])
        if not (folder / f"{today}.html.gz").exists():
            self.fetch(url, "history_page", folder / f"{today}.html.gz", check_html, once=False)
        files = sorted(f for f in folder.glob("*.html.gz") if ".suspicious." not in f.name) if folder.exists() else []
        return parse_history(gzip.decompress(files[-1].read_bytes()).decode("utf-8")) if files else {}

    def season_games(self, league: str, season: int, season_id: int):
        """已结束赛季的赛季页（只存一次）→ 比赛列表。"""
        target = self.out / "season" / league / f"{season}.html.gz"
        self.fetch(SEASON_URL.format(season_id=season_id), "season_page", target, check_html, once=True)
        return parse_season_page(gzip.decompress(target.read_bytes()).decode("utf-8")) if target.exists() else []

    def odds_final(self, league: str, season: int, match_id: int, retry_bad=False) -> str:
        """踢完以后的赔率（初盘 + 收盘），只存一次。"""
        target = self.out / "odds_final" / league / str(season) / f"{match_id}.json.gz"
        status, _ = self.fetch(ODDS_URL.format(match_id=match_id), "odds_final", target,
                               lambda c: check_odds(c, match_id), once=True, retry_bad=retry_bad)
        return status

    def odds_live(self, league: str, match_id: int) -> str:
        """赛前赔率快照，变了才存。"""
        target = self.out / "odds_live" / league / str(match_id) / f"{stamp(self.now())}.json.gz"
        status, _ = self.fetch(ODDS_URL.format(match_id=match_id), "odds_live", target,
                               lambda c: check_odds(c, match_id), once=False)
        return status

    def fixtures(self, league: str, start: datetime, until: datetime, max_pages: int | None = None) -> list:
        """从 start 那天起翻赛程，翻到 until 之后或没有下一页为止（最多 max_pages 页，默认 max_fixture_pages）。"""
        league_id = self.cfg["league_ids"][league]
        cursor = start.strftime("%Y-%m-%d 00:00:00")
        seen, out = set(), []
        for page in range(max_pages or self.cfg["max_fixture_pages"]):
            url = FIXTURES_URL.format(league_id=league_id, start=quote(cursor))
            target = self.out / "fixtures" / league / f"{stamp(self.now())}_{page + 1}.json.gz"
            status, content = self.fetch(url, "fixtures", target, check_fixtures, once=False)
            if content is None:
                break
            data = json.loads(content)
            for f in parse_fixtures(data):
                if f.match_id not in seen:
                    seen.add(f.match_id)
                    out.append(f)
            cursor = next_start(data)
            last_ko = max((f.kickoff_utc for f in out if f.kickoff_utc), default=None)
            if cursor is None or (last_ko and last_ko >= until):
                break
        return out

    def _due(self, url: str, gap_hours: float) -> bool:
        """离上次成功抓这个网址（快照）是否已经过了 gap_hours 小时。留 10 分钟余量，免得定时任务稍早几分钟就跳过。"""
        last = self._last(url, "odds_live", ("saved", "unchanged"))
        if not last:
            return True
        return self.now() - datetime.fromisoformat(last[2]) >= timedelta(hours=gap_hours) - timedelta(minutes=10)

    def _final_if_done(self, league, season, f) -> bool:
        """比赛踢完且开球已过 final_after_hours 小时 → 抓最终赔率。返回是否处理了。"""
        if f.status == "Played" and f.kickoff_utc and \
                self.now() - f.kickoff_utc >= timedelta(hours=self.cfg["final_after_hours"]):
            self.odds_final(league, season, f.match_id)
            return True
        return False

    # ---------- 三种模式 ----------
    def upcoming(self, leagues: list[str]):
        """即将开赛：抓赛程；快开球的比赛按“离开球越近越勤”抓赔率快照；刚踢完的抓一次最终赔率。"""
        now = self.now()
        season = current_season(now)
        rules = self.cfg["odds_snapshot_rules"]
        for lg in leagues:
            games = self.fixtures(lg, now - timedelta(days=self.cfg["upcoming_lookback_days"]),
                                  now + timedelta(days=self.cfg["upcoming_ahead_days"]))
            n_live = n_final = 0
            for f in games:
                if self._final_if_done(lg, season, f):
                    n_final += 1
                    continue
                if f.status != "Fixture" or f.kickoff_utc is None or f.kickoff_utc <= now:
                    continue  # 正在踢、延期、或者刚开球还没踢完：不抓快照
                gap = snapshot_gap((f.kickoff_utc - now).total_seconds() / 3600, rules)
                if gap is not None and self._due(ODDS_URL.format(match_id=f.match_id), gap):
                    self.odds_live(lg, f.match_id)
                    n_live += 1
            log.info("%s：赛程 %d 场，赔率快照 %d 场，最终赔率检查 %d 场", lg, len(games), n_live, n_final)

    def backfill(self, leagues: list[str], first: int, last: int | None = None, retry_bad=False):
        """历史回补：已结束赛季用赛季页，当前赛季用赛程接口；每场比赛的最终赔率只抓一次。可以中途停，下次接着来。"""
        now = self.now()
        this = current_season(now)
        last = this if last is None else min(last, this)
        for lg in leagues:
            seasons = self.history(lg)
            for y in range(last, first - 1, -1):  # 从近往远：近的赛季更有用
                if y == this:
                    games = self.fixtures(lg, datetime(y, 7, 1, tzinfo=timezone.utc), now, max_pages=30)  # 一个赛季约 9 页
                    ids = [f.match_id for f in games if f.status == "Played" and f.kickoff_utc
                           and now - f.kickoff_utc >= timedelta(hours=self.cfg["final_after_hours"])]
                elif y in seasons:
                    ids = [g.match_id for g in self.season_games(lg, y, seasons[y])]
                else:
                    log.info("%s %d/%d：历届赛季页里没有这个赛季，跳过", lg, y, y + 1)
                    continue
                before = dict(self.counts)
                for mid in ids:
                    self.odds_final(lg, y, mid, retry_bad=retry_bad)
                new = {k: v - before.get(k, 0) for k, v in self.counts.items() if v - before.get(k, 0)}
                log.info("%s %d/%d：%d 场；这次 %s", lg, y, y + 1, len(ids),
                         "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in new.items()) or "都已有")

    def probe(self, leagues: list[str], first: int, per_season: int) -> list[dict]:
        """试验：每个联赛每个已结束赛季抽几场，看懂球帝有没有赔率、有几家公司。返回每个赛季一行。"""
        rows = []
        for lg in leagues:
            seasons = self.history(lg)
            for y in sorted(s for s in seasons if s >= first):
                games = self.season_games(lg, y, seasons[y])
                stats = {"联赛": lg, "赛季": y, "赛季页比赛数": len(games), "抽查场数": 0, "有赔率场数": 0,
                         "胜平负公司数": 0, "让球公司数": 0, "大小球公司数": 0, "有竞彩官方": 0, "没拿到赔率文件": 0}
                for g in spread(games, per_season):
                    self.odds_final(lg, y, g.match_id)
                    path = self.out / "odds_final" / lg / str(y) / f"{g.match_id}.json.gz"
                    stats["抽查场数"] += 1
                    if not path.exists():  # 网站没有、失败，或返回了别的比赛（内容可疑）
                        stats["没拿到赔率文件"] += 1
                        continue
                    d = read_json(path)
                    if d.get("euro") or d.get("asia") or d.get("size"):
                        stats["有赔率场数"] += 1
                    stats["胜平负公司数"] = max(stats["胜平负公司数"], len(d.get("euro") or []))
                    stats["让球公司数"] = max(stats["让球公司数"], len(d.get("asia") or []))
                    stats["大小球公司数"] = max(stats["大小球公司数"], len(d.get("size") or []))
                    stats["有竞彩官方"] += any(b.get("name") == "竞彩官方" for b in d.get("euro") or [])
                log.info("%s %d/%d：抽 %d 场，有赔率 %d 场，公司数 胜平负 %d / 让球 %d / 大小球 %d，有竞彩 %d 场",
                         lg, y, y + 1, stats["抽查场数"], stats["有赔率场数"], stats["胜平负公司数"],
                         stats["让球公司数"], stats["大小球公司数"], stats["有竞彩官方"])
                rows.append(stats)
        return rows


STATUS_NAMES = {"saved": "已保存", "unchanged": "没变化", "not_found": "网站没有", "suspicious": "内容可疑",
                "failed": "失败", "skipped": "已有，跳过"}


def setup_logging(log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    handlers = [logging.FileHandler(log_dir / "dongqiudi.log", encoding="utf-8")]
    if sys.stderr is not None:  # 定时任务用 pythonw 运行时没有窗口，只写日志文件
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S", handlers=handlers)


def main():
    parser = argparse.ArgumentParser(description="懂球帝采集（只存原始层）")
    parser.add_argument("mode", choices=["upcoming", "probe", "backfill"])
    parser.add_argument("--leagues", nargs="*", help="联赛代码，例如 E0 SP1；不写就是全部五大联赛")
    parser.add_argument("--from", dest="first", type=int, help="从哪个赛季开始（开始年份）")
    parser.add_argument("--to", dest="last", type=int, help="到哪个赛季为止（开始年份），不写就是到当前赛季")
    parser.add_argument("--retry-bad", action="store_true", help="以前“网站没有”“内容可疑”的也重新试一次")
    args = parser.parse_args()

    s = load_settings()
    cfg = s["dongqiudi"]
    setup_logging(resolve_path(s["paths"]["logs"]))
    leagues = args.leagues or list(cfg["league_ids"])
    unknown = [lg for lg in leagues if lg not in cfg["league_ids"]]
    if unknown:
        parser.error(f"这些联赛还没有懂球帝编号：{unknown}（在 settings.yaml 的 dongqiudi.league_ids 里加）")
    con = connect()
    c = Collector(con, resolve_path(s["paths"]["raw"]), cfg)
    try:
        if args.mode == "upcoming":
            c.upcoming(leagues)
        elif args.mode == "backfill":
            c.backfill(leagues, args.first or cfg["backfill_first_season"], args.last, retry_bad=args.retry_bad)
        else:
            import csv
            rows = c.probe(leagues, args.first or cfg["backfill_first_season"], cfg["probe_matches_per_season"])
            out = resolve_path(s["paths"]["reports"]) / "dongqiudi_probe.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="", encoding="utf-8-sig") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            log.info("试验结果写在 %s", out)
    except Blocked as e:
        log.error("懂球帝拒绝访问或返回了拦截页，程序已停止（不会绕过）：%s", e)
        sys.exit(2)
    finally:
        con.close()
        log.info("本次请求：%s", "，".join(f"{STATUS_NAMES[k]} {v}" for k, v in c.counts.items()) or "没有新的请求")


if __name__ == "__main__":
    main()
