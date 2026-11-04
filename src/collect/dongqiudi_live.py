r"""比赛进行中的实时比分（需求文档 6.2“比赛中实时比分”）。只给网站显示用：不存原始层、不进数据库的比赛表、不进模型。

做什么：
- 五大联赛有比赛快开球、正在踢的时候，每个联赛每 2 分钟最多看一次懂球帝的赛程接口
  （和每小时“即将开赛”用的是同一个接口 api.dongqiudi.com/data/tab/league/new/<联赛编号>，robots.txt 允许），
  从里面读出比分、第几分钟、红牌，写进一个小文件 data/live/scores.json，网站读这个文件显示。
- 没有比赛在时间窗口里时，一次都不请求。

什么时候请求（数字都在 settings.yaml 的 dongqiudi.live）：
- 开球时间从数据库只读取（matches 表 + match_sources 表里来源 dongqiudi 的比赛编号，只看 status = scheduled 的）。
- 一个联赛只有在“有比赛处于开球前 before_minutes 分钟到开球后 after_minutes 分钟之间、而且还没确认踢完”时才请求。
  “确认踢完”有三种：数据库里已经是 finished（每小时任务导入的）；这一轮或以前几轮懂球帝已经写 Played（踢完）；
  懂球帝写的是延期、待定之类（不是“正在踢”也不是“未开赛”）。
- 同一个联赛两次请求至少隔 every_seconds 秒（看 fetch_log 里这个联赛上一次实时请求的时间，几个程序一起跑也算得对）。

遵守的规矩（和 src/collect/dongqiudi.py 一样，直接用它的 Collector 发请求，不另写一套）：
- 所有懂球帝请求（每小时任务、回补、这里）合起来每 5 秒最多一次（数据库 request_slots 表排队，Collector 管）。
- 每次请求都记进 fetch_log（来源 dongqiudi，数据类型 fixtures_live），以后可以数一共请求了几次。
- 遇到 403、429 或拦截页：整个停下，scores.json 写 status=paused，pause_minutes（默认 60）分钟内不再请求。
  不重试绕过，不换 UA、不加 cookie、不登录。每小时任务被拒绝时（fetch_log 里有记录）这里也一样停。
- 返回的内容不存原始文件：只为显示，比分以后由每小时任务存的赛程快照和 football-data 正式记录。

怎么运行：
    .venv\Scripts\python -m src.collect.dongqiudi_live once    # 手动跑一轮，打印摘要（中文）
网站服务器在 settings.yaml 设置 live.live_poller: true 时，用后台线程调用 run_forever()。
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from src.clean.dongqiudi_reader import BEIJING, parse_fixtures
from src.collect.dongqiudi import FIXTURES_URL, SOURCE, Blocked, Collector, check_fixtures
from src.common.config import PROJECT_ROOT, load_settings, resolve_path

DATA_TYPE = "fixtures_live"   # 记进 fetch_log 的数据类型（和每小时任务的 fixtures 分开，数次数、数据状态页都不会混）
STATES = ("Playing", "Played", "Fixture")   # 懂球帝的状态：正在踢 / 踢完 / 未开赛。别的（延期、待定等）不显示
MIN_SLEEP_SECONDS = 5         # run_forever 每轮最少睡几秒，防止算错时间时一直空转

# settings.yaml 里没写 dongqiudi.live 时用的默认值（写了就以 settings.yaml 为准）
DEFAULTS = {"before_minutes": 5, "after_minutes": 150, "every_seconds": 120, "pause_minutes": 60,
            "show_played_minutes": 300, "idle_check_minutes": 10, "output": "data/live/scores.json"}

log = logging.getLogger("dongqiudi_live")


class FetchFailed(Exception):
    """这一次没拿到（连不上、服务器出错 3 次、网站说没有）：只跳过这个联赛这一轮，下一轮再来。不是被拒绝。"""


def live_cfg(settings: dict) -> dict:
    """settings.yaml 的 dongqiudi.live，缺的项用 DEFAULTS 补上。"""
    return {**DEFAULTS, **((settings.get("dongqiudi") or {}).get("live") or {})}


def iso(t: datetime) -> str:
    """标准时间写成文字，例如 2026-11-08T15:00:00+00:00（scores.json 里的时间都是这个样子）。"""
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def bj(t: datetime) -> str:
    """显示用的北京时间，例如 11-08 23:00。"""
    return t.astimezone(BEIJING).strftime("%m-%d %H:%M")


def league_url_prefix(league_id) -> str:
    """一个联赛的赛程接口网址的前半段（问号以前），用来在 fetch_log 里找“这个联赛上一次实时请求”。"""
    return FIXTURES_URL.split("?")[0].format(league_id=league_id) + "?"


def live_url(league_id, now: datetime, cfg: dict) -> str:
    """实时请求的网址：和每小时“即将开赛”同一个接口，start 取“开球后 after_minutes 那么久以前”那一天的 0 点。
    这样昨天深夜开球、今天还在踢的比赛也在这一页里（一页大约是前一周到后四周，已经看过存下来的文件）。"""
    day = (now - timedelta(minutes=cfg["after_minutes"])).astimezone(timezone.utc)
    return FIXTURES_URL.format(league_id=league_id, start=quote(day.strftime("%Y-%m-%d 00:00:00")))


# ---------- 发请求：用懂球帝采集程序的 Collector（限速、重试、fetch_log、403/429 停止） ----------
class LiveFetcher:
    """真正的请求和 fetch_log 读写。测试里换成假的（有同样三个方法：last_request、last_block、get）。

    log_db：fetch_log、request_slots 所在的数据库（平时就是主数据库，和每小时任务共用，限速才能合起来算）。
    只有真要发请求时才用可写方式打开；只是看看上一次请求时间时用只读方式打开，不占写锁。
    """

    def __init__(self, settings: dict, log_db: Path | None = None, session=None, sleep=time.sleep,
                 now=None, raw_dir: Path | None = None):
        self.settings = settings
        self.cfg = settings["dongqiudi"]
        self.live = live_cfg(settings)
        self.log_db = Path(log_db) if log_db else resolve_path(settings["paths"]["database"])
        self.session, self.sleep, self.now = session, sleep, now
        # Collector 要一个原始层文件夹，但这里只用它的 _get、_record，不会往里面写文件
        self.raw_dir = raw_dir or resolve_path(settings["paths"]["raw"])
        self.collector = None   # 第一次真要请求时才建（建的时候会打开可写的数据库连接）

    def _read(self, sql: str, args=()):
        """只读查 fetch_log。数据库还没有、表还没有时当作“没有记录”。"""
        if not self.log_db.exists():
            return None
        con = sqlite3.connect(f"{self.log_db.resolve().as_uri()}?mode=ro", uri=True, timeout=60)
        try:
            return con.execute(sql, args).fetchone()
        except sqlite3.OperationalError as err:
            if "no such table" in str(err):
                return None
            raise
        finally:
            con.close()

    def last_request(self, league: str) -> datetime | None:
        """这个联赛上一次实时请求（不管成功失败）是什么时候。"""
        prefix = league_url_prefix(self.cfg["league_ids"][league])
        row = self._read("select max(fetched_at_utc) from fetch_log where source = ? and data_type = ?"
                         " and substr(url, 1, ?) = ?", (SOURCE, DATA_TYPE, len(prefix), prefix))
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def last_block(self, since: datetime) -> tuple[datetime, str] | None:
        """since 以后懂球帝最近一次拒绝访问（403/429，任何程序）或实时请求遇到拦截页：(时间, 原因)。没有返回 None。
        每小时任务的赛程、历届赛季页、赛季页遇到拦截页也算（那几种它自己也会停下）；
        赔率接口“返回了另一场比赛”这种常见的可疑不算。"""
        row = self._read(
            "select fetched_at_utc, http_status, data_type, note from fetch_log where source = ? and fetched_at_utc >= ?"
            " and ((status = 'failed' and http_status in (403, 429))"
            "      or (status = 'suspicious' and data_type in (?, 'fixtures', 'history_page', 'season_page')))"
            " order by fetched_at_utc desc limit 1", (SOURCE, iso(since), DATA_TYPE))
        if not row:
            return None
        why = f"HTTP {row[1]}" if row[1] in (403, 429) else f"拦截页或内容不对（{row[2]}：{row[3] or '内容可疑'}）"
        return datetime.fromisoformat(row[0]), why

    def get(self, league: str, now: datetime) -> dict:
        """请求一个联赛的赛程接口，返回 JSON。不存原始文件，只在 fetch_log 记一行。
        403/429 或拦截页 → 抛 Blocked（调用的地方整个停下）；其他没拿到 → 抛 FetchFailed（只跳过这个联赛这一轮）。"""
        if self.collector is None:
            from src.common.db import connect   # 用到时才打开可写连接
            kwargs = {"sleep": self.sleep}
            if self.session is not None:
                kwargs["session"] = self.session
            if self.now is not None:
                kwargs["now"] = self.now
            self.collector = Collector(connect(self.log_db), self.raw_dir, self.cfg, **kwargs)
        c = self.collector
        url = live_url(self.cfg["league_ids"][league], now, self.live)
        r = c._get(url)   # 和每小时任务同一个限速（每 5 秒最多一次，几个程序合起来算）；连不上时等 30 秒重试，最多 3 次
        if isinstance(r, str):
            c._record(url, DATA_TYPE, "failed", note=r)
            raise FetchFailed(r)
        if r.status_code in (403, 429):
            c._record(url, DATA_TYPE, "failed", http_status=r.status_code, note="网站拒绝访问，实时比分停止")
            raise Blocked(f"HTTP {r.status_code}")
        if r.status_code != 200:
            c._record(url, DATA_TYPE, "not_found" if r.status_code in (404, 410) else "failed", http_status=r.status_code)
            raise FetchFailed(f"HTTP {r.status_code}")
        problem = check_fixtures(r.content)   # 不是 JSON、没有比赛列表：多半是拦截页
        if problem:
            c._record(url, DATA_TYPE, "suspicious", http_status=200, content=r.content, note=problem)
            raise Blocked(problem)
        # 状态记“已保存”是为了让数据状态页的次数表能数到它（那张表只有已保存、没变化等几列）；
        # file_path 是空的：内容只拿来显示，没存文件
        c._record(url, DATA_TYPE, "saved", http_status=200, content=r.content, note="只用于实时比分显示，没存原始文件")
        return json.loads(r.content)

    def close(self):
        if self.collector is not None:
            self.collector.con.close()
            self.collector = None


# ---------- 读数据库：哪些比赛快开球 ----------
def scheduled_matches(db_path: Path, leagues: list[str], now: datetime, cfg: dict) -> list[dict]:
    """数据库里五大联赛还没踢（status = scheduled）、开球时间在“开球后 after_minutes 以前”之后的比赛（只读打开）。
    只要开球时间是准的（date_precision = day）：时间待定的比赛开球时间是占位的，不能拿来判断现在踢没踢。
    返回 [{dqd_id, league, kickoff}]。"""
    if not db_path.exists():
        raise FileNotFoundError("数据库文件不存在")
    con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=60)
    try:
        since = (now - timedelta(minutes=cfg["after_minutes"])).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        rows = con.execute(
            "select s.source_key, m.competition, m.kickoff_utc from matches m"
            " join match_sources s on s.match_id = m.match_id and s.source = ?"
            f" where m.competition in ({','.join('?' * len(leagues))}) and m.status = 'scheduled'"
            " and m.date_precision = 'day' and m.kickoff_utc is not null and m.kickoff_utc >= ?",
            (SOURCE, *leagues, since)).fetchall()
    finally:
        con.close()
    out = []
    for key, league, ko in rows:
        try:   # 数据库里开球时间写成 2026-11-08 15:00（标准时间）
            out.append({"dqd_id": int(key), "league": league,
                        "kickoff": datetime.strptime(ko, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)})
        except (TypeError, ValueError):
            continue
    return out


# ---------- 读懂球帝的返回 ----------
def _int(v, empty=None):
    """'2' → 2；空的、不是数字 → empty。"""
    s = str(v).strip() if v is not None else ""
    return int(s) if s.isdigit() else empty


def minute_text(m: dict) -> str:
    """第几分钟：'63'；补时 '45+2'；中场休息 '中场'；没开始、踢完是空的。"""
    if m.get("status") != "Playing":
        return ""
    if m.get("minute_period") == "HT":
        return "中场"
    base = str(m.get("minute") or "").strip()
    extra = str(m.get("minute_extra") or "").strip()
    return f"{base}+{extra}" if base and extra else base


RED_CODES = ("RC", "Y2C")   # 事件里的红牌：直接红牌、两黄变一红


def red_cards(m: dict, side: str) -> int:
    """一队几张红牌。懂球帝有 rc_A / rc_B 两个字段，但 2026-11-04 查原始层里两万多条（含 98 条进行中）全是空的；
    红牌其实在事件列表 team_A_events / team_B_events 里（code 是 RC 或 Y2C，452 场踢完的比赛里有 78 张，同一个人不会两种都记）。
    所以数事件里的红牌；万一以后 rc 字段有数了，取两个里大的。"""
    events = m.get(f"team_{side}_events")
    n = sum(1 for e in events if isinstance(e, dict) and e.get("code") in RED_CODES) if isinstance(events, list) else 0
    return max(n, _int(m.get(f"rc_{side}"), 0))


def parse_live(data: dict, league: str) -> tuple[list[dict], dict[int, str]]:
    """赛程接口 JSON → (scores.json 格式的比赛, {比赛编号: 懂球帝状态})。
    比赛只留 Playing / Played / Fixture 三种；状态表包括全部（延期、待定也在里面，用来判断“不用再看”）。
    比分、开球时间、队名用和每小时任务一样的 parse_fixtures 读；分钟、红牌它不读，这里从原始字段补。"""
    raw = {}
    for m in data.get("list") or []:
        try:
            raw[int(m["match_id"])] = m
        except (KeyError, TypeError, ValueError):
            continue
    games, states = [], {}
    for f in parse_fixtures({"list": list(raw.values())}):
        states[f.match_id] = f.status
        if f.status not in STATES or f.kickoff_utc is None:
            continue
        m = raw[f.match_id]
        started = f.status != "Fixture"
        games.append({"dqd_id": f.match_id, "league": league, "home": f.home, "away": f.away,
                      "kickoff_utc": iso(f.kickoff_utc), "state": f.status, "minute": minute_text(m),
                      "home_goals": f.home_goals if started else None, "away_goals": f.away_goals if started else None,
                      # 红牌从事件里数（见 red_cards）。比分用 fs_A / fs_B：进行中时事件列表偶尔比比分慢一点
                      "home_red": red_cards(m, "A"), "away_red": red_cards(m, "B")})
    return games, states


def in_window(g: dict, now: datetime, cfg: dict) -> bool:
    """这场比赛现在该不该显示：
    未开赛 → 开球前 before_minutes 到开球后 after_minutes（晚开球的也留着）；
    正在踢 → 开球后 show_played_minutes 以内（防止旧数据一直挂着）；
    踢完 → 开球后 show_played_minutes 以内（默认 300 分钟 ≈ 踢完以后再留 3 小时左右）。
    格式不对的当作不显示。"""
    try:
        ko = datetime.fromisoformat(g["kickoff_utc"])
        state = g["state"]
    except (KeyError, TypeError, ValueError):
        return False
    if state == "Fixture":
        return ko - timedelta(minutes=cfg["before_minutes"]) <= now <= ko + timedelta(minutes=cfg["after_minutes"])
    if state in ("Playing", "Played"):
        return ko - timedelta(minutes=cfg["before_minutes"]) <= now <= ko + timedelta(minutes=cfg["show_played_minutes"])
    return False


# ---------- 写 scores.json ----------
def write_scores(path: Path, payload: dict):
    """先写到同一个文件夹里的临时文件，再一次性改名成 scores.json：网站任何时候读到的都是完整的一份。
    Windows 上网站正好在读这个文件时改名可能失败，等一下再试几次；最后还是不行就报错（旧文件原样留着）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2)
    finally:
        if os.path.exists(tmp):   # 改名成功后临时文件就不在了；失败时删掉，不留垃圾
            os.remove(tmp)


def read_previous(path: Path) -> list[dict]:
    """上一次写的 scores.json 里的比赛（用来在“这一轮这个联赛没请求”时接着显示，以及记住哪些已经踢完）。
    文件不在、坏了 → 当作没有。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        games = data.get("matches") if isinstance(data, dict) else None
        return [g for g in games if isinstance(g, dict)] if isinstance(games, list) else []
    except (OSError, ValueError):
        return []


def payload(now: datetime, status: str, note: str, matches: list[dict]) -> dict:
    """scores.json 的内容（格式和网站服务器约好了，字段名不改）。"""
    return {"updated_at": iso(now), "status": status, "note": note, "matches": matches}


# ---------- 一轮 ----------
def tick(settings: dict, now: datetime | None = None, fetcher=None, memory: dict | None = None) -> dict:
    """做一轮：看哪些联赛要请求、请求、写 scores.json。返回摘要（字典）。任何意外都不往外抛：记日志、写 status。

    fetcher：发请求的东西，不给就用 LiveFetcher（真的请求）；测试里给假的。
    memory：run_forever 每轮传同一个字典进来，记住“这一场已经确认踢完 / 不用再看”（只在内存里）。
    """
    now = now or datetime.now(timezone.utc)
    cfg = live_cfg(settings)
    out_path = resolve_path(cfg["output"])
    memory = memory if memory is not None else {}
    done = memory.setdefault("done", set())   # 已经确认不用再看的比赛编号
    own = fetcher is None
    summary = {"status": None, "note": "", "requests": 0, "requested": [], "waiting": [], "failed": [],
               "matches": 0, "next_check_utc": None, "output": out_path}
    try:
        fetcher = fetcher or LiveFetcher(settings)   # 放在 try 里：设置缺项等也算“这一轮出错”，不往外抛
        _tick(settings, now, cfg, fetcher, done, out_path, summary)
    except Exception as err:   # 意外（数据库读不了、磁盘满了等）：不让后台线程死掉，记下来，下一轮再来
        log.exception("实时比分这一轮出错")
        summary.update(status="off", note=f"这一轮出错（{type(err).__name__}），过一会儿自动再试",
                       next_check_utc=now + timedelta(seconds=max(60, cfg["every_seconds"])))
        try:
            write_scores(out_path, payload(now, "off", summary["note"], []))
        except Exception:
            log.exception("scores.json 也写不了")
    finally:
        if own and fetcher is not None:
            fetcher.close()
    return summary


def _tick(settings, now, cfg, fetcher, done, out_path, summary):
    leagues = list(settings["dongqiudi"]["league_ids"])   # 只看五大联赛
    every = timedelta(seconds=cfg["every_seconds"])
    before, after = timedelta(minutes=cfg["before_minutes"]), timedelta(minutes=cfg["after_minutes"])
    previous = read_previous(out_path)
    # 上一份文件里已经“踢完”的，不用再看（程序重新启动、手动跑 once 时也记得）
    done.update(g["dqd_id"] for g in previous if g.get("state") == "Played" and isinstance(g.get("dqd_id"), int))

    # 1. 最近被网站拒绝过：pause_minutes 内一次都不请求
    pause = timedelta(minutes=cfg["pause_minutes"])
    blocked = fetcher.last_block(now - pause)
    if blocked:
        t, why = blocked
        resume = t + pause
        note = f"懂球帝 {bj(t)} 拒绝访问或返回拦截页（{why}），暂停到 {bj(resume)}（北京时间），不绕过"
        summary.update(status="paused", note=note, next_check_utc=resume)
        write_scores(out_path, payload(now, "paused", note, []))
        return

    # 2. 哪些联赛现在要看：有比赛在“开球前 before 到开球后 after”之间，而且还没确认踢完
    active, upcoming = {}, []
    for m in scheduled_matches(resolve_path(settings["paths"]["database"]), leagues, now, cfg):
        if m["dqd_id"] in done:
            continue
        if m["kickoff"] - before <= now <= m["kickoff"] + after:
            active.setdefault(m["league"], set()).add(m["dqd_id"])
        elif m["kickoff"] - before > now:
            upcoming.append(m["kickoff"] - before)

    # 3. 每个要看的联赛：离上次请求够 every_seconds 才请求
    fresh, next_times = {}, []
    for lg in leagues:
        if lg not in active:
            continue
        last = fetcher.last_request(lg)
        if last is not None and now - last < every:
            summary["waiting"].append(lg)
            next_times.append(last + every)
            continue
        next_times.append(now + every)
        try:
            data = fetcher.get(lg, now)
        except Blocked as err:
            resume = now + pause
            note = f"懂球帝拒绝访问或返回拦截页（{err}），暂停到 {bj(resume)}（北京时间），不绕过"
            log.error("实时比分：%s", note)
            summary.update(status="paused", note=note, next_check_utc=resume)
            summary["requests"] += 1
            write_scores(out_path, payload(now, "paused", note, []))
            return
        except FetchFailed as err:
            summary["requests"] += 1
            summary["failed"].append(lg)
            log.warning("实时比分：%s 这一轮没拿到（%s），下一轮再试", lg, err)
            continue
        summary["requests"] += 1
        summary["requested"].append(lg)
        games, states = parse_live(data, lg)
        for mid in active[lg]:
            if mid not in states:   # 返回的这一页里没有这场（一页约前一周到后四周，正常不会发生）
                log.warning("实时比分：%s 第 %d 场不在返回的赛程里，这次不再看", lg, mid)
                done.add(mid)
            elif states[mid] not in ("Playing", "Fixture"):   # 踢完了，或者延期、待定：不用再看
                done.add(mid)
        fresh[lg] = [g for g in games if in_window(g, now, cfg)]

    # 4. 拼出要显示的比赛：这一轮请求了的用新的；没请求的（等间隔、没拿到、已经踢完不用看）接着用上一份里还在窗口里的
    shown = []
    for lg in leagues:
        if lg in fresh:
            shown += fresh[lg]
        else:
            shown += [g for g in previous if g.get("league") == lg and in_window(g, now, cfg)]
    shown.sort(key=lambda g: (str(g.get("kickoff_utc")), str(g.get("league")), str(g.get("dqd_id"))))

    status = "ok" if active or shown else "off"
    if status == "off":
        nxt = min(upcoming, default=None)
        note = "现在没有比赛在时间窗口里，不请求" + (f"；下一场 {bj(nxt + before)}（北京时间）开球" if nxt else "")
    else:
        parts = []
        if summary["requested"]:
            parts.append("这一轮更新了 " + "、".join(summary["requested"]))
        if summary["waiting"]:
            parts.append("、".join(summary["waiting"]) + f" 离上次不到 {cfg['every_seconds']} 秒，用上一份")
        if summary["failed"]:
            parts.append("、".join(summary["failed"]) + " 这一轮没拿到，用上一份，下一轮再试")
        note = "；".join(parts) or "比赛都已踢完，显示最后比分"
    write_scores(out_path, payload(now, status, note, shown))

    # 5. 下一次什么时候再看：最早的“联赛间隔到了”或“下一场进入窗口”，最多隔 idle_check_minutes（数据库可能刚导入新赛程）
    candidates = next_times + upcoming + [now + timedelta(minutes=cfg["idle_check_minutes"])]
    summary.update(status=status, note=note, matches=len(shown), next_check_utc=min(candidates))


# ---------- 一直跑（网站服务器的后台线程用） ----------
def run_forever(settings: dict, stop_event, make_fetcher=None, clock=None):
    """一轮接一轮：每轮做完睡到下一次该看的时间（stop_event.set() 可以马上叫醒、退出）。
    任何意外都只记日志、写 status，然后睡一会儿再来，线程不会死掉。
    make_fetcher、clock 只给测试用：造假的请求程序、假的现在时间。"""
    cfg = live_cfg(settings)
    memory = {}
    log.info("实时比分后台线程开始")
    while not stop_event.is_set():
        wait = 60.0
        try:
            now = clock() if clock else datetime.now(timezone.utc)
            summary = tick(settings, now, make_fetcher() if make_fetcher else None, memory)
            nxt = summary.get("next_check_utc")
            if nxt is not None:
                wait = (nxt - (clock() if clock else datetime.now(timezone.utc))).total_seconds()
        except Exception:   # tick 自己已经接住了意外；这里再保险一次（例如造请求程序出错）
            log.exception("实时比分后台线程出错，60 秒后再试")
        wait = min(max(wait, MIN_SLEEP_SECONDS), cfg["idle_check_minutes"] * 60)
        stop_event.wait(wait)
    log.info("实时比分后台线程停止")


# ---------- 手动运行 ----------
def describe(summary: dict) -> str:
    """把一轮的摘要写成中文。"""
    out = summary["output"]
    shown_path = out.relative_to(PROJECT_ROOT).as_posix() if out.is_relative_to(PROJECT_ROOT) else out.name
    lines = [f"状态：{summary['status']}（{summary['note']}）",
             f"这一轮请求懂球帝 {summary['requests']} 次"
             + (f"：更新了 {'、'.join(summary['requested'])}" if summary["requested"] else ""),
             f"显示的比赛：{summary['matches']} 场，写在 {shown_path}"]
    if summary["waiting"]:
        lines.append(f"离上次请求不到间隔、这轮没请求：{'、'.join(summary['waiting'])}")
    if summary["failed"]:
        lines.append(f"没拿到（下一轮再试）：{'、'.join(summary['failed'])}")
    if summary["next_check_utc"]:
        lines.append(f"下一次该看：{bj(summary['next_check_utc'])}（北京时间）")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="懂球帝实时比分（只写 data/live/scores.json，给网站显示）")
    parser.add_argument("mode", choices=["once"], help="once：手动跑一轮并打印摘要")
    parser.parse_args()
    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    handlers = [logging.FileHandler(log_dir / "dongqiudi_live.log", encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S", handlers=handlers)
    print(describe(tick(s)))


if __name__ == "__main__":
    main()
