r"""读懂球帝的原始文件（需求文档 6.2、第 17 节）。这里只把文件“读对”，不进数据库。

懂球帝的几种原始文件（采集程序 src/collect/dongqiudi.py 存在 data/raw/dongqiudi/ 下，都用 gzip 压缩）：
1. 历届赛季页 /zhanbao/lishi/<联赛编号>：列出每个已结束赛季的编号，例如英超 2023/24 = 21198。
2. 赛季页 /zhanbao/saiji/<赛季编号>：这个赛季全部比赛。每场一行：
       <span class="d">05-19</span><a href="/zhanbao/53615071">切尔西 <b>2-1</b> 伯恩茅斯<!----></a>
   日期是北京时间的“月-日”，没有年份；按“第几轮”分组。
3. 赛程接口（JSON）：当前赛季的比赛列表，start_play 是标准时间（UTC）的开球时间，status 是
   Fixture（没踢）/ Playing（正在踢）/ Played（踢完）。
4. 赔率接口（JSON）：一场比赛各家公司的赔率。euro = 胜平负，asia = 让球，size = 大小球。
   每家公司有 begin（初盘）和 now（最新；比赛踢完后就是收盘，已和 football-data 核对过）。

几个要注意的地方（试验时发现的）：
- 让球和大小球的赔率是“香港盘”写法（不含本金），要加 1 才是我们用的小数赔率。
- 让球盘口的正负号和我们数据库相反：懂球帝 draw_value = -0.5 表示“受半球”（主队受让），
  我们数据库里 line 是主队的让球数，主队让球为负（主队受半球 = +0.5），所以 line = -draw_value。
- 赔率时间 ts 是北京时间“月-日 时:分”，没有年份，要按开球时间推年份（resolve_ts）。
- 请求某些编号时，网站会返回另一场比赛（例如要 3000000 返回 53000000）。
  所以返回的 matchId 必须和请求的一样，不一样就当“内容可疑”（采集程序里检查）。
"""
import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

BEIJING = timezone(timedelta(hours=8))

# 赛季页里的“第几轮”标题和每场比赛
ROUND_RE = re.compile(r'<h2><a href="/zhanbao/lunci/(\d+)-(\d+)">')
GAME_RE = re.compile(
    r'<span class="d">(\d\d)-(\d\d)</span><a href="/zhanbao/(\d+)">\s*(.*?)\s*<b>\s*(\d+)\s*-\s*(\d+)\s*</b>\s*(.*?)\s*(?:<!---->)*</a>')
# 历届赛季页里的赛季链接，例如 <a href="/zhanbao/saiji/21198">2023/24</a>
SEASON_LINK_RE = re.compile(r'<a href="/zhanbao/saiji/(\d+)">(\d{4})/(\d{2})</a>')
TAG_RE = re.compile(r"<[^>]+>")


def read_raw(path: Path) -> bytes:
    """读原始文件；.gz 结尾的先解压。"""
    data = Path(path).read_bytes()
    return gzip.decompress(data) if str(path).endswith(".gz") else data


def read_text(path: Path) -> str:
    return read_raw(path).decode("utf-8")


def read_json(path: Path):
    return json.loads(read_raw(path))


# ---------- 历届赛季页、赛季页 ----------
def parse_history(html: str) -> dict[int, int]:
    """历届赛季页 → {赛季开始年份: 懂球帝赛季编号}，例如 {2023: 21198}。进行中的赛季不在里面。"""
    out = {}
    for sid, y1, y2 in SEASON_LINK_RE.findall(html):
        y1 = int(y1)
        if (y1 + 1) % 100 == int(y2):  # 只认“2023/24”这样跨两年的写法
            out[y1] = int(sid)
    return out


@dataclass
class SeasonGame:
    match_id: int      # 懂球帝的比赛编号
    round: int | None  # 第几轮
    month: int         # 北京时间的月、日（没有年份）
    day: int
    home: str          # 中文队名
    away: str
    home_goals: int
    away_goals: int


def parse_season_page(html: str) -> list[SeasonGame]:
    """赛季页 → 全部比赛（只认有比分的；没踢的比赛没有比分，不在已结束赛季的页面里）。"""
    # 先记下每个“第几轮”标题的位置，比赛属于它前面最近的那个标题
    rounds = [(m.start(), int(m.group(2))) for m in ROUND_RE.finditer(html)]
    games, k, current = [], 0, None
    for m in GAME_RE.finditer(html):
        while k < len(rounds) and rounds[k][0] < m.start():
            current = rounds[k][1]
            k += 1
        mo, d, mid, home, hg, ag, away = m.groups()
        games.append(SeasonGame(int(mid), current, int(mo), int(d), TAG_RE.sub("", home).strip(),
                                TAG_RE.sub("", away).strip(), int(hg), int(ag)))
    return games


def count_game_rows(html: str) -> int:
    """页面里一共有几行比赛（不管有没有比分），用来检查 parse_season_page 有没有漏读。"""
    return html.count('<span class="d">')


# ---------- 赔率 ----------
MARKETS = {"euro": "1x2", "asia": "ah", "size": "ou"}  # 懂球帝的名字 → 我们数据库 odds 表的 market


@dataclass
class OddsRow:
    market: str          # 1x2 / ah / ou
    bookmaker: str       # 懂球帝显示的公司名（大多打了星号，例如 澳*、Bet3**）
    cid: str             # 懂球帝的公司编号（比名字稳定，以后对照公司用这个）
    timing: str          # begin 初盘 / now 最新（踢完后 = 收盘）
    line: float | None   # 让球：主队让球数（我们的写法，主队让球为负）；大小球：总进球线；胜平负：空
    home: float | None   # 胜平负：主胜；让球：主队；大小球：大球。都是小数赔率（含本金）
    draw: float | None   # 只有胜平负有
    away: float | None   # 胜平负：客胜；让球：客队；大小球：小球
    ts_text: str         # 原始时间文字（北京时间，没有年份）


def _num(text) -> float | None:
    """文字 → 数字；空的、0 和认不出的都当没有。"""
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _hk(text) -> float | None:
    """香港盘赔率（不含本金）→ 小数赔率（含本金）。"""
    v = _num(text)
    return round(v + 1, 4) if v is not None else None


def ou_line(text) -> float | None:
    """大小球盘口文字 → 数字：'2.5' → 2.5，'3.5/4' → 3.75（两条线各一半）。"""
    try:
        parts = [float(p) for p in str(text).split("/")]
    except ValueError:
        return None
    return sum(parts) / len(parts) if parts and len(parts) <= 2 else None


def asia_line(draw_value) -> float | None:
    """懂球帝让球数（主队让球为正）→ 我们的写法（主队让球为负）。"""
    try:
        v = float(draw_value)
    except (TypeError, ValueError):
        return None
    return -v if v != 0 else 0.0


def parse_odds(data: dict) -> list[OddsRow]:
    """赔率 JSON → 一行一个（市场, 公司, 初盘/最新）。空的赔率跳过。"""
    rows = []
    for key, market in MARKETS.items():
        for b in data.get(key) or []:
            for timing in ("begin", "now"):
                o = b.get(timing) or {}
                if market == "1x2":
                    home, draw, away, line = _num(o.get("homeWin")), _num(o.get("draw")), _num(o.get("awayWin")), None
                elif market == "ah":
                    home, draw, away, line = _hk(o.get("homeWin")), None, _hk(o.get("awayWin")), asia_line(o.get("draw_value"))
                else:
                    home, draw, away, line = _hk(o.get("homeWin")), None, _hk(o.get("awayWin")), ou_line(o.get("draw"))
                if home is None and away is None:
                    continue
                rows.append(OddsRow(market, b.get("name", ""), str(b.get("cid", "")), timing, line, home, draw, away,
                                    o.get("ts", "")))
    return rows


def resolve_ts(ts_text: str, kickoff_utc: datetime) -> datetime | None:
    """'05-19 21:23'（北京时间，没有年份）→ 标准时间。年份取离开球时间最近的那一年。"""
    m = re.fullmatch(r"(\d\d)-(\d\d) (\d\d):(\d\d)", ts_text or "")
    if not m:
        return None
    mo, d, hh, mm = map(int, m.groups())
    best = None
    for y in (kickoff_utc.year - 1, kickoff_utc.year, kickoff_utc.year + 1):
        try:
            t = datetime(y, mo, d, hh, mm, tzinfo=BEIJING).astimezone(timezone.utc)
        except ValueError:  # 2 月 29 日这样的日期在非闰年不存在
            continue
        if best is None or abs(t - kickoff_utc) < abs(best - kickoff_utc):
            best = t
    return best


# ---------- 赛程接口 ----------
@dataclass
class Fixture:
    match_id: int
    kickoff_utc: datetime | None  # 开球时间（标准时间）
    status: str                   # Fixture / Playing / Played（也可能有延期等别的状态）
    home: str
    away: str
    home_goals: int | None
    away_goals: int | None
    gameweek: str


def parse_fixtures(data: dict) -> list[Fixture]:
    """赛程接口 JSON → 比赛列表。"""
    out = []
    for m in data.get("list") or []:
        try:
            ko = datetime.strptime(m.get("start_play", ""), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            ko = None
        hg, ag = m.get("fs_A"), m.get("fs_B")
        out.append(Fixture(int(m["match_id"]), ko, m.get("status", ""), m.get("team_A_name", ""), m.get("team_B_name", ""),
                           int(hg) if str(hg).isdigit() else None, int(ag) if str(ag).isdigit() else None,
                           str(m.get("gameweek", ""))))
    return out


def next_start(data: dict) -> str | None:
    """赛程接口的“下一页”从哪天开始：nextDate 形如 '2026-11-30 00:00:00next'，原样作为下次的 start。"""
    nd = data.get("nextDate") or ""
    return nd if nd.endswith("next") else None
