r"""把数据库里维基百科来的杯赛、欧战比赛，和两个开放数据集交叉核对（需求文档第 16 节“数据质量”）。

为什么要核对：维基百科网页是人写的，格式五花八门，解析程序可能读错日期、比分、主客队。
拿另外两份独立整理的数据对一遍，就知道解析得准不准、错在哪。

  - engsoccerdata（GPL）：欧冠 2000/01–2017/18、英足总杯 2000/01–2018/19、英联赛杯 2000/01–2018/19
  - openfootball（CC0）：欧冠 2011/12 起、欧联 2020/21 起、欧协联 2021/22 起、英足总杯 / 英联赛杯 2018/19–2024/25、
    德国杯 2018/19 起、国王杯 / 意大利杯 2020/21–2024/25
  （下载程序：src/collect/crosscheck_sources.py；文件在 data/raw/engsoccerdata、data/raw/openfootball）

怎么核对：
  1. 读参照数据里的每一场比赛，只留“至少一方是我们的球队”的（和数据库的范围一样）。
  2. 在数据库里找同一赛事、日期相近（精确到天的差 1 天以内；只知道轮次的差 10 天以内）、队名相像的比赛。
     队名在不同数据集里写法不一样（“Arsenal FC”、“Arsenal”），所以比较的是去掉 FC、重音、标点以后的词。
  3. 找到了就比：日期一样吗、90 分钟比分一样吗、加时比分 / 点球一样吗、主客队有没有反。
  4. 反过来，参照数据覆盖的赛季里，数据库有、参照数据没有的比赛也列出来。

结果写进 data/reports/wikipedia_交叉核对.md 和 wikipedia_交叉核对_不一致清单.txt。
运行方法：Claude 运行 `.venv\Scripts\python -m src.clean.wikipedia_crosscheck`（先跑过 src.clean.wikipedia_import）。
"""
import collections
import csv
import datetime as dt
import difflib
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from src.clean.wikipedia_import import load_team_map
from src.common.config import load_settings, resolve_path
from src.common.db import connect

log = logging.getLogger("wikipedia_crosscheck")

# 参照数据里的国家代码 → teams.yaml 的国家
CODE_COUNTRY = {"ENG": "英格兰", "WAL": "英格兰", "GER": "德国", "ITA": "意大利", "ESP": "西班牙", "FRA": "法国", "MON": "法国"}
COMP_COUNTRY = {"FAC": "英格兰", "ELC": "英格兰", "DFB": "德国", "CDR": "西班牙", "CI": "意大利", "CDF": "法国"}


@dataclass
class RefMatch:
    """参照数据里的一场比赛。比分不知道的写 None。"""
    source: str                 # engsoccerdata / openfootball
    competition: str            # 我们的赛事代码（UCL、FAC……）
    season: int
    date: dt.date
    home: str
    away: str
    home_country: str = ""      # 参照数据写了国家代码时，换成 teams.yaml 的国家
    away_country: str = ""
    ft: tuple | None = None     # 90 分钟比分
    aet: tuple | None = None    # 加时后比分
    final: tuple | None = None  # 最后比分（不知道含不含加时的时候只有这个）
    pens: tuple | None = None   # 点球大战
    round_text: str = ""
    line: str = ""              # 原始文件里的位置，方便查

    et_known: bool = True       # True = 参照数据明确说了有没有加时；False = 说不清（例如英联赛杯只有一个比分）
    awarded: bool = False


# ---------------------------------------------------------------- 读参照数据

def _pair(text) -> tuple | None:
    """“3-1” → (3, 1)；NA、空、“away goals” 这种不是比分的 → None。"""
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _na(v) -> bool:
    return v is None or str(v).strip() in ("", "NA")


def read_engsoccerdata(folder: Path, files: dict, first: int) -> list[RefMatch]:
    """engsoccerdata 的三个 CSV。三个文件的列不一样：
      champs.csv    FT = 90 分钟比分，aet = 加时后比分，pens = 点球（有时写 “away goals”，那不是点球）
      facup.csv     aet = yes 时 FT 是加时后的比分（90 分钟比分不知道）；pen = yes 时 hp、vp 是点球比分
      leaguecup.csv 2000 年以后 aet、pens 都是空的，FT 可能含加时也可能不含（说不清）
    """
    out = []
    for code, name in files.items():
        path = folder / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for i, r in enumerate(csv.DictReader(f), start=2):
                if _na(r.get("Date")) or int(r["Season"]) < first:
                    continue
                if not _na(r.get("nonmatch")):
                    continue                     # 不是真比赛（例如对手弃权）
                m = RefMatch(source="engsoccerdata", competition=code, season=int(r["Season"]),
                             date=dt.date.fromisoformat(r["Date"]), home=r["home"], away=r["visitor"],
                             round_text=r.get("round", ""), line=f"{name}:{i}")
                if code == "UCL":
                    m.home_country = CODE_COUNTRY.get(r.get("hcountry", ""), r.get("hcountry", ""))
                    m.away_country = CODE_COUNTRY.get(r.get("vcountry", ""), r.get("vcountry", ""))
                    m.ft = _pair(r["FT"])
                    m.aet = None if _na(r.get("aet")) else _pair(r["aet"])
                    m.pens = None if _na(r.get("pens")) else _pair(r["pens"])
                elif code == "FAC":
                    if str(r.get("aet", "")).strip() == "yes":
                        m.aet = _pair(r["FT"])
                    else:
                        m.ft = _pair(r["FT"])
                    if str(r.get("pen", "")).strip() == "yes" and not _na(r.get("hp")):
                        m.pens = (int(float(r["hp"])), int(float(r["vp"])))
                else:
                    m.final, m.et_known = _pair(r["FT"]), False
                out.append(m)
    return out


_WEEKDAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
_MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_OF_DATE = re.compile(r"^\s*\[?(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})/?\s*(\d{1,2})(?:\s+(\d{4}))?\]?\s*$")
_OF_SCORE = re.compile(r"(?<!\S)(\d+)-(\d+)(?!\S)")
_OF_CODE = re.compile(r"\s*\(([A-Z]{3})\)\s*$")


def _of_year(weekday: int, month: int, day: int, season: int, prev: dt.date | None) -> dt.date | None:
    """openfootball 的日期行常常不写年份（“Wed Oct 3”）：用星期几挑出是赛季里哪一年。"""
    fits = []
    for y in (season, season + 1):
        try:
            d = dt.date(y, month, day)
        except ValueError:
            continue
        if d.weekday() == weekday:
            fits.append(d)
    if len(fits) == 1:
        return fits[0]
    if prev is not None:      # 两年都对得上（或都对不上）：接着上一个日期往后排
        for y in (prev.year, prev.year + 1):
            try:
                d = dt.date(y, month, day)
            except ValueError:
                continue
            if d >= prev:
                return d
    return fits[0] if fits else None


def parse_openfootball_score(part: str) -> dict | None:
    """“3-5 pen. 1-1 a.e.t. (1-1, 1-0)” 这种写法 → 比分。规则（Football.TXT 格式）：
      A-B (半场)                      90 分钟 A-B，没有加时
      A-B a.e.t. (90 分钟, 半场)      加时后 A-B
      A-B pen. C-D a.e.t. (90 分钟…)  点球 A-B，加时后 C-D
      A-B pen. (x-y…)                 点球 A-B；有没有加时说不清，x-y 当 90 分钟比分
    """
    if "[cancelled]" in part or "[postponed]" in part:
        return None
    m = _OF_SCORE.match(part.strip())
    if not m:
        return None
    first = (int(m.group(1)), int(m.group(2)))
    rest = part.strip()[m.end():].strip()
    out = {"ft": None, "aet": None, "pens": None, "et_known": True, "awarded": "[awarded]" in part}
    paren = re.search(r"\((\d+)-(\d+)", rest)
    paren_score = (int(paren.group(1)), int(paren.group(2))) if paren else None
    if rest.startswith("pen."):
        out["pens"] = first
        rest = rest[4:].strip()
        m2 = re.match(r"(\d+)-(\d+)\s*a\.e\.t\.", rest)
        if m2:
            out["aet"] = (int(m2.group(1)), int(m2.group(2)))
        else:
            out["et_known"] = False
        out["ft"] = paren_score
    elif rest.startswith("a.e.t."):
        out["aet"], out["ft"] = first, paren_score
    else:
        out["ft"] = first
    return out


def _split_team(text: str) -> tuple[str, str]:
    """“Arsenal FC (ENG)” → (“Arsenal FC”, “英格兰”)；没写国家代码的国家是空的。"""
    text = text.strip()
    m = _OF_CODE.search(text)
    if not m:
        return text, ""
    return text[:m.start()].strip(), CODE_COUNTRY.get(m.group(1), m.group(1))


def read_openfootball_file(path: Path, code: str, season: int) -> list[RefMatch]:
    """读一个 openfootball 的 Football.TXT 文件。每场比赛一行：
         [开球时间]  主队 v 客队  比分 [@ 球场]
       上面某一行是日期（“Tue Sep 18 2018”或“Wed Oct 3”），再上面 “▪ Group A” 是轮次。"""
    out = []
    day: dt.date | None = None
    rnd = ""
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip() or line.startswith("="):
            continue
        if line.lstrip().startswith("▪"):
            rnd = line.strip("▪ ").strip()
            continue
        dm = _OF_DATE.match(line)
        if dm:
            wd, mon, d, y = _WEEKDAYS[dm.group(1)], _MON[dm.group(2)], int(dm.group(3)), dm.group(4)
            day = dt.date(int(y), mon, d) if y else _of_year(wd, mon, d, season, day)
            continue
        if day is None:
            continue
        body = re.sub(r"^\s*\d{1,2}[:.]\d{2}\s+", "", line)        # 去掉开球时间
        body = re.split(r"\s+@\s+", body)[0]                         # 去掉球场
        if " v " in body:
            home_part, away_part = body.split(" v ", 1)
            # 客队名和比分之间至少两个空格；没有比分的（未赛、取消）没有这一段
            pieces = re.split(r"\s{2,}", away_part.strip(), maxsplit=1)
            if len(pieces) < 2:
                continue
            away_name, score_part = pieces
        else:
            # 另一种写法（早几年的国内杯赛文件）：主队  比分  客队，中间至少两个空格
            pieces = re.split(r"\s{2,}", body.strip())
            if len(pieces) != 3 or not _OF_SCORE.match(pieces[1]):
                continue
            home_part, score_part, away_name = pieces
        sc = parse_openfootball_score(score_part)
        if sc is None:
            continue
        home, hc = _split_team(home_part)
        away, ac = _split_team(away_name)
        out.append(RefMatch(source="openfootball", competition=code, season=season, date=day,
                            home=home, away=away, home_country=hc, away_country=ac,
                            ft=sc["ft"], aet=sc["aet"], pens=sc["pens"], et_known=sc["et_known"],
                            awarded=sc["awarded"], round_text=rnd,
                            line=f"{path.parent.parent.name}/{path.parent.name}/{path.name}:{i}"))
    return out


def read_openfootball(folder: Path, files: dict) -> list[RefMatch]:
    out = []
    for code, (_repo, name, first, last) in files.items():
        for y in range(first, last + 1):
            path = folder / code / str(y) / name
            if path.exists():
                out.extend(read_openfootball_file(path, code, y))
    return out


# ---------------------------------------------------------------- 队名比较

_DROP = {"fc", "afc", "cf", "ac", "as", "ss", "ssc", "acf", "sc", "sv", "fk", "sk", "cd", "ud", "sd", "rcd", "club",
         "football", "futbol", "calcio", "de", "del", "la", "the", "f", "c", "1", "spa", "cfc", "bc", "us", "vfl", "vfb",
         "tsv", "fsv", "bsc", "tsg", "spvgg", "osc", "sad", "aia"}
_SAME = {"utd": "united", "munchen": "munich", "internazionale": "inter", "koln": "cologne", "athletico": "atletico",
         "st": "saint", "olympiakos": "olympiacos", "psg": "paris saint germain", "man": "manchester",
         "wolves": "wolverhampton wanderers", "spurs": "tottenham hotspur", "nikosia": "nicosia",
         "sporting": "sporting", "atl": "atletico", "ath": "athletic"}


def name_key(name: str) -> tuple[str, ...]:
    """队名 → 用来比较的词（去重音、小写、去标点、去 FC 这类词、去年份）。“Bayern München” → (bayern, munich)。"""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"[.'’&/\-]", " ", n)
    words = []
    for w in n.split():
        if w in _DROP or re.fullmatch(r"(18|19|20)\d{2}|\d{2}", w):
            continue
        words.extend(_SAME.get(w, w).split())
    return tuple(words)


def name_similarity(a: tuple, b: tuple) -> float:
    """两个队名（name_key 的结果）有多像，0–1。一样是 1；一个是另一个的一部分（“Dortmund” 和 “Borussia Dortmund”）
    按多出来的词扣一点；其他情况看字母有多像。"""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sa, sb = set(a), set(b)
    ratio = difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    if sa <= sb or sb <= sa:
        small, big = min(len(sa), len(sb)), max(len(sa), len(sb))
        return max(ratio, 0.95 - 0.1 * (big - small))
    return ratio


class Names:
    """我们的球队在各数据集里可能的写法（teams.yaml 的英文名、别名、维基百科条目名），用来判断参照数据里的队是不是我们的。"""

    def __init__(self, settings: dict):
        tm = load_team_map(settings)
        self.country = {k: v["国家"] for k, v in tm.teams.items()}
        self.keys: dict[str, set] = collections.defaultdict(set)     # 球队编号 → 写法
        for team, t in tm.teams.items():
            for n in [team, t["英文"], *t.get("别名", [])]:
                self.keys[team].add(name_key(n))
        for title, team in tm.alias.items():
            self.keys[team].add(name_key(title))
        self.by_key: dict[tuple, set] = collections.defaultdict(set)
        for team, ks in self.keys.items():
            for k in ks:
                self.by_key[k].add(team)
        self._cache: dict[tuple, float] = {}

    def ours(self, name: str, country: str, comp_country: str) -> str | None:
        """参照数据里的这个队是我们的哪支球队（写法完全一样、国家也对得上才算）；不是就返回 None。"""
        want = country or comp_country
        for team in self.by_key.get(name_key(name), ()):
            if not want or want == self.country[team]:
                return team
        return None

    def similarity(self, ref_name: str, db_team: str, db_title: str) -> float:
        """参照数据的队名和数据库里的一支队有多像。数据库里的队：我们的球队看所有写法，外来球队看维基百科条目名。"""
        k = (ref_name, db_team)
        if k not in self._cache:
            rk = name_key(ref_name)
            cands = self.keys.get(db_team) or {name_key(db_title)}
            self._cache[k] = max(name_similarity(rk, c) for c in cands)
        return self._cache[k]


# ---------------------------------------------------------------- 数据库里的比赛

@dataclass
class DbMatch:
    match_id: int
    competition: str
    season: int
    date: dt.date
    precision: str
    round: str
    leg: int
    home: str
    away: str
    status: str
    ft: tuple | None
    aet: tuple | None
    pens: tuple | None
    extra_time: bool


def load_db(con: sqlite3.Connection, codes: set[str]) -> list[DbMatch]:
    q = f"""
        select m.match_id, m.competition, m.season, m.match_date, m.date_precision, m.round, m.leg,
               m.home_team, m.away_team, m.status, s.home_goals, s.away_goals,
               c.extra_time, c.aet_home, c.aet_away, c.pen_home, c.pen_away
        from matches m
        join match_sources ms on ms.match_id = m.match_id and ms.source = 'wikipedia'
        left join match_stats s on s.match_id = m.match_id and s.source = 'wikipedia'
        left join cup_details c on c.match_id = m.match_id and c.source = 'wikipedia'
        where m.competition in ({",".join("?" * len(codes))})"""
    out = []
    for r in con.execute(q, sorted(codes)):
        ft = (r[10], r[11]) if r[10] is not None else None
        aet = (r[13], r[14]) if r[13] is not None else None
        pens = (r[15], r[16]) if r[15] is not None else None
        out.append(DbMatch(r[0], r[1], r[2], dt.date.fromisoformat(r[3]), r[4], r[5], r[6], r[7], r[8], r[9],
                           ft, aet, pens, bool(r[12])))
    return out


def _title(team_id: str) -> str:
    return team_id[5:] if team_id.startswith("wiki:") else team_id


# ---------------------------------------------------------------- 配对、比较

MIN_SIMILARITY = 0.75   # 两队队名都至少这么像，才认为是同一场比赛
SWAP_PENALTY = 0.05     # 主客队反过来才对得上时扣一点（同分时优先认主客一样的）


def window_days(precision: str) -> int:
    """日期差多少天以内算“可能是同一场”：数据库日期精确到天的差 1 天（时区、跨午夜），只知道轮次的差 10 天。"""
    return 1 if precision == "day" else 10


def pair_up(refs: list[RefMatch], dbs: list[DbMatch], names: Names) -> list[tuple[RefMatch, DbMatch, bool]]:
    """把参照数据的比赛和数据库的比赛一对一配起来。返回 [(参照, 数据库, 主客是否反了)]。
    先算所有可能的配对和像的程度，再从最像的开始配，配过的两边都不再用。
    还剩下的再配一轮：同一天、至少一支队像、比分一样（另一支队的写法差太多，例如俄文转写）。"""
    by_day: dict[dt.date, list[DbMatch]] = collections.defaultdict(list)
    for d in dbs:
        by_day[d.date].append(d)
    edges = []
    for ri, r in enumerate(refs):
        for off in range(-10, 11):
            for d in by_day.get(r.date + dt.timedelta(days=off), ()):
                if abs(off) > window_days(d.precision):
                    continue
                hh = names.similarity(r.home, d.home, _title(d.home))
                aa = names.similarity(r.away, d.away, _title(d.away))
                straight = min(hh, aa)
                ha = names.similarity(r.home, d.away, _title(d.away))
                ah = names.similarity(r.away, d.home, _title(d.home))
                swapped = min(ha, ah) - SWAP_PENALTY
                score, sw = (straight, False) if straight >= swapped else (swapped, True)
                if score >= MIN_SIMILARITY:
                    edges.append((-score, abs(off), ri, d.match_id, sw, d))
    edges.sort(key=lambda e: e[:4])
    used_r, used_d, out = set(), set(), []
    for _neg, _off, ri, mid, sw, d in edges:
        if ri in used_r or mid in used_d:
            continue
        used_r.add(ri)
        used_d.add(mid)
        out.append((refs[ri], d, sw))
    # 第二轮：剩下没配上的，同一天、一支队像、比分也一样，就算同一场。
    # 例子：参照写 “CSKA Moskva”，维基百科条目名是 “PFC CSKA Moscow”，名字相似度不够，但同一天 Inter 1-2 这支队，
    # 只能是同一场比赛
    second = []
    for ri, r in enumerate(refs):
        if ri in used_r:
            continue
        rs = {x for x in (r.ft, r.final, r.aet) if x}
        for d in by_day.get(r.date, ()):
            if d.match_id in used_d or d.precision != "day":
                continue
            for sw in (False, True):
                dh, da = (d.away, d.home) if sw else (d.home, d.away)
                ds = {(_flip(x) if sw else x) for x in (d.ft, d.aet) if x}
                best = max(names.similarity(r.home, dh, _title(dh)), names.similarity(r.away, da, _title(da)))
                if best >= MIN_SIMILARITY and rs & ds:
                    second.append((-best, sw, ri, d.match_id, d))
    second.sort(key=lambda e: e[:4])
    for _neg, sw, ri, mid, d in second:
        if ri in used_r or mid in used_d:
            continue
        used_r.add(ri)
        used_d.add(mid)
        out.append((refs[ri], d, sw))
    return out


def _flip(p):
    return (p[1], p[0]) if p else p


def compare(r: RefMatch, d: DbMatch, swapped: bool) -> list[str]:
    """比较一对比赛，返回不一样的地方（空 = 完全一致）。数据库的比分先按参照数据的主客方向摆好。"""
    ft, aet, pens = (_flip(d.ft), _flip(d.aet), _flip(d.pens)) if swapped else (d.ft, d.aet, d.pens)
    out = []
    if swapped:
        out.append("主客队反了")
    if d.precision == "day" and d.date != r.date:
        out.append(f"日期不一样（数据库 {d.date}，参照 {r.date}）")
    if d.status == "scheduled":
        out.append("数据库里还没比分")
        return out
    if r.awarded or d.status == "awarded":
        return out                                  # 判罚比分：各数据集写法不一样，不比
    if r.ft and ft and r.ft != ft:
        out.append(f"90 分钟比分不一样（数据库 {ft[0]}-{ft[1]}，参照 {r.ft[0]}-{r.ft[1]}）")
    if r.aet and not d.extra_time:
        out.append(f"参照有加时（{r.aet[0]}-{r.aet[1]}），数据库没有")
    elif r.aet and aet and r.aet != aet:
        out.append(f"加时比分不一样（数据库 {aet[0]}-{aet[1]}，参照 {r.aet[0]}-{r.aet[1]}）")
    if r.et_known and not r.aet and d.extra_time and not r.pens:
        out.append("数据库有加时，参照没有")
    # 数据库 90 分钟比分不知道、参照又只有一个比分时，参照可能写的是 90 分钟比分：每边都不超过加时后比分就不算错
    maybe_90 = (r.final and ft is None and aet is not None
                and r.final[0] <= aet[0] and r.final[1] <= aet[1])
    if r.final and r.final not in (ft, aet) and not maybe_90:
        shown = f"{ft[0]}-{ft[1]}" if ft else "?"
        if aet:
            shown += f"，加时 {aet[0]}-{aet[1]}"
        out.append(f"比分不一样（数据库 {shown}，参照 {r.final[0]}-{r.final[1]}）")
    if r.pens and not pens:
        out.append(f"参照有点球（{r.pens[0]}-{r.pens[1]}），数据库没有")
    elif r.pens and pens and r.pens != pens:
        out.append(f"点球比分不一样（数据库 {pens[0]}-{pens[1]}，参照 {r.pens[0]}-{r.pens[1]}）")
    elif pens and not r.pens and r.et_known:
        out.append(f"数据库有点球（{pens[0]}-{pens[1]}），参照没有")
    return out


# ---------------------------------------------------------------- 汇总

def issue_kind(text: str) -> str:
    return text.split("（")[0]


def check_source(refs: list[RefMatch], dbs: list[DbMatch], names: Names) -> dict:
    """核对一个参照数据集。返回每个赛事的统计和不一致清单。"""
    out = {}
    by_comp_ref = collections.defaultdict(list)
    for r in refs:
        by_comp_ref[r.competition].append(r)
    by_comp_db = collections.defaultdict(list)
    for d in dbs:
        by_comp_db[d.competition].append(d)
    for code, rs in sorted(by_comp_ref.items()):
        seasons = {r.season for r in rs}
        ds = [d for d in by_comp_db.get(code, []) if d.season in seasons]
        pairs = pair_up(rs, ds, names)
        paired_r = {id(r) for r, _, _ in pairs}
        paired_d = {d.match_id for _, d, _ in pairs}
        cc = COMP_COUNTRY.get(code, "")
        ours = [r for r in rs if names.ours(r.home, r.home_country, cc) or names.ours(r.away, r.away_country, cc)]
        missing = [r for r in ours if id(r) not in paired_r]
        # 数据库有、参照没有：只看参照数据确实覆盖了的（赛季, 轮次），例如 openfootball 早年的欧冠没有资格赛
        covered = {(d.season, d.round.replace("_replay2", "").replace("_replay", "")) for _, d, _ in pairs}
        extra = [d for d in ds if d.match_id not in paired_d
                 and (d.season, d.round.replace("_replay2", "").replace("_replay", "")) in covered]
        issues = []
        kinds: collections.Counter = collections.Counter()
        consistent = round_only = 0
        for r, d, sw in pairs:
            if d.precision == "round":
                round_only += 1
                if abs((d.date - r.date).days) > 0:
                    kinds["只知道轮次（日期不比）"] += 1
            found = compare(r, d, sw)
            if not found:
                consistent += 1
            for f in found:
                kinds[issue_kind(f)] += 1
                issues.append((code, r, d, f))
        # “参照里我们的比赛”= 认得出是我们的球队的 + 靠名字相似配上的（相似配对能认出对照表里没写的写法）
        ours_count = len({id(r) for r in ours} | paired_r)
        out[code] = {"seasons": sorted(seasons), "ref": len(rs), "ours": ours_count, "paired": len(pairs),
                     "missing": missing, "extra": extra, "consistent": consistent, "round_only": round_only,
                     "kinds": kinds, "issues": issues,
                     "date_same": sum(1 for r, d, _ in pairs if d.precision == "day" and d.date == r.date),
                     "date_day": sum(1 for _, d, _ in pairs if d.precision == "day")}
    return out


def run(settings: dict | None = None, con: sqlite3.Connection | None = None) -> dict:
    settings = settings or load_settings()
    c = settings["crosscheck"]
    raw = resolve_path(settings["paths"]["raw"])
    first = settings["wikipedia"]["first_season"]
    refs = {
        "engsoccerdata": read_engsoccerdata(raw / "engsoccerdata", c["engsoccerdata"]["files"], first),
        "openfootball": read_openfootball(raw / "openfootball", c["openfootball"]["files"]),
    }
    own = con is None
    con = con or connect()
    codes = {r.competition for rs in refs.values() for r in rs}
    dbs = load_db(con, codes) if codes else []
    if own:
        con.close()
    names = Names(settings)
    return {src: check_source(rs, dbs, names) for src, rs in refs.items()}


# ---------------------------------------------------------------- 报告

SOURCE_NOTE = {
    "engsoccerdata": "engsoccerdata（GPL，jalapic/engsoccerdata 的 champs / facup / leaguecup）",
    "openfootball": "openfootball（CC0，Football.TXT 格式）",
}


def _pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "—"


def _ref_text(r: RefMatch) -> str:
    sc = r.ft or r.aet or r.final
    s = f"{sc[0]}-{sc[1]}" if sc else "?"
    if r.aet:
        s += " 加时"
    if r.pens:
        s += f" 点球 {r.pens[0]}-{r.pens[1]}"
    return f"{r.date} {r.home} – {r.away} {s}（{r.round_text}，{r.line}）"


def _db_text(d: DbMatch) -> str:
    s = f"{d.ft[0]}-{d.ft[1]}" if d.ft else "?"
    if d.aet:
        s += f" 加时 {d.aet[0]}-{d.aet[1]}"
    if d.pens:
        s += f" 点球 {d.pens[0]}-{d.pens[1]}"
    return f"#{d.match_id} {d.date}{'（轮次日期）' if d.precision == 'round' else ''} {_title(d.home)} – {_title(d.away)} {s} {d.round}/{d.leg}"


def write_report(result: dict, settings: dict) -> Path:
    folder = resolve_path(settings["paths"]["reports"])
    folder.mkdir(parents=True, exist_ok=True)
    md = folder / "wikipedia_交叉核对.md"
    txt = folder / "wikipedia_交叉核对_不一致清单.txt"
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    L = ["# 维基百科杯赛、欧战数据交叉核对", "",
         f"生成时间（北京时间）：{now}。由 `python -m src.clean.wikipedia_crosscheck` 生成，能重新生成。", "",
         "拿两个独立整理的开放数据集和数据库里维基百科来的比赛对比（只比至少一方是我们的球队的比赛）。",
         "“完全一致”= 配上了，而且日期（数据库精确到天的）、90 分钟比分、加时比分、点球、主客队都没有不一样的地方。",
         "数据库里“只知道轮次”的比赛不比日期（它的日期是这一轮的第一天，本来就不是比赛那天）。", ""]
    T = []
    total_pairs = total_ok = 0
    for src, comps in result.items():
        L += [f"## {SOURCE_NOTE.get(src, src)}", "",
              "| 赛事 | 赛季 | 参照里我们的比赛 | 配上 | 参照有、数据库没有 | 数据库有、参照没有 | 完全一致 | 一致率 | 日期一样（精确到天的） | 只知道轮次 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for code, s in comps.items():
            seasons = s["seasons"]
            span = f"{seasons[0]}–{seasons[-1]}" if seasons else ""
            L.append(f"| {code} | {span} | {s['ours']} | {s['paired']} | {len(s['missing'])} | {len(s['extra'])} | "
                     f"{s['consistent']} | {_pct(s['consistent'], s['paired'])} | "
                     f"{s['date_same']}/{s['date_day']} | {s['round_only']} |")
            total_pairs += s["paired"]
            total_ok += s["consistent"]
        L.append("")
        kinds = collections.Counter()
        for s in comps.values():
            kinds.update({k: v for k, v in s["kinds"].items() if k != "只知道轮次（日期不比）"})
        if kinds:
            L += ["不一致的种类：", "", "| 种类 | 条数 |", "|---|---|"]
            L += [f"| {k} | {v} |" for k, v in kinds.most_common()]
            L.append("")
        for code, s in comps.items():
            for c, r, d, f in s["issues"]:
                T.append(f"[{src} {c}] {f}\n    参照：{_ref_text(r)}\n    数据库：{_db_text(d)}")
            for r in s["missing"]:
                T.append(f"[{src} {code}] 参照有、数据库没有\n    参照：{_ref_text(r)}")
            for d in s["extra"]:
                T.append(f"[{src} {code}] 数据库有、参照没有（参照覆盖了这一轮）\n    数据库：{_db_text(d)}")
    L[6:6] = [f"**总的一致率：{total_ok}/{total_pairs} = {_pct(total_ok, total_pairs)}**（配上的比赛里完全一致的比例）", ""]
    L += ["## 不一致清单", "", f"每一条的参照和数据库内容见 `{txt.name}`（{len(T)} 条）。", ""]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    txt.write_text("\n".join(T) + "\n", encoding="utf-8")
    return md


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    result = run(settings)
    path = write_report(result, settings)
    log.info("交叉核对完成，报告在 %s", path)


if __name__ == "__main__":
    main()
