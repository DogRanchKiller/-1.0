"""赛前积分榜：给一个联赛赛季和一个截止时间，算出那时候的积分榜（第 1 阶段：复盘自动标签，需求文档第 15 节）。

为什么要“赛前”积分榜：
  复盘时想知道“这场是不是保级战”，要按开球前球迷看到的积分榜判断，不能按赛季末的最终积分榜。
  例如某队最后一轮才掉进降级区，按最终积分榜看，它前面每一场都像“保级战”；但赛前它可能在中游，根本没人觉得是保级战。
  用最终积分榜，就是拿“后来才知道的事”给以前的比赛贴标签（偷看未来，需求文档第 16 节）。

积分榜怎么算（每一项都只用截止时间之前已经知道比分的比赛）：
  场次 = 胜 + 平 + 负；积分 = 胜 × 胜一场的分 + 平 × 平一场的分 + 扣分（扣分是负数）；
  净胜球 = 进球 − 失球；剩余场次 = 一个赛季每队要踢的场数（双循环 = 2 ×（球队数 − 1））− 已踢场次。
  名次：先比积分，积分一样再按这个联赛的同分规则一项一项比（config/league_rules.yaml，每个联赛不同）。

数据怎么来：
  所有比分都经过取数层 FeatureStore 拿（historical_matches），用途写“复盘”，只拿截止时间之前已经能知道的比分。
  为了快：一个联赛赛季只取一次（截止时间 = 这季最后一个要算的时间点），然后在内存里按“比分能知道的时间”切，
  每个截止时间只用切出来的那部分（见 SeasonTable.at）。切的时候按 known_at_utc 比大小，和取数层的规则完全一样。

扣分：football-data 没有扣分数据，手工录在 config/points_deductions.yaml（每条写生效时间和来源），
  到生效时间才算进积分榜（扣分宣布之前，积分榜上看不到这次扣分）。

做不到和真实规则完全一样的地方（见 config/league_rules.yaml 开头）：
  - 附加赛、抽签、公平竞赛分（红黄牌）这些最后一级的规则没有数据，用到时按队名字母排，并在 tie_unresolved 列标出来；
  - 三队以上同分时，相互战绩先在全部同分的队里算，分出一部分后，剩下的队再在它们之间重新算（大多数联赛就是这样，个别细节可能不同）。
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.config import load_settings, resolve_path
from src.features.known_time import to_utc_text

# 同分时能比的项目（league_rules.yaml 的“同分”里只能写这些）
OVERALL = {"净胜球", "进球", "客场进球", "胜场"}
H2H = {"相互战绩积分", "相互战绩净胜球", "相互战绩进球", "相互战绩客场进球"}
CRITERIA = OVERALL | H2H

# 积分榜一列一个统计，按这个顺序存在数组里
_P, _W, _D, _L, _GF, _GA, _AGF = range(7)


# ================= 规则和扣分 =================
@dataclass
class LeagueRules:
    """一个联赛一个赛季的规则（从 config/league_rules.yaml 读）。"""
    competition: str
    season: int
    win_points: int = 3            # 胜一场得几分
    draw_points: int = 1           # 平一场得几分
    tiebreak: list = field(default_factory=lambda: ["净胜球", "进球"])  # 积分一样时依次比什么
    h2h_needs_all_played: bool = False  # 相互战绩要两队之间的比赛都踢完才用（西甲、意甲）；没踢完就跳过相互战绩
    relegation: int = 3            # 直接降级几队
    relegation_playoff: int = 0    # 要踢降级附加赛的几队（名次紧挨着直接降级区上面）
    source: str = ""               # 规则来源

    @property
    def danger_spots(self) -> int:
        """“保级战”要躲开的名次数 = 直接降级 + 降级附加赛（踢附加赛也算没保住）。"""
        return self.relegation + self.relegation_playoff


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def _rules_file(path: str) -> dict:
    return _load_yaml(Path(path))


def load_rules(competition: str, season: int, settings: dict | None = None, path: Path | None = None) -> LeagueRules:
    """查 league_rules.yaml：这个联赛这个赛季用哪一段规则。赛季不在任何一段里就报错（不猜）。"""
    settings = settings or load_settings()
    path = path or resolve_path(settings["standings"]["rules"])
    raw = _rules_file(str(path))["联赛"]
    if competition not in raw:
        raise ValueError(f"league_rules.yaml 里没有联赛 {competition}")
    for seg in raw[competition]["规则"]:
        lo, hi = seg["赛季"]
        if lo <= season <= hi:
            tb = list(seg["同分"])
            bad = set(tb) - CRITERIA
            if bad:
                raise ValueError(f"league_rules.yaml {competition} {lo}–{hi}：不认识的同分项目 {sorted(bad)}，只能是 {sorted(CRITERIA)}")
            return LeagueRules(competition, season, int(seg.get("胜", 3)), int(seg.get("平", 1)), tb,
                               seg.get("相互战绩要踢完") in (True, "是"), int(seg["降级"]),
                               int(seg.get("降级附加赛", 0)), str(seg.get("来源", "")))
    raise ValueError(f"league_rules.yaml 里 {competition} 没有覆盖 {season} 赛季的规则")


def load_deductions(settings: dict | None = None, path: Path | None = None) -> pd.DataFrame:
    """读扣分记录。每行：competition, season, team, points（扣分是负数，退回是正数）, effective_utc（生效时间 UTC）。"""
    settings = settings or load_settings()
    path = path or resolve_path(settings["standings"]["deductions"])
    rows = (_load_yaml(path) or {}).get("扣分") or []
    out = []
    for r in rows:
        pts = int(r["分数"])
        season = int(r["赛季"])
        out.append({"competition": r["联赛"], "season": season, "team": r["球队"], "points": pts,
                    "effective_utc": _effective_utc(r["生效"], season), "reason": r.get("原因", "")})
    return pd.DataFrame(out, columns=["competition", "season", "team", "points", "effective_utc", "reason"])


def _effective_utc(value, season: int) -> str:
    """扣分的生效时间。
    - 只写了日期（例如 2023-11-17）：按“第二天 00:00 UTC”生效。宣布的具体时刻不知道，宁晚勿早，
      当天的比赛不算进去，免得在宣布之前就“知道”了扣分。
    - 写“赛季结束”：哪天宣布的查不到。只算进赛季结束后的最终积分榜（第二年 7 月 1 日生效），
      赛季中间的积分榜不算。这样会让赛季中间的积分榜少扣了分（不准），但绝不会提前知道扣分。
    """
    s = str(value).strip()
    if s == "赛季结束":
        return f"{season + 1}-07-01 00:00:00"
    if len(s) == 10:  # 只有日期
        return to_utc_text(pd.Timestamp(s) + pd.Timedelta(days=1))
    return to_utc_text(s)


# ================= 积分榜计算（纯计算，不碰数据库；测试直接用） =================
def _rank(teams: list[str], stats: dict[str, np.ndarray], pair_results,
          rules: LeagueRules) -> tuple[list[str], set[str]]:
    """把球队排名次。返回（从第 1 名到最后一名的队名，同分规则用完还分不开的队）。

    stats：每队一个数组 [积分, 胜, 平, 负, 进球, 失球, 客场进球]（积分已经算上扣分）。
    pair_results(主队, 客队)：返回截止时间前这两队（这个主客顺序）已经知道的比分列表 [(主队进球, 客队进球), ...]，
      算相互战绩用。用函数而不是整张表，是为了快：同分的几队之间只有几场比赛，直接查，不用每次扫全季的表。
    做法：先按积分分组；积分一样的一组，按 rules.tiebreak 的顺序一项一项比。
    某一项把一组分成了几小组，每个小组再从头按规则比（相互战绩只在小组里的队之间重新算）。
    """
    unresolved: set[str] = set()

    def overall(team: str, crit: str) -> float:
        s = stats[team]
        return {"净胜球": s[_GF] - s[_GA], "进球": s[_GF], "客场进球": s[_AGF], "胜场": s[_W]}[crit]

    def h2h_values(group: list[str]) -> dict | None:
        """这组队之间的比赛（小联赛）。要求踢完而没踢完时返回 None（跳过相互战绩）。"""
        games = [(h, a, hg, ag) for h in group for a in group if h != a for hg, ag in pair_results(h, a)]
        if rules.h2h_needs_all_played:
            played = {(h, a) for h, a, _, _ in games}
            need = {(h, a) for h in group for a in group if h != a}  # 双循环：每对主客各一场
            if not need <= played:
                return None
        v = {t: {"相互战绩积分": 0.0, "相互战绩净胜球": 0.0, "相互战绩进球": 0.0, "相互战绩客场进球": 0.0} for t in group}
        for h, a, hg, ag in games:
            v[h]["相互战绩进球"] += hg
            v[a]["相互战绩进球"] += ag
            v[a]["相互战绩客场进球"] += ag
            v[h]["相互战绩净胜球"] += hg - ag
            v[a]["相互战绩净胜球"] += ag - hg
            if hg > ag:
                v[h]["相互战绩积分"] += rules.win_points
            elif hg < ag:
                v[a]["相互战绩积分"] += rules.win_points
            else:
                v[h]["相互战绩积分"] += rules.draw_points
                v[a]["相互战绩积分"] += rules.draw_points
        return v

    def order(group: list[str]) -> list[str]:
        """把一组积分相同的队排好。"""
        if len(group) == 1:
            return group
        h2h = None
        for crit in rules.tiebreak:
            if crit in H2H:
                if h2h is None:
                    h2h = h2h_values(group) or {}
                if not h2h:          # 相互战绩没踢完（西甲、意甲规则）→ 这项跳过
                    continue
                key = {t: h2h[t][crit] for t in group}
            else:
                key = {t: overall(t, crit) for t in group}
            if len(set(key.values())) == 1:
                continue             # 这一项大家一样，比下一项
            out = []
            for val in sorted(set(key.values()), reverse=True):
                out += order([t for t in group if key[t] == val])  # 分出来的小组从头再比
            return out
        unresolved.update(group)     # 规则用完还一样：按队名字母排，标出来
        return sorted(group)

    by_points: dict[float, list[str]] = {}
    for t in teams:
        by_points.setdefault(stats[t][_P], []).append(t)
    ranked: list[str] = []
    for pts in sorted(by_points, reverse=True):
        ranked += order(sorted(by_points[pts]))
    return ranked, unresolved


def compute_table(results: pd.DataFrame, teams: list[str], rules: LeagueRules,
                  deductions: dict[str, int] | None = None, games_per_team: int | None = None) -> pd.DataFrame:
    """从比分算积分榜（纯计算）。

    results：要算进去的比赛，列 home_team, away_team, home_goals, away_goals（已经按截止时间切好）。
    teams：这个赛季的全部球队（一场没踢的队也要在榜上）。
    deductions：{队名: 扣分合计（负数）}，已经按截止时间挑好。
    games_per_team：一个赛季每队踢几场；不给就按双循环 2 ×（球队数 − 1）。
    返回每队一行，按名次排好：position, team, played, won, drawn, lost, goals_for, goals_against, goal_diff,
      deduction, points, remaining, tie_unresolved。
    """
    deductions = deductions or {}
    games = games_per_team if games_per_team is not None else 2 * (len(teams) - 1)
    stats = {t: np.zeros(7) for t in teams}
    res = results.dropna(subset=["home_goals", "away_goals"])
    for h, a, hg, ag in res[["home_team", "away_team", "home_goals", "away_goals"]].itertuples(index=False):
        _add(stats[h], hg, ag, home=True, rules=rules)
        _add(stats[a], ag, hg, home=False, rules=rules)
    for t, d in deductions.items():
        stats[t][_P] += d
    pairs: dict[tuple, list] = {}
    for h, a, hg, ag in res[["home_team", "away_team", "home_goals", "away_goals"]].itertuples(index=False):
        pairs.setdefault((h, a), []).append((hg, ag))
    return _finish(teams, stats, lambda h, a: pairs.get((h, a), ()), rules, deductions, games)


def _add(s: np.ndarray, gf, ga, home: bool, rules: LeagueRules) -> None:
    """把一场比赛加到一支队的统计上。"""
    s[_GF] += gf
    s[_GA] += ga
    if not home:
        s[_AGF] += gf
    if gf > ga:
        s[_W] += 1
        s[_P] += rules.win_points
    elif gf == ga:
        s[_D] += 1
        s[_P] += rules.draw_points
    else:
        s[_L] += 1


def _finish(teams, stats, pair_results, rules, deductions, games) -> pd.DataFrame:
    """排名次、整理成表。"""
    ranked, unresolved = _rank(list(teams), stats, pair_results, rules)
    rows = []
    for i, t in enumerate(ranked, start=1):
        s = stats[t]
        played = int(s[_W] + s[_D] + s[_L])
        rows.append({"position": i, "team": t, "played": played, "won": int(s[_W]), "drawn": int(s[_D]),
                     "lost": int(s[_L]), "goals_for": int(s[_GF]), "goals_against": int(s[_GA]),
                     "goal_diff": int(s[_GF] - s[_GA]), "deduction": int(deductions.get(t, 0)),
                     "points": int(s[_P]), "remaining": games - played, "tie_unresolved": t in unresolved})
    return pd.DataFrame(rows)


class SeasonTable:
    """一个联赛赛季的“随时可查”积分榜：比分一次给全，查某个截止时间时只用那时已经知道的比分。

    results：这季的比分（经过 FeatureStore 取来），要有 home_team, away_team, home_goals, away_goals, known_at_utc。
    teams：这季全部球队。deductions：load_deductions() 的表（只留这个联赛这季的就行，多了也会自己筛）。
    快的办法：按“比分能知道的时间”排好序，先算好“前 k 场加完以后每队的累计统计”，
    查截止时间 = 二分查找出“截止前知道了几场”，直接取那一行，不用每次从头加。
    """

    def __init__(self, competition: str, season: int, results: pd.DataFrame, teams: list[str],
                 rules: LeagueRules, deductions: pd.DataFrame | None = None, games_per_team: int | None = None):
        self.competition, self.season, self.rules = competition, season, rules
        self.teams = sorted(teams)
        self.games = games_per_team if games_per_team is not None else 2 * (len(self.teams) - 1)
        res = results.dropna(subset=["home_goals", "away_goals"]).sort_values(["known_at_utc", "match_id"]
                                                                              if "match_id" in results else
                                                                              ["known_at_utc"])
        unknown = (set(res["home_team"]) | set(res["away_team"])) - set(self.teams)
        if unknown:
            raise ValueError(f"{competition} {season}：比分里有不在球队名单里的队 {sorted(unknown)}")
        self.known = res["known_at_utc"].tolist()          # 每场比分能知道的时间，从早到晚
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(res)
        # cum[k] = 前 k 场加完以后每队的 [积分, 胜, 平, 负, 进球, 失球, 客场进球]（不含扣分）
        cum = np.zeros((n + 1, len(self.teams), 7))
        self._pairs: dict[tuple, list] = {}               # (主队, 客队) → [(第几场, 主队进球, 客队进球)]
        for k, (h, a, hg, ag) in enumerate(res[["home_team", "away_team", "home_goals", "away_goals"]]
                                           .itertuples(index=False), start=1):
            cum[k] = cum[k - 1]
            _add(cum[k, idx[h]], hg, ag, home=True, rules=rules)
            _add(cum[k, idx[a]], ag, hg, home=False, rules=rules)
            self._pairs.setdefault((h, a), []).append((k, hg, ag))
        self._cum = cum
        d = deductions if deductions is not None else pd.DataFrame(columns=["competition", "season", "team",
                                                                             "points", "effective_utc"])
        d = d[(d["competition"] == competition) & (d["season"] == season)].sort_values("effective_utc")
        bad = set(d["team"]) - set(self.teams)
        if bad:
            raise ValueError(f"points_deductions.yaml：{competition} {season} 的扣分球队 {sorted(bad)} 这季不在这个联赛里，检查队名")
        self.deductions = d.reset_index(drop=True)
        self._cache: dict = {}

    def at(self, cutoff) -> pd.DataFrame:
        """截止时间那一刻的积分榜（只算比分能知道时间 ≤ 截止时间的比赛，扣分只算生效时间 ≤ 截止时间的）。"""
        from bisect import bisect_right
        cutoff = to_utc_text(cutoff)
        k = bisect_right(self.known, cutoff)
        nd = bisect_right(self.deductions["effective_utc"].tolist(), cutoff)
        key = (k, nd)
        if key not in self._cache:
            ded: dict[str, int] = {}
            for t, p in self.deductions.iloc[:nd][["team", "points"]].itertuples(index=False):
                ded[t] = ded.get(t, 0) + int(p)
            stats = {t: self._cum[k, i].copy() for i, t in enumerate(self.teams)}
            for t, p in ded.items():
                stats[t][_P] += p

            def pair_results(h, a, k=k):
                return [(hg, ag) for j, hg, ag in self._pairs.get((h, a), ()) if j <= k]

            self._cache[key] = _finish(self.teams, stats, pair_results, self.rules, ded, self.games)
        return self._cache[key].copy()

    def final(self) -> pd.DataFrame:
        """全部比分都算上的积分榜（赛季末；核对维基百科用）。"""
        return self.at("2999-01-01")


# ================= 经过取数层取数据 =================
def season_window(season: int) -> tuple[str, str, str]:
    """一个赛季在日历上的范围（UTC），用来向取数层要这季的比赛。

    一般是 [当年 7 月 1 日, 第二年 7 月 1 日)。2019/20 赛季因为疫情踢到 2020 年 7、8 月，
    所以第二年 7 月 1 日时这季还在踢的话，再往后看到 9 月 1 日（第三个值）。
    只按赛季（season 字段）留下这季的比赛，窗口只是为了少取数据。
    """
    return f"{season}-07-01 00:00:00", f"{season + 1}-07-01 00:00:00", f"{season + 1}-09-01 00:00:00"


def season_schedule(fs, competition: str, season: int, usage: str = "复盘") -> pd.DataFrame:
    """这季的赛程（每场一行：match_id、主客队、kickoff_earliest_utc），经过 FeatureStore.prediction_targets 取。

    赛程是赛前就知道的（数据用途字典：比赛信息“赛程”）。这里只用它拿“这季有哪些队、有哪些场次要贴标签”。
    """
    start, end, ext = season_window(season)
    df = fs.prediction_targets(start, end, usage, competition=competition)
    df = df[df["season"] == season]
    # 到 7 月 1 日还在踢（最后一场离 7 月 1 日不到 3 周）→ 再往后取到 9 月 1 日
    if not df.empty and df["kickoff_earliest_utc"].max() >= to_utc_text(pd.Timestamp(end) - pd.Timedelta(days=21)):
        more = fs.prediction_targets(end, ext, usage, competition=competition)
        df = pd.concat([df, more[more["season"] == season]], ignore_index=True)
    return df.sort_values(["kickoff_earliest_utc", "match_id"]).reset_index(drop=True)


def season_table(fs, competition: str, season: int, cutoff, usage: str = "复盘", settings: dict | None = None,
                 schedule: pd.DataFrame | None = None, deductions: pd.DataFrame | None = None) -> SeasonTable:
    """取一个联赛赛季到 cutoff 为止已经能知道的全部比分（经过 FeatureStore，一次取完），做成 SeasonTable。

    cutoff 要 ≥ 之后要查的所有截止时间（一般用这季最后一个要查的时间点）。截止之后的比分根本没取出来，
    所以就算 SeasonTable.at() 写错了，也不可能用到 cutoff 之后的比分。
    usage：默认“复盘”。以后积分榜要当模型输入，改成“A版输入”（数据用途字典里比分对“以前的比赛”允许 A 版输入）。
    """
    settings = settings or fs.settings
    schedule = schedule if schedule is not None else season_schedule(fs, competition, season, usage)
    if schedule.empty:
        raise ValueError(f"{competition} {season} 没有比赛")
    teams = sorted(set(schedule["home_team"]) | set(schedule["away_team"]))
    start, _, _ = season_window(season)
    # 截止时间最晚取到“这季最后一场开球后 3 天”：再往后已经是下个赛季的比赛，不取。
    # 这样问“2023/24 赛季最终积分榜”时，不会顺带取到 2024/25（期末考试赛季）的比分，开发模式也能用。
    season_end = to_utc_text(pd.Timestamp(schedule["kickoff_earliest_utc"].max()) + pd.Timedelta(days=3))
    query_cutoff = min(to_utc_text(cutoff), season_end)
    res = fs.historical_matches(query_cutoff, usage, competition=competition, since=start)
    res = res[res["season"] == season]
    assert (res["known_at_utc"] <= query_cutoff).all(), "取数层返回了截止时间之后的比分（不应该发生）"
    rules = load_rules(competition, season, settings)
    ded = deductions if deductions is not None else load_deductions(settings)
    return SeasonTable(competition, season, res, teams, rules, ded)


def standings_at(fs, competition: str, season: int, cutoff, usage: str = "复盘",
                 settings: dict | None = None) -> pd.DataFrame:
    """最简单的用法：某个联赛赛季在某个截止时间的积分榜。

    例子：standings_at(fs, "E0", 2022, "2023-03-01 00:00") → 2023 年 3 月 1 日 0 点（UTC）时的英超积分榜。
    """
    return season_table(fs, competition, season, cutoff, usage, settings).at(cutoff)
