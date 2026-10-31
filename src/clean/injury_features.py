r"""从 salimt 镜像（Transfermarkt 球员数据）算出每场比赛开球前 24 小时“两队谁伤着”（需求文档第 4 节“步骤 7 怎么做”）。

只做研究实验用（你决定，2026-10-31）：
  - 结果不导入数据库，只写到 data/research/injury_features.csv（在 data/ 下，不上传 GitHub）。
  - 模型和回测要用，必须经过取数层（数据类别「伤停（研究）」），用途只能是「研究实验」，只在开发模式（config/data_usage.yaml）。

原始文件（src/collect/salimt_mirror.py 下载的，data/raw/salimt/）：
  transfers.csv     转会记录：球员、日期、从哪个队（编号）到哪个队。文件里每个球员的记录是新的在前。
  market_value.csv  身价记录：球员、估价日期、身价（欧元）。
  injuries.csv      伤病记录：球员、开始日期、结束日期（还有缺阵天数、场数，这里不用：事后才知道）。
  team_seasons.csv  每个队每个赛季的联赛成绩：用来把 Transfermarkt 的球队编号对上我们的球队。

一、球队对照（Transfermarkt 球队编号 → 我们的球队）
  同一联赛、同一赛季，胜、平、负、进球、失球五个数都一样，就给这一对投一票。
  每个编号只能对到我们的一个队、我们的每个队只能对到一个编号，有冲突就停下报错。
  结果写在 config/team_aliases_transfermarkt.yaml；以后每次运行都重新投一次票，和文件对不上就报错（防止文件被改错）。

二、每场比赛的“开球前情况”
  时间点 as_of_utc = 最早开球 − 24 小时（和回测的 T-24 一样）。“知道的” = 日期严格早于 as_of_utc 那天（英国时间）的记录：
  同一天的转会、估价、受伤一律当作还不知道（宁晚勿早）。
  - 某队有哪些人：每个球员“最后一次在这天之前的转会”转去的队（租借、租借回来都算转会）。
  - 每人身价：这天之前最后一次估价；从没估过价的人算 0。
  - 伤着的人：开始日期早于这天、结束日期晚于这天（还没复出）。没有结束日期的记录不用。
  每队（home_ / away_）四个数：
    inj_n        伤了几个人
    inj_share    伤的人身价占全队身价的比例（全队身价是 0 时空值）
    inj_top_loss “最贵 11 人”掉了多少：log(全队最贵 11 人的身价和) − log(不算伤员时最贵 11 人的身价和)（0 = 主力都没伤）
    top_value    全队最贵 11 人的身价和的 log（对照组只用两队的差）
  first_season 以前的比赛、对不上球队的队，这些数都是空值（树把空值当“不知道”，不当“没人伤”）。

运行方法：Claude 运行 `.venv\Scripts\python -m src.clean.injury_features`
  第一次（还没有对照文件）加 --write-aliases 生成对照文件。
  在工作树里运行、原始文件在主项目文件夹时，用 --raw 指定主项目的 data/raw。
结果：data/research/injury_features.csv（每场一行）、data/reports/伤停特征报告.md
"""
import argparse
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.clean.football_data_import import load_teams
from src.common.config import load_settings, resolve_path
from src.features.known_time import compute_known_times

log = logging.getLogger("injury_features")

STATS = ["w", "d", "l", "gf", "ga"]          # 球队对照用的五个数：胜、平、负、进球、失球
SIDE_COLS = ["inj_n", "inj_share", "inj_top_loss", "top_value"]
COLUMNS = ["match_id", "competition", "season", "as_of_utc",
           *[f"{side}_{c}" for side in ("home", "away") for c in SIDE_COLS]]


# ======================= 一、球队对照 =======================
def our_team_seasons(con: sqlite3.Connection, leagues, first: int, last: int) -> pd.DataFrame:
    """数据库里每队每季的联赛成绩（football-data 的比分，已踢完和判罚比分的比赛）：联赛、赛季、队、胜平负、进失球。"""
    marks = ",".join("?" * len(leagues))
    m = pd.read_sql(
        f"select m.competition, m.season, m.home_team, m.away_team, s.home_goals, s.away_goals from matches m "
        f"join match_stats s on s.match_id = m.match_id and s.source = 'football-data' "
        f"where m.competition in ({marks}) and m.season between ? and ? and m.status in ('finished', 'awarded') "
        f"and s.home_goals is not null",
        con, params=[*leagues, first, last])
    home = m.rename(columns={"home_team": "team", "home_goals": "gf", "away_goals": "ga"})
    away = m.rename(columns={"away_team": "team", "away_goals": "gf", "home_goals": "ga"})
    t = pd.concat([home[["competition", "season", "team", "gf", "ga"]],
                   away[["competition", "season", "team", "gf", "ga"]]], ignore_index=True)
    t["w"], t["d"], t["l"] = (t.gf > t.ga).astype(int), (t.gf == t.ga).astype(int), (t.gf < t.ga).astype(int)
    return t.groupby(["competition", "season", "team"], as_index=False)[STATS].sum()


def tm_team_seasons(raw_dir: Path, leagues: dict, first: int, last: int) -> pd.DataFrame:
    """Transfermarkt 每队每季的联赛成绩，联赛编号换成我们的联赛代码。"""
    ts = pd.read_csv(raw_dir / "salimt" / "team_seasons.csv")
    to_ours = {tm: ours for ours, tm in leagues.items()}
    ts = ts[ts["competition_id"].isin(to_ours) & ts["season_id"].between(first, last)].copy()
    ts["competition"] = ts["competition_id"].map(to_ours)
    ts = ts.rename(columns={"season_id": "season", "season_wins": "w", "season_draws": "d", "season_losses": "l",
                            "season_goals_for": "gf", "season_goals_against": "ga", "team_name": "tm_name"})
    return ts[["competition", "season", "club_id", "tm_name", *STATS]]


def vote(ours: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """五个数都一样的（联赛、赛季）里配对，每对一票。返回 club_id、tm_name、team、票数、哪几季。

    同一联赛同一赛季两个队五个数完全一样时，这一季两边都配不出唯一的一对，这一季不投票（极少见）。"""
    j = ours.merge(tm, on=["competition", "season", *STATS])
    ambiguous = j.duplicated(["competition", "season", "team"], keep=False) | \
        j.duplicated(["competition", "season", "club_id"], keep=False)
    j = j[~ambiguous]
    return (j.groupby(["club_id", "team"])
             .agg(tm_name=("tm_name", "last"), votes=("season", "size"),
                  seasons=("season", lambda s: f"{s.min()}–{s.max()}"))
             .reset_index())


def decide(votes: pd.DataFrame) -> dict[int, str]:
    """每个编号只能对一个队、每个队只能对一个编号；有冲突就报错停下。返回 {club_id: 我们的队}。"""
    bad = votes[votes["club_id"].duplicated(keep=False) | votes["team"].duplicated(keep=False)]
    if not bad.empty:
        raise ValueError("球队对照有冲突（一个编号对到几个队，或一个队对到几个编号）：\n" + bad.to_string(index=False))
    return {int(r.club_id): r.team for r in votes.itertuples()}


def write_aliases(path: Path, votes: pd.DataFrame) -> None:
    """生成 config/team_aliases_transfermarkt.yaml：我们的队 → Transfermarkt 编号（注释里写 Transfermarkt 的队名、票数）。"""
    lines = ["# Transfermarkt（salimt 镜像）的球队编号对照表（需求文档第 4 节“步骤 7 怎么做”、6.1“Transfermarkt 球员数据镜像（salimt）”）",
             "#",
             "#   对照：teams.yaml 的“英文” → Transfermarkt 的球队编号（club_id）",
             "#",
             "# 程序生成（src/clean/injury_features.py --write-aliases）：同联赛同赛季、胜平负和进失球五个数都一样就投一票，",
             "# 每个编号只能对到一个队、每个队只能对到一个编号。以后每次运行都重新投票，和这个文件对不上就报错。",
             "# 每行后面的注释：Transfermarkt 的队名、票数（几个赛季对上）、对上的赛季范围。",
             "对照:"]
    for r in votes.sort_values("team").itertuples():
        lines.append(f"  {yaml.safe_dump(r.team, allow_unicode=True).splitlines()[0]}: {int(r.club_id)}"
                     f"    # {r.tm_name}，{r.votes} 票，{r.seasons}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_aliases(settings: dict) -> dict[int, str]:
    """读对照文件：{club_id: 我们的队}。检查我们的队都在 teams.yaml 里、编号没写两次。"""
    _, names = load_teams(settings)
    known = set(names.values())
    path = resolve_path(settings["paths"]["team_aliases_transfermarkt"])
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    alias: dict[int, str] = {}
    for team, cid in (data.get("对照") or {}).items():
        if team not in known:
            raise ValueError(f"{path.name}：{team} 不在 teams.yaml 里")
        if int(cid) in alias:
            raise ValueError(f"{path.name}：编号 {cid} 写了两次")
        alias[int(cid)] = team
    return alias


def check_aliases(alias: dict[int, str], decided: dict[int, str]) -> None:
    """对照文件和这次投票核对：投票定下的每一对都要和文件一样，文件里也不能多出投票里没有的。"""
    wrong = [f"编号 {cid}：文件写 {alias.get(cid)}，投票是 {team}" for cid, team in sorted(decided.items())
             if alias.get(cid) != team]
    extra = [f"编号 {cid}（{team}）投票里没有" for cid, team in sorted(alias.items()) if cid not in decided]
    if wrong or extra:
        raise ValueError("球队对照文件和投票结果不一致：\n  " + "\n  ".join(wrong + extra))


# ======================= 二、球员：在哪个队、身价、伤没伤 =======================
NEVER = np.iinfo(np.int64).max   # “以后再没转会”的结束日


def _day_numbers(s: pd.Series) -> np.ndarray:
    """日期 → 从 1970-01-01 起第几天（整数，比较快）。"""
    return pd.to_datetime(s).to_numpy("datetime64[D]").astype(np.int64)


class PlayerData:
    """按天查“某队有哪些人、各自身价多少、谁伤着”。只用严格早于查询那天的记录。"""

    def __init__(self, transfers: pd.DataFrame, values: pd.DataFrame, injuries: pd.DataFrame, clubs, top_n: int = 11):
        self.top_n = int(top_n)
        t = transfers.copy()
        t["d"] = pd.to_datetime(t["transfer_date"], errors="coerce")
        t["file_order"] = -np.arange(len(t))       # 文件里每个球员新的在前：同一天的几次转会，文件里靠后的发生得早
        t = t.dropna(subset=["d"]).sort_values(["player_id", "d", "file_order"], kind="stable")
        t["until"] = t.groupby("player_id")["d"].shift(-1)   # 下一次转会的日期 = 在这个队待到哪天
        t = t[t["to_team_id"].isin(set(clubs))]
        self.stints: dict[int, tuple] = {}
        for cid, g in t.groupby("to_team_id"):
            until = g["until"]
            self.stints[int(cid)] = (g["player_id"].to_numpy(np.int64), _day_numbers(g["d"]),
                                     np.where(until.isna(), NEVER, _day_numbers(until.fillna(g["d"]))))
        players = set(t["player_id"])
        v = values[values["player_id"].isin(players)].copy()
        v["day"] = _day_numbers(v["date_unix"])
        v = v.sort_values(["player_id", "day"], kind="stable")
        self.values = {int(p): (g["day"].to_numpy(), g["value"].to_numpy(float)) for p, g in v.groupby("player_id")}
        i = injuries[injuries["player_id"].isin(players)].copy()
        i["f"], i["e"] = pd.to_datetime(i["from_date"], errors="coerce"), pd.to_datetime(i["end_date"], errors="coerce")
        i = i.dropna(subset=["f", "e"])                 # 没有结束日期的不用（多是还没好的）
        self.injuries = {int(p): (_day_numbers(g["f"]), _day_numbers(g["e"])) for p, g in i.groupby("player_id")}

    def squad(self, club: int, day: int) -> np.ndarray:
        """这天（不含）之前最后一次转会转到这个队、到这天还没转走的球员。"""
        if club not in self.stints:
            return np.array([], dtype=np.int64)
        p, start, until = self.stints[club]
        return np.unique(p[(start < day) & (until >= day)])

    def value(self, player: int, day: int) -> float:
        """这天之前最后一次估价；从没估过价算 0。"""
        rec = self.values.get(player)
        if rec is None:
            return 0.0
        k = np.searchsorted(rec[0], day, "left") - 1
        return float(rec[1][k]) if k >= 0 else 0.0

    def injured(self, player: int, day: int) -> bool:
        """这天之前开始受伤、结束日期在这天或之后（还没复出）。
        结束日期正好是这天也算还伤着：“这天复出”是这天才有的消息，和转会、估价一样当作还不知道（宁晚勿早）。"""
        rec = self.injuries.get(player)
        return rec is not None and bool(((rec[0] < day) & (rec[1] >= day)).any())

    def side(self, club: int | None, day: int) -> list[float]:
        """一队这天的四个数：inj_n、inj_share、inj_top_loss、top_value；对不上队、没有人时都是空值。"""
        nan = [np.nan] * len(SIDE_COLS)
        if club is None:
            return nan
        sq = self.squad(club, day)
        if len(sq) == 0:
            return nan
        vals = np.array([self.value(int(p), day) for p in sq])
        hurt = np.array([self.injured(int(p), day) for p in sq])
        total = vals.sum()
        top_all = np.sort(vals)[-self.top_n:].sum()
        top_fit = np.sort(vals[~hurt])[-self.top_n:].sum() if (~hurt).any() else 0.0
        share = vals[hurt].sum() / total if total > 0 else np.nan
        loss = np.log(top_all) - np.log(top_fit) if top_all > 0 and top_fit > 0 else np.nan
        return [float(hurt.sum()), share, loss, np.log(top_all) if top_all > 0 else np.nan]


# ======================= 三、每场比赛 =======================
def load_matches(con: sqlite3.Connection, leagues, first: int, last: int, settings: dict) -> pd.DataFrame:
    """这几个联赛、这几个赛季的全部比赛，加上 as_of_utc（最早开球 − hours_before 小时）和那天的英国日期（第几天）。"""
    marks = ",".join("?" * len(leagues))
    m = pd.read_sql(
        f"select match_id, competition, season, home_team, away_team, match_date, kickoff_utc, date_precision "
        f"from matches where competition in ({marks}) and season between ? and ? order by match_id",
        con, params=[*leagues, first, last])
    f = settings["features"]
    kt = compute_known_times(m, f)
    hours = float(settings["injury_research"]["hours_before_kickoff"])
    as_of = pd.to_datetime(kt["kickoff_earliest_utc"]) - pd.Timedelta(hours=hours)
    m["as_of_utc"] = as_of.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    # “知道的” = 日期严格早于 as_of 那天（英国时间）的记录
    local = as_of.dt.tz_localize("UTC").dt.tz_convert(f["date_only_timezone"]).dt.tz_localize(None).dt.normalize()
    m["day"] = _day_numbers(local).astype(np.int64)
    return m


def build(matches: pd.DataFrame, players: PlayerData, team_to_club: dict[str, int]) -> pd.DataFrame:
    """每场一行：match_id、联赛、赛季、as_of_utc，主客队各四个数。同一队同一天只算一次。"""
    cache: dict = {}
    def side(team, day):
        key = (team, day)
        if key not in cache:
            cache[key] = players.side(team_to_club.get(team), day)
        return cache[key]
    rows = [[r.match_id, r.competition, r.season, r.as_of_utc, *side(r.home_team, r.day), *side(r.away_team, r.day)]
            for r in matches.itertuples()]
    return pd.DataFrame(rows, columns=COLUMNS)


def write_report(path: Path, out: pd.DataFrame, votes: pd.DataFrame, unmapped: list[str]) -> None:
    """覆盖报告：每个联赛、赛季，有数的比例、平均伤了几个人、伤的身价占比、主力掉了多少。"""
    g = out.assign(has=out["home_inj_n"].notna() & out["away_inj_n"].notna(),
                   inj=(out["home_inj_n"] + out["away_inj_n"]) / 2,
                   share=(out["home_inj_share"] + out["away_inj_share"]) / 2,
                   loss=(out["home_inj_top_loss"] + out["away_inj_top_loss"]) / 2)
    t = (g.groupby(["competition", "season"])
          .agg(场数=("match_id", "size"), 两队都有数=("has", "mean"), 平均每队伤几人=("inj", "mean"),
               伤的身价占比=("share", "mean"), 主力身价掉了=("loss", "mean"))
          .reset_index())
    lines = ["# 伤停特征报告（研究实验，src/clean/injury_features.py 生成）", "",
             f"- 球队对照：{len(votes)} 个队对上 Transfermarkt 编号（投票没有冲突）。",
             f"- 这几个赛季出现、但对不上编号的队（这些队的数是空值）：{('、'.join(unmapped)) or '没有'}",
             f"- 比赛：{len(out):,} 场，两队都有数的 {int(g['has'].sum()):,} 场。", "",
             "“主力身价掉了”= log(最贵 11 人身价和) − log(不算伤员时最贵 11 人身价和)，两队平均。", "",
             "| 联赛 | 赛季 | 场数 | 两队都有数 | 平均每队伤几人 | 伤的身价占比 | 主力身价掉了 |", "|---|---|---|---|---|---|---|"]
    for r in t.itertuples(index=False):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.0%} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(settings: dict, db_path: Path | None = None, raw_dir: Path | None = None, write_alias: bool = False) -> dict:
    cfg = settings["injury_research"]
    leagues = dict(cfg["leagues"])                         # 我们的联赛代码 → Transfermarkt 联赛编号
    raw = raw_dir or resolve_path(settings["paths"]["raw"])
    first, last = int(cfg["first_season"]), int(settings["seasons"]["dev_last"])
    m_first, m_last = int(cfg["mapping_seasons"][0]), int(cfg["mapping_seasons"][1])
    path = db_path or resolve_path(settings["paths"]["database"])
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)   # 只读打开，不改数据库
    try:
        ours = our_team_seasons(con, list(leagues), m_first, m_last)
        matches = load_matches(con, list(leagues), first, last, settings)
    finally:
        con.close()
    votes = vote(ours, tm_team_seasons(raw, leagues, m_first, m_last))
    decided = decide(votes)
    alias_path = resolve_path(settings["paths"]["team_aliases_transfermarkt"])
    if write_alias:
        write_aliases(alias_path, votes)
    alias = load_aliases(settings)
    check_aliases(alias, decided)
    team_to_club = {team: cid for cid, team in alias.items()}
    teams = set(matches["home_team"]) | set(matches["away_team"])
    unmapped = sorted(teams - set(team_to_club))

    s = raw / "salimt"
    players = PlayerData(pd.read_csv(s / "transfers.csv", usecols=["player_id", "transfer_date", "to_team_id"]),
                         pd.read_csv(s / "market_value.csv"), pd.read_csv(s / "injuries.csv"),
                         clubs=alias, top_n=int(cfg["top_n"]))
    out = build(matches, players, team_to_club)
    out_csv = resolve_path(settings["paths"]["injury_features"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8", float_format="%.6g")
    reports = resolve_path(settings["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    write_report(reports / "伤停特征报告.md", out, votes, unmapped)
    return {"对上的队": len(votes), "对不上的队": unmapped, "比赛": len(out),
            "两队都有数": int((out["home_inj_n"].notna() & out["away_inj_n"].notna()).sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=Path, help="数据库位置（默认 settings.yaml 的 paths.database）")
    p.add_argument("--raw", type=Path, help="原始数据文件夹（默认 settings.yaml 的 paths.raw）")
    p.add_argument("--write-aliases", action="store_true", help="按投票结果重新生成球队对照文件")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(run(load_settings(), a.db, a.raw, a.write_aliases))


if __name__ == "__main__":
    main()
