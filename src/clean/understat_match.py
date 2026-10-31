r"""把 Understat 射门明细镜像里的比赛和我们数据库里的比赛一场场对上，算出每场的 xG（需求文档 6.1“Understat 射门明细镜像”、第 4 节步骤 4）。

只做研究实验用（你决定，2026-10-31）：
  - 结果不导入数据库，只写到 data/research/understat_xg.csv（在 data/ 下，不上传 GitHub）。
  - 模型和回测要用，必须经过取数层（FeatureStore.historical_matches 的数据类别「Understat xG（研究）」），
    用途只能是「研究实验」，而且只能在开发模式下用（config/data_usage.yaml）。

原始文件：data/raw/understat_mirror/<联赛>/<赛季>.csv，一行是一次射门。这里用到的列：
  match_id（Understat 的比赛编号）、h_team / a_team（主客队）、h_goals / a_goals（比分）、date（日期）、
  h_a（射门的是主队 h 还是客队 a）、result（结果）、xG、situation（点球 Penalty、角球、定位球……）。

几个要注意的地方（2026-10-31 查原始数据发现的）：
  - 乌龙球（result = OwnGoal）记在“踢进乌龙的那名球员”那一边，xG 是 0。
    所以算比分时乌龙球要算给对面；算 xG 时它是 0，加不加都一样；算射门次数时不算（不是进攻方的射门）。
    这样处理后，射门明细加起来的进球数和文件里的比分，13,484 场里有 13,476 场一致（不一致的列在报告里）。
  - 日期不是英国当地时间（例如 2016-09-16 切尔西对利物浦，英国时间 20:00 开球，文件里写 23:00），
    所以和我们的比赛对日期时允许差 1 天。
  - 射正：Understat 的“射正”= 进球（Goal）+ 被扑出（SavedShot）。打中门柱（ShotOnPost）不算射正，和 football-data 的口径接近。

怎么对：
  1. 队名：Understat 用全名（例如 Manchester United），我们用 football-data 的写法（Man United）。
     对照表在 config/team_aliases_understat.yaml（程序“投票”找候选、人工逐条检查后写下）。
     这里每次运行还会重新投一次票（同联赛同赛季、日期差 ≤ 1 天、比分一样的比赛给对应的队名投票），
     投票结果和对照表不一致就报错停下，防止对照表写错。
  2. 比赛：按（联赛、赛季、主队、客队）找：联赛里同一对主客队一个赛季只踢一次。再核对日期（差 ≤ 1 天）和比分，
     不一致的不要，列在报告里。

运行方法：Claude 运行 `.venv\Scripts\python -m src.clean.understat_match`
结果：data/research/understat_xg.csv（每场一行）、data/reports/understat_对照报告.md、data/reports/understat_没对上.csv
"""
import argparse
import csv
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from src.clean.football_data_import import load_teams
from src.common.config import load_settings, resolve_path

log = logging.getLogger("understat_match")

ON_TARGET = {"Goal", "SavedShot"}      # 射正：进了，或者被门将扑出
MIN_VOTES = 3                          # 一个队名至少得几票
MIN_SHARE = 0.6                        # 得票最多的写法要占全部票数的多少
DATE_TOLERANCE_DAYS = 1                # 日期允许差几天（Understat 的日期不是英国时间）
COLUMNS = ["match_id", "competition", "season", "home_team", "away_team", "understat_id", "date_gap_days",
           "home_goals", "away_goals", "home_xg", "away_xg", "home_npxg", "away_npxg",
           "home_shots", "away_shots", "home_sot", "away_sot", "home_pens", "away_pens"]


def read_shots(raw_dir: Path, leagues, first: int, last: int) -> pd.DataFrame:
    """读原始层里全部射门明细，加上 competition（我们的联赛代码）列。没下载的文件跳过。"""
    parts = []
    for code in leagues:
        for season in range(first, last + 1):
            f = raw_dir / "understat_mirror" / code / f"{season}.csv"
            if f.exists():
                parts.append(pd.read_csv(f).assign(competition=code, file_season=season))
    if not parts:
        raise FileNotFoundError(f"{raw_dir / 'understat_mirror'} 里没有射门明细，先运行 src.collect.understat_mirror")
    return pd.concat(parts, ignore_index=True)


def match_totals(shots: pd.DataFrame) -> pd.DataFrame:
    """射门明细 → 每场一行：比分（文件里的）、射门加起来的进球、xG、不含点球的 xG（npxG）、射门、射正、点球次数。

    乌龙球算给对面（见文件开头）。一场比赛一次射门都没有的一边，各项是 0。
    """
    s = shots.copy()
    own = s["result"] == "OwnGoal"
    s["side"] = s["h_a"].where(~own, s["h_a"].map({"h": "a", "a": "h"}))   # 进球算给哪边
    s["is_goal"] = s["result"].isin(["Goal", "OwnGoal"]).astype(int)
    s["is_shot"] = (~own).astype(int)
    s["is_sot"] = s["result"].isin(ON_TARGET).astype(int)
    s["is_pen"] = (s["situation"] == "Penalty").astype(int)
    s["npxg"] = s["xG"].where(s["situation"] != "Penalty", 0.0)
    keys = ["competition", "file_season", "match_id"]
    info = s.groupby(keys).agg(h_team=("h_team", "first"), a_team=("a_team", "first"),
                               h_goals=("h_goals", "first"), a_goals=("a_goals", "first"), date=("date", "first"),
                               n_info=("h_team", "nunique"))
    if (info["n_info"] > 1).any():
        raise ValueError("同一个 Understat 比赛编号出现了不同的主队，原始文件可能有问题")
    per_side = s.groupby(keys + ["side"]).agg(goals=("is_goal", "sum"), xg=("xG", "sum"), npxg=("npxg", "sum"),
                                              shots=("is_shot", "sum"), sot=("is_sot", "sum"), pens=("is_pen", "sum"))
    wide = per_side.unstack("side", fill_value=0)
    out = info.drop(columns="n_info").copy()
    for col in ["goals", "xg", "npxg", "shots", "sot", "pens"]:
        for side, name in (("h", "home"), ("a", "away")):
            v = wide[(col, side)] if (col, side) in wide.columns else 0
            out[f"{name}_{col}"] = pd.Series(v, index=wide.index).reindex(out.index).fillna(0)
    out = out.reset_index().rename(columns={"file_season": "season", "match_id": "understat_id"})
    out["understat_id"] = out["understat_id"].astype(int)
    out["day"] = pd.to_datetime(out["date"]).dt.normalize()
    for c in ["h_goals", "a_goals", "home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot",
              "home_pens", "away_pens"]:
        out[c] = out[c].astype(int)
    out["shots_add_up"] = (out["home_goals"] == out["h_goals"]) & (out["away_goals"] == out["a_goals"])
    return out


def load_db_matches(con: sqlite3.Connection, leagues, first: int, last: int) -> pd.DataFrame:
    """数据库里这几个联赛、这几个赛季的比赛和 football-data 的比分（我们数据库的主比分来源）。"""
    marks = ",".join("?" * len(leagues))
    df = pd.read_sql(
        f"select m.match_id, m.competition, m.season, m.match_date, m.home_team, m.away_team, "
        f"s.home_goals, s.away_goals from matches m "
        f"left join match_stats s on s.match_id = m.match_id and s.source = 'football-data' "
        f"where m.competition in ({marks}) and m.season between ? and ?",
        con, params=[*leagues, first, last])
    df["day"] = pd.to_datetime(df["match_date"])
    return df


def vote(us: pd.DataFrame, db: pd.DataFrame) -> dict:
    """按“同联赛同赛季、日期差 ≤ 1 天、比分一样”投票：{(联赛, Understat 队名): {我们的队名: 票数}}。
    一场 Understat 比赛找到 k 场候选，每场候选给主队、客队的对应各投 1/k 票。"""
    votes: dict = defaultdict(lambda: defaultdict(float))
    idx = defaultdict(list)
    for r in db.dropna(subset=["home_goals"]).itertuples():
        idx[(r.competition, int(r.season), int(r.home_goals), int(r.away_goals))].append(r)
    for r in us.itertuples():
        cands = [m for m in idx[(r.competition, int(r.season), r.h_goals, r.a_goals)]
                 if abs((m.day - r.day).days) <= DATE_TOLERANCE_DAYS]
        for m in cands:
            votes[(r.competition, r.h_team)][m.home_team] += 1 / len(cands)
            votes[(r.competition, r.a_team)][m.away_team] += 1 / len(cands)
    return votes


def decide(votes: dict) -> tuple[dict, list[dict]]:
    """每个 Understat 队名取得票最多的我们的队名。返回（{(联赛, 队名): 我们的队名}, 报告表格）。"""
    mapping, table = {}, []
    for (comp, name), c in sorted(votes.items()):
        total = sum(c.values())
        ranked = sorted(c.items(), key=lambda kv: -kv[1])
        ours, top = ranked[0]
        ok = top >= MIN_VOTES and top / total >= MIN_SHARE
        if ok:
            mapping[(comp, name)] = ours
        table.append({"联赛": comp, "Understat 队名": name, "我们的队名": ours, "得票": round(top, 1),
                      "占比": round(top / total, 3), "第二名": ranked[1][0] if len(ranked) > 1 else "",
                      "结果": "定下" if ok else "票数不够"})
    return mapping, table


def load_aliases(settings: dict) -> dict[str, str]:
    """读 config/team_aliases_understat.yaml：{Understat 队名: 我们的队名}。检查我们的队名都在 teams.yaml 里、一个名字没写两次。"""
    _, names = load_teams(settings)
    ours = set(names.values())
    path = resolve_path(settings["paths"]["team_aliases_understat"])
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    alias: dict[str, str] = {}
    for team, us_names in (data.get("对照") or {}).items():
        if team not in ours:
            raise ValueError(f"{path.name}：{team} 不在 teams.yaml 里")
        for n in ([us_names] if isinstance(us_names, str) else us_names):
            if n in alias:
                raise ValueError(f"{path.name}：Understat 队名 {n} 写了两次")
            alias[n] = team
    return alias


def check_aliases(alias: dict[str, str], decided: dict, all_names: set[tuple[str, str]]) -> None:
    """对照表和投票结果核对：数据里出现的队名都要在对照表里；投票定下来的，必须和对照表一样。不一致就报错停下。"""
    missing = sorted(n for _, n in all_names if n not in alias)
    if missing:
        raise ValueError(f"这些 Understat 队名不在 team_aliases_understat.yaml 里：{missing}")
    wrong = [f"{comp} {n}：对照表写的是 {alias[n]}，投票是 {ours}"
             for (comp, n), ours in sorted(decided.items()) if alias[n] != ours]
    if wrong:
        raise ValueError("对照表和投票结果不一致：\n  " + "\n  ".join(wrong))


def align(us: pd.DataFrame, db: pd.DataFrame, alias: dict[str, str]) -> pd.DataFrame:
    """按（联赛、赛季、主队、客队）找我们的比赛，再核对日期和比分。每场 Understat 比赛一行，“结果”列说明对上没有。"""
    idx = {(r.competition, int(r.season), r.home_team, r.away_team): r for r in db.itertuples()}
    rows = []
    for r in us.itertuples():
        home, away = alias.get(r.h_team), alias.get(r.a_team)
        m = idx.get((r.competition, int(r.season), home, away))
        res, mid, gap = "对上", None, None
        if m is None:
            res = "数据库里没有这场"
        else:
            mid, gap = int(m.match_id), abs((m.day - r.day).days)
            if pd.isna(m.home_goals) or (int(m.home_goals), int(m.away_goals)) != (r.h_goals, r.a_goals):
                res = "比分不同"
            elif gap > DATE_TOLERANCE_DAYS:
                res = f"日期差 {gap} 天"
            elif not r.shots_add_up:
                res = "射门明细加起来和比分不一致"
        rows.append({"结果": res, "match_id": mid, "date_gap_days": gap, "home_team": home, "away_team": away,
                     **{k: getattr(r, k) for k in ["competition", "season", "understat_id", "h_team", "a_team", "date",
                                                   "h_goals", "a_goals", "home_goals", "away_goals",
                                                   "home_xg", "away_xg", "home_npxg", "away_npxg",
                                                   "home_shots", "away_shots", "home_sot", "away_sot",
                                                   "home_pens", "away_pens"]}})
    out = pd.DataFrame(rows)
    ok = out[out["结果"] == "对上"]
    dup = ok["match_id"][ok["match_id"].duplicated()]
    if not dup.empty:
        raise ValueError(f"有几场 Understat 比赛对到了我们同一场比赛：{sorted(dup.unique())[:10]}")
    return out


def write_report(path: Path, aligned: pd.DataFrame, db: pd.DataFrame, table: list[dict], shots: pd.DataFrame) -> None:
    """对照报告（给人看的 Markdown）：每联赛每赛季对上多少场、没对上的原因、数据库里多出来的比赛、队名投票结果。"""
    ok = aligned[aligned["结果"] == "对上"]
    used = set(ok["match_id"].astype(int))
    lines = ["# Understat 射门明细镜像 → 我们的比赛 对照报告", "",
             "程序 `src/clean/understat_match.py` 生成。只做研究实验用，结果不导入数据库（需求文档 6.1“Understat 射门明细镜像”）。", "",
             f"- 射门明细 {len(shots):,} 行，比赛 {len(aligned):,} 场；对上 {len(ok):,} 场。",
             f"- 乌龙球 {int((shots['result'] == 'OwnGoal').sum()):,} 个，算给对面；"
             f"点球 {int((shots['situation'] == 'Penalty').sum()):,} 次，平均 xG {shots.loc[shots['situation'] == 'Penalty', 'xG'].mean():.3f}。",
             "", "## 没对上的原因", ""]
    bad = aligned[aligned["结果"] != "对上"]
    if bad.empty:
        lines.append("没有。")
    for res, n in bad["结果"].value_counts().items():
        lines.append(f"- {res}：{n} 场")
    lines += ["", "## 每个联赛每个赛季", "",
              "“数据库多出来的”= 我们数据库里有、Understat 镜像里没有的比赛（2021/22 镜像只到 2021-12-12，所以多出来很多）。", "",
              "| 联赛 | 赛季 | Understat 场数 | 对上 | 没对上 | 数据库场数 | 数据库多出来的 |", "|---|---|---|---|---|---|---|"]
    for (comp, season), part in aligned.groupby(["competition", "season"]):
        dbp = db[(db["competition"] == comp) & (db["season"] == season)]
        n_ok = int((part["结果"] == "对上").sum())
        extra = int((~dbp["match_id"].isin(used)).sum())
        lines.append(f"| {comp} | {season} | {len(part)} | {n_ok} | {len(part) - n_ok} | {len(dbp)} | {extra} |")
    lines += ["", "## 队名投票", "", "对照表 config/team_aliases_understat.yaml 和投票结果一致才会运行到这里。", "",
              "| 联赛 | Understat 队名 | 我们的队名 | 得票 | 占比 | 结果 |", "|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['联赛']} | {t['Understat 队名']} | {t['我们的队名']} | {t['得票']} | {t['占比']} | {t['结果']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(settings: dict, db_path: Path | None = None) -> dict:
    cfg = settings["understat_mirror"]
    leagues = list(cfg["leagues"])
    first, last = int(cfg["first_season"]), int(cfg["last_season"])
    shots = read_shots(resolve_path(settings["paths"]["raw"]), leagues, first, last)
    us = match_totals(shots)
    path = db_path or resolve_path(settings["paths"]["database"])
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)   # 只读打开，不改数据库
    try:
        db = load_db_matches(con, leagues, first, last)
    finally:
        con.close()
    decided, table = decide(vote(us, db))
    alias = load_aliases(settings)
    names = {(r.competition, n) for r in us.itertuples() for n in (r.h_team, r.a_team)}
    check_aliases(alias, decided, names)
    aligned = align(us, db, alias)

    out_csv = resolve_path(settings["paths"]["understat_xg"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ok = aligned[aligned["结果"] == "对上"].copy()
    ok["match_id"] = ok["match_id"].astype(int)
    ok[COLUMNS].sort_values("match_id").to_csv(out_csv, index=False, encoding="utf-8")
    reports = resolve_path(settings["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    aligned[aligned["结果"] != "对上"].to_csv(reports / "understat_没对上.csv", index=False, encoding="utf-8-sig",
                                             quoting=csv.QUOTE_MINIMAL)
    write_report(reports / "understat_对照报告.md", aligned, db, table, shots)
    return {"比赛": len(aligned), "对上": len(ok), "没对上": aligned.loc[aligned["结果"] != "对上", "结果"].value_counts().to_dict()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=Path, help="数据库位置（默认 settings.yaml 的 paths.database）")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(run(load_settings(), a.db))


if __name__ == "__main__":
    main()
