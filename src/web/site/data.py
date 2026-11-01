"""正式网站要的数据，一次读好（需求文档第 9 节接口 1、3、4）。只读，不出预测、不跑模型、不联网。

从哪里读：
1. 预测存档 predictions 表（接口 1）：直接读。同一场、同一个版本（kind）有几行时，取 prediction_id 最大的那行。
   detail 里的键：预览版（preview）的“融合概率”“融合”（倾向、信心等级、不确定因素）等；定时版本（T-24 / T-6 / T-1）另有
   “计划时点”“晚了几分钟”“模型比分矩阵”“融合比分矩阵”“首发”。矩阵只读不算：没存矩阵的版本，页面上写“这一版没有存比分矩阵”。
2. 数据库里的赛程、变盘记录、竞彩赔率：经过取数层（src/features 的 FeatureStore，实时模式），用途「显示」（config/data_usage.yaml）。
   比分（赛后才有的答案）在数据字典里没有「显示」这个用途，网站上的比分是赛后和预测对照着看的，按「复盘」取。
   中文队名、比赛状态（改期、取消）和预览页一样直接读 teams、matches 表（只给人看，不进模型）。
3. data/review/（接口 3，K2 写）：scores.csv、matches/<match_id>.json、reports/。文件不存在时页面上写“还没有复盘数据”。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureStore
from src.features.errors import FeatureAccessError
from src.features.known_time import to_utc_text

# scores.csv 的列（需求文档第 9 节接口 3）
SCORE_COLUMNS = ["prediction_id", "match_id", "competition", "season", "kickoff_utc", "kind", "source", "home_goals",
                 "away_goals", "p_home", "p_draw", "p_away", "logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score",
                 "logloss_over25", "logloss_btts"]


@dataclass
class SiteData:
    """生成整站要的全部数据。match_id 一律是 int。"""
    now: datetime
    settings: dict
    matches: dict[int, dict] = field(default_factory=dict)       # 要做单场页的比赛：联赛、主客队、开球时间、状态……
    preds: dict[int, dict[str, dict]] = field(default_factory=dict)   # match_id → 版本 → 那一行（detail 已解析成字典）
    results: dict[int, tuple[int, int]] = field(default_factory=dict)  # 已经能知道的比分 (主, 客)
    odds: dict[int, pd.DataFrame] = field(default_factory=dict)   # 开球前的变盘记录（用途「显示」）
    lottery: dict[int, pd.DataFrame] = field(default_factory=dict)
    team_zh: dict[str, str] = field(default_factory=dict)
    schedule: pd.DataFrame | None = None                          # 最近一段时间已经开球的赛程（联赛 + 杯赛、欧战）
    scores: pd.DataFrame | None = None                            # data/review/scores.csv；没有是 None
    scores_note: str | None = None                                # scores.csv 读不了的原因
    reviews: dict[int, dict] = field(default_factory=dict)        # data/review/matches/<match_id>.json
    reports: list[Path] = field(default_factory=list)             # data/review/reports/ 里的周报、月报
    notes: list[str] = field(default_factory=list)                # 读数据时碰到的问题，写在首页上
    comp_names: dict[str, str] = field(default_factory=dict)
    page_ids: set[int] = field(default_factory=set)               # 要做单场页的比赛（见 load 第 4 步）
    levels: dict[int, int | None] = field(default_factory=dict)   # scores.csv 里每个 prediction_id 当时的信心等级

    def comp(self, code: str) -> str:
        return self.comp_names.get(code, code)

    def zh(self, team: str) -> str:
        return self.team_zh.get(team) or team                     # 没有中文名的先显示英文

    def latest(self, mid: int) -> tuple[str, dict] | None:
        """这场比赛最新的一个版本：出得最晚的那版（cutoff_utc 最大；一样时 prediction_id 大的）。"""
        rows = self.preds.get(mid) or {}
        if not rows:
            return None
        kind = max(rows, key=lambda k: (str(rows[k]["cutoff_utc"] or ""), rows[k]["prediction_id"]))
        return kind, rows[kind]


# ---------- 预测存档 ----------
def read_predictions(con, kinds: list[str]) -> dict[int, dict[str, dict]]:
    """每场每个版本取 prediction_id 最大的那行（接口 1）。只要 settings site.versions 里的版本。表还没建时返回空。"""
    q = ",".join("?" * len(kinds))
    sql = f"""select p.* from predictions p join (
                select match_id, kind, max(prediction_id) as pid from predictions where kind in ({q})
                group by match_id, kind) last on last.pid = p.prediction_id"""
    try:
        df = pd.read_sql(sql, con, params=list(kinds))
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return {}
    out: dict[int, dict[str, dict]] = {}
    for r in df.to_dict("records"):
        try:
            r["detail"] = json.loads(r["detail"] or "{}")
        except (TypeError, ValueError):
            r["detail"] = {}                 # detail 坏了也不让整站出错，这一版的细节显示成空
        out.setdefault(int(r["match_id"]), {})[r["kind"]] = r
    return out


def read_detail_levels(con, prediction_ids) -> dict[int, int | None]:
    """战绩页按信心等级分组用：scores.csv 里每个 prediction_id 那一行存的信心等级（detail 的 融合 → 信心等级）。"""
    ids = sorted({int(i) for i in prediction_ids if pd.notna(i)})
    out: dict[int, int | None] = {}
    for k in range(0, len(ids), 500):                # SQLite 一次能放的问号有限，分批查
        part = ids[k:k + 500]
        q = ",".join("?" * len(part))
        try:
            rows = con.execute(f"select prediction_id, detail from predictions where prediction_id in ({q})", part)
        except sqlite3.OperationalError:
            return out
        for pid, detail in rows:
            try:
                out[int(pid)] = (json.loads(detail or "{}").get("融合") or {}).get("信心等级")
            except (TypeError, ValueError, AttributeError):
                out[int(pid)] = None
    return out


def matrix(detail: dict, key: str, tol: float) -> np.ndarray | None:
    """detail 里存的比分矩阵（"模型比分矩阵" / "融合比分矩阵"）→ 数组。没有、格式不对、加起来和 1 差太多都返回 None。

    存档里是 6 位小数，121 个格子加起来可能和 1 差一点点（舍入），差 tol 以内按比例补成正好 1，
    这样才能交给 src/common/markets.py 算各玩法（它只允许差百万分之一）。只是把舍入补回来，不是重新算矩阵。
    """
    raw = detail.get(key)
    if raw is None:
        return None
    try:
        m = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if m.ndim != 2 or min(m.shape) < 2 or np.isnan(m).any() or (m < 0).any() or abs(m.sum() - 1) > tol:
        return None
    return m / m.sum()


# ---------- data/review/ ----------
def read_review(folder: Path, data: SiteData) -> None:
    """读 K2 生成的对账、复盘结果。没有文件夹、没有文件都不报错（页面上写“还没有复盘数据”）。"""
    path = folder / "scores.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            missing = [c for c in SCORE_COLUMNS if c not in df.columns]
            if missing:
                data.scores_note = f"scores.csv 缺这几列：{'、'.join(missing)}（格式见需求文档第 9 节接口 3）"
            else:
                df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
                data.scores = df.dropna(subset=["match_id"]).astype({"match_id": int})
        except (OSError, ValueError, pd.errors.ParserError) as err:
            data.scores_note = f"scores.csv 读不了：{err}"
    for p in sorted((folder / "matches").glob("*.json")) if (folder / "matches").exists() else []:
        try:
            data.reviews[int(p.stem)] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data.notes.append(f"单场复盘 {p.name} 读不了，跳过")
    if (folder / "reports").exists():
        data.reports = sorted((folder / "reports").glob("*.md"), reverse=True)


# ---------- 整站要的数据 ----------
def load(con, settings: dict, now: datetime) -> SiteData:
    """读整站要的数据。con：数据库连接（建议只读打开）；now：现在（UTC，带时区）。"""
    from src.common.config import resolve_path

    cfg = settings["site"]
    data = SiteData(now=now, settings=settings)
    names = {code: name for group in settings["leagues"].values() for code, name in group.items()}
    for code, c in (settings.get("wikipedia", {}).get("competitions") or {}).items():
        names.setdefault(code, str(c.get("名称", code)).split("（")[0])
    data.comp_names = names

    # 1. 预测存档
    data.preds = read_predictions(con, list(cfg["versions"]))

    # 2. 赛程：从“预测过的最早那场”和“球队页要看的最近几天”里更早的那天，到未来几天（用途「显示」）
    fs = FeatureStore(con, "实时", settings)
    nowt = to_utc_text(now)
    start = now - timedelta(days=cfg["team_days_back"])
    kicks = [r["kickoff_utc"] or r["match_date"] for rows in data.preds.values() for r in rows.values()
             if (r["kickoff_utc"] or r["match_date"])]
    if kicks:
        start = min(start, pd.Timestamp(min(kicks)).tz_localize("UTC").to_pydatetime() - timedelta(days=1))
    end = now + timedelta(days=max(cfg["days_ahead"], cfg["team_days_ahead"]))
    targets = fs.prediction_targets(start, end, "显示")
    site_leagues = set(settings["leagues"][cfg["leagues"]])
    for t in targets.to_dict("records"):
        mid = int(t["match_id"])
        kick = t.get("kickoff_utc") if isinstance(t.get("kickoff_utc"), str) else None
        data.matches[mid] = {"match_id": mid, "competition": t["competition"], "season": int(t["season"]),
                             "home_team": t["home_team"], "away_team": t["away_team"], "kickoff_utc": kick,
                             "match_date": str(t["match_date"]), "kickoff_earliest_utc": t["kickoff_earliest_utc"],
                             "date_precision": (t.get("date_precision") if isinstance(t.get("date_precision"), str)
                                                else None) or "day",
                             "site_league": t["competition"] in site_leagues}
    # 预测过、但不在上面时间段里的比赛（例如以前的赛季），用存档那一行的信息
    for mid, rows in data.preds.items():
        if mid not in data.matches:
            r = next(iter(rows.values()))
            data.matches[mid] = {"match_id": mid, "competition": r["competition"], "season": int(r["season"]),
                                 "home_team": r["home_team"], "away_team": r["away_team"], "kickoff_utc": r["kickoff_utc"],
                                 "match_date": str(r["match_date"]), "kickoff_earliest_utc": r["kickoff_utc"] or r["match_date"],
                                 "date_precision": "day", "site_league": r["competition"] in site_leagues}

    # 3. 对账结果（K2）；scores.csv 里有、上面还没有的比赛也做单场页（战绩页要链接过去）
    read_review(resolve_path(cfg["review_dir"]), data)
    if data.scores is not None:
        for r in data.scores.drop_duplicates("match_id").to_dict("records"):
            mid = int(r["match_id"])
            if mid not in data.matches:
                row = con.execute("select competition, season, home_team, away_team, kickoff_utc, match_date"
                                  " from matches where match_id = ?", (mid,)).fetchone()
                if row is None:
                    continue                 # 数据库里没有这场（不该发生），不做页面
                data.matches[mid] = {"match_id": mid, "competition": row[0], "season": int(row[1]), "home_team": row[2],
                                     "away_team": row[3], "kickoff_utc": row[4], "match_date": str(row[5]),
                                     "kickoff_earliest_utc": row[4] or row[5], "date_precision": "day",
                                     "site_league": row[0] in site_leagues}

    # 4. 哪些比赛做单场页：预测过的、对过账的、网站联赛里最近几天和未来几天的（首页上列出来的）
    lo, hi = to_utc_text(now - timedelta(days=cfg["days_back"])), to_utc_text(now + timedelta(days=cfg["days_ahead"]))
    data.page_ids = {mid for mid, m in data.matches.items()
                     if mid in data.preds or (m["site_league"] and lo <= m["kickoff_earliest_utc"] < hi)}
    if data.scores is not None:
        data.page_ids |= set(data.scores["match_id"]) & set(data.matches)
        data.levels = read_detail_levels(con, data.scores["prediction_id"])

    # 4.1 只给人看的：中文队名、比赛状态（和预览页 add_display_info 一样直接读，不进模型）
    ids = list(data.matches)
    teams = {m[k] for m in data.matches.values() for k in ("home_team", "away_team")}
    data.team_zh = {t: n for t, n in con.execute("select team_id, name_zh from teams") if n and t in teams}
    status = {}
    for k in range(0, len(ids), 500):
        part = ids[k:k + 500]
        status.update(dict(con.execute(f"select match_id, status from matches where match_id in ({','.join('?' * len(part))})",
                                       part)))
    for mid, m in data.matches.items():
        m["status"] = status.get(mid)

    # 5. 比分（赛后的答案）：用途「复盘」。实时模式下期末考试赛季（2024/25、2025/26）不能这样取，只取上线后的赛季；
    #    以前赛季的比分（如果有）从 scores.csv 补
    live_first = con.execute("select min(coalesce(kickoff_utc, match_date)) from matches where season >= ?",
                             (int(settings["seasons"]["live_first"]),)).fetchone()[0]
    since = max(to_utc_text(start), to_utc_text(pd.Timestamp(live_first))) if live_first else to_utc_text(start)
    try:
        hist = fs.historical_matches(nowt, "复盘", competition=None, data=("比分",), since=since)
        for r in hist[["match_id", "home_goals", "away_goals"]].dropna().itertuples(index=False):
            data.results[int(r.match_id)] = (int(r.home_goals), int(r.away_goals))
    except (FeatureAccessError, ValueError) as err:
        data.notes.append(f"比分暂时取不到：{str(err).splitlines()[0]}")
    if data.scores is not None:
        for r in data.scores.dropna(subset=["home_goals", "away_goals"]).drop_duplicates("match_id").itertuples(index=False):
            data.results.setdefault(int(r.match_id), (int(r.home_goals), int(r.away_goals)))

    # 6. 赛程（球队页的“最近踢过的”，含杯赛、欧战；用途「显示」）
    data.schedule = fs.schedule(nowt, "显示", since=now - timedelta(days=cfg["team_days_back"]))

    # 7. 要做单场页的比赛的变盘记录、竞彩赔率（用途「显示」；变盘记录只给开球前的价格）
    for mid in sorted(data.page_ids):
        try:
            data.odds[mid] = fs.odds_changes(mid, nowt, "显示")
            data.lottery[mid] = fs.lottery_odds(mid, nowt, "显示")
        except (FeatureAccessError, ValueError) as err:
            data.notes.append(f"比赛 {mid} 的赔率取不到：{str(err).splitlines()[0]}")
    return data
