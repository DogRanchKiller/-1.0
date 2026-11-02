"""赛后对账（需求文档第 15 节、第 9 节“完整链路的分工和接口”接口 1、3）：把预测存档和实际比分对上，算每一版的分数。

做什么：
1. 读预测存档 predictions 表。同一场、同一模型、模型版本、kind 只取 prediction_id 最大的那一行，不改原始存档。
2. 比分、xG、红牌、收盘赔率一律经过取数层 FeatureStore 取，用途「复盘」（config/data_usage.yaml 允许）。
3. 每场已经踢完、有预测的比赛，出这几种“行”（scores.csv 一行 = 一个版本的一种概率）：
   - 每个 kind 一行“模型”：p_home / p_draw / p_away 是存档里模型自己的概率；
   - 这一版有融合结果就再出一行“融合”；
   - 收盘标杆按来源、公司、市场、时点独立保留，prediction_id 和模型身份为空；不随模型版本数增加。
     这场还没有收盘赔率（football-data 还没更新）就不出这一行。
4. 分数（都是越小越好，函数全部用 src/evaluate/metrics.py 现有的）：
   - 胜平负：log loss、Brier、RPS；
   - 比分、大于 2.5 球、双方进球的 log loss：从“模型比分矩阵”/“融合比分矩阵”算（src/common/markets.py）。
     preview 行没有矩阵，这几列留空；收盘赔率行只有大 2.5 球（有 2.5 球那条线的收盘赔率时），比分、双方进球留空。

哪些比赛对账（第 5 节“期末考试”、第 17 节“三种运行模式”）：
- 只对上线后赛季（2026/27 起）的比赛，用“实时”模式的取数层。上线后赛季在实时模式下可以用于「复盘」。
- 期末考试赛季（2024/25、2025/26）：实时模式下不允许「复盘」读，改用考试模式又等于“偷偷考了一次”，所以一律不对账，
  只在摘要里记下跳过了几场。开发赛季的存档不是真实成绩，也跳过。
- 出预测的时间（cutoff_utc）不早于开球的行（存档出错）不对账，记下来。

结果只用来记录和展示，绝不反过来调模型、融合权重、倾向门槛、信心等级的任何参数。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.config import season_role
from src.common.markets import btts, over_under, result_probs
from src.common.odds import devig
from src.evaluate.metrics import brier_per_match, log_loss_per_match, rps_per_match
from src.features import FeatureAccessError, FeatureStore
from src.features.known_time import to_utc_text
from src.features.usage import season_label
from src.review.closing_evidence import COLUMNS as EVIDENCE_COLUMNS, closing_coverage

USAGE = "复盘"                       # 取数层的用途（data_usage.yaml）
SOURCE_MODEL, SOURCE_FUSION, SOURCE_CLOSE = "模型", "融合", "收盘赔率"
CLOSE_KIND = "收盘（{book}）"        # 收盘赔率行的 kind 写法
RESULT_INDEX = {"H": 0, "D": 1, "A": 2}

# scores.csv 的列，严格按接口 3 的顺序
SCORE_COLUMNS = ["prediction_id", "match_id", "competition", "season", "kickoff_utc", "kind", "source",
                 "model", "model_version", "cutoff_utc", "created_at_utc", "late_minutes", "rescheduled",
                 "archive_status", "confidence", "benchmark_source", "bookmaker", "market", "timing", "line",
                 "qualified", "qualification", "evidence_reason",
                 "source_missing", "missing_since_utc", "restored_at_utc", "missing_reason", "last_sources_json",
                 "home_goals", "away_goals", "p_home", "p_draw", "p_away",
                 "logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts"]


# ======================= 读预测存档 =======================
def latest_predictions(con: sqlite3.Connection) -> pd.DataFrame:
    """每个 (比赛、模型、模型版本、预测时点) 只取最新编号；不同模型版本不能互相覆盖。"""
    has = con.execute("select 1 from sqlite_master where type = 'table' and name = 'predictions'").fetchone()
    if not has:
        return pd.DataFrame(columns=["prediction_id", "match_id", "competition", "season", "home_team", "away_team",
                                     "kickoff_utc", "match_date", "kind", "model", "model_version", "cutoff_utc",
                                     "created_at_utc", "p_home", "p_draw", "p_away", "detail"])
    return pd.read_sql("""
        select p.* from predictions p
        join (select match_id, model, model_version, kind, max(prediction_id) as pid from predictions
              group by match_id, model, model_version, kind) last
          on last.pid = p.prediction_id
        order by p.match_id, p.model, p.model_version, p.kind""", con)


# ======================= 概率、矩阵的整理 =======================
def clean_probs(p, tol: float) -> np.ndarray | None:
    """三个概率 → 加起来正好是 1 的数组。存档是 4 位小数，加起来可能差 0.0001，按比例补齐；
    有空值、负数，或者和 1 差超过 tol（review.prob_sum_tol）就当坏数据，返回 None。"""
    if p is None:
        return None
    a = np.asarray(p, dtype=float)
    if a.shape != (3,) or np.isnan(a).any() or (a < 0).any() or abs(a.sum() - 1) > tol:
        return None
    return a / a.sum()


def clean_matrix(m, tol: float) -> np.ndarray | None:
    """比分矩阵（接口 1：11 × 11 二维列表，6 位小数）→ 加起来正好是 1 的数组。

    121 个 6 位小数加起来可能和 1 差几十万分之一，比 markets.py 的检查（百万分之一）松，所以先在这里按 tol 检查、按比例补齐，
    再交给 markets.py。不是二维、有空值负数、和 1 差太多 → None（这一版当作没有矩阵，分数列留空）。
    """
    if m is None:
        return None
    try:
        a = np.asarray(m, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or min(a.shape) < 2 or np.isnan(a).any() or (a < 0).any() or abs(a.sum() - 1) > tol:
        return None
    return a / a.sum()


def score_metrics(p1x2: np.ndarray, matrix: np.ndarray | None, hg: int, ag: int) -> dict:
    """一种概率在这一场的分数。p1x2：[主胜, 平, 客胜]；matrix：比分矩阵（没有就是 None，比分 / 大 2.5 / 双方进球留空）。

    - 胜平负三个分数：metrics.py 的 log_loss_per_match、brier_per_match、rps_per_match（三列概率、结果 0 / 1 / 2）。
    - 比分 log loss：把矩阵拉平成一行（121 个比分各是一个结果），实际比分是第几个就是第几个结果，再用同一个 log loss。
      进了 10 球以上的那一队按最后一行 / 列算（矩阵最后一行 / 列就是“10 球及以上”，接口 1）。
    - 大 2.5 球、双方进球：从矩阵算“是”的概率（markets.over_under 的 win、markets.btts 的 yes），一维 log loss。
    """
    res = 0 if hg > ag else 1 if hg == ag else 2
    out = {"logloss_1x2": float(log_loss_per_match([p1x2], [res])[0]),
           "brier_1x2": float(brier_per_match([p1x2], [res])[0]),
           "rps_1x2": float(rps_per_match([p1x2], [res])[0]),
           "logloss_score": None, "logloss_over25": None, "logloss_btts": None}
    if matrix is not None:
        i, j = min(hg, matrix.shape[0] - 1), min(ag, matrix.shape[1] - 1)
        out["logloss_score"] = float(log_loss_per_match([matrix.ravel()], [i * matrix.shape[1] + j])[0])
        p_over = over_under(matrix, 2.5, "over")["win"]
        out["logloss_over25"] = float(log_loss_per_match([p_over], [int(hg + ag > 2)])[0])
        p_yes = btts(matrix)["yes"]
        out["logloss_btts"] = float(log_loss_per_match([p_yes], [int(hg > 0 and ag > 0)])[0])
    return out


def over25_logloss(p_over: float | None, hg: int, ag: int) -> float | None:
    """收盘赔率的大 2.5 球 log loss（没有 2.5 球那条线就是 None）。"""
    if p_over is None:
        return None
    return float(log_loss_per_match([p_over], [int(hg + ag > 2)])[0])
# ======================= 一版预测 =======================
def archive_metadata(kind: str, detail: dict) -> dict:
    """只解释出预测时已保存的元数据；缺信息写未知，不用今天的设置补算信心或准点状态。"""
    plan, late = detail.get("计划时点"), detail.get("晚了几分钟")
    rescheduled = detail.get("改期重出") is True
    fusion = detail.get("融合")
    confidence = fusion.get("信心等级") if isinstance(fusion, dict) else None
    status = "unknown"
    if kind == "preview":
        status = "preview"
    elif rescheduled:
        status = "rescheduled"
    elif (isinstance(plan, (int, float)) and not isinstance(plan, bool) and np.isfinite(plan)
          and isinstance(late, (int, float)) and not isinstance(late, bool) and np.isfinite(late)
          and plan > 0 and late >= 0 and kind == f"T-{plan:g}"):
        status = "late" if late > 0 else "on_time"
    return {"plan_hours": plan, "late_minutes": late, "rescheduled": rescheduled,
            "archive_status": status, "confidence": confidence}


def parse_version(row: pd.Series, tol: float) -> dict:
    """predictions 表的一行 → 复盘要用的一版预测。detail 里的键按接口 1（定时版本另有“计划时点”“比分矩阵”等，预览版没有）。"""
    try:
        detail = json.loads(row["detail"] or "{}")
    except (TypeError, ValueError):
        detail = {}
    if not isinstance(detail, dict):
        detail = {}
    model_matrix = clean_matrix(detail.get("模型比分矩阵"), tol)
    fusion_matrix = clean_matrix(detail.get("融合比分矩阵"), tol)
    # 融合概率：有融合矩阵就从矩阵算（和其他玩法同一个来源）；没有矩阵（预览版）用存档里的“融合概率”
    fusion_p = result_probs(fusion_matrix) if fusion_matrix is not None else clean_probs(detail.get("融合概率"), tol)
    return {
        "prediction_id": int(row["prediction_id"]), "kind": row["kind"], "model": row["model"],
        "model_version": row["model_version"], "cutoff_utc": to_utc_text(row["cutoff_utc"]),
        "created_at_utc": row["created_at_utc"], "kickoff_utc_at_prediction": row["kickoff_utc"],
        **archive_metadata(row["kind"], detail),
        "model_p": clean_probs([row["p_home"], row["p_draw"], row["p_away"]], tol),
        "model_matrix": model_matrix,
        # 存档里有矩阵、但格式不对（不是二维、加起来不是 1）：分数列留空，摘要里提醒
        "matrix_bad": [k for k, m in (("模型比分矩阵", model_matrix), ("融合比分矩阵", fusion_matrix))
                       if detail.get(k) is not None and m is None],
        "fusion_p": fusion_p, "fusion_matrix": fusion_matrix,
        "fusion_status": detail["融合"].get("状态") if isinstance(detail.get("融合"), dict) else None,
        "odds_p": clean_probs(detail.get("赔率概率"), tol),   # 出这一版时的赔率概率（预览页记的“赔率概率”）
    }


# ======================= 收盘赔率 =======================
def all_closings(odds: pd.DataFrame, books: list[str], method: str) -> list[dict]:
    """把取数层已准许的收盘赔率变成独立标杆，不在这里重复收盘证据的资格判断。

    胜平负和大小 2.5 球分别保留，来源不同也分别保留。没有完整、有效价格的市场不评分。
    """
    if odds.empty:
        return []
    out, seen = [], {}
    for row in odds.to_dict("records"):
        book, market = row.get("bookmaker"), row.get("market")
        if book not in books or market not in ("1x2", "ou"):
            continue
        if market == "ou" and row.get("line") != 2.5:
            continue
        cols = ("home", "draw", "away") if market == "1x2" else ("over", "under")
        try:
            prices = np.array([row.get(k) for k in cols], dtype=float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(prices).all() or (prices <= 1).any():
            continue
        line = 2.5 if market == "ou" else None
        identity = (row.get("source"), book, market, row.get("timing"), line)
        if identity in seen:
            if not np.array_equal(prices, seen[identity]):
                raise ValueError("同一独立收盘标杆有互相矛盾的价格，不能任选一条")
            continue
        seen[identity] = prices
        p = devig(prices, method)
        out.append({"book": book, "bookmaker": book, "benchmark_source": row.get("source"),
                    "market": market, "timing": row.get("timing"), "line": line, "qualified": row.get("qualified"),
                    "qualification": row.get("qualification"), "evidence_reason": row.get("reason"),
                    "p_1x2": p if market == "1x2" else None,
                    "p_over25": float(p[0]) if market == "ou" else None})
    return sorted(out, key=lambda c: (books.index(c["book"]), str(c["benchmark_source"]), c["market"]))


def pick_closing(odds: pd.DataFrame, books: list[str], method: str) -> dict | None:
    """按 review.close_bookmakers 的顺序，取第一家有完整胜平负收盘赔率的公司，去水后返回
    {"book", "p_1x2", "p_over25"}；同一家有 2.5 球那条线的大小球收盘赔率时 p_over25 是去水后“大”的概率，否则 None。"""
    closings = all_closings(odds, books, method)
    c = next((dict(c) for c in closings if c["market"] == "1x2"), None)
    if c is not None:
        ou = next((o for o in closings if o["market"] == "ou" and o["book"] == c["book"]
                   and o["benchmark_source"] == c["benchmark_source"]), None)
        c["p_over25"] = ou["p_over25"] if ou else None
    return c


# ======================= 收集：要复盘的比赛 =======================
@dataclass
class Collected:
    matches: list[dict] = field(default_factory=list)     # 每场一个字典（见 collect）
    skipped: dict = field(default_factory=dict)           # 跳过的原因 → 几场 / 几行
    warnings: list[str] = field(default_factory=list)
    closing_evidence: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=EVIDENCE_COLUMNS))
    closing_coverage: dict = field(default_factory=dict)

    def skip(self, why: str, n: int = 1) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + n


def review_closing_coverage(evidence: pd.DataFrame, match_ids, books) -> dict:
    """复盘只评大小2.5球，其他大小盘口不能填充它的分母；原始候选另存不改。"""
    scoped = evidence.copy()
    outside = scoped["market"].eq("ou") & scoped["line"].notna() & ~scoped["line"].eq(2.5)
    scoped.loc[outside, "qualified"] = False
    for idx in scoped.index[outside]:
        scoped.at[idx, "reasons"] = tuple(reason for reason in scoped.at[idx, "reasons"]
                                          if reason != "registered_source_declared_close") + ("not_ou_2_5",)
    result = closing_coverage(scoped, match_ids, bookmakers=books, markets=("1x2", "ou"))
    for group in result["by_bookmaker_market"]:
        group["line"] = 2.5 if group["market"] == "ou" else None
    return result


def evidence_records(evidence: pd.DataFrame) -> list[dict]:
    """证据写入JSON时空值是null，不输出NaN；完整原因仍然保留。"""
    return evidence.astype(object).where(pd.notna(evidence), None).to_dict("records")


def exclude_conflicting_closes(evidence: pd.DataFrame) -> pd.DataFrame:
    """单个标杆身份出现矛盾价格时排除该身份，评分和覆盖共同使用同一份审查结果。

    当前表的唯一键通常阻止重复；旧库或异常候选仍须防守。不能任取一条，也不能
    因为一家公司出错而连带丢掉另一家公司或其它玩法。原始价格保留供诊断。
    """
    evidence = evidence.copy()
    qualified = evidence[evidence["qualified"].eq(True)].copy()
    qualified["comparison_line"] = qualified["line"].where(qualified["market"].eq("ou"), None)
    keys = ["match_id", "source", "bookmaker", "market", "timing", "comparison_line"]
    for _, group in qualified.groupby(keys, dropna=False, sort=False):
        columns = ["home", "draw", "away"] if group.iloc[0]["market"] == "1x2" else ["over", "under"]
        if len(group[columns].drop_duplicates()) <= 1:
            continue
        evidence.loc[group.index, "qualified"] = False
        evidence.loc[group.index, "qualification"] = "excluded"
        evidence.loc[group.index, "reason"] = "conflicting_close_prices"
        evidence.loc[group.index, "evidence_note"] = "同一来源、公司、玩法、时点的收盘价格互相矛盾；该身份不评分、不计合格覆盖。"
        for idx in group.index:
            evidence.at[idx, "reasons"] = tuple(reason for reason in evidence.at[idx, "reasons"]
                                                if reason != "registered_source_declared_close") + ("conflicting_close_prices",)
    return evidence


def collect(con: sqlite3.Connection, fs: FeatureStore, settings: dict, now) -> Collected:
    """找出“已经踢完、有预测”的比赛，把每场要的东西都准备好（比分、xG、红牌、收盘赔率、各版本预测、开球时间）。

    fs 必须是“实时”模式的取数层；now 是“站在什么时候”复盘（UTC）。比分在 now 之前已经能知道的比赛才算已完赛。
    标签不在这里算（见 match_review.attach_labels）。
    """
    cfg = settings["review"]
    tol = float(cfg["prob_sum_tol"])
    now = to_utc_text(now)
    out = Collected()
    preds = latest_predictions(con)
    if preds.empty:
        out.closing_coverage = review_closing_coverage(out.closing_evidence, [], list(cfg["close_bookmakers"]))
        return out
    preds["season_role"] = [season_role(int(s), settings) for s in preds["season"]]
    for role, why in (("exam", "期末考试赛季的预测（实时模式不允许「复盘」读，不对账）"),
                      ("dev", "开发赛季的预测（不是上线后的真实成绩，不对账）")):
        n = int(preds.loc[preds["season_role"] == role, "match_id"].nunique())
        if n:
            out.skip(why, n)
    preds = preds[preds["season_role"] == "live"]

    # 比分、xG、红牌：每个联赛取一次（以前的比赛 + 用途「复盘」），只留要复盘的那几场
    stats: dict[int, dict] = {}
    for comp, part in preds.groupby("competition"):
        since = (pd.Timestamp(part["match_date"].min()) - pd.Timedelta(days=int(cfg["stats_margin_days"]))).strftime("%Y-%m-%d")
        try:
            hist = fs.historical_matches(now, USAGE, competition=comp, data=("比分", "射门、xG", "角球、黄牌"), since=since)
        except (FeatureAccessError, ValueError) as e:
            out.skip(f"取数层不给 {comp} 的比分（{type(e).__name__}）", int(part["match_id"].nunique()))
            out.warnings.append(f"{comp}：{e}")
            continue
        hist = hist[hist["match_id"].isin(part["match_id"])]
        for r in hist.to_dict("records"):
            stats[int(r["match_id"])] = r

    for mid, part in preds.groupby("match_id", sort=False):
        mid = int(mid)
        s = stats.get(mid)
        if s is None or pd.isna(s.get("home_goals")) or pd.isna(s.get("away_goals")):
            out.skip("还没踢完或还没有比分")
            continue
        kickoff_earliest = s["kickoff_earliest_utc"]
        versions = []
        for _, row in part.iterrows():
            if any(row[k] != s[k] for k in ("competition", "season", "home_team", "away_team")):
                out.skip("预测存档与当前比赛身份不一致（不能仅凭match_id对账）")
                out.warnings.append(f"比赛 {mid} 的 prediction_id {row['prediction_id']} 身份不一致，保留原存档但不评分")
                continue
            v = parse_version(row, tol)
            if v["cutoff_utc"] >= kickoff_earliest:
                out.skip("出预测的时间不早于开球（存档有问题，这一版不对账）")
                out.warnings.append(f"比赛 {mid} 的 {v['kind']}（prediction_id {v['prediction_id']}）"
                                    f"截止时间 {v['cutoff_utc']} 不早于开球 {kickoff_earliest}")
                continue
            if v["model_p"] is None:
                out.skip("模型概率有空值或加起来不是 1（这一版不对账）")
                continue
            for k in v["matrix_bad"]:
                out.warnings.append(f"比赛 {mid} 的 {v['kind']}：{k}格式不对，比分、大 2.5 球、双方进球几列留空")
            versions.append(v)
        if not versions:
            continue
        evidence = pd.DataFrame(columns=EVIDENCE_COLUMNS)
        try:
            evidence = fs.closing_evidence([mid], now, USAGE, bookmaker=list(cfg["close_bookmakers"]), market=["1x2", "ou"])
            evidence = exclude_conflicting_closes(evidence)
            close_odds = evidence[evidence["qualified"].eq(True)]
            closings = all_closings(close_odds, list(cfg["close_bookmakers"]), cfg["close_devig_method"])
            close = pick_closing(close_odds, list(cfg["close_bookmakers"]), cfg["close_devig_method"])
        except (FeatureAccessError, ValueError) as e:
            close, closings = None, []
            out.warnings.append(f"比赛 {mid} 收盘赔率取不到：{e}")
        out.closing_evidence = pd.concat([out.closing_evidence, evidence], ignore_index=True)
        hx, ax = s.get("home_xg"), s.get("away_xg")
        hr, ar = s.get("home_red"), s.get("away_red")
        first = part.iloc[0]
        out.matches.append({
            "match_id": mid, "competition": s["competition"], "season": int(s["season"]),
            "season_label": season_label(int(s["season"])), "home_team": s["home_team"], "away_team": s["away_team"],
            "match_date": s["match_date"],
            # 开球时间按比赛表现在的（改期后是新时间）；只有日期的是空
            "kickoff_utc": to_utc_text(s["kickoff_utc"]) if isinstance(s["kickoff_utc"], str) and s["kickoff_utc"] else None,
            "kickoff_earliest_utc": kickoff_earliest,
            "home_goals": int(s["home_goals"]), "away_goals": int(s["away_goals"]),
            "result": "H" if s["home_goals"] > s["away_goals"] else "D" if s["home_goals"] == s["away_goals"] else "A",
            "stats_source": s.get("stats_source"),
            **{k: s.get(k) if pd.notna(s.get(k)) else None for k in
               ("source_missing", "missing_since_utc", "restored_at_utc", "missing_reason", "last_sources_json")},
            "xg": None if pd.isna(hx) or pd.isna(ax) else (float(hx), float(ax)),
            "red": None if pd.isna(hr) or pd.isna(ar) else (int(hr), int(ar)),
            "close": close, "all_closings": closings, "versions": versions, "prediction_season": int(first["season"]),
            "closing_evidence": evidence_records(evidence),
            "closing_coverage": review_closing_coverage(evidence, [mid], list(cfg["close_bookmakers"])),
        })
    out.matches.sort(key=lambda m: (m["kickoff_earliest_utc"], m["match_id"]))
    out.closing_coverage = review_closing_coverage(out.closing_evidence, [m["match_id"] for m in out.matches],
                                                  list(cfg["close_bookmakers"]))
    return out


# ======================= scores.csv =======================
def kind_rank(kind: str, kinds: list[str]) -> int:
    """kind 在 review.kinds 里的位置（显示顺序）；没写进去的排在最后。"""
    return kinds.index(kind) if kind in kinds else len(kinds)


def score_rows(matches: list[dict], settings: dict) -> pd.DataFrame:
    """预测行保留模型版本和存档状态；标杆按比赛/来源/公司/市场/时点保留，完全独立于模型行。"""
    kinds = list(settings["review"]["kinds"])
    rows = []
    for m in matches:
        hg, ag = m["home_goals"], m["away_goals"]
        base = {"match_id": m["match_id"], "competition": m["competition"], "season": m["season"],
                "kickoff_utc": m["kickoff_utc"], "home_goals": hg, "away_goals": ag,
                **{k: m.get(k) for k in ("source_missing", "missing_since_utc", "restored_at_utc", "missing_reason", "last_sources_json")}}
        for v in sorted(m["versions"], key=lambda v: kind_rank(v["kind"], kinds)):
            for source, p, mat in ((SOURCE_MODEL, v["model_p"], v["model_matrix"]),
                                   (SOURCE_FUSION, v["fusion_p"], v["fusion_matrix"])):
                if p is None:
                    continue
                rows.append({**base, "prediction_id": v["prediction_id"], "kind": v["kind"], "source": source,
                             **{k: v.get(k) for k in ("model", "model_version", "cutoff_utc", "created_at_utc",
                                                      "late_minutes", "rescheduled", "archive_status", "confidence")},
                             "p_home": p[0], "p_draw": p[1], "p_away": p[2], **score_metrics(p, mat, hg, ag)})
        # 旧内存调用方只有 close 时仍可读；新版 all_closings 即使为空也不能回退、伪造缺失标杆。
        closings = m.get("all_closings", [m["close"]] if m.get("close") is not None else [])
        for c in closings:
            p = c["p_1x2"]
            met = score_metrics(p, None, hg, ag) if p is not None else {k: None for k in SCORE_COLUMNS if k.startswith(("logloss_", "brier_", "rps_"))}
            met["logloss_over25"] = over25_logloss(c.get("p_over25"), hg, ag)
            rows.append({**base, "prediction_id": None, "kind": CLOSE_KIND.format(book=c["book"]), "source": SOURCE_CLOSE,
                         **{k: c.get(k) for k in ("benchmark_source", "market", "timing", "line", "qualified", "qualification", "evidence_reason")},
                         "bookmaker": c.get("bookmaker", c["book"]),
                         "p_home": p[0] if p is not None else None,
                         "p_draw": p[1] if p is not None else None, "p_away": p[2] if p is not None else None, **met})
    df = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    df["prediction_id"] = df["prediction_id"].astype("Int64")   # 收盘赔率行为空，其余是整数（不写成 123.0）
    return df
