"""逐场正式预测链路的只读验收，不补预测、不运行模型、不生成评分。

本模块只把已有证据对齐。所有原始预测逐行检查，绝不用最大编号遮掉早期证据。
宽限只解释“为什么还没有记录”，不把迟到改成准点。数据库主库只读；取数层
会在连接的 TEMP 区建立已知时间表。调用方应使用 open_readonly，不能使用会
禁止 TEMP 表的 PRAGMA query_only，也不应传入尚未提交的写事务。
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.config import expand_league_groups, resolve_path, season_role
from src.common.markets import result_probs
from src.features import FeatureAccessError, FeatureStore
from src.features.known_time import compute_known_times
from src.models.registry import model_config
from src.predict.preview import model_version
from src.review.reconcile import (SOURCE_FUSION, SOURCE_MODEL, exclude_conflicting_closes,
                                  parse_version, score_metrics)


IDENTITY = ("match_id", "competition", "season", "home_team", "away_team", "match_date")
PROB_COLS = ("p_home", "p_draw", "p_away")
METRICS = ("logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts")
CLOSE_BOOKS = {"平博 Pinnacle", "多家平均"}
SLOT_LABELS = {
    "not_due": "尚未到计划时点", "awaiting_scheduler": "已到点，等待本轮调度与计算",
    "overdue": "超过调度宽限，仍缺有效存档", "missed_window": "窗口已过，缺有效存档（现策略不补跑）",
    "before_activation": "早于跟踪启用，不追认为漏跑", "not_applicable": "不适用定时验收",
    "on_time": "准点存档", "late": "已存档，但实际迟到", "invalid_archive": "已有存档，但证据不适用或核验失败",
}
SUMMARY_KEYS = ("match_count", "recorded_slot_count", "late_slot_count", "pending_slot_count",
                "overdue_slot_count", "missed_slot_count", "invalid_slot_count", "archived_complete_count",
                "scored_complete_count", "complete_on_time_count", "source_missing_count",
                "closing_complete_count", "due_slot_count")


def _time(value):
    """证据时间保留秒以下精度；截断可能把未来或迟到记录变成准点。"""
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return None
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return None


def _text(ts):
    return None if ts is None else ts.tz_localize(None).isoformat(sep=" ")


def _number(value):
    try:
        if isinstance(value, bool):
            return None
        num = float(value)
        return num if math.isfinite(num) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _int(value):
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _same_time(a, b):
    ta, tb = _time(a), _time(b)
    return ta is not None and tb is not None and ta == tb


def _probs(values, tol):
    nums = [_number(x) for x in values]
    if len(nums) != len(PROB_COLS) or any(x is None or x < 0 or x > 1 for x in nums):
        return None
    total = sum(nums)
    return np.asarray(nums) / total if total > 0 and abs(total - 1) <= tol else None


def _columns(con, table):
    return {row[1] for row in con.execute(f"pragma table_info({table})")}


def _plain(value):
    """把 pandas/numpy 的标量与空值变成严格 JSON 数据。"""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _archive(row, match, hours, due, end, now, scope, tol):
    issues = []
    if any(row.get(k) != match.get(k) for k in IDENTITY):
        issues.append("归档比赛身份与当前比赛不一致（仅能核对已有归档字段，round/leg未归档）")
    if row.get("model") != scope["model"]:
        issues.append("不是当前生产模型")
    if row.get("model_version") != scope["model_version"]:
        issues.append("不是当前生产模型版本")
    kick, old_kick = _time(match.get("kickoff_utc")), _time(row.get("kickoff_utc"))
    if old_kick is None or kick is None or old_kick != kick:
        issues.append("旧赛程或未知开球时间，不能验收当前赛程")
    cutoff, created = _time(row.get("cutoff_utc")), _time(row.get("created_at_utc"))
    for label, ts in (("截止时间", cutoff), ("实际保存时间", created)):
        if ts is None:
            issues.append(f"{label}无效或缺失")
            continue
        if ts > now:
            issues.append(f"{label}晚于评价时刻")
        if (kick is not None and ts >= kick) or (old_kick is not None and ts >= old_kick):
            issues.append(f"{label}不早于开球")
        if due is not None and (ts < due or ts >= end):
            issues.append(f"{label}不在该阶段窗口内")
    if cutoff is not None and created is not None and created < cutoff:
        issues.append("实际保存时间早于截止时间")
    late = (cutoff - due).total_seconds() / 60 if cutoff is not None and due is not None else None
    created_late = (created - due).total_seconds() / 60 if created is not None and due is not None else None
    try:
        detail = json.loads(row.get("detail") or "{}")
        if not isinstance(detail, dict):
            raise ValueError("不是对象")
    except (TypeError, ValueError):
        detail = {}
        issues.append("存档详情不是有效JSON对象")
    if hours is not None:
        plan, saved_late = _number(detail.get("计划时点")), _number(detail.get("晚了几分钟"))
        if plan != hours or saved_late is None or late is None or saved_late < 0 or saved_late != round(late, 1):
            issues.append("时点元数据与真实截止时间不一致或缺失")
    if _probs([row.get(k) for k in PROB_COLS], tol) is None:
        issues.append("模型概率缺失、非有限或不是有效概率分布")
    # 当前错误行应被看见；正常的其他模型、旧赛程历史不污染新赛程完整性。
    current = (all(row.get(k) == match.get(k) for k in IDENTITY)
               and row.get("model") == scope["model"] and row.get("model_version") == scope["model_version"]
               and old_kick is not None and kick is not None and old_kick == kick)
    return {"prediction_id": int(row["prediction_id"]), **{k: row.get(k) for k in
            ("model", "model_version", "kind", "kickoff_utc", "cutoff_utc", "created_at_utc")},
            "valid": not issues, "on_time": due is not None and not issues and cutoff == due and created == due,
            "late_minutes": late, "created_late_minutes": created_late, "issues": issues,
            "current_identity": current, "rescheduled": detail.get("改期重出") is True,
            "lineup_note": detail.get("首发")}


def _slot(match, rows, hours, next_hours, now, scope, tol):
    kickoff = _time(match.get("kickoff_utc"))
    due = kickoff - pd.Timedelta(hours=hours) if kickoff is not None else None
    end = kickoff - pd.Timedelta(hours=next_hours) if kickoff is not None else None
    enabled = _time(scope["enabled_since_utc"])
    applicable = kickoff is not None and match.get("status") not in {"postponed", "cancelled", "awarded"}
    tracked = applicable and enabled is not None and due >= enabled
    archives = [_archive(row, match, hours, due, end, now, scope, tol) for row in rows]
    valid = [a for a in archives if a["valid"]]
    # 选最早真实保存的合格证据；后来错误行保留为异常，不能遮掉它。
    chosen = min(valid, key=lambda a: (_time(a["created_at_utc"]), _time(a["cutoff_utc"]), a["prediction_id"])) if valid else None
    if not applicable or enabled is None:
        state = "not_applicable"
    elif chosen:
        state = "on_time" if chosen["on_time"] else "late"
    elif not tracked:
        state = "before_activation"
    elif archives:
        state = "invalid_archive"
    elif now < due:
        state = "not_due"
    elif now >= end:
        state = "missed_window"
    elif now <= due + pd.Timedelta(minutes=scope["grace_minutes"]):
        state = "awaiting_scheduler"
    else:
        state = "overdue"
    issues = list(dict.fromkeys(issue for a in archives for issue in a["issues"]))
    label = SLOT_LABELS[state]
    if state == "not_applicable":
        label = "精确开球时间未定" if kickoff is None else ("未配置跟踪启用时间" if enabled is None else "比赛取消、延期或判罚，不作定时验收")
    return {"kind": f"T-{hours:g}", "due_at_utc": _text(due), "window_end_utc": _text(end),
            "grace_until_utc": _text(due + pd.Timedelta(minutes=scope["grace_minutes"])) if due is not None else None,
            "state": state, "label": label, "tracked": bool(tracked), "on_time": bool(chosen and chosen["on_time"] and applicable),
            **{k: chosen.get(k) if chosen and applicable else None for k in
               ("prediction_id", "cutoff_utc", "created_at_utc", "late_minutes", "created_late_minutes")},
            "valid_prediction_ids": [a["prediction_id"] for a in valid] if applicable else [],
            "other_prediction_ids": [a["prediction_id"] for a in archives if not chosen or a != chosen],
            "has_invalid_current_archive": any(a["current_identity"] and not a["valid"] for a in archives),
            "issues": issues, "archives": archives}


def _result(match, stats, now):
    known = _time(match.get("result_known_utc"))
    base = {"known_at_utc": _text(known), "home_goals": None, "away_goals": None, "source": None}
    if match.get("status") in {"postponed", "cancelled", "awarded"}:
        return {**base, "state": "not_applicable", "label": "延期、取消或判罚赛果不作常规完赛验收"}
    if known is None or now < known:
        return {**base, "state": "not_due", "label": "尚未到赛果可知时点"}
    if not stats or stats.get("home_goals") is None or stats.get("away_goals") is None:
        return {**base, "state": "awaiting_result", "label": "已到赛果可知时点，等待有效完赛比分"}
    hg, ag = _int(stats.get("home_goals")), _int(stats.get("away_goals"))
    if hg is None or ag is None or hg < 0 or ag < 0:
        return {**base, "state": "invalid", "label": "赛果不是有效的非负整数比分"}
    actual = "H" if hg > ag else "D" if hg == ag else "A"
    if stats.get("result") not in (None, actual):
        return {**base, "state": "invalid", "label": "赛果方向与比分不一致"}
    return {**base, "state": "available", "label": "当前有效完赛比分已可用", "home_goals": hg,
            "away_goals": ag, "source": stats.get("stats_source")}


def _read_scores(folder):
    path = folder / "scores.csv"
    if not path.exists():
        return [], None
    try:
        df = pd.read_csv(path)
        required = {"prediction_id", "match_id", "competition", "season", "kickoff_utc", "kind", "source",
                    "model", "model_version", "cutoff_utc", "created_at_utc", "home_goals", "away_goals", *PROB_COLS, *METRICS}
        missing = required - set(df.columns)
        if missing:
            return [], f"scores.csv缺列：{'、'.join(sorted(missing))}"
        return df.astype(object).where(pd.notna(df), None).to_dict("records"), None
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return [], f"scores.csv无法读取：{exc}"


def _metric_issues(actual, expected, tolerance, prefix):
    issues = []
    for name in METRICS:
        want = expected[name]
        got = _number(actual.get(name))
        if want is None:
            if actual.get(name) is not None:
                issues.append(f"{prefix}{name}不应有值")
        elif got is None or abs(got - want) > tolerance:
            issues.append(f"{prefix}{name}缺失、非有限或与概率及赛果不符")
    return issues


def _score_issues(score, row, source, probs, metrics, result, tolerance):
    issues = []
    for key in ("prediction_id", "match_id", "competition", "season", "kind", "model", "model_version"):
        if score.get(key) != row.get(key):
            issues.append(f"评分{key}身份不一致")
    if score.get("source") != source:
        issues.append("评分概率来源不一致")
    for key in ("kickoff_utc", "cutoff_utc", "created_at_utc"):
        if not _same_time(score.get(key), row.get(key)):
            issues.append(f"评分{key}时间不一致")
    for key in ("home_goals", "away_goals"):
        if _int(score.get(key)) != result[key]:
            issues.append("评分使用的赛果与当前完赛比分不一致")
    for key, want in zip(PROB_COLS, probs):
        got = _number(score.get(key))
        if got is None or abs(got - want) > tolerance:
            issues.append("评分概率与该prediction_id原始存档不一致")
            break
    return issues + _metric_issues(score, metrics, tolerance, "评分")


def _matrix_issues(row, cfg):
    """正式矩阵按实际存档协议验收，通用复盘解析的宽松兼容不能替代完整性。

    正式存档将矩阵保存为 (market_odds.max_goals + 1)² 个六位小数，1x2 则
    保存四位。检查允许的只是这两次舍入误差，不借 review.prob_sum_tol 放宽。
    """
    detail = json.loads(row["detail"])
    size = int(cfg["market_odds"]["max_goals"]) + 1
    half_unit = .5 * 10 ** -int(cfg["schedule"]["matrix_decimals"])
    mass_tolerance = size * size * half_unit + 1e-12
    prob_tolerance = .5e-4 + 2 * mass_tolerance
    fusion = detail.get("融合")
    fusion_expected = (detail.get("融合比分矩阵") is not None
                       or isinstance(fusion, dict) and fusion.get("状态") == "已融合")
    issues = []
    # 赔率不全/未融合时正式协议只存 1x2；模型仍必须有六项评分，不能反向
    # 要求它补造不存在的融合比分矩阵。已声明的坏融合概率也不能悄悄消失。
    if detail.get("融合概率") is not None and _probs(detail["融合概率"], cfg["review"]["prob_sum_tol"]) is None:
        issues.append("融合概率不是有效的胜平负分布")
    for name, raw_probs, required in (("模型比分矩阵", [row.get(k) for k in PROB_COLS], True),
                                      ("融合比分矩阵", detail.get("融合概率"), fusion_expected)):
        raw = detail.get(name)
        if raw is None:
            if required:
                issues.append(f"存档缺少{name}，不能完整核验六项评分")
            continue
        if (not isinstance(raw, list) or len(raw) != size
                or any(not isinstance(line, list) or len(line) != size for line in raw)):
            issues.append(f"{name}不是正式存档要求的{size}×{size}矩阵")
            continue
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
               or x < 0 or x > 1 for line in raw for x in line):
            issues.append(f"{name}含非数值、非有限值或超出[0,1]的概率")
            continue
        matrix = np.asarray(raw, dtype=float)
        total = float(matrix.sum())
        if total <= 0 or abs(total - 1) > mass_tolerance:
            issues.append(f"{name}概率总和超出正式存档舍入误差")
            continue
        # 模型和融合的 1x2 应来自各自同一矩阵，不能拿两份矛盾的分布分别评分。
        if raw_probs is not None:
            probs = _probs(raw_probs, cfg["review"]["prob_sum_tol"])
            if probs is None or not np.allclose(np.asarray(raw_probs, dtype=float), result_probs(matrix / total),
                                                atol=prob_tolerance, rtol=0):
                issues.append(f"{name}与存档胜平负概率不一致")
    return issues


def _review(match, slots, raw_by_pid, scores, score_error, folder, now, cfg):
    result = match["result"]
    base = {"score_complete": False, "review_file_valid": False, "scored_prediction_ids": [], "issues": []}
    if result["state"] != "available":
        return {**base, "state": "not_due" if result["state"] == "not_due" else "awaiting_result", "label": "等待当前有效赛果后核验评分"}
    selected = [s["prediction_id"] for s in slots if s["prediction_id"] is not None]
    if not selected:
        return {**base, "state": "no_valid_archives", "label": "没有当前生产版本的有效正式存档可评分"}
    issues = [score_error] if score_error else []
    path = folder / "matches" / f"{match['match_id']}.json"
    doc, doc_valid = None, False
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(doc, dict):
                raise ValueError("不是JSON对象")
            pairs = {"match_id": "match_id", "联赛": "competition", "season": "season", "主队": "home_team",
                     "客队": "away_team", "比赛日期": "match_date"}
            identity_ok = all(doc.get(a) == match.get(b) for a, b in pairs.items())
            generated = _time(doc.get("生成时间_utc"))
            real = doc.get("实际")
            score_ok = isinstance(real, dict) and real.get("比分") == [result["home_goals"], result["away_goals"]]
            doc_valid = (identity_ok and _same_time(doc.get("开球时间_utc"), match.get("kickoff_utc")) and score_ok
                         and generated is not None and _time(result["known_at_utc"]) <= generated <= now
                         and isinstance(doc.get("各版本"), list))
            if not doc_valid:
                issues.append("复盘JSON比赛身份、当前开球、赛果、生成时间或版本列表核验失败")
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"复盘JSON无法读取：{exc}")
    else:
        issues.append("尚无单场复盘JSON")
    confirmed = []
    for pid in selected:
        row, own = raw_by_pid[pid], []
        try:
            own += _matrix_issues(row, cfg)
            parsed = parse_version(pd.Series(row), cfg["review"]["prob_sum_tol"])
        except (TypeError, ValueError, OverflowError):
            issues.append(f"prediction_id={pid}的矩阵或融合信息无法解释")
            continue
        # 正式定时版本的模型评分必须保留比分矩阵；不能让只有 1x2 的旧 CSV
        # 把 gb_l7（或未来的正式比分模型）伪装成六项指标均已验收。
        if parsed["model_matrix"] is None:
            own.append("存档缺少模型比分矩阵，不能完整核验比分、大2.5球和双方进球")
        if parsed["matrix_bad"]:
            own.append("存档比分矩阵无效，不能完整核验该版本评分")
        entries = [e for e in doc.get("各版本", []) if isinstance(e, dict) and e.get("prediction_id") == pid] if doc_valid else []
        entry = entries[0] if len(entries) == 1 else None
        if entry is None:
            own.append("复盘JSON缺少该prediction_id唯一版本")
        elif (any(entry.get(k) != row[k] for k in ("model", "model_version")) or entry.get("版本") != row["kind"]
              or not _same_time(entry.get("cutoff_utc"), row["cutoff_utc"])
              or not _same_time(entry.get("created_at_utc"), row["created_at_utc"])):
            own.append("复盘JSON版本身份或真实时间不一致")
        for source, probs, matrix in ((SOURCE_MODEL, parsed["model_p"], parsed["model_matrix"]),
                                      (SOURCE_FUSION, parsed["fusion_p"], parsed["fusion_matrix"])):
            if probs is None:
                continue
            metrics = score_metrics(probs, matrix, result["home_goals"], result["away_goals"])
            matched = [s for s in scores if _int(s.get("prediction_id")) == pid and s.get("source") == source]
            if len(matched) != 1:
                own.append(f"{source}评分缺失或同prediction_id存在重复评分")
            else:
                own += _score_issues(matched[0], row, source, probs, metrics, result, cfg["lifecycle"]["score_tolerance"])
            detail = entry.get(source) if entry else None
            if not isinstance(detail, dict) or not isinstance(detail.get("概率"), list) or len(detail["概率"]) != len(PROB_COLS):
                own.append(f"复盘JSON缺少{source}概率与分数")
            else:
                # 单场JSON接口固定四位小数，CSV固定六位，两者不可共用容差。
                if any(_number(got) != round(float(want), 4) for got, want in zip(detail["概率"], probs)):
                    own.append(f"复盘JSON{source}概率与存档不一致")
                rounded = {k: round(v, 4) if v is not None else None for k, v in metrics.items()}
                own += _metric_issues(detail, rounded, 0, f"复盘JSON{source}")
        if not own and doc_valid and not score_error:
            confirmed.append(pid)
        issues.extend(f"prediction_id={pid}：{x}" for x in dict.fromkeys(own))
    full = bool(slots) and len(selected) == len(slots) and len(confirmed) == len(selected) and not issues
    if full:
        state, label = "complete", "三个正式时点的评分与单场复盘已核验"
    elif confirmed:
        state, label = "partial", "部分正式版本已评分，完整链路仍缺证据"
    elif issues and (path.exists() or scores or score_error):
        state, label = "invalid", "已有评分或复盘，但证据核验未通过"
    else:
        state, label = "pending", "等待赛后评分和单场复盘"
    return {"state": state, "label": label, "score_complete": full, "review_file_valid": doc_valid and not issues,
            "scored_prediction_ids": confirmed, "issues": list(dict.fromkeys(issues))}


def _closing(match, evidence, now, books, error):
    kickoff = _time(match.get("kickoff_utc"))
    due = kickoff is not None and now >= kickoff and match.get("status") not in {"postponed", "cancelled", "awarded"}
    groups = []
    for book in books:
        for market in ("1x2", "ou"):
            rows = [r for r in evidence if r.get("match_id") == match["match_id"]
                    and r.get("bookmaker") == book and r.get("market") == market
                    and (market != "ou" or r.get("line") == 2.5)]
            qualified = due and any(r.get("qualified") is True for r in rows)
            state = "error" if error and due else "not_due" if not due else "qualified" if qualified else "missing"
            label = {"error": "收盘证据读取失败", "not_due": "尚未到收盘可知时点或比赛不适用",
                     "qualified": "已有登记来源的合格收盘字段", "missing": "缺少合格收盘证据"}[state]
            groups.append({"bookmaker": book, "market": market, "line": 2.5 if market == "ou" else None,
                           "state": state, "label": label, "qualified": bool(qualified), "candidate_count": len(rows),
                           "reasons": list(dict.fromkeys(str(r.get("evidence_note") or r.get("reason")) for r in rows))})
    count = sum(g["qualified"] for g in groups)
    state = "error" if error and due else "not_due" if not due else "complete" if groups and count == len(groups) else "partial" if count else "missing"
    return {"state": state, "label": {"error": "收盘证据核验失败", "not_due": "收盘尚不可知或比赛不适用",
            "complete": "四组收盘证据齐全", "partial": "部分收盘证据仍缺失", "missing": "四组收盘证据均未齐备"}[state],
            "complete": state == "complete", "groups": groups}


def _build_report(con: sqlite3.Connection, settings: dict, now) -> dict:
    """返回可以直接保存为JSON的报告；只消费现有库和site.review_dir，不写任何产物。

    scope只含当前preview联赛组和实时赛季、site最近/未来窗口。match_id与归档有的
    比赛身份字段一起核对；round/leg只展示当前值，不冒称存档保存过这两个字段。
    """
    now = _time(now)
    report = {"generated_at_utc": _text(_time(datetime.now(timezone.utc))), "evaluation_cutoff_utc": _text(now),
              "scope": {}, "summary": {k: 0 for k in SUMMARY_KEYS}, "matches": [], "warnings": []}
    warnings = report["warnings"]
    try:
        if now is None:
            raise ValueError("评价时刻无效或缺失")
        cfg = settings["lifecycle"]
        enabled = _time(cfg["enabled_since_utc"])
        grace, tolerance = _number(cfg["grace_minutes"]), _number(cfg["score_tolerance"])
        if enabled is None or grace is None or grace < 0 or tolerance is None or tolerance <= 0:
            raise ValueError("跟踪启用时间、宽限或评分容差无效")
        horizons = sorted((float(h) for h in settings["schedule"]["horizons"]), reverse=True)
        if horizons != [24, 6, 1]:
            raise ValueError("正式时点必须完整且唯一配置T-24、T-6、T-1")
        books = settings["review"]["close_bookmakers"]
        if (not isinstance(books, (list, tuple)) or len(books) != 2
                or any(not isinstance(book, str) or not book.strip() for book in books) or set(books) != CLOSE_BOOKS):
            raise ValueError("收盘必须完整且唯一配置平博 Pinnacle、多家平均，四组证据不能缩减或替换")
        size, decimals = _int(settings["market_odds"]["max_goals"]), _int(settings["schedule"]["matrix_decimals"])
        if size is None or size < 2 or decimals is None or decimals < 4:
            raise ValueError("正式比分矩阵尺寸或存档精度配置无效")
        back, ahead = _number(settings["site"]["days_back"]), _number(settings["site"]["days_ahead"])
        if back is None or ahead is None or back < 0 or ahead < 0 or back + ahead <= 0:
            raise ValueError("网站最近/未来天数必须非负且范围非空")
        name = settings["preview"]["model"]
        scope = {"model": name, "model_version": model_version(name, model_config(name, settings)),
                 "leagues": list(expand_league_groups(settings["preview"]["leagues"], settings,
                                                       settings["preview"].get("skip_leagues"))),
                 "enabled_since_utc": _text(enabled), "grace_minutes": grace,
                  "start_utc": _text(now - pd.Timedelta(days=back)),
                  "end_utc": _text(now + pd.Timedelta(days=ahead))}
        report["scope"] = scope
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        warnings.append(f"生命周期配置不可用，不能判为零样本已完成：{exc}")
        return report
    try:
        required = {*IDENTITY, "kickoff_utc", "status", "round", "leg", "date_precision"}
        missing = required - _columns(con, "matches")
        if missing:
            raise ValueError(f"matches缺列：{'、'.join(sorted(missing))}")
        fs = FeatureStore(con, "实时", settings)
        start, end = _time(scope["start_utc"]), _time(scope["end_utc"])
        # 取数层接口按秒查询，先扩大到覆盖范围，再用真实时间收紧到报告的半开窗口。
        targets = fs.prediction_targets(_text(start.floor("s")), _text(end.ceil("s")), "显示", competition=scope["leagues"])
        targets = targets[targets["season"].map(lambda s: season_role(int(s), settings) == "live")]
        in_scope = [start <= (_time(r.get("kickoff_utc")) or _time(r["kickoff_earliest_utc"])) < end
                    for r in targets.to_dict("records")]
        targets = targets.loc[in_scope]
        if targets.empty:
            return report
        ids = [int(x) for x in targets["match_id"]]
        placeholders = ",".join("?" for _ in ids)
        states = {int(mid): state for mid, state in con.execute(
            f"select match_id,status from matches where match_id in ({placeholders})", ids)}
        targets["status"] = targets["match_id"].map(states)
        known = compute_known_times(targets, settings["features"])
        targets = targets.merge(known[["match_id", "result_known_utc"]], on="match_id", validate="one_to_one")
        stats_df = fs.historical_matches(report["evaluation_cutoff_utc"], "复盘", competition=scope["leagues"],
                                         data=("比分",), since=scope["start_utc"])
        stats = {int(r["match_id"]): _plain(r) for r in stats_df.to_dict("records") if int(r["match_id"]) in ids}
    except (sqlite3.Error, pd.errors.DatabaseError, FeatureAccessError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"当前赛程或赛果无法核验：{exc}")
        return report
    pred_cols = {*IDENTITY, "prediction_id", "kickoff_utc", "kind", "model", "model_version",
                 "cutoff_utc", "created_at_utc", "detail", *PROB_COLS}
    missing = pred_cols - _columns(con, "predictions")
    predictions = []
    if missing:
        warnings.append(f"predictions表缺失或缺列：{'、'.join(sorted(missing))}")
    else:
        predictions = pd.read_sql(f"select * from predictions where match_id in ({placeholders}) order by prediction_id", con, params=ids).to_dict("records")
    source_cols = _columns(con, "match_source_state")
    source_states = {}
    if {"match_id", "source_missing"}.issubset(source_cols):
        source_states = {int(r["match_id"]): _plain(r) for r in pd.read_sql(
            f"select * from match_source_state where match_id in ({placeholders})", con, params=ids).to_dict("records")}
    else:
        warnings.append("match_source_state缺失或缺列，来源缺失状态未知")
    source_records = []
    if {"match_id", "source", "source_key", "raw_file", "fetched_at_utc"}.issubset(_columns(con, "match_sources")):
        source_records = pd.read_sql(f"select * from match_sources where match_id in ({placeholders})", con,
                                    params=ids).to_dict("records")
    else:
        warnings.append("match_sources缺失或缺列，不能核对当前比赛来源")
    evidence, close_error = [], None
    started = [int(r["match_id"]) for r in targets.to_dict("records") if _time(r.get("kickoff_utc")) is not None
               and _time(r["kickoff_utc"]) <= now and r["status"] not in {"postponed", "cancelled", "awarded"}]
    if started:
        try:
            required_odds = {"match_id", "source", "bookmaker", "market", "timing", "line", "home", "draw", "away", "over", "under"}
            if not required_odds.issubset(_columns(con, "odds")):
                raise ValueError("odds表缺失或缺列，不能核验收盘证据")
            df = fs.closing_evidence(started, report["evaluation_cutoff_utc"], "复盘", bookmaker=books, market=["1x2", "ou"])
            evidence = _plain(exclude_conflicting_closes(df).to_dict("records"))
        except (sqlite3.Error, pd.errors.DatabaseError, FeatureAccessError, ValueError, KeyError, TypeError) as exc:
            close_error = f"收盘证据无法核验：{exc}"
            warnings.append(close_error)
    folder = resolve_path(settings["site"]["review_dir"])
    scores, score_error = _read_scores(folder)
    if score_error:
        warnings.append(score_error)
    raw_by_pid = {int(r["prediction_id"]): r for r in predictions}
    for record in targets.to_dict("records"):
        m = _plain(record)
        mid = int(m["match_id"])
        own = [r for r in predictions if int(r["match_id"]) == mid]
        source_flag = source_states.get(mid, {}).get("source_missing")
        state_valid = source_flag in (0, 1) and not isinstance(source_flag, (str, bool))
        transition_times = [_time(source_states.get(mid, {}).get(key)) for key in ("missing_since_utc", "restored_at_utc")]
        future_transition = any(t is not None and t > now for t in transition_times)
        current_sources = [r for r in source_records if r["match_id"] == mid
                           and all(isinstance(r.get(k), str) and r[k].strip() for k in ("source", "source_key", "raw_file"))
                           and _time(r.get("fetched_at_utc")) is not None and _time(r["fetched_at_utc"]) <= now]
        state = ("unknown" if future_transition else "missing" if state_valid and source_flag == 1
                 else "unknown" if mid in source_states and not state_valid
                 else "present" if current_sources else "unknown")
        m["source_missing"] = 1 if state == "missing" else 0 if state == "present" else None
        m["source_state"] = {"state": state, "label": {"missing": "来源暂缺，保留最后资料",
                             "present": "已有当前来源，未标记缺源", "unknown": "来源状态未知或当前来源不可核对"}[state],
                             "current_source_count": len(current_sources)}
        m["preview_count"] = sum(r["kind"] == "preview" for r in own)
        m["slots"] = [_slot(m, [r for r in own if r["kind"] == f"T-{h:g}"], h,
                            horizons[i + 1] if i + 1 < len(horizons) else 0, now, scope,
                            settings["review"]["prob_sum_tol"]) for i, h in enumerate(horizons)]
        formal_kinds = {s["kind"] for s in m["slots"]}
        m["other_archives"] = [_archive(r, m, None, None, None, now, scope, settings["review"]["prob_sum_tol"])
                               for r in own if r["kind"] not in formal_kinds]
        m["result"] = _result(m, stats.get(mid), now)
        m["review"] = _review(m, m["slots"], raw_by_pid, scores, score_error, folder, now, settings)
        m["closing"] = _closing(m, evidence, now, books, close_error)
        m["lineup"] = {"state": "unverified", "label": "首发链路尚未验收"}
        m["archives_complete"] = bool(m["slots"]) and all(s["state"] in {"on_time", "late"} for s in m["slots"])
        m["score_complete"] = m["review"]["score_complete"]
        m["complete_on_time"] = (m["archives_complete"] and all(s["on_time"] and not s["has_invalid_current_archive"] for s in m["slots"])
                                  and m["score_complete"] and m["closing"]["complete"] and m["source_state"]["state"] == "present")
        report["matches"].append(m)
    all_slots = [s for m in report["matches"] for s in m["slots"]]
    counts = Counter(s["state"] for s in all_slots)
    report["summary"].update(match_count=len(report["matches"]), recorded_slot_count=counts["on_time"] + counts["late"],
        late_slot_count=counts["late"], pending_slot_count=counts["not_due"] + counts["awaiting_scheduler"],
        overdue_slot_count=counts["overdue"], missed_slot_count=counts["missed_window"],
        invalid_slot_count=sum(s["state"] == "invalid_archive" or s["has_invalid_current_archive"] for s in all_slots),
        due_slot_count=sum(s["tracked"] and _time(s["due_at_utc"]) <= now for s in all_slots),
        archived_complete_count=sum(m["archives_complete"] for m in report["matches"]),
        scored_complete_count=sum(m["score_complete"] for m in report["matches"]),
        complete_on_time_count=sum(m["complete_on_time"] for m in report["matches"]),
        source_missing_count=sum(m["source_missing"] == 1 for m in report["matches"]),
        closing_complete_count=sum(m["closing"]["complete"] for m in report["matches"]))
    return _plain(report)


def build_report(con: sqlite3.Connection, settings: dict, now) -> dict:
    """只读验收，并拒绝把并发更新期间混合读取的证据作为同一份完整结果。

    FeatureStore 的 TEMP 表初始化会提交连接，不能假定入口事务覆盖全程。
    data_version 仅监测其他连接对主库的提交；本模块不写主库，TEMP 表变化
    不会造成误报。发现主库改变就撤回逐场结果，留明确警告供下轮安全重试。
    """
    try:
        before = con.execute("pragma main.data_version").fetchone()[0]
    except sqlite3.Error:
        before = None
    report = _build_report(con, settings, now)
    try:
        after = con.execute("pragma main.data_version").fetchone()[0]
    except sqlite3.Error:
        after = None
    if before is None or after is None or before != after:
        report["warnings"].append("数据库在验收期间发生变化或版本无法核对，已撤回本次逐场结果，请重试")
        report["matches"] = []
        report["summary"] = {key: 0 for key in SUMMARY_KEYS}
    return report
