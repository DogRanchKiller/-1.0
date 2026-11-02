"""网站的报告消费边界：只检查结构和已有结论的一致性，不重新验收原始证据。"""
from __future__ import annotations

import math


KINDS = ("T-24", "T-6", "T-1")
LINEUP_LABEL = "首发链路尚未验收"
STATES = {
    "slots": {"not_due", "awaiting_scheduler", "overdue", "missed_window", "before_activation",
              "not_applicable", "on_time", "late", "invalid_archive"},
    "result": {"not_applicable", "not_due", "awaiting_result", "invalid", "available"},
    "review": {"not_due", "awaiting_result", "no_valid_archives", "complete", "partial", "invalid", "pending"},
    "closing": {"error", "not_due", "complete", "partial", "missing"},
    "group": {"error", "not_due", "qualified", "missing"},
    "source": {"present", "not_flagged", "unknown", "missing"},
}
KNOWN_STATES = set().union(*STATES.values()) | {"unverified"}
MATCH_COUNTS = ("match_count", "archived_complete_count", "scored_complete_count", "closing_complete_count",
                "complete_on_time_count", "source_missing_count")
SLOT_COUNTS = ("recorded_slot_count", "late_slot_count", "pending_slot_count", "overdue_slot_count",
               "missed_slot_count", "invalid_slot_count", "due_slot_count")


def _count(value) -> bool:
    return type(value) is int and value >= 0


def _list_of(value, cls) -> bool:
    return isinstance(value, list) and all(isinstance(item, cls) for item in value)


def _state(value, role) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("state"), str) and value["state"] in STATES[role]
            and isinstance(value.get("label"), str) and bool(value["label"]))


def report_problem(report) -> str | None:
    """返回不能展示为有效报告的原因；不修改报告，因此下载仍保留同一份原始诊断。"""
    if not isinstance(report, dict) or not report:
        return "报告缺失或不是对象"
    if not _list_of(report.get("matches"), dict):
        return "比赛清单格式不完整"
    if not isinstance(report.get("scope"), dict) or not isinstance(report.get("summary"), dict):
        return "范围或汇总格式不完整"
    if not _list_of(report.get("warnings", []), str):
        return "报告提醒格式不完整"
    scope, summary, matches = report["scope"], report["summary"], report["matches"]
    if not _list_of(scope.get("leagues", []), str):
        return "赛事范围格式不完整"
    grace = scope.get("grace_minutes")
    if grace is not None and (type(grace) not in (int, float) or not math.isfinite(grace) or grace < 0):
        return "调度宽限格式不完整"
    keys = MATCH_COUNTS + SLOT_COUNTS
    if any(not _count(summary[key]) for key in keys if key in summary):
        return "汇总计数必须是非负整数"
    if summary.get("match_count") != len(matches) or (matches and any(key not in summary for key in keys)):
        return "汇总分母与比赛清单不一致或字段缺失"
    if any(summary.get(key, 0) > len(matches) for key in MATCH_COUNTS):
        return "场次汇总超过本次比赛分母"
    if any(summary.get(key, 0) > len(matches) * len(KINDS) for key in SLOT_COUNTS):
        return "时点汇总超过本次时点总数"
    if not (summary.get("complete_on_time_count", 0) <= summary.get("scored_complete_count", 0)
            <= summary.get("archived_complete_count", 0) <= len(matches)):
        return "完成、评分与存档汇总不一致"
    if (summary.get("complete_on_time_count", 0) > summary.get("closing_complete_count", 0)
            or summary.get("late_slot_count", 0) > summary.get("recorded_slot_count", 0)
            or summary.get("archived_complete_count", 0) * len(KINDS) > summary.get("recorded_slot_count", 0)
            or sum(summary.get(k, 0) for k in ("recorded_slot_count", "pending_slot_count", "overdue_slot_count",
                                               "missed_slot_count")) > len(matches) * len(KINDS)):
        return "存档、等待或完成汇总互相冲突"
    ids = set()
    for row in matches:
        mid = row.get("match_id")
        if not _count(mid) or mid == 0 or mid in ids:
            return "比赛编号缺失、无效或重复"
        ids.add(mid)
        if any(not isinstance(row.get(key), str) for key in ("home_team", "away_team", "competition")):
            return "比赛身份格式不完整"
        if "source_state" in row and not _state(row["source_state"], "source"):
            return "当前来源状态格式不完整"
        if (not _list_of(row.get("other_archives", []), dict)
                or any(not _list_of(archive.get("issues", []), str) for archive in row.get("other_archives", []))):
            return "其他存档证据格式不完整"
        slots = row.get("slots")
        if (not _list_of(slots, dict) or len(slots) != len(KINDS)
                or sorted(str(slot.get("kind")) for slot in slots) != sorted(KINDS)):
            return "正式时点缺失、重复或未知"
        for slot in slots:
            if not _state(slot, "slots") or not _list_of(slot.get("issues", []), str):
                return "正式时点状态或原因格式不完整"
            if slot["state"] in {"on_time", "late"}:
                if (not _count(slot.get("prediction_id")) or not slot["prediction_id"]
                        or not slot.get("cutoff_utc") or not slot.get("created_at_utc")
                        or (slot.get("on_time") is not (slot["state"] == "on_time"))):
                    return "正式时点完成状态缺少对应存档字段"
            if not _list_of(slot.get("archives", []), dict):
                return "存档证据清单格式不完整"
            if any(not _list_of(archive.get("issues", []), str) for archive in slot.get("archives", [])):
                return "存档证据原因格式不完整"
        if any(not _state(row.get(role), role) for role in ("result", "review", "closing")):
            return "赛果、评分或收盘状态格式不完整"
        review, closing = row["review"], row["closing"]
        if (not _list_of(review.get("scored_prediction_ids"), int)
                or any(type(pid) is not int or pid <= 0 for pid in review["scored_prediction_ids"])
                or not _list_of(review.get("issues", []), str)):
            return "评分证据清单格式不完整"
        if (type(review.get("score_complete")) is not bool
                or (review["state"] == "complete") != review["score_complete"]
                or (review["score_complete"] and review.get("review_file_valid") is not True)):
            return "评分完成状态与核验字段不一致"
        groups = closing.get("groups")
        if not _list_of(groups, dict) or any(not _state(group, "group") for group in groups):
            return "收盘分组状态格式不完整"
        for group in groups:
            if (not _count(group.get("candidate_count")) or not _list_of(group.get("reasons", []), str)
                    or type(group.get("qualified")) is not bool
                    or (group["state"] == "qualified") != group["qualified"]
                    or not isinstance(group.get("bookmaker"), str) or not isinstance(group.get("market"), str)):
                return "收盘分组证据格式不完整"
        if (type(closing.get("complete")) is not bool or (closing["state"] == "complete") != closing["complete"]
                or (closing["complete"] and (not groups or not all(g["qualified"] for g in groups)))):
            return "收盘完成状态与分组证据不一致"
        lineup = row.get("lineup")
        if not isinstance(lineup, dict) or lineup.get("state") != "unverified" or lineup.get("label") != LINEUP_LABEL:
            return "首发链路尚未验收，不能显示完成结论"
        if any(type(row.get(key)) is not bool for key in ("archives_complete", "score_complete", "complete_on_time")):
            return "逐场完成字段格式不完整"
        if (row["score_complete"] != review["score_complete"]
                or (row["complete_on_time"] and not (row["archives_complete"] and row["score_complete"]
                    and closing["complete"] and all(slot.get("on_time") is True for slot in slots)))):
            return "逐场完成结论互相冲突"
    for key, flag in (("archived_complete_count", "archives_complete"), ("scored_complete_count", "score_complete"),
                      ("complete_on_time_count", "complete_on_time")):
        if summary.get(key, 0) != sum(row[flag] for row in matches):
            return "完成汇总与逐场结论不一致"
    if summary.get("closing_complete_count", 0) != sum(row["closing"]["complete"] for row in matches):
        return "收盘汇总与逐场结论不一致"
    return None
