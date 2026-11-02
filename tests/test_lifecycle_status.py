"""验收状态不能把空样本、尚未到点或坏证据显示成全周期通过。"""
from src.jobs import status_page as sp


def report(**changes):
    summary = {
        "match_count": 5, "recorded_slot_count": 5, "late_slot_count": 5,
        "pending_slot_count": 10, "overdue_slot_count": 0, "invalid_slot_count": 0,
        "archived_complete_count": 0, "scored_complete_count": 0, "complete_on_time_count": 0,
    }
    summary.update(changes)
    return {"summary": summary, "warnings": [], "matches": []}


def test_future_stages_and_late_startup_do_not_pass_complete_acceptance():
    check = sp.lifecycle_check(report())
    assert check.level == "unknown"
    assert "全周期" in check.text and "未完成" in check.text
    assert "迟到 5" in check.text and "待完成 10" in check.text


def test_no_sample_means_unknown_not_zero_percent():
    check = sp.lifecycle_check(report(match_count=0, recorded_slot_count=0,
                                      late_slot_count=0, pending_slot_count=0))
    assert check.level == "unknown"
    assert "没有可验收比赛" in check.text and "%" not in check.text


def test_overdue_and_invalid_evidence_warn_independently_of_job_success():
    for key in ("overdue_slot_count", "invalid_slot_count"):
        check = sp.lifecycle_check(report(**{key: 1}))
        assert check.level == "warn"
        assert check.key == "prediction_lifecycle"


def test_readable_but_incomplete_or_broken_report_cannot_be_green():
    broken = report(scored_complete_count=5, complete_on_time_count=5)
    broken["warnings"] = ["读取比赛证据失败"]
    check = sp.lifecycle_check(broken)
    assert check.level == "warn" and "读取比赛证据失败" in check.text


def test_completed_archives_and_scores_still_do_not_claim_accuracy_or_lineups():
    check = sp.lifecycle_check(report(recorded_slot_count=15, late_slot_count=0,
                                      pending_slot_count=0, archived_complete_count=5,
                                      scored_complete_count=5, complete_on_time_count=5))
    assert check.level == "ok"
    assert "已评分 5" in check.text
    assert "准确率" not in check.text and "首发已" not in check.text
