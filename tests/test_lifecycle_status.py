"""验收状态不能把空样本、尚未到点或坏证据显示成全周期通过。"""
import pytest

from src.jobs import status_page as sp


def report(**changes):
    summary = {
        "match_count": 5, "recorded_slot_count": 5, "late_slot_count": 5,
        "pending_slot_count": 10, "overdue_slot_count": 0, "invalid_slot_count": 0,
        "archived_complete_count": 0, "scored_complete_count": 0, "complete_on_time_count": 0,
        "closing_complete_count": 0,
    }
    summary.update(changes)
    return {"summary": summary, "warnings": [], "matches": [{"match_id": n} for n in range(summary["match_count"])]}


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
                                      scored_complete_count=5, closing_complete_count=5, complete_on_time_count=5))
    assert check.level == "ok"
    assert "已评分 5" in check.text
    assert "准确率" not in check.text and "首发已" not in check.text


def test_one_completed_match_does_not_make_the_other_pending_matches_pass():
    check = sp.lifecycle_check(report(recorded_slot_count=6, late_slot_count=3, pending_slot_count=9,
                                      archived_complete_count=1, scored_complete_count=1,
                                      closing_complete_count=1, complete_on_time_count=1))
    assert check.level == "unknown"


def test_missed_window_and_source_loss_need_attention():
    for key in ("missed_slot_count", "source_missing_count"):
        assert sp.lifecycle_check(report(**{key: 1})).level == "warn"


@pytest.mark.parametrize("payload", [None, [], "bad", {"summary": []}, {"summary": "bad"}])
def test_bad_report_shape_is_unavailable_not_an_exception(payload):
    assert sp.lifecycle_check(payload).level == "unknown"


@pytest.mark.parametrize("key", ["missed_slot_count", "source_missing_count", "closing_complete_count", "due_slot_count"])
@pytest.mark.parametrize("value", ["1", -1, True, None, []])
def test_optional_count_when_present_must_be_a_nonnegative_integer(key, value):
    assert sp.lifecycle_check(report(**{key: value})).level == "unknown"


@pytest.mark.parametrize("warnings", ["读取失败", {"error": "失败"}, [None], [1], False])
def test_bad_warning_shape_is_unavailable_not_green(warnings):
    payload = report()
    payload["warnings"] = warnings
    assert sp.lifecycle_check(payload).level == "unknown"


def test_archived_late_match_with_no_pending_slots_is_not_an_inconsistent_count():
    payload = report(match_count=1, recorded_slot_count=3, late_slot_count=3,
                     pending_slot_count=0, archived_complete_count=1, scored_complete_count=1)
    check = sp.lifecycle_check(payload)
    assert check.level == "unknown" and "计数关系异常" not in check.text
    assert "已评分 1" in check.text and "未完成" in check.text


@pytest.mark.parametrize("changes", [
    {"complete_on_time_count": 6}, {"archived_complete_count": 6},
    {"scored_complete_count": 1}, {"late_slot_count": 6},
    {"recorded_slot_count": 16}, {"pending_slot_count": 16},
    {"recorded_slot_count": 10, "pending_slot_count": 10},
])
def test_impossible_count_relationships_cannot_pass(changes):
    assert sp.lifecycle_check(report(**changes)).level == "warn"


def complete_report():
    return report(recorded_slot_count=15, late_slot_count=0, pending_slot_count=0,
                  archived_complete_count=5, scored_complete_count=5, closing_complete_count=5,
                  complete_on_time_count=5)


def test_missing_closing_summary_cannot_support_complete_acceptance():
    payload = complete_report()
    del payload["summary"]["closing_complete_count"]
    assert sp.lifecycle_check(payload).level == "unknown"


def test_all_matches_on_time_conflicts_with_a_late_archive():
    payload = complete_report()
    payload["summary"]["late_slot_count"] = 1
    assert sp.lifecycle_check(payload).level == "warn"


@pytest.mark.parametrize("matches", [[], None, "bad", [None] * 5])
def test_missing_match_evidence_cannot_support_summary(matches):
    payload = complete_report()
    payload["matches"] = matches
    assert sp.lifecycle_check(payload).level == "unknown"
