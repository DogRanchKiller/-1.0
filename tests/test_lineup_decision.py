"""T-1 首发等待合同：只交给纯函数假赛程、假快照和假存档。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.predict.lineup_decision import ArchivedPrediction, Fixture, LineupSnapshot, decide_t1


KICKOFF = datetime(2026, 11, 4, 20, 0, tzinfo=timezone.utc)
STARTERS_HOME = tuple(f"home-{n}" for n in range(1, 12))
STARTERS_AWAY = tuple(f"away-{n}" for n in range(1, 12))


def fixture(**changes):
    values = dict(match_id=42, competition="E0", season=2026, home_team="Home",
                  away_team="Away", kickoff_utc=KICKOFF, status="scheduled")
    values.update(changes)
    return Fixture(**values)


def lineup(**changes):
    values = dict(match_id=42, competition="E0", season=2026, home_team="Home",
                  away_team="Away", kickoff_utc=KICKOFF, source="provider",
                  evidence_key="raw/snapshot-1", first_seen_utc=KICKOFF - timedelta(minutes=55),
                  confirmed=True, home_starters=STARTERS_HOME, away_starters=STARTERS_AWAY)
    values.update(changes)
    return LineupSnapshot(**values)


def archive(**changes):
    kind = changes.pop("kind", "T-1")
    return ArchivedPrediction(fixture=fixture(**changes), kind=kind)


def at(minutes_before, **kwargs):
    return decide_t1(fixture(), KICKOFF - timedelta(minutes=minutes_before), **kwargs)


@pytest.mark.parametrize("minutes,expected", [
    (60.000001, "not_due"), (60, "wait_for_lineup"),
    (30.000001, "wait_for_lineup"), (30, "timeout_release_without_lineup"),
    (17, "timeout_release_without_lineup"),
])
def test_wait_and_timeout_boundaries(minutes, expected):
    assert at(minutes).state == expected


def test_configurable_boundary_and_invalid_config():
    assert at(51, wait_start_minutes=50, timeout_minutes=20).state == "not_due"
    assert at(50, wait_start_minutes=50, timeout_minutes=20).state == "wait_for_lineup"
    assert at(20, wait_start_minutes=50, timeout_minutes=20).state == "timeout_release_without_lineup"
    for start, timeout in [(30, 30), (20, 30), (60, 0), (True, 30)]:
        with pytest.raises(ValueError):
            at(45, wait_start_minutes=start, timeout_minutes=timeout)


def test_complete_confirmed_lineup_releases_at_first_seen_boundary():
    evidence = lineup(first_seen_utc=KICKOFF - timedelta(minutes=60))
    result = at(60, lineup=evidence)
    assert result.state == "release_with_declared_lineup"
    assert result.evidence_key == "raw/snapshot-1"
    assert result.rescheduled_from_utc is None


@pytest.mark.parametrize("changes", [
    {"confirmed": False},
    {"home_starters": STARTERS_HOME[:-1]},
    {"away_starters": STARTERS_AWAY[:-1]},
    {"home_starters": STARTERS_HOME[:-1] + (STARTERS_HOME[0],)},
    {"away_starters": STARTERS_AWAY[:-1] + ("",)},
    {"away_starters": STARTERS_AWAY[:-1] + (STARTERS_HOME[0],)},
    {"source": ""}, {"evidence_key": ""},
    {"first_seen_utc": KICKOFF - timedelta(minutes=29)},
    {"first_seen_utc": None},
    {"source": None}, {"evidence_key": None},
    {"home_starters": None},
    {"kickoff_utc": KICKOFF.astimezone(timezone(timedelta(hours=8)))},
    {"match_id": 43}, {"competition": "SP1"}, {"home_team": "Other"},
    {"kickoff_utc": KICKOFF + timedelta(hours=2)},
])
def test_unverified_lineup_cannot_trigger_early_release(changes):
    assert at(45, lineup=lineup(**changes)).state == "wait_for_lineup"
    assert at(30, lineup=lineup(**changes)).state == "timeout_release_without_lineup"


def test_lineup_available_after_deadline_is_used_at_next_poll():
    # T-30 是不再等待的期限；轮询延迟时仍只使用本次截止前确实看到的证据。
    evidence = lineup(first_seen_utc=KICKOFF - timedelta(minutes=22))
    assert at(17, lineup=evidence).state == "release_with_declared_lineup"
    assert at(23, lineup=evidence).state == "timeout_release_without_lineup"


def test_lineup_first_seen_after_previous_poll_can_release_later():
    evidence = lineup(first_seen_utc=KICKOFF - timedelta(minutes=44))
    assert at(45, lineup=evidence).state == "wait_for_lineup"
    assert at(30, lineup=evidence).state == "release_with_declared_lineup"


@pytest.mark.parametrize("status", ["cancelled", "postponed", "awarded", "played"])
def test_unavailable_fixture_is_skipped(status):
    assert decide_t1(fixture(status=status), KICKOFF - timedelta(minutes=30), lineup=lineup()).state == "skip"


def test_unknown_or_past_kickoff_never_releases():
    assert decide_t1(fixture(kickoff_utc=None), KICKOFF - timedelta(minutes=20)).state == "skip"
    assert decide_t1(fixture(), KICKOFF, lineup=lineup()).state == "skip"
    assert decide_t1(fixture(), KICKOFF + timedelta(minutes=2), lineup=lineup()).state == "skip"


def test_archive_deduplicates_current_kickoff_and_reschedule_may_release():
    old = archive(kickoff_utc=KICKOFF - timedelta(days=1))
    assert at(45, last_t1_archive=archive(), lineup=lineup()).state == "skip"
    decision = at(45, last_t1_archive=old, lineup=lineup())
    assert decision.state == "release_with_declared_lineup"
    assert decision.rescheduled_from_utc == old.fixture.kickoff_utc
    # 旧场次的首发证据不能用于改期后的同一 match_id。
    assert at(45, last_t1_archive=old, lineup=lineup(kickoff_utc=old.fixture.kickoff_utc)).state == "wait_for_lineup"


def test_conflicting_archive_identity_fails_closed():
    assert at(45, last_t1_archive=archive(home_team="Other"), lineup=lineup()).state == "skip"


@pytest.mark.parametrize("target_changes,evidence_changes", [
    ({}, {"match_id": 42.0}), ({}, {"season": 2026.0}),
    ({"match_id": 1}, {"match_id": True}), ({"season": 1}, {"season": True}),
])
def test_lineup_identity_requires_exact_integer_types(target_changes, evidence_changes):
    decision = decide_t1(fixture(**target_changes), KICKOFF - timedelta(minutes=45),
                         lineup=lineup(**{**target_changes, **evidence_changes}))
    assert decision.state == "wait_for_lineup"


@pytest.mark.parametrize("target_changes,archive_changes", [
    ({}, {"match_id": 42.0}), ({}, {"season": 2026.0}),
    ({"match_id": 1}, {"match_id": True}), ({"season": 1}, {"season": True}),
])
def test_archive_identity_requires_exact_integer_types(target_changes, archive_changes):
    decision = decide_t1(fixture(**target_changes), KICKOFF - timedelta(minutes=45),
                         last_t1_archive=archive(**{**target_changes, **archive_changes}),
                         lineup=lineup(**target_changes))
    assert decision.state == "skip" and decision.reason == "archive_identity_mismatch"


@pytest.mark.parametrize("bad_fixture,bad_lineup", [
    ({"match_id": 0}, {"match_id": 0}),
    ({"home_team": ""}, {"home_team": ""}),
    ({"away_team": "Home"}, {"away_team": "Home"}),
    ({"competition": " "}, {"competition": " "}),
])
def test_malformed_fixture_identity_fails_closed(bad_fixture, bad_lineup):
    decision = decide_t1(fixture(**bad_fixture), KICKOFF - timedelta(minutes=45),
                         lineup=lineup(**bad_lineup))
    assert decision.state == "skip" and decision.reason == "invalid_fixture_identity"


def test_other_archive_kind_cannot_be_treated_as_t1():
    decision = at(45, last_t1_archive=archive(kind="T-6"), lineup=lineup())
    assert decision.state == "skip" and decision.reason == "archive_kind_mismatch"


@pytest.mark.parametrize("bad", [datetime(2026, 11, 4, 19, 0),
                                 datetime(2026, 11, 4, 20, 0, tzinfo=timezone(timedelta(hours=8)))])
def test_decision_requires_explicit_utc(bad):
    with pytest.raises(ValueError, match="UTC"):
        decide_t1(fixture(), bad)
    with pytest.raises(ValueError, match="UTC"):
        decide_t1(fixture(kickoff_utc=bad), KICKOFF - timedelta(minutes=30))


def test_lineup_with_non_utc_first_seen_cannot_pass():
    bad = lineup(first_seen_utc=datetime(2026, 11, 4, 19, 0))
    assert at(45, lineup=bad).state == "wait_for_lineup"


@pytest.mark.parametrize("bad", [object(), fixture(), None])
def test_malformed_lineup_object_is_ignored(bad):
    assert at(45, lineup=bad).state == "wait_for_lineup"
    assert at(30, lineup=bad).state == "timeout_release_without_lineup"


@pytest.mark.parametrize("bad", [object(), ArchivedPrediction(fixture=None, kind="T-1"),
                                 ArchivedPrediction(fixture=object(), kind="T-1"),
                                 ArchivedPrediction(fixture=lineup(), kind="T-1")])
def test_malformed_archive_object_fails_closed(bad):
    decision = at(45, last_t1_archive=bad, lineup=lineup())
    assert decision.state == "skip" and decision.reason == "archive_identity_mismatch"
