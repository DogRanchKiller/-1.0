"""离线 T-1 等待决策合同；不采集、预测、读写数据库或接入现有定时任务。

所有时间必须是显式 UTC。``cutoff_utc`` 是本次决定所能看到的截止时刻，
首发只接受在该时刻之前实际首次看到、来源及比赛身份可核对的确认名单。
T-30 是停止等待的期限；如果轮询迟到但决定前已有完整名单，仍使用名单。
这个规则仅属隔离合同，正式接入前仍需确定来源和迟到统计口径。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal


State = Literal["not_due", "wait_for_lineup", "release_with_verified_lineup",
                "timeout_release_without_lineup", "skip"]


@dataclass(frozen=True)
class Fixture:
    match_id: int
    competition: str
    season: int
    home_team: str
    away_team: str
    kickoff_utc: datetime | None
    status: str


@dataclass(frozen=True)
class ArchivedPrediction:
    fixture: Fixture
    kind: str


@dataclass(frozen=True)
class LineupSnapshot:
    """两队的先发保存稳定的来源球员 ID，不用同名球员的显示名判断重复。"""

    match_id: int
    competition: str
    season: int
    home_team: str
    away_team: str
    kickoff_utc: datetime
    source: str
    evidence_key: str
    first_seen_utc: datetime
    confirmed: bool
    home_starters: tuple[str, ...]
    away_starters: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    state: State
    reason: str
    evidence_key: str | None = None
    rescheduled_from_utc: datetime | None = None


def _is_utc(value: object) -> bool:
    return (isinstance(value, datetime) and value.tzinfo is not None
            and value.utcoffset() == timedelta(0))


def _identity(obj: Fixture | LineupSnapshot) -> tuple:
    return (obj.match_id, obj.competition, obj.season, obj.home_team,
            obj.away_team, obj.kickoff_utc)


def _valid_fixture_identity(fixture: Fixture) -> bool:
    names = (fixture.competition, fixture.home_team, fixture.away_team)
    return (type(fixture.match_id) is int and fixture.match_id > 0
            and type(fixture.season) is int and fixture.season > 0
            and all(isinstance(value, str) and bool(value.strip()) and value == value.strip()
                    for value in names)
            and fixture.home_team != fixture.away_team)


def _complete_lineup(snapshot: LineupSnapshot | None, fixture: Fixture, cutoff_utc: datetime) -> bool:
    if (snapshot is None or not _is_utc(snapshot.kickoff_utc)
            or _identity(snapshot) != _identity(fixture)):
        return False
    if (snapshot.confirmed is not True
            or not isinstance(snapshot.source, str) or not snapshot.source.strip()
            or not isinstance(snapshot.evidence_key, str) or not snapshot.evidence_key.strip()
            or not _is_utc(snapshot.first_seen_utc) or snapshot.first_seen_utc > cutoff_utc):
        return False
    if not isinstance(snapshot.home_starters, tuple) or not isinstance(snapshot.away_starters, tuple):
        return False
    players = (*snapshot.home_starters, *snapshot.away_starters)
    return (len(snapshot.home_starters) == len(snapshot.away_starters) == 11
            and all(isinstance(p, str) and bool(p) and p.strip() == p for p in players)
            and len(set(players)) == 22)


def decide_t1(fixture: Fixture, cutoff_utc: datetime, *, last_archive: ArchivedPrediction | None = None,
              lineup: LineupSnapshot | None = None, wait_start_minutes: int = 60,
              timeout_minutes: int = 30) -> Decision:
    """对一场比赛作一次只读决定；调用方负责取证、持久化和发布前复查。"""
    if not _is_utc(cutoff_utc):
        raise ValueError("cutoff_utc 必须是带时区的 UTC 时间")
    if (type(wait_start_minutes) is not int or type(timeout_minutes) is not int
            or not 0 < timeout_minutes < wait_start_minutes):
        raise ValueError("T-1 等待起点必须大于正数的超时分钟数")
    if fixture.kickoff_utc is None:
        return Decision("skip", "unknown_kickoff")
    if not _is_utc(fixture.kickoff_utc):
        raise ValueError("kickoff_utc 必须是带时区的 UTC 时间")
    if not _valid_fixture_identity(fixture):
        return Decision("skip", "invalid_fixture_identity")
    if fixture.status != "scheduled":
        return Decision("skip", "fixture_not_scheduled")
    if cutoff_utc >= fixture.kickoff_utc:
        return Decision("skip", "kickoff_reached")

    rescheduled_from = None
    if last_archive is not None:
        if last_archive.kind != "T-1":
            return Decision("skip", "archive_kind_mismatch")
        if (not _is_utc(last_archive.fixture.kickoff_utc)
                or _identity(last_archive.fixture)[:-1] != _identity(fixture)[:-1]):
            return Decision("skip", "archive_identity_mismatch")
        if last_archive.fixture.kickoff_utc == fixture.kickoff_utc:
            return Decision("skip", "already_archived")
        rescheduled_from = last_archive.fixture.kickoff_utc

    if cutoff_utc < fixture.kickoff_utc - timedelta(minutes=wait_start_minutes):
        return Decision("not_due", "before_wait_start", rescheduled_from_utc=rescheduled_from)
    if _complete_lineup(lineup, fixture, cutoff_utc):
        return Decision("release_with_verified_lineup", "confirmed_lineup_before_cutoff",
                        evidence_key=lineup.evidence_key, rescheduled_from_utc=rescheduled_from)
    if cutoff_utc >= fixture.kickoff_utc - timedelta(minutes=timeout_minutes):
        return Decision("timeout_release_without_lineup", "lineup_unavailable_at_deadline",
                        rescheduled_from_utc=rescheduled_from)
    return Decision("wait_for_lineup", "awaiting_confirmed_lineup", rescheduled_from_utc=rescheduled_from)
