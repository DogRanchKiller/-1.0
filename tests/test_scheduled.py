"""定时预测核心功能测试（需求文档第 9 节）。

覆盖：plan_matches、fold_matrix、matrix_list、RunLock。
全部用假数据，不碰真实数据库（tmp_path 隔离锁文件）。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.predict.scheduled import (
    RunLock,
    fold_matrix,
    kind_of,
    matrix_list,
    plan_matches,
)


# ---- 辅助函数 ----

def _now() -> datetime:
    """固定的测试用"现在"（UTC，不带毫秒）。"""
    return datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ko_text(dt: datetime) -> str:
    """datetime → 数据库里用的 UTC 文字。"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _targets(*kickoffs) -> pd.DataFrame:
    """给定开球时间字符串列表，构造最小版 targets DataFrame（match_id 从 1 起）。"""
    rows = [{"match_id": i + 1, "kickoff_utc": ko} for i, ko in enumerate(kickoffs)]
    return pd.DataFrame(rows)


# ---- plan_matches ----

def test_plan_matches_basic():
    """3 场比赛：一场已过 T-24、一场还没到任何时点、一场已开球。
    期望：1 个 Plan（T-24），counts 里 已开球=1、还没到点=1。
    """
    now = _now()
    # match 1：开球 12 小时后 → ko-24h=now-12h 已过，ko-6h=now+6h 还没到
    ko1 = _ko_text(now + timedelta(hours=12))
    # match 2：开球 30 小时后 → ko-24h=now+6h 还没到
    ko2 = _ko_text(now + timedelta(hours=30))
    # match 3：1 小时前已开球
    ko3 = _ko_text(now - timedelta(hours=1))

    targets = _targets(ko1, ko2, ko3)
    plans, counts = plan_matches(targets, {}, [24, 6, 1], now)

    assert len(plans) == 1
    assert plans[0].match_id == 1
    assert plans[0].hours == 24
    assert counts["已开球"] == 1
    assert counts["还没到点"] == 1


def test_plan_matches_skip_earlier():
    """一场比赛同时过了 T-24 和 T-6：只出 T-6，skipped 里有 24。"""
    now = _now()
    # 开球 3 小时后 → ko-24h=now-21h ≤ now，ko-6h=now-3h ≤ now，ko-1h=now+2h > now
    ko = _ko_text(now + timedelta(hours=3))
    targets = _targets(ko)

    plans, counts = plan_matches(targets, {}, [24, 6, 1], now)

    assert len(plans) == 1, f"应出 1 个计划，实际：{plans}"
    assert plans[0].hours == 6, "同时过了 T-24 和 T-6，只出最晚的 T-6"
    assert 24 in plans[0].skipped, f"T-24 应在 skipped 里，实际 skipped={plans[0].skipped}"


def test_plan_matches_already_stored():
    """同一场、同一版已按当前开球时间存过 → counts['已出过'] += 1，不出新 Plan。"""
    now = _now()
    ko_dt = now + timedelta(hours=12)
    ko = _ko_text(ko_dt)
    targets = _targets(ko)
    mid = 1
    # 模拟"T-24 已按这个开球时间存过"
    stored = {(mid, kind_of(24)): ko}

    plans, counts = plan_matches(targets, stored, [24, 6, 1], now)

    assert len(plans) == 0
    assert counts["已出过"] == 1


# ---- fold_matrix ----

def test_fold_matrix_shape():
    """fold_matrix(3×3 矩阵, n=10) 的结果是 (11, 11)。"""
    m = np.eye(3) / 3
    out = fold_matrix(m, 10)
    assert out.shape == (11, 11), f"期望 (11,11)，实际 {out.shape}"


# ---- matrix_list ----

def test_matrix_list_sum():
    """matrix_list(np.eye(3)/3, 10, 6)：每行和 ≤ 1，全部和 ≈ 1。"""
    m = np.eye(3) / 3
    result = matrix_list(m, 10, 6)

    assert result is not None
    assert len(result) == 11, "应有 11 行"
    assert len(result[0]) == 11, "每行应有 11 列"

    row_sums = [sum(row) for row in result]
    assert all(s <= 1 + 1e-9 for s in row_sums), f"某行和超过 1：{row_sums}"
    total = sum(row_sums)
    assert abs(total - 1.0) < 1e-4, f"全部和偏离 1 太多：{total}"


# ---- RunLock ----

def test_runlock_acquired(tmp_path):
    """锁的三个场景：第一个拿到，第二个拿同路径失败，第一个 release 后第三个能拿到。"""
    lock_path = tmp_path / "run.lock"
    stale = 30.0

    # 场景 1：第一个锁拿到
    lock1 = RunLock(lock_path, stale)
    assert lock1.acquire() is True
    assert lock1.acquired is True
    assert lock_path.exists(), "锁文件应该已创建"

    # 场景 2：第二个锁拿不到（文件已存在，没过期）
    lock2 = RunLock(lock_path, stale)
    assert lock2.acquire() is False
    assert lock2.acquired is False

    # 场景 3：第一个释放后，第三个能拿到
    lock1.release()
    assert not lock_path.exists(), "release 后锁文件应删掉"

    lock3 = RunLock(lock_path, stale)
    assert lock3.acquire() is True
    assert lock3.acquired is True
    lock3.release()


# ---- 定时任务入口 src/jobs/scheduled_predict.py（心跳、锁）----

class FakeModel:
    """假模型：每场都给 50% / 30% / 20%（不用真的拟合，测试快）。"""
    name = "fake"

    def fit(self, history, cutoff):
        pass

    def predict(self, targets, cutoff):
        return [[0.5, 0.3, 0.2] for _ in range(len(targets))]


@pytest.fixture
def job_env(tmp_path):
    """临时数据库 + 临时心跳文件夹 + 临时锁文件，不碰真实的 data/。

    现在 = 2026-11-01 12:00 UTC。比赛 10 在 12 小时后开球（T-24 已到点、T-6 还没到）；比赛 1 已经踢完（历史）。
    """
    import copy
    from src.clean.schema import create_tables
    from src.common.config import load_settings
    from src.common.db import connect

    s = copy.deepcopy(load_settings())
    s["leagues"] = {"top": {"E0": "英超"}, "second": {}}
    s["schedule"]["lock_file"] = str(tmp_path / "jobs" / "scheduled_predict.lock")
    db = tmp_path / "t.db"
    con = connect(db)
    create_tables(con)
    con.executemany("insert into teams (team_id, country, name_zh) values (?, '英格兰', ?)",
                    [("Liverpool", "利物浦"), ("Arsenal", "阿森纳")])
    for mid, d, ko in [(1, "2026-10-03", "2026-10-03 14:00"), (10, "2026-11-02", "2026-11-02 00:00")]:
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                    " status, date_precision, round) values (?, 'E0', 2026, 'Liverpool', 'Arsenal', ?, ?,"
                    " 'scheduled', 'day', ?)", (mid, d, ko, str(mid)))
        con.execute("insert into match_sources values (?, 'football-data', ?, 'x.csv', '2026-10-20 00:00:00')",
                    (mid, f"k{mid}"))
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals) values (1, 'football-data', 2, 1)")
    con.commit()
    con.close()
    return s, db, tmp_path / "jobs"


def test_job_writes_heartbeat_and_stores_once(job_env):
    """入口跑两次：第一次存 1 行 T-24、心跳文件记成功和摘要；第二次“已出过”，不重复存。"""
    import sqlite3
    from src.jobs.heartbeat import read_state
    from src.jobs.scheduled_predict import JOB_NAME, run_job

    s, db, folder = job_env
    now = _now()
    run_job(s, now, db_path=db, folder=folder, model=FakeModel(), code="test")
    st = read_state(JOB_NAME, folder)
    assert st["成功"] is True and st["结束"] is not None and st["错误"] is None
    assert st["摘要"]["新存"] == 1 and st["摘要"]["T-24"] == 1 and st["摘要"]["窗口内比赛"] == 1

    con = sqlite3.connect(db)
    rows = con.execute("select kind, cutoff_utc, detail from predictions").fetchall()
    con.close()
    assert len(rows) == 1 and rows[0][0] == "T-24" and rows[0][1] == "2026-11-01 12:00:00"
    detail = json.loads(rows[0][2])
    assert detail["计划时点"] == 24 and detail["首发"] == "未含首发"
    assert detail["晚了几分钟"] == 720.0          # 计划时间 = 开球前 24 小时 = 昨天 00:00，晚了 12 小时

    run_job(s, now, db_path=db, folder=folder, model=FakeModel(), code="test")
    st = read_state(JOB_NAME, folder)
    assert st["成功"] is True and st["摘要"]["新存"] == 0 and st["摘要"]["已出过"] == 1
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from predictions").fetchone()[0] == 1
    con.close()
    assert not (folder / "scheduled_predict.lock").exists(), "跑完锁文件应该删掉"


def test_job_skips_when_locked(job_env):
    """锁被别的运行拿着：这次跳过，心跳记成功 + “跳过”，不存任何预测。"""
    import sqlite3
    from src.jobs.heartbeat import read_state
    from src.jobs.scheduled_predict import JOB_NAME, run_job

    s, db, folder = job_env
    other = RunLock(s["schedule"]["lock_file"], 60)
    assert other.acquire()
    try:
        run_job(s, _now(), db_path=db, folder=folder, model=FakeModel(), code="test")
    finally:
        other.release()
    st = read_state(JOB_NAME, folder)
    assert st["成功"] is True and st["摘要"]["跳过"] == "上次运行还没结束"
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from predictions").fetchone()[0] == 0
    con.close()
