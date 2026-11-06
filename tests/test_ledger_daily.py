"""定时任务“四本账对账”（src/jobs/ledger_daily.py）：每天把四本账算一遍，写结果文件和心跳文件。"""
import pandas as pd

from src.jobs import ledger_daily
from src.jobs.heartbeat import read_state
from tests.test_ledger import env, run  # noqa: F401（env 是 pytest 的 fixture）


def test_heartbeat_summary_is_short(env):
    rows, summary = run(env)
    got = ledger_daily.heartbeat_summary(rows, summary, env[1])
    assert got["对上账的预测"] == 3                                    # 编号 1、3、4（同一场同一时点取最早的）
    assert got["T-24 胜平负"].startswith("2 场，差 ")                  # T-24 胜平负对上 2 场
    assert len(got) <= 6                                               # 状态页最多显示 6 项
    assert all(isinstance(v, (int, str)) and len(str(v)) <= 40 for v in got.values())


def test_heartbeat_summary_without_rows(env):
    s = env[1]
    rows = pd.DataFrame()
    from src.evaluate.ledger import summarize
    got = ledger_daily.heartbeat_summary(rows, summarize(rows, s), s)
    assert got["对上账的预测"] == 0 and got["T-24 胜平负"] == "0 场"


def test_main_writes_files_and_heartbeat(env, tmp_path, monkeypatch):
    con, s = env
    db = con.execute("pragma database_list").fetchone()[2]               # 临时数据库文件的位置
    con.close()
    s["paths"]["database"] = db
    s["ledger"]["output_dir"] = str(tmp_path / "ledger")
    s["jobs"]["state_dir"] = str(tmp_path / "jobs")
    monkeypatch.setattr(ledger_daily, "load_settings", lambda: s)
    ledger_daily.main()
    assert (tmp_path / "ledger" / "ledger_matches.csv").exists() and (tmp_path / "ledger" / "ledger_summary.csv").exists()
    hb = read_state("ledger", tmp_path / "jobs")
    assert hb["成功"] is True and hb["摘要"]["对上账的预测"] == 3


def test_task_is_registered_and_shown_on_status_page():
    from pathlib import Path
    from src.common.config import load_settings
    root = Path(__file__).resolve().parents[1]
    ps1 = (root / "src" / "jobs" / "register_tasks.ps1").read_text(encoding="utf-8-sig")
    assert "-m src.jobs.ledger_daily" in ps1 and "足球预测_四本账对账" in ps1 and "09:40" in ps1
    s = load_settings()
    assert s["notify"]["jobs"]["ledger"]["name"] == "四本账对账"
    assert "ledger" not in (s["notify"].get("push_jobs") or [])           # 只在状态页显示，出错不推送
