"""推送出口（src/notify/、src/jobs/notify.py）和数据状态页“定时任务”栏目的自动检查。

全部用临时文件夹（tmp_path）和临时数据库，不碰真实数据；发送一律用假的 post，不联网
（下面的 no_network 把真的 requests.post 换成一碰就报错的函数，万一哪里漏了也连不出去）。
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import partial

import pytest
import requests
import yaml

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.jobs import heartbeat
from src.jobs import notify as notify_job
from src.jobs import status_page as sp
from src.notify import compose as cp
from src.notify import send as sd
from src.notify import setup as su
from src.notify.outbox import Outbox

UTC = timezone.utc
NOW = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)          # 北京时间 11-01 18:00
TOKEN = "abcdef0123456789SECRETTOKEN42"                  # 假口令，检查它不出现在任何输出里


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("测试里不许真的联网")
    monkeypatch.setattr(requests, "post", boom)


@pytest.fixture
def env(tmp_path):
    """临时设置：数据库、心跳文件夹、发件箱、状态文件都在 tmp_path 里。"""
    s = load_settings()
    s["paths"] = {**s["paths"], "database": str(tmp_path / "test.db"), "raw": str(tmp_path / "raw"),
                  "backup": str(tmp_path / "backup"), "reports": str(tmp_path / "reports"),
                  "teams": str(tmp_path / "teams.yaml")}
    s["jobs"] = {**s["jobs"], "state_dir": str(tmp_path / "jobs")}
    s["notify"] = {**s["notify"], "outbox_dir": str(tmp_path / "notify" / "outbox"),
                   "state_file": str(tmp_path / "notify" / "state.json")}
    con = sqlite3.connect(tmp_path / "test.db")
    create_tables(con)
    con.executemany("insert into teams values (?, ?, ?)",
                    [("Arsenal", "阿森纳", "英格兰"), ("Chelsea", "切尔西", "英格兰"), ("Valencia", "瓦伦西亚", "西班牙")])
    con.commit()
    yield s, con, tmp_path
    con.close()


DETAIL = {"融合概率": [0.5, 0.3, 0.2],
          "融合": {"信心等级": 3, "不确定因素": ["赛季前几轮"],
                   "倾向": {"result": {"side": 0, "p": 0.5, "lean": True}, "draw": {"p": 0.3, "risk": True},
                            "over25": {"side": "over", "p": 0.55, "lean": True}, "btts": None, "ah": None}}}


def add_pred(con, match_id, kind, kickoff, home="Arsenal", away="Chelsea", comp="E0", p=(0.4, 0.3, 0.3),
             detail=DETAIL) -> int:
    cur = con.execute(
        "insert into predictions (match_id, competition, season, home_team, away_team, kickoff_utc, match_date, kind,"
        " model, model_version, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail)"
        " values (?, ?, 2026, ?, ?, ?, ?, ?, 'gb_l7', 'v1', '2026-11-01 09:00:00', '2026-11-01 09:00:00', ?, ?, ?, ?)",
        (match_id, comp, home, away, kickoff.strftime("%Y-%m-%d %H:%M"), kickoff.date().isoformat(), kind, *p,
         None if detail is None else json.dumps(detail, ensure_ascii=False)))
    con.commit()
    return cur.lastrowid


def messages(s, folder="outbox"):
    box = Outbox.from_settings(s)
    d = {"outbox": box.folder, "sent": box.sent, "failed": box.failed}[folder]
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.exists() else []


def heartbeat_file(tmp, job, **kw):
    data = {"任务": job, "开始": "2026-11-01 09:50:00", "结束": "2026-11-01 09:51:00", "成功": True,
            "摘要": {}, "错误": None, "上次成功": "2026-11-01 09:51:00", **kw}
    (tmp / "jobs").mkdir(exist_ok=True)
    (tmp / "jobs" / f"{job}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class FakePost:
    """假的 requests.post：记下每次调用，按 replies 依次回答（字典 = 返回的 JSON；异常 = 抛出去）。"""

    def __init__(self, *replies):
        self.replies = list(replies) or [{"code": 200, "msg": "请求成功"}]
        self.calls = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append(json)
        r = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(r, BaseException):
            raise r

        class Resp:
            status_code = 200

            def json(self_inner):
                return r
        return Resp()


SECRETS = {"pushplus": {"token": TOKEN}}


# ======================= 预测消息 =======================
def test_merge_into_one(env):
    """一次运行找到的几场比赛合并成一条；中文队名、北京时间、融合结论、模型概率都在。"""
    s, con, tmp = env
    add_pred(con, 1, "T-24", NOW + timedelta(hours=24))
    add_pred(con, 2, "T-24", NOW + timedelta(hours=25), home="Valencia", away="Real Madrid", comp="SP1")
    add_pred(con, 3, "T-6", NOW + timedelta(hours=6), detail=None)
    out = cp.compose(s, NOW)
    assert out["预测消息"] == 1
    [m] = messages(s)
    assert m["种类"] == "预测" and m["标题"] == "足球预测：T-6 1 场，T-24 2 场"
    body = m["正文"]
    assert "英超　阿森纳 对 切尔西" in body and "西甲　瓦伦西亚 对 Real Madrid" in body   # 没中文名的用英文
    assert "11 月 2 日 星期一 18:00（北京时间）" in body                                  # UTC 10:00 = 北京 18:00
    assert "融合结论：主胜 50% / 平 30% / 客胜 20%，信心 ●●● 高" in body
    assert "倾向 主胜 50%" in body and "平局风险高（30%）" in body and "要注意：赛季前几轮" in body
    assert "模型概率：主胜 40% / 平 30% / 客胜 30%（没有融合结论）" in body                # detail 为空的那场
    assert body.index("（T-6）") < body.index("阿森纳 对 切尔西**（T-24）")                # 按开球时间排


def test_no_repeat_and_latest_row(env):
    """推过的不再推；同一场同一版只认 prediction_id 最大的；开过球的、preview 不推；改期重出的新行再推。"""
    s, con, tmp = env
    add_pred(con, 1, "T-24", NOW + timedelta(hours=24), p=(0.9, 0.05, 0.05), detail={})
    add_pred(con, 1, "T-24", NOW + timedelta(hours=24), p=(0.4, 0.3, 0.3), detail={})       # 同一场同一版的新一行
    add_pred(con, 2, "T-1", NOW - timedelta(hours=1))                                       # 已经开球
    add_pred(con, 3, "preview", NOW + timedelta(hours=30))                                  # 预览页的，不推
    assert cp.compose(s, NOW)["预测消息"] == 1
    [m] = messages(s)
    assert "主胜 40%" in m["正文"] and "主胜 90%" not in m["正文"] and "T-1" not in m["正文"]
    assert cp.compose(s, NOW + timedelta(minutes=15))["预测消息"] == 0                       # 再跑一次：没有新的
    pid = add_pred(con, 1, "T-24", NOW + timedelta(hours=48), detail={"改期重出": True})   # 改期，按新时间再出一行
    assert cp.compose(s, NOW + timedelta(minutes=30))["预测消息"] == 1
    assert "改期后重新出的一版" in messages(s)[-1]["正文"]
    state = json.loads((tmp / "notify" / "state.json").read_text(encoding="utf-8"))
    assert str(pid) in state["已推预测"]


def test_outbox_dedup(tmp_path):
    """同样的 种类 + 关键字 不重复生成（发出去挪到 sent/ 以后也不再生成）。"""
    box = Outbox(tmp_path / "outbox")
    m = box.add("故障", "t", "c", key="k", now=NOW)
    assert m and box.add("故障", "t2", "c2", key="k", now=NOW) is None
    [(path, msg)] = box.pending()
    box.mark_sent(path, msg, "2026-11-01 10:00:00")
    assert box.add("故障", "t", "c", key="k", now=NOW) is None and box.pending() == []
    assert box.add("预测", "t", "c", key="k", now=NOW) is not None                         # 种类不同算两条
    with pytest.raises(ValueError):
        box.add("别的", "t", "c", key="k")


def test_long_content_split(env):
    """正文太长时拆成几条，标题写（1/2）。"""
    s, con, tmp = env
    s["notify"]["max_content_chars"] = 400
    for i in range(4):
        add_pred(con, 10 + i, "T-24", NOW + timedelta(hours=24, minutes=i))
    cp.compose(s, NOW)
    ms = [m for _, m in Outbox.from_settings(s).pending()]               # 按发送顺序
    assert len(ms) > 1 and ms[0]["标题"].endswith(f"（1/{len(ms)}）")
    assert all(len(m["正文"]) <= 400 for m in ms)


def test_no_predictions_table_or_db(tmp_path):
    """数据库里没有 predictions 表、或者数据库文件不在：不出错，0 条。"""
    s = load_settings()
    s["notify"] = {**s["notify"], "outbox_dir": str(tmp_path / "o"), "state_file": str(tmp_path / "st.json")}
    sqlite3.connect(tmp_path / "empty.db").close()
    assert cp.compose(s, NOW, db_path=tmp_path / "empty.db", jobs_dir=tmp_path / "jobs")["预测消息"] == 0
    out = cp.compose(s, NOW, db_path=tmp_path / "nothing.db", jobs_dir=tmp_path / "jobs")
    assert out["预测消息"] == 0 and out["说明"] == "找不到数据库" and not (tmp_path / "nothing.db").exists()


# ======================= 发送 =======================
def fill_outbox(s, n):
    box = Outbox.from_settings(s)
    for i in range(n):
        box.add("故障", f"标题{i}", f"正文{i}", key=f"k{i}", now=NOW - timedelta(minutes=n - i))


@pytest.mark.parametrize("secrets", [{}, {"pushplus": {"token": ""}}, {"pushplus": None}, {"push": {"token": "x"}}])
def test_no_token_keeps_outbox(env, secrets):
    """没有口令（没配、空的、格式不对）：消息留在 outbox，不报错，也不发。"""
    s, con, tmp = env
    fill_outbox(s, 2)
    post = FakePost()
    out = sd.send_outbox(s, secrets, NOW, post)
    assert post.calls == [] and len(messages(s)) == 2 and out["推送"].startswith("还没设置微信推送")


def test_send_ok_moves_to_sent(env):
    s, con, tmp = env
    fill_outbox(s, 2)
    post = FakePost()
    out = sd.send_outbox(s, SECRETS, NOW, post)
    assert out["已发"] == 2 and messages(s) == [] and len(messages(s, "sent")) == 2
    call = post.calls[0]
    assert call["token"] == TOKEN and call["channel"] == "wechat" and call["template"] == "markdown"
    assert call["title"] == "标题0"                                       # 先进先发


def test_rate_limits(env):
    """每分钟上限：一次运行最多 4 条，剩下留着；每天上限（北京时间 0 点起算）到了就不发。"""
    s, con, tmp = env
    fill_outbox(s, 6)
    post = FakePost()
    out = sd.send_outbox(s, SECRETS, NOW, post)
    assert len(post.calls) == 4 and out["还在发件箱"] == 2 and "1 分钟内" in out["停下原因"]
    out = sd.send_outbox(s, SECRETS, NOW + timedelta(minutes=15), post)    # 下一次运行接着发
    assert len(post.calls) == 6 and out["还在发件箱"] == 0

    s["notify"]["max_per_day"] = 7
    box = Outbox.from_settings(s)
    box.add("故障", "新", "新", key="新的", now=NOW)
    box.add("故障", "新2", "新2", key="新的2", now=NOW)
    out = sd.send_outbox(s, SECRETS, NOW + timedelta(minutes=30), post)
    assert len(post.calls) == 7 and "每天的上限" in out["停下原因"] and out["还在发件箱"] == 1
    # 北京时间第二天 0 点以后（UTC 16:00）重新计数
    out = sd.send_outbox(s, SECRETS, datetime(2026, 11, 1, 16, 1, tzinfo=UTC), post)
    assert len(post.calls) == 8 and out["还在发件箱"] == 0


def test_retry_limit_then_failed(env):
    """一直失败：每次运行试一次，加上第一次一共 max_retries + 1 次，然后挪到 failed/，写明原因和次数。"""
    s, con, tmp = env
    fill_outbox(s, 1)
    post = FakePost({"code": 905, "msg": "账号未实名"})
    n = int(s["notify"]["max_retries"]) + 1
    for i in range(n):
        out = sd.send_outbox(s, SECRETS, NOW + timedelta(minutes=15 * i), post)
    assert len(post.calls) == n and messages(s) == [] and out["最终失败"] == 1
    [m] = messages(s, "failed")
    assert m["尝试次数"] == n and f"试了 {n} 次" in m["失败原因"]
    assert "code 905" in m["失败原因"] and "实名" in m["失败原因"] and "待核实" in m["失败原因"]


def test_expired_not_sent(env):
    s, con, tmp = env
    Outbox.from_settings(s).add("预测", "旧", "旧", key="旧", now=NOW - timedelta(hours=30))
    post = FakePost()
    out = sd.send_outbox(s, SECRETS, NOW, post)
    assert post.calls == [] and out["过期"] == 1 and "过期" in messages(s, "failed")[0]["失败原因"]


@pytest.mark.parametrize("delay", [timedelta(hours=1), timedelta(hours=2)])
def test_prediction_retry_stops_at_kickoff(env, delay):
    """T-1 推送断网后，到开球或开球后恢复，不再发赛前预测，即使生成还不到 24 小时。"""
    s, con, tmp = env
    add_pred(con, 1, "T-1", NOW + timedelta(hours=1))
    cp.compose(s, NOW)
    offline = FakePost(requests.ConnectionError("测试断网"))
    assert sd.send_outbox(s, SECRETS, NOW, offline)["失败待重试"] == 1
    online = FakePost()
    out = sd.send_outbox(s, SECRETS, NOW + delay, online)
    assert online.calls == []
    assert out["过期"] == 1 and messages(s) == []
    assert "开球" in messages(s, "failed")[0]["失败原因"]


def test_split_prediction_messages_have_separate_deadlines(env):
    """拆开的消息分别按自己的最早开球时间过期，晚场不会被另一条里的早场一起丢掉。"""
    s, con, tmp = env
    s["notify"]["max_content_chars"] = 200
    add_pred(con, 1, "T-1", NOW + timedelta(hours=1))
    add_pred(con, 2, "T-6", NOW + timedelta(hours=6))
    assert cp.compose(s, NOW)["预测消息"] == 2
    post = FakePost()
    out = sd.send_outbox(s, SECRETS, NOW + timedelta(hours=2), post)
    assert out["过期"] == 1 and out["已发"] == 1 and len(post.calls) == 1
    assert "T-6" in post.calls[0]["content"]


def test_legacy_prediction_without_deadline_is_not_sent(env):
    """旧发件箱没有开球期限时无法证明仍是赛前消息，留失败原因，不盲发。"""
    s, con, tmp = env
    Outbox.from_settings(s).add("预测", "旧格式", "未知开球时间", key="旧格式", now=NOW)
    post = FakePost()
    out = sd.send_outbox(s, SECRETS, NOW, post)
    assert post.calls == [] and out["过期"] == 1
    assert "开球期限" in messages(s, "failed")[0]["失败原因"]


def test_network_error_stops_and_masks(env):
    """网络出错：这次停下（不白白用掉后面消息的重试次数）；异常说明里的口令换成 ***。"""
    s, con, tmp = env
    fill_outbox(s, 3)
    post = FakePost(requests.ConnectionError(f"连不上 https://www.pushplus.plus/send?token={TOKEN}"))
    out = sd.send_outbox(s, SECRETS, NOW, post)
    assert len(post.calls) == 1 and out["失败待重试"] == 1 and "网络不通" in out["停下原因"]
    assert TOKEN not in json.dumps(out, ensure_ascii=False) and "***" in out["最近错误"]
    assert TOKEN not in json.dumps(messages(s), ensure_ascii=False)


@pytest.mark.parametrize("reply", [{"code": 500, "msg": "对方暂时出错"}, requests.ConnectionError("测试断网")])
def test_job_send_failure_has_failed_heartbeat_and_can_retry(env, monkeypatch, reply):
    """真实发送尝试失败必须让入口失败；摘要和队列保留，恢复后的下一次运行仍能成功发出。"""
    s, con, tmp = env
    add_pred(con, 1, "T-1", NOW + timedelta(hours=1))
    monkeypatch.setattr(notify_job, "load_settings", lambda: s)
    monkeypatch.setattr(notify_job, "load_secrets", lambda: SECRETS)
    monkeypatch.setattr(notify_job, "job_run", partial(heartbeat.job_run, folder=tmp / "jobs"))
    heartbeat_file(tmp, "notify", 上次成功="2026-11-01 09:00:00")
    with pytest.raises(RuntimeError, match="发送失败"):
        notify_job.main(NOW, FakePost(reply))
    hb = heartbeat.read_state("notify", tmp / "jobs")
    assert hb["成功"] is False and hb["上次成功"] == "2026-11-01 09:00:00"
    assert hb["摘要"]["失败待重试"] == 1 and hb["摘要"]["已发"] == 0
    assert len(messages(s)) == 1 and messages(s)[0]["尝试次数"] == 1
    checks = sp.check_jobs(tmp / "jobs", s, NOW)
    assert next(c for c in checks if c.key.startswith("job_notify_")).level == "error"
    recovered = FakePost()
    out = notify_job.main(NOW + timedelta(minutes=15), recovered)
    assert out["已发"] == 1 and len(recovered.calls) == 1 and not messages(s)
    assert heartbeat.read_state("notify", tmp / "jobs")["成功"] is True


def test_job_final_send_failure_is_failed_not_a_successful_empty_queue(env, monkeypatch):
    """最后一次重试也失败时，消息移到 failed，任务仍必须记失败，不能因待发队列清空变成功。"""
    s, con, tmp = env
    s["notify"]["max_retries"] = 0
    add_pred(con, 1, "T-1", NOW + timedelta(hours=1))
    monkeypatch.setattr(notify_job, "load_settings", lambda: s)
    monkeypatch.setattr(notify_job, "load_secrets", lambda: SECRETS)
    monkeypatch.setattr(notify_job, "job_run", partial(heartbeat.job_run, folder=tmp / "jobs"))
    with pytest.raises(RuntimeError, match="发送失败"):
        notify_job.main(NOW, FakePost({"code": 500, "msg": "暂时出错"}))
    hb = heartbeat.read_state("notify", tmp / "jobs")
    assert hb["成功"] is False and hb["摘要"]["最终失败"] == 1
    assert not messages(s) and len(messages(s, "failed")) == 1


@pytest.mark.parametrize("mode", ["quota", "expired"])
def test_job_quota_or_normal_expiration_is_success(env, monkeypatch, mode):
    """没有实际发送失败时，正常限流或清理过期预测都算成功，不误报系统坏了。"""
    s, con, tmp = env
    add_pred(con, 1, "T-1", NOW + timedelta(hours=1))
    cp.compose(s, NOW)
    run_at = NOW
    if mode == "quota":
        path = sd.resolve_path(s["notify"]["state_file"])
        state = cp.load_state(path)
        state["发送记录"] = [NOW.strftime("%Y-%m-%d %H:%M:%S")] * s["notify"]["max_per_minute"]
        cp.save_state(path, state)
    else:
        run_at += timedelta(hours=1)
    monkeypatch.setattr(notify_job, "load_settings", lambda: s)
    monkeypatch.setattr(notify_job, "load_secrets", lambda: SECRETS)
    monkeypatch.setattr(notify_job, "job_run", partial(heartbeat.job_run, folder=tmp / "jobs"))
    post = FakePost()
    out = notify_job.main(run_at, post)
    assert not post.calls and heartbeat.read_state("notify", tmp / "jobs")["成功"] is True
    if mode == "expired":
        assert out["过期"] == 1
    else:
        assert len(messages(s)) == 1 and "下次运行" in out["停下原因"]


def test_token_never_leaks(env, monkeypatch, capsys):
    """整个定时任务跑一遍（对方回的说明、网络异常里都带着口令）：屏幕、心跳文件、发件箱、状态文件里都没有口令。"""
    s, con, tmp = env
    add_pred(con, 1, "T-24", NOW + timedelta(hours=24))
    heartbeat_file(tmp, "scheduled_predict", 成功=False, 错误=f"RuntimeError: token={TOKEN} 出错")
    monkeypatch.setattr(notify_job, "load_settings", lambda: s)
    monkeypatch.setattr(notify_job, "load_secrets", lambda: SECRETS)
    monkeypatch.setattr(notify_job, "job_run", partial(heartbeat.job_run, folder=tmp / "jobs"))
    post = FakePost({"code": 999, "msg": f"口令 {TOKEN} 不对"}, requests.Timeout(f"timeout {TOKEN}"))
    for now in (NOW, NOW + timedelta(minutes=15)):
        with pytest.raises(RuntimeError, match="发送失败") as exc:
            notify_job.main(now, post)
        assert TOKEN not in str(exc.value)
    assert len(post.calls) >= 2
    assert TOKEN not in capsys.readouterr().out
    for p in tmp.rglob("*"):
        # 跳过数据库和测试自己写的假心跳文件（模拟别的任务出错时错误里带着口令，是输入不是我们的输出）
        if p.is_file() and p.suffix != ".db" and p.name != "scheduled_predict.json":
            assert TOKEN not in p.read_text(encoding="utf-8", errors="ignore"), p
    hb = heartbeat.read_state("notify", tmp / "jobs")
    assert hb["成功"] is False and hb["摘要"]["推送"] == "已设置微信推送"


def test_job_without_token(env, monkeypatch, capsys):
    """没有口令时定时任务照样成功，心跳摘要写“还没设置微信推送”。"""
    s, con, tmp = env
    add_pred(con, 1, "T-24", NOW + timedelta(hours=24))
    monkeypatch.setattr(notify_job, "load_settings", lambda: s)
    monkeypatch.setattr(notify_job, "load_secrets", lambda: {})
    monkeypatch.setattr(notify_job, "job_run", partial(heartbeat.job_run, folder=tmp / "jobs"))
    out = notify_job.main(NOW)
    hb = heartbeat.read_state("notify", tmp / "jobs")
    assert hb["成功"] is True and "还没设置微信推送" in hb["摘要"]["推送"] and out["预测消息"] == 1
    assert len(messages(s)) == 1 and "还没设置微信推送" in capsys.readouterr().out


def test_send_test_messages(env):
    """测试推送：没口令、网络不通、可能没实名、成功，各有中文说明。"""
    s, con, tmp = env
    ok, text = sd.send_test(s, {}, NOW)
    assert not ok and "还没设置口令" in text
    ok, text = sd.send_test(s, SECRETS, NOW, FakePost(requests.ConnectionError(TOKEN)))
    assert not ok and "网络不通" in text and TOKEN not in text
    ok, text = sd.send_test(s, SECRETS, NOW, FakePost({"code": 905, "msg": "x"}))
    assert not ok and "实名" in text
    post = FakePost()
    ok, text = sd.send_test(s, SECRETS, NOW + timedelta(minutes=5), post)
    assert ok and "发送成功" in text and post.calls[0]["title"] == "足球预测系统：测试消息"


# ======================= 故障消息 =======================
def test_fault_and_recovery(env):
    """任务失败推一条；没恢复前不再推（错误换了说法也不推）；恢复推一条“已恢复”；再坏再推。"""
    s, con, tmp = env
    heartbeat_file(tmp, "scheduled_predict", 成功=False, 错误="Traceback...\nValueError: 坏了",
                   上次成功="2026-11-01 09:00:00")
    out = cp.compose(s, NOW)
    assert out["新故障"] == 1 and out["正在提醒的故障"] == 1
    [m] = messages(s)
    assert m["种类"] == "故障" and "定时预测" in m["标题"] and "ValueError: 坏了" in m["正文"]
    heartbeat_file(tmp, "scheduled_predict", 成功=False, 错误="ValueError: 换了说法", 上次成功="2026-11-01 09:00:00")
    assert cp.compose(s, NOW + timedelta(minutes=15))["新故障"] == 0 and len(messages(s)) == 1
    heartbeat_file(tmp, "scheduled_predict", 结束="2026-11-01 10:31:00", 上次成功="2026-11-01 10:31:00")
    out = cp.compose(s, NOW + timedelta(minutes=30))
    assert out["已恢复"] == 1 and out["正在提醒的故障"] == 0
    assert "已恢复" in messages(s)[-1]["标题"] and "已经好了" in messages(s)[-1]["正文"]
    heartbeat_file(tmp, "scheduled_predict", 成功=False, 错误="ValueError: 又坏了")
    assert cp.compose(s, NOW + timedelta(minutes=45))["新故障"] == 1 and len(messages(s)) == 3


def test_stuck_stale_and_never(env):
    """“结束”为空太久 = 卡住；上次成功太久 = 太久没成功；没有心跳文件 = 还没运行过，不算故障。"""
    s, con, tmp = env
    heartbeat_file(tmp, "scheduled_predict", 开始="2026-11-01 08:00:00", 结束=None, 成功=None)
    heartbeat_file(tmp, "review", 上次成功="2026-10-30 00:00:00")
    out = cp.compose(s, NOW)                                                   # site 没有心跳文件
    assert out["新故障"] == 2
    body = messages(s)[0]["正文"]
    assert "还没结束" in body and "超过 18 小时没成功过" in body and "网站" not in body


def test_no_betting_words(env):
    """消息里只写预测和概率：没有“推荐”“下注”“投注”。"""
    s, con, tmp = env
    add_pred(con, 1, "T-24", NOW + timedelta(hours=24))
    add_pred(con, 2, "T-1", NOW + timedelta(hours=1), detail=None)
    heartbeat_file(tmp, "site", 成功=False, 错误="OSError: x")
    cp.compose(s, NOW)
    heartbeat_file(tmp, "site")
    cp.compose(s, NOW + timedelta(minutes=15))
    _, text = sd.send_test(s, SECRETS, NOW, FakePost())
    all_text = json.dumps(messages(s), ensure_ascii=False) + text
    assert len(messages(s)) == 3
    for word in ("推荐", "下注", "投注"):
        assert word not in all_text


# ======================= 设置口令 =======================
def test_setup_keeps_other_keys(tmp_path, capsys):
    f = tmp_path / "secrets.yaml"
    f.write_text("push:\n  token: old\n  retry_waits: [30, 120]\nother: 1\npushplus:\n  note: 留着\n", encoding="utf-8")
    assert su.main(ask=lambda prompt: f"  {TOKEN}  ", path=f)
    data = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert data == {"push": {"token": "old", "retry_waits": [30, 120]}, "other": 1,
                    "pushplus": {"note": "留着", "token": TOKEN}}
    out = capsys.readouterr().out
    assert TOKEN not in out and f"{len(TOKEN)} 个字符" in out
    assert sd.get_token(data) == TOKEN


def test_setup_new_file_and_bad_input(tmp_path, capsys):
    f = tmp_path / "secrets.yaml"
    assert not su.main(ask=lambda p: "", path=f) and not f.exists()
    assert not su.main(ask=lambda p: "abc def ghi jkl mno", path=f) and not f.exists()
    assert su.main(ask=lambda p: TOKEN, path=f)
    assert yaml.safe_load(f.read_text(encoding="utf-8")) == {"pushplus": {"token": TOKEN}}
    f.write_text("- 1\n- 2\n", encoding="utf-8")                              # 不认识的格式：不覆盖
    assert not su.main(ask=lambda p: TOKEN, path=f) and f.read_text(encoding="utf-8") == "- 1\n- 2\n"
    assert TOKEN not in capsys.readouterr().out


# ======================= 数据状态页“定时任务” =======================
def test_status_page_jobs(env, monkeypatch):
    """各任务一行：成功正常、失败出错、太久没成功注意、无心跳写“还没运行过”；新问题进弹窗。"""
    s, con, tmp = env
    heartbeat_file(tmp, "scheduled_predict", 摘要={"新存": 3, "新存的编号": [1, 2]})
    heartbeat_file(tmp, "notify", 成功=False, 错误=f"Traceback\nRuntimeError: token {TOKEN}")
    heartbeat_file(tmp, "review", 上次成功="2026-10-29 00:00:00")
    sc = connect(tmp / "status.db")
    sc.execute("create table matches (match_id text, match_date text, competition text, season integer,"
               " home_team text, away_team text)")
    (tmp / "backup" / "database").mkdir(parents=True)
    (tmp / "reports").mkdir()
    monkeypatch.setattr(sp, "load_secrets", lambda: SECRETS)
    try:
        checks, *_ = sp.build(sc, s, NOW)
    finally:
        sc.close()
    c = {x.name: x for x in checks}
    old = ["懂球帝 每小时赔率任务", "每日备份", "数据库", "待确认的队名", "懂球帝没对上的比赛"]
    assert all(n in c for n in old)                                          # 原有的检查都还在
    assert [x.name for x in checks][-5:] == ["定时任务：数据采集与导入", "定时任务：定时预测", "定时任务：赛后对账、复盘",
                                              "定时任务：网站重新生成", "定时任务：推送"]
    assert c["定时任务：定时预测"].level == "ok" and "新存 3" in c["定时任务：定时预测"].text
    assert "新存的编号" not in c["定时任务：定时预测"].text
    assert c["定时任务：推送"].level == "error" and "RuntimeError" in c["定时任务：推送"].text
    assert TOKEN not in c["定时任务：推送"].text and "***" in c["定时任务：推送"].text
    assert c["定时任务：赛后对账、复盘"].level == "warn" and "超过 18 小时没成功过" in c["定时任务：赛后对账、复盘"].text
    assert c["定时任务：网站重新生成"].level == "ok" and "还没运行过" in c["定时任务：网站重新生成"].text
    fresh = sp.new_alerts(checks, tmp / "alerts.json")
    assert {x.name for x in fresh} >= {"定时任务：推送", "定时任务：赛后对账、复盘"}
    assert "定时任务：网站重新生成" not in {x.name for x in fresh}
    page_html = sp.render_html(checks, {}, {}, [], [], NOW)
    assert "定时任务：网站重新生成" in page_html and "还没运行过" in page_html


def test_status_page_no_heartbeat_folder(env):
    """心跳文件夹整个不在：设置里每个任务都显示“还没运行过”，不算问题。"""
    s, con, tmp = env
    checks = sp.check_jobs(tmp / "no_such_dir", s, NOW)
    assert len(checks) == len(s["notify"]["jobs"]) and all(x.level == "ok" and "还没运行过" in x.text for x in checks)
