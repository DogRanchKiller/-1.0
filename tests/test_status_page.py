"""数据状态页的自动检查。用临时数据库和临时文件夹，不碰真实数据，也不真的弹窗。"""
import os
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import pytest

from src.common.config import load_settings
from src.common.db import connect
from src.jobs import status_page as sp

UTC = timezone.utc
NOW = datetime(2026, 10, 31, 2, 0, tzinfo=UTC)  # 北京时间 10-31 10:00


def log(con, t, data_type, status, http=200, url="https://x/1", source="dongqiudi", note=None):
    con.execute("insert into fetch_log (url, source, data_type, fetched_at_utc, status, http_status, note)"
                " values (?, ?, ?, ?, ?, ?, ?)", (url, source, data_type, t.isoformat(timespec="seconds"),
                                                  status, http, note))
    con.commit()


@pytest.fixture
def env(tmp_path):
    con = connect(tmp_path / "test.db")
    con.execute("create table matches (match_id text, match_date text)")
    con.execute("insert into matches values ('a', '2026-10-29')")
    s = load_settings()
    s = {**s, "paths": {**s["paths"], "raw": str(tmp_path / "raw"), "backup": str(tmp_path / "backup")}}
    (tmp_path / "backup" / "database").mkdir(parents=True)
    return con, s, tmp_path


def make_backup(tmp_path, t):
    f = tmp_path / "backup" / "database" / "football_20261031.db"
    f.write_bytes(b"x")
    os.utime(f, (t.timestamp(), t.timestamp()))


def levels(checks):
    return {c.name: c.level for c in checks}


def test_all_ok(env):
    con, s, tmp = env
    log(con, NOW - timedelta(minutes=40), "fixtures", "saved")
    make_backup(tmp, NOW - timedelta(hours=6))
    checks, today, day, problems, rows = sp.build(con, s, NOW)
    lv = levels(checks)
    assert lv["懂球帝 每小时赔率任务"] == "ok" and lv["每日备份"] == "ok" and lv["有没有被网站拒绝"] == "ok"
    assert today == {"fixtures": {"saved": 1}} and problems == []


def test_problems_found(env):
    """超过 2 小时没成功、被网站拒绝、备份太旧、失败太多，都要标出来。"""
    con, s, tmp = env
    log(con, NOW - timedelta(hours=3), "fixtures", "saved")
    for i in range(4):
        log(con, NOW - timedelta(hours=1, minutes=i), "odds_live", "failed", http=500)
    log(con, NOW - timedelta(minutes=30), "odds_final", "failed", http=403)
    make_backup(tmp, NOW - timedelta(hours=40))
    checks, _, _, problems, _ = sp.build(con, s, NOW)
    lv = levels(checks)
    assert lv["懂球帝 每小时赔率任务"] == "error"
    assert lv["有没有被网站拒绝"] == "error"
    assert lv["失败和内容可疑"] == "warn"      # 4 次普通失败（403 不算在这里）
    assert lv["每日备份"] == "warn"
    assert lv["懂球帝 历史回补"] == "error"    # 最后一条是 403：回补停下了
    assert len(problems) == 5


def test_old_problems_ignored(env):
    """24 小时以前的失败不再算；今天的次数从北京时间 0 点算。"""
    con, s, tmp = env
    log(con, NOW - timedelta(hours=30), "odds_live", "failed", http=403)
    log(con, datetime(2026, 10, 30, 15, 30, tzinfo=UTC), "fixtures", "saved")  # 北京 10-30 23:30：昨天
    log(con, datetime(2026, 10, 30, 16, 30, tzinfo=UTC), "fixtures", "unchanged")  # 北京 10-31 00:30：今天
    checks, today, day, problems, _ = sp.build(con, s, NOW)
    assert levels(checks)["有没有被网站拒绝"] == "ok" and problems == []
    assert today == {"fixtures": {"unchanged": 1}}
    assert day == {"fixtures": {"saved": 1, "unchanged": 1}}


def test_alert_only_once(tmp_path):
    """同一个问题只提醒一次；问题消失后再出现，重新提醒。"""
    f = tmp_path / "alerts.json"
    bad = sp.Check("backup_stale", "每日备份", "warn", "旧")
    ok = sp.Check("backup", "每日备份", "ok", "新")
    assert sp.new_alerts([bad], f) == [bad]
    assert sp.new_alerts([bad], f) == []
    assert sp.new_alerts([ok], f) == []
    assert sp.new_alerts([bad], f) == [bad]


class Texts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.text = [], ""

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.text += data


def test_html_escapes_outside_text(env):
    """网址和说明来自外部，里面的 < > 不能变成网页标签。"""
    con, s, tmp = env
    log(con, NOW - timedelta(minutes=5), "odds_live", "suspicious", url="https://x/<script>alert(1)</script>",
        note="<b>坏</b>")
    checks, today, day, problems, rows = sp.build(con, s, NOW)
    page = sp.render_html(checks, today, day, problems, rows, NOW)
    p = Texts()
    p.feed(page)
    assert "script" not in p.tags and "b" not in p.tags
    assert "<script>alert(1)</script>" in p.text and "内容可疑" in p.text
    assert 'http-equiv="refresh"' in page
    md = sp.render_markdown(checks, today, problems, NOW)
    assert "# 抓取日报 2026-10-31" in md and "内容可疑" in md
