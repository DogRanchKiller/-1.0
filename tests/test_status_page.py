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
    con.execute("create table matches (match_id text, match_date text, competition text, season integer,"
                " home_team text, away_team text)")
    con.execute("insert into matches values ('a', '2026-10-29', 'E0', 2026, 'Arsenal', 'Newcomer')")
    con.execute("insert into matches values ('b', '2005-10-29', 'E0', 2005, 'Arsenal', 'Oldtown')")
    s = load_settings()
    # 报告文件夹、teams.yaml 都换成临时的，不读真实的副本
    s = {**s, "paths": {**s["paths"], "raw": str(tmp_path / "raw"), "backup": str(tmp_path / "backup"),
                        "reports": str(tmp_path / "reports"), "teams": str(tmp_path / "teams.yaml")}}
    (tmp_path / "backup" / "database").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
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
    checks, today, day, problems, rows, lists = sp.build(con, s, NOW)
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
    checks, _, _, problems, _, _ = sp.build(con, s, NOW)
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
    checks, today, day, problems, _, _ = sp.build(con, s, NOW)
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
    checks, today, day, problems, rows, lists = sp.build(con, s, NOW)
    page = sp.render_html(checks, today, day, problems, rows, NOW)
    p = Texts()
    p.feed(page)
    assert "script" not in p.tags and "b" not in p.tags
    assert "<script>alert(1)</script>" in p.text and "内容可疑" in p.text
    assert 'http-equiv="refresh"' in page
    md = sp.render_markdown(checks, today, problems, NOW)
    assert "# 抓取日报 2026-10-31" in md and "内容可疑" in md


# ---------- 待确认的队名、没对上的比赛 ----------
TEAMS_YAML = """球队:
  - {英文: "Arsenal", 中文: '阿森纳', 国家: 英格兰}
  - {英文: "Newcomer", 中文: '', 国家: 英格兰}
  - {英文: "Oldtown", 中文: '', 国家: 英格兰}
"""
NAMES_CSV = ("中文名,英文名,得票,占比,第二名,第二名得票,结果\n"
             "阿森纳,Arsenal,300,0.9,Chelsea,3,定下\n"
             "新来队,Newcomer,4,0.5,Oldtown,3,待人工确认\n")
UNMATCHED_HEAD = "联赛,赛季,懂球帝编号,主队,客队,比分,结果,原始文件\n"


def write_reports(tmp, names=NAMES_CSV, unmatched=UNMATCHED_HEAD, teams=TEAMS_YAML):
    """写导入报告（带 BOM，和导入程序一样）。传 None 就不写这个文件。"""
    for f, text in [("reports/dongqiudi_导入_队名对照.csv", names), ("reports/dongqiudi_导入_没对上.csv", unmatched),
                    ("teams.yaml", teams)]:
        if text is not None:
            (tmp / f).write_text(text, encoding="utf-8-sig" if f.endswith(".csv") else "utf-8")


def by_name(checks):
    return {c.name: c for c in checks}


def test_pending_names_and_unmatched(env):
    """没定下的队名、最近在五大联赛出场却没中文名的，要标出来；很久以前的球队只列出来。"""
    con, s, tmp = env
    write_reports(tmp, unmatched=UNMATCHED_HEAD + "I1,2016,1,萨索洛,佩斯卡拉,2-1,对上（比分不同）,x\n")
    checks, *_, lists = sp.build(con, s, NOW)
    c = by_name(checks)
    assert c["待确认的队名"].level == "warn" and c["待确认的队名"].key == "team_names_pending"
    assert "没定下来的 1 个" in c["待确认的队名"].text and "空着 2 个" in c["待确认的队名"].text
    assert "出场的 1 个" in c["待确认的队名"].text
    names, unmatched = lists
    assert names.total == 3
    assert [r[1] for r in names.rows] == ["新来队", "Newcomer", "Oldtown"]   # 最近出场的排在很久以前的前面
    assert c["懂球帝没对上的比赛"].level == "ok" and "对上（比分不同） 1 场" in c["懂球帝没对上的比赛"].text
    assert unmatched.rows == [["I1", "2016/17", "萨索洛 对 佩斯卡拉", "2-1", "对上（比分不同）"]]


def test_names_ok_and_too_many_unmatched(env):
    """队名都定下、空着的都是很久以前的球队：正常。没对上的超过界限：注意；页面只列前几行，写总数。"""
    con, s, tmp = env
    s = {**s, "status": {**s["status"], "list_rows": 3, "max_unmatched_matches": 4}}
    rows = "".join(f"F1,2019,{i},甲{i},乙{i},,数据库里没有这场,x\n" for i in range(5))
    write_reports(tmp, names=NAMES_CSV.splitlines()[0] + "\n阿森纳,Arsenal,300,0.9,,0,定下\n",
                  unmatched=UNMATCHED_HEAD + rows, teams=TEAMS_YAML.replace("'', 国家: 英格兰}\n  - {英文: \"Oldtown\"",
                                                                            "'新来', 国家: 英格兰}\n  - {英文: \"Oldtown\""))
    checks, *_, lists = sp.build(con, s, NOW)
    c = by_name(checks)
    assert c["待确认的队名"].level == "ok" and "出场的 0 个" in c["待确认的队名"].text
    assert c["懂球帝没对上的比赛"].level == "warn" and c["懂球帝没对上的比赛"].key == "unmatched_many"
    assert lists[1].total == 5 and len(lists[1].rows) == 3
    page = sp.render_html(checks, {}, {}, [], [], NOW, lists)
    assert "一共 5 场，只列了前 3 场" in page


def test_reports_unreadable(env):
    """文件不在、列名改了、行缺格、teams.yaml 坏了：页面写“读不到”，程序不出错。"""
    con, s, tmp = env
    checks, *_, lists = sp.build(con, s, NOW)          # 三个文件都不在
    c = by_name(checks)
    assert c["待确认的队名"].level == "warn" and c["待确认的队名"].key == "team_names_unreadable"
    assert "不存在" in c["待确认的队名"].text and "不存在" in c["懂球帝没对上的比赛"].text
    page = sp.render_html(checks, {}, {}, [], [], NOW, lists)
    assert "读不到" in page

    write_reports(tmp, names="中文,英文\n阿森纳,Arsenal\n", unmatched=UNMATCHED_HEAD + "I1,2016\n", teams="球队: [1, 2")
    checks, *_, lists = sp.build(con, s, NOW)
    c = by_name(checks)
    assert "没有这些列：中文名、英文名、得票、结果" in c["待确认的队名"].text
    assert "teams.yaml 读不懂" in c["待确认的队名"].text
    assert c["懂球帝没对上的比赛"].level == "ok"         # 缺格的行照样列出来，空格当空
    assert lists[1].rows == [["I1", "2016/17", " 对 ", "", ""]]
    md = sp.render_markdown(checks, {}, [], NOW, lists)
    assert "## 待确认的队名" in md and "读不到" in md


def test_listing_escapes_outside_text(env):
    """队名、比赛来自懂球帝，里面的 < > 不能变成网页标签。"""
    con, s, tmp = env
    write_reports(tmp, names=NAMES_CSV.replace("新来队", "<script>x</script>"),
                  unmatched=UNMATCHED_HEAD + "I1,2016,1,<b>主</b>,客,1-0,队名没对上,x\n")
    checks, today, day, problems, rows, lists = sp.build(con, s, NOW)
    p = Texts()
    p.feed(sp.render_html(checks, today, day, problems, rows, NOW, lists))
    assert "script" not in p.tags and "b" not in p.tags
    assert "<script>x</script>" in p.text and "<b>主</b> 对 客" in p.text
