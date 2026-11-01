"""正式网站（src/web/site/、src/jobs/build_site.py）的自动检查。临时数据库 + 临时 data/review/，不碰真实数据。

造的数据（现在 = 2026-11-06 12:00 UTC，网站只看英超 E0）：
  1   2026-11-04 15:00 利物浦 对 阿森纳，已踢完 2–1。有 T-24、T-6、T-1（存了 11 × 11 比分矩阵，T-6 只有模型矩阵）
      和两行预览版（没存矩阵；旧的那行不该被用）。scores.csv、单场复盘 JSON 都有。
  10  2026-11-07 10:00 利物浦 对 曼城，还没踢。T-24 已经出了（晚了 5 分钟），T-6、T-1 还没到时间；有赔率变动。
  11  2026-11-08 15:00 曼城 对 “A/B: C?*”（队名带 Windows 文件名不能用的符号，也没有中文名），没有预测、没有赔率。
  20  2026-10-30 欧冠 阿森纳 对 “A/B: C?*”（杯赛，只有赛程）。
"""
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.common.markets import poisson_score_matrix
from src.jobs import build_site
from src.web.site import pages
from src.web.site.common import TeamFiles, safe_name
from src.web.site.data import matrix

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
WEIRD = "A/B: C?*"


def fake_matrix(home_xg=1.5, away_xg=1.1) -> list:
    """和存档里一样：11 × 11，6 位小数（加起来和 1 差一点点）。"""
    return np.round(poisson_score_matrix(home_xg, away_xg, max_goals=10), 6).tolist()


def fusion(level=3, p=(0.5, 0.27, 0.23)) -> dict:
    return {"状态": "已融合", "w": 0.6, "大小球": {"line": 2.5, "p_over": 0.55, "n_books": 2},
            "亚盘": {"line": -0.5, "home": 1.95, "away": 1.9, "n_books": 2}, "大2.5球": 0.55, "双方进球": 0.52,
            "倾向": {"result": {"side": 0, "p": p[0], "lean": True}, "draw": {"p": p[1], "risk": False},
                     "over25": {"side": "over", "p": 0.55, "lean": True}, "btts": {"side": "yes", "p": 0.52, "lean": False},
                     "ah": {"side": "home", "p": 0.53, "line": -0.5, "lean": True}},
            "信心等级": level, "不确定因素": ["赛季前几轮"], "赔率变动": 0.01}


def scheduled_detail(hours, late=0, fused=True) -> dict:
    return {"计划时点": hours, "晚了几分钟": late, "模型比分矩阵": fake_matrix(1.4, 1.2),
            "融合比分矩阵": fake_matrix() if fused else None, "首发": "未含首发",
            "融合概率": [0.5, 0.27, 0.23] if fused else None, "融合": fusion() if fused else {"状态": "无赔率，未融合"}}


@pytest.fixture
def env(tmp_path):
    s = copy.deepcopy(load_settings())
    s["leagues"] = {"top": {"E0": "英超"}, "second": {}}
    s["site"]["review_dir"] = str(tmp_path / "review")
    db = tmp_path / "t.db"
    con = connect(db)
    create_tables(con)
    con.executemany("insert into teams (team_id, country, name_zh) values (?, '英格兰', ?)",
                    [("Liverpool", "利物浦"), ("Arsenal", "阿森纳"), ("Man City", "曼城"), (WEIRD, None)])
    ms = [(1, "E0", "Liverpool", "Arsenal", "2026-11-04", "2026-11-04 15:00", "finished"),
          (10, "E0", "Liverpool", "Man City", "2026-11-07", "2026-11-07 10:00", "scheduled"),
          (11, "E0", "Man City", WEIRD, "2026-11-08", "2026-11-08 15:00", "scheduled"),
          (20, "UCL", "Arsenal", WEIRD, "2026-10-30", "2026-10-30 20:00", "finished")]
    for mid, comp, h, a, d, ko, st in ms:
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                    " status, round) values (?, ?, 2026, ?, ?, ?, ?, ?, ?)", (mid, comp, h, a, d, ko, st, str(mid)))
        con.execute("insert into match_sources values (?, 'football-data', ?, 'x.csv', '2026-10-20 00:00:00')", (mid, f"k{mid}"))
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals) values (1, 'football-data', 2, 1)")
    preds = [  # (match_id, kind, cutoff_utc, p, detail)
        (1, "preview", "2026-11-01 08:00:00", (0.2, 0.3, 0.5), {}),                       # 旧的预览版，不该被用
        (1, "T-24", "2026-11-03 15:00:00", (0.46, 0.28, 0.26), scheduled_detail(24)),
        (1, "T-6", "2026-11-04 09:00:00", (0.47, 0.28, 0.25), scheduled_detail(6, fused=False)),
        (1, "T-1", "2026-11-04 14:00:00", (0.48, 0.27, 0.25), scheduled_detail(1)),
        (1, "preview", "2026-11-03 20:00:00", (0.45, 0.29, 0.26), {"融合概率": [0.5, 0.27, 0.23], "融合": fusion(2)}),
        (10, "T-24", "2026-11-06 10:05:00", (0.4, 0.3, 0.3), scheduled_detail(24, late=5)),
        (10, "preview", "2026-11-05 20:00:00", (0.41, 0.3, 0.29), {"融合概率": [0.42, 0.3, 0.28], "融合": fusion()}),
    ]
    info = {1: ("Liverpool", "Arsenal", "2026-11-04 15:00", "2026-11-04"), 10: ("Liverpool", "Man City", "2026-11-07 10:00", "2026-11-07")}
    for mid, kind, cut, p, d in preds:
        h, a, ko, md = info[mid]
        con.execute("insert into predictions (match_id, competition, season, home_team, away_team, kickoff_utc, match_date,"
                    " kind, model, model_version, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail)"
                    " values (?, 'E0', 2026, ?, ?, ?, ?, ?, 'gb', 'gb-1', ?, ?, ?, ?, ?, ?)",
                    (mid, h, a, ko, md, kind, cut, cut, *p, json.dumps(d, ensure_ascii=False)))
    oc = [("Bet365", "1x2", None, 2.0, 3.5, 4.0, None, None, "2026-11-05 08:00:00"),
          ("Bet365", "1x2", None, 1.9, 3.6, 4.2, None, None, "2026-11-06 08:00:00"),
          ("威廉希尔", "1x2", None, 2.0, 3.4, 3.9, None, None, "2026-11-06 09:00:00"),
          ("Bet365", "1x2", None, 1.5, 4.0, 6.0, None, None, "2026-11-06 13:00:00"),   # 比“现在”晚，不能用
          ("Bet365", "ah", -0.5, 1.95, None, 1.90, None, None, "2026-11-06 08:00:00"),
          ("Bet365", "ou", 2.75, None, None, None, 1.88, 1.98, "2026-11-06 08:00:00")]
    con.executemany("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw, away,"
                    " over, under, changed_at_utc, fetched_at_utc, raw_file) values (10, 'dongqiudi', ?, ?, ?, ?, ?, ?, ?,"
                    " ?, ?, ?, '2026-11-06 09:00:00', 'x.json.gz')", [(r[0], r[0], *r[1:]) for r in oc])
    con.execute("insert into lottery_odds values (10, 'dongqiudi', '竞彩胜平负', 0, 1.8, 3.3, 3.9, '2026-11-06 09:00:00', 'x')")
    con.commit()
    con.close()
    return db, s, tmp_path


def write_review(tmp_path: Path) -> None:
    rv = tmp_path / "review"
    (rv / "matches").mkdir(parents=True)
    (rv / "reports").mkdir()
    con = connect(tmp_path / "t.db")
    pid = {k: i for i, k in con.execute("select prediction_id, kind from predictions where match_id = 1")}
    con.close()
    rows = []
    for kind in ("T-24", "T-6", "T-1", "preview"):
        for src, p in (("模型", (0.46, 0.28, 0.26)), ("融合", (0.5, 0.27, 0.23))):
            rows.append([pid[kind], 1, "E0", 2026, "2026-11-04 15:00", kind, src, 2, 1, *p, 0.69, 0.55, 0.18, 2.1, 0.68, 0.7])
    rows.append(["", 1, "E0", 2026, "2026-11-04 15:00", "", "收盘赔率", 2, 1, 0.52, 0.26, 0.22, 0.65, 0.52, 0.17, "", "", ""])
    pd.DataFrame(rows, columns=pd.Index(["prediction_id", "match_id", "competition", "season", "kickoff_utc", "kind", "source",
                                         "home_goals", "away_goals", "p_home", "p_draw", "p_away", "logloss_1x2", "brier_1x2",
                                         "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts"])).to_csv(rv / "scores.csv", index=False)
    (rv / "matches" / "1.json").write_text(json.dumps({"误差归类": "主队进攻被低估", "三版对比": [
        {"版本": "T-24", "log loss": 0.69}, {"版本": "T-1", "log loss": 0.66}]}, ensure_ascii=False), encoding="utf-8")
    (rv / "reports" / "2026-W45.md").write_text("# 第 45 周\n\n一共 1 场。\n", encoding="utf-8")


def build(env, **kw):
    db, s, tmp = env
    out = tmp / "data" / "site"
    summary = build_site.build(s, NOW, db_path=db, out_dir=out, **kw)
    return out, summary


def read(out: Path, rel: str) -> str:
    return (out / rel).read_text(encoding="utf-8")


def plain(html: str) -> str:
    """网页里转义过的符号换回来，方便查文字。"""
    return html.replace("&#x27;", "'").replace("&amp;", "&").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")


def all_pages(out: Path) -> list[Path]:
    return sorted(out.rglob("*.html"))


# ---------- 整站 ----------
def test_build_makes_all_pages(env):
    write_review(env[2])
    out, summary = build(env)
    files = {p.relative_to(out).as_posix() for p in all_pages(out)}
    assert {"index.html", "record.html", "match/1.html", "match/10.html", "match/11.html"} <= files
    assert "match/20.html" not in files                   # 杯赛只有赛程，不做单场页
    team_files = {f for f in files if f.startswith("team/")}
    assert team_files == {"team/Liverpool.html", "team/Arsenal.html", "team/Man_City.html", "team/A_B_C.html"}
    assert summary["单场页"] == 3 and summary["球队页"] == 4 and summary["页面合计"] == 9
    assert summary["有复盘数据"] is True and summary["有预测的比赛"] == 2
    assert not (out.parent / build_site.TMP_NAME).exists() and not (out.parent / build_site.OLD_NAME).exists()


def test_no_script_no_advice_words_and_links_work(env):
    write_review(env[2])
    out, _ = build(env)
    for p in all_pages(out):
        html = p.read_text(encoding="utf-8")
        assert "<script" not in html.lower(), p
        assert "推荐" not in html, p                       # 包括“不推荐”：网页上一律不出现这个词
        assert re.fullmatch(r"[A-Za-z0-9._-]+", p.name), p.name    # 文件名都安全
        # 每个相对链接都要能点开（双击 index.html 打开也能用）
        for href in re.findall(r"href='([^']+)'", html):
            target = (p.parent / unquote(href)).resolve()
            assert target.exists(), f"{p.name} 里的链接 {href} 打不开"
    assert "不是下注建议" in read(out, "index.html")


def test_index(env):
    write_review(env[2])
    out, _ = build(env)
    html = plain(read(out, "index.html"))
    assert "利物浦 对 曼城" in html and "11 月 7 日 星期六" in html and "18:00" in html   # 10:00 UTC = 北京 18:00
    assert "开球前 24 小时" in html                       # 第 10 场最新一版是 T-24（比预览版晚出）
    assert "曼城 对 A/B: C?*" in html                     # 没有中文名的队显示英文名
    assert "还没有预测" in html                           # 第 11 场
    assert "利物浦 对 阿森纳" in html and "2–1" in html     # 最近踢完的


def test_match_page_versions_heatmap_markets(env):
    write_review(env[2])
    out, _ = build(env)
    html = plain(read(out, "match/1.html"))
    assert "赛果 2–1" in html and "最新一版：开球前 1 小时" in html
    for kind in ("开球前 24 小时", "开球前 6 小时", "开球前 1 小时", "预览版（手动出的）"):
        assert kind in html
    assert "11-01 16:00" not in html                       # 旧的预览版（11-01 08:00 UTC）不用，用 prediction_id 大的
    assert "11-04 04:00" in html                           # 新的预览版：11-03 20:00 UTC = 北京 11-04 04:00
    assert html.count("<svg") == 5                         # T-24、T-1 各两张（融合、模型），T-6 一张
    assert "融合后的比分概率" in html and "模型自己的比分概率" in html
    assert "这一版没有存比分矩阵" in html                  # 预览版没存矩阵
    assert "大小球 2.5 球（赔率主盘口）" in html and "亚盘 主让半球（-0.5）（赔率主盘口）" in html
    assert "赢全" in html and "走盘" in html and "是 " in html
    assert "最可能的 5 个比分：1–1" in html                # 1.5 对 1.1 的泊松，最可能是 1–1
    assert "猜中" in html
    assert "主队进攻被低估" in html and "收盘赔率" in html  # 复盘 JSON 和 scores.csv
    assert "还没有赔率变动记录" in html                    # 第 1 场没有赔率


def test_match_page_upcoming_odds(env):
    out, _ = build(env)
    html = plain(read(out, "match/10.html"))
    assert "还没踢" in html and "比计划晚 5 分钟" in html
    assert "还没到时间" in html                            # T-6、T-1
    assert "polyline" in html and "初盘（最早的价格）" in html
    assert "6.00" not in html                              # 比“现在”晚的价格（客胜 6.0）不能出现
    assert "竞彩胜平负" in html and "1.80 / 3.30 / 3.90" in html
    assert "还没有复盘数据" in html


def test_match_page_without_prediction(env):
    out, _ = build(env)
    html = plain(read(out, "match/11.html"))
    assert "这场还没有预测" in html and "还没有赔率变动记录" in html and "没有竞彩赔率" in html
    assert "A/B: C?*" in html and "href='../team/A_B_C.html'" in html


def test_team_page(env):
    write_review(env[2])
    out, _ = build(env)
    html = plain(read(out, "team/Arsenal.html"))
    assert "欧冠" in html and "客 对 利物浦" in html and "2–1</span> 负" in html
    assert "有预测、已踢完的 1 场" in html
    weird = plain(read(out, "team/A_B_C.html"))
    assert "接下来 14 天" in weird and "曼城" in weird and "欧冠" in weird


def test_record_page(env):
    write_review(env[2])
    out, _ = build(env)
    html = read(out, "record.html")
    assert "不分版本" in html and "收盘赔率" in html and "0.690" in html
    assert "●●● 高" in html and "●●○ 中" in html           # T-24、T-1 是 3 级，预览版是 2 级
    assert "2026-W45.md" in html and "第 45 周" in html


def test_record_and_review_missing(env):
    out, summary = build(env)
    assert summary["有复盘数据"] is False
    assert "还没有复盘数据" in read(out, "record.html")
    assert "还没有复盘数据" in read(out, "match/1.html")


def test_bad_scores_csv(env):
    rv = env[2] / "review"
    rv.mkdir()
    (rv / "scores.csv").write_text("match_id,foo\n1,2\n", encoding="utf-8")
    out, _ = build(env)
    html = read(out, "record.html")
    assert "scores.csv 缺这几列" in html and "还没有复盘数据" in html


# ---------- 小工具 ----------
def test_safe_name():
    assert safe_name("Man United") == "Man_United"
    assert safe_name("Atlético Madrid") == "Atletico_Madrid"
    assert safe_name(WEIRD) == "A_B_C"
    assert safe_name("CON") == "CON_" and safe_name("nul.x") == "nul.x_"
    assert safe_name("...") == "team" and safe_name("拜仁") == "team"
    assert len(safe_name("x" * 200)) <= 60
    files = TeamFiles(["A/B", "A:B", "a b"])              # 三个都会变成 A_B（Windows 不分大小写）
    names = [files.file(t) for t in ("A/B", "A:B", "a b")]
    assert len({n.lower() for n in names}) == 3 and all(re.fullmatch(r"[A-Za-z0-9._-]+\.html", n) for n in names)
    assert files.file("没有这队") is None


def test_matrix_reader():
    m = matrix({"k": fake_matrix()}, "k", 0.001)
    assert m.shape == (11, 11) and abs(m.sum() - 1) < 1e-12
    assert matrix({}, "k", 0.001) is None and matrix({"k": None}, "k", 0.001) is None
    assert matrix({"k": [[0.5, 0], [0, 0]]}, "k", 0.001) is None        # 加起来不是 1
    assert matrix({"k": [[1.2, -0.2], [0, 0]]}, "k", 0.001) is None      # 有负数
    assert matrix({"k": [0.5, 0.5]}, "k", 0.001) is None                 # 不是二维


def test_bad_matrix_shows_placeholder(env):
    db, s, tmp = env
    con = connect(db)
    d = scheduled_detail(1)
    d["融合比分矩阵"] = [[0.5, 0.0], [0.0, 0.0]]
    d["模型比分矩阵"] = None
    con.execute("update predictions set detail = ? where match_id = 1 and kind = 'T-1'", (json.dumps(d),))
    con.commit()
    con.close()
    out, _ = build(env)
    assert "格式不对" in read(out, "match/1.html")


# ---------- 生成过程 ----------
def test_rebuild_replaces_old_pages(env):
    out, _ = build(env)
    (out / "match" / "999.html").write_text("旧页面", encoding="utf-8")
    build(env)
    assert not (out / "match" / "999.html").exists() and (out / "index.html").exists()


def test_failed_build_keeps_old_site(env, monkeypatch):
    out, _ = build(env)
    before = read(out, "index.html")

    def boom(*a, **k):
        raise RuntimeError("生成到一半出错")
    monkeypatch.setattr(pages, "render_record", boom)
    with pytest.raises(RuntimeError):
        build(env)
    assert read(out, "index.html") == before
    assert not (out.parent / build_site.TMP_NAME).exists()


def test_main_writes_heartbeat(env, monkeypatch, capsys):
    db, s, tmp = env
    s = copy.deepcopy(s)
    s["paths"]["database"] = str(db)
    s["site"]["output_dir"] = str(tmp / "data" / "site")
    s["jobs"]["state_dir"] = str(tmp / "jobs")
    monkeypatch.setattr(build_site, "load_settings", lambda: s)
    monkeypatch.setattr("src.jobs.heartbeat.load_settings", lambda: s)
    summary = build_site.main()
    state = json.loads((tmp / "jobs" / "site.json").read_text(encoding="utf-8"))
    assert state["任务"] == "site" and state["成功"] is True and state["摘要"]["页面合计"] == summary["页面合计"]
    assert "网站生成好了" in capsys.readouterr().out
