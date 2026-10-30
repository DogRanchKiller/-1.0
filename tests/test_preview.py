"""预测页预览版（src/predict/preview.py、src/web/preview_page.py）的自动检查。

用临时数据库，不碰真实数据；模型换成一个假模型（固定给 50% / 30% / 20%），不用凑够 Elo 要的 200 场。
造的数据（现在 = 2026-11-06 12:00 UTC，英超）：
  1、2  2026-10 两场已经踢完（有比分）
  10    2026-11-07 15:00 利物浦 对 阿森纳，有 2 家公司的胜平负、亚盘、大小球
  11    2026-11-08 15:00 曼城 对 <b>坏</b>队（队名带网页符号，检查转义），没有赔率，状态是改期
  12    2026-11-06 11:00 已经开球（比现在早），不应该出现
  13    2026-11-20 15:00 超过 7 天，不应该出现
"""
import copy
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.common.odds import devig
from src.features import FeatureStore
from src.predict.preview import (add_display_info, archive, build_preview, devig_1x2, gap, main_line,
                                 market_summary, model_version, to_1x2)
from src.web.preview_page import ah_text, render

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


class FakeModel:
    """假模型：每场都给 50% / 30% / 20%。"""
    name = "fake"

    def __init__(self, p=(0.5, 0.3, 0.2)):
        self.p = list(p)
        self.fit_cutoff = None

    def fit(self, history, cutoff):
        self.fit_cutoff = cutoff
        self.n_history = len(history)

    def predict(self, targets, cutoff):
        return [list(self.p) for _ in range(len(targets))]


def odds_rows(rows):
    return pd.DataFrame(rows, columns=["bookmaker", "market", "line", "home", "draw", "away", "over", "under",
                                       "changed_at_utc"])


# ---------- 赔率 → 概率 ----------
def test_devig_1x2_average_of_books():
    o = odds_rows([["A", "1x2", None, 2.0, 3.5, 4.0, None, None, "t"],
                   ["B", "1x2", None, 2.1, 3.3, 3.8, None, None, "t"]])
    p = devig_1x2(o)
    want = (devig([2.0, 3.5, 4.0]) + devig([2.1, 3.3, 3.8])) / 2
    assert np.allclose(p, want) and abs(p.sum() - 1) < 1e-9


def test_devig_1x2_skips_bad_rows_and_empty():
    o = odds_rows([["A", "1x2", None, 2.0, 3.5, 4.0, None, None, "t"],
                   ["B", "1x2", None, 1.0, 3.3, 3.8, None, None, "t"],     # 赔率 1.0 不合理，不用
                   ["C", "1x2", None, None, 3.3, 3.8, None, None, "t"]])   # 缺一个，不用
    assert np.allclose(devig_1x2(o), devig([2.0, 3.5, 4.0]))
    assert devig_1x2(o.iloc[0:0]) is None


def test_main_line_most_common():
    o = odds_rows([["A", "ou", 2.5, None, None, None, 1.9, 1.9, "t"],
                   ["B", "ou", 3.0, None, None, None, 2.0, 1.8, "t"],
                   ["C", "ou", 3.0, None, None, None, 2.1, 1.75, "t"]])
    m = main_line(o, "over", "under")
    assert m["line"] == 3.0 and m["n_books"] == 2
    assert m["over"] == pytest.approx(2.05) and m["p_over"] + m["p_under"] == pytest.approx(1)
    want = (devig([2.0, 1.8]) + devig([2.1, 1.75])) / 2
    assert m["p_over"] == pytest.approx(want[0])


def test_main_line_tie_picks_most_balanced():
    # 两条盘口各一家：-0.5 那条两边差不多（1.95 / 1.90），-0.75 那条偏得多，选 -0.5
    o = odds_rows([["A", "ah", -0.75, 2.30, None, 1.65, None, None, "t"],
                   ["B", "ah", -0.5, 1.95, None, 1.90, None, None, "t"]])
    assert main_line(o, "home", "away")["line"] == -0.5
    assert main_line(o.iloc[0:0], "home", "away") is None


def test_gap():
    # 差：主胜 +0.05，平 +0.07，客胜 -0.12 → 差得最多的是客胜，模型更不看好
    assert gap([0.2, 0.25, 0.55], [0.15, 0.18, 0.67], 0.10) == {"outcome": 2, "diff": pytest.approx(-0.12)}
    assert gap([0.3, 0.3, 0.4], [0.32, 0.3, 0.38], 0.10) is None
    assert gap([0.5, 0.3, 0.2], [0.45, 0.3, 0.25], 0.10) is None
    assert gap(None, [0.4, 0.3, 0.3], 0.1) is None and gap([0.4, 0.3, 0.3], None, 0.1) is None


def test_to_1x2_accepts_matrix():
    m = np.zeros((3, 3))
    m[1, 0], m[1, 1], m[0, 1] = 0.5, 0.3, 0.2     # 1:0 主胜、1:1 平、0:1 客胜
    assert to_1x2(m) == pytest.approx([0.5, 0.3, 0.2])
    assert to_1x2(None) is None and to_1x2([0.4, 0.3, 0.3]) == pytest.approx([0.4, 0.3, 0.3])


def test_ah_text():
    assert ah_text(0) == "平手" and ah_text(-0.5) == "主让半球" and ah_text(0.25) == "主受让平半"
    assert ah_text(-1.75) == "主让球半两球" and ah_text(-0.75) == "主让半一"


def test_model_version_changes_with_settings():
    assert model_version("elo", {"k": 20}) == model_version("elo", {"k": 20})
    assert model_version("elo", {"k": 20}) != model_version("elo", {"k": 30})


# ---------- 整个流程（临时数据库） ----------
@pytest.fixture
def env(tmp_path):
    s = copy.deepcopy(load_settings())
    s["leagues"] = {"top": {"E0": "英超"}, "second": {}}
    con = connect(tmp_path / "t.db")
    create_tables(con)
    teams = [("Liverpool", "利物浦"), ("Arsenal", "阿森纳"), ("Man City", "曼城"), ("<b>Bad</b>", None)]
    con.executemany("insert into teams (team_id, country, name_zh) values (?, '英格兰', ?)", teams)
    m = [(1, "Liverpool", "Arsenal", "2026-10-03", "2026-10-03 14:00", "finished", "day"),
         (2, "Man City", "Liverpool", "2026-10-10", "2026-10-10 14:00", "finished", "day"),
         (10, "Liverpool", "Arsenal", "2026-11-07", "2026-11-07 15:00", "scheduled", "day"),
         (11, "Man City", "<b>Bad</b>", "2026-11-08", "2026-11-08 15:00", "postponed", "day"),
         (12, "Arsenal", "Man City", "2026-11-06", "2026-11-06 11:00", "scheduled", "day"),
         (13, "Arsenal", "Liverpool", "2026-11-20", "2026-11-20 15:00", "scheduled", "day"),
         (14, "Arsenal", "<b>Bad</b>", "2026-11-09", None, "scheduled", "round")]
    for mid, h, a, d, ko, st, prec in m:
        # round 写成比赛编号：同一赛季同一对主客队出现两次（测试里方便造），靠轮次区分
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                    " status, date_precision, round) values (?, 'E0', 2026, ?, ?, ?, ?, ?, ?, ?)",
                    (mid, h, a, d, ko, st, prec, str(mid)))
        con.execute("insert into match_sources values (?, 'football-data', ?, 'x.csv', '2026-10-20 00:00:00')",
                    (mid, f"k{mid}"))
    con.executemany("insert into match_stats (match_id, source, home_goals, away_goals) values (?, 'football-data', ?, ?)",
                    [(1, 2, 1), (2, 0, 0)])
    oc = [("Bet365", "1x2", None, 2.0, 3.5, 4.0, None, None, "2026-11-05 08:00:00"),
          ("Bet365", "1x2", None, 1.9, 3.6, 4.2, None, None, "2026-11-06 08:00:00"),   # 后变的这组才是“现在”的
          ("Bet365", "1x2", None, 1.5, 4.0, 6.0, None, None, "2026-11-06 13:00:00"),   # 比“现在”晚，不能用
          ("威廉希尔", "1x2", None, 2.0, 3.4, 3.9, None, None, "2026-11-06 09:00:00"),
          ("Bet365", "ah", -0.5, 1.95, None, 1.90, None, None, "2026-11-06 08:00:00"),
          ("威廉希尔", "ah", -0.5, 1.92, None, 1.93, None, None, "2026-11-06 08:00:00"),
          ("Bet365", "ou", 2.75, None, None, None, 1.88, 1.98, "2026-11-06 08:00:00")]
    con.executemany("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw, away,"
                    " over, under, changed_at_utc, fetched_at_utc, raw_file) values (10, 'dongqiudi', ?, ?, ?, ?, ?, ?, ?,"
                    " ?, ?, ?, '2026-11-06 09:00:00', 'x.json.gz')", [(r[0], r[0], *r[1:]) for r in oc])
    con.commit()
    return con, s


def run_preview(env, model=None):
    con, s = env
    res = build_preview(FeatureStore(con, "实时", s), s, NOW, model=model or FakeModel())
    add_display_info(con, res)
    return res


def test_build_preview_targets_and_odds(env):
    model = FakeModel()
    res = run_preview(env, model)
    assert model.fit_cutoff == "2026-11-06 12:00:00" and model.n_history == 2
    by_id = {r["match_id"]: r for r in res["rows"]}
    assert set(by_id) == {10, 11, 14}            # 已经开球的 12、超过 7 天的 13 不要
    m = by_id[10]["market"]
    assert m["p_1x2"] == pytest.approx(((devig([1.9, 3.6, 4.2]) + devig([2.0, 3.4, 3.9])) / 2).tolist())
    assert m["ah"]["line"] == -0.5 and m["ah"]["n_books"] == 2 and m["ou"]["line"] == 2.75
    assert m["last_change_utc"] == "2026-11-06 09:00:00"
    assert by_id[10]["home_zh"] == "利物浦" and by_id[11]["status"] == "postponed"
    assert by_id[11]["market"]["p_1x2"] is None and by_id[11]["gap"] is None
    assert by_id[14]["date_precision"] == "round" and by_id[14]["kickoff_utc"] is None


def test_gap_flag_uses_threshold(env):
    res = run_preview(env, FakeModel((0.75, 0.15, 0.10)))     # 赔率主胜约 51%，差 20 多个百分点
    g = {r["match_id"]: r["gap"] for r in res["rows"]}
    assert g[10]["outcome"] == 0 and g[10]["diff"] > 0.1
    res = run_preview(env, FakeModel((0.52, 0.27, 0.21)))
    assert {r["match_id"]: r["gap"] for r in res["rows"]}[10] is None


def test_archive_insert_only_and_dedup(env):
    con, _s = env
    res = run_preview(env)
    assert archive(con, res, NOW, "abc1234") == {"新存": 3, "没变化": 0}
    assert archive(con, res, NOW, "abc1234") == {"新存": 0, "没变化": 3}     # 一样的不重复存
    first = con.execute("select prediction_id, p_home from predictions order by prediction_id").fetchall()
    res2 = run_preview(env, FakeModel((0.6, 0.25, 0.15)))
    assert archive(con, res2, NOW, "abc1234") == {"新存": 3, "没变化": 0}    # 概率变了：加新行
    rows = con.execute("select prediction_id, p_home from predictions order by prediction_id").fetchall()
    assert rows[:3] == first and len(rows) == 6                              # 旧的行原样留着
    d = json.loads(con.execute("select detail from predictions where match_id = 10 limit 1").fetchone()[0])
    assert d["代码版本"] == "abc1234" and d["赔率公司数"] == 2 and d["模型设置"]["k"] == 20


def test_archive_skips_rows_without_model(env):
    con, _s = env
    res = run_preview(env)
    for r in res["rows"]:
        r["model_p"] = None
    assert archive(con, res, NOW) == {"新存": 0, "没变化": 0}


def test_render_escapes_and_labels(env):
    _con, s = env
    page = render(run_preview(env), s, NOW)
    assert "<b>Bad</b>" not in page and "&lt;b&gt;Bad&lt;/b&gt;" in page   # 没有中文名的显示英文，而且转义了
    assert "临时模型" in page and "不是下注建议" in page and "改期" in page
    assert "时间待定" in page and "还没有赔率" in page and "主让半球" in page
    assert "推荐下注" not in page and "建议买" not in page
    assert "11 月 7 日 星期六" in page                                       # 2026-11-07 15:00 UTC = 北京 23:00
