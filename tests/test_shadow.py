"""xG 研究第 2 段（需求文档第 4 节“xG 研究第 2 段怎么做”）的自动检查：

  - gbdt 的 xg_source = match_stats：用比赛统计的 home_xg / away_xg 更新 dyn，A版输入能用；和 Understat 版喂同样的数结果一样；
  - 登记表：gb_xg 的设置对，gb_l7、gb_xg_r 的模型版本指纹没变；
  - 定时预测入口：存了 T-24 才出影子预测，截止时间和正式那行一样，不进 predictions；再跑一次不重复存；
    影子模型出错时定时任务照样成功、正式预测照常；正式模型换成假模型又没给影子模型时不出；shadow.enabled = false 时不出；
  - 对比报告：配对、分数和手算的一样，场数不够只列数。
用模拟比赛和临时数据库，不碰真实数据库。
"""
import json
import sqlite3

import numpy as np
import pytest

from src.models.registry import build_model, model_config
from src.predict.preview import model_version
from tests.test_adaptive import promo_history
from tests.test_gbdt import gb
from tests.test_gbdt_xg import CUT, same_dyn, with_xg
from tests.test_scheduled import FakeModel, _now, job_env  # noqa: F401 —— job_env 是 pytest 的 fixture


# ---------------- 模型：xg_source ----------------

def test_match_stats_source_same_as_understat_with_same_numbers():
    """同样的 xG 数，放在 home_xg / away_xg（match_stats）和放在 us_ 两列（understat）结果一样；A版输入能用。"""
    h = with_xg(promo_history(), shift=0.8)
    ms = h.rename(columns={"us_home_xg": "home_xg", "us_away_xg": "away_xg"})
    a = gb(usage="研究实验", xg_mix=0.5)
    b = gb(xg_mix=0.5, xg_source="match_stats")                  # 用途是默认的 A版输入
    a.update(h, CUT)
    b.update(ms, CUT)
    assert same_dyn(a, b) and np.array_equal(np.vstack(a.rows), np.vstack(b.rows), equal_nan=True)
    assert b.data == ("比分", "射门、xG")                          # 不加 Understat 那一类
    c = gb()                                                     # 不写 xg_mix：多给 home_xg 也不用
    c.update(ms, CUT)
    assert not same_dyn(b, c)


def test_xg_source_guards():
    with pytest.raises(ValueError):
        gb(xg_mix=0.5, xg_source="whoscored")
    with pytest.raises(ValueError):
        gb(xg_mix=0.5, xg_source="understat")                    # Understat 还是只许研究实验


def test_registry_gb_xg_and_fingerprints_unchanged():
    base, x = model_config("gb_l7"), model_config("gb_xg")
    assert {k: v for k, v in x.items() if k not in ("xg_mix", "xg_source")} == base
    assert x["xg_mix"] == 0.5 and x["xg_source"] == "match_stats" and "usage" not in x
    m = build_model("gb_xg")
    assert m.usage == "A版输入" and m.xg_cols == ["home_xg", "away_xg"]
    # 上线的 gb_l7、第 1 段的 gb_xg_r：设置一个数都没变（指纹和合并前一样）
    assert model_version("gb_l7", base) == "gb_l7-f62b5466"
    assert "xg_source" not in model_config("gb_xg_r")


# ---------------- 定时预测入口：影子预测 ----------------

class FakeShadow(FakeModel):
    """影子用的假模型：给 40% / 30% / 30%，记下拟合时的截止时间。"""
    name = "fake_xg"

    def __init__(self):
        self.cutoffs = []

    def fit(self, history, cutoff):
        self.cutoffs.append(cutoff)

    def predict(self, targets, cutoff):
        return [[0.4, 0.3, 0.3] for _ in range(len(targets))]


class BrokenShadow(FakeShadow):
    def fit(self, history, cutoff):
        raise RuntimeError("故意出错")


def _rows(db, sql):
    con = sqlite3.connect(db)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_shadow_stored_with_same_cutoff_and_not_in_predictions(job_env):
    from src.jobs.heartbeat import read_state
    from src.jobs.scheduled_predict import JOB_NAME, run_job

    s, db, folder = job_env
    sh = FakeShadow()
    run_job(s, _now(), db_path=db, folder=folder, model=FakeModel(), code="test", shadow_models={"fake_xg": sh})
    st = read_state(JOB_NAME, folder)
    assert st["成功"] is True and st["摘要"]["影子预测"]["新存"] == 1
    main = _rows(db, "select prediction_id, model, cutoff_utc from predictions")
    assert len(main) == 1 and main[0][1] != "fake_xg"            # predictions 里只有正式那行
    shadow = _rows(db, "select main_prediction_id, model, kind, cutoff_utc, p_home, p_draw, p_away, detail"
                       " from shadow_predictions")
    assert len(shadow) == 1
    mid, model, kind, cutoff, *p, detail = shadow[0]
    assert (mid, model, kind, cutoff) == (main[0][0], "fake_xg", "T-24", main[0][2])
    assert sh.cutoffs == [main[0][2]] and p == [0.4, 0.3, 0.3]
    assert json.loads(detail)["其中有xG的场数"] == 0

    # 再跑一次：正式预测“已出过”，影子也不重复
    run_job(s, _now(), db_path=db, folder=folder, model=FakeModel(), code="test", shadow_models={"fake_xg": sh})
    assert _rows(db, "select count(*) from shadow_predictions")[0][0] == 1
    assert sh.cutoffs == [main[0][2]]                            # 没有新存的正式 T-24，就不拟合


def test_shadow_error_does_not_break_main_job(job_env):
    from src.jobs.heartbeat import read_state
    from src.jobs.scheduled_predict import JOB_NAME, run_job

    s, db, folder = job_env
    run_job(s, _now(), db_path=db, folder=folder, model=FakeModel(), code="test",
            shadow_models={"fake_xg": BrokenShadow()})
    st = read_state(JOB_NAME, folder)
    assert st["成功"] is True and st["错误"] is None and st["摘要"]["新存"] == 1
    assert "故意出错" in st["摘要"]["影子预测"]["错误"]
    assert _rows(db, "select count(*) from predictions")[0][0] == 1
    assert _rows(db, "select count(*) from shadow_predictions")[0][0] == 0


def test_shadow_skipped_for_fake_main_model_or_disabled(job_env):
    from src.jobs.heartbeat import read_state
    from src.jobs.scheduled_predict import JOB_NAME, run_job

    s, db, folder = job_env
    run_job(s, _now(), db_path=db, folder=folder, model=FakeModel(), code="test")
    assert "测试" in read_state(JOB_NAME, folder)["摘要"]["影子预测"]["说明"]
    s["shadow"]["enabled"] = False
    run_job(s, _now(), db_path=db, folder=folder, model=FakeModel(), code="test", shadow_models={"x": FakeShadow()})
    assert "没开" in read_state(JOB_NAME, folder)["摘要"]["影子预测"]["说明"]
    assert _rows(db, "select count(*) from shadow_predictions")[0][0] == 0


def test_shadow_only_follows_t24(job_env):
    """正式预测这次存的是 T-6（不是 T-24）：影子不出。"""
    from datetime import timedelta

    from src.jobs.heartbeat import read_state
    from src.jobs.scheduled_predict import JOB_NAME, run_job

    s, db, folder = job_env
    sh = FakeShadow()
    run_job(s, _now() + timedelta(hours=7), db_path=db, folder=folder, model=FakeModel(), code="test",
            shadow_models={"fake_xg": sh})                       # 开球前 5 小时：只出 T-6
    st = read_state(JOB_NAME, folder)
    assert st["摘要"]["T-6"] == 1 and st["摘要"]["影子预测"]["新存"] == 0 and sh.cutoffs == []


# ---------------- 对比报告 ----------------

def test_xg_shadow_report(job_env):
    from src.common.db import connect
    from src.evaluate.xg_shadow import build, scores
    from src.features import FeatureStore, open_readonly
    from src.jobs.scheduled_predict import run_job

    s, db, folder = job_env
    run_job(s, _now(), db_path=db, folder=folder, model=FakeModel(), code="test",
            shadow_models={"fake_xg": FakeShadow()})
    con = connect(db)                                            # 比赛 10 踢完：主队 1-0
    con.execute("update matches set status = 'finished' where match_id = 10")
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                " values (10, 'football-data', 1, 0, 'H')")
    con.commit()
    con.close()
    ro = open_readonly(db)
    try:
        rows, summary = build(FeatureStore(ro, "实时", s), ro, s)
    finally:
        ro.close()
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["差 胜平负 log loss"] == pytest.approx(-np.log(0.4) + np.log(0.5))
    assert np.isnan(r["影子 比分 log loss"])                    # 假模型没给比分矩阵
    main = summary[summary["主比较"]].iloc[0]
    assert main["场数"] == 1 and "场数不够" in main["结论"]
    # 手算一场：[0.5, 0.3, 0.2]、主胜 → Brier 0.38、RPS 0.145（metrics.py 里的例子）
    m = np.full((3, 3), 1 / 9)
    sc = scores([0.5, 0.3, 0.2], m, 2, 0, 1e-15)
    assert sc["Brier"] == pytest.approx(0.38) and sc["RPS"] == pytest.approx(0.145)
    assert sc["比分 log loss"] == pytest.approx(np.log(9))
    assert sc["大 2.5 球 Brier"] == pytest.approx((3 / 9) ** 2)     # 2-0 没过 2.5，矩阵里 i + j ≥ 3 的有 3 格
    assert sc["双方进球 Brier"] == pytest.approx((4 / 9) ** 2)      # 2-0 不是双方进球，i ≥ 1 且 j ≥ 1 的有 4 格


def _pair_row(shadow, main, day, s_ll, m_ll, s_home, m_home):
    """xg_shadow.match_rows 那样的一行（只填汇总要用的列）。"""
    from src.evaluate.xg_shadow import SCORES
    r = {"影子模型": shadow, "正式模型": main, "比赛日期": day, "影子主胜": s_home, "正式主胜": m_home}
    for k in SCORES:
        s, m = (s_ll, m_ll) if k == SCORES[0] else (np.nan, np.nan)
        r[f"影子 {k}"], r[f"正式 {k}"], r[f"差 {k}"] = s, m, s - m
    return r


def test_xg_shadow_pools_across_model_switch(job_env):
    """换模型前后两段：方向固定成 新 − 旧（先当影子的 gb_xg − gb_l7），换之后那段差取反，两段合起来计数。"""
    import pandas as pd
    from src.evaluate.xg_shadow import model_pairs, summarize

    s, _, _ = job_env
    s["shadow"]["min_matches"] = 3
    s["preview"]["model"], s["shadow"]["models"] = "gb_xg", ["gb_l7"]          # 换之后的设置
    # 换之前 2 场：影子 gb_xg、正式 gb_l7；换之后 1 场：影子 gb_l7、正式 gb_xg
    rows = pd.DataFrame([_pair_row("gb_xg", "gb_l7", "2026-11-01", 0.9, 1.0, 0.45, 0.40),
                         _pair_row("gb_xg", "gb_l7", "2026-11-02", 1.1, 1.0, 0.50, 0.52),
                         _pair_row("gb_l7", "gb_xg", "2026-11-09", 1.3, 1.0, 0.30, 0.33)])
    stored = pd.DataFrame({"shadow_id": [3, 1, 2], "shadow_model": ["gb_l7", "gb_xg", "gb_xg"],
                           "main_model": ["gb_xg", "gb_l7", "gb_l7"]})
    pairs = model_pairs(stored, s)
    assert pairs == [("gb_xg", "gb_l7")]                     # 最早的影子是 gb_xg：方向不因后来换模型而反
    main = summarize(rows, s, pairs)
    main = main[main["主比较"]]
    assert len(main) == 1                                     # 一组，不分成两组
    r = main.iloc[0]
    assert (r["新模型"], r["旧模型"]) == ("gb_xg", "gb_l7")
    assert r["场数"] == 3 and r["新当影子的场"] == 2 and r["新当正式的场"] == 1
    # 新 − 旧：−0.1、+0.1、换之后 gb_xg 1.0 − gb_l7 1.3 = −0.3（影子 − 正式 是 +0.3，取反）
    assert r["差的平均"] == pytest.approx(-0.1)
    assert r["新平均分"] == pytest.approx((0.9 + 1.1 + 1.0) / 3)
    assert r["旧平均分"] == pytest.approx((1.0 + 1.0 + 1.3) / 3)
    assert r["主胜概率平均绝对差"] == pytest.approx((0.05 + 0.02 + 0.03) / 3)
    assert not np.isnan(r["区间下限"])                         # 够 3 场才算区间：合起来才够
    # 只有换之前那 2 场时不够 3 场
    only_before = summarize(rows.iloc[:2], s, pairs)
    assert "场数不够" in only_before[only_before["主比较"]].iloc[0]["结论"]
    # 还没有任何影子预测：按设置配（影子模型, 上线模型）
    assert model_pairs(pd.DataFrame(), s) == [("gb_l7", "gb_xg")]
