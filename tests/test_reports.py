"""tests/test_reports.py
测试 src/review/reports.py 里的纯函数：period_keys、period_range、
average_table（空数据）、build_report（样本太少）。
不连数据库，不读文件。
"""
import pandas as pd
import pytest

from src.review.reconcile import SCORE_COLUMNS, SOURCE_MODEL
from src.review.reports import (
    average_table,
    build_report,
    period_keys,
    period_range,
)


# ======================= period_keys =======================
class TestPeriodKeys:
    def test_format(self):
        """2026-10-26 是周一，应该落在 ISO 第 44 周、10 月。
        返回 ("2026-W44", "2026-10")。"""
        day = pd.Timestamp("2026-10-26")
        w, m = period_keys(day)
        assert w == "2026-W44"
        assert m == "2026-10"

    def test_types(self):
        """返回值类型是字符串。"""
        w, m = period_keys(pd.Timestamp("2026-11-01"))
        assert isinstance(w, str)
        assert isinstance(m, str)
        # 月键格式 YYYY-MM
        assert len(m) == 7 and m[4] == "-"
        # 周键格式 YYYY-Www
        assert "-W" in w


# ======================= period_range =======================
class TestPeriodRangeWeek:
    def test_w44_start_end(self):
        """2026-W44：周一 2026-10-26 到周日 2026-11-01。"""
        start, end = period_range("周", "2026-W44")
        assert start == pd.Timestamp("2026-10-26")
        assert end == pd.Timestamp("2026-11-01")


# ======================= average_table =======================
class TestAverageTableEmpty:
    def test_empty_returns_no_data(self):
        """空 DataFrame → 返回 ["（没有数据）"]。"""
        empty = pd.DataFrame(columns=SCORE_COLUMNS)
        out = average_table(empty, ["T-1", "preview"])
        assert out == ["（没有数据）"]


# ======================= build_report（样本太少） =======================
def _make_review(mid, date, comp, home, away, result, hg, ag, kind="T-1", ll=1.0):
    """造一个最小的复盘字典，够 build_report 用。"""
    return {
        "match_id": mid,
        "联赛": comp,
        "比赛日期": date,
        "主队": home, "客队": away,
        "主队中文": None, "客队中文": None,
        "实际": {"比分": [hg, ag], "结果": result},
        "正式版本": kind,
        "各版本": [{
            "版本": kind,
            "prediction_id": mid * 100,
            SOURCE_MODEL: {"logloss_1x2": ll, "概率": [0.4, 0.3, 0.3]},
        }],
        "误差归类": {"类型": "预测偏差型"},
        "开球时间_utc": None,
    }


def _make_scores(reviews):
    """根据 reviews 造一个 scores DataFrame，只有模型行。"""
    rows = []
    for r in reviews:
        v = r["各版本"][0]
        rows.append({
            "prediction_id": v["prediction_id"], "match_id": r["match_id"],
            "competition": r["联赛"], "season": 2026, "kickoff_utc": None,
            "kind": v["版本"], "source": SOURCE_MODEL,
            "home_goals": r["实际"]["比分"][0], "away_goals": r["实际"]["比分"][1],
            "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3,
            "logloss_1x2": v[SOURCE_MODEL]["logloss_1x2"],
            "brier_1x2": 0.2, "rps_1x2": 0.1,
            "logloss_score": None, "logloss_over25": None, "logloss_btts": None,
        })
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


class TestBuildReportSampleSmall:
    def test_sample_too_small(self):
        """2 场（少于 min_matches=300），报告里应该出现"样本太少"。"""
        reviews = [
            _make_review(1, "2026-10-26", "英超", "A队", "B队", "主胜", 2, 1, ll=0.8),
            _make_review(2, "2026-10-27", "西甲", "C队", "D队", "平局", 1, 1, ll=0.9),
        ]
        sc = _make_scores(reviews)
        settings = {
            "review": {
                "kinds": ["T-1", "preview"],
                "official_order": ["T-1", "preview"],
                "errors": {"miss_below": 0.334, "close_move": 0.05,
                           "direction_min": 0.01, "xg_margin": 0.5},
                "reports": {"min_matches": 300, "top_misses": 5},
            }
        }
        now_utc = pd.Timestamp("2026-11-02 00:00", tz="UTC")
        text = build_report("周", "2026-W44", reviews, sc, settings, now_utc)
        assert "样本太少" in text
