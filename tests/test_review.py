"""tests/test_review.py
测试 src/review/reconcile.py 里的纯函数：clean_probs、clean_matrix、
score_metrics、over25_logloss、score_rows（列名检查）、kind_rank。
不连数据库，不读文件。
"""
import numpy as np
import pandas as pd
import pytest

from src.review.reconcile import (
    SCORE_COLUMNS,
    SOURCE_MODEL,
    clean_matrix,
    clean_probs,
    kind_rank,
    over25_logloss,
    score_metrics,
    score_rows,
)

TOL = 0.001   # review.prob_sum_tol 的默认值


# ======================= clean_probs =======================
class TestCleanProbs:
    def test_normal(self):
        """三个概率加起来差一点点（四位小数存档），应该按比例补齐、不返回 None。"""
        arr = clean_probs([0.33, 0.34, 0.33], TOL)
        assert arr is not None
        assert abs(arr.sum() - 1.0) < 1e-9

    def test_none_input(self):
        """输入 None → 直接返回 None。"""
        assert clean_probs(None, TOL) is None

    def test_negative(self):
        """有负数 → 当作坏数据，返回 None。"""
        assert clean_probs([0.5, -0.1, 0.6], TOL) is None

    def test_sum_too_far(self):
        """三个概率加起来 1.5，和 1 差超过 tol → 返回 None。"""
        assert clean_probs([0.5, 0.5, 0.5], TOL) is None


# ======================= clean_matrix =======================
class TestCleanMatrix:
    def _uniform_11x11(self):
        """构造一个合法的 11×11 矩阵，每格均等、加起来是 1。"""
        m = np.ones((11, 11)) / 121.0
        return m.tolist()   # 传列表，模拟存档里读出来的 JSON

    def test_normal_11x11(self):
        arr = clean_matrix(self._uniform_11x11(), TOL)
        assert arr is not None
        assert arr.shape == (11, 11)
        assert abs(arr.sum() - 1.0) < 1e-9

    def test_not_2d(self):
        """一维列表 → 不是二维 → 返回 None。"""
        assert clean_matrix([1.0 / 11] * 11, TOL) is None

    def test_has_none(self):
        """矩阵里有 None → np.asarray 转 float 后出现 NaN → 返回 None。"""
        m = [[None] + [1.0 / 121] * 10] + [[1.0 / 121] * 11] * 10
        assert clean_matrix(m, TOL) is None


# ======================= score_metrics =======================
class TestScoreMetrics:
    def _make_matrix(self):
        """简单的矩阵：只在 (2,1) 格（主队 2 球、客队 1 球）放概率，其他格均分余下部分。"""
        m = np.full((11, 11), 0.01 / 120)
        m[2, 1] = 0.99
        return m

    def test_2_1_logloss_not_none(self):
        """2-1 的比赛，传入合法概率和矩阵，logloss_1x2 和 logloss_score 都不是 None。"""
        p = np.array([0.50, 0.30, 0.20])
        mat = self._make_matrix()
        out = score_metrics(p, mat, hg=2, ag=1)
        assert out["logloss_1x2"] is not None
        assert out["logloss_score"] is not None

    def test_no_matrix_gives_none_for_score(self):
        """没有矩阵时，logloss_score / logloss_over25 / logloss_btts 留 None，胜平负三个分数正常。"""
        p = np.array([0.50, 0.30, 0.20])
        out = score_metrics(p, None, hg=2, ag=1)
        assert out["logloss_1x2"] is not None
        assert out["logloss_score"] is None
        assert out["logloss_over25"] is None
        assert out["logloss_btts"] is None


# ======================= over25_logloss =======================
class TestOver25Logloss:
    def test_with_value(self):
        """给个合法概率，应该返回一个正的浮点数。"""
        val = over25_logloss(0.6, hg=2, ag=1)   # 2+1=3 > 2.5，实际是大球
        assert val is not None
        assert isinstance(val, float)
        assert val > 0

    def test_none_input(self):
        """p_over 是 None → 直接返回 None。"""
        assert over25_logloss(None, hg=2, ag=1) is None


# ======================= score_rows =======================
class TestScoreRows:
    def test_empty_matches_columns(self):
        """空列表输入时，返回的 DataFrame 列名应该和 SCORE_COLUMNS 完全一致。"""
        settings = {"review": {"kinds": ["T-1", "preview"]}}
        df = score_rows([], settings)
        assert list(df.columns) == SCORE_COLUMNS

    def test_empty_matches_no_rows(self):
        """空列表输入时，不应该有任何行。"""
        settings = {"review": {"kinds": ["T-1", "preview"]}}
        df = score_rows([], settings)
        assert len(df) == 0


# ======================= kind_rank =======================
class TestKindRank:
    KINDS = ["T-24", "T-6", "T-1", "preview"]

    def test_t1_before_preview(self):
        """T-1 的排名（index）应该小于 preview 的排名。"""
        assert kind_rank("T-1", self.KINDS) < kind_rank("preview", self.KINDS)

    def test_t24_first(self):
        """T-24 排在最前面，rank = 0。"""
        assert kind_rank("T-24", self.KINDS) == 0

    def test_unknown_kind_last(self):
        """不在列表里的 kind 排在最后（index = len(kinds)）。"""
        assert kind_rank("手动版", self.KINDS) == len(self.KINDS)
