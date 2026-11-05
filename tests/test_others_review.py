"""调研别人的项目（src/collect/others_review.py、src/evaluate/others_demo.py）的检查。不连网、不读真回测。"""
import numpy as np
import pandas as pd
import pytest

from src.collect import others_review as orv
from src.common.config import load_settings
from src.common.markets import poisson_score_matrix
from src.evaluate import others_demo as od


def test_abstract_rebuilt_in_order():
    """OpenAlex 的倒排摘要按位置拼回原文；没有摘要是空字符串。"""
    inv = {"predict": [1], "We": [0], "football": [2, 4], "and": [3]}
    assert orv.abstract(inv) == "We predict football and football"
    assert orv.abstract(None) == ""


def test_arxiv_id_found_in_locations():
    w = {"locations": [{"landing_page_url": "https://doi.org/x"}, {"pdf_url": "https://arxiv.org/pdf/2101.01234v2"}]}
    assert orv._arxiv_id(w) == "2101.01234"
    assert orv._arxiv_id({"locations": [None, {"landing_page_url": None}]}) is None


def test_get_waits_retry_after(monkeypatch):
    """429 时按网站的 Retry-After 多等 5 秒再试；成功就返回。"""
    slept, calls = [], []

    class R:
        def __init__(self, code, ra=None):
            self.status_code, self.headers = code, ({"Retry-After": ra} if ra else {})

    seq = [R(429, "7"), R(200)]
    monkeypatch.setattr(orv.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(orv.requests, "get", lambda *a, **k: calls.append(1) or seq.pop(0))
    orv._last.clear()
    cfg = {"min_interval_seconds": {"x": 0}, "retry_waits": [3, 9], "user_agent": "t"}
    assert orv._get("u", "x", cfg).status_code == 200
    assert len(calls) == 2 and slept[-1] == 12      # max(3, 7 + 5)


def test_demo_rows():
    """检验季的比赛才算；只挑最高的那几成场数对；双选、固定线的命中率按比分核对。"""
    s = load_settings()
    cut = int(s["pick_tiers_study"]["cut_last"])
    rng = np.random.default_rng(3)
    n = 400
    pairs = [(float(rng.uniform(0.5, 2.8)), float(rng.uniform(0.4, 1.8))) for _ in range(n)]
    mats = np.stack([poisson_score_matrix(h, a, 10) for h, a in pairs])
    mats /= mats.sum(axis=(1, 2), keepdims=True)
    hg, ag = rng.poisson([p[0] for p in pairs]), rng.poisson([p[1] for p in pairs])
    t = pd.DataFrame({"season": np.where(np.arange(n) < 100, cut, cut + 1), "home_goals": hg, "away_goals": ag})
    rows = {r["问题"]: r for r in od.demo(t, mats, s)}
    test = np.arange(n) >= 100
    assert rows["胜平负，全部场次"]["场数"] == 300
    assert rows["胜平负，只挑把握分最高的 10%"]["场数"] == 30
    # 固定 2.5 球：按矩阵算大球概率，猜大的一边，和比分对
    i, j = np.indices(mats.shape[1:])
    over = (mats[test] * (i + j > 2.5)).sum((1, 2))
    tot = hg[test] + ag[test]
    expect = np.where(over >= 0.5, tot > 2.5, tot < 2.5).mean()
    assert rows["大小球固定 2.5 球（猜概率大的一边）"]["命中率"] == pytest.approx(expect)
    # 双选一定不比单选差
    assert rows["双选：猜不输（主胜或平 / 客胜或平）"]["命中率"] >= rows["胜平负，全部场次"]["命中率"]
