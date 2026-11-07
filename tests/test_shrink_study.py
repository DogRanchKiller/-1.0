'''复盘问题回测 src/evaluate/shrink_study.py 的检查（需求文档第 15 节“复盘问题回测”）。

都用小的假数据（写在 tmp_path），不碰 data/。
检查：结果换成赢了多少和权重、弱队是哪边、回归找回已知的斜率、一直往弱队偏能查出来、不偏的不会误报、结论按两条写、考试赛季读不进来、整个流程能跑通。
'''
import copy

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.evaluate import shrink_study as ss


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s["compare"]["n_boot"] = 50
    return s


def _table(path, n=6000, bias=0.05, season=2020, seed=1):
    '''假的每场表，像平博的线：弱队那边的真实胜算 td 在 50% 左右，市场 m 就是 td；
    模型在弱队那边多给 bias（bias > 0 就是“把差距估小了”，一直往弱队偏）。'''
    rng = np.random.default_rng(seed)
    td = rng.uniform(0.44, 0.56, n)
    qd = np.clip(td + bias + rng.normal(0, 0.03, n), 0.02, 0.98)
    dog_home = rng.random(n) < 0.3
    market = np.where(np.arange(n) % 2 == 0, "ah", "ou")
    is_ah = market == "ah"
    dog_is_a = is_ah & dog_home              # 大小球的弱队是小球，永远不是 a 那边（大球）
    dog_won = rng.random(n) < td
    d = pd.DataFrame({
        "match_id": np.arange(n), "group": "五大联赛", "competition": "E0", "season": season,
        "match_date": pd.date_range("2020-08-01", periods=n, freq="6h").strftime("%Y-%m-%d"),
        "market": market, "line": np.where(is_ah, np.where(dog_home, 0.75, -0.75), 2.5),
        "m": np.where(dog_is_a, td, 1 - td), "p": np.where(dog_is_a, qd, 1 - qd),
        "out_a": np.where(dog_won == dog_is_a, "win", "loss")})
    d.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_outcome_weights_and_dog_side(tmp_path, settings):
    d = pd.DataFrame({"match_id": [1, 2, 3, 4], "group": "五大联赛", "competition": "E0", "season": 2020,
                      "match_date": "2020-09-01", "market": ["ah", "ah", "ah", "ou"], "line": [0.5, -1.0, 0.0, 2.5],
                      "p": [0.6, 0.3, 0.5, 0.4], "m": [0.5, 0.5, 0.5, 0.5],
                      "out_a": ["half_win", "loss", "push", "half_loss"]})
    path = tmp_path / "pm.csv"
    d.to_csv(path, index=False, encoding="utf-8-sig")
    r = ss.load_rows(path, settings)
    assert r["y"].tolist() == [1, 0, 0, 0] and r["w"].tolist() == [0.5, 1, 0, 0.5]
    # 第 1 场主队受让 → 弱队是主队；第 2 场主队让球 → 弱队是客队；平手盘没有弱队；大小球的弱队是小球
    assert r.loc[0, "q"] == pytest.approx(0.6) and r.loc[0, "yd"] == 1
    assert r.loc[1, "q"] == pytest.approx(0.7) and r.loc[1, "yd"] == 1
    assert np.isnan(r.loc[2, "q"])
    assert r.loc[3, "q"] == pytest.approx(0.6) and r.loc[3, "yd"] == 1


def test_fit_slope_recovers_known_value():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.2 + 1.5 * x)))).astype(float)
    a, b = ss.fit_slope(x, y, np.ones_like(x))
    assert a == pytest.approx(0.2, abs=0.05) and b == pytest.approx(1.5, abs=0.08)


def test_verdict_rules():
    assert ss.verdict(-0.01, -0.02) == ss.BOTH
    assert ss.verdict(-0.01, 0.01) == ss.ONLY_ALL and ss.verdict(0.01, -0.01) == ss.ONLY_PICKED
    assert ss.verdict(0.01, 0.02) == ss.NONE


def test_exam_season_refused(tmp_path, settings):
    path = _table(tmp_path / "pm.csv", n=40, season=2024)
    with pytest.raises(ValueError, match="不是开发赛季"):
        ss.load_rows(path, settings)


def test_dog_bias_detected_and_market_clean(tmp_path, settings):
    d = ss.load_rows(_table(tmp_path / "pm.csv"), settings)
    res = ss.analyze(d, settings)
    for market in ["亚盘", "大小球"]:
        r = res[res["玩法"] == market].set_index("项目")
        assert r.loc["弱队那边整体：实际 − 模型", "值"] == pytest.approx(-0.05, abs=0.02)
        assert abs(r.loc["弱队那边整体：实际 − 平博（参照）", "值"]) < 0.025
        assert r.loc["挑弱队时：实际 − 模型", "上限"] < 0
        assert ss.verdicts(res, "五大联赛")[market] == ss.BOTH


def test_unbiased_model_not_flagged(tmp_path, settings):
    # 不偏、只有误差的模型：弱队那边整体的差在 0 附近；挑弱队时的差偏负（自己挑的那边偏乐观），所以不能只看判断 2
    d = ss.load_rows(_table(tmp_path / "pm.csv", bias=0.0, seed=4), settings)
    res = ss.analyze(d, settings)
    verd = ss.verdicts(res, "五大联赛")
    assert verd["亚盘"] in (ss.NONE, ss.ONLY_PICKED) and verd["大小球"] in (ss.NONE, ss.ONLY_PICKED)
    r = res[res["玩法"] == "亚盘"].set_index("项目")
    assert r.loc["挑弱队时：实际 − 模型", "值"] < 0


def test_main_writes_report(tmp_path, settings, monkeypatch):
    path = _table(tmp_path / "pm.csv", n=400)
    settings["shrink_study"] = {**settings["shrink_study"], "per_match": str(path), "output_dir": str(tmp_path / "out")}
    monkeypatch.setattr(ss, "load_settings", lambda: settings)
    ss.main()
    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## 结论（只看五大联赛）" in text and "亚盘" in text
    assert (tmp_path / "out" / "results.csv").exists()
