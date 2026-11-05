"""复现 Yeung 等 2024 深度学习的自动检查（src/models/yeung_nn.py、src/evaluate/yeung_repro.py）。

全部用编出来的小数据，不碰数据库：
- 小零件算得对不对（RPS、位置标记）；
- “最近 5 场”只用赛前的比赛（同一时刻开球的几场互相看不到比分）；
- 训练之前不出预测，训练之后出的是合法的三个概率，没见过的队也能出；
- 隔多久重新训练；早于 first_season 的比赛不进训练表；
- 评分赛季碰到考试赛季就停下；两次回测的比赛信息对不上就停下。
"""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.common.config import load_settings
from src.evaluate import yeung_repro
from src.models.yeung_nn import YeungModel, positional_encoding, rps_loss


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def small_cfg(settings, **over):
    """原设置，只把训练调小（跑得快），其余照旧。"""
    cfg = copy.deepcopy(settings["repro_yeung"])
    cfg.update({"epochs": 2, "batch_size": 10, "first_season": 2005}, **over)
    return cfg


def match(mid, ko, home, away, hg, ag, season=2010):
    res = "H" if hg > ag else ("A" if hg < ag else "D")
    return {"match_id": mid, "kickoff_earliest_utc": ko, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "result": res, "season": season}


def league(n_weeks=40, start="2010-08-01", season=2010):
    """6 支队，每周 3 场（同一时刻开球），比分随便编（固定种子）。"""
    rng = np.random.default_rng(1)
    teams = list("ABCDEF")
    rows, mid = [], 1
    for w in range(n_weeks):
        ko = (pd.Timestamp(start) + pd.Timedelta(days=7 * w)).strftime("%Y-%m-%d %H:%M:%S")
        order = rng.permutation(teams)
        for k in range(3):
            rows.append(match(mid, ko, order[2 * k], order[2 * k + 1], int(rng.integers(0, 4)), int(rng.integers(0, 3)), season))
            mid += 1
    return pd.DataFrame(rows)


def test_rps_loss_by_hand():
    p = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = torch.tensor([0, 2])
    # 第 1 场主胜：0.5×((0.5−1)² + (0.8−1)²) = 0.145；第 2 场客胜：0.5×(0.2² + 0.5²) = 0.145
    assert rps_loss(p, y).item() == pytest.approx(0.145)


def test_positional_encoding_values():
    pe = positional_encoding(3, 5)
    assert pe.shape == (3, 5, 1)
    # 第 0 个样本：双数场 sin(0)=0，单数场 cos(0)=1
    assert pe[0, :, 0].tolist() == pytest.approx([0, 1, 0, 1, 0])
    # 第 2 个样本第 1 场：cos(2 / 100^(2/5))
    assert pe[2, 1, 0].item() == pytest.approx(np.cos(2 / 100 ** 0.4), abs=1e-6)
    # 出预测时 pos 都是 0
    assert torch.equal(positional_encoding(3, 5, pos0_only=True)[2], pe[0])


def test_strength_uses_only_earlier_matches(settings):
    m = YeungModel("y", small_cfg(settings), 0)
    rows = [match(i, f"2010-08-{i:02d} 15:00:00", "A", "B", 1, 0) for i in range(1, 7)]
    m._process(pd.DataFrame(rows[:5]))
    assert m._strength("A") == 1.0 and m._strength("B") == -1.0
    # 第 5 场时 B 只踢过 4 场，算不出实力 → A 那一场记的“对手实力”是空的
    assert np.isnan(m.recent["A"][-1][2])
    m._process(pd.DataFrame(rows[5:]))
    # 第 6 场记的是 B 在这场之前的实力（−1），不是算上这场以后的
    assert m.recent["A"][-1][2] == -1.0 and m.recent["B"][-1][2] == 1.0
    # 第 6 场训练表里的主队部分：第 1 行是第 5 场（最近的一场在最上面），主客场标记 1
    f6 = m.rows[-1][3]
    assert f6.shape == (5, 8)
    assert f6[0, :2].tolist() == [1, 0] and f6[0, 3] == 1.0 and f6[0, 7] == -1.0


def test_same_kickoff_matches_do_not_see_each_other(settings):
    m = YeungModel("y", small_cfg(settings), 0)
    t2 = "2010-08-08 15:00:00"
    m._process(pd.DataFrame([match(1, "2010-08-01 15:00:00", "A", "B", 2, 0),
                             match(2, t2, "A", "C", 1, 1),
                             match(3, t2, "C", "D", 0, 3)]))       # 编出来的“同一时刻 C 踢两场”，专门看会不会漏
    f2, f3 = m.rows[1][3], m.rows[2][3]
    assert f2[0, :2].tolist() == [2, 0] and np.isnan(f2[1:, :4]).all()   # A 赛前只踢过 1 场：只有第 1 行
    assert np.isnan(f2[:, 4:]).all()            # C 赛前一场都没踢过
    assert np.isnan(f3).all()                   # 第 3 场赛前 C、D 都没有记录：看不到同一时刻那场 A−C 的比分
    # 同一批再喂一次不会重复算
    m._process(pd.DataFrame([match(1, "2010-08-01 15:00:00", "A", "B", 2, 0)]))
    assert len(m.rows) == 3 and len(m.recent["A"]) == 2


def test_season_before_first_season_is_skipped(settings):
    m = YeungModel("y", small_cfg(settings, first_season=2006), 0)
    m._process(pd.DataFrame([match(1, "2005-08-01 15:00:00", "A", "B", 2, 0, season=2005),
                             match(2, "2006-08-01 15:00:00", "A", "B", 1, 0, season=2006)]))
    assert len(m.rows) == 1 and len(m.recent["A"]) == 1


def test_fit_predict_and_retrain_schedule(settings):
    hist = league()
    targets = pd.DataFrame({"home_team": ["A", "C", "Z"], "away_team": ["B", "Z", "Y"]})   # Z、Y 没见过
    m = YeungModel("y", small_cfg(settings), 0)
    assert m.predict(targets, "2011-06-01 00:00:00") == [None] * 3      # 还没训练：不出预测
    m.fit(hist, "2011-06-01 00:00:00")
    assert len(m.train_log) == 1 and len(m.team_index) == 6
    assert m.net.emb.num_embeddings == 7                                 # 多一个“没见过的队”
    p = m.predict(targets, "2011-06-01 00:00:00")
    for q in p:
        assert q is not None and len(q) == 3 and min(q) > 0 and sum(q) == pytest.approx(1.0)
    # 同一个种子再来一遍，结果一模一样
    m2 = YeungModel("y", small_cfg(settings), 0)
    m2.fit(hist, "2011-06-01 00:00:00")
    assert np.allclose(m2.predict(targets, "2011-06-01 00:00:00"), p)
    # 隔 7 天不重新训练，满 91 天重新训练
    m.fit(hist, "2011-06-08 00:00:00")
    assert len(m.train_log) == 1
    m.fit(hist, "2011-08-31 00:00:00")
    assert len(m.train_log) == 2


def test_too_few_rows_no_training(settings):
    m = YeungModel("y", small_cfg(settings, batch_size=1000), 0)
    m.fit(league(), "2011-06-01 00:00:00")
    assert m.net is None and m.predict(pd.DataFrame({"home_team": ["A"], "away_team": ["B"]}), "x") == [None]


def test_check_seasons_rejects_exam(settings):
    s = copy.deepcopy(settings)
    assert yeung_repro.check_seasons(s) == (2014, s["seasons"]["dev_last"])
    s["seasons"]["dev_last"] = s["seasons"]["exam"][0]
    with pytest.raises(ValueError, match="不是开发赛季"):
        yeung_repro.check_seasons(s)


def test_backtest_rejects_unknown_seed(settings):
    with pytest.raises(ValueError, match="种子"):
        yeung_repro.backtest(settings, 99)


def _pred_row(mid, model, horizon, home_goals=1):
    return {"match_id": mid, "model": model, "horizon": horizon, "competition": "E0", "season": 2014,
            "match_date": "2014-08-16", "home_team": "A", "away_team": "B", "home_goals": home_goals,
            "away_goals": 0, "log_loss": 1.0, "rps": 0.2, "brier": 0.6, "hit": 1}


def _write(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "predictions.csv", index=False, encoding="utf-8-sig")


def test_combined_checks_match_info(settings, tmp_path):
    s = copy.deepcopy(settings)
    s["backtest"]["output_dir"] = str(tmp_path)      # 临时文件夹，不碰项目的 data/
    cfg = s["repro_yeung"]
    for seed in cfg["seeds"]:
        _write(tmp_path / cfg["run_name"] / f"s{seed}",
               [_pred_row(1, f"yeung_s{seed}", "T-24"), _pred_row(1, "pinnacle_close", "收盘")])
    _write(tmp_path / cfg["compare_with"], [_pred_row(1, cfg["compare_model"], "T-24")])
    pred = yeung_repro.combined(s)
    # 3 个种子 + 平博收盘只留一份 + gb_l7
    assert sorted(pred["model"]) == sorted([*yeung_repro.model_names(cfg), "pinnacle_close", cfg["compare_model"]])
    # 同一个编号、比分不一样 → 停下
    _write(tmp_path / cfg["compare_with"], [_pred_row(1, cfg["compare_model"], "T-24", home_goals=3)])
    with pytest.raises(ValueError, match="对不上"):
        yeung_repro.combined(s)
