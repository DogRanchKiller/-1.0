"""虚拟投注机器人 bots_v1 的检查（需求文档第 14 节“历史回测第一版规则 bots_v1”）。

都用小的假数据（测试数据库在内存里，回测结果写在 tmp_path），不碰真实数据库和真实回测结果。
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.bots import betting as bt
from src.common.config import load_settings


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s["compare"]["n_boot"] = 200
    return s


@pytest.fixture
def cfg(settings):
    return settings["bots"]


def _bot(cfg, name):
    return next(b for b in cfg["list"] if b["name"] == name)


# ---------------- 选注 ----------------
def test_pick_random(cfg):
    """随机机器人：同一个种子、同一场，重跑结果一样；换场次三个结果都会出现，比例接近 1/3。"""
    bot = _bot(cfg, "对照_随机_平博")
    odds = [2.0, 3.4, 4.0]
    first = [bt.pick_bet(bot, mid, {}, odds, 20261101)[0] for mid in range(3000)]
    again = [bt.pick_bet(bot, mid, {}, odds, 20261101)[0] for mid in range(3000)]
    assert first == again
    share = np.bincount(first, minlength=3) / len(first)
    assert np.all(np.abs(share - 1 / 3) < 0.04)
    other_seed = [bt.pick_bet(bot, mid, {}, odds, 1)[0] for mid in range(3000)]
    assert other_seed != first                                   # 换种子就换一套
    k, p, o = bt.pick_bet(bot, 7, {}, odds, 20261101)
    assert np.isnan(p) and o == odds[k]                          # 对照组没有概率来源


def test_pick_favorite(cfg):
    bot = _bot(cfg, "对照_热门_最高")
    assert bt.pick_bet(bot, 1, {}, [2.5, 3.2, 2.9], 0)[0] == 0
    assert bt.pick_bet(bot, 1, {}, [4.0, 3.5, 1.8], 0)[0] == 2
    assert bt.pick_bet(bot, 1, {}, [2.6, 3.0, 2.6], 0)[0] == 0   # 一样低按主、平、客的顺序
    assert bt.pick_bet(bot, 1, {}, None, 0) is None              # 没有像样的价格不下
    assert bt.pick_bet(bot, 1, {}, [2.0, np.nan, 3.0], 0) is None


def test_pick_home(cfg):
    bot = _bot(cfg, "对照_主队_平博")
    for odds in ([1.3, 5.0, 9.0], [6.0, 4.0, 1.5]):
        k, _, o = bt.pick_bet(bot, 3, {}, odds, 0)
        assert k == 0 and o == odds[0]


def test_pick_top_value(cfg):
    """top 选模型概率最高的；value 选期望收益最大的，过门槛才下；fusion 用融合概率。"""
    top = _bot(cfg, "跟模型_最高")
    probs = {"model": np.array([0.30, 0.25, 0.45]), "fusion": np.array([0.5, 0.3, 0.2])}
    assert bt.pick_bet(top, 1, probs, [3.0, 3.6, 2.4], 0) == (2, 0.45, 2.4)
    v3, v8 = _bot(cfg, "模型价值3_最高"), _bot(cfg, "模型价值8_最高")
    # 期望收益：0.30×3.6−1 = +8%，0.25×3.6−1 = −10%，0.45×2.3−1 = +3.5% → 最大的是主胜 +8%
    odds = [3.6, 3.6, 2.3]
    assert bt.pick_bet(v3, 1, probs, odds, 0)[0] == 0
    assert bt.pick_bet(v8, 1, probs, odds, 0)[0] == 0             # 正好 8%（有计算误差也算过门槛）
    assert bt.pick_bet(v8, 1, probs, [3.5, 3.6, 2.3], 0) is None  # 最大的 +5% < 8%，不下
    fv = _bot(cfg, "融合价值3_最高")
    assert bt.pick_bet(fv, 1, probs, [2.1, 3.0, 4.0], 0)[0] == 0  # 融合 0.5 × 2.1 − 1 = +5%
    assert bt.pick_bet(fv, 1, {"model": probs["model"], "fusion": np.full(3, np.nan)},
                       [2.1, 3.0, 4.0], 0) is None               # 没有融合概率不下


# ---------------- 注额 ----------------
def test_calc_stake_flat(cfg):
    bot = _bot(cfg, "模型价值5_平博")
    assert bt.calc_stake(bot, 0.5, 2.2, 10000, cfg) == 100
    assert bt.calc_stake(bot, 0.1, 9.0, 3000, cfg) == 100        # 成绩用的固定注额不随资金变


def test_calc_stake_kelly(cfg):
    bot = _bot(cfg, "模型凯利_最高")
    # f = (0.5 × 2.2 − 1) ÷ (2.2 − 1) × 0.25 = 0.1 ÷ 1.2 × 0.25 ≈ 2.083%
    assert bt.calc_stake(bot, 0.5, 2.2, 10000, cfg) == pytest.approx(10000 * 0.1 / 1.2 * 0.25)
    # 优势很大：(0.8 × 2 − 1) ÷ 1 × 0.25 = 15% → 封顶 5%
    assert bt.calc_stake(bot, 0.8, 2.0, 10000, cfg) == pytest.approx(500)
    assert bt.calc_stake(bot, 0.4, 2.0, 10000, cfg) == 0          # 期望收益为负 → 0
    # 资金模拟：按当前资金，也封顶 5%
    assert bt.sim_stake(bot, 0.8, 2.0, 6000, cfg) == pytest.approx(300)
    assert bt.sim_stake(_bot(cfg, "跟模型_平博"), 0.5, 2.0, 4000, cfg) == pytest.approx(80)   # min(100, 2%)


# ---------------- 结算、CLV ----------------
def test_settle_bet():
    """买主胜、赔率 2.0、注额 100：主胜赢 +100，平局、客胜输 −100，没有比分算“走” 0。"""
    assert bt.settle_bet(0, "H", 2.0, 100) == ("win", 100.0)
    assert bt.settle_bet(0, "D", 2.0, 100) == ("loss", -100.0)
    assert bt.settle_bet(0, "A", 2.0, 100) == ("loss", -100.0)
    assert bt.settle_bet(0, None, 2.0, 100) == ("push", 0.0)
    assert bt.settle_bet(1, "D", 3.4, 100) == ("win", pytest.approx(240.0))


def test_clv():
    close = [2.0, 3.5, 4.0]
    q = (1 / 2.0) / (1 / 2.0 + 1 / 3.5 + 1 / 4.0)                 # 按比例去水后主胜的概率
    assert bt.clv(0, 2.2, close) == pytest.approx(2.2 * q - 1)
    assert np.isnan(bt.clv(0, 2.2, [np.nan] * 3))                # 没有平博收盘 → 空，不参加平均


# ---------------- 资金模拟 ----------------
def _bets(n, days, profit_per_bet, stake=100.0, prob=0.5, odds=2.0):
    """n 注假注单：第 i 注在 days[i] 那天 12:00 截止，开球后 3 天结果能知道之前就排下一注（同一天的注单还没结算）。"""
    rows = []
    for i in range(n):
        cut = pd.Timestamp(days[i]) + pd.Timedelta(hours=12, minutes=i % 60)
        rows.append({"match_id": i, "cutoff_utc": cut.strftime("%Y-%m-%d %H:%M:%S"),
                     "result_known_utc": (cut + pd.Timedelta(hours=27)).strftime("%Y-%m-%d %H:%M:%S"),
                     "prob": prob, "odds": odds, "stake": stake, "profit": profit_per_bet})
    return pd.DataFrame(rows)


def test_daily_cap(cfg):
    """同一天 25 注固定注额：每注 min(100, 10000 × 2%) = 100，当天合计最多 10000 × 20% = 2000 → 下 20 注，跳过 5 注。"""
    bets = _bets(25, ["2015-09-12"] * 25, 0.0)
    out, sim = bt.simulate_bankroll(bets, _bot(cfg, "跟模型_平博"), cfg)
    assert sim["实际下了几注"] == 20 and sim["单日上限跳过"] == 5
    assert out.loc[out["sim_status"] == bt.SIM_PLACED, "sim_stake"].sum() == pytest.approx(2000)
    # 凯利类：优势大（封顶 5% = 500）→ 当天只下得了 4 注
    out, sim = bt.simulate_bankroll(_bets(6, ["2015-09-12"] * 6, 0.0, prob=0.8), _bot(cfg, "模型凯利_最高"), cfg)
    assert sim["实际下了几注"] == 4 and sim["单日上限跳过"] == 2
    # 第二天重新算
    days = ["2015-09-12"] * 22 + ["2015-09-13"] * 3
    _, sim = bt.simulate_bankroll(_bets(25, days, 0.0), _bot(cfg, "跟模型_平博"), cfg)
    assert sim["实际下了几注"] == 23 and sim["单日上限跳过"] == 2


def test_stop_drawdown(cfg):
    """每天一注、每注输：资金从最高点回撤超过 30% 就停，后面的注都不下，停在哪天要记下。"""
    days = pd.date_range("2015-08-01", periods=60, freq="7D").strftime("%Y-%m-%d").tolist()
    bets = _bets(60, days, -100.0)            # 成绩用的注额 100、输 −100 → 每 1 单位本金 −1
    out, sim = bt.simulate_bankroll(bets, _bot(cfg, "跟模型_平博"), cfg)
    assert sim["停了没有"] == "停了" and sim["停在哪天"] is not None
    assert sim["最大回撤"] > 0.30
    placed = out[out["sim_status"] == bt.SIM_PLACED]
    assert len(placed) == 31                  # 输 30 注正好回撤 30%（不算超过），第 31 注输完 31% → 停
    assert (out.loc[placed.index.max() + 1:, "sim_status"] == bt.SIM_STOPPED).all()
    assert sim["最后资金"] / 10000 < 0.70
    # 一直不输：不会停
    _, sim = bt.simulate_bankroll(_bets(60, days, 0.0), _bot(cfg, "跟模型_平博"), cfg)
    assert sim["停了没有"] == "没停" and sim["实际下了几注"] == 60
