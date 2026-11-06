'''竞彩二次筛选研究 src/evaluate/picks_filter_study.py 的检查（需求文档第 14 节“竞彩二次筛选研究怎么做”）。

两种假数据（都在内存里，不碰 data/）：
  1. 一个小的假数据库（取数层那几步）：
     1 号 2019/20 英超 2019-09-15（周日）14:00 开球：football-data 赛前价格（周五 18:00 英国时间采，T-24 能知道）；懂球帝 Bet365 开球前 10 天的初盘、
       开球前 3 小时的变盘（T-24 时还看不到）；竞彩两次价格（初始、最终）和一条截止时间之后才抓到的（不能用）。
     2 号没有任何价格；9 号是期末考试赛季（2024/25），一碰就报错。
  2. 一张假的研究表（第一层矩阵、竞彩价格、比分都是随机造的），用来查滚动训练、挑比赛、结算、bootstrap。
'''
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.markets import poisson_matrices
from src.common.odds import devig
from src.evaluate import picks_filter_study as pf
from src.features import FeatureStore, SeasonLockedError
from src.features.errors import UsageNotAllowedError

K1 = '2019-09-15 14:00:00'   # 周日：football-data 周五傍晚采的赛前价格，T-24 已经能知道（周六的比赛 T-24 还没采）
K2 = '2019-09-22 14:00:00'
K9 = '2024-08-17 14:00:00'


def at(kick, hours):
    '''开球前 hours 小时的时间文字（UTC）。'''
    return (pd.Timestamp(kick) - pd.Timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s['compare']['n_boot'] = 300
    st = s['picks_filter_study']
    st['min_train_rows'] = 200
    st['n_random'] = 300
    st['lgbm']['n_rounds'] = 30
    st['lgbm']['min_data_in_leaf'] = 20
    st['train_first_result'] = 2015
    st['train_first_handicap'] = 2015
    st['eval_first'] = 2017
    st['eval_last'] = 2018
    st['lottery_cutoff_utc'] = '2026-11-03 00:00:00'
    return s


@pytest.fixture
def fs(settings):
    con = sqlite3.connect(':memory:')
    create_tables(con)
    con.executemany('insert into teams (team_id, country) values (?, ?)',
                    [(t, 'England') for t in ('H1', 'H2', 'H9', 'B')])
    con.executemany('insert into matches (match_id, competition, season, home_team, away_team, match_date,'
                    ' kickoff_utc, status) values (?, ?, ?, ?, ?, ?, ?, ?)',
                    [(i, 'E0', s, 'H' + str(i), 'B', k[:10], k, 'finished')
                     for i, s, k in ((1, 2019, K1), (2, 2019, K2), (9, 2024, K9))])
    con.executemany('insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)',
                    [(1, 'football-data', 2, 1, 'H'), (2, 'football-data', 0, 0, 'D'), (9, 'football-data', 1, 1, 'D')])
    # football-data 赛前价格：平博胜平负、多家平均大小球、Bet365 胜平负和亚盘
    odds = [(1, 'football-data', '平博 Pinnacle', '1x2', 'pre', None, 1.9, 3.6, 4.4, None, None),
            (1, 'football-data', 'Bet365', '1x2', 'pre', None, 1.85, 3.5, 4.33, None, None),
            (1, 'football-data', '多家平均', 'ou', 'pre', 2.5, None, None, None, 1.9, 1.95),
            (1, 'football-data', 'Bet365', 'ah', 'pre', -0.75, 1.95, None, 1.95, None, None),
            (9, 'football-data', '平博 Pinnacle', '1x2', 'pre', None, 2.0, 3.4, 4.0, None, None)]
    con.executemany('insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, over,'
                    ' under) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', odds)
    # 懂球帝 Bet365：开球前 10 天初盘；开球前 3 小时变盘（T-24 还看不到）
    ch = [(1, 'dongqiudi', 'Bet365', 'Bet365', '1x2', None, 2.1, 3.4, 3.6, at(K1, 240)),
          (1, 'dongqiudi', 'Bet365', 'Bet365', '1x2', None, 1.5, 4.0, 6.5, at(K1, 3)),
          (1, 'dongqiudi', 'Bet365', 'Bet365', 'ah', -0.5, 1.9, None, 2.0, at(K1, 240)),
          (1, 'dongqiudi', 'Bet365', 'Bet365', 'ah', -1.25, 1.9, None, 2.0, at(K1, 3))]
    con.executemany('insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw,'
                    ' away, changed_at_utc, fetched_at_utc, raw_file) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [(*r, r[-1], 'x') for r in ch])
    # 竞彩：1 号胜平负初始、最终各一次，让球（-1）只有一次；还有一条截止时间之后才抓到的（不能用）
    lot = [(1, 'dongqiudi', '竞彩胜平负', 0, 1.6, 3.5, 4.5, '2026-10-30 10:00:00'),
           (1, 'dongqiudi', '竞彩胜平负', 0, 1.7, 3.4, 4.1, '2026-10-31 10:00:00'),
           (1, 'dongqiudi', '竞彩胜平负', 0, 9.9, 9.9, 9.9, '2026-11-05 10:00:00'),
           (1, 'dongqiudi', '竞彩让球', -1, 3.6, 3.35, 1.78, '2026-10-30 10:00:00')]
    con.executemany('insert into lottery_odds (match_id, source, product, handicap, home, draw, away, fetched_at_utc,'
                    ' raw_file) values (?, ?, ?, ?, ?, ?, ?, ?, ?)', [(*r, 'x') for r in lot])
    con.commit()
    return FeatureStore(con, '开发', settings)


def base_rows(fs, ids):
    t = pd.DataFrame({'match_id': ids})
    t['cutoff_utc'] = [fs.cutoff_before(int(m), 24, '研究实验') for m in ids]
    k = fs.prediction_targets('2019-06-01 00:00:00', '2020-07-01 00:00:00', '研究实验', competition=['E0'])
    return t.merge(k[['match_id', 'kickoff_earliest_utc']], on='match_id')


# ======================= 不偷看未来 =======================
def test_odds_features_only_before_cutoff(fs, settings):
    out = pf.fetch_odds(fs, base_rows(fs, [1, 2]), settings).set_index('match_id')
    one = out.loc[1]
    # T-24 时懂球帝只看得到初盘：自己的变化是 0；开球前 3 小时那次变盘（1.5 / 4.0 / 6.5、亚盘 -1.25）没用上
    assert one['mv_in_d_home'] == pytest.approx(0.0)
    fd = devig(np.array([[1.85, 3.5, 4.33]]), 'proportional')[0]
    op = devig(np.array([[2.1, 3.4, 3.6]]), 'proportional')[0]
    assert one['mv_d_home'] == pytest.approx(fd[0] - op[0])
    assert one['mv_ah_line_d'] == pytest.approx(-0.75 - (-0.5))
    assert one['mv_open_age_days'] == pytest.approx(10.0)
    assert one['mv_has_dqd'] == 1.0
    assert one['mo_pinnacle'] == 1.0 and one['mk_overround'] == pytest.approx(1 / 1.9 + 1 / 3.6 + 1 / 4.4 - 1)
    # 2 号什么价格都没有：不出错，全是空值
    assert out.loc[2, 'mv_has_dqd'] == 0.0 and np.isnan(out.loc[2, 'mo_p_home']) and np.isnan(out.loc[2, 'mv_d_home'])


def test_cutoff_moves_features(fs, settings):
    '''截止时间挪到开球前 2 小时：那次变盘就看得到了（说明上面的“没用上”确实是截止时间挡住的）。'''
    b = base_rows(fs, [1])
    b['cutoff_utc'] = at(K1, 2)
    one = pf.fetch_odds(fs, b, settings).iloc[0]
    assert one['mv_in_d_home'] > 0.1


def test_exam_season_is_locked(fs, settings):
    b = pd.DataFrame({'match_id': [9], 'cutoff_utc': [at(K9, 24)], 'kickoff_earliest_utc': [K9]})
    with pytest.raises(SeasonLockedError):
        pf.fetch_odds(fs, b, settings)
    with pytest.raises(SeasonLockedError):
        pf.fetch_lottery(fs, [9], settings)


def test_lottery_only_for_settlement(fs, settings):
    lot = pf.fetch_lottery(fs, [1, 2], settings).set_index('product')
    r = lot.loc['竞彩胜平负']
    assert (r['lot_open_home'], r['lot_final_home']) == (1.6, 1.7)    # 截止时间之后的 9.9 没取到
    h = lot.loc['竞彩让球']
    assert h['handicap'] == -1 and h['lot_open_away'] == h['lot_final_away'] == 1.78   # 只有一行 = 没变过
    # 竞彩赔率不能用赛前用途取（数据字典只给「显示」「研究评估」）
    with pytest.raises(UsageNotAllowedError):
        fs.lottery_odds(1, at(K1, 24), '研究实验')
    # 二次筛选模型的特征里没有任何竞彩的列，也没有让球数
    for m in pf.MARKETS:
        for move in (True, False):
            cols = pf.feature_columns(m, move)
            assert not any(c.startswith('lot_') or c in ('handicap', 'ev1', 'ev2') for c in cols)
    assert set(pf.G4).isdisjoint(pf.feature_columns('result', with_move=False))


# ======================= 假的研究表 =======================
def fake_world(seed=1, per_week=12):
    '''随机造：2015/16–2018/19 每周 per_week 场；第一层矩阵、市场概率、竞彩价格、比分。返回 market_table 要的几样。'''
    rng = np.random.default_rng(seed)
    dates = []
    for season in (2015, 2016, 2017, 2018):
        for d in pd.date_range(f'{season}-08-15', f'{season + 1}-05-20', freq='7D'):
            dates += [(season, d)] * per_week
    n = len(dates)
    season = np.array([s for s, _ in dates])
    day = pd.to_datetime([d for _, d in dates]) + pd.to_timedelta(rng.integers(0, 3, n), unit='D')
    strength = rng.normal(0, 0.35, n)
    lh_true, la_true = np.exp(0.3 + strength), np.exp(0.1 - strength)
    mats = poisson_matrices(lh_true * np.exp(rng.normal(0, 0.1, n)), la_true * np.exp(rng.normal(0, 0.1, n)), n=10)
    dyn = poisson_matrices(lh_true * np.exp(rng.normal(0, 0.1, n)), la_true * np.exp(rng.normal(0, 0.1, n)), n=10)
    mk = poisson_matrices(lh_true, la_true, n=10)
    p_mk = pf.handicap_probs(mk, 0)
    hg, ag = rng.poisson(lh_true), rng.poisson(la_true)
    ids = np.arange(1, n + 1)
    kick = day + pd.Timedelta(hours=15)
    base = pd.DataFrame({'match_id': ids, 'competition': rng.choice(['E0', 'SP1', 'D1'], n), 'season': season,
                         'match_date': day.strftime('%Y-%m-%d'),
                         'cutoff_utc': (kick - pd.Timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S'),
                         'kickoff_earliest_utc': kick.strftime('%Y-%m-%d %H:%M:%S'),
                         'home_team': [f'T{i % 40}' for i in ids], 'away_team': [f'T{(i + 7) % 40}' for i in ids],
                         'row': np.arange(n)})
    base['mo_p_home'], base['mo_p_draw'], base['mo_p_away'] = p_mk.T
    base['mo_over25'] = (mk * (np.add.outer(np.arange(11), np.arange(11)) > 2)).sum(axis=(1, 2))
    base['mo_pinnacle'] = 1.0
    base['mk_overround'] = 0.03
    for c in ('mv_d_home', 'mv_d_draw', 'mv_d_away', 'mv_in_d_home', 'mv_in_d_draw', 'mv_in_d_away',
              'mv_ah_line_d', 'mv_open_age_days'):
        base[c] = rng.normal(0, 0.02, n)
    base['mv_has_dqd'] = 1.0
    for c in ('home_rest_days', 'away_rest_days'):
        base[c] = rng.integers(3, 8, n).astype(float)
    base['home_after_euro'] = base['away_after_euro'] = 0.0
    base['home_games'] = base['away_games'] = rng.integers(0, 38, n).astype(float)
    h = np.where(lh_true > la_true * 1.4, -1, 0)
    p_h = pf.handicap_probs(mk, h)
    rows = []
    for prod, p, hc in (('竞彩胜平负', p_mk, np.zeros(n, int)), ('竞彩让球', p_h, h)):
        fin = np.maximum(0.88 / p, 1.01)                       # 真实竞彩价格最低也 > 1
        opn = fin * np.exp(rng.normal(0, 0.03, fin.shape))
        rows.append(pd.DataFrame({'match_id': ids, 'product': prod, 'handicap': hc,
                                  **{f'lot_open_{s}': opn[:, k] for k, s in enumerate(pf.SIDES)},
                                  **{f'lot_final_{s}': fin[:, k] for k, s in enumerate(pf.SIDES)}}))
    lot = pd.concat(rows, ignore_index=True)
    lot = lot[lot['match_id'] % 17 != 0]                     # 少数比赛没有竞彩
    known = (kick + pd.Timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
    results = pd.DataFrame({'match_id': ids, 'home_goals': hg.astype(float), 'away_goals': ag.astype(float),
                            'result_known_utc': known})
    results.loc[results['match_id'] % 101 == 0, ['home_goals', 'away_goals']] = np.nan   # 少数改期（退款）
    return base, {'main': mats, 'dyn': dyn}, lot, results


@pytest.fixture(scope='module')
def world():
    return fake_world()


def test_lottery_perturbation_changes_nothing_but_settlement(world, settings):
    '''竞彩价格乱改（让球数不改）：特征、C 的分数、A / B / C / C− / E 挑的比赛都一样；只有结算的钱变了。'''
    base, mats, lot, results = world
    lot2 = lot.copy()
    rng = np.random.default_rng(5)
    for c in [c for c in lot2 if c.startswith('lot_')]:
        lot2[c] = 1 + (lot2[c] - 1) * rng.uniform(0.5, 2.0, len(lot2))   # 价格都还 > 1（≤ 1 的价格本来就不算能买）
    for m in pf.MARKETS:
        t1 = pf.market_table(base, mats, lot, results, m, settings)
        t2 = pf.market_table(base, mats, lot2, results, m, settings)
        cols = pf.feature_columns(m)
        pd.testing.assert_frame_equal(t1[cols], t2[cols])
        assert (t1['eligible'] == t2['eligible']).all() and (t1['label'].equals(t2['label']))
        w1, w2 = pf.walk_forward(t1, m, settings), pf.walk_forward(t2, m, settings)
        pd.testing.assert_frame_equal(w1, w2)
        u1 = t1[t1['eligible']].merge(w1, on='match_id').sort_values(['week', 'match_id'], ignore_index=True)
        u2 = t2[t2['eligible']].merge(w2, on='match_id').sort_values(['week', 'match_id'], ignore_index=True)
        u1, u2 = u1[u1['c_full'].notna()].reset_index(drop=True), u2[u2['c_full'].notna()].reset_index(drop=True)
        s1, s2 = pf.methods_for(u1, 5), pf.methods_for(u2, 5)
        for name in ('A 全买', 'B 第一层概率前k', 'C 二次筛选前k', 'C− 去掉水位变化前k', 'M 市场前k（参考）'):
            assert (s1[name][0] == s2[name][0]).all() and (s1[name][1] == s2[name][1]).all()
        e1 = pf.random_simulation(u1, 5, 50, 7)['命中率']
        e2 = pf.random_simulation(u2, 5, 50, 7)['命中率']
        assert np.allclose(e1, e2)
        assert not np.allclose(t1['lot_final_home'].fillna(0), t2['lot_final_home'].fillna(0))


def test_walk_forward_uses_only_known_labels(world, settings):
    '''每块的训练行 = 块的截止时间之前已经能知道比分的；把截止时间之后的比分全改掉，第一块的分数不变。'''
    base, mats, lot, results = world
    t = pf.market_table(base, mats, lot, results, 'result', settings)
    wf = pf.walk_forward(t, 'result', settings)
    lab = t[t['label'].notna()]
    for b, g in wf.groupby('block'):
        cut = g['block_cutoff'].iloc[0]
        assert g['n_train'].iloc[0] == (lab['result_known_utc'] <= cut).sum()
        ev = t[t['match_id'].isin(g['match_id'])]
        assert (ev['cutoff_utc'] >= cut).all()                       # 块里每场的截止时间都不早于块的截止时间
    first = wf[wf['block'] == wf['block'].min()]
    cut = first['block_cutoff'].iloc[0]
    t2 = t.copy()
    late = t2['result_known_utc'] > cut
    t2.loc[late, 'label'] = 1 - t2.loc[late, 'label']                # 以后才知道的比分全反过来
    t2.loc[late & ~t2['eligible'], 'p_pick'] = 0.99                  # 以后的、不打分的比赛特征也乱改
    wf2 = pf.walk_forward(t2, 'result', settings)
    f2 = wf2[wf2['block'] == wf2['block'].min()]
    assert np.allclose(first['c_full'].to_numpy(), f2['c_full'].to_numpy())
    assert not np.allclose(wf['c_full'].to_numpy(), wf2['c_full'].to_numpy())   # 后面的块确实变了（说明改动有效）
    assert wf['c_full'].notna().all()


def test_lgbm_is_deterministic(world, settings):
    base, mats, lot, results = world
    t = pf.market_table(base, mats, lot, results, 'handicap', settings)
    pd.testing.assert_frame_equal(pf.walk_forward(t, 'handicap', settings), pf.walk_forward(t, 'handicap', settings))


# ======================= 让球结算 =======================
def test_settle_outcome_handicap():
    hg, ag = [2, 1, 1, 0, 3, None], [1, 0, 1, 0, 0, 1]
    assert list(pf.settle_outcome(hg, ag, 0)) == [0, 0, 1, 1, 0, -1]
    # 主队让 1 球（-1）：赢 1 球 = 让球平，赢 2 球以上才算让球主胜
    assert list(pf.settle_outcome(hg, ag, -1)) == [1, 1, 2, 2, 0, -1]
    # 主队受让 1 球（+1）：输 1 球 = 让球平
    assert list(pf.settle_outcome([0, 0, 1], [1, 2, 1], 1)) == [1, 2, 0]


def test_handicap_probs_matches_settlement():
    m = np.zeros((1, 11, 11))
    m[0, 2, 1], m[0, 1, 1], m[0, 3, 0] = 0.5, 0.3, 0.2              # 2:1、1:1、3:0
    assert np.allclose(pf.handicap_probs(m, 0), [[0.7, 0.3, 0.0]])
    assert np.allclose(pf.handicap_probs(m, -1), [[0.2, 0.5, 0.3]])
    assert np.allclose(pf.handicap_probs(m, -2), [[0.2, 0.0, 0.8]])      # 3:0 让 2 球还赢 1 球
    # 一批矩阵、每场不同的让球数，和逐场算一样
    rng = np.random.default_rng(0)
    mats = poisson_matrices(rng.uniform(0.8, 2.2, 5), rng.uniform(0.6, 1.8, 5), n=10)
    h = np.array([0, -1, 1, -2, 0])
    batch = pf.handicap_probs(mats, h)
    assert np.allclose([pf.handicap_probs(mats[i:i + 1], h[i])[0] for i in range(5)], batch)
    assert np.allclose(batch.sum(axis=1), 1.0)


# ======================= 命中率和 ROI =======================
def bet_frame():
    return pd.DataFrame({'week': ['2019-09-09'] * 2 + ['2019-09-16'] * 2, 'season': 2019, 'match_id': [1, 2, 3, 4],
                         'finished': [True, True, True, False], 'outcome': [0, 2, 1, -1],
                         **{f'lot_final_{s}': [2.0, 3.0, 4.0, 5.0] for s in pf.SIDES},
                         **{f'lot_open_{s}': [2.5, 3.0, 4.0, 5.0] for s in pf.SIDES}})


def test_roi_and_hit_rate():
    d = bet_frame()
    b = pf.settle(d, np.ones(4, bool), np.array([0, 0, 1, 0]))      # 中、不中、中、退款
    assert list(b['stake']) == [1, 1, 1, 0] and list(b['hit']) == [1, 0, 1, 0]
    assert list(b['profit_final']) == [1.0, -1.0, 3.0, 0.0]
    assert list(b['profit_open']) == [1.5, -1.0, 3.0, 0.0]
    weeks = np.array(['2019-09-09', '2019-09-16'])
    w = pf.weekly(b, weeks)
    assert w.tolist() == [[2, 1, 0.0, 0.5], [1, 1, 3.0, 3.0]]
    counts = pf.boot_counts(2, 50, 1)
    est, _, _ = pf.ratio_ci(w[:, 1], w[:, 0], counts, 0.95)
    assert est == pytest.approx(2 / 3)                                # 命中率 = 2 中 / 3 注（退款不算注）
    roi, _, _ = pf.ratio_ci(w[:, 2], w[:, 0], counts, 0.95)
    assert roi == pytest.approx(3.0 / 3)                              # ROI = 总盈亏 3 ÷ 3 注
    row = pf.method_row('x', w, counts, 0.95, pf.CICounter())
    assert row['注数'] == 3 and row['命中率'] == pytest.approx(2 / 3) and row['ROI_初始价'] == pytest.approx(3.5 / 3)
    # 只挑一部分：mask 外的不算
    b2 = pf.settle(d, np.array([True, False, False, False]), 2)
    assert len(b2) == 1 and b2['hit'].iloc[0] == 0 and b2['profit_final'].iloc[0] == -1.0


# ======================= 成块 bootstrap =======================
def test_bootstrap_basic_properties():
    counts = pf.boot_counts(10, 500, 42)
    assert counts.shape == (500, 10) and (counts.sum(axis=1) == 10).all()     # 每次抽 10 周
    assert np.array_equal(counts, pf.boot_counts(10, 500, 42))               # 种子一样，抽样一样
    assert not np.array_equal(counts, pf.boot_counts(10, 500, 43))
    # 每周命中率都是 0.4：区间就是 0.4
    st, hit = np.array([5.0] * 10), np.array([2.0] * 10)
    assert pf.ratio_ci(hit, st, counts, 0.95) == pytest.approx((0.4, 0.4, 0.4))
    # 同一个方法和自己比：差的区间是 0
    w = np.column_stack([st, hit, hit - 3, hit - 3])
    assert pf.ratio_diff_ci(w[:, 1], w[:, 0], w[:, 1], w[:, 0], counts, 0.95) == pytest.approx((0, 0, 0))


def test_bootstrap_resamples_whole_weeks():
    '''两周：第 1 周 3 注全中，第 2 周 1 注没中。整周一起抽，只可能出现 0、3/4、1 三种命中率（拆开抽才会有别的值）。'''
    counts = pf.boot_counts(2, 400, 3)
    st, hit = np.array([3.0, 1.0]), np.array([3.0, 0.0])
    b = pf._ratio(counts @ hit, counts @ st)
    assert set(np.round(b, 6)) <= {0.0, 0.75, 1.0}
    assert len(set(np.round(b, 6))) == 3


# ======================= 每周前 k 场 =======================
def test_select_top_k():
    d = pd.DataFrame({'week': ['a'] * 5 + ['b'] * 2, 'match_id': [5, 4, 3, 2, 1, 7, 6],
                      's': [0.9, 0.5, 0.5, np.nan, 0.1, 0.2, 0.3]})
    m = pf.select_top_k(d, 's', 2)
    # a 周：0.9 的 5 号；两场 0.5 一样高，挑 match_id 小的 3 号；b 周只有两场，全挑；空值不挑
    assert d.loc[m, 'match_id'].tolist() == [5, 3, 7, 6]
    assert pf.select_top_k(d, 's', 10).sum() == 6
    assert d.loc[pf.select_top_k(d, 's', 1), 'match_id'].tolist() == [5, 6]


def test_random_expectation_and_simulation():
    d = bet_frame().iloc[:3].assign(pick=[0, 0, 1])                 # 第 1 周 2 场（1 中 1 不中），第 2 周 1 场（中）
    weeks = np.array(['2019-09-09', '2019-09-16'])
    e = pf.random_expectation(d, 1, weeks)
    assert e.tolist() == [[1.0, 0.5, 0.0, 0.25], [1.0, 1.0, 3.0, 3.0]]   # 第 1 周挑 1 场的期望 = 一半
    sim = pf.random_simulation(d, 1, 4000, 9)
    assert set(np.round(sim['命中率'], 6)) <= {0.5, 1.0}               # 每次正好两注
    assert sim['命中率'].mean() == pytest.approx(0.75, abs=0.02)


def test_run_market_end_to_end(world, settings):
    base, mats, lot, results = world
    t = pf.market_table(base, mats, lot, results, 'handicap', settings)
    ci = pf.CICounter()
    r = pf.run_market(t, 'handicap', settings, ci)
    tab = r['主表'].set_index('方法')
    assert tab.loc['A 全买', '注数'] == t.loc[t['eligible'] & t['finished'], 'match_id'].isin(
        r['每场']['match_id']).sum()
    weeks = r['周数']
    assert tab.loc['B 第一层概率前k', '注数'] <= 5 * weeks
    # 价格 = 0.88 ÷ 市场概率：全买的 ROI 应该在 -12% 附近
    assert -0.3 < tab.loc['A 全买', 'ROI'] < 0.05
    assert r['返奖率_final'] == pytest.approx(0.88, abs=0.01)
    assert {'J1', 'J2', 'J3', '影子'} <= set(r) and ci.n > 20
    assert len(r['每周名单']) <= 5 * weeks
