'''赔率走向研究 src/evaluate/odds_move_study.py 的检查（需求文档第 14 节“赔率走向研究怎么做”）。

用内存里的假数据库（不碰 data/）：只给 T 之前的价格、开球后的价格不用、考试赛季取不到、少于 3 家公司的不进、
没有价格的场不出错、去水后和为 1、热门和 ΔF 的符号、公司对同场别家的差加起来是 0、β 的算法、判断标准。
造的数据：
  1 号 2019/20 英超 2019-09-14 14:00 开球，三家公司，各有开球前 10 天的初盘、开球前 3 小时一次变盘（热门都变短）、开球后一次（不能用）；
  2 号只有两家公司；3 号没有价格；9 号是期末考试赛季（2024/25）。
'''
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.odds import devig
from src.evaluate import odds_move_study as oms
from src.features import FeatureStore, SeasonLockedError

BOOKS = ['Bet365', 'William Hill', 'Interwetten']
K1 = '2019-09-14 14:00:00'
K2 = '2019-09-21 14:00:00'
K3 = '2019-09-28 14:00:00'
K9 = '2024-08-17 14:00:00'
# 每家公司：(开球前 10 天的初盘, 开球前 3 小时的变盘)
PRICES = {
    'Bet365': ((2.0, 3.4, 4.0), (1.8, 3.6, 4.6)),
    'William Hill': ((2.1, 3.4, 3.8), (1.9, 3.6, 4.2)),
    'Interwetten': ((1.9, 3.5, 4.2), (1.8, 3.6, 4.8)),
}


def at(kick, hours):
    '''开球前 hours 小时的时间文字（UTC）；hours 为负就是开球后。'''
    return (pd.Timestamp(kick) - pd.Timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s['compare']['n_boot'] = 200
    return s


@pytest.fixture
def fs(settings):
    con = sqlite3.connect(':memory:')
    create_tables(con)
    con.executemany('insert into teams (team_id, country) values (?, ?)',
                    [(t, 'England') for t in ('H1', 'H2', 'H3', 'H9', 'B')])
    matches = [(1, 2019, K1), (2, 2019, K2), (3, 2019, K3), (9, 2024, K9)]
    con.executemany('insert into matches (match_id, competition, season, home_team, away_team, match_date,'
                    ' kickoff_utc, status) values (?, ?, ?, ?, ?, ?, ?, ?)',
                    [(i, 'E0', s, 'H' + str(i), 'B', k[:10], k, 'finished') for i, s, k in matches])
    rows = []
    for mid, kick, books in ((1, K1, BOOKS), (2, K2, BOOKS[:2]), (9, K9, BOOKS)):
        for bk in books:
            first, second = PRICES[bk]
            for when, price in ((at(kick, 240), first), (at(kick, 3), second), (at(kick, -1), (1.2, 6.0, 12.0))):
                rows.append((mid, 'dongqiudi', bk, bk, '1x2', None, *price, when, when, 'x'))
    con.executemany('insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw,'
                    ' away, changed_at_utc, fetched_at_utc, raw_file) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    rows)
    con.commit()
    return FeatureStore(con, '开发', settings)


def targets(fs):
    return fs.prediction_targets('2019-06-01 00:00:00', '2020-07-01 00:00:00', '机器人下注', competition=['E0'])


def test_collect_uses_only_prices_before_cutoff(fs, settings):
    long = oms.collect(fs, settings, pd.DataFrame({'match_id': [1, 2, 3]}), BOOKS)
    one = long[long['match_id'] == 1]
    assert len(one) == 3 * 3                                     # 三个时点 × 三家公司
    t24 = one[(one['horizon'] == 24) & (one['bookmaker'] == 'Bet365')].iloc[0]
    assert (t24['t_home'], t24['f_home']) == (2.0, 1.8)          # T-24 只知道初盘；最后价格是开球前 3 小时那次
    t1 = one[(one['horizon'] == 1) & (one['bookmaker'] == 'Bet365')].iloc[0]
    assert t1['t_home'] == 1.8                                   # 开球前 3 小时的变盘，T-1 时已经知道
    assert (one['f_home'] > 1.5).all()                           # 开球后那次（1.2）不会混进最后价格
    for h in (24, 6, 1):
        part = one[one['horizon'] == h]
        assert (part['t_at'] <= at(K1, h)).all() and (part['f_at'] <= at(K1, 1 / 60)).all()
    assert (one['first_at'] == at(K1, 240)).all()
    assert 3 not in set(long['match_id'])                        # 没有价格的场不出错，也不出现


def test_consensus_needs_three_books(fs, settings):
    m = targets(fs)
    long = oms.collect(fs, settings, m, BOOKS)
    assert set(long['match_id']) == {1, 2}
    b, per_match = oms.consensus(long, m, settings)
    assert set(per_match['match_id']) == {1} and set(b['match_id']) == {1}


def test_consensus_probabilities_and_sign(fs, settings):
    m = targets(fs)
    b, pm = oms.consensus(oms.collect(fs, settings, m, BOOKS), m, settings)
    row = pm[pm['horizon'] == 24].iloc[0]
    assert sum(row['ct_' + s] for s in oms.ODDS) == pytest.approx(1.0)
    assert sum(row['cf_' + s] for s in oms.ODDS) == pytest.approx(1.0)
    assert row['fav'] == 0                                       # 主胜是热门
    method = settings['backtest']['benchmark_devig_method']
    t = devig(np.array([PRICES[k][0] for k in BOOKS]), method)
    f = devig(np.array([PRICES[k][1] for k in BOOKS]), method)
    assert row['dF'] == pytest.approx(f[:, 0].mean() - t[:, 0].mean())
    assert row['dF'] > 0 and row['d_home'] > 0                   # 热门价格变短：概率变大
    assert row['d_home'] + row['d_draw'] + row['d_away'] == pytest.approx(0.0, abs=1e-12)


def test_company_gap_sums_to_zero_and_signs(fs, settings):
    m = targets(fs)
    b, _ = oms.consensus(oms.collect(fs, settings, m, BOOKS), m, settings)
    sums = b.groupby(['match_id', 'horizon'])['x1'].sum()
    assert np.allclose(sums, 0, atol=1e-12)
    one = b[(b['match_id'] == 1) & (b['horizon'] == 24)].sort_values('x1')
    assert one.iloc[-1]['bookmaker'] == 'Interwetten' and one.iloc[0]['bookmaker'] == 'William Hill'
    assert np.allclose(one['y1'], one['pf_fav'] - one['pt_fav'])


def test_move_table_row(fs, settings):
    m = targets(fs)
    _, pm = oms.consensus(oms.collect(fs, settings, m, BOOKS), m, settings)
    mt = oms.move_table(pm, settings)
    row = mt[(mt['时点'] == 'T-24') & (mt['分法'] == '全部')].iloc[0]
    assert row['场数'] == 1 and row['ΔF平均'] > 0 and row['动了的比例'] == 1.0


def test_exam_season_is_locked(fs, settings):
    with pytest.raises(SeasonLockedError):
        oms.collect(fs, settings, pd.DataFrame({'match_id': [9]}), BOOKS)
    tables = oms.study(fs, settings, books=BOOKS)               # 整套研究在假数据上能跑完，而且不碰考试赛季
    assert 9 not in set(tables['per_match']['match_id'])
    assert set(tables['per_match']['match_id']) == {1}
    assert tables['verdict'].iloc[0]['判断'] == '数据不够判断'    # 只有 1 场，不下结论


def test_beta_ratio(settings):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, 400)
    dates = pd.date_range('2019-08-01', periods=400, freq='D').strftime('%Y-%m-%d')
    seasons = np.where(np.arange(400) < 200, 2019, 2020)
    for k in (0.5, -2.0):
        out = oms.beta_rows('x', x, k * x, dates, seasons, settings)
        assert list(out['哪一份']) == ['合并', '2019', '2020']
        assert np.allclose(out['β'], k)


def test_judge_rules(settings):
    c = settings['odds_move_study']

    def j(mean, lo, hi, by, counts, need_abs=True):
        return oms.judge('x', mean, lo, hi, by, counts, c, need_abs)['判断']

    big = {2019: 1000, 2020: 1000, 2021: 1000, 2022: 1000}
    pos = {s: 0.01 for s in big}
    assert j(0.01, 0.004, 0.016, pos, big).startswith('看到规律')
    assert j(0.01, -0.001, 0.02, pos, big).startswith('没看到可靠规律')                    # 区间跨过 0
    assert j(0.01, 0.004, 0.016, {**pos, 2021: -0.002}, big).startswith('没看到可靠规律')  # 有赛季方向相反
    tiny = {s: 0.003 for s in big}
    assert j(0.003, 0.001, 0.005, tiny, big).startswith('没看到可靠规律')                  # 平均太小
    assert j(0.003, 0.001, 0.005, tiny, big, need_abs=False).startswith('看到规律')        # β 不要求这条
    small = {2019: 1000, 2020: 1000, 2021: 1000, 2022: 50}
    assert j(0.01, 0.004, 0.016, {**pos, 2022: -0.5}, small).startswith('看到规律')       # 不够样本的赛季不管
    assert j(0.01, 0.004, 0.016, pos, {2019: 1000, 2020: 1000}).startswith('数据不够')     # 够样本的赛季不到 3 个
    assert j(0.01, 0.004, 0.016, pos, {s: 300 for s in big}).startswith('数据不够')        # 合计 1200 < 3000


def test_trusted_books_only_checked(settings):
    books = oms.trusted_books(settings)
    assert 'Bet365' in books and '平博 Pinnacle' in books
    assert '澳门' not in books and '易胜博' not in books
