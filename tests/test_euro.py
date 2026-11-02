'''第 8 阶段（欧洲杯赛的跨联赛实力对比）的自动检查。

不碰真实数据库：用手造的小数据，数据库文件放在 pytest 的 tmp_path 里。
检查的东西：
  - 中立场推断、疑似重复比赛合并、没有比分的去掉、国内杯赛不算欧战；
  - 取数层：阶段 2 拦住；阶段 8 + 研究实验放行；只给截止时间前已知的比分；
    2024 赛季（期末考试）开发模式拦住；考试模式不给研究实验；
  - 复制设置改阶段 8，不影响共用的设置；
  - 欧战实力表：模拟数据能找回 δ；对照版 δ 全 0；只用截止前窗口里的比赛；比赛太少不出预测；
    评分拿不到的比赛跳过；中立场只差一个主场优势；
  - dyn 评分快照：联赛赛季够了才有数；
  - 残差研究：只用 T-24 之前已经知道比分的欧战；斜率的 bootstrap 能找回已知斜率；主比较能认出明显更准的。
'''
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.evaluate.euro_study import leg_kind, main_compare, residual_rows, slope_boot
from src.features import FeatureStore, SeasonLockedError, UsageNotAllowedError, open_readonly
from src.features import euro_matches as EM
from src.features.euro_matches import (dev_end_cutoff, drop_round_duplicates, fetch_euro, mark_neutral,
                                       prepare_euro, stage8_settings)
from src.models.dynamic import DynamicModel
from src.models.euro_strength import EuroStrength, dyn_team_table


@pytest.fixture(scope='module')
def settings():
    return load_settings()


@pytest.fixture
def cfg(settings):
    return settings['euro']


def frame(rows) -> pd.DataFrame:
    '''造一张“取数层给的杯赛欧战表”。'''
    cols = ['match_id', 'competition', 'season', 'home_team', 'away_team', 'round', 'leg', 'date_precision',
            'home_goals', 'away_goals', 'result']
    return pd.DataFrame(rows, columns=cols)


# ======================= 整理：中立场、重复、没有比分 =======================
def test_mark_neutral(cfg):
    '''决赛（leg 0）和 2019 赛季 UCL、UEL 的 R16、QF、SF 单场淘汰赛是中立场；其余不是。'''
    df = frame([
        (1, 'UCL', 2022, 'A', 'B', 'F', 0, 'day', 1, 0, 'H'),
        (2, 'UCL', 2019, 'A', 'B', 'QF', 0, 'day', 1, 0, 'H'),
        (3, 'UEL', 2019, 'A', 'B', 'SF', 0, 'day', 1, 0, 'H'),
        (4, 'UCL', 2019, 'A', 'B', 'QF', 1, 'day', 1, 0, 'H'),     # 两回合的第 1 回合：不是
        (5, 'UCL', 2018, 'A', 'B', 'QF', 0, 'day', 1, 0, 'H'),     # 别的赛季：不是
        (6, 'UECL', 2019, 'A', 'B', 'QF', 0, 'day', 1, 0, 'H'),    # UECL 当时还没有：规则里不含
        (7, 'UCL', 2019, 'A', 'B', 'G', 0, 'day', 1, 0, 'H'),      # 小组赛：不是
        (8, 'UCL', 2022, 'A', 'B', 'F', 1, 'day', 1, 0, 'H'),      # leg 不是 0：不是
    ])
    out = mark_neutral(df, cfg)
    assert out['neutral'].tolist() == [True, True, True, False, False, False, False, False]
    assert 'neutral' not in df.columns                                # 不改传进来的表


def test_drop_round_duplicates():
    '''轮次精度的一行 + 日期精度的一行：同赛事、赛季、轮次，比分相同，有一队相同，只有一个候选才去掉。'''
    df = frame([
        (10, 'UEL', 2020, 'Dortmund', 'Odds BK', 'Q2', 1, 'round', 1, 0, 'H'),
        (11, 'UEL', 2020, 'Dortmund', 'Odd', 'Q2', 1, 'day', 1, 0, 'H'),         # 10 的重复
        (12, 'UEL', 2020, 'X', 'Y', 'Q1', 1, 'round', 2, 2, 'D'),
        (13, 'UEL', 2020, 'X', 'Z', 'Q1', 1, 'day', 2, 1, 'H'),                  # 比分不同：不算重复
        (14, 'UEL', 2020, 'M', 'N', 'Q3', 1, 'round', 0, 0, 'D'),
        (15, 'UEL', 2020, 'M', 'P', 'Q3', 1, 'day', 0, 0, 'D'),                  # 两个候选：说不准，不去掉
        (16, 'UEL', 2020, 'Q', 'N', 'Q3', 1, 'day', 0, 0, 'D'),
    ])
    kept, rep = drop_round_duplicates(df)
    assert sorted(kept['match_id']) == [11, 12, 13, 14, 15, 16]
    assert {int(k): bool(v) for k, v in rep.set_index('match_id')['dropped'].items()} == {10: True, 14: False}
    assert int(rep.set_index('match_id').loc[10, 'partner_match_id']) == 11


def test_prepare_euro_filters_scores_and_competitions(cfg):
    raw = frame([
        (1, 'UCL', 2021, 'A', 'B', 'G', 0, 'day', 2, 1, 'H'),
        (2, 'UCL', 2021, 'A', 'C', 'G', 0, 'day', np.nan, np.nan, None),         # 没有比分
        (3, 'FAC', 2021, 'A', 'D', 'R3', 0, 'day', 1, 0, 'H'),                   # 国内杯赛：不算欧战
        (4, 'UEL', 2021, 'E', 'F', 'F', 0, 'day', 0, 0, 'D'),                    # 决赛：中立场
    ])
    df, rep = prepare_euro(raw, cfg)
    assert df['match_id'].tolist() == [1, 4]
    assert df['home_goals'].dtype.kind == 'i'
    assert df['neutral'].tolist() == [False, True]
    assert rep['drop_reason'].tolist() == ['没有比分']
    with pytest.raises(ValueError, match='缺这些列'):
        prepare_euro(raw.drop(columns=['leg']), cfg)


def test_stage8_settings_is_a_copy(settings):
    before = settings['features']['current_stage']
    s8 = stage8_settings(settings)
    assert s8['features']['current_stage'] == 8
    assert settings['features']['current_stage'] == before                     # 共用的设置没动
    assert dev_end_cutoff(settings) == '2024-06-30 00:00:00'


# ======================= 取数层：阶段、截止时间、运行模式 =======================
EURO = [
    # (match_id, 赛事, 赛季, 主队, 客队, 日期, 开球 UTC, 轮次, 回合, 日期精度)
    (1, 'UCL', 2023, 'A', 'B', '2023-11-01', '2023-11-01 20:00', 'G', 0, 'day'),
    (2, 'UCL', 2023, 'B', 'A', '2023-12-05', '2023-12-05 20:00', 'G', 0, 'day'),
    (3, 'UEL', 2023, 'C', 'X', '2024-05-22', '2024-05-22 19:00', 'F', 0, 'day'),     # 决赛
    (4, 'UCL', 2024, 'A', 'C', '2024-09-17', '2024-09-17 19:00', 'G', 0, 'day'),     # 期末考试赛季
    (5, 'FAC', 2023, 'A', 'B', '2023-11-10', '2023-11-10 15:00', 'R3', 0, 'day'),    # 国内杯赛
]
SCORES = {1: (2, 1, 'H'), 2: (0, 0, 'D'), 3: (1, 0, 'H'), 4: (3, 0, 'H'), 5: (1, 1, 'D')}


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / 'football.db'
    con = sqlite3.connect(path)
    create_tables(con)
    con.executemany('insert into teams (team_id, country) values (?, ?)', [(t, 'Test') for t in 'ABCX'])
    con.executemany('insert into matches (match_id, competition, season, home_team, away_team, match_date,'
                    ' kickoff_utc, status, round, leg, date_precision) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [(m[0], m[1], m[2], m[3], m[4], m[5], m[6], 'finished', m[7], m[8], m[9]) for m in EURO])
    con.executemany('insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)',
                    [(m, 'wikipedia', *s) for m, s in SCORES.items()])
    con.commit()
    con.close()
    return path


def make_store(db_path, mode, s) -> FeatureStore:
    return FeatureStore(open_readonly(db_path), mode, s)


def test_stage_2_blocks_euro_scores(db_path, settings):
    fs = make_store(db_path, '开发', settings)                      # 共用设置：阶段 2
    with pytest.raises(UsageNotAllowedError, match='第 8 阶段'):
        fetch_euro(fs, '2023-12-01 00:00:00')


def test_fetch_euro_only_known_scores_and_neutral(db_path, settings):
    s8 = stage8_settings(settings)
    fs = make_store(db_path, '开发', s8)
    df, _ = fetch_euro(fs, '2023-12-01 00:00:00')
    assert df['match_id'].tolist() == [1]                            # 2 号截止时间之后才踢
    df, _ = fetch_euro(fs, dev_end_cutoff(s8))
    assert sorted(df['match_id']) == [1, 2, 3]                       # 4 号是 2024 赛季，5 号是国内杯赛
    assert {int(k): bool(v) for k, v in df.set_index('match_id')['neutral'].items()} == {1: False, 2: False, 3: True}
    assert (df['known_at_utc'] <= dev_end_cutoff(s8)).all()


def test_dev_mode_blocks_exam_season(db_path, settings):
    fs = make_store(db_path, '开发', stage8_settings(settings))
    with pytest.raises(SeasonLockedError):
        fetch_euro(fs, '2024-09-30 00:00:00')


def test_exam_mode_not_for_research(db_path, settings):
    fs = make_store(db_path, '考试', stage8_settings(settings))
    with pytest.raises((UsageNotAllowedError, SeasonLockedError)):
        fetch_euro(fs, '2024-09-30 00:00:00')


# ======================= 欧战实力表 =======================
def euro_sim(n=3000, seed=0):
    '''按已知的 μ = 0.15、h = 0.25、各联赛 δ 模拟跨联赛欧战。返回 (联赛, 真实 δ, 评分表, 比赛表)。'''
    rng = np.random.default_rng(seed)
    leagues = ['E0', 'SP1', 'D1', 'I1', 'F1']
    td = dict(zip(leagues + ['out'], [0.2, 0.1, 0.0, -0.1, -0.2, -0.4]))
    table = {}
    for c in leagues:
        for i in range(10):
            a, d = rng.normal(0, 0.2, 2)
            table[f'{c}_{i}'] = (c, float(a), float(d))
    teams = list(table) + [f'X{i}' for i in range(20)]            # X 开头的是联赛外球队，不在评分表里
    rows = []
    for k in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        ch, ah, dh = table.get(h, ('out', 0.0, 0.0))
        ca, aa, da = table.get(a, ('out', 0.0, 0.0))
        neutral = k % 10 == 0
        g = td[ch] - td[ca]
        lh = np.exp(0.15 + (0 if neutral else 0.25) + ah - da + g)
        la = np.exp(0.15 + aa - dh - g)
        ko = (pd.Timestamp('2020-08-01') + pd.Timedelta(days=int(rng.integers(0, 1200)))).strftime('%Y-%m-%d %H:%M:%S')
        rows.append({'match_id': k, 'home_team': h, 'away_team': a, 'kickoff_earliest_utc': ko,
                     'home_goals': int(rng.poisson(lh)), 'away_goals': int(rng.poisson(la)), 'neutral': neutral})
    return leagues, td, table, pd.DataFrame(rows)


def test_euro_strength_recovers_simulated_deltas():
    leagues, td, table, m = euro_sim()
    es = EuroStrength(leagues, 0.5, 1461, None, 200)
    assert es.add(m, table) == len(m)
    es.fit('2024-01-01 00:00:00')
    assert es.ready and es.converged
    assert es.home == pytest.approx(0.25, abs=0.08)
    d = es.deltas()
    for g in leagues + ['out']:
        assert d['联赛外' if g == 'out' else g] == pytest.approx(td[g], abs=0.08), g      # 真实 δ 五个联赛的平均是 0
    assert d['E0'] > d['D1'] > d['F1'] > d['联赛外']
    assert sum(d[g] for g in leagues) == pytest.approx(0, abs=1e-9)


def test_without_delta_keeps_deltas_zero():
    leagues, _, table, m = euro_sim(n=1000)
    es0 = EuroStrength(leagues, 0.5, 1461, None, 200, with_delta=False)
    es0.add(m, table)
    es0.fit('2024-01-01 00:00:00')
    assert es0.ready and np.allclose(es0.theta[2:], 0)


def test_fit_uses_only_window_before_cutoff():
    leagues, _, table, m = euro_sim()
    es = EuroStrength(leagues, 0.5, 1461, None, 200)
    es.add(m, table)
    cutoff = '2022-01-01 00:00:00'
    es.fit(cutoff)
    ko = pd.to_datetime(m['kickoff_earliest_utc'])
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=1461)
    assert es.n_used == int(((ko >= start) & (ko <= pd.Timestamp(cutoff))).sum())
    assert es.n_used < len(m)


def test_too_few_matches_no_prediction():
    leagues, _, table, m = euro_sim(n=100)
    es = EuroStrength(leagues, 0.5, 1461, None, 200)
    es.add(m, table)
    es.fit('2024-01-01 00:00:00')
    assert not es.ready
    assert es.expected_goals('E0_0', 'F1_0', False, table) is None
    assert es.result_probs('E0_0', 'F1_0', False, table) is None


def test_add_skips_repeats_and_missing_ratings():
    leagues, _, table, m = euro_sim(n=50)
    es = EuroStrength(leagues, 0.5, 1461, None, 20)
    assert es.add(m, table) == 50
    assert es.add(m, table) == 0                                      # 同一场不记第二次
    one = pd.DataFrame([{'match_id': 999, 'home_team': 'E0_0', 'away_team': 'X1',
                         'kickoff_earliest_utc': '2021-01-01 20:00:00', 'home_goals': 1, 'away_goals': 0,
                         'neutral': False}])
    bad = dict(table)
    bad['E0_0'] = ('E0', float('nan'), float('nan'))                  # 这队联赛评分拿不到
    assert es.add(one, bad) == 0 and es.skipped == 1


def test_neutral_only_removes_home_advantage():
    leagues, _, table, m = euro_sim()
    es = EuroStrength(leagues, 0.5, 1461, None, 200)
    es.add(m, table)
    es.fit('2024-01-01 00:00:00')
    home = es.expected_goals('E0_0', 'F1_0', False, table)
    neutral = es.expected_goals('E0_0', 'F1_0', True, table)
    assert home[0] / neutral[0] == pytest.approx(np.exp(es.home))
    assert home[1] == pytest.approx(neutral[1])                       # 客队的进球数不受影响
    p = es.result_probs('E0_0', 'F1_0', False, table)
    assert p.shape == (3,) and p.sum() == pytest.approx(1)
    assert es.expected_margin('E0_0', 'F1_0', False, table) == pytest.approx(home[0] - home[1])


# ======================= dyn 的评分快照 =======================
def league_history(seasons, n_teams=10, seed=3) -> pd.DataFrame:
    '''造一个双循环联赛（E0）的历史比赛，比分是随机泊松，只用来给 dyn 喂数据。'''
    rng = np.random.default_rng(seed)
    rows, mid = [], 0
    for s in seasons:
        day = pd.Timestamp(f'{s}-08-15 14:00:00')
        k = 0
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                k += 1
                mid += 1
                ko = day + pd.Timedelta(days=int(k * 280 / (n_teams * (n_teams - 1))))
                hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
                rows.append({'match_id': mid, 'competition': 'E0', 'season': s, 'home_team': f'T{i}',
                             'away_team': f'T{j}', 'kickoff_earliest_utc': ko.strftime('%Y-%m-%d %H:%M:%S'),
                             'home_goals': hg, 'away_goals': ag,
                             'result': 'H' if hg > ag else ('D' if hg == ag else 'A')})
    return pd.DataFrame(rows)


def test_dyn_team_table_needs_burn_in():
    m = DynamicModel(min_matches=50)
    m.fit(league_history((2018, 2019, 2020, 2021)), '2022-07-01 00:00:00')
    members = {2021: {f'T{i}': 'E0' for i in range(10)}}
    t = dyn_team_table(m, ['E0'], members, 2021)
    assert len(t) == 10 and {v[0] for v in t.values()} == {'E0'}
    assert not np.isnan([v[1] for v in t.values()]).any()
    young = DynamicModel(min_matches=50)                               # 只有 1 个赛季：还在“让估计稳定”，不给数
    young.fit(league_history((2021,)), '2022-07-01 00:00:00')
    t2 = dyn_team_table(young, ['E0'], members, 2021)
    assert len(t2) == 10 and np.isnan([v[1] for v in t2.values()]).all()
    assert dyn_team_table(m, ['SP1'], members, 2021) == {}


# ======================= 残差研究和比较 =======================
def test_residual_rows_only_use_euro_known_before_t24():
    pred = pd.DataFrame([
        # A 主场 3:0，赛前预期净胜球 0.2 → 超预期 2.8（A 是 +2.8，B 是 −2.8）：会算进去
        {'home_team': 'A', 'away_team': 'B', 'home_goals': 3, 'away_goals': 0, 'margin': 0.2,
         'kickoff_earliest_utc': '2020-01-01 19:00:00', 'known_at_utc': '2020-01-01 22:00:00'},
        # 联赛 T-24 之后才踢：不能算
        {'home_team': 'A', 'away_team': 'B', 'home_goals': 5, 'away_goals': 0, 'margin': 0.0,
         'kickoff_earliest_utc': '2020-03-01 19:00:00', 'known_at_utc': '2020-03-01 22:00:00'},
        # T-24 之前已经开球，但比分 T-24 之后才能知道：不能算
        {'home_team': 'A', 'away_team': 'B', 'home_goals': 4, 'away_goals': 0, 'margin': 0.0,
         'kickoff_earliest_utc': '2020-01-31 22:00:00', 'known_at_utc': '2020-02-01 02:00:00'},
        # D 在联赛开球前 2 天踢了欧战：欧战后 4 天内
        {'home_team': 'D', 'away_team': 'E', 'home_goals': 1, 'away_goals': 0, 'margin': 0.0,
         'kickoff_earliest_utc': '2020-01-30 20:00:00', 'known_at_utc': '2020-01-30 23:00:00'},
    ])
    res = pd.DataFrame([
        {'match_id': 1, 'season': 2019, 'home_team': 'A', 'away_team': 'C', 'home_goals': 1, 'away_goals': 0,
         'exp_margin': 0.1, 't24': '2020-02-01 00:00:00'},
        {'match_id': 2, 'season': 2019, 'home_team': 'D', 'away_team': 'F', 'home_goals': 0, 'away_goals': 0,
         'exp_margin': 0.0, 't24': '2020-02-01 00:00:00'},
    ])
    rows = residual_rows(res, pred, [45, 90, 180], 4, 24).set_index('team')
    assert sorted(rows.index) == ['A', 'D']                            # C、F 过去没踢欧战，不进研究
    assert rows.loc['A', 'y'] == pytest.approx(0.9)
    for col in ('x45', 'x90', 'x180'):
        assert rows.loc['A', col] == pytest.approx(2.8)
    assert not rows.loc['A', 'after_euro']
    assert rows.loc['D', 'x90'] == pytest.approx(1.0) and rows.loc['D', 'y'] == pytest.approx(0.0)
    assert rows.loc['D', 'after_euro']


def test_slope_boot_recovers_known_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    blocks = np.repeat(np.arange(100), 20)
    b, a, lo, hi = slope_boot(x, 0.5 * x + rng.normal(size=2000), blocks, 400, 0.95, 7)
    assert b == pytest.approx(0.5, abs=0.1) and lo < 0.5 < hi and lo > 0
    b0, _, lo0, hi0 = slope_boot(x, rng.normal(size=2000), blocks, 400, 0.95, 7)    # 没有关系的数据
    assert abs(b0) < 0.1 and lo0 < hi0


def test_main_compare_verdicts(settings):
    n = 300
    dates = pd.date_range('2018-08-01', periods=n, freq='3D').strftime('%Y-%m-%d')
    pred = pd.DataFrame({'result': ['H'] * n, 'match_date': dates, 'pH': 0.8, 'pD': 0.1, 'pA': 0.1,
                         'qH': 0.4, 'qD': 0.3, 'qA': 0.3})
    assert main_compare(pred, settings)['结论'].tolist() == ['更准'] * 3
    swapped = pred.rename(columns={'pH': 'qH', 'pD': 'qD', 'pA': 'qA', 'qH': 'pH', 'qD': 'pD', 'qA': 'pA'})
    assert main_compare(swapped, settings)['结论'].tolist() == ['更差'] * 3
    same = pred.assign(pH=pred['qH'], pD=pred['qD'], pA=pred['qA'])
    assert main_compare(same, settings)['结论'].tolist() == ['没差别'] * 3


def test_leg_kind():
    assert leg_kind('G', 0) == '小组赛'
    assert leg_kind('Q2', 1) == '第1回合'
    assert leg_kind('Q2', 2) == '第2回合'
    assert leg_kind('F', 0) == '单场'


def test_dyn_team_table_uses_euro_season_membership():
    """新季首轮以前：降级队不能沿用上季联赛组；升班马和仍在联赛的队使用 dyn 的赛前规则。"""
    m = DynamicModel(min_matches=50)
    m.fit(league_history((2018, 2019, 2020, 2021)), '2022-07-01 00:00:00')
    members = {2022: {**{f'T{i}': 'E0' for i in range(1, 10)}, 'Promoted': 'E0'}}
    table = dyn_team_table(m, ['E0'], members, 2022)
    assert 'T0' not in table  # 已降级，即使 dyn 还保存上一季的状态，也必须归到联赛外。
    assert set(table) == set(members[2022])
    for team in ('T1', 'Promoted'):
        assert table[team][0] == 'E0'
        assert table[team][1:] == pytest.approx(m._strength(m.leagues['E0'], 2022, team))
    es = EuroStrength(['E0'])
    assert es._lookup(table, 'T0') == (1, 0.0, 0.0)


def test_known_top_league_without_rating_is_unavailable():
    """明确属于五大联赛但 dyn 尚无该联赛评分，不能伪装成评分为零的联赛外队。"""
    m = DynamicModel(min_matches=50)
    table = dyn_team_table(m, ['E0'], {2022: {'A': 'E0'}}, 2022)
    assert table['A'][0] == 'E0' and np.isnan(table['A'][1:]).all()


def test_season_memberships_uses_schedule_without_results(db_path, settings):
    """当季成员来自取数层赛程，未完赛也可识别；次级联赛球队不进五大联赛成员表。"""
    con = sqlite3.connect(db_path)
    con.executemany('insert into matches (match_id, competition, season, home_team, away_team, match_date,'
                    ' kickoff_utc, status) values (?, ?, 2023, ?, ?, ?, ?, ?)', [
        (20, 'E0', 'A', 'B', '2023-08-12', '2023-08-12 14:00', 'scheduled'),
        (21, 'E1', 'C', 'X', '2023-08-12', '2023-08-12 14:00', 'scheduled')])
    con.commit()
    con.close()
    fs = make_store(db_path, '开发', stage8_settings(settings))
    assert EM.season_memberships(fs, 2023, 2023) == {2023: {'A': 'E0', 'B': 'E0'}}
