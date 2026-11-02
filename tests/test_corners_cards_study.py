'''第 10 阶段角球、黄牌研究（src/evaluate/corners_cards_study.py）的自动检查。

用内存里的临时数据库造一个小联赛，不碰真实数据库、不碰主项目 data/：
  E0，6 个队，每个赛季双循环 10 轮、每轮 3 场（共 30 场），每周六一轮；2021、2022、2023 三个赛季。
  2021 只当历史；2022 用来选岭惩罚；2023 用来评。2023 有一个新队 G（以前没出现过）。
  每场都有 football-data 来源的角球、犯规、黄牌、红牌和裁判（三个裁判轮流）。
  2023 第 1 轮的三场另外有一份懂球帝来源“全是 0”的记录（两边填得一样满，按来源顺序仍取 football-data）；
  2023 第 10 轮第 1 场只有懂球帝来源的 0（真实数据里那是“没有数据”，研究程序必须排除，不能当成 0 个角球）。
  另有一场 2024（期末考试赛季）的比赛：开发模式下不能碰。
'''
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.evaluate import corners_cards_study as cc
from src.features import FeatureAccessError, FeatureStore, UsageNotAllowedError

ATT_C = {'A': 0.25, 'B': 0.10, 'C': 0.0, 'D': -0.10, 'E': -0.20, 'F': -0.25, 'G': 0.0}   # 每队的角球倾向（对数）
ATT_Y = {k: -0.5 * v for k, v in ATT_C.items()}                                          # 每队的黄牌倾向（对数）
REF_EFFECT = {'R1': 0.30, 'R2': 0.0, 'R3': -0.30}                                        # 裁判对黄牌的影响（对数）
SEASON_START = {2021: '2021-08-07', 2022: '2022-08-06', 2023: '2023-08-05'}              # 都是星期六
TEAMS = {2021: list('ABCDEF'), 2022: list('ABCDEF'), 2023: list('ABCDEG')}
ONLY_DQD = 2023090          # 2023 第 10 轮第 1 场：只有懂球帝来源
EXAM_MATCH = 900
STATS_SQL = '''insert into match_stats (match_id, source, home_goals, away_goals, result, home_corners, away_corners,
    home_fouls, away_fouls, home_yellow, away_yellow, home_red, away_red, referee)
    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''


def round_robin(teams):
    '''双循环赛程（轮转法）：返回 [(轮次, [(主, 客), ...]), ...]，前一半和后一半主客互换。'''
    t = list(teams)
    half = []
    for r in range(len(t) - 1):
        pairs = [(t[i], t[-1 - i]) for i in range(len(t) // 2)]
        half.append([(h, a) if r % 2 == 0 else (a, h) for h, a in pairs])
        t = [t[0], t[-1], *t[1:-1]]
    return list(enumerate(half + [[(a, h) for h, a in rnd] for rnd in half]))


def build_rows(override=None):
    '''造比赛和统计。override：{比赛号: (主队角球, 客队角球, 主队黄牌, 客队黄牌)}，用来故意改某一场的答案。
    不管改不改，随机数都照常抽，所以只有被改的那一场不同。'''
    rng = np.random.default_rng(10)
    matches, fd, dqd = [], [], []
    n = 0
    for season, teams in TEAMS.items():
        start = pd.Timestamp(SEASON_START[season])
        for rnd, games in round_robin(teams):
            day = (start + pd.Timedelta(days=7 * rnd)).strftime('%Y-%m-%d')
            for i, (h, a) in enumerate(games):
                mid = season * 1000 + rnd * 10 + i
                ko = [f'{day} 14:00', f'{day} 16:30', None][i]      # 第 3 场只有日期
                matches.append((mid, 'E0', season, h, a, day, ko, 'finished'))
                ref = ['R1', 'R2', 'R3'][n % 3]
                n += 1
                r = REF_EFFECT[ref]
                hc = int(rng.poisson(np.exp(1.75 + ATT_C[h])))
                ac = int(rng.poisson(np.exp(1.55 + ATT_C[a])))
                hy = int(rng.poisson(np.exp(0.75 + ATT_Y[h] + r)))
                ay = int(rng.poisson(np.exp(0.95 + ATT_Y[a] + r)))
                if override and mid in override:
                    hc, ac, hy, ay = override[mid]
                if mid != ONLY_DQD:
                    fd.append((mid, 'football-data', 0, 0, 'D', hc, ac, 11, 12, hy, ay, 0, 0, ref))
                if mid == ONLY_DQD or (season == 2023 and rnd == 0):
                    dqd.append((mid, 'dongqiudi', 0, 0, 'D', 0, 0, 0, 0, 0, 0, 0, 0, None))
    matches.append((EXAM_MATCH, 'E0', 2024, 'A', 'B', '2024-08-17', '2024-08-17 14:00', 'finished'))
    fd.append((EXAM_MATCH, 'football-data', 0, 0, 'D', 6, 5, 10, 10, 2, 1, 0, 0, 'R1'))
    return matches, fd, dqd


def make_fs(settings, override=None):
    matches, fd, dqd = build_rows(override)
    c = sqlite3.connect(':memory:')
    create_tables(c)
    c.executemany('''insert into teams (team_id, country) values (?, 'England')''', [(t,) for t in ATT_C])
    c.executemany('''insert into matches (match_id, competition, season, home_team, away_team, match_date,
        kickoff_utc, status) values (?, ?, ?, ?, ?, ?, ?, ?)''', matches)
    c.executemany(STATS_SQL, fd + dqd)
    c.commit()
    return FeatureStore(c, '开发', settings)


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s['compare']['n_boot'] = 200          # 测试里少抽几次，快一点
    s['corners_cards'].update(leagues=['E0'], min_train_matches=20, baseline_matches=60, train_matches=90,
                              ridge_grid=[5, 20, 80], select_seasons=[2022, 2022], eval_seasons=[2023, 2023],
                              ref_min_matches=5, calibration_bins=4)
    return s


def sim_train(target, n=1500, seed=1):
    '''造一份足够大的训练数据：6 个队随机对阵，角球、黄牌按上面的倾向和裁判影响抽。'''
    rng = np.random.default_rng(seed)
    teams = np.array(list('ABCDEF'))
    home, away = rng.choice(teams, n), rng.choice(teams, n)
    keep = home != away
    home, away = home[keep], away[keep]
    refs = rng.choice(['R1', 'R2', 'R3'], len(home))
    att = ATT_C if target == 'corners' else ATT_Y
    r = np.array([REF_EFFECT[x] for x in refs]) if target == 'yellow' else 0.0
    hc, ac = cc.TARGETS[target]
    mh = np.exp(1.55 + np.array([att[x] for x in home]) + r)
    ma = np.exp(1.40 + np.array([att[x] for x in away]) + r)
    return pd.DataFrame({'home_team': home, 'away_team': away, 'referee': refs,
                         'kickoff_earliest_utc': pd.date_range('2020-01-01', periods=len(home), freq='h').strftime('%Y-%m-%d %H:%M:%S'),
                         hc: rng.poisson(mh), ac: rng.poisson(ma)})


FIT_CFG = {'half_life_days': 1e9, 'ref_ridge': 20, 'ref_min_matches': 5}   # 半衰期很大 = 几乎不衰减
CUT = pd.Timestamp('2020-04-01')


# ---------- 计数分布 ----------
def test_count_nll_matches_scipy():
    y = np.array([0, 3, 9])
    mu = np.array([2.0, 3.5, 8.0])
    np.testing.assert_allclose(cc.count_nll(y, mu, 0.0), -stats.poisson.logpmf(y, mu))
    alpha = 0.2
    r = 1 / alpha
    np.testing.assert_allclose(cc.count_nll(y, mu, alpha), -stats.nbinom.logpmf(y, r, r / (r + mu)))
    # 离散参数趋近 0 时，负二项就是泊松
    np.testing.assert_allclose(cc.count_nll(y, mu, 1e-5), -stats.poisson.logpmf(y, mu), rtol=1e-3)


def test_nb_alpha_moment_estimate():
    rng = np.random.default_rng(3)
    mu, alpha = 10.0, 0.3
    r = 1 / alpha
    y = rng.negative_binomial(r, r / (r + mu), 200000)
    assert cc.nb_alpha(y, np.full(len(y), mu), np.ones(len(y))) == pytest.approx(alpha, abs=0.03)
    y2 = rng.poisson(mu, 200000)
    assert cc.nb_alpha(y2, np.full(len(y2), mu), np.ones(len(y2))) == pytest.approx(0.0, abs=0.02)
    # 比泊松还集中（每场都正好等于均值）：记 0，不能是负数
    assert cc.nb_alpha(np.full(50, 10.0), np.full(50, 10.0), np.ones(50)) == 0.0


# ---------- 拟合和预测 ----------
@pytest.mark.parametrize('target', ['corners', 'yellow'])
def test_fit_recovers_team_means(target):
    train = sim_train(target)
    fit = cc.fit_league(train, target, CUT, FIT_CFG, lam=1.0)
    att = ATT_C if target == 'corners' else ATT_Y
    base_h, base_a = (1.55, 1.40)
    # 黄牌的真实均值里含裁判影响，三个裁判平均下来 exp(r) 的平均
    ref_avg = 1.0 if target == 'corners' else float(np.mean([np.exp(v) for v in REF_EFFECT.values()]))
    for h in 'ABCDEF':
        for a in 'ABCDEF':
            if h == a:
                continue
            true = (np.exp(base_h + att[h]) + np.exp(base_a + att[a])) * ref_avg
            got = cc.predict_means(fit, np.array([h]), np.array([a]))[0]
            assert got == pytest.approx(true, rel=0.12)


def test_ridge_shrinks_team_effects_to_zero():
    fit = cc.fit_league(sim_train('corners'), 'corners', CUT, FIT_CFG, lam=1e8)
    assert np.abs(fit.att).max() < 1e-3 and np.abs(fit.de).max() < 1e-3


def test_new_team_gets_neutral_effect():
    fit = cc.fit_league(sim_train('corners'), 'corners', CUT, FIT_CFG, lam=5.0)
    got = cc.predict_means(fit, np.array(['ZZ']), np.array(['YY']))[0]
    assert got == pytest.approx(np.exp(fit.c + fit.h) + np.exp(fit.c))


def test_referee_effect_recovered_and_rare_referee_dropped():
    train = sim_train('yellow')
    train.loc[train.index[:3], 'referee'] = 'R9'          # 只出现 3 场，不到 ref_min_matches
    fit = cc.fit_league(train, 'yellow', CUT, FIT_CFG, lam=5.0, use_ref=True)
    assert list(fit.refs) == ['R1', 'R2', 'R3']
    assert fit.ref[0] > fit.ref[1] > fit.ref[2]
    assert fit.ref[0] == pytest.approx(0.3, abs=0.1) and fit.ref[2] == pytest.approx(-0.3, abs=0.1)
    home, away = np.array(['A']), np.array(['B'])
    same = cc.predict_means(fit, home, away, np.array(['R9'], dtype=object))
    assert same == pytest.approx(cc.predict_means(fit, home, away))      # 没有裁判项的裁判按 0 算


def test_older_matches_weigh_less():
    '''同样的数据，半衰期越短，越靠近截止时间的比赛说了算：把最近一段时间的角球改成很多，预测应该跟着涨。'''
    train = sim_train('corners')
    recent = train.copy()
    recent.loc[recent.index[-300:], ['home_corners', 'away_corners']] += 4
    slow = cc.fit_league(recent, 'corners', CUT, {**FIT_CFG, 'half_life_days': 1e9}, lam=5.0)
    fast = cc.fit_league(recent, 'corners', CUT, {**FIT_CFG, 'half_life_days': 3.0}, lam=5.0)
    args = (np.array(['C']), np.array(['D']))
    assert cc.predict_means(fast, *args)[0] > cc.predict_means(slow, *args)[0] + 1.0


# ---------- 判定规则、赛季检查 ----------
def test_final_label_rules():
    assert cc.final_label(cc.BETTER, cc.BETTER) == '更准'
    assert cc.final_label(cc.BETTER, cc.SAME) == '可能更准，也可能是巧合'
    assert cc.final_label(cc.WORSE, cc.WORSE) == '更差'
    assert cc.final_label(cc.WORSE, cc.SAME) == '可能更差，也可能是巧合'
    assert cc.final_label(cc.SAME, cc.SAME) == '没差别'


def test_check_seasons_refuses_exam_seasons(settings):
    cc.check_seasons(settings, [2022, 2022], [2023, 2023])
    with pytest.raises(ValueError, match='不是开发赛季'):
        cc.check_seasons(settings, [2022, 2022], [2023, 2024])
    with pytest.raises(ValueError, match='不是开发赛季'):
        cc.check_seasons(settings, [2024, 2024], [2025, 2025])
    with pytest.raises(ValueError, match='之前'):
        cc.check_seasons(settings, [2023, 2023], [2022, 2022])


def test_exam_season_cannot_be_prediction_target(settings):
    fs = make_fs(settings)
    with pytest.raises(FeatureAccessError):
        cc.season_targets(fs, ['E0'], 2023, 2024, settings['corners_cards']['usage'])


# ---------- 取数层的权限和来源 ----------
def test_usage_and_source_rules(settings):
    fs = make_fs(settings)
    with pytest.raises(UsageNotAllowedError):       # 研究用的数据类，别的用途一律不给
        fs.historical_matches('2023-12-31', 'A版输入', competition='E0', data=[cc.CORNERS_DATA])
    hist = fs.historical_matches('2023-12-31', '研究实验', competition='E0', data=[cc.CORNERS_DATA])
    assert EXAM_MATCH not in set(hist['match_id'])             # 考试赛季的比赛不会出现
    src = hist.set_index('match_id')['stats_source']
    assert src.loc[2023000] == 'football-data'                  # 两个来源一样满，按顺序取 football-data
    assert src.loc[ONLY_DQD] == 'dongqiudi'                     # 只有懂球帝的，研究程序要排除它


def test_missing_fouls_do_not_drop_valid_counts_when_dqd_has_zero_placeholders(settings):
    '''football-data 只缺犯规时，角球、黄牌训练和答案都仍用它，不能被更完整的懂球帝 0 挤掉。'''
    baseline = make_fs(settings)
    missing_fouls = make_fs(settings)
    missing_fouls.conn.execute(
        "update match_stats set home_fouls = NULL where match_id in (2023000, 2023001, 2023002) "
        "and source = 'football-data'")
    missing_fouls.conn.commit()
    cfg = settings['corners_cards']
    targets = cc.season_targets(baseline, ['E0'], 2023, 2023, cfg['usage'])
    ridges = {tg: [20.0] for tg in cc.TARGETS}
    before = cc.walk(baseline, cfg, targets, ridges, True, True)
    after = cc.walk(missing_fouls, cfg, targets, ridges, True, True)
    pd.testing.assert_frame_equal(before.preds, after.preds)
    outcomes = cc.fetch_outcomes(missing_fouls, cfg, targets, with_ref=True)
    row = outcomes.set_index('match_id').loc[2023000]
    assert row['stats_source'] == 'football-data' and row['home_corners'] > 0
    assert pd.isna(outcomes.set_index('match_id').loc[ONLY_DQD, 'stats_source'])
    score_before, excluded_before = cc.score(
        cc.add_ref_predictions(before, cc.fetch_outcomes(baseline, cfg, targets, with_ref=True)),
        cc.fetch_outcomes(baseline, cfg, targets, with_ref=True))
    score_after, excluded_after = cc.score(cc.add_ref_predictions(after, outcomes), outcomes)
    pd.testing.assert_frame_equal(score_before, score_after)
    assert excluded_before == excluded_after
    assert excluded_after['corners']['没有答案被排除的场数'] == 1


# ---------- 不偷看未来 ----------
def test_own_answer_never_leaks(settings):
    '''把 2023 第 6 轮某一场的答案改得很夸张：它自己和更早的比赛预测必须一模一样；后面的比赛（已经能知道它了）才会变。'''
    mid = 2023051
    cfg = settings['corners_cards']
    fs1 = make_fs(settings)
    fs2 = make_fs(settings, override={mid: (40, 40, 15, 15)})
    t = cc.season_targets(fs1, ['E0'], 2023, 2023, cfg['usage'])
    ridges = {tg: [20.0] for tg in cc.TARGETS}
    p1 = cc.walk(fs1, cfg, t, ridges, True, False).preds
    p2 = cc.walk(fs2, cfg, t, ridges, True, False).preds
    m = p1.merge(p2, on=['match_id', 'target', 'model'], suffixes=('_1', '_2'))
    assert len(m) == len(p1) == len(p2) and len(m) > 0
    early = m[m['match_id'] < 2023060]
    late = m[m['match_id'] >= 2023070]
    np.testing.assert_allclose(early['mu_1'], early['mu_2'], rtol=1e-12)
    assert (late['mu_1'] - late['mu_2']).abs().max() > 1e-6      # 防止测试是空的：后面的比赛确实用上了这场的结果


# ---------- 整个流程 ----------
def test_run_study_end_to_end(settings, tmp_path):
    fs = make_fs(settings)
    res = cc.run_study(fs)
    assert res['meta']['select_seasons'] == [2022, 2022] and res['meta']['eval_seasons'] == [2023, 2023]
    assert set(res['chosen_ridge']) == {'corners', 'yellow'}
    assert all(v in settings['corners_cards']['ridge_grid'] for v in res['chosen_ridge'].values())
    scored = res['scored']
    assert set(scored['model']) == {cc.M_BASE, cc.M_TEAM, cc.M_NB, cc.M_REF}
    assert scored['mu'].notna().all() and (scored['mu'] > 0).all() and np.isfinite(scored['nll']).all()
    assert set(scored['season']) == {2023}                      # 只给 2023 打分，考试赛季一场都没有
    # 只有懂球帝来源的那一场：有预测、没答案，被排除，而不是当成 0 个角球
    for tg in cc.TARGETS:
        assert res['excluded'][tg]['没有答案被排除的场数'] == 1
    assert ONLY_DQD not in set(scored['match_id'])
    main = res['main']
    assert len(main) == 5 and (main['n'] == 29).all()
    assert set(main['结论']) <= {'更准', '可能更准，也可能是巧合', '没差别', '可能更差，也可能是巧合', '更差'}
    assert (main['ci99_low'] <= main['ci95_low'] + 1e-12).all() and (main['ci99_high'] >= main['ci95_high'] - 1e-12).all()
    assert len(res['calibration']) > 0 and (res['dispersion']['皮尔逊残差平方平均'] > 0).all()
    assert len(res['referees']) == 3                              # 三个裁判，每个都够 ref_min_matches
    out = tmp_path / 'out'
    cc.write_outputs(res, out)
    for name in ('report.txt', 'main.csv', 'scored.csv', 'numbers.json'):
        assert (out / name).exists()
    assert '5 个主比较' in (out / 'report.txt').read_text(encoding='utf-8')
