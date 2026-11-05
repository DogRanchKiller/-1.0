'''第 11 阶段（国际赛开发回测，工作树 INC）的自动检查：国家队 Elo（src/models/intl_elo.py）和研究程序（src/evaluate/intl_study.py）。

不碰真实数据库：用手造的小数据，数据库文件放在 pytest 的 tmp_path 里。
检查的东西：
  - 公式和手算一样（We、G、K、主场优势、P 四舍五入）；中立场不加主场优势（intl_elo_home 照加）；
  - 点球大战按平局算；友谊赛更新评分、但不进有序 logit、不打分；没写进分档的赛事名报错；
  - 有序 logit 只用截止时间前、往前 8 年以内的正式比赛；
  - 研究程序在手造数据库上从头跑到尾；不偷看（改掉截止时间以后的比分，之前的预测一点不变）；
    格子固定（--first 不同，同一场的预测一样）；开发模式碰 2024 年就报错。

INA 合并前的临时做法（合并后改用真的）：
  - match_venue 表：INA 合并前的临时建表，合并后改用 schema（create_tables 会建）。
  - day_world 的“什么时候能知道”：先在这里替 known_time.compute_known_times 补上（规则是和 INA 约好的：
    最早开球 = 前一天 10:00 UTC，最晚开球 = 后一天 12:00 UTC，比分能知道 = 最晚开球 + 3 小时）；
    INA 合并后去掉替身，用取数层自己的。
  - 取数层的 match_venues：先在这里放一个替身（返回 match_id、source、neutral、city、country、tournament）；
    INA 合并后去掉替身，用取数层自己的。
  - settings 的 leagues 加 intl: {INT: 国际赛}：先在测试里复制一份设置加上；INA 合并后设置里本来就有。
'''
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

import src.features.store as store_mod
from src.clean.schema import create_tables
from src.common.config import load_settings
from src.evaluate import intl_study
from src.features import FeatureStore, SeasonLockedError, open_readonly
from src.features.known_time import TIME_FORMAT
from src.models.intl_elo import IntlElo, UnknownTournamentError, round_half_away

# 测试用的 K 分档（真的分档在 settings.yaml 的 intl_study.k_by_tournament）
K_TEST = {60: ['FIFA World Cup'], 40: ['Test Cup qualification'], 30: ['Test Cup'], 20: ['Friendly']}
TEAMS = [f'T{i} NT' for i in range(8)]


def make_elo(**kw) -> IntlElo:
    return IntlElo(K_TEST, **kw)


# ======================= 公式：和手算一样 =======================
def test_formula_by_hand():
    '''两队都是 1500，主场、友谊赛（K 20）、2:0：
    dr = 0 + 100 = 100，We = 1 ÷ (10^(−100/400) + 1) = 0.6401，G = 1.5（赢 2 球），
    P = 20 × 1.5 × (1 − 0.6401) = 10.80 → 四舍五入 11。主队 1511，客队 1489。'''
    e = make_elo()
    we = 1 / (10 ** (-100 / 400) + 1)
    assert we == pytest.approx(0.6401, abs=1e-4)
    p = e.update('2000-01-01', 'A', 'B', 2, 0, False, 'Friendly', '2000-01-02 15:00:00')
    assert p == 11
    assert e.rating('A') == 1511 and e.rating('B') == 1489
    # 再来一场：中立场、世界杯（K 60）、B 4:1 赢 A（B 是主队）。
    # dr = 1489 − 1511 = −22（中立场不加 100），We = 1 ÷ (10^(22/400) + 1)，G = (11 + 3) ÷ 8 = 1.75
    p2 = e.update('2000-02-01', 'B', 'A', 4, 1, True, 'FIFA World Cup', '2000-02-02 15:00:00')
    we2 = 1 / (10 ** (22 / 400) + 1)
    assert we2 == pytest.approx(0.4684, abs=1e-4)                      # P = 105 × 0.5316 = 55.82 → 56
    assert p2 == round_half_away(60 * 1.75 * (1 - we2)) == 56
    assert e.rating('B') == 1489 + 56 and e.rating('A') == 1511 - 56
    # 赢 5 球：G = (11 + 5) ÷ 8 = 2；输了 P 是负数，四舍五入往离 0 远的方向
    f = make_elo(round_points=False)
    p3 = f.update('2000-01-01', 'A', 'B', 0, 5, False, 'Test Cup', '2000-01-02 15:00:00')
    assert p3 == pytest.approx(30 * 2 * (0 - we))
    assert round_half_away(-10.5) == -11 and round_half_away(10.5) == 11 and round_half_away(10.49) == 10


def test_neutral_no_home_advantage():
    '''intl_elo：中立场不加主场优势；intl_elo_home：一律加。非中立场两版一样。'''
    e, h = make_elo(use_neutral=True), make_elo(use_neutral=False)
    assert e.rating_diff('A', 'B', True) == 0 and h.rating_diff('A', 'B', True) == 100
    assert e.rating_diff('A', 'B', False) == 100 and h.rating_diff('A', 'B', False) == 100
    pe = e.update('2000-01-01', 'A', 'B', 1, 1, True, 'Test Cup', '2000-01-02 15:00:00')
    ph = h.update('2000-01-01', 'A', 'B', 1, 1, True, 'Test Cup', '2000-01-02 15:00:00')
    assert pe == 0                                   # 中立场两队一样强、打平：不加不减
    assert ph == round_half_away(30 * (0.5 - 1 / (10 ** (-100 / 400) + 1))) == -4   # 照加主场优势：主队打平算吃亏
    assert e.rec_diff == [0] and h.rec_diff == [100]  # 进有序 logit 的评分差也跟着版本走


def test_shootout_counts_as_draw():
    '''点球大战：martj42 记加时后的平局比分（点球只在 cup_details），Elo 按平局算，和普通平局完全一样。'''
    a, b = make_elo(), make_elo()
    for e in (a, b):
        e.update('2000-01-01', 'A', 'B', 3, 0, False, 'Test Cup', '2000-01-02 15:00:00')   # 先拉开差距
    pa = a.update('2000-02-01', 'A', 'B', 1, 1, False, 'Test Cup', '2000-02-02 15:00:00')   # 1:1，点球 B 赢
    pb = b.update('2000-02-01', 'A', 'B', 1, 1, False, 'Test Cup', '2000-02-02 15:00:00')   # 1:1，普通平局
    assert pa == pb < 0 and a.ratings == b.ratings
    assert a.rec_y[-1] == 1                           # 有序 logit 里也是“平”


def test_friendly_updates_but_not_fitted():
    '''友谊赛照样更新评分（K 20），但不进有序 logit 的记录。'''
    e = make_elo()
    e.update('2000-01-01', 'A', 'B', 1, 0, False, 'Friendly', '2000-01-02 15:00:00')
    assert e.rating('A') > 1500 and e.n_updates == 1
    assert e.rec_y == [] and e.fit_window('2001-01-01 00:00:00')[0].size == 0
    e.update('2000-03-01', 'A', 'B', 1, 0, False, 'Test Cup', '2000-03-02 15:00:00')
    assert len(e.rec_y) == 1 and e.is_official('Test Cup') and not e.is_official('Friendly')


def test_unknown_tournament_raises():
    '''没写进分档的赛事名：直接报错，不猜，评分不动；同一个赛事写进两档也报错；研究程序走之前先查一遍。'''
    e = make_elo()
    with pytest.raises(UnknownTournamentError, match='Mystery Cup'):
        e.update('2000-01-01', 'A', 'B', 1, 0, False, 'Mystery Cup', '2000-01-02 15:00:00')
    assert e.ratings == {} and e.n_updates == 0
    with pytest.raises(ValueError, match='写了两次'):
        IntlElo({30: ['Test Cup'], 40: ['Test Cup']})
    cfg = {'k_by_tournament': K_TEST}
    intl_study.check_k_table(['Friendly', 'Test Cup'], cfg)
    with pytest.raises(UnknownTournamentError, match='Mystery Cup'):
        intl_study.check_k_table(['Friendly', 'Mystery Cup'], cfg)


def test_real_settings_k_table():
    '''真的设置：分档能建起来（没有重复）；友谊赛 20、世界杯 60；没写的赛事报错。'''
    cfg = load_settings()['intl_study']
    e = IntlElo.from_settings(cfg, True, 'intl_elo')
    assert e.k_for('Friendly') == 20 and e.k_for('FIFA World Cup') == 60
    assert cfg['home_advantage'] == 100 and cfg['start_rating'] == 1500 and cfg['fit_years'] == 8
    with pytest.raises(UnknownTournamentError):
        e.k_for('No Such Tournament')


def test_logit_uses_only_official_before_cutoff():
    '''有序 logit：只用正式比赛、比分在截止时间前已经知道、比赛日期在截止往前 8 年以内的。'''
    e = make_elo(min_fit_n=1)
    e.update('2011-06-01', 'A', 'B', 1, 0, False, 'Test Cup', '2011-06-02 15:00:00')   # 8 年以前：不用
    e.update('2013-06-01', 'A', 'B', 0, 0, False, 'Test Cup', '2013-06-02 15:00:00')   # 用
    e.update('2015-06-01', 'A', 'B', 2, 0, False, 'Friendly', '2015-06-02 15:00:00')   # 友谊赛：不用
    e.update('2018-06-01', 'B', 'A', 0, 1, True, 'Test Cup', '2018-06-02 15:00:00')    # 用
    e.update('2019-12-31', 'A', 'B', 1, 2, False, 'Test Cup', '2020-01-01 15:00:00')   # 日期在前，但比分截止后才知道：不用
    d, y = e.fit_window('2020-01-01 00:00:00')
    assert y.tolist() == [1, 2] and len(d) == 2
    e.min_fit_n = 3                                                    # 能用的只有 2 场，比赛不够：不出预测
    assert not e.fit('2020-01-01 00:00:00') and e.fit_n == 2 and e.theta is None
    assert e.predict('A', 'B', False) is None


# ======================= INA 合并前的替身 =======================
_real_compute_known_times = store_mod.compute_known_times


def _known_times_with_day_world(matches: pd.DataFrame, feature_settings: dict) -> pd.DataFrame:
    '''替身：day_world 的比赛按和 INA 约好的规则算；其余照取数层原来的算。INA 合并后去掉，用取数层自己的。'''
    out = _real_compute_known_times(matches, feature_settings)
    if 'date_precision' not in matches.columns:
        return out
    w = (matches['date_precision'] == 'day_world').to_numpy()
    if w.any():
        d = pd.to_datetime(matches.loc[w, 'match_date'], format='%Y-%m-%d')
        earliest, latest = d - pd.Timedelta(hours=14), d + pd.Timedelta(hours=36)   # 前一天 10:00、后一天 12:00（UTC）
        out.loc[w, 'kickoff_earliest_utc'] = earliest.dt.strftime(TIME_FORMAT).to_numpy()
        out.loc[w, 'kickoff_latest_utc'] = latest.dt.strftime(TIME_FORMAT).to_numpy()
        out.loc[w, 'close_known_utc'] = latest.dt.strftime(TIME_FORMAT).to_numpy()
        out.loc[w, 'result_known_utc'] = (latest + pd.Timedelta(hours=3)).dt.strftime(TIME_FORMAT).to_numpy()
    return out


def _match_venues_stand_in(self, cutoff, usage, competition=None, since=None) -> pd.DataFrame:
    '''替身：取数层的 match_venues（赛程信息，最早开球在截止时间之前的比赛）。INA 合并后去掉，用取数层自己的。'''
    from src.features.known_time import to_utc_text
    self.usage.check('比赛信息', '本场', usage)
    sql = (f'select v.match_id, v.source, v.neutral, v.city, v.country, v.tournament, k.season_role, m.season '
           f'from match_venue v join matches m using (match_id) join temp.{self._kt} k using (match_id) '
           f'where k.kickoff_earliest_utc <= ?')
    params = [to_utc_text(cutoff)]
    if competition is not None:
        sql += ' and m.competition = ?'
        params.append(competition)
    if since is not None:
        sql += ' and k.kickoff_earliest_utc >= ?'
        params.append(to_utc_text(since))
    df = pd.read_sql(sql + ' order by v.match_id', self.conn, params=params)
    return self._check_rows_mode(df, usage).drop(columns='season')


@pytest.fixture(scope='module', autouse=True)
def ina_stand_ins():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(store_mod, 'compute_known_times', _known_times_with_day_world)
        mp.setattr(FeatureStore, 'match_venues', _match_venues_stand_in, raising=False)
        yield


@pytest.fixture(scope='module')
def settings():
    '''复制一份设置：leagues 加 intl（INA 合并后设置里本来就有），intl_study 换成测试用的小分档、小门槛。'''
    s = copy.deepcopy(load_settings())
    s['leagues']['intl'] = {'INT': '国际赛'}
    c = s['intl_study']
    c['k_by_tournament'] = copy.deepcopy(K_TEST)
    c['tournament_groups'] = {'世界杯': ['FIFA World Cup'], '预选赛': ['Test Cup qualification']}
    c['min_fit_n'] = 50
    c['main_years'] = [2021, 2023]
    c['neutral_years'] = [2019, 2023]
    c['min_total_n'] = 100
    c['min_year_n'] = 30
    s['compare']['n_boot'] = 200
    return s


# ======================= 手造数据库 =======================
SHOOTOUT_IDS = (40, 41, 42)          # 这几场改成 Test Cup 的 1:1，点球决胜（cup_details 记点球）
DECOY_IDS = (60, 61, 62)             # 这几场另有一个来源 dongqiudi 的错比分，研究只认 martj42


def synthetic_matches(seed: int = 7) -> pd.DataFrame:
    '''8 支队，2012-01-01 起每 4 天一场，到 2023 年底；另加 2023-12-31、2024-01-01 两场（看年底的边界）。
    主客队、中立场、赛事随机；比分按“真实实力差”抽 Poisson。'''
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(-0.6, 0.6, len(TEAMS))))
    dates = list(pd.date_range('2012-01-01', '2023-12-20', freq='4D').strftime('%Y-%m-%d')) + ['2023-12-31', '2024-01-01']
    tours = rng.choice(['Friendly', 'Test Cup', 'Test Cup qualification', 'FIFA World Cup'], size=len(dates),
                       p=[0.35, 0.3, 0.25, 0.1])
    rows = []
    for i, d in enumerate(dates, start=1):
        h, a = rng.choice(TEAMS, size=2, replace=False)
        neutral = bool(rng.random() < 0.3)
        lam_h = np.exp(0.2 + strength[h] - strength[a] + (0 if neutral else 0.25))
        lam_a = np.exp(0.2 + strength[a] - strength[h])
        hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
        t = str(tours[i - 1])
        if i in SHOOTOUT_IDS:
            t, hg, ag = 'Test Cup', 1, 1
        rows.append({'match_id': i, 'match_date': d, 'season': int(d[:4]), 'home_team': h, 'away_team': a,
                     'home_goals': hg, 'away_goals': ag, 'neutral': neutral, 'tournament': t})
    return pd.DataFrame(rows)


def result_letter(hg, ag) -> str:
    return 'H' if hg > ag else ('D' if hg == ag else 'A')


def build_db(path, m: pd.DataFrame, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    create_tables(con)
    # INA 合并前的临时建表，合并后改用 schema
    con.execute('create table if not exists match_venue (match_id integer not null, source text not null, '
                'neutral integer not null, city text, country text, tournament text not null, primary key (match_id, source))')
    con.executemany('insert into teams (team_id, country) values (?, ?)', [(t, 'World') for t in TEAMS])
    con.executemany('insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,'
                    ' status, round, leg, date_precision) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    [(r.match_id, 'INT', r.season, r.home_team, r.away_team, r.match_date, None, 'finished',
                      r.match_date, 0, 'day_world') for r in m.itertuples()])
    con.executemany('insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)',
                    [(r.match_id, 'martj42', r.home_goals, r.away_goals, result_letter(r.home_goals, r.away_goals))
                     for r in m.itertuples()])
    con.executemany('insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)',
                    [(i, 'dongqiudi', 9, 0, 'H') for i in DECOY_IDS])
    con.executemany('insert into match_venue values (?, ?, ?, ?, ?, ?)',
                    [(r.match_id, 'martj42', int(r.neutral), 'City', 'Country', r.tournament) for r in m.itertuples()])
    con.executemany('insert into cup_details (match_id, source, extra_time, aet_home, aet_away, pen_home, pen_away, note)'
                    ' values (?, ?, ?, ?, ?, ?, ?, ?)',
                    [(i, 'martj42', 1, 1, 1, 3, 4, 'shootout winner: away') for i in SHOOTOUT_IDS])
    odds = []
    for r in m[(m['season'] >= 2021) & (m['tournament'] != 'Friendly')].itertuples():
        p = rng.dirichlet([4, 2.5, 3])
        o = 1 / (p * 1.06)
        odds.append((r.match_id, 'dongqiudi-intl', '多家平均', '1x2', 'close', None, *o, None, None))
        odds.append((r.match_id, 'dongqiudi-intl', 'Bet365', '1x2', 'close', None, *(o * 1.01), None, None))
    con.executemany('insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, over, under)'
                    ' values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', odds)
    con.commit()
    con.close()


@pytest.fixture(scope='module')
def matches():
    return synthetic_matches()


@pytest.fixture(scope='module')
def db_path(tmp_path_factory, matches):
    path = tmp_path_factory.mktemp('intl') / 'football.db'
    build_db(path, matches)
    return path


@pytest.fixture(scope='module')
def study(db_path, settings):
    '''在手造数据库上跑一遍（评 2019–2023），再做两个比较。'''
    return intl_study.analyze(intl_study.run(settings, db_path=db_path, first=2019, last=2023))


# ======================= 研究程序：从头跑到尾 =======================
PRED_COLS = intl_study.P1 + intl_study.P0 + ['elo_home', 'elo_away', 'diff', 'elo_home_h', 'elo_away_h', 'diff_h',
                                             'fit_n', 'fit_n_h', 'step_cutoff']


def test_study_end_to_end(study, matches):
    pred = study['pred']
    assert len(pred) > 200 and study['n_friendly_skipped'] > 0
    assert 'Friendly' not in set(pred['tournament'])                       # 友谊赛不打分
    assert pred['season'].between(2019, 2023).all()
    assert 2024 not in set(pred['season']) and matches['match_id'].iloc[-1] not in set(pred['match_id'])
    assert pred[intl_study.P1].notna().all().all() and pred[intl_study.P0].notna().all().all()
    assert np.allclose(pred[intl_study.P1].sum(axis=1), 1) and np.allclose(pred[intl_study.P0].sum(axis=1), 1)
    assert pred['result'].notna().all()                                     # 2023 年最后一场的答案也拿到了
    assert (pred['fit_n'] >= 50).all()
    # 标杆：只有主比较那几年有，去水后加起来是 1；只取了“多家平均”那一行
    main = pred['season'] >= 2021
    assert pred.loc[main, intl_study.PB].notna().all().all() and pred.loc[~main, intl_study.PB].isna().all().all()
    assert np.allclose(pred.loc[main, intl_study.PB].sum(axis=1), 1)
    # 每场归到 T-24 之前最近的格子（最多早 7 天）
    t24, step = pd.to_datetime(pred['t24']), pd.to_datetime(pred['step_cutoff'])
    assert ((step <= t24) & (step > t24 - pd.Timedelta(days=7))).all()
    assert (pd.to_datetime(pred['t24']) == pd.to_datetime(pred['kickoff_earliest_utc']) - pd.Timedelta(hours=24)).all()
    assert study['out_of_order'] == (0, 0) and study['dropped_no_score'] == 0
    # 两个比较都出了表
    assert study['main']['场数'] == int(main.sum()) and study['neutral']['场数'] == len(pred)
    for k in ('main', 'neutral'):
        assert study[k]['结论'] in {'更准', '看不出差别', '更差', intl_study.SMALL_TOTAL}
    text = intl_study.build_report(study)
    assert '一、主比较' in text and '二、中立场比较' in text and '只看看' in text


def test_study_ratings_match_replay(study, matches):
    '''增量更新（只走有比赛要预测的格子）和“从头按日期一场一场重算”结果一样：
    比分只认 martj42（dongqiudi 的错比分没用上）、点球大战按平局。'''
    c_last = study['pred']['step_cutoff'].max()
    known = (pd.to_datetime(matches['match_date']) + pd.Timedelta(hours=39)).dt.strftime(TIME_FORMAT)
    done = matches.assign(known_at_utc=known)[known <= c_last]
    cfg = study['settings']['intl_study']
    for model, use_neutral in ((study['elo'], True), (study['elo_h'], False)):
        r = IntlElo.from_settings(cfg, use_neutral, 'replay')
        r.update_frame(done)
        assert r.ratings == model.ratings and r.n_updates == model.n_updates


def test_shootout_scores_are_draws(db_path, settings):
    '''取数层给的点球大战比分是 1:1、结果 D（点球只在 cup_details）。'''
    fs = FeatureStore(open_readonly(db_path), '开发', settings)
    h = fs.historical_matches('2013-01-01 00:00:00', '研究实验', competition='INT', source='martj42')
    s = h.set_index('match_id').loc[list(SHOOTOUT_IDS)]
    assert (s['result'] == 'D').all() and (s['home_goals'] == s['away_goals']).all()


def test_study_no_look_ahead(tmp_path, matches, settings, db_path):
    '''改掉 2020-06-01 以后的比分：截止时间在这之前的预测一点不变（之后的会变，说明这个检查有用）。'''
    changed = matches.copy()
    late = changed['match_date'] >= '2020-06-01'
    changed.loc[late, ['home_goals', 'away_goals']] = changed.loc[late, ['away_goals', 'home_goals']].to_numpy() + [2, 0]
    other = tmp_path / 'changed.db'
    build_db(other, changed)
    a = intl_study.run(settings, db_path=db_path, first=2019, last=2020)['pred'].set_index('match_id')
    b = intl_study.run(settings, db_path=other, first=2019, last=2020)['pred'].set_index('match_id')
    assert a.index.equals(b.index)
    before = a['step_cutoff'] < '2020-06-01 00:00:00'
    assert before.sum() > 50 and (~before).sum() > 10
    pd.testing.assert_frame_equal(a.loc[before, PRED_COLS], b.loc[before, PRED_COLS])
    assert not np.allclose(a.loc[~before, intl_study.P1].to_numpy(), b.loc[~before, intl_study.P1].to_numpy())


def test_study_grid_fixed(study, db_path, settings):
    '''格子从固定的一天算起：只评 2021 年和评 2019–2023 年，2021 年每场的预测一模一样。'''
    one = intl_study.run(settings, db_path=db_path, first=2021, last=2021)['pred'].set_index('match_id')
    full = study['pred'].set_index('match_id')
    full = full[full['season'] == 2021]
    assert one.index.equals(full.index)
    pd.testing.assert_frame_equal(one[PRED_COLS], full[PRED_COLS])


def test_study_unknown_tournament(tmp_path, matches, settings):
    '''数据库里出现没写进分档的赛事：走之前就报错，不猜。'''
    m = matches.copy()
    m.loc[m['match_id'] == 100, 'tournament'] = 'Mystery Cup'
    path = tmp_path / 'mystery.db'
    build_db(path, m)
    with pytest.raises(UnknownTournamentError, match='Mystery Cup'):
        intl_study.run(settings, db_path=path, first=2023, last=2023)


def test_dev_mode_blocks_2024(db_path, settings, matches):
    '''开发模式碰 2024 年：--last 2024 程序先报错；取数层取到 2024 年的比赛也报错。'''
    with pytest.raises(ValueError, match='期末考试'):
        intl_study.run(settings, db_path=db_path, first=2023, last=2024)
    fs = FeatureStore(open_readonly(db_path), '开发', settings)
    with pytest.raises(SeasonLockedError):
        fs.historical_matches('2024-01-03 00:00:00', '研究实验', competition='INT', since='2023-12-01 00:00:00')
    with pytest.raises(SeasonLockedError):
        fs.cutoff_before(int(matches['match_id'].iloc[-1]), 24, '研究实验')
