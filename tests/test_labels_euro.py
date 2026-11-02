"""复盘标签“欧战后的联赛”的自动检查（需求文档第 15 节，工作树 Q1）。

用内存里的临时数据库，不碰真实数据库。联赛借用 test_labels.py 的 4 队小联赛（E0，赛季 2022），
再加几场杯赛、欧战（X 是对手；杯赛、欧战的对手和比分不影响标签）。时间都是 UTC。

联赛（和 test_labels.py 一样）：
  1 A–T 08-06 14:00   2 C–F 08-06 14:00   3 C–T 08-13 14:00   4 F–A 08-13 14:00
  5 T–F 08-20 14:00   6 A–C 08-20 14:00   7 T–C 10-01 14:00   8 A–F 10-01（只有日期）
杯赛、欧战：
  101 欧联 08-04 19:00 T–X   → 1 号 T（客队）隔 2 天，算
  102 欧冠 08-10 19:00 A–X   → 4 号 A（客队）隔 3 天，算
  103 足总杯 08-16 19:00 T–X → 5 号 T 隔 4 天，但上一场是国内杯赛，不算
  104 欧冠 09-27 19:00 C–X   → 7 号 C（客队）隔 4 天，正好在界限上，算
  105 欧联 09-26 19:00 F–X   → 8 号 F（客队）隔 5 天，超过 4 天，不算
  106 欧联 08-12 19:00 F–Y   → 4 号 F（主队）隔 1 天，算；但提前 24 小时截止时，这场还不知道踢没踢
（105、106 的对手故意不同：同赛事、同赛季、同主客队的两场会撞上 matches 表的唯一约束）
"""
import copy
import sqlite3

import pandas as pd
import pytest
import yaml

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.features import FeatureStore
from src.features.rest_days import RestDays
from src.review import labels as L

A, T, C, F, X, Y = 'Arsenal', 'Tottenham', 'Chelsea', 'Fulham', 'Ajax', 'Benfica'
LEAGUE = [
    # match_id, 主, 客, 日期, 开球 UTC（None = 只有日期）
    (1, A, T, '2022-08-06', '2022-08-06 14:00'),
    (2, C, F, '2022-08-06', '2022-08-06 14:00'),
    (3, C, T, '2022-08-13', '2022-08-13 14:00'),
    (4, F, A, '2022-08-13', '2022-08-13 14:00'),
    (5, T, F, '2022-08-20', '2022-08-20 14:00'),
    (6, A, C, '2022-08-20', '2022-08-20 14:00'),
    (7, T, C, '2022-10-01', '2022-10-01 14:00'),
    (8, A, F, '2022-10-01', None),
]
SCORES = {1: (1, 1), 2: (2, 0), 3: (1, 0), 4: (0, 3), 5: (3, 0), 6: (2, 0), 7: (1, 0), 8: (0, 0)}
CUPS = [
    # match_id, 赛事, 主, 客, 日期, 开球 UTC
    (101, 'UEL', T, X, '2022-08-04', '2022-08-04 19:00'),
    (102, 'UCL', A, X, '2022-08-10', '2022-08-10 19:00'),
    (103, 'FAC', T, X, '2022-08-16', '2022-08-16 19:00'),
    (104, 'UCL', C, X, '2022-09-27', '2022-09-27 19:00'),
    (105, 'UEL', F, X, '2022-09-26', '2022-09-26 19:00'),
    (106, 'UEL', F, Y, '2022-08-12', '2022-08-12 19:00'),
]
# 手算：每场 (主队欧战后, 客队欧战后)
EXPECTED = {1: (False, True), 2: (False, False), 3: (False, False), 4: (True, True),
            5: (False, False), 6: (False, False), 7: (False, True), 8: (False, False)}
NEW_COLS = ['home_after_euro', 'away_after_euro']


def make_con(with_cups: bool = True) -> sqlite3.Connection:
    c = sqlite3.connect(':memory:')
    create_tables(c)
    c.executemany('insert into teams (team_id, country) values (?, ?)',
                  [(t, 'England') for t in (A, T, C, F, X, Y)])
    sql = ('insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,'
           ' status) values (?, ?, 2022, ?, ?, ?, ?, ?)')
    c.executemany(sql, [(m[0], 'E0', m[1], m[2], m[3], m[4], 'finished') for m in LEAGUE])
    if with_cups:
        c.executemany(sql, [(m[0], m[1], m[2], m[3], m[4], m[5], 'finished') for m in CUPS])
    c.executemany('insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, ?, ?, ?, ?)',
                  [(i, 'football-data', h, a, 'H' if h > a else 'A' if h < a else 'D') for i, (h, a) in SCORES.items()])
    c.commit()
    return c


@pytest.fixture
def settings(tmp_path):
    """真实 settings 的拷贝：排名规则、扣分、标签输出都指到临时文件夹。"""
    s = copy.deepcopy(load_settings())
    rules = {'联赛': {'E0': {'规则': [{'赛季': [1993, 9999], '胜': 3, '平': 1, '同分': ['相互战绩积分', '净胜球', '进球'],
                                   '相互战绩要踢完': '否', '降级': 1, '降级附加赛': 0, '来源': '测试'}]}}}
    (tmp_path / 'rules.yaml').write_text(yaml.safe_dump(rules, allow_unicode=True), encoding='utf-8')
    (tmp_path / 'ded.yaml').write_text('扣分: []\n', encoding='utf-8')
    s['standings'] = {'rules': str(tmp_path / 'rules.yaml'), 'deductions': str(tmp_path / 'ded.yaml')}
    s['labels'] = {**s['labels'], 'output_dir': str(tmp_path / 'labels')}
    return s


def flags(df: pd.DataFrame) -> dict:
    return {int(i): (bool(r.home_after_euro), bool(r.away_after_euro)) for i, r in df.iterrows()}


def label(settings, con) -> pd.DataFrame:
    return L.label_all(settings, ['E0'], [2022], conn=con, verbose=False).set_index('match_id')


def test_after_euro_by_hand(settings):
    """每场主队、客队的欧战后，对照上面手算的结果（含隔 4 天算、隔 5 天不算、国内杯赛不算）。"""
    df = label(settings, make_con())
    assert flags(df) == EXPECTED


def test_no_cup_data_means_no_after_euro_and_old_labels_unchanged(settings):
    """没有杯赛、欧战赛程：全是 False；德比、争冠、保级等原有的列和有杯赛时一模一样（杯赛不影响它们）。"""
    settings['labels']['race'] = {'last_rounds': 3, 'gap_points': 6}
    plain = label(settings, make_con(with_cups=False))
    full = label(settings, make_con(with_cups=True))
    assert not plain[NEW_COLS].any().any()
    pd.testing.assert_frame_equal(plain.drop(columns=NEW_COLS), full.drop(columns=NEW_COLS))


def test_game_not_yet_known_at_cutoff_is_not_used(settings):
    """截止时间提前 24 小时：4 号（08-13 14:00 开球）的截止时间是 08-12 14:00，
    F 在 08-12 19:00 的欧联那时还不知道踢没踢，不能用，F 的上一场要找成 08-06 的联赛 → 主队不算欧战后。
    其他场都不受影响（A 的欧冠 08-10 那时已经知道）。"""
    settings['labels'] = {**settings['labels'], 'cutoff_hours_before_kickoff': 24}
    df = label(settings, make_con())
    assert flags(df) == {**EXPECTED, 4: (False, True)}


def test_same_rule_as_model_after_euro_column(settings):
    """标签和梯度提升树用的“欧战后”那一列口径一样：把全部赛程交给 RestDays 算出来的，和标签逐场相同。"""
    con = make_con()
    df = L.label_all(settings, ['E0'], [2022], conn=con, verbose=False)
    rest = RestDays.from_settings(settings)
    rest.add(FeatureStore(con, '开发', settings).schedule('2030-01-01', '休息天数和欧战后标签'))
    model = rest.annotate(df[['match_date', 'home_team', 'away_team']])
    assert (df['home_after_euro'].astype(float) == model['home_after_euro']).all()
    assert (df['away_after_euro'].astype(float) == model['away_after_euro']).all()
    assert df['home_after_euro'].sum() == 1 and df['away_after_euro'].sum() == 3


def test_saved_labels_keep_after_euro_columns(settings):
    """存成文件再读回来，欧战后两列还在，值不变；开发赛季在开发模式下可以读。"""
    df = L.label_all(settings, ['E0'], [2022], conn=make_con(), verbose=False)
    L.save_labels(df, settings)
    back = L.load_labels('开发', settings=settings).set_index('match_id')
    assert flags(back) == EXPECTED
