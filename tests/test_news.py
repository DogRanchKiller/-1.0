"""新闻资料库（第 9 阶段）：src/clean/news_import.py、news_schema.py 和取数层 news_for_match、news_facts_for_match 的自动检查。

全部用临时文件夹里的假文件 + 临时数据库，不碰真实数据（data/）。
造的数据：5 支球队；比赛 1 = 2026/27 曼城对切尔西（2026-11-07 15:00 开球，未赛）；
比赛 2 = 2024/25 利物浦对阿森纳（期末考试赛季，已赛）。“现在”固定是 2026-11-06 12:00 UTC。
"""
import copy
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.clean.news_import import run
from src.clean.news_schema import create_news_tables
from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.features import FeatureStore, LookAheadError, SeasonLockedError, UsageNotAllowedError

UTC = timezone.utc
NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
TEAMS = [('Man City', '曼城'), ('Chelsea', '切尔西'), ('Liverpool', '利物浦'), ('Arsenal', '阿森纳'), ('Man United', '曼联')]


def make_db(tmp_path, name):
    con = connect(tmp_path / name)
    create_tables(con)
    con.executemany("insert into teams (team_id, name_zh, country) values (?, ?, '英格兰')", TEAMS)
    con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status)"
                " values (1, 'E0', 2026, 'Man City', 'Chelsea', '2026-11-07', '2026-11-07 15:00', 'scheduled')")
    con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status)"
                " values (2, 'E0', 2024, 'Liverpool', 'Arsenal', '2025-02-01', '2025-02-01 15:00', 'finished')")
    con.commit()
    return con


@pytest.fixture
def env(tmp_path):
    s = copy.deepcopy(load_settings())
    s['news']['sources'] = {'测试社': {'status': '允许', 'note': '假数据'}, '待定社': {'status': '待定'}, '禁止社': {'status': '不用'}}
    s['news']['inbox_dir'] = str(tmp_path / 'inbox')
    (tmp_path / 'inbox').mkdir()
    return {'con': make_db(tmp_path, 't.db'), 's': s, 'tmp': tmp_path, 'n': 0}


def item(url='http://x/1', **kw):
    row = {'source': '测试社', 'url': url, 'title': '曼城主帅会见记者', 'fetched_at': '2026-11-05 10:00:00'}
    row.update(kw)
    return row


def do_run(env, items=(), facts=()):
    """把假数据写成文件放进收件箱，运行导入程序。"""
    env['n'] += 1
    inbox = env['tmp'] / 'inbox'
    for prefix, rows in (('news_', items), ('news_facts_', facts)):
        if rows:
            text = '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n'
            (inbox / f'{prefix}{env["n"]}.jsonl').write_text(text, encoding='utf-8')
    return run(env['con'], env['s'], raw_root=env['tmp'] / 'raw', report_path=env['tmp'] / 'report.md', now=NOW)


def links(con, url):
    return sorted(con.execute('select l.team_id, l.match_id, l.method from news_links l join news_items i using (news_id)'
                              ' where i.url = ?', (url,)).fetchall())


# ======================= 导入：来源和时间规则 =======================
def test_only_allowed_sources_are_imported(env):
    s = do_run(env, [item('http://x/a', source='待定社'), item('http://x/b', source='禁止社'),
                     item('http://x/c', source='没登记社')])
    assert s['items_new'] == 0 and s['items_rejected'] == 3
    assert env['con'].execute('select count(*) from news_items').fetchone()[0] == 0
    reasons = ' '.join(r[2] for r in s['rejects'])
    assert '没登记' in reasons and '待定' in reasons and '不用' in reasons


def test_default_settings_have_no_registered_source():
    """没经过你决定的网站，数据进不了库：默认配置里一个来源都没有登记。"""
    assert load_settings()['news']['sources'] == {}


def test_time_rules(env):
    rows = [
        item('http://x/1', fetched_at=None),                              # 没有抓到时间
        item('http://x/2', fetched_at='2026-11-05'),                      # 只有日期
        item('http://x/3', fetched_at='2026-11-06 13:00:00'),             # 比现在晚
        item('http://x/4', published_at='不是时间'),                      # 发布时间看不懂
        item('http://x/5', published_at='2026-11-05'),                    # 好的，发布时间只有日期
        item('http://x/6', fetched_at='2026-11-05T11:00:00+01:00'),       # 好的，带时区要换算
    ]
    s = do_run(env, rows)
    assert s['items_new'] == 2 and s['items_rejected'] == 4
    con = env['con']
    assert con.execute("select published_at_utc from news_items where url = 'http://x/5'").fetchone()[0] == '2026-11-06 00:00:00'
    assert con.execute("select fetched_at_utc from news_items where url = 'http://x/6'").fetchone()[0] == '2026-11-05 10:00:00'


def test_bad_lines_are_reported_not_fatal(env):
    inbox = env['tmp'] / 'inbox'
    (inbox / 'news_bad.jsonl').write_text('不是 json\n[1, 2]\n' + json.dumps(item(), ensure_ascii=False) + '\n', encoding='utf-8')
    s = run(env['con'], env['s'], raw_root=env['tmp'] / 'raw', report_path=env['tmp'] / 'report.md', now=NOW)
    assert s['items_new'] == 1 and s['items_rejected'] == 2
    assert (env['tmp'] / 'report.md').read_text(encoding='utf-8').count('news_bad.jsonl') >= 2


# ======================= 导入：重复、原始层、重复运行 =======================
def test_duplicates_keep_first_row_and_rerun_is_stable(env):
    first = item(fetched_at='2026-11-05 10:00:00')
    again = item(fetched_at='2026-11-05 08:00:00')               # 同一条，抓到时间更早：不能改成更早
    renamed = item(title='曼城换了新标题')                          # 同一链接标题改了：算新的一行
    s1 = do_run(env, [first, again, renamed])
    assert s1['items_new'] == 2 and s1['items_dup'] == 1
    assert env['con'].execute("select fetched_at_utc from news_items where title = '曼城主帅会见记者'").fetchone()[0] == '2026-11-05 10:00:00'
    n_links = env['con'].execute('select count(*) from news_links').fetchone()[0]
    s2 = run(env['con'], env['s'], raw_root=env['tmp'] / 'raw', report_path=env['tmp'] / 'report.md', now=NOW)   # 同一批文件再来一次
    assert s2['items_new'] == 0 and s2['items_dup'] == 3
    assert env['con'].execute('select count(*) from news_items').fetchone()[0] == 2
    assert env['con'].execute('select count(*) from news_links').fetchone()[0] == n_links   # 重复出现不补关联
    assert len(list((env['tmp'] / 'raw').rglob('*.gz'))) == 1      # 同样的文件内容只存一份


def test_raw_file_is_archived_and_recorded(env):
    do_run(env, [item()])
    gz = list((env['tmp'] / 'raw').rglob('*.gz'))
    assert len(gz) == 1 and gz[0].parent.name == '2026-11-06'
    recorded = env['con'].execute('select raw_file from news_items').fetchone()[0]
    assert recorded.endswith(gz[0].name)


def test_create_tables_is_repeatable_and_keeps_data(env):
    do_run(env, [item()])
    create_news_tables(env['con'])
    create_news_tables(env['con'])
    assert env['con'].execute('select count(*) from news_items').fetchone()[0] == 1


# ======================= 导入：关联球队、比赛 =======================
def test_links_from_source_title_and_match_ids(env):
    rows = [
        item('http://x/a', title='曼城对阵切尔西前瞻'),
        item('http://x/b', title='周末焦点', teams=['利物浦', '不存在队'], match_ids=[1, 999]),
        item('http://x/c', title='曼联转会传闻'),            # 曼联不能被认成曼城
    ]
    s = do_run(env, rows)
    con = env['con']
    assert links(con, 'http://x/a') == [('Chelsea', 0, '标题认出'), ('Man City', 0, '标题认出')]
    assert links(con, 'http://x/b') == [('', 1, '来源自带'), ('Liverpool', 0, '来源自带')]
    assert links(con, 'http://x/c') == [('Man United', 0, '标题认出')]
    assert s['unknown_teams'] == {'不存在队': 1} and s['bad_match_ids'] == 1


def test_ambiguous_name_is_not_recognised(env):
    env['con'].execute("insert into teams (team_id, name_zh, country) values ('Man City 2', '曼城', '英格兰')")
    env['con'].commit()
    do_run(env, [item('http://x/a', title='曼城主帅会见记者')])
    assert links(env['con'], 'http://x/a') == []      # “曼城”对到两支球队，不认


# ======================= 导入：结构化信息 =======================
def fact(**kw):
    row = {'team': '曼城', 'fact_type': 'coach_change', 'effective_date': '2026-11-04', 'person': '某某',
           'recorded_at': '2026-11-06 08:00:00', 'news_source': '测试社', 'news_url': 'http://x/n1'}
    row.update(kw)
    return row


def test_facts_import_rules(env):
    facts = [
        fact(),                                              # 好的
        fact(fact_type='transfer'),                          # 种类不对
        fact(effective_date='2026/11/04'),                   # 日期写法不对
        fact(team='火星队'),                                  # 球队认不出
        fact(recorded_at=None),                              # 没有录入时间
        fact(recorded_at='2026-11-06 09:00:00'),             # 重复：只留先录入的
        fact(news_url='http://x/不存在'),                     # 对不到来源新闻
        fact(news_source=None, news_url=None, person='乙', method='乱写'),      # method 不对
    ]
    s = do_run(env, [item('http://x/n1')], facts)
    assert s['facts_new'] == 1 and s['facts_dup'] == 1 and s['facts_rejected'] == 6
    assert s['unknown_teams'] == {'火星队': 1}
    row = env['con'].execute('select recorded_at_utc, news_id from news_facts').fetchone()
    assert row[0] == '2026-11-06 08:00:00' and row[1] is not None


def test_rule_fact_needs_source_but_manual_fact_can_be_entered_with_no_sources(env):
    """来源表为空时，规则产物不能绕过来源登记；用户明确手动写下的一件事仍可录入。"""
    env['s']['news']['sources'] = {}
    s = do_run(env, facts=[
        fact(person='规则产物', method='规则', news_source=None, news_url=None),
        fact(person='人工录入', method='人工', news_source=None, news_url=None),
    ])
    assert s['facts_new'] == 1 and s['facts_rejected'] == 1
    assert env['con'].execute('select person, method, news_id from news_facts').fetchall() == [
        ('人工录入', '人工', None)]
    assert '规则' in s['rejects'][0][2] and '来源新闻' in s['rejects'][0][2]


@pytest.mark.parametrize('source_status', [None, '待定', '不用'])
def test_rule_fact_rechecks_source_permission_after_news_was_imported(env, source_status):
    """新闻以前导入过，不代表现在仍允许从这个来源导入新整理的信息。"""
    do_run(env, items=[item('http://x/n1')])
    if source_status is None:
        env['s']['news']['sources'].pop('测试社')
    else:
        env['s']['news']['sources']['测试社']['status'] = source_status
    s = do_run(env, facts=[fact(method='规则')])
    assert s['facts_new'] == 0 and s['facts_rejected'] == 1
    assert env['con'].execute('select count(*) from news_facts').fetchone()[0] == 0
    assert any('来源' in why and '允许' in why for _, _, why in s['rejects'])


def test_rule_fact_links_allowed_imported_news(env):
    s = do_run(env, items=[item('http://x/n1')], facts=[fact(method='规则')])
    assert s['facts_new'] == 1 and s['facts_rejected'] == 0
    assert env['con'].execute(
        'select f.method, i.source, i.url from news_facts f join news_items i using (news_id)'
    ).fetchall() == [('规则', '测试社', 'http://x/n1')]


# ======================= 取数层：什么时候能知道 =======================
def test_news_visible_only_after_fetched_time(env):
    do_run(env, [item('http://x/1', title='曼城新闻'),
                 item('http://x/2', title='切尔西新闻', published_at='2026-11-05 12:00:00'),   # 发布比抓到晚：按发布时间
                 item('http://x/3', title='利物浦新闻')])                                      # 和比赛 1 无关
    fs = FeatureStore(env['con'], '实时', env['s'])
    assert fs.news_for_match(1, '2026-11-05 09:59:59', '显示').empty
    got = fs.news_for_match(1, '2026-11-05 10:00:00', '显示')
    assert list(got['url']) == ['http://x/1'] and got['known_at_utc'].iloc[0] == '2026-11-05 10:00:00'
    got = fs.news_for_match(1, '2026-11-05 12:00:00', '显示')
    assert set(got['url']) == {'http://x/1', 'http://x/2'}
    assert (got['known_at_utc'] <= '2026-11-05 12:00:00').all()
    assert got.set_index('url').loc['http://x/2', 'known_at_utc'] == '2026-11-05 12:00:00'
    only_new = fs.news_for_match(1, '2026-11-05 12:00:00', '显示', days=1 / 24)
    assert list(only_new['url']) == ['http://x/2']


def test_facts_known_time_is_latest_of_record_and_news(env):
    facts = [
        fact(),                                                                  # 录入 11-06 08:00，新闻 11-05 10:00 → 11-06 08:00
        fact(team='切尔西', person='乙', recorded_at='2026-11-04 09:00:00'),       # 录入早、新闻晚 → 按新闻抓到时间 11-05 10:00
        fact(team='曼城', fact_type='long_injury', effective_date='2026-10-01', person='丙', news_source=None, news_url=None,
             recorded_at='2026-11-06 11:00:00'),                                 # 生效日早，但是事后才录入 → 11-06 11:00 才能用
    ]
    do_run(env, [item('http://x/n1')], facts)
    fs = FeatureStore(env['con'], '实时', env['s'])
    assert fs.news_facts_for_match(1, '2026-11-05 09:59:59', '显示').empty
    got = fs.news_facts_for_match(1, '2026-11-05 10:00:00', '显示')
    assert list(got['person']) == ['乙']
    got = fs.news_facts_for_match(1, '2026-11-06 08:00:00', '显示')
    assert set(got['person']) == {'某某', '乙'}
    got = fs.news_facts_for_match(1, '2026-11-06 12:00:00', '显示')
    assert set(got['person']) == {'某某', '乙', '丙'} and (got['known_at_utc'] <= '2026-11-06 12:00:00').all()
    assert list(fs.news_facts_for_match(1, '2026-11-06 12:00:00', '显示', fact_type='long_injury')['person']) == ['丙']


def test_missing_tables_give_empty_result(tmp_path):
    con = make_db(tmp_path, 'b.db')            # 没建新闻表
    s = copy.deepcopy(load_settings())
    fs = FeatureStore(con, '实时', s)
    news = fs.news_for_match(1, '2026-11-05 10:00:00', '显示')
    facts = fs.news_facts_for_match(1, '2026-11-05 10:00:00', '显示')
    assert news.empty and 'known_at_utc' in news.columns and facts.empty and 'known_at_utc' in facts.columns


# ======================= 取数层：用途、阶段、赛前、运行模式 =======================
def test_usage_stage_lookahead_and_mode_checks(env):
    do_run(env, [item('http://x/n1')], [fact()])
    fs = FeatureStore(env['con'], '实时', env['s'])
    with pytest.raises(UsageNotAllowedError):                                   # 新闻标题模型不用
        fs.news_for_match(1, '2026-11-06 12:00:00', 'A版输入')
    with pytest.raises(UsageNotAllowedError):                                   # 结构化信息要到第 9 阶段才能当模型输入
        fs.news_facts_for_match(1, '2026-11-06 12:00:00', 'A版输入')
    s9 = copy.deepcopy(env['s'])
    s9['features']['current_stage'] = 9
    fs9 = FeatureStore(env['con'], '实时', s9)
    assert len(fs9.news_facts_for_match(1, '2026-11-06 12:00:00', 'A版输入')) == 1      # 开球前可以
    with pytest.raises(LookAheadError):                                                  # 开球后取本场信息做赛前预测 = 偷看未来
        fs9.news_facts_for_match(1, '2026-11-07 15:00:00', 'A版输入')
    dev = FeatureStore(env['con'], '开发', env['s'])
    with pytest.raises(SeasonLockedError):                                      # 开发模式读不了期末考试赛季的比赛
        dev.news_for_match(2, '2025-02-01 12:00:00', '显示')
