"""取数层：新闻资料库（需求文档 7.1“怎么防止偷看未来”）。

两个函数，FeatureStore 里的同名方法（news_for_match、news_facts_for_match）直接调它们。
模型和回测不直接读 news_items / news_links / news_facts 三张表，只能从这里取。

“什么时候能知道”（每行都带 known_at_utc，截止时间早于它就拿不到）：
  新闻标题：抓到时间、发布时间里更晚的一个（发布时间比抓到时间还晚多半是网站时间写错，宁晚勿早）；
  结构化信息：录入时间、来源新闻的抓到时间、来源新闻的发布时间里最晚的一个
            （事后回头整理旧新闻，整理出来的信息当时并没有，所以录入时间也要算）。
检查和别的“本场”数据一样走 FeatureStore._target：用途按数据字典、赛前用途的截止时间必须早于开球、
运行模式检查赛季（开发模式读不了期末考试赛季）。
新闻表还没建（还没运行过导入程序）时返回空表，不报错：那时确实没有新闻。
"""
import pandas as pd

from src.features.known_time import to_utc_text

NEWS_COLUMNS = ['news_id', 'source', 'url', 'title', 'published_at_utc', 'fetched_at_utc', 'language', 'known_at_utc']
FACT_COLUMNS = ['fact_id', 'team_id', 'fact_type', 'effective_date', 'person', 'note', 'news_id', 'method',
                'recorded_at_utc', 'known_at_utc']


def _has_tables(conn, names: list[str]) -> bool:
    found = {r[0] for r in conn.execute("select name from sqlite_master where type = 'table'")}
    return all(n in found for n in names)


def _teams_of(conn, match_id: int) -> tuple[str, str]:
    return conn.execute('select home_team, away_team from matches where match_id = ?', (int(match_id),)).fetchone()


def _earliest_known(cutoff: str, days) -> str | None:
    """days 给了：只要“能知道的时间”不早于 截止时间 - days 天 的（省得拿回一大堆旧新闻，不影响防偷看）。"""
    if days is None:
        return None
    if float(days) <= 0:
        raise ValueError('days 要大于 0')
    return to_utc_text(pd.Timestamp(cutoff) - pd.Timedelta(days=float(days)))


def news_for_match(store, match_id: int, cutoff, usage: str, days=None) -> pd.DataFrame:
    """和这场比赛有关的新闻：关联了这场比赛，或者关联了主队、客队的，截止时间之前已经能知道的。

    用途要在数据字典“新闻标题”里允许（现在只有复盘、显示，模型不用）。从新到旧排。
    """
    cutoff = to_utc_text(cutoff)
    store._target(match_id, cutoff, usage, ['新闻标题'])
    if not _has_tables(store.conn, ['news_items', 'news_links']):
        return pd.DataFrame(columns=NEWS_COLUMNS)
    home, away = _teams_of(store.conn, match_id)
    known = "max(i.fetched_at_utc, coalesce(i.published_at_utc, ''))"
    where, params = [f'{known} <= ?'], [cutoff]
    earliest = _earliest_known(cutoff, days)
    if earliest is not None:
        where.append(f'{known} >= ?')
        params.append(earliest)
    sql = f"""
        select i.news_id, i.source, i.url, i.title, i.published_at_utc, i.fetched_at_utc, i.language,
               {known} as known_at_utc
        from news_items i
        where {' and '.join(where)}
          and exists (select 1 from news_links l where l.news_id = i.news_id
                      and (l.match_id = ? or l.team_id in (?, ?)))
        order by known_at_utc desc, i.news_id desc"""
    return pd.read_sql(sql, store.conn, params=[*params, int(match_id), home, away])


def news_facts_for_match(store, match_id: int, cutoff, usage: str, fact_type=None) -> pd.DataFrame:
    """主队、客队的结构化信息（主帅更换、核心长期伤缺），截止时间之前已经能知道的。

    用途要在数据字典“新闻结构化信息”里允许；当模型输入要到第 9 阶段（settings.yaml 的 features.current_stage），
    而且第 5 节“第 4 阶段以后才加的改进”的影子运行通过以后才上线。
    effective_date 是事情生效的日子，可能比 known_at_utc 晚（提前宣布）或早（事后才录入），由调用方自己判断怎么用。
    """
    cutoff = to_utc_text(cutoff)
    store._target(match_id, cutoff, usage, ['新闻结构化信息'])
    if not _has_tables(store.conn, ['news_facts', 'news_items']):
        return pd.DataFrame(columns=FACT_COLUMNS)
    home, away = _teams_of(store.conn, match_id)
    known = "max(f.recorded_at_utc, coalesce(i.fetched_at_utc, ''), coalesce(i.published_at_utc, ''))"
    where, params = [f'{known} <= ?', 'f.team_id in (?, ?)'], [cutoff, home, away]
    if fact_type is not None:
        where.append('f.fact_type = ?')
        params.append(fact_type)
    sql = f"""
        select f.fact_id, f.team_id, f.fact_type, f.effective_date, f.person, f.note, f.news_id, f.method,
               f.recorded_at_utc, {known} as known_at_utc
        from news_facts f left join news_items i on i.news_id = f.news_id
        where {' and '.join(where)}
        order by known_at_utc, f.fact_id"""
    return pd.read_sql(sql, store.conn, params=params)
