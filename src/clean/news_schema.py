"""新闻资料库的三张表（需求文档 7.1“新闻资料库怎么做”）。

三张表：
  news_items  一条新闻一行：来源、链接、标题、发布时间、抓到时间（不存正文和图片）
  news_links  这条新闻说的是哪支球队、哪场比赛（一条新闻可以关联多个）
  news_facts  从新闻里整理出的“一件事”：主帅更换、核心长期伤缺

为什么单独放一个文件、不改 schema.py：schema.py 是别的阶段共用的，这里只加不改。
建表语句全部是 create table if not exists / create index if not exists：
可以重复运行，已经有的表和里面的数据一律不动（不删、不改）。
时间都是标准时间（UTC）写成文字 'YYYY-MM-DD HH:MM:SS'，文字排序就是时间排序，数据库里可以直接比大小。
"""
import sqlite3

NEWS_TABLES = """
-- 新闻条目：一条新闻一行。同一来源、同一链接、同一标题只留先导入的那一行（不改它，包括抓到时间）。
create table if not exists news_items (
    news_id          integer primary key,
    source           text not null,   -- 来源名，必须是 settings.yaml 的 news.sources 里登记为“允许”的
    url              text not null,   -- 新闻链接
    title            text not null,   -- 新闻标题
    published_at_utc text,            -- 网站写的发布时间（可能没有；只有日期的按第二天 00:00 算，宁晚勿早）
    fetched_at_utc   text not null,   -- 抓到时间：我们第一次看到它的时间，必填。防偷看未来靠它
    language         text,            -- 语言，例如 zh、en
    raw_file         text,            -- 导入的原始文件（压缩）放在哪
    imported_at_utc  text not null,   -- 导入程序处理它的时间
    unique (source, url, title)
);
create index if not exists news_items_fetched on news_items (fetched_at_utc);

-- 关联：这条新闻说的是哪支球队、哪场比赛。球队和比赛二选一（另一个留默认值 '' 或 0）。
-- 只在第一次导入这条新闻时建；同一条新闻以后再出现不补（事后补的关联会让旧新闻在更早的截止时间出现）。
create table if not exists news_links (
    news_id  integer not null references news_items(news_id),
    team_id  text not null default '',     -- 统一球队编号（teams 表）；关联比赛时是 ''
    match_id integer not null default 0,   -- 比赛编号（matches 表）；关联球队时是 0
    method   text not null,                -- 怎么关联的：来源自带 / 标题认出 / 人工
    primary key (news_id, team_id, match_id)
);
create index if not exists news_links_team on news_links (team_id);
create index if not exists news_links_match on news_links (match_id);

-- 结构化信息：一行 = 一支球队的一件事。
-- “能知道的时间”取三个时间里最晚的：录入时间、来源新闻的抓到时间、来源新闻的发布时间（取数层算）。
-- 事后回头整理旧新闻，整理出来的信息当时并没有，所以录入时间也要算进去。
create table if not exists news_facts (
    fact_id         integer primary key,
    team_id         text not null references teams(team_id),
    fact_type       text not null,            -- coach_change 主帅更换 / long_injury 核心长期伤缺（settings.yaml news.fact_types）
    effective_date  text not null,            -- 事情哪天生效 yyyy-mm-dd
    person          text not null default '', -- 涉及的人（新帅、伤员）
    note            text,                     -- 备注（“核心”怎么认的、来源说明等）
    news_id         integer references news_items(news_id),   -- 从哪条新闻来；你手动录入的可以没有
    method          text not null,            -- 人工 / 规则
    recorded_at_utc text not null,            -- 录入时间：这一行被写下来的时间，必填
    imported_at_utc text not null,            -- 导入程序处理它的时间
    unique (team_id, fact_type, effective_date, person)
);
create index if not exists news_facts_team on news_facts (team_id, fact_type);
"""


def create_news_tables(con: sqlite3.Connection) -> None:
    """建新闻资料库的三张表。可以重复运行：已经有的表不动、里面的数据不删。"""
    con.executescript(NEWS_TABLES)
