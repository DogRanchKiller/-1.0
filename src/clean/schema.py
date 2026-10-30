"""数据库里清洗层的正式表（需求文档第 17 节“统一标准”“按数据类型放进正式表”）。

历史数据和以后爬到的实时数据放在同一套表里，用“来源”字段区分，不分两个数据库。
这一版只有 football-data 用得到的表；变盘记录、首发、伤停、竞彩等表到第 1b 阶段再加。

比赛编号（match_id）一旦分配就不变：以后的预测存档要用它找到比赛。
所以重建数据时，比赛表是“有就更新、没有才新增”，不是删掉重来；
其他表（赛果统计、赔率）是从原始层算出来的，按来源整批删掉重导。
"""
import sqlite3

TABLES = """
-- 球队：一行一支球队。内容来自 config/teams.yaml，每次导入时按它重写
create table if not exists teams (
    team_id   text primary key,   -- 统一的英文名，例如 Man United
    name_zh   text,               -- 中文名（第 1b 阶段填）
    country   text not null
);

-- 队名对照：各个来源的写法 → 统一球队编号（需求文档 6.4）
create table if not exists team_names (
    source    text not null,      -- 来源，例如 football-data
    name      text not null,      -- 这个来源里的写法
    team_id   text not null references teams(team_id),
    primary key (source, name)
);

-- 比赛：一行一场比赛。同一联赛同一赛季同一对主客队只有一场（需求文档第 16 节“同一场比赛的匹配”）
create table if not exists matches (
    match_id       integer primary key,
    competition    text not null,   -- 赛事代码，联赛用 football-data 的代码，例如 E0
    season         integer not null,-- 赛季开始年份，2023 表示 2023/24
    home_team      text not null references teams(team_id),
    away_team      text not null references teams(team_id),
    match_date     text not null,   -- 比赛日期（英国当地日期，yyyy-mm-dd）
    kickoff_utc    text,            -- 开球时间（标准时间 yyyy-mm-dd HH:MM）；2019/20 以前 football-data 没有，是空的
    status         text not null,   -- scheduled 未赛 / finished 已赛 / postponed 改期 / cancelled 取消
    unique (competition, season, home_team, away_team)
);

-- 各来源的比赛对应到哪个统一比赛编号，以及这条数据从哪个原始文件来、什么时候拿到的
create table if not exists match_sources (
    match_id       integer not null references matches(match_id),
    source         text not null,
    source_key     text not null,   -- 这个来源里怎么认这场比赛（football-data 没有编号，用 联赛|日期|主队|客队）
    raw_file       text,            -- 原始文件位置（相对于项目文件夹）
    fetched_at_utc text,            -- 原始文件的下载时间
    primary key (source, source_key)
);

-- 赛果和统计：一行一场比赛的一个来源。全部是赛后才知道的数据
create table if not exists match_stats (
    match_id    integer not null references matches(match_id),
    source      text not null,
    home_goals integer, away_goals integer,          -- 全场比分（90 分钟含补时）
    result      text,                                -- H 主胜 / D 平 / A 客胜
    ht_home_goals integer, ht_away_goals integer,    -- 半场比分（暂不用）
    home_shots integer, away_shots integer,
    home_shots_on_target integer, away_shots_on_target integer,
    home_corners integer, away_corners integer,
    home_fouls integer, away_fouls integer,
    home_yellow integer, away_yellow integer,
    home_red integer, away_red integer,
    home_xg real, away_xg real,
    referee     text,
    primary key (match_id, source)
);

-- 赔率：一行是“一场比赛 × 一家公司 × 一种玩法 × 一个时点”的一组赔率（欧洲小数赔率）
create table if not exists odds (
    match_id    integer not null references matches(match_id),
    source      text not null,
    bookmaker   text not null,   -- 公司名，按数据字典的写法，例如 平博 Pinnacle、多家平均
    market      text not null,   -- 1x2 胜平负 / ou 大小球 / ah 亚盘
    timing      text not null,   -- pre 赛前（football-data：开赛前 1–3 天）/ close 收盘（开球时）
    line        real,            -- 盘口：大小球是几球（2.5）；亚盘是主队让球数，负数 = 主队让球（-0.5 主让半球）
    home real, draw real, away real,  -- 胜平负：主胜 平 客胜；亚盘：home = 主队赔率，away = 客队赔率
    over real, under real,            -- 大小球：大 小
    primary key (match_id, source, bookmaker, market, timing)
) without rowid;
-- without rowid：数据直接按主键排好存放，不再另存一份主键索引。
-- 实测（2026-10-30，189 万组赔率）这张表从约 330 MB 降到 135 MB。
-- 没有另建“按公司查”的索引：现在按公司统计都要扫整张表，索引帮不上忙，只占地方。
"""


def create_tables(con: sqlite3.Connection) -> None:
    """建表（已经有的表不动）。"""
    con.executescript(TABLES)
