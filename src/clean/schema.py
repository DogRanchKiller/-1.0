"""数据库里清洗层的正式表（需求文档第 17 节“统一标准”“按数据类型放进正式表”）。

历史数据和以后爬到的实时数据放在同一套表里，用“来源”字段区分，不分两个数据库。
football-data、维基百科（杯赛）、懂球帝（变盘记录、彩票赔率）用到的表；首发、伤停等表以后再加。

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

-- 比赛：一行一场比赛（需求文档第 16 节“同一场比赛的匹配”）。
-- 联赛：同一联赛同一赛季同一对主客队只有一场，round 记空字符串 ''，leg 记 0。
-- 杯赛、欧战：同一对球队一个赛季可能碰好几次（小组赛一次、淘汰赛两回合、重赛），
--   所以要加上 round（哪一轮）和 leg（第几回合）才能分清。
create table if not exists matches (
    match_id       integer primary key,
    competition    text not null,   -- 赛事代码，联赛用 football-data 的代码（例如 E0），杯赛用 settings.yaml wikipedia 里的代码（例如 UCL）
    season         integer not null,-- 赛季开始年份，2023 表示 2023/24
    home_team      text not null references teams(team_id),
    away_team      text not null references teams(team_id),
    match_date     text not null,   -- 比赛日期（比赛当地的日期，yyyy-mm-dd）
    kickoff_utc    text,            -- 开球时间（标准时间 yyyy-mm-dd HH:MM）；不知道时区或来源没有时是空的
    status         text not null,   -- scheduled 未赛 / finished 已赛 / awarded 判罚比分 / postponed 改期 / cancelled 取消
    round          text not null default '',  -- 轮次：联赛是 ''；杯赛例如 Q2 第二轮资格赛、G 小组赛、R16 十六强、F 决赛，
                                              -- 重赛后面加 _replay（代码含义见 src/clean/wikipedia_import.py 的 ROUND_RULES）
    leg            integer not null default 0,-- 第几回合：0 = 单场；1、2 = 两回合的第 1、2 回合
    date_precision text not null default 'day',-- 日期精确到：day 具体哪一天 / round 只知道这一轮从哪天开始（match_date 记这一轮的第一天）
    unique (competition, season, home_team, away_team, round, leg)
);

-- 杯赛的加时、点球（只有杯赛、欧战有，一场比赛一个来源一行）。
-- match_stats 里的比分一律是 90 分钟（含补时）的比分，和联赛口径一样；踢了加时的，加时后的比分在这里。
create table if not exists cup_details (
    match_id      integer not null references matches(match_id),
    source        text not null,
    extra_time    integer not null,  -- 1 = 踢了加时（含金球、银球）
    aet_home integer, aet_away integer,   -- 加时后的比分（没踢加时就是空的）
    pen_home integer, pen_away integer,   -- 点球大战比分（没有就是空的）
    note          text,              -- 例如 golden goal 金球、awarded 判罚比分
    primary key (match_id, source)
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

-- 变盘记录：一行是“一场比赛 × 一家公司 × 一种玩法”在某个时间的一组价格（需求文档第 17 节“变盘记录”）。
-- 和 odds 表的区别：odds 只有“赛前”“收盘”两个固定时点；这里每行带网站显示的变盘时间，
-- 同一家公司同一场可以有很多行（懂球帝的初盘、最新，加上我们每小时存的快照里看到的每个价格）。
-- 赔率都是小数赔率（含本金），盘口写法和 odds 表一样。
create table if not exists odds_changes (
    match_id       integer not null references matches(match_id),
    source         text not null,   -- 来源，例如 dongqiudi
    bookmaker      text not null,   -- 公司名，按公司对照表换过（config/bookmakers_dongqiudi.yaml）；没对照的照原样
    bookmaker_raw  text not null,   -- 网站上原样的公司名（例如 Bet3**），以后改对照表时有据可查
    market         text not null,   -- 1x2 胜平负 / ou 大小球 / ah 亚盘
    line           real,            -- 盘口：大小球是几球；亚盘是主队让球数，负数 = 主队让球；胜平负是空
    home real, draw real, away real,-- 胜平负：主胜 平 客胜；亚盘：home = 主队，away = 客队
    over real, under real,          -- 大小球：大 小
    changed_at_utc text not null,   -- 网站显示的变盘时间（标准时间 yyyy-mm-dd HH:MM:SS，网站只到分钟）。取数层按它算能不能用
    fetched_at_utc text not null,   -- 我们第一次看到这个价格的时间（标准时间 yyyy-mm-dd HH:MM:SS）
    raw_file       text not null    -- 第一次看到它的原始文件
);
-- 同一个价格在好几个快照里重复出现时只留一行（比赛 + 公司 + 玩法 + 盘口 + 时间 + 赔率都一样算重复，导入程序里去重）
create index if not exists odds_changes_match on odds_changes (match_id, bookmaker, market, changed_at_utc);

-- 彩票赔率（竞彩、北单）：只显示，不进模型（数据用途字典“竞彩赔率”）。
-- 懂球帝的彩票赔率没有变盘时间，只能按“我们什么时候看到的”算能不能知道。
create table if not exists lottery_odds (
    match_id       integer not null references matches(match_id),
    source         text not null,
    product        text not null,   -- 竞彩胜平负 / 竞彩让球 / 北单
    handicap       integer not null,-- 主队让几球，写法和亚盘一样：-1 = 主队让 1 球，+1 = 主队受让 1 球；不让球是 0
    home real, draw real, away real,-- 主胜 平 客胜（让球的按让球后的结果算）
    fetched_at_utc text not null,   -- 我们第一次看到这组赔率的时间（标准时间 yyyy-mm-dd HH:MM:SS）。取数层按它算能不能用
    raw_file       text not null
);
create index if not exists lottery_odds_match on lottery_odds (match_id, product, fetched_at_utc);
"""


def create_tables(con: sqlite3.Connection) -> None:
    """建表（已经有的表不动），旧版的比赛表自动升级（见 migrate_matches）。"""
    con.executescript(TABLES)
    migrate_matches(con)


# 升级后比赛表的列（顺序和上面建表语句一致）
MATCH_COLUMNS = ["match_id", "competition", "season", "home_team", "away_team", "match_date",
                 "kickoff_utc", "status", "round", "leg", "date_precision"]


def migrate_matches(con: sqlite3.Connection) -> bool:
    """把旧版比赛表（唯一键只有 联赛、赛季、主队、客队）升级成带 round、leg、date_precision 的新版。

    SQLite 不能直接修改唯一键，只能“建新表 → 原样复制 → 删旧表 → 新表改名”（SQLite 官方文档推荐的做法）。
    复制时 match_id 原样带过去，所以已有比赛的编号一个都不变；旧比赛（都是联赛）round 记 ''、leg 记 0、
    date_precision 记 day。整个过程在一个事务里，中途出错就全部不算。
    已经是新版就什么都不做，返回 False；升级了返回 True。
    """
    cols = [r[1] for r in con.execute("pragma table_info(matches)")]
    if "round" in cols:
        return False
    before = con.execute("select count(*), coalesce(sum(match_id), 0) from matches").fetchone()
    new_table = TABLES.split("create table if not exists matches (")[1].split(");")[0]
    con.commit()
    fk_was = con.execute("pragma foreign_keys").fetchone()[0]
    con.execute("pragma foreign_keys = off")   # 其他表引用了 matches，换表期间先关掉外键检查
    try:
        with con:
            con.execute(f"create table matches_new ({new_table})")
            old = [c for c in MATCH_COLUMNS if c in cols]
            con.execute(f"insert into matches_new ({', '.join(old)}) select {', '.join(old)} from matches")
            con.execute("drop table matches")
            con.execute("alter table matches_new rename to matches")
            after = con.execute("select count(*), coalesce(sum(match_id), 0) from matches").fetchone()
            if after != before:
                raise RuntimeError(f"比赛表升级前后对不上：之前 {before}，之后 {after}")
    finally:
        con.execute(f"pragma foreign_keys = {'on' if fk_was else 'off'}")   # 恢复原来的设置
    return True
