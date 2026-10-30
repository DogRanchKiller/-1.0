"""数据库连接和公共表。

数据库是一个 SQLite 文件（位置在 settings.yaml 的 paths.database）。
这里只放各模块共用的表；各类比赛数据的表在清洗模块里建。
"""
import sqlite3
from pathlib import Path

from src.common.config import load_settings, resolve_path

# 抓取记录：每次下载或抓取都记一行，成功失败都记（需求文档第 17 节“原样存档”）
FETCH_LOG_SCHEMA = """
create table if not exists fetch_log (
    id             integer primary key,
    url            text not null,
    source         text not null,   -- 来源，例如 football-data
    data_type      text not null,   -- 数据类型，例如 league_csv
    fetched_at_utc text not null,   -- 抓取时间（标准时间）
    status         text not null,   -- saved 已保存 / unchanged 没变化 / not_found 网站没有 / suspicious 内容可疑 / failed 失败
    http_status    integer,
    file_path      text,            -- 保存位置（相对于项目文件夹）
    bytes          integer,
    sha256         text,            -- 文件指纹：内容一样指纹就一样，用来判断有没有变化
    note           text
)
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """打开数据库（不存在就新建），并确保公共表存在。"""
    path = db_path or resolve_path(load_settings()["paths"]["database"])
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(FETCH_LOG_SCHEMA)
    con.commit()
    return con
