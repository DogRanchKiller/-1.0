"""整批重导时保护预测的比赛身份和最后已知赛果，不改动预测存档本身。

三个导入器共用同一套规则。所有写操作都参加调用者的导入事务，
这里不建表、不提交；任一步失败时，来源、统计、缺源标记一起回滚。
"""
import json
import sqlite3
from datetime import datetime, timezone


# 预测的所有 kind 都受保护，不能只保护正式时点而漏掉 preview。
PROTECTED_MISSING = ("match_id in (select match_id from predictions)"
                     " and match_id not in (select match_id from match_sources)")
UNPREDICTED_MISSING = ("select match_id from matches"
                       " where match_id not in (select match_id from match_sources)"
                       " and match_id not in (select match_id from predictions)")
RESULT_TABLES = ("match_stats", "cup_details")


def new_match_ids(con: sqlite3.Connection):
    """在调用者的写事务内分配新编号，同时避开旧孤立预测占用过的编号。

    SQLite 的普通整数主键可能复用已删掉的最大编号。新编号必须高于当前比赛和所有历史
    预测引用的编号；不能据自然信息把孤立预测接到新插入的比赛。已有自然键触发更新时，
    导入器仍保留原 match_id，本次候选编号只被跳过。这里不写序列表、不另开或提交事务。
    """
    highest = con.execute(
        "select max(mid) from (select coalesce(max(match_id), 0) as mid from matches"
        " union all select coalesce(max(match_id), 0) as mid from predictions)").fetchone()[0]
    candidate = int(highest) + 1
    while candidate <= 2**63 - 1:
        yield candidate
        candidate += 1
    raise OverflowError("比赛编号已超出 SQLite 整数范围，不能冒险复用历史预测的编号")


def snapshot_prediction_sources(con: sqlite3.Connection) -> dict[int, list[dict]]:
    """换掉来源前保存预测场的出处，只在确实发生缺源时写进状态表。"""
    previous: dict[int, list[dict]] = {}
    for mid, source, key, raw, fetched in con.execute(
            "select match_id, source, source_key, raw_file, fetched_at_utc from match_sources"
            " where match_id in (select match_id from predictions) order by match_id, source, source_key"):
        previous.setdefault(mid, []).append({"source": source, "source_key": key,
                                             "raw_file": raw, "fetched_at_utc": fetched})
    return previous


def delete_replaced_results(con: sqlite3.Connection, source: str,
                            tables: tuple[str, ...] = ("match_stats",),
                            competitions: list[str] | None = None) -> None:
    """先换完 match_sources，再删除本来源可重建的统计。

有预测且本轮没有任何当前来源的场次留下最后已知资料；其余仍按原规则重导。
维基百科只重导指定赛事时，沿用原来的赛事范围。
"""
    if any(table not in RESULT_TABLES for table in tables):
        raise ValueError("只允许替换赛果统计或杯赛详情表")
    scope, params = "", [source]
    if competitions is not None:
        ph = ",".join("?" * len(competitions))
        scope = f" and match_id in (select match_id from matches where competition in ({ph}))"
        params.extend(competitions)
    for table in tables:
        con.execute(f"delete from {table} where source=?{scope} and not ({PROTECTED_MISSING})", params)


def reconcile_source_lifecycle(con: sqlite3.Connection, previous_sources: dict[int, list[dict]],
                               now_utc: str | None = None) -> dict[str, int]:
    """换完来源和统计后处理缺源/恢复，并只清理没有预测的无来源比赛。

缺源时不触碰 matches、match_stats、cup_details 或 predictions。来源恢复时，
清掉该场没有当前来源对应的旧保留统计，避免旧比分盖住新来源的正常更新。
旧库里已经找不到比赛的预测不猜自然键、不新建比赛，仅返回数量供核查。
"""
    now = now_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    missing = [r[0] for r in con.execute(f"select match_id from matches where {PROTECTED_MISSING}")]
    new_missing = 0
    for mid in missing:
        last = json.dumps(previous_sources.get(mid, []), ensure_ascii=False, sort_keys=True)
        cur = con.execute(
            "insert into match_source_state (match_id, source_missing, missing_since_utc, restored_at_utc,"
            " missing_reason, last_sources_json) values (?, 1, ?, null, 'no_current_match_sources', ?)"
            " on conflict(match_id) do update set source_missing=1, missing_since_utc=excluded.missing_since_utc,"
            " restored_at_utc=null, missing_reason=excluded.missing_reason, last_sources_json=excluded.last_sources_json"
            " where match_source_state.source_missing=0", (mid, now, last))
        new_missing += cur.rowcount

    restoring = ("select match_id from match_source_state where source_missing=1"
                 " and match_id in (select match_id from match_sources)")
    for table in RESULT_TABLES:
        con.execute(f"delete from {table} where match_id in ({restoring}) and not exists"
                    f" (select 1 from match_sources s where s.match_id={table}.match_id and s.source={table}.source)")
    restored = con.execute(
        f"update match_source_state set source_missing=0, missing_reason=null, restored_at_utc=?"
        f" where match_id in ({restoring})", (now,)).rowcount

    # 无预测的垃圾不因这次保护而永久堆积；先清理派生子表，兼容启用外键的连接。
    for table in (*RESULT_TABLES, "odds", "odds_changes", "lottery_odds", "lineups", "match_source_state"):
        con.execute(f"delete from {table} where match_id in ({UNPREDICTED_MISSING})")
    deleted = con.execute(f"delete from matches where match_id in ({UNPREDICTED_MISSING})").rowcount
    unmatched = con.execute("select count(distinct match_id) from predictions"
                            " where match_id not in (select match_id from matches)").fetchone()[0]
    return {"missing": len(missing), "new_missing": new_missing,
            "restored": restored, "deleted": deleted, "unmatched_predictions": unmatched}
