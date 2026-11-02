"""用三个真实写库入口验证预测身份保全；全部输入与数据库都是临时假数据。"""
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.clean import dongqiudi_import as dqd
from src.clean import football_data_import as fd
from src.clean import wikipedia_import as wiki
from src.clean.schema import create_tables


TEAMS = [{"英文": t, "国家": "英格兰"} for t in "ABC"]
PROTECTED = ("A", "B", 2, 1)
CONTROL = ("B", "C", 1, 1)
UNUSED = ("C", "A", 0, 1)
FETCHED = "2026-11-01 00:00:00"


@dataclass
class Importer:
    name: str

    @property
    def competition(self):
        return "FAC" if self.name == "wikipedia" else "E0"

    def write(self, con, fixtures):
        """直接调用真实导入器，避免读取主项目的配置、文件与数据库。"""
        if self.name == "football-data":
            matches, stats = [], []
            for i, (home, away, hg, ag) in enumerate(fixtures):
                matches.append({"row": i, "competition": "E0", "season": 2026,
                                "home_team": home, "away_team": away, "match_date": "2026-11-01",
                                "kickoff_utc": "2026-11-01 15:00", "status": "finished",
                                "source_key": f"E0|2026-11-01|{home}|{away}",
                                "raw_file": "synthetic.csv", "fetched_at_utc": FETCHED})
                stats.append({**dict.fromkeys(fd.STAT_FIELDS), "row": i,
                              "home_goals": hg, "away_goals": ag,
                              "result": "H" if hg > ag else "A" if hg < ag else "D"})
            mcols = ["row", "competition", "season", "home_team", "away_team", "match_date",
                     "kickoff_utc", "status", "source_key", "raw_file", "fetched_at_utc"]
            return fd.write_database(con, TEAMS, {t: t for t in "ABC"},
                                     pd.DataFrame(matches, columns=mcols),
                                     pd.DataFrame(stats, columns=["row", *fd.STAT_FIELDS]),
                                     pd.DataFrame(columns=["row", *fd.ODDS_COLS[1:]]))
        if self.name == "dongqiudi":
            links = []
            for home, away, hg, ag in fixtures:
                did = {("A", "B"): 101, ("B", "C"): 102, ("C", "A"): 103}[home, away]
                mid = self.mid(con, home, away)
                fx = SimpleNamespace(status="Played", home_goals=hg, away_goals=ag,
                                     kickoff_utc=datetime(2026, 11, 1, 15, tzinfo=timezone.utc))
                seen = dqd.Seen("E0", 2026, fx, "synthetic.json", FETCHED)
                cand = dqd.Candidate(did, "E0", 2026, home, away, hg, ag,
                                     "synthetic.json", FETCHED, seen)
                links.append(dqd.Link(cand, mid, "新增（当前赛季）" if mid is None else "对上"))
            return dqd.write_db(con, links, {t: t for t in "ABC"}, {}, [], [],
                                3, ZoneInfo("Europe/London"))
        rows = []
        for home, away, hg, ag in fixtures:
            rows.append(wiki.Row("FAC", 2026, "F", 0, home, away, "2026-11-01", "day",
                                 "2026-11-01 15:00", "finished", hg, ag,
                                 "H" if hg > ag else "A" if hg < ag else "D",
                                 1, hg + 1, ag, None, None, "synthetic extra time", "synthetic.html",
                                 "synthetic page", home, away, "England", "England"))
        tm = wiki.TeamMap({t: t for t in "ABC"}, {}, {t["英文"]: t for t in TEAMS})
        return wiki.write_database(con, rows, tm, {"FAC": {"国家": "英格兰"}},
                                   {"synthetic.html": FETCHED})

    def mid(self, con, home="A", away="B"):
        row = con.execute("select match_id from matches where competition=? and season=2026"
                          " and home_team=? and away_team=?", (self.competition, home, away)).fetchone()
        return row[0] if row else None


@pytest.fixture(params=["football-data", "dongqiudi", "wikipedia"])
def importer(request):
    return Importer(request.param)


@pytest.fixture(params=[0, 1], ids=["fk_off", "fk_on"])
def con(request, tmp_path):
    db = sqlite3.connect(tmp_path / "synthetic.db")
    db.execute(f"pragma foreign_keys={request.param}")
    create_tables(db)
    db.executemany("insert into teams (team_id, country) values (?, '英格兰')", [(t,) for t in "ABC"])
    db.commit()
    yield db
    db.close()


def predict(con, mid, kind="preview"):
    """原样留存全部列，包含正式时点和模型版本，不能只数预测行数。"""
    con.execute("insert into predictions (match_id, competition, season, home_team, away_team, kickoff_utc,"
                " match_date, kind, model, model_version, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail)"
                " select match_id, competition, season, home_team, away_team, kickoff_utc, match_date, ?,"
                " 'test-model', 'test-v1', '2026-10-31 15:00', '2026-10-31 15:01', 0.5, 0.3, 0.2, ?"
                " from matches where match_id=?", (kind, '{"frozen_input":"original"}', mid))
    con.commit()


def records(con, table, mid=None):
    where = " where match_id=?" if mid is not None else ""
    return con.execute(f"select * from {table}{where} order by 1, 2", (mid,) if mid is not None else ()).fetchall()


@pytest.mark.parametrize("kind", ["preview", "T-24", "T-6", "T-1"])
def test_source_disappears_and_recovers_without_losing_prediction(con, importer, kind):
    importer.write(con, [PROTECTED, CONTROL, UNUSED])
    mid, unused = importer.mid(con), importer.mid(con, "C", "A")
    predict(con, mid, kind)
    predictions = records(con, "predictions")
    match, stats, cup = (records(con, t, mid) for t in ("matches", "match_stats", "cup_details"))

    importer.write(con, [CONTROL])
    # 先断言实际损坏：旧实现会直接丢掉比赛，或在启用外键时清空赛果后留下孤立预测。
    assert records(con, "matches", mid) == match
    assert records(con, "predictions") == predictions
    assert records(con, "match_stats", mid) == stats
    assert records(con, "cup_details", mid) == cup
    assert records(con, "matches", unused) == []
    assert records(con, "match_stats", unused) == []
    assert records(con, "match_sources", mid) == []
    state = con.execute("select source_missing, missing_since_utc, restored_at_utc, missing_reason,"
                        " last_sources_json from match_source_state where match_id=?", (mid,)).fetchone()
    assert state[:4] == (1, state[1], None, "no_current_match_sources")
    assert state[1]
    last = json.loads(state[4])
    assert len(last) == 1 and last[0]["source"] == importer.name
    assert last[0]["fetched_at_utc"] == FETCHED
    assert last[0]["raw_file"].startswith("synthetic")

    importer.write(con, [CONTROL])
    assert con.execute("select source_missing, missing_since_utc, restored_at_utc, missing_reason,"
                       " last_sources_json from match_source_state where match_id=?", (mid,)).fetchone() == state
    assert records(con, "matches", mid) == match
    assert records(con, "predictions") == predictions
    assert records(con, "match_stats", mid) == stats

    importer.write(con, [("A", "B", 3, 0), CONTROL])
    assert importer.mid(con) == mid
    assert records(con, "predictions") == predictions
    assert con.execute("select home_goals, away_goals from match_stats where match_id=?", (mid,)).fetchall() == [(3, 0)]
    restored = con.execute("select source_missing, missing_since_utc, restored_at_utc, missing_reason"
                           " from match_source_state where match_id=?", (mid,)).fetchone()
    assert restored[0:2] == (0, state[1]) and restored[2] and restored[3] is None
    importer.write(con, [("A", "B", 3, 0), CONTROL])
    assert con.execute("select source_missing, missing_since_utc, restored_at_utc, missing_reason"
                       " from match_source_state where match_id=?", (mid,)).fetchone() == restored
    assert con.execute("pragma foreign_key_check").fetchall() == []


def test_other_source_keeps_match_active_and_normal_replacement(con, importer):
    importer.write(con, [PROTECTED, CONTROL])
    mid = importer.mid(con)
    predict(con, mid)
    con.execute("insert into match_sources values (?, 'other', 'other-id', 'other.raw', ?)", (mid, FETCHED))
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                " values (?, 'other', 4, 0, 'H')", (mid,))
    con.commit()
    importer.write(con, [CONTROL])
    assert importer.mid(con) == mid
    # 仍有当前来源时，删除本来源旧资料的原规则继续生效。
    assert con.execute("select source, home_goals from match_stats where match_id=?", (mid,)).fetchall() == [("other", 4)]
    assert con.execute("select * from match_source_state where match_id=? and source_missing=1", (mid,)).fetchall() == []


def test_old_schema_adds_lifecycle_state_without_changing_rows(tmp_path):
    db = sqlite3.connect(tmp_path / "old.db")
    create_tables(db)
    Importer("football-data").write(db, [PROTECTED])
    mid = db.execute("select match_id from matches").fetchone()[0]
    predict(db, mid, "T-24")
    before = {t: records(db, t) for t in ("matches", "match_sources", "match_stats", "predictions")}
    db.execute("drop table if exists match_source_state")
    db.commit()
    create_tables(db)
    create_tables(db)
    assert db.execute("select * from match_source_state").fetchall() == []
    assert {t: records(db, t) for t in before} == before
    db.close()


@pytest.mark.parametrize("transition", ["missing", "restoring"])
def test_import_and_marker_roll_back_together(con, importer, transition):
    importer.write(con, [PROTECTED, CONTROL, UNUSED])
    mid = importer.mid(con)
    predict(con, mid, "T-1")
    if transition == "restoring":
        importer.write(con, [CONTROL])
        # AFTER 触发器等本次恢复标记确实已写才报错，不能只测试导入刚开始就失败。
        con.execute("create trigger fail_lifecycle after update on match_source_state"
                    " when new.source_missing=0 begin select raise(abort, 'synthetic rollback'); end")
        incoming = [("A", "B", 3, 0), ("B", "C", 4, 0)]
    else:
        con.execute("create trigger fail_lifecycle after insert on match_source_state"
                    " begin select raise(abort, 'synthetic rollback'); end")
        incoming = [("B", "C", 4, 0)]
    con.commit()
    tables = ("teams", "team_names", "matches", "match_sources", "match_stats", "cup_details",
              "odds", "odds_changes", "lottery_odds", "lineups", "predictions", "match_source_state")
    before = {t: records(con, t) for t in tables}
    with pytest.raises(sqlite3.IntegrityError, match="synthetic rollback"):
        importer.write(con, incoming)
    assert {t: records(con, t) for t in tables} == before
    assert con.execute("pragma foreign_key_check").fetchall() == []


def test_other_source_restores_and_replaces_retained_result(con, importer):
    importer.write(con, [PROTECTED, CONTROL])
    mid = importer.mid(con)
    predict(con, mid)
    original_predictions = records(con, "predictions")
    importer.write(con, [CONTROL])
    con.execute("insert into match_sources values (?, 'new-source', 'new-key', 'new.raw', ?)", (mid, FETCHED))
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                " values (?, 'new-source', 5, 0, 'H')", (mid,))
    con.commit()
    # 任何导入器都应识别来源恢复；旧保留赛果不能遮住新来源的答案。
    importer.write(con, [CONTROL])
    assert importer.mid(con) == mid
    assert records(con, "predictions") == original_predictions
    assert con.execute("select source, home_goals, away_goals from match_stats where match_id=?", (mid,)).fetchall() == [
        ("new-source", 5, 0)]
    assert records(con, "cup_details", mid) == []
    assert con.execute("select source_missing, missing_reason from match_source_state where match_id=?", (mid,)).fetchone() == (0, None)


def test_empty_import_protects_only_predicted_matches(con, importer):
    importer.write(con, [PROTECTED, CONTROL, UNUSED])
    mid = importer.mid(con)
    predict(con, mid, "T-24")
    before = {t: records(con, t, mid) for t in ("matches", "predictions", "match_stats", "cup_details")}
    importer.write(con, [])
    assert {t: records(con, t, mid) for t in before} == before
    assert con.execute("select match_id from matches").fetchall() == [(mid,)]
    assert con.execute("select source_missing from match_source_state where match_id=?", (mid,)).fetchone() == (1,)


@pytest.mark.parametrize("orphan_id", [1, 999])
def test_new_matches_never_reuse_orphan_prediction_ids(con, importer, orphan_id):
    from src.clean.source_lifecycle import reconcile_source_lifecycle

    # 高编号用例让现有最大编号刚好比孤立预测小一，复现 SQLite 自动分配下一编号的碰撞。
    existing_id = orphan_id - 1 if orphan_id > 1 else None
    if existing_id is not None:
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date,"
                    " kickoff_utc, status, round) values (?, ?, 2026, 'B', 'C', '2026-11-01',"
                    " '2026-11-01 15:00', 'finished', ?)",
                    (existing_id, importer.competition, "F" if importer.name == "wikipedia" else ""))
    con.execute("insert into predictions (match_id, competition, season, home_team, away_team, match_date,"
                " kind, model, model_version, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail)"
                " values (?, ?, 2026, 'A', 'B', '2026-11-01', 'preview', 'old-model', 'old-v1',"
                " '2026-10-31 15:00', '2026-10-31 15:01', 0.5, 0.3, 0.2, ?)",
                (orphan_id, importer.competition, '{"frozen_input":"orphan-original"}'))
    con.commit()
    original_predictions = records(con, "predictions")

    # 即使新行的自然信息恰好与旧存档一致，也不能猜它就是旧比赛而替预测复用编号。
    importer.write(con, [PROTECTED, CONTROL])
    assert records(con, "matches", orphan_id) == []
    assert importer.mid(con) > orphan_id
    if existing_id is not None:
        assert importer.mid(con, "B", "C") == existing_id
    assert records(con, "predictions") == original_predictions
    assert records(con, "match_sources", orphan_id) == []
    assert records(con, "match_source_state", orphan_id) == []
    assert reconcile_source_lifecycle(con, {})["unmatched_predictions"] == 1
    assert con.execute("pragma foreign_key_check").fetchall() == []


def test_lifecycle_transitions_keep_episode_time_and_do_not_guess_old_orphans(con):
    from src.clean.source_lifecycle import reconcile_source_lifecycle, snapshot_prediction_sources

    importer = Importer("football-data")
    importer.write(con, [PROTECTED])
    mid = importer.mid(con)
    predict(con, mid)
    # 旧版本已经删了比赛的预测：即使自然信息相同，也不替它猜编号或改原存档。
    con.execute("insert into predictions select 999, 999, competition, season, home_team, away_team, kickoff_utc,"
                " match_date, kind, model, model_version, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail"
                " from predictions limit 1")
    original_predictions = records(con, "predictions")
    previous = snapshot_prediction_sources(con)
    con.execute("delete from match_sources")
    first = reconcile_source_lifecycle(con, previous, "2026-11-02 00:00:00")
    assert first == {"missing": 1, "new_missing": 1, "restored": 0, "deleted": 0, "unmatched_predictions": 1}
    first_state = records(con, "match_source_state")
    repeat = reconcile_source_lifecycle(con, {}, "2026-11-02 01:00:00")
    assert repeat["new_missing"] == 0 and records(con, "match_source_state") == first_state
    con.execute("insert into match_sources values (?, 'restored', 'r', 'restored.raw', ?)", (mid, FETCHED))
    assert reconcile_source_lifecycle(con, {}, "2026-11-02 02:00:00")["restored"] == 1
    assert con.execute("select restored_at_utc from match_source_state where match_id=?", (mid,)).fetchone() == ("2026-11-02 02:00:00",)
    again_previous = snapshot_prediction_sources(con)
    con.execute("delete from match_sources")
    assert reconcile_source_lifecycle(con, again_previous, "2026-11-03 00:00:00")["new_missing"] == 1
    again = con.execute("select missing_since_utc, restored_at_utc, last_sources_json from match_source_state"
                        " where match_id=?", (mid,)).fetchone()
    assert again[0:2] == ("2026-11-03 00:00:00", None)
    assert json.loads(again[2])[0]["source"] == "restored"
    assert con.execute("select match_id from matches").fetchall() == [(mid,)]
    assert records(con, "predictions") == original_predictions


def test_wikipedia_partial_import_preserves_other_competition(con):
    importer = Importer("wikipedia")
    importer.write(con, [PROTECTED, CONTROL])
    con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, status, round)"
                " values (100, 'UCL', 2026, 'C', 'A', '2026-11-01', 'finished', 'F')")
    con.execute("insert into match_sources values (100, 'wikipedia', 'ucl-key', 'ucl.raw', ?)", (FETCHED,))
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                " values (100, 'wikipedia', 3, 1, 'H')")
    con.commit()
    mid = importer.mid(con)
    predict(con, mid)
    before_ucl = {t: records(con, t, 100) for t in ("matches", "match_sources", "match_stats")}
    tm = wiki.TeamMap({}, {}, {t["英文"]: t for t in TEAMS})
    wiki.write_database(con, [], tm, {"FAC": {}, "UCL": {}}, {}, codes=["FAC"])
    assert {t: records(con, t, 100) for t in before_ucl} == before_ucl
    assert con.execute("select source_missing from match_source_state where match_id=?", (mid,)).fetchone() == (1,)
