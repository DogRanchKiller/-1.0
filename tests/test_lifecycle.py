"""逐场生命周期反例：只用内存库和临时复盘文件，不运行生产任务。"""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.models.registry import model_config
from src.predict.preview import model_version
from src.review.reconcile import parse_version, score_metrics


KO = "2026-11-03 12:00:00"


@pytest.fixture
def setup(tmp_path):
    con = sqlite3.connect(":memory:")
    create_tables(con)
    con.executemany("insert into teams(team_id,country) values (?, 'England')", [("A",), ("B",)])
    con.execute("insert into matches(match_id,competition,season,home_team,away_team,match_date,kickoff_utc,status) "
                "values(1,'E0',2026,'A','B','2026-11-03',?,'scheduled')", (KO,))
    cfg = load_settings()
    cfg["site"]["review_dir"] = str(tmp_path)
    cfg["lifecycle"] = {"enabled_since_utc": "2026-11-02 01:47:37", "grace_minutes": 20,
                        "score_tolerance": 0.0000011}
    con.commit()
    yield con, cfg, tmp_path
    con.close()


def archive(setup, pid=1, kind="T-24", cutoff="2026-11-02 12:00:00", created=None, **changes):
    con, cfg, _ = setup
    h = int(kind[2:]) if kind.startswith("T-") else None
    late = (pd.Timestamp(cutoff) - (pd.Timestamp(KO) - pd.Timedelta(hours=h))).total_seconds() / 60 if h else None
    row = {"prediction_id": pid, "match_id": 1, "competition": "E0", "season": 2026,
           "home_team": "A", "away_team": "B", "match_date": "2026-11-03", "kickoff_utc": KO,
           "kind": kind, "model": cfg["preview"]["model"],
           "model_version": model_version(cfg["preview"]["model"], model_config(cfg["preview"]["model"], cfg)),
           "cutoff_utc": cutoff, "created_at_utc": created or cutoff,
           "p_home": .5, "p_draw": .3, "p_away": .2,
           "detail": json.dumps({"计划时点": h, "晚了几分钟": round(late, 1) if late is not None else None,
                                 "首发": "未含首发"}, ensure_ascii=False)}
    row.update(changes)
    con.execute(f"insert into predictions({','.join(row)}) values({','.join('?' for _ in row)})", list(row.values()))
    con.commit()
    return row


def report(setup, now="2026-11-02 12:30:00"):
    from src.review.lifecycle import build_report
    return build_report(setup[0], setup[1], now)


def slot(setup, now="2026-11-02 12:30:00", kind="T-24"):
    return next(x for x in report(setup, now)["matches"][0]["slots"] if x["kind"] == kind)


@pytest.mark.parametrize("now,state", [("2026-11-02 11:59:59", "not_due"),
    ("2026-11-02 12:00:00", "awaiting_scheduler"), ("2026-11-02 12:20:00", "awaiting_scheduler"),
    ("2026-11-02 12:20:01", "overdue"), ("2026-11-03 06:00:00", "missed_window")])
def test_missing_slot_distinguishes_future_grace_overdue_and_closed_window(setup, now, state):
    assert slot(setup, now)["state"] == state


def test_activation_does_not_retroactively_accuse_old_slots(setup):
    setup[1]["lifecycle"]["enabled_since_utc"] = "2026-11-02 13:00:00"
    assert slot(setup)["state"] == "before_activation"
    archive(setup, cutoff="2026-11-02 13:01:00")
    s = slot(setup, "2026-11-02 13:02:00")
    assert s["state"] == "late" and not s["on_time"] and not s["tracked"]


def test_grace_never_relabels_two_minute_or_one_second_delay_as_on_time(setup):
    archive(setup, cutoff="2026-11-02 12:02:00", created="2026-11-02 12:03:00")
    s = slot(setup)
    assert s["state"] == "late" and s["late_minutes"] == 2 and s["created_late_minutes"] == 3
    archive(setup, pid=2, cutoff="2026-11-02 12:00:01")
    s = slot(setup)
    assert s["state"] == "late" and s["late_minutes"] > 0 and not s["on_time"]


def test_real_creation_delay_is_independent_of_cutoff_metadata(setup):
    archive(setup, created="2026-11-02 12:01:06")
    s = slot(setup)
    assert s["state"] == "late" and s["late_minutes"] == 0 and s["created_late_minutes"] == 1.1


@pytest.mark.parametrize("changes,fragment", [
    ({"home_team": "wrong"}, "身份"), ({"kickoff_utc": "2026-11-04 12:00:00"}, "旧赛程"),
    ({"model_version": "old-version"}, "模型版本"), ({"model": "other"}, "模型"),
    ({"created_at_utc": "2026-11-04 12:00:00"}, "开球"),
    ({"created_at_utc": "2026-11-02 12:31:00"}, "评价时刻"),
    ({"created_at_utc": "2026-11-02 11:59:59"}, "早于截止"),
    ({"p_home": float("inf")}, "概率"),
    ({"detail": '{"计划时点": 24, "晚了几分钟": 100}'}, "元数据"),
])
def test_invalid_archive_cannot_complete_a_stage(setup, changes, fragment):
    archive(setup, **changes)
    s = slot(setup)
    assert s["state"] == "invalid_archive" and s["prediction_id"] is None
    assert any(fragment in x for x in s["issues"])


def test_next_stage_boundary_rejects_backfilled_and_late_created_previous_stage(setup):
    archive(setup, cutoff="2026-11-03 06:00:00")
    archive(setup, pid=2, created="2026-11-03 06:00:00")
    s = slot(setup, "2026-11-03 06:01:00")
    assert s["state"] == "invalid_archive"
    assert len(s["archives"]) == 2 and all(not x["valid"] for x in s["archives"])


def test_all_rows_are_examined_and_future_latest_cannot_hide_valid_earlier_row(setup):
    archive(setup)
    archive(setup, pid=2, created="2026-11-04 12:00:00")
    s = slot(setup)
    assert s["state"] == "on_time" and s["prediction_id"] == 1
    assert s["valid_prediction_ids"] == [1] and len(s["archives"]) == 2
    assert s["archives"][1]["issues"]


def test_preview_never_fills_formal_slot(setup):
    archive(setup, kind="preview")
    r = report(setup)
    assert r["matches"][0]["preview_count"] == 1
    assert r["summary"]["recorded_slot_count"] == 0


@pytest.mark.parametrize("sql", ["update matches set kickoff_utc=null", "update matches set status='cancelled'",
                                  "update matches set status='postponed'"])
def test_imprecise_or_inactive_fixture_is_not_a_missed_run(setup, sql):
    setup[0].execute(sql)
    setup[0].commit()
    assert all(s["state"] == "not_applicable" for s in report(setup)["matches"][0]["slots"])


def finish_and_save_review(setup, matrix=None):
    con, cfg, folder = setup
    con.execute("update matches set status='finished'")
    con.execute("insert into match_stats(match_id,source,home_goals,away_goals,result) values(1,'football-data',2,1,'H')")
    rows = [archive(setup, pid=i, kind=f"T-{h}", cutoff=str(pd.Timestamp(KO) - pd.Timedelta(hours=h)))
            for i, h in enumerate([24, 6, 1], 1)]
    if matrix is None:
        matrix = np.zeros((11, 11))
        matrix[0, 0], matrix[0, 1], matrix[1, 0] = .3, .2, .5
        matrix = matrix.tolist()
    scores, entries = [], []
    for row in rows:
        row["detail"] = json.dumps({"计划时点": int(row["kind"][2:]), "晚了几分钟": 0,
                                    "首发": "未含首发", "模型比分矩阵": matrix})
        con.execute("update predictions set detail=? where prediction_id=?", (row["detail"], row["prediction_id"]))
        v = parse_version(pd.Series(row), cfg["review"]["prob_sum_tol"])
        met = score_metrics(v["model_p"], v["model_matrix"], 2, 1)
        scores.append({**{k: row[k] for k in ["prediction_id", "match_id", "competition", "season", "kickoff_utc", "kind",
                                             "model", "model_version", "cutoff_utc", "created_at_utc"]},
                       "source": "模型", "home_goals": 2, "away_goals": 1,
                       "p_home": .5, "p_draw": .3, "p_away": .2, **met})
        entries.append({"prediction_id": row["prediction_id"], "版本": row["kind"], "model": row["model"],
                        "model_version": row["model_version"], "cutoff_utc": row["cutoff_utc"],
                        "created_at_utc": row["created_at_utc"],
                        "模型": {"概率": [.5, .3, .2], **{k: round(v, 4) if v is not None else None for k, v in met.items()}}})
    pd.DataFrame(scores).to_csv(folder / "scores.csv", index=False, float_format="%.6f")
    doc = {"match_id": 1, "联赛": "E0", "season": 2026, "主队": "A", "客队": "B",
           "比赛日期": "2026-11-03", "开球时间_utc": KO, "实际": {"比分": [2, 1]}, "各版本": entries,
           "生成时间_utc": "2026-11-03 15:01:00"}
    (folder / "matches").mkdir(exist_ok=True)
    (folder / "matches" / "1.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    con.commit()
    return doc


def test_valid_scores_and_json_do_not_imply_closing_or_lineup_complete(setup):
    finish_and_save_review(setup)
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert m["archives_complete"] and m["score_complete"]
    assert m["review"]["state"] == "complete" and m["review"]["review_file_valid"]
    assert m["closing"]["state"] == "missing" and len(m["closing"]["groups"]) == 4
    assert not m["complete_on_time"] and m["lineup"]["state"] == "unverified"


def test_missing_model_score_matrix_cannot_count_as_complete(setup):
    finish_and_save_review(setup)
    setup[0].execute("update predictions set detail=json_object('计划时点', cast(substr(kind, 3) as integer), '晚了几分钟', 0, '首发', '未含首发')")
    setup[0].commit()
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert not m["score_complete"]
    assert m["review"]["state"] in {"partial", "invalid"}
    assert any("矩阵" in x for x in m["review"]["issues"])


@pytest.mark.parametrize("column,value", [("prediction_id", 99), ("model_version", "old"), ("kind", "preview"),
    ("kickoff_utc", "2026-11-04 12:00:00"), ("home_goals", 0), ("p_home", .7),
    ("logloss_1x2", float("inf")), ("brier_1x2", .99)])
def test_stale_or_corrupt_score_is_not_a_completed_stage(setup, column, value):
    finish_and_save_review(setup)
    path = setup[2] / "scores.csv"
    scores = pd.read_csv(path)
    scores.loc[0, column] = value
    scores.to_csv(path, index=False)
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert not m["score_complete"] and m["review"]["state"] in {"partial", "invalid"}


@pytest.mark.parametrize("change", ["team", "kickoff", "score", "pid", "future", "probability"])
def test_json_has_independent_identity_and_time_validation(setup, change):
    doc = finish_and_save_review(setup)
    if change == "team": doc["主队"] = "wrong"
    if change == "kickoff": doc["开球时间_utc"] = "2026-11-04 12:00:00"
    if change == "score": doc["实际"]["比分"] = [0, 1]
    if change == "pid": doc["各版本"][0]["prediction_id"] = 99
    if change == "future": doc["生成时间_utc"] = "2026-11-03 17:00:00"
    if change == "probability": doc["各版本"][0]["模型"]["概率"] = [.6, .2, .2]
    (setup[2] / "matches" / "1.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert not m["score_complete"] and m["review"]["issues"]


def test_closing_groups_are_independent_and_only_registered_ou25_is_qualified(setup):
    finish_and_save_review(setup)
    con = setup[0]
    con.execute("insert into match_sources values(1,'football-data','E0:1','data/raw/fd.csv','2026-11-03 15:00:00')")
    con.execute("insert into odds(match_id,source,bookmaker,market,timing,line,home,draw,away) "
                "values(1,'football-data','多家平均','1x2','close',null,2,3.5,4)")
    con.execute("insert into odds(match_id,source,bookmaker,market,timing,line,over,under) "
                "values(1,'football-data','多家平均','ou','close',3.5,1.9,2.0)")
    con.commit()
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert m["closing"]["state"] == "partial"
    assert sum(g["qualified"] for g in m["closing"]["groups"]) == 1
    assert not m["complete_on_time"]


def test_readonly_main_database_and_json_safe_zero_samples(setup):
    con = setup[0]
    con.execute("delete from matches")
    con.commit()
    denied = []
    def authorize(action, _a, _b, db, _trigger):
        if db == "main" and action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                                        sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE}:
            denied.append(action)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    con.set_authorizer(authorize)
    r = report(setup)
    assert r["summary"]["match_count"] == 0 and r["summary"]["complete_on_time_count"] == 0
    json.dumps(r, allow_nan=False)
    assert not denied


def test_missing_prediction_table_is_explicit_warning_not_clean_empty_archive(setup):
    setup[0].execute("drop table predictions")
    setup[0].commit()
    r = report(setup)
    assert r["warnings"] and "predictions" in " ".join(r["warnings"])


def add_all_closes(setup):
    con, cfg, _ = setup
    con.execute("insert into match_sources values(1,'football-data','E0:1','data/raw/fd.csv','2026-11-03 15:00:00')")
    for book in cfg["review"]["close_bookmakers"]:
        con.execute("insert into odds(match_id,source,bookmaker,market,timing,home,draw,away) "
                    "values(1,'football-data',?,'1x2','close',2,3.5,4)", (book,))
        con.execute("insert into odds(match_id,source,bookmaker,market,timing,line,over,under) "
                    "values(1,'football-data',?,'ou','close',2.5,1.9,2.0)", (book,))
    con.commit()


@pytest.mark.parametrize("source_state", ["missing_table", "missing_column", "missing_row", None, 2, -1, "bad"])
def test_unknown_source_state_never_counts_as_complete(setup, source_state):
    finish_and_save_review(setup)
    add_all_closes(setup)
    con = setup[0]
    con.execute("drop table match_source_state")
    if source_state in {"missing_table", "missing_column", "missing_row"}:
        con.execute("delete from match_sources")
    if source_state != "missing_table":
        columns = "match_id integer" if source_state == "missing_column" else "match_id integer, source_missing"
        con.execute(f"create table match_source_state({columns})")
        if source_state not in {"missing_column", "missing_row"}:
            con.execute("insert into match_source_state values(1,?)", (source_state,))
    con.commit()
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert m["archives_complete"] and m["score_complete"]
    assert m["source_state"]["state"] == "unknown"
    assert not m["complete_on_time"]


@pytest.mark.parametrize("state_schema", ["missing_table", "missing_column", "missing_row", "restored"])
def test_current_source_allows_sparse_missing_state_table_to_complete(setup, state_schema):
    finish_and_save_review(setup)
    add_all_closes(setup)
    if state_schema == "restored":
        setup[0].execute("insert into match_source_state(match_id,source_missing,missing_since_utc,last_sources_json) "
                         "values(1,0,'2026-11-01 00:00:00','[]')")
    elif state_schema != "missing_row":
        setup[0].execute("drop table match_source_state")
        if state_schema == "missing_column":
            setup[0].execute("create table match_source_state(match_id integer)")
    setup[0].commit()
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert m["source_state"]["state"] == "present" and m["complete_on_time"]


def test_explicit_restored_flag_without_current_source_is_still_unknown(setup):
    setup[0].execute("insert into match_source_state(match_id,source_missing,missing_since_utc,last_sources_json) "
                     "values(1,0,'2026-11-01 00:00:00','[]')")
    setup[0].commit()
    m = report(setup)["matches"][0]
    assert m["source_state"]["state"] == "unknown" and not m["complete_on_time"]


@pytest.mark.parametrize("defect", ["wrong_shape", "wrong_1x2", "excess_rounding", "string_cell", "bool_cell"])
def test_gb_l7_matrix_must_match_archive_contract_even_if_old_scores_agree(setup, defect):
    matrix = np.zeros((11, 11))
    matrix[0, 0], matrix[0, 1], matrix[1, 0] = .3, .2, .5
    if defect == "wrong_shape": matrix = matrix[:2, :2]
    if defect == "wrong_1x2": matrix[0, 0], matrix[0, 1] = .25, .25
    if defect == "excess_rounding": matrix *= .9995
    matrix = matrix.tolist()
    if defect == "string_cell": matrix[0][0] = "0.3"
    if defect == "bool_cell": matrix[10][10] = False
    finish_and_save_review(setup, matrix)
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert not m["score_complete"]
    assert any("矩阵" in issue for issue in m["review"]["issues"])


@pytest.mark.parametrize("metric", ["logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts"])
@pytest.mark.parametrize("artifact", ["csv", "json"])
def test_every_metric_is_independently_required(setup, metric, artifact):
    doc = finish_and_save_review(setup)
    if artifact == "csv":
        path = setup[2] / "scores.csv"
        scores = pd.read_csv(path)
        scores.loc[0, metric] = float("nan")
        scores.to_csv(path, index=False)
    else:
        del doc["各版本"][0]["模型"][metric]
        (setup[2] / "matches" / "1.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert not m["score_complete"] and any(metric in issue for issue in m["review"]["issues"])


@pytest.mark.parametrize("method", ["prediction_targets", "historical_matches", "closing_evidence"])
def test_feature_access_rejection_becomes_explicit_warning(setup, monkeypatch, method):
    from src.features import FeatureAccessError
    from src.review import lifecycle
    def reject(*args, **kwargs):
        raise FeatureAccessError("fixture feature access rejected")
    monkeypatch.setattr(lifecycle.FeatureStore, method, reject)
    r = report(setup, "2026-11-03 16:00:00")
    assert any("fixture feature access rejected" in warning for warning in r["warnings"])
    assert r["summary"]["complete_on_time_count"] == 0
    if method == "closing_evidence":
        assert r["matches"][0]["closing"]["state"] == "error"


def test_evaluation_cutoff_preserves_subseconds_and_generation_uses_wall_clock(setup, monkeypatch):
    from datetime import datetime, timezone
    from src.review import lifecycle
    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 11, 4, 1, 2, 3, 456789, tzinfo=timezone.utc)
    monkeypatch.setattr(lifecycle, "datetime", FixedClock)
    archive(setup, cutoff="2026-11-02 12:00:00.200000")
    r = report(setup, "2026-11-02T20:00:00.500000+08:00")
    assert r["evaluation_cutoff_utc"] == "2026-11-02 12:00:00.500000"
    assert r["generated_at_utc"] == "2026-11-04 01:02:03.456789"
    assert r["matches"][0]["slots"][0]["state"] == "late"


def test_preview_and_unconfigured_kinds_remain_visible_without_filling_slots(setup):
    archive(setup)
    archive(setup, pid=2, kind="preview")
    archive(setup, pid=3, kind="legacy-preview", created="2026-11-03 12:01:00")
    m = report(setup)["matches"][0]
    retained = [a for s in m["slots"] for a in s["archives"]] + m["other_archives"]
    assert {a["prediction_id"] for a in retained} == {1, 2, 3}
    assert {a["prediction_id"] for a in m["other_archives"]} == {2, 3}
    assert m["other_archives"][1]["issues"]
    assert m["preview_count"] == 1 and not m["archives_complete"]


@pytest.mark.parametrize("books", [[], ["多家平均"], ["多家平均", "多家平均"], ["多家平均", None], ["多家平均", "other"]])
def test_four_closing_groups_cannot_be_reduced_by_invalid_configuration(setup, books):
    setup[1]["review"]["close_bookmakers"] = books
    r = report(setup)
    assert any("收盘" in warning for warning in r["warnings"])
    assert r["summary"]["closing_complete_count"] == 0


@pytest.mark.parametrize("horizons", [[24], [24, 6], [24, 6, 1, .5], [24, 6, 1, 1]])
def test_formal_three_slot_scope_cannot_silently_change(setup, horizons):
    setup[1]["schedule"]["horizons"] = horizons
    r = report(setup)
    assert any("时点" in warning for warning in r["warnings"])
    assert r["summary"]["archived_complete_count"] == 0


def save_partial_fusion_review(setup, status):
    doc = finish_and_save_review(setup)
    con, cfg, folder = setup
    scores = pd.read_csv(folder / "scores.csv")
    fusion_scores = []
    for row in pd.read_sql("select * from predictions", con).to_dict("records"):
        detail = json.loads(row["detail"])
        detail.update({"融合": {"状态": status}, "融合概率": [.5, .3, .2], "融合比分矩阵": None})
        con.execute("update predictions set detail=? where prediction_id=?", (json.dumps(detail), row["prediction_id"]))
        metrics = score_metrics(np.array([.5, .3, .2]), None, 2, 1)
        original = scores[scores["prediction_id"] == row["prediction_id"]].iloc[0].to_dict()
        fusion_scores.append({**original, "source": "融合", **metrics})
        entry = next(entry for entry in doc["各版本"] if entry["prediction_id"] == row["prediction_id"])
        entry["融合"] = {"概率": [.5, .3, .2], **{key: round(value, 4) if value is not None else None
                                                for key, value in metrics.items()}}
    pd.concat([scores, pd.DataFrame(fusion_scores)], ignore_index=True).to_csv(folder / "scores.csv", index=False, float_format="%.6f")
    (folder / "matches" / "1.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    con.commit()


@pytest.mark.parametrize("status", ["赔率不全", "无赔率，未融合"])
def test_deliberate_partial_fusion_keeps_model_six_metric_acceptance(setup, status):
    save_partial_fusion_review(setup, status)
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert m["score_complete"]


def test_claimed_full_fusion_requires_fusion_matrix(setup):
    save_partial_fusion_review(setup, "已融合")
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert not m["score_complete"] and any("融合比分矩阵" in issue for issue in m["review"]["issues"])


def test_invalid_declared_fusion_probability_cannot_silently_disappear(setup):
    finish_and_save_review(setup)
    con = setup[0]
    for pid, raw in con.execute("select prediction_id,detail from predictions").fetchall():
        detail = json.loads(raw)
        detail.update({"融合": {"状态": "赔率不全"}, "融合概率": [float("nan"), .3, .2]})
        con.execute("update predictions set detail=? where prediction_id=?", (json.dumps(detail), pid))
    con.commit()
    m = report(setup, "2026-11-03 16:00:00")["matches"][0]
    assert not m["score_complete"] and any("融合" in issue for issue in m["review"]["issues"])


@pytest.mark.parametrize("table", ["match_stats", "match_sources", "odds"])
def test_missing_evidence_tables_are_explicit_warnings_not_unhandled_errors(setup, table):
    setup[0].execute(f"drop table {table}")
    setup[0].commit()
    r = report(setup, "2026-11-03 16:00:00")
    assert r["warnings"] and r["summary"]["complete_on_time_count"] == 0


def test_current_source_fetched_after_cutoff_cannot_make_source_present(setup):
    setup[0].execute("insert into match_sources values(1,'football-data','E0:1','data/raw/fd.csv','2026-11-02 12:30:00.001')")
    setup[0].commit()
    m = report(setup)["matches"][0]
    assert m["source_state"]["state"] == "unknown" and m["source_missing"] is None


def test_known_missing_source_is_separate_from_unknown_even_with_stale_current_mapping(setup):
    setup[0].execute("insert into match_sources values(1,'football-data','E0:1','data/raw/fd.csv','2026-11-02 12:00:00')")
    setup[0].execute("insert into match_source_state(match_id,source_missing,missing_since_utc,last_sources_json) "
                     "values(1,1,'2026-11-02 12:20:00','[]')")
    setup[0].commit()
    r = report(setup)
    assert r["matches"][0]["source_state"]["state"] == "missing"
    assert r["summary"]["source_missing_count"] == 1


def test_readonly_nonempty_database_with_scores_does_not_write_main_or_artifacts(setup):
    finish_and_save_review(setup)
    add_all_closes(setup)
    before = {path: path.read_bytes() for path in setup[2].rglob("*") if path.is_file()}
    denied = []
    def authorize(action, _a, _b, db, _trigger):
        if db == "main" and action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                                        sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE}:
            denied.append(action)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    setup[0].set_authorizer(authorize)
    r = report(setup, "2026-11-03 16:00:00")
    assert r["summary"]["scored_complete_count"] == 1 and not r["warnings"]
    assert not denied and before == {path: path.read_bytes() for path in setup[2].rglob("*") if path.is_file()}


@pytest.mark.parametrize("now", [None, "bad date", 123, float("nan")])
def test_invalid_evaluation_cutoff_is_explicit_warning(setup, now):
    r = report(setup, now)
    assert r["evaluation_cutoff_utc"] is None and any("评价时刻" in warning for warning in r["warnings"])


def test_report_scope_uses_exact_half_open_window(setup):
    con = setup[0]
    con.executemany("insert into matches(match_id,competition,season,home_team,away_team,match_date,kickoff_utc,status,round) "
                    "values(?,'E0',2026,'A','B',?,?,'scheduled',?)", [
        (2, "2026-10-30", "2026-10-30 12:30:00", "2"),
        (3, "2026-11-09", "2026-11-09 12:30:00", "3"),
    ])
    con.commit()
    r = report(setup, "2026-11-02 12:30:00.500000")
    assert {m["match_id"] for m in r["matches"]} == {1, 3}


@pytest.mark.parametrize("key,value", [("days_back", -1), ("days_ahead", -1), ("days_ahead", float("inf"))])
def test_invalid_site_window_is_a_warning_not_a_clean_empty_scope(setup, key, value):
    setup[1]["site"][key] = value
    r = report(setup)
    assert r["warnings"] and not r["matches"]


def test_real_readonly_sqlite_file_allows_feature_temp_tables_without_main_writes(setup):
    from src.features import open_readonly
    from src.review.lifecycle import build_report
    archive(setup)
    path = setup[2] / "readonly.db"
    with sqlite3.connect(path) as copy:
        setup[0].backup(copy)
    before = path.read_bytes()
    readonly = open_readonly(path)
    try:
        r = build_report(readonly, setup[1], "2026-11-02 12:30:00")
        assert r["summary"]["recorded_slot_count"] == 1 and not r["warnings"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            readonly.execute("update matches set status='cancelled'")
    finally:
        readonly.close()
    assert path.read_bytes() == before


@pytest.mark.parametrize("flag,missing,restored", [(1, "2026-11-02 12:31:00", None),
                                                   (0, "2026-11-01 12:00:00", "2026-11-02 12:31:00")])
def test_future_source_lifecycle_transition_is_not_current_evidence(setup, flag, missing, restored):
    setup[0].execute("insert into match_sources values(1,'football-data','E0:1','data/raw/fd.csv','2026-11-02 12:00:00')")
    setup[0].execute("insert into match_source_state(match_id,source_missing,missing_since_utc,restored_at_utc,last_sources_json) "
                     "values(1,?,?,?,'[]')", (flag, missing, restored))
    setup[0].commit()
    m = report(setup)["matches"][0]
    assert m["source_state"]["state"] == "unknown" and m["source_missing"] is None


def test_concurrent_main_database_change_withdraws_mixed_snapshot_acceptance(setup, monkeypatch):
    from src.features import open_readonly
    from src.review import lifecycle
    archive(setup)
    path = setup[2] / "concurrent.db"
    with sqlite3.connect(path) as copy:
        setup[0].backup(copy)
    original = lifecycle.FeatureStore.prediction_targets
    def changing_targets(*args, **kwargs):
        targets = original(*args, **kwargs)
        with sqlite3.connect(path) as writer:
            writer.execute("update matches set status='postponed' where match_id=1")
        return targets
    monkeypatch.setattr(lifecycle.FeatureStore, "prediction_targets", changing_targets)
    readonly = open_readonly(path)
    try:
        r = lifecycle.build_report(readonly, setup[1], "2026-11-02 12:30:00")
    finally:
        readonly.close()
    assert any("验收期间" in warning for warning in r["warnings"])
    assert not r["matches"] and all(count == 0 for count in r["summary"].values())
