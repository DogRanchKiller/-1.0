"""解释是当次预测附件：只存新记录，不改模型概率、去重和旧记录。"""
import copy
import json

import pytest

from src.features import FeatureStore
from src.predict.preview import archive, build_preview, prediction_detail
from src.predict.scheduled import run_scheduled
from test_preview import FakeModel, NOW, env


class ExplainModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.fit_calls = self.predict_calls = self.explain_calls = 0
        self.options = None

    def fit(self, history, cutoff):
        self.fit_calls += 1
        super().fit(history, cutoff)

    def predict(self, targets, cutoff):
        self.predict_calls += 1
        return super().predict(targets, cutoff)

    def predict_with_explanations(self, targets, cutoff, *, options=None):
        self.explain_calls += 1
        self.options = options
        return (FakeModel.predict(self, targets, cutoff),
                [{"schema": "gbdt-tree-shap-v1", "status": "unavailable",
                  "reason": "validation_failed"} for _ in range(len(targets))])


def test_explained_prediction_uses_one_fit_and_shared_call_without_changing_numbers(env):
    con, settings = env
    baseline = build_preview(FeatureStore(con, "实时", settings), settings, NOW, FakeModel())
    model = ExplainModel()
    result = build_preview(FeatureStore(con, "实时", settings), settings, NOW, model)
    assert model.explain_calls == 1 and model.fit_calls == 1 and model.predict_calls == 0
    assert model.options["integration_points"] == settings["explanation"]["integration_points"]
    for row, old in zip(result["rows"], baseline["rows"]):
        assert {key: value for key, value in row.items() if key != "explanation"} == {
            key: value for key, value in old.items() if key != "explanation"}
        assert row["explanation"]["identity"] == {
            **{key: row[key] for key in ("match_id", "competition", "season", "home_team", "away_team")},
            "model": result["model"], "model_version": result["model_version"], "cutoff_utc": result["cutoff_utc"]}


def test_disabled_and_unsupported_keep_prediction_and_explicit_reason(env):
    con, settings = env
    result = build_preview(FeatureStore(con, "实时", settings), settings, NOW, FakeModel())
    assert all(r["explanation"]["reason"] == "unsupported_model" for r in result["rows"])
    settings["explanation"]["enabled"] = False
    model = ExplainModel()
    disabled = build_preview(FeatureStore(con, "实时", settings), settings, NOW, model)
    assert model.explain_calls == 0 and model.predict_calls == 1
    assert all(r["explanation"]["reason"] == "disabled" for r in disabled["rows"])
    assert [r["model_p"] for r in result["rows"]] == [r["model_p"] for r in disabled["rows"]]


def test_explanation_is_shared_by_preview_and_scheduled_archives(env):
    con, settings = env
    result = build_preview(FeatureStore(con, "实时", settings), settings, NOW, ExplainModel())
    detail = prediction_detail(result, result["rows"][0], "test")
    assert detail["模型解释"] == result["rows"][0]["explanation"]
    archive(con, result, NOW, "test")
    saved = json.loads(con.execute("select detail from predictions order by prediction_id limit 1").fetchone()[0])
    assert saved["模型解释"] == detail["模型解释"]
    later = NOW.replace(hour=16)  # 23小时后开球，真实当前时点，不伪造准点T-24。
    summary = run_scheduled(FeatureStore(con, "实时", settings), con, settings, later,
                            model=ExplainModel(), code="test", created=later)
    assert summary["新存"] == 1
    saved = json.loads(con.execute("select detail from predictions where kind='T-24'").fetchone()[0])
    assert saved["模型解释"]["identity"]["cutoff_utc"] == "2026-11-06 16:00:00"
    assert saved["模型解释"]["reason"] == "validation_failed"
    json.dumps(saved["模型解释"], allow_nan=False)


def test_new_explanation_never_backfills_or_changes_deduplication(env):
    con, settings = env
    result = build_preview(FeatureStore(con, "实时", settings), settings, NOW, FakeModel())
    for row in result["rows"]:
        row.pop("explanation", None)
    first = archive(con, result, NOW, "old")
    before = con.execute("select * from predictions order by prediction_id").fetchall()
    explained = build_preview(FeatureStore(con, "实时", settings), settings, NOW, ExplainModel())
    counts = archive(con, explained, NOW, "new")
    assert counts["新存"] == 0 and counts["没变化"] == first["新存"]
    assert con.execute("select * from predictions order by prediction_id").fetchall() == before
    # 正常概率发生变化会产生新行，附件只随这条新记录保存。
    changed = copy.deepcopy(explained)
    changed["rows"][0]["model_p"] = [0.51, 0.29, 0.20]
    assert archive(con, changed, NOW, "new")["新存"] == 1
    last = json.loads(con.execute("select detail from predictions order by prediction_id desc limit 1").fetchone()[0])
    assert last["模型解释"]["identity"]["match_id"] == changed["rows"][0]["match_id"]


@pytest.mark.parametrize("bad", [None, [], {"schema": "gbdt-tree-shap-v1", "status": "available", "x": float("nan")}])
def test_invalid_attachment_cannot_break_archive_or_emit_nonfinite_json(env, bad):
    con, settings = env
    result = build_preview(FeatureStore(con, "实时", settings), settings, NOW, FakeModel())
    result["rows"][0]["explanation"] = bad
    detail = prediction_detail(result, result["rows"][0], "test")
    assert detail["模型解释"]["status"] == "unavailable"
    json.dumps(detail["模型解释"], allow_nan=False)
