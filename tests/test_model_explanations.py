"""逐场解释只用模拟比赛/小型真实树；原预测、历史数据和存档都不改。"""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.models.goals import score_matrices
from tests.test_adaptive import promo_history
from tests.test_gbdt import gb
from tests.test_goals_models import targets


@pytest.fixture
def trained():
    model = gb()
    cutoff = "2021-08-01 00:00:00"
    history = promo_history()
    model.fit(history[history["kickoff_earliest_utc"] < cutoff], cutoff)
    assert model.booster is not None
    return model, cutoff


def prediction_targets():
    return targets([(11, "E0", 2021, "T3", "T4"), (12, "E0", 2016, "T1", "T2"),
                    (13, "E0", 2021, "T5", "NEW")]).assign(kickoff_earliest_utc="2021-08-20 14:00:00")


def assert_closed(explanation, matrix):
    assert explanation["status"] == "available", explanation
    assert explanation["schema"] == "gbdt-tree-shap-v1"
    assert explanation["method"] == "tree_shap_raw_log_lambda"
    assert explanation["conversion"] == "straight_log_path_v1"
    assert explanation["reason"] is None
    assert set(explanation["checks"]) == {"raw_max_abs_error", "goal_max_abs_error", "home_win_abs_error",
                                          "integration_max_abs_error", "matrix_max_abs_error"}
    assert max(explanation["checks"].values()) < 1e-6
    for side in ("home", "away"):
        terms = [t for t in explanation["terms"] if t["side"] == side]
        assert [t["feature"] for t in terms] == explanation["feature_names"]
        assert explanation["raw_base"][side] + sum(t["raw_contribution"] for t in terms) == pytest.approx(
            explanation["raw_prediction"][side], abs=1e-6)
        key = side + "_goals"
        assert explanation["base"][key] + sum(t["goal_contribution"] for t in terms) == pytest.approx(
            explanation["prediction"][key], abs=1e-6)
    assert explanation["base"]["home_win"] + sum(t["home_win_contribution"] for t in explanation["terms"]) == pytest.approx(
        explanation["prediction"]["home_win"], abs=1e-6)
    assert explanation["prediction"]["home_win"] == pytest.approx(float(np.tril(matrix, -1).sum()), abs=1e-12)
    json.dumps(explanation, allow_nan=False)


def test_real_tree_uses_same_feature_matrix_without_refitting_or_changing_prediction(trained, monkeypatch):
    model, cutoff = trained
    target = prediction_targets()
    plain = copy.deepcopy(model).predict(target, cutoff)
    before = model.booster.model_to_string()
    calls, feature_calls = [], []
    predict, features = model.booster.predict, model._features

    def spy_predict(X, *args, **kwargs):
        calls.append((X, kwargs.copy()))
        return predict(X, *args, **kwargs)

    def spy_features(*args, **kwargs):
        feature_calls.append(args)
        return features(*args, **kwargs)

    monkeypatch.setattr(model.booster, "predict", spy_predict)
    monkeypatch.setattr(model, "_features", spy_features)
    monkeypatch.setattr(model, "fit", lambda *a, **k: pytest.fail("解释不能拟合模型"))
    predictions, explanations = model.predict_with_explanations(target, cutoff)
    assert len(predictions) == len(explanations) == len(target)
    assert len(feature_calls) == 2
    assert predictions[1] is None and explanations[1]["reason"] == "no_prediction"
    assert len(calls) == 3
    assert all(X is calls[0][0] for X, _ in calls)
    assert sum(bool(kw.get("pred_contrib")) for _, kw in calls) == 1
    assert sum(bool(kw.get("raw_score")) for _, kw in calls) == 1
    assert model.booster.model_to_string() == before
    for i in (0, 2):
        assert np.array_equal(predictions[i], plain[i])
        assert_closed(explanations[i], predictions[i])
        assert explanations[i]["model_fingerprint"] == hashlib.sha256(before.encode("utf-8")).hexdigest()
        assert explanations[i]["model_trained_at_utc"] == cutoff
        # 缺失的休息天数是null；主/客顺序不能错位。
        assert all(t["value"] is None for t in explanations[i]["terms"] if t["feature"].startswith("rest_days_"))


def test_plain_predict_has_no_explanation_cost(trained, monkeypatch):
    model, cutoff = trained
    from src.models import explain
    monkeypatch.setattr(explain, "explain_batch", lambda *a, **k: pytest.fail("普通predict不算解释"))
    predict = model.booster.predict
    calls = []

    def spy(X, *args, **kwargs):
        calls.append(kwargs)
        return predict(X, *args, **kwargs)

    monkeypatch.setattr(model.booster, "predict", spy)
    assert model.predict(prediction_targets(), cutoff)[0] is not None
    assert calls == [{}]


@pytest.mark.parametrize("failure", ["contrib_error", "bad_shape", "bad_raw", "feature_order", "bad_one_row"])
def test_explanation_failure_preserves_original_predictions(trained, monkeypatch, failure):
    model, cutoff = trained
    target = prediction_targets()
    expected = copy.deepcopy(model).predict(target, cutoff)
    predict = model.booster.predict

    def broken(X, *args, **kwargs):
        result = predict(X, *args, **kwargs)
        if kwargs.get("pred_contrib"):
            if failure == "contrib_error":
                raise RuntimeError("解释计算失败")
            if failure == "bad_shape":
                return result[:, :-1]
            if failure == "bad_one_row":
                result[0, 0] = np.nan
        if kwargs.get("raw_score") and failure == "bad_raw":
            result = result + 0.1
        return result

    monkeypatch.setattr(model.booster, "predict", broken)
    if failure == "feature_order":
        monkeypatch.setattr(model.booster, "feature_name", lambda: list(reversed([model.names[i] for i in model.use])))
    got, explanations = model.predict_with_explanations(target, cutoff)
    for i in (0, 2):
        assert np.array_equal(got[i], expected[i])
    assert explanations[0]["status"] == "unavailable"
    assert explanations[0]["reason"] in ("explanation_failed", "validation_failed")
    assert explanations[1]["reason"] == "no_prediction"
    if failure == "bad_one_row":
        assert explanations[2]["status"] == "available"
    json.dumps(explanations, allow_nan=False)


def test_disabled_untrained_empty_and_unsupported_targets(trained, monkeypatch):
    model, cutoff = trained
    plain = copy.deepcopy(model).predict(prediction_targets(), cutoff)
    predict = model.booster.predict
    calls = []

    def spy(X, *args, **kwargs):
        calls.append(kwargs)
        return predict(X, *args, **kwargs)

    monkeypatch.setattr(model.booster, "predict", spy)
    got, explanation = model.predict_with_explanations(prediction_targets(), cutoff, options={"enabled": False})
    assert np.array_equal(got[0], plain[0]) and calls == [{}]
    assert explanation[0]["reason"] == "disabled"
    assert explanation[1]["reason"] == "no_prediction"
    assert gb().predict_with_explanations(prediction_targets(), cutoff)[1][0]["reason"] == "no_prediction"
    assert model.predict_with_explanations(prediction_targets().iloc[:0], cutoff) == ([], [])
    multi = gb(target="result")
    history = promo_history()
    multi.fit(history[history["kickoff_earliest_utc"] < cutoff], cutoff)
    expected = copy.deepcopy(multi).predict(prediction_targets(), cutoff)
    got, explanation = multi.predict_with_explanations(prediction_targets(), cutoff)
    assert np.array_equal(got[0], expected[0])
    assert explanation[0]["reason"] == "unsupported_target"
    assert explanation[1]["reason"] == "no_prediction"


def hand_explanation(phi, base=(0.25, -0.15), *, rho=-0.12, max_goals=10, options=None, mutate=None):
    from src.models.explain import poisson_explanation
    phi, base = np.array(phi, float), np.array(base, float)
    raw = base + phi.sum(axis=1)
    lam = np.exp(raw)
    matrix = score_matrices(lam[0], lam[1], rho, 0.0, max_goals)[0]
    kwargs = dict(feature_names=[f"f{i}" for i in range(phi.shape[1])], values=np.zeros_like(phi),
                  contributions=np.column_stack([phi, base]), raw=raw, lambdas=lam, matrix=matrix,
                  rho=rho, max_goals=max_goals, model_fingerprint="a" * 64, library_version="test", options=options)
    if mutate:
        mutate(kwargs)
    return poisson_explanation(**kwargs), matrix


@pytest.mark.parametrize("phi", [[[0.2, -0.2], [-0.1, 0.1]], [[1e-14, -2e-14], [0., 0.]],
                                 [[0., 0.], [0., 0.]], [[0.8, -0.2], [-0.4, 0.1]]])
def test_path_conversion_closes_with_cancellation_zero_and_regular_changes(phi):
    explanation, matrix = hand_explanation(phi)
    assert_closed(explanation, matrix)


def test_probability_contributions_include_away_goal_effect_and_match_tail_folding():
    explanation, matrix = hand_explanation([[0., 0.], [0.4, 0.2]], max_goals=2)
    assert_closed(explanation, matrix)
    home = [t for t in explanation["terms"] if t["side"] == "home"]
    away = [t for t in explanation["terms"] if t["side"] == "away"]
    assert all(t["home_win_contribution"] == 0 for t in home)
    assert all(t["home_win_contribution"] < 0 and t["goal_contribution"] > 0 for t in away)


@pytest.mark.parametrize("change", ["matrix", "lambda", "raw", "nan_contribution", "infinite_input"])
def test_inconsistent_or_nonfinite_evidence_is_unavailable(change):
    def mutate(kwargs):
        if change == "matrix":
            kwargs["matrix"] = kwargs["matrix"].T.copy()
        elif change == "lambda":
            kwargs["lambdas"] = kwargs["lambdas"] + 0.1
        elif change == "raw":
            kwargs["raw"] = kwargs["raw"] + 0.1
        elif change == "nan_contribution":
            kwargs["contributions"][0, 0] = np.nan
        else:
            kwargs["values"][0, 0] = np.inf
    explanation, _ = hand_explanation([[0.2, -0.1], [-0.1, 0.3]], mutate=mutate)
    assert explanation == {"schema": "gbdt-tree-shap-v1", "status": "unavailable", "reason": "validation_failed"}


@pytest.mark.parametrize("options", [{"integration_points": 1}, {"integration_points": True},
                                     {"derivative_step": 0}, {"atol": float("nan")}])
def test_invalid_numerical_options_do_not_produce_claimed_explanation(options):
    explanation, _ = hand_explanation([[0.2], [-0.2]], options=options)
    assert explanation["status"] == "unavailable"


def test_integration_checks_both_resolutions_and_derivative_steps(monkeypatch):
    from src.models import explain
    original, calls = explain._path_integral, []

    def spy(base, delta, rho, max_goals, points, step):
        calls.append((points, step))
        return original(base, delta, rho, max_goals, points, step)

    monkeypatch.setattr(explain, "_path_integral", spy)
    explanation, matrix = hand_explanation([[0.4, -0.1], [-0.2, 0.1]])
    assert_closed(explanation, matrix)
    assert {(64, 1e-5), (128, 1e-5), (128, 5e-6)} <= set(calls)


def test_unconverged_integral_is_not_rescaled_to_hide_residual(monkeypatch):
    from src.models import explain
    original = explain._path_integral

    def unconverged(base, delta, rho, max_goals, points, step):
        out = original(base, delta, rho, max_goals, points, step)
        return out + (0.01 if points == 64 else 0)

    monkeypatch.setattr(explain, "_path_integral", unconverged)
    explanation, _ = hand_explanation([[0.3], [-0.2]])
    assert explanation["status"] == "unavailable" and explanation["reason"] == "validation_failed"


@pytest.mark.parametrize("changed", ["points", "step"])
def test_reported_integration_error_is_maximum_individual_contribution_error(monkeypatch, changed):
    from src.models import explain
    original = explain._path_integral
    calls = {}

    def perturbed(base, delta, rho, max_goals, points, step):
        out = original(base, delta, rho, max_goals, points, step)
        if (changed == "points" and points == 64) or (changed == "step" and points == 128 and step == 1e-5):
            out = out + np.array([1e-8, -1e-8])
        calls[(points, step)] = out
        return out

    monkeypatch.setattr(explain, "_path_integral", perturbed)
    phi = np.array([[0.4, -0.3], [-0.2, 0.1]])
    explanation, matrix = hand_explanation(phi)
    assert_closed(explanation, matrix)
    quadrature = phi * (calls[(128, 1e-5)] - calls[(64, 1e-5)])[:, None]
    derivative = phi * (calls[(128, 5e-6)] - calls[(128, 1e-5)])[:, None]
    assert explanation["checks"]["integration_max_abs_error"] == pytest.approx(
        max(np.abs(quadrature).max(), np.abs(derivative).max()), rel=1e-12, abs=1e-16)


@pytest.mark.parametrize("changed", ["points", "step"])
def test_opposite_individual_errors_cannot_cancel_convergence_check(monkeypatch, changed):
    from src.models import explain
    original = explain._path_integral

    def perturbed(base, delta, rho, max_goals, points, step):
        out = original(base, delta, rho, max_goals, points, step)
        if (changed == "points" and points == 64) or (changed == "step" and points == 128 and step == 1e-5):
            out = out + 0.01
        return out

    monkeypatch.setattr(explain, "_path_integral", perturbed)
    explanation, _ = hand_explanation([[0.3, -0.3], [-0.2, 0.2]])
    assert explanation["status"] == "unavailable" and explanation["reason"] == "validation_failed"


def test_dc_clipped_scores_are_reconstructed_and_nonsmooth_path_fails_closed():
    explanation, matrix = hand_explanation([[0.3], [-0.2]], base=(1.7, 1.5), rho=-0.3)
    assert_closed(explanation, matrix)
    assert matrix[0, 1] == matrix[1, 0] == 0
    # 这条路径穿越DC截零折点，默认64/128点不能达到容差，就只能缺解释。
    unavailable, _ = hand_explanation([[1.5], [-0.2]], base=(0., 0.), rho=-0.3)
    assert unavailable["status"] == "unavailable" and unavailable["reason"] == "validation_failed"
