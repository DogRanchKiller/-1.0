"""同一棵泊松树、同一特征行的解释；只读计算，不拟合、不取数、不修改预测。

LightGBM原生TreeSHAP在log(lambda)上相加。这里保留原数，并沿主客两侧
共同的直线路径换算为球数、主胜概率贡献；换算后的数不是这两个空间的
原生SHAP，也不是因果效应。基准不是联赛平均，rho始终固定为本场使用值。
任何形状、数值或重构检查失败，都只让解释不可用，不给残差另找一个因素。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

from src.models.goals import score_matrices

SCHEMA = "gbdt-tree-shap-v1"


class _InvalidExplanation(ValueError):
    """解释证据或数值精度没有通过检查；不影响原预测。"""


def unavailable(reason: str) -> dict:
    return {"schema": SCHEMA, "status": "unavailable", "reason": reason}


def _require(ok) -> None:
    if not ok:
        raise _InvalidExplanation("解释未通过一致性检查")


def _options(options) -> tuple[int, float, float]:
    cfg = {} if options is None else options
    _require(isinstance(cfg, Mapping))
    points = cfg.get("integration_points", 64)
    step, atol = cfg.get("derivative_step", 0.00001), cfg.get("atol", 0.000001)
    _require(isinstance(points, (int, np.integer)) and not isinstance(points, (bool, np.bool_)) and points >= 2)
    _require(not isinstance(step, (bool, np.bool_)) and not isinstance(atol, (bool, np.bool_)))
    step, atol = float(step), float(atol)
    _require(np.isfinite(step) and step > 0 and np.isfinite(atol) and atol > 0)
    return int(points), step, atol


def _max_abs(values) -> float:
    array = np.asarray(values, dtype=float)
    _require(np.isfinite(array).all())
    return float(np.max(np.abs(array), initial=0.0))


@lru_cache(maxsize=8)
def _quadrature(points: int) -> tuple[np.ndarray, np.ndarray]:
    """只缓存积分节点，不缓存比赛、树或特征；相同点数不用每场重新算节点。"""
    nodes, weights = leggauss(points)
    return (nodes + 1.0) / 2.0, weights / 2.0


def _home_win(log_rates: np.ndarray, rho: float, max_goals: int) -> np.ndarray:
    rates = np.exp(log_rates)
    # 必须复用正式预测函数，包括尾部折入、DC截零和归一化，不能换成独立泊松近似。
    matrices = score_matrices(rates[:, 0], rates[:, 1], rho, 0.0, max_goals)
    _require(np.isfinite(matrices).all())
    return np.tril(matrices, -1).sum(axis=(1, 2))


def _path_integral(base, delta, rho, max_goals, points, step) -> np.ndarray:
    """在双侧log强度的共同路径上积分偏导；返回主、客两个概率/log单位系数。"""
    nodes, weights = _quadrature(points)
    path = base[None, :] + nodes[:, None] * delta[None, :]
    result = []
    for side in range(2):
        plus, minus = path.copy(), path.copy()
        plus[:, side] += step
        minus[:, side] -= step
        derivative = (_home_win(plus, rho, max_goals) - _home_win(minus, rho, max_goals)) / (2.0 * step)
        result.append(float(weights @ derivative))
    return np.asarray(result)


def _trained_time(value) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.isoformat(sep=" ")


def poisson_explanation(*, feature_names, values, contributions, raw, lambdas, matrix, rho,
                        max_goals, model_fingerprint, library_version, model_trained_at_utc=None,
                        options=None) -> dict:
    """校验并换算一场的已提取证据；主、客两行次序固定，不调用模型或数据层。"""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _poisson_explanation(feature_names=feature_names, values=values, contributions=contributions,
                raw=raw, lambdas=lambdas, matrix=matrix, rho=rho, max_goals=max_goals,
                model_fingerprint=model_fingerprint, library_version=library_version,
                model_trained_at_utc=model_trained_at_utc, options=options)
    except (_InvalidExplanation, ValueError, TypeError, OverflowError, FloatingPointError):
        return unavailable("validation_failed")
    except Exception:
        # 解释是旁路；不可预期的解释故障也不能取消本来已算好的预测。
        return unavailable("explanation_failed")


def _poisson_explanation(*, feature_names, values, contributions, raw, lambdas, matrix, rho,
                         max_goals, model_fingerprint, library_version, model_trained_at_utc, options):
    points, step, atol = _options(options)
    names = list(feature_names)
    _require(bool(names) and all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names))
    values, contributions = np.asarray(values, dtype=float), np.asarray(contributions, dtype=float)
    raw, lambdas, matrix = (np.asarray(x, dtype=float) for x in (raw, lambdas, matrix))
    _require(values.shape == (2, len(names)) and contributions.shape == (2, len(names) + 1))
    _require(raw.shape == (2,) and lambdas.shape == (2,))
    # NaN输入代表缺失，原树可以处理；无限输入和非有限输出没有可信解释。
    _require(not np.isinf(values).any() and np.isfinite(contributions).all())
    _require(np.isfinite(raw).all() and np.isfinite(lambdas).all() and (lambdas > 0).all())
    _require(isinstance(max_goals, (int, np.integer)) and not isinstance(max_goals, (bool, np.bool_)) and max_goals >= 1)
    max_goals, rho = int(max_goals), float(rho)
    _require(np.isfinite(rho) and matrix.shape == (max_goals + 1, max_goals + 1))
    _require(np.isfinite(matrix).all() and (matrix >= 0).all() and abs(float(matrix.sum()) - 1.0) <= atol)
    _require(isinstance(model_fingerprint, str) and len(model_fingerprint) == 64)
    _require(isinstance(library_version, str) and bool(library_version))

    phi, base = contributions[:, :-1], contributions[:, -1]
    delta, base_goals = raw - base, np.exp(base)
    raw_error = max(_max_abs(base + phi.sum(axis=1) - raw), _max_abs(np.log(lambdas) - raw))
    _require(raw_error <= atol)
    # expm1在delta很小时保留精度；恰好为0时使用极限1。
    factor = np.divide(np.expm1(delta), delta, out=np.ones_like(delta), where=delta != 0)
    goal_terms = phi * (base_goals * factor)[:, None]
    goal_error = max(_max_abs(np.exp(raw) - lambdas),
                     _max_abs(base_goals + goal_terms.sum(axis=1) - lambdas))
    reconstructed = score_matrices(lambdas[0], lambdas[1], rho, 0.0, max_goals)[0]
    matrix_error = _max_abs(reconstructed - matrix)
    _require(goal_error <= atol and matrix_error <= atol)

    # 默认64/128点检验积分收敛，并把导数步长减半检验差分收敛。
    coarse = _path_integral(base, delta, rho, max_goals, points, step)
    fine = _path_integral(base, delta, rho, max_goals, points * 2, step)
    final = _path_integral(base, delta, rho, max_goals, points * 2, step / 2.0)
    probability_terms = phi * final[:, None]
    quadrature_error = _max_abs(phi * (fine - coarse)[:, None])
    derivative_error = _max_abs(phi * (final - fine)[:, None])
    # 必须逐项检查，两侧/正负贡献相消不能掩盖不收敛。公开误差与贡献使用同一概率单位。
    integration_error = max(quadrature_error, derivative_error)
    _require(quadrature_error <= atol and derivative_error <= atol)
    # 对未缩放的偏导积分也作保守检查，避免很小的phi隐藏数值失稳。
    _require(_max_abs(fine - coarse) <= atol and _max_abs(final - fine) <= atol)
    baseline_probability = float(_home_win(base[None, :], rho, max_goals)[0])
    predicted_probability = float(np.tril(matrix, -1).sum())
    probability_error = abs(baseline_probability + float(probability_terms.sum()) - predicted_probability)
    _require(integration_error <= atol and probability_error <= atol)
    terms = [{"side": side, "feature": name, "value": None if np.isnan(values[s, f]) else float(values[s, f]),
              "raw_contribution": float(phi[s, f]), "goal_contribution": float(goal_terms[s, f]),
              "home_win_contribution": float(probability_terms[s, f])}
             for s, side in enumerate(("home", "away")) for f, name in enumerate(names)]
    result = {"schema": SCHEMA, "status": "available", "reason": None,
              "method": "tree_shap_raw_log_lambda", "conversion": "straight_log_path_v1",
              "model_fingerprint": model_fingerprint, "library_version": library_version,
              "model_trained_at_utc": _trained_time(model_trained_at_utc),
              "context": {"rho": rho, "max_goals": max_goals},
              "base": {"home_goals": float(base_goals[0]), "away_goals": float(base_goals[1]),
                       "home_win": baseline_probability},
              "prediction": {"home_goals": float(lambdas[0]), "away_goals": float(lambdas[1]),
                             "home_win": predicted_probability},
              "raw_base": dict(zip(("home", "away"), map(float, base))),
              "raw_prediction": dict(zip(("home", "away"), map(float, raw))),
              "feature_names": names, "terms": terms,
              "checks": {"raw_max_abs_error": raw_error, "goal_max_abs_error": goal_error,
                         "home_win_abs_error": probability_error, "integration_max_abs_error": integration_error,
                         "matrix_max_abs_error": matrix_error}}
    json.dumps(result, allow_nan=False)       # 真正序列化前再检查一次，不能把NaN写入存档。
    return result


def explain_batch(booster, X, lambdas, matrices, contexts, feature_names, trained_at, *, options=None) -> list[dict]:
    """对本次预测原样使用的X读取贡献；一次取全批，转换错误只隔离出错那场。"""
    count = len(matrices)
    if isinstance(options, Mapping) and options.get("enabled") is False:
        return [unavailable("disabled") for _ in range(count)]
    try:
        _options(options)
        _require(list(booster.feature_name()) == list(feature_names))
        _require(X.shape == (2 * count, len(feature_names)))
        contributions = np.asarray(booster.predict(X, pred_contrib=True), dtype=float)
        raw = np.asarray(booster.predict(X, raw_score=True), dtype=float)
        _require(contributions.shape == (2 * count, len(feature_names) + 1) and raw.shape == (2 * count,))
        _require(np.shape(lambdas) == (count, 2) and len(contexts) == count)
        fingerprint = hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()
    except (_InvalidExplanation, ValueError, TypeError, OverflowError, FloatingPointError):
        return [unavailable("validation_failed") for _ in range(count)]
    except Exception:
        return [unavailable("explanation_failed") for _ in range(count)]
    return [poisson_explanation(feature_names=feature_names, values=X[2 * i:2 * i + 2],
                contributions=contributions[2 * i:2 * i + 2], raw=raw[2 * i:2 * i + 2], lambdas=lambdas[i],
                matrix=matrix, rho=contexts[i]["rho"], max_goals=contexts[i]["max_goals"],
                model_fingerprint=fingerprint, library_version=lgb.__version__, model_trained_at_utc=trained_at,
                options=options) for i, matrix in enumerate(matrices)]
