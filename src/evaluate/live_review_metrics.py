"""真实存档的描述统计；只消费复盘成绩，不读数据库、不训练或改变模型。

模型身份、预测时点、实际归档状态分开统计。每个标杆只保存一次，再与每个
模型组做同场内连接；没有匹配证据时保留空值。校准按主胜/平/客胜分别分箱，
每个结果的样本数都是比赛数，不能把三列展开后称为三倍的比赛。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluate.metrics import calibration_table


MODEL_SOURCES = ("模型", "融合")
CLOSE_SOURCE = "收盘赔率"
UNKNOWN = "未知"
GROUP_COLUMNS = ("model", "model_version", "kind", "source", "archive_status")
BENCHMARK_COLUMNS = ("benchmark_source", "bookmaker", "market", "timing")
METRICS = ("logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts")
MARKET_METRICS = {"1x2": METRICS[:3], "ou": ("logloss_over25",)}
PROBS = ("p_home", "p_draw", "p_away")
TAG_COLUMNS = ("德比", "争冠战", "保级战", "home_after_euro", "away_after_euro")


def _scalar(value):
    """把 pandas/numpy 标量换成严格 JSON 能表示的值。"""
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _prepare(scores: pd.DataFrame) -> pd.DataFrame:
    """旧文件的身份空值明确记未知；重复证据与互相矛盾的比分直接报错。"""
    df = scores.copy()
    for c in ("match_id", "home_goals", "away_goals", "line", *METRICS, *PROBS):
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not df.empty:
        mids = df["match_id"].to_numpy(dtype=float)
        if not np.isfinite(mids).all() or not (mids == np.floor(mids)).all():
            raise ValueError("复盘成绩的 match_id 必须是非空整数")
    for c in (*GROUP_COLUMNS, *BENCHMARK_COLUMNS, "confidence", "competition"):
        if c not in df:
            df[c] = UNKNOWN
        df[c] = df[c].map(lambda v: UNKNOWN if _scalar(v) in (None, "") else _scalar(v))
    model = df[df["source"].isin(MODEL_SOURCES)]
    # 同组同场只允许一条成绩；绝不通过 first/平均消掉重复身份。
    if model.duplicated(["match_id", "model", "model_version", "kind", "source"]).any():
        raise ValueError("同一模型版本、时点和来源存在重复比赛成绩")
    close = df[df["source"] == CLOSE_SOURCE]
    if close.duplicated(["match_id", *BENCHMARK_COLUMNS, "line"]).any():
        raise ValueError("同一来源、公司、玩法、时点存在重复收盘标杆")
    for _, part in df.groupby("match_id", sort=False):
        results = part[["home_goals", "away_goals"]].dropna().drop_duplicates()
        if len(results) > 1:
            raise ValueError("同场模型与标杆的比分不一致，不能比较")
    return df


def _columns(group_by: Sequence[str] | None) -> list[str]:
    """额外分组只能加细，不能通过少传一列把不同模型身份合并。"""
    return list(dict.fromkeys([*GROUP_COLUMNS, *(group_by or [])]))


def _groups(df: pd.DataFrame, columns: list[str]):
    for key, part in df.groupby(columns, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        yield dict(zip(columns, (_scalar(x) for x in key))), part


def _note(n: int, min_matches: int) -> str:
    if not n:
        return "无匹配证据"
    return "样本太少，不能下结论" if n < min_matches else "仅描述统计，不能据此直接改模型"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _summary(df: pd.DataFrame, group_by, min_matches: int) -> list[dict]:
    out = []
    for group, part in _groups(df[df["source"].isin(MODEL_SOURCES)], _columns(group_by)):
        n = int(part["match_id"].nunique())
        metrics = {}
        for name in METRICS:
            values = _numeric(part[name]).dropna()
            metrics[name] = {"n_matches": len(values), "missing_matches": n - len(values),
                             "mean": float(values.mean()) if len(values) else None}
        available = any(v["n_matches"] for v in metrics.values())
        out.append({"group": group, "n_matches": n, "n_rows": len(part),
                    "status": "available" if available else "unavailable",
                    "note": _note(n if available else 0, min_matches), "metrics": metrics})
    return out


def score_summary(scores: pd.DataFrame, *, group_by=None, min_matches: int) -> list[dict]:
    """每种模型身份的各指标均值及有效/缺失比赛数。group_by 只增加维度。"""
    return _summary(_prepare(scores), group_by, min_matches)


def _comparisons(df: pd.DataFrame, benchmark_books: Sequence[str], min_matches: int) -> list[dict]:
    models = df[df["source"].isin(MODEL_SOURCES)]
    closes = df[(df["source"] == CLOSE_SOURCE) & (df["timing"] == "close")]
    # 旧 CSV 没有来源证据时不冒充 football-data 的合格收盘。
    closes = closes[closes["benchmark_source"] != UNKNOWN]
    if "qualified" in closes:
        closes = closes[closes["qualified"].eq(True)]
    closes = closes[(closes["market"] == "1x2") |
                    ((closes["market"] == "ou") & closes["line"].eq(2.5))]
    identities = [group for group, _ in _groups(closes, list(BENCHMARK_COLUMNS))]
    # 即使一家公司整批没有报价，也要给每个模型组留下“无匹配证据”。
    for book in benchmark_books:
        for market in MARKET_METRICS:
            if not any(i["bookmaker"] == book and i["market"] == market for i in identities):
                identities.append(dict(zip(BENCHMARK_COLUMNS, ("football-data", book, market, "close"))))
    out = []
    for group, part in _groups(models, list(GROUP_COLUMNS)):
        for identity in identities:
            mask = pd.Series(True, index=closes.index)
            for col, value in identity.items():
                mask &= closes[col].eq(value)
            bench = closes[mask & closes["match_id"].isin(part["match_id"])]
            common = part.merge(bench, on="match_id", how="inner", suffixes=("_model", "_benchmark"),
                                validate="one_to_one")
            measures = {}
            for name in MARKET_METRICS[identity["market"]]:
                a, b = _numeric(common[name + "_model"]), _numeric(common[name + "_benchmark"])
                both = a.notna() & b.notna()
                a, b = a[both], b[both]
                delta = a - b
                measures[name] = {"n_matches": len(a), "model_mean": float(a.mean()) if len(a) else None,
                                  "benchmark_mean": float(b.mean()) if len(b) else None,
                                  "mean_delta": float(delta.mean()) if len(a) else None,
                                  "model_better": int((delta < 0).sum()),
                                  "benchmark_better": int((delta > 0).sum()), "ties": int((delta == 0).sum())}
            # 两个玩法的匹配分母互相独立，以该玩法主指标两边都有为准。
            n = measures[MARKET_METRICS[identity["market"]][0]]["n_matches"]
            out.append({"group": group, "benchmark": identity, "n_model": len(part),
                        "n_benchmark": len(bench), "n_matches": n, "missing_matches": len(part) - n,
                        "status": "available" if n else "unavailable", "note": _note(n, min_matches),
                        "metrics": measures})
    return out


def matched_comparisons(scores: pd.DataFrame, *, benchmark_books: Sequence[str], min_matches: int) -> list[dict]:
    """每个模型身份分别与每家公司/玩法做同场比较，差=模型−标杆。"""
    return _comparisons(_prepare(scores), benchmark_books, min_matches)


def _calibration(df: pd.DataFrame, group_by, n_bins: int) -> list[dict]:
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("校准箱数必须是正整数")
    out = []
    for group, part in _groups(df[df["source"].isin(MODEL_SOURCES)], _columns(group_by)):
        p = part[list(PROBS)].to_numpy(dtype=float)
        goals = part[["home_goals", "away_goals"]].to_numpy(dtype=float)
        valid = (np.isfinite(p).all(axis=1) & (p >= 0).all(axis=1) & (p <= 1).all(axis=1)
                 & np.isclose(p.sum(axis=1), 1.0) & np.isfinite(goals).all(axis=1)
                 & (goals >= 0).all(axis=1) & (goals == np.floor(goals)).all(axis=1))
        p, goals = p[valid], goals[valid]
        y = np.where(goals[:, 0] > goals[:, 1], 0, np.where(goals[:, 0] == goals[:, 1], 1, 2))
        outcomes = []
        for index, name in enumerate(("主胜", "平局", "客胜")):
            if len(p):
                table = calibration_table(p[:, index], (y == index).astype(int), n_bins)
                bins = [{key: _scalar(value) for key, value in row.items()} for row in table.to_dict("records")]
                weighted_gap = sum(b["n"] * abs(b["gap"]) for b in bins if b["n"])
                ece = weighted_gap / len(p)
            else:
                bins = [{"bin_low": i / n_bins, "bin_high": (i + 1) / n_bins, "n": 0,
                         "mean_pred": None, "observed": None, "gap": None} for i in range(n_bins)]
                ece = None
            outcomes.append({"outcome": name, "ece": ece, "bins": bins})
        out.append({"group": group, "n_matches": len(p), "missing_matches": len(part) - len(p),
                    "status": "available" if len(p) else "unavailable", "outcomes": outcomes})
    return out


def calibration_by_group(scores: pd.DataFrame, *, n_bins: int, group_by=None) -> list[dict]:
    """按三个结果各自分箱，保留空箱；缺概率/比分的场次另记缺失。"""
    return _calibration(_prepare(scores), group_by, n_bins)


def _tag_rows(df: pd.DataFrame, labels: Mapping | None) -> pd.DataFrame:
    rows = []
    for row in df[df["source"].isin(MODEL_SOURCES)].to_dict("records"):
        label = (labels or {}).get(int(row["match_id"]))
        if label is None:
            label = (labels or {}).get(str(int(row["match_id"])))
        label = label if isinstance(label, Mapping) and "算不出" not in label else {}
        for tag in TAG_COLUMNS:
            value = _scalar(label.get(tag))
            if tag == "德比" and tag in label:
                value = bool(value)  # 没有德比名明确表示非德比；缺整条标签才是未知。
            elif value is not None and not isinstance(value, bool):
                # 旧标签的 0/1 可读取，其他文本不能凭非空误当成 True。
                value = bool(value) if value in (0, 1) else None
            rows.append({**row, "tag": tag, "tag_value": UNKNOWN if value is None else value})
    return pd.DataFrame(rows, columns=[*df.columns, "tag", "tag_value"])


def build_live_metrics(scores: pd.DataFrame, *, benchmark_books: Sequence[str], n_bins: int,
                       min_matches: int, labels: Mapping | None = None) -> dict:
    """给任务、报告和网页共用的 JSON 数据；参数沿用配置，绝不依据当前结果调门槛。

    labels 直接接 attach_labels 的 {match_id: 标签字典}。旧数据缺模型身份、信心
    等级或标签时列入“未知”，不会补成当前版本或非标签比赛。
    """
    df = _prepare(scores)
    confidence = _summary(df, ["confidence"], min_matches)
    confidence_calibration = _calibration(df, ["confidence"], n_bins)
    for row, cal in zip(confidence, confidence_calibration):
        row["calibration"] = cal
    model = df[df["source"].isin(MODEL_SOURCES)]
    all_close = df[df["source"] == CLOSE_SOURCE]
    close = all_close[all_close["match_id"].isin(model["match_id"])]
    return {"groups": _summary(df, None, min_matches),
            "comparisons": _comparisons(df, benchmark_books, min_matches),
            "calibration": _calibration(df, None, n_bins), "confidence": confidence,
            "competitions": _summary(df, ["competition"], min_matches),
            "tags": _summary(_tag_rows(df, labels), ["tag", "tag_value"], min_matches),
            "counts": {"reviewed_matches": int(model["match_id"].nunique()), "model_rows": len(model),
                       "benchmark_rows": len(close), "benchmark_matches": int(close["match_id"].nunique()),
                       "orphan_benchmark_rows": len(all_close) - len(close)}}
