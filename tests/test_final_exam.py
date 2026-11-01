"""期末考试（预测部分）src/evaluate/final_exam.py：只考一次的检查、五条检查在假数据上能算、算得对。不碰真数据库。"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.evaluate import final_exam as fe
from src.evaluate import metrics
from src.evaluate.confidence_study import build as cs_build
from src.models.goals import score_matrices

SETTINGS = load_settings()


def _probs(m):
    """矩阵 → [主胜, 平, 客胜, 大 2.5 球]。"""
    i, j = np.indices(m.shape)
    return np.array([(m * (i > j)).sum(), (m * (i == j)).sum(), (m * (i < j)).sum(), (m * (i + j > 2)).sum()])


def _fake(model_noise=0.4, market_noise=0.03, seasons=(2024, 2025), n_teams=14, seed=5):
    """造一次假考试：一个联赛，每季换 1 个队（升班马）。真实比分按“真”的 λ 抽；模型的 λ 噪声大，赔率的 λ 噪声小。
    平博收盘 = 赔率；每 9 场没有平博（胜平负用多家平均）；第 1 场没有赔率（不融合，信心等级降一级）。"""
    rng = np.random.default_rng(seed)
    rows, mats, odds, mid = [], [], [], 0
    for si, s in enumerate(seasons):
        teams = [f"T{k}" for k in range(si, si + n_teams)]
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        for k, (h, a) in enumerate(fixtures):
            mid += 1
            lh, la = rng.uniform(0.8, 2.2), rng.uniform(0.6, 1.8)
            hg, ag = rng.poisson(lh), rng.poisson(la)
            m = score_matrices(np.array([lh * np.exp(rng.normal(0, model_noise))]),
                               np.array([la * np.exp(rng.normal(0, model_noise))]))[0].astype(np.float32)
            mk = score_matrices(np.array([lh * np.exp(rng.normal(0, market_noise))]),
                                np.array([la * np.exp(rng.normal(0, market_noise))]), rho=-0.05)[0]
            p = _probs(m.astype(float) / m.sum())
            q = _probs(mk)
            y = 0 if hg > ag else (1 if hg == ag else 2)
            date = pd.Timestamp(f"{s}-08-10") + pd.Timedelta(days=k // 5 * 7)
            base = {"match_id": mid, "competition": "E0", "season": s, "match_date": date.strftime("%Y-%m-%d"),
                    "kickoff_utc": (date + pd.Timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "home_team": h, "away_team": a, "result": "HDA"[y], "home_goals": hg, "away_goals": ag}

            def scored(p3):
                p3 = np.asarray(p3, float)[None, :3]
                yy = np.array([y])
                return {"p_home": p3[0, 0], "p_draw": p3[0, 1], "p_away": p3[0, 2],
                        "log_loss": metrics.log_loss_per_match(p3, yy)[0], "brier": metrics.brier_per_match(p3, yy)[0],
                        "rps": metrics.rps_per_match(p3, yy)[0]}
            rows.append({**base, "model": "gb_l7", "horizon": "T-24", **scored(p),
                         "cutoff_utc": (date - pd.Timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%SZ")})
            pin = mid % 9 != 0
            if pin:
                rows.append({**base, "model": "pinnacle_close", "horizon": "收盘", **scored(q)})
            rows.append({**base, "model": "market_avg", "horizon": "收盘", **scored(q)})
            mats.append(m)
            odds.append({"match_id": mid, "mo_p_home": q[0], "mo_p_draw": q[1], "mo_p_away": q[2], "mo_over25": q[3],
                         "mo_pinnacle": float(pin), "ah_line": -0.25 if lh > la else 0.25, "ah_home_odds": 1.9,
                         "ah_away_odds": 1.95})
    odds[0].update(mo_p_home=np.nan, mo_p_draw=np.nan, mo_p_away=np.nan, mo_over25=np.nan)
    pred = pd.DataFrame(rows)
    m = pred.drop_duplicates("match_id")
    promoted = pd.Series(m["season"].to_numpy() > seasons[0], index=m["match_id"].to_numpy())  # 简化：第二季都算有升班马
    return pred, np.arange(1, mid + 1), np.stack(mats), pd.DataFrame(odds), promoted


def test_check_not_done(tmp_path):
    s = copy.deepcopy(SETTINGS)
    s["final_exam"]["done"] = False
    fe.check_not_done(s, tmp_path)                     # 都没有：可以考
    (tmp_path / fe.DONE_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(fe.ExamAlreadyDone):            # 结果文件夹里有 exam_done.json：考过了
        fe.check_not_done(s, tmp_path)
    s["final_exam"]["done"] = True
    with pytest.raises(fe.ExamAlreadyDone):            # settings 写了 done：考过了
        fe.check_not_done(s, tmp_path / "别的文件夹")


def test_run_exam_end_to_end():
    pred, ids, mats, odds, promoted = _fake()
    res = fe.run_exam(pred, ids, mats, odds, promoted, SETTINGS)
    s = res["summary"]
    n = 2 * 14 * 13
    assert s["数据"]["融合"] == n - 1                                    # 第 1 场没有赔率
    assert s["第 2 条 融合 对 只用模型"]["通过"] is True                 # 赔率噪声小得多 → 融合更准
    assert s["第 2 条 融合 对 只用模型"]["w"] == SETTINGS["fusion"]["w"]
    f = res["fusion"]
    assert {"全部", "联赛", "赛季", "胜平负用不用平博"} == set(f["分法"])
    assert set(f.loc[f["分法"] == "赛季", "哪一份"]) == {"2024/25", "2025/26"}
    # 融合 对 平博收盘：只比两边都有的比赛（每 9 场没有平博）
    r = f[(f["分法"] == "全部") & (f["base"] == fe.FUSED) & (f["new"] == "pinnacle_close") & (f["metric"] == "log_loss")]
    assert int(r["n"].iloc[0]) == n - 1 - n // 9
    g = res["w_grid"]
    assert len(g[g["分法"] == "全部"]) == 21
    # 第 1 条：两个标杆，全部 + 两季
    m = res["model"]
    assert set(m["base"]) == {"pinnacle_close", "market_avg"} and set(m["哪一份"]) == {"全部", "2024/25", "2025/26"}
    assert int(m[(m["base"] == "market_avg") & (m["哪一份"] == "全部")]["n"].iloc[0]) == n
    # 第 3 条：五个玩法都有一行，门槛是现在的
    lv = res["leans"].set_index("玩法")
    assert list(lv.index) == ["result", "draw", "over25", "btts", "ah"]
    assert lv.loc["result", "门槛"] == SETTINGS["lean"]["result"]
    assert set(res["lean_bands"]["玩法"]) == set(lv.index)
    # 第 4 条：现在没有降级因素 → 有赔率的都是“高”，没赔率的那场是“中”；“中”没有 d → 比不了
    lt = res["levels"].set_index("等级")
    assert lt["场数"].sum() == n and lt.loc["●●○ 中", "场数"] == 1 and lt.loc["●●○ 中", "有 d 的场数"] == 0
    assert s["第 4 条 信心等级"]["“高”的 d 不大于“中”“低”"] == "比不了"
    assert "算数（只报告，不改规则）" in res["factors"].columns
    assert {"2024/25", "2025/26"} <= set(res["factors"]["哪一份"])
    # 第 5 条：每个玩法每档的场数加起来 = 融合的场数
    c = res["calibration"]
    assert (c.groupby("玩法")["n"].sum() == n - 1).all()
    json.dumps(s, ensure_ascii=False, default=str)                      # 摘要能存成 json


def test_fusion_fails_when_odds_worse():
    pred, ids, mats, odds, promoted = _fake(model_noise=0.03, market_noise=0.5, seed=8)
    res = fe.run_exam(pred, ids, mats, odds, promoted, SETTINGS)
    assert res["summary"]["第 2 条 融合 对 只用模型"]["通过"] is False


def test_lean_check_matches_definition():
    pred, ids, mats, odds, promoted = _fake()
    t, model, market, _ = fe.fu.build(pred, ids, mats, odds, SETTINGS)
    fused = fe.fu.mix(model, market, SETTINGS["fusion"]["w"])
    s = copy.deepcopy(SETTINGS)
    s["lean"]["btts"] = None
    leans, _ = fe.lean_check(t, fused, s)
    lv = leans.set_index("玩法")
    assert lv.loc["btts", "通过"] == "没有门槛"
    # 胜平负：自己按定义再算一次
    strength, value = fe.fu.lean_values(t, fused)["result"]
    k = strength >= s["lean"]["result"] - 1e-12
    assert lv.loc["result", "有倾向的场数"] == k.sum()
    assert abs(lv.loc["result", "实际的平均"] - value[k].mean()) < 1e-12
    assert abs(lv.loc["result", "偏向那边发生的比例"] - (value[k] > 0).mean()) < 1e-12
    assert lv.loc["result", "通过"] == (lv.loc["result", "区间下限"] > 0)
    assert fe.hit_rate("ah", np.array([1.0, 0.5, -1.0, 0.0])) == pytest.approx(1.5 / 2.5)


def test_calibration_bins():
    t = pd.DataFrame({"y": [0, 1, 2, 0], "over": [1.0, 0, 1, 0], "btts_yes": [1.0, 1, 0, 0]})
    p3 = np.array([[1.0, 0, 0], [0.05, 0.9, 0.05], [0.2, 0.3, 0.5], [0.15, 0.35, 0.5]])
    fused = {"p3": p3, "over25": np.array([0.55, 0.55, 0.55, 0.55]), "btts": np.array([0.1, 0.2, 0.3, 0.4])}
    c = fe.calibration_check(t, fused, 0.1)
    home = c[c["玩法"] == "主胜"].set_index("下沿")
    assert home.loc[0.9, "n"] == 1 and home.loc[0.9, "实际比例"] == 1.0     # 概率 1.0 算进最后一档
    assert home.loc[0.1, "n"] == 1 and home.loc[0.1, "实际比例"] == 1.0     # 0.15 在 0.1 那档（第 4 场主胜）
    assert home.loc[0.2, "n"] == 1 and home.loc[0.2, "实际比例"] == 0.0     # 正好 0.2 算进 0.2 那档（下沿 ≤ 概率）
    over = c[c["玩法"] == "大 2.5 球"]
    assert len(over) == 1 and over["n"].iloc[0] == 4 and over["实际比例"].iloc[0] == 0.5


def test_confidence_needs_promoted_for_every_match_and_w1():
    pred, ids, mats, odds, promoted = _fake()
    with pytest.raises(ValueError):
        cs_build(pred, odds, SETTINGS, promoted=promoted.iloc[:-5])        # 有比赛不知道有没有升班马
    s = copy.deepcopy(SETTINGS)
    s["fusion"]["w"] = 0.5
    with pytest.raises(ValueError):
        fe.confidence_check(pred, odds, promoted, s)                      # build 按 w = 1 算融合


def test_reference_check(tmp_path):
    rows = [{"match_id": i, "season": s, "model": "gb_l7", "horizon": "T-24", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
            for i, s in [(1, 2022), (2, 2023), (3, 2024)]]
    ref = pd.DataFrame(rows)
    ref.to_csv(tmp_path / "predictions.csv", index=False, encoding="utf-8-sig")
    full = ref.copy()
    full.loc[1, ["p_home", "p_away"]] = [0.48, 0.22]                  # 2023 那场差 0.02；2024 是打分赛季，不比
    full.loc[2, "p_home"] = 0.9
    r = fe.reference_check(full, tmp_path, "gb_l7", 2024)
    assert r["两边都有"] == 2 and abs(r["胜平负概率最大差"] - 0.02) < 1e-12


class _Args:
    def __init__(self, dry_run=False, reference="ref", skip_backtest=False, reason=None):
        self.dry_run, self.reference, self.skip_backtest, self.reason = dry_run, reference, skip_backtest, reason


def test_exam_guards(tmp_path):
    s = copy.deepcopy(SETTINGS)
    s["final_exam"]["done"] = False
    clean = {"commit": "abc", "有没提交的改动": False}
    assert fe.exam_guards(s, tmp_path, {"commit": "x", "有没提交的改动": True}, _Args(dry_run=True)) is None  # 试跑不查
    with pytest.raises(RuntimeError):                                   # 有没提交的改动
        fe.exam_guards(s, tmp_path, {"commit": "abc", "有没提交的改动": True}, _Args())
    with pytest.raises(RuntimeError):                                   # 查不到 git 状态
        fe.exam_guards(s, tmp_path, {"commit": None, "有没提交的改动": None}, _Args())
    with pytest.raises(RuntimeError):                                   # 没给 --reference
        fe.exam_guards(s, tmp_path, clean, _Args(reference=None))
    with pytest.raises(RuntimeError):                                   # 没开考过就想重跑
        fe.exam_guards(s, tmp_path, clean, _Args(skip_backtest=True, reason="修 bug"))
    rec = fe.exam_guards(s, tmp_path, clean, _Args())                   # 开考：写开考记录
    assert (tmp_path / fe.STARTED_FILE).exists() and rec["重跑"] == [] and rec["git"] == clean
    with pytest.raises(fe.ExamAlreadyDone):                             # 开考过，不能当新考试再来
        fe.exam_guards(s, tmp_path, clean, _Args())
    with pytest.raises(RuntimeError):                                   # 重跑没写原因
        fe.exam_guards(s, tmp_path, clean, _Args(skip_backtest=True))
    s2 = copy.deepcopy(s)
    s2["lean"]["result"] = 0.05
    with pytest.raises(RuntimeError):                                   # 开考后改了设置
        fe.exam_guards(s2, tmp_path, clean, _Args(skip_backtest=True, reason="修 bug"))
    s3 = copy.deepcopy(s)
    s3["final_exam"]["done"] = False                                    # done 不算设置的一部分
    rec = fe.exam_guards(s3, tmp_path, clean, _Args(skip_backtest=True, reason="修 bug"))
    assert len(rec["重跑"]) == 1 and rec["重跑"][0]["原因"] == "修 bug"
    saved = json.loads((tmp_path / fe.STARTED_FILE).read_text(encoding="utf-8"))
    assert len(saved["重跑"]) == 1
    (tmp_path / fe.DONE_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(fe.ExamAlreadyDone):                             # 考完了，重跑也不行
        fe.exam_guards(s, tmp_path, clean, _Args(skip_backtest=True, reason="修 bug"))


def test_check_reference():
    ok = {"胜平负概率最大差": 0.0, "两边都没出预测的不算，一边出一边没出的场数": 0, "两边都有": 5, "参考的场数": 5}
    fe.check_reference(ok)
    for bad in ({"胜平负概率最大差": 1e-6}, {"两边都没出预测的不算，一边出一边没出的场数": 1}, {"两边都有": 4},
                {"胜平负概率最大差": None}):
        with pytest.raises(RuntimeError):
            fe.check_reference({**ok, **bad})
    with pytest.raises(RuntimeError):
        fe.check_reference(None)


def test_reference_check_one_side(tmp_path):
    rows = [{"match_id": i, "season": 2022, "model": "gb_l7", "horizon": "T-24", "p_home": 0.5, "p_draw": 0.3,
             "p_away": 0.2} for i in (1, 2, 3)]
    ref = pd.DataFrame(rows)
    ref.to_csv(tmp_path / "predictions.csv", index=False, encoding="utf-8-sig")
    full = ref.copy()
    full.loc[0, ["p_home", "p_draw", "p_away"]] = np.nan              # 这次没出预测、参考出了
    full = full[full["match_id"] != 3]                                  # 这次少一场
    r = fe.reference_check(full, tmp_path, "gb_l7", 2024)
    assert r["两边都没出预测的不算，一边出一边没出的场数"] == 1 and r["胜平负概率最大差"] == 0.0
    assert r["两边都有"] == 2 and r["参考的场数"] == 3
    with pytest.raises(RuntimeError):
        fe.check_reference(r)


def test_promoted_history():
    pred, *_ = _fake(seasons=(2023, 2024, 2025))
    m = pred.drop_duplicates("match_id").drop(columns=["model", "horizon"])
    calls = []

    class FS:
        def prediction_targets(self, start, end, usage, competition=None):
            calls.append((start, end, usage, competition))
            return m

    p = fe.promoted_history(FS(), 2024, 2025, SETTINGS)
    b = SETTINGS["backtest"]
    assert calls == [(f"2023-{b['window_start_md']}", f"2026-{b['window_end_md']}", "A版输入",
                      list(SETTINGS["leagues"]["top"]))]
    assert set(p.index) == set(m["match_id"])                           # 2023 也在（只用来看上赛季在不在）
    s24 = m.loc[m["season"] == 2024, "match_id"]
    s23 = m.loc[m["season"] == 2023, "match_id"]
    assert p.loc[s24].any() and not p.loc[s23].any()                    # 2024 有升班马；2023 是第一季，看不到上一季


def test_fusion_passed_only_looks_at_model_vs_fused():
    row = {"分法": "全部", "哪一份": "全部", "metric": "log_loss"}
    t = pd.DataFrame([{**row, "base": fe.MODEL, "new": fe.FUSED, "verdict": "没差别"},
                      {**row, "base": fe.FUSED, "new": "pinnacle_close", "verdict": fe.BETTER}])
    assert fe.fusion_passed(t) is False                                 # 融合对平博那行更准，不算
    t.loc[0, "verdict"] = fe.BETTER
    assert fe.fusion_passed(t) is True
    assert fe.fusion_passed(t[t["base"] == fe.FUSED]) is False          # 没有那一行 = 不通过


def test_lean_check_no_lean_matches():
    pred, ids, mats, odds, promoted = _fake()
    t, model, market, _ = fe.fu.build(pred, ids, mats, odds, SETTINGS)
    fused = fe.fu.mix(model, market, SETTINGS["fusion"]["w"])
    s = copy.deepcopy(SETTINGS)
    s["lean"]["over25"] = 0.99                                          # 门槛高到没有一场
    leans, _ = fe.lean_check(t, fused, s)
    lv = leans.set_index("玩法")
    assert lv.loc["over25", "通过"] == "没有有倾向的比赛" and lv.loc["over25", "有倾向的场数"] == 0


def test_confidence_pair_ok():
    pred, ids, mats, odds, promoted = _fake()
    pre = pred[pred["model"] == "market_avg"].assign(horizon="赛前代替")
    pre = pre[pre["match_id"] <= 40]                                    # 前 40 场有多家平均赛前
    _, _, check = fe.confidence_check(pd.concat([pred, pre]), odds, promoted, SETTINGS)
    want = sum(1 for i in range(2, 41) if i % 9 != 0)                   # 第 1 场没赔率；每 9 场没有平博
    assert check["两家都有赛前的比赛（“没有平博”那条能不能算）"] == want
