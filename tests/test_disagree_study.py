"""模型和赔率意见不一样时谁准 src/evaluate/disagree_study.py 的检查（只用合成数据，不读真实回测、不读数据库）。

核对：区间算法和 compare.py 一样、分档界限、各类标签、赛季阶段、只许开发赛季、模型设置改过要停、
多重比较的倍数、可信程度的规则、整套跑通（模型真的更准 / 更差两种情况）、跟随比例 β 的含义、报告不出现不该有的词。
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.evaluate import disagree_study as ds
from src.evaluate.compare import bootstrap_mean_diff, block_ids


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s["compare"]["n_boot"] = 300
    return s


def _probs(rng, n, conc=(9, 6, 6)):
    return rng.dirichlet(conc, n)


def _season_matches(comp, season, n_teams, start):
    """一个双循环赛季的赛程：n_teams 个队，每轮 n_teams/2 场，每周一轮。"""
    teams = [f"{comp}{i}" for i in range(n_teams)]
    rows, day = [], pd.Timestamp(start)
    rounds = []
    for r in range(n_teams - 1):                     # 轮转法排单循环
        pairs = [(teams[i], teams[n_teams - 1 - i]) for i in range(n_teams // 2)]
        rounds.append(pairs)
        teams = [teams[0], teams[-1], *teams[1:-1]]
    rounds += [[(b, a) for a, b in pr] for pr in rounds]
    for pr in rounds:
        for h, a in pr:
            rows.append({"competition": comp, "season": season, "match_date": day.strftime("%Y-%m-%d"),
                         "kickoff_utc": "", "home_team": h, "away_team": a})
        day += pd.Timedelta(days=7)
    return rows


def make_data(truth="odds", seed=0, seasons=(2010, 2012, 2016, 2018), exam=False):
    """合成回测结果：真实结果按 truth 那边的概率抽。truth='odds' → 赔率是真的、模型加了噪声；'model' → 反过来。"""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for comp, start in [("E0", f"{s}-08-10"), ("D1", f"{s}-08-12")]:
            rows += _season_matches(comp, s, 20, start)
    if exam:
        rows += _season_matches("E0", 2024, 4, "2024-08-10")
    m = pd.DataFrame(rows)
    m["match_id"] = np.arange(1, len(m) + 1)
    n = len(m)
    true = _probs(rng, n)
    noisy = np.clip(true + rng.normal(0, 0.08, (n, 3)), 0.02, None)
    noisy /= noisy.sum(axis=1, keepdims=True)
    model, odds = (noisy, true) if truth == "odds" else (true, noisy)
    res = np.array(["H", "D", "A"])[[rng.choice(3, p=p) for p in true]]
    base = m.assign(result=res, benchmark_period="x")
    mp = base.assign(model="gb_l7", horizon="T-24", p_home=model[:, 0], p_draw=model[:, 1], p_away=model[:, 2])
    close = (odds + 0.5 * (model - odds))                    # 收盘正好往模型走一半 → β = 0.5
    cp = base.assign(model="pinnacle_close", horizon="收盘", p_home=close[:, 0], p_draw=close[:, 1], p_away=close[:, 2])
    pred = pd.concat([mp, cp], ignore_index=True)
    o = pd.DataFrame({"match_id": m["match_id"], "mo_p_home": odds[:, 0], "mo_p_draw": odds[:, 1],
                      "mo_p_away": odds[:, 2], "mo_over25": 0.5, "mo_pinnacle": (m["season"] >= 2012).astype(float)})
    o.loc[o.index[:3], "mo_p_home"] = np.nan                # 3 场没有赛前赔率
    return pred, o


# ---------------- 小零件 ----------------
def test_boot_means_same_as_compare(settings):
    """和 compare.bootstrap_mean_diff 同一个算法、同一个种子：分位数应完全一样。"""
    rng = np.random.default_rng(1)
    diff = rng.normal(0.01, 0.2, 600)
    dates = pd.Series(pd.date_range("2015-08-01", periods=600, freq="D").strftime("%Y-%m-%d"))
    blocks = block_ids(dates, "week")
    c = settings["compare"]
    mean, lo, hi = bootstrap_mean_diff(diff, blocks, c["n_boot"], c["confidence"], c["seed"])
    means = ds.boot_means(diff, blocks, c["n_boot"], c["seed"])
    assert np.allclose(np.quantile(means, [0.025, 0.975]), [lo, hi])
    with pytest.raises(ValueError):
        ds.boot_means([], [], 10, 1)


def test_bins_and_labels():
    assert ds._bin_labels([0.03, 0.06, 0.10]) == ["3% 以内", "3%–6%", "6%–10%", "10% 以上"]
    labels = ds._bin_labels([0.03, 0.06])
    got = ds._cut(np.array([0.0, 0.0299, 0.03, 0.0599, 0.06, 0.5]), [0.03, 0.06], labels)
    assert list(got) == ["3% 以内", "3% 以内", "3%–6%", "3%–6%", "6% 以上", "6% 以上"]   # 正好在分界上算上面一档


def test_group_labels(settings):
    cfg = settings["disagree_study"]
    df = pd.DataFrame({
        "p_home": [0.52, 0.31, 0.70], "p_draw": [0.30, 0.25, 0.15], "p_away": [0.18, 0.44, 0.15],
        "q_home": [0.40, 0.35, 0.65], "q_draw": [0.30, 0.20, 0.20], "q_away": [0.30, 0.45, 0.15],
        "q_pinnacle": [1.0, 0.0, 1.0], "season": [2010, 2016, 2023], "competition": ["E0", "SP1", "XX"]})
    g = ds.add_groups(df, cfg, settings)
    assert np.allclose(g["gap"], [0.12, 0.05, 0.05])
    assert list(g["gap_group"]) == ["10% 以上", "3%–6%", "3%–6%"]
    assert list(g["strength"]) == ["势均力敌", "一方稍强", "强弱分明"]       # 赔率最大 0.40 / 0.45 / 0.65
    assert list(g["draw_level"]) == ["平局概率高", "平局概率低", "平局概率低"]
    assert list(g["side"]) == ["模型更看好主胜", "模型更看好平局", "模型更看好主胜"]
    assert list(g["favdir"]) == ["模型更看好热门", "模型更看好冷门", "模型更看好热门"]
    assert list(g["source"]) == ["平博赛前", "多家平均赛前", "平博赛前"]
    assert list(g["period"]) == ["2010–2013", "2014–2023", "2014–2023"]
    assert list(g["league"]) == ["英超", "西甲", "XX"]


def test_season_phase():
    """4 个队、双循环 6 轮、每轮 2 场；前 1 轮 = 前 2 场早期，最后 2 场后期，其余中段。"""
    m = pd.DataFrame(_season_matches("E0", 2015, 4, "2015-08-01"))
    m["match_id"] = np.arange(len(m))
    ph = ds.season_phase(m.sample(frac=1, random_state=0), 1)       # 打乱顺序也要按时间排
    assert list(ph.sort_index()) == ["早期"] * 2 + ["中段"] * 8 + ["后期"] * 2


def test_dev_only_and_models_check(settings):
    assert ds.check_dev_only([2005, 2023], settings) == [2005, 2023]
    with pytest.raises(ValueError, match="不是开发赛季"):
        ds.check_dev_only([2023, 2024], settings)
    old = {"设置": {"models": copy.deepcopy({k: v for k, v in settings["models"].items() if k != "gbdt_league_groups"})}}
    assert "gbdt" in ds.check_models_settings(old, settings)          # 回测以后新加的项不影响
    old["设置"]["models"]["gbdt"] = {"改过": 1}
    with pytest.raises(ValueError, match="不一样"):
        ds.check_models_settings(old, settings)
    with pytest.raises(ValueError, match="没有 models"):
        ds.check_models_settings({}, settings)


def test_exam_season_in_file_stops(settings):
    pred, odds = make_data(exam=True)
    with pytest.raises(ValueError, match="不是开发赛季"):
        ds.run(pred, odds, settings)


def test_bonferroni_and_credibility():
    assert ds.bonferroni_z(27, 0.95) == pytest.approx(3.11, abs=0.01)
    assert ds.bonferroni_z(1, 0.95) == pytest.approx(1.96, abs=0.01)
    base = {"结论": ds.MODEL_BETTER, "校正后上限": -0.001, "平均差_2005–2013": -0.01, "平均差_2014–2023": -0.02}
    assert ds.credibility(base) == ds.SURE
    assert ds.credibility({**base, "校正后上限": 0.001}) == ds.MAYBE         # 校正后跨过 0
    assert ds.credibility({**base, "平均差_2014–2023": 0.003}) == ds.MAYBE   # 有一段是正数
    assert ds.credibility({**base, "平均差_2014–2023": np.nan}) == ds.SURE   # 这组只有一段：只看有的那段
    assert ds.credibility({**base, "结论": ds.UNCLEAR}) == ""


# ---------------- 整套 ----------------
def test_run_odds_true(settings):
    """赔率是真的、模型加噪声：全部比赛“赔率更准”，意见不一样的比赛也是；没有一组“比较可信”的模型更准。"""
    pred, odds = make_data("odds")
    res = ds.run(pred, odds, settings)
    g, info = res["groups"], res["info"]
    assert info["场数"]["没有赛前赔率的场数"] == 3
    assert info["场数"]["比较的场数"] == info["场数"]["模型预测场数"] - 3
    allrow = g[(g["范围"] == "全部比赛") & (g["分类"] == "全部")].iloc[0]
    assert allrow["结论"] == ds.ODDS_BETTER and allrow["差"] > 0
    assert (g["可信程度"] != ds.SURE).all()
    assert set(g["分类"]) >= {"分歧大小", "联赛", "两队强弱差距", "平局概率", "赛季阶段", "模型比赔率多给哪项", "热门冷门",
                            "赔率来源", "两段"}
    assert set(g.loc[g["分类"] == "赛季阶段", "组"]) == {"早期", "中段", "后期"}
    # 区间都把平均差包在里面，校正后的更宽
    assert ((g["下限"] <= g["差"]) & (g["差"] <= g["上限"])).all()
    assert ((g["校正后下限"] <= g["下限"] + 1e-12) & (g["校正后上限"] >= g["上限"] - 1e-12)).all()
    c = res["concentration"]
    assert c.loc[c["分类"] == "联赛", "场数"].sum() == info["场数"]["比较的场数"]


def test_run_model_true(settings):
    """反过来：模型是真的、赔率加噪声 → 全部比赛和意见不一样的比赛都是“模型更准”，而且“比较可信”。"""
    pred, odds = make_data("model", seed=2)
    g = ds.run(pred, odds, settings)["groups"]
    row = g[g["范围"].str.startswith("意见不一样") & (g["分类"] == "全部")].iloc[0]
    assert row["结论"] == ds.MODEL_BETTER
    allrow = g[(g["范围"] == "全部比赛") & (g["分类"] == "全部")].iloc[0]
    assert allrow["结论"] == ds.MODEL_BETTER
    assert (g.loc[g["算进多重比较"] & (g["结论"] == ds.MODEL_BETTER), "可信程度"] != "").all()
    assert (g["可信程度"] == ds.SURE).any()


def test_follow_beta(settings):
    """收盘 = 赛前 + 0.5 ×（模型 − 赛前）→ β 正好 0.5、往模型方向走的比例 100%；只用平博赛前的比赛。"""
    pred, odds = make_data("odds")
    res = ds.run(pred, odds, settings)
    f = res["follow"]
    row = f[f["组"] == "全部"].iloc[0]
    assert row["跟随比例β"] == pytest.approx(0.5, abs=1e-9)
    assert row["结论"] == "赔率往模型那边走"
    assert row["往模型方向走的比例"] == pytest.approx(1.0)
    df = res["frame"]
    assert row["场数"] == int(((df["q_pinnacle"] == 1) & df["c_home"].notna()).sum())


def test_report(settings, tmp_path):
    pred, odds = make_data("odds")
    res = ds.run(pred, odds, settings)
    res["info"].update({"生成时间": "test", "数据": {"predictions.csv": {"路径": "合成"}}})
    text = ds.write_report(res, tmp_path).read_text(encoding="utf-8")
    assert "谁准" in text and "多重比较" in text and "跟随比例" in text
    assert "推荐" not in text and "下注建议" not in text


# ---------------- 审查后补的：容差、核对、只许开发模式、档名个数 ----------------
def test_renormalize_tolerance():
    x = pd.DataFrame({"a": [0.5, 0.3, np.nan], "b": [0.3, 0.3, 0.2], "c": [0.200001, 0.4, 0.8]})
    got = ds.renormalize(x, 1e-4)
    assert np.allclose(got.iloc[:2].sum(axis=1), 1.0)                 # 四舍五入的小差补成正好 1
    assert got.iloc[2].isna().all()                                   # 有空值的行不算（三项都是空）
    x.loc[1, "c"] = 0.41                                              # 加起来 1.01：不是四舍五入，要停
    with pytest.raises(ValueError, match="离 1 太远"):
        ds.renormalize(x, 1e-4)


def test_log_loss_check(settings):
    pred, odds = make_data("odds")
    m = pred["model"] == "gb_l7"
    y = ds.metrics.encode_results(pred.loc[m, "result"])
    pred.loc[m, "log_loss"] = ds.metrics.log_loss_per_match(pred.loc[m, ds.P].to_numpy(float), y)
    res = ds.run(pred, odds, settings)                                # 记下的和重算的一样：照常跑
    assert res["info"]["场数"]["重算的模型 log loss 和回测记下的最大差"] < 1e-9
    pred.loc[pred.index[m][10], "log_loss"] += 0.01                   # 有一场对不上：停下
    with pytest.raises(ValueError, match="重算的模型 log loss"):
        ds.run(pred, odds, settings)


def test_label_count_must_match(settings):
    pred, odds = make_data("odds")
    settings["disagree_study"]["fav_bins"] = [0.40, 0.50, 0.60]      # 分出 4 档，名字还是 3 个
    with pytest.raises(ValueError, match="名字却有 3 个"):
        ds.run(pred, odds, settings)


def _write_backtest(tmp_path, settings, mode):
    import json
    pred, odds = make_data("odds")
    base = tmp_path / "bt"
    base.mkdir()
    pred.to_csv(base / "predictions.csv", index=False, encoding="utf-8-sig")
    odds.to_csv(base / "fusion_odds.csv", index=False, encoding="utf-8-sig")
    info = {"运行模式": mode, "设置": {"models": copy.deepcopy(settings["models"])}}
    (base / "run_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    settings["disagree_study"]["base_dir"] = str(base)
    settings["disagree_study"]["output_dir"] = str(tmp_path / "out")
    return settings


def test_main_dev_only(settings, tmp_path):
    s = _write_backtest(tmp_path, settings, "final_exam")
    with pytest.raises(ValueError, match="不是开发模式"):
        ds.main(s)
    assert not (tmp_path / "out").exists()                            # 停下时什么都没写


def test_main_writes_outputs(settings, tmp_path):
    s = _write_backtest(tmp_path, settings, "dev")
    ds.main(s)
    out = tmp_path / "out"
    for f in ("groups.csv", "concentration.csv", "follow.csv", "report.md", "run_info.json"):
        assert (out / f).exists()
