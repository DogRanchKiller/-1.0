"""开发回测和平博收盘比（src/evaluate/asia_study.py）的自动检查：日职中超（asia_study）、次级联赛（second_study）。

不用数据库、不读真的回测结果：在临时文件夹里造 predictions.csv（每场一行模型、一行平博），
检查合在一起的组、样本少的规则、考试赛季锁住、“差距之差”的 bootstrap（同一周一起抽）、五大联赛重跑核对、报告。
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.evaluate import asia_study as a

MODELS = ["gb_l7", "dyn_q1e5_b01"]


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def fake_pred(codes, seasons, per_week=4, weeks=30, gap=0.02, seed=0, start="2014-08-04"):
    """造一次回测的结果：每个联赛、每季 weeks 周、每周 per_week 场。
    平博 log loss 约 1.0；模型 = 平博 + gap + 噪声（gap 可以按联赛给：dict）。"""
    rng = np.random.default_rng(seed)
    rows, mid = [], 0
    for code in codes:
        g = gap[code] if isinstance(gap, dict) else gap
        for s in seasons:
            base = pd.Timestamp(f"{s}-{start[5:]}")
            for w in range(weeks):
                for k in range(per_week):
                    mid += 1
                    date = (base + pd.Timedelta(days=7 * w + k % 3)).strftime("%Y-%m-%d")
                    bench = 1.0 + rng.normal(0, 0.3)
                    info = {"match_id": mid, "competition": code, "season": s, "match_date": date}
                    rows.append({**info, "model": "pinnacle_close", "horizon": "收盘", "log_loss": bench,
                                 "brier": bench * 0.6, "hit": float(rng.random() < 0.5)})
                    for m in MODELS:
                        ll = bench + g + rng.normal(0, 0.05)
                        rows.append({**info, "model": m, "horizon": "T-24", "log_loss": ll,
                                     "brier": ll * 0.6, "hit": float(rng.random() < 0.45)})
    return pd.DataFrame(rows)


def write_run(root, name, pred):
    d = root / name
    d.mkdir(parents=True)
    pred.to_csv(d / "predictions.csv", index=False, encoding="utf-8-sig")


def second_settings(settings, tmp_path):
    """用真的 second_study 设置，只把读写的位置换到临时文件夹。"""
    s = copy.deepcopy(settings)
    s["backtest"]["output_dir"] = str(tmp_path / "backtest")
    s["second_study"]["output"] = str(tmp_path / "reports" / "次级.md")
    s["asia_study"]["output"] = str(tmp_path / "reports" / "亚洲.md")
    return s


# ---------------- 设置 ----------------
def test_second_study_settings_match_leagues_and_models(settings):
    """次级联赛就是 leagues.second 那五个，参照就是 leagues.top 那五个；模型名都是 settings 里真有的（没有编）。"""
    c = settings["second_study"]
    second, top = list(settings["leagues"]["second"]), list(settings["leagues"]["top"])
    assert list(c["runs"]) == second
    assert c["pooled"]["次级联赛合计"] == second
    assert c["gap_reference"]["codes"] == top
    assert c["gap_reference"]["same_country"] == dict(zip(second, top))
    m = settings["models"]
    assert set(c["models"]) == {"gb_l7", "dyn_q1e5_b01"}
    assert "gb_l7" in m["gbdt_variants"] and "dyn_q1e5_b01" in m["dynamic_variants"]
    assert settings["preview"]["model"] == "gb_l7"            # 上线的就是 gb_l7
    assert m["gbdt"]["dyn"] == "dyn_q1e5_b01"                 # gb_l7 里带的比分模型
    assert c["horizon"] == "T-24" and c["benchmark"] == "pinnacle_close"


# ---------------- 考试赛季锁住 ----------------
def test_exam_season_in_run_is_refused(settings, tmp_path):
    """回测结果里混进 2024（考试赛季）的比赛，读的时候直接报错，不拿来比。"""
    s = second_settings(settings, tmp_path)
    write_run(tmp_path / "backtest", "bad", fake_pred(["E1"], [2023, 2024], weeks=3))
    with pytest.raises(ValueError, match="不是开发赛季"):
        a.read_run("bad", s)
    with pytest.raises(ValueError, match="2025"):
        a.check_dev_seasons(pd.DataFrame({"season": [2022, 2025]}), s, "测试")
    a.check_dev_seasons(pd.DataFrame({"season": [2014, 2023]}), s, "测试")   # 开发赛季不报错


# ---------------- 合在一起的组、样本少 ----------------
def test_pooled_group_and_small_sample_rule(settings):
    """合计组的场数 = 各联赛之和；少于 min_total_n 场的，区间照算，结论写“样本少，不下结论”。"""
    cfg = dict(settings["second_study"], min_total_n=200)
    pred = fake_pred(["E1", "D2"], [2020, 2021], per_week=2, weeks=30, gap=0.03)   # 每个联赛 120 场
    cmp_cfg = dict(settings["compare"], n_boot=300)
    _, one = a.league_tables(pred, "E1", cfg, cmp_cfg)
    _, both = a.league_tables(pred, "合计", cfg, cmp_cfg, codes=["E1", "D2"])
    assert (one["n"] == 120).all() and (both["n"] == 240).all()
    assert (one["log_loss_verdict"] == a.SMALL).all()
    assert (both["log_loss_verdict"] == "更差").all()           # 240 场、差 0.03，区间整个 > 0
    assert not one["log_loss_lo"].isna().any()                 # 样本少也照样算区间


# ---------------- 差距之差的 bootstrap ----------------
def test_bootstrap_gap_value_and_verdict(settings):
    """a 的平均差 0.05、b 的 0.02：差距之差 ≈ 0.03，区间整个 > 0 → 差得更多；反过来 → 差得更少；一样 → 看不出大小。"""
    rng = np.random.default_rng(3)
    weeks = np.repeat([f"w{i:03d}" for i in range(300)], 5)
    da, db = 0.05 + rng.normal(0, 0.05, len(weeks)), 0.02 + rng.normal(0, 0.05, len(weeks))
    c = settings["compare"]
    g, lo, hi = a.bootstrap_gap(da, weeks, db, weeks, 1000, c["confidence"], c["seed"])
    assert g == pytest.approx(da.mean() - db.mean())
    assert lo < g < hi and a.gap_verdict(lo, hi) == a.GAP_MORE
    _, lo2, hi2 = a.bootstrap_gap(db, weeks, da, weeks, 1000, c["confidence"], c["seed"])
    assert a.gap_verdict(lo2, hi2) == a.GAP_LESS
    _, lo3, hi3 = a.bootstrap_gap(da, weeks, da + rng.normal(0, 0.05, len(weeks)), weeks, 1000,
                                  c["confidence"], c["seed"])
    assert a.gap_verdict(lo3, hi3) == a.GAP_SAME
    # 同样的种子，同样的区间
    assert a.bootstrap_gap(da, weeks, db, weeks, 1000, c["confidence"], c["seed"]) == (g, lo, hi)


def test_bootstrap_gap_draws_the_same_weeks_for_both_groups():
    """两组每周有一个共同的偏差（同一周一起偏）：一起抽，相减时抵消，区间很窄；
    如果两组各抽各的，区间会宽得多。用来确认真的是“同一周一起抽”。"""
    rng = np.random.default_rng(5)
    k = 200
    shock = rng.normal(0, 0.2, k)                     # 每周共同的偏差，比每场的噪声大很多
    weeks_a, weeks_b = np.repeat(np.arange(k), 4), np.repeat(np.arange(k), 6)
    da = shock[weeks_a] + rng.normal(0, 0.01, len(weeks_a))
    db = shock[weeks_b] + rng.normal(0, 0.01, len(weeks_b))
    _, lo, hi = a.bootstrap_gap(da, weeks_a, db, weeks_b, 1000, 0.95, 1)
    # 各抽各的：b 的周号换一套名字，两组就没有共同的周了
    _, lo_i, hi_i = a.bootstrap_gap(da, weeks_a, db, np.array([f"b{w}" for w in weeks_b]), 1000, 0.95, 1)
    assert (hi - lo) < 0.25 * (hi_i - lo_i)


def test_bootstrap_gap_refuses_empty():
    with pytest.raises(ValueError, match="没有可比较"):
        a.bootstrap_gap([], [], [0.1], ["w1"], 10, 0.95, 1)


# ---------------- 整个跑一遍 ----------------
def make_second_runs(s, tmp_path, top_gap=0.015):
    second, top = list(s["leagues"]["second"]), list(s["leagues"]["top"])
    seasons = list(range(2014, 2024))
    sec = fake_pred(second, seasons, per_week=3, weeks=34, gap={"E1": 0.04, "SP2": 0.015, "D2": 0.015,
                                                               "I2": 0.015, "F2": 0.015}, seed=1)
    top_pred = fake_pred(top, seasons, per_week=3, weeks=34, gap=top_gap, seed=2)
    # 和真的五大联赛一样：有 2 场没有平博收盘（模型照样出了预测）
    no_pin = top_pred["match_id"].drop_duplicates().iloc[[5, 50]]
    top_pred = top_pred[~(top_pred["match_id"].isin(no_pin) & (top_pred["model"] == "pinnacle_close"))]
    write_run(tmp_path / "backtest", s["second_study"]["runs"]["E1"][0], sec)
    write_run(tmp_path / "backtest", s["second_study"]["gap_reference"]["run"], top_pred)
    # 重跑核对要的数：按第 4 节步骤 6 的算法填（模型按自己出了预测的全部比赛平均，平博按有平博的全部比赛平均）
    pin = top_pred[top_pred["model"] == "pinnacle_close"]["log_loss"]
    ref = {}
    for m in MODELS:
        mine = top_pred[top_pred["model"] == m]["log_loss"]
        ref[m] = {"model": mine.mean(), "pinnacle": pin.mean(), "n": len(mine)}
    s["second_study"]["five_league_reference"] = ref
    return sec, top_pred


def test_second_study_end_to_end(settings, tmp_path):
    """次级联赛整套：合计表有 5 个联赛 + 合计组；英冠差 0.04（五大联赛 0.015）判“差得更多”，其他四个“看不出大小”；
    报告、csv 都写出来。"""
    s = second_settings(settings, tmp_path)
    s["compare"] = dict(s["compare"], n_boot=500)
    _, top_pred = make_second_runs(s, tmp_path)
    seasons, totals, out = a.run(s, "second_study")
    groups = list(s["leagues"]["second"]) + ["次级联赛合计"]
    assert set(totals["league"]) == set(groups) and set(totals["model"]) == set(MODELS)
    pooled = totals[totals["league"] == "次级联赛合计"]["n"].iloc[0]
    assert pooled == totals[totals["league"].isin(s["leagues"]["second"])]["n"].sum() / len(MODELS)
    gaps = pd.read_csv(out.with_name(out.stem + "_差距之差.csv"), encoding="utf-8-sig")
    e1 = gaps[gaps["group"] == "E1"]
    assert (e1["verdict"] == a.GAP_MORE).all() and (e1["gap"] > 0.015).all()
    others = gaps[gaps["group"].isin(["SP2", "D2", "I2", "F2"])]
    assert (others["verdict"] == a.GAP_SAME).all()
    assert (gaps["reference"] == "五大联赛").all()
    # 参照组只算两边都有平博收盘的比赛：造的 2 场没有平博收盘的不算
    assert (gaps["ref_n"] == top_pred["match_id"].nunique() - 2).all()
    same = pd.read_csv(out.with_name(out.stem + "_同国顶级联赛.csv"), encoding="utf-8-sig")
    assert dict(zip(same["group"], same["reference"])) == s["second_study"]["gap_reference"]["same_country"]
    eras = pd.read_csv(out.with_name(out.stem + "_前后两段.csv"), encoding="utf-8-sig")
    assert set(eras["era"]) == {"前", "后"} and "五大联赛" in set(eras["group"])
    text = out.read_text(encoding="utf-8")
    assert "次级联赛开发回测" in text and "比五大联赛差得多还是少" in text and "英冠" in text and "一样" in text
    era_lines = [l for l in text.splitlines() if l.startswith("| 五大联赛 | gb_l7 |")]
    assert [l.split(" | ")[2] for l in era_lines] == ["前", "后"]   # 前后两段表：先“前”再“后”
    assert "推荐" not in text and "下注" not in text


@pytest.mark.parametrize("key, change", [("n", 1), ("model", 0.001)])
def test_reproduce_mismatch_stops(settings, tmp_path, key, change):
    """五大联赛重跑和第 4 节的数对不上（场数或分数），直接报错，不往下出报告。"""
    s = second_settings(settings, tmp_path)
    s["compare"] = dict(s["compare"], n_boot=50)
    make_second_runs(s, tmp_path)
    s["second_study"]["five_league_reference"]["gb_l7"][key] += change
    with pytest.raises(ValueError, match="不一样"):
        a.run(s, "second_study")
    assert not (tmp_path / "reports" / "次级.md").exists()


def test_asia_section_still_works_without_extras(settings, tmp_path):
    """日职、中超那一段没有 pooled、gap_reference：照原来的样子出报告（和五大联赛并排那张表），不算差距之差。"""
    s = second_settings(settings, tmp_path)
    s["compare"] = dict(s["compare"], n_boot=50)
    pred = fake_pred(["JPN", "CHN"], [2021, 2022, 2023], per_week=3, weeks=30)
    for runs in s["asia_study"]["runs"].values():
        for i, r in enumerate(runs):
            if not (tmp_path / "backtest" / r).exists():
                write_run(tmp_path / "backtest", r, pred if i == 0 else pred[pred["model"] != "dyn_q1e5_b01"])
    _, totals, out = a.run(s)
    assert set(totals["league"]) == {"JPN", "CHN"}
    assert (totals.groupby("league")["n"].first() == 270).all()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# 日职、中超开发回测") and "和五大联赛比（log loss 差" in text
    assert "差距之差" not in text
    assert not out.with_name(out.stem + "_差距之差.csv").exists()


def test_kor_study_market_avg_main_and_pinnacle_look_only(settings, tmp_path):
    """韩K 那一段：主标杆是多家平均（报告写“多家平均”）；和平博比只看看（结论写“只看看”，只算有平博的比赛）；
    有和日职、中超、五大联赛并排的表。"""
    s = second_settings(settings, tmp_path)
    s["compare"] = dict(s["compare"], n_boot=50)
    s["kor_study"]["output"] = str(tmp_path / "reports" / "韩K.md")
    pred = fake_pred(["KOR"], [2021, 2022, 2023], per_week=3, weeks=30)
    avg = pred[pred["model"] == "pinnacle_close"].assign(model="market_avg")
    keep_pin = pred["model"] != "pinnacle_close"
    keep_pin |= pred["season"] == 2023                      # 平博只有 2023 有（90 场）
    write_run(tmp_path / "backtest", "kor_dev", pd.concat([pred[keep_pin], avg], ignore_index=True))
    _, totals, out = a.run(s, "kor_study")
    assert set(totals["league"]) == {"KOR"} and (totals["n"] == 270).all()
    assert (totals["log_loss_verdict"] == a.SMALL).all()    # 270 < min_total_n 1000：不下结论
    look = pd.read_csv(out.with_name(out.stem + "_只看看_平博.csv"), encoding="utf-8-sig")
    assert (look["n"] == 90).all() and (look["log_loss_verdict"] == a.LOOK).all()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# 韩K 开发回测") and "多家平均" in text and "只看看：和平博收盘比" in text
    assert "和日职、中超、五大联赛并排" in text and "| 日职 | dyn_q1e5_b01 | 平博收盘 | 3,132 |" in text
    assert "推荐" not in text and "下注" not in text
