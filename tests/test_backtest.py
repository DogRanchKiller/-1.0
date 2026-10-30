"""回测框架（src/evaluate/backtest.py）和模型比较（src/evaluate/compare.py）的自动检查。

用内存里的临时数据库（schema.py 建表）造一个小联赛，不碰真实数据库：
  E0，6 个队，每个赛季双循环 10 轮、每轮 3 场（共 30 场），每周六一轮。
  2021/22：A B C D E F；2022/23：F 降级、G 升级。回测 2022/23（2021/22 只当历史）。
  每轮第 1 场 14:00 UTC 开球，第 2 场 16:30，第 3 场只有日期（检查“只有日期”的截止时间算法）。
  2022/23 第 5 轮第 1 场改期（postponed，没有比分）：赛前不知道它会改期，照样要出预测，只是没法打分。
  2022/23 每轮第 1 场有平博收盘赔率 2.0 / 3.5 / 4.0（参考标杆）。
  另有一场 2024/25（期末考试赛季）的比赛：开发模式下回测不能碰它。
模型的最少场数调成 20、Elo 不留“热身赛季”，这样小联赛也能出预测。
"""
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.odds import devig_proportional
from src.evaluate import compare as cmp
from src.evaluate.backtest import BENCHMARK, run_backtest, save
from src.features import FeatureStore, SeasonLockedError, UsageNotAllowedError
from src.models.base import Model
from src.models.frequency import FrequencyModel
from src.models.registry import build_model

STRENGTH = {"A": 0.6, "B": 0.3, "C": 0.1, "D": -0.1, "E": -0.3, "F": -0.5, "G": -0.4}
SEASON_START = {2021: "2021-08-07", 2022: "2022-08-06"}       # 都是星期六
TEAMS = {2021: list("ABCDEF"), 2022: list("ABCDEG")}
POSTPONED = 2022_05_0                                          # 2022/23 第 5 轮第 1 场
EXAM_MATCH = 900
SCORE_COLS = {"home_goals", "away_goals", "result", "status", "known_at_utc"}


def round_robin(teams):
    """双循环赛程（轮转法）：返回 [(轮次, [(主, 客), ...]), ...]，前一半和后一半主客互换。"""
    t = list(teams)
    half = []
    for r in range(len(t) - 1):
        pairs = [(t[i], t[-1 - i]) for i in range(len(t) // 2)]
        half.append([(h, a) if r % 2 == 0 else (a, h) for h, a in pairs])
        t = [t[0], t[-1], *t[1:-1]]
    return list(enumerate(half + [[(a, h) for h, a in rnd] for rnd in half]))


def build_rows():
    rng = np.random.default_rng(2022)
    matches, stats, odds = [], [], []
    for season, teams in TEAMS.items():
        start = pd.Timestamp(SEASON_START[season])
        for rnd, games in round_robin(teams):
            day = (start + pd.Timedelta(days=7 * rnd)).strftime("%Y-%m-%d")
            for i, (h, a) in enumerate(games):
                mid = season * 1000 + rnd * 10 + i
                ko = [f"{day} 14:00", f"{day} 16:30", None][i]
                status = "postponed" if mid == POSTPONED else "finished"
                matches.append((mid, "E0", season, h, a, day, ko, status))
                if status == "finished":
                    d = STRENGTH[h] - STRENGTH[a]
                    hg, ag = int(rng.poisson(np.exp(0.35 + 0.5 * d))), int(rng.poisson(np.exp(0.1 - 0.5 * d)))
                    stats.append((mid, hg, ag, "H" if hg > ag else ("D" if hg == ag else "A")))
                if season == 2022 and i == 0:
                    odds.append((mid, 2.0, 3.5, 4.0))
    matches.append((EXAM_MATCH, "E0", 2024, "A", "B", "2024-08-17", "2024-08-17 14:00", "finished"))
    stats.append((EXAM_MATCH, 1, 0, "H"))
    return matches, stats, odds


MATCHES, STATS, ODDS = build_rows()
TARGET_IDS = sorted(m[0] for m in MATCHES if m[2] == 2022)


@pytest.fixture
def settings(tmp_path):
    s = copy.deepcopy(load_settings())
    s["models"]["min_matches"] = 20
    s["models"]["elo"]["burn_in_seasons"] = 0
    s["backtest"]["output_dir"] = str(tmp_path)      # 结果存到临时文件夹，不写项目的 data/
    return s


@pytest.fixture
def fs(settings):
    c = sqlite3.connect(":memory:")
    create_tables(c)
    c.executemany("insert into teams (team_id, country) values (?, 'England')", [(t,) for t in STRENGTH])
    c.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                  " status) values (?, ?, ?, ?, ?, ?, ?, ?)", MATCHES)
    c.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                  " values (?, 'football-data', ?, ?, ?)", STATS)
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away)"
                  " values (?, 'football-data', '平博 Pinnacle', '1x2', 'close', null, ?, ?, ?)", ODDS)
    c.commit()
    return FeatureStore(c, "开发", settings)


CONST = np.array([0.45, 0.27, 0.28])


class Const(Model):
    """老实的常数模型：不拟合任何东西，每场都预测 CONST。下面的作弊模型都从它改。"""

    name = "honest"

    def fit(self, history, cutoff):
        pass

    def predict(self, targets, cutoff):
        return [CONST] * len(targets)


class Spy(Model):
    """不预测，只记账：每次 fit / update 收到了什么历史、predict 收到了什么目标。按调用顺序记在 log 里。"""

    name = "spy"

    def __init__(self):
        self.log = []

    def fit(self, history, cutoff):
        self.log.append(("fit", cutoff, history.copy()))

    def update(self, new_rows, cutoff):
        self.log.append(("update", cutoff, new_rows.copy()))

    def predict(self, targets, cutoff):
        self.log.append(("predict", cutoff, targets.copy()))
        return [CONST] * len(targets)


# ======================= 不偷看未来 =======================
def test_history_never_contains_future_or_targets(fs):
    """模型每次收到的历史：每行 known_at_utc ≤ 截止时间；要预测的比赛一场都不在里面；目标表里没有比分、状态。
    增量取数（since）拼起来的历史，和每次在这个截止时间“从头全取一遍”完全一样（不漏、不多）。"""
    spy = Spy()
    run_backtest(fs, "E0", 2022, 2022, [spy], benchmark=False)
    seen: set[int] = set()
    n_checked = 0
    for kind, cutoff, df in spy.log:
        if kind in ("fit", "update"):
            assert (df["known_at_utc"] <= cutoff).all(), f"{kind} @ {cutoff} 收到了截止后才知道的比赛"
            if kind == "fit":
                full = fs.historical_matches(cutoff, "A版输入", competition=["E0"])
                assert set(df["match_id"]) == set(full["match_id"]), f"fit @ {cutoff} 的历史和从头取的不一样"
                seen = set(df["match_id"])
                n_checked += 1
            else:
                seen |= set(df["match_id"])
        else:
            assert not SCORE_COLS & set(df.columns), "目标表里不能有比分、状态"
            assert not seen & set(df["match_id"]), f"predict @ {cutoff}：要预测的比赛在历史里"
            assert (df["kickoff_earliest_utc"] > cutoff).all()
            assert EXAM_MATCH not in set(df["match_id"])
    assert n_checked >= 5


def test_cheat_model_using_wrong_usage_is_blocked(fs):
    """作弊模型 1：自己拿着取数层，用“A版输入”去读本场比分 → 取数层直接报错，回测停下。"""

    class CheatUsage(Const):
        name = "cheat_usage"

        def predict(self, targets, cutoff):
            for mid in targets["match_id"]:
                fs.match_result(int(mid), cutoff, "A版输入")
            return [CONST] * len(targets)

    with pytest.raises(UsageNotAllowedError):
        run_backtest(fs, "E0", 2022, 2022, [CheatUsage()], benchmark=False)


def test_cheat_model_peeking_before_kickoff_gets_nothing(fs):
    """作弊模型 2：换成“评估”用途读本场比分（用途允许），但截止时间是开球前 → 比分还不能知道，拿到空表。
    它想“知道答案就给答案 99%”，结果一场都没拿到，预测和老实的常数模型一模一样，分数也一样。"""

    class CheatPeek(Const):
        name = "cheat_peek"
        hits = 0

        def predict(self, targets, cutoff):
            out = []
            for mid in targets["match_id"]:
                r = fs.match_result(int(mid), cutoff, "评估")
                if r.empty:
                    out.append(CONST)
                else:
                    CheatPeek.hits += 1
                    out.append({"H": [0.98, 0.01, 0.01], "D": [0.01, 0.98, 0.01],
                                "A": [0.01, 0.01, 0.98]}[r.iloc[0]["result"]])
            return out

    res = run_backtest(fs, "E0", 2022, 2022, [CheatPeek(), Const()], benchmark=False)
    assert CheatPeek.hits == 0
    p = res.predictions.set_index(["match_id", "horizon", "model"])["log_loss"].unstack("model")
    pd.testing.assert_series_equal(p["cheat_peek"], p["honest"], check_names=False)


def test_exam_season_is_locked_in_dev_mode(fs):
    """开发模式下，回测范围碰到期末考试赛季（2024/25）→ 取数层报错（第 17 节三种运行模式）。"""
    with pytest.raises(SeasonLockedError):
        run_backtest(fs, "E0", 2024, 2024, [Spy()], benchmark=False)


# ======================= 周期重新拟合 =======================
def test_refit_weekly_and_only_on_data_before_cutoff(fs, settings):
    """每 refit_days（7）天才重新拟合一次；每次拟合的截止时间 ≤ 用它出预测的截止时间；
    用每次拟合的截止时间从头取历史、新造一个频率模型拟合，得到的概率和回测里的一模一样
    → 回测里的模型确实只用了那个截止时间之前的数据。"""
    res = run_backtest(fs, "E0", 2022, 2022, [build_model("frequency", settings)], benchmark=False)
    p = res.predictions
    assert (p["fit_cutoff_utc"] <= p["cutoff_utc"]).all()
    fits = sorted(p["fit_cutoff_utc"].unique())
    gaps = pd.Series(pd.to_datetime(fits)).diff().dropna()
    assert (gaps >= pd.Timedelta(days=settings["backtest"]["refit_days"])).all()
    assert len(fits) >= 5
    for fc, part in p.groupby("fit_cutoff_utc"):
        fresh = FrequencyModel(window_days=settings["models"]["frequency"]["window_days"], min_matches=20)
        fresh.fit(fs.historical_matches(fc, "A版输入", competition=["E0"]), fc)
        np.testing.assert_allclose(part[["p_home", "p_draw", "p_away"]].to_numpy(),
                                   np.tile(fresh.table["E0"], (len(part), 1)))


def test_elo_ratings_only_see_results_known_before_cutoff(fs, settings):
    """Elo 每场赛后加减分：回测结束时 Elo 处理过的比赛 = 最后一个截止时间之前能知道结果的比赛，
    不会多出“预测时还不知道”的比赛（最后几轮的结果在最后一次预测之后才知道，不该算进去）。"""
    elo = build_model("elo", settings)
    res = run_backtest(fs, "E0", 2022, 2022, [elo], benchmark=False)
    last = res.predictions["cutoff_utc"].max()
    known = set(fs.historical_matches(last, "A版输入", competition=["E0"])["match_id"])
    assert elo.processed <= known
    assert elo.processed >= {m for m in known if m < 2022000}   # 上赛季的全部用上了


# ======================= 结果表、汇总、保存 =======================
def test_results_table_summary_and_save(fs, settings, tmp_path):
    models = [build_model(n, settings) for n in ("frequency", "frequency_poisson", "elo")]
    res = run_backtest(fs, "E0", 2022, 2022, models)
    p = res.predictions
    mods = p[p["model"] != BENCHMARK]
    # 每场 × 3 个时点 × 3 个模型一行
    assert len(mods) == len(TARGET_IDS) * 3 * 3
    assert set(mods["match_id"]) == set(TARGET_IDS)
    assert mods["p_home"].notna().all()                          # 小联赛也都出了预测
    np.testing.assert_allclose(mods[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    # 截止时间：14:00 开球 → T-24 = 前一天 14:00；只有日期（2022-08-06，英国夏令时）→ 当天 00:00 英国 = 前一天 23:00 UTC，再往前 24 小时
    t24 = mods[mods["horizon"] == "T-24"].drop_duplicates("match_id").set_index("match_id")["cutoff_utc"]
    assert t24[2022000] == "2022-08-05 14:00:00"
    assert t24[2022002] == "2022-08-04 23:00:00"
    # 只有比分矩阵的模型才有大小球、双方进球概率
    assert mods.loc[mods["model"] == "frequency_poisson", "p_over25"].notna().all()
    assert mods.loc[mods["model"] == "frequency", "p_over25"].isna().all()
    # 改期的比赛：照样出了预测，但没有结果、不打分
    post = p[p["match_id"] == POSTPONED]
    assert len(post) > 0 and post["result"].isna().all() and post["log_loss"].isna().all()
    # 平博收盘：去水后的概率，时点“收盘”
    bench = p[p["model"] == BENCHMARK]
    assert (bench["horizon"] == "收盘").all() and len(bench) == len(ODDS)
    np.testing.assert_allclose(bench[["p_home", "p_draw", "p_away"]].to_numpy()[0],
                               devig_proportional(np.array([2.0, 3.5, 4.0])))
    # 汇总：每个模型在同一批（有结果的 29 场）比赛上比
    s = res.summary
    allrow = s[(s["level"] == "全部") & (s["horizon"] == "T-24") & (s["model"] != BENCHMARK)]
    assert set(allrow["model"]) == {"frequency", "frequency_poisson", "elo"}
    assert (allrow["n"] == len(TARGET_IDS) - 1).all()
    assert set(s.loc[s["level"] == "联赛×赛季", "season"]) == {"2022"}
    assert not res.summary_vs_benchmark.empty and not res.calibration.empty
    # 保存到 settings 里的 output_dir（这里是临时文件夹）
    out = save(res, "unit", settings)
    assert out.parent == tmp_path
    for f in ("predictions.csv", "summary.csv", "summary_vs_benchmark.csv", "calibration.csv", "run_info.json",
              "ratings_elo.csv"):
        assert (out / f).exists(), f
    back = cmp.load_predictions(out)
    assert len(back) == len(p)


# ======================= bootstrap 比较 =======================
def fake_scores(n_weeks=60, per_week=10, seed=0):
    """造两个模型的每场分数表（和 predictions.csv 的列一样）：每周 10 场，同一周的比赛有一个共同的“运气”。"""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2020-08-08", periods=n_weeks, freq="7D"), per_week)
    week_luck = np.repeat(rng.normal(0, 0.05, n_weeks), per_week)
    base = 1.0 + week_luck + rng.normal(0, 0.3, len(dates))
    return dates, base, rng


def frame(dates, scores_by_model):
    rows = []
    for model, s in scores_by_model.items():
        rows.append(pd.DataFrame({"match_id": np.arange(len(dates)), "match_date": dates.strftime("%Y-%m-%d"),
                                  "model": model, "horizon": "T-24", "log_loss": s, "brier": s / 2, "rps": s / 5}))
    return pd.concat(rows, ignore_index=True)


def test_bootstrap_identical_models_no_difference(settings):
    dates, base, _ = fake_scores()
    r = cmp.compare(frame(dates, {"a": base, "b": base.copy()}), "a", "b", settings=settings)
    assert (r["diff"] == 0).all() and (r["verdict"] == cmp.SAME).all()


def test_bootstrap_clearly_better_and_worse(settings):
    """b 每场都比 a 低 0.05 左右（再加一点噪音）→ 更准；反过来比 → 更差。"""
    dates, base, rng = fake_scores()
    better = base - 0.05 + rng.normal(0, 0.05, len(base))
    pred = frame(dates, {"a": base, "b": better})
    r = cmp.compare(pred, "a", "b", settings=settings)
    assert (r["verdict"] == cmp.BETTER).all()
    assert (r["ci_high"] < 0).all() and r.loc[r["primary"], "metric"].item() == "log_loss"
    assert (cmp.compare(pred, "b", "a", settings=settings)["verdict"] == cmp.WORSE).all()


def test_bootstrap_pure_noise_no_difference(settings):
    """b 和 a 只差纯噪音（平均差是 0）→ 没差别。"""
    dates, base, rng = fake_scores(seed=3)
    noisy = base + rng.normal(0, 0.2, len(base))
    r = cmp.compare(frame(dates, {"a": base, "b": noisy}), "a", "b", settings=settings)
    assert r.loc[r["metric"] == "log_loss", "verdict"].item() == cmp.SAME


def test_bootstrap_same_seed_same_interval():
    d = np.random.default_rng(5).normal(-0.01, 0.2, 300)
    blocks = np.repeat(np.arange(30), 10)
    assert cmp.bootstrap_mean_diff(d, blocks, 500, 0.95, 42) == cmp.bootstrap_mean_diff(d, blocks, 500, 0.95, 42)


def test_week_blocks_start_on_monday():
    """按周分块：周一到周日是同一块（块名 = 那周的周一）。周末和下周一的补赛不在一块。"""
    b = cmp.block_ids(pd.Series(["2024-01-08", "2024-01-13", "2024-01-14", "2024-01-15"]), "week")
    assert list(b) == ["2024-01-08", "2024-01-08", "2024-01-08", "2024-01-15"]
    assert len(set(cmp.block_ids(pd.Series(["2024-01-13", "2024-01-14"]), "day"))) == 2


# ======================= 两个标杆、分段统计 =======================
# 2022/23 每轮第 1 场有平博收盘（上面的 ODDS）。再给“多家平均”造几种情况，覆盖每个分段：
#   第 0–3 轮第 1 场：多家平均收盘 + 赛前都有 → 用收盘（有收盘就不用赛前）→ 平博收盘+平均收盘
#   第 6–9 轮第 1 场：只有多家平均赛前 → 赛前代替 → 平博收盘+平均赛前
#   第 0–3 轮第 2 场：只有多家平均收盘 → 只有平均收盘
#   第 6–9 轮第 2 场：只有多家平均赛前 → 只有平均赛前
#   其余：没有标杆（第 5 轮第 1 场改期，第 4 轮第 1 场只有平博收盘）
AVG_CLOSE = [2022_000 + r * 10 + i for r in range(4) for i in (0, 1)]
AVG_PRE = [2022_000 + r * 10 + i for r in (0, 1, 2, 3, 6, 7, 8, 9) for i in (0,)] + \
          [2022_000 + r * 10 + 1 for r in (6, 7, 8, 9)]


@pytest.fixture
def fs_two(fs):
    c = fs.conn
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away)"
                  " values (?, 'football-data', '多家平均', '1x2', 'close', null, 1.8, 3.6, 4.5)",
                  [(m,) for m in AVG_CLOSE])
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away)"
                  " values (?, 'football-data', '多家平均', '1x2', 'pre', null, 2.2, 3.3, 3.4)",
                  [(m,) for m in AVG_PRE])
    c.commit()
    fs.refresh()
    return fs


def test_two_benchmarks_and_periods(fs_two, settings):
    from src.evaluate import backtest as bt

    res = run_backtest(fs_two, "E0", 2022, 2022, [build_model("frequency", settings)])
    p = res.predictions
    avg = p[p["model"] == bt.BENCHMARK_AVG].set_index("match_id")
    # 有收盘平均就用收盘，没有才用赛前代替；两种时点分开写
    assert set(avg.index[avg["horizon"] == bt.CLOSE]) == set(AVG_CLOSE)
    assert set(avg.index[avg["horizon"] == bt.PRE_SUBSTITUTE]) == set(AVG_PRE) - set(AVG_CLOSE)
    np.testing.assert_allclose(avg.loc[2022_000, RESULT_COLS_T], devig_proportional(np.array([1.8, 3.6, 4.5])))
    np.testing.assert_allclose(avg.loc[2022_060, RESULT_COLS_T], devig_proportional(np.array([2.2, 3.3, 3.4])))
    # 每场的分段
    period = p.drop_duplicates("match_id").set_index("match_id")["benchmark_period"]
    assert period[2022_000] == bt.PERIOD_BOTH and period[2022_060] == bt.PERIOD_PIN_PRE
    assert period[2022_001] == bt.PERIOD_AVG and period[2022_061] == bt.PERIOD_PRE
    assert period[2022_040] == bt.PERIOD_PIN and period[2022_002] == bt.PERIOD_NONE
    # 分段汇总：每一段里模型和这一段的标杆在同一批比赛上（n 一样）
    s = res.summary_by_period
    top = s[(s["level"] == "全部") & s["horizon"].isin(["T-24", bt.CLOSE, bt.PRE_SUBSTITUTE])]
    for per, n in [(bt.PERIOD_BOTH, 4), (bt.PERIOD_PIN_PRE, 4), (bt.PERIOD_AVG, 4), (bt.PERIOD_PRE, 4)]:
        part = top[top["benchmark_period"] == per]
        assert (part["n"] == n).all() and len(part) == (3 if per in (bt.PERIOD_BOTH, bt.PERIOD_PIN_PRE) else 2), per
    assert res.info["多家平均用收盘的比赛数"] == 8 and res.info["多家平均用赛前代替的比赛数"] == 8
    # 默认比较：两个标杆各时点分开比模型；两个都有收盘时再比平博 vs 多家平均
    pairs = cmp.default_pairs(p)
    assert ("frequency", bt.BENCHMARK_AVG, "T-24", bt.PRE_SUBSTITUTE) in pairs
    assert ("frequency", bt.BENCHMARK_AVG, "T-24", bt.CLOSE) in pairs
    assert (bt.BENCHMARK_AVG, bt.BENCHMARK, bt.CLOSE, bt.CLOSE) in pairs
    assert all(base not in bt.BENCHMARKS for base, _, base_hz, _ in pairs if base_hz == "T-24")
    out = save(res, "two", settings)
    assert (out / "summary_by_benchmark_period.csv").exists()


def test_benchmark_devig_method_setting(fs_two, settings):
    """标杆的去水方法按 backtest.benchmark_devig_method；不影响 calc.devig_method。"""
    from src.common.odds import devig_shin
    from src.evaluate import backtest as bt

    settings["backtest"]["benchmark_devig_method"] = "shin"
    res = run_backtest(fs_two, "E0", 2022, 2022, [build_model("frequency", settings)])
    pin = res.predictions[res.predictions["model"] == bt.BENCHMARK]
    np.testing.assert_allclose(pin[RESULT_COLS_T].to_numpy()[0], devig_shin(np.array([2.0, 3.5, 4.0])))
    assert res.info["标杆去水方法"] == "shin"


RESULT_COLS_T = ["p_home", "p_draw", "p_away"]


def test_week_blocks_give_wider_interval_when_weeks_share_luck():
    """同一周的比赛分数差有共同的波动（例如某周冷门特别多）时，按单场抽样会把区间算得太窄；按周抽才对。"""
    rng = np.random.default_rng(11)
    weeks = np.repeat(np.arange(80), 10)
    d = np.repeat(rng.normal(0, 0.1, 80), 10) + rng.normal(0, 0.05, 800)
    _, lo_w, hi_w = cmp.bootstrap_mean_diff(d, weeks, 2000, 0.95, 1)
    _, lo_m, hi_m = cmp.bootstrap_mean_diff(d, np.arange(800), 2000, 0.95, 1)
    assert hi_w - lo_w > 1.5 * (hi_m - lo_m)
