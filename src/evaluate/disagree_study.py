"""模型和赔率意见不一样时谁准（需求文档第 4 节“模型和赔率意见不一样时谁准：怎么比”，跑之前写下，不临时加）。

一、要回答什么
  上线的 gb_l7（A 版，不用赔率）和赛前赔率（去水后）看法不一样的比赛里，谁更准？
  分歧集中在哪类比赛？有没有哪一类比赛模型反而比赔率准？只是研究、只描述，不改任何上线的东西。

二、数据从哪来
  不重新跑回测。直接读 W9 那次开发回测（w9_gb_l7）存下的文件（从工作树 W9 原样复制到 data/backtest/w9_gb_l7/，
  sha256 见结果的 run_info.json；路径写在 settings.yaml 的 disagree_study.base_dir）：
  - predictions.csv：每场 gb_l7 在 T-24 的胜平负概率、真实结果；还有平博收盘（去水后，只拿来看看）。
  - fusion_odds.csv：同一次回测经取数层（src/features，用途「融合」）按“本场”取的 T-24 能看到的赛前赔率，已经按比例去水。
    有平博赛前用平博，没有用多家平均赛前（和上线的融合同一条规则）。
  这些都是“截止时间之前”算出来的，这里只是把它们放在一起比，不再取任何新数据。
  用之前核对三件事：每场都是开发赛季（season_role），回测时的 models 设置和现在一样，文件的 sha256 记下来。

三、怎么比
  每场：模型的 log loss − 赔率的 log loss（负数 = 模型更准）。
  每组：这个差的平均，按周成块 bootstrap 2000 次得到 95% 区间（和 compare.py 同一个种子、同一个算法）。
  分歧 = 主胜、平、客胜三个概率里模型和赔率差得最多的那个（绝对值）；分歧 ≥ 3% 叫“意见不一样”。
  分类（联赛、强弱差距、平局概率、赛季阶段、模型多给哪项、热门冷门、赔率来源、两段）都在“意见不一样”的比赛里比。

四、为什么还要多重比较校正
  一共比 27 个区间。就算两边其实一样准，每个 95% 区间也有 5% 的机会碰巧不跨 0，27 个里平均会有 1 个多。
  所以“模型更准”的组还要再过两道检查才写“比较可信”：
  （1）Bonferroni 校正：区间放宽到 1 − 5% ÷ 27 的置信水平（约 3.1 倍标准差）还整个在 0 以下；
  （2）两段（2005–2013、2014–2023）各自的平均差都是负数（不是只在一段里碰巧）。
  否则写“可能是碰巧”。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.disagree_study
结果放在 settings.yaml 的 disagree_study.output_dir（data/backtest/disagree_study/，不进 Git）。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.evaluate import metrics
from src.evaluate.compare import block_ids
from src.evaluate.market_checks import ratio_ci

P = ["p_home", "p_draw", "p_away"]      # 模型（gb_l7）的主胜、平、客胜概率
Q = ["q_home", "q_draw", "q_away"]      # 赛前赔率（去水后）
C = ["c_home", "c_draw", "c_away"]      # 平博收盘（去水后，只看看）

MODEL_BETTER, ODDS_BETTER, UNCLEAR = "模型更准", "赔率更准", "看不出"
SURE, MAYBE = "比较可信", "可能是碰巧"

# 8 类分组：（类名，列名）。顺序就是报告里的顺序
DIMENSIONS = [("联赛", "league"), ("两队强弱差距", "strength"), ("平局概率", "draw_level"), ("赛季阶段", "phase"),
              ("模型比赔率多给哪项", "side"), ("热门冷门", "favdir"), ("赔率来源", "source"), ("两段", "period")]


# ======================= 读数据、核对 =======================
def sha256(path: Path) -> str:
    """文件的指纹：内容变了一点，这串字就完全不一样。记下来，以后能核对用的是不是同一份文件。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_models_settings(run_info: dict, settings: dict) -> list[str]:
    """回测时记下的 models 设置，和现在 settings.yaml 的 models 每一项都要一样（不一样说明模型改过，旧结果不代表现在的模型）。

    只比回测时就有的项：之后新加的项（例如次级联赛用的 gbdt_league_groups）回测时还没有，不影响五大联赛。
    返回比过的项名；有不一样的就停下报错。
    """
    old = run_info.get("设置", {}).get("models")
    if not old:
        raise ValueError("回测的 run_info.json 里没有 models 设置，没法核对模型有没有改过")
    cur = settings["models"]
    bad = [k for k in old if old[k] != cur.get(k)]
    if bad:
        raise ValueError(f"回测时的 models 设置和现在不一样：{bad}，这次回测不代表现在的模型")
    return list(old)


def check_dev_only(seasons, settings: dict) -> list[int]:
    """每个赛季都要是开发赛季，有一个不是（例如期末考试赛季）就停下。返回用到的赛季。"""
    ss = sorted({int(s) for s in seasons})
    bad = [s for s in ss if season_role(s, settings) != "dev"]
    if bad:
        raise ValueError(f"这些赛季不是开发赛季，不能用：{bad}")
    return ss


def season_phase(m: pd.DataFrame, rounds: int) -> pd.Series:
    """每场是这个联赛这个赛季的早期 / 中段 / 后期（index = match_id）。

    和 early_rounds.py 一样按开球时间排（早年没有开球时间用比赛日期，同一时间按比赛编号），不用网站上的“第几轮”。
    前 rounds 轮 = 前（球队数 ÷ 2 × rounds）场 → 早期；最后同样多场 → 后期；中间 → 中段。
    改期的比赛按真正踢的时间排（例如推迟到赛季末踢的，算后期）。
    """
    m = m.drop_duplicates("match_id")[["match_id", "competition", "season", "match_date", "kickoff_utc",
                                       "home_team", "away_team"]].copy()
    ko = pd.to_datetime(m["kickoff_utc"], errors="coerce")
    m["order_time"] = ko.fillna(pd.to_datetime(m["match_date"]))
    m = m.sort_values(["competition", "season", "order_time", "match_id"])
    g = m.groupby(["competition", "season"])
    m["nth"] = g.cumcount()                                    # 这个赛季第几场（从 0 数）
    m["n_all"] = g["match_id"].transform("size")               # 这个赛季一共几场
    teams = m.melt(id_vars=["competition", "season"], value_vars=["home_team", "away_team"])
    n_teams = teams.groupby(["competition", "season"])["value"].nunique().rename("n_teams")
    m = m.join(n_teams, on=["competition", "season"])
    k = m["n_teams"] // 2 * rounds
    phase = np.where(m["nth"] < k, "早期", np.where(m["nth"] >= m["n_all"] - k, "后期", "中段"))
    return pd.Series(phase, index=m["match_id"].to_numpy())


def renormalize(x: pd.DataFrame, tol: float) -> pd.DataFrame:
    """每行三项除以它们的和（空值的行不动）。先查：和 1 差超过 tol（设置的 sum_tol）就报错，不悄悄改大的错。"""
    tot = x.sum(axis=1, min_count=len(x.columns))
    bad = (tot - 1).abs() > tol
    if bad.any():
        raise ValueError(f"{int(bad.sum())} 场的概率加起来离 1 太远（最大差 {float((tot - 1).abs().max()):.2e}）")
    return x.div(tot, axis=0)


def build_frame(pred: pd.DataFrame, odds: pd.DataFrame, cfg: dict, settings: dict) -> tuple[pd.DataFrame, dict]:
    """把模型、赛前赔率、平博收盘接成每场一行，算好每场的 log loss。返回（表，场数统计）。"""
    m = pred[(pred["model"] == cfg["model"]) & (pred["horizon"] == cfg["horizon"])].copy()
    if m.empty:
        raise ValueError(f"predictions.csv 里没有 {cfg['model']}@{cfg['horizon']}")
    if m["match_id"].duplicated().any():
        raise ValueError("同一场比赛有两行模型预测，不知道用哪个")
    seasons = check_dev_only(pred["season"], settings)          # 整个文件都查（不只模型那几行）
    m["phase"] = m["match_id"].map(season_phase(m, int(cfg["phase_rounds"])))  # 用全部比赛排轮次，再去掉没赔率的
    keep = ["match_id", "competition", "season", "match_date", "result", "phase", *P,
            *(["log_loss"] if "log_loss" in m else [])]
    m = m[keep]
    o = odds[["match_id", *cfg["odds_cols"], cfg["pinnacle_col"]]].copy()
    o.columns = ["match_id", *Q, "q_pinnacle"]
    c = pred[pred["model"] == cfg["close_model"]][["match_id", *P]].drop_duplicates("match_id")
    c.columns = ["match_id", *C]
    df = m.merge(o, on="match_id", how="left").merge(c, on="match_id", how="left")
    no_odds = df[Q].isna().any(axis=1)
    df = df[~no_odds].reset_index(drop=True)
    for cols in (P, Q, C):                                     # csv 里只存了 6 位小数，三项加起来和 1 差一点点 → 补成正好 1
        df[cols] = renormalize(df[cols], float(cfg["sum_tol"]))
    y = metrics.encode_results(df["result"])
    df["ll_m"] = metrics.log_loss_per_match(df[P].to_numpy(float), y)
    df["ll_q"] = metrics.log_loss_per_match(df[Q].to_numpy(float), y)
    df["diff"] = df["ll_m"] - df["ll_q"]                         # 负数 = 模型更准
    if "log_loss" in df:                                       # 核对：重算的模型 log loss 和回测记下的一样
        worst = float((df["ll_m"] - df["log_loss"]).abs().max())
        if worst > float(cfg["ll_tol"]):
            raise ValueError(f"重算的模型 log loss 和回测记下的不一样，最大差 {worst:.2e}")
        counts_check = worst
    else:
        counts_check = None
    has_c = df[C].notna().all(axis=1).to_numpy()
    df["ll_c"] = np.nan
    if has_c.any():
        df.loc[has_c, "ll_c"] = metrics.log_loss_per_match(df.loc[has_c, C].to_numpy(float), y[has_c])
    df["is_draw"] = (y == 1).astype(float)
    counts = {"模型预测场数": int(len(m)), "没有赛前赔率的场数": int(no_odds.sum()), "比较的场数": int(len(df)),
              "有平博收盘的场数": int(has_c.sum()), "赛季": [seasons[0], seasons[-1]],
              "重算的模型 log loss 和回测记下的最大差": counts_check}
    return df, counts


# ======================= 分组 =======================
def _bin_labels(edges: list[float]) -> list[str]:
    """分界 [0.03, 0.06] → ['3% 以内', '3%–6%', '6% 以上']。"""
    pct = [f"{e * 100:g}%" for e in edges]
    return [f"{pct[0]} 以内", *[f"{a}–{b}" for a, b in zip(pct, pct[1:])], f"{pct[-1]} 以上"]


def _cut(x: np.ndarray, edges: list[float], labels: list[str]) -> np.ndarray:
    """按分界分档：x < 第一个分界 → 第一档 …… ≥ 最后一个分界 → 最后一档。"""
    if len(labels) != len(edges) + 1:                          # n 个分界分出 n + 1 档，名字个数要对上
        raise ValueError(f"分界 {list(edges)} 分出 {len(edges) + 1} 档，名字却有 {len(labels)} 个：{list(labels)}")
    return np.asarray(labels, dtype=object)[np.searchsorted(np.asarray(edges, float), x, side="right")]


def add_groups(df: pd.DataFrame, cfg: dict, settings: dict) -> pd.DataFrame:
    """每场加上分歧大小和 8 类分组的标签（都只用开球前就知道的东西：两边的概率、联赛、赛季、开球时间）。"""
    df = df.copy()
    pm, pq = df[P].to_numpy(float), df[Q].to_numpy(float)
    d = pm - pq
    df["gap"] = np.abs(d).max(axis=1)
    df["gap_group"] = _cut(df["gap"].to_numpy(), cfg["gap_bins"], _bin_labels(cfg["gap_bins"]))
    fav = pq.max(axis=1)
    df["fav_q"] = fav
    df["strength"] = _cut(fav, cfg["fav_bins"], cfg["fav_labels"])         # 档的名字在设置里，个数要和分界对上
    df["draw_level"] = _cut(pq[:, 1], cfg["draw_bins"], cfg["draw_labels"])
    df["side"] = np.asarray(["模型更看好主胜", "模型更看好平局", "模型更看好客胜"], dtype=object)[d.argmax(axis=1)]
    fav_i = pq.argmax(axis=1)                                   # 赔率里的热门（最大的那项）
    df["favdir"] = np.where(d[np.arange(len(d)), fav_i] > 0, "模型更看好热门", "模型更看好冷门")
    df["source"] = np.where(df["q_pinnacle"] == 1, "平博赛前", "多家平均赛前")
    split = int(cfg["period_split"])
    first = int(df["season"].min())                            # 两段的名字：最早的赛季–分界前一季、分界–开发赛季最后一季
    df["period"] = np.where(df["season"] < split, f"{first}–{split - 1}", f"{split}–{settings['seasons']['dev_last']}")
    names = {**settings["leagues"]["top"], **settings["leagues"].get("second", {})}
    df["league"] = df["competition"].map(names).fillna(df["competition"])
    return df


# ======================= 区间 =======================
def boot_means(diff, blocks, n_boot: int, seed: int) -> np.ndarray:
    """按周成块 bootstrap 的 n_boot 个平均差（和 compare.bootstrap_mean_diff 同一个算法、同一个种子，只是把每次的平均都留下来，
    这样还能算标准差，用来做多重比较校正）。"""
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        raise ValueError("没有可比较的比赛")
    _, inv = np.unique(np.asarray(blocks), return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    k = len(sums)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    step = max(1, 2_000_000 // k)
    for i in range(0, n_boot, step):
        idx = rng.integers(0, k, size=(min(step, n_boot - i), k))
        means[i:i + len(idx)] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return means


def verdict(lo: float, hi: float) -> str:
    """差 = 模型 − 赔率：区间整个在 0 以下 → 模型更准；整个在 0 以上 → 赔率更准；跨过 0 → 看不出。"""
    if hi < 0:
        return MODEL_BETTER
    if lo > 0:
        return ODDS_BETTER
    return UNCLEAR


def bonferroni_z(n_tests: int, confidence: float) -> float:
    """多重比较校正后用几倍标准差：n_tests 个区间一共只允许 (1 − confidence) 的机会碰巧出错，
    每个区间分到 (1 − confidence) ÷ n_tests，两边各一半。27 个、95% → 约 3.11 倍。"""
    alpha = (1 - confidence) / n_tests
    return NormalDist().inv_cdf(1 - alpha / 2)


def credibility(row: dict) -> str:
    """“模型更准”的组再过两道检查（需求文档写下的规则）：
    （1）校正后的区间（平均 ± z × 标准差）整个在 0 以下；（2）这组里有比赛的每一段，平均差都是负数。
    都过了 → 比较可信；否则 → 可能是碰巧。不是“模型更准”的组不用判断（空）。"""
    if row["结论"] != MODEL_BETTER:
        return ""
    periods = [v for k, v in row.items() if k.startswith("平均差_") and pd.notna(v)]
    ok = row["校正后上限"] < 0 and len(periods) > 0 and all(v < 0 for v in periods)
    return SURE if ok else MAYBE


def compare_row(sub: pd.DataFrame, settings: dict, z: float, periods: list[str]) -> dict:
    """一组比赛：场数、两边平均 log loss、平均差和 95% 区间、校正后的区间、模型更准的场数比例、两段各自的平均差、平局概率。"""
    c = settings["compare"]
    means = boot_means(sub["diff"], block_ids(sub["match_date"], c["block"]), c["n_boot"], c["seed"])
    alpha = (1 - c["confidence"]) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    mean, sd = float(sub["diff"].mean()), float(means.std(ddof=1))
    row = {"场数": len(sub), "周数": int(pd.Series(block_ids(sub["match_date"], c["block"])).nunique()),
           "模型 log loss": float(sub["ll_m"].mean()), "赔率 log loss": float(sub["ll_q"].mean()),
           "差": mean, "下限": float(lo), "上限": float(hi), "结论": verdict(float(lo), float(hi)),
           "校正后下限": mean - z * sd, "校正后上限": mean + z * sd,
           "模型更准的场数比例": float((sub["diff"] < 0).mean()), "平均分歧": float(sub["gap"].mean())}
    for p in periods:
        s = sub[sub["period"] == p]
        row[f"场数_{p}"] = len(s)
        row[f"平均差_{p}"] = float(s["diff"].mean()) if len(s) else np.nan
    row["模型平局概率"] = float(sub["p_draw"].mean())
    row["赔率平局概率"] = float(sub["q_draw"].mean())
    row["实际平局比例"] = float(sub["is_draw"].mean())
    cl = sub.dropna(subset=["ll_c"])
    row["有收盘的场数"] = len(cl)
    row["模型比收盘差"] = float((cl["ll_m"] - cl["ll_c"]).mean()) if len(cl) else np.nan
    row["赔率比收盘差"] = float((cl["ll_q"] - cl["ll_c"]).mean()) if len(cl) else np.nan
    row["可信程度"] = ""
    row["可信程度"] = credibility(row)
    return row


# ======================= 三张表 =======================
def level_order(df: pd.DataFrame, col: str, cfg: dict) -> list[str]:
    """每类里各组的先后（报告按这个顺序排）；表里没有的组跳过。"""
    fixed = {"gap_group": _bin_labels(cfg["gap_bins"]), "strength": list(cfg["fav_labels"]),
             "draw_level": list(cfg["draw_labels"]), "phase": ["早期", "中段", "后期"],
             "side": ["模型更看好主胜", "模型更看好平局", "模型更看好客胜"], "favdir": ["模型更看好热门", "模型更看好冷门"],
             "source": ["平博赛前", "多家平均赛前"]}
    have = list(pd.unique(df[col]))
    if col in fixed:
        return [v for v in fixed[col] if v in have]
    return sorted(have)


def groups_table(df: pd.DataFrame, cfg: dict, settings: dict) -> tuple[pd.DataFrame, float]:
    """每组一行的比较表。返回（表，校正用的 z）。

    范围：全部比赛（只作参照）；全部比赛按分歧四档（算进 27 个）；意见不一样（分歧 ≥ 3%）的全部（参照）和 8 类各组（算进 27 个）；
    分歧 ≥ 6% 的全部和 8 类各组（只看看，不算进 27 个）。
    """
    z = bonferroni_z(int(cfg["n_tests"]), settings["compare"]["confidence"])
    periods = level_order(df, "period", cfg)
    rows = []

    def add(scope: str, dim: str, level: str, sub: pd.DataFrame, counted: bool) -> None:
        if len(sub) == 0:
            return
        r = compare_row(sub, settings, z, periods)
        rows.append({"范围": scope, "分类": dim, "组": level, "算进多重比较": counted, **r})

    add("全部比赛", "全部", "全部", df, False)
    for lv in level_order(df, "gap_group", cfg):
        add("全部比赛", "分歧大小", lv, df[df["gap_group"] == lv], True)
    for scope, lo, counted in [(f"意见不一样（分歧 ≥ {cfg['disagree_min']:.0%}）", cfg["disagree_min"], True),
                               (f"分歧 ≥ {cfg['disagree_strong']:.0%}（只看看）", cfg["disagree_strong"], False)]:
        part = df[df["gap"] >= lo]
        add(scope, "全部", "全部", part, False)
        for dim, col in DIMENSIONS:
            for lv in level_order(part, col, cfg):
                add(scope, dim, lv, part[part[col] == lv], counted)
    out = pd.DataFrame(rows)
    return out, z


def concentration_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """分歧集中在哪（描述，不算区间）：每类每组全部比赛的场数、占全部的比例、分歧 ≥ 3% / ≥ 6% 的比例、平均分歧。"""
    lo, hi = cfg["disagree_min"], cfg["disagree_strong"]
    rows = [{"分类": "全部", "组": "全部", "场数": len(df), "占全部": 1.0,
             f"分歧≥{lo:.0%}的比例": float((df["gap"] >= lo).mean()), f"分歧≥{hi:.0%}的比例": float((df["gap"] >= hi).mean()),
             "平均分歧": float(df["gap"].mean()), f"意见不一样的场数里占": 1.0}]
    n_dis = int((df["gap"] >= lo).sum())
    for dim, col in DIMENSIONS:
        for lv in level_order(df, col, cfg):
            s = df[df[col] == lv]
            rows.append({"分类": dim, "组": lv, "场数": len(s), "占全部": len(s) / len(df),
                         f"分歧≥{lo:.0%}的比例": float((s["gap"] >= lo).mean()),
                         f"分歧≥{hi:.0%}的比例": float((s["gap"] >= hi).mean()), "平均分歧": float(s["gap"].mean()),
                         f"意见不一样的场数里占": float((s["gap"] >= lo).sum() / n_dis) if n_dis else np.nan})
    return pd.DataFrame(rows)


def follow_table(df: pd.DataFrame, cfg: dict, settings: dict) -> pd.DataFrame:
    """只看看：赔率会不会往 gb_l7 那边走（第 5 节检验三的跟随比例 β）。

    只用赛前赔率是平博、平博收盘也有的比赛（同一家公司，m 只是这一家自己的变化）。
    每场 d = 模型 − 赛前，m = 收盘 − 赛前（三个概率）。β = Σ(d·m) ÷ Σ(d·d)：0 = 收盘的变化和模型无关；1 = 正好走到模型的位置。
    另外：分歧 ≥ 3%、赔率变了的比赛里，往模型方向走（d·m > 0）的比例。
    """
    g0 = df[(df["q_pinnacle"] == 1) & df[C].notna().all(axis=1)].copy()
    if g0.empty:
        return pd.DataFrame()
    d = g0[P].to_numpy(float) - g0[Q].to_numpy(float)
    m = g0[C].to_numpy(float) - g0[Q].to_numpy(float)
    g0["dm"], g0["dd"] = (d * m).sum(axis=1), (d * d).sum(axis=1)
    g0["moved"] = np.abs(m).sum(axis=1) > 1e-9
    lo = cfg["disagree_min"]
    parts = [("全部", g0), (f"分歧 ≥ {lo:.0%}", g0[g0["gap"] >= lo])]
    parts += [(lv, g0[g0["gap_group"] == lv]) for lv in level_order(g0, "gap_group", cfg)]
    rows = []
    for name, g in parts:
        if g.empty:
            continue
        beta, blo, bhi = ratio_ci(g["dm"], g["dd"], g["match_date"], settings)
        row = {"组": name, "场数": len(g), "跟随比例β": beta, "下限": blo, "上限": bhi,
               "结论": "赔率往模型那边走" if blo > 0 else ("赔率往反方向走" if bhi < 0 else "看不出")}
        big = g[(g["gap"] >= lo) & g["moved"]]
        if len(big):
            share, slo, shi = ratio_ci((big["dm"] > 0).astype(float), np.ones(len(big)), big["match_date"], settings)
            row.update({"分歧够大且赔率变了的场数": len(big), "往模型方向走的比例": share, "比例下限": slo, "比例上限": shi})
        rows.append(row)
    return pd.DataFrame(rows)


# ======================= 报告 =======================
def _f(x, nd: int = 4, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def _md(rows: list[list[str]], head: list[str]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(res: dict, out_dir: Path) -> Path:
    """把结果写成一份中文 md 报告（给人看；数都在同目录的 csv 里）。"""
    g, conc, fol, info = res["groups"], res["concentration"], res["follow"], res["info"]
    periods = info["两段"]
    L = [f"# 模型（{info['模型']}）和赔率意见不一样时谁准", "",
         f"规则见需求文档第 4 节“模型和赔率意见不一样时谁准：怎么比”（跑之前写下）。生成时间 {info['生成时间']}。", "",
         f"- 数据：{info['数据']['predictions.csv']['路径']}（开发回测，T-24），赛季 {info['场数']['赛季'][0]}–{info['场数']['赛季'][1]}，"
         f"全部是开发赛季；比较 {info['场数']['比较的场数']:,} 场（{info['场数']['没有赛前赔率的场数']} 场没有赛前赔率，不算）。",
         "- 差 = 模型 log loss − 赔率 log loss，负数 = 模型更准；区间 = 按周成块 bootstrap 95%。",
         f"- 多重比较：{info['算进多重比较的区间数']} 个区间，校正后用 ±{info['校正倍数z']:.2f} 倍 bootstrap 标准差。", ""]
    better = g[(g["结论"] == MODEL_BETTER) & g["算进多重比较"]]
    if better.empty:
        L += ["**一句话：算进多重比较的组里，没有一组模型比赔率准。**", ""]
    else:
        items = [f"{r['分类']}·{r['组']}（{r['可信程度']}）" for _, r in better.iterrows()]
        L += [f"**一句话：模型比赔率准的组：{'、'.join(items)}；其余都是赔率更准或看不出。**", ""]

    def table(sub: pd.DataFrame, with_period: bool) -> str:
        head = ["范围 / 分类", "组", "场数", "模型", "赔率", "差", "95% 区间", "结论", "可信程度", "模型更准的场数比例"]
        if with_period:
            head += [f"差（{p}）" for p in periods]
        rows = []
        for _, r in sub.iterrows():
            row = [r["分类"] if r["分类"] != "全部" else r["范围"], r["组"], f"{r['场数']:,}", _f(r["模型 log loss"], sign=False),
                   _f(r["赔率 log loss"], sign=False), _f(r["差"]), f"{_f(r['下限'])} 到 {_f(r['上限'])}", r["结论"],
                   r["可信程度"] or "", _pct(r["模型更准的场数比例"])]
            if with_period:
                row += [f"{_f(r.get(f'平均差_{p}'))}（{int(r[f'场数_{p}']):,} 场）" if r[f"场数_{p}"] else "—"
                        for p in periods]
            rows.append(row)
        return _md(rows, head)

    L += ["## 1. 全部比赛和分歧大小", "", table(g[g["范围"] == "全部比赛"], True), ""]
    for scope in [s for s in g["范围"].unique() if s != "全部比赛"]:
        L += [f"## {'2' if '只看看' not in scope else '3'}. {scope}：按类分", "", table(g[g["范围"] == scope], True), ""]
    L += ["## 4. 分歧集中在哪（全部比赛，描述）", ""]
    cols = [c for c in conc.columns if c not in ("分类", "组")]
    L += [_md([[r["分类"], r["组"], f"{r['场数']:,}"] + [_pct(r[c]) for c in cols[1:]]
               for _, r in conc.iterrows()], ["分类", "组", "场数", *cols[1:]]), ""]
    dis = g[g["范围"].str.startswith("意见不一样")]
    L += ["## 5. 平局概率：模型、赔率给的平均和实际（意见不一样的比赛，只看看）", "",
          _md([[r["分类"], r["组"], f"{r['场数']:,}", _pct(r["模型平局概率"]), _pct(r["赔率平局概率"]), _pct(r["实际平局比例"])]
               for _, r in dis.iterrows()], ["分类", "组", "场数", "模型给平局", "赔率给平局", "实际平局"]), ""]
    L += ["## 6. 离平博收盘多远（有收盘的比赛，只看看；正数 = 比收盘差）", "",
          _md([[r["范围"] if r["分类"] == "全部" else r["分类"], r["组"], f"{r['有收盘的场数']:,}", _f(r["模型比收盘差"]),
                _f(r["赔率比收盘差"])] for _, r in g[g["分类"].isin(["全部", "分歧大小"])].iterrows()],
              ["范围 / 分类", "组", "有收盘的场数", "模型 − 收盘", "赛前赔率 − 收盘"]), ""]
    if len(fol):
        L += ["## 7. 赔率会不会往模型那边走（平博赛前 → 平博收盘，只看看）", "",
              _md([[r["组"], f"{r['场数']:,}", _f(r["跟随比例β"]), f"{_f(r['下限'])} 到 {_f(r['上限'])}", r["结论"],
                    _pct(r.get("往模型方向走的比例")), f"{_pct(r.get('比例下限'))} 到 {_pct(r.get('比例上限'))}"]
                   for _, r in fol.iterrows()],
                  ["组", "场数", "跟随比例 β", "95% 区间", "结论", "往模型方向走的比例", "区间"]), ""]
    L += ["## 注意", "",
          "- 各组共用同一批比赛（一场比赛同时在联赛、强弱、平局等几类里），区间不是互相独立的证据。",
          "- log loss 更小不等于能赚钱：赔率有抽水，这里没算任何下注。结果不改任何上线的东西。", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ======================= 主流程 =======================
def run(pred: pd.DataFrame, odds: pd.DataFrame, settings: dict) -> dict:
    """算三张表（不写文件）。pred = predictions.csv，odds = fusion_odds.csv。"""
    cfg = settings["disagree_study"]
    df, counts = build_frame(pred, odds, cfg, settings)
    df = add_groups(df, cfg, settings)
    groups, z = groups_table(df, cfg, settings)
    n_counted = int(groups["算进多重比较"].sum())
    info = {"模型": cfg["model"], "时点": cfg["horizon"], "场数": counts, "校正倍数z": z,
            "算进多重比较的区间数": n_counted, "设置里写的区间数": int(cfg["n_tests"]),
            "两段": level_order(df, "period", cfg)}
    if n_counted != int(cfg["n_tests"]):
        info["提醒"] = f"实际算了 {n_counted} 个区间，和设置里写的 {cfg['n_tests']} 个不一样（有的组没有比赛）；校正仍按设置里的数"
    return {"frame": df, "groups": groups, "concentration": concentration_table(df, cfg),
            "follow": follow_table(df, cfg, settings), "info": info}


def main(settings: dict | None = None) -> None:
    """settings 不给就读 settings.yaml（测试时传进来）。相对路径按项目文件夹算。"""
    t0 = time.time()
    settings = settings or load_settings()
    cfg = settings["disagree_study"]
    base = resolve_path(cfg["base_dir"])
    files = {"predictions.csv": base / "predictions.csv", "fusion_odds.csv": base / "fusion_odds.csv",
             "run_info.json": base / "run_info.json"}
    for name, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"找不到 {f}（settings.yaml 的 disagree_study.base_dir）")
    run_info = json.loads(files["run_info.json"].read_text(encoding="utf-8"))
    if run_info.get("运行模式") != "dev":                         # 只用开发模式跑的回测
        raise ValueError(f"回测不是开发模式跑的：{run_info.get('运行模式')}")
    checked = check_models_settings(run_info, settings)
    pred = pd.read_csv(files["predictions.csv"], encoding="utf-8-sig", low_memory=False)
    odds = pd.read_csv(files["fusion_odds.csv"], encoding="utf-8-sig")
    res = run(pred, odds, settings)
    out = resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res["info"].update({
        "生成时间": time.strftime("%Y-%m-%d %H:%M"),
        "数据": {k: {"路径": str(v), "大小": v.stat().st_size, "sha256": sha256(v)} for k, v in files.items()},
        "核对过的 models 设置": checked, "回测的运行模式": run_info.get("运行模式"),
        "设置": {"disagree_study": cfg, "compare": settings["compare"]}})
    for name in ("groups", "concentration", "follow"):
        res[name].to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    res["info"]["耗时秒"] = round(time.time() - t0, 1)
    (out / "run_info.json").write_text(json.dumps(res["info"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    path = write_report(res, out)
    g = res["groups"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(g[["范围", "分类", "组", "场数", "差", "下限", "上限", "结论", "可信程度"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"报告：{path}")


if __name__ == "__main__":
    main()
