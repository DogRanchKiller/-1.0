"""选方向研究（需求文档第 4 节“选方向研究怎么做”“事先定的比较”，跑之前写下，跑完不改）。

问的是：每场选方向（主要是让球主盘口选哪边，其次胜平负选哪项）能不能比“照赔率选”更常选对？能多多少、稳不稳？
实时伤停来源值不值得接？只是研究：不改模型、fusion.w、倾向门槛、信心等级、网站、预测存档、定时任务。

做法（细节见需求文档）：
  1. 读两次开发回测（都不重跑，只读）：w9_gb_l7（gb_l7 逐场概率 + T-24 赛前赔率 fusion_odds.csv），
     伤停研究 I7 的 s7_eval（gb_l7_r、gb_inj 逐场概率，用途「研究实验」）。先按赛季筛，考试赛季一行都不进计算；
  2. 五大联赛 2014–2023、有让球主盘口的比赛；每个来源（赔率、gb_l7、gb_l7_r、gb_inj）的比分矩阵
     都用它自己的主胜、平、客胜、大 2.5 球四个概率反推（fit_market_dc，和第 5 节“市场比分矩阵”一样）；
  3. 五个候选 A–E 的参数只在 2014–2019 上选（选参数的函数拿不到 2020–2023 的比赛），让球、胜平负、大小球各自选；
  4. 2020–2023 检验：候选命中率 − 赔率命中率，按周成块 bootstrap，三条都过才写“过了”；另外报告校准、log loss、分组（只是看看）。
结果写到 settings.yaml direction_study.output_dir：summary.json、report.md、picks.csv、grid.csv、calibration.csv、groups.csv、run_info.json。

运行：.venv\\Scripts\\python -m src.evaluate.direction_study
      数据不在程序所在的文件夹时（例如在工作树里跑）：加 --root <项目文件夹>，相对路径都从那里算。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.common.markets import dc_matrices, dc_rho_bounds, fit_market_dc
from src.common.settlement import settle_asian_handicap, settle_margin, settle_over_under
from src.evaluate.compare import block_ids
from src.evaluate.disagree_study import check_dev_only, sha256
from src.evaluate.fusion_study import FOUR
from src.evaluate.pick_tiers_study import implied_lines, ou_grid
from src.features.usage import DataUsage, season_label
from src.web.site.picks import VALUE

N = 10                                   # 比分矩阵 0–10 球（最后一行 / 列是“10 个及以上”），和回测一样
PCOLS = ["p_home", "p_draw", "p_away", "p_over25"]   # 每个来源存的四个概率
MARKETS = ("ah", "1x2", "ou")
MARKET_NAMES = {"ah": "让球（主盘口）", "1x2": "胜平负", "ou": "大小球（主盘口）"}
CANDIDATES = {"A": "A. 修正模型自己的偏差", "B": "B. 模型往赔率拉", "C": "C. 赔率拿不准时听模型",
              "D": "D. gb_inj 自己", "E": "E. 赔率 + 伤停带来的变化"}
SIDE_1X2 = ("主胜", "平", "客胜")

# 把比分矩阵（拉平成 121 格）换成“净胜球”“总进球”“主胜 / 平 / 客胜”的分布：乘一个 0/1 表就行，一批比赛一次算完
_I, _J = np.indices((N + 1, N + 1))
AH_X = np.arange(-N, N + 1)              # 净胜球（主 − 客）−10 … +10
OU_X = np.arange(2 * N + 1)              # 总进球 0 … 20
DIFF_IND = ((_I - _J).ravel()[:, None] == AH_X[None, :]).astype(float)       # 121 × 21
TOT_IND = ((_I + _J).ravel()[:, None] == OU_X[None, :]).astype(float)        # 121 × 21
P3_IND = np.stack([(_I > _J).ravel(), (_I == _J).ravel(), (_I < _J).ravel()], axis=1).astype(float)
OVER25 = ((_I + _J).ravel() > 2).astype(float)


@dataclass
class Data:
    """一批比赛（每场一行）和算好的东西。按行切（take）时每样都一起切，保证对得上。"""
    t: pd.DataFrame                       # 每场一行：比赛、比分、让球主盘口、每个来源的四个概率
    lam: dict                             # 来源 → (λ主, λ客, rho)：反推出来的比分矩阵参数
    setup: dict                           # 每场的线、每种进球结果的结算分值、实际结算分值
    probs: dict                           # 来源 → {"ah": 主队让球胜算, "ou": 大球胜算, "1x2": 三个概率}
    checks: dict = field(default_factory=dict)


def take(d: Data, mask) -> Data:
    """只留 mask 为 True 的比赛（选参数时只给 2014–2019 的就是用这个切出来的）。"""
    mask = np.asarray(mask, bool)
    return Data(t=d.t[mask].reset_index(drop=True),
                lam={k: tuple(x[mask] for x in v) for k, v in d.lam.items()},
                setup={k: v[mask] for k, v in d.setup.items()},
                probs={k: {m: x[mask] for m, x in v.items()} for k, v in d.probs.items()},
                checks=d.checks)


# ======================= 赛季：考试赛季一律不碰 =======================
def seasons_of(settings: dict) -> tuple[int, int, int]:
    """(第一个赛季, 定参数的最后一季, 检验的最后一季)。检验的最后一季超过开发赛季就停下（期末考试赛季不能用）。"""
    c = settings["direction_study"]
    first, plast, tlast = int(c["first_season"]), int(c["param_last"]), int(c["test_last"])
    if tlast > int(settings["seasons"]["dev_last"]):
        raise ValueError(f"direction_study.test_last = {tlast} 超过了开发赛季的最后一季，不能用期末考试赛季")
    if not first <= plast < tlast:
        raise ValueError(f"赛季设置不对：first_season {first}、param_last {plast}、test_last {tlast}")
    check_dev_only(range(first, tlast + 1), settings)
    return first, plast, tlast


def source_names(settings: dict) -> list[str]:
    """四个来源：赔率（market）、gb_l7、gb_l7_r、gb_inj。"""
    c = settings["direction_study"]
    return ["market", c["base_model"], *c["inj_models"]]


def read_run(run_dir: Path, models, settings: dict) -> pd.DataFrame:
    """读一次回测的 predictions.csv：先按赛季、联赛、模型、时点筛（考试赛季的行不往下传），再查剩下的全是开发赛季。"""
    c = settings["direction_study"]
    first, _, last = seasons_of(settings)
    full = pd.read_csv(Path(run_dir) / "predictions.csv", encoding="utf-8-sig", low_memory=False)
    keep = (full["season"].between(first, last) & full["competition"].isin(c["leagues"])
            & full["model"].isin(list(models)) & (full["horizon"] == c["horizon"]))
    p = full[keep].reset_index(drop=True)
    check_dev_only(p["season"].unique(), settings)
    return p


def check_run_info(run_info: dict, models, settings: dict) -> dict:
    """回测记下的设置和现在一样吗？运行模式要是开发；用到的几个模型的设置（gbdt 和 gbdt_variants 里那几项）要和现在一样。

    只比用到的模型：I7 之后 gbdt_variants 又加过别的变体，那些和这次无关。不一样就停下（旧结果不代表现在的模型）。
    """
    if run_info.get("运行模式") != "dev":
        raise ValueError(f"回测的运行模式是 {run_info.get('运行模式')!r}，这次只能用开发模式的回测")
    old = run_info.get("设置", {}).get("models")
    if not old:
        raise ValueError("回测的 run_info.json 里没有 models 设置，没法核对模型有没有改过")
    cur = settings["models"]
    checked = []
    if old.get("gbdt") != cur.get("gbdt"):
        raise ValueError("回测时的 models.gbdt 设置和现在不一样，这次回测不代表现在的模型")
    checked.append("gbdt")
    for m in models:
        if m == "gbdt":
            continue
        a, b = old.get("gbdt_variants", {}).get(m), cur.get("gbdt_variants", {}).get(m)
        if a is None or a != b:
            raise ValueError(f"模型 {m} 的设置回测时是 {a}，现在是 {b}，不一样，停下")
        checked.append(f"gbdt_variants.{m}")
    return {"比过的设置": checked, "回测的模型": run_info.get("模型"), "回测的赛季": run_info.get("赛季")}


def check_usage(seasons, settings: dict) -> str:
    """伤停那份（用途「研究实验」，只在开发模式）：按数据用途字典查，开发模式能不能把这些赛季用于这个用途。"""
    c = settings["direction_study"]
    usage = c["inj_usage"]
    du = DataUsage(settings, mode="dev")
    du.check_usage_name(usage)
    for s in sorted({int(x) for x in seasons}):
        du.check_mode("dev", season_role(s, settings), usage, f"{season_label(s)} 赛季的伤停研究预测")
    for m in c["inj_models"]:
        got = settings["models"]["gbdt_variants"][m].get("usage")
        if got != usage:
            raise ValueError(f"{m} 的用途是 {got!r}，不是 {usage!r}")
    return usage


def build_table(base: pd.DataFrame, inj: pd.DataFrame, odds: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, dict]:
    """每场一行：比赛、比分、让球主盘口，加上每个来源的四个概率（列名 来源:p_home 这样）。

    要：有比分；胜平负和大 2.5 球的赔率都有；让球主盘口（两边赔率都在 fusion_study.ah_main_odds 里，和 SEL 一样）；
    I7 那份里 gb_l7_r 和 gb_inj 都有。
    """
    c = settings["direction_study"]
    lo_o, hi_o = settings["fusion_study"]["ah_main_odds"]
    b = base[base["model"] == c["base_model"]].dropna(subset=["result", *PCOLS]).drop_duplicates("match_id")
    t = b[["match_id", "competition", "season", "match_date", "home_goals", "away_goals"]].merge(odds, on="match_id", how="left")
    chk = {"gb_l7 有预测的比赛": int(len(t))}
    ok = t[FOUR].notna().all(axis=1)
    main = t[["ah_home_odds", "ah_away_odds"]].apply(lambda x: x.between(lo_o, hi_o)).all(axis=1) & t["ah_line"].notna()
    chk["胜平负或大 2.5 球赔率缺"] = int((~ok).sum())
    chk["没有让球主盘口"] = int((ok & ~main).sum())
    t = t[ok & main].copy()
    t = t.merge(b[["match_id", *PCOLS]].rename(columns={k: f"{c['base_model']}:{k}" for k in PCOLS}), on="match_id")
    for m in c["inj_models"]:
        part = inj[inj["model"] == m].drop_duplicates("match_id")
        g = part[["match_id", "home_goals", "away_goals", *PCOLS]].rename(
            columns={"home_goals": f"{m}:hg", "away_goals": f"{m}:ag", **{k: f"{m}:{k}" for k in PCOLS}})
        n0 = len(t)
        t = t.merge(g, on="match_id", how="inner")
        chk[f"I7 里没有 {m} 的比赛"] = int(n0 - len(t))
        if not ((t[f"{m}:hg"] == t["home_goals"]) & (t[f"{m}:ag"] == t["away_goals"])).all():
            raise ValueError(f"{m} 那份的比分和 w9_gb_l7 对不上，两次回测不是同一批比赛？")
        t = t.drop(columns=[f"{m}:hg", f"{m}:ag"])
    for i, k in enumerate(PCOLS):
        t[f"market:{k}"] = t[FOUR[i]]
    t = t.dropna(subset=[f"{s}:{k}" for s in source_names(settings) for k in PCOLS])
    chk["用到的比赛"] = int(len(t))
    return t.sort_values("match_id", ignore_index=True), chk


# ======================= 比分矩阵和各玩法的胜算 =======================
def fit_source(target: np.ndarray, tol: float | None, name: str) -> tuple[tuple, dict]:
    """一个来源的四个概率 → (λ主, λ客, rho)。再用这三个数算回四个概率，和原来的比：
    tol 给了（模型）就要求最多差 tol，超过就停下；赔率不卡（rho 取到边界时平局对不上，和 fusion_study 一样只报告）。"""
    target = np.asarray(target, float).copy()
    s = target[:, :3].sum(axis=1, keepdims=True)
    if np.abs(s - 1).max() > 1e-4:
        raise ValueError(f"{name} 的主胜 + 平 + 客胜和 1 差了 {np.abs(s - 1).max():.2e}，概率不对")
    target[:, :3] /= s                                         # csv 只存 6 位小数，补成正好是 1
    lh, la, rho = fit_market_dc(target, n=N)
    got = flat(dc_matrices(lh, la, rho, n=N))
    err = np.abs(np.column_stack([got @ P3_IND, got @ OVER25]) - target)
    lo, hi = dc_rho_bounds(lh, la)
    info = {"还原误差（最大）": float(err.max()), "还原误差（平均）": float(err.mean()),
            "还原误差超过门槛的场数": int((err.max(axis=1) > (tol if tol is not None else 1e-3)).sum()),
            "rho 取了边界的场数": int(((rho <= lo + 1e-12) | (rho >= hi - 1e-12)).sum())}
    if tol is not None and err.max() > tol:
        raise ValueError(f"{name} 反推的比分矩阵算回来和存的概率最多差 {err.max():.2e}，超过 {tol}，停下")
    return (lh, la, rho), info


def flat(mats: np.ndarray) -> np.ndarray:
    """比分矩阵（场数 × 11 × 11）拉平成场数 × 121。"""
    return mats.reshape(len(mats), -1)


def ah_values(line: float) -> np.ndarray:
    """这条主队盘口上，净胜球 −10 … +10 时买主队的结算分值（赢 1、赢半 0.5、走 0、输半 −0.5、输 −1）。买客队正好相反。"""
    return np.array([VALUE[settle_margin(int(d), float(line))] for d in AH_X])


def ou_values(line: float) -> np.ndarray:
    """这条大小球线上，总进球 0 … 20 时买大的结算分值。买小正好相反。"""
    return np.array([VALUE[settle_margin(int(k), -float(line))] for k in OU_X])


def table_for(lines: np.ndarray, fn) -> np.ndarray:
    """每场一行的结算分值表：相同的线只算一次。"""
    uniq = {float(x): fn(float(x)) for x in np.unique(lines)}
    return np.stack([uniq[float(x)] for x in lines])


def up_share(dist: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """胜算 = (赢 + 赢半×0.5) ÷ (赢 + 赢半×0.5 + 输 + 输半×0.5)，和网站、markets.win_share 一样（走盘不算）。
    dist：每场净胜球（或总进球）的分布；vals：每场同样长度的结算分值。"""
    up = (dist * np.clip(vals, 0, None)).sum(axis=1)
    down = (dist * np.clip(-vals, 0, None)).sum(axis=1)
    tot = up + down
    if (tot <= 0).any():
        raise ValueError("有比赛这条线上只会走盘，算不出胜算")
    return up / tot


def market_probs(lam: tuple, setup: dict) -> dict:
    """(λ主, λ客, rho) → 每场 主队让球胜算、大球胜算（都在这场的那条线上）、胜平负三个概率。"""
    m = flat(dc_matrices(*lam, n=N))
    return {"ah": up_share(m @ DIFF_IND, setup["ah_vals"]),
            "ou": up_share(m @ TOT_IND, setup["ou_vals"]),
            "1x2": m @ P3_IND}


def prepare(t: pd.DataFrame, settings: dict) -> Data:
    """每场一行的表 → Data：每个来源反推比分矩阵参数；每场的让球线、大小球线（赔率矩阵推出来的主盘口）和结算分值。
    选方向、算命中率都只用这里算好的东西。"""
    c = settings["direction_study"]
    check_dev_only(t["season"].unique(), settings)
    t = t.sort_values("match_id", ignore_index=True)
    lam, checks = {}, {}
    for s in source_names(settings):
        lam[s], checks[s] = fit_source(t[[f"{s}:{k}" for k in PCOLS]].to_numpy(float),
                                       None if s == "market" else float(c["refit_tol"]), s)
    hg, ag = t["home_goals"].to_numpy(int), t["away_goals"].to_numpy(int)
    ah_line = t["ah_line"].to_numpy(float)
    m_market = flat(dc_matrices(*lam["market"], n=N))
    ou_line, _ = implied_lines(m_market @ TOT_IND, ou_grid(settings))   # 和 SEL 一样：赔率矩阵大球胜算最接近 50% 的线
    setup = {"ah_line": ah_line, "ah_vals": table_for(ah_line, ah_values),
             "ou_line": ou_line, "ou_vals": table_for(ou_line, ou_values),
             # 实际结算：买主队 / 买大的分值。买客队、买小正好是相反数（同一条线两边输赢对调，走盘还是走盘）
             "v_ah_home": np.array([VALUE[settle_asian_handicap(int(h), int(a), float(ln), "home")["outcome"]]
                                    for h, a, ln in zip(hg, ag, ah_line)]),
             "v_ou_over": np.array([VALUE[settle_over_under(int(h), int(a), float(ln), "over")["outcome"]]
                                    for h, a, ln in zip(hg, ag, ou_line)]),
             "real_1x2": np.where(hg > ag, 0, np.where(hg == ag, 1, 2)),
             "season": t["season"].to_numpy(int),
             "week": block_ids(t["match_date"], settings["compare"]["block"])}
    probs = {s: market_probs(lam[s], setup) for s in lam}
    return Data(t=t, lam=lam, setup=setup, probs=probs, checks={"反推比分矩阵": checks})


# ======================= 选方向、结算 =======================
def side_of(market: str, p) -> np.ndarray:
    """方向：让球、大小球 = 主队（大球）胜算 ≥ 50% 选主队（大），记 +1，否则 −1；胜平负 = 三个概率最大的那项（0 主胜、1 平、2 客胜，一样大取前面的）。

    先四舍五入到 9 位小数再比：赔率正好五五开的比赛（例如平手盘、主客概率一样），算出来的胜算会是 0.4999999999 或 0.5000000001，
    不四舍五入的话选哪边就由计算误差定了；四舍五入以后一律按规则选主队 / 大 / 前面那项。"""
    p = np.round(np.asarray(p, float), 9)
    if market == "1x2":
        return np.argmax(p, axis=1)
    return np.where(p >= 0.5, 1, -1)


def claim_of(market: str, p, side) -> np.ndarray:
    """选出那一边写的胜算（概率）。"""
    if market == "1x2":
        return np.asarray(p)[np.arange(len(side)), side]
    p = np.asarray(p)
    return np.where(side == 1, p, 1 - p)


def value_of(market: str, side, setup: dict) -> np.ndarray:
    """每场结算分值：赢 1、赢半 0.5、走盘 0、输半 −0.5、输 −1；胜平负 中 1、没中 −1。"""
    if market == "ah":
        return side * setup["v_ah_home"]
    if market == "ou":
        return side * setup["v_ou_over"]
    return np.where(side == setup["real_1x2"], 1.0, -1.0)


def hit_rate(v) -> float:
    """命中率 = 赢的分 ÷ 有输赢的分（走盘 0 分不算），和战绩页、SEL 一样。"""
    v = np.asarray(v, float)
    den = np.abs(v).sum()
    return float(np.clip(v, 0, None).sum() / den) if den > 0 else float("nan")


def near_even(market: str, p) -> np.ndarray:
    """赔率离五五开多远：让球、大小球 = |胜算 − 50%|；胜平负 = 最大和第二大的概率差。候选 C 用。"""
    if market == "1x2":
        s = np.sort(np.asarray(p), axis=1)
        return s[:, 2] - s[:, 1]
    return np.abs(np.asarray(p) - 0.5)


# ======================= 五个候选 =======================
def from_logs(log_h, log_a, rho, setup: dict) -> dict:
    """改过的 log λ → 各玩法胜算。rho 超出这场能取的范围时取边界（和 fit_market_dc 一样）。"""
    lh, la = np.exp(log_h), np.exp(log_a)
    lo, hi = dc_rho_bounds(lh, la)
    return market_probs((lh, la, np.clip(rho, lo, hi)), setup)


def grid(spec) -> np.ndarray:
    lo, hi, step = (float(x) for x in spec)
    return np.round(np.arange(lo, hi + step / 2, step), 6)


def spaces(x: str, settings: dict) -> list[tuple[tuple, list[dict]]]:
    """每个候选要试的参数：[(这组参数给哪几个玩法用, 参数表)]。A 的让球、胜平负改差 s，大小球改总 m，分开试。"""
    g = settings["direction_study"]["grids"]
    if x == "A":
        kb = [{"k": float(k), "b": float(b), "d": 0.0} for k in grid(g["a_k"]) for b in grid(g["a_b"])]
        return [(("ah", "1x2"), kb), (("ou",), [{"k": 1.0, "b": 0.0, "d": float(v)} for v in grid(g["a_d"])])]
    one = {"B": [{"a": float(v)} for v in grid(g["b_a"])], "C": [{"tau": float(v)} for v in grid(g["c_tau"])],
           "D": [{}], "E": [{"c": float(v)} for v in grid(g["e_c"])]}[x]
    return [(MARKETS, one)]


def closeness(x: str, prm: dict) -> tuple:
    """一样高时取最接近“不改”的：A 先 k 离 1 最近、再 b 离 0 最近（大小球 d 离 0 最近）；B、C、E 取最小的 a、τ、c。"""
    if x == "A":
        return abs(prm["k"] - 1), abs(prm["b"]), abs(prm["d"])
    return tuple(abs(v) for v in prm.values())


def decide(d: Data, x: str, prm: dict, settings: dict) -> dict:
    """候选 x 用参数 prm 时每场的方向 → {玩法: (方向, 写的胜算或 None, 胜算或 None)}。C 没有自己的概率，后两项是 None。"""
    c = settings["direction_study"]
    base, (ref, inj) = c["base_model"], c["inj_models"]
    if x == "C":
        pm, pb = d.probs["market"], d.probs[base]
        return {m: (np.where(near_even(m, pm[m]) < prm["tau"], side_of(m, pb[m]), side_of(m, pm[m])), None, None)
                for m in MARKETS}
    if x == "A":
        lh, la, rho = d.lam[base]
        s, mm = (np.log(lh) - np.log(la)) / 2, (np.log(lh) + np.log(la)) / 2
        s2, m2 = prm["k"] * s + prm["b"], mm + prm["d"]
        p = from_logs(m2 + s2, m2 - s2, rho, d.setup)
    elif x == "B":
        mh, ma, mrho = d.lam["market"]
        lh, la, _ = d.lam[base]
        p = from_logs(np.log(mh) + prm["a"] * (np.log(lh) - np.log(mh)),
                      np.log(ma) + prm["a"] * (np.log(la) - np.log(ma)), mrho, d.setup)
    elif x == "E":
        mh, ma, mrho = d.lam["market"]
        (rh, ra, _), (ih, ia, _) = d.lam[ref], d.lam[inj]
        p = from_logs(np.log(mh) + prm["c"] * (np.log(ih) - np.log(rh)),
                      np.log(ma) + prm["c"] * (np.log(ia) - np.log(ra)), mrho, d.setup)
    elif x == "D" or x in d.probs:        # D（gb_inj）和对照（赔率、gb_l7、gb_l7_r）：直接用那个来源自己的胜算
        p = d.probs[inj if x == "D" else x]
    else:
        raise ValueError(f"不认识的候选 {x!r}")
    out = {}
    for m in MARKETS:
        side = side_of(m, p[m])
        out[m] = (side, claim_of(m, p[m], side), p[m])
    return out


def select(dp: Data, settings: dict) -> tuple[dict, list[dict]]:
    """在定参数的赛季上给每个候选、每个玩法选参数（命中率最高的；一样高取最接近“不改”的）。

    dp 只能有定参数赛季（2014–2019）的比赛：拿到检验季的比赛就停下（检验季不准拿来选参数）。
    返回 (选的参数, 每个参数在定参数赛季上的命中率表)。
    """
    _, plast, _ = seasons_of(settings)
    if len(dp.t) == 0 or (dp.setup["season"] > plast).any() or (dp.t["season"] > plast).any():
        raise ValueError(f"选参数只能用 {plast} 及以前的赛季，拿到了检验季的比赛，停下")
    chosen, rows = {}, []
    for x in CANDIDATES:
        chosen[x] = {}
        for markets, plist in spaces(x, settings):
            best = {m: None for m in markets}
            for prm in plist:
                dec = decide(dp, x, prm, settings)
                for m in markets:
                    r = hit_rate(value_of(m, dec[m][0], dp.setup))
                    rows.append({"候选": x, "玩法": m, "参数": json.dumps(prm), "定参数赛季命中率": r})
                    rank = (-round(r, 12), closeness(x, prm))
                    if best[m] is None or rank < best[m][0]:
                        best[m] = (rank, prm, r)
            for m in markets:
                chosen[x][m] = {"参数": best[m][1], "定参数赛季命中率": best[m][2]}
    return chosen, rows


# ======================= 检验季：差、区间、判断 =======================
def boot_ratio(num, den, weeks, settings: dict) -> tuple[float, float, float, float]:
    """Σnum ÷ Σden 和它的区间：按周成块 bootstrap（抽周，再把抽到的周的 num、den 各自加起来相除）。
    种子、次数、置信水平用 settings 的 compare；每次调用都从同一个种子开始，所以几个候选抽到的是同一批周（成对比较）。
    返回 (点估计, 区间下限, 区间上限, bootstrap 标准差)。"""
    c = settings["compare"]
    _, inv = np.unique(np.asarray(weeks), return_inverse=True)
    sn = np.bincount(inv, weights=np.asarray(num, float))
    sd = np.bincount(inv, weights=np.asarray(den, float))
    k, n_boot = len(sn), int(c["n_boot"])
    rng = np.random.default_rng(int(c["seed"]))
    r = np.empty(n_boot)
    step = max(1, 2_000_000 // k)
    for i in range(0, n_boot, step):
        idx = rng.integers(0, k, size=(min(step, n_boot - i), k))
        r[i:i + len(idx)] = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    alpha = (1 - float(c["confidence"])) / 2
    lo, hi = np.quantile(r, [alpha, 1 - alpha])
    return float(sn.sum() / sd.sum()), float(lo), float(hi), float(r.std(ddof=1))


def diff_parts(v_c, v_m) -> tuple[np.ndarray, np.ndarray]:
    """两个选法的命中率差 = Σ(候选赢的分 − 赔率赢的分) ÷ Σ有输赢的分。同一条线两边“有输赢的分”一样，这里核对一下。"""
    v_c, v_m = np.asarray(v_c, float), np.asarray(v_m, float)
    if not np.allclose(np.abs(v_c), np.abs(v_m)):
        raise ValueError("两个选法每场“有输赢的分”不一样，不是同一条线？")
    return np.clip(v_c, 0, None) - np.clip(v_m, 0, None), np.abs(v_m)


def compare(v_c, v_m, d: Data, settings: dict) -> dict:
    """候选和赔率在这批比赛上比：命中率、差、95% 区间、标准差、每个赛季的差。"""
    num, den = diff_parts(v_c, v_m)
    diff, lo, hi, sd = boot_ratio(num, den, d.setup["week"], settings)
    seasons = {}
    for s in sorted(np.unique(d.setup["season"])):
        k = d.setup["season"] == s
        seasons[int(s)] = float(num[k].sum() / den[k].sum())
    return {"场数": int(len(v_c)), "命中率": hit_rate(v_c), "赔率命中率": hit_rate(v_m),
            "差": diff, "区间": [lo, hi], "标准差": sd, "每个赛季的差": seasons}


def judge(res: dict, settings: dict) -> dict:
    """事先定的三条：95% 区间下限 > 0；四个检验赛季至少三个差 > 0；差 ≥ +0.5 个百分点。
    三条都过 = “过了”，再看 Bonferroni（差 − 2.576 × 标准差 > 0）写“比较可信”或“可能是碰巧”；否则“没看到”。"""
    c = settings["direction_study"]
    ok1 = res["区间"][0] > 0
    pos = sum(v > 0 for v in res["每个赛季的差"].values())
    ok2 = pos >= int(c["min_positive_seasons"])
    ok3 = res["差"] >= float(c["min_gain"]) - 1e-12
    bonf = res["差"] - float(c["bonferroni_z"]) * res["标准差"]
    verdict = "没看到"
    if ok1 and ok2 and ok3:
        verdict = "过了，比较可信" if bonf > 0 else "过了，可能是碰巧"
    return {"下限 > 0": bool(ok1), "差 > 0 的赛季数": int(pos), "赛季够": bool(ok2), "差够大": bool(ok3),
            "Bonferroni 下限": float(bonf), "结论": verdict}


def bin_labels(edges) -> list[str]:
    """档名：[0.5, 0.525, 0.55, 0.6] → 50%–52.5%、52.5%–55%、55%–60%、60% 以上。"""
    e = [f"{float(x) * 100:g}%" for x in edges]
    return [f"{e[i]}–{e[i + 1]}" for i in range(len(e) - 1)] + [f"{e[-1]} 以上"]


def bin_of(x, edges) -> np.ndarray:
    """每场落在哪一档（标签）。edges 是每档的下边，最后一个往上是一档。"""
    labels = bin_labels(edges)
    k = np.clip(np.searchsorted(np.asarray(edges, float), np.asarray(x, float), side="right") - 1, 0, len(labels) - 1)
    return np.array(labels, dtype=object)[k]


def calibration(name: str, m: str, claim, v, settings: dict) -> list[dict]:
    """校准：选出那一边写的胜算平均多少、实际命中率多少（总的 + 按写的胜算分档）。"""
    edges = settings["direction_study"]["calib_bins"]["three_way" if m == "1x2" else "two_way"]
    claim, v = np.asarray(claim, float), np.asarray(v, float)
    rows = []
    for label, k in [("全部", np.ones(len(v), bool))] + [(b, bin_of(claim, edges) == b) for b in bin_labels(edges)]:
        if k.sum() == 0:
            continue
        r = hit_rate(v[k])
        rows.append({"来源": name, "玩法": m, "档": label, "场数": int(k.sum()),
                     "写的胜算": float(claim[k].mean()), "实际命中率": r, "差（实际 − 写的）": r - float(claim[k].mean())})
    return rows


def logloss_parts(m: str, p_c, p_m, d: Data) -> tuple[np.ndarray, np.ndarray]:
    """每场 log loss 差（候选 − 赔率）和权重。胜平负：−log(实际那项的概率)，每场权重 1。
    让球（主队胜算 p）：主队赢 −log p、赢半 0.5·(−log p)、输半 0.5·(−log(1−p))、输 −log(1−p)，走盘权重 0。"""
    eps = 1e-12
    if m == "1x2":
        r = d.setup["real_1x2"]
        ll = lambda p: -np.log(np.clip(np.asarray(p)[np.arange(len(r)), r], eps, 1))
        return ll(p_c) - ll(p_m), np.ones(len(r))
    v = d.setup["v_ah_home"]
    w = np.abs(v)

    def ll(p):
        p = np.clip(np.asarray(p, float), eps, 1 - eps)
        return np.where(v > 0, -np.log(p), -np.log(1 - p)) * w
    return ll(p_c) - ll(p_m), w


def logloss(m: str, p_c, p_m, d: Data, settings: dict) -> dict:
    num, w = logloss_parts(m, p_c, p_m, d)
    eps = 1e-12
    if m == "1x2":
        r = d.setup["real_1x2"]
        base = float(-np.log(np.clip(np.asarray(p_m)[np.arange(len(r)), r], eps, 1)).mean())
    else:
        v = d.setup["v_ah_home"]
        p = np.clip(np.asarray(p_m, float), eps, 1 - eps)
        base = float((np.where(v > 0, -np.log(p), -np.log(1 - p)) * np.abs(v)).sum() / np.abs(v).sum())
    diff, lo, hi, _ = boot_ratio(num, w, d.setup["week"], settings)
    return {"赔率": base, "候选": base + diff, "差（候选 − 赔率）": diff, "区间": [lo, hi]}


def ah_price_side(t: pd.DataFrame) -> np.ndarray:
    """对照（只报告）：让球两边去水后的赔率哪边低选哪边（一样低选主队）。"""
    ph, pa = 1 / t["ah_home_odds"].to_numpy(float), 1 / t["ah_away_odds"].to_numpy(float)
    return np.where(ph / (ph + pa) >= 0.5, 1, -1)


def study(d: Data, settings: dict, log=print) -> dict:
    """整个研究：2014–2019 选参数 → 2020–2023 检验。d 是 prepare 出来的全部比赛（只有开发赛季）。

    返回 summary（写 json、报告用）和几张表（picks、grid、calibration、groups）。
    """
    first, plast, tlast = seasons_of(settings)
    c = settings["direction_study"]
    base, (ref, inj) = c["base_model"], c["inj_models"]
    check_dev_only(d.setup["season"], settings)
    if (d.setup["season"] < first).any() or (d.setup["season"] > tlast).any():
        raise ValueError(f"比赛的赛季超出了 {first}–{tlast}，停下")
    in_param = d.setup["season"] <= plast
    dp, dt = take(d, in_param), take(d, ~in_param)       # 选参数只给 dp（2014–2019）；dt 只在参数定下以后才用
    log(f"定参数 {first}–{plast}：{len(dp.t)} 场；检验 {plast + 1}–{tlast}：{len(dt.t)} 场")
    chosen, grid_rows = select(dp, settings)
    log("参数选好了：" + "；".join(f"{x} " + "、".join(f"{m} {json.dumps(chosen[x][m]['参数'])}" for m in MARKETS)
                                for x in CANDIDATES))

    # 检验季：每个候选按每个玩法自己的参数出方向
    def run(dd: Data, x: str, m: str):
        prm = chosen[x][m]["参数"] if x in CANDIDATES else {}
        return decide(dd, x, prm, settings)[m]

    test, refs, calib, ll, picks = {}, {}, [], {}, {"match_id": dt.t["match_id"].to_numpy(), "season": dt.setup["season"],
                                                   "competition": dt.t["competition"].to_numpy(),
                                                   "match_date": dt.t["match_date"].to_numpy(),
                                                   "ah_line": dt.setup["ah_line"], "ou_line": dt.setup["ou_line"]}
    mk = {m: run(dt, "market", m) for m in MARKETS}
    vm = {m: value_of(m, mk[m][0], dt.setup) for m in MARKETS}
    param_mk = {m: hit_rate(value_of(m, run(dp, "market", m)[0], dp.setup)) for m in MARKETS}
    for name in ["market", base, ref]:
        refs[name] = {}
        for m in MARKETS:
            side, claim, p = run(dt, name, m)
            v = value_of(m, side, dt.setup)
            refs[name][m] = compare(v, vm[m], dt, settings)
            calib += calibration(name, m, claim, v, settings)
            picks[f"{name}_{m}_side"], picks[f"{name}_{m}_v"] = side, v
    vp = value_of("ah", ah_price_side(dt.t), dt.setup)
    refs["让球价格"] = {"ah": compare(vp, vm["ah"], dt, settings)}
    for x in CANDIDATES:
        test[x], ll[x] = {}, {}
        for m in MARKETS:
            side, claim, p = run(dt, x, m)
            v = value_of(m, side, dt.setup)
            res = compare(v, vm[m], dt, settings)
            res["定参数赛季命中率"] = chosen[x][m]["定参数赛季命中率"]
            res["定参数赛季赔率命中率"] = param_mk[m]
            res["参数"] = chosen[x][m]["参数"]
            if m in ("ah", "1x2"):
                res["判断"] = judge(res, settings)
            test[x][m] = res
            picks[f"{x}_{m}_side"], picks[f"{x}_{m}_v"] = side, v
            if claim is None:
                calib.append({"来源": x, "玩法": m, "档": "不适用（没有自己的概率）"})
            else:
                calib += calibration(x, m, claim, v, settings)
                if m in ("ah", "1x2"):
                    ll[x][m] = logloss(m, p, mk[m][2], dt, settings)
    groups = group_rows(dt, test, vm, picks, settings)
    summary = {"规则": "选方向研究（需求文档第 4 节“选方向研究怎么做”“事先定的比较”）",
               "定参数赛季": [first, plast], "检验赛季": [plast + 1, tlast],
               "场数": {"定参数": int(len(dp.t)), "检验": int(len(dt.t))},
               "每个检验赛季的场数": {int(s): int((dt.setup["season"] == s).sum()) for s in np.unique(dt.setup["season"])},
               "检验": test, "对照": refs, "log loss": ll, "伤停来源": injury_decision(test),
               "检查": d.checks}
    return {"summary": summary, "picks": pd.DataFrame(picks), "grid": pd.DataFrame(grid_rows),
            "calibration": pd.DataFrame(calib), "groups": pd.DataFrame(groups)}


def injury_decision(test: dict) -> str:
    """事先定：D 或 E 在让球或胜平负上“过了”，才写值得去找实时伤停来源（是上限）；都没看到就写只为选方向不值得接。"""
    hit = [f"{x} 的{MARKET_NAMES[m]}" for x in ("D", "E") for m in ("ah", "1x2")
           if test[x][m]["判断"]["结论"].startswith("过了")]
    if hit:
        return ("、".join(hit) + " 过了：对选方向有用，值得去找实时伤停来源。"
                "注意这是上限（伤停开始日期是事后整理的），实时来源的效果只会更小。")
    return "D、E 在让球和胜平负上都没看到：只为选方向，不值得接实时伤停来源（概率准不准是另一回事，I7 已经回答过）。"


def group_rows(dt: Data, test: dict, vm: dict, picks: dict, settings: dict) -> list[dict]:
    """只是看看（不能拿来挑结论）：检验季按赛季、联赛、分歧大小、伤停改了多少分开，每个候选让球、胜平负的差和区间。"""
    c = settings["direction_study"]
    base, (ref, inj) = c["base_model"], c["inj_models"]
    pr = dt.probs
    by = {"赛季": np.array([season_label(int(s)) for s in dt.setup["season"]], dtype=object),
          "联赛": dt.t["competition"].to_numpy(object),
          "分歧大小": bin_of(np.abs(pr[base]["ah"] - pr["market"]["ah"]), c["diverge_bins"]),
          "伤停改了多少": bin_of(np.abs(pr[inj]["ah"] - pr[ref]["ah"]), c["inj_bins"])}
    rows = []
    for gname, lab in by.items():
        order = {"分歧大小": bin_labels(c["diverge_bins"]), "伤停改了多少": bin_labels(c["inj_bins"])}.get(gname)
        for g in (order if order else sorted(set(lab))):
            if not (lab == g).any():
                continue
            k = lab == g
            sub = take(dt, k)
            for x in CANDIDATES:
                for m in ("ah", "1x2"):
                    res = compare(picks[f"{x}_{m}_v"][k], vm[m][k], sub, settings)
                    rows.append({"分组": gname, "组": g, "候选": x, "玩法": m, "场数": res["场数"],
                                 "候选命中率": res["命中率"], "赔率命中率": res["赔率命中率"],
                                 "差": res["差"], "区间下限": res["区间"][0], "区间上限": res["区间"][1]})
    return rows


# ======================= 读真实数据 =======================
def load(settings: dict, root: Path, log=print) -> Data:
    """读两次回测（只读）→ prepare。root 是项目文件夹，设置里的相对路径从这里算。"""
    c = settings["direction_study"]
    run_dir, inj_dir = root / c["run"], root / c["inj_run"]
    info = {}
    for key, folder, models in (("w9_gb_l7", run_dir, [c["base_model"]]), ("s7_eval", inj_dir, c["inj_models"])):
        ri = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
        info[key] = check_run_info(ri, models, settings)
    base = read_run(run_dir, [c["base_model"]], settings)
    inj = read_run(inj_dir, c["inj_models"], settings)
    usage = check_usage(inj["season"].unique(), settings)
    odds = pd.read_csv(run_dir / "fusion_odds.csv", encoding="utf-8-sig")
    t, chk = build_table(base, inj, odds, settings)
    log(f"用到 {len(t)} 场（{int(t['season'].min())}–{int(t['season'].max())}），反推比分矩阵……")
    d = prepare(t, settings)
    shas = {f"{rel}/{f}": sha256(folder / f) for rel, folder, names in
            ((c["run"], run_dir, ("predictions.csv", "fusion_odds.csv", "run_info.json")),
             (c["inj_run"], inj_dir, ("predictions.csv", "run_info.json"))) for f in names}
    d.checks.update({"建表": chk, "回测核对": info, "伤停那份的用途": usage, "文件 sha256": shas})
    return d


# ======================= 报告 =======================
def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def pp(x) -> str:
    return f"{x * 100:+.2f}"


def ci(r) -> str:
    return f"{pp(r[0])} 到 {pp(r[1])}"


def prm_text(x: str, m: str, prm: dict) -> str:
    if x == "A":
        return f"d = {prm['d']:+.2f}" if m == "ou" else f"k = {prm['k']:.2f}，b = {prm['b']:+.2f}"
    if not prm:
        return "（没有参数）"
    k, v = next(iter(prm.items()))
    return {"a": f"a = {v:.2f}", "tau": f"τ = {v:.3f}", "c": f"c = {v:.2f}"}[k]


def market_table(s: dict, m: str, verdict: bool) -> list[str]:
    head = "| 候选 | 参数 | 定参数赛季：候选 / 赔率 | 检验季命中率 | 赔率 | 差（百分点） | 95% 区间 | 差 > 0 的赛季 |"
    sep = "|---|---|---|---|---|---|---|---|"
    if verdict:
        head, sep = head + " Bonferroni 下限 | 结论 |", sep + "---|---|"
    out = [head, sep]
    for x, name in CANDIDATES.items():
        r = s["检验"][x][m]
        pos = sum(v > 0 for v in r["每个赛季的差"].values())
        row = (f"| {name} | {prm_text(x, m, r['参数'])} | {pct(r['定参数赛季命中率'])} / {pct(r['定参数赛季赔率命中率'])} "
               f"| {pct(r['命中率'])} | {pct(r['赔率命中率'])} | {pp(r['差'])} | {ci(r['区间'])} | {pos} / {len(r['每个赛季的差'])} |")
        if verdict:
            j = r["判断"]
            row += f" {pp(j['Bonferroni 下限'])} | {j['结论']} |"
        out.append(row)
    return out


def report(res: dict, settings: dict) -> str:
    """报告（中文，给不会编程的人看）。只描述结果，不写“推荐下注”这类话。"""
    s = res["summary"]
    base = settings["direction_study"]["base_model"]
    f, p = s["定参数赛季"], s["检验赛季"]
    L = ["# 选方向研究", "",
         f"规则：需求文档第 4 节“选方向研究怎么做”“事先定的比较”（跑之前写下，跑完没改）。"
         f"五大联赛 T-24、有让球主盘口的比赛；{f[0]}–{f[1]} 定参数（{s['场数']['定参数']} 场），"
         f"{p[0]}–{p[1]} 检验（{s['场数']['检验']} 场）。期末考试赛季没读。", "",
         "差 = 候选命中率 − 照赔率选的命中率（百分点）；区间是按周成块 bootstrap 的 95% 区间。"
         "“过了”要三条都过：区间下限 > 0、四个检验赛季至少三个差 > 0、差至少 +0.5 个百分点。", "", "## 结论", ""]
    for m in ("ah", "1x2"):
        got = [f"{x}（{s['检验'][x][m]['判断']['结论']}）" for x in CANDIDATES
               if s["检验"][x][m]["判断"]["结论"].startswith("过了")]
        L.append(f"- {MARKET_NAMES[m]}：" + ("、".join(got) if got else "五个候选都没看到比照赔率选更常选对。"))
    L += [f"- 实时伤停来源：{s['伤停来源']}", "", "## 让球（主盘口）：检验季", "", *market_table(s, "ah", True),
          "", "## 胜平负：检验季", "", *market_table(s, "1x2", True),
          "", "## 大小球（主盘口）：检验季（只报告，不下结论）", "", *market_table(s, "ou", False), "",
          "## 对照（只报告）", "", "| 选法 | 玩法 | 检验季命中率 | 差（百分点） | 95% 区间 |", "|---|---|---|---|---|"]
    names = {"market": "照赔率选", base: f"{base} 自己", "让球价格": "让球价格（去水后赔率低的一边）"}
    for name, by in s["对照"].items():
        for m, r in by.items():
            L.append(f"| {names.get(name, name + ' 自己')} | {MARKET_NAMES[m]} | {pct(r['命中率'])} | {pp(r['差'])} | {ci(r['区间'])} |")
    L += ["", "## 每个检验赛季的差（百分点）", "", "| 候选 | 玩法 | " + " | ".join(str(x) for x in s["每个检验赛季的场数"]) + " |",
          "|---|---|" + "---|" * len(s["每个检验赛季的场数"])]
    for x in CANDIDATES:
        for m in MARKETS:
            L.append(f"| {x} | {MARKET_NAMES[m]} | " + " | ".join(pp(v) for v in s["检验"][x][m]["每个赛季的差"].values()) + " |")
    L += ["", "## log loss（检验季，候选 − 赔率；负数 = 候选的概率更准；C 没有自己的概率，不适用）", "",
          "| 候选 | 玩法 | 赔率 | 候选 | 差 | 95% 区间 |", "|---|---|---|---|---|---|"]
    for x, by in s["log loss"].items():
        for m, r in by.items():
            L.append(f"| {x} | {MARKET_NAMES[m]} | {r['赔率']:.4f} | {r['候选']:.4f} | {r['差（候选 − 赔率）']:+.4f} "
                     f"| {r['区间'][0]:+.4f} 到 {r['区间'][1]:+.4f} |")
    cal = res["calibration"]
    L += ["", "## 校准（检验季，全部比赛：写的胜算和实际命中率）", "", "| 来源 | 玩法 | 写的胜算 | 实际命中率 | 差 |", "|---|---|---|---|---|"]
    for _, r in cal[cal["档"] == "全部"].iterrows():
        L.append(f"| {names.get(r['来源'], CANDIDATES.get(r['来源'], r['来源']))} | {MARKET_NAMES[r['玩法']]} | {pct(r['写的胜算'])} | {pct(r['实际命中率'])} | {pp(r['差（实际 − 写的）'])} |")
    L += ["", "分档的校准见 calibration.csv；按联赛、分歧大小、伤停改了多少分开的差见 groups.csv（只是看看，不能拿来挑结论）。",
          "", "## 事先知道的限制", "",
          "- 伤停的开始日期是事后整理的，可能比消息公开早一两天；表是 2026 年的版本；停赛不在表里。所以 D、E 的结果是上限。",
          "- 回测是 T-24；上线用开球前最后一版赔率，赔率更准，模型能补的只会更少。",
          "- 这只是“选方向选对的比例”，不算赔率和盈亏，不说明能赚钱。", ""]
    return "\n".join(L)


def write(res: dict, out: Path, settings: dict, extra: dict) -> None:
    """结果只写进 out 这一个文件夹。"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res["summary"], ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out / "report.md").write_text(report(res, settings), encoding="utf-8")
    for k in ("picks", "grid", "calibration", "groups"):
        res[k].to_csv(out / f"{k}.csv", index=False, encoding="utf-8-sig")
    info = {"时间（UTC）": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "设置": {k: settings[k] for k in ("direction_study", "compare", "seasons")}, **extra}
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="选方向研究（需求文档第 4 节）")
    ap.add_argument("--root", default=None, help="项目文件夹（数据在它的 data/ 下）；不给就用程序所在的文件夹")
    args = ap.parse_args(argv)
    settings = load_settings()
    root = Path(args.root) if args.root else resolve_path(".")
    d = load(settings, root)
    res = study(d, settings)
    out = root / settings["direction_study"]["output_dir"]
    write(res, out, settings, {"检查": d.checks})
    print(report(res, settings))
    print(f"结果写在 {out}")


if __name__ == "__main__":
    main()
