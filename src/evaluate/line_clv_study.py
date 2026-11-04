'''亚盘、大小球收盘线研究（需求文档第 14 节“亚盘、大小球收盘线研究怎么做”：规则先写下，再写这个程序，再跑）。

一、要回答什么
  模型和赛前盘口意见不一样时，收盘会不会往模型那边走？
  走得够多、够稳，才说明模型知道一些市场赛前还没算进去的东西。bots_v1 已经答了胜平负，这里只做亚盘和大小球，分五大联赛、次级联赛两组。
  只是研究：虚拟注、不连账户、不写下注建议，不改任何上线的东西。

二、几个名词（都是“主队这边”或“大球这边”的胜算：赢半、输半折一半，走盘不算）
  m：平博赛前这条线去水；p：模型比分矩阵在同一条线上的胜算；a：多家平均赛前同一条线去水；
  c：收盘时同一条线的胜算（亚盘收盘线变了的，用收盘胜平负 + 大小球反推的市场矩阵量“两条线差多少”，再加到收盘去水上）。
  回归：y = c − m，x₁ = p − m，x₂ = a − m，y = b₀ + b₁·x₁ + b₂·x₂。b₁ > 0 = 控制了别家价格以后，收盘还往模型那边走。
  CLV = 赛前赔率 × 收盘时下的那一边的胜算 − 1（> 0 = 拿到了比收盘好的价格）。

三、数据（全部经取数层，开发模式）
  概率：data/backtest/<回测>/predictions.csv 和 matrices_gb_l7.npz，读进来先按赛季筛、再查全是开发赛季。
  赛前价格：用途「机器人下注」，开球前 24 小时，每一行“能知道的时间”不晚于截止时间。
  收盘价格、比分：用途「机器人结算」，开球后 72 小时。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.line_clv_study
结果放在 data/backtest/line_clv/：verdict.csv 主判断；beta.csv 回归（主判断和只看看的）；bots.csv 虚拟注成绩；
  approx_check.csv 亚盘换线近似的核对；per_match.csv 每场一行；report.md 中文报告；run_info.json
'''
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.common.markets import asian_handicap, dc_matrices, fit_market_dc, over_under
from src.common.odds import devig
from src.common.settlement import settle_asian_handicap, settle_over_under
from src.evaluate.compare import block_ids
from src.evaluate.market_checks import ratio_ci
from src.evaluate.odds_move_study import boot_mean
from src.features import FeatureStore, LookAheadError, open_readonly
from src.features.known_time import to_utc_text

MARKETS = {"ah": "亚盘", "ou": "大小球"}
SIDE_A = {"ah": "home", "ou": "over"}      # “这一边”：亚盘主队、大小球大球
SIDE_B = {"ah": "away", "ou": "under"}     # 另一边
PRICE = {"ah": ("home", "away"), "ou": ("over", "under")}   # odds 表里两个赔率的列名

# ======================= 1. 概率 =======================
def load_group(name: str, g: dict, settings: dict, run_dir=None) -> tuple[pd.DataFrame, np.ndarray]:
    '''读一组联赛的回测概率和比分矩阵 → (每场一行：match_id、competition、season、match_date、cutoff_utc, 矩阵数组)。

    第一步按赛季筛（考试赛季一行都不往下传），再检查剩下的全是开发赛季，否则停下报错。
    矩阵算出的胜平负和 predictions.csv 差太多（两个文件不是同一次回测）也停下。run_dir 给了就从那里读（测试用）。
    '''
    c = settings["line_clv_study"]
    if int(c["last_season"]) > int(settings["seasons"]["dev_last"]):
        raise ValueError(f"line_clv_study.last_season = {c['last_season']} 超过了开发赛季的最后一季，不能用考试赛季")
    d = resolve_path(settings["backtest"]["output_dir"]) / g["run"] if run_dir is None else run_dir
    cols = ["match_id", "competition", "season", "match_date", "horizon", "cutoff_utc", "model",
            "p_home", "p_draw", "p_away"]
    pred = pd.read_csv(d / "predictions.csv", usecols=cols, dtype={"cutoff_utc": str, "match_date": str})
    pred = pred[(pred["model"] == c["model"]) & (pred["horizon"] == c["horizon"])]
    pred = pred[pred["season"].between(int(c["first_season"]), int(c["last_season"]))]
    bad = sorted(int(s) for s in pred["season"].unique() if season_role(int(s), settings) != "dev")
    if bad:
        raise ValueError(f"{name}：筛完以后还有不是开发赛季的行 {bad}，停下")
    pred = pred[pred["competition"].isin(g["leagues"])].dropna(subset=["p_home"]).drop_duplicates("match_id")
    npz = np.load(d / f"matrices_{c['model']}.npz")
    pos = pd.Series(np.arange(len(npz["match_id"])), index=npz["match_id"])
    out = pred[pred["match_id"].isin(pos.index)].sort_values("match_id", ignore_index=True)
    mats = npz["matrix"][pos.loc[out["match_id"]].to_numpy()].astype(float)
    mats /= mats.sum(axis=(1, 2), keepdims=True)
    i, j = np.indices(mats.shape[1:])
    p_home = (mats * (i > j)).sum(axis=(1, 2))
    gap = float(np.abs(p_home - out["p_home"].to_numpy(float)).max()) if len(out) else 0.0
    if gap > 1e-4:
        raise ValueError(f"{name}：比分矩阵算出的主胜和 predictions.csv 差了 {gap:.2e}，两个文件不是同一次回测？")
    out["group"] = name
    return out[["match_id", "group", "competition", "season", "match_date", "cutoff_utc"]], mats


def second_check(settings: dict, run_dir=None) -> dict:
    '''次级联赛重跑的回测先和 L2 的数核对：gb_l7 T-24 和平博收盘同一批比赛，模型的胜平负 log loss 差不超过 tol。'''
    c = settings["line_clv_study"]["second_check"]
    d = resolve_path(settings["backtest"]["output_dir"]) / c["run"] if run_dir is None else run_dir
    pred = pd.read_csv(d / "predictions.csv", usecols=["match_id", "horizon", "model", "log_loss"])
    a = pred[(pred["model"] == settings["line_clv_study"]["model"]) & (pred["horizon"] == "T-24")].dropna()
    b = pred[(pred["model"] == "pinnacle_close") & (pred["horizon"] == "收盘")].dropna()
    both = a.merge(b, on="match_id", suffixes=("", "_bench"))
    got = float(both["log_loss"].mean())
    ok = abs(got - float(c["log_loss"])) <= float(c["tol"])
    info = {"重跑的 log loss": round(got, 5), "L2 的数": c["log_loss"], "场数": int(len(both)), "对得上": bool(ok)}
    if not ok:
        raise ValueError(f"次级联赛重跑的回测和 L2 对不上：{info}，停下查")
    return info


# ======================= 2. 赔率和比分 =======================
def collect(fs: FeatureStore, matches: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, dict]:
    '''每场取赛前（开球前 24 小时能看到的）、收盘、比分 → 每场一行的原始价格表。

    列：match_id、cutoff_utc、pre_<公司简称>_<玩法>_line/x/y（x、y 是两个赔率：亚盘主、客，大小球大、小）、
    pre_<公司简称>_1x2_h/d/a、close_pin_ah_line/x/y、close_pin_ou_line/x/y、close_pin_1x2_h/d/a、home_goals、away_goals。
    '''
    c = settings["line_clv_study"]
    pin, avg = c["book"], c["consensus_book"]
    short = {pin: "pin", avg: "avg"}
    after = pd.Timedelta(hours=float(c["hours_before_kickoff"]) + float(c["settle_after_kickoff_hours"]))
    rows, n_unknown, n_cut_diff = [], 0, 0
    for mid, old_cut in matches[["match_id", "cutoff_utc"]].itertuples(index=False):
        try:
            cut = fs.cutoff_before(int(mid), float(c["hours_before_kickoff"]), c["usage_bet"])
        except ValueError:                 # 数据库里没有这场（取数层不认）
            n_unknown += 1
            continue
        if isinstance(old_cut, str) and old_cut and to_utc_text(old_cut) != cut:
            n_cut_diff += 1                # 只数，不停：回测的截止时间和现在算的不一样，说明数据库变过
        r = {"match_id": int(mid), "cutoff_utc": cut}
        o = fs.pre_match_odds(int(mid), cut, c["usage_bet"], bookmaker=[pin, avg], market=["ah", "ou", "1x2"])
        if len(o) and (o["known_at_utc"].astype(str) > cut).any():
            raise LookAheadError(f"取数层返回了截止时间 {cut} 之后才能知道的赔率（比赛 {mid}），停下")
        o = o[o["source"] == c["odds_source"]]
        for x in o.itertuples(index=False):
            k = f"pre_{short[x.bookmaker]}_{x.market}"
            if x.market == "1x2":          # 赛前胜平负只用来算“网站倾向”那条只看看的规则
                r[k + "_h"], r[k + "_d"], r[k + "_a"] = x.home, x.draw, x.away
                continue
            a_col, b_col = PRICE[x.market]
            r[k + "_line"], r[k + "_x"], r[k + "_y"] = x.line, getattr(x, a_col), getattr(x, b_col)
        when = to_utc_text(pd.Timestamp(cut) + after)
        cl = fs.closing_odds(int(mid), when, c["usage_settle"], bookmaker=pin, market=["ah", "ou", "1x2"])
        cl = cl[cl["source"] == c["odds_source"]]
        for x in cl.itertuples(index=False):
            if x.market == "1x2":
                r["close_pin_1x2_h"], r["close_pin_1x2_d"], r["close_pin_1x2_a"] = x.home, x.draw, x.away
            else:
                a_col, b_col = PRICE[x.market]
                k = f"close_pin_{x.market}"
                r[k + "_line"], r[k + "_x"], r[k + "_y"] = x.line, getattr(x, a_col), getattr(x, b_col)
        res = fs.match_result(int(mid), when, c["usage_settle"])
        if not res.empty and pd.notna(res.iloc[0]["home_goals"]):
            r["home_goals"], r["away_goals"] = int(res.iloc[0]["home_goals"]), int(res.iloc[0]["away_goals"])
        rows.append(r)
    return pd.DataFrame(rows), {"取数层不认的比赛": n_unknown, "截止时间和回测不一样的比赛": n_cut_diff}

# ======================= 3. 每场的概率 =======================
def two_way(x, y, rng, method: str) -> np.ndarray:
    '''两个赔率 → “这一边”的去水胜算；赔率不全、≤ 1，或倒数之和不在 rng 范围（当作录错）的是 nan。'''
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (np.nan_to_num(x) > 1) & (np.nan_to_num(y) > 1)
    out = np.full(len(x), np.nan)
    if ok.any():
        s = 1 / x[ok] + 1 / y[ok]
        good = (s >= rng[0]) & (s <= rng[1])
        p = devig(np.stack([x[ok], y[ok]], axis=1), method)[:, 0]
        out[np.flatnonzero(ok)[good]] = p[good]
    return out


def three_way_ok(o3: np.ndarray, rng) -> np.ndarray:
    '''胜平负三个赔率都有、都 > 1、倒数之和在 rng 范围。'''
    ok = np.isfinite(o3).all(axis=1) & (np.nan_to_num(o3) > 1).all(axis=1)
    s = np.where(ok, (1 / np.where(ok[:, None], o3, 2.0)).sum(axis=1), 0.0)
    return ok & (s >= rng[0]) & (s <= rng[1])


def market_mats(o3: np.ndarray, over: np.ndarray, ok: np.ndarray, settings: dict) -> dict[int, np.ndarray]:
    '''胜平负赔率 + 大 2.5 球去水胜算 → 市场比分矩阵（fit_market_dc，和融合同一个方法）。返回 {行号: 矩阵}，只算 ok 的行。'''
    idx = np.flatnonzero(ok & np.isfinite(over))
    if len(idx) == 0:
        return {}
    c = settings["line_clv_study"]
    p3 = devig(o3[idx], c["devig_method"])
    n = int(settings["market_odds"]["max_goals"])
    lh, la, rho = fit_market_dc(np.column_stack([p3, over[idx]]), n=n, ou_line=float(c["ou_line"]))
    m = dc_matrices(lh, la, rho, n)
    return {int(k): m[t] for t, k in enumerate(idx)}


def share(mat, market: str, line: float) -> float:
    '''比分矩阵在这条线上“这一边”（亚盘主队、大小球大球）的胜算。'''
    f = asian_handicap if market == "ah" else over_under
    return float(f(mat, float(line), SIDE_A[market])["win_share"])


def expected(mat, market: str, line: float, side: str, odds: float) -> float:
    '''模型比分矩阵算的期望盈亏（每 1 单位本金，含走盘、赢半、输半），用实际赔率。'''
    f = asian_handicap if market == "ah" else over_under
    return float(f(mat, float(line), side, odds)["expected_profit"])


def settle(market: str, hg: int, ag: int, line: float, side: str, odds: float) -> tuple[str, float]:
    '''按真实比分结算 → (结果, 每 1 单位本金的盈亏)。'''
    f = settle_asian_handicap if market == "ah" else settle_over_under
    r = f(int(hg), int(ag), float(line), side, float(odds))
    return r["outcome"], float(r["profit"])


def per_match(raw: pd.DataFrame, info: pd.DataFrame, model_mats: dict[int, np.ndarray], settings: dict
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''原始价格表 → 每场 × 玩法一行：m、p、a、c，两边的赔率、模型期望盈亏、收盘胜算、真实结算，另加只看看的“网站倾向”胜算。

    返回 (每场一行的表, 亚盘换线近似的核对表)。
    '''
    c = settings["line_clv_study"]
    q = settings["quality"]["overround"]
    meth = c["devig_method"]
    lo_clip, hi_clip = c["clip"]
    df = raw.merge(info.drop(columns="cutoff_utc", errors="ignore"), on="match_id", how="left")
    for col in ["pre_pin_ah_line", "pre_pin_ah_x", "pre_pin_ah_y", "pre_avg_ah_line", "pre_avg_ah_x", "pre_avg_ah_y",
                "pre_pin_ou_line", "pre_pin_ou_x", "pre_pin_ou_y", "pre_avg_ou_line", "pre_avg_ou_x", "pre_avg_ou_y",
                "pre_pin_1x2_h", "pre_pin_1x2_d", "pre_pin_1x2_a", "close_pin_ah_line", "close_pin_ah_x",
                "close_pin_ah_y", "close_pin_ou_line", "close_pin_ou_x", "close_pin_ou_y",
                "close_pin_1x2_h", "close_pin_1x2_d", "close_pin_1x2_a", "home_goals", "away_goals"]:
        if col not in df:
            df[col] = np.nan
    # 收盘、赛前的市场矩阵（收盘的给亚盘换线用；赛前的只给“网站倾向”用）
    c3 = df[["close_pin_1x2_h", "close_pin_1x2_d", "close_pin_1x2_a"]].to_numpy(float)
    c_over = two_way(df["close_pin_ou_x"], df["close_pin_ou_y"], q["2way"], meth)
    close_mats = market_mats(c3, c_over, three_way_ok(c3, q["1x2"]), settings)
    p3 = df[["pre_pin_1x2_h", "pre_pin_1x2_d", "pre_pin_1x2_a"]].to_numpy(float)
    p_over = two_way(df["pre_pin_ou_x"], df["pre_pin_ou_y"], q["2way"], meth)
    pre_mats = market_mats(p3, p_over, three_way_ok(p3, q["1x2"]), settings)

    rows, approx = [], []
    for mk in MARKETS:
        m = two_way(df[f"pre_pin_{mk}_x"], df[f"pre_pin_{mk}_y"], q["2way"], meth)
        a = two_way(df[f"pre_avg_{mk}_x"], df[f"pre_avg_{mk}_y"], q["2way"], meth)
        cd = two_way(df[f"close_pin_{mk}_x"], df[f"close_pin_{mk}_y"], q["2way"], meth)
        for i, r in enumerate(df.itertuples(index=False)):
            line = getattr(r, f"pre_pin_{mk}_line")
            mat = model_mats.get(int(r.match_id))
            if not np.isfinite(m[i]) or mat is None or not np.isfinite(line):
                continue
            a_i = a[i] if getattr(r, f"pre_avg_{mk}_line") == line else np.nan   # 多家平均要同一条线才能比
            cline = getattr(r, f"close_pin_{mk}_line")
            moved = bool(np.isfinite(cline) and cline != line)
            c_i = np.nan
            if np.isfinite(cd[i]) and np.isfinite(cline):
                if not moved:
                    c_i = cd[i]
                elif i in close_mats:
                    cm = close_mats[i]
                    c_i = float(np.clip(cd[i] + share(cm, mk, line) - share(cm, mk, cline), lo_clip, hi_clip))
                if mk == "ah" and not moved and i in close_mats:   # 核对换线近似：线没变的也用矩阵算一次
                    approx.append({"group": r.group, "direct": cd[i], "matrix": share(close_mats[i], mk, line)})
            ox, oy = getattr(r, f"pre_pin_{mk}_x"), getattr(r, f"pre_pin_{mk}_y")
            row = {"match_id": int(r.match_id), "group": r.group, "competition": r.competition,
                   "season": int(r.season), "match_date": r.match_date, "market": mk, "line": float(line),
                   "close_line": cline, "moved": moved, "m": m[i], "p": share(mat, mk, line), "a": a_i, "c": c_i,
                   "odds_a": float(ox), "odds_b": float(oy),
                   "ev_a": expected(mat, mk, line, SIDE_A[mk], ox), "ev_b": expected(mat, mk, line, SIDE_B[mk], oy),
                   "site": share(pre_mats[i], mk, line) if i in pre_mats else np.nan}
            if np.isfinite(r.home_goals):
                for s, side, odds in (("a", SIDE_A[mk], ox), ("b", SIDE_B[mk], oy)):
                    row[f"out_{s}"], row[f"profit_{s}"] = settle(mk, r.home_goals, r.away_goals, line, side, odds)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["clv_a"] = out["odds_a"] * out["c"] - 1
    out["clv_b"] = out["odds_b"] * (1 - out["c"]) - 1
    return out, pd.DataFrame(approx, columns=["group", "direct", "matrix"])

# ======================= 4. 回归：收盘往模型那边走了吗 =======================
def ols_boot(y, X, dates, settings: dict, boot: bool = True) -> dict:
    '''最小二乘 y = b₀ + X·b，返回 b₁（X 第一列的系数）和它的 95% 区间（按周成块 bootstrap：抽周、重新拟合）。

    做法：每周先把 XᵀX、Xᵀy 加好，抽周以后把抽到的周的加起来再解方程，和逐场重新拟合结果一样，但快得多。
    '''
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    out = {"场数": int(len(y)), "b1": np.nan, "b1_下限": np.nan, "b1_上限": np.nan, "b2": np.nan}
    if len(y) < X.shape[1] + 2:
        return out
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    out["b1"] = float(b[1])
    out["b2"] = float(b[2]) if len(b) > 2 else np.nan
    if not boot:
        return out
    c = settings["compare"]
    _, inv = np.unique(block_ids(pd.Series(dates), c["block"]), return_inverse=True)
    k, d = int(inv.max()) + 1, X.shape[1]
    xtx = np.zeros((k, d, d))
    xty = np.zeros((k, d))
    np.add.at(xtx, inv, X[:, :, None] * X[:, None, :])
    np.add.at(xty, inv, X * y[:, None])
    rng = np.random.default_rng(c["seed"])
    b1 = np.empty(int(c["n_boot"]))
    for t in range(len(b1)):
        idx = rng.integers(0, k, size=k)
        b1[t] = (np.linalg.pinv(xtx[idx].sum(axis=0)) @ xty[idx].sum(axis=0))[1]
    alpha = (1 - float(c["confidence"])) / 2
    out["b1_下限"], out["b1_上限"] = (float(v) for v in np.quantile(b1, [alpha, 1 - alpha]))
    return out


def beta_table(pm: pd.DataFrame, settings: dict) -> pd.DataFrame:
    '''每组 × 玩法：主判断的回归（有 x₂），加只看看的（不放 x₂、按赛季、按联赛、亚盘线变没变）。

    用的场：m、p、a、c 四个都有的。所有分法用同一批场，方便对照。
    '''
    rows = []
    for (grp, mk), s in pm.groupby(["group", "market"], sort=False):
        s = s.dropna(subset=["m", "p", "a", "c"])
        y, x1, x2 = s["c"] - s["m"], s["p"] - s["m"], s["a"] - s["m"]
        base = {"组": grp, "玩法": MARKETS[mk]}
        rows.append({**base, "分法": "主判断", "取值": "全部（有 x₂）",
                     **ols_boot(y, np.column_stack([x1, x2]), s["match_date"], settings)})
        rows.append({**base, "分法": "不放 x₂", "取值": "全部",
                     **ols_boot(y, x1.to_numpy()[:, None], s["match_date"], settings)})
        parts = [("赛季", str(k), v) for k, v in s.groupby("season")]
        parts += [("联赛", k, v) for k, v in s.groupby("competition")]
        if mk == "ah":
            parts += [("线变没变", "变了" if k else "没变", v) for k, v in s.groupby("moved")]
        for how, val, v in parts:
            X = np.column_stack([v["p"] - v["m"], v["a"] - v["m"]])
            rows.append({**base, "分法": how, "取值": val,
                         **ols_boot(v["c"] - v["m"], X, v["match_date"], settings, boot=how != "赛季")})
    return pd.DataFrame(rows)

# ======================= 5. 虚拟注 =======================
def picks(s: pd.DataFrame, settings: dict) -> dict[str, np.ndarray]:
    '''每条规则每场买哪边：1 = 这一边（主队 / 大球），-1 = 另一边，0 = 不下。挑选只用赛前能知道的东西（模型、平博赛前赔率）。'''
    c = settings["line_clv_study"]
    ev_a, ev_b = s["ev_a"].to_numpy(float), s["ev_b"].to_numpy(float)
    best = np.where(ev_a >= ev_b, 1, -1)
    big = np.maximum(ev_a, ev_b)
    out = {f"模型价值 {round(e * 100)}%": np.where(big >= e, best, 0)
           for e in [float(c["main_edge"])] + [float(v) for v in c["look_edges"]]}
    p = s["p"].to_numpy(float)
    out["跟模型"] = np.where(p > 0.5, 1, np.where(p < 0.5, -1, 0))
    seed = int(c["random_seed"])
    out["随机"] = np.array([1 if np.random.default_rng([seed, int(mid)]).random() < 0.5 else -1
                          for mid in s["match_id"]], dtype=int)
    out["总买主队/大球"] = np.ones(len(s), dtype=int)
    lean = float(c["site_lean"])
    site = s["site"].to_numpy(float)
    out["网站倾向"] = np.where(site >= 0.5 + lean, 1, np.where(site <= 0.5 - lean, -1, 0))
    return out


def bot_stats(s: pd.DataFrame, pick: np.ndarray, settings: dict) -> dict:
    '''一条规则的成绩：CLV 平均和区间、和同一批比赛买另一边的 CLV 差、ROI 和区间、赢的比例。'''
    on = pick != 0
    t = s[on]
    side = pick[on]
    clv = np.where(side == 1, t["clv_a"], t["clv_b"])
    other = np.where(side == 1, t["clv_b"], t["clv_a"])
    out = {"下注场数": int(on.sum())}
    ok = np.isfinite(clv)
    out["有收盘的场数"] = int(ok.sum())
    if ok.sum() >= 2:
        out["CLV"], out["CLV_下限"], out["CLV_上限"] = boot_mean(clv[ok], t["match_date"][ok], settings)
        out["比另一边"], out["比另一边_下限"], out["比另一边_上限"] = boot_mean(
            (clv - other)[ok], t["match_date"][ok], settings)
    if "profit_a" in t:
        prof = np.where(side == 1, t["profit_a"], t["profit_b"]).astype(float)
        outc = np.where(side == 1, t["out_a"], t["out_b"])
        done = np.isfinite(prof)
        out["结算场数"] = int(done.sum())
        if done.sum() >= 2:
            out["ROI"], out["ROI_下限"], out["ROI_上限"] = ratio_ci(prof[done], np.ones(done.sum()),
                                                                 t["match_date"][done], settings)
            res = pd.Series(outc[done])
            decided = res[res != "push"]
            out["赢的比例"] = float(decided.isin(["win", "half_win"]).mean()) if len(decided) else np.nan
    return out


def bot_table(pm: pd.DataFrame, settings: dict) -> pd.DataFrame:
    '''每组 × 玩法 × 规则一行。第一条规则（模型价值 main_edge）是主判断，其余只看看。'''
    rows = []
    for (grp, mk), s in pm.groupby(["group", "market"], sort=False):
        s = s.reset_index(drop=True)
        for name, pick in picks(s, settings).items():
            rows.append({"组": grp, "玩法": MARKETS[mk], "规则": name, **bot_stats(s, pick, settings)})
    return pd.DataFrame(rows)

# ======================= 6. 结论 =======================
def verdict(beta: pd.DataFrame, bots: pd.DataFrame, settings: dict) -> pd.DataFrame:
    '''主判断：4 组（亚盘 / 大小球 × 两组联赛）各一行，按事先写下的三条 + CLV 区间下结论。'''
    c = settings["line_clv_study"]
    main_bot = f"模型价值 {round(float(c['main_edge']) * 100)}%"
    rows = []
    for (grp, mk), b in beta.groupby(["组", "玩法"], sort=False):
        top = b[b["分法"] == "主判断"].iloc[0]
        ss = b[b["分法"] == "赛季"]
        n_pos = int((ss["b1"] > 0).sum())
        rule1 = bool(top["b1_下限"] > 0 and n_pos >= int(c["min_positive_seasons"]) and top["b1"] >= float(c["min_b1"]))
        bt = bots[(bots["组"] == grp) & (bots["玩法"] == mk) & (bots["规则"] == main_bot)].iloc[0]
        rule2 = bool(bt.get("CLV_下限", np.nan) > 0)
        if rule1 and rule2:
            words = "模型在这一组能预判收盘"
        elif rule1:
            words = "方向对，但不够抵掉抽水"
        elif rule2:
            words = "没看到（CLV 过了但回归没过，按规则不算）"
        else:
            words = "没看到"
        roi_lo, roi_hi = bt.get("ROI_下限", np.nan), bt.get("ROI_上限", np.nan)
        luck = "可能只是运气" if not (roi_lo > 0 or roi_hi < 0) else ("区间整个 > 0" if roi_lo > 0 else "区间整个 < 0")
        suspicious = bool(top["b1"] > float(c["suspicious_b1"]) or rule2)
        rows.append({"组": grp, "玩法": mk, "场数": int(top["场数"]), "b1": top["b1"], "b1_下限": top["b1_下限"],
                     "b1_上限": top["b1_上限"], "b2": top["b2"], "b1>0 的赛季数": n_pos, "赛季数": int(len(ss)),
                     "判断1 收盘往模型走": rule1, "下注场数": bt["下注场数"], "CLV": bt.get("CLV", np.nan),
                     "CLV_下限": bt.get("CLV_下限", np.nan), "CLV_上限": bt.get("CLV_上限", np.nan),
                     "判断2 价格比收盘好": rule2, "ROI": bt.get("ROI", np.nan), "ROI_下限": roi_lo, "ROI_上限": roi_hi,
                     "ROI 说明": luck, "结论": words, "要先查偷看未来": suspicious})
    return pd.DataFrame(rows)


def approx_table(ap: pd.DataFrame) -> pd.DataFrame:
    '''亚盘换线近似的核对：线没变的场，矩阵算的胜算 − 直接去水，平均差、平均绝对差。'''
    rows = []
    for grp, s in ap.groupby("group", sort=False):
        d = s["matrix"] - s["direct"]
        rows.append({"组": grp, "场数": int(len(s)), "平均差": float(d.mean()), "平均绝对差": float(d.abs().mean()),
                     "差的 95% 分位": float(d.abs().quantile(0.95))})
    return pd.DataFrame(rows)


def _f(v, nd=3) -> str:
    '''数字 → 带正负号的文字；没有的写空。'''
    if v is None or not np.isfinite(float(v)):
        return ""
    return f"{float(v):+.{nd}f}"


def _p(v) -> str:
    '''比例 → 百分数文字（+1.2%）；没有的写空。'''
    if v is None or not np.isfinite(float(v)):
        return ""
    return f"{float(v) * 100:+.1f}%"


def _ci(lo, hi, pct=True) -> str:
    g = _p if pct else _f
    return f"{g(lo)} ~ {g(hi)}" if g(lo) else ""


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)

def count_table(pm: pd.DataFrame, n_matches: dict[str, int]) -> pd.DataFrame:
    '''每组 × 玩法用到多少场：有模型和平博赛前价格的、有收盘的、四个概率都有（进回归）的、亚盘线变了的。'''
    rows = []
    for (grp, mk), s in pm.groupby(["group", "market"], sort=False):
        rows.append({"组": grp, "玩法": MARKETS[mk], "回测里的比赛": n_matches.get(grp, 0),
                     "有平博赛前价格": int(len(s)), "有收盘胜算": int(s["c"].notna().sum()),
                     "进回归（m p a c 都有）": int(s[["m", "p", "a", "c"]].notna().all(axis=1).sum()),
                     "收盘线变了": int(s["moved"].sum()), "有比分": int(s.get("profit_a", pd.Series(dtype=float)).notna().sum())})
    return pd.DataFrame(rows)


def report(ver: pd.DataFrame, beta: pd.DataFrame, bots: pd.DataFrame, approx: pd.DataFrame,
           counts: pd.DataFrame, info: dict) -> str:
    '''中文 markdown 报告：结论表在最前，后面是回归、虚拟注、只看看的分法、核对。'''
    out = ["# 亚盘、大小球收盘线研究", "",
           "规则见需求文档第 14 节“亚盘、大小球收盘线研究怎么做”（跑之前写下）。只是研究：虚拟注、平博价格、开发赛季 2019/20–2023/24。", "",
           "## 结论（主判断，4 组）", ""]
    out.append(md_table(
        ["组", "玩法", "场数", "b₁", "b₁ 区间", "b₁>0 的赛季", "判断1", "下注场数", "CLV", "CLV 区间", "判断2",
         "ROI", "ROI 区间", "结论"],
        [[r["组"], r["玩法"], r["场数"], _f(r["b1"]), _ci(r["b1_下限"], r["b1_上限"], False),
          f"{r['b1>0 的赛季数']}/{r['赛季数']}", "过" if r["判断1 收盘往模型走"] else "不过", r["下注场数"],
          _p(r["CLV"]), _ci(r["CLV_下限"], r["CLV_上限"]), "过" if r["判断2 价格比收盘好"] else "不过",
          _p(r["ROI"]), _ci(r["ROI_下限"], r["ROI_上限"]), r["结论"] + ("（ROI " + r["ROI 说明"] + "）")]
         for _, r in ver.iterrows()]))
    if ver["要先查偷看未来"].any():
        out += ["", "**注意**：有一组 b₁ > 0.5 或模型价值的 CLV 区间整个 > 0，按规则要先当作偷看未来去查，查完才能用这个结论。"]
    out += ["", "## 回归（y = c − m，x₁ = p − m，x₂ = a − m）", "",
            "b₁：模型比平博赛前多看 1 个百分点，收盘平均往模型那边走多少个百分点（×100 看）。b₂：多家平均的同样意思。", ""]
    look = beta[beta["分法"] != "赛季"]
    out.append(md_table(["组", "玩法", "分法", "取值", "场数", "b₁", "b₁ 区间", "b₂"],
                        [[r["组"], r["玩法"], r["分法"], r["取值"], r["场数"], _f(r["b1"]),
                          _ci(r["b1_下限"], r["b1_上限"], False), _f(r["b2"])] for _, r in look.iterrows()]))
    out += ["", "分赛季（只有点估计，用来数“5 个赛季里几个 b₁ > 0”）：", ""]
    ss = beta[beta["分法"] == "赛季"]
    out.append(md_table(["组", "玩法", "赛季", "场数", "b₁", "b₂"],
                        [[r["组"], r["玩法"], r["取值"], r["场数"], _f(r["b1"]), _f(r["b2"])] for _, r in ss.iterrows()]))
    out += ["", "## 虚拟注（平博赛前价格，每注 1 单位）", "",
            "第一条“模型价值 5%”是主判断，其余只看看（几十个数里碰巧有一两个不含 0 很正常）。"
            "“比另一边”：同一批比赛，买的这边 CLV − 另一边 CLV。", ""]
    out.append(md_table(["组", "玩法", "规则", "下注场数", "CLV", "CLV 区间", "比另一边", "ROI", "ROI 区间", "赢的比例"],
                        [[r["组"], r["玩法"], r["规则"], r["下注场数"], _p(r.get("CLV")),
                          _ci(r.get("CLV_下限"), r.get("CLV_上限")), _p(r.get("比另一边")), _p(r.get("ROI")),
                          _ci(r.get("ROI_下限"), r.get("ROI_上限")),
                          "" if not np.isfinite(r.get("赢的比例", np.nan)) else f"{r['赢的比例'] * 100:.1f}%"]
                         for _, r in bots.iterrows()]))
    out += ["", "## 核对", "", "亚盘换线的近似（线没变的场，市场矩阵算的胜算 − 直接去水；越接近 0 越说明换线的那部分算得可靠）：", ""]
    out.append(md_table(["组", "场数", "平均差", "平均绝对差", "差的 95% 分位"],
                        [[r["组"], r["场数"], _f(r["平均差"], 4), f"{r['平均绝对差']:.4f}", f"{r['差的 95% 分位']:.4f}"]
                         for _, r in approx.iterrows()]))
    out += ["", "场数：", ""]
    out.append(md_table(list(counts.columns), counts.astype(str).values.tolist()))
    out += ["", "其他：" + json.dumps(info, ensure_ascii=False)]
    return "\n".join(out) + "\n"


def study(fs: FeatureStore, settings: dict, run_dirs: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    '''整个研究：每组读概率 → 取价格 → 每场概率 → 回归、虚拟注 → 结论。run_dirs 给了就从那里读回测（测试用）。'''
    c = settings["line_clv_study"]
    pms, aps, n_matches, info = [], [], {}, {}
    for name, g in c["groups"].items():
        matches, mats = load_group(name, g, settings, (run_dirs or {}).get(name))
        n_matches[name] = int(len(matches))
        model_mats = {int(m): mats[i] for i, m in enumerate(matches["match_id"])}
        raw, got = collect(fs, matches, settings)
        info[name] = got
        if raw.empty:
            continue
        pm, ap = per_match(raw, matches, model_mats, settings)
        pms.append(pm)
        aps.append(ap)
    pm = pd.concat(pms, ignore_index=True)
    ap = pd.concat(aps, ignore_index=True)
    beta = beta_table(pm, settings)
    bots = bot_table(pm, settings)
    tables = {"verdict": verdict(beta, bots, settings), "beta": beta, "bots": bots,
              "approx_check": approx_table(ap), "counts": count_table(pm, n_matches), "per_match": pm}
    return tables, info


def main():
    ap = argparse.ArgumentParser(description="亚盘、大小球收盘线研究（需求文档第 14 节）")
    ap.add_argument("--db", default=None, help="数据库文件（不写就用 settings.yaml 的 paths.database）")
    ap.add_argument("--out", default=None, help="结果文件夹（不写就用 settings.yaml 的 line_clv_study.output_dir）")
    a = ap.parse_args()
    settings = load_settings()
    c = settings["line_clv_study"]
    check = second_check(settings)
    print("次级联赛回测核对：", check)
    fs = FeatureStore(open_readonly(a.db), "开发", settings)
    tables, info = study(fs, settings)
    info = {"次级联赛回测核对": check, "取数": info, "设置": c}
    out_dir = resolve_path(a.out or c["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.6f")
    text = report(tables["verdict"], tables["beta"], tables["bots"], tables["approx_check"], tables["counts"], info)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(tables["verdict"].to_string(index=False))
    print(tables["approx_check"].to_string(index=False))
    print(f"结果在 {out_dir}")


if __name__ == "__main__":
    main()
