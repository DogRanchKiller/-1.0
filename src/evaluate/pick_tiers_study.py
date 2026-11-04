"""四个玩法的“把握”等级（需求文档第 5 节“四个玩法的‘把握’等级怎么定”，跑之前写下）。

问的是：每场四个玩法（胜平负、让球、大小球、比分）都给一个方向，同样这么写的预测，历史上中了几成？
能不能分出“更有把握 / 一般 / 没什么把握”三档，每档写出来的可能性和实际命中率对不对得上？

做法（细节见需求文档）：
  1. 读回测 w9_gb_l7 的 T-24（五大联赛 2012–2023，都是开发赛季；考试赛季一行都不读）；
  2. 按 fusion.w（= 1，和网站一样）用赛前赔率反推市场比分矩阵；“模型自己”那列用 gb_l7 的比分矩阵；
  3. 每场四个方向 + 把握分：胜平负 = 最大那项的概率；让球 = 多家平均主盘口上胜算较大一边的胜算；
     大小球 = “推出来的主盘口”（0.5–6 每隔 0.25，大球胜算最接近 50% 的那条）上胜算较大一边的胜算；比分 = 最大 4 格合起来的概率；
  4. 2012–2019 的三等分点当切点（高 / 中 / 低），2020–2023 检验：高档 − 低档命中率的区间、每档的校准；
  5. 推出来的主盘口和懂球帝开球前的真主盘口核对一次（只报告，2018 起）。
结果写到 settings.yaml pick_tiers_study.output_dir：summary.json、report.md、picks.csv（每场一行）。

运行：.venv\\Scripts\\python -m src.evaluate.pick_tiers_study
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.common.markets import asian_handicap
from src.common.settlement import settle_asian_handicap, settle_margin, settle_over_under
from src.evaluate import fusion_study as fu
from src.evaluate.compare import load_predictions
from src.evaluate.picks_reference import fused_matrices, model_matrices
from src.web.site.picks import SCORE_TOP, VALUE, best_scores

CHANNELS = ("1x2", "ah", "ou", "score")
NAMES = {"1x2": "胜平负", "ah": "让球（亚盘）", "ou": "大小球（主盘口）", "score": f"比分 {SCORE_TOP} 个"}
TIERS = ("高", "中", "低")
SIDE_1X2 = ("主胜", "平", "客胜")


# ======================= 读数据 =======================
def load(settings: dict, run_dir=None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """读回测 → (每场一行的比赛表, 市场比分矩阵, 模型比分矩阵, 检查信息)。

    第一步按赛季、联赛筛（考试赛季一行都不往下传），再检查剩下的全是开发赛季，否则停下报错。
    比赛表、矩阵的筛法和融合研究（fusion_study.build）一样：有比分、有模型矩阵、胜平负和大 2.5 球赔率都有。
    """
    c = settings["pick_tiers_study"]
    if int(c["test_last"]) > int(settings["seasons"]["dev_last"]):
        raise ValueError(f"pick_tiers_study.test_last = {c['test_last']} 超过了开发赛季的最后一季，不能用考试赛季")
    d = resolve_path(settings["backtest"]["output_dir"]) / c["run"] if run_dir is None else run_dir
    full = load_predictions(d)
    pred = full[full["season"].between(int(c["first_season"]), int(c["test_last"]))
                & full["competition"].isin(c["leagues"])].reset_index(drop=True)
    bad = sorted(int(s) for s in pred["season"].unique() if season_role(int(s), settings) != "dev")
    if bad:
        raise ValueError(f"筛完以后还有不是开发赛季的行 {bad}，停下")
    npz = np.load(d / f"matrices_{settings['fusion_study']['model']}.npz")
    odds = pd.read_csv(d / "fusion_odds.csv", encoding="utf-8-sig")
    t, _model, _market, check = fu.build(pred, npz["match_id"], npz["matrix"], odds, settings)
    market = fused_matrices(t, npz["match_id"], npz["matrix"], float(settings["fusion"]["w"]))
    own = model_matrices(t, npz["match_id"], npz["matrix"])
    keep = ["match_id", "competition", "season", "match_date", "home_goals", "away_goals", "ah_line"]
    return t[keep].reset_index(drop=True), market, own, check


# ======================= 大小球：任意线上的胜算、推出来的主盘口 =======================
def ou_grid(settings: dict) -> np.ndarray:
    lo, hi, step = (float(x) for x in settings["pick_tiers_study"]["ou_lines"])
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def totals(mats: np.ndarray) -> np.ndarray:
    """每场总进球的分布（场数 × (2n+1)），第 k 列 = 总进球 k 的概率。最后一行 / 列是“n 个及以上”，和 over_under 一样当成 n。"""
    n = mats.shape[1] - 1
    i, j = np.indices(mats.shape[1:])
    return np.stack([(mats * (i + j == k)).sum(axis=(1, 2)) for k in range(2 * n + 1)], axis=1)


def over_shares(tot: np.ndarray, lines) -> np.ndarray:
    """每场在每条线上“大”的胜算（场数 × 线数）。胜算 = (赢 + 赢半×0.5) ÷ (赢 + 赢半×0.5 + 输 + 输半×0.5)，和网站一样。
    按真实规则结算（settle_margin）：买大 = 总进球 k、盘口 −线。"""
    out = np.empty((len(tot), len(lines)))
    for c, line in enumerate(lines):
        v = np.array([VALUE[settle_margin(k, -float(line))] for k in range(tot.shape[1])])
        up, down = tot @ np.clip(v, 0, None), tot @ np.clip(-v, 0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, c] = np.where(up + down > 0, up / np.where(up + down > 0, up + down, 1), np.nan)
    return out


def implied_lines(tot: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """推出来的主盘口：每场在 grid 里大球胜算最接近 50% 的那条（一样近取小的）。返回 (线, 这条线上大的胜算)。"""
    sh = over_shares(tot, grid)
    gap = np.where(np.isnan(sh), np.inf, np.abs(sh - 0.5))
    k = np.argmin(gap, axis=1)                     # argmin 一样时取前面的 = 小的线
    return grid[k], sh[np.arange(len(sh)), k]


# ======================= 每场四个方向、把握分、结果 =======================
def picks(t: pd.DataFrame, mats: np.ndarray, settings: dict, ou_line=None) -> pd.DataFrame:
    """每场一行：四个玩法的方向、把握分（p_<玩法>）、结算的分值（v_<玩法>：中 1、赢半 0.5、走 0、输半 −0.5、没中 −1）。

    ou_line 不给 = 用这批矩阵自己推出来的主盘口；给了（每场一个数）就用那条线（模型自己那列和市场用同一条线）。
    另算旧规则“大小球固定 2.5”的方向和分值（v_ou25），只做对照。
    """
    hg, ag = t["home_goals"].to_numpy(int), t["away_goals"].to_numpy(int)
    i, j = np.indices(mats.shape[1:])
    p3 = np.stack([(mats * (i > j)).sum((1, 2)), (mats * (i == j)).sum((1, 2)), (mats * (i < j)).sum((1, 2))], axis=1)
    side = np.argmax(p3, axis=1)                    # 一样大时取前面的：主胜 > 平 > 客胜
    real = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    out = pd.DataFrame({"match_id": t["match_id"].to_numpy(), "season": t["season"].to_numpy(),
                        "competition": t["competition"].to_numpy(), "match_date": t["match_date"].to_numpy(),
                        "home_goals": hg, "away_goals": ag})
    out["pick_1x2"] = [SIDE_1X2[s] for s in side]
    out["p_1x2"] = p3[np.arange(len(p3)), side]
    out["v_1x2"] = np.where(side == real, 1.0, -1.0)

    # 让球：多家平均主盘口（fusion_study.build 已经把不是主盘口的线换成空值）
    pa, va, sa = [], [], []
    for k, line in enumerate(t["ah_line"].to_numpy(float)):
        if np.isnan(line):
            pa.append(np.nan), va.append(np.nan), sa.append(None)
            continue
        share = asian_handicap(mats[k], float(line), "home")["win_share"]
        s = "home" if share >= 0.5 else "away"
        pa.append(share if s == "home" else 1 - share), sa.append(s)
        va.append(VALUE[settle_asian_handicap(int(hg[k]), int(ag[k]), float(line), s)["outcome"]])
    out["ah_line"], out["pick_ah"], out["p_ah"], out["v_ah"] = t["ah_line"].to_numpy(float), sa, pa, va

    # 大小球：主盘口
    tot = totals(mats)
    if ou_line is None:
        line, over = implied_lines(tot, ou_grid(settings))
    else:
        line = np.asarray(ou_line, float)
        over = np.array([over_shares(tot[k:k + 1], [line[k]])[0, 0] for k in range(len(tot))])
    s_ou = np.where(over >= 0.5, "over", "under")
    out["ou_line"], out["pick_ou"] = line, s_ou
    out["p_ou"] = np.where(s_ou == "over", over, 1 - over)
    out["v_ou"] = [VALUE[settle_over_under(int(h), int(a), float(ln), s)["outcome"]] for h, a, ln, s in zip(hg, ag, line, s_ou)]
    over25 = over_shares(tot, [2.5])[:, 0]
    out["v_ou25"] = np.where((over25 >= 0.5) == (hg + ag > 2), 1.0, -1.0)

    # 比分：最大的 SCORE_TOP 格（和网站同一个函数）
    best = [best_scores(m) for m in mats]
    out["pick_score"] = ["/".join(f"{h}-{a}" for h, a in b["scores"]) for b in best]
    out["p_score"] = [b["p"] for b in best]
    out["v_score"] = [1.0 if (int(h), int(a)) in set(b["scores"]) else -1.0 for h, a, b in zip(hg, ag, best)]
    return out


# ======================= 分档、命中率、bootstrap =======================
def cuts(p: pd.Series) -> list[float]:
    """把握分的三等分点（1/3、2/3 分位数），保留 3 位小数。只用定切点的赛季算。"""
    q = p.dropna().quantile([1 / 3, 2 / 3]).to_numpy(float)
    return [round(float(q[0]), 3), round(float(q[1]), 3)]


def tier(p: pd.Series, cut: list[float]) -> pd.Series:
    """把握分 ≥ 上切点 = 高，< 下切点 = 低，中间 = 中。没有把握分（这场没有这个玩法）是空。"""
    t = np.where(p >= cut[1], "高", np.where(p >= cut[0], "中", "低"))
    return pd.Series(np.where(p.isna(), None, t), index=p.index)


def hit_rate(v) -> float:
    """命中率 = 赢的分 ÷ 有输赢的分（走盘 0 分不算），和战绩页一样。没有可算的是 nan。"""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    total = np.abs(v).sum()
    return float(np.clip(v, 0, None).sum() / total) if total > 0 else float("nan")


def week_index(dates: pd.Series) -> np.ndarray:
    """按周分块：同一周（周一到周日）的比赛在一块。"""
    d = pd.to_datetime(dates)
    return ((d - pd.Timestamp("2000-01-03")).dt.days // 7).to_numpy()


def tier_stats(df: pd.DataFrame, ch: str, settings: dict) -> dict:
    """一个玩法在 df（检验季）上的每档：场数、命中率、平均把握分、差（平均把握分 − 命中率）及区间；高 − 低 的区间。

    按周成块 bootstrap：每次有放回地抽同样多的周，抽到的周里每档的分加起来再算。
    """
    c = settings["pick_tiers_study"]
    d = df[df[f"tier_{ch}"].notna() & df[f"v_{ch}"].notna()]
    weeks = week_index(d["match_date"])
    uw, wk = np.unique(weeks, return_inverse=True)
    agg = {}
    for tr in TIERS:
        m = (d[f"tier_{ch}"] == tr).to_numpy()
        v, p = d[f"v_{ch}"].to_numpy(float), d[f"p_{ch}"].to_numpy(float)
        agg[tr] = np.stack([np.bincount(wk, (np.clip(v, 0, None)) * m, len(uw)),
                            np.bincount(wk, np.abs(v) * m, len(uw)),
                            np.bincount(wk, p * m, len(uw)), np.bincount(wk, m.astype(float), len(uw))])
    rng = np.random.default_rng(int(c["random_seed"]))
    draws = rng.integers(0, len(uw), size=(int(c["boot"]), len(uw)))
    counts = np.stack([np.bincount(r, minlength=len(uw)) for r in draws])          # 每次抽到每周几次
    out, boot_rate = {}, {}
    for tr in TIERS:
        a = agg[tr]
        s = counts @ a.T                                                           # (次数, 4)
        with np.errstate(invalid="ignore", divide="ignore"):
            br, bp = s[:, 0] / s[:, 1], s[:, 2] / s[:, 3]
        boot_rate[tr] = br
        n = int(a[3].sum())
        rate = a[0].sum() / a[1].sum() if a[1].sum() else np.nan        # 命中率（走盘不算）
        mp = a[2].sum() / a[3].sum() if a[3].sum() else np.nan          # 平均把握分（每场一样算）
        lo, hi = np.nanpercentile(bp - br, [2.5, 97.5])
        out[tr] = {"场数": n, "命中率": float(rate), "平均把握分": float(mp), "差": float(mp - rate),
                   "差_下限": float(lo), "差_上限": float(hi)}
    gap = boot_rate["高"] - boot_rate["低"]
    lo, hi = np.nanpercentile(gap, [2.5, 97.5])
    out["高减低"] = {"差": out["高"]["命中率"] - out["低"]["命中率"], "下限": float(lo), "上限": float(hi)}
    out["全部"] = {"场数": int(len(d)), "命中率": hit_rate(d[f"v_{ch}"]), "平均把握分": float(d[f"p_{ch}"].mean())}
    return out


def verdict(st: dict, settings: dict) -> dict:
    """按事先定的两条判断：分得开（高 − 低 区间下限 > 0，且 高 ≥ 中 ≥ 低）；每档准不准（差的区间含 0、|差| ≤ calib_tol）。"""
    tol = float(settings["pick_tiers_study"]["calib_tol"])
    r = [st[tr]["命中率"] for tr in TIERS]
    apart = bool(st["高减低"]["下限"] > 0 and r[0] >= r[1] >= r[2])
    calib = {tr: bool(st[tr]["差_下限"] <= 0 <= st[tr]["差_上限"] and abs(st[tr]["差"]) <= tol) for tr in TIERS}
    return {"分得开": apart, "准不准": calib}


def by_group(df: pd.DataFrame, ch: str, key: str) -> pd.DataFrame:
    """检验季按赛季或联赛分开：每档的场数和命中率（只报告）。"""
    rows = []
    for g, d in df.groupby(key):
        row = {key: g}
        for tr in TIERS:
            x = d[d[f"tier_{ch}"] == tr][f"v_{ch}"]
            row[f"{tr}_场数"], row[f"{tr}_命中率"] = int(x.notna().sum()), hit_rate(x)
        rows.append(row)
    return pd.DataFrame(rows)


def controls(cut_df: pd.DataFrame, test: pd.DataFrame) -> dict:
    """对照（检验季，只报告）：一直猜主胜；旧规则大小球固定 2.5；一直猜定切点赛季最常见的 4 个比分。"""
    common = (cut_df["home_goals"].astype(str) + "-" + cut_df["away_goals"].astype(str)).value_counts().index[:SCORE_TOP]
    real = test["home_goals"].astype(str) + "-" + test["away_goals"].astype(str)
    return {"一直猜主胜": hit_rate(np.where(test["home_goals"] > test["away_goals"], 1.0, -1.0)),
            "大小球固定2.5": hit_rate(test["v_ou25"]),
            "最常见4个比分": list(common), "一直猜最常见4个比分": hit_rate(np.where(real.isin(common), 1.0, -1.0))}


def tiered(df: pd.DataFrame, cut_mask: np.ndarray, settings: dict) -> tuple[pd.DataFrame, dict]:
    """在 cut_mask 那些行（定切点的赛季）上定每个玩法的切点，给所有行分档。返回 (加了 tier_ 列的表, 切点)。"""
    df, cut = df.copy(), {}
    for ch in CHANNELS:
        cut[ch] = cuts(df.loc[cut_mask, f"p_{ch}"])
        df[f"tier_{ch}"] = tier(df[f"p_{ch}"], cut[ch])
    return df, cut


# ======================= 推出来的主盘口 vs 懂球帝真主盘口（只报告） =======================
def line_check(fs, per: pd.DataFrame, mats: np.ndarray, settings: dict, log=print) -> dict:
    """2018 起有懂球帝大小球记录的比赛：开球前最后的价格（取数层 latest_odds，用途见 settings）按 preview.main_line 定真主盘口，
    和推出来的线比：一样的比例、平均差、两条线上方向一样的比例；两条线上各自的命中率（同一批比赛）。
    这里用的是开球前的价格，不是 T-24，只用来看“推出来的线像不像真的”，不进任何一场的方向和等级。"""
    from src.predict.preview import main_line      # 放这里：只有这里用，preview 要连带读模型那一套

    c = settings["pick_tiers_study"]
    first = int(c["check_first_season"])
    start = f"{first}-{settings['backtest']['window_start_md']}"
    end = f"{int(c['test_last']) + 1}-{settings['backtest']['window_end_md']}"
    kick = fs.prediction_targets(start, end, c["usage"], competition=c["leagues"])
    kick = kick.set_index("match_id")["kickoff_earliest_utc"]
    pos = {mid: k for k, mid in enumerate(per["match_id"])}
    rows = []
    for mid in per.loc[per["season"] >= first, "match_id"]:
        if mid not in kick.index:
            continue
        cutoff = (pd.Timestamp(kick[mid]) - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        o = fs.latest_odds(int(mid), cutoff, c["usage"], market="ou")
        ml = main_line(o, "over", "under") if len(o) else None
        if ml is None:
            continue
        k = pos[mid]
        rows.append({"k": k, "real_line": ml["line"]})
    if not rows:
        return {"场数": 0}
    chk = pd.DataFrame(rows)
    sub = per.iloc[chk["k"]].reset_index(drop=True)
    on_real = picks(sub, mats[chk["k"].to_numpy()], settings, ou_line=chk["real_line"].to_numpy())
    log(f"  核对主盘口：{len(chk)} 场有懂球帝大小球")
    return {"场数": int(len(chk)),
            "线一样的比例": float((np.abs(sub["ou_line"].to_numpy() - chk["real_line"].to_numpy()) < 1e-6).mean()),
            "平均差": float(np.abs(sub["ou_line"].to_numpy() - chk["real_line"].to_numpy()).mean()),
            "方向一样的比例": float((sub["pick_ou"].to_numpy() == on_real["pick_ou"].to_numpy()).mean()),
            "推出来的线命中率": hit_rate(sub["v_ou"]), "真线命中率": hit_rate(on_real["v_ou"]),
            "真线平均把握分": float(on_real["p_ou"].mean()), "推出来的线平均把握分": float(sub["p_ou"].mean()),
            "真线分布": {f"{k:g}": int(v) for k, v in chk["real_line"].value_counts().sort_index().items()}}


# ======================= 整个研究 =======================
def study(fs, settings: dict, run_dir=None, log=print) -> dict:
    """返回结果（summary.json 的内容）和每场的表：{"summary": ..., "picks": 市场那列, "model": 模型自己那列}。fs = None 时不做主盘口核对。"""
    c = settings["pick_tiers_study"]
    t, market, own, check = load(settings, run_dir)
    log(f"读到 {len(t)} 场（{t['season'].min()}–{t['season'].max()}）")
    per = picks(t, market, settings)
    mine = picks(t, own, settings, ou_line=per["ou_line"].to_numpy())   # 模型自己：同一条大小球线
    cut_mask = (per["season"] <= int(c["cut_last"])).to_numpy()
    out = {"规则": "四个玩法把握等级（需求文档第 5 节）", "回测": c["run"], "w": float(settings["fusion"]["w"]),
           "定切点赛季": [int(c["first_season"]), int(c["cut_last"])], "检验赛季": [int(c["cut_last"]) + 1, int(c["test_last"])],
           "数据检查": {k: check[k] for k in ("有预测的比赛", "没有模型矩阵", "融合", "亚盘线不是主盘口、不用的场数", "没有亚盘线的场数")}}
    tables = {}
    for col, df in (("市场", per), ("模型自己", mine)):
        df, cut = tiered(df, cut_mask, settings)
        test = df[~cut_mask]
        res = {"切点": cut, "定切点场数": int(cut_mask.sum()), "检验场数": int(len(test)), "玩法": {}}
        for ch in CHANNELS:
            st = tier_stats(test, ch, settings)
            res["玩法"][ch] = {"每档": st, "判断": verdict(st, settings),
                              "按赛季": by_group(test, ch, "season").to_dict("records"),
                              "按联赛": by_group(test, ch, "competition").to_dict("records")}
        res["对照"] = controls(df[cut_mask], test)
        out[col] = res
        tables[col] = df
    out["主盘口核对"] = line_check(fs, per, market, settings, log) if fs is not None else {"场数": 0, "说明": "没做"}
    out["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {"summary": out, "picks": tables["市场"], "model": tables["模型自己"]}


def _p(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def report(res: dict) -> str:
    s = res["summary"]
    L = [f"# 四个玩法的“把握”等级（{s['created_utc']} UTC）", "",
         f"回测 {s['回测']}，w = {s['w']:g}；切点用 {s['定切点赛季'][0]}–{s['定切点赛季'][1]}，检验 {s['检验赛季'][0]}–{s['检验赛季'][1]}。", ""]
    for col in ("市场", "模型自己"):
        r = s[col]
        L += [f"## {col}（定切点 {r['定切点场数']} 场，检验 {r['检验场数']} 场）", ""]
        for ch in CHANNELS:
            g, v = r["玩法"][ch]["每档"], r["玩法"][ch]["判断"]
            L += [f"### {NAMES[ch]}：切点 {r['切点'][ch][0]:.3f} / {r['切点'][ch][1]:.3f}；"
                  f"分得开：{'是' if v['分得开'] else '否'}；全部 {g['全部']['场数']} 场 命中率 {_p(g['全部']['命中率'])}", "",
                  "| 档 | 场数 | 命中率 | 平均把握分 | 差（把握分 − 命中率） | 差的区间 | 准不准 |", "|---|---|---|---|---|---|---|"]
            for tr in TIERS:
                x = g[tr]
                L.append(f"| {tr} | {x['场数']} | {_p(x['命中率'])} | {_p(x['平均把握分'])} | {x['差']:+.1%} | "
                         f"{x['差_下限']:+.1%} 到 {x['差_上限']:+.1%} | {'准' if v['准不准'][tr] else '不准'} |")
            h = g["高减低"]
            L += ["", f"高 − 低：{h['差']:+.1%}（区间 {h['下限']:+.1%} 到 {h['上限']:+.1%}）", ""]
        k = r["对照"]
        L += [f"对照（检验季）：一直猜主胜 {_p(k['一直猜主胜'])}；大小球固定 2.5 {_p(k['大小球固定2.5'])}；"
              f"一直猜 {'、'.join(k['最常见4个比分'])} {_p(k['一直猜最常见4个比分'])}", ""]
    m = s["主盘口核对"]
    if m.get("场数"):
        L += ["## 推出来的大小球主盘口 vs 懂球帝开球前的真主盘口（只报告）", "",
              f"{m['场数']} 场：线一样 {_p(m['线一样的比例'])}，平均差 {m['平均差']:.3f} 球，方向一样 {_p(m['方向一样的比例'])}；"
              f"命中率 推出来的线 {_p(m['推出来的线命中率'])}、真线 {_p(m['真线命中率'])}；"
              f"平均把握分 推出来的线 {_p(m['推出来的线平均把握分'])}、真线 {_p(m['真线平均把握分'])}。", "",
              "真线分布：" + "、".join(f"{k} 球 {v}" for k, v in m["真线分布"].items()), ""]
    return "\n".join(L) + "\n"


def main() -> None:
    from src.features import FeatureStore, open_readonly

    ap = argparse.ArgumentParser(description="四个玩法的把握等级（需求文档第 5 节）")
    ap.add_argument("--db", default=None, help="数据库文件（默认 settings.yaml 的 paths.database）")
    args = ap.parse_args()
    settings = load_settings()
    fs = FeatureStore(open_readonly(args.db), "开发", settings)
    res = study(fs, settings)
    out = resolve_path(settings["pick_tiers_study"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res["summary"], ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    res["picks"].to_csv(out / "picks.csv", index=False, encoding="utf-8-sig")
    res["model"].to_csv(out / "picks_model.csv", index=False, encoding="utf-8-sig")
    text = report(res)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"结果放在 {out}")


if __name__ == "__main__":
    main()
