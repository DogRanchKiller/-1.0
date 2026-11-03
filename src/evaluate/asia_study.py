'''第 11 阶段：日职、中超、次级联赛的开发回测和平博收盘比（需求文档第 4 节“第 11 阶段”两段）。

同一个程序按 settings 里的一个“设置段”跑：
  asia_study：日职、中超（工作树 J1）；
  second_study：次级联赛（英冠、西乙、德乙、意乙、法乙，工作树 L2）。
  两段的写法一样；second_study 多几项（都是可选的，asia_study 没写就不算）：
    pooled：几个联赛合在一起再算一次（“次级联赛合计”）；
    min_total_n：一个联赛合计少于这么多场，只列数、不下结论；
    gap_reference：和另一组联赛（五大联赛）比“差距之差”，同一周一起抽的 bootstrap（见 bootstrap_gap）；
    era_split：按这个赛季分前后两段只看看（次级联赛 2017/18 起才有射门数据）。
  kor_study：韩K（工作树 KR）。也是同一种写法，多几项可选的（别的段没写就和原来一样）：
    bench_label：报告表头里标杆叫什么（不写是“平博”；韩K 主标杆是多家平均，写“多家平均”）；
    look_benchmarks：只看看的另一个标杆（例如平博收盘），合计照算区间，但结论一律写“只看看”；
    side_by_side：别的联赛已经写下的数（日职、中超、五大联赛），只并排放，不算差距之差。

这个程序不重新预测，只读已经跑好的回测结果（data/backtest/<名字>/predictions.csv）：
  回测本身用 src.evaluate.backtest 跑（经过取数层、开发模式，碰不到 2024、2025 考试赛季）。
  读进来以后再用 season_role 查一遍：每一行都必须是开发赛季，不是就报错（第二道保险）。

做什么：
  1. 每个模型和平博收盘去水概率只比“两边都有分数”的比赛（同一批比赛，才公平）；
  2. 每个联赛分赛季、合计各算一次：场数、模型和平博的 log loss、Brier、命中率，差 = 模型 − 平博；
  3. 合计再用成块 bootstrap（按周，src/evaluate/compare.py）算 95% 区间，给“更准 / 没差别 / 更差”；
     log loss、Brier 越小越好；命中率越大越好，所以命中率的结论倒过来看（差 > 0 才算模型更好）；
  4. 和五大联赛的差距（settings 的 asia_study.five_league_reference）并排放。
结果写一份中文 markdown 报告（settings 的 asia_study.output，data/ 不进 Git），表也存 csv。

用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.asia_study                          （日职、中超）
  .venv\\Scripts\\python -m src.evaluate.asia_study --section second_study   （次级联赛）
'''
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.evaluate.compare import block_ids, bootstrap_mean_diff, verdict

NAMES = {"log_loss": "log loss", "brier": "Brier", "hit": "命中率"}
BENCH_NAMES = {"pinnacle_close", "market_avg"}   # 回测结果里这两个“模型”其实是标杆
SMALL = "样本少，不下结论"
LOOK = "只看看"
GAP_MORE, GAP_SAME, GAP_LESS = "差得更多", "看不出大小", "差得更少"


def paired_rows(pred: pd.DataFrame, model: str, horizon: str, bench: str) -> pd.DataFrame:
    '''模型（时点 horizon）和标杆（时点“收盘”）都有分数的比赛，每场一行：model_<分数>、bench_<分数>。'''
    cols = ["match_id", "competition", "season", "match_date", "log_loss", "brier", "hit"]
    a = pred[(pred["model"] == model) & (pred["horizon"] == horizon)][cols].dropna(subset=["log_loss"])
    b = pred[(pred["model"] == bench) & (pred["horizon"] == "收盘")][["match_id", "log_loss", "brier", "hit"]]
    b = b.dropna(subset=["log_loss"])
    return a.merge(b, on="match_id", suffixes=("_model", "_bench"))


def summarize(p: pd.DataFrame, metrics: list[str]) -> dict:
    '''一组比赛的场数和每个分数的两边平均、差（模型 − 标杆）。'''
    out = {"n": len(p)}
    for m in metrics:
        out[f"{m}_model"] = p[f"{m}_model"].mean()
        out[f"{m}_bench"] = p[f"{m}_bench"].mean()
        out[f"{m}_diff"] = out[f"{m}_model"] - out[f"{m}_bench"]
    return out


def model_verdict(metric: str, lo: float, hi: float) -> str:
    '''差 = 模型 − 平博。log loss / Brier 小的好，直接用 compare.verdict；命中率大的好，把区间反过来再判断。'''
    if metric == "hit":
        return verdict(-hi, -lo)
    return verdict(lo, hi)


def league_tables(pred: pd.DataFrame, league: str, cfg: dict, cmp_cfg: dict,
                  codes: list[str] | None = None, look_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''一个联赛：分赛季的表、合计的表（合计带 bootstrap 区间和结论）。

    league：表里写的名字（联赛代码，或者合在一起的组名，例如“次级联赛合计”）；
    codes：这一组包括哪几个联赛代码，不写就是 [league]。
    合计少于 cfg 的 min_total_n 场（写了才查）：区间照算，结论写“样本少，不下结论”。
    look_only：只看看的标杆（kor_study 的 look_benchmarks），区间照算，结论一律写“只看看”。
    '''
    metrics = cfg["metrics"]
    bench = cfg["benchmark"]
    codes = codes or [league]
    min_total = cfg.get("min_total_n")
    models = [m for m in pred["model"].unique() if m not in BENCH_NAMES]
    season_rows, total_rows = [], []
    for model in models:
        p = paired_rows(pred[pred["competition"].isin(codes)], model, cfg["horizon"], bench)
        if p.empty:
            continue
        for season, g in p.groupby("season"):
            season_rows.append({"league": league, "model": model, "season": int(season), **summarize(g, metrics)})
        row = {"league": league, "model": model, "seasons": f"{int(p['season'].min())}–{int(p['season'].max())}",
               **summarize(p, metrics)}
        blocks = block_ids(p["match_date"], cmp_cfg["block"])
        for m in metrics:
            d = p[f"{m}_model"] - p[f"{m}_bench"]
            _, lo, hi = bootstrap_mean_diff(d, blocks, cmp_cfg["n_boot"], cmp_cfg["confidence"], cmp_cfg["seed"])
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
            if look_only:
                row[f"{m}_verdict"] = LOOK
            else:
                row[f"{m}_verdict"] = model_verdict(m, lo, hi) if min_total is None or len(p) >= min_total else SMALL
        total_rows.append(row)
    return pd.DataFrame(season_rows), pd.DataFrame(total_rows)


def check_dev_seasons(pred: pd.DataFrame, settings: dict, where: str) -> None:
    '''读进来的回测结果每一行都必须是开发赛季（season_role == "dev"）。考试赛季 2024、2025 是锁住的，碰到直接报错。'''
    bad = sorted({int(s) for s in pred["season"].dropna().unique() if season_role(int(s), settings) != "dev"})
    if bad:
        raise ValueError(f"{where} 里有不是开发赛季的比赛：{bad}。考试赛季锁住，不能拿来比")


def gap_verdict(lo: float, hi: float) -> str:
    '''差距之差 =（这组 模型 − 平博）−（参照组 模型 − 平博）：整个区间 > 0 差得更多，< 0 差得更少，跨过 0 看不出大小。'''
    if lo > 0:
        return GAP_MORE
    if hi < 0:
        return GAP_LESS
    return GAP_SAME


def bootstrap_gap(d_a, blocks_a, d_b, blocks_b, n_boot: int, confidence: float, seed: int
                  ) -> tuple[float, float, float]:
    '''两组比赛的“平均差之差”（a 的平均 − b 的平均）的成块 bootstrap 区间：返回 (差之差, 区间下限, 区间上限)。

    为什么同一周一起抽：次级联赛和五大联赛同一周的比赛共享一些东西（同一次重新拟合、国家队比赛日前后、冬歇），
    a、b 分开各抽各的，会把这种“一起偏”当成独立的运气。所以把两组的周合在一起编号，每次重抽一批周，
    这批周里 a 的比赛算 a 的平均、b 的比赛算 b 的平均，再相减。
    抽到的一批周里某一组一场都没有（几乎不会，几百周里只有个别周只有一组有比赛）时报错，不悄悄算出 nan。
    '''
    d_a, d_b = np.asarray(d_a, dtype=float), np.asarray(d_b, dtype=float)
    if len(d_a) == 0 or len(d_b) == 0:
        raise ValueError("没有可比较的比赛")
    keys, inv = np.unique(np.concatenate([np.asarray(blocks_a), np.asarray(blocks_b)]), return_inverse=True)
    ia, ib = inv[:len(d_a)], inv[len(d_a):]
    k = len(keys)
    sa, na = np.bincount(ia, weights=d_a, minlength=k), np.bincount(ia, minlength=k).astype(float)
    sb, nb = np.bincount(ib, weights=d_b, minlength=k), np.bincount(ib, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    step = max(1, 2_000_000 // k)          # 一次抽多少轮：控制内存（和 compare.bootstrap_mean_diff 一样）
    for i in range(0, n_boot, step):
        idx = rng.integers(0, k, size=(min(step, n_boot - i), k))
        ca, cb = na[idx].sum(axis=1), nb[idx].sum(axis=1)
        if (ca == 0).any() or (cb == 0).any():
            raise ValueError("重抽的一批周里有一组一场比赛都没有，算不了平均")
        out[i:i + len(idx)] = sa[idx].sum(axis=1) / ca - sb[idx].sum(axis=1) / cb
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(out, [alpha, 1 - alpha])
    return float(d_a.mean() - d_b.mean()), float(lo), float(hi)


def gap_table(groups: dict[str, pd.DataFrame], ref: dict[str, pd.DataFrame], pairs: list[tuple[str, str]],
              models: list[str], cfg: dict, cmp_cfg: dict) -> pd.DataFrame:
    '''“差距之差”表。groups、ref：组名 → 这组的预测（已经按联赛筛好）；pairs：[(这组, 参照组)]。
    只算主分数（compare.primary_metric），每个模型一行。'''
    metric = cmp_cfg["primary_metric"]
    rows = []
    for name, ref_name in pairs:
        for model in models:
            a = paired_rows(groups[name], model, cfg["horizon"], cfg["benchmark"])
            b = paired_rows(ref[ref_name], model, cfg["horizon"], cfg["benchmark"])
            if a.empty or b.empty:
                continue
            da, db = a[f"{metric}_model"] - a[f"{metric}_bench"], b[f"{metric}_model"] - b[f"{metric}_bench"]
            g, lo, hi = bootstrap_gap(da, block_ids(a["match_date"], cmp_cfg["block"]),
                                      db, block_ids(b["match_date"], cmp_cfg["block"]),
                                      cmp_cfg["n_boot"], cmp_cfg["confidence"], cmp_cfg["seed"])
            # 样本少的规则和合计表一样：这一组少于 min_total_n 场（写了才查），区间照算，结论写“样本少，不下结论”
            small = cfg.get("min_total_n") is not None and len(a) < cfg["min_total_n"]
            rows.append({"group": name, "reference": ref_name, "model": model, "metric": metric,
                         "n": len(a), "diff": float(da.mean()), "ref_n": len(b), "ref_diff": float(db.mean()),
                         "gap": g, "lo": lo, "hi": hi, "verdict": SMALL if small else gap_verdict(lo, hi)})
    return pd.DataFrame(rows)


def era_table(groups: dict[str, pd.DataFrame], models: list[str], cfg: dict, split: int) -> pd.DataFrame:
    '''按赛季分前后两段（< split、≥ split）只看看：每组、每个模型的场数、模型和平博、差。不算区间、不下结论。'''
    rows = []
    for name, pred in groups.items():
        for model in models:
            p = paired_rows(pred, model, cfg["horizon"], cfg["benchmark"])
            for era, g in (("前", p[p["season"] < split]), ("后", p[p["season"] >= split])):
                if g.empty:
                    continue
                label = f"{int(g['season'].min())}–{int(g['season'].max())}"
                rows.append({"group": name, "model": model, "era": era, "seasons": label,
                             **summarize(g, cfg["metrics"])})
    return pd.DataFrame(rows)


def read_run(run: str, settings: dict | None = None, cache: dict | None = None) -> pd.DataFrame:
    '''读一次回测的 predictions.csv（data/backtest/<run>/）。读完查一遍全是开发赛季。
    cache：同一次运行里几个联赛共用一次回测时，只读一次文件。'''
    if cache is not None and run in cache:
        return cache[run]
    path = resolve_path(f"{(settings or load_settings())['backtest']['output_dir']}/{run}") / "predictions.csv"
    pred = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    check_dev_seasons(pred, settings or load_settings(), f"回测 {run}")
    if cache is not None:
        cache[run] = pred
    return pred


def load_league(cfg: dict, league: str, settings: dict | None = None, cache: dict | None = None) -> pd.DataFrame:
    '''把一个联赛的几次回测拼起来，只留这个联赛的比赛。同一个模型在两次回测里都有时，只留第一次的（名单前面的）。'''
    parts, seen = [], set()
    for run in cfg["runs"][league]:
        pred = read_run(run, settings, cache)
        pred = pred[pred["competition"] == league]
        bench_rows = pred["model"].isin(BENCH_NAMES)
        keep = bench_rows & (not parts)        # 标杆只取第一次回测的（新的一列，不改 bench_rows 本身）
        keep = keep | (~bench_rows & ~pred["model"].isin(seen))
        seen |= set(pred.loc[~bench_rows, "model"].unique())
        parts.append(pred[keep])
    return pd.concat(parts, ignore_index=True)


def _f(v, nd=4, sign=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def group_order(cfg: dict) -> list[str]:
    '''报告里各组的先后：runs 里联赛的顺序，再接 pooled 里合在一起的组。'''
    return list(cfg["runs"]) + list((cfg.get("pooled") or {}))


def report(seasons: pd.DataFrame, totals: pd.DataFrame, cfg: dict, league_names: dict,
           extra: dict | None = None) -> str:
    '''中文 markdown 报告：合计表、和五大联赛并排（或差距之差）、只看看的表、分赛季表。
    extra：second_study 才有的几张表（gaps 差距之差、same_country 和同国顶级联赛比、eras 分前后两段、
    reproduce 五大联赛重跑核对），没有就不写。'''
    extra = extra or {}
    order = {g: i for i, g in enumerate(group_order(cfg))}
    name = lambda g: league_names.get(g, g)   # noqa: E731（联赛代码 → 中文名，合计组本来就是中文）
    bl = cfg.get("bench_label", "平博")       # 标杆在表头里的名字（kor_study 是“多家平均”）
    out = [f"# {cfg.get('title', '日职、中超开发回测：模型和平博收盘比')}", "",
           f"时点 {cfg['horizon']}；只比模型和{bl}收盘去水概率都有的比赛。差 = 模型 − {bl}：",
           "log loss、Brier 差 > 0 是模型更差；命中率差 < 0 是模型更差。区间是按周成块 bootstrap 的 95% 区间。", ""]
    out += ["## 合计", ""]
    head = ["联赛", "模型", "赛季", "场数", "模型 log loss", bl, "差", "95% 区间", "结论",
            "模型 Brier", bl, "差", "模型命中率", bl, "差"]
    rows = []
    t_sorted = totals.assign(_o=totals["league"].map(order)).sort_values(["_o", "log_loss_diff"])
    for _, r in t_sorted.iterrows():
        rows.append([name(r["league"]), r["model"], r["seasons"], str(int(r["n"])),
                     _f(r["log_loss_model"]), _f(r["log_loss_bench"]), _f(r["log_loss_diff"], sign=True),
                     f"{_f(r['log_loss_lo'], sign=True)} 到 {_f(r['log_loss_hi'], sign=True)}", r["log_loss_verdict"],
                     _f(r["brier_model"]), _f(r["brier_bench"]), _f(r["brier_diff"], sign=True),
                     _f(r["hit_model"], 3), _f(r["hit_bench"], 3), _f(r["hit_diff"], 3, sign=True)])
    out += [md_table(head, rows), ""]
    for label, look in extra.get("look", []):
        out += [f"## 只看看：和{label}收盘比（只算两边都有的比赛，不下结论）", ""]
        rows = [[name(r["league"]), r["model"], r["seasons"], str(int(r["n"])), _f(r["log_loss_model"]),
                 _f(r["log_loss_bench"]), _f(r["log_loss_diff"], sign=True),
                 f"{_f(r['log_loss_lo'], sign=True)} 到 {_f(r['log_loss_hi'], sign=True)}"]
                for _, r in look.iterrows()]
        out += [md_table(["联赛", "模型", "赛季", "场数", "模型 log loss", label, "差", "95% 区间"], rows)
                if rows else "（没有两边都有的比赛）", ""]
    if cfg.get("side_by_side"):
        out += [side_by_side_md(cfg, totals, name), ""]
    if "reproduce" in extra:
        out += ["## 五大联赛重跑核对（和第 4 节步骤 6 的数比）", "", extra["reproduce"], ""]
    if "gaps" in extra and not extra["gaps"].empty:
        out += ["## 比五大联赛差得多还是少（结论看这张）", "",
                "差距之差 =（这一组 模型 − 平博）−（五大联赛 模型 − 平博），log loss。> 0 是这一组离平博更远。",
                "区间：次级联赛和五大联赛同一周的比赛一起抽（按周成块 bootstrap，95%）。", ""]
        out += [gap_md(extra["gaps"], name), ""]
    else:
        ref = cfg.get("five_league_reference") or {}
        if ref:
            out += ["## 和五大联赛比（log loss 差 = 模型 − 平博）", ""]
            head = ["模型", "五大联赛 2014–2023", "场数"] + [name(lg) for lg in group_order(cfg)]
            rows = []
            for model, v in ref.items():
                cells = [model, _f(v["model"] - v["pinnacle"], sign=True), str(v["n"])]
                for lg in group_order(cfg):
                    t = totals[(totals["league"] == lg) & (totals["model"] == model)]
                    cells.append(f"{_f(t['log_loss_diff'].iloc[0], sign=True)}（{int(t['n'].iloc[0])} 场）" if len(t) else "")
                rows.append(cells)
            out += [md_table(head, rows), ""]
    if "same_country" in extra and not extra["same_country"].empty:
        out += ["## 只看看：和同一国家的顶级联赛比（差距之差，log loss）", ""]
        out += [gap_md(extra["same_country"], name), ""]
    if "eras" in extra and not extra["eras"].empty:
        split = cfg["era_split"]
        out += [f"## 只看看：{split} 赛季以前 / 以后（次级联赛 {split}/{(split + 1) % 100:02d} 起才有射门数据）", ""]
        head = ["组", "模型", "段", "赛季", "场数", "模型 log loss", "平博", "差"]
        e = extra["eras"].assign(_o=extra["eras"]["group"].map(order).fillna(len(order)),
                                 _e=extra["eras"]["era"].map({"前": 0, "后": 1}))   # “前”在“后”前面（按汉字编码排会反过来）
        rows = [[name(r["group"]), r["model"], r["era"], r["seasons"], str(int(r["n"])), _f(r["log_loss_model"]),
                 _f(r["log_loss_bench"]), _f(r["log_loss_diff"], sign=True)]
                for _, r in e.sort_values(["_o", "group", "model", "_e"]).iterrows()]
        out += [md_table(head, rows), ""]
    out += ["## 分赛季（log loss、Brier、命中率；只看看）", ""]
    head = ["联赛", "模型", "赛季", "场数", "模型 log loss", bl, "差", "Brier 差", "命中率差", "备注"]
    rows = []
    s_sorted = seasons.assign(_o=seasons["league"].map(order)).sort_values(["_o", "model", "season"])
    for _, r in s_sorted.iterrows():
        note = "样本少" if r["n"] < cfg["min_season_n"] else ""
        rows.append([name(r["league"]), r["model"], str(r["season"]), str(int(r["n"])),
                     _f(r["log_loss_model"]), _f(r["log_loss_bench"]), _f(r["log_loss_diff"], sign=True),
                     _f(r["brier_diff"], sign=True), _f(r["hit_diff"], 3, sign=True), note])
    out += [md_table(head, rows), ""]
    return "\n".join(out)


def side_by_side_md(cfg: dict, totals: pd.DataFrame, name) -> str:
    '''和别的联赛并排（只看看）：这次算的合计 + settings 里 side_by_side 写下的别的联赛的数（标杆可能不一样，表里写明）。'''
    bl = cfg.get("bench_label", "平博")
    head = ["联赛", "模型", "标杆", "场数", "模型 log loss", "标杆 log loss", "差", "95% 区间", "数从哪来"]
    rows = []
    for _, r in totals.iterrows():
        rows.append([name(r["league"]), r["model"], f"{bl}收盘", str(int(r["n"])), _f(r["log_loss_model"]),
                     _f(r["log_loss_bench"]), _f(r["log_loss_diff"], sign=True),
                     f"{_f(r['log_loss_lo'], sign=True)} 到 {_f(r['log_loss_hi'], sign=True)}", "这次"])
    for v in cfg["side_by_side"]:
        ci = f"{_f(v['lo'], sign=True)} 到 {_f(v['hi'], sign=True)}" if v.get("lo") is not None else ""
        rows.append([v["league"], v["model"], v["bench"], f"{int(v['n']):,}", _f(v["model_ll"]), _f(v["bench_ll"]),
                     _f(v["model_ll"] - v["bench_ll"], sign=True), ci, v["source"]])
    return "\n".join(["## 只看看：和日职、中超、五大联赛并排（log loss，差 = 模型 − 标杆）", "",
                      "别的联赛的标杆是平博收盘，这里是" + bl + "收盘，口径不一样：只并排看，不算差距之差。", "",
                      md_table(head, rows)])


def gap_md(g: pd.DataFrame, name) -> str:
    '''差距之差表 → markdown。'''
    head = ["这一组", "参照", "模型", "场数", "这一组 差", "参照场数", "参照 差", "差距之差", "95% 区间", "结论"]
    rows = [[name(r["group"]), name(r["reference"]), r["model"], str(int(r["n"])), _f(r["diff"], sign=True),
             str(int(r["ref_n"])), _f(r["ref_diff"], sign=True), _f(r["gap"], sign=True),
             f"{_f(r['lo'], sign=True)} 到 {_f(r['hi'], sign=True)}", r["verdict"]] for _, r in g.iterrows()]
    return md_table(head, rows)


def reproduce_check(ref_pred: pd.DataFrame, cfg: dict, tol: float) -> str:
    '''五大联赛重跑的合计（模型、平博、场数）和 five_league_reference 写的数比。
    场数不一样、或者 log loss 差超过 tol，直接报错（规则：不一样先查原因，不往下做）。返回一段说明给报告。

    按第 4 节步骤 6 的算法核对：模型的数是“模型出了预测的全部比赛”的平均（18,085 场），
    平博的数是“有平博收盘的全部比赛”的平均（少 2 场没有平博，18,083 场）。两边不是同一批比赛，
    所以这里分开算（不用 paired_rows）。报告里另外写一行“同一批比赛”的数（后面的表都用同一批）。'''
    lines = []
    m_rows = ref_pred[ref_pred["horizon"] == cfg["horizon"]].dropna(subset=["log_loss"])
    b_rows = ref_pred[(ref_pred["model"] == cfg["benchmark"]) & (ref_pred["horizon"] == "收盘")]
    b_rows = b_rows.dropna(subset=["log_loss"])
    got_b = b_rows["log_loss"].mean()
    for model, v in (cfg.get("five_league_reference") or {}).items():
        mine = m_rows[m_rows["model"] == model]
        got_m = mine["log_loss"].mean()
        ok = len(mine) == v["n"] and abs(got_m - v["model"]) <= tol and abs(got_b - v["pinnacle"]) <= tol
        p = paired_rows(ref_pred, model, cfg["horizon"], cfg["benchmark"])
        lines.append(f"- {model}：重跑 {len(mine)} 场、模型 {got_m:.5f}，平博 {len(b_rows)} 场 {got_b:.5f}；"
                     f"第 4 节 {v['n']} 场，模型 {v['model']:.5f}、平博 {v['pinnacle']:.5f}。{'一样' if ok else '不一样'}。"
                     f"同一批比赛（两边都有）{len(p)} 场：模型 {p['log_loss_model'].mean():.5f}、"
                     f"平博 {p['log_loss_bench'].mean():.5f}")
        if not ok:
            raise ValueError("五大联赛重跑和第 4 节的数不一样，先查原因：\n" + "\n".join(lines))
    return "\n".join(lines)


def run(settings: dict | None = None, section: str = "asia_study") -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    '''按 settings 的一个设置段跑（asia_study 日职中超 / second_study 次级联赛），写报告和 csv。'''
    s = settings or load_settings()
    cfg = s[section]
    names = {c: n for g in s["leagues"].values() for c, n in g.items()}
    cache: dict = {}
    groups = {league: load_league(cfg, league, s, cache) for league in cfg["runs"]}
    for pooled_name, codes in (cfg.get("pooled") or {}).items():
        groups[pooled_name] = pd.concat([groups[c] for c in codes], ignore_index=True)
    seasons, totals = [], []
    for g, pred in groups.items():
        codes = (cfg.get("pooled") or {}).get(g)
        a, b = league_tables(pred, g, cfg, s["compare"], codes=codes)
        seasons.append(a)
        totals.append(b)
    seasons, totals = pd.concat(seasons, ignore_index=True), pd.concat(totals, ignore_index=True)
    extra = {}
    looks = []
    for bench, label in (cfg.get("look_benchmarks") or {}).items():
        lcfg = {**cfg, "benchmark": bench}
        parts = [league_tables(pred, g, lcfg, s["compare"], codes=(cfg.get("pooled") or {}).get(g), look_only=True)[1]
                 for g, pred in groups.items()]
        looks.append((label, pd.concat(parts, ignore_index=True)))
    if looks:
        extra["look"] = looks
    gref = cfg.get("gap_reference")
    if gref:
        ref_pred = read_run(gref["run"], s)
        ref_pred = ref_pred[ref_pred["competition"].isin(gref["codes"])]
        extra["reproduce"] = reproduce_check(ref_pred, cfg, gref["reproduce_tol"])
        models = [m for m in cfg["models"]]
        ref = {gref["name"]: ref_pred}
        extra["gaps"] = gap_table(groups, ref, [(g, gref["name"]) for g in groups], models, cfg, s["compare"])
        same = gref.get("same_country") or {}
        if same:
            ref_by_code = {top: ref_pred[ref_pred["competition"] == top] for top in same.values()}
            extra["same_country"] = gap_table(groups, ref_by_code, list(same.items()), models, cfg, s["compare"])
        if cfg.get("era_split"):
            extra["eras"] = era_table({**groups, gref["name"]: ref_pred}, models, cfg, cfg["era_split"])
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report(seasons, totals, cfg, names, extra), encoding="utf-8")
    kw = {"index": False, "encoding": "utf-8-sig", "float_format": "%.6f"}
    seasons.to_csv(out.with_name(out.stem + "_分赛季.csv"), **kw)
    totals.to_csv(out.with_name(out.stem + "_合计.csv"), **kw)
    for label, look in extra.get("look", []):
        look.to_csv(out.with_name(f"{out.stem}_只看看_{label}.csv"), **kw)
    for key, label in (("gaps", "差距之差"), ("same_country", "同国顶级联赛"), ("eras", "前后两段")):
        if key in extra and not extra[key].empty:
            extra[key].to_csv(out.with_name(f"{out.stem}_{label}.csv"), **kw)
    return seasons, totals, out


def main() -> None:
    ap = argparse.ArgumentParser(description="开发回测和平博收盘比（读已经跑好的回测）")
    ap.add_argument("--section", default="asia_study",
                    help="settings 里哪一段：asia_study（日职、中超，默认）/ second_study（次级联赛）/ kor_study（韩K）")
    a = ap.parse_args()
    _, totals, out = run(section=a.section)
    cols = ["league", "model", "n", "log_loss_model", "log_loss_bench", "log_loss_diff", "log_loss_lo",
            "log_loss_hi", "log_loss_verdict", "brier_diff", "hit_diff"]
    pd.set_option("display.width", 220)
    print(totals[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n报告在 {out}")


if __name__ == "__main__":
    main()
