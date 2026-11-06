"""历史基线：冻结的版本在开发赛季按四个通道和同一时刻的市场各比一遍（需求文档第 16 节“历史基线怎么算（第 3 步）”，跑之前写下的规则）。

一、比哪个版本
  现在冻结的 gb_l7、T-24、开发赛季 2005–2023、五大联赛。直接读 v1.12 新时间规则下的回测（baseline.run，默认 tc_fix_gb_l7）：
  predictions.csv 和每场整张比分矩阵 matrices_gb_l7.npz，不重新训练。

二、市场价（经取数层，开发模式，用途「评估」）
  每场按自己的 T-24 截止时间取 football-data 赛前价（能知道的时间按采集时间规则）。T-24 还拿不到的场不比。
  同一场再按“截止时间 + late_hours 小时”取一次，只用来数“有赛前价、但 T-24 还没采到”的场。

三、四个通道（差 = 模型分数 − 市场分数，小于 0 = 模型更准）
  1. 胜平负：市场 = 平博赛前，没有用多家平均（market_odds.result_books 的顺序）。log loss。
  2. 亚盘让球：平博赛前（下结论）、多家平均赛前（只报告）各一份。线 = 这家赛前那条线，两边赔率都在 ah_main_odds 里才算主盘口。
     市场胜算 = 两边按比例去水后主队那边；模型胜算 = 模型矩阵在这条线上主队那边（赢半、输半折半个，走盘不计）。
  3. 亚盘大小球：平博赛前 2.5（下结论）、多家平均赛前 2.5（只报告）。“大”那边的胜算。
  4. 比分：市场 = 胜平负（第 1 条那家）+ 大 2.5 球（多家平均）还原的市场比分矩阵（fit_market_dc，和融合研究一样）。
     −ln(实际比分那一格)。
  亚盘、大小球每场的分数和四本账一样（src/evaluate/ledger.py 的 line_score）。

四、区间和结论
  差的平均按周成块 bootstrap（compare 段的设置）。每个通道下结论的那一份（下结论的那家、全部场次）给“更准 / 没差别 / 更差”；
  按联赛、两段、胜平负按市场是哪家，都只报告。

五、结果（data/backtest/<回测名>/，不进 Git）
  baseline_odds.csv       取到的赔率（下次直接读；设置变了或加 --refresh-odds 重新取）
  baseline_matches.csv    每场每个通道每家市场一行（以后的改进版按场和这里的“模型分数”比）
  baseline_summary.csv    每个通道每家市场每一份的场数、两边平均分、差、区间、结论、命中率
  baseline_excluded.csv   没比的场：按通道、市场、原因、周几数
  baseline_check.json     场数核对、市场比分矩阵还原误差

只读：不改模型、冻结的设置、网站、定时任务。
用法：
  .venv\\Scripts\\python -m src.evaluate.baseline
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.markets import dc_matrices, dc_rho_bounds, fit_market_dc
from src.common.odds import devig
from src.common.settlement import over_under_margin, settle_margin
from src.evaluate.compare import block_ids, bootstrap_mean_diff, load_predictions, verdict
from src.evaluate.fusion_study import P3, derive
from src.evaluate.ledger import RESULT_NAMES, line_hit, line_score
from src.features import FeatureStore, open_readonly
from src.features.market_odds import _check_known, odds_features
from src.features.known_time import to_utc_text

ODDS_COLS = ["match_id", "source", "bookmaker", "market", "line", "home", "draw", "away", "over", "under"]
WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
OK = "比了"
# 没比的原因
LATE, NONE, NOT_MAIN = "有赛前价、T-24 还没采到", "没有这家的赛前价", "亚盘线不是主盘口"
# baseline_matches.csv 的列。模型概率、市场概率：胜平负、比分是给实际结果的概率；亚盘是主队那边、大小球是“大”那边的胜算
ROW_COLS = ["match_id", "联赛", "赛季", "比赛日期", "周几", "两段", "通道", "市场", "下结论的那一份", "市场价来自",
            "比分", "线", "结算", "模型概率", "市场概率", "模型分数", "市场分数", "差", "模型命中", "模型份数",
            "市场命中", "市场份数"]


# ======================= 每个通道比哪几家 =======================
def channel_keys(settings: dict) -> list[tuple[str, str, bool]]:
    """(通道, 市场, 是不是下结论的那一份)。胜平负、比分的市场是“平博优先、没有用多家平均”这一份。"""
    b = settings["baseline"]
    keys = [("胜平负", b["result_label"], True)]
    for channel in ("亚盘让球", "亚盘大小球"):
        keys += [(channel, book, book == b["main_book"]) for book in b["line_books"]]
    return keys + [("比分", b["result_label"], True)]


# ======================= 取赔率（经取数层） =======================
def fetch_odds(fs, targets: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每场（targets：match_id、cutoff_utc）取两次赛前价：截止时间（“当时”）、截止时间 + late_hours（“后来”，只用来数没采到的）。

    只要 market_odds.source 这个来源、line_books 和胜平负要的几家、三种玩法。返回一行 = 一场 × 一家 × 一种玩法 × 哪一次。
    “当时”那一次检查取到的“能知道的时间”不晚于截止时间。
    """
    b, cfg = settings["baseline"], settings["market_odds"]
    books = list(dict.fromkeys([*cfg["result_books"], cfg["ou_book"], *b["line_books"]]))
    parts = []
    for mid, cutoff in targets[["match_id", "cutoff_utc"]].itertuples(index=False):
        cutoff = to_utc_text(cutoff)
        late = to_utc_text(pd.Timestamp(cutoff) + pd.Timedelta(hours=float(b["late_hours"])))
        for when, cut in (("当时", cutoff), ("后来", late)):
            o = fs.pre_match_odds(int(mid), cut, b["usage"], bookmaker=books, market=["1x2", "ou", "ah"])
            if when == "当时":
                _check_known(o, cut, "本场赔率")
            o = o[o["source"] == cfg["source"]][ODDS_COLS]
            parts.append(o.assign(when=when))
    if not parts:
        return pd.DataFrame(columns=[*ODDS_COLS, "when"])
    return pd.concat(parts, ignore_index=True)


# ======================= 赔率 → 每个通道每家市场的价格 =======================
def _line_table(odds: pd.DataFrame, book: str, market: str, settings: dict) -> pd.DataFrame:
    """亚盘（market='ah'）或大 2.5 球（'ou'）：这家每场一行 match_id、line、share（去水后主队 / 大那边）、main（是不是主盘口）。"""
    a, b = ("home", "away") if market == "ah" else ("over", "under")
    o = odds[(odds["bookmaker"] == book) & (odds["market"] == market)].dropna(subset=["line", a, b])
    o = o[(o[[a, b]] > 1).all(axis=1)]
    if market == "ou":
        o = o[np.isclose(o["line"].astype(float), float(settings["market_odds"]["ou_line"]))]
    o = o.drop_duplicates("match_id")
    share = devig(o[[a, b]].to_numpy(float), settings["market_odds"]["devig_method"])[:, 0] if len(o) else []
    out = pd.DataFrame({"match_id": o["match_id"].to_numpy(int), "line": o["line"].to_numpy(float), "share": share})
    if market == "ah":
        lo, hi = settings["fusion_study"]["ah_main_odds"]
        out["main"] = o[[a, b]].apply(lambda c: c.between(lo, hi)).all(axis=1).to_numpy()
    else:
        out["main"] = True
    return out


def market_tables(odds: pd.DataFrame, settings: dict) -> dict:
    """一次取到的赔率 → {(通道, 市场): 每场一行的表}。胜平负：mo_p_home / draw / away、mo_pinnacle；
    比分：再加 mo_over25（还原市场矩阵要的四个数）；亚盘、大小球：line、share、main。"""
    feats = odds_features(odds, settings["market_odds"])
    res = feats.dropna(subset=["mo_p_home", "mo_p_draw", "mo_p_away"])
    out = {}
    for channel, book, _ in channel_keys(settings):
        if channel == "胜平负":
            out[(channel, book)] = res[["match_id", "mo_p_home", "mo_p_draw", "mo_p_away", "mo_pinnacle"]]
        elif channel == "比分":
            out[(channel, book)] = res.dropna(subset=["mo_over25"])[
                ["match_id", "mo_p_home", "mo_p_draw", "mo_p_away", "mo_over25", "mo_pinnacle"]]
        else:
            out[(channel, book)] = _line_table(odds, book, "ah" if channel == "亚盘让球" else "ou", settings)
    return out


def status_of(match_ids: np.ndarray, now: pd.DataFrame, later: pd.DataFrame) -> np.ndarray:
    """每场这个通道这家市场比不比：比了 / 亚盘线不是主盘口 / 有赛前价、T-24 还没采到 / 没有这家的赛前价。"""
    ids = pd.Index(match_ids)
    st = np.full(len(ids), NONE, dtype=object)
    st[ids.isin(later["match_id"])] = LATE
    if "main" in now:
        st[ids.isin(now.loc[~now["main"], "match_id"])] = NOT_MAIN
        st[ids.isin(now.loc[now["main"], "match_id"])] = OK
    else:
        st[ids.isin(now["match_id"])] = OK
    return st


# ======================= 模型矩阵在一条线上的胜算 =======================
_VALUE_UP = {"win": 1.0, "half_win": 0.5, "push": 0.0, "half_loss": 0.0, "loss": 0.0}
_VALUE_DOWN = {"win": 0.0, "half_win": 0.0, "push": 0.0, "half_loss": 0.5, "loss": 1.0}


def line_shares(dist: np.ndarray, values: np.ndarray, line: np.ndarray, to_margin) -> np.ndarray:
    """dist：每场一行的分布（净胜球或总进球，值在 values 里）；line：每场的线；to_margin(值, 线) → (净胜球, 盘口)。

    胜算 = (赢 + 赢半 × 0.5) ÷ (赢 + 赢半 × 0.5 + 输 + 输半 × 0.5)，走盘不计（和 markets.win_share 一样）。全是走盘的是空值。
    同一条线的场一起算，所以三万多场也很快。
    """
    share = np.full(len(dist), np.nan)
    for ln in np.unique(line[~np.isnan(line)]):
        k = line == ln
        outcomes = [settle_margin(*to_margin(int(v), float(ln))) for v in values]
        up = dist[k] @ np.array([_VALUE_UP[o] for o in outcomes])
        down = dist[k] @ np.array([_VALUE_DOWN[o] for o in outcomes])
        with np.errstate(invalid="ignore", divide="ignore"):
            share[k] = np.where(up + down > 0, up / (up + down), np.nan)
    return share


def total_dist(mats: np.ndarray) -> np.ndarray:
    """一批比分矩阵 → 每场总进球 0 … 2n 的分布。"""
    i, j = np.indices(mats.shape[1:])
    return np.stack([(mats * (i + j == s)).sum(axis=(1, 2)) for s in range(2 * (mats.shape[1] - 1) + 1)], axis=1)


def _ah_margin(diff: int, line: float):        # 主队那边：净胜球 = 主 − 客，盘口 = 线
    return diff, line


def _ou_margin(total: int, line: float):       # 大那边
    return over_under_margin(total, line, "over")


def _top_hits(mats: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    """比分命中（只显示）：每场概率最大的那一格是不是实际比分（超过最大进球数的并进最后一行 / 列）。"""
    n = mats.shape[1] - 1
    flat = mats.reshape(len(mats), -1).argmax(axis=1)
    return (((flat // (n + 1)) == np.minimum(hg, n)) & ((flat % (n + 1)) == np.minimum(ag, n))).astype(float)


# ======================= 主流程：每场每个通道一行 =======================
def build(pred: pd.DataFrame, ids: np.ndarray, mats: np.ndarray, odds: pd.DataFrame, settings: dict
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """返回 (每场每个通道每家市场一行, 汇总, 没比的场, 检查信息)。odds 是 fetch_odds 的结果。"""
    b = settings["baseline"]
    leagues = b["leagues"]
    if isinstance(leagues, str):                     # 写组名（例如 top）时换成这一组的联赛代码
        leagues = list(settings["leagues"][leagues])
    t = pred[(pred["model"] == b["model"]) & (pred["horizon"] == b["horizon"]) & pred["competition"].isin(leagues)]
    t = t.dropna(subset=["result", "home_goals", "away_goals", *P3]).drop_duplicates("match_id")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    no_mat = ~t["match_id"].isin(pos.index)
    if no_mat.any():
        raise ValueError(f"{int(no_mat.sum())} 场有预测、没有模型矩阵：矩阵和预测不是同一次回测？")
    t = t.sort_values("match_id", ignore_index=True)
    m = mats[pos.loc[t["match_id"]].to_numpy()].astype(float)
    m /= m.sum(axis=(1, 2), keepdims=True)
    hg, ag = t["home_goals"].to_numpy(int), t["away_goals"].to_numpy(int)
    model = derive(m, hg, ag)
    # 模型矩阵算出来的胜平负要和 predictions.csv 一样（矩阵存的是同一次预测；float32 存的，差一点点）
    gap = float(np.abs(model["p3"] - t[P3].to_numpy(float)).max())
    if gap > 1e-4:
        raise ValueError(f"模型矩阵算出的胜平负和 predictions.csv 差了 {gap:.2e}，矩阵和预测不是同一次回测？")
    model["p3"] = model["p3"] / model["p3"].sum(axis=1, keepdims=True)      # float 误差补成正好 1
    model["total"], model["top_hit"] = total_dist(m), _top_hits(m, hg, ag)
    model["y"] = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))           # 0 主胜、1 平、2 客胜

    now = market_tables(odds[odds["when"] == "当时"], settings)
    later = market_tables(odds[odds["when"] == "后来"], settings)
    d = pd.to_datetime(t["match_date"])
    split = b["period_split"]
    base = pd.DataFrame({"match_id": t["match_id"], "联赛": t["competition"], "赛季": t["season"],
                         "比赛日期": t["match_date"], "周几": [WEEKDAYS[k] for k in d.dt.day_of_week],
                         "两段": np.where(t["season"] < split, f"{split} 之前", f"{split} 起"),
                         "比分": [f"{h}-{a}" for h, a in zip(hg, ag)]})
    rows, excluded = [], []
    check = {"有预测、有比分的场": int(len(t)), "模型矩阵和 predictions.csv 胜平负最大差": gap}
    for channel, book, main in channel_keys(settings):
        st = status_of(t["match_id"].to_numpy(), now[(channel, book)], later[(channel, book)])
        g = base.assign(状态=st)
        excluded.append(g[g["状态"] != OK].groupby(["状态", "周几"]).size().rename("场数").reset_index()
                        .assign(通道=channel, 市场=book))
        check[f"{channel}（{book}）"] = {r: int((st == r).sum()) for r in (OK, LATE, NONE, NOT_MAIN)}
        k = np.flatnonzero(st == OK)
        if len(k):
            mk = now[(channel, book)].set_index("match_id").loc[t["match_id"].to_numpy()[k]]
            r = base.iloc[k].reset_index(drop=True).assign(通道=channel, 市场=book, 下结论的那一份=main)
            rows.append(score_channel(channel, book, r, mk, model, k, hg, ag, settings, check))
    rows = pd.concat(rows, ignore_index=True).reindex(columns=ROW_COLS) if rows else pd.DataFrame(columns=ROW_COLS)
    excluded = pd.concat(excluded, ignore_index=True)[["通道", "市场", "状态", "周几", "场数"]]
    return rows, summarize(rows, settings), excluded, check


def score_channel(channel: str, book: str, r: pd.DataFrame, mk: pd.DataFrame, model: dict, k: np.ndarray,
                  hg: np.ndarray, ag: np.ndarray, settings: dict, check: dict) -> pd.DataFrame:
    """一个通道一家市场、比了的那些场（下标 k）：两边的概率、分数、差、命中。r 是这些场的基本信息，mk 是这些场的市场价。"""
    eps = settings["calc"]["logloss_eps"]
    hk, ak = hg[k], ag[k]
    if channel in ("胜平负", "比分"):
        r["市场价来自"] = np.where(mk["mo_pinnacle"].to_numpy() == 1, "平博", "多家平均")
        r["模型份数"] = r["市场份数"] = 1.0
    else:
        r["市场价来自"] = book
    if channel == "胜平负":
        y = model["y"][k]
        q = mk[["mo_p_home", "mo_p_draw", "mo_p_away"]].to_numpy(float)
        r["结算"] = [RESULT_NAMES[v] for v in y]
        r["模型概率"], r["市场概率"] = model["p3"][k, y], q[np.arange(len(k)), y]
        r["模型命中"] = (model["p3"][k].argmax(axis=1) == y).astype(float)
        r["市场命中"] = (q.argmax(axis=1) == y).astype(float)
    elif channel == "比分":
        n = model["total"].shape[1] // 2
        target = mk[["mo_p_home", "mo_p_draw", "mo_p_away", "mo_over25"]].to_numpy(float)
        lh, la, rho = fit_market_dc(target, n=n)
        mm = dc_matrices(lh, la, rho, n=n)
        got = derive(mm, hk, ak)
        err = np.abs(np.column_stack([got["p3"], got["over25"]]) - target)
        lo, hi = dc_rho_bounds(lh, la)
        check["市场比分矩阵"] = {"还原误差（最大）": float(err.max()), "还原误差（平均）": float(err.mean()),
                              "还原误差超过 0.001 的场数": int((err.max(axis=1) > 1e-3).sum()),
                              "rho 取了边界的场数": int(((rho <= lo + 1e-12) | (rho >= hi - 1e-12)).sum())}
        r["模型概率"], r["市场概率"] = model["cell"][k], got["cell"]
        r["模型命中"], r["市场命中"] = model["top_hit"][k], _top_hits(mm, hk, ak)
    else:
        line = mk["line"].to_numpy(float)
        n = model["total"].shape[1] // 2
        if channel == "亚盘让球":
            share = line_shares(model["diff"][k], np.arange(-n, n + 1), line, _ah_margin)
            outcome = [settle_margin(*_ah_margin(int(h - a), float(ln))) for h, a, ln in zip(hk, ak, line)]
        else:
            share = line_shares(model["total"][k], np.arange(2 * n + 1), line, _ou_margin)
            outcome = [settle_margin(*_ou_margin(int(h + a), float(ln))) for h, a, ln in zip(hk, ak, line)]
        q = mk["share"].to_numpy(float)
        r["线"], r["结算"], r["模型概率"], r["市场概率"] = line, outcome, share, q
        r["模型分数"] = [line_score(s, o, eps) for s, o in zip(share, outcome)]
        r["市场分数"] = [line_score(s, o, eps) for s, o in zip(q, outcome)]
        for side, s in (("模型", share), ("市场", q)):
            hits = [line_hit(v, o) for v, o in zip(s, outcome)]
            r[f"{side}命中"], r[f"{side}份数"] = [h for h, _ in hits], [w for _, w in hits]
    if channel in ("胜平负", "比分"):
        r["模型分数"] = -np.log(np.maximum(r["模型概率"].to_numpy(float), eps))
        r["市场分数"] = -np.log(np.maximum(r["市场概率"].to_numpy(float), eps))
    r["差"] = r["模型分数"] - r["市场分数"]
    return r


# ======================= 汇总：每个通道每家市场每一份 =======================
def summarize(rows: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """全部、每个联赛、两段，胜平负和比分再按市场是哪家分开。只有下结论的那一份的“全部”给结论，其余“只报告”。"""
    cmp = settings["compare"]
    out = []
    for channel, book, main in channel_keys(settings):
        g = rows[(rows["通道"] == channel) & (rows["市场"] == book)]
        g = g[g["差"].notna()]
        parts = [("全部", "全部", g)]
        parts += [("联赛", v, x) for v, x in g.groupby("联赛")]
        parts += [("两段", v, x) for v, x in g.groupby("两段")]
        if channel in ("胜平负", "比分"):
            parts += [("市场价来自", v, x) for v, x in g.groupby("市场价来自")]
        for how, which, x in parts:
            r = {"通道": channel, "市场": book, "分法": how, "哪一份": which, "场数": len(x)}
            if len(x):
                mean, lo, hi = bootstrap_mean_diff(x["差"], block_ids(x["比赛日期"], cmp["block"]),
                                                   cmp["n_boot"], cmp["confidence"], cmp["seed"])
                r.update({"赛季从": int(x["赛季"].min()), "赛季到": int(x["赛季"].max()),
                          "模型平均分": x["模型分数"].mean(), "市场平均分": x["市场分数"].mean(),
                          "差的平均": mean, "区间下限": lo, "区间上限": hi,
                          "结论": verdict(lo, hi) if main and how == "全部" else "只报告"})
                for side in ("模型", "市场"):
                    w = x[f"{side}份数"].sum()
                    r[f"{side}命中率"] = x[f"{side}命中"].sum() / w if w > 0 else np.nan
            out.append(r)
    return pd.DataFrame(out)


def main() -> None:
    settings = load_settings()
    b = settings["baseline"]
    ap = argparse.ArgumentParser(description="历史基线：冻结的版本在开发赛季按四个通道和市场各比一遍（只读）")
    ap.add_argument("--run", default=b["run"], help="回测结果文件夹名（data/backtest/ 下面，要有 matrices_<模型>.npz）")
    ap.add_argument("--db", default=None, help="数据库文件（不写 = settings.yaml 的 paths.database）")
    ap.add_argument("--refresh-odds", action="store_true", help="不用上次存的 baseline_odds.csv，重新取赔率")
    a = ap.parse_args()
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / a.run
    pred = load_predictions(run_dir)
    npz = np.load(run_dir / f"matrices_{b['model']}.npz")
    ids, mats = npz["match_id"], npz["matrix"]
    targets = pred[(pred["model"] == b["model"]) & (pred["horizon"] == b["horizon"])][["match_id", "cutoff_utc"]]
    targets = targets.drop_duplicates("match_id")
    cache, key_file = run_dir / "baseline_odds.csv", run_dir / "baseline_odds_settings.json"
    # 取赔率用到的设置变了（换公司、换线、换用途、采集时间规则、换数据库）就重新取，不用旧的
    key = json.dumps({"market_odds": settings["market_odds"], "baseline": {k: b[k] for k in ("usage", "line_books",
                      "late_hours", "model", "horizon")}, "db": str(a.db),
                      "pre_odds_hours": settings["features"].get("pre_odds_hours_before_kickoff"),
                      "pre_odds_collection": settings["features"].get("pre_odds_collection")},
                     ensure_ascii=False, sort_keys=True)
    odds = None
    if cache.exists() and key_file.exists() and key_file.read_text(encoding="utf-8") == key and not a.refresh_odds:
        odds = pd.read_csv(cache, encoding="utf-8-sig")
    if odds is None:
        fs = FeatureStore(open_readonly(a.db), "开发", settings)
        odds = fetch_odds(fs, targets, settings)
        odds.to_csv(cache, index=False, encoding="utf-8-sig")
        key_file.write_text(key, encoding="utf-8")
    rows, summary, excluded, check = build(pred, ids, mats, odds, settings)
    rows.to_csv(run_dir / "baseline_matches.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    summary.to_csv(run_dir / "baseline_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    excluded.to_csv(run_dir / "baseline_excluded.csv", index=False, encoding="utf-8-sig")
    (run_dir / "baseline_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(json.dumps(check, ensure_ascii=False, indent=2))
    s = summary[summary["分法"] == "全部"].drop(columns=["分法", "哪一份"])
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"结果存在 {run_dir}")


if __name__ == "__main__":
    main()
