"""机器人回测的取数（需求文档第 14 节“范围”“下注时点和价格”“概率来源”“结算”）。

三样东西，都不偷看未来：
1. 概率：W9 那次回测（settings bots.run，例如 w9_gb_l7）的 predictions.csv 和 matrices_<模型>.npz。
   读进来第一步按赛季筛掉开发赛季以外的行，再检查剩下的全是开发赛季（season_role = dev），否则停下报错。
   模型的胜平负由比分矩阵算，和 predictions.csv 核对（差太多说明两个文件不是同一次回测）。
2. 下注时的赔率：取数层用途「机器人下注」，按“本场”、截止时间 = 开球前 24 小时取赛前胜平负；
   每一行“能知道的时间”都不能晚于截止时间，否则停下报错。收盘赔率、比分这时一律拿不到。
3. 结算：取数层用途「机器人结算」，站在开球后 72 小时取本场比分和平博收盘胜平负。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import resolve_path, season_role
from src.common.odds import devig, overround
from src.features import FeatureStore, LookAheadError
from src.features.known_time import to_utc_text

RES = ["home", "draw", "away"]
P3 = ["p_home", "p_draw", "p_away"]
MATRIX_TOL = 1e-4      # 矩阵算出的胜平负和 predictions.csv 最多差多少（矩阵按 float32 存，差一点点）


# ======================= 1. 概率 =======================
def load_bot_predictions(run: str, settings: dict, run_dir=None) -> tuple[pd.DataFrame, dict]:
    """读 data/backtest/<run>/ 的回测概率和比分矩阵 → (每场一行的表, 检查信息)。

    表的列：match_id、competition、season、match_date、cutoff_utc（回测时的截止时间）、p_home / p_draw / p_away（模型）。
    只留：settings bots 的模型和时点、联赛、first_season–last_season、predictions.csv 和矩阵都有的比赛。
    run_dir 给了就从那里读（测试用）。
    """
    cfg = settings["bots"]
    d = resolve_path(settings["backtest"]["output_dir"]) / run if run_dir is None else run_dir
    cols = ["match_id", "competition", "season", "match_date", "horizon", "cutoff_utc", "model", *P3]
    pred = pd.read_csv(d / "predictions.csv", usecols=cols, dtype={"cutoff_utc": str, "match_date": str})
    pred = pred[(pred["model"] == cfg["model"]) & (pred["horizon"] == cfg["horizon"])]
    # 第一步：按赛季筛（考试赛季一行都不往下传），再检查剩下的全是开发赛季
    if int(cfg["last_season"]) > int(settings["seasons"]["dev_last"]):
        raise ValueError(f"bots.last_season = {cfg['last_season']} 超过了开发赛季的最后一季 "
                         f"{settings['seasons']['dev_last']}，机器人回测不能用考试赛季")
    n_all = int(len(pred))
    pred = pred[pred["season"].between(int(cfg["first_season"]), int(cfg["last_season"]))]
    roles = {int(s): season_role(int(s), settings) for s in pred["season"].unique()}
    bad = sorted(s for s, r in roles.items() if r != "dev")
    if bad:
        raise ValueError(f"筛完以后还有不是开发赛季的行：{bad}，停下")
    pred = pred[pred["competition"].isin(cfg["leagues"])]
    pred = pred.dropna(subset=P3).drop_duplicates("match_id")

    npz = np.load(d / f"matrices_{cfg['model']}.npz")
    ids, mats = npz["match_id"], npz["matrix"]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    has = pred["match_id"].isin(pos.index)
    out = pred[has].sort_values("match_id", ignore_index=True)
    m = mats[pos.loc[out["match_id"]].to_numpy()].astype(float)
    m /= m.sum(axis=(1, 2), keepdims=True)
    i, j = np.indices(m.shape[1:])
    p3 = np.stack([(m * (i > j)).sum(axis=(1, 2)), (m * (i == j)).sum(axis=(1, 2)), (m * (i < j)).sum(axis=(1, 2))],
                  axis=1)
    gap = float(np.abs(p3 - out[P3].to_numpy(float)).max()) if len(out) else 0.0
    if gap > MATRIX_TOL:
        raise ValueError(f"比分矩阵算出的胜平负和 predictions.csv 差了 {gap:.2e}，两个文件不是同一次回测？")
    out[P3] = p3                       # 模型概率用矩阵算的（第 14 节“概率来源”）
    info = {"predictions.csv 里这个模型和时点的行数": n_all, "开发赛季、这几个联赛有概率的比赛": int(len(pred)),
            "没有比分矩阵的比赛": int((~has).sum()), "有概率也有矩阵的比赛": int(len(out)),
            "矩阵和 predictions.csv 胜平负最大差": gap, "赛季": sorted(roles)}
    return out[["match_id", "competition", "season", "match_date", "cutoff_utc", *P3]], info


# ======================= 2. 下注时的赔率 =======================
def _price_ok(o3: np.ndarray, rng: list) -> np.ndarray:
    """这组胜平负赔率像不像样：三个都有、都 > 1，倒数之和在 quality.overround 的这一档里。"""
    ok = ~np.isnan(o3).any(axis=1) & (np.nan_to_num(o3, nan=0.0) > 1).all(axis=1)
    res = np.zeros(len(o3), dtype=bool)
    if ok.any():
        s = overround(o3[ok])
        res[ok] = (s >= float(rng[0])) & (s <= float(rng[1]))
    return res


def load_bot_odds(fs: FeatureStore, matches: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """每场在截止时间（开球前 24 小时）能看到的赛前胜平负 → (比赛表, 每种价格一张表, 融合概率)。

    比赛表：matches 里取数层认的比赛，加 cutoff_utc（取数层按「机器人下注」重算；和回测的截止时间对不上就报错）。
    每种价格（settings bots.price_books：平博 / 最高）一张表，和比赛表一行对一行：
      home、draw、away、bookmaker、known_at_utc、ok（价格像样才下）。
    融合概率：fusion_books 按顺序选第一家三个赔率都齐的，按比例去水（w = 1，融合就是赔率概率）；没有的场是空值。
    """
    cfg = settings["bots"]
    usage = cfg["usage_bet"]
    books = {name: b["bookmaker"] for name, b in cfg["price_books"].items()}
    want = list(dict.fromkeys([*books.values(), *cfg["fusion_books"]]))
    keep, cutoffs, parts = [], [], []
    n_unknown = 0
    for mid, old_cut in matches[["match_id", "cutoff_utc"]].itertuples(index=False):
        try:
            cut = fs.cutoff_before(int(mid), float(cfg["hours_before_kickoff"]), usage)
        except ValueError:                 # 数据库里没有这场（取数层不认），不下
            n_unknown += 1
            continue
        if isinstance(old_cut, str) and old_cut and to_utc_text(old_cut) != cut:
            raise ValueError(f"比赛 {mid} 的截止时间：回测里是 {old_cut}，取数层现在算出 {cut}，"
                             f"数据库和回测不是同一版？停下")
        o = fs.pre_match_odds(int(mid), cut, usage, bookmaker=want, market="1x2")
        if len(o) and (o["known_at_utc"].astype(str) > cut).any():
            raise LookAheadError(f"取数层返回了截止时间 {cut} 之后才能知道的赔率（比赛 {mid}），停下")
        parts.append(o[o["source"] == cfg["odds_source"]])
        keep.append(int(mid))
        cutoffs.append(cut)
    m = matches[matches["match_id"].isin(keep)].copy()
    m["cutoff_utc"] = m["match_id"].map(dict(zip(keep, cutoffs)))
    m = m.reset_index(drop=True)
    allo = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["match_id", "bookmaker", *RES, "known_at_utc"])
    allo = allo.dropna(subset=RES).drop_duplicates(["match_id", "bookmaker"])

    def table(book: str) -> pd.DataFrame:
        t = allo[allo["bookmaker"] == book][["match_id", *RES, "known_at_utc"]]
        t = m[["match_id"]].merge(t, on="match_id", how="left")
        t["bookmaker"] = book
        return t

    odds, info = {}, {"取数层不认、不下的比赛": n_unknown, "取数层认的比赛": int(len(m))}
    q = settings["quality"]["overround"]
    for name, b in cfg["price_books"].items():
        t = table(b["bookmaker"])
        o3 = t[RES].to_numpy(dtype=float)
        t["ok"] = _price_ok(o3, q[b["overround"]])
        has = ~np.isnan(o3).any(axis=1)
        info[f"{name}：有赛前胜平负的比赛"] = int(has.sum())
        info[f"{name}：价格不像样、不下的比赛"] = int((has & ~t["ok"]).sum())
        odds[name] = t
    # 融合概率（w = 1）：按 fusion_books 的顺序，第一家三个赔率都齐、都 > 1 的去水
    fusion = np.full((len(m), 3), np.nan)
    fusion_book = pd.Series([None] * len(m), dtype=object)
    for b in cfg["fusion_books"]:
        o3 = table(b)[RES].to_numpy(dtype=float)
        free = np.isnan(fusion).any(axis=1) & ~np.isnan(o3).any(axis=1)
        free &= (np.nan_to_num(o3, nan=0.0) > 1).all(axis=1)
        if free.any():
            fusion[free] = devig(o3[free], cfg["devig_method"])
            fusion_book[free] = b
    info["有融合概率的比赛"] = int((~np.isnan(fusion).any(axis=1)).sum())
    info["融合用平博的比赛"] = int((fusion_book == cfg["fusion_books"][0]).sum())
    fus = pd.DataFrame(fusion, columns=P3).assign(match_id=m["match_id"], fusion_book=fusion_book)
    return m, odds, fus


# ======================= 3. 结算用的比分和收盘赔率 =======================
def load_settlement(fs: FeatureStore, matches: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """站在开球后 settle_after_kickoff_hours 小时（用途「机器人结算」），每场取比分和平博收盘胜平负。

    截止时间 = 开球前 24 小时，所以“开球后 72 小时” = 截止时间 + 24 + 72 小时（只有日期的比赛也一样，都从最早开球算）。
    返回每场一行：match_id、result（H/D/A，没有比分是空）、home_goals、away_goals、result_known_utc
    （比分能知道的时间；没有比分的按结算时间算，资金模拟里这一注在那时退回）、close_home / close_draw / close_away。
    """
    cfg = settings["bots"]
    usage = cfg["usage_settle"]
    after = pd.Timedelta(hours=float(cfg["hours_before_kickoff"]) + float(cfg["settle_after_kickoff_hours"]))
    rows = []
    for mid, cut in matches[["match_id", "cutoff_utc"]].itertuples(index=False):
        when = to_utc_text(pd.Timestamp(cut) + after)
        r = fs.match_result(int(mid), when, usage)
        if not r.empty and pd.notna(r.iloc[0]["result"]):
            x = r.iloc[0]
            res, hg, ag, known = x["result"], int(x["home_goals"]), int(x["away_goals"]), str(x["known_at_utc"])
        else:
            res, hg, ag, known = None, None, None, when
        c = fs.closing_odds(int(mid), when, usage, bookmaker=cfg["clv_bookmaker"], market="1x2")
        c = c[c["source"] == cfg["odds_source"]].dropna(subset=RES)
        close = c.iloc[0][RES].to_numpy(dtype=float) if len(c) else np.full(3, np.nan)
        rows.append((int(mid), res, hg, ag, known, *close))
    return pd.DataFrame(rows, columns=["match_id", "result", "home_goals", "away_goals", "result_known_utc",
                                       "close_home", "close_draw", "close_away"])
