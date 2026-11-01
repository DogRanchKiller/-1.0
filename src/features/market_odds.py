"""步骤 8 B 版的赔率特征（需求文档第 4 节“步骤 8 怎么做：B 版（赔率作为输入）”）。

一、算什么（每场一行，都按主队的角度写；梯度提升树再按“这队 / 对手”翻过来用）
  mo_p_home、mo_p_draw、mo_p_away：胜平负赛前赔率去水后的三个概率。
      选哪家：settings.yaml market_odds.result_books 按顺序，前面的公司有就用前面的（平博 → 多家平均，
      和“融合用哪家赔率”、三个检验同一条规则）。赔率缺一个、或不大于 1 的当作没有。
  mo_pinnacle：胜平负用的是不是平博（1 / 0）。
  mo_over25：大小球 2.5 赛前赔率去水后“大 2.5 球”的概率（market_odds.ou_book，多家平均）。
  mo_loglam_home、mo_loglam_away：市场预期进球（取 log）。用独立泊松比分矩阵反推：
      找主队、客队预期进球，让矩阵的主胜、平、客胜、大 2.5 球四个概率和上面四个数最接近（差的平方和最小），
      和第 5 节“市场比分矩阵”同一个办法（src/common/markets.py 的 fit_market_lambdas）。
      没有大小球的比赛不反推（只有胜平负三个数定不出两个预期进球的大小），是空值。
  去水都用 market_odds.devig_method（按比例，和回测标杆一样）。

二、从哪拿（都经过取数层，规矩和别的数据一样）
  以前的比赛：FeatureStore.historical_odds（比分能知道才给），history_odds_features 取。
  要预测的比赛：FeatureStore.pre_match_odds 按“本场”取（检查用途、运行模式、截止时间早于开球；
      赛前赔率开球前 24 小时才算能知道），attach_odds_features 接到目标比赛那张表上。
  只用 market_odds.source 这个来源（football-data）：别的来源的赛前赔率采集时间不一样，混在一起训练和预测对不上。
  两个函数都检查：取到的每一行“能知道的时间”都不晚于截止时间，否则停下报错。

三、为什么放在取数层这边
  模型不碰取数层（src/models/base.py 的两条规矩），赔率特征由回测框架取好、算好，再交给模型。
  数据只往一个方向流，这里不能引用模型（src/models），所以反推预期进球用的是 src/common/markets.py。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.markets import fit_market_lambdas
from src.common.odds import devig
from src.features.errors import LookAheadError

ODDS_COLS = ["mo_p_home", "mo_p_draw", "mo_p_away", "mo_over25", "mo_loglam_home", "mo_loglam_away", "mo_pinnacle"]
RES = ["home", "draw", "away"]
OU = ["over", "under"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"match_id": pd.Series(dtype=int), **{c: pd.Series(dtype=float) for c in ODDS_COLS}})


def odds_features(odds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """赔率表（取数层给的，一行 = 一场 × 一家公司 × 一种玩法）→ 每场一行特征（match_id + ODDS_COLS）。

    cfg：settings.yaml 的 market_odds 段。没有任何可用赔率的比赛不出现在结果里（接到比赛表上时是空值）。
    """
    if odds is None or odds.empty:
        return _empty()
    o = odds[odds["source"] == cfg["source"]]
    method = cfg["devig_method"]
    pin_book = cfg["pinnacle_book"]              # mo_pinnacle = 胜平负用的是不是平博（按名字认，改了顺序也不会弄错）
    # 胜平负：按 result_books 的顺序，每场用排在最前面、三个赔率都齐的那家
    parts, seen = [], set()
    for book in cfg["result_books"]:
        r = o[(o["bookmaker"] == book) & (o["market"] == "1x2")].dropna(subset=RES)
        r = r[(r[RES] > 1).all(axis=1) & ~r["match_id"].isin(seen)].drop_duplicates("match_id")
        if r.empty:
            continue
        p = devig(r[RES].to_numpy(dtype=float), method)
        parts.append(pd.DataFrame({"match_id": r["match_id"].to_numpy(int), "mo_p_home": p[:, 0],
                                   "mo_p_draw": p[:, 1], "mo_p_away": p[:, 2],
                                   "mo_pinnacle": float(book == pin_book)}))
        seen.update(int(x) for x in r["match_id"])
    res = pd.concat(parts, ignore_index=True) if parts else _empty()[["match_id", "mo_p_home", "mo_p_draw",
                                                                     "mo_p_away", "mo_pinnacle"]]
    # 大小球 2.5（一家公司一场只要一行）
    u = o[(o["bookmaker"] == cfg["ou_book"]) & (o["market"] == "ou")]
    u = u[np.isclose(pd.to_numeric(u["line"], errors="coerce"), float(cfg["ou_line"]))].dropna(subset=OU)
    u = u[(u[OU] > 1).all(axis=1)].drop_duplicates("match_id")
    over = devig(u[OU].to_numpy(dtype=float), method)[:, 0] if len(u) else np.zeros(0)
    ou = pd.DataFrame({"match_id": u["match_id"].to_numpy(int), "mo_over25": over})
    out = res.merge(ou, on="match_id", how="outer")
    if out.empty:
        return _empty()
    out["match_id"] = out["match_id"].astype(int)
    # 市场预期进球：胜平负和大 2.5 球都有的比赛才反推
    four = ["mo_p_home", "mo_p_draw", "mo_p_away", "mo_over25"]
    both = out[four].notna().all(axis=1).to_numpy()
    out["mo_loglam_home"], out["mo_loglam_away"] = np.nan, np.nan
    if both.any():
        lh, la = fit_market_lambdas(out.loc[both, four].to_numpy(float), int(cfg["max_goals"]))
        out.loc[both, "mo_loglam_home"], out.loc[both, "mo_loglam_away"] = np.log(lh), np.log(la)
    return out[["match_id", *ODDS_COLS]].astype({c: float for c in ODDS_COLS}).sort_values("match_id",
                                                                                         ignore_index=True)


def _check_known(odds: pd.DataFrame, cutoff: str, what: str) -> None:
    if len(odds) and (odds["known_at_utc"].astype(str) > str(cutoff)).any():
        raise LookAheadError(f"取数层返回了截止时间 {cutoff} 之后才能知道的{what}，回测停止")


def _odds_query(cfg: dict) -> dict:
    books = list(dict.fromkeys([*cfg["result_books"], cfg["ou_book"]]))
    return {"bookmaker": books, "market": ["1x2", "ou"]}


def history_odds_features(fs, cutoff: str, usage: str, competition=None, since=None) -> pd.DataFrame:
    """截止时间之前已经踢完的联赛比赛的赔率特征（和 historical_matches 同一个截止时间、同一个 since 规则）。"""
    cfg = fs.settings["market_odds"]
    o = fs.historical_odds(cutoff, usage, timing="pre", competition=competition, since=since, **_odds_query(cfg))
    _check_known(o, cutoff, "历史比赛赔率")
    return odds_features(o, cfg)


def attach_odds_features(fs, targets: pd.DataFrame, cutoff: str, usage: str) -> pd.DataFrame:
    """给要预测的比赛接上本场的赔率特征（截止时间之前能看到的赛前赔率；没有就是空值）。行的顺序不变。"""
    cfg = fs.settings["market_odds"]
    q = _odds_query(cfg)
    parts = []
    for mid in targets["match_id"].drop_duplicates():
        o = fs.pre_match_odds(int(mid), cutoff, usage, **q)
        _check_known(o, cutoff, "本场赔率")
        parts.append(o)
    feats = odds_features(pd.concat(parts, ignore_index=True) if parts else None, cfg)
    base = targets.drop(columns=[c for c in ODDS_COLS if c in targets])
    out = base.merge(feats, on="match_id", how="left")
    out.index = targets.index
    return out
