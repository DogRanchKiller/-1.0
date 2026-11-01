"""预测页的“结论”：融合概率、倾向、信心等级、不确定因素（需求文档第 5 节“预测页怎么接融合、倾向、信心等级”）。

这个文件只做计算：拿到模型的比分矩阵和“融合用的赔率”（src/predict/preview.py 的 fusion_odds 算好的去水概率），
算出每场的结论。取赔率、存档在 preview.py，网页的文字在 src/web/preview_page.py。

和回测用同一套东西，这样回测定下的 w、门槛、等级规则上线后才适用：
  - 市场比分矩阵：src/common/markets.py 的 fit_market_dc（W9 回测用的同一个）；
  - 前几轮、升班马：src/evaluate 里信心等级回测用的 early_flags、promoted_flags；
  - 等级：confidence_study 的 levels（从 3 开始，碰上一个算数的因素降一级，没有赔率再降，最低 1）。
数字（w、门槛、降级因素、分歧多大算大）都在 settings.yaml 的 fusion、lean、confidence、confidence_study 段。

“倾向”只是“哪边更可能发生”，不是“值得买”：赔率已经把这个可能性算进去了，再扣掉抽水，多数倾向没有利润空间（第 3 节）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.markets import (asian_handicap, check_score_matrix, dc_matrices, fit_market_dc, over_under,
                                result_probs)
from src.evaluate.confidence_study import FACTOR_NAMES, LEVEL_NAMES, levels, promoted_flags
from src.evaluate.early_rounds import early_flags

FUSED, PARTIAL, NO_ODDS = "已融合", "赔率不全", "无赔率，未融合"
# 不确定因素（只提醒，不改等级；第 3 节）：标记名 → 页面上的说法
UNCERTAIN = {"gap": "模型和赔率分歧大", "early": "赛季前几轮", "promoted": "有升班马",
             "draw_risk": "平局风险高", "odds_move": "赔率最近大幅变动"}
EPS = 1e-12               # 比门槛时的容差，和回测（fusion_study）一样


# ======================= 前几轮、升班马 =======================
def season_flags(hist: pd.DataFrame, targets: pd.DataFrame, settings: dict) -> dict[int, dict]:
    """每场要预测的比赛：是不是这个联赛这个赛季的前几轮、有没有升班马。返回 {match_id: {"early": …, "promoted": …}}。

    和信心等级回测同一个数法（early_flags、promoted_flags），只是比赛表换成“现在以前的历史比赛 + 要预测的比赛”。
    注意：刚开球、比分还没进来的几场两边都不在，前几轮会少数几场（只在第 10 轮前后差一两场）。
    """
    cols = ["match_id", "competition", "season", "match_date", "kickoff_utc", "home_team", "away_team"]
    if targets.empty:
        return {}
    m = pd.concat([hist[cols], targets[cols]], ignore_index=True).drop_duplicates("match_id")
    early = early_flags(m, settings["confidence_study"]["early_rounds"])
    promoted = promoted_flags(m)
    return {int(i): {"early": bool(early[i]), "promoted": bool(promoted[i])} for i in targets["match_id"]}


# ======================= 融合 =======================
def market_matrix(fo: dict, n: int) -> tuple[np.ndarray, float]:
    """融合用的赔率（胜平负 + 大小球胜算）→ 市场比分矩阵（和 W9 回测同一个反推方法），和还原误差（四个数差得最多的）。"""
    target = [*fo["p_1x2"], fo["ou"]["p_over"]]
    lh, la, rho = fit_market_dc([target], n=n, ou_line=fo["ou"]["line"])
    m = dc_matrices(lh, la, rho, n)[0]
    got = [*result_probs(m), _over_share(m, fo["ou"]["line"])]
    return m, float(np.abs(np.array(got) - target).max())


def _over_share(m: np.ndarray, line: float) -> float:
    """比分矩阵在大小球 line 上“大”的胜算（2.5 球就是大 2.5 球的概率）。"""
    return float(over_under(m, line, "over")["win_share"])


def _same_shape(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """模型的比分矩阵换成和市场矩阵一样大（才能按格子相加）：大了把多出来的球数并进最后一行 / 列（“n 球及以上”），小了补 0。"""
    out = np.zeros(shape)
    r, c = shape
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            out[min(i, r - 1), min(j, c - 1)] += m[i, j]
    return out


def fuse(pred, fo: dict, settings: dict) -> dict:
    """模型的预测（比分矩阵、三个概率或 None）+ 融合用的赔率 → 融合结果。

    返回 status（已融合 / 赔率不全 / 无赔率，未融合；模型和赔率都没有是 None）、matrix（各玩法从它算；没有是 None）、
    p_1x2、over25、btts（没有是 None）、fit_error（市场矩阵的还原误差）。
      - 胜平负、大小球赔率都有：反推市场矩阵，融合矩阵 = w × 市场 + (1 − w) × 模型（模型没有比分矩阵时 w 当 1）；
        （市场矩阵算到 settings.yaml market_odds.max_goals 球；模型矩阵大小不一样时先换成一样大）
      - 只有胜平负：胜平负按 w 融合（模型没有比分矩阵时 w 当 1），其他玩法不写倾向（不凭空补大小球）；
      - 胜平负也没有：用模型自己的概率，信心等级另外降级。
    """
    w = float(settings["fusion"]["w"])
    model_m = None if pred is None or np.asarray(pred).ndim != 2 else check_score_matrix(pred)
    model_p = None if pred is None else (result_probs(model_m) if model_m is not None else np.asarray(pred, float))
    out = {"status": None, "matrix": None, "p_1x2": None, "over25": None, "btts": None, "fit_error": None, "w": w}
    if fo["p_1x2"] is None:
        if model_p is None:
            return out
        out.update(status=NO_ODDS, matrix=model_m, p_1x2=[float(x) for x in model_p])
    elif fo["ou"] is None:
        # 只有胜平负：胜平负按 w 混合，不凭空补别的玩法。模型没有比分矩阵时和“两样都有”一样 w 当 1
        p = np.asarray(fo["p_1x2"], float)
        if model_m is not None:
            p = w * p + (1 - w) * model_p
        out.update(status=PARTIAL, p_1x2=[float(x) for x in p / p.sum()])
        return out
    else:
        mk, err = market_matrix(fo, int(settings["market_odds"]["max_goals"]))
        m = mk if model_m is None or w == 1 else w * mk + (1 - w) * _same_shape(model_m, mk.shape)
        out.update(status=FUSED, matrix=m, fit_error=err, p_1x2=[float(x) for x in result_probs(m)])
    m = out["matrix"]
    if m is not None:
        out["over25"] = _over_share(m, 2.5)
        out["btts"] = float(m[1:, 1:].sum())
    return out


# ======================= 倾向 =======================
def leans(f: dict, ah_line: float | None, settings: dict) -> dict:
    """每个玩法的倾向（门槛 settings.yaml 的 lean，和 W9 回测的算法一样）。

    每个玩法：{"side": 偏向哪边, "p": 那边的概率（亚盘是胜算）, "lean": 够不够门槛}；算不出来是 None。
      result：side 0/1/2（主胜/平/客胜），p 最高的那个，够门槛 = 最高 − 第二高 ≥ lean.result
      draw：{"p": 平局概率, "risk": ≥ lean.draw}
      over25 / btts：side "over"/"under"、"yes"/"no"，够门槛 = 偏向那边 − 0.5 ≥ 门槛
      ah：side "home"/"away"，在 ah_line（主队的线）上较大的胜算；ah_line 是 None（没有主盘口，或赔率不在 1.5–2.7）时不算
    门槛是 null 的玩法一律不够（“无明显倾向”）。
    """
    th = settings["lean"]

    def enough(gap, key):
        # 减 1e-12：和回测 fusion_study 一样，正好等于门槛的算够（0.44 − 0.40 在电脑里是 0.03999…，不减就算不够）
        return th.get(key) is not None and gap >= th[key] - EPS

    out = {"result": None, "draw": None, "over25": None, "btts": None, "ah": None}
    if f["p_1x2"] is None:
        return out
    p = np.asarray(f["p_1x2"], float)
    order = np.argsort(-p, kind="stable")
    out["result"] = {"side": int(order[0]), "p": float(p[order[0]]),
                     "lean": bool(enough(p[order[0]] - p[order[1]], "result"))}
    out["draw"] = {"p": float(p[1]), "risk": bool(th.get("draw") is not None and p[1] >= th["draw"] - EPS)}
    for key, yes, no in (("over25", "over", "under"), ("btts", "yes", "no")):
        if f[key] is not None:
            side_yes = f[key] >= 0.5
            q = f[key] if side_yes else 1 - f[key]
            out[key] = {"side": yes if side_yes else no, "p": float(q), "lean": bool(enough(q - 0.5, key))}
    if f["matrix"] is not None and ah_line is not None:
        share = asian_handicap(f["matrix"], ah_line, "home")["win_share"]
        if not np.isnan(share):
            home = share >= 0.5
            q = share if home else 1 - share
            out["ah"] = {"side": "home" if home else "away", "p": float(q), "line": float(ah_line),
                         "lean": bool(enough(q - 0.5, "ah"))}
    return out


# ======================= 信心等级、不确定因素 =======================
def odds_move(pair) -> float | None:
    """(N 小时前, 现在) 同几家公司的胜平负概率（preview.py 的 move_pair），三项里变得最多的那项变了多少。没有可比的是 None。"""
    if pair is None:
        return None
    before_p, now_p = pair
    return float(np.abs(np.asarray(now_p, float) - np.asarray(before_p, float)).max())


def conclude(pred, model_p, fo: dict, move_pair, season: dict, settings: dict) -> dict:
    """一场比赛的结论。pred：模型的预测（比分矩阵 / 三个概率 / None）；model_p：模型的胜平负；
    fo：现在融合用的赔率；move_pair：看赔率变动用的 (N 小时前, 现在) 胜平负概率（没有是 None）；season：season_flags 这场的那一项。

    返回 status、w、p_1x2（融合概率，页面的主结论）、over25、btts、fit_error、leans、level（3/2/1，没有结论是 None）、
    level_name、flags（四个候选降级因素 + 平局风险高 + 赔率大幅变动）、uncertain（要提醒的不确定因素，页面上的说法）、odds_move。
    """
    conf, st = settings["confidence"], settings["confidence_study"]
    unknown = set(conf["downgrade_factors"]) - set(FACTOR_NAMES)
    if unknown:
        raise ValueError(f"settings.yaml confidence.downgrade_factors 里有不认识的因素：{sorted(unknown)}"
                         f"（只能是 {list(FACTOR_NAMES)}）")
    f = fuse(pred, fo, settings)
    ln = leans(f, fo["ah"]["line"] if fo.get("ah") else None, settings)
    has_odds = fo["p_1x2"] is not None
    gap = (model_p is not None and has_odds
           and float(np.abs(np.asarray(model_p, float) - np.asarray(fo["p_1x2"], float)).max()) >= st["gap"])
    move = odds_move(move_pair)
    flags = {"gap": bool(gap), "early": bool(season.get("early", False)), "promoted": bool(season.get("promoted", False)),
             "no_pinnacle": bool(has_odds and not fo["pinnacle"]),
             "draw_risk": bool(ln["draw"] and ln["draw"]["risk"]),
             "odds_move": bool(move is not None and move >= settings["preview"]["odds_move"]["min_change"])}
    level = None
    if f["status"] is not None:
        one = pd.DataFrame([{**flags, "has_odds": has_odds}])
        level = int(levels(one, list(conf["downgrade_factors"]), int(conf["no_odds_downgrade"]))[0])
    return {"status": f["status"], "w": f["w"], "p_1x2": f["p_1x2"], "over25": f["over25"], "btts": f["btts"],
            "fit_error": f["fit_error"], "leans": ln, "level": level, "level_name": LEVEL_NAMES.get(level),
            "flags": flags, "uncertain": [UNCERTAIN[k] for k in UNCERTAIN if flags[k]], "odds_move": move}
