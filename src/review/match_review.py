"""单场复盘（需求文档第 15 节“单场复盘”）：每场已经踢完、有预测的比赛写一个 data/review/matches/<match_id>.json。

内容（文件格式写在需求文档第 15 节“单场复盘文件的格式”）：
1. 各版本预测（T-24 / T-6 / T-1 / 预览）和收盘赔率的概率、分数；哪一个最接近结果（胜平负 log loss 最小）。
2. 实际比分、xG、红牌。
3. 赔率是否往模型的方向走（出预测时的赔率 → 收盘赔率）。
4. 误差原因自动归类（运气型 / 信息缺失型 / 意外事件型 / 预测偏差型），判断规则见 classify_error 和第 15 节。
5. 自动标签（德比、争冠、保级）：调用 src/review/labels.py 的 label_season 现算，不改它。

数据都从 reconcile.collect 来（比分、xG、红牌、收盘赔率都经过取数层，用途「复盘」）。
结果只用来记录和展示，不反过来调任何参数。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureStore
from src.features.standings import load_deductions
from src.review.labels import label_season, load_derbies
from src.review.reconcile import CLOSE_KIND, SOURCE_CLOSE, SOURCE_FUSION, SOURCE_MODEL, kind_rank, score_metrics, \
    over25_logloss

BEIJING = timezone(timedelta(hours=8))
OUTCOMES = ["主胜", "平局", "客胜"]
RESULT_INDEX = {"H": 0, "D": 1, "A": 2}
KIND_NAMES = {"T-24": "赛前 24 小时版", "T-6": "赛前 6 小时版", "T-1": "赛前 1 小时版", "preview": "预览版"}


# ======================= 小工具 =======================
def bj_text(utc_text: str | None) -> str | None:
    """UTC 文字 → 北京时间文字（只用来显示）。"""
    if not utc_text:
        return None
    t = pd.Timestamp(utc_text)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_convert(BEIJING).strftime("%Y-%m-%d %H:%M")


def r4(x) -> float | None:
    """保留 4 位小数；空值、NaN → None（JSON 里写 null）。"""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def p_list(p) -> list | None:
    return None if p is None else [r4(v) for v in p]


def write_json(path: Path, data: dict) -> None:
    """先写临时文件再改名，不会留下写了一半的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def team_names_zh(con: sqlite3.Connection, teams) -> dict:
    """球队中文名（teams 表的 name_zh，和预览页一样只用来显示）。没有表、没有中文名 → 不给，显示时用英文名。"""
    teams = sorted(set(teams))
    if not teams:
        return {}
    try:
        q = ",".join("?" * len(teams))
        return {t: n for t, n in con.execute(f"select team_id, name_zh from teams where team_id in ({q})", teams) if n}
    except sqlite3.Error:
        return {}


# ======================= 自动标签 =======================
def attach_labels(fs: FeatureStore, matches: list[dict], settings: dict) -> dict[int, dict]:
    """每场的自动标签。按（联赛, 赛季）调用 labels.label_season 现算（实时模式取数层，用途「复盘」）。

    算不出来（缺德比名单、积分规则没写这个联赛等）不让整个复盘失败：这几场的标签写“算不出：原因”。
    返回 {match_id: {"德比": ..., "争冠战": ..., "保级战": ..., "赛前名次": ...} 或 {"算不出": 原因}}。
    """
    out: dict[int, dict] = {}
    if not matches:
        return out
    try:
        derbies = load_derbies(settings)
        deductions = load_deductions(settings)
    except Exception as e:   # noqa: BLE001  —— 配置文件读不了，所有比赛都写原因
        return {m["match_id"]: {"算不出": f"{type(e).__name__}：{e}"} for m in matches}
    groups: dict[tuple, list[int]] = {}
    for m in matches:
        groups.setdefault((m["competition"], m["season"]), []).append(m["match_id"])
    for (comp, season), ids in groups.items():
        try:
            lab = label_season(fs, comp, season, derbies, settings, deductions)
        except Exception as e:   # noqa: BLE001
            for i in ids:
                out[i] = {"算不出": f"{type(e).__name__}：{e}"}
            continue
        lab = lab.set_index("match_id") if not lab.empty else lab
        for i in ids:
            if lab.empty or i not in lab.index:
                out[i] = {"算不出": "标签程序没有给这场比赛算（赛程里找不到）"}
                continue
            r = lab.loc[i]
            out[i] = {
                "德比": str(r["derby_name"]) if bool(r["is_derby"]) else None,
                "争冠战": bool(r["is_title_race"]), "保级战": bool(r["is_relegation_race"]),
                "赛前名次": {"主队": int(r["home_position"]), "客队": int(r["away_position"])},
                "赛前积分": {"主队": int(r["home_points"]), "客队": int(r["away_points"])},
                "同分排名没分出来": bool(r["tie_unresolved"]),
            }
    return out


# ======================= 选正式版、误差归类、赔率方向 =======================
def official_version(versions: list[dict], order: list[str]) -> dict:
    """正式成绩用哪一版：按 review.official_order 取这场有的第一个 kind（T-1 → T-6 → T-24 → 预览）；
    都没有（表里出现了别的 kind）就用 prediction_id 最大的那版。
    第 15 节说“有手动缺阵调整时用调整后的版本”：现在还没有调整版，以后加了 kind 写进 official_order 最前面即可。"""
    by_kind = {v["kind"]: v for v in versions}
    for k in order:
        if k in by_kind:
            return by_kind[k]
    return max(versions, key=lambda v: v["prediction_id"])


def classify_error(m: dict, v: dict, cfg: dict) -> dict:
    """误差原因归类（第 15 节的表，具体规则写在第 15 节“误差原因怎么归类”）。看的是正式那一版的“模型概率”。

    1. 先看是不是“明显没料到”：模型给实际结果的概率 < errors.miss_below（约三分之一）才归类，否则“没有明显失误”。
    2. 按表的顺序看符合哪几条，第一条符合的就是“类型”，其余的也列出来：
       - 运气型：按 xG 看的结果（两队 xG 差 ≥ errors.xg_margin 算多的那队赢，否则算平）不是实际结果，
         而且模型给“xG 结果”的概率比给实际结果的高（xG 支持模型，比分不支持）。
       - 信息缺失型：收盘时实际结果的概率比出预测时的赔率高 errors.close_move 以上（市场后来知道了模型不知道的事）。
         “首发有重要缺阵”还没有数据判断，暂时不看。
       - 意外事件型：这场有红牌。数据里没有红牌、点球发生在第几分钟，“早早”暂时没法判断，有红牌就算。
       - 预测偏差型：以上都不是。
    3. 缺数据没法判断的条目写进“没法判断”（例如没有 xG、还没有收盘赔率），这时“预测偏差型”可能是数据不够造成的。
    """
    res = RESULT_INDEX[m["result"]]
    p = v["model_p"]
    p_res = float(p[res])
    out = {"看的版本": v["kind"], "模型给实际结果的概率": r4(p_res), "类型": None, "符合": [], "没法判断": []}
    miss = float(cfg["miss_below"])
    if p_res >= miss:
        out["类型"] = "没有明显失误"
        out["说明"] = f"模型给实际结果（{OUTCOMES[res]}）{p_res:.0%}，不低于 {miss:.0%}，不归类"
        return out
    # 运气型
    if m["xg"] is None:
        out["没法判断"].append("运气型：这场没有 xG")
    else:
        hx, ax = m["xg"]
        d, mg = hx - ax, float(cfg["xg_margin"])
        xg_res = 0 if d >= mg else 2 if d <= -mg else 1
        if xg_res != res and p[xg_res] > p[res]:
            out["符合"].append({"类型": "运气型", "依据": f"xG {hx:.2f} 比 {ax:.2f}，按 xG 看是{OUTCOMES[xg_res]}；"
                                                         f"模型给{OUTCOMES[xg_res]} {p[xg_res]:.0%}，比实际结果高"})
    # 信息缺失型
    c = m["close"]
    if c is None:
        out["没法判断"].append("信息缺失型：这场还没有收盘赔率")
    elif v["odds_p"] is None:
        out["没法判断"].append("信息缺失型：这一版没有记出预测时的赔率")
    else:
        move = float(c["p_1x2"][res] - v["odds_p"][res])
        if move >= float(cfg["close_move"]):
            out["符合"].append({"类型": "信息缺失型", "依据": f"实际结果（{OUTCOMES[res]}）的赔率概率从出预测时 "
                                                           f"{v['odds_p'][res]:.0%} 涨到收盘 {c['p_1x2'][res]:.0%}"})
    # 意外事件型
    if m["red"] is None:
        out["没法判断"].append("意外事件型：这场没有红牌数据（点球、第几分钟都没有数据）")
    elif sum(m["red"]) > 0:
        out["符合"].append({"类型": "意外事件型", "依据": f"红牌：主队 {m['red'][0]}、客队 {m['red'][1]}"
                                                       "（没有第几分钟的数据，有红牌就算）"})
    out["类型"] = out["符合"][0]["类型"] if out["符合"] else "预测偏差型"
    out["说明"] = f"模型给实际结果（{OUTCOMES[res]}）只有 {p_res:.0%}，归为{out['类型']}"
    if out["类型"] == "预测偏差型" and out["没法判断"]:
        out["说明"] += "（有几条缺数据没法判断，可能不是真的偏差）"
    return out


def odds_direction(v: dict, close: dict | None, cfg: dict) -> dict:
    """赔率是否往模型的方向走：看模型和出预测时赔率分歧最大的那个结果，收盘时赔率概率往哪边动。"""
    if v["odds_p"] is None:
        return {"结论": "没法判断", "原因": "这一版没有记出预测时的赔率"}
    if close is None:
        return {"结论": "没法判断", "原因": "这场还没有收盘赔率"}
    o, c, mp = np.asarray(v["odds_p"]), np.asarray(close["p_1x2"]), np.asarray(v["model_p"])
    gap = mp - o
    k = int(np.argmax(np.abs(gap)))
    move = float(c[k] - o[k])
    dmin = float(cfg["direction_min"])
    if abs(gap[k]) < dmin:
        verdict = "模型和当时赔率几乎一样，谈不上方向"
    elif abs(move) < dmin:
        verdict = "赔率基本没动"
    elif np.sign(move) == np.sign(gap[k]):
        verdict = "往模型的方向走"
    else:
        verdict = "往模型的反方向走"
    return {"看的结果": OUTCOMES[k], "模型": r4(mp[k]), "出预测时赔率": r4(o[k]), "收盘赔率": r4(c[k]),
            "收盘赔率公司": close["book"], "结论": verdict}


# ======================= 拼一场的复盘 =======================
def version_entry(m: dict, v: dict) -> dict:
    """一版预测：概率、分数（模型、融合各一组）。"""
    hg, ag = m["home_goals"], m["away_goals"]
    e = {"版本": v["kind"], "名称": KIND_NAMES.get(v["kind"], v["kind"]), "prediction_id": v["prediction_id"],
         "模型": v["model"], "模型版本": v["model_version"],
         "截止时间_utc": v["cutoff_utc"], "截止时间_北京": bj_text(v["cutoff_utc"]),
         "计划时点": v["plan_hours"], "晚了几分钟": v["late_minutes"], "改期重出": v["rescheduled"],
         "出预测时的赔率概率": p_list(v["odds_p"]), "融合状态": v["fusion_status"]}
    for name, p, mat in ((SOURCE_MODEL, v["model_p"], v["model_matrix"]), (SOURCE_FUSION, v["fusion_p"], v["fusion_matrix"])):
        if p is None:
            e[name] = None
            continue
        sc = score_metrics(p, mat, hg, ag)
        e[name] = {"概率": p_list(p), "有比分矩阵": mat is not None, **{k: r4(x) for k, x in sc.items()}}
    return e


def close_entry(m: dict) -> dict | None:
    c = m["close"]
    if c is None:
        return None
    sc = score_metrics(c["p_1x2"], None, m["home_goals"], m["away_goals"])
    sc["logloss_over25"] = over25_logloss(c["p_over25"], m["home_goals"], m["away_goals"])
    return {"版本": CLOSE_KIND.format(book=c["book"]), "公司": c["book"], "概率": p_list(c["p_1x2"]),
            "大2.5球概率": r4(c["p_over25"]), **{k: r4(x) for k, x in sc.items()}}


def closest(official: dict, close: dict | None) -> dict:
    """正式那一版的模型、融合，和收盘赔率，谁的胜平负 log loss 最小（最接近结果）。"""
    cands = [(SOURCE_MODEL, official[SOURCE_MODEL]), (SOURCE_FUSION, official[SOURCE_FUSION]), (SOURCE_CLOSE, close)]
    cands = [(n, e["logloss_1x2"]) for n, e in cands if e is not None and e.get("logloss_1x2") is not None]
    best = min(cands, key=lambda x: x[1])
    return {"最接近": best[0], "比较的是": [n for n, _ in cands], "胜平负log_loss": {n: x for n, x in cands},
            "说明": "只有模型一个来源，没有可比的" if len(cands) == 1 else f"{best[0]}最接近结果（胜平负 log loss 最小）"}


def info_trend(entries: list[dict]) -> dict:
    """“信息越多是否越准”：T-24 → T-6 → T-1 模型的胜平负 log loss 是不是一路变小。预览版出的时间不固定，不比。"""
    seq = [(e["版本"], e[SOURCE_MODEL]["logloss_1x2"]) for e in entries
           if e["版本"] in ("T-24", "T-6", "T-1") and e[SOURCE_MODEL] is not None]
    if len(seq) < 2:
        return {"结论": "定时版本不到两版，没法比", "各版": dict(seq)}
    xs = [x for _, x in seq]
    if all(b < a for a, b in zip(xs, xs[1:])):
        verdict = "越来越准"
    elif all(b > a for a, b in zip(xs, xs[1:])):
        verdict = "越来越不准"
    else:
        verdict = "有升有降"
    return {"结论": verdict, "各版": dict(seq)}


def build_review(m: dict, labels: dict | None, names_zh: dict, settings: dict, generated_utc: str) -> dict:
    """一场比赛的复盘内容（写进 matches/<match_id>.json，周报月报也用它）。"""
    cfg = settings["review"]
    kinds = list(cfg["kinds"])
    vs = sorted(m["versions"], key=lambda v: kind_rank(v["kind"], kinds))
    entries = [version_entry(m, v) for v in vs]
    off_v = official_version(vs, list(cfg["official_order"]))
    off = next(e for e in entries if e["prediction_id"] == off_v["prediction_id"])
    close = close_entry(m)
    err = classify_error(m, off_v, cfg["errors"])
    home = names_zh.get(m["home_team"], m["home_team"])
    away = names_zh.get(m["away_team"], m["away_team"])
    near = closest(off, close)
    res = RESULT_INDEX[m["result"]]
    line = (f"{home} {m['home_goals']}-{m['away_goals']} {away}：{KIND_NAMES.get(off['版本'], off['版本'])}模型给"
            f"{OUTCOMES[res]} {off_v['model_p'][res]:.0%}，{err['类型']}")
    if near["比较的是"][1:]:
        line += f"；{near['最接近']}最接近结果"
    return {
        "match_id": m["match_id"], "联赛": m["competition"], "赛季": m["season_label"], "season": m["season"],
        "主队": m["home_team"], "客队": m["away_team"], "主队中文": names_zh.get(m["home_team"]),
        "客队中文": names_zh.get(m["away_team"]), "比赛日期": m["match_date"],
        "开球时间_utc": m["kickoff_utc"], "开球时间_北京": bj_text(m["kickoff_utc"]),
        "一句话": line,
        "实际": {"比分": [m["home_goals"], m["away_goals"]], "结果": OUTCOMES[res],
                 "xG": None if m["xg"] is None else [r4(x) for x in m["xg"]],
                 "红牌": None if m["red"] is None else list(m["red"]), "比分数据来源": m["stats_source"]},
        "正式版本": off["版本"],
        "各版本": entries,
        "收盘赔率": close,
        "最接近结果": near,
        "信息越多越准吗": info_trend(entries),
        "赔率方向": odds_direction(off_v, m["close"], cfg["errors"]),
        "误差归类": err,
        "自动标签": labels,
        "生成时间_utc": generated_utc,
        "说明": "单场结果大多是运气，只用来记录和展示，不据此改模型或任何参数。",
    }


def write_match_files(out_dir: Path, reviews: list[dict]) -> int:
    """每场写一个 <match_id>.json；全部重算，所以这次没有的旧文件（例如存档被删）一起删掉。返回写了几个。"""
    folder = Path(out_dir) / "matches"
    folder.mkdir(parents=True, exist_ok=True)
    keep = {f"{r['match_id']}.json" for r in reviews}
    for p in folder.glob("*.json"):
        if p.name not in keep and p.stem.isdigit():
            p.unlink()
    for r in reviews:
        write_json(folder / f"{r['match_id']}.json", r)
    return len(reviews)
