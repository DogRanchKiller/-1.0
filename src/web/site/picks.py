"""四个玩法的预测和命中率（规则 picks_v1，需求文档第 3 节“四个玩法的预测和命中率”）。

每场比赛用开球前最新的那一版存档（和单场页、战绩页同一版），四个玩法各给一个预测：
  胜平负：最后的胜平负概率里最大的那项（一样大时按 主胜 > 平 > 客胜）；
  让球（亚盘）：存档里写的亚盘倾向那一边（主盘口上胜算较大的一边）；接融合以前的旧存档没有倾向，就用“模型亚盘主队胜算”；
    新存档没写亚盘倾向（没有主盘口，或两边赔率不在 1.5–2.7）就不预测；
  大小球：固定看 2.5 球，大 2.5 球的概率 ≥ 50% 预测“大”，否则“小”（旧存档只在大小球主盘口正好是 2.5 时用模型的胜算）；
  比分：存下来的比分矩阵里最大的那一格（先用融合后的，没有用模型自己的）。
这些都是从“只加不改”的预测存档算出来的，不另外存表；规则改了就另起 picks_v2，旧的照留。

注意：
- 不管有没有到“写倾向”的门槛，四个玩法都给预测（这样每场都能算命中率）；没到门槛的，页面上标“接近五五开”。
- “预测”只是“哪个更可能”，不是“值得买”：赔率已经把可能性算进去了。页面上不写任何下注建议。
- 比分预测和胜平负预测可能对不上（例如胜平负预测主胜，最可能的一个比分却是 1–1）：
  平局的概率集中在 0–0、1–1 几个格子，主胜的概率分散在 1–0、2–0、2–1……很多格子，所以单个最大的格子常常是平局。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.common.config import resolve_path
from src.common.settlement import settle_asian_handicap, settle_over_under
from src.web.site.common import pct0
from src.web.site.data import matrix

RULE = "picks_v1"
OU_LINE = 2.5                      # 大小球固定看这条线（picks_v1 定下的，存档里融合后只存了“大2.5球”）
MARKETS = ("1x2", "ah", "ou", "score")
MARKET_NAMES = {"1x2": "胜平负", "ah": "让球（亚盘）", "ou": "大小球 2.5", "score": "比分"}
# 结算结果 → 算命中率用的分值（和期末考试算亚盘命中率的写法一样）：赢 1、赢半 0.5、走 0、输半 −0.5、输 −1
VALUE = {"win": 1.0, "half_win": 0.5, "push": 0.0, "half_loss": -0.5, "loss": -1.0}
OUTCOME_TEXT = {"win": "中", "half_win": "赢半", "push": "走盘", "half_loss": "输半", "loss": "没中"}
# 存档“融合.状态”里表示真的用上了赔率的两种（和 src/predict/conclusion.py 的 FUSED、PARTIAL 一样）。
# “无赔率，未融合”时存档的“融合概率”其实就是模型自己的概率，不能当成赔率
ODDS_USED = ("已融合", "赔率不全")


# ---------- 读存档 ----------
def _num(x) -> float | None:
    """存档里的一个数：是有限的数字就返回 float，否则 None（缺了、写成文字、NaN 都算没有）。"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fusion(row: dict) -> dict:
    f = (row.get("detail") or {}).get("融合")
    return f if isinstance(f, dict) else {}


def used_odds(row: dict) -> bool:
    """这一版最后的概率有没有用上赔率（融合.状态 是“已融合”或“赔率不全”）。接融合以前的旧版、没有赔率的版本都是 False。"""
    return _fusion(row).get("状态") in ODDS_USED


def _old(row: dict) -> bool:
    """接融合以前的旧存档：没有“融合.倾向”这一块。只有这种才退回去用模型在赔率主盘口上的胜算。"""
    return not isinstance(_fusion(row).get("倾向"), dict)


def _src(row: dict) -> str:
    return "融合" if used_odds(row) else "模型"


def _lean(row: dict, key: str) -> dict | None:
    """存档里某个玩法的倾向（conclusion.leans 存的 {"side", "p", "lean", ...}）；没有或格式不对返回 None。"""
    ln = _fusion(row).get("倾向")
    item = ln.get(key) if isinstance(ln, dict) else None
    return item if isinstance(item, dict) else None


# ---------- 四个玩法各自的预测 ----------
def pick_1x2(row: dict) -> dict | None:
    """胜平负：最后的概率（有融合用融合的，没有用模型的）里最大的那项。"""
    fp = (row.get("detail") or {}).get("融合概率")
    if isinstance(fp, list) and len(fp) == 3 and all(_num(x) is not None for x in fp):
        p = [float(x) for x in fp]          # 没有赔率时这里存的就是模型的概率，来源看 used_odds
    else:
        p = [_num(row.get(k)) for k in ("p_home", "p_draw", "p_away")]
        if any(x is None for x in p):
            return None
    src = _src(row)
    side = int(np.argmax(p))            # np.argmax 一样大时取前面的：主胜 > 平 > 客胜
    lean = _lean(row, "result")
    enough = bool(lean.get("lean")) if lean is not None and lean.get("side") == side else None
    return {"side": side, "p": p[side], "source": src, "lean": enough}


def pick_ah(row: dict) -> dict | None:
    """让球（亚盘）：主盘口上胜算较大的那一边。line 永远是主队的盘口（负数 = 主队让球）。"""
    lean = _lean(row, "ah")
    if lean is not None and lean.get("side") in ("home", "away") and _num(lean.get("line")) is not None \
            and _num(lean.get("p")) is not None:
        return {"side": lean["side"], "line": float(lean["line"]), "p": float(lean["p"]), "source": _src(row),
                "lean": bool(lean.get("lean"))}
    if not _old(row):
        return None                     # 新存档没写亚盘倾向 = 没有合格的主盘口，和历史参考一样不预测
    # 旧存档（接融合之前）：只存了模型在赔率主盘口上的主队胜算
    d = row.get("detail") or {}
    line, share = _num(d.get("亚盘主盘口")), _num(d.get("模型亚盘主队胜算"))
    if line is None or share is None:
        return None
    home = share >= 0.5
    return {"side": "home" if home else "away", "line": line, "p": share if home else 1 - share, "source": "模型",
            "lean": None}


def pick_ou(row: dict) -> dict | None:
    """大小球：固定 2.5 球，大 2.5 球的概率 ≥ 50% 预测“大”。"""
    lean = _lean(row, "over25")
    over = None
    if lean is not None and lean.get("side") in ("over", "under") and _num(lean.get("p")) is not None:
        q = float(lean["p"])
        over, src, enough = (q if lean["side"] == "over" else 1 - q), _src(row), bool(lean.get("lean"))
    elif _num(_fusion(row).get("大2.5球")) is not None:
        over, src, enough = float(_fusion(row)["大2.5球"]), _src(row), None
    elif _old(row):
        # 旧存档：模型在赔率大小球主盘口上的胜算，只有主盘口正好是 2.5 时才能用
        d = row.get("detail") or {}
        if _num(d.get("大小球主盘口")) == OU_LINE and _num(d.get("模型大球胜算")) is not None:
            over, src, enough = float(d["模型大球胜算"]), "模型", None
    if over is None:
        return None
    side = "over" if over >= 0.5 else "under"
    return {"side": side, "line": OU_LINE, "p": over if side == "over" else 1 - over, "source": src, "lean": enough}


def pick_score(row: dict, tol: float) -> dict | None:
    """比分：存下来的比分矩阵里概率最大的那一格（先用融合后的矩阵，没有用模型自己的）。没存矩阵的版本不预测。"""
    d = row.get("detail") or {}
    for key, src in (("融合比分矩阵", "融合"), ("模型比分矩阵", "模型")):
        m = matrix(d, key, tol)
        if m is not None:
            h, a = np.unravel_index(int(np.argmax(m)), m.shape)
            return {"score": (int(h), int(a)), "p": float(m[h, a]), "source": src if used_odds(row) else "模型",
                    "lean": None}
    return None


def match_picks(row: dict, settings: dict) -> dict:
    """一版存档的四个预测：{"1x2": ..., "ah": ..., "ou": ..., "score": ...}，做不出来的那项是 None。"""
    tol = settings["site"]["matrix_sum_tol"]
    return {"1x2": pick_1x2(row), "ah": pick_ah(row), "ou": pick_ou(row), "score": pick_score(row, tol)}


# ---------- 对比分 ----------
def settle(market: str, pick: dict | None, score) -> str | None:
    """预测和 90 分钟比分 (主, 客) 一比：win / half_win / push / half_loss / loss。没预测、没比分是 None。"""
    if pick is None or score is None:
        return None
    h, a = int(score[0]), int(score[1])
    if market == "1x2":
        side = 0 if h > a else (1 if h == a else 2)
        return "win" if pick["side"] == side else "loss"
    if market == "ah":
        return settle_asian_handicap(h, a, pick["line"], pick["side"])["outcome"]
    if market == "ou":
        return settle_over_under(h, a, pick["line"], pick["side"])["outcome"]
    if market == "score":
        return "win" if tuple(pick["score"]) == (h, a) else "loss"
    raise ValueError(f"不认识的玩法：{market}")


def hit_rate(outcomes: list[str]) -> float | None:
    """命中率 = 赢的分 ÷ 有输赢的分：(赢 + 赢半×0.5) ÷ (赢 + 赢半×0.5 + 输 + 输半×0.5)，走盘不算。没有可算的是 None。
    胜平负、比分只有中 / 没中，就是 中的场数 ÷ 场数。"""
    v = [VALUE[o] for o in outcomes]
    total = sum(abs(x) for x in v)
    return None if total == 0 else sum(max(x, 0) for x in v) / total


def tally(outcomes: list[str]) -> dict:
    """一个玩法的汇总：场数、每种结果几场、命中率。"""
    return {"n": len(outcomes), **{k: outcomes.count(k) for k in VALUE}, "rate": hit_rate(outcomes)}


# ---------- 网站上算命中率用哪些比赛 ----------
def before_kickoff(data, mid: int):
    """这场开球以前出的最新一版：(开球时间, kind, row)。开球时间读不出是 "开球时间不明"；没有开球以前的版本是 None。
    “最新”和 SiteData.latest 一样：cutoff_utc 最大，一样时 prediction_id 大的。正常时就是 latest（定时预测开球后不出）。"""
    from src.features.known_time import to_utc_text   # 放这里：只有这里用

    rows = list((data.preds.get(mid) or {}).values())
    kick = (data.matches.get(mid) or {}).get("kickoff_utc") or (rows[0].get("kickoff_utc") if rows else None)
    try:
        kick = to_utc_text(kick)
        ok = [r for r in rows if to_utc_text(r["cutoff_utc"]) < kick]
    except (TypeError, ValueError):
        return "开球时间不明"
    if not ok:
        return None
    row = max(ok, key=lambda r: (str(r["cutoff_utc"] or ""), r["prediction_id"]))
    return kick, row["kind"], row


def finished(data) -> tuple[list[dict], dict]:
    """算命中率的比赛：有 90 分钟比分、开球前出过预测的。每场用最新的那一版（和单场页一样）。

    返回 (比赛列表（开球时间从近到远），没算的场数和原因)。比赛列表每项：
    {"mid", "kickoff", "kind", "row", "score", "picks", "outcomes"}。
    开球以后才出的版本不用（取开球以前的最新一版，before_kickoff）；只有开球以后的版本、开球时间不明的，不算，单独记场数。
    """
    out, skipped = [], {"只有开球以后的版本": 0, "开球时间不明": 0}
    for mid, score in data.results.items():
        if data.latest(mid) is None:
            continue
        got = before_kickoff(data, mid)
        if got is None:
            skipped["只有开球以后的版本"] += 1
            continue
        if got == "开球时间不明":
            skipped["开球时间不明"] += 1
            continue
        kick, kind, row = got
        picks = match_picks(row, data.settings)
        out.append({"mid": mid, "kickoff": kick, "kind": kind, "row": row, "score": score, "picks": picks,
                    "outcomes": {k: settle(k, picks[k], score) for k in MARKETS}})
    out.sort(key=lambda x: (x["kickoff"], x["mid"]), reverse=True)
    return out, {k: v for k, v in skipped.items() if v}


def summarize(items: list[dict]) -> dict:
    """网站上的汇总：每个玩法 tally 一次（没预测的那场不算进这个玩法），另算“一直猜主胜”的对照。"""
    out = {k: tally([x["outcomes"][k] for x in items if x["outcomes"][k] is not None]) for k in MARKETS}
    out["always_home"] = tally(["win" if x["score"][0] > x["score"][1] else "loss" for x in items])
    out["n_matches"] = len(items)
    out["model_only"] = sum(1 for x in items if not used_odds(x["row"]))   # 这一版没用上赔率（旧版、没有赔率）
    return out


# ---------- 历史参考（src/evaluate/picks_reference.py 存的 summary.json） ----------
def load_reference(settings: dict) -> tuple[dict | None, str | None]:
    """读历史参考。返回 (内容, 读不了的原因)；还没算过是 (None, "还没算")。格式不对就不用，写原因。"""
    path = resolve_path(settings["picks"]["reference_dir"]) / "summary.json"
    if not path.exists():
        return None, "还没算历史参考"
    try:
        ref = json.loads(Path(path).read_text(encoding="utf-8"))
        seasons = ref.get("seasons")
        ok = (ref.get("rule") == RULE and isinstance(ref.get("markets"), dict)
              and all(_num(ref["markets"][k]["rate"]) is not None and int(ref["markets"][k]["n"]) > 0 for k in MARKETS)
              and _num(ref["always_home"]["rate"]) is not None
              and isinstance(seasons, list) and len(seasons) == 2 and all(type(x) is int for x in seasons)
              and type(ref.get("n_matches")) is int)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        ok = False
    if not ok:
        return None, "历史参考文件格式不对，没有用"
    return ref, None


# ---------- 页面上的写法 ----------
def pick_text(market: str, pick: dict | None, home: str, away: str) -> str:
    """一个预测的大白话（未转义；调用的地方负责转义）。没预测写“这一版没法预测”。"""
    if pick is None:
        return "这一版没法预测"
    if market == "1x2":
        return (f"主胜（{home}）", "平局", f"客胜（{away}）")[pick["side"]]
    if market == "ah":
        # 换成预测那一边自己的盘口：主队 -0.5 = 客队 +0.5
        line = pick["line"] if pick["side"] == "home" else -pick["line"]
        team = home if pick["side"] == "home" else away
        return f"{team} {'平手' if line == 0 else f'{line:+g}'}"
    if market == "ou":
        return f"{'大' if pick['side'] == 'over' else '小'} {pick['line']:g} 球"
    return f"{pick['score'][0]}–{pick['score'][1]}"


def outcome_html(outcome: str | None) -> str:
    if outcome is None:
        return "—"
    cls = "hit" if VALUE[outcome] > 0 else ("miss" if VALUE[outcome] < 0 else "muted")
    return f"<span class='{cls}'>{OUTCOME_TEXT[outcome]}</span>"


def prob_html(market: str, pick: dict | None) -> str:
    """预测那一边的可能性；存档里写着“没到倾向门槛”的，后面淡字提醒（胜平负写“和第二名差得不多”，让球、大小球写“接近五五开”）。"""
    if pick is None:
        return "—"
    note = ""
    if pick.get("lean") is False:
        note = " <span class='small muted'>和第二名差得不多</span>" if market == "1x2" else " <span class='small muted'>接近五五开</span>"
    return f"{pct0(pick['p'])}{note}"


def rate_text(t: dict) -> str:
    return "—" if t["rate"] is None else f"{t['rate']:.0%}"


def counts_text(t: dict) -> str:
    """每种结果几场：胜平负、比分写“中 x、没中 y”；让球、大小球再加赢半、走、输半（有才写）。"""
    parts = [f"中 {t['win']}"]
    parts += [f"{OUTCOME_TEXT[k]} {t[k]}" for k in ("half_win", "push", "half_loss") if t[k]]
    parts.append(f"没中 {t['loss']}")
    return "、".join(parts)
