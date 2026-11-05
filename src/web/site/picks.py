"""四个玩法的预测和命中率（规则 picks_v2，需求文档第 3 节“四个玩法的预测和命中率”）。

每场比赛用开球前最新的那一版存档（和单场页、战绩页同一版），四个玩法各给一个预测：
  胜平负：最后的胜平负概率里最大的那项（一样大时按 主胜 > 平 > 客胜）；
  让球（亚盘）：存档里写的亚盘倾向那一边（主盘口上胜算较大的一边）；接融合以前的旧存档没有倾向，就用“模型亚盘主队胜算”；
    新存档没写亚盘倾向（没有主盘口，或两边赔率不在 1.5–2.7）就不预测；
  大小球：固定看 2.5 球，大 2.5 球的概率 ≥ 50% 预测“大”，否则“小”（旧存档只在大小球主盘口正好是 2.5 时用模型的胜算）；
  比分：存下来的比分矩阵里最大的 4 格（先用融合后的，没有用模型自己的），实际比分是其中任意一个就算中。
这些都是从“只加不改”的预测存档算出来的，不另外存表；规则改了就另起一版，旧的照留。
另有“模型自己”一列（model_picks）：同样四条规则，只用模型自己的概率和比分矩阵，不看赔率，单独记命中率（需求文档 v0.88）。
picks_v1 和 picks_v2 只差比分：picks_v1 只给最大的 1 格（历史参考在 reference_dir/summary.json，照留），
picks_v2 给最大的 4 格（历史参考在 reference_dir/picks_v2/summary.json）。
文件后半段是 picks_v3（需求文档第 5 节“四个玩法的预测 picks_v3”，2026-11-05）：单场页、首页用它，
大小球改看主盘口、预览版按存的赔率概率重算比分矩阵，胜平负、大小球、比分加“把握”等级（SEL 的切点和历史命中率，
数字在 settings.yaml 的 picks_v3 块）。战绩页和 /api 原来的 picks 照旧用上面的 picks_v2。

注意：
- 不管有没有到“写倾向”的门槛，四个玩法都给预测（这样每场都能算命中率）；没到门槛的，页面上标“接近五五开”。
- “预测”只是“哪个更可能”，不是“值得买”：赔率已经把可能性算进去了。页面上不写任何下注建议。
- 比分预测和胜平负预测可能对不上（例如胜平负预测主胜，最可能的比分里却有 1–1、0–0）：
  平局的概率集中在 0–0、1–1 几个格子，主胜的概率分散在 1–0、2–0、2–1……很多格子，所以单个最大的格子常常是平局。
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common.config import resolve_path
from src.common.settlement import settle_asian_handicap, settle_over_under
from src.web.site.common import pct0
from src.web.site.data import matrix

RULE = "picks_v2"
SCORE_TOP = 4                      # 比分给最可能的几个（规则的一部分：picks_v1 是 1 个，picks_v2 是 4 个）
OU_LINE = 2.5                      # 大小球固定看这条线（picks_v1 定下的，存档里融合后只存了“大2.5球”）
MARKETS = ("1x2", "ah", "ou", "score")
MARKET_NAMES = {"1x2": "胜平负", "ah": "让球（亚盘）", "ou": "大小球 2.5", "score": f"比分 {SCORE_TOP} 个"}
# 单场页“四个玩法的预测”表从上到下的顺序（只管显示，2026-11-05 你定的：亚盘放最前面当主结论）。
# 算命中率、存参考、接口返回用的还是上面的 MARKETS（顺序不动，免得战绩页、历史参考、/api 跟着变）。
DISPLAY_ORDER = ("ah", "1x2", "ou", "score")
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


def best_scores(m: np.ndarray) -> dict:
    """矩阵里概率最大的 SCORE_TOP 格：{"scores": [(主, 客), ...]（从大到小）, "p": 这几格加起来的概率}。
    网站和历史参考（src/evaluate/picks_reference.py）都用这一个，保证同一套规则。
    一样大时按矩阵里的先后取（主队进球少的在前，再按客队进球少的）：用稳定排序，每次排出来都一样。"""
    m = np.asarray(m, dtype=float)
    order = np.argsort(-m, axis=None, kind="stable")[:SCORE_TOP]
    hs, as_ = np.unravel_index(order, m.shape)
    return {"scores": [(int(h), int(a)) for h, a in zip(hs, as_)], "p": float(m[hs, as_].sum())}


def pick_score(row: dict, tol: float) -> dict | None:
    """比分：存下来的比分矩阵里概率最大的 SCORE_TOP 格（先用融合后的矩阵，没有用模型自己的）。没存矩阵的版本不预测。"""
    d = row.get("detail") or {}
    for key, src in (("融合比分矩阵", "融合"), ("模型比分矩阵", "模型")):
        m = matrix(d, key, tol)
        if m is not None:
            return {**best_scores(m), "source": src if used_odds(row) else "模型", "lean": None}
    return None


def match_picks(row: dict, settings: dict) -> dict:
    """一版存档的四个预测：{"1x2": ..., "ah": ..., "ou": ..., "score": ...}，做不出来的那项是 None。"""
    tol = settings["site"]["matrix_sum_tol"]
    return {"1x2": pick_1x2(row), "ah": pick_ah(row), "ou": pick_ou(row), "score": pick_score(row, tol)}


# ---------- “模型自己”的四个预测（只看我们的模型，不看赔率；和上面那列对比用） ----------
def model_picks(row: dict, settings: dict) -> dict:
    """同一版存档里，只用模型自己的数（p_home/p_draw/p_away、模型比分矩阵）按同样的四条规则给的预测。

    上线的概率是赔率（fusion.w = 1，模型没比赔率准），所以上面那列其实是“赔率怎么看”；这一列是“模型怎么看”，
    单独记命中率，看模型自己到底差多少。规则：
      胜平负：模型三个概率里最大的；
      让球：和上面那列同一条线（存档倾向写的线；新存档没写倾向就不预测，接融合以前的旧存档用亚盘主盘口），模型比分矩阵在这条线上胜算较大的一边；
        没有矩阵时（旧存档）用存档里“模型亚盘主队胜算”（那是亚盘主盘口上的）；
      大小球：模型比分矩阵里大 2.5 球的概率；没有矩阵时只有大小球主盘口正好是 2.5 才用“模型大球胜算”；
      比分：模型比分矩阵里最大的 SCORE_TOP 格。
    """
    from src.common.markets import asian_handicap, over_under   # 放这里：只有这里用

    d = row.get("detail") or {}
    m = matrix(d, "模型比分矩阵", settings["site"]["matrix_sum_tol"])
    out = {k: None for k in MARKETS}
    p = [_num(row.get(k)) for k in ("p_home", "p_draw", "p_away")]
    if all(x is not None for x in p):
        side = int(np.argmax(p))
        out["1x2"] = {"side": side, "p": p[side], "source": "模型", "lean": None}
    lean = _lean(row, "ah")
    line = _num(lean.get("line")) if lean is not None else None
    if line is None and _old(row):
        line = _num(d.get("亚盘主盘口"))     # 只有接融合以前的旧存档才退回亚盘主盘口；新存档没写倾向 = 不预测
    share = None
    if m is not None and line is not None:
        share = _num(asian_handicap(m, line, "home")["win_share"])
    elif m is None and line is not None and line == _num(d.get("亚盘主盘口")) and _num(d.get("模型亚盘主队胜算")) is not None:
        share = float(d["模型亚盘主队胜算"])
    if share is not None:
        home = share >= 0.5
        out["ah"] = {"side": "home" if home else "away", "line": line, "p": share if home else 1 - share,
                     "source": "模型", "lean": None}
    over = None
    if m is not None:
        over = _num(over_under(m, OU_LINE, "over")["win_share"])
    elif _num(d.get("大小球主盘口")) == OU_LINE:
        over = _num(d.get("模型大球胜算"))
    if over is not None:
        side = "over" if over >= 0.5 else "under"
        out["ou"] = {"side": side, "line": OU_LINE, "p": over if side == "over" else 1 - over, "source": "模型", "lean": None}
    if m is not None:
        out["score"] = {**best_scores(m), "source": "模型", "lean": None}
    return out


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
        return "win" if (h, a) in {tuple(s) for s in pick["scores"]} else "loss"   # 实际比分是其中任意一个就算中
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
    {"mid", "kickoff", "kind", "row", "score", "picks", "outcomes", "model_picks", "model_outcomes"}（后两个是“模型自己”那列）。
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
        picks, mine = match_picks(row, data.settings), model_picks(row, data.settings)
        out.append({"mid": mid, "kickoff": kick, "kind": kind, "row": row, "score": score, "picks": picks,
                    "outcomes": {k: settle(k, picks[k], score) for k in MARKETS},
                    "model_picks": mine, "model_outcomes": {k: settle(k, mine[k], score) for k in MARKETS}})
    out.sort(key=lambda x: (x["kickoff"], x["mid"]), reverse=True)
    return out, {k: v for k, v in skipped.items() if v}


def summarize(items: list[dict]) -> dict:
    """网站上的汇总：每个玩法 tally 一次（没预测的那场不算进这个玩法），另算“一直猜主胜”的对照，
    和“模型自己”那列每个玩法的 tally（放在 out["model"]）。"""
    out = {k: tally([x["outcomes"][k] for x in items if x["outcomes"][k] is not None]) for k in MARKETS}
    out["model"] = {k: tally([x["model_outcomes"][k] for x in items if x["model_outcomes"][k] is not None]) for k in MARKETS}
    out["always_home"] = tally(["win" if x["score"][0] > x["score"][1] else "loss" for x in items])
    out["n_matches"] = len(items)
    out["model_only"] = sum(1 for x in items if not used_odds(x["row"]))   # 这一版没用上赔率（旧版、没有赔率）
    return out


# ---------- 历史参考（src/evaluate/picks_reference.py 存的 summary.json） ----------
def reference_path(settings: dict) -> Path:
    """这一版规则的历史参考文件：reference_dir/<规则名>/summary.json（picks_v1 的在 reference_dir/summary.json，照留不读）。"""
    return resolve_path(settings["picks"]["reference_dir"]) / RULE / "summary.json"


def load_reference(settings: dict) -> tuple[dict | None, str | None]:
    """读历史参考。返回 (内容, 读不了的原因)；还没算过是 (None, "还没算")。格式不对就不用，写原因。"""
    path = reference_path(settings)
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
    # “模型自己”那列的历史参考是后来加的：格式不对只是不显示这一项，不影响上面那列
    mm = ref.get("model_markets")
    try:
        good = isinstance(mm, dict) and all(_num(mm[k]["rate"]) is not None and int(mm[k]["n"]) > 0 for k in MARKETS)
    except (KeyError, TypeError, ValueError):
        good = False
    if not good:
        ref = {**ref, "model_markets": None}
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
    return "、".join(f"{h}–{a}" for h, a in pick["scores"])


def ah_sentence(pick: dict | None, home: str, away: str) -> str | None:
    """亚盘倾向的一句话（未转义；调用的地方负责转义）。首页每场那一句、单场页的大字结论都用它（2026-11-05 起亚盘放最前面）。

    pick 是 pick_ah 给的那个（主盘口上胜算较大的那一边）；没有合格的主盘口（pick 是 None）返回 None，
    调用的地方退回原来的胜平负写法。盘口的中文叫法用预览页的 ah_text（和“比分概率和各玩法”里一样），
    换成被看好那一队自己的盘口：主队 -0.5 是“主让半球”→ 主队“让半球”；客队那边是 +0.5 →“主受让半球”→ 客队“受让半球”。
    例：
      倾向利物浦让半球赢盘，胜算 53%
      倾向阿森纳受让平半赢盘，胜算 55%
      倾向利物浦赢盘（平手盘），胜算 54%（平手没有谁让谁）
      利物浦让半球赢盘的胜算 51%，接近五五开（存档写着没到倾向门槛：lean 是 False，门槛不在这里重算）
      利物浦让半球赢盘的胜算 51%（这一版没有存倾向）（接融合以前的旧存档：lean 是 None，不判断够不够倾向）
    “胜算”= 赢一半算一半、走盘不算时这一边赢的机会（和需求文档第 3 节“方向怎么定”一样）。只说哪边更可能，不是值得买。
    """
    from src.web.preview_page import ah_text   # 放这里：只有这里用，免得顶上多一层导入

    if pick is None:
        return None
    team = home if pick["side"] == "home" else away
    # 被看好那一队自己的盘口：line 永远是主队的，客队那边正负号反过来
    own = pick["line"] if pick["side"] == "home" else -pick["line"]
    name = ah_text(own)                        # 例：主让半球、主受让平半、平手
    level = name[1:] if name.startswith("主") else name   # 去掉“主”字，变成这一队自己“让半球”“受让平半”
    p = pct0(pick["p"])
    if level == "平手":
        what, back = f"平手盘{team}赢盘", f"倾向{team}赢盘（平手盘），胜算 {p}"
    else:
        what, back = f"{team}{level}赢盘", f"倾向{team}{level}赢盘，胜算 {p}"
    if pick.get("lean") is False:
        return f"{what}的胜算 {p}，接近五五开"
    if pick.get("lean") is None:
        return f"{what}的胜算 {p}（这一版没有存倾向）"
    return back


def outcome_html(outcome: str | None) -> str:
    if outcome is None:
        return "—"
    cls = "hit" if VALUE[outcome] > 0 else ("miss" if VALUE[outcome] < 0 else "muted")
    return f"<span class='{cls}'>{OUTCOME_TEXT[outcome]}</span>"


def prob_html(market: str, pick: dict | None) -> str:
    """预测那一边的可能性；存档里写着“没到倾向门槛”的，后面淡字提醒（胜平负写“和第二名差得不多”，让球、大小球写“接近五五开”）。
    比分写这几个比分加起来的可能性。"""
    if pick is None:
        return "—"
    note = f" <span class='small muted'>{SCORE_TOP} 个合起来</span>" if market == "score" else ""
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


# ======================= picks_v3：把握等级（需求文档第 5 节“四个玩法的预测 picks_v3”） =======================
# 上面 picks_v2 的函数一个都不改（战绩页、历史参考、/api 原有的 picks 照旧用它们）。下面是另加的：
# 单场页“四个玩法的预测”表、首页每场的“把握”一行和筛选、/api 新加的 picks_v3 和 latest.tiers 用。
# 数字（主盘口可用的线、切点、每档历史命中率）都在 settings.yaml 的 picks_v3 块，照抄 SEL 的结果，不重新调。
# 和 picks_v2 比：胜平负、让球的方向一样；大小球改看主盘口（不再固定 2.5）；预览版没存矩阵时按存的数重算赔率矩阵；
# 胜平负、大小球、比分加“把握”等级（高 / 中 / 低）。让球分不开，不分档，只写总的历史命中率。
RULE_V3 = "picks_v3"
TIERS = ("high", "mid", "low")                     # 档的代码：网页 data-tier-* 的值、/api 里的值
TIER_NAMES = {"high": "高", "mid": "中", "low": "低"}
TIERED = ("1x2", "ou", "score")                    # 分档的玩法（SEL：这三个分得开；让球分不开）
MARKET_NAMES_V3 = {**MARKET_NAMES, "ou": "大小球（主盘口）"}


def ou_lines(settings: dict) -> np.ndarray:
    """大小球主盘口可用的线（settings.yaml picks_v3.ou_lines：从 0.5 到 6，每隔 0.25）。"""
    lo, hi, step = (float(x) for x in settings["picks_v3"]["ou_lines"])
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def good_line(x, settings: dict) -> float | None:
    """存档里的大小球主盘口能不能用：是数字、正好是 ou_lines 里的一条（0.5–6 之间、0.25 的整数倍）。能用返回它，不能用 None。"""
    x = _num(x)
    if x is None:
        return None
    grid = ou_lines(settings)
    hit = grid[np.isclose(grid, x, rtol=0, atol=1e-9)]
    return float(hit[0]) if len(hit) else None


def _from_odds(row: dict) -> bool:
    """这一版的胜平负概率是不是赔率算的：融合状态是“已融合”或“赔率不全”，而且 w = 1（w = 1 时融合的就是赔率的数）。"""
    return used_odds(row) and _num(_fusion(row).get("w")) == 1.0


@lru_cache(maxsize=4096)
def _rebuild(p_1x2: tuple, line: float, p_over: float, n: int) -> np.ndarray:
    """按存的胜平负概率和大小球（线、大球胜算）重新反推市场比分矩阵：和出这一版时同一个方法（src/predict/conclusion.py 的 market_matrix）。
    同样的几个数算出来的矩阵一样，所以记下来（lru_cache）：首页、单场页、接口反复用同一版时不用重算。
    记下来的矩阵大家共用，设成只读，谁也改不了它。"""
    from src.predict.conclusion import market_matrix   # 放这里：只有这里用，免得网站一导入就带上预测那一套

    m, _err = market_matrix({"p_1x2": list(p_1x2), "ou": {"line": line, "p_over": p_over}}, n)
    m = np.array(m, dtype=float)
    m.setflags(write=False)
    return m


def market_matrix_v3(row: dict, settings: dict) -> tuple[np.ndarray | None, str | None]:
    """这一版赔率算的比分矩阵和它从哪来：(矩阵, "存档") / (矩阵, "重算") / (None, None)。

    1. 存档里的“融合比分矩阵”（定时版 T-24 / T-6 / T-1 有；w = 1 时它就是赔率反推的市场矩阵）；
    2. 没存矩阵（预览版只存了数）：融合状态是“已融合”、w = 1、存了融合用的大小球（线和大球胜算）时，
       用存的融合概率和这条大小球重新反推（_rebuild）。只在内存里算，不写回存档；
    3. 别的情况（没用上赔率、只有胜平负赔率、数不全）：None。
    """
    f = _fusion(row)
    if f.get("状态") != "已融合" or _num(f.get("w")) != 1.0:
        return None, None
    d = row.get("detail") or {}
    m = matrix(d, "融合比分矩阵", settings["site"]["matrix_sum_tol"])
    if m is not None:
        return m, "存档"
    fp, ou = d.get("融合概率"), f.get("大小球")
    if not (isinstance(fp, list) and len(fp) == 3 and all(_num(x) is not None and float(x) > 0 for x in fp)):
        return None, None
    if not isinstance(ou, dict) or _num(ou.get("line")) is None or _num(ou.get("p_over")) is None \
            or not 0 < float(ou["p_over"]) < 1:
        return None, None
    try:
        m = _rebuild(tuple(float(x) for x in fp), float(ou["line"]), float(ou["p_over"]),
                     int(settings["market_odds"]["max_goals"]))
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        return None, None
    return (m, "重算") if np.isfinite(m).all() else (None, None)


def _over_share(m: np.ndarray, line: float) -> float | None:
    """矩阵 m 在大小球 line 上“大”的胜算（赢半算一半、走盘不算，和战绩页、SEL 一样）。算不出来是 None。"""
    from src.common.markets import over_under   # 放这里：只有 v3 用

    return _num(over_under(m, float(line), "over")["win_share"])


def implied_line(m: np.ndarray, settings: dict) -> float | None:
    """推出来的主盘口：ou_lines 里大球胜算最接近 50% 的那条（一样近取小的），和 SEL 定切点时同一个算法
    （src/evaluate/pick_tiers_study.py 的 implied_lines；那边 import 了这个文件，这里不能反过来 import 它）。"""
    best = None
    for line in ou_lines(settings):
        s = _over_share(m, float(line))
        if s is None:
            continue
        gap = abs(s - 0.5)
        if best is None or gap < best[0]:          # 只有更近才换：一样近时留着前面的（小的线）
            best = (gap, float(line))
    return None if best is None else best[1]


def ou_main_line(row: dict, m: np.ndarray | None, settings: dict) -> tuple[float | None, str | None]:
    """大小球用哪条主盘口：(线, 从哪来)。
    “存档”= 这一版存的“大小球主盘口”（出这一版时懂球帝多家公司开得最多的那条，就是真主盘口），要合格（good_line）；
    “推算”= 存档没有或不合格时，用矩阵 m 推出来的（implied_line）；都没有是 (None, None)。不现查懂球帝（截止以后的价格对不上这一版）。"""
    line = good_line((row.get("detail") or {}).get("大小球主盘口"), settings)
    if line is not None:
        return line, "存档"
    if m is not None:
        line = implied_line(m, settings)
        if line is not None:
            return line, "推算"
    return None, None


def tier_of(market: str, p, settings: dict) -> str | None:
    """把握分 → 档：≥ 上切点 = high，≥ 下切点 = mid，其余 = low（切点在 settings.yaml picks_v3.tiers.<玩法>.cuts）。
    不分档的玩法（让球）、没有把握分的是 None。"""
    p = _num(p)
    if p is None or market not in TIERED:
        return None
    lo, hi = (float(x) for x in settings["picks_v3"]["tiers"][market]["cuts"])
    return "high" if p >= hi else ("mid" if p >= lo else "low")


def _ou_from(over: float | None, line: float, src: str) -> dict | None:
    if over is None:
        return None
    side = "over" if over >= 0.5 else "under"          # 正好 50% 算“大”（和 SEL 一样）
    return {"side": side, "line": line, "p": over if side == "over" else 1 - over, "source": src, "lean": None}


def match_picks_v3(row: dict, settings: dict) -> dict:
    """一版存档的 picks_v3 四个预测：{"1x2", "ah", "ou", "score"}，做不出来的是 None。
    每项在 picks_v2 那些键之外加 "tier"（high / mid / low；不分档、不是赔率算的是 None）；
    大小球再加 "line_from"（主盘口从哪来：存档 / 推算），大小球、比分加 "matrix"（矩阵从哪来：存档 / 重算 / 模型）。

    胜平负、让球：和 picks_v2 一样（pick_1x2、pick_ah）。胜平负的概率是赔率算的（_from_odds）才分档。
    大小球：主盘口（ou_main_line）上矩阵胜算较大的一边；矩阵先用赔率算的（market_matrix_v3），没有用模型比分矩阵（不分档）；
      连模型矩阵也没有时，只有用的是存档的主盘口，才用存档的“模型大球胜算”（它就是在这条线上算的；不分档）。
    比分：同一个矩阵里最大的 SCORE_TOP 格；赔率算的矩阵才分档。
    """
    d = row.get("detail") or {}
    out = {"1x2": pick_1x2(row), "ah": pick_ah(row), "ou": None, "score": None}
    if out["1x2"] is not None:
        out["1x2"]["tier"] = tier_of("1x2", out["1x2"]["p"], settings) if _from_odds(row) else None
    if out["ah"] is not None:
        out["ah"]["tier"] = None                        # 让球不分档
    mk, how = market_matrix_v3(row, settings)
    if mk is not None:
        m, src = mk, "融合"
    else:
        m, src, how = matrix(d, "模型比分矩阵", settings["site"]["matrix_sum_tol"]), "模型", "模型"
        if m is None:
            how = None
    line, line_from = ou_main_line(row, m, settings)
    if line is not None:
        if m is not None:
            ou = _ou_from(_over_share(m, line), line, src)
        elif line_from == "存档":
            ou, how = _ou_from(_num(d.get("模型大球胜算")), line, "模型"), "模型"
        else:
            ou = None
        if ou is not None:
            ou.update(tier=tier_of("ou", ou["p"], settings) if mk is not None else None, line_from=line_from, matrix=how)
            out["ou"] = ou
    if m is not None:
        top = best_scores(m)
        out["score"] = {**top, "source": src, "lean": None,
                        "tier": tier_of("score", top["p"], settings) if mk is not None else None, "matrix": how}
    return out


def model_picks_v3(row: dict, settings: dict, ou_line: float | None) -> dict:
    """“模型自己”一列（picks_v3）：胜平负、让球、比分和 model_picks 一样；大小球换成 ou_line（和预测那列同一条主盘口）：
    模型比分矩阵在这条线上算；没有矩阵时，只有 ou_line 就是存档的“大小球主盘口”才用存档的“模型大球胜算”。
    不分档。让球、大小球加 "overstated": True：模型在这两个玩法上写的胜算历史上偏高（SEL），页面上只写方向，
    数字还留在接口里（以后核对用）。"""
    d = row.get("detail") or {}
    out = {k: (None if v is None else dict(v)) for k, v in model_picks(row, settings).items()}
    m = matrix(d, "模型比分矩阵", settings["site"]["matrix_sum_tol"])
    over = None
    if ou_line is not None:
        if m is not None:
            over = _over_share(m, ou_line)
        elif good_line(d.get("大小球主盘口"), settings) == ou_line:
            over = _num(d.get("模型大球胜算"))
    out["ou"] = _ou_from(over, ou_line, "模型") if ou_line is not None else None
    for k in MARKETS:
        if out[k] is not None:
            out[k]["tier"] = None
            if k in ("ah", "ou"):
                out[k]["overstated"] = True
    return out


# ---------- picks_v3 页面上的写法 ----------
def rate1(x: float) -> str:
    """命中率写一位小数（大小球三档只差一两个百分点，取整就看不出来了）。"""
    return f"{float(x) * 100:.1f}%"


def test_seasons_text(settings: dict) -> str:
    """历史命中率是哪几季的：例“2020/21–2023/24 四个赛季”（赛季用开始年份存，2020 = 2020/21）。"""
    a, b = (int(x) for x in settings["picks_v3"]["test_seasons"])
    return f"{a}/{(a + 1) % 100:02d}–{b}/{(b + 1) % 100:02d} 共 {b - a + 1} 个赛季"


def tier_info(market: str, pick: dict | None, settings: dict) -> dict | None:
    """这一个预测所在那一档的历史数：{"tier", "name", "rate", "n", "mean_p"}；不分档、没预测是 None。"""
    t = pick.get("tier") if pick else None
    if t not in TIERS or market not in TIERED:
        return None
    h = settings["picks_v3"]["tiers"][market][t]
    return {"tier": t, "name": TIER_NAMES[t], "rate": float(h["rate"]), "n": int(h["n"]), "mean_p": float(h["mean_p"])}


def tier_html(market: str, pick: dict | None, settings: dict) -> str:
    """单场页表里“把握”那一格（转义好的 HTML；里面只有程序自己写的字和数字）。
    例：把握高  历史命中 69.4%（2,388 场）。让球写“不分档”和总的历史命中率；不是赔率算的写“不分档”。
    这一档平均写的可能性和历史命中率差超过 calib_tol（比分中档），写明以历史命中为准；大小球写三档差几个百分点。"""
    c = settings["picks_v3"]
    if pick is None:
        return "—"
    if market == "ah":
        return (f"不分档 <span class='small muted'>历史命中 {rate1(c['ah_all']['rate'])}（{int(c['ah_all']['n']):,} 场），"
                "这个玩法看不出哪场更有把握</span>")
    info = tier_info(market, pick, settings)
    if info is None:
        return "<span class='small muted'>不是赔率算的，不分档</span>"
    note = ""
    if abs(info["mean_p"] - info["rate"]) > float(c["calib_tol"]):
        how = "偏低" if info["mean_p"] < info["rate"] else "偏高"
        note = f"；这一档写出来的可能性历史上{how}（平均写 {rate1(info['mean_p'])}），以历史命中为准"
    if market == "ou":
        t = c["tiers"]["ou"]
        note += f"；三档只差 {(float(t['high']['rate']) - float(t['low']['rate'])) * 100:.0f} 个百分点"
    return (f"<span class='tier tier-{info['tier']}'>把握{info['name']}</span> "
            f"<span class='small muted'>历史命中 {rate1(info['rate'])}（{info['n']:,} 场）{note}</span>")


def tier_line(ps: dict, settings: dict, home: str, away: str) -> str | None:
    """首页每场那一格的“把握”一行（纯文字，调用的地方转义）。ps 是 match_picks_v3 的结果。三个分档的玩法都没有档时是 None。
    例：把握：胜平负 高（历史命中 69.4%）、大 2.75 球 中（51.0%）、比分 低（38.4%）、让球不分档"""
    parts = []
    for k in TIERED:
        info = tier_info(k, ps.get(k), settings)
        if info is None:
            continue
        label = pick_text("ou", ps[k], home, away) if k == "ou" else {"1x2": "胜平负", "score": "比分"}[k]
        num = f"历史命中 {rate1(info['rate'])}" if not parts else rate1(info["rate"])
        parts.append(f"{label} {info['name']}（{num}）")
    if not parts:
        return None
    return "把握：" + "、".join(parts) + "、让球不分档"


def tier_codes(ps: dict) -> dict:
    """{"1x2": "high"/"mid"/"low"/None, "ou": ..., "score": ...}：首页 data-tier-* 和 /api 的 latest.tiers 用。"""
    return {k: (ps.get(k) or {}).get("tier") for k in TIERED}
