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

注意：
- 不管有没有到“写倾向”的门槛，四个玩法都给预测（这样每场都能算命中率）；没到门槛的，页面上标“接近五五开”。
- “预测”只是“哪个更可能”，不是“值得买”：赔率已经把可能性算进去了。页面上不写任何下注建议。
- 比分预测和胜平负预测可能对不上（例如胜平负预测主胜，最可能的比分里却有 1–1、0–0）：
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
