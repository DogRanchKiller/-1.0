"""预测页预览版：算未来几天比赛的模型概率和赔率概率，存进预测存档，生成网页（需求文档第 8 节“早期简易页”）。

做什么：
1. 站在“现在”这个时间点（实时模式），通过取数层拿到现在之前已经能知道的比分（模型要的话还有射门、射正和
   休息天数），拟合模型（settings.yaml 的 preview.model；2026-10-31 从 Elo 换成 dc，同一天又换成 gb_l7，仍是临时模型）。
2. 取未来 N 天开球的联赛比赛，模型给出每场的主胜 / 平 / 客胜概率。
3. 每场取懂球帝各家公司“现在”的赔率，去掉抽水后换成概率，几家公司取平均（叫“赔率概率”）。
   模型给比分矩阵时，再算模型在赔率的亚盘、大小球主盘口上的胜算，和赔率那边对比。
4. 模型和赔率的概率差得多的场次标出来（只是“两边看法不同”，不是下注建议）。
   再按回测定下的规则写“结论”（需求文档第 5 节“预测页怎么接融合、倾向、信心等级”，src/predict/conclusion.py）：
   融合概率（w = 1 时就是赔率反推的比分矩阵）、各玩法的倾向、信心等级、不确定因素。
5. 模型的预测和结论存进预测存档（predictions 表，只加不改不删）；和上次存的一样就不重复存。
6. 生成网页 data/reports/preview.html（只是本机一个文件，不连网，不开网站服务）。

运行：.venv\\Scripts\\python -m src.predict.preview
平时双击“查看预测预览.bat”：先把懂球帝新抓的赔率导入数据库，再运行这个程序，最后打开网页。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean.schema import create_tables
from src.common.config import load_settings, resolve_path
from src.common.db import connect
from src.common.markets import asian_handicap, over_under, result_probs
from src.common.odds import devig
from src.features import FeatureStore, open_readonly
from src.features.known_time import to_utc_text
from src.features.rest_days import RestDays
from src.models.registry import build_model, model_config
from src.predict.conclusion import conclude, season_flags

KIND = "preview"          # 预测存档里这一种预测的名字
OUTCOMES = ("主胜", "平", "客胜")


# ---------- 赔率 → 概率 ----------
def valid_1x2(rows: pd.DataFrame) -> pd.DataFrame:
    """能用的胜平负赔率：三个都有、都大于 1（不合理的那家不用）。"""
    o = rows[["home", "draw", "away"]].dropna()
    return o[(o > 1).all(axis=1)]


def devig_1x2(rows: pd.DataFrame) -> np.ndarray | None:
    """几家公司的胜平负赔率 → 每家去水后的概率，再取平均。没有可用的赔率返回 None。

    去水方法按 settings.yaml 的 calc.devig_method（默认按比例）。每家去水后三个概率加起来都是 1，
    取平均以后加起来还是 1。
    """
    o = valid_1x2(rows)
    if o.empty:
        return None
    return devig(o.to_numpy(dtype=float)).mean(axis=0)


def main_line(rows: pd.DataFrame, a: str, b: str) -> dict | None:
    """让球或大小球的“主盘口”和这条盘口上的平均赔率、去水后的两边概率。

    a、b 是两边赔率的列名（让球：home / away；大小球：over / under）。
    各家公司开的盘口不一定一样（例如多数开半球，个别开平半），所以：
      1. 主盘口 = 开这条盘口的公司最多的那条；
      2. 一样多时，选两边概率最接近一半一半的那条（盘口本来就是为了让两边差不多）；
      3. 只用开这条盘口的公司：赔率取平均（给人看），去水后的概率也取平均。
    注意：整数球（例如平手、2 球）和四分之一球（例如平半、2.75）会有“走盘”（退钱）或“赢一半”，
    这里的两边概率只是“不算走盘时，哪边赢的机会”，不是精确的赢盘概率。
    """
    o = rows[["line", a, b]].dropna()
    o = o[(o[a] > 1) & (o[b] > 1)]
    if o.empty:
        return None
    count = Counter(o["line"].round(2))
    best = max(count.values())
    candidates = []
    for line, n in count.items():
        if n != best:
            continue
        g = o[o["line"].round(2) == line]
        p = devig(g[[a, b]].to_numpy(dtype=float)).mean(axis=0)
        candidates.append((abs(p[0] - 0.5), line, g, p))
    _, line, g, p = min(candidates, key=lambda c: (c[0], c[1]))
    return {"line": float(line), a: float(g[a].mean()), b: float(g[b].mean()),
            f"p_{a}": float(p[0]), f"p_{b}": float(p[1]), "n_books": int(len(g))}


def market_summary(odds: pd.DataFrame) -> dict:
    """一场比赛各家公司“现在”的赔率（取数层 latest_odds 的结果）→ 页面上要的赔率概率。"""
    out = {"n_books": int(odds["bookmaker"].nunique()) if len(odds) else 0,
           "last_change_utc": odds["changed_at_utc"].max() if len(odds) else None}
    one = odds[odds["market"] == "1x2"]
    p = devig_1x2(one)
    out["p_1x2"] = None if p is None else [float(x) for x in p]
    out["n_books_1x2"] = int(one[["home", "draw", "away"]].dropna().shape[0])
    out["ah"] = main_line(odds[odds["market"] == "ah"], "home", "away")
    out["ou"] = main_line(odds[odds["market"] == "ou"], "over", "under")
    return out


def fusion_odds(odds: pd.DataFrame, settings: dict) -> dict:
    """一场比赛融合用的赔率（取数层 latest_odds、用途「融合」的结果）→ 去水后的概率（需求文档第 5 节“预测页怎么接融合……”）。

    和回测“融合用哪家赔率”同一条规则：
      胜平负：有平博就只用平博；没有就各家去水再平均。
      大小球：开 2.5 球这条线的各家去水再平均；没有一家开 2.5 时，主盘口 ≥ 2.75 就用主盘口（按“大球胜算”反推，rho 不影响），
              主盘口更低就当作没有（rho 会影响，反推不准）。
      亚盘：主盘口（开的公司最多的那条），平均赔率两边都在 fusion_study.ah_main_odds（1.5–2.7）里才用来写倾向。
    返回 {"p_1x2": [主胜, 平, 客胜] 或 None, "pinnacle": 胜平负用的是不是平博, "n_books_1x2": 几家,
          "ou": {"line", "p_over", "n_books"} 或 None, "ah": {"line", "home", "away", "n_books"} 或 None}。
    """
    pin = settings["market_odds"]["pinnacle_book"]
    one = odds[odds["market"] == "1x2"]
    pin_rows = valid_1x2(one[one["bookmaker"] == pin])
    use = pin_rows if len(pin_rows) else valid_1x2(one)
    p = devig(use.to_numpy(dtype=float)).mean(axis=0) if len(use) else None
    out = {"p_1x2": None if p is None else [float(x) for x in p], "pinnacle": bool(len(pin_rows)),
           "n_books_1x2": int(len(use)), "ou": None, "ah": None}
    ou = odds[odds["market"] == "ou"][["line", "over", "under"]].dropna()
    ou = ou[(ou["over"] > 1) & (ou["under"] > 1)]
    line = settings["market_odds"]["ou_line"]
    at = ou[ou["line"].round(2) == line]
    if len(at):
        out["ou"] = {"line": float(line), "p_over": float(devig(at[["over", "under"]].to_numpy(float)).mean(axis=0)[0]),
                     "n_books": int(len(at))}
    else:
        main = main_line(odds[odds["market"] == "ou"], "over", "under")
        if main and main["line"] > line:
            out["ou"] = {"line": main["line"], "p_over": main["p_over"], "n_books": main["n_books"]}
    ah = main_line(odds[odds["market"] == "ah"], "home", "away")
    lo, hi = settings["fusion_study"]["ah_main_odds"]
    if ah and lo <= ah["home"] <= hi and lo <= ah["away"] <= hi:
        out["ah"] = {"line": ah["line"], "home": ah["home"], "away": ah["away"], "n_books": ah["n_books"]}
    return out


def move_pair(now: pd.DataFrame, before: pd.DataFrame, settings: dict) -> tuple[list, list] | None:
    """看“赔率最近有没有大幅变动”用的一对胜平负概率：(N 小时前, 现在)。

    只比两个时刻都有胜平负赔率的那几家公司（免得“多了几家公司”或“平博刚开盘”被当成赔率变了）：
    两个时刻都有平博就只比平博，不然这几家每家去水再平均。一家都对不上返回 None（不算变动）。
    """
    a = now[now["market"] == "1x2"].set_index("bookmaker")
    b = before[before["market"] == "1x2"].set_index("bookmaker")
    a, b = valid_1x2(a), valid_1x2(b)
    common = sorted(set(a.index) & set(b.index))
    pin = settings["market_odds"]["pinnacle_book"]
    if pin in common:
        common = [pin]
    if not common:
        return None
    return ([float(x) for x in devig(b.loc[common].to_numpy(float)).mean(axis=0)],
            [float(x) for x in devig(a.loc[common].to_numpy(float)).mean(axis=0)])


def gap(model_p, market_p, threshold: float) -> dict | None:
    """模型和赔率概率差得最多的那个结果。差距没超过界限返回 None。

    返回 {"outcome": 0/1/2（主胜/平/客胜）, "diff": 模型减赔率（正数 = 模型更看好这个结果）}。
    """
    if model_p is None or market_p is None:
        return None
    d = np.asarray(model_p, dtype=float) - np.asarray(market_p, dtype=float)
    i = int(np.argmax(np.abs(d)))
    return {"outcome": i, "diff": float(d[i])} if abs(d[i]) >= threshold else None


# ---------- 算预测 ----------
def to_1x2(pred) -> list[float] | None:
    """模型给的结果 → [主胜, 平, 客胜]。模型可能给三个概率，也可能给比分矩阵（以后的泊松、DC），也可能不给（None）。"""
    if pred is None:
        return None
    a = np.asarray(pred, dtype=float)
    return [float(x) for x in (a if a.ndim == 1 else result_probs(a))]


def model_lines(pred, market: dict) -> dict:
    """模型在赔率那边的“主盘口”上怎么看：亚盘主队的胜算、大小球“大”的胜算。

    只有给比分矩阵的模型（dc 等）才算得出来；只给胜平负三个概率的模型（Elo）或没有盘口时是 None。
    胜算 = (赢 + 赢半×0.5) ÷ (赢 + 赢半×0.5 + 输 + 输半×0.5)，走盘不计（src/common/markets.py 的 win_share），
    这样和赔率那边“不算走盘时哪边赢的机会”是同一个意思，两个数可以直接对比。
    """
    out = {"ah_home": None, "ou_over": None}
    if pred is None or np.asarray(pred).ndim != 2:
        return out
    if market.get("ah"):
        out["ah_home"] = asian_handicap(pred, market["ah"]["line"], "home")["win_share"]
    if market.get("ou"):
        out["ou_over"] = over_under(pred, market["ou"]["line"], "over")["win_share"]
    # 全部都是走盘时没有胜算（nan），当作没算出来
    return {k: None if v is None or np.isnan(v) else float(v) for k, v in out.items()}


def text_or_none(x) -> str | None:
    """数据库里的空值读进来可能是 None，也可能是 NaN（不是数字），统一成 None。"""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def model_version(name: str, cfg: dict) -> str:
    """模型版本 = 名字 + 设置的指纹（设置里任何一个数改了，指纹就变）。例：elo-3f2a9c1b。"""
    text = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return f"{name}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"


def model_inputs(fs: FeatureStore, settings: dict, model, cutoff: str, end: str,
                 leagues: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """交给模型的两张表：截止时间之前的历史比赛（训练用）、[cutoff, end) 开球的比赛（要预测的）。

    和回测框架（src/evaluate/backtest.py 的 run_backtest）取数的方法一样，这样模型上线后看到的数和回测时一样：
      - 用途按模型自己写的 usage（现在都是“A版输入”）；
      - 数据种类按模型自己写的 data：例如 gb_l7 要 ('比分', '射门、xG')，就把射门、射正也取来；
      - 模型要休息天数（uses_rest_days）时，取截止时间之前已经开球的全部赛程（联赛 + 杯赛、欧战），
        给两张表都加上休息天数、欧战后四列（src/features/rest_days.py）。
    回测是一个截止时间一个截止时间往前走、一批一批取；这里只有“现在”这一个截止时间，一次取完。
    tests/test_backtest.py 的 test_preview_gives_model_same_inputs_as_backtest 检查同一个截止时间两边交给模型的数一样。
    """
    if getattr(model, "uses_pre_odds", False) or getattr(model, "injury", False):
        # 预测页只给 A 版取数：B 版（要赛前赔率）、伤停研究模型在这里拿不到它们要的数，
        # 硬跑的话这些特征全是空值，和训练时看到的不一样，出的预测不可信，所以直接停下
        raise ValueError(f"预测页不支持 {model.name}：它要赛前赔率或伤停，预测页只给 A 版模型取数")
    usage = getattr(model, "usage", "A版输入")
    data = tuple(getattr(model, "data", ("比分",)))
    hist = fs.historical_matches(cutoff, usage, competition=leagues, data=data)
    targets = fs.prediction_targets(cutoff, end, "A版输入", competition=leagues)
    if getattr(model, "uses_rest_days", False):
        sched = fs.schedule(cutoff, "休息天数和欧战后标签")
        rest = RestDays.from_settings(settings)
        rest.add(sched)
        hist, targets = rest.annotate(hist), rest.annotate(targets)
    return hist, targets


def build_preview(fs: FeatureStore, settings: dict, now: datetime, model=None) -> dict:
    """站在 now 这个时间点，算未来几天比赛的模型概率、赔率概率和结论。不写数据库。

    fs：取数层（实时模式）。model：不给就按 settings.yaml 的 preview.model 新造一个（测试里可以换成假模型）。
    每场多出两项：fusion_odds（融合用的赔率，fusion_odds 的结果）、conclusion（conclusion.conclude 的结果）。
    计算全部在 compute_predictions 里（定时预测 src/predict/scheduled.py 也用它，两边出的数一样）。
    """
    return compute_predictions(fs, settings, now, model)[0]


def compute_predictions(fs: FeatureStore, settings: dict, now: datetime, model=None) -> tuple[dict, dict]:
    """预览页和定时预测共用的计算（2026-11-01 从 build_preview 抽出来，内容没改）。

    返回 (result, preds)：
      result：build_preview 的结果（页面、存档用）；
      preds：{match_id: 模型原样给的预测}（比分矩阵 / 三个概率 / None）。定时预测要把整张比分矩阵存进存档，
             result 里只有胜平负，所以另外交出来；放在 result 外面，预览页的结果一点不变。
    """
    cfg = settings["preview"]
    cutoff = to_utc_text(now)
    end = to_utc_text(now + timedelta(days=cfg["days_ahead"]))
    leagues = settings["leagues"][cfg["leagues"]]
    name = cfg["model"]
    mcfg = model_config(name, settings)
    model = model or build_model(name, settings)

    # 1~2. 截止“现在”已经能知道结果的比赛、未来几天开球的比赛（开球时间已经过了的不算），整理成和回测一样的样子
    hist, targets = model_inputs(fs, settings, model, cutoff, end, list(leagues))
    # 拟合模型，给每场出概率
    model.fit(hist, cutoff)
    probs = model.predict(targets, cutoff) if len(targets) else []

    gap_th = settings["confidence_study"]["gap"]           # “差距大”和信心等级的“分歧大”同一个数
    before = to_utc_text(now - timedelta(hours=cfg["odds_move"]["hours"]))
    season = season_flags(hist, targets, settings)
    rows = []
    preds = {}
    for t, p in zip(targets.to_dict("records"), probs):
        mid = int(t["match_id"])
        preds[mid] = p
        mp = to_1x2(p)
        # 3. 各家公司“现在”的赔率。用途写“显示”：只给人看，不进模型
        market = market_summary(fs.latest_odds(mid, cutoff, "显示"))
        # 4. 融合用的赔率（用途「融合」）：现在的；和 N 小时前同几家公司比，看赔率有没有大幅变动
        odds_now = fs.latest_odds(mid, cutoff, "融合")
        fo = fusion_odds(odds_now, settings)
        move = move_pair(odds_now, fs.latest_odds(mid, before, "融合"), settings)
        rows.append({
            "match_id": int(t["match_id"]), "competition": t["competition"], "league": leagues[t["competition"]],
            "season": int(t["season"]), "home_team": t["home_team"], "away_team": t["away_team"],
            "kickoff_utc": text_or_none(t["kickoff_utc"]), "match_date": t["match_date"],
            "date_precision": text_or_none(t.get("date_precision")) or "day",
            "model_p": mp, "model_lines": model_lines(p, market),
            # “差距”和信心等级的“分歧大”一样，和融合用的胜平负比
            "market": market, "gap": gap(mp, fo["p_1x2"], gap_th),
            "fusion_odds": fo, "conclusion": conclude(p, mp, fo, move, season.get(mid, {}), settings)})
    result = {"cutoff_utc": cutoff, "end_utc": end, "rows": rows, "model": name, "model_cfg": mcfg,
              "model_version": model_version(name, mcfg), "hist_n": int(len(hist)),
              "hist_last_date": hist["match_date"].max() if len(hist) else None,
              # 其中有几场的比分是 football-data 还没更新、先用懂球帝的
              "hist_n_dongqiudi": int((hist["stats_source"] == "dongqiudi").sum()) if len(hist) else 0,
              "gap_threshold": gap_th, "days_ahead": cfg["days_ahead"], "odds_move": cfg["odds_move"],
              "fusion_w": float(settings["fusion"]["w"]), "lean": settings["lean"]}
    return result, preds


def add_display_info(con, result: dict) -> None:
    """加上只给人看的信息：中文队名、比赛状态（改期、取消）。

    这两样不进模型，所以直接从数据库读，不经过取数层。状态特别要注意：赛前不知道一场比赛会不会改期，
    取数层的 prediction_targets 故意不给状态（防止回测“碰巧”避开改期的比赛）；这里只是页面上提醒一下。
    """
    ids = {r["match_id"] for r in result["rows"]}
    teams = {r[k] for r in result["rows"] for k in ("home_team", "away_team")}
    zh = {}
    if teams:
        q = ",".join("?" * len(teams))
        zh = {t: n for t, n in con.execute(f"select team_id, name_zh from teams where team_id in ({q})", list(teams)) if n}
    status = {}
    if ids:
        q = ",".join("?" * len(ids))
        status = dict(con.execute(f"select match_id, status from matches where match_id in ({q})", list(ids)))
    for r in result["rows"]:
        r["home_zh"] = zh.get(r["home_team"], r["home_team"])   # 没有中文名的先显示英文
        r["away_zh"] = zh.get(r["away_team"], r["away_team"])
        r["status"] = status.get(r["match_id"])


def code_version() -> str | None:
    """现在的程序是 Git 里哪一版（提交编号的前 7 位）；有没提交的改动时后面加“+改动”。拿不到就是空。"""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return (head + ("+改动" if dirty else "")) or None


# ---------- 预测存档 ----------
def conclusion_key(c: dict, fused) -> list:
    """结论里要存档的部分（判断“和上次一样”用）：状态、融合概率、大 2.5 球、双方进球（4 位小数）、亚盘那条线、信心等级。

    只比融合概率不够：例如大小球赔率刚出来，从“赔率不全”变成“已融合”，胜平负几乎不变，但多了大小球、亚盘的倾向。
    """
    ah = (c.get("leans") or {}).get("ah")
    r4 = (lambda x: None if x is None else round(x, 4))
    return [c.get("status"), fused, r4(c.get("over25")), r4(c.get("btts")), ah["line"] if ah else None, c.get("level")]


def archive(con, result: dict, now: datetime, code: str | None = None) -> dict:
    """把模型的预测存进 predictions 表。只加不改不删。

    同一场比赛、同一个模型版本，模型概率（4 位小数）、开球时间、结论（conclusion_key）都和上次存的一样，就不再存一行
    （每次打开预览页都会运行一次，不去重的话表会白白变大）。模型没给概率的比赛不存。
    p_home / p_draw / p_away 列仍是模型自己的概率（回头和赛果对比“模型准不准”）；融合后的概率、倾向、信心等级
    记在 detail 的“融合概率”“融合”里。
    返回 {"新存": 几行, "没变化": 几行}。
    """
    created = to_utc_text(now)
    counts = {"新存": 0, "没变化": 0}
    with con:
        for r in result["rows"]:
            if r["model_p"] is None:
                continue
            p, fused, key = archive_key(r)
            last = con.execute("select p_home, p_draw, p_away, kickoff_utc, detail from predictions where match_id = ?"
                               " and kind = ? and model_version = ? order by prediction_id desc limit 1",
                               (r["match_id"], KIND, result["model_version"])).fetchone()
            # 模型概率、开球时间、结论都没变才算“和上次一样”（赔率变了结论就变，要存一行新的）
            if (last and [round(x, 4) for x in last[:3]] == p and last[3] == r["kickoff_utc"]
                    and json.loads(last[4] or "{}").get("结论指纹") == key):
                counts["没变化"] += 1
                continue
            insert_prediction(con, result, r, KIND, created, p, prediction_detail(result, r, code))
            counts["新存"] += 1
    return counts


# 下面三个函数 2026-11-01 从 archive 抽出来（内容没改），定时预测 src/predict/scheduled.py 也用，两边存的格式一样
def archive_key(r: dict) -> tuple[list, list | None, list]:
    """一场比赛要存档的三样：模型概率（4 位小数）、融合概率（4 位小数，没有是 None）、结论指纹（conclusion_key）。"""
    p = [round(x, 4) for x in r["model_p"]]
    c = r.get("conclusion") or {}
    fused = None if c.get("p_1x2") is None else [round(x, 4) for x in c["p_1x2"]]
    return p, fused, conclusion_key(c, fused)


def prediction_detail(result: dict, r: dict, code: str | None) -> dict:
    """存档 detail 列的内容（预览页的那些键）。定时预测在这之外再加自己的几个键。"""
    _p, fused, key = archive_key(r)
    c = r.get("conclusion") or {}
    m = r["market"]
    lines = r.get("model_lines") or {}
    fo = r.get("fusion_odds") or {}
    return {"模型设置": result["model_cfg"], "代码版本": code, "拟合用的比赛场数": result["hist_n"],
            "比分截至": result["hist_last_date"], "赔率概率": m["p_1x2"], "赔率公司数": m["n_books_1x2"],
            "赔率最后变动": m["last_change_utc"],
            # 模型在赔率主盘口上的胜算（只记下来，不参与判断“和上次一样”）
            "亚盘主盘口": m["ah"]["line"] if m["ah"] else None, "模型亚盘主队胜算": lines.get("ah_home"),
            "大小球主盘口": m["ou"]["line"] if m["ou"] else None, "模型大球胜算": lines.get("ou_over"),
            # 融合后的结论（需求文档第 5 节“预测页怎么接融合、倾向、信心等级”）
            "融合概率": fused, "结论指纹": key,
            "融合": {"状态": c.get("status"), "w": c.get("w"), "胜平负用平博": fo.get("pinnacle"),
                     "胜平负几家": fo.get("n_books_1x2"), "大小球": fo.get("ou"), "亚盘": fo.get("ah"),
                     "大2.5球": c.get("over25"), "双方进球": c.get("btts"), "市场矩阵还原误差": c.get("fit_error"),
                     "倾向": c.get("leans"), "信心等级": c.get("level"), "降级候选因素": c.get("flags"),
                     "不确定因素": c.get("uncertain"), "赔率变动": c.get("odds_move")}}


def insert_prediction(con, result: dict, r: dict, kind: str, created: str, p: list, detail: dict) -> None:
    """往 predictions 表加一行（只加不改）。截止时间用 result 的 cutoff_utc（真正算这一版的时间）。"""
    con.execute(
        "insert into predictions (match_id, competition, season, home_team, away_team, kickoff_utc, match_date,"
        " kind, model, model_version, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail)"
        " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (r["match_id"], r["competition"], r["season"], r["home_team"], r["away_team"], r["kickoff_utc"],
         r["match_date"], kind, result["model"], result["model_version"], result["cutoff_utc"], created,
         *p, json.dumps(detail, ensure_ascii=False)))


def main() -> None:
    from src.web.preview_page import render   # 网页的样子放在 src/web，这里只管算

    s = load_settings()
    now = datetime.now(timezone.utc)
    ro = open_readonly()
    try:
        result = build_preview(FeatureStore(ro, "实时", s), s, now)
        add_display_info(ro, result)
    finally:
        ro.close()
    con = connect()
    try:
        create_tables(con)          # 第一次运行时建 predictions 表
        counts = archive(con, result, now, code_version())
    finally:
        con.close()
    out = resolve_path(s["preview"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(result, s, now), encoding="utf-8")
    rows = result["rows"]
    print(f"未来 {result['days_ahead']} 天：{len(rows)} 场比赛；模型有概率 {sum(r['model_p'] is not None for r in rows)} 场；"
          f"有赔率 {sum(r['market']['p_1x2'] is not None for r in rows)} 场；差距大 {sum(r['gap'] is not None for r in rows)} 场")
    status = Counter(r["conclusion"]["status"] for r in rows if r["conclusion"]["status"])
    print("结论：" + ("，".join(f"{k} {v} 场" for k, v in status.items()) or "没有"))
    print(f"预测存档：新存 {counts['新存']} 行，和上次一样没重复存 {counts['没变化']} 行")
    print(f"网页：{out}")


if __name__ == "__main__":
    main()
