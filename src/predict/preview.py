"""预测页预览版：算未来几天比赛的模型概率和赔率概率，存进预测存档，生成网页（需求文档第 8 节“早期简易页”）。

做什么：
1. 站在“现在”这个时间点（实时模式），通过取数层拿到现在之前已经能知道的比分，拟合模型（现在是 Elo，临时模型）。
2. 取未来 N 天开球的联赛比赛，模型给出每场的主胜 / 平 / 客胜概率。
3. 每场取懂球帝各家公司“现在”的赔率，去掉抽水后换成概率，几家公司取平均（叫“赔率概率”）。
4. 模型和赔率的概率差得多的场次标出来（只是“两边看法不同”，不是下注建议）。
5. 模型的预测存进预测存档（predictions 表，只加不改不删）；和上次存的一样就不重复存。
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
from src.common.markets import result_probs
from src.common.odds import devig
from src.features import FeatureStore, open_readonly
from src.features.known_time import to_utc_text
from src.models.registry import build_model, model_config

KIND = "preview"          # 预测存档里这一种预测的名字
OUTCOMES = ("主胜", "平", "客胜")


# ---------- 赔率 → 概率 ----------
def devig_1x2(rows: pd.DataFrame) -> np.ndarray | None:
    """几家公司的胜平负赔率 → 每家去水后的概率，再取平均。没有可用的赔率返回 None。

    去水方法按 settings.yaml 的 calc.devig_method（默认按比例）。每家去水后三个概率加起来都是 1，
    取平均以后加起来还是 1。
    """
    o = rows[["home", "draw", "away"]].dropna()
    o = o[(o > 1).all(axis=1)]          # 赔率必须大于 1，不合理的那家不用
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


def text_or_none(x) -> str | None:
    """数据库里的空值读进来可能是 None，也可能是 NaN（不是数字），统一成 None。"""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def model_version(name: str, cfg: dict) -> str:
    """模型版本 = 名字 + 设置的指纹（设置里任何一个数改了，指纹就变）。例：elo-3f2a9c1b。"""
    text = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return f"{name}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"


def build_preview(fs: FeatureStore, settings: dict, now: datetime, model=None) -> dict:
    """站在 now 这个时间点，算未来几天比赛的模型概率和赔率概率。不写数据库。

    fs：取数层（实时模式）。model：不给就按 settings.yaml 的 preview.model 新造一个（测试里可以换成假模型）。
    """
    cfg = settings["preview"]
    cutoff = to_utc_text(now)
    end = to_utc_text(now + timedelta(days=cfg["days_ahead"]))
    leagues = settings["leagues"][cfg["leagues"]]
    name = cfg["model"]
    mcfg = model_config(name, settings)
    model = model or build_model(name, settings)

    # 1. 截止“现在”已经能知道比分的比赛 → 拟合模型
    hist = fs.historical_matches(cutoff, "A版输入", competition=list(leagues))
    model.fit(hist, cutoff)
    # 2. 未来几天开球的比赛（开球时间已经过了的不算）
    targets = fs.prediction_targets(cutoff, end, "A版输入", competition=list(leagues))
    probs = model.predict(targets, cutoff) if len(targets) else []

    rows = []
    for t, p in zip(targets.to_dict("records"), probs):
        mp = to_1x2(p)
        # 3. 各家公司“现在”的赔率。用途写“显示”：只给人看，不进模型
        market = market_summary(fs.latest_odds(int(t["match_id"]), cutoff, "显示"))
        rows.append({
            "match_id": int(t["match_id"]), "competition": t["competition"], "league": leagues[t["competition"]],
            "season": int(t["season"]), "home_team": t["home_team"], "away_team": t["away_team"],
            "kickoff_utc": text_or_none(t["kickoff_utc"]), "match_date": t["match_date"],
            "date_precision": text_or_none(t.get("date_precision")) or "day",
            "model_p": mp, "market": market, "gap": gap(mp, market["p_1x2"], cfg["gap_threshold"])})
    return {"cutoff_utc": cutoff, "end_utc": end, "rows": rows, "model": name, "model_cfg": mcfg,
            "model_version": model_version(name, mcfg), "hist_n": int(len(hist)),
            "hist_last_date": hist["match_date"].max() if len(hist) else None,
            "gap_threshold": cfg["gap_threshold"], "days_ahead": cfg["days_ahead"]}


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
def archive(con, result: dict, now: datetime, code: str | None = None) -> dict:
    """把模型的预测存进 predictions 表。只加不改不删。

    同一场比赛、同一个模型版本，概率（保留 4 位小数）和开球时间都和上次存的一样，就不再存一行
    （每次打开预览页都会运行一次，不去重的话表会白白变大）。模型没给概率的比赛不存。
    返回 {"新存": 几行, "没变化": 几行}。
    """
    created = to_utc_text(now)
    counts = {"新存": 0, "没变化": 0}
    with con:
        for r in result["rows"]:
            if r["model_p"] is None:
                continue
            p = [round(x, 4) for x in r["model_p"]]
            last = con.execute("select p_home, p_draw, p_away, kickoff_utc from predictions where match_id = ? and kind = ?"
                               " and model_version = ? order by prediction_id desc limit 1",
                               (r["match_id"], KIND, result["model_version"])).fetchone()
            if last and [round(x, 4) for x in last[:3]] == p and last[3] == r["kickoff_utc"]:
                counts["没变化"] += 1
                continue
            m = r["market"]
            detail = {"模型设置": result["model_cfg"], "代码版本": code, "拟合用的比赛场数": result["hist_n"],
                      "比分截至": result["hist_last_date"], "赔率概率": m["p_1x2"], "赔率公司数": m["n_books_1x2"],
                      "赔率最后变动": m["last_change_utc"]}
            con.execute(
                "insert into predictions (match_id, competition, season, home_team, away_team, kickoff_utc, match_date,"
                " kind, model, model_version, cutoff_utc, created_at_utc, p_home, p_draw, p_away, detail)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r["match_id"], r["competition"], r["season"], r["home_team"], r["away_team"], r["kickoff_utc"],
                 r["match_date"], KIND, result["model"], result["model_version"], result["cutoff_utc"], created,
                 *p, json.dumps(detail, ensure_ascii=False)))
            counts["新存"] += 1
    return counts


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
    print(f"预测存档：新存 {counts['新存']} 行，和上次一样没重复存 {counts['没变化']} 行")
    print(f"网页：{out}")


if __name__ == "__main__":
    main()
