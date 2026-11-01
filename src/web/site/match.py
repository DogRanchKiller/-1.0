"""单场页 match/<match_id>.html（需求文档第 3 节“输出”、第 8 节）。

一页从上到下：
- 简要（不用点开就看得到）：最新一版的结论（胜平负概率、信心等级、各玩法倾向）、最多 3 条要点。
- 详细（点标题展开，<details>，不用脚本）：各版本对比（T-24 / T-6 / T-1 / 预览版）、比分概率热力图和各玩法、
  赔率变动图、各家公司的初盘和最新价格、竞彩赔率、赛后复盘。
比分概率和各玩法（亚盘、大小球、双方进球）只从存档里存的比分矩阵算（src/common/markets.py），不重新跑模型；
这一版没存矩阵就写“这一版没有存比分矩阵”，不另外算。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.common import markets
from src.common.odds import devig
from src.features.usage import season_label
from src.web.preview_page import ah_text, lean_texts
from src.web.site.common import (NOT_ADVICE, OUTCOMES, STATUS_TAGS, TeamFiles, bj_text, details, e, heatmap_svg,
                                 kickoff_text, level_name, line_chart_svg, page, pct0, pct1, placeholder, table)
from src.web.site.data import SiteData, matrix

VERSION_HOURS = {"T-24": 24, "T-6": 6, "T-1": 1}
VERSION_NAMES = {"T-24": "开球前 24 小时", "T-6": "开球前 6 小时", "T-1": "开球前 1 小时", "preview": "预览版（手动出的）"}
LEAN_KEYS = ("result", "draw", "over25", "btts", "ah")
MARKET_NAMES = {"1x2": "胜平负", "ah": "亚盘", "ou": "大小球"}


# ---------- 读存档里的一行（几页共用） ----------
def version_name(kind: str) -> str:
    return VERSION_NAMES.get(kind, kind)


def fusion(row: dict) -> dict:
    f = (row.get("detail") or {}).get("融合")
    return f if isinstance(f, dict) else {}


def final_probs(row: dict) -> tuple[list[float], str]:
    """这一版最后给的胜平负概率：有“融合概率”用融合的，没有（没赔率、没融合）用模型自己的。返回 (概率, "融合"/"模型")。"""
    fp = (row.get("detail") or {}).get("融合概率")
    if isinstance(fp, list) and len(fp) == 3 and all(isinstance(x, (int, float)) for x in fp):
        return [float(x) for x in fp], "融合"
    return [float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])], "模型"


def leans(row: dict) -> list[str]:
    """各玩法的倾向，一项一句（和预览页同一个写法）。存档里缺了某一项就不写那一项。"""
    ln = fusion(row).get("倾向")
    if not isinstance(ln, dict):
        return []
    try:
        return lean_texts({"leans": {k: ln.get(k) for k in LEAN_KEYS}})
    except (KeyError, TypeError, IndexError):
        return []


def level(row: dict):
    return fusion(row).get("信心等级")


def uncertain(row: dict) -> list[str]:
    u = fusion(row).get("不确定因素")
    return [str(x) for x in u] if isinstance(u, list) else []


def side_of(score: tuple[int, int]) -> int:
    """比分 → 0 主胜 / 1 平 / 2 客胜。"""
    h, a = score
    return 0 if h > a else (1 if h == a else 2)


def hit_html(row: dict, score) -> str:
    """胜平负最大的那项有没有猜中（只是看个大概；更全的对账看战绩页的 log loss）。"""
    if score is None:
        return "—"
    p, _ = final_probs(row)
    ok = int(np.argmax(p)) == side_of(score)
    return f"<span class='{'hit' if ok else 'miss'}'>{'猜中' if ok else '没猜中'}</span>"


def probs_text(p) -> str:
    return " / ".join(f"{OUTCOMES[i]} {pct0(x)}" for i, x in enumerate(p))


def conclusion_html(row: dict, with_leans: bool = True) -> str:
    """一版的结论：胜平负概率、信心等级、倾向。首页和单场页共用（单场页上面另列要点，不重复写倾向）。"""
    p, src = final_probs(row)
    st = fusion(row).get("状态")
    bits = [f"<span class='probs'>{e(probs_text(p))}</span> <span class='tag'>{e(level_name(level(row)))}</span>"]
    if src == "模型":
        bits.append(" <span class='tag warn'>只有模型（没和赔率融合）</span>")
    elif st and st != "已融合":
        bits.append(f" <span class='tag warn'>{e(st)}</span>")
    lt = leans(row) if with_leans else []
    if lt:
        bits.append("<br><span class='small'>" + "；".join(e(t) for t in lt) + "</span>")
    return "".join(bits)


def score_text(data: SiteData, mid: int) -> str:
    s = data.results.get(mid)
    return f"{s[0]}–{s[1]}" if s else ""


def planned_time(info: dict, kind: str) -> datetime | None:
    """定时版本的计划出的时间（最早可能的开球时间往前推 24 / 6 / 1 小时）。"""
    if kind not in VERSION_HOURS or not info.get("kickoff_earliest_utc"):
        return None
    k = pd.Timestamp(info["kickoff_earliest_utc"]).tz_localize("UTC").to_pydatetime()
    return k - timedelta(hours=VERSION_HOURS[kind])


def utc_dt(text) -> datetime:
    return pd.Timestamp(text).tz_localize("UTC").to_pydatetime()


def num2(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.2f}"


def line_name(line) -> str:
    """亚盘盘口的写法：中文叫法 + 数字，例 主让半球（-0.5）。"""
    if line is None or (isinstance(line, float) and np.isnan(line)):
        return "—"
    return f"{ah_text(float(line))}（{'0' if float(line) == 0 else f'{float(line):+g}'}）"


# ---------- 各版本对比 ----------
def versions_html(data: SiteData, mid: int) -> str:
    """T-24 / T-6 / T-1 / 预览版一版一行：什么时候出的、模型和融合后的概率、信心等级、倾向、不确定因素、首发。"""
    info, rows = data.matches[mid], data.preds.get(mid) or {}
    score = data.results.get(mid)
    out = []
    for kind in data.settings["site"]["versions"]:
        r = rows.get(kind)
        if r is None:
            plan = planned_time(info, kind)
            why = ("还没到时间" if plan is not None and plan > data.now else "这一版没有出") if kind in VERSION_HOURS \
                else "没有出预览版"
            out.append([e(version_name(kind)), f"<span class='muted'>{e(why)}</span>", "", "", "", "", "", ""])
            continue
        d = r.get("detail") or {}
        late = d.get("晚了几分钟")
        late_txt = "" if late in (None, 0) else f"<br><span class='small muted'>比计划晚 {int(late)} 分钟</span>"
        p, src = final_probs(r)
        has_m = "融合比分矩阵" if d.get("融合比分矩阵") is not None else ("模型比分矩阵" if d.get("模型比分矩阵") is not None else None)
        extra = "；".join(x for x in [*uncertain(r), str(d.get("首发") or "")] if x) or "—"
        out.append([
            e(version_name(kind)),
            e(bj_text(r["cutoff_utc"])) + late_txt,
            e(probs_text([r["p_home"], r["p_draw"], r["p_away"]])),
            e(probs_text(p)) if src == "融合" else "<span class='muted'>没融合</span>",
            e(level_name(level(r))),
            "<br>".join(e(t) for t in leans(r)) or "—",
            e(extra) + (f"<br><span class='small muted'>存了{e(has_m)}</span>" if has_m else
                        "<br><span class='small muted'>没存比分矩阵</span>"),
            hit_html(r, score),
        ])
    note = ("<p class='small muted'>“出的时间”是北京时间，是真正算出这一版的时间。各版本只用当时已经知道的数据，"
            "所以越晚的版本赔率越新；“首发”还没有接进来时写“未含首发”。猜中 = 胜平负里概率最大的那项和赛果一样。</p>")
    return table(["版本", "出的时间", "模型自己", "和赔率融合后", "信心", "倾向", "不确定因素、首发", "胜平负猜中"], out) + note


# ---------- 比分概率、各玩法（只从存的矩阵算） ----------
def version_matrices(row: dict, tol: float) -> tuple[list[tuple[str, np.ndarray]], str | None]:
    """这一版存的矩阵：[(名字, 矩阵), ...]，融合的在前。一个都没有时第二项写原因。"""
    d = row.get("detail") or {}
    got, bad = [], []
    for key, name in (("融合比分矩阵", "融合后"), ("模型比分矩阵", "模型自己")):
        if d.get(key) is None:
            continue
        m = matrix(d, key, tol)
        if m is None:
            bad.append(key)
        else:
            got.append((name, m))
    if got:
        return got, None
    if bad:
        return [], f"这一版存的{'、'.join(bad)}格式不对（不是概率表或加起来不是 1），不显示"
    return [], "这一版没有存比分矩阵"


def markets_html(mats: list[tuple[str, np.ndarray]], fus: dict, cfg: dict) -> str:
    """各玩法的概率，全部从比分矩阵算：胜平负、大小球、亚盘、双方进球。每个矩阵一列，方便对比。"""
    names = [n for n, _ in mats]
    rows = []
    for i, o in enumerate(OUTCOMES):
        cells = []
        for _, m in mats:
            p = float(markets.result_probs(m)[i])
            cells.append(f"{pct1(p)}<br><span class='small muted'>公平赔率 {num2(1 / p) if p > 0 else '—'}</span>")
        rows.append([e(f"胜平负：{o}"), *cells])
    ou_main = (fus.get("大小球") or {}).get("line") if isinstance(fus.get("大小球"), dict) else None
    ou_lines = sorted({float(x) for x in cfg["ou_lines"]} | ({float(ou_main)} if ou_main is not None else set()))
    for line in ou_lines:
        cells = []
        for _, m in mats:
            r = markets.over_under(m, line, "over")
            cells.append(f"大 {pct1(r['win_share'])} / 小 {pct1(1 - r['win_share'])}"
                         + (f"<br><span class='small muted'>走盘 {pct1(r['push'])}</span>" if r["push"] > 1e-9 else ""))
        tag = "（赔率主盘口）" if ou_main is not None and float(ou_main) == line else ""
        rows.append([e(f"大小球 {line:g} 球{tag}"), *cells])
    ah = fus.get("亚盘") if isinstance(fus.get("亚盘"), dict) else {}
    ah_main = ah.get("line")
    base = float(ah_main) if ah_main is not None else 0.0
    ah_lines = sorted({round(base + float(x), 2) for x in [0.0, *cfg["ah_extra"]]})
    for line in ah_lines:
        cells = []
        for _, m in mats:
            r = markets.asian_handicap(m, line, "home")
            cells.append(f"主 {pct1(r['win_share'])} / 客 {pct1(1 - r['win_share'])}<br><span class='small muted'>"
                         f"主队这边：赢全 {pct0(r['win'])}、赢半 {pct0(r['half_win'])}、走盘 {pct0(r['push'])}、"
                         f"输半 {pct0(r['half_loss'])}、输全 {pct0(r['loss'])}</span>")
        tag = "（赔率主盘口）" if ah_main is not None and float(ah_main) == line else ""
        rows.append([e(f"亚盘 {line_name(line)}{tag}"), *cells])
    cells = []
    for _, m in mats:
        b = markets.btts(m)
        cells.append(f"是 {pct1(b['yes'])} / 否 {pct1(b['no'])}")
    rows.append([e("双方进球"), *cells])
    note = ("<p class='small muted'>大小球、亚盘的百分比是“胜算”：赢半、输半各算一半，走盘（退钱）不算（第 3 节“方向怎么定”）。"
            "公平赔率 = 1 ÷ 概率，是“没有抽水时应该开多少”，只是换个写法，不是买不买的意思。</p>")
    return table(["玩法", *names], rows) + note


def scores_html(data: SiteData, mid: int) -> str:
    """每一版一个可以展开的小节：热力图 + 最可能的几个比分 + 各玩法。最新一版默认展开。"""
    cfg = data.settings["site"]
    rows = data.preds.get(mid) or {}
    if not rows:
        return placeholder("这场还没有预测")
    latest = data.latest(mid)[0]
    order = sorted(rows, key=lambda k: (str(rows[k]["cutoff_utc"] or ""), rows[k]["prediction_id"]), reverse=True)
    parts = []
    for kind in order:
        r = rows[kind]
        mats, why = version_matrices(r, cfg["matrix_sum_tol"])
        if not mats:
            parts.append(details(f"{version_name(kind)}的比分概率", placeholder(why), open_=kind == latest))
            continue
        maps = []
        for name, m in mats:
            tops = "、".join(f"{h}–{a} {pct1(p)}" for h, a, p in markets.top_scores(m, cfg["top_scores"]))
            maps.append(f"<div class='card pad'><b>{e(name)}的比分概率</b>"
                        f"{heatmap_svg(m, cfg['heatmap_goals'], f'{version_name(kind)} {name}的比分概率')}"
                        f"<p class='small'>最可能的 {cfg['top_scores']} 个比分：{e(tops)}</p></div>")
        inner = f"<div class='grid2'>{''.join(maps)}</div>" + markets_html(mats, fusion(r), cfg)
        parts.append(details(f"{version_name(kind)}的比分概率和各玩法", inner, open_=kind == latest))
    note = (f"<p class='small muted'>热力图：行是主队进几球、列是客队进几球，格子里是百分比，越亮越可能；"
            f"{cfg['heatmap_goals'] + 1}+ 那一格是 {cfg['heatmap_goals'] + 1} 球及以上加在一起。</p>")
    return "".join(parts) + note


# ---------- 赔率变动 ----------
def odds_timeline(odds: pd.DataFrame) -> list[tuple[datetime, np.ndarray, int]]:
    """几家公司胜平负赔率的变动 → 每次有一家变动时，当时各家最新价格去水后的平均概率：[(时间, [主, 平, 客], 几家), ...]。"""
    o = odds[(odds["market"] == "1x2")].dropna(subset=["home", "draw", "away"])
    o = o[(o[["home", "draw", "away"]] > 1).all(axis=1)]
    if o.empty:
        return []
    probs = devig(o[["home", "draw", "away"]].to_numpy(dtype=float))
    current: dict[str, np.ndarray] = {}
    out = []
    for book, t, p in zip(o["bookmaker"], o["changed_at_utc"], probs):
        current[book] = p
        avg = np.mean(list(current.values()), axis=0)
        if out and out[-1][0] == utc_dt(t):
            out[-1] = (utc_dt(t), avg, len(current))     # 同一分钟有几家变动，只留这一分钟最后的样子
        else:
            out.append((utc_dt(t), avg, len(current)))
    return out


def at_time(line, t: datetime):
    """时间线上 t 那一刻的样子（t 之前最后一次变动）；那时还没有赔率返回 None。"""
    last = None
    for p in line:
        if p[0] > t:
            break
        last = p
    return last


def odds_html(data: SiteData, mid: int) -> str:
    """赔率变动图（几家胜平负去水后的平均概率）+ 初盘、T-24、T-6、T-1、最新几个时点的平均。"""
    odds = data.odds.get(mid)
    if odds is None or odds.empty:
        return placeholder("还没有赔率变动记录")
    line = odds_timeline(odds)
    if not line:
        return placeholder("还没有胜平负赔率")
    info = data.matches[mid]
    marks = []
    for kind in ("T-24", "T-6", "T-1"):
        t = planned_time(info, kind)
        if t is not None and line[0][0] <= t <= data.now:      # 只画已经过去、而且那时已经有赔率的时点
            marks.append((kind, t))
    series = [(o, [(t, float(p[i])) for t, p, _ in line]) for i, o in enumerate(OUTCOMES)]
    chart = line_chart_svg(series, marks, "胜平负赔率去水后的平均概率")
    points = [("初盘（最早的价格）", line[0][0])]
    points += [(f"{k}（{version_name(k)}）", t) for k, t in marks]
    points.append(("最新", line[-1][0]))
    rows = []
    for name, t in points:
        p = at_time(line, t)
        if p is not None:
            rows.append([e(name), e(bj_text(p[0].strftime("%Y-%m-%d %H:%M"))), e(probs_text(p[1])), str(p[2])])
    note = ("<p class='small muted'>每家公司的胜平负赔率先去掉抽水（按比例），再把当时各家的最新价格取平均；"
            "价格在下一次变动前不变，所以画成台阶。只用开球前的价格。虚线是开球前 24 / 6 / 1 小时。</p>")
    return (f"<div class='card pad'>{chart}</div>{note}"
            + table(["时点", "价格的时间", "平均概率", "几家"], rows))


def books_html(data: SiteData, mid: int) -> str:
    """每家公司每种玩法的初盘和最新价格、变了几次。"""
    odds = data.odds.get(mid)
    if odds is None or odds.empty:
        return placeholder("还没有赔率")

    def price(r) -> str:
        if r["market"] == "1x2":
            return f"{num2(r['home'])} / {num2(r['draw'])} / {num2(r['away'])}"
        if r["market"] == "ah":
            return f"{line_name(r['line'])} 主 {num2(r['home'])} / 客 {num2(r['away'])}"
        return f"{float(r['line']):g} 球 大 {num2(r['over'])} / 小 {num2(r['under'])}" if pd.notna(r["line"]) else "—"

    rows = []
    for (book, mk), g in odds.groupby(["bookmaker", "market"], sort=True):
        first, last = g.iloc[0], g.iloc[-1]
        rows.append([e(book), e(MARKET_NAMES.get(mk, mk)), e(price(first)), e(price(last)),
                     e(bj_text(last["changed_at_utc"])), str(len(g) - 1)])
    return table(["公司", "玩法", "初盘", "最新", "最新价格的时间", "变了几次"], rows) + \
        "<p class='small muted'>胜平负写“主胜 / 平 / 客胜”的赔率。价格来自懂球帝，只有开球前的。</p>"


def lottery_html(data: SiteData, mid: int) -> str:
    """竞彩、北单的赔率（只能显示，不进模型）。"""
    lo = data.lottery.get(mid)
    if lo is None or lo.empty:
        return placeholder("没有竞彩赔率（不是每场都开）")
    rows = []
    for (prod, hc), g in lo.groupby(["product", "handicap"], sort=True, dropna=False):
        r = g.iloc[-1]
        hc_txt = "—" if pd.isna(hc) else f"{float(hc):+g}"
        rows.append([e(prod), e(hc_txt), e(f"{num2(r['home'])} / {num2(r['draw'])} / {num2(r['away'])}"),
                     e(bj_text(r["fetched_at_utc"]))])
    return table(["玩法", "让球", "主胜 / 平 / 客胜", "抓到的时间"], rows)


# ---------- 赛后复盘 ----------
def json_html(x, depth: int = 0) -> str:
    """把复盘 JSON（格式由 K2 定）照原样排成网页：字典 → 两列，列表 → 一行一项，字典组成的列表 → 表格。"""
    if depth > 5:
        return e(str(x))
    if isinstance(x, dict):
        if not x:
            return "—"
        items = "".join(f"<dt>{e(k)}</dt><dd>{json_html(v, depth + 1)}</dd>" for k, v in x.items())
        return f"<dl class='kv'>{items}</dl>"
    if isinstance(x, list):
        if not x:
            return "—"
        if all(isinstance(i, dict) for i in x):
            keys = list(dict.fromkeys(k for i in x for k in i))
            body = "".join("<tr>" + "".join(f"<td>{json_html(i.get(k), depth + 1)}</td>" for k in keys) + "</tr>" for i in x)
            head = "".join(f"<th>{e(k)}</th>" for k in keys)
            return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        if all(isinstance(i, (int, float, str, bool)) or i is None for i in x):
            return e("、".join("—" if i is None else (f"{i:.4g}" if isinstance(i, float) else str(i)) for i in x))
        return "<ul class='points'>" + "".join(f"<li>{json_html(i, depth + 1)}</li>" for i in x) + "</ul>"
    if isinstance(x, float):
        return e(f"{x:.4g}")
    if isinstance(x, bool):
        return "是" if x else "否"
    return "—" if x is None else e(str(x))


def review_html(data: SiteData, mid: int) -> str:
    """赛后对账（scores.csv 里这场的几行）+ 单场复盘（matches/<match_id>.json）。都没有写“还没有复盘数据”。"""
    parts = []
    if data.scores is not None:
        s = data.scores[data.scores["match_id"] == mid]
        if not s.empty:
            rows = []
            for r in s.to_dict("records"):
                rows.append([e(version_name(str(r["kind"])) if pd.notna(r["kind"]) else "—"), e(r["source"]),
                             e(probs_text([r["p_home"], r["p_draw"], r["p_away"]])),
                             *(e(f"{r[c]:.3f}" if pd.notna(r[c]) else "—")
                               for c in ("logloss_1x2", "brier_1x2", "rps_1x2", "logloss_score", "logloss_over25", "logloss_btts"))])
            parts.append(table(["版本", "概率来自", "胜平负概率", "胜平负 log loss", "Brier", "RPS", "比分 log loss",
                                "大小 2.5 log loss", "双方进球 log loss"], rows))
            parts.append("<p class='small muted'>log loss、Brier、RPS 都是越小越准（第 7 节）。“收盘赔率”一行是拿开球那一刻的赔率当对照。</p>")
    rv = data.reviews.get(mid)
    if rv is not None:
        parts.append(f"<div class='card pad'>{json_html(rv)}</div>")
    return "".join(parts) or placeholder("还没有复盘数据")


# ---------- 整页 ----------
def key_points(row: dict) -> list[str]:
    """要点，最多 3 条：有倾向的玩法、平局风险、不确定因素。"""
    pts = [t for t in leans(row) if "无明显倾向" not in t]
    pts += [f"要注意：{u}" for u in uncertain(row)]
    return pts[:3] or ["各玩法都没有明显倾向"]


def render(data: SiteData, mid: int, teams: TeamFiles) -> str:
    """单场页的整页 HTML。页面在 match/ 文件夹里，回到网站根目录要 "../"。"""
    info = data.matches[mid]
    pre = "../"
    home, away = info["home_team"], info["away_team"]
    title = f"{data.zh(home)} 对 {data.zh(away)}"
    tag = STATUS_TAGS.get(info.get("status"))
    score = score_text(data, mid)
    state = (f"<span class='score'>赛果 {e(score)}</span>" if score
             else (f"<span class='tag warn'>{e(tag)}</span>" if tag else "<span class='tag'>还没踢</span>"))
    head = (f"<h1>{teams.link(home, data.zh(home), pre)} <span class='accent'>对</span> {teams.link(away, data.zh(away), pre)}</h1>"
            f"<p class='muted'>{e(data.comp(info['competition']))} {e(season_label(info['season']))} 赛季 · "
            f"{e(kickoff_text(info))}（北京时间） · {state}</p>")
    latest = data.latest(mid)
    if latest is None:
        summary = f"<div class='card pad'>{placeholder('这场还没有预测')}</div>"
    else:
        kind, row = latest
        pts = "".join(f"<li>{e(t)}</li>" for t in key_points(row))
        summary = (f"<h2>结论</h2><div class='card pad'><p class='small muted'>最新一版：{e(version_name(kind))}，"
                   f"{e(bj_text(row['cutoff_utc']))} 出的</p><p>{conclusion_html(row, with_leans=False)}</p>"
                   f"<ul class='points'>{pts}</ul>{f'<p>{hit_html(row, data.results.get(mid))}</p>' if score else ''}</div>")
    has_review = data.reviews.get(mid) is not None or (
        data.scores is not None and bool((data.scores["match_id"] == mid).any()))
    body = "".join([
        head, summary, f"<div class='note'>{NOT_ADVICE}</div>",
        details("各版本对比（开球前 24 / 6 / 1 小时、预览版）", versions_html(data, mid)),
        details("比分概率和各玩法（亚盘、大小球、双方进球）", scores_html(data, mid)),
        details("赔率变动", odds_html(data, mid)),
        details("各家公司的初盘和最新价格", books_html(data, mid)),
        details("竞彩赔率", lottery_html(data, mid)),
        details("赛后复盘", review_html(data, mid), open_=has_review),
    ])
    return page(f"{title} - 足球预测", body, pre, data.now, data.settings["site"]["refresh_minutes"])
