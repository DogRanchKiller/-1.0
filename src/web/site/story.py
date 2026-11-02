"""单场页最上面“看得懂”的几块（需求文档第 3 节“结论链”“四块分析”、第 8 节）。

给不看技术细节的人：
- 一句话结论 + 一条三色概率条（绿 = 主胜，灰 = 平，蓝 = 客胜；只用样式画，不用图片和脚本）。
- “结论是怎么得出来的”：6 步，每一步都填这场比赛自己的数字（模型猜几球 → 比分 → 胜平负 → 看赔率 → 合并 → 倾向和信心）。
- 四块分析：实力对比、近期状态与赛程、伤停与首发、盘口与水位解读。
- 影响最大的 3 个因素：把模型解释里同一类的特征加在一起，用大白话写。

只用这一版存档里已经有的东西（比分矩阵、融合记录、模型解释）和开球前的赔率记录，不重新跑模型；
缺什么就写“这一版没有存……”，不拿别的版本或别的比赛来补。
"""
from __future__ import annotations

import math

import numpy as np

from src.common import markets
from src.web.explanation import validate_explanation
from src.web.preview_page import ah_text
from src.web.site.common import OUTCOMES, bar_html, e, level_name, pct0, pct1
from src.web.site.data import SiteData
from src.web.site.match import final_probs, fusion, odds_timeline, version_matrices

# 降级候选因素的中文名（src/predict 存的“降级候选因素”里的键）
FACTOR_NAMES = {"gap": "模型和赔率分歧大", "early": "赛季前几轮", "promoted": "有升班马", "no_pinnacle": "没有平博赔率",
                "draw_risk": "平局风险高", "odds_move": "赔率变动大"}

# 模型解释里的特征 → 大白话的“一类因素”。同一类的贡献加在一起再排大小。
GROUPS = [("dyn_home", "主场优势和联赛特点"), ("is_home", "主场优势和联赛特点"), ("league", "主场优势和联赛特点"),
          ("dyn_", "两队最近的进攻、防守评分"), ("elo", "Elo 实力评分"), ("pi_", "主客场实力评分"),
          ("table_", "本赛季积分榜"), ("form", "近几场的进球、失球、射正和积分"), ("ewm", "近几场的进球、失球、射正和积分"),
          ("rest_days", "休息天数和赛程"), ("after_euro", "休息天数和赛程"), ("promoted", "升班马"),
          ("inj_", "伤停"), ("top_value", "主力身价"), ("mkt_", "市场赔率")]


def _num(x) -> float | None:
    """存档里的数字：是有限的数就返回 float，否则 None（缺了、写错了都当没有）。"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _pp(x: float) -> str:
    """概率差 → “+4.6 个百分点”。"""
    v = round(x * 100, 1)
    return f"{0.0 if v == 0 else v:+.1f} 个百分点"


def bar_legend(data: SiteData, home: str, away: str) -> str:
    return (f"<p class='small legend'><span><i class='s0'></i>{e(data.zh(home))}赢</span><span><i class='s1'></i>平局</span>"
            f"<span><i class='s2'></i>{e(data.zh(away))}赢</span></p>")


# ---------- 一句话结论 ----------
def headline(data: SiteData, row: dict, home: str, away: str) -> str:
    """例：“佛罗伦萨赢的可能稍大（40%）”；差不多时写“两边差不多，没有明显倾向”。只描述可能性，不给建议。"""
    p, _ = final_probs(row)
    ln = fusion(row).get("倾向") or {}
    res = ln.get("result") if isinstance(ln, dict) else None
    names = [data.zh(home), "平局", data.zh(away)]
    if isinstance(res, dict) and res.get("lean") and res.get("side") in (0, 1, 2):
        side = res["side"]
        text = "平局的可能最大" if side == 1 else f"{names[side]}赢的可能更大"
        text += f"（{pct0(p[side])}）"
    else:
        text = "两边差不多，没有明显倾向"
    draw = ln.get("draw") if isinstance(ln, dict) else None
    if isinstance(draw, dict) and draw.get("risk"):
        text += f"；平局风险高（{pct0(p[1])}）"
    return text


# ---------- 结论是怎么得出来的 ----------
def chain_html(data: SiteData, row: dict, home: str, away: str) -> str:
    """6 步，每步一句话 + 这场的数字。"""
    cfg = data.settings
    site = cfg["site"]
    d = row.get("detail") or {}
    fus = fusion(row)
    hn, an = data.zh(home), data.zh(away)
    steps = []

    # 第 1～3 步：模型自己（只看模型比分矩阵；没存就说没存）
    mats = dict(version_matrices(row, site["matrix_sum_tol"])[0])
    model_m = mats.get("模型自己")
    if model_m is not None:
        g = np.arange(model_m.shape[0])
        xh, xa = float(model_m.sum(axis=1) @ g), float(model_m.sum(axis=0) @ g)
        steps.append(("模型先猜两队各进几球",
                      f"把两队的实力评分、近期进球失球、射正、休息天数等交给模型，它估计：{e(hn)}平均进 <b>{xh:.2f}</b> 球，"
                      f"{e(an)}平均进 <b>{xa:.2f}</b> 球（这叫“预期进球”）。"))
        tops = "、".join(f"{h}–{a}（{pct1(q)}）" for h, a, q in markets.top_scores(model_m, 3))
        steps.append(("由此算出每个比分的可能性",
                      f"按这两个数算出 0–0、1–0、1–1……每个比分的可能，最可能的 3 个是：{e(tops)}。"
                      "就算是最可能的比分，可能性也不高，足球很难猜准比分。"))
        mp = markets.result_probs(model_m)
        o25 = markets.over_under(model_m, 2.5)["win"]       # 2.5 球没有走盘，“赢”就是总进球 ≥ 3
        bt = markets.btts(model_m)["yes"]
        extra = []
        if _num(o25) is not None:
            extra.append(f"总进球 3 个及以上（大 2.5 球）{pct0(o25)}")
        if _num(bt) is not None:
            extra.append(f"双方都进球 {pct0(bt)}")
        steps.append(("把比分加起来，得到模型自己的胜平负",
                      f"所有“{e(hn)}进得多”的比分加起来就是{e(hn)}赢的可能，其他同理："
                      f"<b>{e(hn)}赢 {pct0(mp[0])} / 平 {pct0(mp[1])} / {e(an)}赢 {pct0(mp[2])}</b>"
                      + (f"；{'，'.join(e(x) for x in extra)}。" if extra else "。")))
    else:
        steps.append(("模型自己的计算", f"这一版没有存模型的比分矩阵，只有模型的胜平负：{e(_probs3(row_probs(row), hn, an))}。"))

    # 第 4 步：赔率
    mkt = d.get("赔率概率")
    books = _num(fus.get("胜平负几家")) or _num(d.get("赔率公司数"))
    if isinstance(mkt, list) and len(mkt) == 3 and all(_num(x) is not None for x in mkt):
        steps.append(("再看博彩公司的赔率",
                      f"{int(books) if books else '几'} 家公司的胜平负赔率，先去掉公司的抽水，再平均成概率："
                      f"<b>{e(_probs3(mkt, hn, an))}</b>。赔率里已经包含了伤停、首发传闻等很多模型不知道的信息。"))
    elif fus.get("状态") == "无赔率，未融合" or not fus:
        steps.append(("再看博彩公司的赔率", "这一版出的时候还没有任何赔率，只能用模型自己的概率。"))
    else:
        steps.append(("再看博彩公司的赔率", "这一版用到了赔率，但没有单独存赔率算出来的概率。"))

    # 第 5 步：合并
    p, src = final_probs(row)
    w = _num(fus.get("w"))
    info = (cfg.get("preview") or {}).get("model_info") or {}
    if src == "融合" and w is not None:
        why = ""
        if all(k in info for k in ("backtest_period", "backtest_n", "backtest_model", "backtest_pinnacle")):
            why = (f"为什么这么分：用 {e(info['backtest_period'])} 年 {int(info['backtest_n'])} 场比赛回测，模型的平均误差（log loss，"
                   f"越小越准）是 {float(info['backtest_model']):.3f}，平博收盘赔率是 {float(info['backtest_pinnacle']):.3f}，"
                   "赔率比模型准，所以按回测选出来的比例更信赔率。")
        tail = []
        for key, name in (("大2.5球", "大 2.5 球"), ("双方进球", "双方都进球")):
            if _num(fus.get(key)) is not None:
                tail.append(f"{name} {pct0(fus[key])}")
        steps.append(("两边合在一起，得到最后的概率",
                      f"合并时赔率占 {w * 100:g}%、模型占 {(1 - w) * 100:g}%。{why}"
                      f"结果：<b>{e(_probs3(p, hn, an))}</b>" + (f"；{'，'.join(tail)}。" if tail else "。")
                      + ("<br><span class='small'>赔率占 100% 时，最后的概率就是赔率的概率；模型那几步留着作对照，"
                         "看模型和市场哪里看法不一样。</span>" if w == 1 else "")))
    else:
        steps.append(("最后的概率", f"没有和赔率合并，直接用模型的：<b>{e(_probs3(p, hn, an))}</b>。"))

    # 第 6 步：倾向和信心
    steps.append(("按门槛写倾向，再定信心", lean_rule_text(cfg, row, p) + confidence_text(cfg, row)))
    items = "".join(f"<li><b class='step'>{e(t)}</b><div>{body}</div></li>" for t, body in steps)
    return f"<ol class='chain'>{items}</ol>"


def row_probs(row: dict) -> list[float]:
    return [float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])]


def _probs3(p, hn: str, an: str) -> str:
    return f"{hn}赢 {pct0(p[0])} / 平 {pct0(p[1])} / {an}赢 {pct0(p[2])}"


def lean_rule_text(cfg: dict, row: dict, p) -> str:
    """第 6 步前半：门槛是多少、这场差多少、所以写什么。门槛读 settings.yaml 的 lean。"""
    th = cfg.get("lean") or {}
    order = sorted(range(3), key=lambda i: -p[i])
    gap = p[order[0]] - p[order[1]]
    parts = []
    r = _num(th.get("result"))
    if r is not None:
        verdict = (f"够门槛，写“倾向 {OUTCOMES[order[0]]}”" if gap >= r - 1e-12 else "不够门槛，写“无明显倾向”")
        parts.append(f"胜平负：最高的（{OUTCOMES[order[0]]} {pct0(p[order[0]])}）比第二高的（{OUTCOMES[order[1]]} "
                     f"{pct0(p[order[1]])}）高 {gap * 100:.1f} 个百分点，门槛是 {r * 100:g} 个百分点，{verdict}。")
    dr = _num(th.get("draw"))
    if dr is not None:
        risk = p[1] >= dr - 1e-12
        parts.append(f"平局 {pct0(p[1])}，{'达到' if risk else '没到'} {dr * 100:g}% 这条线，"
                     f"{'标“平局风险高”' if risk else '不标平局风险'}。")
    o = _num(th.get("over25"))
    if o is not None:
        parts.append(f"大小球、双方进球、亚盘：偏向那边的概率比 50% 多 {o * 100:g} 个百分点以上才写倾向。")
    return e("".join(parts))


def confidence_text(cfg: dict, row: dict) -> str:
    """第 6 步后半：信心等级怎么来的（规则读 settings.yaml 的 confidence），这场碰上了哪些因素。"""
    fus = fusion(row)
    lv = fus.get("信心等级")
    conf = cfg.get("confidence") or {}
    used = [FACTOR_NAMES.get(k, k) for k in conf.get("downgrade_factors") or []]
    flags = fus.get("降级候选因素")
    hit = [FACTOR_NAMES.get(k, k) for k, x in flags.items() if x] if isinstance(flags, dict) else []
    rule = "信心从 ●●● 高开始。"
    if used:
        rule += f"碰上这些因素每个降一级：{'、'.join(used)}。"
    else:
        rule += ("“模型和赔率分歧大”“赛季前几轮”“有升班马”“没有平博赔率”这几个因素在开发赛季回测里都没有让预测明显变差，"
                 "所以现在一个都不降级；")
    rule += f"只有一家赔率都没有时降 {int(_num(conf.get('no_odds_downgrade')) or 0)} 级。"
    here = f"这场：{'碰上了 ' + '、'.join(hit) + '（只作提醒）' if hit else '没有碰上要提醒的因素'}，信心 {level_name(lv)}。"
    if lv == 3 and not used:
        here += "所以几乎每场都是 ●●● 高，这不等于“很有把握”，只说明还没找到能区分把握大小的信号。"
    return "<br>" + e(rule + here)


# ---------- 四块分析 ----------
def _terms(row: dict, atol: float) -> dict[tuple[str, str], float]:
    """模型解释里每个特征的输入值：{(主/客那一条, 特征名): 值}。解释没存或校验不过返回空的（页面写“没有”，不猜）。"""
    payload, _ = validate_explanation((row.get("detail") or {}).get("模型解释"), row, atol=atol)
    if payload is None:
        return {}
    return {(t["side"], t["feature"]): x for t in payload["terms"] if (x := _num(t.get("value"))) is not None}


def _li(items: list[str]) -> str:
    items = [x for x in items if x]
    return ("<ul class='points'>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>") if items else ""


def strength_block(v: dict, hn: str, an: str) -> str:
    """实力对比：Elo、主客场实力评分、动态进攻评分、本赛季积分榜。值都是“这一版截止时间”那一刻的。"""
    out = []
    eh, ea = v.get(("home", "elo_own")), v.get(("home", "elo_opp"))
    if eh is not None and ea is not None:
        rh, ra = round(eh), round(ea)        # 差值用显示出来的整数算，读者自己减也对得上
        gap = "两队一样" if rh == ra else f"{e(hn if rh > ra else an)}高 {abs(rh - ra)} 分"
        out.append(f"Elo 实力评分（0 ≈ 本联赛平均水平）：{e(hn)} {rh:+d}，{e(an)} {ra:+d}，{gap}。")
    egd = v.get(("home", "pi_egd"))
    if egd is not None:
        out.append(f"按主客场实力评分，{e(hn)}在主场预计净胜 {egd:+.2f} 球。")
    # dyn_loglam_for / against：动态评分模型对这场的 log（预期进球），已经算上对手防守和主场优势（src/models/dynamic.py）
    lf, la = v.get(("home", "dyn_loglam_for")), v.get(("home", "dyn_loglam_against"))
    if lf is not None and la is not None:
        out.append(f"动态评分（按两队最近的进攻、防守随时更新）估计这场：{e(hn)}约进 {math.exp(lf):.2f} 球，"
                   f"{e(an)}约进 {math.exp(la):.2f} 球（已经算上对手防守和主场）。")
    for side, name in (("home", hn), ("away", an)):
        n, ppg, gd = (v.get((side, k)) for k in ("table_played_own", "table_ppg_own", "table_gdpg_own"))
        if n is not None and ppg is not None and gd is not None and n > 0:
            out.append(f"{e(name)}本赛季 {n:.0f} 轮：场均 {ppg:.2f} 分，场均净胜球 {gd:+.2f}。")
    return _li(out)


def form_block(v: dict, hn: str, an: str) -> str:
    """近期状态与赛程：最近 5 场的场均（可以跨赛季）、休息天数、是否刚踢完欧战。"""
    out = []
    for side, name in (("home", hn), ("away", an)):
        pts, gf, ga, sf = (v.get((side, f"form5_{k}_own")) for k in ("pts", "gf", "ga", "sotf"))
        if None not in (pts, gf, ga):
            s = f"{e(name)}最近 5 场：场均 {pts:.1f} 分、进 {gf:.1f} 球、丢 {ga:.1f} 球"
            out.append(s + (f"、射正 {sf:.1f} 次。" if sf is not None else "。"))
    for side, name in (("home", hn), ("away", an)):
        rest = v.get((side, "rest_days_own"))
        if rest is not None:
            euro = "，刚踢完欧战" if v.get((side, "after_euro_own")) else ""
            out.append(f"{e(name)}距上一场休息 {rest:.0f} 天{euro}。")
    return _li(out)


def odds_block(data: SiteData, mid: int, row: dict, hn: str, an: str) -> str:
    """盘口与水位解读：亚盘、大小球主盘口，平均水位折成的胜算，和模型算的对比；胜平负赔率从初盘到最新怎么动。"""
    d, fus = row.get("detail") or {}, fusion(row)
    out = []
    ah = fus.get("亚盘")
    if isinstance(ah, dict) and None not in (_num(ah.get("line")), _num(ah.get("home")), _num(ah.get("away"))) \
            and ah["home"] > 1 and ah["away"] > 1:
        ph = (1 / ah["home"]) / (1 / ah["home"] + 1 / ah["away"])
        s = (f"亚盘主盘口 {e(ah_text(float(ah['line'])))}（{int(_num(ah.get('n_books')) or 0)} 家平均）："
             f"{e(hn)}赔率 {ah['home']:.2f}，{e(an)}赔率 {ah['away']:.2f}。赔率越低，公司认为那边越可能赢盘；"
             f"去掉抽水，{e(hn)}赢盘的胜算约 {pct0(ph)}")
        m = _num(d.get("模型亚盘主队胜算"))
        if m is not None and _num(d.get("亚盘主盘口")) == float(ah["line"]):
            s += f"，模型自己算的是 {pct0(m)}"
        out.append(s + "。")
    ou = fus.get("大小球")
    if isinstance(ou, dict) and _num(ou.get("line")) is not None and _num(ou.get("p_over")) is not None:
        out.append(f"大小球 {float(ou['line']):g} 球（{int(_num(ou.get('n_books')) or 0)} 家）：去掉抽水后，"
                   f"总进球超过 {float(ou['line']):g} 球的胜算约 {pct0(ou['p_over'])}。")
    odds = data.odds.get(mid)
    line = odds_timeline(odds) if odds is not None and not odds.empty else []
    if len(line) >= 2:
        (_, p0, _), (_, p1, n1) = line[0], line[-1]
        i = int(np.argmax(np.abs(p1 - p0)))
        move = "基本没动" if abs(p1[i] - p0[i]) < 0.01 else f"变得最多的是{OUTCOMES[i]}：{pct0(p0[i])} → {pct0(p1[i])}"
        out.append(f"胜平负赔率从初盘到最新（{n1} 家平均），{move}。赔率大幅变动常常说明有新消息（伤停、首发等）。")
    return _li(out)


def analysis_html(data: SiteData, mid: int, row: dict, home: str, away: str) -> str:
    """四块分析，两列卡片。哪一块没有数据就写清楚没有，不编。"""
    hn, an = data.zh(home), data.zh(away)
    v = _terms(row, (data.settings.get("explanation") or {}).get("atol", 1e-6))
    no_terms = "<p class='small muted'>这一版没有存模型输入（模型解释），没法列出。</p>"
    lineup = (row.get("detail") or {}).get("首发")
    blocks = [
        ("实力对比", strength_block(v, hn, an) or no_terms),
        ("近期状态与赛程", form_block(v, hn, an) or no_terms),
        ("伤停与首发", _li(["模型现在不看伤停和首发（还没有可靠的免费数据接进来）。"
                            + ("这一版出的时候没有首发信息。" if lineup == "未含首发" else ""),
                            "赔率会很快反映伤停和首发消息，合并后的概率里间接包含了这部分。"])),
        ("盘口与水位解读", odds_block(data, mid, row, hn, an) or "<p class='small muted'>还没有盘口数据。</p>"),
    ]
    return "<div class='grid2'>" + "".join(f"<div class='card pad'><h3>{e(t)}</h3>{b}</div>" for t, b in blocks) + "</div>"


# ---------- 影响最大的几个因素 ----------
def group_of(feature: str) -> str:
    for prefix, name in GROUPS:
        if feature.startswith(prefix):
            return name
    return "其他"


def factors_html(row: dict, hn: str, atol: float, n: int = 3) -> str:
    """把模型解释里所有条目按“一类因素”加起来（主队那条和客队那条都算），看谁把主队赢的可能推高、拉低最多。"""
    payload, why = validate_explanation((row.get("detail") or {}).get("模型解释"), row, atol=atol)
    if payload is None:
        return f"<p class='small muted'>{e(why)}，没法列出影响最大的因素。</p>"
    sums: dict[str, float] = {}
    for t in payload["terms"]:
        c = _num(t.get("home_win_contribution"))
        if c is not None:
            g = group_of(str(t.get("feature")))
            sums[g] = sums.get(g, 0.0) + c
    top = sorted(sums.items(), key=lambda kv: -abs(kv[1]))[:n]
    base, pred = payload["base"]["home_win"], payload["prediction"]["home_win"]
    items = [f"{e(name)}：让{e(hn)}赢的可能{'多' if c > 0 else '少'}了 {abs(c) * 100:.1f} 个百分点" for name, c in top]
    return (f"<p>模型对“一场普通比赛”先估{e(hn)}赢 {pct1(base)}，看完这场的数据后改成 {pct1(pred)}"
            f"（{e(_pp(pred - base))}）。影响最大的 {len(top)} 类：</p>{_li(items)}"
            "<p class='small muted'>这是模型算账时每类数据“推了多少”，说明模型怎么想，不代表因果。</p>")


# ---------- 合起来 ----------
def hero_html(data: SiteData, mid: int, row: dict) -> str:
    """结论卡片最上面：一句话 + 大号概率条 + 颜色说明。"""
    info = data.matches[mid]
    home, away = info["home_team"], info["away_team"]
    p, _ = final_probs(row)
    return (f"<p class='headline'>{e(headline(data, row, home, away))}</p>"
            f"{bar_html(p, big=True)}{bar_legend(data, home, away)}")


def story_html(data: SiteData, mid: int, row: dict) -> str:
    """结论卡片下面：结论链、四块分析、影响最大的因素。"""
    info = data.matches[mid]
    home, away = info["home_team"], info["away_team"]
    atol = (data.settings.get("explanation") or {}).get("atol", 1e-6)
    return "".join([
        "<h2>结论是怎么得出来的</h2>", f"<div class='card pad'>{chain_html(data, row, home, away)}</div>",
        "<h2>四块分析</h2>", analysis_html(data, mid, row, home, away),
        "<h2>影响最大的几个因素（模型自己）</h2>", f"<div class='card pad'>{factors_html(row, data.zh(home), atol)}</div>",
    ])
