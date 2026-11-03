"""虚拟投注页 bots.html：虚拟投注机器人历史回测（bots_v1）的结果（需求文档第 8 节“虚拟投注页”、第 14 节）。

这一页只做一件事：把回测已经算好的结果文件读进来，排成好懂的网页。不重算、不下注、不连任何账户。
读的文件都在 settings.yaml 的 site.bots_dir（现在是 data/backtest/bots/bots_v1/，由 src/evaluate/bot_backtest.py 生成）：
  summary.csv    每个机器人一行：排名、下了几注、ROI 和区间、CLV 和区间……（排行榜）
  by_season.csv  每个机器人、每个赛季一行（画“按赛季累计盈亏”的小图、算“几个赛季赚”）
  by_period.csv  每个机器人分两段（2012 之前 / 2012 起）各一行
  bankroll.csv   资金模拟：从 10000 开始按风险规则下，最后剩多少、停没停
  check.json     场数、赛季检查、随机下注的理论值（算“抽水”那一段）
很大的每注明细 bets.csv 不读（网页用不到）。

文件不在、或者 summary.csv 格式不对：这一页写“还没有虚拟投注的结果”，网站其他页照常生成。
其余几个文件缺了或格式不对，只是对应的那一块不显示，写明原因。

几个词（页面上也写了）：
- ROI（投资回报率）= 盈亏 ÷ 总注额。-2% 就是每下 100 平均亏 2。
- 95% 区间：把历史按周打乱重抽很多次，ROI 大多落在这个范围里。区间跨过 0，说明赚还是亏看不出来，可能只是运气。
- CLV（收盘线价值）= 下注赔率 × 平博收盘去掉抽水后这个结果的可能性 − 1（src/bots/betting.py clv），
  也就是拿下注赔率和平博收盘去掉抽水后的“公平价格”比。正好按平博收盘价下注，CLV 约等于负的抽水（约 −2% 到 −3%），
  所以要和“对照_随机_平博”（每场随便买平博的价格）的 CLV 对比着看。它受赛果运气影响小。

页面上任何一块出了意外的错：整页换成“虚拟投注结果格式不对”的占位页，不让整个网站生成失败（render）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import resolve_path
from src.web.site.common import details, e, page, placeholder, table

NO_RESULT = "还没有虚拟投注的结果"

# 每个文件至少要有这些列（回测程序 src/bots/stats.py 写出来的列名），少了就当格式不对
SUMMARY_COLS = ("名次", "机器人", "对照组", "价格", "规则", "概率来源", "注额", "投注场数", "命中率", "平均赔率",
                "盈亏", "ROI", "ROI下限", "ROI上限", "有CLV的注数", "CLV平均", "CLV下限", "CLV上限", "最大回撤", "最长连输")
SUMMARY_NUM = ("投注场数", "命中率", "平均赔率", "盈亏", "ROI", "ROI下限", "ROI上限", "有CLV的注数",
               "CLV平均", "CLV下限", "CLV上限", "最大回撤", "最长连输")
# “名次”单独处理：有 CLV 的注不够 bots.min_clv_bets 的机器人，回测程序在这一格写字“注数不够，不排”（src/bots/stats.py TOO_FEW），
# 所以名次里不是数字的一律当“没排名”（空值），不当格式不对；别的数字列有字还是当格式不对
SUMMARY_RANK = ("名次",)
SEASON_COLS = ("机器人", "赛季", "投注场数", "盈亏", "ROI")
PERIOD_COLS = ("机器人", "两段", "投注场数", "ROI", "ROI下限", "ROI上限")
BANK_COLS = ("机器人", "初始资金", "最后资金", "最高点", "最大回撤", "停了没有", "停在哪天", "实际下了几注")

# 价格那一列的大白话（summary.csv 的“价格”是 settings.yaml bots.price_books 的名字）
BOOK_TEXT = {"平博": "平博赛前价格", "最高": "多家最高（事后挑的）"}


def _csv(path: Path, cols: tuple[str, ...], num: tuple[str, ...] = (), coerce: tuple[str, ...] = (),
         required: tuple[str, ...] = ("机器人",)) -> pd.DataFrame:
    """读一个结果 csv（回测写的是带 BOM 的 UTF-8）。下面几种都当格式不对（抛 ValueError）：
    少列；num 里的数字列有不是数字的字（空格子可以，变成 NaN，页面上写“—”）；required 里的列有空格子（例如机器人名字、赛季）。
    coerce 里的列：不是数字的一律变成 NaN，不报错（只用于“名次”）。"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"机器人": str})
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} 少了列：{'、'.join(missing)}")
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for c in coerce:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    empty = [c for c in required if df[c].isna().any()]
    if empty:
        raise ValueError(f"{path.name} 这几列有空格子：{'、'.join(empty)}")
    return df


def load(settings: dict) -> tuple[dict | None, str]:
    """读回测结果。返回 (内容, 说明)：读不了时内容是 None，说明写原因（页面上就写这个）。

    内容是 {"dir": 文件夹, "summary": 表, "season": 表或 None, "period": 表或 None, "bank": 表或 None,
           "check": dict 或 None, "problems": [哪几个文件没用上、为什么]}。
    """
    rel = (settings.get("site") or {}).get("bots_dir")
    if not rel:
        return None, NO_RESULT
    folder = resolve_path(rel)
    if not (folder / "summary.csv").exists():
        return None, NO_RESULT
    try:
        summary = _csv(folder / "summary.csv", SUMMARY_COLS, SUMMARY_NUM, coerce=SUMMARY_RANK)
    except OSError as err:                        # 读不了：只写错误种类，不写说明（说明里常带电脑上的路径，不能上网页）
        return None, f"{NO_RESULT}（结果文件 summary.csv 读不了：{type(err).__name__}）"
    except ValueError as err:                     # 编码不对、列不对、数字不对（pandas 的解析错误也是 ValueError）：写原因
        return None, f"{NO_RESULT}（结果文件 summary.csv 格式不对，没有用：{err}）"
    if summary.empty:
        return None, NO_RESULT
    out = {"dir": folder, "summary": summary, "season": None, "period": None, "bank": None, "check": None, "problems": []}
    # (放在哪、文件名、要有的列、数字列、不能空的列)
    parts = (("season", "by_season.csv", SEASON_COLS, ("赛季", "投注场数", "盈亏", "ROI"), ("机器人", "赛季")),
             ("period", "by_period.csv", PERIOD_COLS, ("投注场数", "ROI", "ROI下限", "ROI上限"), ("机器人", "两段")),
             ("bank", "bankroll.csv", BANK_COLS, ("初始资金", "最后资金", "最高点", "最大回撤", "实际下了几注"), ("机器人",)))
    for key, name, cols, num, required in parts:
        if not (folder / name).exists():
            out["problems"].append(f"没有 {name}，这一块不显示")
            continue
        try:
            out[key] = _csv(folder / name, cols, num, required=required)
        except (OSError, ValueError):
            out["problems"].append(f"{name} 格式不对，这一块不显示")
    try:
        check = json.loads((folder / "check.json").read_text(encoding="utf-8"))
        out["check"] = check if isinstance(check, dict) else None
        if out["check"] is None:
            out["problems"].append("check.json 格式不对，场数和抽水那一段不显示")
    except FileNotFoundError:
        out["problems"].append("没有 check.json，场数和抽水那一段不显示")
    except (OSError, ValueError):
        out["problems"].append("check.json 格式不对，场数和抽水那一段不显示")
    return out, ""


# ---------- 数字的写法 ----------
def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def txt(v) -> str:
    """文字格子：空的（NaN、None、空字符串）写“—”，免得页面上出现字面的 nan。"""
    return str(v) if _ok(v) and str(v) != "" else "—"


def spct(v, digits: int = 1) -> str:
    """带正负号的百分比，负号用“−”：-0.0355 → −3.5%。空的写“—”。"""
    if not _ok(v):
        return "—"
    s = f"{float(v) * 100:+.{digits}f}%"
    return s.replace("-", "−")


def count(v) -> str:
    """注数、场数：1234 → 1,234；空的写“—”。"""
    return f"{int(v):,}" if _ok(v) else "—"


def money(v) -> str:
    """虚拟资金：1234.5 → 1,235；负数用“−”。"""
    return f"{float(v):,.0f}".replace("-", "−") if _ok(v) else "—"


def season_name(y) -> str:
    """赛季用开始年份表示：2012 → 2012/13。"""
    y = int(y)
    return f"{y}/{(y + 1) % 100:02d}"


# ---------- 每个机器人一句话说规则 ----------
def rule_text(row: dict, bot_cfg: dict | None) -> str:
    """机器人的规则，一句大白话（未转义；调用的地方负责转义）。

    规则种类、概率来源、注额来自 summary.csv；期望收益门槛（3% / 5% / 8%）summary.csv 里没有，
    从 settings.yaml bots.list 里同名的那一项拿（拿不到就不写具体数）。
    """
    rule, prob, stake = row.get("规则"), row.get("概率来源"), row.get("注额")
    if rule == "random":
        text = "对照组：每场在主胜、平、客胜里随便挑一个（三个机会一样），不看任何信息"
    elif rule == "favorite":
        text = "对照组：每场买赔率最低的那边（大家最看好的热门）"
    elif rule == "home":
        text = "对照组：每场都买主队赢"
    elif rule == "top" and prob == "model":
        text = "每场买我们的模型觉得最可能的那个结果，不管赔率划不划算"
    elif rule == "value":
        edge = (bot_cfg or {}).get("edge")
        bar = f"至少高出 {float(edge):.0%}" if _ok(edge) else "高出门槛"
        if prob == "fusion":
            text = (f"拿平博的价格去掉抽水当“真实可能性”（2012 年以前没有平博，用多家平均），"
                    f"哪家的赔率比平博去掉抽水后的公平赔率{bar}就买哪家；其实是在比价，不靠模型")
        else:
            text = f"模型算的可能性 × 赔率 − 1（期望收益）{bar}才买，买期望收益最大的那个结果"
    else:
        text = f"规则 {txt(rule)}"
    if stake == "kelly":
        text += "；每注下多少按四分之一凯利公式（觉得越划算下得越多，最多资金的 5%）"
    return text


# ---------- 表格里的格子 ----------
def roi_cell(row: dict) -> str:
    """ROI 和 95% 区间，再写一句“看得出赚亏没有”。"""
    roi, lo, hi = row.get("ROI"), row.get("ROI下限"), row.get("ROI上限")
    if not _ok(roi):
        return "—"
    if _ok(lo) and lo > 0:
        verdict = "<span class='tag'>区间整个大于 0</span>"
    elif _ok(hi) and hi < 0:
        verdict = "<span class='tag warn'>确实亏</span>"
    else:
        verdict = "<span class='small muted'>区间跨过 0：看不出赚亏，可能只是运气</span>"
    rng = f"{spct(lo)} 到 {spct(hi)}" if _ok(lo) and _ok(hi) else "—"
    return f"<b>{e(spct(roi))}</b><br><span class='small'>{e(rng)}</span><br>{verdict}"


def random_clv(summary: pd.DataFrame) -> float | None:
    """“对照_随机_平博”（规则 random、价格平博）的 CLV 平均：每场随便买平博的价格，CLV 大约就是负的抽水。没有返回 None。"""
    g = summary[(summary["规则"].astype(str) == "random") & (summary["价格"].astype(str) == "平博") & summary["CLV平均"].notna()]
    return float(g["CLV平均"].iloc[0]) if len(g) else None


def clv_cell(row: dict, ref: float | None = None) -> str:
    """CLV 平均和 95% 区间。ref 是“对照_随机_平博”的 CLV（拿来对比）。用“多家最高”价格的，写明价格未必下得到。"""
    clv, lo, hi, n = row.get("CLV平均"), row.get("CLV下限"), row.get("CLV上限"), row.get("有CLV的注数")
    if not _ok(clv):
        return "<span class='muted'>没有平博收盘，算不了</span>"
    if _ok(lo) and lo > 0:
        verdict = "高于平博收盘的公平价格"
    elif _ok(hi) and hi < 0:
        verdict = "低于平博收盘的公平价格" + (f"（随便买平博是 {spct(ref)}）" if _ok(ref) else "")
    else:
        verdict = "和平博收盘的公平价格看不出高低"
    rng = f"{spct(lo)} 到 {spct(hi)}" if _ok(lo) and _ok(hi) else "—"
    note = ("<br><span class='small' style='color:var(--amber)'>多家最高是事后挑的价格，下注时未必下得到</span>"
            if row.get("价格") == "最高" and _ok(lo) and lo > 0 else "")
    count = f"（{int(n):,} 注有收盘）" if _ok(n) else ""
    return (f"<b>{e(spct(clv))}</b><br><span class='small'>{e(rng)}</span><br>"
            f"<span class='small muted'>{e(verdict + count)}</span>{note}")


def bank_cell(r: dict | None) -> str:
    """资金模拟：10000 开始，最后剩多少、最高到过多少、停没停。"""
    if r is None:
        return "<span class='muted'>—</span>"
    stopped = str(r.get("停了没有")) == "停了"
    end = f"{money(r['初始资金'])} → <b>{money(r['最后资金'])}</b>"
    peak = f"最高到过 {money(r['最高点'])}，最多回撤 {r['最大回撤'] * 100:.0f}%" if _ok(r.get("最大回撤")) else ""
    stop = (f"<span class='tag warn'>{e(txt(r.get('停在哪天')))} 停了</span>" if stopped
            else "<span class='tag'>一直没停</span>")
    n = f"下了 {int(r['实际下了几注']):,} 注" if _ok(r.get("实际下了几注")) else ""
    return f"{end}<br><span class='small'>{e(peak)}</span><br>{stop} <span class='small muted'>{e(n)}</span>"


def curve_svg(points: list[tuple[int, float]], seasons: list[int], lo: float, hi: float, name: str) -> str:
    """一个机器人“按赛季累计盈亏”的小折线图（内嵌 SVG，不用脚本）。

    points：[(赛季, 到这个赛季末累计盈亏), ...]；seasons：所有机器人共用的赛季（横轴一样，几张图能对着看）；
    lo、hi：所有机器人共用的纵轴范围（同一把尺子，免得小的波动看起来很大）。虚线是 0：线在虚线下面 = 累计亏。
    """
    if not points or not seasons:
        return "<span class='muted'>—</span>"
    w, h, pad = 168, 54, 4
    span = max(hi - lo, 1.0)
    pos = {s: i for i, s in enumerate(seasons)}

    def x(s):
        return pad + pos[s] / max(len(seasons) - 1, 1) * (w - 2 * pad)

    def y(v):
        return pad + (hi - v) / span * (h - 2 * pad)

    first = points[0][0]
    xy = [f"{x(first):.1f},{y(0):.1f}"] + [f"{x(s):.1f},{y(v):.1f}" for s, v in points]   # 从 0 开始画
    end = points[-1][1]
    color = "#34d399" if end >= 0 else "#f87171"
    tip = f"{name}：{season_name(first)} 到 {season_name(points[-1][0])} 累计 {money(end)}"
    return (f"<svg class='chart' viewBox='0 0 {w} {h}' style='max-width:{w}px' role='img' aria-label='{e(tip)}'>"
            f"<line x1='{pad}' x2='{w - pad}' y1='{y(0):.1f}' y2='{y(0):.1f}' stroke='#5f7468' stroke-dasharray='3 3'/>"
            f"<polyline fill='none' stroke='{color}' stroke-width='1.8' points='{' '.join(xy)}'><title>{e(tip)}</title></polyline>"
            f"</svg><span class='small'>{e('累计 ' + money(end))}</span>")


# ---------- 最上面的结论（按结果文件里的数算，不写死） ----------
def _random_theory(check: dict | None, book: str) -> dict | None:
    """check.json 里“对照_随机_<价格>”的理论值和实际值；没有或格式不对返回 None。"""
    try:
        for name, v in (check or {}).get("随机机器人的理论值", {}).items():
            if name.endswith("_" + book) and _ok(v["理论ROI"]) and _ok(v["实际ROI"]):
                return {"name": str(name), "theory": float(v["理论ROI"]), "actual": float(v["实际ROI"]),
                        "ci": [float(x) for x in v.get("区间", [])][:2], "inside": v.get("理论值在区间里")}
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


def conclusions(summary: pd.DataFrame, bank: pd.DataFrame | None, check: dict | None) -> list[str]:
    """读者最需要知道的几句（HTML，已转义）。每句都是从结果文件里数出来的，条件不成立就不写那一句。"""
    rows = summary.to_dict("records")
    n = len(rows)
    pos = [r for r in rows if _ok(r["ROI下限"]) and r["ROI下限"] > 0]
    neg = [r for r in rows if _ok(r["ROI上限"]) and r["ROI上限"] < 0]
    out = []
    if not pos:
        out.append(f"<b>{n} 个机器人没有一个 ROI 显著为正</b>：没有一个的 95% 区间整个大于 0。"
                   f"{len(neg)} 个区间整个小于 0（确实亏），其余 {n - len(neg)} 个区间跨过 0，看不出赚亏，可能只是运气。")
    else:
        names = "、".join(e(r["机器人"]) for r in pos)
        out.append(f"{n} 个机器人里有 {len(pos)} 个 ROI 区间整个大于 0（{names}）。这么多机器人一起比，碰巧有一两个“看起来赚钱”是正常的，"
                   "要在期末考试赛季再过一次才算。")
    pin = _random_theory(check, "平博")
    if pin:
        out.append(f"<b>抽水是真实成本</b>：用平博的价格每场随便买一个，理论上每下 100 亏 {abs(pin['theory']) * 100:.1f}"
                   f"（这就是抽水），回测实际 {e(spct(pin['actual']))}。想赚钱，先得比庄家准出这么多。")
    # CLV 是和平博收盘去掉抽水后的公平价格比：按平博的价格下注，什么都不想也是负的（约等于负的抽水），
    # 所以拿“对照_随机_平博”的 CLV 当参照，看模型挑的注比随便买好多少
    ref = random_clv(summary)
    model_pin = [r for r in rows if r["价格"] == "平博" and r["概率来源"] == "model"
                 and _ok(r["CLV平均"]) and _ok(r["CLV上限"])]
    if model_pin and all(r["CLV上限"] < 0 for r in model_pin):
        lo, hi = min(r["CLV平均"] for r in model_pin), max(r["CLV平均"] for r in model_pin)
        rng = e(spct(lo)) if f"{lo:.3f}" == f"{hi:.3f}" else f"{e(spct(lo))} 到 {e(spct(hi))}"
        text = (f"<b>模型没有打败平博收盘</b>：用模型、按平博价格下注的机器人，CLV 都显著为负（{rng}）。"
                "CLV 是拿下注赔率和平博收盘去掉抽水后的公平价格比，按平博的价格下注本来就带着抽水，")
        if _ok(ref):
            # 用页面上显示的两个数（保留 1 位小数）相减，读者心算对得上：−1.9% 和 −2.4% → 0.5 个百分点
            gap = float(f"{hi * 100:.1f}") - float(f"{ref * 100:.1f}")
            text += (f"每场随便买平博的价格 CLV 是 {e(spct(ref))}；模型挑的注最好也只比随便买多 {gap:.1f} 个百分点"
                     if gap > 0 else f"每场随便买平博的价格 CLV 是 {e(spct(ref))}，模型挑的注没有比随便买好")
            text += "，还是负的：模型挑注的眼光没有超过平博收盘。"
        else:
            text += "模型挑的注没能把这部分抽水赚回来。"
        out.append(text)
    best = [r for r in rows if r["价格"] == "最高" and r["概率来源"] == "fusion" and _ok(r["CLV平均"])
            and _ok(r["CLV下限"]) and r["CLV下限"] > 0]
    if best:
        top = max(best, key=lambda r: r["CLV平均"])
        cross = sum(1 for r in best if not (_ok(r["ROI下限"]) and r["ROI下限"] > 0) and not (_ok(r["ROI上限"]) and r["ROI上限"] < 0))
        up = sum(1 for r in best if _ok(r["ROI下限"]) and r["ROI下限"] > 0)
        if cross == len(best):
            roi_txt = "而且它们的 ROI 区间仍然跨过 0。"
        elif up:
            roi_txt = f"其中 {up} 个 ROI 区间整个大于 0，但价格未必下得到，这个 ROI 当不了真。"
        else:
            roi_txt = f"它们的 ROI 区间有 {cross} 个跨过 0，其余整个小于 0。"
        out.append(f"<b>排在最前面的“融合价值 × 多家最高”是比价，不是模型准</b>：CLV 最高 {e(spct(top['CLV平均']))}，"
                   "因为它专挑比平博去掉抽水后的公平赔率高的那家。这些价格是事后从多家里挑的，<b>下注时未必下得到</b>，"
                   "公司也会限制这样赢钱的账户；" + roi_txt)
    if bank is not None and len(bank):
        stopped = int((bank["停了没有"].astype(str) == "停了").sum())
        start = money(bank["初始资金"].iloc[0])
        who = "全部" if stopped == len(bank) else f"{stopped} 个"
        out.append(f"资金模拟：每个机器人从 {start} 虚拟资金开始按风险规则下，{len(bank)} 个里{who}因为从最高点回撤太多被停掉。")
    return out


def vig_section(check: dict | None) -> str:
    """“为什么抽水是真实成本”：用随机机器人的理论值和实际值讲。"""
    pin, best = _random_theory(check, "平博"), _random_theory(check, "最高")
    if not pin and not best:
        return placeholder("没有随机下注的理论值（check.json 没有或格式不对）")
    items, inside = [], []
    for t, label in ((pin, "平博的价格"), (best, "多家里挑最高的价格")):
        if t:
            ci = f"（区间 {spct(t['ci'][0])} 到 {spct(t['ci'][1])}）" if len(t["ci"]) == 2 else ""
            items.append(f"<li>{e(label)}：理论 {e(spct(t['theory']))}，回测实际 {e(spct(t['actual']) + ci)}。</li>")
            inside.append((t["name"], t["inside"]))
    # “理论值在区间里”按 check.json 里回测程序自己记的字段写：都是 true 才说“没发现算错”；有 false 就点名；没有这个字段就不写
    if inside and all(v is True for _, v in inside):
        verdict = "随机下注的理论值都落在回测的区间里，没发现回测程序算错。"
    elif any(v is False for _, v in inside):
        bad = "、".join(n for n, v in inside if v is False)
        verdict = f"{bad} 的理论值不在回测的区间里，要先查回测程序。"
    else:
        verdict = ""
    return ("<div class='card pad'><p>公司开出的三个赔率，换算成可能性（1 ÷ 赔率）加起来总是超过 100%，多出来的就是公司的抽水。"
            "所以闭着眼睛随便买，长期就是亏这么多：</p>"
            f"<ul class='points'>{''.join(items)}</ul>"
            f"<p class='small'>{e(verdict)}"
            + ("“多家最高”比平博少亏很多，是因为每个结果都挑了最高的那家，把抽水几乎挑没了；"
               "这只是事后比价，实际下注时很难同时在几家都下到最高价。" if best else "")
            + "</p></div>")


# ---------- 每个机器人的简短统计（两段、每个赛季） ----------
def bot_detail_rows(summary: pd.DataFrame, season: pd.DataFrame | None, period: pd.DataFrame | None) -> list[list[str]]:
    """每个机器人一行：两段的 ROI、几个赛季赚、最好和最差的赛季、最大回撤（固定注额）、最长连输。"""
    labels = sorted(period["两段"].astype(str).unique()) if period is not None else []
    rows = []
    for r in summary.to_dict("records"):
        name = r["机器人"]
        cells = [e(name)]
        if period is not None:
            p = period[period["机器人"] == name]
            for lab in labels:
                g = p[p["两段"].astype(str) == lab]
                if g.empty:
                    cells.append("<span class='muted'>没下注</span>")
                else:
                    g0 = g.iloc[0]
                    cells.append(f"{e(spct(g0['ROI']))}<br><span class='small muted'>{e(spct(g0['ROI下限']))} 到 "
                                 f"{e(spct(g0['ROI上限']))}，{e(count(g0['投注场数']))} 注</span>")
        if season is not None:
            g = season[(season["机器人"] == name) & (season["投注场数"] > 0)].sort_values("赛季")
            g = g[g["ROI"].notna() & g["盈亏"].notna()]    # 空的赛季不算
            if g.empty:
                cells += ["—", "—"]
            else:
                win = int((g["盈亏"] > 0).sum())
                b, w = g.loc[g["ROI"].idxmax()], g.loc[g["ROI"].idxmin()]
                cells += [f"{win} / {len(g)}",
                          e(f"最好 {season_name(b['赛季'])} {spct(b['ROI'])}；最差 {season_name(w['赛季'])} {spct(w['ROI'])}")]
        cells += [e(money(r["最大回撤"])), e(f"{int(r['最长连输'])} 注") if _ok(r["最长连输"]) else "—"]
        rows.append(cells)
    head = (["机器人"] + [f"ROI（{lab}）" for lab in labels]
            + (["赚钱的赛季 / 下了注的赛季", "最好、最差的赛季（ROI）"] if season is not None else [])
            + ["最大回撤（累计盈亏从最高点往下掉最多多少）", "最长连输"])
    return [head] + rows


# ---------- 整页 ----------
SAFETY = ("<b>全是虚拟的：不用真钱，不连任何博彩账户，只是历史回测。</b>这里只说“历史上照这些规则下会怎样”，不是下注建议，"
          "也不鼓励下注。中国大陆合法的只有体育彩票和福利彩票。")

def guide(settings: dict, summary: pd.DataFrame, bank: pd.DataFrame | None) -> str:
    """“几个词”那一块（HTML，已转义）。资金模拟的数从 settings.yaml bots 段和 bankroll.csv 拿，不写死。"""
    b = settings.get("bots") or {}
    ref = random_clv(summary)
    clv_ref = f"（这一版“对照_随机_平博”是 {spct(ref)}）" if _ok(ref) else ""
    stop = b.get("stop_drawdown")
    stop_txt = f"{float(stop):.0%}" if _ok(stop) else "规定的比例"
    # 停的时候还没结算的注：下注在开球前 hours_before_kickoff 小时，结果要开球后 settle_after_kickoff_hours 小时才算
    before, after = b.get("hours_before_kickoff"), b.get("settle_after_kickoff_hours")
    days = (f"约 {(float(before) + float(after)) / 24:.0f} 天的注没结算（结果要开球后 {float(after):.0f} 小时才算）"
            if _ok(before) and _ok(after) else "一些注没结算")
    dd = bank["最大回撤"].dropna() if bank is not None else pd.Series(dtype=float)
    dd_txt = (f"（这一版 {dd.min() * 100:.0f}% 到 {dd.max() * 100:.0f}%）" if len(dd) and dd.min() != dd.max()
              else (f"（这一版 {dd.min() * 100:.0f}%）" if len(dd) else ""))
    start = money(bank["初始资金"].iloc[0]) if bank is not None and len(bank) else "10,000"
    return ("<div class='card pad guide'><b>几个词</b><ul class='points small'>"
            "<li><b>ROI</b>（投资回报率）= 盈亏 ÷ 总注额。−2% 就是平均每下 100 亏 2。</li>"
            "<li><b>95% 区间</b>：把历史按周打乱、重抽很多次算出来的范围。区间跨过 0，就是赚还是亏看不出来，可能只是运气。</li>"
            "<li><b>CLV</b>（收盘线价值）= 下注赔率 × 平博收盘去掉抽水后这个结果的可能性 − 1，也就是拿下注赔率和"
            "<b>平博收盘去掉抽水后的公平价格</b>比（收盘是开球前最后的价格，一般最准）。正好按平博收盘价下注，CLV 约等于负的抽水"
            f"{e(clv_ref)}，所以按平博价格下注的机器人 CLV 是负的很正常，要和随便买比。"
            "CLV 比 ROI 少受赛果运气影响，所以排行榜按它排。只有 2012/13 起有平博收盘。</li>"
            "<li><b>价格</b>：“平博”是平博一家的赛前价格；“多家最高”是每个结果各挑一家最高的赔率，事后挑的，下注时未必下得到，偏乐观。</li>"
            f"<li><b>资金模拟</b>：从 {e(start)} 虚拟资金开始，单注、单日都有上限，从最高点回撤超过 {e(stop_txt)} 就停。只报告，不参加排名。"
            f"停的时候手上还有{e(days)}，这些注照常结算，所以“最多回撤”可能比 {e(stop_txt)} 多出不少{e(dd_txt)}。"
            "<b>最后资金比起始多，不代表这个机器人赚钱</b>，要看上面固定注额的 ROI。</li>"
            "</ul></div>")


def _sort_key(r: dict):
    rank = r["名次"]
    return (0, rank) if _ok(rank) else (1, -(r["CLV平均"] if _ok(r["CLV平均"]) else -9.0))


def curves(season: pd.DataFrame | None) -> tuple[dict[str, list[tuple[int, float]]], list[int], float, float]:
    """每个机器人按赛季的累计盈亏，和所有机器人共用的横轴（赛季）、纵轴范围。"""
    if season is None or season.empty:
        return {}, [], 0.0, 0.0
    seasons = sorted(int(s) for s in season["赛季"].unique())
    out = {}
    # 先去掉盈亏空的赛季再累加（不然那个赛季之后全是 NaN，图里会写出 nan）
    for name, g in season[season["盈亏"].notna()].sort_values("赛季").groupby("机器人", sort=False):
        cum = g["盈亏"].cumsum()
        out[str(name)] = [(int(s), float(v)) for s, v in zip(g["赛季"], cum)]
    vals = [v for pts in out.values() for _, v in pts] + [0.0]
    return out, seasons, min(vals), max(vals)


HEAD = "<h1>虚拟投注 <span class='accent'>历史回测</span></h1>"
BAD_FORMAT = "虚拟投注结果格式不对"


def _placeholder_page(data, why: str) -> str:
    """只有标题、“全是虚拟的”和一句原因的占位页。"""
    body = HEAD + f"<div class='note'>{SAFETY}</div>" + placeholder(why)
    return page("虚拟投注 - 足球预测", body, "", data.now, data.settings["site"]["refresh_minutes"])


def render(data) -> str:
    """虚拟投注页 bots.html。data 是 SiteData（这里只用 settings 和 now）。

    结果文件是另一个程序写的，可能有想不到的毛病（空格子、类型不对……）。这里出了任何错，都只把这一页换成
    “虚拟投注结果格式不对”的占位页，网站别的页照常生成（所以这里故意接住所有 Exception；只写错误的种类，不写细节，
    免得把电脑上的路径写到网页上）。
    """
    return render_checked(data)[0]


def render_checked(data) -> tuple[str, str | None]:
    """和 render 一样，另外告诉调用的地方有没有走兜底：返回 (网页, 出错种类)；正常时出错种类是 None。
    build_site 用它把“虚拟投注页出错”写进生成摘要和首页提示，让状态页、首页都看得出来。"""
    try:
        return _render(data), None
    except Exception as err:   # noqa: BLE001 —— 故意的：这一页坏了不能拖垮整个网站
        kind = type(err).__name__
        return _placeholder_page(data, f"{BAD_FORMAT}，这一页先不显示（{kind}）。别的页不受影响。"), kind


def _render(data) -> str:
    cfg = data.settings["site"]
    res, why = load(data.settings)
    if res is None:
        return _placeholder_page(data, why)
    summary, season, period, bank, check = res["summary"], res["season"], res["period"], res["bank"], res["check"]
    bots_cfg = {b.get("name"): b for b in (data.settings.get("bots") or {}).get("list", []) if isinstance(b, dict)}
    bank_by = {str(r["机器人"]): r for r in bank.to_dict("records")} if bank is not None else {}
    cv, seasons, lo, hi = curves(season)
    ref = random_clv(summary)          # “对照_随机_平博”的 CLV，CLV 格子里拿来对比
    rows = []
    for r in sorted(summary.to_dict("records"), key=_sort_key):
        name = str(r["机器人"])
        tag = " <span class='tag'>对照组</span>" if str(r.get("对照组")) == "是" else ""
        # 名次空 = 回测程序写的“注数不够，不排”（有 CLV 的注不够 bots.min_clv_bets）
        rank = str(int(r["名次"])) if _ok(r["名次"]) else "<span class='small muted'>注数不够，不排</span>"
        hit = (f"{r['命中率'] * 100:.1f}%<br><span class='small muted'>平均赔率 {r['平均赔率']:.2f}</span>"
               if _ok(r["命中率"]) and _ok(r["平均赔率"]) else "—")
        book = txt(r["价格"])
        rows.append([rank, f"<b>{e(name)}</b>{tag}", e(rule_text(r, bots_cfg.get(name))),
                     e(BOOK_TEXT.get(book, book)), e(count(r["投注场数"])), hit,
                     roi_cell(r), clv_cell(r, ref), curve_svg(cv.get(name, []), seasons, lo, hi, name),
                     bank_cell(bank_by.get(name))])
    board = table(["名次", "机器人", "规则（一句话）", "价格", "下了几注", "命中率", "ROI（95% 区间）", "CLV（95% 区间）",
                   "按赛季累计盈亏", "资金模拟"], rows)
    n = len(summary)
    pos = int((summary["ROI下限"] > 0).sum())
    stopped = int((bank["停了没有"].astype(str) == "停了").sum()) if bank is not None else None
    tiles = (f"<div class='stats'><div class='stat'><b>{n}</b><span>个机器人</span></div>"
             f"<div class='stat hi'><b>{pos}</b><span>个 ROI 显著为正</span></div>"
             + (f"<div class='stat'><b>{stopped} / {len(bank)}</b><span>资金模拟里被停掉</span></div>" if bank is not None else "")
             + "</div>")
    # 回测范围（从 check.json 读）
    scope = ""
    if check:
        try:
            s0, s1 = (int(x) for x in check["赛季"])
            counts = check.get("场数")
            n_m = counts.get("有概率也有矩阵的比赛") if isinstance(counts, dict) else None   # 场数不是字典：只是不写场数
            scope = (f"五大联赛 {season_name(s0)}–{season_name(s1)} 赛季（只用开发赛季，没碰期末考试的 2024/25、2025/26），"
                     f"只玩胜平负，开球前 24 小时下注，概率来自模型 {txt(check.get('模型'))}（时点 {txt(check.get('时点'))}）"
                     + (f"，{int(n_m):,} 场比赛" if _ok(n_m) else "") + f"。版本 {txt(check.get('版本'))}，结果不再改。")
        except (AttributeError, KeyError, TypeError, ValueError):
            scope = ""
    curve_note = (("<p class='small'>“按赛季累计盈亏”：固定每注 100 虚拟资金（凯利类按初始资金 10,000 算注额），一个赛季一个点，"
                   "所有小图用同一把尺子。虚线是 0，线在虚线下面就是累计亏。绿线 = 最后累计是赚的，红线 = 亏的。</p>")
                  if cv else "")
    det = bot_detail_rows(summary, season, period)
    detail = details("每个机器人的简短统计（分两段、每个赛季）",
                     "<p class='small'>2012/13 以前没有平博赔率，用平博价格的机器人那一段没下注。两段、每个赛季只报告，不当结论："
                     "一段赚一段亏很常见。</p>" + table(det[0], det[1:]))
    problems = "".join(f"<p class='small muted'>{e(p)}</p>" for p in res["problems"])
    body = (HEAD
            + (f"<p class='muted'>{e(scope)}</p>" if scope else "")
            + f"<div class='note'>{SAFETY}</div>"
            + "<div class='card pad'><b>先看结论：这一页是看“为什么很难赢”，不是下注参考</b><ul class='points'>"
            + "".join(f"<li>{c}</li>" for c in conclusions(summary, bank, check)) + "</ul></div>"
            + tiles + guide(data.settings, summary, bank)
            + "<h2>机器人排行榜（按 CLV 从高到低）</h2>" + board + curve_note
            + "<h2>为什么说抽水是真实成本</h2>" + vig_section(check)
            + "<h2>每个机器人的细节</h2>" + detail
            + "<p class='small muted'>历史回测赚钱不代表以后能赚：赔率会变、公司会限制赢钱的账户，开发赛季还被用来挑过模型。"
              "排名靠前的规则要在期末考试赛季再过一次才算"
            + ("，这一版没有找到值得拿去考的" if pos == 0 else "")
            + "。每注明细太大，网站上不放。赔率和比分来自 football-data.co.uk。</p>"
            + problems)
    return page("虚拟投注 - 足球预测", body, "", data.now, cfg["refresh_minutes"])
