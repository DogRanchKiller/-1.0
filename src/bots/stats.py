"""机器人的成绩、排行榜、合理性检查、中文报告（需求文档第 14 节“评价指标”“排名”“检查结果合不合理”“结论怎么写”）。

成绩都按“成绩用的注额”算（固定 100；凯利按初始资金 10000 算，不复利），和资金模拟分开。
区间都按周成块 bootstrap（settings compare 的次数、置信水平、种子；同一周的注单一起抽）：
  ROI 抽周后算 Σ盈亏 ÷ Σ注额；CLV 抽周后算平均（每注一样重）。
全是虚拟投注；报告只说“历史上这样下会怎样”，不写“推荐下注”。
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from src.evaluate.compare import block_ids, bootstrap_mean_diff
from src.evaluate.market_checks import ratio_ci
from src.web.theme import BASE_CSS, VIEWPORT

LUCK = "可能只是运气"            # ROI 区间跨过 0
GOOD_PRICE = "价格确实比收盘好"   # CLV 区间整个大于 0
TOO_FEW = "注数不够，不排"


def _roi_ci(b: pd.DataFrame, settings: dict) -> tuple[float, float, float]:
    """ROI = Σ盈亏 ÷ Σ注额 和 95% 区间。只有一周的注单时区间算不出来，是空值。"""
    if b.empty or b["stake"].sum() <= 0:
        return np.nan, np.nan, np.nan
    if b["match_date"].nunique() < 2:
        return float(b["profit"].sum() / b["stake"].sum()), np.nan, np.nan
    return ratio_ci(b["profit"].to_numpy(float), b["stake"].to_numpy(float), b["match_date"], settings)


def _clv_ci(b: pd.DataFrame, settings: dict) -> tuple[float, float, float]:
    """CLV 平均和 95% 区间（只用有 CLV 的注）。"""
    x = b.dropna(subset=["clv"])
    if x.empty:
        return np.nan, np.nan, np.nan
    if x["match_date"].nunique() < 2:
        return float(x["clv"].mean()), np.nan, np.nan
    c = settings["compare"]
    return bootstrap_mean_diff(x["clv"].to_numpy(float), block_ids(x["match_date"], c["block"]),
                               int(c["n_boot"]), float(c["confidence"]), int(c["seed"]))


def max_drawdown(profits) -> float:
    """累计盈亏曲线（从 0 开始）从最高点往下最多掉多少（金额）。"""
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(profits, dtype=float))])
    return float((np.maximum.accumulate(cum) - cum).max())


def longest_losing_streak(outcomes) -> int:
    """最长连输几注。“走”（退回本金）不打断，也不算输。"""
    best = cur = 0
    for o in outcomes:
        if o == "loss":
            cur += 1
            best = max(best, cur)
        elif o == "win":
            cur = 0
    return best


def bot_stats(b: pd.DataFrame, settings: dict) -> dict:
    """一个机器人（或它的一段）的成绩，b 是它的注单。"""
    b = b.sort_values(["cutoff_utc", "match_id"], kind="stable")
    settled = b[b["outcome"] != "push"]
    roi, roi_lo, roi_hi = _roi_ci(b, settings)
    clv_m, clv_lo, clv_hi = _clv_ci(b, settings)
    has_clv = b["clv"].notna()
    return {"投注场数": int(len(b)), "走（退回）": int((b["outcome"] == "push").sum()),
            "命中率": float((settled["outcome"] == "win").mean()) if len(settled) else np.nan,
            "平均赔率": float(b["odds"].mean()) if len(b) else np.nan,
            "总注额": float(b["stake"].sum()), "盈亏": float(b["profit"].sum()),
            "ROI": roi, "ROI下限": roi_lo, "ROI上限": roi_hi,
            "有CLV的注数": int(has_clv.sum()), "CLV平均": clv_m, "CLV下限": clv_lo, "CLV上限": clv_hi,
            "CLV>0的比例": float((b.loc[has_clv, "clv"] > 0).mean()) if has_clv.any() else np.nan,
            "最大回撤": max_drawdown(b["profit"]), "最长连输": longest_losing_streak(b["outcome"])}


def _flags(row: dict) -> tuple[str, str]:
    luck = LUCK if not (row["ROI下限"] > 0 or row["ROI上限"] < 0) else ""
    good = GOOD_PRICE if row["CLV下限"] > 0 else ""
    return luck, good


def summarize(bets: pd.DataFrame, bots: list[dict], settings: dict) -> pd.DataFrame:
    """每个机器人一行，就是排行榜：按 CLV 平均从高到低，有 CLV 的注 ≥ min_clv_bets 才排名次，不够的列在下面。"""
    cfg = settings["bots"]
    rows = []
    for bot in bots:
        s = bot_stats(bets[bets["bot"] == bot["name"]], settings)
        luck, good = _flags(s)
        rows.append({"机器人": bot["name"], "对照组": "是" if bot.get("control") else "", "价格": bot["book"],
                     "规则": bot["rule"], "概率来源": bot.get("prob", ""), "注额": bot["stake"], **s,
                     "ROI标记": luck, "CLV标记": good})
    t = pd.DataFrame(rows)
    ranked = t["有CLV的注数"] >= int(cfg["min_clv_bets"])
    top = t[ranked].sort_values("CLV平均", ascending=False, kind="stable")
    top.insert(0, "名次", range(1, len(top) + 1))
    rest = t[~ranked].assign(名次=TOO_FEW)
    return pd.concat([top, rest[top.columns]], ignore_index=True)


def by_group(bets: pd.DataFrame, bots: list[dict], settings: dict, key: str) -> pd.DataFrame:
    """按两段（key = 'period'）或每个赛季（key = 'season'）分开的成绩，只报告。"""
    split = int(settings["bots"]["period_split"])
    b = bets.assign(period=np.where(bets["season"] < split, f"{split} 之前", f"{split} 起"))
    rows = []
    for bot in bots:
        mine = b[b["bot"] == bot["name"]]
        for g, part in mine.groupby(key, sort=True):
            s = bot_stats(part, settings)
            luck, good = _flags(s)
            rows.append({"机器人": bot["name"], "两段" if key == "period" else "赛季": g, **s,
                         "ROI标记": luck, "CLV标记": good})
    return pd.DataFrame(rows)


def random_theory(odds: dict, settings: dict) -> dict:
    """随机下注的理论 ROI：每场 1/3 × Σ(去水概率 × 赔率) − 1，用这个价格本身按比例去水。
    去水概率 × 赔率 = 1 ÷ 倒数之和，所以每场 = 1 ÷ 倒数之和 − 1；固定注额，所以 ROI = 各场平均。"""
    out = {}
    for name, t in odds.items():
        ok = t["ok"].to_numpy(bool)
        o3 = t.loc[ok, ["home", "draw", "away"]].to_numpy(float)
        out[name] = float((1 / (1 / o3).sum(axis=1) - 1).mean()) if len(o3) else np.nan
    return out


def checks(summary: pd.DataFrame, theory: dict, bots: list[dict]) -> dict:
    """跑完先看这两条（第 14 节“检查结果合不合理”）。"""
    s = summary.set_index("机器人")
    rnd = {}
    for bot in bots:
        if bot["rule"] != "random":
            continue
        r = s.loc[bot["name"]]
        th = theory.get(bot["book"], np.nan)
        rnd[bot["name"]] = {"理论ROI": th, "实际ROI": r["ROI"], "区间": [r["ROI下限"], r["ROI上限"]],
                            "理论值在区间里": bool(r["ROI下限"] <= th <= r["ROI上限"])}
    # 模型价值 × 平博的 CLV 显著为正 → 先当作偷看未来去查（第 16 节）
    suspect = [b["name"] for b in bots if b.get("prob") == "model" and b["book"] == "平博"
               and s.loc[b["name"], "CLV下限"] > 0]
    return {"随机机器人的理论值": rnd, "模型用平博价格CLV显著为正（要先查是不是偷看未来）": suspect}


# ======================= 中文报告（只用 CSS，样式用 src/web/theme.py） =======================
def _fmt(v, kind: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="muted">—</span>'
    if kind == "pct":
        return f"{v * 100:+.2f}%"
    if kind == "share":
        return f"{v * 100:.1f}%"
    if kind == "money":
        return f"{v:,.0f}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(r[c], k)}</td>" for c, k in cols) + "</tr>"
                   for _, r in df.iterrows())
    return f'<div class="card scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


BOARD_COLS = [("名次", ""), ("机器人", ""), ("对照组", ""), ("投注场数", ""), ("命中率", "share"), ("平均赔率", ""),
              ("盈亏", "money"), ("ROI", "pct"), ("ROI下限", "pct"), ("ROI上限", "pct"), ("有CLV的注数", ""),
              ("CLV平均", "pct"), ("CLV下限", "pct"), ("CLV上限", "pct"), ("CLV>0的比例", "share"),
              ("最大回撤", "money"), ("最长连输", ""), ("ROI标记", ""), ("CLV标记", "")]


def report_html(summary: pd.DataFrame, by_period: pd.DataFrame, bankroll: pd.DataFrame, check: dict,
                settings: dict) -> str:
    """一页中文报告：怎么看、合理性检查、排行榜、两段、资金模拟。"""
    cfg = settings["bots"]
    rnd = "".join(f"<li>{html.escape(k)}：理论 {_fmt(v['理论ROI'], 'pct')}，实际 {_fmt(v['实际ROI'], 'pct')}"
                  f"（区间 {_fmt(v['区间'][0], 'pct')} 到 {_fmt(v['区间'][1], 'pct')}）"
                  f"{'' if v['理论值在区间里'] else ' <span class=\"tag warn\">不在区间里，先查程序</span>'}</li>"
                  for k, v in check["随机机器人的理论值"].items())
    sus = check["模型用平博价格CLV显著为正（要先查是不是偷看未来）"]
    sus_html = (f'<p><span class="tag warn">要先查</span> {html.escape("、".join(sus))} 的 CLV 区间整个大于 0：'
                f"模型比平博收盘还准，先当作偷看未来去查（第 16 节）。</p>" if sus else
                "<p>模型用平博价格的机器人，CLV 区间都没有整个大于 0（没有“比收盘还准”的可疑信号）。</p>")
    period_cols = [("机器人", ""), ("两段", ""), ("投注场数", ""), ("ROI", "pct"), ("ROI下限", "pct"),
                   ("ROI上限", "pct"), ("有CLV的注数", ""), ("CLV平均", "pct"), ("CLV下限", "pct"), ("CLV上限", "pct")]
    bank_cols = [("机器人", ""), ("初始资金", "money"), ("最后资金", "money"), ("最高点", "money"),
                 ("最大回撤", "share"), ("停了没有", ""), ("停在哪天", ""), ("实际下了几注", ""), ("单日上限跳过", "")]
    n = check.get("场数", {})
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{VIEWPORT}
<title>虚拟投注机器人历史回测 {html.escape(cfg['version'])}</title><style>{BASE_CSS}</style></head>
<body><div class="wrap">
<h1>虚拟投注机器人 <span class="accent">历史回测 {html.escape(cfg['version'])}</span></h1>
<p class="small">五大联赛 {cfg['first_season']}/{(cfg['first_season'] + 1) % 100:02d}–{cfg['last_season']}/{(cfg['last_season'] + 1) % 100:02d}（只用开发赛季），
胜平负，开球前 24 小时下注，概率来自回测 {html.escape(cfg['run'])}（{html.escape(cfg['model'])}）。
有概率的比赛 {n.get('有概率也有矩阵的比赛', '—')} 场，取数层认的 {n.get('取数层认的比赛', '—')} 场。</p>
<div class="note"><b>全是虚拟投注，不连任何账户。</b>这里只说“历史上这样下会怎样”，不是下注建议。<br>
历史回测赚钱不代表以后能赚：赔率会变、公司会限制赢钱的账户、“多家最高”是每个结果各挑一家最高的价格，实际未必同时下得到（偏乐观），
开发赛季还被用来选过模型和融合权重。排名靠前的规则要在期末考试赛季再过一次才算；这一版不碰考试赛季。<br>
19 个机器人、每个两个区间，碰巧有一两个“看起来赚钱”是正常的（多重比较）。</div>
<h2>先看：结果合不合理</h2>
<div class="card" style="padding:14px 18px"><ul>{rnd}</ul>{sus_html}</div>
<h2>排行榜（按 CLV 平均从高到低）</h2>
<p class="small">CLV = 下注赔率 × 平博收盘去水后这个结果的概率 − 1，大于 0 = 买到了比收盘更好的价格；
只算有平博收盘的比赛（2012/13 起）。有 CLV 的注不到 {cfg['min_clv_bets']} 注的不排名次，列在下面。
ROI = 盈亏 ÷ 总注额（固定注额 {cfg['flat_stake']}，凯利按初始资金 {cfg['bankroll']:,} 算）。区间是 95%，按周成块 bootstrap。</p>
{_table(summary, BOARD_COLS)}
<h2>两段分开看（只报告）</h2>
<p class="small">{cfg['period_split']} 之前没有平博赔率，平博的机器人一注都没有、也没有 CLV。</p>
{_table(by_period, period_cols) if not by_period.empty else '<p class="muted">没有注单</p>'}
<h2>资金模拟（只报告，不参加排名）</h2>
<p class="small">每个机器人从 {cfg['bankroll']:,} 开始，按截止时间从早到晚下注；单注固定类 min({cfg['flat_stake']}, 资金 × {cfg['flat_cap']:.0%})、
凯利类最多资金 × {cfg['kelly_cap']:.0%}；同一天合计最多当天第一注时资金的 {cfg['daily_cap']:.0%}；从最高点回撤超过 {cfg['stop_drawdown']:.0%} 就停（不恢复）。</p>
{_table(bankroll, bank_cols)}
<footer class="small">生成程序 src/evaluate/bot_backtest.py；规则见需求文档第 14 节“历史回测第一版规则 {html.escape(cfg['version'])}”。</footer>
</div></body></html>"""
