"""周报、月报（需求文档第 15 节“汇总复盘”）：data/review/reports/周报_YYYY-Www.md、月报_YYYY-MM.md。

- 按北京时间的开球日期分周（ISO 周，周一到周日）、分月。
- 内容：场数；每个版本 × 来源（模型 / 融合 / 收盘赔率）的平均 log loss、Brier、RPS；
  同场比较（正式版本的模型、融合和收盘赔率，只比都有的那几场）；最大的几次失误；误差归类分布。
- 场数少于 review.reports.min_matches 时写“样本太少，不能下结论”。
- 每周报告只用来发现异常、提醒；不能据此改模型（第 15 节表格）。不写下注建议。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.review.match_review import BEIJING, KIND_NAMES, OUTCOMES, RESULT_INDEX
from src.review.reconcile import SOURCE_CLOSE, SOURCE_FUSION, SOURCE_MODEL, kind_rank

METRICS = [("logloss_1x2", "胜平负 log loss"), ("brier_1x2", "Brier"), ("rps_1x2", "RPS"),
           ("logloss_score", "比分 log loss"), ("logloss_over25", "大 2.5 球 log loss"), ("logloss_btts", "双方进球 log loss")]
SOURCE_ORDER = {SOURCE_MODEL: 0, SOURCE_FUSION: 1, SOURCE_CLOSE: 2}


def bj_day(utc_text: str) -> pd.Timestamp:
    """UTC 开球时间 → 北京时间的日期（分周、分月用）。"""
    t = pd.Timestamp(utc_text)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_convert(BEIJING).normalize().tz_localize(None)


def period_keys(day: pd.Timestamp) -> tuple[str, str]:
    """（周的名字, 月的名字），例如 ("2026-W44", "2026-10")。"""
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}", day.strftime("%Y-%m")


def period_range(kind: str, key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """一周 / 一个月的第一天和最后一天（北京时间日期）。"""
    if kind == "周":
        start = pd.Timestamp(datetime.strptime(key + "-1", "%G-W%V-%u"))
        return start, start + pd.Timedelta(days=6)
    start = pd.Timestamp(key + "-01")
    return start, start + pd.offsets.MonthEnd(0)


def fmt(x, digits: int = 4) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def average_table(sc: pd.DataFrame, kinds: list[str]) -> list[str]:
    """每个版本 × 来源一行：场数和各项平均分数（空的不算进平均；整列都空写“—”）。"""
    if sc.empty:
        return ["（没有数据）"]
    g = sc.groupby(["kind", "source"], sort=False)
    keys = sorted(g.groups.keys(), key=lambda k: (k[1] == SOURCE_CLOSE, kind_rank(k[0], kinds), SOURCE_ORDER.get(k[1], 9)))
    rows = []
    for k in keys:
        part = g.get_group(k)
        rows.append([KIND_NAMES.get(k[0], k[0]), k[1], part["match_id"].nunique()]
                    + [fmt(pd.to_numeric(part[c], errors="coerce").mean()) for c, _ in METRICS])
    return md_table(["版本", "来源", "场数"] + [n for _, n in METRICS], rows)


def official_rows(sc: pd.DataFrame, reviews: list[dict]) -> pd.DataFrame:
    """每场正式那一版的模型行、融合行，加上这场的收盘赔率行。列 who = 模型 / 融合 / 收盘赔率。"""
    pids = {r["match_id"]: next(e["prediction_id"] for e in r["各版本"] if e["版本"] == r["正式版本"]) for r in reviews}
    keep = sc[(sc["source"] == SOURCE_CLOSE) |
              (sc["prediction_id"].notna() & (sc["prediction_id"] == sc["match_id"].map(pids)))]
    return keep.assign(who=keep["source"])


def pair_table(off: pd.DataFrame) -> list[str]:
    """同场比较：两个来源都有的那几场，平均胜平负 log loss，和哪边更小的场数。"""
    piv = off.pivot_table(index="match_id", columns="who", values="logloss_1x2", aggfunc="first")
    rows = []
    for a, b in ((SOURCE_MODEL, SOURCE_CLOSE), (SOURCE_FUSION, SOURCE_CLOSE), (SOURCE_MODEL, SOURCE_FUSION)):
        if a not in piv or b not in piv:
            rows.append([f"{a} 对 {b}", 0, "—", "—", "—"])
            continue
        both = piv[[a, b]].dropna()
        rows.append([f"{a} 对 {b}", len(both), fmt(both[a].mean() if len(both) else None),
                     fmt(both[b].mean() if len(both) else None),
                     f"{int((both[a] < both[b]).sum())} : {int((both[a] > both[b]).sum())}" if len(both) else "—"])
    return md_table(["比较", "两边都有的场数", "前者平均", "后者平均", "前者更小 : 后者更小（场）"], rows)


def miss_table(reviews: list[dict], n: int) -> list[str]:
    """最大的几次失误：正式版本模型的胜平负 log loss 最大的几场。"""
    items = []
    for r in reviews:
        e = next(e for e in r["各版本"] if e["版本"] == r["正式版本"])
        ll = e[SOURCE_MODEL]["logloss_1x2"]
        res = RESULT_INDEX[{"主胜": "H", "平局": "D", "客胜": "A"}[r["实际"]["结果"]]]
        items.append((ll, r, e[SOURCE_MODEL]["概率"][res]))
    items.sort(key=lambda x: -x[0])
    rows = [[r["比赛日期"], r["联赛"], f"{r['主队中文'] or r['主队']} {r['实际']['比分'][0]}-{r['实际']['比分'][1]} "
             f"{r['客队中文'] or r['客队']}", KIND_NAMES.get(r["正式版本"], r["正式版本"]),
             f"{p:.0%}", fmt(ll), r["误差归类"]["类型"]] for ll, r, p in items[:n]]
    if not rows:
        return ["（没有数据）"]
    return md_table(["日期", "联赛", "比赛", "版本", "模型给实际结果", "胜平负 log loss", "误差归类"], rows)


def error_table(reviews: list[dict]) -> list[str]:
    counts = pd.Series([r["误差归类"]["类型"] for r in reviews]).value_counts()
    if counts.empty:
        return ["（没有数据）"]
    return md_table(["类型", "场数"], [[k, int(v)] for k, v in counts.items()])


def build_report(kind: str, key: str, reviews: list[dict], sc: pd.DataFrame, settings: dict, now_utc) -> str:
    """一份周报 / 月报的 Markdown 文字。kind：“周”或“月”。"""
    cfg = settings["review"]["reports"]
    kinds = list(settings["review"]["kinds"])
    start, end = period_range(kind, key)
    today = pd.Timestamp(now_utc).tz_convert(BEIJING).normalize().tz_localize(None)
    n = len(reviews)
    title = f"{'周报' if kind == '周' else '月报'} {key}"
    lines = [f"# 赛后复盘{title}", "",
             f"- 时间段（北京时间开球日期）：{start:%Y-%m-%d} 至 {end:%Y-%m-%d}",
             f"- 生成时间（北京时间）：{pd.Timestamp(now_utc).tz_convert(BEIJING):%Y-%m-%d %H:%M}"]
    if end >= today:
        lines.append("- 这段时间还没过完，后面的比赛踢完会补进来（每次全部重算）。")
    comps = pd.Series([r["联赛"] for r in reviews]).value_counts()
    lines.append(f"- 对账场数：{n}（" + "、".join(f"{c} {int(v)} 场" for c, v in comps.items()) + "）")
    if n < int(cfg["min_matches"]):
        lines.append(f"- **样本太少，不能下结论**（{n} 场，少于 {cfg['min_matches']} 场）。下面的数字只用来发现异常，"
                     "例如某个联赛突然变差、数据漏抓。")
    lines += ["", "分数都是越小越好。log loss 看概率给得准不准，Brier、RPS 是另两种算法（RPS 考虑主胜、平、客胜的先后顺序）。", "",
              "## 各版本平均分数", ""] + average_table(sc, kinds)
    lines += ["", "## 同场比较（正式版本）", "",
              "每场的正式版本按 T-1 → T-6 → T-24 → 预览版的顺序取。只比两边都有的那几场，这样才公平。", ""]
    lines += pair_table(official_rows(sc, reviews))
    lines += ["", f"## 最大的 {cfg['top_misses']} 次失误", ""] + miss_table(reviews, int(cfg["top_misses"]))
    lines += ["", "## 误差归类", "", "规则见需求文档第 15 节“误差原因怎么归类”。", ""] + error_table(reviews)
    lines += ["", "---", "这份报告只用来记录和展示，不能据此直接改模型或任何参数（要改先写进改进清单、用历史数据回测）。", ""]
    return "\n".join(lines)


def write_reports(out_dir: Path, reviews: list[dict], sc: pd.DataFrame, settings: dict, now_utc) -> list[Path]:
    """按周、按月各写一份；全部重算，这次没有的旧报告删掉。返回写了的文件。"""
    folder = Path(out_dir) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in reviews:
        w, m = period_keys(bj_day(r["开球时间_utc"] or r["比赛日期"] + " 12:00:00"))
        groups.setdefault(("周", w), []).append(r)
        groups.setdefault(("月", m), []).append(r)
    written = []
    for (kind, key), rs in sorted(groups.items()):
        ids = {r["match_id"] for r in rs}
        text = build_report(kind, key, rs, sc[sc["match_id"].isin(ids)], settings, now_utc)
        path = folder / f"{'周报' if kind == '周' else '月报'}_{key}.md"
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        written.append(path)
    for p in list(folder.glob("周报_*.md")) + list(folder.glob("月报_*.md")):
        if p not in written:
            p.unlink()
    return written
