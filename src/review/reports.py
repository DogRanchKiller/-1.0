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

from src.evaluate.live_review_metrics import build_live_metrics, matched_comparisons, score_summary
from src.review.match_review import BEIJING, RESULT_INDEX
from src.review.reconcile import EVIDENCE_COLUMNS, SOURCE_CLOSE, SOURCE_MODEL, review_closing_coverage

METRICS = [("logloss_1x2", "胜平负 log loss"), ("brier_1x2", "Brier"), ("rps_1x2", "RPS"),
           ("logloss_score", "比分 log loss"), ("logloss_over25", "大 2.5 球 log loss"), ("logloss_btts", "双方进球 log loss")]
EXTRA_NAMES = {"confidence": "保存的信心等级", "competition": "联赛", "tag": "标签", "tag_value": "标签取值"}
TAG_NAMES = {"home_after_euro": "主队刚踢欧战", "away_after_euro": "客队刚踢欧战"}
COVERAGE_REASONS = {"no_record": "没有候选记录", "missing_review_evidence": "旧复盘没有证据明细",
                    "not_ou_2_5": "不是2.5球盘口", "fetched_after_cutoff": "下载晚于复盘截止",
                    "not_verified_close": "赛前观察价未证实为收盘", "not_close": "来源未标收盘",
                    "missing_provenance": "缺少来源凭据", "ambiguous_provenance": "来源凭据不唯一",
                    "missing_raw_file": "缺少原始文件路径", "missing_fetched_at_utc": "缺少下载时间",
                    "missing_source_key": "缺少来源比赛编号", "invalid_odds": "赔率不完整或无效",
                    "conflicting_close_prices": "同一标杆的收盘价格冲突",
                    "unregistered_close_source": "来源未登记收盘语义", "not_yet_closed": "尚未到收盘时点",
                    "in_play": "开球后的价格", "not_observed_pregame": "未在赛前观察到"}


def extra_text(key, value) -> str:
    """额外分组的中文显示；内部列名和取值保留原样。"""
    if key == "tag":
        return TAG_NAMES.get(value, str(value))
    if key == "tag_value" and isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


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


def metric_text(metric: dict, digits: int = 4) -> str:
    """均值旁边写本指标有效分母；没有分数也明确0场有效，不借组总场数充数。"""
    valid = metric["n_matches"]
    total = valid + metric["missing_matches"]
    return f"{fmt(metric['mean'], digits)}（有效{valid}/{total}）"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def group_cells(group: dict) -> list[str]:
    """模型身份、预测时点、归档状态分别展示，空值不补成当前模型。"""
    return [str(group.get(k) or "未知") for k in ("model", "model_version", "kind", "source", "archive_status")]


def live_metrics(sc: pd.DataFrame, settings: dict, labels=None) -> dict:
    """任务、报告、网页共用统计模块及配置，不在展示层另算一套指标。"""
    return build_live_metrics(sc, benchmark_books=settings["review"].get("close_bookmakers", []),
                              n_bins=int(settings.get("calc", {}).get("calibration_bins", 10)),
                              min_matches=int(settings["review"]["reports"]["min_matches"]), labels=labels)


def coverage_for_reviews(reviews: list[dict], settings: dict, match_ids=None) -> dict:
    """报告与网页只从这批单场证据汇总；旧JSON缺明细保持未知，不用其他比赛补齐。"""
    mids = list(dict.fromkeys(match_ids if match_ids is not None else [r["match_id"] for r in reviews]))
    books = list(settings["review"].get("close_bookmakers", []))
    if not books:
        return {"match_count": len(mids), "by_bookmaker_market": []}
    by_match = {r["match_id"]: r for r in reviews}
    rows = []
    for mid in mids:
        review = by_match.get(mid) or {}
        if "收盘证据" in review:
            rows.extend(review["收盘证据"])
        else:
            rows.extend({"match_id": mid, "bookmaker": book, "market": market, "qualified": False,
                         "reasons": ("missing_review_evidence",)} for book in books for market in ("1x2", "ou"))
    evidence = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
    return review_closing_coverage(evidence, mids, books)


def coverage_cells(coverage: dict) -> list[list]:
    rows = []
    for group in coverage["by_bookmaker_market"]:
        reasons = "、".join(f"{COVERAGE_REASONS.get(reason, reason)} {count}场"
                           for reason, count in group["absence_reasons"].items()) or "—"
        rows.append([group["bookmaker"], "胜平负" if group["market"] == "1x2" else "大2.5球",
                     group["match_count"], group["qualified_match_count"],
                     group["match_count"] - group["qualified_match_count"],
                     "—" if group["coverage"] is None else f"{group['coverage']:.1%}", reasons])
    return rows


def average_table(sc: pd.DataFrame, kinds: list[str], summaries=None) -> list[str]:
    """每个模型身份、预测时点、实际归档状态一行；独立标杆在同场比较中展示。"""
    if sc.empty:
        return ["（没有数据）"]
    summaries = score_summary(sc, min_matches=0) if summaries is None else summaries
    rows = [group_cells(item["group"]) + [item["n_matches"]]
            + [metric_text(item["metrics"][c]) for c, _ in METRICS] + [item["note"]] for item in summaries]
    return md_table(["模型", "模型版本", "预测时点", "概率来源", "归档状态", "场数"]
                    + [n for _, n in METRICS] + ["证据说明"], rows)


def official_entries(review: dict):
    """每个模型版本各选自己的正式存档；旧 JSON 只能回读明确选定的单条。"""
    if "各模型版本复盘" in review:
        for item in review["各模型版本复盘"]:
            entry = next((e for e in review["各版本"] if e["prediction_id"] == item["正式prediction_id"]), None)
            if entry is not None:
                yield entry, item
    else:
        entry = next((e for e in review["各版本"] if e["版本"] == review["正式版本"]), None)
        if entry is not None:
            yield entry, review


def official_rows(sc: pd.DataFrame, reviews: list[dict]) -> pd.DataFrame:
    """每场每个模型版本的正式行，加上独立收盘标杆；不以时点名合并身份。"""
    pids = {e["prediction_id"] for r in reviews for e, _ in official_entries(r)}
    mids = {r["match_id"] for r in reviews}
    keep = sc[sc["match_id"].isin(mids) & ((sc["source"] == SOURCE_CLOSE) | sc["prediction_id"].isin(pids))]
    return keep.assign(who=keep["source"])


def pair_table(off: pd.DataFrame, comparisons=None) -> list[str]:
    """每个模型组与每个独立标杆逐场比较；零匹配保留为空，不写成零分。"""
    comparisons = matched_comparisons(off, benchmark_books=[], min_matches=0) if comparisons is None else comparisons
    rows = []
    for item in comparisons:
        benchmark = item["benchmark"]
        for metric, values in item["metrics"].items():
            rows.append(group_cells(item["group"]) + [benchmark["benchmark_source"], benchmark["bookmaker"],
                        benchmark["market"], benchmark["timing"], metric, item["n_model"], item["n_benchmark"],
                        item["n_model"] - values["n_matches"], values["n_matches"],
                        fmt(values["model_mean"]), fmt(values["benchmark_mean"]), fmt(values["mean_delta"]),
                        f"{values['model_better']} : {values['benchmark_better']} : {values['ties']}" if values["n_matches"] else "—",
                        item["note"]])
    return md_table(["模型", "模型版本", "预测时点", "概率来源", "归档状态", "标杆来源", "公司", "玩法", "标杆时点",
                     "指标", "模型场数", "标杆覆盖", "缺失场数", "同场数", "模型均值", "标杆均值", "差值（模型减标杆）", "模型较好 : 标杆较好 : 相同", "证据说明"], rows)


def miss_table(reviews: list[dict], n: int) -> list[str]:
    """最大的几次失误：正式版本模型的胜平负 log loss 最大的几场。"""
    items = []
    for r in reviews:
        for entry, official in official_entries(r):
            ll = entry[SOURCE_MODEL]["logloss_1x2"]
            res = RESULT_INDEX[{"主胜": "H", "平局": "D", "客胜": "A"}[r["实际"]["结果"]]]
            items.append((ll, r, entry, official, entry[SOURCE_MODEL]["概率"][res]))
    items.sort(key=lambda x: -x[0])
    rows = [[r["比赛日期"], r["联赛"], f"{r['主队中文'] or r['主队']} {r['实际']['比分'][0]}-{r['实际']['比分'][1]} "
             f"{r['客队中文'] or r['客队']}", entry.get("model", "未知"), entry.get("model_version", "未知"),
             entry["版本"], entry.get("archive_status", "unknown"), f"{p:.0%}", fmt(ll), official["误差归类"]["类型"],
             r["实际"].get("来源说明", "") or "—"] for ll, r, entry, official, p in items[:n]]
    if not rows:
        return ["（没有数据）"]
    return md_table(["日期", "联赛", "比赛", "模型", "模型版本", "预测时点", "归档状态", "模型给实际结果", "胜平负 log loss", "误差归类", "赛果来源说明"], rows)


def error_table(reviews: list[dict]) -> list[str]:
    counts = pd.Series([(entry.get("model", "未知"), entry.get("model_version", "未知"), entry["版本"],
                         entry.get("archive_status", "unknown"), official["误差归类"]["类型"])
                        for r in reviews for entry, official in official_entries(r)], dtype=object).value_counts()
    if counts.empty:
        return ["（没有数据）"]
    return md_table(["模型", "模型版本", "预测时点", "归档状态", "类型", "场数"], [[*k, int(v)] for k, v in counts.items()])


def build_report(kind: str, key: str, reviews: list[dict], sc: pd.DataFrame, settings: dict, now_utc) -> str:
    """一份周报 / 月报的 Markdown 文字。kind：“周”或“月”。"""
    cfg = settings["review"]["reports"]
    kinds = list(settings["review"]["kinds"])
    metrics = live_metrics(sc, settings, {r["match_id"]: r.get("自动标签") for r in reviews})
    official_metrics = live_metrics(official_rows(sc, reviews), settings)
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
    source_missing = sum(r.get("实际", {}).get("source_missing") in (True, 1) for r in reviews)
    if source_missing:
        lines.append(f"- {source_missing}场来源缺失；赛果为最后已知资料。")
    if n < int(cfg["min_matches"]):
        lines.append(f"- **样本太少，不能下结论**（{n} 场，少于 {cfg['min_matches']} 场）。下面的数字只用来发现异常，"
                     "例如某个联赛突然变差、数据漏抓。")
    lines += ["", "分数都是越小越好。log loss 看概率给得准不准，Brier、RPS 是另两种算法（RPS 考虑主胜、平、客胜的先后顺序）。", "",
              "## 各版本平均分数", ""] + average_table(sc, kinds, metrics["groups"])
    lines += ["", "## 同场比较（正式版本）", "",
              "每场每个模型版本分别按 T-1 → T-6 → T-24 → 预览版选正式存档；准点、晚出、改期、预览、未知分别统计。"
              "两家收盘标杆独立比较，只比同场都有的指标。收盘指来源明确标记的字段，不表示恰好在开球瞬间抓到。", ""]
    lines += pair_table(official_rows(sc, reviews), official_metrics["comparisons"])
    lines += ["", "## 收盘证据覆盖", "", "分母为本报告复盘比赛集合；每家公司、玩法独立列出，缺失原因可以重叠。", ""]
    lines += md_table(["公司", "玩法", "复盘场数", "标杆覆盖", "缺失场数", "覆盖率", "缺失原因"],
                      coverage_cells(coverage_for_reviews(reviews, settings)))
    calibration_rows = [group_cells(item["group"]) + [outcome["outcome"], item["n_matches"], fmt(outcome["ece"])]
                        for item in metrics["calibration"] for outcome in item["outcomes"]]
    lines += ["", "## 概率校准", "", "主胜、平局、客胜分别校准；每项样本数均为比赛数。", ""]
    lines += md_table(["模型", "模型版本", "预测时点", "概率来源", "归档状态", "结果", "场数", "校准误差 ECE"], calibration_rows)
    for title, key, extra in (("按保存的信心等级", "confidence", ["confidence"]),
                              ("按联赛", "competitions", ["competition"]),
                              ("按比赛标签", "tags", ["tag", "tag_value"])):
        rows = [group_cells(item["group"]) + [extra_text(x, item["group"][x]) for x in extra] + [item["n_matches"],
                metric_text(item["metrics"]["logloss_1x2"]), metric_text(item["metrics"]["brier_1x2"]), item["note"]] for item in metrics[key]]
        lines += ["", f"## {title}", ""] + md_table(["模型", "模型版本", "预测时点", "概率来源", "归档状态", *[EXTRA_NAMES[x] for x in extra],
                                                     "场数", "胜平负 log loss", "Brier", "证据说明"], rows)
        if key == "confidence":
            rows = [group_cells(item["group"]) + [item["group"]["confidence"], outcome["outcome"],
                    item["calibration"]["n_matches"], fmt(outcome["ece"])]
                    for item in metrics["confidence"] for outcome in item["calibration"]["outcomes"]]
            lines += ["", "### 按信心等级校准", ""] + md_table(
                ["模型", "模型版本", "预测时点", "概率来源", "归档状态", "保存的信心等级", "结果", "场数", "校准误差 ECE"], rows)
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
