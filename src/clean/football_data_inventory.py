"""football-data 列清点和覆盖表（需求文档第 17 节“历史数据下载后怎么整理”第 2 步）。

做什么：
1. 读原始层里每个联赛、每个赛季的 CSV（当前赛季读最新一份快照）。
2. 每一列算“有数据的比赛占多少”（非空比例）。
3. 按数据字典（config/football_data_columns.yaml）把列归到“数据项”，算数据项的覆盖：
   一场比赛算“有”，要求这个数据项的每个部分（例如胜平负的主胜、平、客胜）都至少有一列有值。
   这样早年列名（BbAvH）和后来列名（AvgH）会算成同一个数据项，不会因为改名看起来断档。
4. 检查有没有数据字典里没登记的列、日期写法，和读文件时发现的问题。

结果写进 data/reports/（能随时重新生成，不进 Git）：
- football_data_列覆盖.csv       每个联赛 × 赛季 × 列 的非空比例
- football_data_数据项覆盖.csv   每个联赛 × 赛季 × 数据项 的覆盖比例
- football_data_覆盖汇总.md      给人看的汇总：每个数据项在每个联赛从哪个赛季起有

运行：.venv\\Scripts\\python -m src.clean.football_data_inventory
"""
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.clean.football_data_reader import keep_league_rows, read_football_data_csv
from src.common.config import load_football_data_columns, load_settings, resolve_path

log = logging.getLogger("inventory")


def season_from_code(code: str) -> int:
    """'9394' → 1993，'0405' → 2004，'2627' → 2026（赛季用开始年份表示）。"""
    a = int(code[:2])
    return 1900 + a if a >= 90 else 2000 + a


def season_label(year: int) -> str:
    """2004 → '2004/05'。"""
    return f"{year}/{(year + 1) % 100:02d}"


def season_files(raw_dir: Path, league: str) -> dict[int, Path]:
    """一个联赛的 {赛季: 文件}。已完赛赛季是 E0/2425.csv；当前赛季是 E0/2627/ 文件夹，取日期最新的快照。"""
    folder = raw_dir / "football-data" / league
    out = {season_from_code(p.stem): p for p in folder.glob("*.csv") if p.stem.isdigit()}
    for d in folder.iterdir() if folder.exists() else []:
        snaps = sorted(d.glob("*.csv")) if d.is_dir() else []
        if snaps:
            out[season_from_code(d.name)] = snaps[-1]
    return dict(sorted(out.items()))


def item_parts(columns: dict) -> dict[str, dict[str, list[str]]]:
    """数据字典 → {数据项: {部分: [列名, ...]}}。"""
    items: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for col, info in columns.items():
        items[info["数据项"]][info["部分"] or "值"].append(col)
    return items


def item_coverage(table: pd.DataFrame, parts: dict[str, list[str]]) -> float:
    """一个数据项在一张表里的覆盖比例：每个部分都至少有一列有值的比赛占多少。"""
    if len(table) == 0:
        return 0.0
    ok = pd.Series(True, index=table.index)
    for cols in parts.values():
        present = [c for c in cols if c in table.columns]
        if not present:
            return 0.0
        ok &= table[present].notna().any(axis=1)
    return float(ok.mean())


def date_formats(dates: pd.Series) -> Counter:
    """统计日期写法：dd/mm/yy、dd/mm/yyyy，或其他。"""
    def kind(s):
        if s is None:
            return "空"
        if re.fullmatch(r"\d{2}/\d{2}/\d{2}", s):
            return "dd/mm/yy"
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
            return "dd/mm/yyyy"
        return "其他"
    return Counter(kind(s) for s in dates)


def stable_ranges(cov: dict[int, float], threshold: float) -> str:
    """把 {赛季: 覆盖比例} 变成“有数据的赛季段”，例如 '2000/01–2026/27' 或 '2012/13–2019/20、2021/22–2026/27'。"""
    years = [y for y, v in sorted(cov.items()) if v >= threshold]
    if not years:
        best = max(cov.values(), default=0)
        return "没有" if best == 0 else f"没有（最多 {best:.0%}）"
    runs, start, prev = [], years[0], years[0]
    for y in years[1:]:
        if y != prev + 1:
            runs.append((start, prev))
            start = y
        prev = y
    runs.append((start, prev))
    return "、".join(season_label(a) if a == b else f"{season_label(a)}–{season_label(b)}" for a, b in runs)


def run(settings: dict | None = None):
    """清点全部文件，返回 (列覆盖表, 数据项覆盖表, 问题清单, 没登记的列, 日期写法)。"""
    settings = settings or load_settings()
    raw_dir = resolve_path(settings["paths"]["raw"])
    columns = load_football_data_columns(settings)
    items = item_parts(columns)
    leagues = {**settings["leagues"]["top"], **settings["leagues"]["second"]}

    col_rows, item_rows, issues = [], [], []
    unregistered: dict[str, list[str]] = defaultdict(list)
    dates: dict[str, Counter] = defaultdict(Counter)
    for lg in leagues:
        for season, path in season_files(raw_dir, lg).items():
            r = read_football_data_csv(path)
            t = keep_league_rows(r.table, lg, r.issues)
            if len(t) == 0:
                issues += [f"{lg} {season_label(season)}：{i}" for i in r.issues]
                continue                                   # 整个文件都不是这个联赛的，不算这一季
            where = f"{lg} {season_label(season)}"
            issues += [f"{where}：{i}" for i in r.issues]
            for c in t.columns:
                if c not in columns:
                    unregistered[c].append(where)
                col_rows.append({"联赛": lg, "赛季": season, "列": c, "比赛数": len(t),
                                 "非空比例": round(float(t[c].notna().mean()), 4)})
            for item, parts in items.items():
                item_rows.append({"联赛": lg, "赛季": season, "数据项": item, "比赛数": len(t),
                                  "覆盖比例": round(item_coverage(t, parts), 4)})
            if "Date" in t.columns:
                dates[f"{season_label(season)}"] += date_formats(t["Date"])
    return pd.DataFrame(col_rows), pd.DataFrame(item_rows), issues, dict(unregistered), dict(dates)


# 汇总表里列出的数据项（按第 17 节第 3 步的类别排）。其余数据项都在 数据项覆盖.csv 里。
SUMMARY_ITEMS = [
    ("比赛信息", ["开球时间", "裁判"]),
    ("赛果", ["全场比分", "半场比分"]),
    ("比赛统计", ["射门", "射正", "角球", "犯规", "黄牌", "红牌", "xG"]),
    ("赛前赔率", ["平博 Pinnacle 胜平负 赛前", "Bet365 胜平负 赛前", "多家平均 胜平负 赛前", "多家最高 胜平负 赛前",
              "平博 Pinnacle 大小球 2.5 球 赛前", "多家平均 大小球 2.5 球 赛前",
              "亚盘盘口 赛前", "平博 Pinnacle 亚盘 赛前", "多家平均 亚盘 赛前", "Betfair 交易所 胜平负 赛前"]),
    ("收盘赔率", ["平博 Pinnacle 胜平负 收盘", "Bet365 胜平负 收盘", "多家平均 胜平负 收盘",
              "平博 Pinnacle 大小球 2.5 球 收盘", "多家平均 大小球 2.5 球 收盘",
              "亚盘盘口 收盘", "平博 Pinnacle 亚盘 收盘", "多家平均 亚盘 收盘"]),
]
GRID_ITEMS = ["射门", "开球时间", "平博 Pinnacle 胜平负 赛前", "平博 Pinnacle 胜平负 收盘",
              "多家平均 胜平负 赛前", "多家平均 胜平负 收盘", "亚盘盘口 赛前", "xG"]


def summary_markdown(item_cov: pd.DataFrame, leagues: dict, threshold: float, issues, unregistered, dates) -> str:
    """给人看的汇总（Markdown）。"""
    lines = [f"# football-data 覆盖汇总（自动生成）", "",
             f"格子里是“有数据的赛季段”：这个赛季里至少 {threshold:.0%} 的比赛有这项数据才算有。"
             "2026/27 是进行中的赛季，算到最新快照。", ""]
    for group_name, group in (("五大联赛", list(leagues)[:5]), ("次级联赛", list(leagues)[5:])):
        lines += [f"## {group_name}", "", "| 类别 | 数据项 | " + " | ".join(leagues[g] for g in group) + " |",
                  "|---|---|" + "---|" * len(group)]
        for cat, names in SUMMARY_ITEMS:
            for it in names:
                cells = []
                for lg in group:
                    s = item_cov[(item_cov["数据项"] == it) & (item_cov["联赛"] == lg)]
                    cells.append(stable_ranges(dict(zip(s["赛季"], s["覆盖比例"])), threshold))
                lines.append(f"| {cat} | {it} | " + " | ".join(cells) + " |")
        lines.append("")
    first = list(leagues)[0]
    lines += [f"## {leagues[first]}逐季（有数据的比赛占多少）", "",
              "| 赛季 | 场数 | " + " | ".join(GRID_ITEMS) + " |", "|---|---|" + "---|" * len(GRID_ITEMS)]
    s = item_cov[item_cov["联赛"] == first]
    for season in sorted(s["赛季"].unique()):
        row = s[s["赛季"] == season].set_index("数据项")
        cells = [f"{row.loc[it, '覆盖比例']:.0%}" if it in row.index else "" for it in GRID_ITEMS]
        lines.append(f"| {season_label(season)} | {int(row['比赛数'].iloc[0])} | " + " | ".join(cells) + " |")
    lines += ["", "## 日期写法", ""]
    lines += [f"- {k}：" + "，".join(f"{f} {n} 场" for f, n in v.items()) for k, v in dates.items() if len(v) > 1 or "dd/mm/yy" not in v]
    lines += ["", "## 读文件时处理过的问题", ""] + [f"- {i}" for i in issues or ["没有"]]
    lines += ["", "## 数据字典里没登记的列", ""]
    lines += [f"- {c}：{'、'.join(w[:3])}{' 等' if len(w) > 3 else ''}" for c, w in unregistered.items()] or ["- 没有"]
    return "\n".join(lines) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    out_dir = resolve_path(settings["paths"]["reports"])
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = settings["inventory"]["full_coverage"]
    leagues = {**settings["leagues"]["top"], **settings["leagues"]["second"]}

    log.info("开始清点 football-data 的原始文件……")
    col_cov, item_cov, issues, unregistered, dates = run(settings)
    col_cov.to_csv(out_dir / "football_data_列覆盖.csv", index=False, encoding="utf-8-sig")
    item_cov.to_csv(out_dir / "football_data_数据项覆盖.csv", index=False, encoding="utf-8-sig")
    (out_dir / "football_data_覆盖汇总.md").write_text(
        summary_markdown(item_cov, leagues, threshold, issues, unregistered, dates), encoding="utf-8")

    files = col_cov.groupby(["联赛", "赛季"])["比赛数"].first()
    log.info("清点了 %d 个文件、%d 场比赛、%d 个不同的列名。", len(files), files.sum(), col_cov["列"].nunique())
    log.info("读文件时处理过的问题 %d 条；数据字典里没登记的列 %d 个%s。", len(issues), len(unregistered),
             "：" + "、".join(unregistered) if unregistered else "")
    log.info("结果在 %s", out_dir)


if __name__ == "__main__":
    main()
