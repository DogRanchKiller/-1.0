"""懂球帝历史回补的进度。

只看本机已经存下的文件和抓取记录，不联网，随时可以运行，不会影响正在跑的回补程序。

运行：.venv\\Scripts\\python -m src.jobs.dongqiudi_progress
或者双击项目文件夹里的“查看懂球帝回补进度.bat”。

每个联赛每个赛季一行：
- 应有：已结束的赛季，看懂球帝赛季页上一共几场；进行中的赛季还不知道，写“进行中”。
- 已抓：已经存下赔率文件的场数（加上网站说没有这场的）。
- 有赔率：存下的文件里，至少有一家公司赔率的场数。
结果同时写进 data/reports/dongqiudi_回补进度.md。
"""
import gzip
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.clean.dongqiudi_reader import BEIJING, parse_history, parse_season_page, read_json
from src.collect.dongqiudi import SOURCE, current_season
from src.common.config import load_settings, resolve_path
from src.common.db import connect

RUNNING_MINUTES = 3  # 最近几分钟内还有新的抓取记录，就认为回补程序还在跑


def season_ids(raw: Path, league: str, season: int) -> set[int] | None:
    """已结束赛季：赛季页上的全部比赛编号。赛季页还没抓到就返回 None。"""
    page = raw / "season" / league / f"{season}.html.gz"
    if not page.exists():
        return None
    return {g.match_id for g in parse_season_page(gzip.decompress(page.read_bytes()).decode("utf-8"))}


def listed_seasons(raw: Path, league: str) -> set[int] | None:
    """懂球帝历届赛季列表里有哪些赛季（用最新一份）。还没抓过列表就返回 None。"""
    files = sorted(f for f in (raw / "history" / league).glob("*.html.gz") if ".suspicious." not in f.name)
    return set(parse_history(gzip.decompress(files[-1].read_bytes()).decode("utf-8"))) if files else None


def has_odds(path: Path) -> bool:
    d = read_json(path)
    return bool(d.get("euro") or d.get("asia") or d.get("size"))


def not_found_ids(con) -> set[int]:
    """抓取记录里“网站没有”的比赛编号（这种比赛没有文件，只有记录）。"""
    ids = set()
    for (url,) in con.execute("select url from fetch_log where source = ? and data_type = 'odds_final' "
                              "and status = 'not_found'", (SOURCE,)):
        ids.add(int(url.split("/index/")[1].split("?")[0]))
    return ids


def season_rows(con, raw: Path, leagues: list[str], first: int, now: datetime) -> list[dict]:
    this = current_season(now)
    missing = not_found_ids(con)
    rows = []
    for lg in leagues:
        listed = listed_seasons(raw, lg)
        for y in range(this, first - 1, -1):  # 和回补程序同一个顺序：从近往远
            folder = raw / "odds_final" / lg / str(y)
            files = [f for f in folder.glob("*.json.gz") if ".suspicious." not in f.name] if folder.exists() else []
            got = {int(f.name.split(".")[0]) for f in files}
            with_odds = sum(has_odds(f) for f in files)
            if y == this:
                total, done = None, len(got)
            else:
                ids = season_ids(raw, lg, y)
                total = len(ids) if ids is not None else None
                done = len((got | missing) & ids) if ids is not None else len(got)
            rows.append({"联赛": lg, "赛季": f"{y}/{(y + 1) % 100:02d}", "进行中": y == this,
                         "懂球帝没有": y != this and listed is not None and y not in listed,
                         "应有": total, "已抓": done, "有赔率": with_odds})
    return rows


def run_state(con, now: datetime) -> tuple[str, datetime | None]:
    """回补程序现在的状态：在跑 / 被网站拒绝后停了 / 没在跑。"""
    last = con.execute("select fetched_at_utc, status, http_status from fetch_log where source = ? "
                       "order by id desc limit 1", (SOURCE,)).fetchone()
    if last is None:
        return "还没开始", None
    t = datetime.fromisoformat(last[0])
    if last[1] == "failed" and last[2] in (403, 429):
        return "已停止：懂球帝拒绝访问（程序按规矩停下，不会绕过）", t
    last_backfill = con.execute("select max(fetched_at_utc) from fetch_log where source = ? "
                                "and data_type = 'odds_final'", (SOURCE,)).fetchone()[0]
    if last_backfill and now - datetime.fromisoformat(last_backfill) < timedelta(minutes=RUNNING_MINUTES):
        return "正在运行", t
    return "没在运行（已经补完，或者中途关过机）", t


def report(rows: list[dict], state: str, last: datetime | None, interval: float, now: datetime) -> str:
    """把进度写成一段中文（Markdown 表格），屏幕上看和存文件用同一份。"""
    left = sum(r["应有"] - r["已抓"] for r in rows if r["应有"] is not None)
    unknown = [r for r in rows if r["应有"] is None and not r["进行中"] and not r["懂球帝没有"]]  # 赛季页还没抓到，场数不知道
    done = sum(r["已抓"] for r in rows)
    lines = [
        "# 懂球帝历史回补进度",
        "",
        f"查看时间：{now.astimezone(BEIJING):%Y-%m-%d %H:%M}（北京时间）",
        f"状态：{state}",
    ]
    if last:
        lines.append(f"最后一次抓取：{last.astimezone(BEIJING):%Y-%m-%d %H:%M:%S}")
    lines.append(f"已抓 {done} 场；已结束赛季还剩 {left} 场"
                 + (f"，另有 {len(unknown)} 个赛季还没开始（场数要等抓到赛季页才知道）" if unknown else ""))
    if left and state == "正在运行":
        finish = now + timedelta(seconds=left * interval)
        lines.append(f"按每 {interval:g} 秒一场估计，还要约 {left * interval / 3600:.1f} 小时，"
                     f"大约 {finish.astimezone(BEIJING):%m-%d %H:%M} 补完" + ("（不算没开始的赛季）" if unknown else ""))
    lines += ["", "| 联赛 | 赛季 | 应有 | 已抓 | 有赔率 | 进度 |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["进行中"]:
            total, pct = "进行中", "—"
        elif r["懂球帝没有"]:
            total, pct = "—", "懂球帝没有这个赛季"
        elif r["应有"] is None:
            total, pct = "?", "还没开始"
        else:
            total = str(r["应有"])
            pct = "完成" if r["已抓"] >= r["应有"] else f"{r['已抓'] / r['应有']:.0%}" if r["应有"] else "—"
        lines.append(f"| {r['联赛']} | {r['赛季']} | {total} | {r['已抓']} | {r['有赔率']} | {pct} |")
    lines += ["", "说明：进行中的赛季只算已经踢完的比赛；“有赔率”是存下的文件里至少有一家公司赔率的场数。"]
    return "\n".join(lines) + "\n"


def main():
    s = load_settings()
    cfg = s["dongqiudi"]
    now = datetime.now(timezone.utc)
    con = connect()
    try:
        rows = season_rows(con, resolve_path(s["paths"]["raw"]) / SOURCE, list(cfg["league_ids"]),
                           cfg["backfill_first_season"], now)
        state, last = run_state(con, now)
    finally:
        con.close()
    text = report(rows, state, last, cfg["min_interval_seconds"], now)
    out = resolve_path(s["paths"]["reports"]) / "dongqiudi_回补进度.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    sys.stdout.write(text + f"\n（这份结果也存在 {out}）\n")


if __name__ == "__main__":
    main()
