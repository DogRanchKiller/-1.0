"""数据状态页 + 每日抓取报告 + 出问题时弹窗提醒（需求文档第 6.2、8 节）。

为什么要有：采集程序每小时自己在后台跑，网站改版、被拦、电脑关机，都会让数据悄悄断掉，而且断掉的那段找不回来。
这一页把“每个采集程序上次成功是什么时候、最近有没有失败”放在一起看，有问题就弹窗。

做什么：
1. 只读抓取记录（fetch_log）、备份文件夹、数据库、懂球帝导入报告和 teams.yaml，不联网，不改任何数据。
   懂球帝导入报告（data/reports/dongqiudi_导入_队名对照.csv、dongqiudi_导入_没对上.csv）是另一个程序写的，
   以后列可能会变：文件不在、列名不对时，页面上写“读不到”，不让整页出错。
2. 生成网页 data/reports/status.html（双击“查看数据状态.bat”打开；页面开着时每 10 分钟自己刷新）。
   只是一个本机文件，没有开网站服务，别的电脑访问不到。
3. 当天的报告另存一份 data/reports/抓取日报/日期.md（一天里多次运行就覆盖，留下的是当天最后一次的）。
4. 出现新的问题时弹一个 Windows 提示框。同一个问题只提醒一次，问题消失后再出现会再提醒。

运行：.venv\\Scripts\\python -m src.jobs.status_page
由 Windows 定时任务“足球预测_数据状态”每小时自动运行。
"""
import csv
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from src.clean.dongqiudi_reader import BEIJING
from src.common.config import load_settings, resolve_path
from src.common.db import connect
from src.jobs.dongqiudi_progress import run_state, season_rows

LEVEL_NAMES = {"ok": "正常", "warn": "注意", "error": "出错"}
DATA_TYPE_NAMES = {"fixtures": "懂球帝 赛程", "odds_live": "懂球帝 赛前赔率快照", "odds_final": "懂球帝 赛后最终赔率",
                   "history_page": "懂球帝 历届赛季列表", "season_page": "懂球帝 赛季页",
                   "match_detail": "懂球帝 比赛详情页（xG）",
                   "league_csv": "football-data 联赛文件", "notes": "football-data 说明文件",
                   "understat_shots": "Understat 射门明细（GitHub 镜像）"}
STATUS_NAMES = {"saved": "已保存", "unchanged": "没变化", "not_found": "网站没有", "suspicious": "内容可疑",
                "failed": "失败"}


@dataclass
class Check:
    key: str     # 问题的编号，用来记“这个问题提醒过没有”
    name: str    # 检查项
    level: str   # ok 正常 / warn 注意 / error 出错
    text: str    # 给人看的说明


@dataclass
class Listing:
    """页面下半部分的一张清单（待确认的队名、没对上的比赛）。"""
    title: str
    columns: list[str]       # 表头
    rows: list[list]         # 只放要显示的前几行
    total: int               # 一共有几条（可能比 rows 多）
    unit: str                # “个”“场”
    errors: list[str]        # 读不到的文件和原因，页面上照写


def bj(t: datetime) -> str:
    """标准时间 → 北京时间文字。"""
    return t.astimezone(BEIJING).strftime("%m-%d %H:%M")


def ago(t: datetime, now: datetime) -> str:
    """多久以前，例如“25 分钟前”“3.5 小时前”。"""
    minutes = (now - t).total_seconds() / 60
    if minutes < 90:
        return f"{minutes:.0f} 分钟前"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} 小时前"
    return f"{minutes / 60 / 24:.1f} 天前"


def last_time(con, sql: str, args=()) -> datetime | None:
    row = con.execute(sql, args).fetchone()
    return datetime.fromisoformat(row[0]) if row and row[0] else None


# ---------- 各项检查 ----------
def check_dongqiudi_hourly(con, now: datetime, cfg: dict) -> Check:
    """每小时的懂球帝任务：看赛程最近一次请求成功（已保存 / 没变化）是什么时候。"""
    t = last_time(con, "select max(fetched_at_utc) from fetch_log where source = 'dongqiudi' and data_type = 'fixtures'"
                       " and status in ('saved', 'unchanged')")
    name = "懂球帝 每小时赔率任务"
    if t is None:
        return Check("dqd_hourly_never", name, "error", "还没有成功过")
    hours = (now - t).total_seconds() / 3600
    if hours > cfg["hourly_stale_hours"]:
        return Check("dqd_hourly_stale", name, "error",
                     f"上次成功是 {bj(t)}（{ago(t, now)}）。超过 {cfg['hourly_stale_hours']} 小时没成功："
                     "可能是电脑关过机或睡眠，或者程序出错了，看 logs/dongqiudi.log")
    return Check("dqd_hourly", name, "ok", f"上次成功 {bj(t)}（{ago(t, now)}）")


def check_blocked(con, now: datetime) -> Check:
    """最近 24 小时有没有被网站拒绝（403、429）。被拒绝时采集程序会自己停下，要人来看。"""
    since = (now - timedelta(hours=24)).isoformat(timespec="seconds")
    row = con.execute("select source, fetched_at_utc, http_status from fetch_log where status = 'failed'"
                      " and http_status in (403, 429) and fetched_at_utc >= ? order by id desc limit 1",
                      (since,)).fetchone()
    name = "有没有被网站拒绝"
    if row:
        return Check(f"blocked_{row[0]}", name, "error",
                     f"{row[0]} 在 {bj(datetime.fromisoformat(row[1]))} 返回 {row[2]}，程序已按规矩停下，不会绕过。"
                     "告诉 Claude 一起看原因")
    return Check("blocked", name, "ok", "最近 24 小时没有")


def check_failures(con, now: datetime, cfg: dict) -> Check:
    """最近 24 小时的失败和内容可疑（网站拒绝的除外，上一项单独看）。偶尔一两次是网络抖动，多了要看。"""
    since = (now - timedelta(hours=24)).isoformat(timespec="seconds")
    failed, suspicious = con.execute(
        "select sum(status = 'failed' and coalesce(http_status, 0) not in (403, 429)), sum(status = 'suspicious')"
        " from fetch_log where fetched_at_utc >= ?", (since,)).fetchone()
    failed, suspicious = failed or 0, suspicious or 0
    name = "失败和内容可疑"
    text = f"最近 24 小时：失败 {failed} 次，内容可疑 {suspicious} 次"
    if suspicious or failed > cfg["max_failures_24h"]:
        return Check("failures", name, "warn", text + "。下面“最近的问题”里有网址；内容可疑的文件单独存着，不会被用到")
    return Check("failures", name, "ok", text)


def check_backfill(con, raw: Path, dq_cfg: dict, now: datetime) -> tuple[Check, list[dict]]:
    """历史回补：还剩多少、程序还在不在跑（复用 dongqiudi_progress 的统计）。"""
    rows = season_rows(con, raw, list(dq_cfg["league_ids"]), dq_cfg["backfill_first_season"], now)
    state, _ = run_state(con, now)
    left = sum(r["应有"] - r["已抓"] for r in rows if r["应有"] is not None)
    unknown = sum(r["应有"] is None and not r["进行中"] and not r["懂球帝没有"] for r in rows)
    name = "懂球帝 历史回补"
    if state.startswith("已停止"):
        return Check("backfill_blocked", name, "error", state), rows
    if left == 0 and unknown == 0:
        return Check("backfill", name, "ok", "已经补完"), rows
    if state == "正在运行":
        hours = left * dq_cfg["min_interval_seconds"] / 3600
        return Check("backfill", name, "ok", f"正在运行，还剩 {left} 场，约 {hours:.1f} 小时"), rows
    return Check("backfill_stopped", name, "warn",
                 f"没在运行，但还剩 {left} 场" + (f"（另有 {unknown} 个赛季没开始）" if unknown else "")
                 + "。多半是中途关过机，告诉 Claude 重新启动，会接着补"), rows


def check_backup(backup_root: Path, now: datetime, cfg: dict) -> Check:
    """每日备份：看备份文件夹里最新一份数据库备份是什么时候写的。"""
    files = sorted((backup_root / "database").glob("football_*.db"), key=lambda f: f.stat().st_mtime)
    name = "每日备份"
    if not files:
        return Check("backup_none", name, "error", f"{backup_root} 里还没有数据库备份")
    t = datetime.fromtimestamp(files[-1].stat().st_mtime, timezone.utc)
    size = files[-1].stat().st_size / 1024 / 1024
    text = f"最新一份 {files[-1].name}（{size:.0f} MB），写于 {bj(t)}（{ago(t, now)}）"
    if (now - t).total_seconds() / 3600 > cfg["backup_stale_hours"]:
        return Check("backup_stale", name, "warn", text + "。超过一天多没备份：可能是凌晨 3:30 电脑没开，看 logs/backup.log")
    return Check("backup", name, "ok", text)


def check_database(con) -> Check:
    """数据库：比赛有多少场、最近一场是哪天（football-data 导入的已结束比赛）。"""
    n, last = con.execute("select count(*), max(match_date) from matches").fetchone()
    return Check("database", "数据库", "ok", f"比赛 {n:,} 场，最近一场 {last or '—'}")


def check_football_data(con, now: datetime) -> Check:
    """football-data 联赛文件：手动（让 Claude）运行，只显示上次下载时间，不报警。"""
    t = last_time(con, "select max(fetched_at_utc) from fetch_log where source = 'football-data'"
                       " and status in ('saved', 'unchanged')")
    text = f"上次下载 {bj(t)}（{ago(t, now)}）" if t else "还没下载过"
    return Check("football_data", "football-data 联赛文件", "ok", text + "。这个不定时自动下载，需要新数据时让 Claude 运行")


# ---------- 懂球帝导入留下的问题（待确认队名、没对上的比赛） ----------
def read_report_csv(path: Path, need: list[str]) -> tuple[list[dict] | None, str | None]:
    """读导入程序写的 CSV。返回（每行一个字典, None）；读不到返回（None, 原因）。

    这些 CSV 是 src/clean/dongqiudi_import.py 写的，以后列可能会改。所以这里不信任文件：
    文件不在、打不开、编码不对、缺了要用的列，都只返回原因，由页面写“读不到”。
    """
    try:
        # utf-8-sig：导入程序写的文件开头带 BOM（让 Excel 认出中文），这样读时会去掉
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in need if c not in (reader.fieldnames or [])]
            if missing:
                return None, f"{path.name} 里没有这些列：{'、'.join(missing)}（导入程序的报告格式可能改了）"
            return list(reader), None
    except FileNotFoundError:
        return None, f"{path.name} 不存在（懂球帝导入还没运行过，或者报告改名了）"
    except (OSError, UnicodeDecodeError, csv.Error) as err:
        return None, f"{path.name} 打不开：{err}"


def season_text(s: str) -> str:
    """赛季开始年份 → “2016/17”。不是年份就原样显示。"""
    return f"{s}/{(int(s) + 1) % 100:02d}" if s.isdigit() else s


def cell(row: dict, col: str) -> str:
    """取 CSV 一行里的一格。行比表头短时那一格是 None，当作空文字。"""
    return str(row.get(col) or "").strip()


def blank_chinese_names(path: Path) -> tuple[list[dict] | None, str | None]:
    """teams.yaml 里“中文”还空着的球队。返回（[{英文, 国家}], None）；读不到返回（None, 原因）。"""
    try:
        teams = yaml.safe_load(path.read_text(encoding="utf-8"))["球队"]
        return [{"英文": str(t["英文"]), "国家": str(t.get("国家") or "")} for t in teams
                if not str(t.get("中文") or "").strip()], None
    except FileNotFoundError:
        return None, f"{path.name} 不存在"
    except (OSError, UnicodeDecodeError, yaml.YAMLError, KeyError, TypeError) as err:
        return None, f"{path.name} 读不懂：{err!r}"


def recent_top_teams(con, dq_cfg: dict) -> set[str]:
    """懂球帝要抓的联赛里，回补起始赛季以来出场过的球队。这些球队没有中文名，网页上就只能显示英文。"""
    leagues = list(dq_cfg["league_ids"])
    ph = ",".join("?" * len(leagues))
    args = [*leagues, int(dq_cfg["backfill_first_season"])]
    try:
        return {t for (t,) in con.execute(
            f"select home_team from matches where competition in ({ph}) and season >= ?"
            f" union select away_team from matches where competition in ({ph}) and season >= ?", args + args)}
    except Exception:   # 数据库还没有这些列（例如测试用的小数据库）：当作没有，不影响别的检查
        return set()


def check_team_names(con, reports: Path, teams_yaml: Path, settings: dict) -> tuple[Check, Listing]:
    """待确认的队名，两个来源：
    1. 懂球帝导入的队名对照里“结果”不是“定下”的：这些中文名没对上我们的球队，它们的比赛没导入。
    2. teams.yaml 里中文名还空着的：大多是次级联赛、以前的球队，网页暂时用不到；
       只有最近几季在五大联赛出场的（例如刚升级的）才要尽快补，超过界限才标出来。
    """
    cfg, limit = settings["status"], settings["status"]["list_rows"]
    name = "待确认的队名"
    rows, errors, parts = [], [], []
    # 1. 懂球帝队名对照里没定下来的
    table, err_csv = read_report_csv(reports / "dongqiudi_导入_队名对照.csv", ["中文名", "英文名", "得票", "结果"])
    pending = [] if err_csv else [r for r in table if cell(r, "结果") != "定下"]
    if err_csv:
        errors.append(err_csv)
    else:
        parts.append(f"懂球帝队名对照里没定下来的 {len(pending)} 个")
        rows += [["懂球帝队名对照", cell(r, "中文名"),
                  f"最像 {cell(r, '英文名')}（{cell(r, '得票')} 票），{cell(r, '结果') or '没写结果'}"] for r in pending]
    # 2. teams.yaml 里中文名空着的；最近在五大联赛出场的排前面
    blank, err_yaml = blank_chinese_names(teams_yaml)
    relevant = []
    if err_yaml:
        errors.append(err_yaml)
    else:
        used = recent_top_teams(con, settings["dongqiudi"])
        since = int(settings["dongqiudi"]["backfill_first_season"])
        relevant = [b for b in blank if b["英文"] in used]
        parts.append(f"teams.yaml 中文名空着 {len(blank)} 个（其中 {since}/{(since + 1) % 100:02d} 起在五大联赛出场的"
                     f" {len(relevant)} 个）")
        for b in sorted(blank, key=lambda b: b["英文"] not in used):
            note = "最近在五大联赛出场过，要补" if b["英文"] in used else "最近没在五大联赛出场，暂时用不到"
            rows.append(["teams.yaml 中文名空着", b["英文"], f"{b['国家']}，{note}"])

    text = "；".join(parts)
    listing = Listing(name, ["来源", "队名", "说明"], rows[:limit], len(rows), "个", errors)
    # 读不到和数量超了用不同的编号：从一种情况变成另一种时，会重新弹窗
    if errors:
        return Check("team_names_unreadable", name, "warn",
                     (text + "。" if text else "") + "读不到：" + "；".join(errors)), listing
    if len(pending) > cfg["max_pending_team_names"] or len(relevant) > cfg["max_blank_recent_team_names"]:
        return Check("team_names_pending", name, "warn",
                     text + "。没定下来的队名，它们的比赛没导入；最近出场的没中文名，网页上只能显示英文。"
                     "告诉 Claude，你认中文队名，可以一起确认"), listing
    return Check("team_names", name, "ok", text), listing


def check_unmatched(reports: Path, settings: dict) -> tuple[Check, Listing]:
    """懂球帝导入时没对上的比赛（比分不同、重复、队名没对上、数据库里没有这场）。

    现在的几场都查过原因（判罚比分、附加赛，见进度记录），所以界限不是 0：超过界限多半是出了新情况
    （例如某个队改了名、某个赛季整季对不上），要看一下。
    """
    cfg, limit = settings["status"], settings["status"]["list_rows"]
    name = "懂球帝没对上的比赛"
    table, err = read_report_csv(reports / "dongqiudi_导入_没对上.csv", ["联赛", "赛季", "主队", "客队", "结果"])
    if err:
        return (Check("unmatched_unreadable", name, "warn", "读不到：" + err),
                Listing(name, [], [], 0, "场", [err]))
    by_result: dict[str, int] = {}
    for r in table:
        k = cell(r, "结果") or "没写结果"
        by_result[k] = by_result.get(k, 0) + 1
    text = f"一共 {len(table)} 场" + ("：" + "，".join(f"{k} {v} 场" for k, v in by_result.items()) if table else "")
    rows = [[cell(r, "联赛"), season_text(cell(r, "赛季")), f"{cell(r, '主队')} 对 {cell(r, '客队')}", cell(r, "比分"),
             cell(r, "结果")] for r in table]
    listing = Listing(name, ["联赛", "赛季", "比赛", "懂球帝比分", "结果"], rows[:limit], len(rows), "场", [])
    if len(table) > cfg["max_unmatched_matches"]:
        return (Check("unmatched_many", name, "warn",
                      text + f"。超过 {cfg['max_unmatched_matches']} 场，可能有队改了名或整季对不上，告诉 Claude 一起看"),
                listing)
    return Check("unmatched", name, "ok", text), listing


# ---------- 抓取次数和最近的问题 ----------
def count_table(con, since: datetime) -> dict[str, dict[str, int]]:
    """从 since 到现在，每种数据各个结果几次：{数据类型: {结果: 次数}}。"""
    table: dict[str, dict[str, int]] = {}
    for dtype, status, n in con.execute(
            "select data_type, status, count(*) from fetch_log where fetched_at_utc >= ? group by 1, 2",
            (since.isoformat(timespec="seconds"),)):
        table.setdefault(dtype, {})[status] = n
    return table


def recent_problems(con, now: datetime, limit: int = 20) -> list[tuple]:
    """最近 24 小时失败或内容可疑的请求（时间、来源、数据类型、结果、HTTP 状态码、网址、说明），新的在前。"""
    since = (now - timedelta(hours=24)).isoformat(timespec="seconds")
    return con.execute("select fetched_at_utc, source, data_type, status, http_status, url, note from fetch_log"
                       " where status in ('failed', 'suspicious') and fetched_at_utc >= ? order by id desc limit ?",
                       (since, limit)).fetchall()


def today_start(now: datetime) -> datetime:
    """北京时间今天 0 点（换成标准时间，用来和 fetch_log 比较）。"""
    return now.astimezone(BEIJING).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


# ---------- 生成网页 ----------
CSS = """
body { font-family: "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2d1f; background: #f4f8f4; }
h1 { color: #1b5e20; } h2 { color: #2e7d32; margin-top: 28px; }
table { border-collapse: collapse; background: #fff; margin: 8px 0; }
th, td { border: 1px solid #c8dcc8; padding: 6px 10px; text-align: left; font-size: 14px; }
th { background: #e3efe3; }
.ok { color: #1b5e20; } .warn { color: #e65100; font-weight: bold; } .error { color: #c62828; font-weight: bold; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.dot.ok { background: #43a047; } .dot.warn { background: #fb8c00; } .dot.error { background: #e53935; }
.small { color: #607060; font-size: 13px; } .url { word-break: break-all; font-size: 12px; }
"""


def e(x) -> str:
    """把任何值转成可以安全放进网页的文字（网址、说明来自外部，里面可能有 < > & 这类符号）。"""
    return html.escape("" if x is None else str(x))


def counts_html(table: dict[str, dict[str, int]]) -> str:
    if not table:
        return "<p class='small'>这段时间没有抓取记录。</p>"
    head = "".join(f"<th>{e(STATUS_NAMES[s])}</th>" for s in STATUS_NAMES)
    body = ""
    for dtype in sorted(table):
        cells = "".join(f"<td>{table[dtype].get(s, '') or ''}</td>" for s in STATUS_NAMES)
        body += f"<tr><td>{e(DATA_TYPE_NAMES.get(dtype, dtype))}</td>{cells}</tr>"
    return f"<table><tr><th>数据</th>{head}</tr>{body}</table>"


def backfill_html(rows: list[dict]) -> str:
    """回补进度按联赛汇总一行（逐赛季的明细在 data/reports/dongqiudi_回补进度.md）。"""
    out = "<table><tr><th>联赛</th><th>已结束赛季应有</th><th>已抓</th><th>有赔率</th><th>还没开始的赛季</th></tr>"
    for lg in dict.fromkeys(r["联赛"] for r in rows):
        rs = [r for r in rows if r["联赛"] == lg and not r["进行中"]]
        total = sum(r["应有"] or 0 for r in rs)
        done = sum(r["已抓"] for r in rs)
        odds = sum(r["有赔率"] for r in rs)
        todo = sum(r["应有"] is None and not r["懂球帝没有"] for r in rs)
        out += f"<tr><td>{e(lg)}</td><td>{total}</td><td>{done}</td><td>{odds}</td><td>{todo or ''}</td></tr>"
    return out + "</table>"


def listing_html(lst: Listing) -> str:
    """一张清单：读不到的原因、表格（只列前几行）、“一共几个，只列了前几个”。里面的文字都来自外部文件，全部转义。"""
    out = "".join(f"<p class='warn'>读不到：{e(x)}</p>" for x in lst.errors)
    if lst.rows:
        head = "".join(f"<th>{e(c)}</th>" for c in lst.columns)
        body = "".join("<tr>" + "".join(f"<td>{e(v)}</td>" for v in r) + "</tr>" for r in lst.rows)
        out += f"<table><tr>{head}</tr>{body}</table>"
        more = f"，只列了前 {len(lst.rows)} {lst.unit}" if lst.total > len(lst.rows) else ""
        out += f"<p class='small'>一共 {lst.total} {lst.unit}{more}。</p>"
    elif not lst.errors:
        out += "<p class='small'>没有。</p>"
    return out


def render_html(checks: list[Check], today: dict, day: dict, problems: list[tuple], rows: list[dict],
                now: datetime, lists: list[Listing] = ()) -> str:
    worst = max(checks, key=lambda c: ["ok", "warn", "error"].index(c.level)).level
    summary = {"ok": "一切正常", "warn": "有需要注意的地方", "error": "有地方出错了"}[worst]
    items = "".join(f"<tr><td><span class='dot {c.level}'></span>{e(c.name)}</td>"
                    f"<td class='{c.level}'>{e(LEVEL_NAMES[c.level])}</td><td>{e(c.text)}</td></tr>" for c in checks)
    probs = "".join(f"<tr><td>{e(bj(datetime.fromisoformat(p[0])))}</td><td>{e(p[1])}</td>"
                    f"<td>{e(DATA_TYPE_NAMES.get(p[2], p[2]))}</td><td>{e(STATUS_NAMES.get(p[3], p[3]))}</td>"
                    f"<td>{e(p[4])}</td><td class='url'>{e(p[5])}</td><td>{e(p[6])}</td></tr>" for p in problems)
    probs_html = (f"<table><tr><th>时间</th><th>来源</th><th>数据</th><th>结果</th><th>状态码</th><th>网址</th>"
                  f"<th>说明</th></tr>{probs}</table>") if problems else "<p class='small'>没有。</p>"
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="600">
<title>足球预测 数据状态</title><style>{CSS}</style></head><body>
<h1>数据状态：<span class="{worst}">{summary}</span></h1>
<p class="small">更新时间 {e(f"{now.astimezone(BEIJING):%Y-%m-%d %H:%M}")}（北京时间）。
这一页每小时自动重新生成一次，开着的话每 10 分钟自动刷新。它只是本机的一个文件，不连网。</p>
<h2>各项检查</h2>
<table><tr><th>项目</th><th>状态</th><th>说明</th></tr>{items}</table>
<h2>今天的抓取次数（北京时间 0 点起）</h2>{counts_html(today)}
<h2>最近 24 小时的抓取次数</h2>{counts_html(day)}
<h2>最近 24 小时的问题</h2>{probs_html}
{"".join(f"<h2>{e(lst.title)}</h2>{listing_html(lst)}" for lst in lists)}
<p class="small">来源：data/reports/dongqiudi_导入_队名对照.csv、dongqiudi_导入_没对上.csv（懂球帝导入时生成，
重新导入才会更新）和 config/teams.yaml。</p>
<h2>懂球帝历史回补</h2>{backfill_html(rows)}
<p class="small">逐赛季明细：data/reports/dongqiudi_回补进度.md（双击“查看懂球帝回补进度.bat”）。<br>
“已保存”= 拿到了新内容存下来；“没变化”= 和上次一样，没重复存；“网站没有”= 懂球帝那边没有这场；
“内容可疑”= 拿到的东西不像正常数据，单独存放不使用；“失败”= 没拿到（网络或网站问题）。</p>
</body></html>
"""


def render_markdown(checks: list[Check], today: dict, problems: list[tuple], now: datetime,
                    lists: list[Listing] = ()) -> str:
    """当天的抓取日报（Markdown）。同一天内每小时覆盖一次，所以最后一版就是全天的结果。"""
    lines = [f"# 抓取日报 {now.astimezone(BEIJING):%Y-%m-%d}", "",
             f"最后更新 {now.astimezone(BEIJING):%H:%M}（北京时间）", "", "## 各项检查", "",
             "| 项目 | 状态 | 说明 |", "|---|---|---|"]
    lines += [f"| {c.name} | {LEVEL_NAMES[c.level]} | {c.text} |" for c in checks]
    lines += ["", "## 今天的抓取次数", "", "| 数据 | " + " | ".join(STATUS_NAMES.values()) + " |",
              "|---" * (len(STATUS_NAMES) + 1) + "|"]
    for dtype in sorted(today):
        lines.append(f"| {DATA_TYPE_NAMES.get(dtype, dtype)} | "
                     + " | ".join(str(today[dtype].get(s, "")) for s in STATUS_NAMES) + " |")
    if not today:
        lines.append("| （今天没有抓取记录） |" + " |" * len(STATUS_NAMES))
    lines += ["", "## 最近 24 小时的问题", ""]
    lines += [f"- {bj(datetime.fromisoformat(p[0]))} {p[1]} {DATA_TYPE_NAMES.get(p[2], p[2])} "
              f"{STATUS_NAMES.get(p[3], p[3])} {p[4] or ''} {p[5]} {p[6] or ''}".rstrip() for p in problems] or ["没有。"]
    for lst in lists:
        lines += ["", f"## {lst.title}", ""]
        lines += [f"读不到：{x}" for x in lst.errors]
        lines += ["- " + "，".join(str(v) for v in r if v != "") for r in lst.rows]
        if lst.rows:
            more = f"，只列了前 {len(lst.rows)} {lst.unit}" if lst.total > len(lst.rows) else ""
            lines += ["", f"一共 {lst.total} {lst.unit}{more}。"]
        elif not lst.errors:
            lines.append("没有。")
    return "\n".join(lines) + "\n"


# ---------- 弹窗提醒 ----------
def popup(title: str, text: str) -> None:
    """弹一个 Windows 提示框。另开一个不带黑窗口的 Python 去弹，这个程序不用等用户点“确定”就能结束。"""
    exe = Path(sys.executable).with_name("pythonw.exe")
    code = ("import ctypes, sys; "
            "ctypes.windll.user32.MessageBoxW(0, sys.argv[2], sys.argv[1], 0x40030)")  # 置顶 + 感叹号图标
    subprocess.Popen([str(exe if exe.exists() else sys.executable), "-c", code, title, text],
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0), close_fds=True)


def new_alerts(checks: list[Check], state_file: Path) -> list[Check]:
    """找出新出现的问题，并记住现在有哪些问题。

    同一个问题只提醒一次：上次运行时已经有的问题，这次不再弹窗。
    问题消失后从记录里删掉，以后再出现会重新提醒。
    """
    old = set(json.loads(state_file.read_text(encoding="utf-8"))) if state_file.exists() else set()
    bad = [c for c in checks if c.level != "ok"]
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(sorted(c.key for c in bad), ensure_ascii=False), encoding="utf-8")
    return [c for c in bad if c.key not in old]


# ---------- 主程序 ----------
def build(con, settings: dict, now: datetime) -> tuple[list[Check], dict, dict, list[tuple], list[dict], list[Listing]]:
    """做全部检查、统计次数。网页和日报都用这一份结果。"""
    cfg = settings["status"]
    raw = resolve_path(settings["paths"]["raw"]) / "dongqiudi"
    reports = resolve_path(settings["paths"]["reports"])
    backfill, rows = check_backfill(con, raw, settings["dongqiudi"], now)
    names, names_list = check_team_names(con, reports, resolve_path(settings["paths"]["teams"]), settings)
    unmatched, unmatched_list = check_unmatched(reports, settings)
    checks = [check_dongqiudi_hourly(con, now, cfg), check_blocked(con, now), check_failures(con, now, cfg),
              backfill, check_backup(resolve_path(settings["paths"]["backup"]), now, cfg),
              check_database(con), check_football_data(con, now), names, unmatched]
    today = count_table(con, today_start(now))
    day = count_table(con, now - timedelta(hours=24))
    return checks, today, day, recent_problems(con, now), rows, [names_list, unmatched_list]


def main(now: datetime | None = None, notify=popup) -> list[Check]:
    s = load_settings()
    now = now or datetime.now(timezone.utc)
    con = connect()
    try:
        checks, today, day, problems, rows, lists = build(con, s, now)
    finally:
        con.close()
    reports = resolve_path(s["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    page = reports / "status.html"
    page.write_text(render_html(checks, today, day, problems, rows, now, lists), encoding="utf-8")
    daily = reports / "抓取日报" / f"{now.astimezone(BEIJING):%Y-%m-%d}.md"
    daily.parent.mkdir(parents=True, exist_ok=True)
    daily.write_text(render_markdown(checks, today, problems, now, lists), encoding="utf-8")
    fresh = new_alerts(checks, reports / "status_alerts.json")
    if fresh:
        notify("足球预测：数据有问题",
               "\n\n".join(f"【{c.name}】{c.text}" for c in fresh) + f"\n\n详情打开：{page}")
    for c in checks:
        print(f"[{LEVEL_NAMES[c.level]}] {c.name}：{c.text}")
    print(f"\n网页：{page}\n日报：{daily}")
    return checks


if __name__ == "__main__":
    main()
