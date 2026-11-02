"""只读的数据健康检查：请求成功、文件变化、导入和可用证据分别统计。

这里只消费数据库、原始 CSV 表头和采集/复盘清单，不采集、不导入、不重算收盘资格。
没有可验证的分页凭据时，未来赛程数量只能叫“已知”，不能据此承诺完整赛历。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.clean.dongqiudi_reader import BEIJING
from src.collect.football_data import current_season, season_code
from src.common.config import load_football_data_columns, resolve_path


@dataclass
class Check:
    key: str
    name: str
    level: str                 # ok / unknown（无可评估证据）/ warn / error
    text: str


def _time(value) -> datetime | None:
    """旧清单用无时区的 UTC 字符串，新清单也接受显式 UTC；无效值保持未知。"""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _bj(value: datetime | None) -> str:
    return value.astimezone(BEIJING).strftime("%m-%d %H:%M") if value else "未知"


def _season(now: datetime) -> int:
    # 自动维护按北京时间日期运行；7月1日零时不能因为UTC仍在6月30日而检查上一季。
    return current_season(now.astimezone(BEIJING))


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("文件顶层不是对象")
    return data


def _manifest(settings: dict, name: str, group: str) -> tuple[dict, str | None]:
    path = resolve_path(settings["jobs"]["state_dir"]) / f"{name}.json"
    try:
        data = _json(path)
        if type(data.get("schema")) is not int or data["schema"] != 1 or not isinstance(data.get(group), dict):
            raise ValueError("清单版本或分组不符")
        return data[group], None
    except (OSError, ValueError) as exc:
        return {}, f"{name} 清单不可用（{type(exc).__name__}）"


def _scopes(settings: dict, source: str) -> list[str]:
    if source == "football-data":
        configured = settings.get("football_data", {}).get("refresh_leagues")
        return list(dict.fromkeys(configured if configured is not None else
                                 [lg for group in settings["leagues"].values() for lg in group]))
    if source == "dongqiudi":
        return list(settings["dongqiudi"]["league_ids"])
    return list(settings.get("cup_refresh", {}).get("competitions", []))


def _config(settings: dict) -> dict:
    cfg = settings["status"]["data_health"]
    for key in ("football_data_stale_hours", "dongqiudi_stale_hours", "cup_stale_hours",
                "result_lookback_days", "result_due_hours", "upcoming_days", "closing_snapshot_stale_hours"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须是正数")
    return cfg


def database_check(con, now: datetime | None = None) -> Check:
    """总量、已完成且有比分的日期、未赛日期是不同指标；旧库缺表时不回退成绿色。"""
    day = (now or datetime.now(timezone.utc)).astimezone(BEIJING).date().isoformat()
    try:
        n = con.execute("select count(*) from matches").fetchone()[0]
        completed = con.execute(
            "select max(m.match_date) from matches m where m.status='finished' and m.match_date<=? and exists "
            "(select 1 from match_stats s where s.match_id=m.match_id and s.home_goals>=0 and s.away_goals>=0)", (day,)
        ).fetchone()[0]
        future_finished = con.execute("select count(*) from matches where status='finished' and match_date>?", (day,)).fetchone()[0]
        pending, future = con.execute(
            "select count(*),max(match_date) from matches where status='scheduled'"
        ).fetchone()
    except sqlite3.Error:
        return Check("database", "数据库", "warn", "数据结构不完整，已完成赛果与未赛赛程统计未知")
    return Check("database", "数据库", "warn" if future_finished else "ok" if n and completed else "unknown",
                 f"比赛记录 {n:,} 场；最近已完成赛果 {completed or '未知'}；"
                 f"未赛赛程 {pending:,} 场，未赛赛程最远 {future or '未知'}。未赛日期不代表赛果更新"
                 + (f"；未来日期却标已赛 {future_finished} 场，需核对日期/状态" if future_finished else ""))


def _fetch_scope(row: dict, settings: dict, season: int) -> tuple[str, str] | None:
    """用来源自己的 URL 字段识别联赛/赛季；notes、旧季和其他联赛不能刷新本项。"""
    url = urlparse(row["url"])
    if row["source"] == "football-data" and row["data_type"] == "league_csv":
        if url.hostname not in ("football-data.co.uk", "www.football-data.co.uk"):
            return None
        parts = url.path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["mmz4281", season_code(season)] and parts[2].endswith(".csv"):
            return "football-data", parts[2][:-4]
    if row["source"] == "dongqiudi" and row["data_type"] == "fixtures":
        if url.hostname != "api.dongqiudi.com":
            return None
        league = next((lg for lg, lid in settings["dongqiudi"]["league_ids"].items()
                       if url.path == f"/data/tab/league/new/{lid}"), None)
        requested = _time(parse_qs(url.query).get("start", [None])[0])
        # 此接口没有赛季字段，使用明确的请求起始日期归季；跨季无法确认的窗口保持未知。
        if league and requested and current_season(requested) == season:
            return "dongqiudi", league
    return None


def _source_logs(con, settings: dict, now: datetime) -> tuple[dict, str | None]:
    season = _season(now)
    try:
        cursor = con.execute(
            "select source,data_type,url,fetched_at_utc,status,file_path,sha256 from fetch_log "
            "where (source='football-data' and data_type='league_csv') "
            "or (source='dongqiudi' and data_type='fixtures' and fetched_at_utc>=?)",
            (f"{season}-07-01",))
        names = [item[0] for item in cursor.description]
        found = {}
        for values in cursor:
            row = dict(zip(names, values))
            scope = _fetch_scope(row, settings, season)
            if scope:
                found.setdefault(scope, []).append(row)
        return found, None
    except (sqlite3.Error, ValueError, TypeError):
        return {}, "抓取记录无法读取"


def _source_dates(con, source: str, league: str, season: int, now: datetime) -> str:
    try:
        day = con.execute(
            "select max(m.match_date) from matches m join match_stats s on s.match_id=m.match_id "
            "where m.competition=? and m.season=? and m.status='finished' and s.source=? and m.match_date<=? "
            "and s.home_goals>=0 and s.away_goals>=0 and exists (select 1 from match_sources ms "
            "where ms.match_id=m.match_id and ms.source=s.source)", (league, season, source, now.astimezone(BEIJING).date().isoformat())
        ).fetchone()[0]
        return f"库内本来源最近已完成赛果 {day or '暂无'}"
    except sqlite3.Error:
        return "库内本来源赛果日期未知（表结构不完整）"


def _column_groups(settings: dict) -> dict:
    """沿用已有数据字典的公司、时点和列名，不把一般赔率或赛前列当收盘列。"""
    groups = {(book, market): {} for book in settings["review"]["close_bookmakers"] for market in ("1x2", "ou")}
    for column, meta in load_football_data_columns(settings).items():
        item = meta.get("数据项", "")
        market = "1x2" if "胜平负" in item else "ou" if "大小球 2.5" in item else None
        key = (meta.get("公司"), market)
        if meta.get("时点") == "收盘" and key in groups:
            groups[key].setdefault(meta.get("部分"), set()).add(column)
    return groups


def _file_columns(settings: dict, league: str, season: int, raw_file, sha, groups: dict) -> tuple[list[str], str]:
    """只读本联赛本季原始 CSV；表头齐全仅是字段能力，不替代逐场证据资格。"""
    if not isinstance(raw_file, str) or not raw_file:
        return ["尚无联赛文件记录，收盘字段能力未知"], "unknown"
    try:
        path = resolve_path(raw_file).resolve()
        folder = (resolve_path(settings["paths"]["raw"]) / "football-data" / league / season_code(season)).resolve()
        if not path.is_relative_to(folder) or path.suffix.lower() != ".csv":
            return ["文件路径不属于本联赛本季，收盘字段能力未知"], "warn"
        if sha and hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            return ["原始文件指纹与保存记录不符，收盘字段能力未知"], "warn"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            headers = set(next(csv.reader(stream), []))
        if "Div" not in headers:
            raise ValueError("不是联赛 CSV")
    except (OSError, ValueError, csv.Error):
        return ["联赛文件无法读取，收盘字段能力未知"], "warn"
    texts, level = [], "ok"
    for (book, market), parts in groups.items():
        needed = ("主胜", "平", "客胜") if market == "1x2" else ("大", "小")
        label = "胜平负" if market == "1x2" else "大小2.5"
        if any(part not in parts for part in needed):
            texts.append(f"{book} {label}：字段定义未知")
            level = "unknown" if level == "ok" else level
            continue
        missing = ["/".join(sorted(parts[part])) for part in needed if not (parts[part] & headers)]
        texts.append(f"{book} {label}：" + ("文件缺列（" + "、".join(missing) + "）" if missing else "列齐全"))
        if missing:
            level = "warn"
    texts.append("字段存在不代表本次复盘已覆盖")
    return texts, level


def check_sources(con, settings: dict, now: datetime) -> list[Check]:
    cfg, season = _config(settings), _season(now)
    logs, logs_error = _source_logs(con, settings, now)
    try:
        columns = _column_groups(settings)
    except (OSError, ValueError, KeyError, TypeError):
        columns = {}
    out = []
    for source, name, group, threshold in (
        ("football-data", "football_data_sources", "leagues", "football_data_stale_hours"),
        ("dongqiudi", "dongqiudi_sources", "leagues", "dongqiudi_stale_hours"),
        ("wikipedia", "cup_sources", "competitions", "cup_stale_hours"),
    ):
        entries, manifest_error = _manifest(settings, name, group)
        for league in _scopes(settings, source):
            entry = entries.get(league)
            entry = entry if isinstance(entry, dict) and type(entry.get("season")) is int and entry["season"] == season else {}
            values = logs.get((source, league), [])
            timed = [(t, row) for row in values if (t := _time(row.get("fetched_at_utc"))) and t <= now]
            timed.sort(key=lambda item: item[0])
            successes = [(t, row) for t, row in timed if row["status"] in ("saved", "unchanged")]
            changed_rows = [(t, row) for t, row in successes if row["status"] == "saved"]
            checked = successes[-1][0] if successes else None
            changed = changed_rows[-1][0] if changed_rows else None
            saved = changed_rows[-1][1] if changed_rows else {}
            imported = _time(entry.get("last_import_success_utc"))
            last_success = entry.get("last_success")
            if isinstance(last_success, dict) and last_success.get("season") != season:
                imported = None
            manifest_checked = _time(entry.get("last_checked_utc"))
            manifest_changed = _time(entry.get("last_changed_utc"))
            status = timed[-1][1]["status"] if timed else entry.get("status")
            # 杯赛清单的 last_checked 是本次尝试，可能失败；文案不能叫“成功检查”。
            if manifest_checked and manifest_checked <= now and (source == "wikipedia" or entry.get("status") in ("saved", "unchanged", "ok")):
                checked = max(t for t in (checked, manifest_checked) if t is not None)
            if manifest_changed and manifest_changed <= now:
                changed = max(t for t in (changed, manifest_changed) if t is not None)
            text, level = [], "ok"
            if checked:
                prefix = "最近检查（含失败）" if source == "wikipedia" else "最近成功检查"
                text.append(f"{prefix} {_bj(checked)}")
                if (now - checked).total_seconds() > cfg[threshold] * 3600:
                    text.append("检查已过期")
                    level = "warn"
            else:
                text.append("本季尚无成功检查")
                level = "warn"
            text.append(f"内容保存 {_bj(changed)}（本机保存时间，不是网站发布时间）")
            if imported and imported <= now:
                text.append(f"导入成功 {_bj(imported)}")
                if changed and changed > imported:
                    text.append("内容保存晚于导入，最新内容尚未确认入库")
                    level = "warn"
            else:
                text.append("导入时间未知")
                level = "unknown" if level == "ok" else level
            if any(t and t > now for t in (imported, manifest_checked, manifest_changed)) or any(
                    not _time(row.get("fetched_at_utc")) or _time(row["fetched_at_utc"]) > now for row in values):
                text.append("记录时间异常，不能确认新鲜度")
                level = "warn"
            if status:
                text.append(f"最近请求 {status}")
                if status not in ("saved", "unchanged", "ok"):
                    level = "warn"
            if source == "wikipedia":
                text.append("本次配置页检查完成" if entry.get("complete") is True and status == "ok" else "本次配置页检查未完整确认")
                text.append("页面检查完成不证明整个赛历完整")
            if logs_error and source != "wikipedia":
                text.append(logs_error)
                level = "warn"
            if manifest_error:
                text.append(manifest_error)
            text.append(_source_dates(con, source, league, season, now))
            if source == "football-data":
                # 成功清单明确记录本轮实际导入的文件；unchanged 不产生新 saved 行，旧日志路径可能已经迁移。
                # 只有清单覆盖到最近保存事件时才优先它，后续下载但尚未导入的新文件仍按新抓取记录检查。
                use_manifest = bool(entry.get("raw_file") and imported and imported <= now and
                                    manifest_checked and manifest_checked <= now and
                                    (not changed_rows or manifest_checked >= changed_rows[-1][0]))
                raw_file = entry.get("raw_file") if use_manifest else saved.get("file_path")
                sha = entry.get("sha256") if use_manifest else saved.get("sha256")
                file_text, file_level = _file_columns(settings, league, season,
                                                    raw_file, sha, columns)
                text += file_text if columns else ["收盘字段定义不可用，字段能力未知"]
                if file_level == "warn" or not columns:
                    level = "warn"
                elif file_level == "unknown" and level == "ok":
                    level = "unknown"
            out.append(Check(f"data_{source}_{league}_{season}", f"数据来源：{source} {league} {season}/{(season+1)%100:02}",
                             level, "；".join(text)))
    return out


def _match_rows(con, league: str, season: int) -> list[dict]:
    cursor = con.execute(
        "select m.match_id,m.match_date,m.kickoff_utc,m.status,m.date_precision,"
        "max(case when s.source='football-data' then 1 else 0 end) as fd,"
        "max(case when s.source='dongqiudi' then 1 else 0 end) as dqd "
        "from matches m left join match_stats s on s.match_id=m.match_id and s.home_goals>=0 and s.away_goals>=0 "
        "and exists (select 1 from match_sources ms where ms.match_id=m.match_id and ms.source=s.source) "
        "where m.competition=? and m.season=? group by m.match_id", (league, season))
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, values)) for values in cursor]


def check_results(con, settings: dict, now: datetime) -> list[Check]:
    cfg, season = _config(settings), _season(now)
    local_day = now.astimezone(BEIJING).date().isoformat()
    since = (now.astimezone(BEIJING) - timedelta(days=cfg["result_lookback_days"])).date().isoformat()
    due = now - timedelta(hours=cfg["result_due_hours"])
    out = []
    for league in _scopes(settings, "dongqiudi"):
        key, name = f"results_{league}_{season}", f"近期赛果：{league} {season}/{(season+1)%100:02}"
        try:
            rows = [r for r in _match_rows(con, league, season) if since <= r["match_date"] <= local_day]
        except (sqlite3.Error, TypeError):
            out.append(Check(key, name, "warn", "赛果覆盖未知（表结构不完整）"))
            continue
        eligible = [r for r in rows if r["status"] == "finished" or
                    (r["status"] == "scheduled" and r["date_precision"] == "day" and
                     (kickoff := _time(r["kickoff_utc"])) and kickoff <= due)]
        ambiguous = sum(r["status"] == "scheduled" and (r["date_precision"] != "day" or not _time(r["kickoff_utc"])) for r in rows)
        fd = sum(r["fd"] for r in eligible)
        fallback = sum(bool(r["dqd"]) and not r["fd"] for r in eligible)
        missing = sum(not r["dqd"] and not r["fd"] for r in eligible)
        level = "warn" if missing or fallback else "ok" if eligible else "unknown"
        text = (f"最近 {cfg['result_lookback_days']:g} 天库内已知已赛或应结束 {len(eligible)} 场；"
                f"football-data 已有比分 {fd} 场；懂球帝补位 {fallback} 场；两源均缺比分 {missing} 场。"
                "补位表示主来源尚未覆盖，不等于两源都缺；应结束按明确开球时间和宽限判断，仍待核实比赛状态。"
                f"时间不明确而未判逾期 {ambiguous} 场。只核对库内已知比赛，不承诺外部赛历完整")
        out.append(Check(key, name, level, text))
    return out


def check_future(con, settings: dict, now: datetime) -> list[Check]:
    cfg, season = _config(settings), _season(now)
    until = now + timedelta(days=cfg["upcoming_days"])
    out = []
    for league in dict.fromkeys(_scopes(settings, "dongqiudi") + _scopes(settings, "wikipedia")):
        key, name = f"future_{league}_{season}", f"未来赛程：{league} {season}/{(season+1)%100:02}"
        try:
            rows = con.execute("select match_date,kickoff_utc,date_precision from matches where competition=? "
                               "and season=? and status='scheduled'", (league, season)).fetchall()
            dates = [day for day, ko, precision in rows if
                     (now <= parsed <= until if (parsed := _time(ko)) and precision == "day" else
                      now.date().isoformat() <= day <= until.date().isoformat())]
        except (sqlite3.Error, TypeError):
            out.append(Check(key, name, "warn", "未来已知赛程统计未知（表结构不完整）"))
            continue
        text = (f"未来 {cfg['upcoming_days']:g} 天已知赛程 {len(dates)} 场"
                + (f"，日期 {min(dates)} 至 {max(dates)}" if dates else "；可能处于休赛期或尚未拿到赛程")
                + "；完整性未验证（没有逐页覆盖凭据）。日期或轮次不精确的场次只计为已知；零场本身不判抓取故障")
        out.append(Check(key, name, "unknown", text))
    return out


def _count(value) -> bool:
    return type(value) is int and value >= 0


def _coverage_groups(data: dict, books: list[str]) -> list[dict]:
    """只接受与既有 coverage_summary 一致的分母，坏快照不能拿部分数字拼成覆盖率。"""
    cov = data["closing_coverage"]
    n, q, ids = cov["match_count"], cov["qualified_match_count"], cov["requested_match_ids"]
    if not _count(n) or not _count(q) or q > n or not isinstance(ids, list) or len(ids) != n or \
            any(type(mid) is not int or mid <= 0 for mid in ids) or len(set(ids)) != n:
        raise ValueError("总分母或成员无效")
    groups = cov["by_bookmaker_market"]
    expected = {(book, market) for book in books for market in ("1x2", "ou")}
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError("公司玩法分组不完整")
    actual = set()
    for row in [cov] + groups:
        rn, rq, ratio = row["match_count"], row["qualified_match_count"], row["coverage"]
        if not _count(rn) or rn != n or not _count(rq) or rq > n:
            raise ValueError("分组分母不一致")
        if (not n and ratio is not None) or (n and (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or
                not math.isfinite(ratio) or not math.isclose(ratio, rq / n))):
            raise ValueError("覆盖率与分子分母不一致")
    for row in groups:
        identity = (row["bookmaker"], row["market"])
        if identity not in expected or identity in actual or (row["market"] == "ou" and row.get("line") != 2.5):
            raise ValueError("玩法、盘口或重复分组无效")
        reasons = row["absence_reasons"]
        if not isinstance(reasons, dict) or any(not isinstance(k, str) or not _count(v) or v > n - row["qualified_match_count"]
                                                for k, v in reasons.items()):
            raise ValueError("缺失原因无效")
        actual.add(identity)
    counts = [r["qualified_match_count"] for r in groups]
    if actual != expected or (counts and not max(counts) <= q <= min(n, sum(counts))):
        raise ValueError("合计与分组不一致")
    return groups


def check_closing(settings: dict, now: datetime) -> list[Check]:
    cfg = _config(settings)
    path = resolve_path(settings["review"]["output_dir"]) / "live_metrics.json"
    try:
        data = _json(path)
        generated, cutoff = _time(data.get("generated_at")), _time(data.get("evaluation_cutoff"))
        if not generated or not cutoff or not cutoff <= generated <= now:
            raise ValueError("缺少可信生成时间/评估截止，或时间次序无效")
        if (now - min(generated, cutoff)).total_seconds() > cfg["closing_snapshot_stale_hours"] * 3600:
            raise ValueError("复盘快照已过期")
        groups = _coverage_groups(data, list(settings["review"]["close_bookmakers"]))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # 与正常的0/0不同：坏/旧文件是需要检查的状态，文字明确覆盖未知，不能默认为0%。
        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return [Check("closing_snapshot", "合格收盘：复盘快照", "warn", f"覆盖未知：{reason}。等待重新生成可信复盘快照")]
    out = []
    reason_names = {"no_record": "无记录", "in_play": "赛中或赛后观测", "not_observed_pregame": "未在赛前观测",
                    "not_ou_2_5": "不是大小2.5", "not_verified_close": "未验证为收盘"}
    for row in groups:
        book, market = row["bookmaker"], row["market"]
        n, q = row["match_count"], row["qualified_match_count"]
        label = "胜平负" if market == "1x2" else "大小2.5"
        state = "无可评估样本（0/0）" if not n else f"{q}/{n}（{q/n:.1%}）" + ("，无合格收盘证据" if not q else "")
        text = f"{state}；快照生成 {_bj(generated)}，评估截止 {_bj(cutoff)}。分母只取这次复盘已保存的比赛集合"
        if row["absence_reasons"]:
            text += "；缺失原因（可重叠）：" + "、".join(f"{reason_names.get(k, k)} {v}" for k, v in row["absence_reasons"].items())
        text += "。覆盖不是预测准确率，也不能据此认定胜过市场"
        out.append(Check(f"closing_{book}_{market}", f"合格收盘：{book} {label}",
                         "unknown" if not n else "warn" if q < n else "ok", text))
    return out


def build_checks(con, settings: dict, now: datetime) -> list[Check]:
    """所有入口均不写库/文件；缺配置或坏数据保留未知，不能让整页崩溃或默认为正常。"""
    out = []
    for name, fn in (("sources", check_sources), ("results", check_results), ("future", check_future)):
        try:
            out += fn(con, settings, now)
        except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            out.append(Check(f"health_{name}", f"数据检查：{name}", "warn", f"数据健康未知（配置或结构不完整：{type(exc).__name__}）"))
    try:
        out += check_closing(settings, now)
    except (KeyError, TypeError, ValueError) as exc:
        out.append(Check("closing_snapshot", "合格收盘：复盘快照", "warn", f"覆盖未知（配置不完整：{type(exc).__name__}）"))
    return out
