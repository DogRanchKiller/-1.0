"""决定“要推什么”，写进发件箱（需求文档第 9 节“推送：已定的做法”）。

两种消息：
1. 预测：预测存档（predictions 表，接口 1）里新出的 T-24 / T-6 / T-1。
   - 同一场、同一版（kind）只认 prediction_id 最大的那行（改期重出时是新的一行，会再推一次）。
   - 只推还没开球的比赛（第一次运行时不会把以前所有的预测都推一遍）。
   - 这一次运行找到的多场比赛合并成一条消息（周末几十场，一场一条会超过每天的条数上限）。太长才拆成几条。
   - 推过的 prediction_id 记在状态文件（settings 的 notify.state_file）里，不重复推。
   - 内容只有比赛、开球时间（北京时间）、概率和结论，不写任何下注建议。
2. 故障：读定时任务的心跳文件（接口 2），用 watch.py 判断。新出现的故障推一条；同一个故障没恢复前不再推；
   恢复了推一条“已恢复”。一次运行里的几个故障合并成一条。

状态文件的内容：
    {"已推预测": {"prediction_id": "开球时间"}, "正在提醒的故障": {"任务:类型": {...}}, "发送记录": [...（send.py 用）]}
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.clean.dongqiudi_reader import BEIJING
from src.common.config import resolve_path
from src.jobs.heartbeat import read_state
from src.notify.outbox import Outbox, now_text
from src.notify.watch import FAULT_NAMES, bj, job_status

OUTCOMES = ("主胜", "平", "客胜")
WEEKDAYS = "一二三四五六日"
FOOTER = "以上只是模型和赔率算出的概率，比赛结果不确定；未含首发、伤停。"


# ======================= 状态文件 =======================
def load_state(path: Path) -> dict:
    """读状态文件；没有或坏了就从空的开始（坏了最多重复推一次，不会漏掉以后的）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_state(path: Path, state: dict) -> None:
    """先写临时文件再改名（写到一半断电不会留下半个文件）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ======================= 预测消息 =======================
def _utc(text) -> datetime | None:
    """数据库里的开球时间文字（例如 2026-11-07 20:00）→ 带 UTC 时区的时间。"""
    if not text:
        return None
    try:
        t = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def new_predictions(con, kinds: list[str], pushed: dict, now: datetime) -> list[dict]:
    """预测存档里还没推过、还没开球的定时版本。同一场同一版只取 prediction_id 最大的那行（接口 1）。

    数据库里还没有 predictions 表（从没出过预测）时返回空列表。
    """
    if not kinds:
        return []
    has = con.execute("select 1 from sqlite_master where type = 'table' and name = 'predictions'").fetchone()
    if not has:
        return []
    q = ",".join("?" * len(kinds))
    sql = ("select prediction_id, match_id, competition, season, home_team, away_team, kickoff_utc, kind,"
           " p_home, p_draw, p_away, detail from predictions where prediction_id in ("
           f" select max(prediction_id) from predictions where kind in ({q}) group by match_id, kind)")
    cols = ["prediction_id", "match_id", "competition", "season", "home_team", "away_team", "kickoff_utc", "kind",
            "p_home", "p_draw", "p_away", "detail"]
    out = []
    for row in con.execute(sql, list(kinds)):
        r = dict(zip(cols, row))
        ko = _utc(r["kickoff_utc"])
        if str(r["prediction_id"]) in pushed or ko is None or ko <= now:
            continue
        try:
            r["detail"] = json.loads(r["detail"] or "{}")
        except ValueError:
            r["detail"] = {}
        if not isinstance(r["detail"], dict):
            r["detail"] = {}
        r["kickoff"] = ko
        out.append(r)
    _chinese_names(con, out)
    out.sort(key=lambda r: (r["kickoff"], r["competition"], r["home_team"], r["kind"]))
    return out


def _chinese_names(con, rows: list[dict]) -> None:
    """加上中文队名（teams 表的 name_zh），没有就用英文。teams 表不在也不出错。"""
    teams = {r[k] for r in rows for k in ("home_team", "away_team")}
    zh = {}
    if teams:
        try:
            q = ",".join("?" * len(teams))
            zh = {t: n for t, n in con.execute(f"select team_id, name_zh from teams where team_id in ({q})",
                                               list(teams)) if n}
        except sqlite3.Error:
            zh = {}
    for r in rows:
        r["home_zh"] = zh.get(r["home_team"], r["home_team"])
        r["away_zh"] = zh.get(r["away_team"], r["away_team"])


def _pct(p) -> str:
    return f"{float(p) * 100:.0f}%"


def _probs(p) -> str | None:
    """[主胜, 平, 客胜] → “主胜 45% / 平 27% / 客胜 28%”；不是三个数就返回 None。"""
    try:
        vals = [float(x) for x in p]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3:
        return None
    return " / ".join(f"{n} {_pct(v)}" for n, v in zip(OUTCOMES, vals))


def _lean_lines(fusion: dict) -> list[str]:
    """结论里的倾向（和预测页同一套说法，用 src/web/preview_page.py 的 lean_texts）。格式不对就不写。"""
    from src.web.preview_page import lean_texts
    leans = fusion.get("倾向")
    if not isinstance(leans, dict):
        return []
    try:
        return lean_texts({"leans": {k: leans.get(k) for k in ("result", "draw", "over25", "btts", "ah")}})
    except (KeyError, TypeError, IndexError, ValueError):
        return []


def match_block(r: dict, leagues: dict) -> str:
    """一场比赛的一段（Markdown）。"""
    ko = r["kickoff"].astimezone(BEIJING)
    league = leagues.get(r["competition"], r["competition"])
    lines = [f"**{league}　{r['home_zh']} 对 {r['away_zh']}**（{r['kind']}）",
             f"开球：{ko.month} 月 {ko.day} 日 星期{WEEKDAYS[ko.weekday()]} {ko:%H:%M}（北京时间）"]
    d = r["detail"]
    fusion = d.get("融合") if isinstance(d.get("融合"), dict) else {}
    model = _probs([r["p_home"], r["p_draw"], r["p_away"]])
    fused = _probs(d.get("融合概率")) if d.get("融合概率") is not None else None
    if fused:
        level = fusion.get("信心等级")
        level_name = {3: "●●● 高", 2: "●●○ 中", 1: "●○○ 低"}.get(level)
        lines.append(f"融合结论：{fused}" + (f"，信心 {level_name}" if level_name else ""))
        lines += _lean_lines(fusion)
        unsure = fusion.get("不确定因素")
        if isinstance(unsure, list) and unsure:
            lines.append("要注意：" + "、".join(str(x) for x in unsure))
        if model:
            lines.append(f"模型自己：{model}")
    elif model:
        lines.append(f"模型概率：{model}（没有融合结论）")
    else:
        lines.append("这一版没有概率")
    if d.get("改期重出"):
        lines.append("（改期后重新出的一版）")
    return "  \n".join(lines)


def prediction_messages(rows: list[dict], settings: dict) -> list[tuple[str, str]]:
    """把这次找到的预测合并成消息：[(标题, 正文)]。正文超过 max_content_chars 时拆成几条。"""
    if not rows:
        return []
    cfg = settings["notify"]
    leagues = {k: v for group in settings["leagues"].values() for k, v in group.items()}
    blocks = [match_block(r, leagues) for r in rows]
    limit = int(cfg["max_content_chars"]) - len(FOOTER) - 10
    chunks, cur = [], []
    for b in blocks:
        if cur and len("\n\n".join(cur + [b])) > limit:
            chunks.append(cur)
            cur = []
        cur.append(b)
    chunks.append(cur)
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    base = "足球预测：" + "，".join(f"{k} {v} 场" for k, v in kinds.items())
    out = []
    for i, c in enumerate(chunks):
        title = base + (f"（{i + 1}/{len(chunks)}）" if len(chunks) > 1 else "")
        out.append((title[:int(cfg["max_title_chars"])], "\n\n".join(c) + "\n\n" + FOOTER))
    return out


def compose_predictions(con, outbox: Outbox, state: dict, settings: dict, now: datetime) -> int:
    """找新出的预测，合并写进发件箱，记下推过的编号。返回写了几条消息。"""
    cfg = settings["notify"]
    pushed = state.setdefault("已推预测", {})
    # 开球后 pushed_keep_days 天的编号删掉（那时早就不会再出这场的新版本了）
    keep = now - timedelta(days=float(cfg["pushed_keep_days"]))
    for pid, ko in list(pushed.items()):
        t = _utc(ko)
        if t is None or t < keep:
            del pushed[pid]
    rows = new_predictions(con, list(cfg["kinds"]), pushed, now)
    n = 0
    msgs = prediction_messages(rows, settings)
    ids = sorted(r["prediction_id"] for r in rows)
    for i, (title, content) in enumerate(msgs):
        # 关键字 = 这条里的预测编号；同样的编号不会再生成第二条
        if outbox.add("预测", title, content, key=f"预测 {ids} 第 {i + 1} 条", now=now):
            n += 1
    for r in rows:
        pushed[str(r["prediction_id"])] = r["kickoff_utc"]
    return n


# ======================= 故障消息 =======================
def compose_faults(outbox: Outbox, state: dict, settings: dict, now: datetime, jobs_dir: Path,
                   secret: str | None = None) -> dict:
    """读心跳文件，新故障推一条、恢复推一条。返回 {"新故障": n, "已恢复": n, "正在提醒": n}。"""
    cfg = settings["notify"]
    active = state.setdefault("正在提醒的故障", {})
    current = {}
    for job in cfg["push_jobs"]:
        st = job_status(job, read_state(job, jobs_dir), now, settings, secret)
        for f in st.faults:
            current[f"{job}:{f.type}"] = {"任务": job, "名字": st.name, "类型": f.type, "说明": f.text}
    new = {k: v for k, v in current.items() if k not in active}
    gone = {k: v for k, v in active.items() if k not in current}
    if new:
        lines = [f"- **{v['名字']}**：{v['说明']}" for v in new.values()]
        title = "足球预测：定时任务出问题（" + "、".join(dict.fromkeys(v["名字"] for v in new.values())) + "）"
        content = ("\n".join(lines) + f"\n\n发现时间：{bj(now)}（北京时间）。打开“查看数据状态.bat”看详情；"
                   "同一个问题恢复前不再重复提醒。")
        key = "故障 " + " ".join(f"{k}@{now_text(now)}" for k in sorted(new))
        outbox.add("故障", title[:int(cfg["max_title_chars"])], content, key=key, now=now)
        for k, v in new.items():
            active[k] = {**v, "发现时间": now_text(now)}
    if gone:
        lines = [f"- **{v['名字']}**：{FAULT_NAMES.get(v['类型'], v['类型'])}，已经好了（{v['发现时间'][5:16]} UTC 发现）"
                 for v in gone.values()]
        title = "足球预测：定时任务已恢复（" + "、".join(dict.fromkeys(v["名字"] for v in gone.values())) + "）"
        key = "恢复 " + " ".join(f"{k}@{v['发现时间']}" for k, v in sorted(gone.items()))
        outbox.add("故障", title[:int(cfg["max_title_chars"])],
                   "\n".join(lines) + f"\n\n确认时间：{bj(now)}（北京时间）。", key=key, now=now)
        for k in gone:
            del active[k]
    return {"新故障": len(new), "已恢复": len(gone), "正在提醒": len(active)}


# ======================= 一次运行 =======================
def open_db_readonly(path: Path) -> sqlite3.Connection | None:
    """只读打开数据库；文件不存在返回 None（不新建空库）。"""
    path = Path(path)
    if not path.exists():
        return None
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def compose(settings: dict, now: datetime, db_path: Path | None = None, jobs_dir: Path | None = None,
            secret: str | None = None) -> dict:
    """compose 这一层的一次运行：预测 + 故障，写进发件箱，存状态文件。返回摘要（中文键）。"""
    cfg = settings["notify"]
    outbox = Outbox.from_settings(settings)
    state_path = resolve_path(cfg["state_file"])
    state = load_state(state_path)
    db_path = db_path or resolve_path(settings["paths"]["database"])
    jobs_dir = jobs_dir or resolve_path(settings["jobs"]["state_dir"])
    summary = {}
    con = open_db_readonly(db_path)
    if con is None:
        summary["预测消息"] = 0
        summary["说明"] = "找不到数据库"
    else:
        try:
            summary["预测消息"] = compose_predictions(con, outbox, state, settings, now)
        finally:
            con.close()
    f = compose_faults(outbox, state, settings, now, jobs_dir, secret)
    summary.update({"新故障": f["新故障"], "已恢复": f["已恢复"], "正在提醒的故障": f["正在提醒"]})
    save_state(state_path, state)
    return summary
