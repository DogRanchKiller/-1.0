"""逐场周期证据页。只展示只读验收报告，不补预测、不重算成绩、不推断首发通过。"""
from __future__ import annotations

from src.web.site.common import bj_text, details, e, match_href, page, placeholder, table
from src.web.site.data import SiteData


KINDS = ("T-24", "T-6", "T-1")
MARKETS = {"1x2": "胜平负", "ah": "亚盘", "ou": "大小球"}
ATTENTION = {"late", "overdue", "missed_window", "invalid_archive", "invalid", "error"}
PENDING = {"not_due", "awaiting_scheduler", "awaiting_result", "pending", "partial", "missing"}


def stamp(value) -> str:
    """人读北京时间到秒，原始 UTC 保留在提示中；不丢失 JSON 中的时间精度。"""
    if not value:
        return "—"
    try:
        label = bj_text(value, "%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError):
        return f"<span class='tag warn'>时间格式未知：{e(value)}</span>"
    return f"<span title='{e(value)} UTC'>{e(label)}</span>"


def state_html(item: dict | None) -> str:
    item = item or {}
    state = item.get("state")
    cls = " warn" if state in ATTENTION else (" temp" if state in PENDING else "")
    return f"<span class='tag{cls}'>{e(item.get('label') or '证据未知')}</span>"


def issues_html(items) -> str:
    return "<ul class='points small'>" + "".join(f"<li>{e(v)}</li>" for v in items) + "</ul>" if items else ""


def number(value) -> str:
    return "—" if value is None else e(value)


def minutes(value) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "—"


def match_report(data: SiteData, mid: int) -> dict | None:
    return next((row for row in data.lifecycle.get("matches", []) if row.get("match_id") == mid), None)


def match_reference(data: SiteData, mid: int) -> str:
    """单场页只引用同一次生成的报告，不从最新预览版推断周期进度。"""
    row = match_report(data, mid)
    if row is None:
        return ""
    slots = {slot.get("kind"): slot for slot in row.get("slots", [])}
    states = " · ".join(f"{e(kind)} {state_html(slots.get(kind))}" for kind in KINDS)
    return (f"<div class='note'><a href='../lifecycle.html#match-{int(mid)}'>本场逐场验收记录</a>"
            f"<p>{states}</p><p class='small'>{state_html(row.get('review'))} · "
            f"{state_html(row.get('closing'))} · {e((row.get('lineup') or {}).get('label') or '首发链路尚未验收')}</p></div>")


def fixture_name(data: SiteData, row: dict) -> str:
    return f"{data.zh(row.get('home_team') or '主队未知')} 对 {data.zh(row.get('away_team') or '客队未知')}"


def overview(data: SiteData, matches: list[dict]) -> str:
    rows = []
    for row in matches:
        mid = int(row["match_id"])
        fixture = f"<a href='#match-{mid}'>{e(fixture_name(data, row))}</a>"
        slots = {slot.get("kind"): slot for slot in row.get("slots", [])}
        rows.append([stamp(row.get("kickoff_utc")), fixture,
                     *(state_html(slots.get(kind)) for kind in KINDS),
                     state_html(row.get("result")), state_html(row.get("review")), state_html(row.get("closing"))])
    return table(["开球（北京时间）", "比赛 · 点开看证据", "T-24", "T-6", "T-1", "赛果", "评分", "收盘证据"], rows)


def slot_rows(row: dict) -> str:
    slots = {slot.get("kind"): slot for slot in row.get("slots", [])}
    rows = []
    for kind in KINDS:
        slot = slots.get(kind) or {}
        timing = (f"截止晚 {minutes(slot.get('late_minutes'))} 分钟<br>"
                  f"写入晚 {minutes(slot.get('created_late_minutes'))} 分钟") if slot.get("prediction_id") else "—"
        rows.append([e(kind), stamp(slot.get("due_at_utc")), state_html(slot) + issues_html(slot.get("issues")),
                     number(slot.get("prediction_id")), stamp(slot.get("cutoff_utc")),
                     stamp(slot.get("created_at_utc")), timing])
    return table(["时点", "计划时间", "存档核验", "存档编号", "数据截止时间", "实际写入时间", "实际迟到"], rows)


def archive_rows(row: dict) -> str:
    rows = []
    for slot in row.get("slots", []):
        for archive in slot.get("archives", []):
            validity = "证据有效" if archive.get("valid") is True else "证据无效或未知"
            rows.append([number(archive.get("prediction_id")), e(archive.get("kind") or slot.get("kind") or "—"),
                         e(f"{archive.get('model') or '未知'} / {archive.get('model_version') or '未知'}"),
                         stamp(archive.get("kickoff_utc")), stamp(archive.get("cutoff_utc")),
                         stamp(archive.get("created_at_utc")), e(validity) + issues_html(archive.get("issues"))])
    if not rows:
        return ""
    return details("全部同场定时存档核验（保留重复、旧开球时间与版本冲突）",
                   table(["存档编号", "时点", "模型 / 版本", "存档开球时间", "数据截止", "实际写入", "证据核验"], rows))


def closing_rows(closing: dict) -> str:
    groups = closing.get("groups") or []
    if not groups:
        return placeholder("收盘分组证据未知")
    rows = []
    for group in groups:
        rows.append([e(group.get("bookmaker") or "未知"), e(MARKETS.get(group.get("market"), group.get("market") or "未知")),
                     number(group.get("line")), state_html(group), number(group.get("candidate_count")),
                     issues_html(group.get("reasons")) or "—"])
    return table(["公司", "市场", "盘口", "证据状态", "候选记录数", "原因"], rows)


def match_evidence(data: SiteData, row: dict) -> str:
    mid = int(row["match_id"])
    name = fixture_name(data, row)
    heading = f"{e(data.comp(row.get('competition') or '未知赛事'))} · {e(name)} · 比赛编号 {mid}"
    if mid in data.page_ids:
        heading += f" · <a href='{match_href(mid, '')}'>单场预测与复盘</a>"
    missing = "<p class='tag warn'>当前来源缺失；只展示现有证据</p>" if row.get("source_missing") else ""
    result, review, closing = (row.get(key) or {} for key in ("result", "review", "closing"))
    scores = f"{number(result.get('home_goals'))}–{number(result.get('away_goals'))}"
    scored_ids = "、".join(str(pid) for pid in review.get("scored_prediction_ids", [])) or "暂无"
    content = (f"<p>{heading}</p>{missing}{slot_rows(row)}"
               f"<h3>赛果与评分</h3><p>{state_html(result)} · 比分 {scores} · "
               f"赛果已知时间 {stamp(result.get('known_at_utc'))} · 来源 {e(result.get('source') or '未知')}</p>"
               f"<p>{state_html(review)} · 已评分存档编号：{e(scored_ids)}</p>{issues_html(review.get('issues'))}"
               f"<h3>收盘证据</h3><p>{state_html(closing)}</p>{closing_rows(closing)}"
               f"<p class='small'>{e((row.get('lineup') or {}).get('label') or '首发链路尚未验收')}</p>"
               f"{archive_rows(row)}")
    return f"<section id='match-{mid}'>" + details(f"{name} · 比赛 {mid} 的完整证据", content) + "</section>"


def render(data: SiteData) -> str:
    report = data.lifecycle
    heading = "<h1>足球预测 <span class='accent'>逐场验收</span></h1>"
    if not report or not isinstance(report.get("matches"), list):
        body = heading + placeholder("验收报告暂不可用；未据此判断任何比赛完成。")
        return page("逐场验收 - 足球预测", body, "", data.now, data.settings["site"]["refresh_minutes"])
    scope, summary = report.get("scope") or {}, report.get("summary") or {}
    competitions = "、".join(data.comp(code) for code in scope.get("leagues", [])) or "未知"
    scope_html = (f"<p class='small'>当前模型 {e(scope.get('model') or '未知')} / {e(scope.get('model_version') or '未知')}；"
                  f"赛事 {e(competitions)}。比赛范围 {stamp(scope.get('start_utc'))} 至 {stamp(scope.get('end_utc'))}。</p>"
                  f"<p class='small'>启用时间 {stamp(scope.get('enabled_since_utc'))}；证据快照 "
                  f"{stamp(report.get('generated_at_utc'))}。页面时间均为北京时间。</p>")
    counters = [("match_count", "本次范围内比赛"), ("archived_complete_count", "三时点有效存档齐备"),
                ("scored_complete_count", "三时点评分齐备"), ("closing_complete_count", "四组收盘证据齐备"),
                ("complete_on_time_count", "按时且评分/收盘证据齐备")]
    stats = "<div class='stats'>" + "".join(
        f"<div class='stat'><b>{number(summary.get(key))}</b><span>{e(label)}</span></div>" for key, label in counters) + "</div>"
    counts = [("recorded_slot_count", "已存档"), ("late_slot_count", "迟到"), ("pending_slot_count", "等待中"),
              ("overdue_slot_count", "到点超宽限未存档"), ("missed_slot_count", "已错过窗口"),
              ("invalid_slot_count", "存档无效"), ("source_missing_count", "来源缺失比赛")]
    count_text = " · ".join(f"{label} {number(summary.get(key))}" for key, label in counts)
    note = (f"<div class='note'>到点后 {number(scope.get('grace_minutes'))} 分钟的宽限只用于等待调度和判漏，"
            "实际迟到仍单独显示。未来时点保留等待状态；预览版、旧开球时间和其他模型版本不补正式时点。"
            "三时点存档齐备、评分齐备和收盘证据齐备分别统计；首发链路尚未验收。</div>")
    matches = report["matches"]
    body = (heading + scope_html + stats + f"<p class='small'>{count_text}</p>" + note
            + "<p class='small'><a href='lifecycle.json'>下载本次逐场证据快照（JSON）</a></p>"
            + issues_html(report.get("warnings"))
            + (overview(data, matches) + "".join(match_evidence(data, row) for row in matches) if matches
               else placeholder("本次范围内没有可验收比赛；尚无匹配证据。")))
    return page("逐场验收 - 足球预测", body, "", data.now, data.settings["site"]["refresh_minutes"])
