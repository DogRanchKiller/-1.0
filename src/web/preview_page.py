"""预测页预览版的网页（需求文档第 8 节“早期简易页”）。只负责把 src/predict/preview.py 算好的结果排成网页。

生成的是一个本机的静态文件（data/reports/preview.html），不开网站服务，别的电脑访问不到；
没有详情页、动画、自动刷新（第 8 节：这些等第 6 阶段的正式网页）。
样子（深色底、绿色、手机上一场一张卡片）用 src/web/theme.py 的共用样式。
网页里的队名、公司名等文字都先转义（e 函数），防止里面的 < > & 之类的符号把网页弄乱。
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from src.clean.dongqiudi_reader import BEIJING
from src.web.theme import BASE_CSS, VIEWPORT

WEEKDAYS = "一二三四五六日"
OUTCOMES = ("主胜", "平", "客胜")
STATUS_TAGS = {"postponed": "改期", "cancelled": "取消", "abandoned": "中断", "awarded": "判定结果"}
# 亚盘盘口的中文叫法（按主队让球的绝对值；0.25 一档）
AH_NAMES = ["平手", "平半", "半球", "半一", "一球", "一球球半", "球半", "球半两球", "两球", "两球两球半", "两球半",
            "两球半三球", "三球"]

# 这一页特有的样子；共用的颜色、字体、表格、卡片在 src/web/theme.py
CSS = BASE_CSS + """
.bar { display: flex; width: 100%; max-width: 180px; height: 6px; margin-top: 6px; border-radius: 3px;
       overflow: hidden; background: var(--line); gap: 2px; }
.bar span { transition: width .6s ease; }
.bar span:nth-child(1) { background: var(--green); } .bar span:nth-child(2) { background: var(--draw); }
.bar span:nth-child(3) { background: var(--blue); }
.legend { display: inline-flex; gap: 14px; flex-wrap: wrap; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.probs { white-space: nowrap; }
td.match { font-weight: 600; font-size: 15px; }
td.time { white-space: nowrap; color: var(--text); }
.gap { display: inline-block; color: var(--lime); background: rgba(190, 242, 100, .1); font-weight: 600;
       font-size: 13px; padding: 3px 10px; border-radius: 8px; border: 1px solid rgba(190, 242, 100, .35); }
tr.has-gap td:first-child { box-shadow: inset 3px 0 0 var(--lime); }
.model-line { color: var(--green); font-weight: 600; }
td.time, td.league { white-space: nowrap; }
/* 屏幕窄（手机、平板、窄窗口）时：表格变成一场一张卡片，两项并排；每一项上面写这一项叫什么（data-label） */
@media (max-width: 1100px) {
  .card:has(table.matches) { background: none; border: none; box-shadow: none; }
  table.matches thead { display: none; }
  table.matches, table.matches tbody { display: block; }
  table.matches tr { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px 18px;
                     background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
                     padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 6px 20px rgba(0, 0, 0, .3); }
  table.matches tr:hover { background: var(--card); }
  table.matches td { display: block; border: none; padding: 0; min-width: 0; }
  table.matches td::before { content: attr(data-label); display: block; color: var(--faint); font-size: 11.5px;
                             letter-spacing: .5px; margin-bottom: 2px; }
  table.matches td.match { grid-column: 1 / -1; order: -2; font-size: 17px; padding-bottom: 8px;
                           border-bottom: 1px solid var(--line); }
  table.matches td.match::before { display: none; }
  table.matches td.time, table.matches td.league { order: -1; }
  tr.has-gap td:first-child { box-shadow: none; }
  table.matches tr.has-gap { border-color: rgba(190, 242, 100, .45); box-shadow: inset 3px 0 0 var(--lime), 0 6px 20px rgba(0, 0, 0, .3); }
  .bar { max-width: none; }
}
/* 手机：每行两项 */
@media (max-width: 600px) {
  table.matches tr { grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 14px; }
}
"""


def e(x) -> str:
    """把任何值转成可以安全放进网页的文字。"""
    return html.escape("" if x is None else str(x))


def pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def to_bj(utc_text: str | None) -> datetime | None:
    """标准时间文字（yyyy-mm-dd HH:MM:SS）→ 北京时间。空的返回 None。"""
    if not utc_text:
        return None
    return pd.Timestamp(utc_text).tz_localize("UTC").tz_convert(BEIJING).to_pydatetime()


def ah_text(line: float) -> str:
    """亚盘盘口 → 中文叫法。line 是主队让球数，负数 = 主队让球。例：-0.5 → 主让半球；0.25 → 主受让平半。"""
    k = round(abs(line) * 4)
    name = AH_NAMES[k] if k < len(AH_NAMES) else f"{abs(line):g} 球"
    if k == 0:
        return name
    return ("主让" if line < 0 else "主受让") + name


def probs_cell(p) -> str:
    """三个概率 + 一条三色小条（绿 = 主胜，灰 = 平，蓝 = 客胜）。"""
    if p is None:
        return "<span class='muted'>没算出来</span>"
    nums = " / ".join(pct(x) for x in p)
    bar = "".join(f"<span style='width:{x * 100:.1f}%'></span>" for x in p)
    return f"<div><span class='probs'>{nums}</span><div class='bar'>{bar}</div></div>"


def gap_cell(g, threshold: float) -> str:
    if g is None:
        return "<span class='muted'>—</span>"
    word = "更看好" if g["diff"] > 0 else "更不看好"
    return f"<span class='gap'>模型{word}{OUTCOMES[g['outcome']]}，差 {abs(g['diff']) * 100:.0f} 个百分点</span>"


def line_cell(m: dict | None, kind: str, model_share: float | None = None) -> str:
    """让球 / 大小球：主盘口、几家平均赔率、去水后两边的概率；模型算得出时，再加一行模型在这条盘口上的胜算。"""
    if m is None:
        return "<span class='muted'>没有</span>"
    if kind == "ah":
        num = "0" if m["line"] == 0 else f"{m['line']:+g}"
        head = f"{e(ah_text(m['line']))}（{num}）"
        body = f"主 {m['home']:.2f} / 客 {m['away']:.2f}<br>赔率 {pct(m['p_home'])} / {pct(m['p_away'])}"
    else:
        head = f"{m['line']:g} 球"
        body = f"大 {m['over']:.2f} / 小 {m['under']:.2f}<br>赔率 {pct(m['p_over'])} / {pct(m['p_under'])}"
    model = ("" if model_share is None
             else f"<br><b class='model-line'>模型 {pct(model_share)} / {pct(1 - model_share)}</b>")
    return f"<div>{head}<br><span class='small'>{body}，{m['n_books']} 家{model}</span></div>"


def backtest_text(info: dict, label: str) -> str:
    """模型的回测成绩那一句。settings.yaml 里没写回测成绩时不写这一句。"""
    if not all(info.get(k) for k in ("backtest_n", "backtest_model", "backtest_pinnacle")):
        return ""
    # 回测用的赛季（settings 的 backtest_period）；没写就不写赛季，免得换了模型还显示旧模型的赛季
    period = f"{e(info['backtest_period'])} 赛季" if info.get("backtest_period") else ""
    return (f"在 {period}五大联赛有平博收盘赔率的 {int(info['backtest_n']):,} 场比赛上回测，"
            f"{e(label)} 的 log loss 是 {float(info['backtest_model']):.3f}，平博收盘赔率是 {float(info['backtest_pinnacle']):.3f}"
            f"（越小越准），也就是说模型还不如收盘赔率准。")


def lines_text(has_lines: bool) -> str:
    """“亚盘、大小球主盘口”那一列里，模型那一行怎么看。"""
    if not has_lines:
        return "模型这两项还算不出来（模型只给胜平负，要给比分概率的模型才算得出；或者这几场都还没开盘）。"
    return ("粗体的“模型”一行：同一条盘口上，模型算出的两边胜算（从每种比分的概率算，赢半、输半算一半，走盘不计），"
            "和上面赔率那一行是同一个意思，可以直接比。")


def day_key(r: dict) -> tuple[str, str]:
    """一场比赛放在哪一天下面：(排序用的日期, 显示的标题)。

    有开球时间的按北京时间的日期；只知道“这一轮从哪天开始”的放进“时间待定”（日期是往前挪过的，不是真的比赛日）。
    """
    if r["date_precision"] == "round":
        return "9999", "时间待定（网站还没公布具体时间）"
    t = to_bj(r["kickoff_utc"])
    d = t.date() if t else pd.Timestamp(r["match_date"]).date()
    return d.isoformat(), f"{d.month} 月 {d.day} 日 星期{WEEKDAYS[d.weekday()]}"


def time_cell(r: dict) -> str:
    if r["date_precision"] == "round":
        d = pd.Timestamp(r["match_date"])
        return f"约 {d.month:02d}-{d.day:02d} 起"
    t = to_bj(r["kickoff_utc"])
    return t.strftime("%H:%M") if t else "未定"


def row_html(r: dict, threshold: float, label: str) -> str:
    tag = STATUS_TAGS.get(r.get("status"))
    tag_html = f"<span class='tag warn'>{e(tag)}</span>" if tag else ""
    match = (f"<span title='{e(r['home_team'])} 对 {e(r['away_team'])}'>{e(r['home_zh'])} 对 {e(r['away_zh'])}</span>"
             f"{tag_html}")
    m = r["market"]
    lines = r.get("model_lines") or {}
    if m["p_1x2"] is None:
        market = "<span class='muted'>还没有赔率</span>"
    else:
        market = f"<div>{probs_cell(m['p_1x2'])}<span class='small'>{m['n_books_1x2']} 家平均</span></div>"
    # data-label：手机上表格变成卡片时，写在每一项前面的名字（见 CSS 里的 @media）
    cells = [("开球", "time", e(time_cell(r))), ("联赛", "league", e(r["league"])), ("比赛", "match", match),
             (f"模型 {label}", "", probs_cell(r["model_p"])), ("赔率概率", "", market),
             ("差距", "", gap_cell(r["gap"], threshold)),
             ("亚盘", "", line_cell(m["ah"], "ah", lines.get("ah_home"))),
             ("大小球", "", line_cell(m["ou"], "ou", lines.get("ou_over")))]
    tds = "".join(f"<td data-label='{e(name)}'{f' class={cls}' if cls else ''}>{body}</td>" for name, cls, body in cells)
    return f"<tr{' class=has-gap' if r['gap'] is not None else ''}>{tds}</tr>"


def render(result: dict, settings: dict, now: datetime) -> str:
    """整页网页（HTML 文字）。"""
    rows = result["rows"]
    th = result["gap_threshold"]
    groups: dict[tuple[str, str], list] = {}
    for r in rows:
        groups.setdefault(day_key(r), []).append(r)
    # 模型的名字、一句话说明、回测成绩都在 settings.yaml 的 preview.model_info，换模型时改那里
    info = settings["preview"].get("model_info") or {}
    label = info.get("label") or result["model"]
    has_lines = any((r.get("model_lines") or {}).get(k) is not None for r in rows for k in ("ah_home", "ou_over"))
    head = ("<thead><tr><th>开球<br>（北京时间）</th><th>联赛</th><th>比赛（主 对 客）</th>"
            f"<th>模型 {e(label)}<span class='tag temp'>临时模型</span><br>主胜 / 平 / 客胜</th>"
            "<th>赔率概率<br>主胜 / 平 / 客胜</th><th>差距</th><th>亚盘主盘口</th><th>大小球主盘口</th></tr></thead>")
    # 每一天一张卡片；卡片里是一张表，手机上每一行变成一张小卡片
    body = "".join(f"<h2>{e(title)}<span class='small'>{len(grp)} 场</span></h2>"
                   f"<div class='card'><table class='matches'>{head}"
                   f"<tbody>{''.join(row_html(r, th, label) for r in grp)}</tbody></table></div>"
                   for (_k, title), grp in sorted(groups.items()))
    if not rows:
        body = "<div class='note'>未来几天没有比赛（或者赛程还没导入）。</div>"
    changes = [r["market"]["last_change_utc"] for r in rows if r["market"]["last_change_utc"]]
    last_odds = to_bj(max(changes)) if changes else None
    n_model = sum(r["model_p"] is not None for r in rows)
    n_market = sum(r["market"]["p_1x2"] is not None for r in rows)
    n_gap = sum(r["gap"] is not None for r in rows)
    hist_last = result["hist_last_date"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
{VIEWPORT}
<title>足球预测 预览版</title><style>{CSS}</style></head><body><div class="wrap">
<h1>五大联赛未来 {result['days_ahead']} 天 · <span class="accent">预测预览版</span></h1>
<p class="small">生成时间 {e(f"{now.astimezone(BEIJING):%Y-%m-%d %H:%M}")}（北京时间）。
共 {len(rows)} 场：模型算出概率 {n_model} 场，有赔率 {n_market} 场，差距大 {n_gap} 场。
这一页只是本机的一个文件，不连网，不会自动刷新；要看最新的，重新双击“查看预测预览.bat”。</p>
<div class="stats"><div class="stat"><b>{len(rows)}</b><span>场比赛</span></div>
<div class="stat"><b>{n_model}</b><span>模型算出</span></div>
<div class="stat"><b>{n_market}</b><span>有赔率</span></div>
<div class="stat hi"><b>{n_gap}</b><span>差距大</span></div></div>
<p class="small legend"><span><i style="background:var(--green)"></i>主胜</span>
<span><i style="background:var(--draw)"></i>平</span><span><i style="background:var(--blue)"></i>客胜</span>
<span><i style="background:var(--lime)"></i>模型和赔率差距大（亮绿边）</span></p>
<div class="note"><b>只给自己看，不是下注建议。</b>页面只显示“模型怎么看”和“赔率怎么看”，不推荐买哪一边。<br>
<b>模型是临时的：</b>现在用 {e(label)}{f"（{e(info['desc'])}）" if info.get("desc") else ""}，不知道伤停、轮换、首发、天气。
{backtest_text(info, label)}<br>
<b>比分截至 {e(hist_last)}：</b>模型用到这一天为止的比分（{result['hist_n']:,} 场）。
比分以 football-data.co.uk 为准，它每周只更新两三次；它还没更新的最近比赛（这次 {result.get('hist_n_dongqiudi', 0)} 场）
先用懂球帝赛程里的比分，懂球帝赛程每 6 小时抓一次，所以一般赛后半天内就能算进去。</div>
<p class="small">模型版本 {e(result['model_version'])}；预测存进了预测存档（只加不删，以后复盘用）。
赔率最后一次变动：{e(f"{last_odds:%m-%d %H:%M}") if last_odds else "没有"}（北京时间）。</p>
{body}
<h2>怎么看这张表</h2><div class="note">
“赔率概率”：懂球帝上各家公司现在的胜平负赔率，每家去掉抽水（按比例）换成概率，再几家取平均。<br>
“差距”：模型和赔率概率在某个结果上差 {th * 100:.0f} 个百分点以上就标出来，只说明两边看法不同，不说明谁对。
{th * 100:.0f} 个百分点这个界限是先定的，还没有用数据验证过。<br>
“亚盘主盘口”“大小球主盘口”：各家公司开的盘口不一定一样，取开的公司最多的那条；下面是这几家的平均赔率（含本金），
和去掉抽水后两边的概率。平手、整数球和四分之一球有走盘（退钱）或赢一半，这个概率只是“不算走盘时哪边赢的机会”。
{lines_text(has_lines)}<br>
“时间待定”：网站还没公布具体开球时间，日期是“这一轮大约从哪天开始”。鼠标停在队名上能看到英文名。
</div>
<footer class="small">数据来源：赔率来自懂球帝（dongqiudi.com），比分来自 football-data.co.uk，最近几天的比分来自懂球帝。只供个人学习，不转发。</footer>
</div></body></html>
"""
