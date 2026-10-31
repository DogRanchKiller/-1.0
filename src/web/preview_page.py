"""预测页预览版的网页（需求文档第 8 节“早期简易页”）。只负责把 src/predict/preview.py 算好的结果排成网页。

生成的是一个本机的静态文件（data/reports/preview.html），不开网站服务，别的电脑访问不到；
没有详情页、动画、自动刷新（第 8 节：这些等第 6 阶段的正式网页）。
网页里的队名、公司名等文字都先转义（e 函数），防止里面的 < > & 之类的符号把网页弄乱。
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from src.clean.dongqiudi_reader import BEIJING

WEEKDAYS = "一二三四五六日"
OUTCOMES = ("主胜", "平", "客胜")
STATUS_TAGS = {"postponed": "改期", "cancelled": "取消", "abandoned": "中断", "awarded": "判定结果"}
# 亚盘盘口的中文叫法（按主队让球的绝对值；0.25 一档）
AH_NAMES = ["平手", "平半", "半球", "半一", "一球", "一球球半", "球半", "球半两球", "两球", "两球两球半", "两球半",
            "两球半三球", "三球"]

CSS = """
body { font-family: "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2d1f; background: #f4f8f4; }
h1 { color: #1b5e20; } h2 { color: #2e7d32; margin-top: 28px; }
table { border-collapse: collapse; background: #fff; margin: 8px 0; }
th, td { border: 1px solid #c8dcc8; padding: 6px 10px; text-align: left; font-size: 14px; vertical-align: top; }
th { background: #e3efe3; }
.small { color: #607060; font-size: 13px; } .muted { color: #9aa89a; }
.note { background: #fff8e1; border: 1px solid #ffcc80; padding: 10px 14px; margin: 12px 0; font-size: 14px; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 12px; margin-left: 4px; }
.tag.temp { background: #fff3e0; color: #e65100; border: 1px solid #ffb74d; }
.tag.warn { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
.gap { color: #e65100; font-weight: bold; }
.bar { display: flex; width: 150px; height: 8px; margin-top: 4px; border-radius: 4px; overflow: hidden; }
.bar span:nth-child(1) { background: #43a047; } .bar span:nth-child(2) { background: #bdbdbd; }
.bar span:nth-child(3) { background: #1e88e5; }
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
    return f"{nums}<div class='bar'>{bar}</div>"


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
    model = "" if model_share is None else f"<br><b>模型 {pct(model_share)} / {pct(1 - model_share)}</b>"
    return f"{head}<br><span class='small'>{body}，{m['n_books']} 家{model}</span>"


def backtest_text(info: dict, label: str) -> str:
    """模型的回测成绩那一句。settings.yaml 里没写回测成绩时不写这一句。"""
    if not all(info.get(k) for k in ("backtest_n", "backtest_model", "backtest_pinnacle")):
        return ""
    return (f"在 2005–2023 赛季五大联赛有平博收盘赔率的 {int(info['backtest_n']):,} 场比赛上回测，"
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


def row_html(r: dict, threshold: float) -> str:
    tag = STATUS_TAGS.get(r.get("status"))
    tag_html = f"<span class='tag warn'>{e(tag)}</span>" if tag else ""
    match = (f"<span title='{e(r['home_team'])} 对 {e(r['away_team'])}'>{e(r['home_zh'])} 对 {e(r['away_zh'])}</span>"
             f"{tag_html}")
    m = r["market"]
    lines = r.get("model_lines") or {}
    if m["p_1x2"] is None:
        market = "<span class='muted'>还没有赔率</span>"
    else:
        market = f"{probs_cell(m['p_1x2'])}<span class='small'>{m['n_books_1x2']} 家平均</span>"
    return (f"<tr><td>{e(time_cell(r))}</td><td>{e(r['league'])}</td><td>{match}</td>"
            f"<td>{probs_cell(r['model_p'])}</td><td>{market}</td><td>{gap_cell(r['gap'], threshold)}</td>"
            f"<td>{line_cell(m['ah'], 'ah', lines.get('ah_home'))}</td>"
            f"<td>{line_cell(m['ou'], 'ou', lines.get('ou_over'))}</td></tr>")


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
    head = ("<tr><th>开球<br>（北京时间）</th><th>联赛</th><th>比赛（主 对 客）</th>"
            f"<th>模型 {e(label)}<span class='tag temp'>临时模型</span><br>主胜 / 平 / 客胜</th>"
            "<th>赔率概率<br>主胜 / 平 / 客胜</th><th>差距</th><th>亚盘主盘口</th><th>大小球主盘口</th></tr>")
    body = "".join(f"<h2>{e(title)}</h2><table>{head}{''.join(row_html(r, th) for r in grp)}</table>"
                   for (_k, title), grp in sorted(groups.items()))
    if not rows:
        body = "<p>未来几天没有比赛（或者赛程还没导入）。</p>"
    changes = [r["market"]["last_change_utc"] for r in rows if r["market"]["last_change_utc"]]
    last_odds = to_bj(max(changes)) if changes else None
    n_model = sum(r["model_p"] is not None for r in rows)
    n_market = sum(r["market"]["p_1x2"] is not None for r in rows)
    n_gap = sum(r["gap"] is not None for r in rows)
    hist_last = result["hist_last_date"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>足球预测 预览版</title><style>{CSS}</style></head><body>
<h1>五大联赛未来 {result['days_ahead']} 天 · 预测预览版</h1>
<p class="small">生成时间 {e(f"{now.astimezone(BEIJING):%Y-%m-%d %H:%M}")}（北京时间）。
共 {len(rows)} 场：模型算出概率 {n_model} 场，有赔率 {n_market} 场，差距大 {n_gap} 场。
这一页只是本机的一个文件，不连网，不会自动刷新；要看最新的，重新双击“查看预测预览.bat”。</p>
<div class="note"><b>只给自己看，不是下注建议。</b>页面只显示“模型怎么看”和“赔率怎么看”，不推荐买哪一边。<br>
<b>模型是临时的：</b>现在用 {e(label)}{f"（{e(info['desc'])}）" if info.get("desc") else ""}，不知道伤停、轮换、首发、天气。
{backtest_text(info, label)}<br>
<b>比分截至 {e(hist_last)}：</b>模型用到这一天为止的比分（{result['hist_n']:,} 场）。
比分以 football-data.co.uk 为准，它每周只更新两三次；它还没更新的最近比赛（这次 {result.get('hist_n_dongqiudi', 0)} 场）
先用懂球帝赛程里的比分，懂球帝赛程每 6 小时抓一次，所以一般赛后半天内就能算进去。</div>
<p class="small">模型版本 {e(result['model_version'])}；预测存进了预测存档（只加不删，以后复盘用）。
赔率最后一次变动：{e(f"{last_odds:%m-%d %H:%M}") if last_odds else "没有"}（北京时间）。</p>
{body}
<h2>怎么看这张表</h2>
<p class="small">
“赔率概率”：懂球帝上各家公司现在的胜平负赔率，每家去掉抽水（按比例）换成概率，再几家取平均。<br>
“差距”：模型和赔率概率在某个结果上差 {th * 100:.0f} 个百分点以上就标出来，只说明两边看法不同，不说明谁对。
{th * 100:.0f} 个百分点这个界限是先定的，还没有用数据验证过。<br>
“亚盘主盘口”“大小球主盘口”：各家公司开的盘口不一定一样，取开的公司最多的那条；下面是这几家的平均赔率（含本金），
和去掉抽水后两边的概率。平手、整数球和四分之一球有走盘（退钱）或赢一半，这个概率只是“不算走盘时哪边赢的机会”。
{lines_text(has_lines)}<br>
“时间待定”：网站还没公布具体开球时间，日期是“这一轮大约从哪天开始”。鼠标停在队名上能看到英文名。
</p>
<p class="small">数据来源：赔率来自懂球帝（dongqiudi.com），比分来自 football-data.co.uk，最近几天的比分来自懂球帝。只供个人学习，不转发。</p>
</body></html>
"""
