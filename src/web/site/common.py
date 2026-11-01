"""正式网站几页共用的小工具：北京时间、百分比、安全的文件名、页面外框、内嵌 SVG 图。

几个词：
- 相对链接：页面之间用 "match/123.html"、"../index.html" 这种写法互相链接，不写 C:\\... 这种完整路径，
  所以整个 data/site 文件夹搬到哪里、双击 index.html 打开都能点。
- SVG：一种用文字描述图形的格式（“在这里画一个方块、那里画一条线”），可以直接写在网页里，不用图片文件、不用脚本。
  热力图和赔率变动图都是这里用 Python 拼出来的 SVG。
- 转义（e 函数）：队名、公司名里可能有 < > & ' 这种符号，直接放进网页会把网页弄乱，先换成网页认得的写法。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from urllib.parse import quote

import numpy as np

from src.clean.dongqiudi_reader import BEIJING
from src.web.preview_page import e, pct, to_bj   # 和预览页共用：转义、百分比、标准时间 → 北京时间
from src.web.theme import BASE_CSS, SITE_CSS, VIEWPORT

OUTCOMES = ("主胜", "平", "客胜")
WEEKDAYS = "一二三四五六日"
STATUS_TAGS = {"postponed": "改期", "cancelled": "取消", "abandoned": "中断", "awarded": "判定结果"}
# 信心等级的叫法和回测（src/evaluate/confidence_study.py 的 LEVEL_NAMES）一样，网页这边不另外定
LEVEL_NAMES = {3: "●●● 高", 2: "●●○ 中", 1: "●○○ 低"}
# Windows 文件名里不能用的名字（不管后面加什么扩展名都不行）
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


# ---------- 数字、时间 ----------
def pct1(p) -> str:
    """概率 → 带一位小数的百分比，例 0.4567 → 45.7%。空的写“—”。"""
    return "—" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p * 100:.1f}%"


def pct0(p) -> str:
    """概率 → 整数百分比，空的写“—”。"""
    return "—" if p is None or (isinstance(p, float) and np.isnan(p)) else pct(p)


def bj_text(utc_text, fmt: str = "%m-%d %H:%M") -> str:
    """标准时间文字 → 北京时间文字。空的写“—”。"""
    t = to_bj(utc_text) if utc_text else None
    return t.strftime(fmt) if t else "—"


def kickoff_text(info: dict, with_date: bool = True) -> str:
    """一场比赛的开球时间（北京时间）。只知道“这一轮从哪天开始”的写“约 11-09 起”。"""
    if info.get("date_precision") == "round" or not info.get("kickoff_utc"):
        d = str(info.get("match_date") or "")[:10]
        return f"约 {d[5:]} 起" if info.get("date_precision") == "round" else (d[5:] or "未定")
    t = to_bj(info["kickoff_utc"])
    return f"{t:%m-%d} 星期{WEEKDAYS[t.weekday()]} {t:%H:%M}" if with_date else f"{t:%H:%M}"


def day_title(info: dict) -> tuple[str, str]:
    """首页按哪一天分组：(排序用的键, 标题)。时间待定的放最后一组。"""
    if info.get("date_precision") == "round" or not info.get("kickoff_utc"):
        return "9999", "时间待定（网站还没公布具体开球时间）"
    d = to_bj(info["kickoff_utc"]).date()
    return d.isoformat(), f"{d.month} 月 {d.day} 日 星期{WEEKDAYS[d.weekday()]}"


def level_name(level) -> str:
    return LEVEL_NAMES.get(level, "—") if level is not None else "—"


# ---------- 安全的文件名 ----------
def safe_name(text: str) -> str:
    """球队名 → 能当 Windows 文件名的一段文字。

    规则：带音调的字母换成不带的（Atlético → Atletico）；只留字母、数字、减号、下划线、点，其他（空格、' / \\ : * ? " < > |）
    都换成下划线；去掉开头结尾的点和下划线；CON、NUL 这种 Windows 保留名后面加下划线；空了就用“team”；最长 60 个字。
    """
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", t).strip("._")[:60].strip("._")
    if not t:
        t = "team"
    if t.split(".")[0].upper() in _RESERVED:
        t += "_"
    return t


class TeamFiles:
    """每支球队的页面文件名。两支队换成同一个文件名时（例如只差一个符号），后来的加上一小段编号，保证不撞。"""

    def __init__(self, teams):
        self.names: dict[str, str] = {}
        used: set[str] = set()
        for team in sorted(set(teams)):
            name = safe_name(team)
            if name.lower() in used:       # Windows 文件名不分大小写，按小写查重
                name = f"{name}_{hashlib.sha1(team.encode('utf-8')).hexdigest()[:6]}"
            used.add(name.lower())
            self.names[team] = name

    def file(self, team: str) -> str | None:
        """team/ 文件夹里的文件名（例 Man_United.html）；没有这支队的页面返回 None。"""
        return f"{self.names[team]}.html" if team in self.names else None

    def link(self, team: str, label: str, prefix: str) -> str:
        """指向球队页的链接；prefix 是从当前页面到网站根目录的路径（首页 ""，子文件夹里的页面 "../"）。没有页面就只写名字。"""
        f = self.file(team)
        if f is None:
            return e(label)
        return f"<a href='{e(prefix + 'team/' + quote(f))}'>{e(label)}</a>"


def match_href(match_id: int, prefix: str) -> str:
    return e(f"{prefix}match/{int(match_id)}.html")


# ---------- 页面外框 ----------
NOT_ADVICE = ("<b>只给自己看，不是下注建议。</b>页面只写“哪边更可能发生”和“有多可信”：赔率已经把可能性算进去了，"
              "再扣掉抽水，多数“倾向”都没有利润空间，“倾向”不等于“值得买”。")


def page(title: str, body: str, prefix: str, now: datetime, refresh_minutes: int) -> str:
    """一整页网页。prefix：从这一页到网站根目录的相对路径（首页 ""，match/、team/ 里的页面 "../"）。

    页面开着时每 refresh_minutes 分钟浏览器自己重新加载一次（<meta http-equiv="refresh">，不是脚本），
    定时任务也是每 15 分钟重新生成一次，所以开着的页面会跟着变。
    """
    nav = (f"<nav class='top'><span class='brand'>足球预测</span><a href='{e(prefix)}index.html'>即将开赛</a>"
           f"<a href='{e(prefix)}record.html'>战绩</a></nav>")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
{VIEWPORT}
<meta http-equiv="refresh" content="{int(refresh_minutes) * 60}">
<title>{e(title)}</title><style>{BASE_CSS}{SITE_CSS}</style></head><body><div class="wrap">
{nav}
{body}
<footer class="small">生成时间 {e(f"{now.astimezone(BEIJING):%Y-%m-%d %H:%M}")}（北京时间）。这是本机的静态文件，不连网；
数据来源：赔率来自懂球帝（dongqiudi.com），比分来自 football-data.co.uk 和懂球帝。只供个人学习，不转发。</footer>
</div></body></html>
"""


def details(title: str, inner: str, open_: bool = False) -> str:
    """“详细”一层：点标题展开、再点收起（<details>，浏览器自带，不用脚本）。"""
    return (f"<details class='more'{' open' if open_ else ''}><summary>{e(title)}</summary>"
            f"<div class='inner'>{inner}</div></details>")


def placeholder(text: str) -> str:
    return f"<div class='placeholder'>{e(text)}</div>"


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    """一张表。head 是表头文字（会转义）；rows 里每格是已经排好的 HTML（调用的地方负责转义）。
    某一格要带样式时，直接给一整格 "<td class='match'>...</td>"，这里就不再包一层。"""
    th = "".join(f"<th>{e(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='card'><div class='scroll'><table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div></div>"


# ---------- 内嵌 SVG：比分热力图 ----------
def fold_matrix(m: np.ndarray, n: int) -> np.ndarray:
    """比分矩阵并成 (n + 2) × (n + 2)：0 到 n 球各一格，最后一格是“n + 1 球及以上”（第 3 节：热力图 0-0 到 5-5）。"""
    out = np.zeros((n + 2, n + 2))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            out[min(i, n + 1), min(j, n + 1)] += m[i, j]
    return out


def heatmap_svg(m: np.ndarray, n: int, title: str) -> str:
    """比分概率热力图：行 = 主队进几球，列 = 客队进几球，格子越亮概率越大，格子里写百分比。"""
    f = fold_matrix(np.asarray(m, float), n)
    k = n + 2
    cell, left, top = 46, 52, 44
    w, h = left + k * cell + 8, top + k * cell + 8
    peak = float(f.max()) or 1.0
    labels = [str(i) for i in range(n + 1)] + [f"{n + 1}+"]
    parts = [f"<svg class='chart' viewBox='0 0 {w} {h}' role='img' aria-label='{e(title)}'>",
             f"<text x='{left + k * cell / 2}' y='14' text-anchor='middle'>客队进球 →</text>",
             f"<text x='14' y='{top + k * cell / 2}' text-anchor='middle' transform='rotate(-90 14 {top + k * cell / 2})'>主队进球 →</text>"]
    for j in range(k):
        parts.append(f"<text x='{left + j * cell + cell / 2}' y='{top - 8}' text-anchor='middle'>{labels[j]}</text>")
    for i in range(k):
        parts.append(f"<text x='{left - 10}' y='{top + i * cell + cell / 2 + 4}' text-anchor='end'>{labels[i]}</text>")
        for j in range(k):
            p = float(f[i, j])
            a = 0.06 + 0.94 * (p / peak) ** 0.8          # 越大越亮；0.8 次方让小概率的格子也看得出深浅
            x, y = left + j * cell, top + i * cell
            parts.append(f"<rect x='{x + 1}' y='{y + 1}' width='{cell - 2}' height='{cell - 2}' rx='5' "
                         f"fill='#34d399' fill-opacity='{a:.3f}'><title>{labels[i]}-{labels[j]}：{p * 100:.2f}%</title></rect>")
            if p >= 0.0005:
                cls = "cell-text dark" if a > 0.55 else "cell-text"
                parts.append(f"<text class='{cls}' x='{x + cell / 2}' y='{y + cell / 2 + 4}' text-anchor='middle'>"
                             f"{p * 100:.1f}</text>")
    parts.append("</svg>")
    return "".join(parts)


# ---------- 内嵌 SVG：折线图（赔率变动） ----------
SERIES_COLORS = ("#34d399", "#94a3b8", "#60a5fa")   # 主胜绿、平灰、客胜蓝（和预览页的三色条一样）


def line_chart_svg(series: list[tuple[str, list[tuple[datetime, float]]]], marks: list[tuple[str, datetime]],
                   title: str) -> str:
    """阶梯折线图：每条线 = (名字, [(时间, 概率), ...])，价格在下一次变动之前保持不变，所以画成台阶。

    marks：竖的虚线，例如 ("开球前 24 小时", 时间)。横轴是北京时间，纵轴是概率（自动按数据的范围放大）。
    """
    pts = [p for _, s in series for p in s]
    if not pts:
        return ""
    times = [t for t, _ in pts] + [t for _, t in marks]
    t0, t1 = min(times), max(times)
    span = max((t1 - t0).total_seconds(), 3600.0)
    vals = [v for _, v in pts]
    lo, hi = max(0.0, min(vals) - 0.03), min(1.0, max(vals) + 0.03)
    if hi - lo < 0.1:
        mid = (hi + lo) / 2
        lo, hi = max(0.0, mid - 0.05), min(1.0, mid + 0.05)
    w, h, left, right, top, bottom = 760, 280, 52, 16, 16, 40

    def x(t):
        return left + (t - t0).total_seconds() / span * (w - left - right)

    def y(v):
        return top + (hi - v) / (hi - lo) * (h - top - bottom)

    parts = [f"<svg class='chart' viewBox='0 0 {w} {h}' role='img' aria-label='{e(title)}'>"]
    for k in range(5):                                       # 横的参考线和纵轴的百分比
        v = lo + (hi - lo) * k / 4
        parts.append(f"<line class='axis' x1='{left}' x2='{w - right}' y1='{y(v):.1f}' y2='{y(v):.1f}'/>"
                     f"<text x='{left - 6}' y='{y(v) + 4:.1f}' text-anchor='end'>{v * 100:.0f}%</text>")
    for k in range(5):                                       # 横轴的时间
        t = datetime.fromtimestamp(t0.timestamp() + span * k / 4, tz=t0.tzinfo)
        parts.append(f"<text x='{x(t):.1f}' y='{h - 14}' text-anchor='middle'>{t.astimezone(BEIJING):%m-%d %H:%M}</text>")
    for name, t in marks:
        parts.append(f"<line x1='{x(t):.1f}' x2='{x(t):.1f}' y1='{top}' y2='{h - bottom}' stroke='#fbbf24' "
                     f"stroke-opacity='.55' stroke-dasharray='4 4'/><text x='{x(t) + 3:.1f}' y='{top + 10}' "
                     f"style='fill:#fbbf24'>{e(name)}</text>")
    for idx, (name, s) in enumerate(series):
        if not s:
            continue
        s = sorted(s)
        xy = []
        for k, (t, v) in enumerate(s):
            if k:
                xy.append(f"{x(t):.1f},{y(s[k - 1][1]):.1f}")       # 台阶：先横着走到变动的时间，再竖着变
            xy.append(f"{x(t):.1f},{y(v):.1f}")
        xy.append(f"{x(t1):.1f},{y(s[-1][1]):.1f}")                 # 最后一个价格画到图的右边
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(xy)}'>"
                     f"<title>{e(name)}</title></polyline>")
    parts.append("</svg>")
    legend = "".join(f"<span><i style='background:{SERIES_COLORS[i % 3]}'></i>{e(n)}</span>" for i, (n, _) in enumerate(series))
    return "".join(parts) + f"<p class='small legend'>{legend}<span><i style='background:#fbbf24'></i>开球前 24 / 6 / 1 小时</span></p>"
