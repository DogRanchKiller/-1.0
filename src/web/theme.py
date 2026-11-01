"""本机网页共用的样子（需求文档第 8 节“风格”）：深色底 + 绿色，电脑、手机都能看。

预测预览页（src/web/preview_page.py）和数据状态页（src/jobs/status_page.py）都用这里的 BASE_CSS，
各页自己特有的样子再加在后面。以后第 6 阶段做正式网页，也从这里接着改，保证几页看起来是一套。

几个词：
- CSS：写“网页长什么样”的规则（颜色、字号、间距），和网页内容分开写。
- 颜色写成变量（--green 这种），想整体换颜色时只改最上面的 :root 一处。
- @media (max-width: 760px)：屏幕窄于 760 像素（手机）时另用的规则。
- 只用样式，不用脚本（JavaScript）：页面是本机静态文件，不需要脚本，也更安全。
"""

# 网页 <head> 里都要加的一行：告诉手机浏览器按手机屏幕宽度排版，不要把电脑版缩小显示
VIEWPORT = '<meta name="viewport" content="width=device-width, initial-scale=1">'

BASE_CSS = """
:root {
  color-scheme: dark;
  --bg: #0a1310;        /* 页面底色：接近黑的墨绿 */
  --card: #101c16;      /* 卡片、表格底色 */
  --card-2: #142219;    /* 表头、强调块 */
  --line: #1d3027;      /* 分隔线 */
  --text: #e2ebe5;      /* 正文 */
  --sub: #94a99d;       /* 次要文字 */
  --faint: #5f7468;     /* 更淡的文字（没有数据时） */
  --green: #34d399;     /* 主色 */
  --green-2: #10b981;
  --lime: #bef264;
  --amber: #fbbf24;     /* 提醒 */
  --red: #f87171;       /* 出错 */
  --blue: #60a5fa;      /* 客胜 */
  --draw: #64776c;      /* 平局 */
  --radius: 14px;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; padding: 32px clamp(14px, 4vw, 48px) 48px;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
  font-size: 15px; line-height: 1.6; color: var(--text);
  background: radial-gradient(1100px 520px at 0% -10%, rgba(16, 185, 129, .16), transparent 70%),
              radial-gradient(900px 500px at 100% 0%, rgba(52, 211, 153, .07), transparent 70%), var(--bg);
  background-attachment: fixed;
  font-variant-numeric: tabular-nums;   /* 数字等宽，竖着对得齐 */
}
.wrap { max-width: 1280px; margin: 0 auto; }
h1 { font-size: clamp(22px, 3vw, 30px); font-weight: 650; margin: 0 0 6px; letter-spacing: .5px; }
h1 .accent { color: var(--green); }
h2 { font-size: 17px; font-weight: 600; color: var(--green); margin: 34px 0 12px;
     display: flex; align-items: center; gap: 10px; }
h2::before { content: ""; width: 4px; height: 18px; border-radius: 2px;
             background: linear-gradient(var(--lime), var(--green-2)); }
a { color: var(--green); }
.small { color: var(--sub); font-size: 13px; }
.muted { color: var(--faint); }

/* 卡片：一块有圆角、细边框的区域 */
.card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
        box-shadow: 0 8px 28px rgba(0, 0, 0, .35); }
.scroll { overflow-x: auto; }           /* 表格太宽时，在卡片里左右滑动 */

/* 表格 */
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { background: var(--card-2); color: var(--sub); font-weight: 500; font-size: 12.5px; text-align: left;
     padding: 11px 14px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 12px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }
tr:last-child td { border-bottom: none; }
tbody tr { transition: background .2s; }
tbody tr:hover { background: rgba(52, 211, 153, .05); }
.card > table, .card > .scroll { border-radius: var(--radius); overflow: hidden; }

/* 小标签 */
.tag { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; margin-left: 6px;
       border: 1px solid; vertical-align: 1px; white-space: nowrap; }
.tag.temp { color: var(--amber); border-color: rgba(251, 191, 36, .4); background: rgba(251, 191, 36, .08); }
.tag.warn { color: var(--red); border-color: rgba(248, 113, 113, .4); background: rgba(248, 113, 113, .08); }

/* 顶部的几个数字 */
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0 6px; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 10px 16px;
        min-width: 110px; }
.stat b { display: block; font-size: 22px; font-weight: 650; color: var(--text); }
.stat span { color: var(--sub); font-size: 12.5px; }
.stat.hi b { color: var(--green); }

/* 说明块 */
.note { background: var(--card); border: 1px solid var(--line); border-left: 3px solid var(--amber);
        border-radius: 10px; padding: 14px 18px; margin: 18px 0; font-size: 14px; color: var(--sub); }
.note b { color: var(--text); }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line); }

@media (max-width: 760px) {
  body { padding-top: 20px; font-size: 14px; }
  td, th { padding: 10px 12px; }
  .stat { flex: 1 1 calc(50% - 12px); min-width: 0; }
}
"""

# ---------- 正式网站（src/web/site/，工作树 K3 加）：只加新样式，上面 BASE_CSS 原样不动 ----------
# 几页共用：顶部导航、可以展开的“详细”块（<details>，不用脚本也能点开收起）、热力图和折线图的外框、单场页的版本对比
SITE_CSS = """
nav.top { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center; margin-bottom: 18px; font-size: 14px; }
nav.top a { text-decoration: none; color: var(--sub); padding: 4px 12px; border-radius: 999px; border: 1px solid var(--line); }
nav.top a:hover { color: var(--green); border-color: rgba(52, 211, 153, .45); }
nav.top .brand { color: var(--green); font-weight: 650; border: none; padding-left: 0; }
details.more { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); margin: 14px 0;
               padding: 0 18px; }
details.more > summary { cursor: pointer; padding: 13px 0; color: var(--green); font-weight: 600; list-style: none; }
details.more > summary::-webkit-details-marker { display: none; }
details.more > summary::before { content: "▸ "; color: var(--sub); }
details.more[open] > summary::before { content: "▾ "; }
details.more[open] > summary { border-bottom: 1px solid var(--line); margin-bottom: 12px; }
details.more > .inner { padding-bottom: 16px; }
.points { margin: 8px 0 0; padding-left: 20px; }
.points li { margin: 4px 0; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.pad { padding: 16px 18px; }
.chart { width: 100%; height: auto; display: block; }
.chart text { fill: var(--sub); font-size: 12px; font-family: inherit; }
.chart .cell-text { fill: var(--text); font-size: 11.5px; }
.chart .cell-text.dark { fill: #062016; font-weight: 600; }
.chart .axis { stroke: var(--line); stroke-width: 1; }
.placeholder { color: var(--faint); border: 1px dashed var(--line); border-radius: 10px; padding: 18px; text-align: center; }
.num { text-align: right; white-space: nowrap; }
.score { font-weight: 650; color: var(--lime); white-space: nowrap; }
.hit { color: var(--green); } .miss { color: var(--red); }
td.match a, td a.match { color: var(--text); text-decoration: none; font-weight: 600; }
td.match a:hover, td a.match:hover { color: var(--green); }
dl.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 6px 0; font-size: 14px; }
dl.kv dt { color: var(--sub); } dl.kv dd { margin: 0; }
@media (max-width: 760px) { dl.kv { grid-template-columns: 1fr; } dl.kv dd { margin-bottom: 6px; } }
.legend { display: inline-flex; gap: 14px; flex-wrap: wrap; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
"""
