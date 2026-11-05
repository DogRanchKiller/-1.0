// 本站自己的脚本（需求文档第 8 节“前端脚本”、第 16 节“安全”）。只在“服务器现读现出”的网页上引用（/static/site.js）。
//
// 它做的事（都是在服务器生成好的网页上“加东西”，网页本身不靠它）：
//   1. 首页：按联赛、日期、有没有预测、信心等级筛选；按开球时间或概率排序；按球队名搜索。
//   2. 战绩页：每场的四个预测按玩法、联赛、中没中筛选。
//   3. 虚拟投注页：点表头排序；按价格、是不是对照组、名字筛选；点“按赛季累计盈亏”的小图看每个赛季的数。
//   4. 概率条、比分热力图：鼠标放上去（手机上点一下）显示具体的数。
//   5. 单场页：在“开球前 24 / 6 / 1 小时”几版之间切换，看概率怎么变（数据来自 /api/match/<编号>）。
//   6. 每 30 秒（settings.yaml 的 live.poll_seconds）问一次 /api/version 和 /api/live：
//      即时比分只改比分那一小格；数据版本变了，就把这一页重新取一次、只换掉正文（做法见下面“自动更新”一段）。
//
// 规则（第 16 节）：
//   - 没有这个脚本、或者脚本出错，网页照样完整（服务器生成的就是完整的页面，这里只是增强）。每一块都单独 try/catch，一块坏了不影响别的。
//   - 往网页里放东西一律用 createElement / textContent，不用“把一段文字当成网页代码塞进去”的那类写法，
//     这样数据里就算有奇怪的字符，也只会被当成文字显示，不会变成可以执行的代码。
//   - 不引用外部的任何东西，只连本站自己的地址（网站的安全规则 CSP 也只允许这样）。
//   - 不写下注建议。
"use strict";

(function () {
  // ======================================================================
  // 第一部分：纯计算（不碰网页，可以单独拿出来检查，tests/test_site_js.py 用 node 跑它们）
  // ======================================================================

  // 信心等级 1 / 2 / 3 → 显示的字（和服务器 src/web/site/common.py 的 LEVEL_NAMES 一样）
  var LEVEL_NAMES = { 3: "●●● 高", 2: "●●○ 中", 1: "●○○ 低" };
  // 战绩页每格的结果代码 → 中文（和 src/web/site/picks.py 的 OUTCOME_TEXT 一样）
  var OUTCOME_TEXT = { win: "中", half_win: "赢半", push: "走盘", half_loss: "输半", loss: "没中" };
  // 胜平负三项的名字（概率条、版本切换都用这个顺序：主队赢、平局、客队赢）
  var OUTCOMES = ["主队赢", "平局", "客队赢"];

  // 即时比分文件里的进球数、红牌数：只认 0–99 的整数（和服务器 match.py 的 _count 一样），别的当作“没有”（返回 null）
  function count(x) {
    return (typeof x === "number" && isFinite(x) && Math.floor(x) === x && x >= 0 && x < 100) ? x : null;
  }

  // 一场即时比分要显示的几小块，按顺序：[{cls: 样式, text: 文字, title: 鼠标提示}, ...]。不显示时返回 []。
  // 和服务器生成的样子一样（match.py 的 live_html）：“进行中 63'” + “2–1” + “红牌 0–1”。
  // flags.paused = 即时比分程序停了（“实时比分暂停”），flags.stale = 很久没更新（“很久没更新”）。
  function liveParts(item, flags) {
    flags = flags || {};
    if (!item || typeof item !== "object") return [];
    var state = item.state;
    if (state !== "Playing" && state !== "Played") return [];      // Fixture（还没开踢）不显示
    var out = [];
    if (state === "Playing") {
      var minute = item.minute ? String(item.minute).slice(0, 8) : "";      // 和服务器一样：0、空的都当作没写
      // 分钟是纯数字时加一撇（63'），别的写法（“中场”之类）原样写
      minute = /^[0-9]+$/.test(minute) ? " " + minute + "'" : (minute ? " " + minute : "");
      out.push({ cls: "tag live", text: "进行中" + minute, title: "懂球帝即时比分，不是最后的赛果" });
    } else {
      out.push({ cls: "tag live", text: "完场（即时）", title: "懂球帝即时比分，正式赛果以复盘对账为准" });
    }
    var hg = count(item.home_goals), ag = count(item.away_goals);
    if (hg !== null && ag !== null) out.push({ cls: "score", text: hg + "–" + ag });
    var hr = count(item.home_red), ar = count(item.away_red);
    if (hr !== null && ar !== null && (hr || ar)) out.push({ cls: "tag warn", text: "红牌 " + hr + "–" + ar });
    if (flags.paused) out.push({ cls: "tag temp", text: "实时比分暂停", title: "即时比分程序暂时停了，这是停之前的比分" });
    else if (flags.stale) out.push({ cls: "tag temp", text: "很久没更新", title: "即时比分文件有一阵没更新了，比分可能不是最新的" });
    return out;
  }

  // 命中率：中算 1 场、赢半和输半各算半场、走盘不算（和战绩页 src/web/site/picks.py 的 hit_rate 一样）。没有能算的场次返回 null
  function hitRate(c) {
    var won = (c.win || 0) + 0.5 * (c.half_win || 0);
    var lost = (c.loss || 0) + 0.5 * (c.half_loss || 0);
    return won + lost > 0 ? won / (won + lost) : null;
  }

  // 排序时比较两个值：空的（null、""、不是数）永远排最后，不管正着排还是倒着排
  function compareValues(a, b, desc) {
    var ea = a === null || a === undefined || a === "" || (typeof a === "number" && !isFinite(a));
    var eb = b === null || b === undefined || b === "" || (typeof b === "number" && !isFinite(b));
    if (ea || eb) return ea === eb ? 0 : (ea ? 1 : -1);
    var r;
    if (typeof a === "number" && typeof b === "number") r = a - b;
    else r = String(a).localeCompare(String(b), "zh-CN");
    return desc ? -r : r;
  }

  // 首页每行 data-p="0.412,0.270,0.318"（主胜,平,客胜）→ [0.412, 0.27, 0.318]；不对的返回 null
  function parseProbs(text) {
    if (!text) return null;
    var p = String(text).split(",").map(Number);
    if (p.length !== 3) return null;
    for (var i = 0; i < 3; i++) if (!isFinite(p[i])) return null;
    return p;
  }

  // 按概率排序时用哪个数：home 主胜 / draw 平 / away 客胜 / max 三个里最大的（“最明显”的比赛排前面）
  function probKey(probs, mode) {
    if (!probs) return null;
    if (mode === "home") return probs[0];
    if (mode === "draw") return probs[1];
    if (mode === "away") return probs[2];
    if (mode === "max") return Math.max(probs[0], probs[1], probs[2]);
    return null;
  }

  // 首页一行要不要显示。row = {league, pred, level, tiers, text}（text = 这一行的字 + 两队原名，小写；
  //   tiers = {"1x2": ..., ou: ..., score: ...}，每个是 "high" / "mid" / "low" 或 ""（行上的 data-tier-*，picks_v3 的把握等级）），
  // f = 筛选条件 {league: 联赛代码或 "", pred: "" / "1" / "0", level: "" / "1" / "2" / "3" / "none",
  //   tier: "" / "any" / "1x2" / "ou" / "score" / "none", q: 搜索的字}
  //   tier："any" = 只看高把握（胜平负或比分是高档；大小球的高档只比低档高 3 个百分点，不算进来），
  //         "1x2" / "ou" / "score" = 这个玩法是高档，"none" = 三个玩法都没有把握等级（不是赔率算的、还没预测）
  function rowMatches(row, f) {
    if (f.league && row.league !== f.league) return false;
    if (f.pred && row.pred !== f.pred) return false;
    if (f.level) {
      if (f.level === "none" ? row.level : row.level !== f.level) return false;
    }
    if (f.tier) {
      var t = row.tiers || {};
      if (f.tier === "any") {
        if (t["1x2"] !== "high" && t.score !== "high") return false;
      } else if (f.tier === "none") {
        if (t["1x2"] || t.ou || t.score) return false;
      } else if (t[f.tier] !== "high") return false;
    }
    var q = String(f.q || "").toLowerCase().trim();
    if (q) {
      var words = q.split(/\s+/);       // 空格隔开的几个词都要找得到（例如“利物浦 曼城”）
      for (var i = 0; i < words.length; i++) if (row.text.indexOf(words[i]) < 0) return false;
    }
    return true;
  }

  // 战绩页：一格的结果代码属于哪一类（筛选“中 / 没中 / 一半”用）
  function outcomeGroup(code) {
    if (code === "win") return "hit";
    if (code === "loss") return "miss";
    if (code === "half_win" || code === "half_loss" || code === "push") return "part";
    return "none";                         // 没预测、还没结算
  }

  // UTC 时间文字（"2026-11-07T10:00:00+00:00"）→ 北京时间 "11-07 18:00"；读不懂返回 ""
  function bjText(iso) {
    var t = Date.parse(iso || "");
    if (!isFinite(t)) return "";
    var d = new Date(t + 8 * 3600 * 1000);       // 北京时间 = UTC + 8 小时（中国不用夏令时）
    function two(n) { return (n < 10 ? "0" : "") + n; }
    return two(d.getUTCMonth() + 1) + "-" + two(d.getUTCDate()) + " " + two(d.getUTCHours()) + ":" + two(d.getUTCMinutes());
  }

  // 概率 → “41.2%”；空的写“—”
  function pct1(p) {
    return typeof p === "number" && isFinite(p) ? (p * 100).toFixed(1) + "%" : "—";
  }

  // 两版之间某一项变了多少：“↑ 2.3 个百分点” / “↓ 1.0 个百分点” / “没变”；没有上一版返回 ""
  function diffText(now, before) {
    if (typeof now !== "number" || typeof before !== "number" || !isFinite(now) || !isFinite(before)) return "";
    var d = (now - before) * 100;
    if (Math.abs(d) < 0.05) return "没变";
    return (d > 0 ? "↑ " : "↓ ") + Math.abs(d).toFixed(1) + " 个百分点";
  }

  // 虚拟资金：1234.5 → “1,235”，负数用“−”（和 src/web/site/bots.py 的 money 一样）；sign = true 时正数前面加“+”
  function money(v, sign) {
    if (typeof v !== "number" || !isFinite(v)) return "—";
    // 取整和 Python 一样：正好在 .5 时取偶数（1234.5 → 1234，1235.5 → 1236）；-0.4 这种写“−0”
    var n = Math.round(v);
    if (Math.abs(v % 1) === 0.5 && n % 2 !== 0) n -= 1;
    var s = Math.abs(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    if (n < 0 || (n === 0 && v < 0)) return "−" + s;
    return (sign && n > 0 ? "+" : "") + s;
  }

  // 一个数组是不是三个正常的数（概率）
  function isProbs(p) {
    if (!p || p.length !== 3) return false;
    for (var i = 0; i < 3; i++) if (typeof p[i] !== "number" || !isFinite(p[i])) return false;
    return true;
  }

  // 从 /api/match/<编号> 的结果里挑出要切换的几版。
  // 同一场可能有好几个模型（或者同一个模型的新旧版本）各出了几版，混在一起比没有意义，
  // 所以只留和“最新一版”（单场页上结论用的那一版）同一个模型、同一个模型版本的；对不上时用排在最后的那个模型。
  // 返回 {versions: [...], latestIndex: 最新一版在里面的位置}；一版都没有返回 null。
  function pickVersions(api) {
    if (!api || !api.versions || !api.versions.length) return null;
    var all = api.versions.filter(function (v) { return v && isProbs(v.final_probs); });
    if (!all.length) return null;
    var lt = api.latest || {};
    var hit = null;
    for (var i = 0; i < all.length; i++) {
      if (all[i].kind === lt.kind && all[i].cutoff_utc === lt.cutoff_utc) hit = all[i];
    }
    var ref = hit || all[all.length - 1];
    var same = all.filter(function (v) { return v.model === ref.model && v.model_version === ref.model_version; });
    var idx = same.indexOf(ref);
    return { versions: same, latestIndex: idx < 0 ? same.length - 1 : idx };
  }

  // 一版按钮上写的名字：T-24 → “开球前 24 小时”这种（服务器给的 kind_name）；同一种出了几次的，后面加截止时间区分
  function versionLabels(versions) {
    var seen = {};
    versions.forEach(function (v) { seen[v.kind] = (seen[v.kind] || 0) + 1; });
    return versions.map(function (v) {
      var name = v.kind_name || v.kind || "?";
      return seen[v.kind] > 1 ? name + "（" + bjText(v.cutoff_utc) + "）" : name;
    });
  }

  var API = {
    count: count, liveParts: liveParts, hitRate: hitRate, compareValues: compareValues, parseProbs: parseProbs,
    probKey: probKey, rowMatches: rowMatches, outcomeGroup: outcomeGroup, bjText: bjText, pct1: pct1,
    diffText: diffText, money: money, pickVersions: pickVersions, versionLabels: versionLabels,
    LEVEL_NAMES: LEVEL_NAMES, OUTCOME_TEXT: OUTCOME_TEXT
  };
  // 在 node 里（自动检查）：只把上面的计算交出去，下面碰网页的部分不跑
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof document === "undefined" || typeof window === "undefined") return;

  // ======================================================================
  // 第二部分：碰网页的小工具
  // ======================================================================

  // 新建一个网页元素：el("span", "tag live", "进行中 63'")。text 用 textContent 放，永远只当文字
  function el(tag, cls, text) {
    var x = document.createElement(tag);
    if (cls) x.className = cls;
    if (text !== undefined && text !== null) x.textContent = String(text);
    return x;
  }

  // 清空一个元素里的所有东西（一个一个拿掉）
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function each(list, fn) {
    Array.prototype.forEach.call(list, fn);
  }

  // 每一块功能单独跑：出错只在浏览器的“控制台”里记一句，不影响网页和别的功能
  function safely(name, fn) {
    try { fn(); } catch (err) {
      if (window.console && console.warn) console.warn("site.js：" + name + " 出错了", err);
    }
  }

  // 记住筛选、排序这些选择：先放在内存里（自动换内容时用），再试着放进 sessionStorage
  // （这个标签页关掉就没了；浏览器不让用时就只放内存里）
  var memory = {};
  function loadState(key, defaults) {
    var got = memory[key];
    if (!got) {
      try { got = JSON.parse(window.sessionStorage.getItem("site:" + key) || "null"); } catch (err) { got = null; }
    }
    var out = {};
    for (var k in defaults) {
      if (Object.prototype.hasOwnProperty.call(defaults, k)) {
        out[k] = got && typeof got[k] === typeof defaults[k] ? got[k] : defaults[k];
      }
    }
    return out;
  }
  function saveState(key, state) {
    memory[key] = state;
    try { window.sessionStorage.setItem("site:" + key, JSON.stringify(state)); } catch (err) { /* 浏览器不让存：只放内存里 */ }
  }

  // 一个“标签 + 下拉框”：options = [[值, 显示的字], ...]。key 是自动换内容后找回“光标在哪个框里”用的
  function selectField(label, key, options, value, onChange) {
    var wrap = el("label", "js-field");
    wrap.appendChild(el("span", "js-label", label));
    var s = el("select");
    s.setAttribute("data-js-key", key);
    options.forEach(function (o) {
      var opt = el("option", "", o[1]);
      opt.value = o[0];
      s.appendChild(opt);
    });
    s.value = value;
    if (s.value !== value) s.value = options.length ? options[0][0] : "";   // 记住的选项这次没有了（比如那一天过去了）：回到第一项
    s.addEventListener("change", function () { onChange(s.value); });
    wrap.appendChild(s);
    return { wrap: wrap, input: s };
  }

  // 一个搜索框：边打字边筛选
  function searchField(label, key, value, placeholder, onInput) {
    var wrap = el("label", "js-field js-search");
    wrap.appendChild(el("span", "js-label", label));
    var s = el("input");
    s.type = "search";
    s.setAttribute("data-js-key", key);
    s.setAttribute("autocomplete", "off");
    s.placeholder = placeholder;
    s.value = value;
    s.addEventListener("input", function () { onInput(s.value); });
    wrap.appendChild(s);
    return { wrap: wrap, input: s };
  }

  // 一个普通按钮（type=button：不会提交表单）
  function button(text, cls, onClick) {
    var b = el("button", cls, text);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
  }

  // 一行工具栏（筛选、排序都放在里面）。data-js-toolbar 标出是脚本加的，自动换内容时会跟着旧内容一起换掉、再重新加
  function toolbar(name) {
    var bar = el("div", "js-toolbar");
    bar.setAttribute("data-js-toolbar", name);
    bar.setAttribute("role", "group");
    bar.setAttribute("aria-label", "筛选和排序（只改这一页怎么显示，不改数据）");
    return bar;
  }

  // ======================================================================
  // 第三部分：首页的筛选、排序、搜索
  // ======================================================================
  // 首页每一场是表格（table.list）里的一行，行上有服务器写好的 data-*：
  //   data-league（联赛代码）、data-league-name（联赛中文名）、data-teams（两队原名）、
  //   data-pred（有没有预测 1 / 0）、data-level（信心 1 / 2 / 3）、data-p（主胜,平,客胜 的概率）、
  //   data-tier-1x2 / data-tier-ou / data-tier-score（把握等级 high / mid / low，不分档的没有这个属性）。
  // “即将开赛”按日期分成好几张表，每张表前面是一个 <h2> 日期标题；最后一张是“最近踢完的”。
  // 筛选只是把不要的行藏起来（hidden），排序只在每张表里面调换行的顺序。数据不动，刷新网页就回到原样。
  function enhanceIndex() {
    var tables = document.querySelectorAll(".wrap table.list");
    if (!tables.length || !document.querySelector(".wrap table.list tr[data-pred]")) return;   // 不是首页（球队页的表没有 data-pred）
    var groups = [];
    each(tables, function (t) {
      var card = t.closest(".card") || t;
      var h = card.previousElementSibling;
      if (!h || h.tagName !== "H2") h = null;
      var rows = Array.prototype.slice.call(t.tBodies.length ? t.tBodies[0].rows : []);
      rows.forEach(function (r, i) { r.setAttribute("data-js-order", String(i)); });   // 原来的顺序（“按开球时间”时还原）
      groups.push({ table: t, card: card, head: h, title: h ? h.textContent.trim() : "", rows: rows });
    });
    // 联赛：按出现的先后列出来
    var leagues = [], seen = {};
    groups.forEach(function (g) {
      g.rows.forEach(function (r) {
        var code = r.getAttribute("data-league") || "";
        if (code && !seen[code]) { seen[code] = true; leagues.push([code, r.getAttribute("data-league-name") || code]); }
      });
    });
    var st = loadState("index", { league: "", day: "", pred: "", level: "", tier: "", sort: "time", q: "" });
    var bar = toolbar("index");
    var info = el("span", "js-count");
    info.setAttribute("aria-live", "polite");
    var empty = el("div", "placeholder js-empty", "没有符合筛选条件的比赛。点“全部显示”回到原样。");
    empty.hidden = true;

    function set(k) { return function (v) { st[k] = v; saveState("index", st); apply(); }; }
    var days = [["", "全部日期"]].concat(groups.filter(function (g) { return g.title; })
      .map(function (g) { return [g.title, g.title]; }));
    var fields = [
      selectField("联赛", "index-league", [["", "全部联赛"]].concat(leagues), st.league, set("league")),
      selectField("日期", "index-day", days, st.day, set("day")),
      selectField("预测", "index-pred", [["", "有没有都看"], ["1", "只看有预测的"], ["0", "只看还没预测的"]], st.pred, set("pred")),
      selectField("信心", "index-level", [["", "全部"], ["3", LEVEL_NAMES[3]], ["2", LEVEL_NAMES[2]], ["1", LEVEL_NAMES[1]], ["none", "没有信心等级"]],
        st.level, set("level")),
      selectField("把握", "index-tier", [["", "全部"], ["any", "只看高把握（胜平负或比分）"], ["1x2", "胜平负把握高"],
        ["score", "比分把握高"], ["ou", "大小球把握高（三档差得很小）"], ["none", "没有把握等级"]], st.tier, set("tier")),
      selectField("排序", "index-sort", [["time", "按开球时间"], ["home", "主队赢的可能从大到小"], ["draw", "平局的可能从大到小"],
        ["away", "客队赢的可能从大到小"], ["max", "最明显的在前（三项里最大的那个）"]], st.sort, set("sort")),
      searchField("搜索球队", "index-q", st.q, "输入球队名，中文英文都行", set("q"))
    ];
    // 下拉框的值可能被换成了第一项（记住的那一天已经没有了）：按实际显示的改回来
    st.league = fields[0].input.value; st.day = fields[1].input.value; st.tier = fields[4].input.value;
    st.sort = fields[5].input.value;
    fields.forEach(function (f) { bar.appendChild(f.wrap); });
    bar.appendChild(button("全部显示", "js-reset", function () {
      st = { league: "", day: "", pred: "", level: "", tier: "", sort: "time", q: "" };
      saveState("index", st);
      fields[0].input.value = ""; fields[1].input.value = ""; fields[2].input.value = "";
      fields[3].input.value = ""; fields[4].input.value = ""; fields[5].input.value = "time"; fields[6].input.value = "";
      apply();
    }));
    bar.appendChild(info);
    // 工具栏放在第一张比赛表的标题前面（“怎么看这张表”的下面）
    var first = groups[0].head || groups[0].card;
    first.parentNode.insertBefore(bar, first);
    bar.parentNode.insertBefore(empty, bar.nextSibling);

    function apply() {
      var shown = 0, total = 0;
      groups.forEach(function (g) {
        var dayOk = !st.day || g.title === st.day;
        var n = 0;
        g.rows.forEach(function (r) {
          var text = (r.textContent + " " + (r.getAttribute("data-teams") || "")).toLowerCase();
          var tiers = { "1x2": r.getAttribute("data-tier-1x2") || "", ou: r.getAttribute("data-tier-ou") || "",
            score: r.getAttribute("data-tier-score") || "" };
          var ok = dayOk && rowMatches({ league: r.getAttribute("data-league") || "", pred: r.getAttribute("data-pred") || "",
            level: r.getAttribute("data-level") || "", tiers: tiers, text: text }, st);
          r.hidden = !ok;
          if (ok) n++;
        });
        total += g.rows.length;
        shown += n;
        // 一张表一场都不剩：连上面的日期标题一起藏起来
        //（“最近踢完的”没有比赛时服务器写的是一句提示、不是表格，不在这里面，照常显示）
        var hide = n === 0;
        g.card.hidden = hide;
        if (g.head) g.head.hidden = hide;
        sortRows(g);
      });
      info.textContent = "显示 " + shown + " / " + total + " 场";
      empty.hidden = shown > 0 || total === 0;
    }

    function sortRows(g) {
      var body = g.table.tBodies[0];
      if (!body) return;
      var rows = g.rows.slice();
      rows.sort(function (a, b) {
        var r = 0;
        if (st.sort !== "time") {
          r = compareValues(probKey(parseProbs(a.getAttribute("data-p")), st.sort),
                            probKey(parseProbs(b.getAttribute("data-p")), st.sort), true);   // 大的在前，没有概率的在最后
        }
        return r || (Number(a.getAttribute("data-js-order")) - Number(b.getAttribute("data-js-order")));
      });
      rows.forEach(function (r) { body.appendChild(r); });
    }
    apply();
  }

  // ======================================================================
  // 第四部分：战绩页“每场的四个预测”的筛选
  // ======================================================================
  // 表 table.picks-record：每行带 data-league（联赛代码）、data-league-name；每个玩法那一格带
  //   data-market（1x2 胜平负 / ah 让球 / ou 大小球 / score 比分）、data-outcome（网站那一列的结果）、
  //   data-model-outcome（“模型自己”的结果）：win 中 / half_win 赢半 / push 走盘 / half_loss 输半 / loss 没中 / none 没预测。
  // 选了某个玩法：只显示这个玩法那一列，并且可以按“中没中”筛选，下面算一下显示出来的这些场的命中率。
  function enhanceRecord() {
    var t = document.querySelector(".wrap table.picks-record");
    if (!t || !t.tBodies.length) return;
    var rows = Array.prototype.slice.call(t.tBodies[0].rows);
    if (!rows.length) return;
    var MARKETS = [["1x2", "胜平负"], ["ah", "让球"], ["ou", "大小球 2.5"], ["score", "比分预测"]];
    // 每个玩法在第几列（按第一行的格子找）
    var col = {};
    each(rows[0].cells, function (c, i) { var m = c.getAttribute("data-market"); if (m) col[m] = i; });
    var leagues = [], seen = {};
    rows.forEach(function (r) {
      var code = r.getAttribute("data-league") || "";
      if (code && !seen[code]) { seen[code] = true; leagues.push([code, r.getAttribute("data-league-name") || code]); }
    });
    var st = loadState("record", { market: "", league: "", result: "", who: "site" });
    var bar = toolbar("record");
    var info = el("p", "small js-count");
    info.setAttribute("aria-live", "polite");
    function set(k) { return function (v) { st[k] = v; saveState("record", st); apply(); }; }
    var fMarket = selectField("玩法", "record-market", [["", "四个玩法都看"]].concat(MARKETS), st.market, set("market"));
    var fLeague = selectField("联赛", "record-league", [["", "全部联赛"]].concat(leagues), st.league, set("league"));
    var fResult = selectField("结果", "record-result", [["", "全部"], ["hit", "中"], ["miss", "没中"],
      ["part", "走盘、赢半、输半"], ["none", "没预测"]], st.result, set("result"));
    var fWho = selectField("看哪一列", "record-who", [["site", "网站的预测"], ["model", "模型自己"]], st.who, set("who"));
    st.league = fLeague.input.value;
    [fMarket, fLeague, fResult, fWho].forEach(function (f) { bar.appendChild(f.wrap); });
    bar.appendChild(button("全部显示", "js-reset", function () {
      st = { market: "", league: "", result: "", who: "site" };
      saveState("record", st);
      fMarket.input.value = ""; fLeague.input.value = ""; fResult.input.value = ""; fWho.input.value = "site";
      apply();
    }));
    var card = t.closest(".card") || t;
    card.parentNode.insertBefore(bar, card);
    card.parentNode.insertBefore(info, card);

    function apply() {
      var m = st.market;
      // 没选玩法时“结果”“看哪一列”不能用（四个玩法的结果各不一样，放在一起筛没意义）
      fResult.input.disabled = !m;
      fWho.input.disabled = !m;
      var attr = st.who === "model" ? "data-model-outcome" : "data-outcome";
      var counts = { win: 0, half_win: 0, push: 0, half_loss: 0, loss: 0, none: 0 };
      var shown = 0;
      rows.forEach(function (r) {
        var ok = !st.league || r.getAttribute("data-league") === st.league;
        var code = "none";
        if (m && col[m] !== undefined && r.cells[col[m]]) {
          code = r.cells[col[m]].getAttribute(attr) || "none";
          if (ok && st.result && outcomeGroup(code) !== st.result) ok = false;
        }
        r.hidden = !ok;
        if (ok) {
          shown++;
          counts[OUTCOME_TEXT[code] ? code : "none"]++;
        }
      });
      // 只显示选中的那个玩法那一列（表头也一起）
      var head = t.tHead && t.tHead.rows.length ? t.tHead.rows[0] : null;
      for (var k in col) {
        if (!Object.prototype.hasOwnProperty.call(col, k)) continue;
        var hide = !!m && k !== m;
        if (head && head.cells[col[k]]) head.cells[col[k]].hidden = hide;
        rows.forEach(function (r) { if (r.cells[col[k]]) r.cells[col[k]].hidden = hide; });
      }
      var text = "显示 " + shown + " / " + rows.length + " 场";
      if (m) {
        var hr = hitRate(counts);
        text += "。这些场里" + (st.who === "model" ? "“模型自己”" : "网站的预测") + "：中 " + counts.win + "、赢半 " + counts.half_win +
          "、走盘 " + counts.push + "、输半 " + counts.half_loss + "、没中 " + counts.loss + "、没预测 " + counts.none +
          "；命中率 " + (hr === null ? "—" : (hr * 100).toFixed(1) + "%") +
          "（赢半、输半各算半场，走盘不算；只是下面这张表里的场次，不是上面的全部）";
      }
      info.textContent = text;
    }
    apply();
  }

  // ======================================================================
  // 第五部分：虚拟投注页：点表头排序、筛选、点小图看每个赛季
  // ======================================================================
  // 表 table.bots-board：每行带 data-book（价格）、data-control（是不是对照组 1 / 0）；
  // 数字那几格带 data-sort（排序用的数，没有的格子排最后）。“按赛季累计盈亏”的小图外面是 <span class="curve" data-curve="…">，
  // data-curve 里是每个赛季 [赛季, 当季盈亏, 累计]。
  var openCurves = {};        // 哪几个机器人的“每个赛季”是展开的（自动换内容后照样展开）
  function enhanceBots() {
    var t = document.querySelector(".wrap table.bots-board");
    if (!t || !t.tBodies.length) return;
    var body = t.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    if (!rows.length) return;
    rows.forEach(function (r, i) { r.setAttribute("data-js-order", String(i)); });
    var st = loadState("bots", { col: -1, desc: false, book: "", control: "", q: "" });
    var heads = t.tHead && t.tHead.rows.length ? t.tHead.rows[0].cells : [];
    // 表头每一格换成一个按钮（手机上也点得到；键盘用 Tab 也能选到）
    each(heads, function (th, i) {
      var label = th.textContent;
      clear(th);
      var b = button(label, "js-sort", function () {
        if (st.col === i) st.desc = !st.desc;
        else { st.col = i; st.desc = hasNumbers(i) && i !== 0; }     // 数字列第一次点：大的在前；名次：小的在前
        saveState("bots", st);
        apply();
      });
      b.appendChild(el("span", "js-arrow"));
      th.appendChild(b);
    });
    function hasNumbers(i) {
      return rows.some(function (r) { return r.cells[i] && r.cells[i].hasAttribute("data-sort"); });
    }
    function value(r, i) {
      var c = r.cells[i];
      if (!c) return null;
      if (c.hasAttribute("data-sort")) return Number(c.getAttribute("data-sort"));
      if (hasNumbers(i)) return null;                  // 数字列里没有数的格子（“—”）：排最后
      var s = c.textContent.trim();
      return s === "—" ? null : s;
    }

    var books = [], seen = {};
    rows.forEach(function (r) {
      var b = r.getAttribute("data-book") || "";
      if (b && !seen[b]) { seen[b] = true; books.push([b, b]); }
    });
    var bar = toolbar("bots");
    var info = el("span", "js-count");
    info.setAttribute("aria-live", "polite");
    function set(k) { return function (v) { st[k] = v; saveState("bots", st); apply(); }; }
    var fBook = selectField("价格", "bots-book", [["", "全部"]].concat(books), st.book, set("book"));
    var fCtl = selectField("对照组", "bots-control", [["", "都看"], ["0", "不看对照组"], ["1", "只看对照组"]], st.control, set("control"));
    var fQ = searchField("搜索", "bots-q", st.q, "机器人名字或规则里的字", set("q"));
    st.book = fBook.input.value;
    [fBook, fCtl, fQ].forEach(function (f) { bar.appendChild(f.wrap); });
    bar.appendChild(button("回到原样", "js-reset", function () {
      st = { col: -1, desc: false, book: "", control: "", q: "" };
      saveState("bots", st);
      fBook.input.value = ""; fCtl.input.value = ""; fQ.input.value = "";
      apply();
    }));
    bar.appendChild(info);
    var card = t.closest(".card") || t;
    card.parentNode.insertBefore(bar, card);

    // 点小图（或者用键盘选中后按回车、空格）：在小图下面展开 / 收起每个赛季的数
    each(t.querySelectorAll(".curve[data-curve]"), function (c) {
      var data;
      try { data = JSON.parse(c.getAttribute("data-curve")); } catch (err) { return; }
      if (!data || !data.seasons || !data.seasons.length) return;
      c.setAttribute("tabindex", "0");
      c.setAttribute("role", "button");
      c.setAttribute("aria-expanded", "false");
      c.setAttribute("aria-label", "看每个赛季的盈亏：" + (data.name || ""));
      c.classList.add("js-clickable");
      var panel = null;
      function toggle(open) {
        if (open && !panel) {
          panel = curvePanel(data);
          c.parentNode.insertBefore(panel, c.nextSibling);
        } else if (!open && panel) {
          panel.parentNode.removeChild(panel);
          panel = null;
        }
        c.setAttribute("aria-expanded", open ? "true" : "false");
        if (open) openCurves[data.name] = true; else delete openCurves[data.name];
      }
      c.addEventListener("click", function () { toggle(!panel); });
      c.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggle(!panel); }
      });
      if (openCurves[data.name]) toggle(true);
    });

    function apply() {
      var shown = 0;
      var q = String(st.q || "").toLowerCase().trim();
      rows.forEach(function (r) {
        var ok = (!st.book || r.getAttribute("data-book") === st.book) &&
          (!st.control || r.getAttribute("data-control") === st.control) &&
          (!q || r.textContent.toLowerCase().indexOf(q) >= 0);          // 名字、规则都在这一行的字里
        r.hidden = !ok;
        if (ok) shown++;
      });
      var sorted = rows.slice();
      if (st.col >= 0) {
        sorted.sort(function (a, b) {
          return compareValues(value(a, st.col), value(b, st.col), st.desc) ||
            Number(a.getAttribute("data-js-order")) - Number(b.getAttribute("data-js-order"));
        });
      }
      sorted.forEach(function (r) { body.appendChild(r); });
      each(heads, function (th, i) {
        var on = i === st.col;
        th.setAttribute("aria-sort", on ? (st.desc ? "descending" : "ascending") : "none");
        var arrow = th.querySelector(".js-arrow");
        if (arrow) arrow.textContent = on ? (st.desc ? " ▼" : " ▲") : "";
      });
      info.textContent = "显示 " + shown + " / " + rows.length + " 个机器人" + (st.col >= 0 ? "（点表头可以换排序，再点一下倒过来）" : "（点表头排序）");
    }
    apply();
  }

  // 每个赛季的小表：赛季 / 当季盈亏 / 到季末累计
  function curvePanel(data) {
    var box = el("div", "js-curve-panel");
    var tb = el("table", "js-mini");
    var head = el("tr");
    ["赛季", "当季盈亏", "累计"].forEach(function (h) { head.appendChild(el("th", "", h)); });
    var thead = el("thead");
    thead.appendChild(head);
    tb.appendChild(thead);
    var tbody = el("tbody");
    data.seasons.forEach(function (s) {
      var tr = el("tr");
      tr.appendChild(el("td", "", s[0]));
      tr.appendChild(el("td", typeof s[1] === "number" ? (s[1] >= 0 ? "hit" : "miss") : "", money(s[1], true)));
      tr.appendChild(el("td", "", money(s[2], true)));
      tbody.appendChild(tr);
    });
    tb.appendChild(tbody);
    box.appendChild(tb);
    box.appendChild(el("p", "small muted", "虚拟投注，单位是虚拟资金；再点一下小图收起"));
    return box;
  }

  // ======================================================================
  // 第六部分：概率条、比分热力图上显示具体的数
  // ======================================================================
  // 电脑：鼠标放上去显示、移开消失。手机（没有“鼠标放上去”）：点一下显示，再点一下或点别处消失。
  // 只有一个浮动的小框（.js-tip），所有概率条、热力图格子共用。事件挂在整个网页上，自动换内容后不用重新挂。
  var tip = null, tipOwner = null, lastPointer = "mouse";

  function tipLines(target) {
    // 概率条：.pbar 里三段 .seg，每段的宽度（style 里的 width:41.2%）就是那一项的概率
    var bar = target.closest(".pbar");
    if (bar) {
      var segs = bar.querySelectorAll(".seg");
      var on = target.closest(".seg");
      var out = [];
      each(segs, function (s, i) {
        var w = parseFloat(s.style.width);
        out.push({ text: (OUTCOMES[i] || "") + " " + (isFinite(w) ? w.toFixed(1) + "%" : "—"), on: s === on });
      });
      return out;
    }
    // 热力图格子：服务器写在 <title> 里的“2-1：12.34%”，脚本第一次碰到时挪到 data-tip（免得浏览器自己的提示和这个重复）
    var cell = target.closest("rect");
    if (cell && cell.closest("svg.chart")) {
      var t = cell.getAttribute("data-tip");
      if (!t) {
        var title = cell.querySelector("title");
        if (!title) return null;
        t = title.textContent;
        cell.setAttribute("data-tip", t);
        cell.removeChild(title);
      }
      return [{ text: "比分 " + t, on: true }];
    }
    return null;
  }

  function showTip(target, x, y) {
    var lines = tipLines(target);
    if (!lines) return hideTip();
    if (!tip) {
      tip = el("div", "js-tip");
      tip.setAttribute("role", "tooltip");
      document.body.appendChild(tip);
    }
    clear(tip);
    lines.forEach(function (l) { tip.appendChild(el("div", l.on ? "on" : "", l.text)); });
    tip.hidden = false;
    tipOwner = target.closest(".pbar") || target.closest("rect");
    // 放在手指 / 鼠标的右上方，不超出屏幕
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
    var left = Math.min(Math.max(8, x + 12), vw - w - 8);
    var top = y - h - 12 < 8 ? y + 18 : y - h - 12;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.min(Math.max(8, top), vh - h - 8) + "px";
  }

  function hideTip() {
    if (tip) tip.hidden = true;
    tipOwner = null;
  }

  function isTipTarget(t) {
    return t && t.closest && (t.closest(".pbar") || (t.closest("rect") && t.closest("svg.chart") && !t.closest(".curve")));
  }

  function setupTips() {
    var hasPointer = typeof window.PointerEvent === "function";
    if (hasPointer) {
      document.addEventListener("pointerdown", function (ev) { lastPointer = ev.pointerType || "mouse"; }, true);
      document.addEventListener("pointermove", function (ev) {
        if (ev.pointerType !== "mouse") return;
        if (isTipTarget(ev.target)) showTip(ev.target, ev.clientX, ev.clientY);
        else if (tipOwner) hideTip();
      });
    }
    // 点一下：手机、平板（或者没有 pointer 事件的旧浏览器）用它开关提示；用鼠标点的不管（鼠标放上去就有了）
    document.addEventListener("click", function (ev) {
      if (hasPointer && lastPointer === "mouse") return;
      var t = ev.target;
      if (!isTipTarget(t)) { hideTip(); return; }
      var owner = t.closest(".pbar") || t.closest("rect");
      if (tipOwner === owner && tip && !tip.hidden) hideTip();
      else showTip(t, ev.clientX, ev.clientY);
    });
    window.addEventListener("scroll", hideTip, { passive: true });
    window.addEventListener("resize", hideTip);
    // 让概率条、热力图格子在手机上点得到（iOS 的浏览器要有 cursor:pointer 才把“点一下”交给网页）
    document.documentElement.classList.add("js-tips");
  }

  // ======================================================================
  // 第七部分：单场页的“各版本切换”
  // ======================================================================
  // 单场页结论卡片下面有一个空的 <div class="version-switch" data-version-switch="比赛编号">（没有脚本时什么都不显示）。
  // 脚本从 /api/match/<编号> 取这场的各版本（开球前 24 / 6 / 1 小时、预览版），放几个按钮：
  // 点一下看那一版的三色条、三个概率、和上一版比变了多少、信心等级和倾向，下面一张小表列出每一版，看概率怎么一步步变。
  var chosenVersion = {};        // 每场选的是哪一版（自动换内容后还选着它）
  var versionCache = {};         // 每场取到的数据：{version: 数据版本, api: 内容}；数据版本没变就不再取

  function enhanceVersions(dataVersion) {
    each(document.querySelectorAll("[data-version-switch]"), function (box) {
      var mid = box.getAttribute("data-version-switch");
      if (!/^[0-9]{1,12}$/.test(mid)) return;
      var hit = versionCache[mid];
      if (hit && hit.version === dataVersion) { renderVersions(box, mid, hit.api); return; }
      getJSON("/api/match/" + mid).then(function (res) {
        if (res.status === 401) { poller.loginExpired(); return; }
        if (res.status !== 200 || !res.body) return;              // 取不到：这一块就不显示，网页其余部分不受影响
        versionCache[mid] = { version: res.body.data_version, api: res.body };
        if (box.isConnected) renderVersions(box, mid, res.body);
      }).catch(function () { /* 连不上：不显示这一块 */ });
    });
  }

  function renderVersions(box, mid, api) {
    var got = pickVersions(api);
    clear(box);
    if (!got) return;
    var vs = got.versions, labels = versionLabels(vs);
    var pick = chosenVersion[mid];
    var idx = typeof pick === "number" && pick < vs.length ? pick : got.latestIndex;
    var home = api.home || "主队", away = api.away || "客队";
    var names = [home + "赢", "平局", away + "赢"];

    var card = el("div", "card pad js-versions");
    card.appendChild(el("b", "", "各版本切换"));
    card.appendChild(el("p", "small muted", vs.length > 1
      ? "这场出过 " + vs.length + " 版预测（同一个模型）。点下面的按钮，看那一版的概率和上一版比变了多少。上面的结论用的是最新一版。"
      : "这场到现在只出过一版预测，等开球前 6 小时、1 小时的版本出来以后，这里可以切换着看。"));
    var btns = el("div", "js-vbtns");
    btns.setAttribute("role", "group");
    btns.setAttribute("aria-label", "选一版");
    vs.forEach(function (v, i) {
      var b = button(labels[i] + (i === got.latestIndex ? "（最新）" : ""), "js-vbtn" + (i === idx ? " on" : ""), function () {
        chosenVersion[mid] = i;
        renderVersions(box, mid, api);
        var again = box.querySelector(".js-vbtn.on");
        if (again) again.focus();
      });
      b.setAttribute("aria-pressed", i === idx ? "true" : "false");
      btns.appendChild(b);
    });
    card.appendChild(btns);

    var v = vs[idx], prev = idx > 0 ? vs[idx - 1] : null;
    var p = v.final_probs;
    // 三色条：和上面结论卡片里的一样（每段的宽度就是概率）
    var bar = el("div", "pbar big");
    p.forEach(function (x, i) {
      var seg = el("span", "seg s" + i, x >= 0.12 ? Math.round(x * 100) + "%" : "");
      seg.style.width = (x * 100).toFixed(1) + "%";
      bar.appendChild(seg);
    });
    card.appendChild(bar);
    var list = el("ul", "js-vprobs");
    p.forEach(function (x, i) {
      var li = el("li");
      li.appendChild(el("span", "", names[i] + " "));
      li.appendChild(el("b", "", pct1(x)));
      var d = prev ? diffText(x, prev.final_probs[i]) : "";
      if (d) li.appendChild(el("span", "small " + (d.charAt(0) === "↑" ? "js-up" : d.charAt(0) === "↓" ? "js-down" : "muted"),
        "（比" + (labels[idx - 1] || "上一版") + " " + d + "）"));
      list.appendChild(li);
    });
    card.appendChild(list);
    var meta = [];
    meta.push("信心：" + (LEVEL_NAMES[v.confidence] || "—"));
    meta.push(v.source === "model" ? "只有模型（没和赔率融合）" : "赔率和模型合起来的概率");
    if (v.cutoff_utc) meta.push("用 " + bjText(v.cutoff_utc) + "（北京时间）之前的数据");
    card.appendChild(el("p", "small", meta.join("；")));
    if (v.leans && v.leans.length) card.appendChild(el("p", "small", "倾向：" + v.leans.join("；")));
    if (v.uncertain && v.uncertain.length) card.appendChild(el("p", "small muted", "要注意：" + v.uncertain.join("；")));

    // 每一版的概率排成一张小表，选中的那一行亮一点
    if (vs.length > 1) {
      var tb = el("table", "js-mini");
      var hr = el("tr");
      ["版本", names[0], names[1], names[2], "信心"].forEach(function (h) { hr.appendChild(el("th", "", h)); });
      var thead = el("thead");
      thead.appendChild(hr);
      tb.appendChild(thead);
      var tbody = el("tbody");
      vs.forEach(function (w, i) {
        var tr = el("tr", i === idx ? "on" : "");
        tr.appendChild(el("td", "", labels[i]));
        w.final_probs.forEach(function (x, k) {
          var d = i > 0 ? diffText(x, vs[i - 1].final_probs[k]) : "";
          var td = el("td", "", pct1(x));
          if (d && d !== "没变") td.appendChild(el("span", "small " + (d.charAt(0) === "↑" ? "js-up" : "js-down"), " " + d.split(" ")[0]));
          tr.appendChild(td);
        });
        tr.appendChild(el("td", "", LEVEL_NAMES[w.confidence] || "—"));
        tbody.appendChild(tr);
      });
      tb.appendChild(tbody);
      var scroll = el("div", "scroll");
      scroll.appendChild(tb);
      card.appendChild(scroll);
    }
    box.appendChild(card);
  }

  // ======================================================================
  // 第八部分：和服务器说话（只连本站自己的地址）
  // ======================================================================
  // 带登录的 cookie（同一个网站，浏览器自动带）；不用缓存；最多等 15 秒。
  // 返回 {status: 状态码, body: 解析好的内容（不是 JSON 时为 null）}
  function request(url, accept) {
    var ctl = typeof AbortController === "function" ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, 15000) : null;
    return fetch(url, { credentials: "same-origin", cache: "no-store", redirect: "follow",
                        headers: { Accept: accept }, signal: ctl ? ctl.signal : undefined })
      .then(function (res) {
        if (timer) clearTimeout(timer);
        return res;
      }, function (err) {
        if (timer) clearTimeout(timer);
        throw err;
      });
  }

  function getJSON(url) {
    return request(url, "application/json").then(function (res) {
      var type = res.headers.get("Content-Type") || "";
      if (type.indexOf("application/json") < 0) return { status: res.status, body: null };
      return res.json().then(function (body) { return { status: res.status, body: body }; },
                             function () { return { status: res.status, body: null }; });
    });
  }

  // ======================================================================
  // 第九部分：自动更新（换正文）
  // ======================================================================
  // 数据版本（/api/version 的 data_version）变了的时候怎么更新？两种做法里选了“把这一页重新取一次，只换掉正文”：
  //   - 网页长什么样只在服务器（Python）那一处写；要是在脚本里再按接口的数字“只改变了的地方”，同样的页面就得写两遍，
  //     两边早晚会对不上（比如服务器改了一句话，脚本里还是旧的）。
  //   - 数据版本最多一分钟变一次（服务器 live.min_reload_seconds），平时一页几十 KB，取一次不费事。
  //   - 换的只是 .wrap 里的正文：上面的“首页 / 数据状态 / 退出登录”、自动更新那一行不动；
  //     换之前记下滚到哪了、哪些“详细”是展开的、光标在哪个框里、筛选排序选了什么，换完照原样恢复，看的人几乎觉察不到；
  //     有变化的那几行闪一下，提醒“这里变了”。
  //   - 取来的是本站自己生成的网页，放进来之前还是把 <script>、on 开头的属性这些能执行的东西去掉（多一道保险）。
  function fetchPage() {
    return request(location.pathname + location.search, "text/html").then(function (res) {
      // 登录过期：服务器把网页请求转到登录页（fetch 自己跟过去了）。算“登录过期”，不跳过去
      if (res.status === 401 || (res.redirected && new URL(res.url, location.href).pathname === "/login")) return { login: true };
      var type = res.headers.get("Content-Type") || "";
      if (res.status !== 200 || type.indexOf("text/html") < 0) return { failed: true };
      return res.text().then(function (text) {
        var doc = new DOMParser().parseFromString(text, "text/html");
        var wrap = doc.querySelector(".wrap");
        if (!wrap) return { failed: true };          // 不是我们的网页（比如出错页）：这次不换
        var meta = doc.querySelector("meta[name='site-data-version']");
        return { wrap: wrap, title: doc.title, version: meta ? meta.getAttribute("content") : null };
      });
    });
  }

  // 去掉能执行的东西：<script>、<iframe> 这些整块拿掉；on 开头的属性（onclick 之类）、以 javascript 加冒号开头的链接拿掉
  function sanitize(root) {
    each(root.querySelectorAll("script, iframe, object, embed, base, meta, link"), function (x) { x.parentNode.removeChild(x); });
    each(root.querySelectorAll("*"), function (x) {
      for (var i = x.attributes.length - 1; i >= 0; i--) {
        var a = x.attributes[i];
        var name = a.name.toLowerCase();
        if (name.indexOf("on") === 0) x.removeAttribute(a.name);
        else if ((name === "href" || name === "src" || name === "xlink:href" || name === "action") &&
                 /^\s*javascript:/i.test(a.value)) x.removeAttribute(a.name);
      }
    });
  }

  // 换正文之前记下：滚到哪（以屏幕上第一场比赛、第一个标题为准）、哪些“详细”展开着、光标在哪个框里、各行的字
  function saveView(wrap) {
    var view = { anchor: null, top: 0, open: {}, focus: null, rows: {} };
    var marks = wrap.querySelectorAll("[data-match-id], h2");
    var count = {};
    for (var i = 0; i < marks.length; i++) {
      var m = marks[i];
      var key = markKey(m, count);
      var r = m.getBoundingClientRect();
      if (!view.anchor && r.bottom > 0 && r.height > 0) { view.anchor = key; view.top = r.top; }
    }
    var seen = {};
    each(wrap.querySelectorAll("details"), function (d) {
      var s = d.querySelector("summary");
      var k = (s ? s.textContent.trim() : "") + "#" + (seen[s ? s.textContent : ""] = (seen[s ? s.textContent : ""] || 0) + 1);
      view.open[k] = d.open;
    });
    var a = document.activeElement;
    if (a && wrap.contains(a) && a.getAttribute("data-js-key")) {
      view.focus = { key: a.getAttribute("data-js-key"), start: a.selectionStart, end: a.selectionEnd };
    }
    each(wrap.querySelectorAll("tr[data-match-id]"), function (r) { view.rows[r.getAttribute("data-match-id")] = r.textContent; });
    return view;
  }

  // 一个“位置记号”的名字：比赛那一行用比赛编号；标题用标题的字（同样的字加序号）
  function markKey(m, count) {
    var k = m.hasAttribute("data-match-id") ? "m:" + m.tagName + ":" + m.getAttribute("data-match-id") : "h:" + m.textContent.trim();
    count[k] = (count[k] || 0) + 1;
    return k + "#" + count[k];
  }

  function restoreView(wrap, view) {
    var seen = {};
    each(wrap.querySelectorAll("details"), function (d) {
      var s = d.querySelector("summary");
      var k = (s ? s.textContent.trim() : "") + "#" + (seen[s ? s.textContent : ""] = (seen[s ? s.textContent : ""] || 0) + 1);
      if (Object.prototype.hasOwnProperty.call(view.open, k)) d.open = view.open[k];
    });
    if (view.anchor) {
      var marks = wrap.querySelectorAll("[data-match-id], h2"), count = {};
      for (var i = 0; i < marks.length; i++) {
        if (markKey(marks[i], count) === view.anchor) {
          window.scrollBy(0, marks[i].getBoundingClientRect().top - view.top);
          break;
        }
      }
    }
    if (view.focus) {
      var f = wrap.querySelector("[data-js-key='" + view.focus.key.replace(/[^a-z0-9-]/gi, "") + "']");
      if (f) {
        f.focus({ preventScroll: true });
        try { if (typeof view.focus.start === "number") f.setSelectionRange(view.focus.start, view.focus.end); } catch (err) { /* 下拉框没有光标位置 */ }
      }
    }
    // 字变了的那几行闪一下
    each(wrap.querySelectorAll("tr[data-match-id]"), function (r) {
      var old = view.rows[r.getAttribute("data-match-id")];
      if (old !== undefined && old !== r.textContent) flash(r);
    });
  }

  function flash(node) {
    node.classList.add("js-changed");
    setTimeout(function () { node.classList.remove("js-changed"); }, 2600);
  }

  // 把 .wrap 里的正文换成新取来的那份，然后恢复位置、重新加上筛选工具栏这些
  function swapWrap(got) {
    var wrap = document.querySelector(".wrap");
    if (!wrap) return;
    sanitize(got.wrap);
    hideTip();
    var view = saveView(wrap);
    var frag = document.createDocumentFragment();
    each(Array.prototype.slice.call(got.wrap.childNodes), function (n) { frag.appendChild(document.importNode(n, true)); });
    clear(wrap);
    wrap.appendChild(frag);
    if (got.title) document.title = got.title;
    applyLive(lastLive);            // 先把即时比分放回去（不然刚换上的是服务器那一刻的比分）
    enhancePage();                  // 筛选、排序这些按刚才选的重新做一遍
    restoreView(wrap, view);
  }

  // ======================================================================
  // 第十部分：即时比分
  // ======================================================================
  // 页面上能放即时比分的地方带 data-live-slot="比赛编号"（首页“状态 / 赛果”那一格、单场页标题下面）。
  // 每次拿到 /api/live 就把里面的 .live-score 换成新的；没有比分了就拿掉，把“还没踢”之类（data-live-placeholder）露出来。
  // /api/live 的 match_ids = {"懂球帝编号": 我们的比赛编号}，按它对上页面上的比赛（对不上的不显示）。
  var lastLive = null;
  function applyLive(info) {
    if (!info) return;
    var status = info.status;
    var on = status === "ok" || status === "paused" || status === "error" || status === "unknown";
    var paused = status === "paused";
    var stale = !!info.stale || status === "error" || status === "unknown";
    var byMid = {};
    if (on && info.match_ids && info.matches && info.matches.length) {
      info.matches.forEach(function (item) {
        if (!item || typeof item !== "object") return;
        var key = item.dqd_id;
        if (typeof key !== "number" && typeof key !== "string") return;
        var mid = info.match_ids[String(key)];
        if (typeof mid !== "number") return;
        byMid[mid] = byMid[mid] || [];
        byMid[mid].push(item);
      });
    }
    var slots = document.querySelectorAll("[data-live-slot]");
    var shownLive = 0;
    each(slots, function (slot) {
      var mid = Number(slot.getAttribute("data-live-slot"));
      // 页面上这场的联赛（同一个懂球帝编号在文件里出现两次、联赛不一样时，只认联赛对得上的那条，和服务器一样）
      var owner = document.querySelector("[data-match-id='" + mid + "'][data-league]");
      var league = owner ? owner.getAttribute("data-league") : "";
      var item = null;
      (byMid[mid] || []).forEach(function (x) {
        if (typeof x.league === "string" && x.league && league && x.league !== league) return;
        item = x;
      });
      var parts = liveParts(item, { paused: paused, stale: stale });
      var old = slot.querySelector(".live-score");
      var before = old ? old.textContent : "";
      if (old) old.parentNode.removeChild(old);
      var hold = slot.querySelector("[data-live-placeholder]");
      if (parts.length) {
        var box = el("span", "live-score");
        parts.forEach(function (p, i) {
          if (i) box.appendChild(document.createTextNode(" "));
          var s = el("span", p.cls, p.text);
          if (p.title) s.title = p.title;
          box.appendChild(s);
        });
        slot.insertBefore(box, slot.firstChild);
        if (hold) hold.hidden = true;
        if (before && before !== box.textContent) flash(box);      // 比分、分钟变了：闪一下
        shownLive++;
      } else if (hold) {
        hold.hidden = false;
      }
    });
    // 页面上方的提醒（只在这一页有即时比分的位置时才写）
    var note = "";
    if (slots.length && paused) note = "实时比分暂停" + (info.note ? "（" + String(info.note).slice(0, 80) + "）" : "") + "：比分可能不是最新的。";
    else if (shownLive && stale) note = "即时比分很久没更新了，比分可能不是最新的。";
    livebar.setLive(note);
  }

  // ======================================================================
  // 第十一部分：每 30 秒问一次服务器
  // ======================================================================
  // 问 /api/version（数据版本）和 /api/live（即时比分）：
  //   - 即时比分：只改比分那几格（上面的 applyLive）。
  //   - 数据版本变了：重新取这一页、换正文（上面的 swapWrap）。网页最开始的版本写在 <meta name="site-data-version">。
  //   - 网页切到后台（看不见）时不问，切回来时到点了马上问一次。
  //   - 服务器说“没登录”（401）：停下不再问，只在上面写一句，不跳转、不弹窗。
  //   - 连不上：下一次隔得久一点再问（最多 4 倍），连着 3 次连不上写一句提醒。
  var livebar = {
    box: null, main: null, live: null,
    make: function () {
      if (this.box) return;
      var box = el("div", "js-livebar");
      box.setAttribute("aria-live", "polite");
      this.main = el("span", "small muted");
      this.live = el("span", "js-livenote");
      this.live.hidden = true;
      box.appendChild(this.main);
      box.appendChild(this.live);
      var nav = document.querySelector("nav.lan-bar");
      var wrap = document.querySelector(".wrap");
      if (nav && nav.parentNode) nav.parentNode.insertBefore(box, nav.nextSibling);
      else if (wrap && wrap.parentNode) wrap.parentNode.insertBefore(box, wrap);
      else return;
      this.box = box;
    },
    setMain: function (text) { if (this.main) this.main.textContent = text; },
    setLive: function (text) {
      if (!this.live) return;
      this.live.textContent = text;
      this.live.hidden = !text;
    }
  };

  function hhmm(d) {
    function two(n) { return (n < 10 ? "0" : "") + n; }
    return two(d.getHours()) + ":" + two(d.getMinutes()) + ":" + two(d.getSeconds());
  }

  var poller = {
    every: 30,           // 隔几秒问一次（网页里的 <meta name="site-poll-seconds">，没有时 30）
    known: null,         // 现在这一页是哪一版数据做的
    timer: null,
    stopped: false,
    busy: false,
    fails: 0,
    nextAt: 0,           // 下一次该问的时间（毫秒）

    start: function () {
      var m = document.querySelector("meta[name='site-poll-seconds']");
      var n = m ? Number(m.getAttribute("content")) : 30;
      this.every = isFinite(n) && n >= 5 ? n : 30;
      var v = document.querySelector("meta[name='site-data-version']");
      this.known = v ? v.getAttribute("content") : null;
      livebar.setMain("自动更新：每 " + this.every + " 秒看一次有没有新数据和即时比分，不用自己刷新。");
      var self = this;
      document.addEventListener("visibilitychange", function () {
        if (self.stopped) return;
        if (document.hidden) { clearTimeout(self.timer); self.timer = null; return; }
        self.schedule(Math.max(0, self.nextAt - Date.now()));     // 切回来：到点了马上问，没到就等剩下的时间
      });
      this.schedule(this.every * 1000);
    },

    schedule: function (ms) {
      clearTimeout(this.timer);
      this.timer = null;
      if (this.stopped) return;
      this.nextAt = Date.now() + ms;
      if (document.hidden) return;                               // 看不见时不问（切回来时再排）
      var self = this;
      this.timer = setTimeout(function () { self.tick(); }, ms);
    },

    tick: function () {
      if (this.stopped || this.busy) return;
      this.busy = true;
      var self = this;
      Promise.all([getJSON("/api/version"), getJSON("/api/live")]).then(function (res) {
        var ver = res[0], live = res[1];
        if (ver.status === 401 || live.status === 401) { self.loginExpired(); return null; }
        if (live.status === 200 && live.body) { lastLive = live.body; safely("即时比分", function () { applyLive(live.body); }); }
        var v = ver.status === 200 && ver.body ? ver.body.data_version : null;
        if (!v) throw new Error("没拿到数据版本");
        self.fails = 0;
        livebar.setMain("自动更新：每 " + self.every + " 秒看一次，上次看是 " + hhmm(new Date()) + "（本机时间）。");
        if (self.known === null) { self.known = v; return null; }
        if (v === self.known) return null;
        // 数据变了：重新取这一页、换正文
        return fetchPage().then(function (got) {
          if (got.login) { self.loginExpired(); return; }
          if (got.failed) return;                                   // 这次没取到：下次再试（版本还是旧的）
          safely("换正文", function () { swapWrap(got); });
          self.known = got.version || v;
          livebar.setMain("自动更新：数据刚换成新的（" + hhmm(new Date()) + "，本机时间），每 " + self.every + " 秒看一次。");
        });
      }).catch(function () {
        self.fails++;
        if (self.fails >= 3) livebar.setMain("自动更新：连着 " + self.fails + " 次连不上网站，过一会儿会再试（页面上是上次的内容）。");
      }).then(function () {
        self.busy = false;
        if (!self.stopped) self.schedule(self.every * 1000 * Math.min(4, 1 + self.fails));
      });
    },

    // 登录过期（服务器回 401）：不再问，只写一句；不跳到登录页、不弹窗，页面上的内容照样能看
    loginExpired: function () {
      this.stopped = true;
      clearTimeout(this.timer);
      this.timer = null;
      livebar.setMain("自动更新停了：登录已经过期。刷新网页会转到登录页，登录后接着更新。");
    }
  };

  // ======================================================================
  // 第十二部分：数据状态页上的操作按钮（需求文档第 8 节“网站上的操作按钮”）
  // ======================================================================
  // 服务器在数据状态页上插了一块“操作”（.act-panel），每个按钮是一个普通表单，没有脚本时照样能交（交完跳回这一页）。
  // 有脚本时：拦下表单，用 fetch 发同样的 POST（同样的表单字段，带着 csrf 防伪口令），不用整页跳转；
  // 有在跑的按钮就每隔 data-poll-seconds 秒问一次 /api/actions，都跑完了就停。所有字都用 textContent 放。
  function actionRow(name) {
    var rows = document.querySelectorAll(".act-row");
    for (var i = 0; i < rows.length; i++) if (rows[i].getAttribute("data-action") === name) return rows[i];
    return null;
  }

  // 按钮下面那一句（state：idle / running / done / error，样式按它换颜色）；正在跑时按钮点不了
  function setActionState(name, state, text) {
    var row = actionRow(name);
    if (!row) return;
    var span = row.querySelector(".act-state");
    var btn = row.querySelector("button");
    if (span) { span.textContent = String(text); span.setAttribute("data-state", String(state)); }
    if (btn) btn.disabled = state === "running";
  }

  // 用服务器给的每个按钮的情况更新页面；返回“还有没有在跑的”
  function applyActions(list) {
    var running = false;
    each(list || [], function (a) {
      if (!a || typeof a.name !== "string") return;
      setActionState(a.name, a.state, typeof a.text === "string" ? a.text : "");
      if (a.state === "running") running = true;
    });
    return running;
  }

  // “操作”标题下面的一句提示（和没有脚本时跳回来写的那句一样）
  function showActionNotice(text) {
    var panel = document.querySelector(".act-panel");
    if (!panel) return;
    var p = panel.querySelector(".act-notice");
    if (!p) {
      p = el("p", "act-notice");
      p.setAttribute("role", "status");
      var h = panel.querySelector("h2");
      panel.insertBefore(p, h ? h.nextSibling : panel.firstChild);
    }
    p.textContent = String(text);
  }

  // 发按钮的表单（POST）。只发到本站 /api/action/ 开头的地址；最多等 15 秒。返回 {status, body}
  function postAction(url, body) {
    var ctl = typeof AbortController === "function" ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, 15000) : null;
    return fetch(url, { method: "POST", credentials: "same-origin", cache: "no-store",
                        headers: { Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded" },
                        body: body, signal: ctl ? ctl.signal : undefined })
      .then(function (res) {
        if (timer) clearTimeout(timer);
        var type = res.headers.get("Content-Type") || "";
        if (type.indexOf("application/json") < 0) return { status: res.status, body: null };
        return res.json().then(function (b) { return { status: res.status, body: b }; },
                               function () { return { status: res.status, body: null }; });
      }, function (err) {
        if (timer) clearTimeout(timer);
        throw err;
      });
  }

  // 有按钮在跑时隔几秒问一次 /api/actions；网页切到后台时隔久一点（10 倍）；都跑完了、登录过期了就停
  var actionPoll = {
    every: 3, timer: null, busy: false, fails: 0,
    schedule: function () {
      clearTimeout(this.timer);
      var self = this;
      var ms = this.every * 1000 * (document.hidden ? 10 : 1) * Math.min(4, 1 + this.fails);
      this.timer = setTimeout(function () { self.tick(); }, ms);
    },
    tick: function () {
      if (this.busy) return;
      this.busy = true;
      var self = this;
      getJSON("/api/actions").then(function (got) {
        self.busy = false;
        if (got.status === 401) { showActionNotice("登录已经过期：刷新网页会转到登录页，登录后再看。"); return; }
        if (got.status === 200 && got.body) {
          self.fails = 0;
          if (applyActions(got.body.actions)) self.schedule();
          return;
        }
        if (got.status >= 500 && ++self.fails < 20) self.schedule();      // 服务器一时出错：过一会儿再问；别的（404 等）停下
      }, function () {
        self.busy = false;
        if (++self.fails < 20) self.schedule();                            // 连不上：过一会儿再问
      });
    }
  };

  function enhanceActions() {
    var panel = document.querySelector(".act-panel");
    if (!panel || panel.getAttribute("data-js-actions")) return;
    panel.setAttribute("data-js-actions", "1");
    var n = Number(panel.getAttribute("data-poll-seconds"));
    actionPoll.every = isFinite(n) && n >= 1 ? n : 3;
    each(panel.querySelectorAll("form.act-form"), function (form) {
      form.addEventListener("submit", function (ev) {
        var row = form.parentNode;
        var name = row && row.getAttribute ? row.getAttribute("data-action") : null;
        var url = form.getAttribute("action") || "";
        if (!name || url !== "/api/action/" + name) return;               // 对不上就不拦，让浏览器照常提交
        ev.preventDefault();
        var input = form.querySelector("input[name='csrf']");
        var btn = form.querySelector("button");
        var label = btn ? btn.textContent : name;
        if (btn) btn.disabled = true;
        setActionState(name, "running", "正在发给服务器……");
        postAction(url, "csrf=" + encodeURIComponent(input ? input.value : "")).then(function (got) {
          var b = got.body;
          if (got.status === 401) {
            setActionState(name, "idle", "登录已经过期：刷新网页会转到登录页，登录后再点。");
            return;
          }
          var text = b && typeof b.text === "string" ? b.text : "没能启动（服务器回 " + got.status + "）。";
          showActionNotice("“" + label + "”：" + text);
          if (b && b.actions) {
            if (applyActions(b.actions)) actionPoll.schedule();
          } else {
            actionPoll.tick();                                             // 没带每个按钮的情况（例如页面过期）：问一次，恢复成真实的样子
          }
        }, function () {
          showActionNotice("“" + label + "”：连不上网站，不知道点没点成；过一会儿刷新这一页看看。");
          actionPoll.tick();
        });
      });
    });
    // 页面打开时就有在跑的（刚才点过、刷新了页面）：接着问
    if (panel.querySelector(".act-state[data-state='running']")) actionPoll.schedule();
  }

  // ======================================================================
  // 第十三部分：开始
  // ======================================================================
  // 每一块单独做、单独 try/catch。换正文以后也再调一次（工具栏这些会重新加上，按刚才选的恢复）
  function enhancePage() {
    safely("首页筛选", enhanceIndex);
    safely("战绩页筛选", enhanceRecord);
    safely("虚拟投注页排序", enhanceBots);
    safely("版本切换", function () { enhanceVersions(poller.known); });
  }

  function init() {
    if (!document.querySelector(".wrap")) return;               // 不是本站的网页（比如登录页）：什么都不做
    safely("提示框", setupTips);
    safely("操作按钮", enhanceActions);
    // 数据状态页没有 site-poll-seconds 这个 meta：它不是现读现出的页面（本来每 10 分钟整页刷新），不开下面“自动更新”那一套
    if (!document.querySelector("meta[name='site-poll-seconds']")) return;
    safely("自动更新那一行", function () { livebar.make(); });
    // 先读网页里写的数据版本（版本切换的缓存用它）
    var v = document.querySelector("meta[name='site-data-version']");
    poller.known = v ? v.getAttribute("content") : null;
    enhancePage();
    safely("自动更新", function () { poller.start(); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
