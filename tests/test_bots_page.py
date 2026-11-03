"""虚拟投注页 bots.html（src/web/site/bots.py）的自动检查。只用临时文件夹里造的几行结果，不读真实回测结果。

造的结果（和 src/bots/stats.py 写出来的列名一样，几个数是随便定的）：
  融合价值5_最高   多家最高价格，CLV 区间整个大于 0（比价），ROI 区间跨过 0
  模型价值5_平博   平博价格，CLV、ROI 区间都整个小于 0
  对照_随机_平博   对照组，ROI 整个小于 0
  A<b>&"x'        名字带网页符号（要转义），注数不够、没排名次
"""
import copy
import json
import re
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.common.config import load_settings
from src.web.site import bots
from src.web.site.common import page
from src.web.site.data import SiteData

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
WEIRD = "A<b>&\"x'"


def summary_rows() -> list[dict]:
    base = {"对照组": "", "走（退回）": 0, "总注额": 0.0, "CLV>0的比例": 0.5, "ROI标记": "", "CLV标记": ""}
    rows = [
        {"名次": 1, "机器人": "融合价值5_最高", "价格": "最高", "规则": "value", "概率来源": "fusion", "注额": "flat",
         "投注场数": 800, "命中率": 0.16, "平均赔率": 9.4, "盈亏": -700.0, "ROI": -0.0088, "ROI下限": -0.066, "ROI上限": 0.057,
         "有CLV的注数": 400, "CLV平均": 0.09, "CLV下限": 0.084, "CLV上限": 0.096, "最大回撤": 3300.0, "最长连输": 45},
        {"名次": 2, "机器人": "模型价值5_平博", "价格": "平博", "规则": "value", "概率来源": "model", "注额": "flat",
         "投注场数": 600, "命中率": 0.26, "平均赔率": 5.2, "盈亏": -3700.0, "ROI": -0.0618, "ROI下限": -0.0947, "ROI上限": -0.0282,
         "有CLV的注数": 600, "CLV平均": -0.0208, "CLV下限": -0.0229, "CLV上限": -0.0187, "最大回撤": 4000.0, "最长连输": 38},
        {"名次": 3, "机器人": "对照_随机_平博", "对照组": "是", "价格": "平博", "规则": "random", "概率来源": "", "注额": "flat",
         "投注场数": 900, "命中率": 0.33, "平均赔率": 3.98, "盈亏": -3200.0, "ROI": -0.0354, "ROI下限": -0.057, "ROI上限": -0.0128,
         "有CLV的注数": 900, "CLV平均": -0.0238, "CLV下限": -0.0251, "CLV上限": -0.0226, "最大回撤": 3500.0, "最长连输": 25},
        # 名次和回测程序 src/bots/stats.py 一样写字“注数不够，不排”（不是空格子）
        {"名次": "注数不够，不排", "机器人": WEIRD, "价格": "最高", "规则": "top", "概率来源": "model", "注额": "kelly",
         "投注场数": 50, "命中率": 0.5, "平均赔率": 2.0, "盈亏": 10.0, "ROI": 0.002, "ROI下限": -0.05, "ROI上限": 0.06,
         "有CLV的注数": 20, "CLV平均": 0.01, "CLV下限": -0.02, "CLV上限": 0.03, "最大回撤": 200.0, "最长连输": 5},
    ]
    return [{**base, **r} for r in rows]


def write_results(folder, rows=None, files=("by_season", "by_period", "bankroll", "check")) -> None:
    """在 folder 里造一份结果（带 BOM 的 UTF-8，和回测程序写的一样）。files 决定除 summary.csv 以外造哪几个。"""
    folder.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else summary_rows()
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
    names = [r["机器人"] for r in rows]
    if "by_season" in files:
        season = [{"机器人": n, "赛季": y, "投注场数": 100, "盈亏": pl, "ROI": pl / 10000}
                  for n in names for y, pl in ((2012, 300.0), (2013, -900.0))]
        pd.DataFrame(season).to_csv(folder / "by_season.csv", index=False, encoding="utf-8-sig")
    if "by_period" in files:
        period = [{"机器人": n, "两段": "2012 起", "投注场数": 200, "ROI": -0.03, "ROI下限": -0.06, "ROI上限": 0.01} for n in names]
        pd.DataFrame(period).to_csv(folder / "by_period.csv", index=False, encoding="utf-8-sig")
    if "bankroll" in files:
        bank = [{"机器人": n, "初始资金": 10000.0, "最后资金": 7000.0, "最高点": 10500.0, "最大回撤": 0.33, "停了没有": "停了",
                 "停在哪天": "2012-09-27", "实际下了几注": 120, "单日上限跳过": 3} for n in names]
        bank[0]["最大回撤"] = 0.43                    # 停的时候还有没结算的注，回撤会比 30% 多不少
        bank[-1].update({"停了没有": "没停", "停在哪天": None, "最后资金": 10100.0})
        pd.DataFrame(bank).to_csv(folder / "bankroll.csv", index=False, encoding="utf-8-sig")
    if "check" in files:
        check = {"版本": "bots_v1", "模型": "gb_l7", "时点": "T-24", "赛季": [2012, 2013],
                 "场数": {"有概率也有矩阵的比赛": 1234},
                 "随机机器人的理论值": {"对照_随机_平博": {"理论ROI": -0.0243, "实际ROI": -0.0354,
                                                          "区间": [-0.057, -0.0128], "理论值在区间里": True}}}
        (folder / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def site(tmp_path):
    s = copy.deepcopy(load_settings())
    s["site"]["bots_dir"] = str(tmp_path / "bots_v1")
    return SiteData(NOW, s), tmp_path / "bots_v1"


def text_of(html: str) -> str:
    return html.replace("&#x27;", "'").replace("&amp;", "&").replace("&quot;", '"')


def check_safe(html: str) -> None:
    """全站的规矩：没有脚本、没有“推荐”、写明虚拟和不鼓励下注。"""
    assert "<script" not in html.lower()
    assert "推荐" not in html
    assert not re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", html, re.I)   # 空值写“—”，不出现字面的 nan
    assert "不用真钱" in html and "不连任何博彩账户" in html and "只是历史回测" in html
    assert "体育彩票和福利彩票" in html and "不鼓励下注" in html


def test_no_results_folder(site):
    data, _ = site
    html = bots.render(data)
    check_safe(html)
    assert "还没有虚拟投注的结果" in html and "<table" not in html


def test_no_setting(site):
    data, _ = site
    del data.settings["site"]["bots_dir"]
    assert "还没有虚拟投注的结果" in bots.render(data)


def test_bad_summary_is_not_used(site):
    data, folder = site
    folder.mkdir(parents=True)
    pd.DataFrame([{"机器人": "x", "ROI": "不是数字"}]).to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
    html = bots.render(data)
    assert "还没有虚拟投注的结果" in html and "格式不对" in html and "<table" not in html
    rows = summary_rows()
    rows[0]["ROI"] = "abc"                                  # 列都在、但数字列里有字
    write_results(folder, rows)
    assert "还没有虚拟投注的结果" in bots.render(data)


def test_full_page(site):
    data, folder = site
    write_results(folder)
    (folder / "bets.csv").write_text("这不是 csv\x00", encoding="utf-8")   # 每注明细不读，坏了也没关系
    html = bots.render(data)
    check_safe(html)
    t = text_of(html)
    # 最上面的结论：按文件里的数算
    assert "4 个机器人没有一个 ROI 显著为正" in t and "2 个区间整个小于 0" in t
    assert "抽水是真实成本" in t and "每下 100 亏 2.4" in t and "−3.5%" in t
    assert "模型没有打败平博收盘" in t and "−2.1%" in t
    # CLV 和“对照_随机_平博”（−2.4%）对比，不说成“买到的价格比收盘差”
    assert "随便买平博的价格 CLV 是 −2.4%" in t and "只比随便买多 0.3 个百分点" in t and "比收盘差" not in t
    assert "下注时未必下得到" in t and "+9.0%" in t and "ROI 区间仍然跨过 0" in t
    assert "4 个里3 个因为从最高点回撤太多被停掉" in t
    assert t.index("先看结论") < t.index("机器人排行榜")
    # 回测范围、规则一句话（门槛从 settings.yaml bots.list 拿）、四分之一凯利、对照组
    assert "2012/13–2013/14" in t and "gb_l7" in t and "1,234 场" in t and "期末考试" in t
    assert "比平博去掉抽水后的公平赔率至少高出 5%" in t and "其实是在比价" in t and "四分之一凯利" in t and "对照组" in t
    # 排行榜：有名次的按名次，注数不够的放最后
    board = t.index("机器人排行榜")                      # 结论和“几个词”里也提到对照组，从排行榜开始找
    assert (t.index("融合价值5_最高", board) < t.index("模型价值5_平博", board) < t.index("对照_随机_平博", board)
            < t.index("注数不够，不排", board))
    assert "确实亏" in t and "看不出赚亏" in t and "高于平博收盘的公平价格" in t
    assert "低于平博收盘的公平价格（随便买平博是 −2.4%）" in t and "和平博收盘的公平价格看不出高低" in t
    # 几个词：CLV 是和去掉抽水后的公平价格比；资金模拟的说明（天数从 settings 算，回撤范围从 bankroll.csv 算）
    assert "平博收盘去掉抽水后的公平价格" in t and "CLV 约等于负的抽水" in t and "“对照_随机_平博”是 −2.4%" in t
    assert "约 4 天的注没结算（结果要开球后 72 小时才算）" in t and "这一版 33% 到 43%" in t
    assert "最后资金比起始多，不代表这个机器人赚钱" in t and "固定注额的 ROI" in t
    assert "没发现回测程序算错" in t and "算得对" not in t
    # 资金模拟、小图（每个机器人一张）、两段和每个赛季
    assert "10,000 → <b>7,000</b>" in t and "2012-09-27 停了" in t and "一直没停" in t and "最多回撤 43%" in t
    assert html.count("<svg") == 4 and "累计 −600" in t
    assert "ROI（2012 起）" in t and "1 / 2" in t and "最好 2012/13" in t and "最差 2013/14" in t
    assert "这一版没有找到值得拿去考的" in t


def test_names_are_escaped(site):
    data, folder = site
    write_results(folder)
    html = bots.render(data)
    assert WEIRD not in html and "A&lt;b&gt;&amp;&quot;x&#x27;" in html


def test_only_summary(site):
    """只有 summary.csv：排行榜照常，缺的几块写原因、不显示。"""
    data, folder = site
    write_results(folder, files=())
    html = bots.render(data)
    check_safe(html)
    assert "机器人排行榜" in html and "<svg" not in html
    for name in ("by_season.csv", "by_period.csv", "bankroll.csv", "check.json"):
        assert f"{name}" in html
    assert "<b>抽水是真实成本</b>" not in html and "没有随机下注的理论值" in html


def test_bad_check_json(site):
    data, folder = site
    write_results(folder)
    (folder / "check.json").write_text("[1, 2", encoding="utf-8")
    html = bots.render(data)
    assert "check.json 格式不对" in html and "机器人排行榜" in html


def test_positive_roi_is_not_called_none(site):
    """真有区间整个大于 0 的：结论改成提醒“可能是碰巧”，不写“没有一个”。"""
    data, folder = site
    rows = summary_rows()
    rows[0].update({"ROI": 0.05, "ROI下限": 0.01, "ROI上限": 0.09})
    write_results(folder, rows)
    t = text_of(bots.render(data))
    assert "没有一个 ROI 显著为正" not in t and "1 个 ROI 区间整个大于 0" in t and "碰巧" in t
    assert "这一版没有找到值得拿去考的" not in t and "区间整个大于 0" in t


def test_rank_text_means_not_ranked(site):
    """名次写字“注数不够，不排”（回测程序就是这么写的）：页面照常生成，那个机器人显示没排名、排在最后；别的数字列有字照样算格式不对。"""
    data, folder = site
    write_results(folder)
    res, why = bots.load(data.settings)
    assert why == "" and res["summary"]["名次"].isna().sum() == 1
    t = text_of(bots.render(data))
    assert "机器人排行榜" in t and "格式不对" not in t
    board = t.index("机器人排行榜")
    weird = bots.e(WEIRD)                               # 名字在网页里是转义过的
    assert t.index("对照_随机_平博", board) < t.index("注数不够，不排", board) < t.index(text_of(weird), board)   # 没排名的在最后


def test_theory_check_field_decides_sentence(site):
    """“没发现回测程序算错”按 check.json 的“理论值在区间里”写：false 点名要查，没有这个字段就不写。"""
    data, folder = site
    write_results(folder)
    check = json.loads((folder / "check.json").read_text(encoding="utf-8"))
    check["随机机器人的理论值"]["对照_随机_平博"]["理论值在区间里"] = False
    (folder / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")
    t = text_of(bots.render(data))
    assert "没发现回测程序算错" not in t and "对照_随机_平博 的理论值不在回测的区间里，要先查回测程序" in t
    del check["随机机器人的理论值"]["对照_随机_平博"]["理论值在区间里"]
    (folder / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")
    t = text_of(bots.render(data))
    assert "没发现回测程序算错" not in t and "要先查回测程序" not in t and "抽水是真实成本" in t


def test_best_price_roi_sentence_follows_data(site):
    """“ROI 区间仍然跨过 0”看“融合价值 × 多家最高”那几行的数：整个大于 0 时不这么说。"""
    data, folder = site
    rows = summary_rows()
    rows[0].update({"ROI": 0.05, "ROI下限": 0.01, "ROI上限": 0.09})
    write_results(folder, rows)
    t = text_of(bots.render(data))
    assert "ROI 区间仍然跨过 0" not in t and "其中 1 个 ROI 区间整个大于 0，但价格未必下得到" in t


def test_empty_cells_show_dash(site):
    """summary 有空的投注场数、命中率、价格：只是那几格写“—”，页面照常，没有字面的 nan。"""
    data, folder = site
    rows = summary_rows()
    rows[1].update({"投注场数": None, "命中率": None, "价格": None, "最长连输": None})
    write_results(folder, rows)
    html = bots.render(data)
    check_safe(html)
    assert "机器人排行榜" in html and "格式不对" not in html and "—" in html


def test_empty_season_only_hides_that_part(site):
    """by_season.csv 赛季空：只是小图和赛季统计不显示，写原因；排行榜照常。"""
    data, folder = site
    write_results(folder)
    season = pd.read_csv(folder / "by_season.csv", encoding="utf-8-sig")
    season.loc[0, "赛季"] = None
    season.to_csv(folder / "by_season.csv", index=False, encoding="utf-8-sig")
    html = bots.render(data)
    check_safe(html)
    assert "by_season.csv 格式不对" in html and "机器人排行榜" in html and "<svg" not in html


def test_check_counts_not_a_dict(site):
    """check.json 的“场数”不是字典：只是不写场数，别的照常。"""
    data, folder = site
    write_results(folder)
    check = json.loads((folder / "check.json").read_text(encoding="utf-8"))
    check["场数"] = [1, 2, 3]
    (folder / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")
    t = text_of(bots.render(data))
    check_safe(t)
    assert "2012/13–2013/14" in t and "1,234 场" not in t and "机器人排行榜" in t


def test_unexpected_error_gives_placeholder(site, monkeypatch):
    """这一页出了想不到的错：只换成“虚拟投注结果格式不对”的占位页，不抛出去。"""
    data, folder = site
    write_results(folder)

    def boom(*args, **kwargs):
        raise RuntimeError("C:/某个路径/不该出现在网页上")

    monkeypatch.setattr(bots, "curves", boom)
    html = bots.render(data)
    check_safe(html)
    assert "虚拟投注结果格式不对" in html and "RuntimeError" in html and "某个路径" not in html and "<table" not in html


def test_render_checked_reports_error_kind(site, monkeypatch):
    """render_checked：正常时出错种类是 None；走了兜底时告诉调用的地方是哪种错（build_site 写进摘要）。"""
    data, folder = site
    write_results(folder)
    html, kind = bots.render_checked(data)
    assert kind is None and "机器人排行榜" in html

    def boom(*args, **kwargs):
        raise KeyError("x")

    monkeypatch.setattr(bots, "conclusions", boom)
    html, kind = bots.render_checked(data)
    assert kind == "KeyError" and "虚拟投注结果格式不对" in html


def test_os_error_hides_path(site, monkeypatch):
    """summary.csv 读不了（OSError）：只写错误种类，不把说明里的路径写到网页上。"""
    data, folder = site
    write_results(folder)

    def no_read(path, *args, **kwargs):
        raise PermissionError(13, "拒绝访问", "C:/某个人的电脑/summary.csv")

    monkeypatch.setattr(bots, "_csv", no_read)
    html = bots.render(data)
    assert "还没有虚拟投注的结果" in html and "PermissionError" in html and "某个人的电脑" not in html


def test_empty_profit_season_not_nan(site):
    """by_season 某个赛季盈亏空：去掉那一行再累加，小图照画，SVG 里没有 nan。"""
    data, folder = site
    write_results(folder)
    season = pd.read_csv(folder / "by_season.csv", encoding="utf-8-sig")
    season.loc[0, "盈亏"] = None
    season.to_csv(folder / "by_season.csv", index=False, encoding="utf-8-sig")
    html = bots.render(data)
    check_safe(html)
    assert html.count("<svg") == 4 and "格式不对" not in html


def test_nav_links_to_bots_page():
    assert "href='bots.html'" in page("x", "", "", NOW, 15)
    assert "href='../bots.html'" in page("x", "", "../", NOW, 15)
