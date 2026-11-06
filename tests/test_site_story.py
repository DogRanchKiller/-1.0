"""单场页“看得懂”的几块（src/web/site/story.py）和首页、战绩页的简化。

用 tests/test_site.py 的临时数据库（利物浦 对 阿森纳 等）和 tests/test_explanation_pages.py 的小解释载荷，不碰真实数据。
"""
import html
import re

from src.web.site import story
from src.web.site.common import TeamFiles, bar_html
from src.web.site.match import render as render_match
from src.web.site.pages import index_guide
from tests.test_explanation_pages import site_data
from tests.test_site import build, env, plain, read, write_review  # noqa: F401  （env 是 pytest 的 fixture）


def text_of(page: str) -> str:
    """去掉标签、换回转义过的符号，只留文字，方便查句子。"""
    return html.unescape(re.sub(r"<[^>]+>", "", page))


def test_match_page_tells_the_story_with_this_matchs_numbers(env):
    env[1]["site"]["show_confidence"] = True          # 这条查“显示信心等级”时第 6 步怎么写；不显示的见下面另一条
    out, _ = build(env)
    page = read(out, "match/1.html")
    t = text_of(page)
    # 一句话 + 大号概率条（融合概率 50 / 27 / 23，最高比第二高多 23 个百分点 ≥ 4，写倾向）
    assert "利物浦赢的可能更大（50%）" in t and "class='pbar big'" in page
    # 结论链 6 步，用这一版自己的数：模型矩阵是 1.4 对 1.2 的泊松
    assert "结论是怎么得出来的" in t and page.count("<li><b class='step'>") == 6
    assert "利物浦平均进 1.40 球" in t and "阿森纳平均进 1.20 球" in t
    assert "最可能的 3 个是：1–1" in t
    # 造的数据模型叫 gb，不是选比例用的 gb_l7：不把 gb_l7 的回测成绩写成这一版的理由
    assert "合并时赔率占 60%、模型占 40%" in t and "用 gb_l7 模型回测选出来的（这一版用的是 gb）" in t and "0.979" not in t
    assert "这一版用到了赔率，但没有单独存赔率算出来的概率" in t       # 造的数据没存“赔率概率”，不编
    assert "高 23.0 个百分点，现在的门槛是 4 个百分点。这一版写的是“倾向 主胜”。" in t
    assert "平局 27.0%，平局风险的线是 28%，这一版没标平局风险" in t      # 线读 settings.yaml lean.draw（0.28）
    assert "按门槛写倾向，再定信心" in t and "所以几乎每场都是 ●●● 高" in t
    # 四块分析：没存模型解释的写明没有；伤停与首发写明模型不看
    for title in ("实力对比", "近期状态与赛程", "伤停与首发", "盘口与水位解读"):
        assert f"<h3>{title}</h3>" in page
    assert "这一版没有存模型输入（模型解释），没法列出" in t
    assert "模型现在不看伤停和首发" in t and "这一版出的时候没有首发信息" in t
    assert "亚盘主盘口 主让半球（2 家平均）" in t and "大小球 2.5 球（2 家）" in t
    assert "这一版没有保存模型解释，没法列出影响最大的因素" in t
    # 技术信息挪到最下面的“技术细节”，不在结论卡片里
    assert t.index("流程记录和技术细节") > t.index("赛后复盘")
    assert "gb / gb-1" not in t[:t.index("更多数据")]
    assert page.count("<svg") == 5 and "<script" not in page.lower() and "推荐" not in page


def test_story_uses_saved_explanation_inputs_and_groups_factors():
    data = site_data()
    # site_data 里最新一版（T-1）没存解释；只留存了解释的 T-24，看结论链用的是它自己的输入
    data.preds[10] = {k: v for k, v in data.preds[10].items() if k[2] == "T-24"}
    page = render_match(data, 10, TeamFiles(data.team_zh))
    t = text_of(page)
    assert "利物浦距上一场休息 7 天" in t                       # rest_days_own（主队那一条）
    assert "模型对“一场普通比赛”先估利物浦赢 40.0%，看完这场的数据后改成 44.0%（+4.0 个百分点）" in t
    # 主队那条、客队那条都算：休息天数 2.4 + 1.2 + 0.8 + 0.4 = 4.8；近期状态 -0.9 + 0.3 - 0.4 + 0.2 = -0.8；合计 +4.0
    assert "休息天数和赛程：让利物浦赢的可能多了 4.8 个百分点" in t
    assert "近几场的进球、失球、射正和积分：让利物浦赢的可能少了 0.8 个百分点" in t
    assert "合并时赔率占 100%、模型占 0%" in t and "最后的概率就是赔率的概率" in t


def test_bad_explanation_is_not_used_for_story():
    data = site_data()
    row = next(iter(data.preds[10].values()))
    row["detail"]["模型解释"]["prediction"]["home_win"] = .8
    data.preds[10] = {k: v for k, v in data.preds[10].items() if v is row}
    t = text_of(render_match(data, 10, TeamFiles(data.team_zh)))
    assert "没法列出影响最大的因素" in t and "距上一场休息" not in t


def test_headline_without_lean_and_draw_risk():
    data = site_data()
    row = next(iter(data.preds[10].values()))
    row["detail"]["融合概率"] = [.36, .3, .34]
    row["detail"]["融合"]["倾向"] = {"result": {"side": 0, "p": .36, "lean": False}, "draw": {"p": .3, "risk": True}}
    assert story.headline(data, row, "Liverpool", "Arsenal") == "两边差不多，没有明显倾向；平局风险高（30%）"


def test_elo_gap_uses_rounded_numbers_and_factor_groups():
    t = story.strength_block({("home", "elo_own"): -64.2, ("home", "elo_opp"): -54.7}, "甲", "乙")
    assert "甲 -64，乙 -55，乙高 9 分" in t
    assert story.group_of("dyn_home") == "主场优势和联赛特点"
    assert story.group_of("dyn_loglam_for") == "两队最近的进攻、防守评分"
    assert story.group_of("ewm5_pts_opp") == story.group_of("form5_gf_own")
    assert story.group_of("unknown_x") == "其他"


def test_probability_bar_widths_and_labels():
    big = bar_html([.5, .3, .2], big=True)
    assert "width:50.0%'>50%" in big and "width:20.0%'>20%" in big
    small = bar_html([.5, .3, .2])
    assert "width:50.0%'></span>" in small                     # 小号条不写字


def test_index_cell_is_short_and_record_starts_with_plain_summary(env):
    write_review(env[2])
    out, _ = build(env)
    index = read(out, "index.html")
    assert "怎么看这张表" in index and "class='pbar'" in index
    assert "开球前 24 小时这一版" in plain(index)
    assert "title='gb / gb-1" in index                         # 模型版本只在悬停提示里
    record = text_of(read(out, "record.html"))
    assert "四个玩法的命中率" in record and "对照：一直猜主胜" in record
    assert record.index("四个玩法的命中率") < record.index("详细统计表")



def test_index_guide_says_where_probabilities_come_from():
    conf = {"downgrade_factors": [], "no_odds_downgrade": 1}
    text = plain(index_guide({"fusion": {"w": 1.0}, "confidence": conf}))     # 没写 site.show_confidence：当 true
    assert "全部来自博彩公司的赔率" in text and "几乎每场都是 ●●● 高" in text
    off = plain(index_guide({"fusion": {"w": 1.0}, "confidence": conf, "site": {"show_confidence": False}}))
    assert "全部来自博彩公司的赔率" in off and "●●●" not in off and "信心等级" not in off
    assert "概率来自去水赔率" in off and "只有模型（没和赔率融合）" in off
    assert "全部来自" not in plain(index_guide({"fusion": {"w": 0.7}, "confidence": conf}))   # 模型也占一部分时不能这么写


def one_row():
    """site_data 里比赛 10 的一版，只留它一版（方便单独改存档内容看页面怎么写）。"""
    data = site_data()
    row = next(iter(data.preds[10].values()))
    data.preds[10] = {k: v for k, v in data.preds[10].items() if v is row}
    return data, row


def test_old_row_without_fusion_does_not_claim_no_lean():
    # 接融合之前存的旧版没有“融合”：不能写“两边差不多”，只说最可能的是哪个，并注明没存倾向
    data, row = one_row()
    row["detail"].pop("融合", None)
    row["detail"].pop("融合概率", None)
    row.update(p_home=.5, p_draw=.27, p_away=.23)
    head = story.headline(data, row, "Liverpool", "Arsenal")
    assert head == "最可能是利物浦赢（50%；这一版没有存倾向）"
    t = text_of(story.chain_html(data, row, "Liverpool", "Arsenal"))
    assert "这一版没有存倾向；按现在的门槛够" in t and "平局风险的线是 28%，这一版没有存平局风险。" in t
    assert "这一版这一版" not in t and "两边差不多" not in t
    assert "信心" not in t                                       # 网站不显示信心等级（site.show_confidence: false）
    data.settings["site"]["show_confidence"] = True
    t = text_of(story.chain_html(data, row, "Liverpool", "Arsenal"))
    assert "这一版没有存信心等级和提醒因素。" in t and "没有碰上要提醒的因素" not in t


def test_stored_lean_wins_over_recomputed_threshold():
    # 4 位小数的概率差正好在门槛边上：页面跟着存档写，并说明为什么和重算的不一样
    data, row = one_row()
    row["detail"]["融合概率"] = [.42, .20, .38]
    row["detail"]["融合"]["倾向"] = {"result": {"side": 0, "p": .42, "lean": False}, "draw": {"p": .2, "risk": False}}
    assert story.headline(data, row, "Liverpool", "Arsenal") == "两边差不多，没有明显倾向"
    t = text_of(story.lean_rule_text(data.settings, row, [.42, .20, .38]))
    assert "高 4.0 个百分点" in t and "这一版写的是“无明显倾向”。（以出这一版时定下的为准" in t


def test_draw_shown_with_one_decimal_near_line():
    data, row = one_row()
    row["detail"]["融合"]["倾向"] = {"result": {"side": 0, "p": .4, "lean": False}, "draw": {"p": .2796, "risk": False}}
    t = text_of(story.lean_rule_text(data.settings, row, [.40, .2796, .3204]))
    assert "平局 28.0%，平局风险的线是 28%，这一版没标平局风险" in t and "没到 28%" not in t
    assert "高 8.0 个百分点" in t                                # 40.0 − 32.0，用显示出来的数减


def test_odds_probability_step_and_no_odds_downgrade():
    data, row = one_row()
    row["detail"]["赔率概率"] = [.45, .3, .25]
    row["detail"]["赔率公司数"] = 12
    t = text_of(story.chain_html(data, row, "Liverpool", "Arsenal"))
    assert "12 家公司的胜平负赔率，先去掉公司的抽水，再平均成概率：利物浦赢 45% / 平 30% / 阿森纳赢 25%" in t
    row["detail"]["融合"] = {"状态": "无赔率，未融合", "信心等级": 2, "降级候选因素": {"early": True}}
    t = text_of(story.confidence_text(data.settings, row))
    assert "这一版一家赔率都没有，降 1 级" in t and "赛季前几轮（只作提醒，不降级）" in t and "●●○ 中" in t


def test_model_backtest_shown_only_for_the_model_that_chose_w():
    data, row = one_row()
    row["model"] = data.settings["fusion"]["model"]
    t = text_of(story.chain_html(data, row, "Liverpool", "Arsenal"))
    assert "0.979" in t and "0.964" in t


def test_odds_move_uses_only_prices_before_cutoff_each_book_against_itself():
    import pandas as pd
    o = pd.DataFrame([("A", "1x2", 2.0, 3.5, 4.0, "2026-11-05 08:00:00"), ("A", "1x2", 1.9, 3.6, 4.2, "2026-11-06 08:00:00"),
                      ("B", "1x2", 2.0, 3.4, 3.9, "2026-11-06 09:00:00"),       # B 后来才开，不算“变动”
                      ("A", "1x2", 1.2, 6.0, 9.0, "2026-11-06 13:00:00")],      # 截止以后的，不能用
                     columns=["bookmaker", "market", "home", "draw", "away", "changed_at_utc"])
    p0, p1, n = story.cutoff_move(o, "2026-11-06 10:00:00")
    assert n == 2
    from src.common.odds import devig
    a0, a1, b = (devig([[2.0, 3.5, 4.0]])[0], devig([[1.9, 3.6, 4.2]])[0], devig([[2.0, 3.4, 3.9]])[0])
    assert abs(p0[0] - (a0[0] + b[0]) / 2) < 1e-12 and abs(p1[0] - (a1[0] + b[0]) / 2) < 1e-12
    assert story.cutoff_move(o, "2026-11-01 00:00:00") is None


def test_form_uses_real_game_count():
    v = {("home", "form5_pts_own"): 1.5, ("home", "form5_gf_own"): 1.0, ("home", "form5_ga_own"): 2.0,
         ("home", "form_n_own"): 2.0}
    assert "甲最近 2 场：场均 1.5 分" in story.form_block(v, "甲", "乙")
