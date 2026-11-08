"""四个玩法的预测 picks_v3（把握等级上网站，工作树 P3，需求文档第 5 节“四个玩法的预测 picks_v3”）的自动检查。全是造的数据。

要检查的：
- 切点和分档：settings.yaml picks_v3 的数照抄 SEL（切点、每档命中率和场数），≥ 上切点 = 高、≥ 下切点 = 中，让球不分档。
- 大小球用哪条主盘口：存档的“大小球主盘口”合格就用它；没有或不合格（不在 0.5–6、不是 0.25 的倍数）用矩阵推出来的线，
  推的算法和 SEL（src/evaluate/pick_tiers_study.py）一模一样。
- 用哪个矩阵：存档的融合矩阵 → 按存的赔率概率重算（预览版）→ 模型矩阵（不分档）；只有赔率算的才分档。
- “模型自己”一列：不分档，大小球和预测那列同一条主盘口，让球、大小球标“写高了”，页面上只写方向；
  让球再写“历史上偏向受让一方”（shrink_study，不改模型）。
- 页面：单场页表里有把握和历史命中率、让球写不分档；首页那一格有“把握”一行、行上有 data-tier-*（值只有 high / mid / low）；
  没有脚本时页面照样完整；全站没有“推荐”。
- 单场页「多家平均」「分歧」（2026-11-08）：市场一侧必须用存档「赔率概率」（懂球帝多家平均），不能用「融合概率」（有平博时那是平博）；
  差 ≥ site.disagree_gap（6%）标分歧；没有多家平均写「没有市场价」。
- picks_v2 的结果不变（match_picks / model_picks 和以前一样，不带档），战绩页照旧；/api 原有字段不变，只多了 picks_v3、latest.tiers。
- 冻结的设置块没动：加 picks_v3 不改期末考试记下的设置。
"""
import copy
import html
import json
import re

import numpy as np
import pytest

from src.common.config import load_settings
from src.common.markets import poisson_score_matrix
from src.evaluate import final_exam as fe
from src.evaluate import pick_tiers_study as pts
from src.predict.conclusion import market_matrix
from src.web.site import picks as pk
from src.web.site.common import TeamFiles
from src.web.site.match import picks_html, render as render_match
from src.web.site.pages import index_guide, latest_cell, row_attrs
from tests.test_explanation_pages import site_data
from tests.test_site import all_pages, build, env, fake_matrix, read, write_review  # noqa: F401  （env 是 pytest 的 fixture）
from tests.test_serve import tree  # noqa: F401  （pytest 的 fixture）
from tests.test_serve_live import LiveRunning, get_json, world  # noqa: F401  （world 是 pytest 的 fixture）
from tests.test_site_js import JS, NODE, run_node

S = load_settings()


def text_of(page: str) -> str:
    """去掉标签、换回转义过的符号，只留文字。"""
    return html.unescape(re.sub(r"<[^>]+>", "", page))


def market(p=(0.62, 0.22, 0.16), line=2.5, p_over=0.55):
    """造一版“赔率算的”比分矩阵：按胜平负 + 大小球反推（和出预测时同一个方法）。"""
    m, _ = market_matrix({"p_1x2": list(p), "ou": {"line": line, "p_over": p_over}}, int(S["market_odds"]["max_goals"]))
    return m


def row_of(*, status="已融合", w=1.0, fp=(0.62, 0.22, 0.16), ou=(2.5, 0.55), main=2.75, fused=True, model=True,
           model_over=None, lean_ah=True):
    """造一版存档。fused = 存不存融合比分矩阵（定时版存，预览版不存）；ou = 融合用的大小球 (线, 大球胜算)，None = 没存；
    main = 存档的“大小球主盘口”；model = 存不存模型比分矩阵。"""
    f = {"状态": status, "w": w, "倾向": {"result": {"side": 0, "p": fp[0], "lean": True}, "draw": {"p": fp[1], "risk": False},
                                         "over25": None, "btts": None,
                                         "ah": {"side": "home", "p": 0.53, "line": -0.5, "lean": True} if lean_ah else None}}
    if ou is not None:
        f["大小球"] = {"line": ou[0], "p_over": ou[1], "n_books": 3}
    d = {"融合概率": list(fp), "融合": f, "大小球主盘口": main, "模型大球胜算": model_over}
    if fused and ou is not None:
        d["融合比分矩阵"] = np.round(market(fp, *ou), 6).tolist()
    if model:
        d["模型比分矩阵"] = fake_matrix(1.2, 1.4)
    return {"prediction_id": 1, "kind": "T-1", "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3, "detail": d}


# ---------- 设置照抄 SEL、分档 ----------
def test_settings_are_sel_numbers():
    c = S["picks_v3"]
    assert c["cut_seasons"] == [2012, 2019] and c["test_seasons"] == [2020, 2023] and c["test_n"] == 7228
    t = c["tiers"]
    assert t["1x2"]["cuts"] == [0.447, 0.574] and t["ou"]["cuts"] == [0.512, 0.524] and t["score"]["cuts"] == [0.405, 0.443]
    assert (t["1x2"]["high"]["rate"], t["1x2"]["high"]["n"]) == (0.6943, 2388)
    assert (t["ou"]["mid"]["rate"], t["ou"]["mid"]["n"]) == (0.5105, 2548)
    assert (t["score"]["low"]["rate"], t["score"]["low"]["n"]) == (0.3836, 3308)
    assert c["ah_all"] == {"rate": 0.5285, "n": 7228}
    for k in pk.TIERED:                                  # 每个玩法三档加起来就是检验季的场数
        assert sum(t[k][x]["n"] for x in pk.TIERS) == c["test_n"] == t[k]["all"]["n"]
    assert set(t) == set(pk.TIERED) and "ah" not in t    # 让球不分档


def test_tier_cut_points():
    tier = pk.tier_of
    assert tier("1x2", 0.574, S) == "high" and tier("1x2", 0.5739, S) == "mid"
    assert tier("1x2", 0.447, S) == "mid" and tier("1x2", 0.4469, S) == "low"
    assert tier("ou", 0.524, S) == "high" and tier("ou", 0.52, S) == "mid" and tier("ou", 0.5, S) == "low"
    assert tier("score", 0.443, S) == "high" and tier("score", 0.41, S) == "mid" and tier("score", 0.3, S) == "low"
    assert tier("ah", 0.6, S) is None and tier("1x2", None, S) is None and tier("1x2", float("nan"), S) is None
    # 和 SEL 的 tier 一样（同一组切点）
    import pandas as pd
    p = pd.Series([0.3, 0.447, 0.5, 0.574, 0.7])
    want = pts.tier(p, S["picks_v3"]["tiers"]["1x2"]["cuts"]).map({"高": "high", "中": "mid", "低": "low"}).tolist()
    assert [tier("1x2", x, S) for x in p] == want


# ---------- 大小球用哪条主盘口 ----------
def test_main_line_uses_archive_then_falls_back_to_implied():
    m = market()
    # 1. 存档的主盘口合格：就用它（不管矩阵推出来是哪条）
    assert pk.ou_main_line(row_of(main=2.75), m, S) == (2.75, "存档")
    assert pk.ou_main_line(row_of(main=3), m, S) == (3.0, "存档")
    # 2. 没有、不合格（不是 0.25 的倍数、超出 0.5–6、不是数字）：推出来的线
    want = pk.implied_line(m, S)
    for bad in (None, 2.6, 0.25, 6.5, "2.5", float("nan"), True):
        assert pk.ou_main_line(row_of(main=bad), m, S) == (want, "推算"), bad
    # 3. 连矩阵也没有：没有线
    assert pk.ou_main_line(row_of(main=None), None, S) == (None, None)


def test_implied_line_same_as_sel():
    """推出来的主盘口和 SEL 定切点时一样（pick_tiers_study.implied_lines），线和胜算都一样。"""
    mats = [market(), market((0.3, 0.3, 0.4), 2.5, 0.42), market((0.75, 0.15, 0.1), 3.25, 0.5),
            np.asarray(poisson_score_matrix(2.2, 1.9, max_goals=10))]
    tot = pts.totals(np.stack([np.asarray(x) for x in mats]))
    lines, shares = pts.implied_lines(tot, pts.ou_grid({"pick_tiers_study": S["pick_tiers_study"]}))
    for m, line, share in zip(mats, lines, shares):
        got = pk.implied_line(np.asarray(m), S)
        assert got == float(line)
        assert abs(pk._over_share(np.asarray(m), got) - share) < 1e-9
    assert np.array_equal(pk.ou_lines(S), pts.ou_grid(S))      # 两边试的线一样


def test_ou_pick_on_main_line_matches_sel_side_and_share():
    """主盘口上大的胜算 ≥ 50% 写“大”，把握分是那一边的胜算（赢半算一半、走盘不算），和 SEL 的 picks 一样。"""
    r = row_of(main=2.75)
    ou = pk.match_picks_v3(r, S)["ou"]
    m = pk.market_matrix_v3(r, S)[0]
    over = pts.over_shares(pts.totals(m[None]), [2.75])[0, 0]
    assert ou["line"] == 2.75 and ou["line_from"] == "存档" and ou["matrix"] == "存档"
    assert ou["side"] == ("over" if over >= 0.5 else "under")
    assert abs(ou["p"] - (over if over >= 0.5 else 1 - over)) < 1e-9
    assert ou["tier"] == pk.tier_of("ou", ou["p"], S)


# ---------- 用哪个矩阵、哪些分档 ----------
def test_matrix_from_archive_then_rebuilt_then_model():
    stored = row_of(fused=True)
    m1, how1 = pk.market_matrix_v3(stored, S)
    assert how1 == "存档"
    # 预览版：没存矩阵，按存的赔率概率重算，和存下来的一样（存档是 6 位小数）
    preview = row_of(fused=False)
    m2, how2 = pk.market_matrix_v3(preview, S)
    assert how2 == "重算" and np.abs(m1 - m2).max() < 1e-5
    assert not m2.flags.writeable                                  # 记下来的矩阵大家共用，不让改
    a, b = pk.match_picks_v3(stored, S), pk.match_picks_v3(preview, S)
    for k in pk.MARKETS:                                            # 除了“矩阵从哪来”和小数点后几位，结论一样
        da = {x: y for x, y in a[k].items() if x not in ("matrix", "p")}
        db = {x: y for x, y in b[k].items() if x not in ("matrix", "p")}
        assert da == db and abs(a[k]["p"] - b[k]["p"]) < 1e-4, k
    assert b["score"]["matrix"] == "重算" and b["score"]["tier"] in pk.TIERS
    # 不是 w = 1、不是“已融合”、没存融合用的大小球：没有赔率矩阵，用模型矩阵、不分档
    for r in (row_of(fused=False, w=0.6), row_of(fused=False, status="赔率不全"), row_of(fused=False, ou=None),
              row_of(fused=False, status="无赔率，未融合")):
        assert pk.market_matrix_v3(r, S) == (None, None)
        ps = pk.match_picks_v3(r, S)
        assert ps["score"]["matrix"] == "模型" and ps["score"]["tier"] is None and ps["score"]["source"] == "模型"
        assert ps["ou"]["tier"] is None and ps["ou"]["source"] == "模型"
    # 胜平负：“赔率不全”也是赔率算的，照样分档；没有赔率不分档
    assert pk.match_picks_v3(row_of(fused=False, status="赔率不全"), S)["1x2"]["tier"] == "high"
    assert pk.match_picks_v3(row_of(status="无赔率，未融合"), S)["1x2"]["tier"] is None
    assert pk.match_picks_v3(row_of(fused=False, w=0.6), S)["1x2"]["tier"] is None


def test_no_matrix_uses_model_share_only_on_archived_line():
    """连模型矩阵也没有：只有用的是存档的主盘口，才用“模型大球胜算”（它就是在这条线上算的）；不分档，比分不预测。"""
    r = row_of(fused=False, ou=None, model=False, main=2.25, model_over=0.47)
    ps = pk.match_picks_v3(r, S)
    assert ps["ou"]["line"] == 2.25 and ps["ou"]["side"] == "under" and abs(ps["ou"]["p"] - 0.53) < 1e-12
    assert ps["ou"]["tier"] is None and ps["ou"]["source"] == "模型" and ps["score"] is None
    assert pk.match_picks_v3(row_of(fused=False, ou=None, model=False, main=None, model_over=0.47), S)["ou"] is None


def test_tiers_and_ah_has_none():
    ps = pk.match_picks_v3(row_of(), S)
    assert ps["1x2"]["tier"] == "high"                             # 0.62 ≥ 0.574
    assert ps["ah"] is not None and ps["ah"]["tier"] is None        # 让球不分档
    assert ps["ah"] == {**pk.pick_ah(row_of()), "tier": None}       # 方向和 picks_v2 一样
    assert pk.tier_codes(ps) == {"1x2": "high", "ou": ps["ou"]["tier"], "score": ps["score"]["tier"]}
    assert ps["score"]["scores"] == pk.best_scores(pk.market_matrix_v3(row_of(), S)[0])["scores"]


def test_model_column_same_line_no_tier_overstated():
    r = row_of(main=3.0)
    ps = pk.match_picks_v3(r, S)
    mine = pk.model_picks_v3(r, S, ps["ou"]["line"])
    assert mine["ou"]["line"] == 3.0 and all(mine[k]["tier"] is None for k in pk.MARKETS)
    assert mine["ah"]["overstated"] and mine["ou"]["overstated"]
    assert "overstated" not in mine["1x2"] and "overstated" not in mine["score"]
    v2 = pk.model_picks(r, S)
    for k in ("1x2", "ah", "score"):                               # 这三个和 picks_v2 的“模型自己”一样
        assert {x: y for x, y in mine[k].items() if x not in ("tier", "overstated")} == v2[k]
    m = np.asarray(fake_matrix(1.2, 1.4))
    over = pk._over_share(m / m.sum(), 3.0)
    assert abs(mine["ou"]["p"] - (over if over >= 0.5 else 1 - over)) < 1e-9
    # 没有模型矩阵：线和存档主盘口一样才用“模型大球胜算”
    r = row_of(model=False, main=2.75, model_over=0.58)
    assert pk.model_picks_v3(r, S, 2.75)["ou"]["p"] == 0.58
    assert pk.model_picks_v3(r, S, 2.5)["ou"] is None and pk.model_picks_v3(r, S, None)["ou"] is None


# ---------- picks_v2 不变、存档不改 ----------
def test_picks_v2_unchanged_and_rows_not_modified():
    r = row_of(fused=False)
    before = copy.deepcopy(r)
    v2 = pk.match_picks(r, S)
    pk.match_picks_v3(r, S)
    pk.model_picks_v3(r, S, 2.75)
    assert r == before                                             # 只在内存里算，不写回存档
    assert pk.match_picks(r, S) == v2 and pk.RULE == "picks_v2" and pk.OU_LINE == 2.5
    # picks_v2 照旧：没存融合矩阵就不给大小球（不重算），比分用模型矩阵；也不带把握等级
    assert v2["ou"] is None and v2["score"]["source"] == "模型"
    assert all("tier" not in (q or {}) for q in v2.values())
    assert all("tier" not in (v2[k] or {}) for k in pk.MARKETS)
    assert pk.MARKET_NAMES["ou"] == "大小球 2.5" and pk.MARKET_NAMES_V3["ou"] == "大小球（主盘口）"


def test_frozen_settings_unchanged_by_picks_v3():
    """picks_v3 是新块，不在期末考试记下的设置里；冻结块（fusion、lean、confidence、final_exam）和去掉 picks_v3 时一模一样。"""
    assert "picks_v3" not in fe.FROZEN
    s2 = copy.deepcopy(S)
    del s2["picks_v3"]
    assert fe.frozen_settings(S) == fe.frozen_settings(s2)
    assert S["fusion"]["w"] == 1.0
    frozen = copy.deepcopy({k: S[k] for k in ("fusion", "lean", "confidence", "final_exam")})
    pk.match_picks_v3(row_of(fused=False), S)
    assert {k: S[k] for k in frozen} == frozen                     # 算的时候也不改设置


# ---------- 页面上的写法 ----------
def test_tier_html_wording():
    ps = pk.match_picks_v3(row_of(), S)
    t = text_of(pk.tier_html("1x2", ps["1x2"], S))
    assert t == "把握高 历史命中 69.4%（2,388 场）"
    t = text_of(pk.tier_html("ah", ps["ah"], S))
    assert "不分档" in t and "52.8%（7,228 场）" in t and "看不出哪场更有把握" in t
    assert "三档只差 3 个百分点" in text_of(pk.tier_html("ou", ps["ou"], S))
    mid = {**ps["score"], "tier": "mid"}                           # 比分中档：写出来偏低，以历史命中为准
    t = text_of(pk.tier_html("score", mid, S))
    assert "45.7%" in t and "偏低" in t and "42.3%" in t and "以历史命中为准" in t
    assert "偏" not in text_of(pk.tier_html("1x2", ps["1x2"], S))     # 准的档不写
    assert text_of(pk.tier_html("1x2", {**ps["1x2"], "tier": None}, S)) == "不是赔率算的，不分档"
    assert pk.tier_html("1x2", None, S) == "—"
    assert pk.test_seasons_text(S) == "2020/21–2023/24 共 4 个赛季"


def test_tier_line_on_index():
    ps = pk.match_picks_v3(row_of(main=2.75), S)
    line = pk.tier_line(ps, S, "利物浦", "阿森纳")
    assert line.startswith("把握：胜平负 高（历史命中 69.4%）、")
    assert pk.pick_text("ou", ps["ou"], "利物浦", "阿森纳") in line and "2.75" in line and line.endswith("、让球不分档")
    assert pk.tier_line(pk.match_picks_v3(row_of(status="无赔率，未融合"), S), S, "a", "b") is None


def page_data(**kw):
    """site_data 的第 10 场：最新一版（T-1）换成造的存档。"""
    data = site_data()
    row = data.latest(10)[1]
    new = row_of(**kw)
    row["detail"].update(new["detail"])
    row["detail"]["融合"] = new["detail"]["融合"]
    for k in ("融合比分矩阵", "模型比分矩阵"):
        if k not in new["detail"]:
            row["detail"].pop(k, None)
    return data, row


def test_match_page_table_shows_tiers():
    data, row = page_data(fused=False, main=2.75)
    t = text_of(picks_html(data, 10, row))
    assert "把握（历史命中）" in t and "大小球（主盘口）" in t and "2.75 球" in t
    assert "把握高 历史命中 69.4%（2,388 场）" in t
    assert "不分档 历史命中 52.8%" in t and "按存的赔率概率重算" in t
    assert "只写方向：模型写的胜算历史上偏高（高档写 61.1%，实际 49.9%）" in t
    assert "历史上偏向受让一方（弱队那边实际比模型低 2.8 个百分点）" in t
    assert t.count("弱队那边实际比模型低 2.8 个百分点") == 1       # 只写在让球那一格，大小球不写
    assert "（高档写 57.4%，实际 50.6%）" in t
    assert "平博同样偏低 1.4 个百分点" in t
    assert "固定 2.5 球" in t and "不说明赔率划算" in t and "推荐" not in t
    assert "多家平均" in t and "分歧" in t
    assert "没有市场价" in t                                      # row_of 没存「赔率概率」
    assert "懂球帝各家" in t and "6 个百分点" in t
    page = render_match(data, 10, TeamFiles(data.team_zh))
    assert "<script" not in page.lower() and "推荐" not in page
    # 存档没存主盘口：写“按概率推出来的线”
    data, row = page_data(main=None)
    assert "这一版没存主盘口，按概率推出来的线" in text_of(picks_html(data, 10, row))
    # 没用上赔率：不分档
    data, row = page_data(status="无赔率，未融合")
    assert "把握高" not in text_of(picks_html(data, 10, row)) and "不是赔率算的，不分档" in text_of(picks_html(data, 10, row))


def test_index_cell_and_row_attrs():
    data, row = page_data(fused=False, main=2.75)
    cell = text_of(latest_cell(data, 10))
    assert "把握：胜平负 高（历史命中 69.4%）" in cell and cell.index("赢盘") < cell.index("把握：")
    attrs = row_attrs(data, data.matches[10])
    ps = pk.match_picks_v3(row, data.settings)
    assert "data-tier-1x2='high'" in attrs
    assert f"data-tier-ou='{ps['ou']['tier']}'" in attrs and f"data-tier-score='{ps['score']['tier']}'" in attrs
    assert set(re.findall(r"data-tier-[a-z0-9]+='([^']*)'", attrs)) <= set(pk.TIERS)
    data, row = page_data(status="无赔率，未融合")
    assert "data-tier-" not in row_attrs(data, data.matches[10])
    assert "把握：这一版不是赔率算的，不分档" in text_of(latest_cell(data, 10))
    guide = text_of(index_guide(S))
    assert "把握 高 / 中 / 低" in guide and "2020/21–2023/24" in guide and "7,228 场" in guide and "推荐" not in guide


def test_whole_site_pages(env):
    """整站生成一次：单场页有把握一列，首页行上的 data-tier-* 只有 high / mid / low，全站没有“推荐”、静态页没有脚本。"""
    write_review(env[2])
    out, _ = build(env)
    page = read(out, "match/1.html")
    assert "把握（历史命中）" in page and "大小球（主盘口）" in page
    assert "多家平均" in page and "分歧" in page                  # 2026-11-08 单场页新两列
    idx = read(out, "index.html")
    assert set(re.findall(r"data-tier-[a-z0-9]+='([^']*)'", idx)) <= set(pk.TIERS)
    for f in all_pages(out):
        h = f.read_text(encoding="utf-8")
        assert "推荐" not in h, f.name
        assert "<script" not in h.lower(), f.name
    record = read(out, "record.html")
    body = record.split("</style>", 1)[1]                              # 去掉共用的样式表（里面的注释写着“把握等级”）
    assert "大小球 2.5" in body and "把握" not in body                   # 战绩页照旧按 picks_v2


# ---------- /api：原有字段不变，只多了 picks_v3、latest.tiers ----------
def test_api_adds_fields_only(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        status, m = get_json(srv, "/api/match/1", h)
        assert status == 200
        assert set(m["picks"]) == {"pick", "model"}                               # 原有的 picks（picks_v2）照旧
        assert m["picks"]["pick"]["ou"] is None or m["picks"]["pick"]["ou"]["line"] == 2.5
        assert all("tier" not in (v or {}) for v in m["picks"]["pick"].values())
        v3 = m["picks_v3"]
        assert v3["rule"] == "picks_v3" and v3["seasons"] == [2020, 2023]
        assert set(v3["pick"]) == set(v3["model"]) == {"1x2", "ah", "ou", "score"}
        for k, q in v3["pick"].items():
            if q is not None:
                assert q["tier"] in (*pk.TIERS, None), k
                assert {"text", "p", "source", "outcome", "tier", "tier_rate", "tier_n"} <= set(q), k
        if v3["pick"]["ah"] is not None:
            assert v3["pick"]["ah"]["tier"] is None and v3["pick"]["ah"]["tier_rate"] == 0.5285
        assert all(q is None or q["tier"] is None for q in v3["model"].values())
        assert set(m["latest"]["tiers"]) == {"1x2", "ou", "score"}
        status, body = get_json(srv, "/api/matches", h)
        for x in body["upcoming"] + body["recent"]:
            if x["latest"] is not None:
                assert {"kind", "probs", "source", "confidence", "leans", "tiers"} <= set(x["latest"])
        assert "推荐" not in json.dumps(body, ensure_ascii=False) + json.dumps(m, ensure_ascii=False)


# ---------- 首页筛选（site.js 的 rowMatches，有 node 才跑） ----------
@pytest.mark.skipif(NODE is None, reason="这台电脑上没有 node（不是项目依赖），跳过脚本逻辑的检查")
def test_js_tier_filter():
    got = run_node(
        "const r = (t) => ({league:'E0', pred:'1', level:'3', text:'x', tiers: t});"
        "const hi1 = r({'1x2':'high', ou:'', score:'low'}), hiS = r({'1x2':'mid', ou:'low', score:'high'}),"
        "      hiO = r({'1x2':'low', ou:'high', score:'mid'}), none = r({'1x2':'', ou:'', score:''});"
        "const f = (tier) => [hi1, hiS, hiO, none].map(x => S.rowMatches(x, {tier: tier}));"
        "console.log(JSON.stringify({all: f(''), any: f('any'), x1: f('1x2'), ou: f('ou'), sc: f('score'), none: f('none'),"
        "  old: S.rowMatches({league:'E0', pred:'1', level:'3', text:'x'}, {tier:'any'})}));")
    assert got["all"] == [True, True, True, True]
    assert got["any"] == [True, True, False, False]           # 只看高把握：胜平负或比分高；大小球高不算
    assert got["x1"] == [True, False, False, False] and got["sc"] == [False, True, False, False]
    assert got["ou"] == [False, False, True, False] and got["none"] == [False, False, False, True]
    assert got["old"] is False                                 # 行上没有 data-tier-*：不算高把握


def test_js_index_has_tier_select_and_reset():
    """首页工具栏多了“把握”下拉框；“全部显示”把它也清掉（字段顺序：联赛、日期、预测、信心、把握、排序、搜索）。"""
    assert '"index-tier"' in JS and "只看高把握" in JS
    assert 'tier: "", sort: "time"' in JS
    assert 'fields[4].input.value = ""; fields[5].input.value = "time"; fields[6].input.value = "";' in JS
    assert 'r.getAttribute("data-tier-1x2")' in JS and "innerHTML" not in JS and "推荐" not in JS


# ---------- 单场页「模型自己」vs「多家平均」分歧（需求文档第 1 节、第 12 节第 1 步） ----------
def disagree_row(*, p=(0.50, 0.30, 0.20), market=(0.44, 0.30, 0.26), fusion_p=(0.99, 0.005, 0.005),
                 ah_line=-0.5, ah_odds=(1.80, 2.00), model_ah=None,
                 ou_line=2.5, ou_over=0.55, model_ou=None, model=False, extra=None):
    """造一版专门查分歧的存档。融合概率故意和市场、模型都不一样，用来证明市场一侧不用融合概率。"""
    r = row_of(fused=False, main=ou_line, ou=(ou_line, ou_over), model=model, fp=fusion_p)
    r["p_home"], r["p_draw"], r["p_away"] = (float(x) for x in p)
    d = r["detail"]
    d["赔率概率"] = [float(x) for x in market]
    d["融合概率"] = [float(x) for x in fusion_p]
    d["亚盘主盘口"] = ah_line
    d["融合"]["亚盘"] = {"line": ah_line, "home": ah_odds[0], "away": ah_odds[1], "n_books": 3}
    if model_ah is not None:
        d["模型亚盘主队胜算"] = model_ah
    if model_ou is not None:
        d["模型大球胜算"] = model_ou
    if extra:
        d.update(extra)
    return r


def test_disagree_gap_matches_study_settings():
    """门槛和检验二最高一组、信心等级「分歧大」、分歧研究「强分歧」是同一个 6%。site 不在期末考试冻结块里。"""
    assert S["site"]["disagree_gap"] == 0.06
    assert S["site"]["disagree_gap"] == S["confidence_study"]["gap"] == S["disagree_study"]["disagree_strong"]
    assert pk.disagree_gap(S) == 0.06
    assert "site" not in fe.FROZEN


def test_match_disagree_1x2_uses_max_of_three():
    """胜平负：三项里差得最多的那一项 ≥ 6% 才标分歧；正好 6% 算过门槛。"""
    r = disagree_row(p=(0.50, 0.30, 0.20), market=(0.44, 0.30, 0.26))   # 主胜差正好 6%
    got = pk.match_disagree(r, S)["1x2"]
    assert got["no_market"] is False and got["no_model"] is False
    assert got["disagree"] is True and got["index"] == 0
    assert got["diff"] == pytest.approx(0.06)
    assert got["market"] == [0.44, 0.30, 0.26]
    just_under = disagree_row(p=(0.50, 0.30, 0.20), market=(0.441, 0.30, 0.259))
    assert pk.match_disagree(just_under, S)["1x2"]["disagree"] is False


def test_match_disagree_does_not_use_fusion_probs():
    """市场一侧必须用「赔率概率」。融合概率故意写成和模型一样（差 0），若误用融合概率就不会标分歧。"""
    r = disagree_row(p=(0.50, 0.30, 0.20), market=(0.40, 0.30, 0.30), fusion_p=(0.50, 0.30, 0.20))
    assert pk.market_avg_1x2(r) == [0.40, 0.30, 0.30]
    got = pk.match_disagree(r, S)["1x2"]
    assert got["disagree"] is True and got["market"] == [0.40, 0.30, 0.30]
    assert got["diff"] == pytest.approx(0.10)


def test_match_disagree_missing_market_and_html():
    """没有多家平均：这一格写「没有市场价」，分歧格写「—」，不标分歧。赔率里有 0 也当没有。"""
    r = row_of()                                                   # 没存「赔率概率」
    got = pk.match_disagree(r, S)["1x2"]
    assert got["no_market"] is True and got["disagree"] is False
    assert "没有市场价" in pk.avg_html("1x2", got, "主", "客")
    assert pk.disagree_html("1x2", got) == "—"
    zero = disagree_row(market=(0.5, 0.5, 0.0))
    assert pk.market_avg_1x2(zero) is None
    assert pk.match_disagree(zero, S)["1x2"]["no_market"] is True


def test_match_disagree_ah_and_ou_on_main_line():
    """亚盘、大小球：主盘口上模型胜算和多家平均去水后胜算差 ≥ 6%；线和主盘口对不上当没有市场价。

    过门槛那一侧用 7 个百分点，不卡正好 6%：去水后再 float 一圈，share+0.06 可能略低于门槛。
    """
    from src.common.odds import devig

    share = float(devig([1.80, 2.00])[0])
    over = disagree_row(model=False, model_ah=share + 0.07, ah_odds=(1.80, 2.00))
    got = pk.match_disagree(over, S)["ah"]
    assert got["no_market"] is False and got["disagree"] is True
    assert got["line"] == -0.5 and abs(got["diff"]) >= 0.06
    under = disagree_row(model=False, model_ah=share + 0.05, ah_odds=(1.80, 2.00))
    assert pk.match_disagree(under, S)["ah"]["disagree"] is False
    mismatch = disagree_row(model=False, model_ah=0.70, extra={"亚盘主盘口": -0.25})
    assert pk.match_disagree(mismatch, S)["ah"]["no_market"] is True   # 融合.亚盘还是 -0.5

    ou_over = disagree_row(model=False, ou_line=2.5, ou_over=0.50, model_ou=0.57)
    got = pk.match_disagree(ou_over, S)["ou"]
    assert got["disagree"] is True and got["line"] == 2.5 and abs(got["diff"]) >= 0.06
    ou_under = disagree_row(model=False, ou_line=2.5, ou_over=0.50, model_ou=0.55)
    assert pk.match_disagree(ou_under, S)["ou"]["disagree"] is False
    ou_miss = disagree_row(model=False, ou_line=2.75, ou_over=0.50, model_ou=0.70)
    # row_of 的 大小球主盘口是 2.75，融合.大小球也被写成 2.75；改成线和主盘口故意错开
    ou_miss["detail"]["融合"]["大小球"] = {"line": 2.5, "p_over": 0.50, "n_books": 3}
    assert pk.match_disagree(ou_miss, S)["ou"]["no_market"] is True


def test_match_disagree_score_top_or_gap():
    """比分：两边最可能的那个比分不一样，或同一个比分上概率差 ≥ 6%，都标分歧。"""
    p, line, p_over = (0.45, 0.28, 0.27), 2.5, 0.52
    km = np.array(pk._rebuild(p, line, p_over, int(S["market_odds"]["max_goals"])))
    n = int(S["market_odds"]["max_goals"]) + 1
    assert km.shape == (n, n)

    # 1. 最可能的比分不一样：把市场矩阵里最大的两格对调，当模型矩阵
    swapped = km.copy()
    order = np.argsort(-swapped, axis=None, kind="stable")
    a0, a1 = np.unravel_index(int(order[0]), swapped.shape), np.unravel_index(int(order[1]), swapped.shape)
    swapped[a0], swapped[a1] = swapped[a1], swapped[a0]
    r = disagree_row(p=(0.40, 0.30, 0.30), market=p, ou_line=line, ou_over=p_over, model=False)
    r["detail"]["模型比分矩阵"] = np.round(swapped, 6).tolist()
    got = pk.match_disagree(r, S)["score"]
    assert got["no_market"] is False and got["same_top"] is False and got["disagree"] is True

    # 2. 同一个比分，模型那一格比市场高 7 个百分点
    boosted = km.copy()
    h, a, kp = pk._top_score(km)
    rest = boosted.copy()
    rest[h, a] = 0.0
    s = float(rest.sum())
    boosted = rest * ((s - 0.07) / s)
    boosted[h, a] = kp + 0.07
    r["detail"]["模型比分矩阵"] = np.round(boosted, 6).tolist()
    got = pk.match_disagree(r, S)["score"]
    assert got["same_top"] is True and got["disagree"] is True
    assert abs(got["diff"]) >= 0.06


def test_match_page_flags_disagree_on_1x2():
    """单场页：胜平负差 ≥ 6% 时「分歧」那一格带标签；市场格不写「没有市场价」。"""
    data, row = page_data(fused=False, main=2.75)
    row["p_home"], row["p_draw"], row["p_away"] = 0.50, 0.30, 0.20
    row["detail"]["赔率概率"] = [0.40, 0.30, 0.30]
    row["detail"]["融合概率"] = [0.50, 0.30, 0.20]                 # 若误用融合概率，差是 0，不会标分歧
    html = picks_html(data, 10, row)
    assert "tag disagree" in html and "主胜 模型" in html
    assert "没有市场价" in html                                     # 亚盘 / 大小球 / 比分这版没存市场价，照样写
    assert "推荐" not in html
