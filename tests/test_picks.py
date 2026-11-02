"""四个玩法的预测和命中率 picks_v2（src/web/site/picks.py、战绩页、单场页）的自动检查。全是造的数据。"""
import json

import numpy as np
import pytest

from src.common.config import load_settings
from src.web.site import picks as pk
from tests.test_site import build, env, fake_matrix, read  # noqa: F401  （env 是 pytest 的 fixture）

TOL = load_settings()["site"]["matrix_sum_tol"]


def test_new_row_uses_stored_leans_and_fused_matrix():
    m = np.zeros((11, 11))
    m[2, 0], m[1, 1] = 0.3, 0.7
    row = {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3,
           "detail": {"融合概率": [0.2, 0.3, 0.5], "融合比分矩阵": m.tolist(), "模型比分矩阵": fake_matrix(),
                      "融合": {"状态": "已融合", "倾向": {"result": {"side": 2, "p": 0.5, "lean": False},
                                       "ah": {"side": "away", "line": 0.25, "p": 0.56, "lean": True},
                                       "over25": {"side": "under", "p": 0.6, "lean": True}}}}}
    p = pk.match_picks(row, {"site": {"matrix_sum_tol": TOL}})
    assert p["1x2"] == {"side": 2, "p": 0.5, "source": "融合", "lean": False}      # 用融合概率，不用模型的
    assert p["ah"] == {"side": "away", "line": 0.25, "p": 0.56, "source": "融合", "lean": True}
    assert p["ou"]["side"] == "under" and p["ou"]["p"] == pytest.approx(0.6) and p["ou"]["line"] == 2.5
    assert p["score"]["source"] == "融合" and p["score"]["scores"][:2] == [(1, 1), (2, 0)]   # 先用融合后的矩阵，大的在前
    assert len(p["score"]["scores"]) == pk.SCORE_TOP and p["score"]["p"] == pytest.approx(1.0)  # 4 格合起来
    assert pk.pick_text("score", p["score"], "利物浦", "阿森纳").startswith("1–1、2–0、")
    assert "4 个合起来" in pk.prob_html("score", p["score"])
    assert "和第二名差得不多" in pk.prob_html("1x2", p["1x2"])                      # 没到倾向门槛，淡字提醒
    assert pk.pick_text("ah", p["ah"], "利物浦", "阿森纳") == "阿森纳 -0.25"          # 换成客队自己的盘口


def test_old_row_falls_back_to_model_numbers():
    row = {"p_home": 0.3, "p_draw": 0.3, "p_away": 0.4,
           "detail": {"亚盘主盘口": -0.5, "模型亚盘主队胜算": 0.45, "大小球主盘口": 2.5, "模型大球胜算": 0.52}}
    p = pk.match_picks(row, {"site": {"matrix_sum_tol": TOL}})
    assert p["1x2"] == {"side": 2, "p": 0.4, "source": "模型", "lean": None}
    assert p["ah"] == {"side": "away", "line": -0.5, "p": pytest.approx(0.55), "source": "模型", "lean": None}
    assert p["ou"]["side"] == "over" and p["ou"]["source"] == "模型"
    assert p["score"] is None                                                     # 没存矩阵不预测比分
    assert pk.pick_text("ah", p["ah"], "利物浦", "阿森纳") == "阿森纳 +0.5"
    # 大小球主盘口不是 2.5，模型的胜算是另一条线上的，不能当 2.5 球用
    row["detail"]["大小球主盘口"] = 2.75
    assert pk.pick_ou(row) is None
    # 概率缺一项（或写成文字）就不预测胜平负
    assert pk.pick_1x2({"p_home": 0.5, "p_draw": "x", "p_away": 0.2, "detail": {}}) is None


def test_settle_quarter_and_integer_lines_and_hit_rate():
    home_q = {"side": "home", "line": -0.25}
    assert pk.settle("ah", home_q, (1, 1)) == "half_loss" and pk.settle("ah", home_q, (2, 1)) == "win"
    assert pk.settle("ah", {"side": "away", "line": -0.75}, (2, 1)) == "half_loss"
    assert pk.settle("ah", {"side": "home", "line": -1.0}, (2, 1)) == "push"
    assert pk.settle("ou", {"side": "over", "line": 2.5}, (2, 1)) == "win"
    four = {"scores": [(1, 1), (1, 0), (2, 1), (0, 0)]}
    assert pk.settle("score", four, (2, 1)) == "win" and pk.settle("score", four, (0, 0)) == "win"   # 任意一个对上就中
    assert pk.settle("score", four, (0, 1)) == "loss" and pk.settle("score", four, (1, 2)) == "loss"
    assert pk.settle("1x2", {"side": 1}, (0, 0)) == "win" and pk.settle("1x2", None, (0, 0)) is None
    # 赢 1 + 赢半 0.5 = 1.5；没中 1 + 输半 0.5 = 1.5；走盘不算 → 50%
    assert pk.hit_rate(["win", "half_win", "push", "loss", "half_loss"]) == pytest.approx(0.5)
    assert pk.hit_rate(["push"]) is None and pk.hit_rate([]) is None
    t = pk.tally(["win", "push", "loss", "loss"])
    assert t["n"] == 4 and t["push"] == 1 and t["rate"] == pytest.approx(1 / 3)
    assert pk.counts_text(t) == "中 1、走盘 1、没中 2"


def test_reference_missing_bad_and_good(tmp_path):
    s = {"picks": {"reference_dir": str(tmp_path)}}
    good = {"rule": pk.RULE, "markets": {k: {"n": 10, "rate": 0.5} for k in pk.MARKETS}, "always_home": {"rate": 0.43},
            "seasons": [2024, 2025], "n_matches": 10}
    # picks_v1 的历史参考放在文件夹本身：新规则不读它（比分那一项的意思不一样了）
    (tmp_path / "summary.json").write_text(json.dumps({**good, "rule": "picks_v1"}), encoding="utf-8")
    assert pk.load_reference(s) == (None, "还没算历史参考")
    tmp_path = tmp_path / pk.RULE
    tmp_path.mkdir()
    (tmp_path / "summary.json").write_text(json.dumps({"rule": "picks_v0"}), encoding="utf-8")
    assert pk.load_reference(s) == (None, "历史参考文件格式不对，没有用")
    (tmp_path / "summary.json").write_text(json.dumps({**good, "rule": "picks_v1"}), encoding="utf-8")
    assert pk.load_reference(s) == (None, "历史参考文件格式不对，没有用")     # 规则名对不上不用
    (tmp_path / "summary.json").write_text(json.dumps({**good, "seasons": "2024"}), encoding="utf-8")
    assert pk.load_reference(s)[0] is None                                       # 赛季写错也不用（页面标题要用）
    (tmp_path / "summary.json").write_text(json.dumps(good), encoding="utf-8")
    assert pk.load_reference(s)[0]["always_home"]["rate"] == 0.43


def test_record_and_match_page_show_four_picks_without_scores_csv(env):
    out, _ = build(env)                                      # 没写 data/review/，没有 scores.csv
    record = read(out, "record.html")
    assert "四个玩法的命中率" in record and "每场的四个预测（最近 1 场）" in record
    assert "历史参考：还没算历史参考。" in record
    # 比赛 1：最新一版 T-1，主胜（中）、利物浦 -0.5（2–1 中）、大 2.5 球（中）、比分 4 个（以 1–1 打头）
    assert "主胜（利物浦）" in record and "利物浦 -0.5" in record and "1–1、" in record and "比分 4 个" in record
    pages = [p.read_text(encoding="utf-8") for p in (out / "match").glob("*.html")]
    assert any("四个玩法的预测" in p and "还没踢" in p for p in pages)          # 比赛 10 还没踢
    assert all("推荐" not in p for p in pages + [record])


class FakeData:
    """finished() / before_kickoff() 用到的几样：一场 2–1 踢完的比赛和它的几版存档。"""
    settings = {"site": {"matrix_sum_tol": TOL}}
    results = {1: (2, 1)}
    matches = {1: {"kickoff_utc": "2026-11-04 15:00"}}

    def __init__(self, rows):
        self.preds = {1: {i: r for i, r in enumerate(rows)}}

    def latest(self, mid):
        rows = list(self.preds[mid].values())
        r = max(rows, key=lambda r: (r["cutoff_utc"], r["prediction_id"]))
        return r["kind"], r


def version(pid, cutoff, p):
    return {"prediction_id": pid, "kind": "T-1", "cutoff_utc": cutoff, "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "detail": {}}


def test_uses_last_version_before_kickoff_and_skips_only_late_ones():
    # 开球后多出了一版：不用它，用开球以前的最新一版（客胜那版 → 胜平负没中）
    items, skipped = pk.finished(FakeData([version(1, "2026-11-04 14:00:00", (0.2, 0.3, 0.5)),
                                           version(2, "2026-11-04 15:30:00", (0.6, 0.2, 0.2))]))
    assert skipped == {} and items[0]["row"]["prediction_id"] == 1 and items[0]["outcomes"]["1x2"] == "loss"
    # 只有开球以后的版本：整场不算，记场数
    items, skipped = pk.finished(FakeData([version(2, "2026-11-04 15:00:00", (0.6, 0.2, 0.2))]))
    assert items == [] and skipped == {"只有开球以后的版本": 1}


def test_no_odds_version_is_counted_as_model_only():
    # 没有赔率时存档的“融合概率”就是模型自己的概率（src/predict/conclusion.py），不能当成赔率
    from src.predict import conclusion
    assert set(pk.ODDS_USED) == {conclusion.FUSED, conclusion.PARTIAL}
    row = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
           "detail": {"融合概率": [0.5, 0.3, 0.2], "融合比分矩阵": fake_matrix(),
                      "融合": {"状态": conclusion.NO_ODDS, "w": 1.0, "倾向": {"result": {"side": 0, "p": 0.5, "lean": True}}}}}
    p = pk.match_picks(row, {"site": {"matrix_sum_tol": TOL}})
    assert {p[k]["source"] for k in ("1x2", "score")} == {"模型"} and not pk.used_odds(row)
    # 新存档没写亚盘、大小球倾向：不退回去用模型在主盘口上的胜算（和历史参考一样不预测）
    row["detail"].update({"亚盘主盘口": -0.5, "模型亚盘主队胜算": 0.6, "大小球主盘口": 2.5, "模型大球胜算": 0.6})
    assert p["ah"] is None and pk.pick_ah(row) is None and pk.pick_ou(row) is None
    items = [{"row": row, "picks": p, "score": (1, 0), "outcomes": {k: pk.settle(k, p[k], (1, 0)) for k in pk.MARKETS}}]
    assert pk.summarize(items)["model_only"] == 1


def test_tie_goes_to_home_then_draw():
    assert pk.pick_1x2({"p_home": 0.35, "p_draw": 0.35, "p_away": 0.3, "detail": {}})["side"] == 0
    assert pk.pick_1x2({"p_home": 0.3, "p_draw": 0.35, "p_away": 0.35, "detail": {}})["side"] == 1


def test_reference_rule_matches_site_rule():
    """历史参考（picks_reference.picks_for）是从矩阵另算的一份，和网站从存档里取的要一样。"""
    from src.evaluate.picks_reference import picks_for
    from src.predict import conclusion
    s = load_settings()
    for xg in ((1.6, 0.9), (0.8, 1.7), (1.2, 1.2)):
        m = np.asarray(fake_matrix(*xg))
        m = m / m.sum()
        i, j = np.indices(m.shape)
        f = {"p_1x2": [m[i > j].sum(), m[i == j].sum(), m[i < j].sum()], "matrix": m,
             "over25": float(m[i + j > 2].sum()), "btts": float(m[1:, 1:].sum())}
        row = {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3,
               "detail": {"融合概率": f["p_1x2"], "融合比分矩阵": m.tolist(),
                          "融合": {"状态": conclusion.FUSED, "倾向": conclusion.leans(f, -0.25, s)}}}
        site, ref = pk.match_picks(row, s), picks_for(m, -0.25)
        for k in pk.MARKETS:
            for key in ref[k]:
                if key == "scores":                          # 比分是一串 (主, 客)，要一模一样（顺序也一样）
                    assert site[k][key] == ref[k][key] and len(ref[k][key]) == pk.SCORE_TOP, (xg, k)
                else:
                    assert site[k][key] == pytest.approx(ref[k][key]), (xg, k, key)


def test_best_scores_order_and_ties():
    m = np.zeros((11, 11))
    m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[2, 1], m[3, 3] = 0.1, 0.2, 0.2, 0.3, 0.15, 0.05
    b = pk.best_scores(m)
    assert b["scores"] == [(1, 1), (0, 1), (1, 0), (2, 1)]          # 0–1 和 1–0 一样大：主队进球少的在前
    assert b["p"] == pytest.approx(0.85) and (3, 3) not in b["scores"] and (0, 0) not in b["scores"]


def test_why_points_say_where_the_probability_comes_from():
    from src.web.site.story import why_points

    class D:
        settings = load_settings()

        @staticmethod
        def zh(t):
            return {"H": "主队<b>", "A": "客队"}[t]

    base = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
    no_odds = why_points(D, {**base, "detail": {"融合概率": [0.5, 0.3, 0.2], "融合": {"状态": "无赔率，未融合", "w": 1.0}}},
                         "H", "A")
    assert "只是模型自己算的（无赔率，未融合）" in no_odds[0] and len(no_odds) == 1 and "主队&lt;b&gt;" in no_odds[0]
    pin = why_points(D, {**base, "detail": {"融合概率": [0.45, 0.3, 0.25],
                                            "融合": {"状态": "已融合", "w": 1.0, "胜平负用平博": True, "胜平负几家": 1}}},
                     "H", "A")
    assert "从平博的赔率（去掉抽水）算出来的" in pin[0] and "和赔率差不多" in pin[1]
    old = why_points(D, {**base, "detail": {}}, "H", "A")
    assert "那时还没接赔率融合" in old[0]


def test_match_page_says_why_a_version_is_not_counted():
    from src.web.site.match import picks_html
    from tests.test_explanation_pages import site_data

    data = site_data()
    row = data.latest(10)[1]
    data.results[10] = (1, 0)
    assert "不算进战绩页" not in picks_html(data, 10, row)                       # 正常：开球前出的
    data.matches[10]["kickoff_utc"] = "2000-01-01 00:00:00"
    assert "这一版是开球以后才出的" in picks_html(data, 10, row)
    data.matches[10]["kickoff_utc"] = None
    for r in data.preds[10].values():
        r["kickoff_utc"] = None
    assert "开球时间不明" in picks_html(data, 10, row)
