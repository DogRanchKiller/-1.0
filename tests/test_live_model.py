"""上线模型换成 gb_xg（需求文档第 4 节“换成 gb_xg 上线”，2026-11-07）：几处设置要互相对得上。

- 预测页（定时预测、网站新出的预测）、四本账用同一个模型：gb_xg；
- gb_l7 反过来当影子模型并排出 T-24，影子模型不能和正式模型一样（一样就会同一个模型存两遍）；
- gb_xg 和 gb_l7 只差 dyn 用不用 xG 更新，树的设置一样；联赛编号含 asia 以后指纹换成 gb_l7-1c589ce8 / gb_xg-f1ceca4f（2026-11-07 以前的存档仍是旧指纹）；
- 融合、倾向、信心等级没重选：fusion.model 还记着 w 当初是用 gb_l7 选的。
"""
from src.common.config import load_settings
from src.models.registry import model_config
from src.predict.preview import model_version

S = load_settings()


def test_live_model_is_gb_xg_and_ledger_follows():
    assert S["preview"]["model"] == "gb_xg"
    assert S["ledger"]["model"] == S["preview"]["model"]
    assert S["preview"]["model_info"]["label"].endswith("gb_xg")


def test_gb_l7_runs_as_shadow_not_the_live_model():
    assert S["shadow"]["enabled"] is True
    assert list(S["shadow"]["models"]) == ["gb_l7"]
    assert S["preview"]["model"] not in S["shadow"]["models"]


def test_gb_xg_is_gb_l7_plus_xg_and_fingerprints():
    """gb_xg 只比 gb_l7 多 xG；联赛编号含 asia 以后指纹是新的（2026-11-08 六联赛接入）。"""
    base, x = model_config("gb_l7", S), model_config("gb_xg", S)
    assert {k: v for k, v in x.items() if k not in ("xg_mix", "xg_source")} == base
    assert "JPN" in base["leagues"] and "CHN" in base["leagues"] and "KOR" in base["leagues"]
    assert model_version("gb_l7", base) == "gb_l7-1c589ce8"
    assert model_version("gb_xg", x) == "gb_xg-f1ceca4f"


def test_fusion_lean_confidence_not_reselected():
    # w、倾向门槛、信心等级都是用 gb_l7 选的，这次不重选（第 1 节冻结）；网站会照实写“这个比例是用 gb_l7 选的”
    assert S["fusion"]["model"] == "gb_l7" and S["fusion"]["w"] == 1.0
    assert S["confidence_study"]["model"] == "gb_l7" and S["fusion_study"]["model"] == "gb_l7"
