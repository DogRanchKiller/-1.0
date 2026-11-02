"""逐场解释的共享网页展示：只检查已经给出的数字，不取数、不跑模型、不补旧档。

原始 SHAP 是对数进球贡献；球数、主胜概率是预测时沿路径换算并保存的贡献。
因此页面分别标明单位，并独立检查加和，不能把载荷里的“检查通过”直接当作证据。
"""
from __future__ import annotations

import hashlib
import html
import math
import re


SCHEMA = "gbdt-tree-shap-v1"
IDENTITY_KEYS = ("match_id", "model", "model_version", "competition", "season", "home_team", "away_team", "cutoff_utc")
CHECK_KEYS = ("raw_max_abs_error", "goal_max_abs_error", "home_win_abs_error",
              "integration_max_abs_error", "matrix_max_abs_error")
INVALID = "模型解释数据校验未通过，已隔离解释；原预测照常显示。"
IDENTITY_ERROR = "模型解释身份与这一版预测不一致，已隔离解释；原预测照常显示。"
MISSING = "这一版没有保存模型解释"
REASONS = {
    "disabled": "这一版未启用模型解释",
    "unsupported_model": "这一版模型暂不支持逐场解释",
    "unsupported_target": "这一版预测目标暂不支持逐场解释",
    "no_prediction": "这一版没有模型预测，无法提供解释",
    "explanation_failed": "这一版模型解释生成失败，原预测照常显示",
    "validation_failed": "这一版模型解释校验未通过，原预测照常显示",
}

# 样式只由页面入口引入一次；同页每版的 radio 使用独立名字，切换互不影响。
EXPLANATION_CSS = """
.model-explanation { min-width: 0; }
.model-explanation p { margin: 8px 0 12px; }
.model-explanation .ex-note { color: var(--sub); font-size: 13px; }
.model-explanation .ex-fusion { border-left: 3px solid var(--amber); padding: 6px 12px; }
.explanation-switch { border: 0; padding: 0; margin: 14px 0; min-width: 0; }
.explanation-switch > legend { color: var(--sub); font-size: 13px; padding: 0 0 6px; }
.explanation-switch > input { position: absolute; opacity: 0; width: 1px; height: 1px; }
.explanation-switch > label { display: inline-block; cursor: pointer; border: 1px solid var(--line);
  border-radius: 7px; padding: 7px 12px; margin: 0 6px 10px 0; font-size: 13px; color: var(--sub); }
.explanation-switch > input:checked + label { color: var(--green); border-color: var(--green); background: var(--card-2); }
.explanation-switch > input:focus-visible + label { outline: 2px solid var(--amber); outline-offset: 3px; }
.explanation-switch > .ex-panel-goals { display: none; }
.explanation-switch > .ex-unit-goals:checked ~ .ex-panel-prob { display: none; }
.explanation-switch > .ex-unit-goals:checked ~ .ex-panel-goals { display: block; }
.ex-goal-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 330px), 1fr)); gap: 18px; }
.ex-panel { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-top: 4px; }
.ex-panel h3 { font-size: 15px; color: var(--text); margin: 0 0 8px; }
.ex-total { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 20px; color: var(--sub); font-size: 13px; }
.ex-total b { color: var(--text); }
.ex-factors { margin: 14px 0; }
.ex-factor { display: grid; grid-template-columns: minmax(130px, 1.5fr) minmax(80px, 1fr) max-content;
  align-items: center; gap: 10px; margin: 9px 0; font-size: 13px; }
.ex-factor-name { overflow-wrap: anywhere; }
.ex-track { position: relative; height: 12px; background: var(--card-2); border-radius: 3px; }
.ex-track::before { content: ''; position: absolute; left: 50%; height: 100%; width: 1px; background: var(--sub); }
.ex-bar { position: absolute; height: 100%; border-radius: 2px; background: var(--green); }
.ex-bar.negative { background: var(--amber); }
.ex-number { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.ex-all { margin-top: 14px; }
.ex-all > summary, .ex-preview > summary { cursor: pointer; color: var(--green); padding: 10px 0; }
.ex-all .scroll { overflow-x: auto; margin-top: 10px; border: 1px solid var(--line); border-radius: 8px; }
.ex-all table { min-width: 760px; }
.ex-all code { display: block; color: var(--sub); font-size: 11px; overflow-wrap: anywhere; }
.ex-audit { margin-top: 12px; overflow-wrap: anywhere; }
.ex-preview { padding: 0 16px; border: 1px solid var(--line); border-radius: 10px; margin-top: 10px; background: var(--card); }
.ex-preview[open] { padding-bottom: 16px; }
.ex-missing { color: var(--sub); border: 1px dashed var(--line); padding: 14px; border-radius: 8px; }
@media (max-width: 600px) {
  .ex-factor { grid-template-columns: minmax(0, 1fr) max-content; gap: 4px 10px; }
  .ex-track { grid-column: 1 / -1; grid-row: 2; }
  .ex-number { grid-column: 2; grid-row: 1; }
  .ex-panel { padding: 10px; }
}
"""


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _finite(value) -> bool:
    """JSON 数字必须有限，True/False 不能冒充 1/0。"""
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _json_value(value) -> bool:
    """包括未展示的额外字段在内，也不能藏 NaN、Infinity 或非 JSON 对象。"""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return _finite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_value(item) for key, item in value.items())
    return False


def _same_identity(identity, row: dict) -> bool:
    if not isinstance(identity, dict):
        return False
    for key in IDENTITY_KEYS:
        left, right = identity.get(key), row.get(key)
        if key in ("match_id", "season"):
            if not isinstance(left, int) or isinstance(left, bool) or not isinstance(right, int) or isinstance(right, bool):
                return False
        elif not isinstance(left, str) or not left or not isinstance(right, str) or not right:
            return False
        if left != right:
            return False
    return True


def _close(left: float, right: float, atol: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= atol


def _check_available(payload: dict, row: dict, atol: float) -> bool:
    """核对结构、每一项和端点，独立重算加和；绝不重算模型或 SHAP。"""
    if payload.get("reason") is not None or payload.get("method") != "tree_shap_raw_log_lambda" \
            or payload.get("conversion") != "straight_log_path_v1":
        return False
    if not isinstance(payload.get("model_fingerprint"), str) or not re.fullmatch(r"[0-9a-f]{64}", payload["model_fingerprint"]):
        return False
    if not isinstance(payload.get("library_version"), str) or not payload["library_version"]:
        return False
    trained = payload.get("model_trained_at_utc")
    if trained is not None and (not isinstance(trained, str) or not trained):
        return False
    context = payload["context"]
    if not isinstance(context, dict) or not _finite(context.get("rho")) or type(context.get("max_goals")) is not int \
            or context["max_goals"] < 1:
        return False
    names, terms = payload["feature_names"], payload["terms"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) \
            or len(set(names)) != len(names) or not isinstance(terms, list) or len(terms) != 2 * len(names):
        return False
    seen = set()
    for term in terms:
        if not isinstance(term, dict) or term.get("side") not in ("home", "away") or term.get("feature") not in names:
            return False
        pair = (term["side"], term["feature"])
        if pair in seen or "value" not in term or (term["value"] is not None and not _finite(term["value"])):
            return False
        seen.add(pair)
        if any(not _finite(term.get(key)) for key in ("raw_contribution", "goal_contribution", "home_win_contribution")):
            return False
    if seen != {(side, name) for side in ("home", "away") for name in names}:
        return False
    for name in ("base", "prediction"):
        values = payload[name]
        if not isinstance(values, dict) or any(not _finite(values.get(key)) for key in ("home_goals", "away_goals", "home_win")):
            return False
        if values["home_goals"] <= 0 or values["away_goals"] <= 0 or not 0 <= values["home_win"] <= 1:
            return False
    for name in ("raw_base", "raw_prediction"):
        values = payload[name]
        if not isinstance(values, dict) or any(not _finite(values.get(side)) for side in ("home", "away")):
            return False
    for side in ("home", "away"):
        own = [term for term in terms if term["side"] == side]
        raw_base, raw_end = payload["raw_base"][side], payload["raw_prediction"][side]
        base, end = payload["base"][f"{side}_goals"], payload["prediction"][f"{side}_goals"]
        if not _close(raw_base + math.fsum(term["raw_contribution"] for term in own), raw_end, atol) \
                or not _close(base + math.fsum(term["goal_contribution"] for term in own), end, atol) \
                or not _close(math.exp(raw_base), base, atol) or not _close(math.exp(raw_end), end, atol):
            return False
    if not _close(payload["base"]["home_win"] + math.fsum(term["home_win_contribution"] for term in terms),
                  payload["prediction"]["home_win"], atol):
        return False
    # 存档 p_home 四舍五入到四位小数，不能用它要求解释保留更多精度，也不能放宽到任意差值。
    if not _finite(row.get("p_home")) or not 0 <= row["p_home"] <= 1 \
            or not _close(payload["prediction"]["home_win"], row["p_home"], 5.1e-5):
        return False
    checks = payload["checks"]
    return isinstance(checks, dict) and all(_finite(checks.get(key)) and 0 <= checks[key] <= atol for key in CHECK_KEYS)


def validate_explanation(payload, row: dict, *, atol: float = 1e-6) -> tuple[dict | None, str | None]:
    """返回（有效载荷, 不可用原因）；解释有问题时只隔离解释，原预测继续显示。"""
    if payload is None:
        return None, MISSING
    if not isinstance(payload, dict) or not _json_value(payload) or not _finite(atol) or atol <= 0:
        return None, INVALID
    if payload.get("schema") != SCHEMA:
        return None, "不支持的模型解释格式，原预测照常显示。"
    if payload.get("status") not in ("available", "unavailable"):
        return None, INVALID
    if payload["status"] == "available" or "identity" in payload:
        if not _same_identity(payload.get("identity"), row):
            return None, IDENTITY_ERROR
    if payload["status"] == "unavailable":
        reason = payload.get("reason")
        return (None, REASONS.get(reason, "这一版模型解释不可用（原因未识别）")) if isinstance(reason, str) else (None, INVALID)
    try:
        if not _check_available(payload, row, atol):
            return None, INVALID
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, INVALID
    return payload, None


def feature_label(side: str, feature: str) -> str:
    """特征里的 own/opp 是该条输出的本队/对手，客队输出时必须调换主客名称。"""
    own, opp = ("主队", "客队") if side == "home" else ("客队", "主队")
    direct = {
        "league": "联赛编号", "is_home": f"{own}主客场标记", "dyn_home": "联赛主场优势",
        "dyn_loglam_for": f"{own}动态预期进球（对数）", "dyn_loglam_against": f"{opp}动态预期进球（对数）",
        "dyn_sd_for": f"{own}动态进球估计不确定度", "dyn_sd_against": f"{opp}动态进球估计不确定度",
        "elo_diff": f"{own}相对{opp}的Elo差（含主场项）", "pi_egd": f"{own}评分对应的预期净胜球",
        "top_value_diff": f"{own}相对{opp}的主力身价差",
        "mkt_p_win": f"{own}市场胜率", "mkt_p_draw": "市场平局概率", "mkt_p_loss": f"{opp}市场胜率",
        "mkt_over25": "市场大2.5球概率", "mkt_loglam_for": f"{own}市场预期进球（对数）",
        "mkt_loglam_against": f"{opp}市场预期进球（对数）", "mkt_pinnacle": "市场是否采用平博",
    }
    if feature in direct:
        return direct[feature]
    match = re.fullmatch(r"(.+)_(own|opp)", feature)
    if match:
        stem, whose = match.groups()
        team = own if whose == "own" else opp
        words = {"elo": "Elo评分", "pi": "主客场实力评分", "promoted": "升班马标记", "rest_days": "休息天数",
                 "after_euro": "欧战后赛程标记", "table_played": "本季联赛场数", "table_ppg": "本季场均积分",
                 "table_gdpg": "本季场均净胜球", "table_rank": "联赛名次比例", "form_n": "近期状态可用场数",
                 "inj_n": "伤停人数", "inj_share": "伤停球员占比", "inj_top_loss": "主力伤停损失"}
        if stem in words:
            return team + words[stem]
        form = re.fullmatch(r"(form|ewm)(\d+)_(gf|ga|sotf|sota|pts)", stem)
        if form:
            kind, window, stat = form.groups()
            word = {"gf": "进球", "ga": "失球", "sotf": "射正", "sota": "被射正", "pts": "积分"}[stat]
            period = f"近{window}场场均" if kind == "form" else f"衰减场均（半衰期{window}场）"
            return f"{team}{period}{word}"
    return f"其他特征（{feature}）"


def _signed(value: float, *, goals: bool = False) -> str:
    number = value if goals else value * 100
    digits = 3 if goals else 2
    # 小到显示精度以下的负值显示 0，避免出现容易误读的 -0.00。
    if round(number, digits) == 0:
        number = 0.0
    return f"{number:+.{digits}f} {'球' if goals else '个百分点'}"


def _fusion_note(fusion: dict | None) -> str:
    saved = fusion if isinstance(fusion, dict) else {}
    status, weight = saved.get("状态", saved.get("status")), saved.get("w")
    if status == "无赔率，未融合":
        return "没有赔率，当前结论使用模型概率。这里的贡献只描述模型计算，不改变信心等级。"
    if status in ("已融合", "赔率不全"):
        if _finite(weight) and weight == 1:
            return "这版市场权重为100%，模型未参与该融合概率。以下解释对应模型自己的概率。"
        if _finite(weight) and 0 <= weight <= 1:
            return f"这版融合概率的市场权重为{weight * 100:g}%，模型权重为{(1 - weight) * 100:g}%。以下贡献仅解释模型部分，未换算为融合贡献。"
        return "这版已有融合概率，以下解释仅对应模型自己的概率；存档没有可核对的融合权重。"
    return "这版没有保存可核对的融合状态；以下仅展示模型自己的解释。"


def _factor_chart(terms: list[dict], *, goals: bool, top_factors: int) -> str:
    """按主胜贡献的绝对值排序，保留前三项和其余项的真实和，不把校验残差塞进去。"""
    ordered = sorted(terms, key=lambda term: abs(term["home_win_contribution"]), reverse=True)
    key = "goal_contribution" if goals else "home_win_contribution"
    shown = [(f"{feature_label(term['side'], term['feature'])}（{'主' if term['side'] == 'home' else '客'}队进球模型）", term[key])
             for term in ordered[:top_factors]]
    others = ordered[top_factors:]
    if others:
        shown.append((f"其他因素（{len(others)} 项）", math.fsum(term[key] for term in others)))
    largest = max((abs(value) for _, value in shown), default=0.0)
    parts = []
    for label, value in shown:
        width = 0.0 if largest == 0 else abs(value) / largest * 48
        left = 50.0 if value >= 0 else 50.0 - width
        parts.append(f"<div class='ex-factor'><span class='ex-factor-name'>{_escape(label)}</span>"
                     f"<span class='ex-track' aria-hidden='true'><span class='ex-bar{' negative' if value < 0 else ''}' "
                     f"style='left:{left:.3f}%;width:{width:.3f}%'></span></span>"
                     f"<span class='ex-number'>{_escape(_signed(value, goals=goals))}</span></div>")
    return "<div class='ex-factors'>" + "".join(parts) + "</div>"


def _all_terms(payload: dict) -> str:
    rows = []
    for term in payload["terms"]:
        value = "输入缺失" if term["value"] is None else f"{term['value']:.6g}"
        name = feature_label(term["side"], term["feature"])
        cells = ["主队进球" if term["side"] == "home" else "客队进球",
                 _escape(name) + f"<code>{_escape(term['feature'])}</code>", _escape(value),
                 f"{term['raw_contribution']:+.6f}", _signed(term["goal_contribution"], goals=True),
                 _signed(term["home_win_contribution"])]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    heads = ["对应输出", "因素", "当时输入值", "原始贡献（对数进球）", "球数贡献", "主胜贡献"]
    table = ("<div class='scroll'><table class='explanation-full'><thead><tr>"
             + "".join(f"<th>{_escape(head)}</th>" for head in heads) + "</tr></thead><tbody>"
             + "".join(rows) + "</tbody></table></div>")
    audit = (f"<p class='small ex-audit'>解释格式 {_escape(payload['schema'])}；LightGBM {_escape(payload['library_version'])}。"
             f"模型训练时间（UTC）：{_escape(payload.get('model_trained_at_utc') or '未知')}。<br>"
             f"模型指纹：{_escape(payload['model_fingerprint'])}</p>")
    return f"<details class='ex-all'><summary>完整条目（{len(rows)} 项）与计算记录</summary>{table}{audit}</details>"


def render_explanation(payload, row: dict, *, uid: str, fusion: dict | None = None, top_factors: int = 3,
                       atol: float = 1e-6, missing_text: str = MISSING) -> str:
    """返回静态片段，调用方引入 EXPLANATION_CSS；uid 在当前页面中必须唯一。"""
    valid, reason = validate_explanation(payload, row, atol=atol)
    if valid is None:
        message = missing_text if payload is None else reason
        return f"<section class='model-explanation'><div class='ex-missing'>{_escape(message)}</div></section>"
    count = top_factors if type(top_factors) is int and top_factors > 0 else 3
    token = "ex-" + hashlib.sha256(str(uid).encode("utf-8")).hexdigest()[:20]
    base, end, terms = valid["base"], valid["prediction"], valid["terms"]
    prob = ("<div class='ex-panel-prob ex-panel'><h3>模型自己的主胜概率</h3>"
            f"<div class='ex-total'><span>解释基准 <b>{base['home_win'] * 100:.2f}%</b></span>"
            f"<span>模型预测 <b>{end['home_win'] * 100:.2f}%</b></span></div>"
            f"{_factor_chart(terms, goals=False, top_factors=count)}"
            f"<p class='ex-note'>各项合计 {_escape(_signed(math.fsum(t['home_win_contribution'] for t in terms)))}。"
            f"按主胜贡献绝对值列前{count}项，其他因素合并；正号提高主胜概率，负号降低主胜概率。</p></div>")
    goals = []
    for side, word in (("home", "主队"), ("away", "客队")):
        own = [term for term in terms if term["side"] == side]
        goals.append(f"<div class='ex-panel'><h3>{word}预期进球</h3><div class='ex-total'>"
                     f"<span>解释基准 <b>{base[side + '_goals']:.3f} 球</b></span>"
                     f"<span>模型预测 <b>{end[side + '_goals']:.3f} 球</b></span></div>"
                     f"{_factor_chart(own, goals=True, top_factors=count)}"
                     f"<p class='ex-note'>各项合计 {_escape(_signed(math.fsum(t['goal_contribution'] for t in own), goals=True))}。"
                     f"正号提高{word}预期进球，负号降低；本侧按主胜贡献绝对值列前{count}项。</p></div>")
    return ("<section class='model-explanation'><p><b>SHAP贡献经路径换算</b></p>"
            "<p class='ex-note'>从解释基准到这版模型预测，观察各因素对应的贡献。球数和百分点为路径换算，"
            "不是这两个单位下的原生精确SHAP；不代表因果或移除该因素的效果。进球基准不是真实平均进球。</p>"
            f"<p class='ex-fusion ex-note'>{_escape(_fusion_note(fusion))}</p>"
            "<fieldset class='explanation-switch'><legend>贡献单位</legend>"
            f"<input class='ex-unit-prob' type='radio' id='{token}-prob' name='{token}' checked>"
            f"<label for='{token}-prob'>主胜概率贡献（百分点）</label>"
            f"<input class='ex-unit-goals' type='radio' id='{token}-goals' name='{token}'>"
            f"<label for='{token}-goals'>预期进球贡献（球）</label>{prob}"
            f"<div class='ex-panel-goals'><div class='ex-goal-grid'>{''.join(goals)}</div></div></fieldset>"
            f"{_all_terms(valid)}</section>")
