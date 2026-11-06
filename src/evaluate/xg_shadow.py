"""影子预测对比：影子模型和正式模型，同一截止时间的 T-24 比谁更准（需求文档第 4 节“xG 研究第 2 段怎么做”）。
2026-11-07 以前正式是 gb_l7、影子是 gb_xg；以后正式是 gb_xg、影子是 gb_l7（“换成 gb_xg 上线”）。
两段比的是同一对模型、同一个截止时间，所以按“模型对”合起来算，方向固定（见二），换模型不从头计数。

一、比哪些场
  shadow_predictions 表里的每一行，配上它记的 main_prediction_id 那行正式预测（predictions 表，同一个截止时间）。
  比赛已经踢完、取数层用途「评估」拿得到比分的才算。

二、分数（差 = 新 − 旧，小于 0 = 新模型更准）
  “新”= 这对模型里先当影子的那个（gb_xg 先当影子、和 gb_l7 比），“旧”= 另一个。
  不管那一行谁是正式、谁是影子，都按 新 − 旧 算：换模型以后影子 − 正式 正好反过来，那几行把差取反。
  每场一行里也照实记下那一行谁是影子、谁是正式。
  主比较：胜平负 log loss。另外报告：Brier、RPS、比分 log loss、大 2.5 球 Brier、双方进球 Brier（后三个从各自存的比分矩阵算）。
  还报告两边预测差多少（主胜概率平均绝对差）：差得很少时，分数也不会差多少。
  存档的概率是 4 位小数，加起来和 1 差一点点，算分前先除以和。

三、区间和结论
  差的平均按周成块 bootstrap（src/evaluate/compare.py 同一个函数、settings.yaml compare 段同样的设置）。
  少于 shadow.min_matches 场只列数、不算区间、不下结论。下了结论也只是交给你看，换不换你定（事先写下的规则）。

四、结果（settings.yaml shadow.output_dir，data/ 不进 Git）
  xg_shadow_matches.csv   每场一行：两边的概率、比分、六个分数
  xg_shadow_summary.csv   每对模型、每个分数一行：场数（其中新模型当影子 / 当正式各几场）、两边平均分、差、区间、结论

只读：不改存档、模型、网站、定时任务。
用法：
  .venv\\Scripts\\python -m src.evaluate.xg_shadow
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate.compare import block_ids, bootstrap_mean_diff, verdict
from src.features import FeatureStore, open_readonly
from src.features.known_time import to_utc_text

USAGE = "评估"
SCORES = ["胜平负 log loss", "Brier", "RPS", "比分 log loss", "大 2.5 球 Brier", "双方进球 Brier"]


def load_pairs(con) -> pd.DataFrame:
    """影子预测配上对应的那行正式预测：每行一场一个影子模型，两边的胜平负、比分矩阵都在。"""
    q = """select s.shadow_id, s.main_prediction_id, s.match_id, s.competition, s.season, s.home_team, s.away_team,
                  s.match_date, s.kind, s.model as shadow_model, s.model_version as shadow_version, s.cutoff_utc,
                  s.p_home as s_home, s.p_draw as s_draw, s.p_away as s_away, s.detail as s_detail,
                  p.model as main_model, p.model_version as main_version, p.cutoff_utc as main_cutoff,
                  p.p_home as m_home, p.p_draw as m_draw, p.p_away as m_away, p.detail as m_detail
           from shadow_predictions s join predictions p on p.prediction_id = s.main_prediction_id
           order by s.shadow_id"""
    df = pd.read_sql(q, con)
    bad = df[df["cutoff_utc"] != df["main_cutoff"]]
    if len(bad):
        # 影子预测按规则和正式那行同一个截止时间；对不上说明存错了，不能拿来比
        raise ValueError(f"有 {len(bad)} 行影子预测的截止时间和正式预测不一样（shadow_id {bad['shadow_id'].tolist()[:5]}）")
    return df


def _matrix(detail_text) -> np.ndarray | None:
    """detail 里的模型比分矩阵（除以和，免得 6 位小数的舍入影响）；没有是 None。"""
    d = json.loads(detail_text) if detail_text else {}
    m = d.get("模型比分矩阵")
    if not m:
        return None
    m = np.asarray(m, dtype=float)
    return m / m.sum()


def scores(p, m, hg: int, ag: int, eps: float) -> dict:
    """一边的六个分数。p：胜平负三个概率；m：比分矩阵（可以是 None，比分相关的三个分数就是空）。"""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    y = 0 if hg > ag else (1 if hg == ag else 2)
    hot = np.eye(3)[y]
    cum_p, cum_y = np.cumsum(p)[:2], np.cumsum(hot)[:2]
    out = {"胜平负 log loss": -np.log(max(p[y], eps)), "Brier": float(((p - hot) ** 2).sum()),
           "RPS": float(((cum_p - cum_y) ** 2).sum() / 2),
           "比分 log loss": np.nan, "大 2.5 球 Brier": np.nan, "双方进球 Brier": np.nan}
    if m is not None:
        n_h, n_a = m.shape
        i, j = np.indices(m.shape)
        out["比分 log loss"] = -np.log(max(float(m[min(hg, n_h - 1), min(ag, n_a - 1)]), eps))
        over = float(m[(i + j) >= 3].sum())
        btts = float(m[(i >= 1) & (j >= 1)].sum())
        out["大 2.5 球 Brier"] = (over - float(hg + ag >= 3)) ** 2
        out["双方进球 Brier"] = (btts - float(hg >= 1 and ag >= 1)) ** 2
    return out


def match_rows(fs: FeatureStore, pairs: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """每对 → 一行（踢完、拿得到比分的才有）。"""
    eps = settings["calc"]["logloss_eps"]
    now = to_utc_text(datetime.now(timezone.utc))
    rows = []
    for _, r in pairs.iterrows():
        res = fs.match_result(int(r["match_id"]), now, USAGE)
        if res.empty or pd.isna(res.iloc[0]["home_goals"]) or pd.isna(res.iloc[0]["away_goals"]):
            continue
        hg, ag = int(res.iloc[0]["home_goals"]), int(res.iloc[0]["away_goals"])
        sp, mp = [r["s_home"], r["s_draw"], r["s_away"]], [r["m_home"], r["m_draw"], r["m_away"]]
        s = scores(sp, _matrix(r["s_detail"]), hg, ag, eps)
        m = scores(mp, _matrix(r["m_detail"]), hg, ag, eps)
        row = {"shadow_id": int(r["shadow_id"]), "main_prediction_id": int(r["main_prediction_id"]),
               "match_id": int(r["match_id"]), "联赛": r["competition"], "赛季": int(r["season"]),
               "主队": r["home_team"], "客队": r["away_team"], "比赛日期": r["match_date"], "时点": r["kind"],
               "截止时间_utc": r["cutoff_utc"], "影子模型": r["shadow_model"], "影子版本": r["shadow_version"],
               "正式模型": r["main_model"], "正式版本": r["main_version"], "比分": f"{hg}-{ag}",
               "影子主胜": sp[0], "影子平": sp[1], "影子客胜": sp[2], "正式主胜": mp[0], "正式平": mp[1], "正式客胜": mp[2]}
        for k in SCORES:
            row[f"影子 {k}"], row[f"正式 {k}"] = s[k], m[k]
            row[f"差 {k}"] = s[k] - m[k]                      # 影子 − 正式（这一行的实际角色）；汇总时再换成 新 − 旧
        rows.append(row)
    return pd.DataFrame(rows)


def model_pairs(pairs: pd.DataFrame, settings: dict) -> list[tuple[str, str]]:
    """每对模型的（新, 旧）。新 = 这对里先当影子的那个（按 shadow_id 最早的一行），不管后来谁上线。
    pairs 是 load_pairs 的结果（含没踢完的场，免得只有换模型以后的场踢完时把方向定反）。
    还没有任何影子预测时：settings 里的每个影子模型配当前上线模型。"""
    if pairs is None or pairs.empty:
        live = settings["preview"]["model"]
        return [(m, live) for m in settings["shadow"]["models"]]
    out, seen = [], set()
    for _, r in pairs.sort_values("shadow_id").iterrows():
        key = frozenset((r["shadow_model"], r["main_model"]))
        if key not in seen:
            seen.add(key)
            out.append((r["shadow_model"], r["main_model"]))
    return out


def oriented(rows: pd.DataFrame, new: str, old: str) -> pd.DataFrame:
    """这对模型的场，换成 新 / 旧 两边：新 <分数>、旧 <分数>、差 <分数> = 新 − 旧，新主胜、旧主胜、新当影子。"""
    if rows.empty:
        return rows
    a = rows[(rows["影子模型"] == new) & (rows["正式模型"] == old)]   # 新模型当影子（换之前）
    b = rows[(rows["影子模型"] == old) & (rows["正式模型"] == new)]   # 新模型当正式（换之后）
    parts = []
    for g, new_is_shadow in ((a, True), (b, False)):
        if g.empty:
            continue
        n_side, o_side = ("影子", "正式") if new_is_shadow else ("正式", "影子")
        d = pd.DataFrame({"比赛日期": g["比赛日期"].values, "新当影子": new_is_shadow,
                          "新主胜": g[f"{n_side}主胜"].values, "旧主胜": g[f"{o_side}主胜"].values})
        for k in SCORES:
            d[f"新 {k}"] = g[f"{n_side} {k}"].values
            d[f"旧 {k}"] = g[f"{o_side} {k}"].values
            d[f"差 {k}"] = d[f"新 {k}"] - d[f"旧 {k}"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0]


def summarize(rows: pd.DataFrame, settings: dict, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """每对模型、每个分数一行，差 = 新 − 旧（换模型前后两段合起来）。少于 shadow.min_matches 场只列数，不算区间、不下结论。
    pairs：model_pairs 给的（新, 旧）；不给就按 settings 配（影子模型, 上线模型）。"""
    need, cmp = int(settings["shadow"]["min_matches"]), settings["compare"]
    if pairs is None:
        pairs = model_pairs(None, settings)
    out = []
    for new, old in pairs:
        g = oriented(rows, new, old)
        for k in SCORES:
            ok = g[g[f"差 {k}"].notna()] if len(g) else g
            n_sh = int(ok["新当影子"].sum()) if len(ok) else 0
            r = {"新模型": new, "旧模型": old, "分数": k, "主比较": k == SCORES[0], "场数": len(ok),
                 "新当影子的场": n_sh, "新当正式的场": len(ok) - n_sh}
            if len(ok):
                r.update({"新平均分": ok[f"新 {k}"].mean(), "旧平均分": ok[f"旧 {k}"].mean(),
                          "差的平均": ok[f"差 {k}"].mean(),
                          "主胜概率平均绝对差": (ok["新主胜"] - ok["旧主胜"]).abs().mean()})
            if len(ok) >= need:
                mean, lo, hi = bootstrap_mean_diff(ok[f"差 {k}"], block_ids(ok["比赛日期"], cmp["block"]),
                                                   cmp["n_boot"], cmp["confidence"], cmp["seed"])
                r.update({"区间下限": lo, "区间上限": hi, "结论": verdict(lo, hi) if k == SCORES[0] else "只报告"})
            else:
                r.update({"区间下限": np.nan, "区间上限": np.nan,
                          "结论": f"场数不够（{len(ok)} 场，要 {need} 场），只列数" if len(ok) else "还没有踢完的场"})
            out.append(r)
    return pd.DataFrame(out)


def build(fs: FeatureStore, con, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """存档 → 每场一行 → 汇总。数据库里还没有 shadow_predictions 表（定时预测还没跑过新版）时两张都是空的。"""
    has = con.execute("select 1 from sqlite_master where type = 'table' and name = 'shadow_predictions'").fetchone()
    pairs = load_pairs(con) if has else pd.DataFrame()
    rows = match_rows(fs, pairs, settings) if len(pairs) else pd.DataFrame()
    return rows, summarize(rows, settings, model_pairs(pairs, settings))


def main() -> None:
    ap = argparse.ArgumentParser(description="影子预测对比：gb_xg − gb_l7，同一截止时间的 T-24，换模型前后合起来（只读）")
    ap.add_argument("--db", help="数据库文件（不写 = settings.yaml 的 paths.database）")
    a = ap.parse_args()
    settings = load_settings()
    con = open_readonly(a.db)
    fs = FeatureStore(con, "实时", settings)          # 影子预测是上线后的比赛（2026/27 起），用实时模式
    rows, summary = build(fs, con, settings)
    out = resolve_path(settings["shadow"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "xg_shadow_matches.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "xg_shadow_summary.csv", index=False, encoding="utf-8-sig")
    print(f"踢完、两边都有的场：{len(rows)}，写到 {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
