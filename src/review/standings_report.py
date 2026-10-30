"""用维基百科的最终积分榜核对我们算的积分榜（复盘自动标签的一部分，需求文档第 15 节）。

做法：
  1. 我们这边：经过取数层（src.features.FeatureStore，用途“复盘”）拿这个联赛赛季全部比分，
     用 src.features.standings 算出赛季结束后的最终积分榜（扣分也算进去）。
  2. 维基百科那边：standings_check.py 下载好的页面里的积分榜（只读本地文件，这里不联网）。
  3. 两边的队名写法不一样（我们 "Man City"，维基 "Manchester City"），
     所以按“胜、平、负、进球、失球”这 5 个数字配对：同一个赛季里两队这 5 个数字完全一样的情况几乎没有。
  4. 配上以后比三样：
       比赛数据一样吗（5 个数字配不上 = 我们数据库少了 / 多了比赛，或者比分不一样）
       积分一样吗（不一样 = 有扣分我们没录，或者录错了）
       名次一样吗（积分一样但名次不一样 = 同分排名规则和官方不一样）
结果写到 data/reports/积分榜核对_汇总.csv（每个联赛赛季一行）和 积分榜核对_差异.csv（每个不一致的球队一行）。

运行：.venv\\Scripts\\python -m src.review.standings_check --report
"""
from __future__ import annotations

import pandas as pd

from src.common.config import resolve_path, season_role
from src.features import FeatureStore, open_readonly
from src.features.standings import load_deductions, season_schedule, season_table
from src.review.labels import MODE_FOR_ROLE, USAGE
from src.review.standings_check import load_page

KEY = ["won", "drawn", "lost", "goals_for", "goals_against"]


def compare_season(ours: pd.DataFrame, wiki: pd.DataFrame) -> tuple[dict, list[dict]]:
    """比一个赛季。ours = 我们的最终积分榜，wiki = 维基百科的积分榜。返回（汇总一行, 差异若干行）。"""
    w = wiki.rename(columns={"gf": "goals_for", "ga": "goals_against", "pos": "wiki_position",
                             "points": "wiki_points"}).copy()
    for c in KEY:
        w[c] = w[c].astype(int)
    o = ours.copy()
    o["key"] = o[KEY].astype(int).astype(str).agg("-".join, axis=1)
    w["key"] = w[KEY].astype(str).agg("-".join, axis=1)
    m = o.merge(w[["key", "team_wiki", "wiki_position", "wiki_points"]], on="key", how="outer", indicator="side")
    diffs = []
    for r in m.itertuples(index=False):
        if r.side != "both":
            side = "只在我们这边" if r.side == "left_only" else "只在维基百科"
            diffs.append({"类型": "比赛数据不一样", "球队": r.team if isinstance(r.team, str) else "",
                          "维基队名": r.team_wiki if isinstance(r.team_wiki, str) else "",
                          "说明": f"{side}：胜平负进失 {r.key}"})
            continue
        if int(r.points) != int(r.wiki_points):
            diffs.append({"类型": "积分不一样", "球队": r.team, "维基队名": r.team_wiki,
                          "说明": f"我们 {int(r.points)}（已扣 {int(r.deduction)}），维基 {int(r.wiki_points)}，差 {int(r.wiki_points) - int(r.points)}"})
        elif int(r.position) != int(r.wiki_position):
            diffs.append({"类型": "名次不一样", "球队": r.team, "维基队名": r.team_wiki,
                          "说明": f"我们第 {int(r.position)}，维基第 {int(r.wiki_position)}（同 {int(r.points)} 分"
                                  f"{'，我们的规则没分开' if r.tie_unresolved else ''}）"})
    both = m[m["side"] == "both"]
    row = {"球队数": len(ours), "维基球队数": len(wiki), "比赛数据配上": len(both),
           "积分一致": int((both["points"].astype(int) == both["wiki_points"].astype(int)).sum()),
           "名次一致": int((both["position"].astype(int) == both["wiki_position"].astype(int)).sum())}
    row["完全一致"] = row["名次一致"] == row["积分一致"] == row["比赛数据配上"] == len(ours) == len(wiki)
    return row, diffs


def write_report(settings: dict, comps: list[str], seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = open_readonly(resolve_path(settings["paths"]["database"]))
    stores = {}
    deductions = load_deductions(settings)
    summary, details = [], []
    for comp in comps:
        for s in seasons:
            page = load_page(comp, s, settings)
            if page is None:
                summary.append({"联赛": comp, "赛季": s, "说明": "没有维基百科页面"})
                continue
            role = season_role(s, settings)
            fs = stores.setdefault(role, FeatureStore(conn, MODE_FOR_ROLE[role], settings))
            sched = season_schedule(fs, comp, s, USAGE)
            if sched.empty:
                summary.append({"联赛": comp, "赛季": s, "说明": "数据库里没有这个赛季"})
                continue
            st = season_table(fs, comp, s, "2999-01-01", USAGE, settings, schedule=sched, deductions=deductions)
            row, diffs = compare_season(st.final(), page["table"])
            summary.append({"联赛": comp, "赛季": s, **row, "页面": page["url"]})
            details += [{"联赛": comp, "赛季": s, **d} for d in diffs]
    out = resolve_path(settings["paths"]["reports"])
    out.mkdir(parents=True, exist_ok=True)
    sm, dt = pd.DataFrame(summary), pd.DataFrame(details, columns=["联赛", "赛季", "类型", "球队", "维基队名", "说明"])
    sm.to_csv(out / "积分榜核对_汇总.csv", index=False, encoding="utf-8-sig")
    dt.to_csv(out / "积分榜核对_差异.csv", index=False, encoding="utf-8-sig")
    ok = sm.get("完全一致", pd.Series(dtype=bool)).fillna(False).astype(bool)
    print(f"核对 {len(sm)} 个联赛赛季，完全一致 {int(ok.sum())} 个；差异 {len(dt)} 条（按类型：{dt['类型'].value_counts().to_dict()}）")
    print(f"写到 {out / '积分榜核对_汇总.csv'} 和 {out / '积分榜核对_差异.csv'}")
    return sm, dt
