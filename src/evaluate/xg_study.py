r"""学习小分析：xG 是什么、准不准、比射门和进球好在哪（需求文档第 4 节步骤 4 之前的“先看看数据”）。

只读本机的 Understat 射门明细镜像（data/raw/understat_mirror）和对上号的每场 xG（data/research/understat_xg.csv），
只出报告，不进数据库、不当模型输入。这是“描述统计”，不是回测，所以不经过取数层（没有“截止时间”的问题）；
真正比较模型准不准的实验另外按需求文档第 5 节的办法做。

四个问题：
  1. 射门图：在哪里射门 xG 高？xG 高的地方，实际进球是不是也多？
  2. 校准：xG = 0.3 的射门，是不是真的大约 30% 进了？
  3. 运气会不会回归：上半季进球比 xG 多的队（“运气好”或者“射术好”），下半季还多吗？
  4. 预测下半季：用上半季的 进球 / 射门推算 / xG，哪个最能预测下半季的进球、失球？
     射门推算 = a × 射正 + b × 射偏（a、b 用全部数据拟合，只有 2 个数、几万场比赛，对结果影响可以忽略）。

只用完整的赛季（2014/15–2020/21）；2021/22 镜像只有半季，不用。

运行方法：Claude 运行 `.venv\Scripts\python -m src.evaluate.xg_study`
结果：data/reports/xg_study.html（图在 data/reports/xg_study/ 里），数字另存 data/reports/xg_study/numbers.json
"""
import html
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # 不开窗口，只存图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.clean.understat_match import match_totals, read_shots  # noqa: E402
from src.common.config import load_settings, resolve_path  # noqa: E402

log = logging.getLogger("xg_study")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # 图上的中文
plt.rcParams["axes.unicode_minus"] = False
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0   # 标准球场（米）。Understat 的 X、Y 是 0–1 的比例：X = 1 是对方球门，Y 是横向
XG_BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


# ---------------------------------------------------------------- 数据

def team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """每场比赛 → 每队一行（这队的进球、失球、xG、对手 xG、射门、射正……），按日期排好，标上是这队这个赛季的上半还是下半。"""
    sides = []
    for me, op in (("home", "away"), ("away", "home")):
        sides.append(pd.DataFrame({
            "competition": matches["competition"], "season": matches["season"], "team": matches[f"{me}_team"],
            "day": matches["day"], "gf": matches[f"{me}_goals"], "ga": matches[f"{op}_goals"],
            "xf": matches[f"{me}_xg"], "xa": matches[f"{op}_xg"],
            "sf": matches[f"{me}_shots"], "sotf": matches[f"{me}_sot"],
            "sa": matches[f"{op}_shots"], "sota": matches[f"{op}_sot"]}))
    t = pd.concat(sides, ignore_index=True).sort_values(["competition", "season", "team", "day"], ignore_index=True)
    keys = ["competition", "season", "team"]
    k = t.groupby(keys).cumcount()
    n = t.groupby(keys)["day"].transform("size")
    t["half"] = np.where(k < n // 2, 1, 2)
    return t


def shot_coefficients(t: pd.DataFrame) -> tuple[float, float]:
    """进球 ≈ a × 射正 + b × 射偏（没有常数项，最小二乘）。返回 (a, b)。"""
    x = np.c_[t["sotf"], t["sf"] - t["sotf"]]
    (a, b), *_ = np.linalg.lstsq(x, t["gf"].to_numpy(float), rcond=None)
    return float(a), float(b)


def halves(t: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """每队每赛季上半、下半的场均：进球 gf、失球 ga、xG xf / xa、射门推算 pf / pa。列名例如 gf_1、gf_2。"""
    t = t.assign(pf=a * t["sotf"] + b * (t["sf"] - t["sotf"]), pa=a * t["sota"] + b * (t["sa"] - t["sota"]))
    g = t.groupby(["competition", "season", "team", "half"])[["gf", "ga", "xf", "xa", "pf", "pa"]].mean().unstack("half")
    g.columns = [f"{c}_{h}" for c, h in g.columns]
    return g.reset_index()


def corr(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def compute(shots: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """算报告里要的全部数字。shots：射门明细（不含乌龙球）；matches：对上号的完整赛季比赛（带 day 日期列）。"""
    out: dict = {"射门数": int(len(shots)), "比赛数": int(len(matches))}
    goal = shots["result"] == "Goal"
    out["xG 总和"] = round(float(shots["xG"].sum()), 1)
    out["进球总数（不含乌龙）"] = int(goal.sum())
    b = pd.cut(shots["xG"], XG_BINS, include_lowest=True)
    cal = shots.assign(goal=goal).groupby(b, observed=True).agg(射门=("xG", "size"), 平均xG=("xG", "mean"), 实际进球率=("goal", "mean"))
    out["校准"] = [{"xG 区间": str(k), **{c: (int(v) if c == "射门" else round(float(v), 3)) for c, v in r.items()}}
                 for k, r in cal.iterrows()]
    pen = shots["situation"] == "Penalty"
    out["点球"] = {"次数": int(pen.sum()), "平均 xG": round(float(shots.loc[pen, "xG"].mean()), 3),
                 "实际进球率": round(float(goal[pen].mean()), 3)}

    t = team_rows(matches)
    a, bb = shot_coefficients(t)
    out["射门推算系数"] = {"射正": round(a, 3), "射偏": round(bb, 3)}
    h = halves(t, a, bb)
    out["队伍赛季数"] = int(len(h))
    out["预测下半季"] = {
        "进球": {"上半季进球": corr(h["gf_1"], h["gf_2"]), "上半季射门推算": corr(h["pf_1"], h["gf_2"]),
                 "上半季 xG": corr(h["xf_1"], h["gf_2"])},
        "失球": {"上半季失球": corr(h["ga_1"], h["ga_2"]), "上半季射门推算": corr(h["pa_1"], h["ga_2"]),
                 "上半季 xG": corr(h["xa_1"], h["ga_2"])},
        "净胜球": {"上半季净胜球": corr(h["gf_1"] - h["ga_1"], h["gf_2"] - h["ga_2"]),
                   "上半季射门推算": corr(h["pf_1"] - h["pa_1"], h["gf_2"] - h["ga_2"]),
                   "上半季 xG": corr(h["xf_1"] - h["xa_1"], h["gf_2"] - h["ga_2"])},
    }
    luck1, luck2 = h["gf_1"] - h["xf_1"], h["gf_2"] - h["xf_2"]
    slope, icpt = np.polyfit(luck1, luck2, 1)
    out["运气回归"] = {"相关系数": corr(luck1, luck2), "斜率": float(slope),
                     "上半季最幸运的 20 队，上半季场均多进": float(luck1.nlargest(20).mean()),
                     "这 20 队下半季场均多进": float(luck2[luck1.nlargest(20).index].mean()),
                     "上半季最不幸的 20 队，上半季场均少进": float(-luck1.nsmallest(20).mean()),
                     "这 20 队下半季场均少进": float(-luck2[luck1.nsmallest(20).index].mean()),
                     "最幸运的 5 队": "、".join(f"{h.loc[i, 'team']} {int(h.loc[i, 'season'])}/{(int(h.loc[i, 'season']) + 1) % 100:02d}"
                                            f"（下半季 {luck2[i]:+.2f}）" for i in luck1.nlargest(5).index)}
    season = t.groupby(["competition", "season", "team"])[["gf", "xf", "sf", "sotf"]].sum()
    out["整季：射门和 xG 的相关"] = {"射门": corr(season["sf"], season["xf"]), "射正": corr(season["sotf"], season["xf"]),
                                "进球": corr(season["gf"], season["xf"])}
    keep = h.assign(luck1=luck1, luck2=luck2)
    out["_halves"] = keep   # 画图用，不写进 json
    return out


def draw_pitch(ax) -> None:
    """画半个球场（对方球门在上面）：边线、禁区、小禁区、点球点、球门。单位米。"""
    w, half = PITCH_WIDTH, PITCH_LENGTH / 2
    kw = {"color": "#555", "lw": 1}
    ax.plot([0, w, w, 0, 0], [half, half, PITCH_LENGTH, PITCH_LENGTH, half], **kw)
    for depth, width in ((16.5, 40.32), (5.5, 18.32)):
        x0 = (w - width) / 2
        ax.plot([x0, x0, x0 + width, x0 + width], [PITCH_LENGTH, PITCH_LENGTH - depth, PITCH_LENGTH - depth, PITCH_LENGTH], **kw)
    ax.plot([(w - 7.32) / 2, (w + 7.32) / 2], [PITCH_LENGTH + 0.6] * 2, color="k", lw=3)
    ax.plot(w / 2, PITCH_LENGTH - 11, "o", color="#555", ms=2)
    ax.set_xlim(-1, w + 1)
    ax.set_ylim(half + 10, PITCH_LENGTH + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_shot_map(shots: pd.DataFrame, path: Path) -> None:
    """左：每个位置射门的平均 xG；右：每个位置射门实际进球的比例。两张图越像，说明 xG 越靠谱。"""
    x = shots["Y"] * PITCH_WIDTH            # 横向
    y = shots["X"] * PITCH_LENGTH           # 纵向，对方球门在 105 米处
    goal = (shots["result"] == "Goal").astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, values, title in ((axes[0], shots["xG"], "平均 xG（模型估的进球概率）"), (axes[1], goal, "实际进球比例")):
        draw_pitch(ax)
        hb = ax.hexbin(x, y, C=values, reduce_C_function=np.mean, gridsize=(34, 27), mincnt=30,
                       extent=(0, PITCH_WIDTH, PITCH_LENGTH / 2, PITCH_LENGTH), cmap="magma_r", vmin=0, vmax=0.5)
        ax.set_title(title)
    fig.colorbar(hb, ax=axes, shrink=0.8, label="概率（0.5 及以上同色）")
    fig.suptitle("在哪里射门更容易进（五大联赛 2014/15–2020/21，不含乌龙球；格子里射门少于 30 次的不画）")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(cal: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    xs = [c["平均xG"] for c in cal]
    ys = [c["实际进球率"] for c in cal]
    ax.plot([0, 1], [0, 1], "--", color="#999", label="完全准（xG = 实际进球率）")
    ax.plot(xs, ys, "o-", color="#c0392b", label="Understat xG")
    for c in cal:
        ax.annotate(f"{c['射门']:,} 次", (c["平均xG"], c["实际进球率"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("这一组射门的平均 xG")
    ax.set_ylabel("这一组射门实际进球的比例")
    ax.set_title("xG 准不准（按 xG 大小分组）")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_luck(h: pd.DataFrame, slope: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(h["luck1"], h["luck2"], s=10, alpha=0.5, color="#2c7fb8")
    lim = max(h["luck1"].abs().max(), h["luck2"].abs().max()) * 1.05
    xs = np.array([-lim, lim])
    ax.plot(xs, xs, "--", color="#999", label="完全保持（斜率 1）")
    ax.plot(xs, slope * xs, color="#c0392b", label=f"实际（斜率 {slope:.2f}）")
    ax.axhline(0, color="#ddd", lw=0.8)
    ax.axvline(0, color="#ddd", lw=0.8)
    ax.set_xlabel("上半季：场均进球 − 场均 xG")
    ax.set_ylabel("下半季：场均进球 − 场均 xG")
    ax.set_title("进球比 xG 多（“运气”或“射术”）会不会保持")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_predict(pred: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    groups = list(pred)
    names = ["用实际进球 / 失球", "用射门推算", "用 xG"]
    colors = ["#95a5a6", "#f39c12", "#c0392b"]
    width = 0.26
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [list(pred[g].values())[i] for g in groups]
        bars = ax.bar(np.arange(len(groups)) + (i - 1) * width, vals, width, label=name, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(groups)), [f"预测下半季的{g}" for g in groups])
    ax.set_ylim(0, 1)
    ax.set_ylabel("相关系数（越接近 1 越能预测）")
    ax.set_title("用上半季的哪个数，最能预测下半季")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render(n: dict) -> str:
    """报告网页。数字都从 n 来；解释的话按数字的大小选说法，不写死结论。"""
    e = html.escape
    p = n["预测下半季"]
    cal = n["校准"]
    luck = n["运气回归"]
    a, b = n["射门推算系数"]["射正"], n["射门推算系数"]["射偏"]
    best = {g: max(v, key=v.get) for g, v in p.items()}
    rows = "".join(f"<tr><td>{e(c['xG 区间'])}</td><td>{c['射门']:,}</td><td>{c['平均xG']:.3f}</td><td>{c['实际进球率']:.3f}</td></tr>"
                   for c in cal)
    pred_rows = "".join(
        f"<tr><td>{e(g)}</td>" + "".join(f"<td{' class=best' if k == best[g] else ''}>{v:.3f}</td>" for k, v in vals.items()) + "</tr>"
        for g, vals in p.items())
    keep = luck["斜率"]
    xg_wins = all(best[g] == "上半季 xG" for g in p)
    sc = n["整季：射门和 xG 的相关"]
    return f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>xG 学习小分析</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:980px;margin:24px auto;padding:0 16px;line-height:1.7;color:#222}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:3px 10px;text-align:right}}th{{background:#f4f4f4}}
td.best{{font-weight:bold;background:#fdecea}}img{{max-width:100%}}.note{{color:#666;font-size:90%}}</style>
<h1>xG 学习小分析</h1>
<p class="note">数据：GitHub 上的 Understat 射门明细镜像（只放本机，只做学习和研究），五大联赛 2014/15–2020/21 完整赛季，
{n['比赛数']:,} 场比赛、{n['射门数']:,} 次射门（不含乌龙球）。程序 <code>src/evaluate/xg_study.py</code>，只是描述统计，不是回测。</p>

<h2>1. xG 是什么</h2>
<p>xG（预期进球）= 一次射门“平均来说”能进的概率。Understat 用射门的位置、角度、是不是头球、是传中还是直塞等，
在几十万次射门上学出来：例如点球的 xG 约 {n['点球']['平均 xG']:.2f}（实际 {n['点球']['次数']:,} 个点球进了 {n['点球']['实际进球率']:.0%}）。
一队一场的 xG = 这场所有射门的 xG 加起来，意思是“按机会的质量，这场平均该进几个”。</p>

<h2>2. 在哪里射门更容易进</h2>
<img src="xg_study/shot_map.png" alt="射门图">
<p>颜色越深，进球概率越高。左边是 xG 估的，右边是实际进了多少。两张图形状很像：越靠近球门、越正对球门，越容易进；
禁区外的远射，大多不到 5%。右图边缘更“花”，是因为每个格子的射门次数有限，实际进球比例有运气成分；xG 把这些运气平均掉了。</p>

<h2>3. xG 准不准（校准）</h2>
<img src="xg_study/calibration.png" alt="校准图">
<table><tr><th>xG 区间</th><th>射门次数</th><th>平均 xG</th><th>实际进球率</th></tr>{rows}</table>
<p>点越贴近虚线越准。全部射门的 xG 加起来是 {n['xG 总和']:,.0f}，实际进了 {n['进球总数（不含乌龙）']:,} 个
（{'xG 略高一点' if n['xG 总和'] > n['进球总数（不含乌龙）'] else 'xG 略低一点'}，差 {abs(n['xG 总和'] / n['进球总数（不含乌龙）'] - 1):.1%}）。
这和需求文档第 5 节说的“校准”是同一个意思：模型说 30% 的事，长期看应该真的发生约 30%。</p>

<h2>4. 运气会不会回归</h2>
<img src="xg_study/luck.png" alt="运气回归散点图">
<p>每个点是一支球队的一个赛季（共 {n['队伍赛季数']} 个）。横轴：上半季进球比 xG 多多少（场均）；纵轴：下半季多多少。
如果“进球比 xG 多”全是射术，点会沿着虚线（斜率 1）；如果全是运气，下半季和上半季无关（斜率 0）。
实际斜率是 <b>{keep:.2f}</b>（相关系数 {luck['相关系数']:.2f}）：上半季多出来的，下半季平均只剩约 {keep:.0%}，
其余 {1 - keep:.0%} 回到了 xG 附近。</p>
<p>具体看两头：上半季最“幸运”的 20 队场均比 xG 多进 {luck['上半季最幸运的 20 队，上半季场均多进']:.2f} 个，下半季只多 {luck['这 20 队下半季场均多进']:.2f} 个；
最“不幸”的 20 队上半季场均少进 {luck['上半季最不幸的 20 队，上半季场均少进']:.2f} 个，下半季只少 {luck['这 20 队下半季场均少进']:.2f} 个。
幸运的那头保持得多一些，不幸的那头几乎全部回去了。这符合“有好射手的队能常年多进一点，但没有哪个队会常年射得特别差”的说法，
不过这里只能看出这个趋势，证明不了。上半季最幸运的几队：{e(luck['最幸运的 5 队'])}。总的来说，大部分是运气。
这就是需求文档第 4 节说的“回归均值”：只看最近几场的进球，会把运气当成实力。</p>

<h2>5. 用上半季的哪个数，最能预测下半季</h2>
<img src="xg_study/predict.png" alt="预测下半季">
<table><tr><th></th><th>用实际进球 / 失球</th><th>用射门推算</th><th>用 xG</th></tr>{pred_rows}</table>
<p>数字是相关系数（同一队上半季的场均 和 下半季的场均，越接近 1 说明越能预测）。
“射门推算”= {a:.2f} × 射正 {'+' if b >= 0 else '−'} {abs(b):.2f} × 射偏：一个射正大约值 {a:.2f} 个进球，射偏{'几乎不值' if abs(b) < 0.03 else f'值 {b:.2f} 个'}。
{'三项都是 xG 最好，射门推算居中，实际进球最差' if xg_wins else '最好的一项见表里加粗的格子'}。
意思是：想知道一队“真实”的进攻和防守水平，xG 比进球更靠谱，只看射门也比只看进球好一些。</p>
<p>整个赛季加起来，各队的 xG 和射门次数的相关系数是 {sc['射门']:.2f}，和射正是 {sc['射正']:.2f}，和进球是 {sc['进球']:.2f}。</p>

<h2>6. 这对我们的模型意味着什么</h2>
<p>现在的 dc 模型用的是实际比分。上面第 5 点说明，用 xG 或射门来估球队实力，可能比用比分更准。
但这里是很粗的看法（相关系数、半个赛季一组），不能说明“预测比赛结果”会好多少。
下一步按需求文档第 5 节的规矩做回测：dc 用比分 / 用射门推算 / 用 xG / 比分和 xG 各一半，在同样的比赛上比 log loss，事先写好规则再跑。</p>
<p class="note">注意：Understat 的 xG 只到 2021 年 12 月，上线后没有这个来源（懂球帝的 xG 是另一家的模型，2025 年 12 月起才有）。
所以 Understat 只用来回答“xG 值多少”这个研究问题；能上线的是射门（football-data 一直有）。</p>
"""


def run(settings: dict) -> dict:
    cfg = settings["understat_mirror"]
    last_full = int(cfg["last_season"]) - 1   # 最后一季只有半季
    shots_all = read_shots(resolve_path(settings["paths"]["raw"]), list(cfg["leagues"]), int(cfg["first_season"]), last_full)
    xg = pd.read_csv(resolve_path(settings["paths"]["understat_xg"]))
    xg = xg[xg["season"] <= last_full]
    days = match_totals(shots_all)[["understat_id", "day"]]
    matches = xg.merge(days, on="understat_id", how="left")
    # 只用对上号的比赛的射门（没对上的那几场明细不全）
    shots = shots_all[shots_all["match_id"].astype(int).isin(set(xg["understat_id"])) & (shots_all["result"] != "OwnGoal")]
    n = compute(shots, matches)
    h = n.pop("_halves")
    reports = resolve_path(settings["paths"]["reports"])
    img = reports / "xg_study"
    img.mkdir(parents=True, exist_ok=True)
    plot_shot_map(shots, img / "shot_map.png")
    plot_calibration(n["校准"], img / "calibration.png")
    plot_luck(h, n["运气回归"]["斜率"], img / "luck.png")
    plot_predict(n["预测下半季"], img / "predict.png")
    (img / "numbers.json").write_text(json.dumps(n, ensure_ascii=False, indent=1), encoding="utf-8")
    (reports / "xg_study.html").write_text(render(n), encoding="utf-8")
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = run(load_settings())
    print(json.dumps({k: v for k, v in n.items() if k != "校准"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
