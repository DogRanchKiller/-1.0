"""拿我们自己的数据演示“高命中率”是怎么来的（需求文档第 5 节“别人的项目”第四部分，跑之前写下；只报告，不判对错）。

同一批检验季比赛（SEL 那份：w9_gb_l7、五大联赛 2020–2023、w = 1 的市场矩阵），不新跑模型，只换问题、挑场次：
  - 胜平负全部场次；只挑把握分（最大那项的概率）最高的 10%、5%；
  - 双选：猜不输（“主胜或平”“客胜或平”里概率大的那个）；
  - 大小球固定 1.5、2.5、3.5 球：猜概率大的一边。
看换了问题、挑了场次以后，命中率会变成多少。

运行：.venv\\Scripts\\python -m src.evaluate.others_demo --run-dir <回测文件夹>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate import pick_tiers_study as pt


def demo(t: pd.DataFrame, mats: np.ndarray, settings: dict) -> list[dict]:
    """t、mats 是 pick_tiers_study.load 给的比赛表和市场比分矩阵；只用检验季（cut_last 之后）的比赛。"""
    c, d = settings["pick_tiers_study"], settings["others_review"]["demo"]
    test = (t["season"] > int(c["cut_last"])).to_numpy()
    t, mats = t[test].reset_index(drop=True), mats[test]
    hg, ag = t["home_goals"].to_numpy(int), t["away_goals"].to_numpy(int)
    i, j = np.indices(mats.shape[1:])
    p3 = np.stack([(mats * (i > j)).sum((1, 2)), (mats * (i == j)).sum((1, 2)), (mats * (i < j)).sum((1, 2))], axis=1)
    real = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    side = np.argmax(p3, axis=1)
    p = p3[np.arange(len(p3)), side]
    hit = side == real
    rows = [{"问题": "胜平负，全部场次", "场数": len(t), "命中率": float(hit.mean()), "平均概率": float(p.mean())}]
    for share in d["top_shares"]:
        n = int(round(len(t) * float(share)))
        top = np.argsort(-p, kind="stable")[:n]   # 把握分最高的 n 场
        rows.append({"问题": f"胜平负，只挑把握分最高的 {share:.0%}", "场数": n,
                     "命中率": float(hit[top].mean()), "平均概率": float(p[top].mean())})
    # 双选：主胜或平（不是客胜）、客胜或平（不是主胜），挑概率大的那个
    home_dc, away_dc = p3[:, 0] + p3[:, 1], p3[:, 2] + p3[:, 1]
    pick_home = home_dc >= away_dc
    dc_hit = np.where(pick_home, real != 2, real != 0)
    rows.append({"问题": "双选：猜不输（主胜或平 / 客胜或平）", "场数": len(t), "命中率": float(dc_hit.mean()),
                 "平均概率": float(np.where(pick_home, home_dc, away_dc).mean())})
    over = pt.over_shares(pt.totals(mats), d["ou_fixed"])
    for k, line in enumerate(d["ou_fixed"]):
        pick_over = over[:, k] >= 0.5
        h = np.where(pick_over, hg + ag > line, hg + ag < line)
        rows.append({"问题": f"大小球固定 {line} 球（猜概率大的一边）", "场数": len(t), "命中率": float(h.mean()),
                     "平均概率": float(np.where(pick_over, over[:, k], 1 - over[:, k]).mean()),
                     "猜大球的比例": float(pick_over.mean())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="回测文件夹（默认 settings.yaml backtest.output_dir / pick_tiers_study.run）")
    a = ap.parse_args()
    s = load_settings()
    t, market, _own, _check = pt.load(s, None if a.run_dir is None else resolve_path(a.run_dir))
    rows = demo(t, market, s)
    out = resolve_path(s["others_review"]["demo"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['问题']}：{r['场数']} 场，命中率 {r['命中率']:.1%}，平均概率 {r['平均概率']:.1%}")


if __name__ == "__main__":
    main()
