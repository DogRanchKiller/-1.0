"""第 8 阶段：欧洲杯赛比赛的整理（需求文档第 4 节“第 8 阶段怎么做”里的“规则”）。

取数层 FeatureStore.cup_results 给出“截止时间之前、比分已经能知道”的杯赛和欧战比赛。这里在它的结果上再整理，
每一步都只看每一行自己的内容，不看截止时间以后的任何数据：
  1. 只留 settings.yaml euro.competitions 里的欧战（UCL、UEL、UECL）；
  2. 没有比分的去掉（判罚比分、完赛但缺比分）；
  3. 疑似重复的去掉（规则第 4 条，见 drop_round_duplicates）；
  4. 标出推断的中立场（规则第 2 条，见 mark_neutral）。
比分一律是 90 分钟比分（match_stats），加时、点球不取。

为什么要自己复制一份设置（stage8_settings）：取数层的阶段检查要求“杯赛欧战比分”要到第 8 阶段才能当模型输入，
而共用的 settings.yaml 里 features.current_stage 现在是 2（改它会同时放开别的数据，合并后由你定）。
研究程序只在自己手里的一份设置里把阶段写成 8，用途也只用“研究实验”（只在开发模式，不上线、不进期末考试）。
"""
from __future__ import annotations

import copy

import pandas as pd

from src.common.config import load_settings

EURO_STAGE = 8
USAGE = "研究实验"
# 取数层给的表里，这里要用到的列（round、leg、date_precision 是 matches 表的列，取数层原样带出来）
NEEDED = ["match_id", "competition", "season", "home_team", "away_team", "round", "leg", "date_precision",
          "home_goals", "away_goals", "result"]
REPORT_COLS = ["match_id", "home_team", "away_team", "drop_reason", "partner_match_id", "partner_home", "partner_away",
               "dropped"]


def stage8_settings(settings: dict | None = None) -> dict:
    """复制一份设置，只把 features.current_stage 改成 8（不动共用的 settings.yaml）。交给 FeatureStore 用。"""
    s = copy.deepcopy(settings or load_settings())
    s["features"]["current_stage"] = EURO_STAGE
    return s


def dev_end_cutoff(settings: dict) -> str:
    """“开发赛季的欧战都踢完了”的截止时间：最后一个开发赛季的下一年的 euro.dev_end_month_day（默认 6 月 30 日）。
    那时 2023/24 的欧战都踢完了，2024/25 的资格赛还没开始，所以开发模式取数不会碰到期末考试赛季。"""
    last = int(settings["seasons"]["dev_last"])
    return f"{last + 1}-{settings['euro']['dev_end_month_day']} 00:00:00"


def mark_neutral(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """加一列 neutral（True = 中立场）。库里没有中立场标记，只能推断（规则第 2 条，推测，不是事实）：
    决赛（round = final_round、leg = 0）；以及 single_leg_season 赛季里 single_leg_competitions 的
    single_leg_rounds 单场淘汰赛（2019/20 疫情后八月在一个国家补赛）。其余都当主客场。"""
    rule = cfg["neutral"]                                # settings.yaml euro.neutral 这一小段
    out = df.copy()
    leg0 = out["leg"] == 0
    final = leg0 & (out["round"] == rule["final_round"])
    special = (leg0 & (out["season"] == int(rule["single_leg_season"])) & out["round"].isin(rule["single_leg_rounds"])
               & out["competition"].isin(rule["single_leg_competitions"]))
    out["neutral"] = (final | special).to_numpy()
    return out


def drop_round_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """疑似重复的比赛（规则第 4 条）：同一场比赛因为队名写法不同（例如 Odd / Odds BK）被存了两次，
    一行只知道轮次开始日（date_precision = round），另一行精确到日。
    判断：同赛事、同赛季、同轮次，两行比分完全一样，而且主队相同或客队相同，就当同一场，留日期精度那行。
    只有一个候选才去掉；有几个候选说明说不准，不去掉，只在报告里记一笔。
    返回 (留下的表, 报告)；报告每个疑似行一行，dropped 写明真的去掉了没有。"""
    day = df[df["date_precision"] != "round"]
    rows, gone = [], set()
    for r in df[df["date_precision"] == "round"].to_dict("records"):
        same = day[(day["competition"] == r["competition"]) & (day["season"] == r["season"])
                   & (day["round"] == r["round"]) & (day["home_goals"] == r["home_goals"])
                   & (day["away_goals"] == r["away_goals"])
                   & ((day["home_team"] == r["home_team"]) | (day["away_team"] == r["away_team"]))]
        if len(same) == 0:
            continue
        p = same.iloc[0]
        if len(same) == 1:
            gone.add(r["match_id"])
            reason = "疑似重复：轮次精度的一行和日期精度的一行，同赛事、赛季、轮次，比分相同，有一队相同"
        else:
            reason = "疑似重复但候选不止一行，没去掉"
        rows.append((r["match_id"], r["home_team"], r["away_team"], reason, int(p["match_id"]) if len(same) == 1 else pd.NA,
                     p["home_team"], p["away_team"], len(same) == 1))
    report = pd.DataFrame(rows, columns=REPORT_COLS)
    return df[~df["match_id"].isin(gone)], report


def prepare_euro(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """整理取数层给的杯赛欧战表。返回 (整理好的表, 报告)：表多一列 neutral，比分是整数；
    报告记下没有比分的行和疑似重复的行（dropped = False 的是“疑似但没去掉”）。"""
    missing = [c for c in NEEDED if c not in raw.columns]
    if missing:
        raise ValueError(f"取数层给的表缺这些列：{missing}（matches 表要有 round、leg、date_precision，比分要用“杯赛欧战比分”取）")
    df = raw[raw["competition"].isin(cfg["competitions"])]
    no_score = df["home_goals"].isna() | df["away_goals"].isna()
    reports = []
    if no_score.any():
        gone = df[no_score]
        reports.append(pd.DataFrame({"match_id": gone["match_id"], "home_team": gone["home_team"],
                                     "away_team": gone["away_team"], "drop_reason": "没有比分", "partner_match_id": pd.NA,
                                     "partner_home": pd.NA, "partner_away": pd.NA, "dropped": True}))
    df = df[~no_score]
    df, dup = drop_round_duplicates(df)
    if not dup.empty:
        reports.append(dup)
    df = mark_neutral(df, cfg).reset_index(drop=True)
    for c in ("home_goals", "away_goals"):
        df[c] = df[c].astype(int)
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame(columns=REPORT_COLS)
    return df, report


def fetch_euro(fs, cutoff, usage: str = USAGE, since=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """经取数层拿截止时间之前比分已经能知道的欧战比赛，再整理。fs 要用 stage8_settings 造（见文件开头）。
    返回 (整理好的表, 报告)。只轮次开始日的比赛，比分按这一轮第 7 天以后才算知道（取数层的规则，这里不改）。"""
    cfg = fs.settings["euro"]
    raw = fs.cup_results(cutoff, usage, competition=cfg["competitions"], since=since)
    return prepare_euro(raw, cfg)
