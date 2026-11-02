"""机器人怎么选注、注额多少、怎么结算、资金模拟（需求文档第 14 节“历史回测第一版规则 bots_v1”）。

这里只有计算，不碰数据库、不读文件：输入是已经取好的概率和赔率，测试可以直接喂假数据。
全是虚拟投注，不连任何账户。

几个词：
- 结果用序号表示：0 = 主胜，1 = 平，2 = 客胜（和 src/evaluate/metrics.py 一样）。
- 赔率是欧洲小数赔率，例如 2.10 = 下 1 元、猜中拿回 2.10 元（含本金）。
- 期望收益（EV）= 概率 × 赔率 − 1。例：概率 50%、赔率 2.10 → 0.5 × 2.10 − 1 = +5%。
- 凯利公式：长期让资金增长最快的下注比例 f = (p × o − 1) ÷ (o − 1)。全凯利波动太大，这里只下四分之一，再封顶 5%。
- CLV（收盘线价值）= 下注赔率 × 平博收盘去水后这个结果的概率 − 1。> 0 说明买到了比收盘更好的价格。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.odds import devig
from src.common.settlement import profit

OUTCOMES = ("H", "D", "A")             # 序号 → 数据库里的结果写法
OUTCOME_ZH = ("主胜", "平", "客胜")
EV_TOL = 1e-12                         # 期望收益和门槛比大小时容许的计算误差（0.05 算出来可能是 0.04999999…）

# 资金模拟里每一注的状态
SIM_PLACED = "下了"
SIM_DAILY_CAP = "单日上限跳过"
SIM_STOPPED = "已停"


# ======================= 选注 =======================
def random_pick(seed: int, match_id: int) -> int:
    """随机机器人：用（种子, match_id）定一个 0 / 1 / 2，等概率。同一场每次重跑都一样，和下注顺序无关。"""
    return int(np.random.default_rng([int(seed), int(match_id)]).integers(0, 3))


def _valid(arr) -> bool:
    """三个数都有（不是空值）。"""
    return arr is not None and len(arr) == 3 and not np.isnan(np.asarray(arr, dtype=float)).any()


def pick_bet(bot: dict, match_id: int, probs: dict, odds, seed: int):
    """按机器人的规则（bot['rule']）选这场下哪个结果。

    probs：{'model': 模型三个概率, 'fusion': 融合三个概率}（没有的是 None）；odds：这个机器人的价格（三个赔率，None = 这场没有像样的价格）。
    返回 (选哪个 0/1/2, 用的概率, 赔率)，不下返回 None。对照组没有概率来源，“用的概率”是 nan。
    规则：
      random   三选一，等概率（种子 + match_id 定）
      favorite 赔率最低的（一样低按主、平、客的顺序取第一个）
      home     每场主胜
      top      概率最高的（一样高取第一个）
      value    期望收益 = 概率 × 赔率 − 1，三个里最大的那个 ≥ 门槛 edge 才下，下最大的那个（一样大取第一个）
    """
    if not _valid(odds):
        return None
    o = np.asarray(odds, dtype=float)
    rule = bot["rule"]
    src = bot.get("prob")
    p = None
    if src is not None:
        p = probs.get(src)
        if not _valid(p):
            return None             # 这个机器人要的概率这场没有（例如没有融合赔率）
        p = np.asarray(p, dtype=float)
    if rule == "random":
        k = random_pick(seed, match_id)
    elif rule == "favorite":
        k = int(np.argmin(o))       # argmin / argmax 碰到一样的取第一个 = 主、平、客的顺序
    elif rule == "home":
        k = 0
    elif rule == "top":
        k = int(np.argmax(p))
    elif rule == "value":
        ev = p * o - 1
        k = int(np.argmax(ev))
        if ev[k] < float(bot["edge"]) - EV_TOL:
            return None
    else:
        raise ValueError(f"不认识的选注规则 {rule!r}（只能是 random / favorite / home / top / value）")
    return k, (float(p[k]) if p is not None else float("nan")), float(o[k])


# ======================= 注额 =======================
def kelly_fraction(prob: float, odds: float, fraction: float, cap: float) -> float:
    """四分之一凯利：f = (p × o − 1) ÷ (o − 1) × fraction，最多 cap，期望收益 ≤ 0 时是 0。"""
    f = (prob * odds - 1) / (odds - 1) * fraction
    return float(min(max(f, 0.0), cap))


def calc_stake(bot: dict, prob: float, odds: float, bankroll: float, cfg: dict) -> float:
    """成绩用的注额（不随资金变）。cfg = settings['bots']。

    flat  固定 cfg['flat_stake']（100）；
    kelly 四分之一凯利 × bankroll。成绩按初始资金算（调用时传 cfg['bankroll']），不复利，结果和下注顺序无关。
    """
    if bot["stake"] == "flat":
        return float(cfg["flat_stake"])
    if bot["stake"] == "kelly":
        return kelly_fraction(prob, odds, float(cfg["kelly_fraction"]), float(cfg["kelly_cap"])) * float(bankroll)
    raise ValueError(f"不认识的注额方式 {bot['stake']!r}（只能是 flat / kelly）")


def sim_stake(bot: dict, prob: float, odds: float, bankroll: float, cfg: dict) -> float:
    """资金模拟用的注额（按当前资金）。
    固定注额类：min(100, 当前资金 × 2%)；凯利类：min(凯利注额按当前资金算, 当前资金 × 5%)。"""
    if bot["stake"] == "flat":
        return float(min(float(cfg["flat_stake"]), bankroll * float(cfg["flat_cap"])))
    k = kelly_fraction(prob, odds, float(cfg["kelly_fraction"]), float(cfg["kelly_cap"])) * bankroll
    return float(min(k, bankroll * float(cfg["kelly_cap"])))


# ======================= 结算 =======================
def settle_bet(pick: int, result: str | None, odds: float, stake: float = 1.0) -> tuple[str, float]:
    """胜平负结算 → (结果, 盈亏)。猜中 = 注额 × (赔率 − 1)，猜错 = −注额（用 settlement.profit）。
    没有比分（改期、中止、取消、判罚）：退回本金，盈亏 0，记“走”。"""
    if result is None or (isinstance(result, float) and np.isnan(result)) or result not in OUTCOMES:
        return "push", 0.0
    outcome = "win" if OUTCOMES[int(pick)] == result else "loss"
    return outcome, float(stake) * profit(outcome, float(odds))


def clv(pick: int, odds: float, close_odds, method: str = "proportional") -> float:
    """CLV = 下注赔率 × 收盘去水后这个结果的概率 − 1。没有三个收盘赔率时是 nan（不参加 CLV 平均）。"""
    if not _valid(close_odds) or (np.asarray(close_odds, dtype=float) <= 1).any():
        return float("nan")
    q = devig(np.asarray(close_odds, dtype=float), method)
    return float(odds * q[int(pick)] - 1)


# ======================= 一个机器人从头跑到尾 =======================
BET_COLS = ["bot", "match_id", "competition", "season", "match_date", "cutoff_utc", "pick", "pick_zh", "bookmaker",
            "odds", "odds_known_utc", "rule", "prob", "ev", "p_model", "p_fusion", "stake", "result", "outcome",
            "profit", "close_home", "close_draw", "close_away", "clv", "result_known_utc"]


def run_bot(bot: dict, matches: pd.DataFrame, probs: dict, odds: dict, settings: dict) -> tuple[pd.DataFrame, dict]:
    """跑一个机器人：每场最多一注，返回 (每注明细, 资金模拟摘要)。

    matches：每场一行（match_id、competition、season、match_date、cutoff_utc、result、result_known_utc、
             close_home / close_draw / close_away = 平博收盘，没有是空值）；
    probs：{'model': 每场三个概率（场数 × 3）, 'fusion': 同上，没有的场是 nan}，行和 matches 对齐；
    odds：{价格名（平博 / 最高）: 表，和 matches 对齐，列 home、draw、away、ok（价格像不像样）、bookmaker、known_at_utc}。
    每注明细里的 stake、profit 是成绩用的（固定 100 / 凯利按初始资金）；sim_* 几列是资金模拟。
    """
    cfg = settings["bots"]
    book = odds[bot["book"]]
    o3 = book[["home", "draw", "away"]].to_numpy(dtype=float)
    ok = book["ok"].to_numpy(dtype=bool)
    pm = np.asarray(probs["model"], dtype=float)
    pf = np.asarray(probs["fusion"], dtype=float)
    close = matches[["close_home", "close_draw", "close_away"]].to_numpy(dtype=float)
    seed = int(cfg["random_seed"])
    method = cfg["devig_method"]
    rows = []
    for i, m in enumerate(matches.itertuples(index=False)):
        if not ok[i]:
            continue                                   # 这组价格不像样（或没有），这场不下
        got = pick_bet(bot, int(m.match_id), {"model": pm[i], "fusion": pf[i]}, o3[i], seed)
        if got is None:
            continue
        k, p, o = got
        stake = calc_stake(bot, p, o, float(cfg["bankroll"]), cfg)
        if stake <= 0:
            continue                                   # 凯利算出 0（期望收益 ≤ 0），不下
        res = m.result if isinstance(m.result, str) else None
        outcome, pl = settle_bet(k, res, o, stake)
        rows.append((bot["name"], int(m.match_id), m.competition, int(m.season), m.match_date, m.cutoff_utc, k,
                     OUTCOME_ZH[k], book["bookmaker"].iloc[i], o, book["known_at_utc"].iloc[i], bot["rule"],
                     p, (p * o - 1) if not np.isnan(p) else np.nan, float(pm[i][k]), float(pf[i][k]), stake, res,
                     outcome, pl, *close[i], clv(k, o, close[i], method), m.result_known_utc))
    bets = pd.DataFrame(rows, columns=BET_COLS)
    bets, sim = simulate_bankroll(bets, bot, cfg)
    return bets, sim


# ======================= 资金模拟（只报告，不参加排名） =======================
def simulate_bankroll(bets: pd.DataFrame, bot: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """从 cfg['bankroll'] 开始，按截止时间从早到晚下注（第 14 节“风险控制”）。

    - 下每一注前，先结算“结果能知道的时间”早于这一注截止时间的注单（还没结算的注单不算进资金）；
    - 单注：sim_stake（按当前资金）；
    - 单日（截止时间同一个 UTC 日期）合计不超过这一天第一注时资金 × daily_cap，超出的这一注跳过；
    - 资金从最高点回撤超过 stop_drawdown 就停（不恢复），记下停在哪天；已经下了的注照常结算。
    返回 (加了 sim_stake、sim_status、sim_profit 三列的明细, 摘要)。
    """
    start = float(cfg["bankroll"])
    if bets.empty:
        out = bets.assign(sim_stake=pd.Series(dtype=float), sim_status=pd.Series(dtype=str),
                          sim_profit=pd.Series(dtype=float))
        return out, {"机器人": bot["name"], "初始资金": start, "最后资金": start, "最高点": start, "最大回撤": 0.0,
                     "停了没有": "没停", "停在哪天": None, "实际下了几注": 0, "单日上限跳过": 0}
    order = bets.sort_values(["cutoff_utc", "match_id"], kind="stable").index
    bankroll = peak = start
    max_dd = 0.0
    stopped_at: str | None = None
    pending: list[tuple[str, int, float]] = []          # (结果能知道的时间, 行号, 盈亏)
    day, day_base, day_used = None, 0.0, 0.0
    sim_stake_col = pd.Series(0.0, index=bets.index)
    sim_status = pd.Series("", index=bets.index, dtype=object)
    sim_profit = pd.Series(0.0, index=bets.index)

    def settle_until(t: str | None) -> None:
        """结算能知道的时间早于 t 的注单（t = None：全部结算），每结算一注检查一次回撤。"""
        nonlocal bankroll, peak, max_dd, stopped_at
        pending.sort()
        while pending and (t is None or pending[0][0] < t):
            known, _, pl = pending.pop(0)
            bankroll += pl
            peak = max(peak, bankroll)
            dd = 1 - bankroll / peak
            max_dd = max(max_dd, dd)
            # 留一点计算误差：1 − 7000 ÷ 10000 算出来是 0.30000000000000004，正好 30% 不算“超过”
            if stopped_at is None and dd > float(cfg["stop_drawdown"]) + EV_TOL:
                stopped_at = str(known)[:10]

    for idx in order:
        r = bets.loc[idx]
        cutoff = str(r["cutoff_utc"])
        settle_until(cutoff)
        if stopped_at is not None:
            sim_status[idx] = SIM_STOPPED
            continue
        d = cutoff[:10]
        if d != day:                                     # 新的一天：记下这一天第一注时的资金
            day, day_base, day_used = d, bankroll, 0.0
        s = sim_stake(bot, float(r["prob"]), float(r["odds"]), bankroll, cfg)
        if day_used + s > day_base * float(cfg["daily_cap"]) + 1e-9:
            sim_status[idx] = SIM_DAILY_CAP
            continue
        day_used += s
        sim_stake_col[idx] = s
        sim_status[idx] = SIM_PLACED
        # 每 1 单位本金的盈亏 × 模拟注额（“走”的是 0）
        pl = float(r["profit"]) / float(r["stake"]) * s if float(r["stake"]) else 0.0
        sim_profit[idx] = pl
        pending.append((str(r["result_known_utc"]), int(idx), pl))
    settle_until(None)
    out = bets.assign(sim_stake=sim_stake_col, sim_status=sim_status, sim_profit=sim_profit)
    summary = {"机器人": bot["name"], "初始资金": start, "最后资金": round(bankroll, 2), "最高点": round(peak, 2),
               "最大回撤": round(max_dd, 4), "停了没有": "停了" if stopped_at else "没停", "停在哪天": stopped_at,
               "实际下了几注": int((sim_status == SIM_PLACED).sum()),
               "单日上限跳过": int((sim_status == SIM_DAILY_CAP).sum())}
    return out, summary
