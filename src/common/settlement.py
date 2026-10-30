"""亚盘让球、大小球的结算（需求文档第 3 节、第 14 节“结算按真实规则”）。

盘口正负号（和数据库 odds 表一致，见 src/clean/schema.py 的 line 列）：
- 亚盘的 line 是“主队让球数”，负数 = 主队让球。-0.5 = 主队让半球；+0.25 = 主队受让平半（客队让平半）。
  football-data 的 AHh / BbAHh 列就是这个写法，导入时原样存（半一“-0.5,-1”换成 -0.75）。
  2026-10-30 用数据库核对过：多家平均赛前赔率里，主胜赔率低于 1.5（主队很强）的 7,758 场，97.3% 的盘口是负数；
  主胜赔率 3 以上（主队弱）的 13,491 场，90.0% 是正数。例如 2021/22 曼城主场对诺维奇，盘口 -2.5。
- 客队那边的盘口就是主队的相反数：主队 -0.75 等于客队 +0.75。这里的函数都按“主队的 line”传进来，
  再用 side='home' / 'away' 说明买的是哪边，不用自己换符号。
- 大小球的 line 是总进球数的线，例如 2.5、2.75，永远是正数。

结算规则：
- 整数盘（0、-1、2 等）和半球盘（-0.5、2.5 等）：整注按一条线结算。整数盘打平这条线时“走盘”，退回本金。
- 四分之一盘（-0.25、-0.75、2.75 等）：本金拆成两半，分别按相邻的两条线结算。
  例：-0.75 = 一半下 -0.5、一半下 -1。主队刚好赢 1 球：-0.5 那半赢，-1 那半走盘 → 合起来叫“赢半”。
- 结果五种：赢（win）、赢半（half_win）、走（push）、输半（half_loss）、输（loss）。

盈亏都是“每下 1 单位本金”的净盈亏（不含本金），赔率是欧洲小数赔率 o：
  赢 = o − 1；赢半 = (o − 1) / 2；走 = 0；输半 = −0.5；输 = −1。
比赛都按 90 分钟（含补时）的比分算，加时和点球不算（需求文档第 16 节“比赛异常”）。
"""
import math

OUTCOMES = ("win", "half_win", "push", "half_loss", "loss")
OUTCOME_ZH = {"win": "赢", "half_win": "赢半", "push": "走", "half_loss": "输半", "loss": "输"}


def split_line(line: float) -> tuple[float, float]:
    """把盘口拆成两半各自的线。整数盘、半球盘两半一样；四分之一盘拆成相邻的两条线。

    例：-0.75 → (-1.0, -0.5)；0.25 → (0.0, 0.5)；2.75 → (2.5, 3.0)；-0.5 → (-0.5, -0.5)。
    盘口必须是 0.25 的倍数（数据库导入时已经检查过，这里再检查一次）。
    """
    quarters = line * 4
    if not math.isclose(quarters, round(quarters), abs_tol=1e-9):
        raise ValueError(f"盘口 {line} 不是 0.25 的倍数")
    quarters = round(quarters)
    if quarters % 2 == 0:                       # 0.5 的倍数：整数盘或半球盘，不用拆
        return (quarters / 4, quarters / 4)
    return ((quarters - 1) / 4, (quarters + 1) / 4)   # 四分之一盘：往下、往上各 0.25


def settle_margin(margin: float, line: float) -> str:
    """通用结算：margin 是“你买的这一边”的净胜球（或大小球里 总进球数），line 是你这边的盘口。

    每一半：margin + 线 > 0 赢，= 0 走，< 0 输。两半合起来：
    赢+赢 = 赢；赢+走 = 赢半；走+走 = 走；走+输 = 输半；输+输 = 输。
    （两半的线只差 0.5，而进球是整数，所以不会一半赢一半输。）
    """
    results = []
    for half in split_line(line):
        x = margin + half
        results.append(1 if x > 1e-9 else (-1 if x < -1e-9 else 0))
    return {2: "win", 1: "half_win", 0: "push", -1: "half_loss", -2: "loss"}[sum(results)]


def profit(outcome: str, odds: float) -> float:
    """一种结果 + 欧洲赔率 → 每 1 单位本金的净盈亏。例：赔率 1.90 赢 = +0.90，赢半 = +0.45，输半 = −0.5。"""
    if odds <= 1:
        raise ValueError("赔率必须大于 1（欧洲小数赔率）")
    return {"win": odds - 1, "half_win": (odds - 1) / 2, "push": 0.0,
            "half_loss": -0.5, "loss": -1.0}[outcome]


def asian_handicap_margin(home_goals: int, away_goals: int, line: float, side: str = "home") -> tuple[float, float]:
    """亚盘：换成“你买的这一边”的净胜球和盘口。line 永远是主队的盘口（数据库写法）。

    买主队：净胜球 = 主进球 − 客进球，盘口 = line。
    买客队：净胜球 = 客进球 − 主进球，盘口 = −line（主队 -0.75 就是客队 +0.75）。
    """
    if side == "home":
        return home_goals - away_goals, line
    if side == "away":
        return away_goals - home_goals, -line
    raise ValueError("side 只能是 'home'（买主队）或 'away'（买客队）")


def settle_asian_handicap(home_goals: int, away_goals: int, line: float, side: str = "home",
                          odds: float | None = None) -> dict:
    """亚盘让球结算。返回 {'outcome': 结果, 'profit': 每 1 单位本金的净盈亏}；不给赔率时 profit 为 None。

    例（手算见 tests/test_calc.py）：主队 -0.25、赔率 1.90。
    - 主队赢 1–0：两半（0 和 -0.5）都赢 → 赢，+0.90
    - 平 1–1：0 那半走盘、-0.5 那半输 → 输半，−0.5
    - 主队输：两半都输 → 输，−1
    """
    margin, my_line = asian_handicap_margin(home_goals, away_goals, line, side)
    outcome = settle_margin(margin, my_line)
    return {"outcome": outcome, "profit": None if odds is None else profit(outcome, odds)}


def over_under_margin(total_goals: int, line: float, side: str = "over") -> tuple[float, float]:
    """大小球：换成和亚盘一样的“净胜球 + 盘口”，这样两种玩法共用一套结算规则。

    买大：赢的条件是 总进球 > 线，即 总进球 + (−线) > 0 → 净胜球 = 总进球，盘口 = −线。
    买小：赢的条件是 总进球 < 线，即 −总进球 + 线 > 0 → 净胜球 = −总进球，盘口 = +线。
    """
    if line < 0:
        raise ValueError("大小球的线不能是负数")
    if side == "over":
        return total_goals, -line
    if side == "under":
        return -total_goals, line
    raise ValueError("side 只能是 'over'（大）或 'under'（小）")


def settle_over_under(home_goals: int, away_goals: int, line: float, side: str = "over",
                      odds: float | None = None) -> dict:
    """大小球结算。返回 {'outcome': 结果, 'profit': 每 1 单位本金的净盈亏}。

    例：大 2.75、赔率 2.00，进 3 球：2.5 那半赢、3.0 那半走盘 → 赢半，+0.50。
    """
    margin, my_line = over_under_margin(home_goals + away_goals, line, side)
    outcome = settle_margin(margin, my_line)
    return {"outcome": outcome, "profit": None if odds is None else profit(outcome, odds)}
