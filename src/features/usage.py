"""数据用途字典：读 config/data_usage.yaml，检查“这类数据能不能用在这个用途上”（需求文档第 17 节“怎么保证不用错”）。

人看 data_usage.yaml 就知道每种数据能用在哪里；这里的代码读同一个文件做检查，两边不会对不上。
检查分三种，不通过就报错（报错类型见 errors.py）：
1. 用途检查：这类数据在这个范围（以前的比赛 / 本场）允不允许这个用途。例如 A 版不能拿赔率。
2. 阶段检查：有些数据要到某个阶段才能当模型输入。例如角球、黄牌第 10 阶段起。
3. 模式检查：开发 / 考试 / 实时三种模式，各自能把哪类赛季的数据用在哪些用途上。
   另外，标了“只在开发模式: 是”的用途（研究实验），考试、实时模式下一律不能用（构造时给了 mode 才查）。
"""
from pathlib import Path

import yaml

from src.common.config import load_settings, resolve_path
from src.features.errors import SeasonLockedError, UsageNotAllowedError

# 两个范围（见 data_usage.yaml 开头的说明）
PAST = "以前的比赛"
THIS = "本场"

# 运行模式：代码里用英文，调用时中文英文都认
MODE_NAMES = {"dev": "开发", "exam": "考试", "live": "实时"}
_MODE_ALIASES = {"dev": "dev", "开发": "dev", "exam": "exam", "考试": "exam", "live": "live", "实时": "live"}
ROLE_NAMES = {"dev": "开发", "exam": "期末考试", "live": "上线后"}

# data_usage.yaml 里“什么时候能知道”可以写的规则名（具体算法在 known_time.py）
KNOWN_RULES = {"赛程", "开球", "赛后", "赛前赔率", "收盘", "抓取时间", "变盘时间"}


def normalize_mode(mode: str) -> str:
    """把 '开发' / 'dev' 这类写法统一成 'dev'、'exam'、'live'。写错了直接报错，不猜。"""
    if mode not in _MODE_ALIASES:
        raise ValueError(f"运行模式只能是 开发 / 考试 / 实时（dev / exam / live），收到的是：{mode!r}")
    return _MODE_ALIASES[mode]


def season_label(season: int) -> str:
    """2023 → '2023/24'，报错信息里给人看。"""
    return f"{season}/{(season + 1) % 100:02d}"


class DataUsage:
    """数据用途字典。构造时读文件并检查文件本身写得对不对（用途名拼错、规则名不认识，都在这里就报出来）。"""

    def __init__(self, settings: dict | None = None, path: Path | None = None, mode: str | None = None):
        # mode：用这个字典的程序在什么运行模式下（取数层会给）。给了就检查“只在开发模式”的用途
        self.mode = normalize_mode(mode) if mode is not None else None
        self.settings = settings or load_settings()
        f = self.settings["features"]
        path = path or resolve_path(f["data_usage"])
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        self.usages: dict = raw["用途"]          # 用途名 → {赛前, 模型输入, 说明}
        self.data: dict = raw["数据"]            # 数据类别 → 登记信息
        self.stage: int = int(f["current_stage"])  # 现在做到第几阶段
        self.modes: dict = {}                    # 'dev' → {'dev': {用途...}, 'exam': {...}, 'live': {...}}
        self._validate(raw)

    # ---------- 读文件时的自检 ----------
    def _validate(self, raw: dict) -> None:
        known = set(self.usages)
        self.dev_only = {k for k, v in self.usages.items() if v.get("只在开发模式") == "是"}
        for name, item in self.data.items():
            for scope in (PAST, THIS):
                if scope not in item:
                    raise ValueError(f"data_usage.yaml：「{name}」没写“{scope}”的用途（不允许任何用途就写 []）")
                bad = set(item[scope]) - known
                if bad:
                    raise ValueError(f"data_usage.yaml：「{name}」的“{scope}”里有不认识的用途 {sorted(bad)}")
            if item.get("什么时候能知道") not in KNOWN_RULES:
                raise ValueError(f"data_usage.yaml：「{name}」的“什么时候能知道”必须是 {sorted(KNOWN_RULES)} 之一")
        for zh, cfg in raw["运行模式"].items():
            mode = normalize_mode(zh)
            self.modes[mode] = {}
            for role in ROLE_NAMES:
                value = cfg[role]
                allowed = known if value == "全部" else set(value)
                if allowed - known:
                    raise ValueError(f"data_usage.yaml：运行模式「{zh}」的 {role} 里有不认识的用途 {sorted(allowed - known)}")
                if mode != "dev" and value != "全部" and allowed & self.dev_only:
                    raise ValueError(f"data_usage.yaml：运行模式「{zh}」里写了只在开发模式能用的用途 {sorted(allowed & self.dev_only)}")
                self.modes[mode][role] = allowed
        if set(self.modes) != set(MODE_NAMES):
            raise ValueError("data_usage.yaml：“运行模式”要把 开发、考试、实时 三种都写上")

    # ---------- 查询 ----------
    def check_usage_name(self, usage: str) -> None:
        """调用方必须声明用途，而且必须是字典里有的。拼错、没写都报错，不按“随便用”处理。"""
        if usage not in self.usages:
            raise UsageNotAllowedError(
                f"用途「{usage}」不认识。取数据时必须声明用途，只能是：{'、'.join(self.usages)}"
                f"（定义在 config/data_usage.yaml 的“用途”）")
        if usage in self.dev_only and self.mode not in (None, "dev"):
            raise UsageNotAllowedError(
                f"不允许：用途「{usage}」只能在开发模式下用，现在是{MODE_NAMES[self.mode]}模式。\n"
                f"  原因：{self.usages[usage].get('说明', '')}")

    def is_pre_match(self, usage: str) -> bool:
        """这个用途是不是“开球前做决定”（是的话，本场数据的截止时间必须早于开球）。"""
        self.check_usage_name(usage)
        return self.usages[usage]["赛前"] == "是"

    def columns(self, data_type: str) -> list[str]:
        """这类数据在数据库里对应哪些列（只有比分、统计这类有“列”）。"""
        return list(self.item(data_type).get("列", []))

    def item(self, data_type: str) -> dict:
        if data_type not in self.data:
            raise ValueError(f"数据类别「{data_type}」没在 config/data_usage.yaml 里登记。能用的：{'、'.join(self.data)}")
        return self.data[data_type]

    def check(self, data_type: str, scope: str, usage: str) -> None:
        """用途检查 + 阶段检查。不允许就抛 UsageNotAllowedError，信息里写清楚允许什么、为什么。"""
        self.check_usage_name(usage)
        item = self.item(data_type)
        allowed = item[scope]
        reason = item.get("注意", "")
        if usage not in allowed:
            raise UsageNotAllowedError(
                f"不允许：{'以前比赛' if scope == PAST else '本场'}的「{data_type}」不能用于「{usage}」。\n"
                f"  允许的用途：{'、'.join(allowed) if allowed else '没有'}。\n"
                f"  原因：{reason}\n"
                f"  （规则在 config/data_usage.yaml，依据需求文档第 17 节“数据用途”）")
        start = item.get("模型输入从第几阶段起")
        if start is not None and self.usages[usage]["模型输入"] == "是" and self.stage < int(start):
            raise UsageNotAllowedError(
                f"不允许：「{data_type}」要到第 {start} 阶段才能当模型输入（这次声明的用途是「{usage}」），"
                f"现在是第 {self.stage} 阶段（settings.yaml 的 features.current_stage）。\n"
                f"  原因：{reason}")

    def check_mode(self, mode: str, role: str, usage: str | None, what: str) -> None:
        """模式检查：这个运行模式能不能把这类赛季（dev / exam / live）的数据用于这个用途。

        usage 为 None 表示“随便哪种用途”——只要这个模式对这类赛季有任何一种允许的用途就算通过。
        what 是报错时说“碰到了什么”，例如“2024/25 赛季 380 场比赛”。
        """
        allowed = self.modes[mode][role]
        ok = bool(allowed) if usage is None else usage in allowed
        if ok:
            return
        if not allowed:
            detail = f"{MODE_NAMES[mode]}模式完全不能读{ROLE_NAMES[role]}赛季，连把它当预测目标都不行"
        else:
            detail = f"{MODE_NAMES[mode]}模式下{ROLE_NAMES[role]}赛季只能用于：{'、'.join(sorted(allowed))}"
        hint = {
            "dev": "开发模式只能用 2023/24 及以前的开发赛季：把截止时间提前到期末考试赛季开始之前，或者改用考试模式（第 4 阶段末只考一次）。",
            "exam": "考试模式不能读上线后的赛季（那是以后的真实成绩）。",
            "live": "实时模式不在期末考试赛季上打分、调参（需求文档第 5 节“锁住只针对评分和调参”）。",
        }[mode]
        raise SeasonLockedError(f"不允许：{what}，属于{ROLE_NAMES[role]}赛季。{detail}。\n  {hint}")
