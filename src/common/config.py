"""读取配置文件。

所有模块都从这里拿配置，不在代码里写死路径和数字，以后搬到云服务器只改 config/settings.yaml。
"""
from pathlib import Path

import yaml

# 项目文件夹 = 这个文件往上三层（src/common/config.py → 项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_settings() -> dict:
    """读 config/settings.yaml，返回一个字典。"""
    with open(CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_secrets() -> dict:
    """读 config/secrets.yaml。文件不存在时返回空字典（还没配推送等密钥时也能运行）。"""
    path = CONFIG_DIR / "secrets.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(p: str) -> Path:
    """配置里的路径：相对路径按项目文件夹算，绝对路径（如备份位置）原样使用。"""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_football_data_columns(settings: dict | None = None) -> dict:
    """读 football-data 的数据字典（config/football_data_columns.yaml），返回 {列名: 这一列的登记信息}。"""
    path = resolve_path((settings or load_settings())["paths"]["football_data_columns"])
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["列"]


def season_role(season_start_year: int, settings: dict | None = None) -> str:
    """判断一个赛季的用途：'dev'（开发）、'exam'（期末考试）、'live'（上线后）。

    赛季用开始年份表示，例如 2023 表示 2023/24。规则见需求文档第 17 节。
    """
    s = (settings or load_settings())["seasons"]
    if season_start_year in s["exam"]:
        return "exam"
    if season_start_year >= s["live_first"]:
        return "live"
    if season_start_year <= s["dev_last"]:
        return "dev"
    raise ValueError(f"赛季 {season_start_year} 不属于任何用途，检查 settings.yaml 的 seasons 配置")
