"""每日备份（需求文档第 16 节）。

1. 数据库：每天复制一份，文件名带日期，保留最近 N 天（N 在 settings.yaml 的 backup.keep_days）。
2. 爬到的原始页面：只复制新增的文件，不整份复制（原始层只增不改，旧文件不会变）。

运行方法：
    .venv\\Scripts\\python -m src.jobs.backup
由 Windows 任务计划程序每天自动运行。
"""
import logging
import shutil
import sqlite3
from datetime import date, timedelta

from src.common.config import load_settings, resolve_path

log = logging.getLogger("backup")


def backup_database(db_path, backup_dir, keep_days, today=None):
    """备份数据库并删掉超过保留天数的旧备份。返回新备份的路径；数据库还不存在时返回 None。"""
    today = today or date.today()
    if not db_path.exists():
        log.info("数据库还不存在（第 1 阶段才建），跳过")
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"football_{today:%Y%m%d}.db"
    # 用 SQLite 自带的备份功能，不直接复制文件：复制时如果程序正在写数据库，复制出来的文件可能是坏的
    src, dst = sqlite3.connect(db_path), sqlite3.connect(target)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()

    oldest_kept = today - timedelta(days=keep_days - 1)
    for old in backup_dir.glob("football_*.db"):
        day = date(int(old.stem[9:13]), int(old.stem[13:15]), int(old.stem[15:17]))
        if day < oldest_kept:
            old.unlink()
            log.info("删除过期备份 %s", old.name)
    return target


def backup_raw(raw_dir, backup_dir):
    """只复制备份里还没有的原始文件。返回新复制的文件数。"""
    copied = 0
    for f in raw_dir.rglob("*"):
        if not f.is_file():
            continue
        target = backup_dir / f.relative_to(raw_dir)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied += 1
    return copied


def main():
    s = load_settings()
    log_dir = resolve_path(s["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_dir / "backup.log", encoding="utf-8"), logging.StreamHandler()],
    )
    backup_root = resolve_path(s["paths"]["backup"])
    db = backup_database(resolve_path(s["paths"]["database"]), backup_root / "database", s["backup"]["keep_days"])
    if db:
        log.info("数据库已备份到 %s", db)
    n = backup_raw(resolve_path(s["paths"]["raw"]), backup_root / "raw")
    log.info("原始文件新复制 %d 个", n)


if __name__ == "__main__":
    main()
