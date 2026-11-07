"""第 0 阶段的自动检查：配置、赛季用途、备份。

运行方法：.venv\\Scripts\\python -m pytest
"""
import sqlite3
from datetime import date

import pytest

from src.common.config import expand_league_groups, load_settings, resolve_path, season_role
from src.jobs.backup import backup_database, backup_raw


def test_settings_load():
    s = load_settings()
    assert set(s["leagues"]["top"]) == {"E0", "SP1", "D1", "I1", "F1"}


def test_expand_league_groups_daily_scope():
    """每日预测和网页：五大 + 英冠西乙德乙 + 日职中超韩K；意乙法乙去掉。"""
    s = load_settings()
    want = {"E0", "SP1", "D1", "I1", "F1", "E1", "SP2", "D2", "JPN", "CHN", "KOR"}
    for key in ("preview", "site"):
        got = expand_league_groups(s[key]["leagues"], s, s[key].get("skip_leagues"))
        assert set(got) == want
        assert "I2" not in got and "F2" not in got
        assert got["E1"] == "英冠" and got["JPN"] == "日职"
    # 训练：梯度提升树联赛编号含 asia，意乙法乙仍在 second 里
    train = expand_league_groups(s["models"]["gbdt_league_groups"], s)
    assert want <= set(train)
    assert {"I2", "F2"} <= set(train)
    # 一个组名也能展开（旧写法）
    assert set(expand_league_groups("top", s)) == set(s["leagues"]["top"])


@pytest.mark.parametrize("season, role", [
    (2005, "dev"), (2023, "dev"),     # 2023/24 及以前是开发赛季
    (2024, "exam"), (2025, "exam"),   # 2024/25、2025/26 是期末考试
    (2026, "live"), (2030, "live"),   # 2026/27 起是上线后
])
def test_season_role(season, role):
    assert season_role(season) == role


def test_relative_and_absolute_paths():
    assert resolve_path("data/raw").is_absolute()
    assert str(resolve_path("D:/足球预测系统_备份")).startswith("D:")


def test_backup_database_keeps_only_recent(tmp_path):
    db = tmp_path / "football.db"
    con = sqlite3.connect(db)
    con.execute("create table t (x)")
    con.execute("insert into t values (1)")
    con.commit()
    con.close()
    out = tmp_path / "backup"
    for d in range(1, 6):  # 连续 5 天备份，只保留 3 天
        backup_database(db, out, keep_days=3, today=date(2026, 11, d))
    names = sorted(p.name for p in out.glob("*.db"))
    assert names == ["football_20261103.db", "football_20261104.db", "football_20261105.db"]
    assert sqlite3.connect(out / names[-1]).execute("select x from t").fetchone() == (1,)


def test_backup_database_missing_db(tmp_path):
    assert backup_database(tmp_path / "none.db", tmp_path / "b", keep_days=30) is None


def test_backup_raw_copies_only_new_files(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "backup"
    (raw / "球探" / "变盘").mkdir(parents=True)
    (raw / "球探" / "变盘" / "a.html.gz").write_bytes(b"1")
    assert backup_raw(raw, out) == 1
    (raw / "球探" / "变盘" / "b.html.gz").write_bytes(b"2")
    assert backup_raw(raw, out) == 1  # 第二次只复制新文件 b
    assert (out / "球探" / "变盘" / "b.html.gz").read_bytes() == b"2"
