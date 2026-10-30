"""football-data 读文件、列清点、数据字典的自动检查。用临时造的小文件，不读真实数据。"""
import pandas as pd
import pytest

from src.clean.football_data_inventory import item_coverage, season_files, season_from_code, stable_ranges
from src.clean.football_data_reader import read_football_data_csv
from src.common.config import load_football_data_columns


def write(tmp_path, text, encoding="utf-8", name="t.csv"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


# ---------- 读文件 ----------
def test_trailing_commas_keep_the_match(tmp_path):
    """行尾多余逗号：这场比赛不能丢（pandas 默认读法会把它当坏行）。"""
    p = write(tmp_path, "Div,HomeTeam,AwayTeam\nE0,Arsenal,Chelsea,,,\nE0,Leeds,Hull\n")
    r = read_football_data_csv(p)
    assert len(r.table) == 2
    assert r.issues == []                                   # 多出来的格子是空的，不算问题


def test_values_beyond_header_are_reported(tmp_path):
    p = write(tmp_path, "Div,HomeTeam,AwayTeam\nI2,Bari,Ascoli,3,0\n")
    r = read_football_data_csv(p)
    assert list(r.table.columns) == ["Div", "HomeTeam", "AwayTeam"]
    assert len(r.issues) == 1 and "第 2 行" in r.issues[0]


def test_unnamed_columns_dropped(tmp_path):
    p = write(tmp_path, "Div,HomeTeam,,\nE0,Arsenal,,x\n")
    r = read_football_data_csv(p)
    assert list(r.table.columns) == ["Div", "HomeTeam"]
    assert "没有列名" in r.issues[0]


def test_duplicate_lb_header_fixed(tmp_path):
    """德乙 2004/05 的表头写成 LB,LB,LB，要按位置改回 LBH、LBD、LBA，且只记一条问题。"""
    p = write(tmp_path, "Div,IWA,LB,LB,LB,SBH\nD2,3.1,2.25,2,1.909,2.1\n")
    r = read_football_data_csv(p)
    assert list(r.table.columns) == ["Div", "IWA", "LBH", "LBD", "LBA", "SBH"]
    assert r.table.loc[0, "LBA"] == "1.909"
    assert len(r.issues) == 1


def test_encodings(tmp_path):
    bom = write(tmp_path, "\ufeffDiv,HomeTeam\nD1,Köln\n", name="a.csv")  # 开头带 BOM 标记
    old = write(tmp_path, "Div,HomeTeam\nD1,Köln\n", encoding="cp1252", name="b.csv")
    for p, enc in ((bom, "utf-8"), (old, "cp1252")):
        r = read_football_data_csv(p)
        assert r.encoding == enc
        assert list(r.table.columns) == ["Div", "HomeTeam"]  # BOM 不会粘在第一个列名上
        assert r.table.loc[0, "HomeTeam"] == "Köln"


def test_blank_rows_and_spaces(tmp_path):
    p = write(tmp_path, "Div,HomeTeam\n,,,\nE0, Arsenal \nE0,\n")
    t = read_football_data_csv(p).table
    assert list(t["HomeTeam"]) == ["Arsenal", None]          # 去掉空格；空格子是 None；全空行丢掉


# ---------- 清点 ----------
def test_season_from_code():
    assert [season_from_code(c) for c in ("9394", "9900", "0405", "2627")] == [1993, 1999, 2004, 2026]


def test_item_coverage_merges_renamed_columns():
    """早年列名 BbAvH 和后来的 AvgH 是同一个数据项；三个部分都有才算这场有。"""
    parts = {"主胜": ["BbAvH", "AvgH"], "平": ["BbAvD", "AvgD"], "客胜": ["BbAvA", "AvgA"]}
    old = pd.DataFrame({"BbAvH": ["2", "2"], "BbAvD": ["3", None], "BbAvA": ["4", "4"]})
    assert item_coverage(old, parts) == 0.5
    new = pd.DataFrame({"AvgH": ["2"], "AvgD": ["3"], "AvgA": ["4"]})
    assert item_coverage(new, parts) == 1.0
    assert item_coverage(pd.DataFrame({"HS": ["1"]}), parts) == 0.0


def test_stable_ranges():
    cov = {2010: 0.95, 2011: 1.0, 2012: 0.3, 2013: 0.92, 2014: 1.0}
    assert stable_ranges(cov, 0.9) == "2010/11–2011/12、2013/14–2014/15"
    assert stable_ranges({2010: 0.0}, 0.9) == "没有"
    assert stable_ranges({2010: 0.5}, 0.9) == "没有（最多 50%）"


def test_season_files_uses_latest_snapshot(tmp_path):
    folder = tmp_path / "football-data" / "E0"
    (folder / "2627").mkdir(parents=True)
    for name in ("2425.csv", "2627/2026-10-30.csv", "2627/2026-11-06.csv"):
        (folder / name).write_text("Div\n")
    files = season_files(tmp_path, "E0")
    assert list(files) == [2024, 2026]
    assert files[2026].name == "2026-11-06.csv"


# ---------- 数据字典（读真实的 config/football_data_columns.yaml）----------
@pytest.fixture(scope="module")
def columns():
    return load_football_data_columns()


def test_dictionary_fields(columns):
    for col, info in columns.items():
        for key in ("含义", "类别", "时点", "数据项", "部分", "用途", "依据"):
            assert key in info, f"{col} 缺 {key}"
        assert info["时点"] in ("赛前", "收盘", "赛后"), col


def test_closing_odds_marked_closing(columns):
    """防偷看未来：收盘赔率列必须标“收盘”；角球 HC/AC、Coral 的 CLH 等名字里有 C 但不是收盘。"""
    for c in ("PSCH", "B365CH", "AvgCH", "MaxC>2.5", "AvgCAHH", "AHCh", "BFECA", "CLCH"):
        assert columns[c]["时点"] == "收盘", c
    for c in ("HC", "AC", "CLH", "PSH", "AvgH", "AHh", "BbAvH"):
        assert columns[c]["时点"] != "收盘", c
    for c in ("FTHG", "FTAG", "FTR", "HS", "HST", "HxG", "HY", "HR"):
        assert columns[c]["时点"] == "赛后", c


def test_renamed_columns_share_item(columns):
    """notes.txt 写明的改名：PH→PSH、BbAv→Avg、BbMx→Max、BbAHh→AHh，要归到同一个数据项。"""
    assert columns["BbAvH"]["数据项"] == columns["AvgH"]["数据项"]
    assert columns["BbMx>2.5"]["数据项"] == columns["Max>2.5"]["数据项"]
    assert columns["BbAHh"]["数据项"] == columns["AHh"]["数据项"]
    assert columns["P>2.5"]["数据项"].startswith("平博") and columns["PSH"]["数据项"].startswith("平博")
