"""读取 football-data 的原始 CSV，处理文件本身的格式问题（需求文档第 17 节“历史数据下载后怎么整理”第 4 步）。

这一步只把文件“读对”，不改数据含义：所有值都先当文字读进来，换算数字、日期、队名是后面清洗的事。

下载后逐个文件检查时发现的问题，以及这里的处理办法：
1. 编码不统一：大部分文件是 UTF-8（有的开头带 BOM 标记），7 个旧文件是西欧编码（德国队名里的 ö、ü、ß）。
   → 先按 UTF-8 读，读不通再按西欧编码（cp1252）读。
2. 行尾多余逗号：约 3900 行比表头多出几个空格子。用 pandas 默认读法会把这些行当“坏行”报错或跳过，整场比赛就丢了。
   → 自己按表头长度截断；多出来的格子如果有值，记进“问题清单”。
3. 没有列名的列：表头末尾常有一串空列名。→ 丢掉；这些列里有值也记进问题清单。
4. 重复列名：德乙 2004/05 的表头把立博（Ladbrokes）的三列都写成 LB。→ 按位置改回 LBH、LBD、LBA，并记下。
5. 全空的行（只有逗号）：丢掉。
6. 值前后的空格（包括网页里常见的不换行空格）：去掉。
"""
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ReadResult:
    """读一个文件的结果：表格 + 问题清单。"""
    path: Path
    table: pd.DataFrame                              # 每列都是文字，空格子是 None
    encoding: str                                    # 实际用的编码
    issues: list[str] = field(default_factory=list)  # 读的过程中发现、并已处理的问题


def decode(raw: bytes) -> tuple[str, str]:
    """把文件内容变成文字。先试 UTF-8（自动去掉 BOM），不行再用西欧编码 cp1252。"""
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


def fix_header(names: list[str], issues: list[str]) -> list[str]:
    """整理表头：去空格；重复的列名按位置改正。空列名保持空字符串，后面丢掉。"""
    names = [n.strip() for n in names]
    out = list(names)
    seen: dict[str, int] = {}
    for i, n in enumerate(names):
        if not n:
            continue
        if n in seen:
            first = seen[n]
            if out[i] != n:
                continue  # 这一列已经在前面一起改过了
            # 已知情况：一家公司的胜平负三列都只写了公司缩写（如 LB LB LB）→ 改成 LBH LBD LBA
            if names[first:first + 3] == [n, n, n] and not {n + "H", n + "D", n + "A"} & set(names):
                out[first:first + 3] = [n + "H", n + "D", n + "A"]
                issues.append(f"表头有 3 个 {n}，按位置改成 {n}H、{n}D、{n}A")
            elif out[i] == n:  # 其他重复：加编号，留给人看
                out[i] = f"{n}__重复{i}"
                issues.append(f"表头第 {i + 1} 列和前面重名（{n}），改名为 {out[i]}")
        else:
            seen[n] = i
    return out


def read_football_data_csv(path: Path) -> ReadResult:
    """读一个 football-data 的 CSV 文件，返回 ReadResult。"""
    text, encoding = decode(Path(path).read_bytes())
    rows = list(csv.reader(io.StringIO(text)))
    issues: list[str] = []
    if not rows:
        return ReadResult(Path(path), pd.DataFrame(), encoding, ["文件是空的"])

    header = fix_header(rows[0], issues)
    width = len(header)
    keep = [i for i, n in enumerate(header) if n]          # 有列名的列
    unnamed = [i for i, n in enumerate(header) if not n]   # 没有列名的列

    records = []
    for line_no, row in enumerate(rows[1:], start=2):     # 行号按文件里的算（第 1 行是表头）
        row = [v.strip() for v in row]
        if not any(row):
            continue                                       # 全空的行
        extra = row[width:]
        if any(extra):
            issues.append(f"第 {line_no} 行比表头多出的格子里有值 {[v for v in extra if v]}，已丢弃")
        row = (row + [""] * width)[:width]                 # 短了补空，长了截断
        lost = [row[i] for i in unnamed if row[i]]
        if lost:
            issues.append(f"第 {line_no} 行没有列名的格子里有值 {lost}，已丢弃")
        records.append([row[i] or None for i in keep])

    table = pd.DataFrame(records, columns=[header[i] for i in keep], dtype=object)
    return ReadResult(Path(path), table, encoding, issues)


def keep_league_rows(table: pd.DataFrame, league: str, issues: list[str]) -> pd.DataFrame:
    """只留下 Div（联赛代码）和文件所属联赛一致的行。

    为什么要这一步：网站上西乙（SP2）1993/94–1995/96 的三个文件，内容其实是西甲（SP1）同一季的比赛，
    Div 一列写的也是 SP1。不去掉的话，同一场西甲比赛会被算两次，还会被当成西乙比赛。
    """
    if len(table) == 0 or "Div" not in table.columns:
        return table
    wrong = table["Div"] != league
    if wrong.any():
        seen = sorted(set(table.loc[wrong, "Div"].fillna("空")))
        issues.append(f"{int(wrong.sum())} 行的联赛代码是 {seen}，不是 {league}，已去掉")
    return table[~wrong].reset_index(drop=True)
