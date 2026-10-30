"""用维基百科的赛季最终积分榜核对我们算的积分榜（第 1 阶段：复盘自动标签）。

为什么要核对：积分榜是自己从比分一场一场加出来的，加错了不会报错。维基百科每个联赛每个赛季有一页，
页面上有最终积分榜（场次、胜平负、进失球、积分、名次），还写了“排名规则”和扣分的注释。
拿它和我们算的对一遍：
- 胜平负、进失球对不上 → 我们的比分数据少了或错了；
- 积分对不上（积分 ≠ 3×胜 + 平）→ 多半是扣分，要录进 config/points_deductions.yaml；
- 名次对不上（积分一样时谁在前）→ 同分排名规则（config/league_rules.yaml）写得不对。

访问规则（2026-10-30 查过，需求文档 6.1 已登记“维基百科：robots.txt 允许；内容 CC BY-SA”）：
- 只访问普通页面 https://en.wikipedia.org/wiki/页面名。robots.txt 禁止 /w/ 和 /api/（程序接口），不用它们。
- 低频：两次请求之间至少隔 settings.yaml 的 scrape.min_interval_seconds 秒；下载过的页面存在原始层，不重复下载。
- 页面原样压缩存进 data/raw/wikipedia/standings/联赛/赛季.html.gz（原始层只增不改）。

运行（在项目文件夹里）：
    .venv\\Scripts\\python -m src.review.standings_check --fetch     下载还没下载的页面（约 330 页，半小时左右）
    .venv\\Scripts\\python -m src.review.standings_check --report    核对，报告写到 data/reports/
"""
import argparse
import gzip
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.common.config import load_settings, resolve_path

USER_AGENT = "football-learning-project/0.1 (personal study; low frequency)"
BASE = "https://en.wikipedia.org/wiki/"


def season_text(season: int) -> str:
    """2022 → '2022–23'；1999 → '1999–2000'（维基百科页面名的写法，中间是长横线 –）。"""
    end = season + 1
    return f"{season}–{end % 100:02d}" if end % 100 != 0 else f"{season}–{end}"


def candidate_titles(comp: str, season: int) -> list[str]:
    """一个联赛赛季在维基百科可能的页面名，按顺序试。改过名的联赛，旧名字的页面一般会自动跳到新名字。"""
    s = season_text(season)
    names = {
        "E0": ["Premier League", "FA Premier League"],
        "E1": ["EFL Championship", "Football League Championship", "Football League First Division"],
        "SP1": ["La Liga"],
        "SP2": ["Segunda División"],
        "D1": ["Bundesliga"],
        "D2": ["2. Bundesliga"],
        "I1": ["Serie A"],
        "I2": ["Serie B"],
        "F1": ["Ligue 1", "French Division 1"],
        "F2": ["Ligue 2", "French Division 2"],
    }[comp]
    return [f"{s} {n}" for n in names]


def raw_path(comp: str, season: int, settings: dict) -> Path:
    return resolve_path(settings["standings_check"]["raw_dir"]) / comp / f"{season}.html.gz"


def fetch_all(settings: dict, comps: list[str], seasons: list[int]) -> None:
    """下载还没有的页面。每次请求后等 min_interval_seconds 秒。"""
    wait = float(settings["scrape"]["min_interval_seconds"])
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    for comp in comps:
        for season in seasons:
            path = raw_path(comp, season, settings)
            if path.exists() or path.with_suffix(".missing").exists():
                continue
            got = None
            for title in candidate_titles(comp, season):
                url = BASE + title.replace(" ", "_")
                r = sess.get(url, timeout=60)
                time.sleep(wait)
                if r.status_code in (403, 429):
                    # 网站拒绝访问或嫌我们太快：整个停下，不重试、不绕过（和其他采集程序同一个规矩）
                    raise SystemExit(f"维基百科返回 {r.status_code}（{url}），已停止下载。过一段时间再试，或告诉 Claude")
                if r.status_code == 200 and parse_table(r.text) is not None:
                    got = (url, r.text)
                    break
            path.parent.mkdir(parents=True, exist_ok=True)
            if got is None:
                # 记下“这页没有积分榜”，下次不再请求
                path.with_suffix(".missing").write_text("没找到带积分榜的页面", encoding="utf-8")
                print(f"{comp} {season}: 没找到")
                continue
            url, html = got
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(html)
            meta = {"url": url, "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
            path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            print(f"{comp} {season}: {url}")


def _num(x) -> float | None:
    """维基百科表格里的数字：负号是长的 −，有的带脚注 [a]。"""
    s = re.sub(r"\[.*?\]", "", str(x)).replace("−", "-").replace("–", "-").replace("+", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


NEED = ["Pos", "Team", "Pld", "W", "D", "L", "GF", "GA", "Pts"]


def parse_table(html: str) -> dict | None:
    """从页面里找最终积分榜。返回 {'table': 表格, 'rules': 排名规则那段文字, 'notes': 表格下面的注释}；找不到返回 None。

    认表格的办法：表头里有 名次、球队、场次、胜、平、负、进球、失球、积分，行数 10–24。
    """
    from bs4 import BeautifulSoup  # 解析网页（beautifulsoup4 + lxml，见 requirements.txt）

    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all("table"):
        heads = [h.get_text(" ", strip=True) for h in t.find_all("th", limit=14)]
        # 球队那一列有时叫 Club，有时表头带着“v t e”（维基百科模板的查看 / 讨论 / 编辑链接）
        heads = ["Team" if h.startswith(("Team", "Club")) else h for h in heads]
        if not all(any(h == n or h.startswith(n + " ") for h in heads) for n in NEED):
            continue
        try:
            df = pd.read_html(io.StringIO(str(t)))[0]
        except ValueError:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]
        df.columns = [re.sub(r"\[.*?\]", "", str(c)).strip() for c in df.columns]
        df = df.rename(columns={c: "Team" for c in df.columns if c.startswith(("Team", "Club"))})
        if not all(c in df.columns for c in NEED) or not 10 <= len(df) <= 24:
            continue
        out = pd.DataFrame({"pos": df["Pos"].map(_num), "team_wiki": df["Team"].astype(str)})
        for c, name in [("Pld", "played"), ("W", "won"), ("D", "drawn"), ("L", "lost"), ("GF", "gf"),
                        ("GA", "ga"), ("Pts", "points")]:
            out[name] = df[c].map(_num)
        # 最后一列“晋级或降级”（例如 Relegation to EFL Championship），用来核对降级名额
        note_cols = [c for c in df.columns if c.startswith(("Qualification", "Promotion", "Relegation"))]
        out["note"] = df[note_cols[0]].fillna("").astype(str).to_numpy() if note_cols else ""
        if out[["pos", "won", "drawn", "lost", "gf", "ga", "points"]].isna().any().any():
            continue
        # 表格后面的文字：Source …… Rules for classification …… Notes ……
        after = t.parent.get_text(" ", strip=True) if t.parent else ""
        i = after.find("Rules for classification")
        rules = after[i:i + 1500] if i >= 0 else ""
        j = after.find("Notes:")
        notes = after[j:j + 3000] if j >= 0 else ""
        return {"table": out, "rules": rules, "notes": notes}
    return None


def load_page(comp: str, season: int, settings: dict) -> dict | None:
    path = raw_path(comp, season, settings)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        parsed = parse_table(f.read())
    if parsed is not None:
        meta = path.with_suffix(".json")
        parsed["url"] = json.loads(meta.read_text(encoding="utf-8"))["url"] if meta.exists() else ""
    return parsed


def main() -> None:
    p = argparse.ArgumentParser(description="用维基百科核对积分榜")
    p.add_argument("--fetch", action="store_true", help="下载还没下载的页面")
    p.add_argument("--report", action="store_true", help="核对并写报告")
    p.add_argument("--comps", default=None, help="联赛代码，逗号分隔；不写 = 全部")
    p.add_argument("--seasons", default=None, help="赛季范围，例如 2005-2025；不写 = settings 里的范围")
    a = p.parse_args()
    settings = load_settings()
    cfg = settings["standings_check"]
    comps = a.comps.split(",") if a.comps else [c for g in settings["leagues"].values() for c in g]
    lo, hi = (a.seasons or f"{cfg['first_season']}-{cfg['last_season']}").split("-")
    seasons = list(range(int(lo), int(hi) + 1))
    if a.fetch:
        fetch_all(settings, comps, seasons)
    if a.report:
        from src.review.standings_report import write_report  # 核对报告（要用到积分榜计算）
        write_report(settings, comps, seasons)


if __name__ == "__main__":
    main()
