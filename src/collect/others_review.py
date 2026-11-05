r"""调研别人的足球预测项目：下载候选论文、GitHub 项目的原始资料（需求文档第 5 节“别人的项目”、6.1“调研别人的项目用的网站”）。

只下载，不打分；打分是人（Claude）读完以后写进 调研/别人的项目调研.md。

三个来源（2026-11-05 看过 robots.txt）：
  1. OpenAlex 接口（api.openalex.org，robots.txt 允许所有程序，数据 CC0）：按检索词找论文，按被引次数排。
  2. GitHub 接口（api.github.com）：GitHub 的 robots.txt 禁止网页搜索 /search，让程序用接口；不登录每小时 60 次（搜索每分钟 10 次）。
     按检索词找项目（按星数排），再读每个项目的 README。
  3. arXiv（只读 /abs/编号 页，robots.txt 允许，要求每 15 秒最多 1 次）：给 OpenAlex 里有 arXiv 编号的论文补全文。

保存：data/raw/others_review/ 下（不进 Git），原样保存接口返回的内容；已经存在的文件不再下载（中途停了下次接着下）。
不写数据库。

运行方法（Claude 运行）：
  .venv\Scripts\python -m src.collect.others_review search        第一步：论文、项目的候选清单（只下一个来源：search-openalex / search-github）
  .venv\Scripts\python -m src.collect.others_review readme 甲/乙 …  第二步：读选中项目的 README（“用户名/项目名”）
  .venv\Scripts\python -m src.collect.others_review arxiv 2101.00001 …  第二步：读选中论文的 arXiv 页面
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

from src.common.config import load_settings, resolve_path

_last: dict[str, float] = {}   # 每个网站上一次请求的时间，用来保证间隔


def _get(url: str, site: str, cfg: dict, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    """按网站的最小间隔发一次 GET；失败（连不上、5xx、429）按 retry_waits 等了再试，都失败就报错停下。"""
    wait = cfg["min_interval_seconds"][site] - (time.time() - _last.get(site, 0))
    if wait > 0:
        time.sleep(wait)
    h = {"User-Agent": cfg["user_agent"], **(headers or {})}
    ask = 0   # 上一次 429 时网站在 Retry-After 里要求等的秒数
    for extra in [0, *cfg["retry_waits"]]:
        time.sleep(max(extra, ask))
        _last[site] = time.time()
        try:
            r = requests.get(url, params=params, headers=h, timeout=60)
        except requests.RequestException as e:
            print(f"  连不上 {url}：{e}")
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After", "")
            ask = int(ra) + 5 if ra.isdigit() else 0   # 按网站要求的多等 5 秒
            print(f"  {url} 返回 {r.status_code}，等一会再试（网站要求等 {ra or '?'} 秒）")
            continue
        return r
    raise RuntimeError(f"{url} 多次失败，停下")


def _slug(s: str) -> str:
    """文件名里不能有的字符换成下划线。"""
    return re.sub(r"[^\w.-]+", "_", s).strip("_")


def abstract(inv: dict | None) -> str:
    """OpenAlex 的摘要是“词 → 出现位置”的倒排表，按位置拼回原文。没有摘要返回空字符串。"""
    if not inv:
        return ""
    pos = {i: w for w, idx in inv.items() for i in idx}
    return " ".join(pos[i] for i in sorted(pos))


def search(cfg: dict, out: Path, sources=("openalex", "github")) -> None:
    """论文、项目的候选清单：每个检索词一个 json（原样），再汇总成 candidates.json（只留打分要用的几项）。
    sources 只下其中一个来源时，另一个来源已经下好的文件照样汇总进来。"""
    out.mkdir(parents=True, exist_ok=True)
    papers, repos = {}, {}
    for q in cfg["openalex_queries"]:
        if "openalex" not in sources and not (out / f"openalex_{_slug(q)}.json").exists():
            continue
        f = out / f"openalex_{_slug(q)}.json"
        if not f.exists():
            # 只在题目和摘要里找（全文找再按被引排，前面全是不相干的高被引论文，2026-11-05 改）
            r = _get(cfg["openalex_url"], "openalex", cfg, params={
                "filter": f"type:article,title_and_abstract.search:{q}", "sort": "cited_by_count:desc",
                "per-page": cfg["openalex_per_query"]})
            r.raise_for_status()
            f.write_text(r.text, encoding="utf-8")
        for w in json.loads(f.read_text(encoding="utf-8"))["results"]:
            ids = w.get("ids") or {}
            loc = (w.get("primary_location") or {}).get("source") or {}
            papers.setdefault(w["id"], {
                "检索词": q, "题目": w.get("title"), "年份": w.get("publication_year"), "被引": w.get("cited_by_count"),
                "期刊": loc.get("display_name"), "doi": w.get("doi"), "arxiv": _arxiv_id(w),
                "开放全文": (w.get("open_access") or {}).get("oa_url"), "摘要": abstract(w.get("abstract_inverted_index")),
                "openalex": ids.get("openalex")})
        print(f"OpenAlex「{q}」：累计 {len(papers)} 篇")
    for q in cfg["github_queries"]:
        if "github" not in sources and not (out / f"github_{_slug(q)}.json").exists():
            continue
        f = out / f"github_{_slug(q)}.json"
        if not f.exists():
            r = _get(cfg["github_search_url"], "github", cfg, params={
                "q": q, "sort": "stars", "order": "desc", "per_page": cfg["github_per_query"]},
                headers={"Accept": "application/vnd.github+json"})
            r.raise_for_status()
            f.write_text(r.text, encoding="utf-8")
        for it in json.loads(f.read_text(encoding="utf-8"))["items"]:
            repos.setdefault(it["full_name"], {
                "检索词": q, "项目": it["full_name"], "星数": it.get("stargazers_count"), "说明": it.get("description"),
                "语言": it.get("language"), "最后更新": it.get("pushed_at"), "链接": it.get("html_url")})
        print(f"GitHub「{q}」：累计 {len(repos)} 个")
    res = {"论文": sorted(papers.values(), key=lambda x: -(x["被引"] or 0)),
           "项目": sorted(repos.values(), key=lambda x: -(x["星数"] or 0))}
    (out / "candidates.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"候选清单：论文 {len(papers)} 篇、项目 {len(repos)} 个 → {out / 'candidates.json'}")


def _arxiv_id(w: dict) -> str | None:
    """OpenAlex 记录里找 arXiv 编号（各个收录位置的链接里有 arxiv.org/abs/编号 的）。"""
    for loc in w.get("locations") or []:
        for k in ("landing_page_url", "pdf_url"):
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", (loc or {}).get(k) or "")
            if m:
                return m.group(1)
    return None


def readme(cfg: dict, out: Path, names: list[str]) -> None:
    """读选中项目的 README（接口直接给原文），存 readme/用户名__项目名.md。"""
    d = out / "readme"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        f = d / f"{_slug(n.replace('/', '__'))}.md"
        if f.exists():
            continue
        r = _get(f"{cfg['github_repo_url']}/{n}/readme", "github", cfg, headers={"Accept": "application/vnd.github.raw"})
        if r.status_code == 404:
            print(f"  {n}：没有 README")
            f.write_text("", encoding="utf-8")
            continue
        r.raise_for_status()
        f.write_text(r.text, encoding="utf-8")
        print(f"  {n}：{len(r.text):,} 个字符")


def arxiv(cfg: dict, out: Path, ids: list[str]) -> None:
    """读选中论文的 arXiv 页面：先 /html/编号（新论文有网页版全文），没有再 /abs/编号（只有摘要）。"""
    d = out / "arxiv"
    d.mkdir(parents=True, exist_ok=True)
    for i in ids:
        f = d / f"{_slug(i)}.html"
        if f.exists():
            continue
        r = _get(f"{cfg['arxiv_url']}/html/{i}", "arxiv", cfg)
        kind = "全文"
        if r.status_code != 200:
            r = _get(f"{cfg['arxiv_url']}/abs/{i}", "arxiv", cfg)
            kind = "摘要页"
        r.raise_for_status()
        f.write_text(r.text, encoding="utf-8")
        print(f"  {i}：{kind}，{len(r.text):,} 个字符")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["search", "search-openalex", "search-github", "readme", "arxiv"])
    ap.add_argument("names", nargs="*", help="readme：用户名/项目名；arxiv：论文编号")
    a = ap.parse_args()
    s = load_settings()
    cfg = s["others_review"]
    out = resolve_path(s["paths"]["raw"]) / cfg["raw_subdir"]
    {"search": lambda: search(cfg, out), "search-openalex": lambda: search(cfg, out, ("openalex",)),
     "search-github": lambda: search(cfg, out, ("github",)), "readme": lambda: readme(cfg, out, a.names),
     "arxiv": lambda: arxiv(cfg, out, a.names)}[a.what]()


if __name__ == "__main__":
    main()
