"""生成正式网站（静态页面）：data/site/ 里的首页、单场页、球队页、战绩页（需求文档第 8 节、第 9 节接口 4）。

运行：.venv\\Scripts\\python -m src.jobs.build_site
定时：任务计划程序“足球预测_生成网站”每 15 分钟跑一次（src/jobs/register_tasks.ps1）。双击根目录的“查看网站.bat”也会先生成一次再打开。

只读数据库、预测存档（predictions 表）、data/review/，不出预测、不跑模型、不联网。
每次全部重新生成：先写进一个临时文件夹，全部写完再换上去。这样生成到一半出错时，原来的网站还在；
浏览器正开着的页面也不会读到写了一半的文件。
每次运行写心跳文件 data/jobs/site.json（第 9 节接口 2），数据状态页、提醒程序看它知道网站多久没更新了。
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import load_settings, resolve_path
from src.features import open_readonly
from src.jobs.heartbeat import job_run
from src.web.site import data as site_data
from src.web.site import match, pages
from src.web.site.common import TeamFiles

TMP_NAME, OLD_NAME = "site_tmp_build", "site_old"   # 临时文件夹、换下来的旧网站，和 data/site 放在同一层


def _remove(path: Path, parent: Path) -> None:
    """删掉一个临时文件夹。只删 parent 下面、名字是上面两个之一的文件夹，防止路径写错时删到别的东西。"""
    if path.exists() and path.parent == parent and path.name in (TMP_NAME, OLD_NAME):
        shutil.rmtree(path)


def _swap(tmp: Path, out: Path) -> None:
    """把新生成的 tmp 换成 out：旧的先改名成 site_old，新的改名成 out，再删掉旧的。

    改名几乎是一瞬间的事。万一改名失败（例如 Windows 上有程序正打开着旧文件夹里的文件），
    就退一步：把新文件一个一个复制过去覆盖，再删掉新网站里已经没有的旧页面。
    """
    parent = out.parent
    old = parent / OLD_NAME
    _remove(old, parent)
    try:
        if out.exists():
            os.replace(out, old)
        os.replace(tmp, out)
    except OSError:
        if not out.exists() and old.exists():
            os.replace(old, out)               # 旧的已经挪开、新的没换上：先把旧的放回去
        shutil.copytree(tmp, out, dirs_exist_ok=True)
        keep = {p.relative_to(tmp) for p in tmp.rglob("*")}
        for p in sorted(out.rglob("*"), reverse=True):
            if p.relative_to(out) not in keep:
                p.unlink() if p.is_file() else p.rmdir()
        _remove(tmp, parent)
    _remove(old, parent)


def write_site(data: site_data.SiteData, folder: Path, final_dir: Path) -> dict:
    """把整站写进 folder。final_dir 是最后换上去的位置（算战绩页里周报的相对链接用）。返回每种页面几页。"""
    (folder / "match").mkdir(parents=True)
    (folder / "team").mkdir()
    teams = {data.matches[mid][k] for mid in data.page_ids for k in ("home_team", "away_team")}
    files = TeamFiles(teams)
    for mid in sorted(data.page_ids):
        (folder / "match" / f"{int(mid)}.html").write_text(match.render(data, mid, files), encoding="utf-8")
    for team in sorted(teams):
        (folder / "team" / files.file(team)).write_text(pages.render_team(data, team, files), encoding="utf-8")
    (folder / "index.html").write_text(pages.render_index(data, files), encoding="utf-8")
    (folder / "record.html").write_text(pages.render_record(data, final_dir), encoding="utf-8")
    return {"首页": 1, "单场页": len(data.page_ids), "球队页": len(teams), "战绩页": 1}


def build(settings: dict | None = None, now: datetime | None = None, db_path: Path | str | None = None,
          out_dir: Path | str | None = None) -> dict:
    """生成整站，返回摘要（几页、几场有预测、有没有复盘数据）。参数不给就按 settings.yaml，测试里给临时的。"""
    s = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    out = Path(out_dir) if out_dir else resolve_path(s["site"]["output_dir"])
    con = open_readonly(db_path or resolve_path(s["paths"]["database"]))
    try:
        data = site_data.load(con, s, now)
    finally:
        con.close()
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / TMP_NAME
    _remove(tmp, parent)                       # 上次中途出错留下的临时文件夹
    try:
        counts = write_site(data, tmp, out)
    except BaseException:
        _remove(tmp, parent)
        raise
    _swap(tmp, out)
    return {**counts, "页面合计": sum(counts.values()), "有预测的比赛": len(data.preds),
            "有复盘数据": data.scores is not None, "提示": len(data.notes)}


def main() -> dict:
    s = load_settings()
    with job_run("site") as run:
        summary = build(s)
        run.summary.update(summary)
    out = resolve_path(s["site"]["output_dir"])
    print(f"网站生成好了：首页 1 页、单场页 {summary['单场页']} 页、球队页 {summary['球队页']} 页、战绩页 1 页，"
          f"共 {summary['页面合计']} 页。")
    print(f"有预测的比赛 {summary['有预测的比赛']} 场；复盘数据：{'有' if summary['有复盘数据'] else '还没有'}。")
    if summary["提示"]:
        print(f"读数据时有 {summary['提示']} 条提示，写在首页最下面。")
    print(f"打开：{out / 'index.html'}")
    return summary


if __name__ == "__main__":
    main()
