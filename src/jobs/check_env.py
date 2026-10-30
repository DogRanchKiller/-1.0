"""第 0 阶段环境检查：一条命令检查环境是不是都准备好了。

运行方法（在项目文件夹里）：
    .venv\\Scripts\\python -m src.jobs.check_env

每一项打印“通过”或“失败 + 原因”，最后汇总。
"""
import importlib
import sqlite3
import sys
import tempfile
from pathlib import Path

import requests

from src.common.config import PROJECT_ROOT, load_settings, resolve_path

results = []  # 每项检查的结果：(名称, 是否通过, 说明)


def check(name, func):
    """运行一项检查，出错时记成失败，不让整个脚本停下。"""
    try:
        detail = func()
        results.append((name, True, detail or ""))
    except Exception as e:  # 检查脚本要把所有问题一次报完，所以这里接住所有错误
        results.append((name, False, f"{type(e).__name__}: {e}"))


def python_version():
    v = sys.version_info
    assert v >= (3, 12), f"需要 Python 3.12 以上，现在是 {v.major}.{v.minor}"
    assert ".venv" in sys.executable, f"没有用项目的虚拟环境运行，现在用的是 {sys.executable}"
    return f"{v.major}.{v.minor}.{v.micro}（虚拟环境）"


def libraries():
    names = ["pandas", "numpy", "scipy", "requests", "yaml", "pytest"]
    versions = [f"{n} {importlib.import_module(n).__version__}" for n in names]
    return "，".join(versions)


def settings_file():
    s = load_settings()
    return f"{len(s['leagues']['top'])} 个联赛，考试赛季 {s['seasons']['exam']}"


def folders():
    needed = ["config", "data/raw", "logs", "tests", "学习笔记", "src/collect", "src/features"]
    missing = [d for d in needed if not (PROJECT_ROOT / d).is_dir()]
    assert not missing, f"缺少文件夹：{missing}"
    return f"{len(needed)} 个关键文件夹都在"


def chinese_path():
    """项目路径有中文，检查数据库和文件读写是否正常（需求文档第 17 节的【待核实】）。"""
    data_dir = resolve_path(load_settings()["paths"]["raw"]).parent
    with tempfile.TemporaryDirectory(dir=data_dir, prefix="检查_") as tmp:
        db = Path(tmp) / "测试.db"
        con = sqlite3.connect(db)
        con.execute("create table t (队名 text)")
        con.execute("insert into t values ('曼城')")
        con.commit()
        value = con.execute("select 队名 from t").fetchone()[0]
        con.close()
        assert value == "曼城", "中文读写结果不对"
    return f"在 {data_dir} 下建数据库、写中文、读回都正常"


def network(url):
    def _check():
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (personal study)"})
        r.raise_for_status()
        return f"HTTP {r.status_code}，{r.elapsed.total_seconds():.1f} 秒"
    return _check


def main():
    check("Python 版本", python_version)
    check("依赖库", libraries)
    check("配置文件", settings_file)
    check("文件夹结构", folders)
    check("中文路径读写", chinese_path)
    # football-data、Understat 的 robots.txt 不允许 AI 程序访问，这里不检查它们（见需求文档 6.1）。
    # football-data 能不能下载，由下载程序 src.collect.football_data 运行时报告。
    check("访问 GitHub", network("https://github.com"))
    check("访问清华 pip 镜像", network("https://pypi.tuna.tsinghua.edu.cn/simple/pandas/"))

    for name, ok, detail in results:
        print(f"[{'通过' if ok else '失败'}] {name}：{detail}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n共 {len(results)} 项，失败 {failed} 项。" + ("环境准备好了。" if not failed else "请把上面的失败信息发给 Claude。"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
