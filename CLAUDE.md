# 项目说明（给 Claude 看）

足球比赛预测系统，用户的主要目的是学习数据科学和机器学习。用户完全不会编程：Claude 写代码、自己运行、把结果用中文讲清楚；用户看结果、提问。
不要让用户自己开终端运行命令，除非必须用户本人操作（例如登录）。

网站的 robots.txt 不允许 AI 程序访问时，先告诉用户，由用户决定；同时提供可以双击的 .bat 让用户自己运行。
- football-data.co.uk：robots.txt 禁止 AI 爬虫，但用户决定（2026-10-30）由 Claude 替用户运行下载程序 `src.collect.football_data`，
  保持低频、只下个人学习要用的文件。
- understat.com：禁止所有程序 → 不用。
- 新网站开工前先看 robots.txt 和使用条款，写进需求文档 6.1 / 6.2。

## 每次开工先做

1. 读 `进度记录.md` 最后几条，知道做到哪、下一步是什么。
2. 需要细节时读 `需求文档.md` 对应章节。需求文档是唯一依据。

## 运行

- 都在项目文件夹里运行，用虚拟环境的 Python：`.venv\Scripts\python`（Git Bash 里写 `.venv/Scripts/python`）。
- 程序都按模块方式运行：`.venv\Scripts\python -m src.jobs.check_env`。
- 自动检查：`.venv\Scripts\python -m pytest`。
- 装新库：先加进 `requirements.txt` 并写明用途，装好后 `pip freeze > requirements-lock.txt`。pip 已配置清华镜像（`.venv\pip.ini`）。
- 定时任务登记：`powershell -ExecutionPolicy Bypass -File src\jobs\register_tasks.ps1`（文件必须保存为带 BOM 的 UTF-8，否则中文任务名会乱码）。

## 约定

- 目录结构和数据分层按需求文档第 17 节。数据只往一个方向流：采集 → 原始层 → 清洗 → 数据库 → 特征 → 模型 → 出预测。
- 所有模型和回测都必须经过 `src/features` 取数据，只拿“截止时间”之前的数据；用途按数据字典检查（第 17 节）。
- 路径、数字都放 `config/settings.yaml`，不写死在代码里。密钥放 `config/secrets.yaml`（不进 Git）。
- 内部时间一律 UTC，显示用北京时间。
- 赛季用开始年份表示（2023 = 2023/24）；赛季用途用 `src.common.config.season_role` 判断。
- 代码注释用中文、写得详细，给不会编程的人看。
- 不写“推荐下注”；虚拟投注不连真实账户；爬虫低频，开工前看使用条款和 robots.txt。

## 改动流程

- 实际做法和需求文档不同时，当次改文档，在第 19 节修改记录记一行，版本号加 0.01。
- 每做完一段：更新 `进度记录.md`，跑 pytest，提交并推送到 GitHub。
- GitHub 仓库是公开的（DogRanchKiller/-1.0）：数据、数据库、日志、密钥都不能提交（见 `.gitignore`）。
