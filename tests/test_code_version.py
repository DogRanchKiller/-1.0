"""Git 元数据读取失败不能阻断预测，中文文件名在无窗口 Python 下也要能读。"""
import subprocess
import sys

import pytest

from src.predict.preview import code_version


@pytest.fixture
def fake_git(monkeypatch):
    """保留真实管道解码和退出码检查，用 Python 子进程代替 Git 输出。"""
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "_text_encoding", lambda: "gbk")

    def install(status=b"", failed_command=None):
        def run(args, **kwargs):
            output = b"abc1234\n" if args[1] == "rev-parse" else status
            exit_code = 128 if args[1] == failed_command else 0
            script = f"import sys; sys.stdout.buffer.write({output!r}); sys.exit({exit_code})"
            return real_run([sys.executable, "-c", script], **kwargs)

        monkeypatch.setattr(subprocess, "run", run)

    return install


@pytest.mark.parametrize("status,expected", [
    (b"", "abc1234"),
    (" M 需.md\n".encode("utf-8"), "abc1234+改动"),
    (b" M bad-\xff.md\n", "abc1234+改动"),
])
def test_code_version_reads_git_output_under_gbk_default(fake_git, status, expected):
    """默认 GBK 不能妨碍 UTF-8 中文或非标准文件名字节的脏状态判断。"""
    fake_git(status)
    assert code_version() == expected


@pytest.mark.parametrize("failed_command", ["rev-parse", "status"])
def test_code_version_failed_git_is_unknown(fake_git, failed_command):
    """任一 Git 命令失败都不能把版本标成干净或仍可确认的提交。"""
    fake_git(failed_command=failed_command)
    assert code_version() is None


@pytest.mark.parametrize("missing_command", ["rev-parse", "status"])
def test_code_version_missing_stdout_is_unknown(monkeypatch, missing_command):
    """保护后台管道没有文本的情况，避免 None.strip() 把预测入口打断。"""
    def run(args, **kwargs):
        output = None if args[1] == missing_command else ("abc1234\n" if args[1] == "rev-parse" else "")
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert code_version() is None


def test_code_version_empty_head_is_unknown(monkeypatch):
    def run(args, **kwargs):
        output = "\n" if args[1] == "rev-parse" else " M 需.md\n"
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert code_version() is None


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 10)])
def test_code_version_unavailable_git_is_unknown(monkeypatch, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    assert code_version() is None
