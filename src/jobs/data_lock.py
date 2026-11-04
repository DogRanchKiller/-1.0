"""数据更新共用的进程锁：持有进程退出后由操作系统释放，不按文件年龄抢锁。"""
from contextlib import contextmanager
import errno
import math
import os
import re
import time

from src.common.config import load_settings, resolve_path


def _try_lock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_path(settings, suffix):
    """锁文件在哪：数据库文件旁边，名字 = 数据库文件名 + suffix。拿锁和下面的“看一眼”都用这一个，保证是同一个文件。"""
    database = resolve_path(settings["paths"]["database"]).resolve()
    return database.with_name(database.name + suffix)


@contextmanager
def _file_lock(settings, suffix):
    """同一数据库的指定锁互斥；等待超时抛错。

    文件保留在数据库旁，仅作为锁的载体。不能删除它或按mtime判断残留，
    否则两个进程可能锁到不同的文件。不同配置只要指向同一库，就共享同一把锁。
    """
    cfg = settings.get("data_refresh") or load_settings()["data_refresh"]
    timeout = float(cfg["lock_timeout_seconds"])
    poll = float(cfg["lock_poll_seconds"])
    if not math.isfinite(timeout) or timeout < 0 or not math.isfinite(poll) or poll <= 0:
        raise ValueError("数据更新锁的等待时间须非负，轮询间隔须为正有限数")
    path = _lock_path(settings, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    # a+b不会截断已被其他进程使用的锁文件；Windows允许锁定文件末尾之后的字节。
    with path.open("a+b") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_lock(handle)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("数据更新锁仍被占用；本次未执行锁内操作") from exc
                time.sleep(min(poll, remaining))
        try:
            yield
        finally:
            _unlock(handle)


def data_write_lock(settings):
    """只在写入比赛/来源事务时共享；网络等待不能阻塞其他来源的采集。"""
    return _file_lock(settings, ".refresh.lock")


def _source_suffix(source):
    if not isinstance(source, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", source):
        raise ValueError("来源锁名称只能使用字母、数字、下划线和短横线")
    return f".refresh.{source}.lock"


def source_refresh_lock(settings, source):
    """单来源从采集至导入独占；放在心跳外，重复手工启动不能覆盖在跑任务的心跳。"""
    return _file_lock(settings, _source_suffix(source))


def lock_busy(settings, source=None):
    """不等待地看一眼：这把锁现在是不是被别的程序拿着（网站上的操作按钮用，需求文档第 8 节“网站上的操作按钮”）。

    source 为 None 看导入用的那把（data_write_lock），否则看这个来源的那把（source_refresh_lock）。
    做法：马上试着拿一下，拿到了就立刻放掉、回 False（没人用）；拿不到回 True（有人在用）。
    锁文件还不存在时（这个来源从没跑过）当作没人用，不为了看一眼去新建文件。
    只是“看一眼”：看完到真的去拿之间，别的程序还可能抢先，所以真正干活的程序照样自己拿锁。
    """
    path = _lock_path(settings, ".refresh.lock" if source is None else _source_suffix(source))
    try:
        handle = path.open("r+b")
    except FileNotFoundError:
        return False
    with handle:
        try:
            _try_lock(handle)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            return True
        _unlock(handle)
        return False
