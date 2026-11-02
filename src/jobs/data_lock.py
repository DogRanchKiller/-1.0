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
    database = resolve_path(settings["paths"]["database"]).resolve()
    path = database.with_name(database.name + suffix)
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


def source_refresh_lock(settings, source):
    """单来源从采集至导入独占；放在心跳外，重复手工启动不能覆盖在跑任务的心跳。"""
    if not isinstance(source, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", source):
        raise ValueError("来源锁名称只能使用字母、数字、下划线和短横线")
    return _file_lock(settings, f".refresh.{source}.lock")
