"""推送的定时任务入口（需求文档第 9 节“推送：已定的做法”）。

每次运行做两件事：
1. compose：找新出的 T-24 / T-6 / T-1 预测（合并成一条）、定时任务的故障和恢复，写进发件箱 data/notify/outbox/。
2. send：有口令就按频率上限推到微信（pushplus）；没设置口令时消息只留在发件箱，不算出错。
用 job_run("notify") 包起来，每次运行写心跳文件 data/jobs/notify.json（数据状态页读它）。
本轮真的尝试发送但失败时，先保留发送摘要和重试队列，再让任务失败，状态页不能显示“成功”。
没有口令、达到频率上限、正常清理过期消息不算发送失败。
心跳文件、屏幕上都只写条数和中文说明，不写口令。

运行：.venv\\Scripts\\python -m src.jobs.notify
由 Windows 定时任务“足球预测_推送”每 15 分钟自动运行（不弹黑窗口）。
"""
from datetime import datetime, timezone

from src.common.config import load_secrets, load_settings
from src.jobs.heartbeat import job_run
from src.notify.compose import compose
from src.notify.send import get_token, mask, send_outbox


def main(now: datetime | None = None, post=None) -> dict:
    """跑一次。now、post 给测试用（假的时间、假的发送）。返回摘要。"""
    settings = load_settings()
    secrets = load_secrets()
    token = get_token(secrets)
    now = now or datetime.now(timezone.utc)
    with job_run("notify") as run:
        c = compose(settings, now, secret=token)
        s = send_outbox(settings, secrets, now, post)
        run.summary.update({**c, **s})
        failed = int(s.get("失败待重试", 0)) + int(s.get("最终失败", 0))
        if failed:
            # send_outbox 已经保存尝试次数、队列和频率状态。这里只改变任务的成功/失败判断，
            # 不再尝试发送，也不删除待重试消息；异常让心跳记失败、命令行以非零状态退出。
            reason = mask(s.get("最近错误") or "发送服务没有确认成功", token)
            raise RuntimeError(f"本次推送有 {failed} 条发送失败：{reason}")
    summary = dict(run.summary)
    print("推送：" + "，".join(f"{k} {mask(v, token)}" for k, v in summary.items()))
    return summary


if __name__ == "__main__":
    main()
