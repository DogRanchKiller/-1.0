"""发送层：把发件箱里的消息真的推到微信（pushplus 推送加，微信公众号渠道；需求文档第 9 节“推送：已定的做法”）。

以后换推送渠道，只换这个文件。

pushplus 的规则（官网写的，2026-11-01 记下）：
- 接口：POST https://www.pushplus.plus/send，JSON 里放 token、title、content、template、channel。
  返回的 JSON 里 code 是 200 表示成功，别的 code 是失败，msg 是原因。
- 要先在官网实名认证才能发到微信；实名用户微信渠道每天 200 条；1 分钟最多 5 次；同样内容 1 小时最多 3 条；
  标题最多 100 字、正文最多 2 万字。超额后账号会被限制。
- 我们按保守值：每分钟 4 条、每天 150 条（settings.yaml 的 notify，里面写了理由）。

口令（token）保密：
- 只从 config/secrets.yaml 的 pushplus.token 读（不进 Git）。
- 不写进日志、心跳文件、屏幕输出、发件箱文件、异常说明：所有要显示或保存的文字都先经过 mask()，把口令换成 ***。
- 网络出错时的异常说明里可能带着请求内容，也一样先换掉再用。

没有口令时：消息只留在 outbox，不报错，摘要写“还没设置微信推送”。
有口令时：每次运行按顺序发，到了每分钟或每天的上限就停，剩下的下次运行再发；
一条失败了下次运行再试，最多再试 max_retries 次，还不行挪到 failed/（写明原因和尝试次数）。
"""
import sys
from datetime import datetime, timedelta, timezone

import requests

from src.clean.dongqiudi_reader import BEIJING
from src.common.config import load_secrets, load_settings, resolve_path
from src.notify.compose import load_state, save_state
from src.notify.outbox import Outbox, now_text, parse_time

NO_TOKEN = "还没设置微信推送"
# pushplus 返回码的中文提示。官网的完整码表没有核实过，只写常见的、并注明待核实；别的码照抄对方的 msg
CODE_HINTS = {
    905: "可能还没在 pushplus 官网实名认证（实名后才能发到微信）【待核实】",
}


def get_token(secrets: dict | None) -> str | None:
    """secrets.yaml 里的 pushplus.token；没配或是空的返回 None。"""
    pp = (secrets or {}).get("pushplus")
    token = pp.get("token") if isinstance(pp, dict) else None
    token = str(token).strip() if token is not None else ""
    return token or None


def mask(text, token: str | None) -> str:
    """把文字里的口令换成 ***（显示、保存任何文字之前都先过一遍）。"""
    text = "" if text is None else str(text)
    return text.replace(token, "***") if token else text


def code_hint(code, msg) -> str:
    """失败时给人看的中文说明。"""
    hint = CODE_HINTS.get(code) if isinstance(code, int) else None
    base = f"pushplus 返回 code {code}" + (f"，说明：{msg}" if msg else "")
    return base + (f"。{hint}" if hint else "。这个返回码的意思还没核实，看上面对方写的说明")


def post_message(token: str, title: str, content: str, cfg: dict, post=None) -> dict:
    """发一条。返回 {"ok": True/False, "reason": 中文原因（已去掉口令）, "network": 是不是网络问题, "code": 返回码}。

    post 默认是 requests.post；测试时换成假的，不联网。
    """
    post = post or requests.post
    body = {"token": token, "title": title, "content": content,
            "template": cfg["template"], "channel": cfg["channel"]}
    try:
        resp = post(cfg["api_url"], json=body, timeout=float(cfg["request_timeout_seconds"]))
    except requests.RequestException as e:
        return {"ok": False, "network": True, "code": None,
                "reason": "网络不通或对方没回应：" + mask(f"{type(e).__name__}: {e}", token)[:300]}
    except Exception as e:      # 其他意外：说明里也可能带着请求内容，同样去掉口令
        return {"ok": False, "network": True, "code": None,
                "reason": "发送时出错：" + mask(f"{type(e).__name__}: {e}", token)[:300]}
    status = getattr(resp, "status_code", None)
    try:
        data = resp.json()
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {"ok": False, "network": False, "code": None,
                "reason": f"pushplus 回的内容看不懂（HTTP {status}）"}
    code, msg = data.get("code"), mask(data.get("msg"), token)[:200]
    if code == 200:
        return {"ok": True, "network": False, "code": 200, "reason": None}
    return {"ok": False, "network": False, "code": code, "reason": code_hint(code, msg)}


# ======================= 频率限制 =======================
def _records(state: dict, now: datetime) -> list[datetime]:
    """最近发过的时间（发送记录）；两天以前的删掉。"""
    keep = now - timedelta(days=2)
    out = []
    for t in state.get("发送记录", []):
        try:
            x = parse_time(t)
        except (TypeError, ValueError):
            continue
        if x >= keep:
            out.append(x)
    state["发送记录"] = [now_text(x) for x in out]
    return out


def quota(state: dict, now: datetime, cfg: dict) -> str | None:
    """现在还能不能发；能发返回 None，不能发返回原因。每次真的请求都算一次（成功失败都算）。"""
    recs = _records(state, now)
    day0 = now.astimezone(BEIJING).replace(hour=0, minute=0, second=0, microsecond=0)
    if sum(t >= day0 for t in recs) >= int(cfg["max_per_day"]):
        return f"今天（北京时间）已经发了 {cfg['max_per_day']} 条，到了每天的上限，剩下的明天再发"
    if sum(t > now - timedelta(seconds=60) for t in recs) >= int(cfg["max_per_minute"]):
        return f"1 分钟内已经发了 {cfg['max_per_minute']} 条，剩下的下次运行再发"
    return None


def _record(state: dict, now: datetime) -> None:
    state.setdefault("发送记录", []).append(now_text(now))


# ======================= 发送发件箱 =======================
def send_outbox(settings: dict, secrets: dict | None, now: datetime | None = None, post=None) -> dict:
    """把 outbox 里的消息按顺序发出去。返回摘要（中文键，里面不会有口令）。"""
    cfg = settings["notify"]
    now = now or datetime.now(timezone.utc)
    box = Outbox.from_settings(settings)
    token = get_token(secrets)
    pending = box.pending()
    summary = {"待发": len(pending), "已发": 0, "失败待重试": 0, "最终失败": 0, "过期": 0}
    if not token:
        summary["推送"] = NO_TOKEN + f"（消息留在发件箱 {box.folder.name}/，设置后会接着发）"
        return summary
    state_path = resolve_path(cfg["state_file"])
    state = load_state(state_path)
    expire = timedelta(hours=float(cfg["expire_hours"]))
    stop = None
    try:
        for path, msg in pending:
            try:
                created = parse_time(msg["创建时间"])
            except (KeyError, TypeError, ValueError):
                created = now
            if now - created > expire:
                box.mark_failed(path, msg, f"过期：生成超过 {cfg['expire_hours']} 小时还没发出去", now_text(now))
                summary["过期"] += 1
                continue
            stop = quota(state, now, cfg)
            if stop:
                break
            title = str(msg.get("标题", ""))[:int(cfg["max_title_chars"])]
            content = str(msg.get("正文", ""))[:int(cfg["max_content_chars"])]
            res = post_message(token, title, content, cfg, post)
            _record(state, now)
            msg["尝试次数"] = int(msg.get("尝试次数") or 0) + 1
            if res["ok"]:
                msg["上次错误"] = None
                box.mark_sent(path, msg, now_text(now))
                summary["已发"] += 1
                continue
            msg["上次错误"] = res["reason"]
            if msg["尝试次数"] > int(cfg["max_retries"]):
                box.mark_failed(path, msg, f"试了 {msg['尝试次数']} 次都没发出去，最后一次：{res['reason']}",
                                now_text(now))
                summary["最终失败"] += 1
            else:
                box.update(path, msg)
                summary["失败待重试"] += 1
            summary["最近错误"] = res["reason"]
            if res["network"]:
                stop = "网络不通，剩下的下次运行再发"      # 网络不通时后面的也发不出去，不白白用掉重试次数
                break
    finally:
        save_state(state_path, state)
    if stop:
        summary["停下原因"] = stop
    summary["还在发件箱"] = len(box.pending())
    summary["推送"] = "已设置微信推送"
    return {k: mask(v, token) if isinstance(v, str) else v for k, v in summary.items()}


# ======================= 发一条测试推送 =======================
def send_test(settings: dict, secrets: dict | None, now: datetime | None = None, post=None) -> tuple[bool, str]:
    """发“足球预测系统：测试消息”。返回 (成功没有, 中文说明)。也算进每分钟、每天的条数。"""
    cfg = settings["notify"]
    now = now or datetime.now(timezone.utc)
    token = get_token(secrets)
    if not token:
        return False, "还没设置口令：先双击“设置微信推送.bat”，把 pushplus 网站上的 token 粘贴进去"
    state_path = resolve_path(cfg["state_file"])
    state = load_state(state_path)
    stop = quota(state, now, cfg)
    if stop:
        save_state(state_path, state)
        return False, "这次没发：" + stop
    t = now.astimezone(BEIJING)
    res = post_message(token, "足球预测系统：测试消息",
                       f"这是一条测试消息，发送时间 {t:%Y-%m-%d %H:%M}（北京时间）。收到说明微信推送设置好了。", cfg, post)
    _record(state, now)
    save_state(state_path, state)
    if res["ok"]:
        return True, "发送成功：请看微信里“pushplus 推送加”公众号有没有收到“足球预测系统：测试消息”"
    reason = mask(res["reason"], token)
    if res["network"]:
        return False, "没发出去，网络不通：看看电脑能不能上网。详细：" + reason
    return False, "没发出去：" + reason + "。如果还没实名认证，先到 pushplus 官网实名，再试一次"


def main(argv: list[str] | None = None) -> None:
    """命令行：python -m src.notify.send test → 发一条测试推送（“发一条测试推送.bat”用）。"""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["test"]:
        print("用法：python -m src.notify.send test   （发一条测试推送）")
        return
    ok, text = send_test(load_settings(), load_secrets())
    print(("【成功】" if ok else "【没成功】") + text)


if __name__ == "__main__":
    main()
