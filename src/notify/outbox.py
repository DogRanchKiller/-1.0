"""推送消息的“发件箱”：一条消息一个 JSON 文件（需求文档第 9 节接口 5）。

为什么用文件、不直接发：
- 发送可能失败（没网、对方限流、还没设置口令）。先写成文件，发不出去时消息还在，下次接着发。
- 你可以直接打开文件夹看“系统想告诉我什么”，就算还没设置微信推送也能看。

三个文件夹（都在 settings.yaml 的 notify.outbox_dir 旁边）：
- outbox/：还没发出去的。
- sent/：已经发出去的（文件里多了“发送时间”“尝试次数”）。
- failed/：最终没发出去的（多了“失败原因”“尝试次数”）。

一条消息的内容：
    {"id": "20261101T100000Z_预测_1a2b3c4d5e6f", "种类": "预测" 或 "故障",
     "标题": "...", "正文": "...（Markdown）", "创建时间": "2026-11-01 10:00:00"（UTC）,
     "关键字": "用来判断是不是同一条", "指纹": "种类 + 关键字 的哈希", "尝试次数": 0, "上次错误": null}

同样内容不重复生成：种类 + 关键字 算一个哈希（指纹），outbox / sent / failed 里已经有这个指纹的文件就不再写。
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import resolve_path

TIME_FMT = "%Y-%m-%d %H:%M:%S"
KINDS = ("预测", "故障")


def now_text(now: datetime | None = None) -> str:
    """UTC 时间写成文本（和心跳文件同一个格式）。"""
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(TIME_FMT)


def parse_time(text: str) -> datetime:
    """把上面的文本读回成带时区（UTC）的时间。"""
    return datetime.strptime(text, TIME_FMT).replace(tzinfo=timezone.utc)


class Outbox:
    """发件箱的三个文件夹。folder 是 outbox/ 本身；sent/、failed/ 在它旁边。"""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.sent = self.folder.parent / "sent"
        self.failed = self.folder.parent / "failed"

    @classmethod
    def from_settings(cls, settings: dict) -> "Outbox":
        return cls(resolve_path(settings["notify"]["outbox_dir"]))

    # ---------- 写 ----------
    @staticmethod
    def fingerprint(kind: str, key: str) -> str:
        """种类 + 关键字 的哈希，取前 12 位（够区分，文件名也不会太长）。"""
        return hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:12]

    def exists(self, fp: str) -> bool:
        """三个文件夹里有没有这个指纹的消息（文件名以 _指纹.json 结尾）。"""
        return any(d.exists() and any(d.glob(f"*_{fp}.json")) for d in (self.folder, self.sent, self.failed))

    def add(self, kind: str, title: str, content: str, key: str, now: datetime | None = None,
            *, expires_at: datetime | None = None) -> dict | None:
        """写一条新消息。同样的（种类 + 关键字）已经有了就不写，返回 None；写了返回这条消息。"""
        if kind not in KINDS:
            raise ValueError(f"消息种类只能是 {KINDS}，收到 {kind!r}")
        fp = self.fingerprint(kind, key)
        if self.exists(fp):
            return None
        created = now_text(now)
        msg = {"id": f"{created.replace('-', '').replace(':', '').replace(' ', 'T')}Z_{kind}_{fp}",
               "种类": kind, "标题": title, "正文": content, "创建时间": created,
               "关键字": key, "指纹": fp, "尝试次数": 0, "上次错误": None}
        if expires_at is not None:
            # 预测按这条消息里最早的开球时间失效；重试发送时也必须检查，不能只看生成后 24 小时。
            msg["过期时间"] = now_text(expires_at)
        self._write(self.folder / f"{msg['id']}.json", msg)
        return msg

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        """先写临时文件再改名：写到一半断电，也不会留下半个文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    # ---------- 读 ----------
    def pending(self) -> list[tuple[Path, dict]]:
        """还没发出去的消息，按创建时间从早到晚；同一时间的按标题排（拆成几条的“（1/3）”“（2/3）”按顺序发）。
        读不了的文件跳过。"""
        out = []
        if self.folder.exists():
            for p in self.folder.glob("*.json"):
                try:
                    out.append((p, json.loads(p.read_text(encoding="utf-8"))))
                except (OSError, ValueError):
                    continue
        out.sort(key=lambda x: (str(x[1].get("创建时间", "")), str(x[1].get("标题", "")), x[0].name))
        return out

    # ---------- 发完以后 ----------
    def update(self, path: Path, msg: dict) -> None:
        """改写 outbox 里这条消息（例如尝试次数加 1）。"""
        self._write(path, msg)

    def mark_sent(self, path: Path, msg: dict, sent_at: str) -> Path:
        """发出去了：记下发送时间，挪到 sent/。"""
        return self._move(path, {**msg, "发送时间": sent_at}, self.sent)

    def mark_failed(self, path: Path, msg: dict, reason: str, at: str) -> Path:
        """最终没发出去：记下原因和时间，挪到 failed/。"""
        return self._move(path, {**msg, "失败原因": reason, "失败时间": at}, self.failed)

    def _move(self, path: Path, msg: dict, folder: Path) -> Path:
        target = folder / path.name
        self._write(target, msg)       # 先写到新地方，再删旧的：中途断电最多留两份，不会一份都没有
        path.unlink(missing_ok=True)
        return target
