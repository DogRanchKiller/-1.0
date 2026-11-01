"""定时出 T-24 / T-6 / T-1 预测（需求文档第 9 节“完整链路的分工和接口”接口 1、第 9 节“定时出 T-24 / T-6 / T-1 预测”）。

做什么（每 15 分钟由 src/jobs/scheduled_predict.py 调用一次）：
1. 通过取数层拿“现在”起、最长时点（24 小时）以内开球的五大联赛比赛。
2. 每场比赛、每个时点（settings.yaml 的 schedule.horizons：24、6、1 小时）算“计划时间” = 开球时间 − 几小时。
   计划时间已经到了、这一版还没出过的，就是“到点了还没出”。规则：
   - 已经开球的不出（开球以后再出就是偷看了比赛过程）。
   - 只有日期、没有具体开球时间的比赛不出（没法算“开球前 24 小时”），摘要里记“开球时间未定”。
   - 同一场几个时点同时到点（例如电脑关机后补跑：开机时离开球只剩 3 小时，T-24 和 T-6 都过了），只出最晚的那个（T-6），
     前面的（T-24）不补，写进本次心跳摘要的“跳过的时点”。补出一份“T-24”其实是 3 小时前的信息，名不副实。
   - 同一场、同一版本已经存过，而且开球时间没变：不再出（每版只存一行，存了就不改）。
   - 开球时间变了（改期）：按新的开球时间重新判断，到点了就再出一行，detail 写“改期重出”。
3. 要出的比赛，用和预览页完全一样的一套函数算（src/predict/preview.py 的 compute_predictions：同一个模型、同样取数、
   同样融合、倾向、信心等级）。截止时间 = 实际运行的时间（不是计划时间）。
4. 存进 predictions 表，kind = "T-24" / "T-6" / "T-1"。detail 在预览页的键之外加：
   计划时点、晚了几分钟、模型比分矩阵、融合比分矩阵（11 × 11，6 位小数）、首发（"未含首发"）。

还没有的东西：首发。需求文档第 2 节说 T-1 要等首发公布再出；系统现在还没有实时首发来源，所以 T-1 在开球前 1 小时到点就出，
detail 写“未含首发”。首发来源定下来以后再改成“开球前 60 分钟起反复检查，拿到首发或到开球前 30 分钟再出”。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import resolve_path
from src.common.markets import check_score_matrix
from src.features import FeatureStore
from src.features.known_time import to_utc_text
from src.predict.conclusion import FUSED, fuse
from src.predict.preview import archive_key, compute_predictions, insert_prediction, prediction_detail

NO_LINEUP = "未含首发"


def kind_of(hours: int) -> str:
    """时点 → 预测存档里的名字：24 → "T-24"。"""
    return f"T-{int(hours)}"


def parse_utc(text) -> datetime:
    """数据库里的时间文字（UTC，例如 '2026-11-01 14:30' 或 '2026-11-01 14:30:00'）→ 带时区的 datetime。"""
    ts = pd.Timestamp(text)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


# ======================= 哪些比赛到点了 =======================
@dataclass
class Plan:
    """一场比赛这次要出的那一版。"""
    match_id: int
    kickoff_utc: str
    hours: int                    # 计划时点：开球前几小时
    due_utc: str                  # 计划时间 = 开球 − hours
    late_minutes: float           # 比计划晚了几分钟
    skipped: list[int] = field(default_factory=list)   # 同时到点、没出的更早时点（例如 [24]）
    rescheduled_from: str | None = None                  # 改期重出时，上次存这一版时的开球时间


def plan_matches(targets: pd.DataFrame, stored: dict, horizons, now: datetime) -> tuple[list[Plan], dict]:
    """决定这次每场出哪一版。纯计算，不碰数据库（检查里直接拿假数据测）。

    targets：要看的比赛（至少有 match_id、kickoff_utc 两列）。
    stored：{(match_id, kind): 上次存这一版时的开球时间}（同一场同一版取 prediction_id 最大的那行）。
    horizons：时点（小时），例如 [24, 6, 1]。now：现在（带时区）。
    返回 (要出的, 计数)。计数：开球时间未定、已开球、还没到点、已出过（都只是数场次）。
    """
    hs = sorted({int(h) for h in horizons}, reverse=True)     # 从早到晚：24、6、1
    counts = {"开球时间未定": 0, "已开球": 0, "还没到点": 0, "已出过": 0}
    plans = []
    for t in targets.to_dict("records"):
        mid, ko_text = int(t["match_id"]), t.get("kickoff_utc")
        if ko_text is None or (isinstance(ko_text, float) and np.isnan(ko_text)):
            counts["开球时间未定"] += 1
            continue
        ko = parse_utc(ko_text)
        if ko <= now:
            counts["已开球"] += 1
            continue
        # 已经到点的时点里，最晚的那个（离开球最近）。hs 从早到晚排，最后一个到点的就是它
        due = [h for h in hs if ko - timedelta(hours=h) <= now]
        if not due:
            counts["还没到点"] += 1
            continue
        h = due[-1]
        last_ko = stored.get((mid, kind_of(h)))
        if last_ko is not None and parse_utc(last_ko) == ko:
            counts["已出过"] += 1
            continue
        # 更早到点、却没按这次的开球时间出过的时点：不补，记下来
        skipped = [x for x in due[:-1]
                   if stored.get((mid, kind_of(x))) is None or parse_utc(stored[(mid, kind_of(x))]) != ko]
        due_at = ko - timedelta(hours=h)
        plans.append(Plan(match_id=mid, kickoff_utc=str(ko_text), hours=h, due_utc=to_utc_text(due_at),
                          late_minutes=round((now - due_at).total_seconds() / 60, 1), skipped=skipped,
                          rescheduled_from=None if last_ko is None else str(last_ko)))
    return plans, counts


def stored_kickoffs(con, match_ids, kinds) -> dict:
    """预测存档里每场每个定时版本最后一次存的开球时间：{(match_id, kind): kickoff_utc}。

    predictions 是我们自己写的预测存档（不是比赛数据），直接读，不经过取数层（和预览页判断“和上次一样”一样）。
    """
    ids, kinds = [int(i) for i in match_ids], list(kinds)
    if not ids:
        return {}
    q1, q2 = ",".join("?" * len(ids)), ",".join("?" * len(kinds))
    sql = (f"select match_id, kind, kickoff_utc from predictions where prediction_id in ("
           f" select max(prediction_id) from predictions where match_id in ({q1}) and kind in ({q2})"
           f" group by match_id, kind)")
    return {(int(m), k): ko for m, k, ko in con.execute(sql, [*ids, *kinds])}


# ======================= 比分矩阵 =======================
def fold_matrix(m, n: int) -> np.ndarray:
    """比分矩阵换成 (n + 1) × (n + 1)：超过 n 球的并进最后一行 / 列（“n 球及以上”），不够大的补 0。"""
    m = np.asarray(m, dtype=float)
    out = np.zeros((n + 1, n + 1))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            out[min(i, n), min(j, n)] += m[i, j]
    return out


def matrix_list(m, n: int, decimals: int) -> list[list[float]] | None:
    """比分矩阵 → 存档用的二维列表（(n + 1) × (n + 1)，decimals 位小数）。没有矩阵是 None。

    舍入以后加起来和 1 会差一点点（121 格每格最多差 0.0000005，最多差约 0.00006），读的程序不用再归一。
    """
    if m is None or np.asarray(m).ndim != 2:
        return None
    return [[round(float(x), decimals) for x in row] for row in fold_matrix(check_score_matrix(m), n)]


# ======================= 出预测、存档 =======================
def run_scheduled(fs: FeatureStore, con, settings: dict, now: datetime, model=None, code: str | None = None,
                  created: datetime | None = None) -> dict:
    """站在 now 这个时间点，把到点了还没出的 T-24 / T-6 / T-1 出掉、存进 predictions 表。返回摘要（写进心跳文件）。

    fs：取数层（实时模式）；con：能写的数据库连接（存档）；now：现在（--now 可以指定，测试用）。
    model：不给就按 preview.model 新造（和预览页同一个；测试里换成假模型）。code：代码版本（存进 detail）。
    created：这一行真正写进存档的时间，不给就是电脑的当前时间（--now 指定过去的时间时，cutoff 是 now，写入时间仍是真实时间）。
    没有要出的比赛时不拟合模型（拟合一次约 1 分钟），直接返回。
    """
    cfg = settings["schedule"]
    hs = sorted({int(h) for h in cfg["horizons"]}, reverse=True)
    if settings["preview"]["days_ahead"] * 24 < hs[0]:
        raise ValueError("settings.yaml 的 preview.days_ahead 比最早的时点还短，定时预测拿不到要出的比赛")
    cutoff = to_utc_text(now)
    now = parse_utc(cutoff)                      # 秒以下舍掉，和截止时间一致
    leagues = list(settings["leagues"][settings["preview"]["leagues"]])
    # 最早时点那么久以内开球的比赛（多看 1 秒：正好在 24 小时整开球的也算到点）
    end = to_utc_text(now + timedelta(hours=hs[0], seconds=1))
    targets = fs.prediction_targets(cutoff, end, "A版输入", competition=leagues)
    kinds = [kind_of(h) for h in hs]
    plans, counts = plan_matches(targets, stored_kickoffs(con, targets["match_id"], kinds), hs, now)
    summary = {"现在": cutoff, "窗口内比赛": int(len(targets)), "新存": 0, **{k: 0 for k in kinds},
               "改期重出": 0, "跳过的时点": [], "模型没给概率": 0, **counts, "最晚了几分钟": None, "新存的编号": []}
    for p in plans:
        if p.skipped:
            summary["跳过的时点"].append({"match_id": p.match_id, "跳过": [kind_of(x) for x in p.skipped],
                                         "这次出": kind_of(p.hours)})
    if not plans:
        return summary

    # 和预览页完全同一套计算（同一个模型、取数、融合、倾向、信心等级）；截止时间 = 现在
    result, preds = compute_predictions(fs, settings, now, model)
    rows = {r["match_id"]: r for r in result["rows"]}
    n, dec = int(settings["market_odds"]["max_goals"]), int(cfg["matrix_decimals"])
    created_text = to_utc_text(created or datetime.now(timezone.utc))
    with con:
        for p in plans:
            r = rows.get(p.match_id)
            if r is None or r["model_p"] is None:
                summary["模型没给概率"] += 1
                continue
            prob, _fused, _key = archive_key(r)
            pred = preds[p.match_id]
            # 融合矩阵：conclude 只交出融合后的概率，矩阵用同样的输入再算一次 fuse（同样的输入出同样的矩阵）
            f = fuse(pred, r["fusion_odds"], settings)
            if f["status"] != r["conclusion"]["status"]:
                raise RuntimeError(f"比赛 {p.match_id}：重算的融合状态和结论不一样（{f['status']} / {r['conclusion']['status']}）")
            detail = prediction_detail(result, r, code)
            detail.update({"计划时点": p.hours, "晚了几分钟": p.late_minutes,
                           "模型比分矩阵": matrix_list(pred, n, dec),
                           "融合比分矩阵": matrix_list(f["matrix"], n, dec) if f["status"] == FUSED else None,
                           "首发": NO_LINEUP})
            if p.rescheduled_from is not None:
                detail["改期重出"] = True
                detail["原开球时间"] = p.rescheduled_from
                summary["改期重出"] += 1
            insert_prediction(con, result, r, kind_of(p.hours), created_text, prob, detail)
            summary["新存"] += 1
            summary[kind_of(p.hours)] += 1
            summary["新存的编号"].append(int(con.execute("select last_insert_rowid()").fetchone()[0]))
            late = summary["最晚了几分钟"]
            summary["最晚了几分钟"] = p.late_minutes if late is None else max(late, p.late_minutes)
    summary["模型版本"] = result["model_version"]
    return summary


# ======================= 锁：防止两次运行重叠 =======================
class RunLock:
    """锁文件（settings.yaml 的 schedule.lock_file）。拿到锁才运行，运行完删掉。

    为什么要锁：每 15 分钟跑一次，偶尔一次跑得慢（导入赔率 + 拟合模型约 2 分钟，电脑忙时更久），
    下一次又开始了，两次会对同一场比赛各存一行。
    残留锁：程序被强行关掉时锁文件删不掉。锁文件超过 schedule.lock_stale_minutes 分钟没更新就当作残留，先挪开再拿锁。
    用法：
        with RunLock(path, stale_minutes) as lock:
            if not lock.acquired: ...（别的运行正在跑，这次跳过）
    """

    def __init__(self, path: Path, stale_minutes: float):
        self.path = Path(path)
        self.stale_seconds = float(stale_minutes) * 60
        self.token = uuid.uuid4().hex          # 这次运行的记号：只删自己的锁
        self.acquired = False
        self.cleared_stale = False             # 这次有没有清掉残留锁

    def _try_create(self) -> bool:
        try:
            # O_EXCL：文件已经有了就失败。两次运行同时来，只有一个能建成
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"记号": self.token, "进程": os.getpid(),
                                "开始": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False))
        return True

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._try_create():
            self.acquired = True
            return True
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:            # 刚好被别人删了：再试一次
            age = None
        if age is not None and age > self.stale_seconds:
            # 先改名再删：两次运行同时发现残留锁，只有一个改名成功，另一个照样拿不到锁
            junk = self.path.with_name(f"{self.path.name}.stale-{self.token}")
            try:
                os.replace(self.path, junk)
                junk.unlink(missing_ok=True)
                self.cleared_stale = True
            except OSError:
                pass
        self.acquired = self._try_create()
        return self.acquired

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if json.loads(self.path.read_text(encoding="utf-8")).get("记号") == self.token:
                self.path.unlink()
        except (OSError, ValueError):
            pass
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def lock_from_settings(settings: dict) -> RunLock:
    cfg = settings["schedule"]
    return RunLock(resolve_path(cfg["lock_file"]), cfg["lock_stale_minutes"])


def main() -> None:
    """直接运行这个文件时，转给真正的定时任务入口 src/jobs/scheduled_predict.py（心跳、锁、数据库连接都在那里）。"""
    from src.jobs.scheduled_predict import main as job_main   # 放在函数里：那个文件要 import 这个文件
    job_main()


if __name__ == "__main__":
    main()
