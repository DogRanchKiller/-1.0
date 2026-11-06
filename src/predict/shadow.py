"""影子预测：定时预测存了正式模型的 T-24 以后，用同一个截止时间再出一份别的模型（需求文档第 4 节“xG 研究第 2 段怎么做”）。

影子模型写在 settings.yaml shadow.models。2026-11-07 以前正式模型是 gb_l7、影子是 gb_xg（懂球帝 xG 版）；
那天你决定换成 gb_xg 上线（需求文档第 4 节“换成 gb_xg 上线”），从此正式是 gb_xg、影子是 gb_l7。

做什么（src/jobs/scheduled_predict.py 在 run_scheduled 之后调 run_shadow）：
1. 看这次新存进 predictions 表的那几行：时点是 settings.yaml shadow.kinds（T-24）的，才跟着出。
2. 按这几行的截止时间（cutoff_utc，一次运行里都一样），用和正式预测一样的取数（src/predict/preview.py 的 model_inputs，
   同样的联赛、同样往后看 preview.days_ahead 天）拟合影子模型，给窗口里的比赛出预测，挑出这几场。
   为什么窗口要一样：模型出预测时会按要预测的比赛换赛季等（会动状态），预测的比赛一样，两边才好比。
3. 存进 shadow_predictions 表（src/clean/schema.py），不进 predictions：网站、推送、复盘、四本账读 predictions 时很多地方不分模型。
   每行记对应那行正式预测的 prediction_id；同一行正式预测、同一个影子模型只存一次（表上有唯一约束）。

出错不影响正式预测：调用的地方（scheduled_predict.run_job）用 safe_run_shadow，出错只记进心跳摘要，定时任务照样算成功。
漏出的（出错、关机）不补：补出来的截止时间就和正式预测那行不一样了。
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timedelta, timezone

from src.features import FeatureStore
from src.features.known_time import to_utc_text
from src.models.registry import build_model, model_config
from src.predict.preview import model_inputs, model_version, to_1x2
from src.predict.scheduled import matrix_list, parse_utc

MAIN_COLUMNS = ["prediction_id", "match_id", "competition", "season", "home_team", "away_team", "kickoff_utc",
                "match_date", "kind", "cutoff_utc"]


def main_rows(con, prediction_ids, kinds) -> list[dict]:
    """predictions 表里这几行（只要 shadow.kinds 里的时点）。predictions 是我们自己写的存档，直接读，不经过取数层。"""
    ids = [int(i) for i in prediction_ids]
    if not ids or not kinds:
        return []
    sql = (f"select {', '.join(MAIN_COLUMNS)} from predictions where prediction_id in ({','.join('?' * len(ids))})"
           f" and kind in ({','.join('?' * len(kinds))}) order by prediction_id")
    return [dict(zip(MAIN_COLUMNS, r)) for r in con.execute(sql, [*ids, *kinds])]


def already_stored(con, prediction_ids, model: str) -> set[int]:
    """这几行正式预测里，这个影子模型已经存过的（main_prediction_id）。"""
    ids = [int(i) for i in prediction_ids]
    if not ids:
        return set()
    sql = (f"select main_prediction_id from shadow_predictions where model = ?"
           f" and main_prediction_id in ({','.join('?' * len(ids))})")
    return {int(r[0]) for r in con.execute(sql, [model, *ids])}


def xg_rows(hist, model) -> int:
    """历史比赛里两队 xG 都有的场数（看懂球帝 xG 有没有真的进到模型里）。不用 xG 的模型是 0。"""
    cols = getattr(model, "xg_cols", None)
    if not getattr(model, "xg_mix", 0) or not cols or not all(c in hist for c in cols):
        return 0
    return int(hist[cols].notna().all(axis=1).sum())


def run_shadow(fs: FeatureStore, con, settings: dict, stored_ids, models=None, code: str | None = None,
               created: datetime | None = None) -> dict:
    """给这次新存的正式预测（stored_ids）出影子预测、存进 shadow_predictions。返回摘要（写进心跳文件的“影子预测”）。

    models：{名字: 模型对象}，不给就按 settings.yaml shadow.models 新造（测试里换成假模型）。
    created：写进存档的时间，不给就是电脑的当前时间。
    """
    cfg = settings["shadow"]
    summary = {"新存": 0}
    rows = main_rows(con, stored_ids, list(cfg["kinds"]))
    if not rows:
        summary["说明"] = "这次没有新存的 " + "、".join(cfg["kinds"])
        return summary
    cutoffs = sorted({r["cutoff_utc"] for r in rows})
    if len(cutoffs) != 1:
        # 一次定时预测只有一个截止时间（run_scheduled 的“现在”）；不止一个说明调错了，不硬出
        raise ValueError(f"这几行正式预测的截止时间不一样：{cutoffs}")
    cutoff = cutoffs[0]
    now = parse_utc(cutoff)
    end = to_utc_text(now + timedelta(days=settings["preview"]["days_ahead"]))
    leagues = list(settings["leagues"][settings["preview"]["leagues"]])
    n, dec = int(settings["market_odds"]["max_goals"]), int(settings["schedule"]["matrix_decimals"])
    created_text = to_utc_text(created or datetime.now(timezone.utc))
    names = list(models) if models is not None else list(cfg["models"])
    for name in names:
        start = time.perf_counter()
        info = {"新存": 0, "已存过": 0, "模型没给概率": 0, "不在窗口里": 0}
        summary[name] = info
        done = already_stored(con, [r["prediction_id"] for r in rows], name)
        todo = [r for r in rows if r["prediction_id"] not in done]
        info["已存过"] = len(rows) - len(todo)
        if not todo:
            continue
        model = models[name] if models is not None else build_model(name, settings)
        mcfg = model_config(name, settings) if models is None else {}      # 测试的假模型没有设置
        version = model_version(name, mcfg)
        # 和正式预测一样的取数、一样的窗口（cutoff 到 cutoff + days_ahead）
        hist, targets = model_inputs(fs, settings, model, cutoff, end, leagues)
        model.fit(hist, cutoff)
        probs = model.predict(targets, cutoff) if len(targets) else []
        preds = {int(t["match_id"]): p for t, p in zip(targets.to_dict("records"), probs)}
        n_xg = xg_rows(hist, model)
        with con:
            for r in todo:
                mid = int(r["match_id"])
                if mid not in preds:
                    info["不在窗口里"] += 1
                    continue
                pred = preds[mid]
                p = to_1x2(pred)
                if p is None:
                    info["模型没给概率"] += 1
                    continue
                detail = {"模型设置": mcfg, "代码版本": code, "拟合用的比赛场数": int(len(hist)),
                          "其中有xG的场数": n_xg, "模型比分矩阵": matrix_list(pred, n, dec)}
                con.execute(
                    "insert into shadow_predictions (main_prediction_id, match_id, competition, season, home_team,"
                    " away_team, kickoff_utc, match_date, kind, model, model_version, cutoff_utc, created_at_utc,"
                    " p_home, p_draw, p_away, detail) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["prediction_id"], mid, r["competition"], r["season"], r["home_team"], r["away_team"],
                     r["kickoff_utc"], r["match_date"], r["kind"], name, version, cutoff, created_text,
                     *[round(float(x), 4) for x in p], json.dumps(detail, ensure_ascii=False)))
                info["新存"] += 1
                summary["新存"] += 1
        info.update({"模型版本": version, "其中有xG的场数": n_xg, "用了几秒": round(time.perf_counter() - start, 1)})
    return summary


def safe_run_shadow(*args, **kwargs) -> dict:
    """run_shadow 包一层：出错时返回 {"错误": ...}，不往外抛（影子预测出错不能让正式预测的定时任务算失败）。"""
    try:
        return run_shadow(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 —— 故意全接住：影子预测只是对比用，任何错都不能影响正式预测
        return {"错误": f"{type(e).__name__}: {e}", "位置": traceback.format_exc(limit=3)}
