"""检查步骤 7 的分组报告（src/evaluate/injury_study.py）：分三份对不对、快照晚于截止时间要停下。"""
import pandas as pd
import pytest

from src.evaluate.injury_study import injury_groups


def _pred(ids, cutoff="2020-01-01 12:00:00"):
    # 每场两行（两个模型），分组只看比赛本身
    rows = [{"match_id": i, "model": m, "cutoff_utc": cutoff} for i in ids for m in ("a", "b")]
    return pd.DataFrame(rows)


def _inj(losses, as_of="2020-01-01 12:00:00"):
    # losses: match_id → (主队主力损失, 客队主力损失)
    return pd.DataFrame([{"match_id": i, "as_of_utc": as_of, "home_inj_top_loss": h, "away_inj_top_loss": a}
                         for i, (h, a) in losses.items()])


def test_three_groups_by_bigger_side_and_missing():
    # 1：两队都没伤；2–10：有伤，取两队里大的那个；11：伤得最重；12：没有伤停数
    losses = {1: (0.0, 0.0), **{i: (0.0, i / 100) for i in range(2, 11)}, 11: (0.5, 0.1)}
    g = injury_groups(_pred([*losses, 12]), _inj(losses), quantile=0.9)
    assert g[1] == "两队主力都没伤"
    assert g[2] == "有主力伤，不多"            # 客队 0.02 > 主队 0：看大的那个
    assert g[11].startswith("主力伤得多")       # 有伤的 10 场里最重的一成
    assert g[10] == "有主力伤，不多"
    assert g[12] == "没有伤停数"
    assert len(g) == 12                        # 每场一行（两个模型不重复算）


def test_snapshot_after_cutoff_stops():
    with pytest.raises(ValueError, match="晚于截止时间"):
        injury_groups(_pred([1]), _inj({1: (0.1, 0.0)}, as_of="2020-01-01 12:00:01"))
