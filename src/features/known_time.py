"""每场比赛的各类数据“什么时候能知道”（需求文档第 16 节“不偷看未来”）。

回测时假装站在过去某个时间点（截止时间）做预测，只能用那时已经知道的数据。
所以每条数据都要有一个“能知道的时间”，和截止时间比：能知道的时间 ≤ 截止时间，才能用。

规则（数字都在 settings.yaml 的 features 段）：
- 比分、射门等赛后数据：开球时间 + result_available_after_kickoff_hours（默认 3 小时：踢 2 小时，留 1 小时更新数据）。
- 收盘赔率：开球那一刻。所以本场的收盘赔率，开球前任何时候都拿不到。
- 赛前赔率：开球前 pre_odds_hours_before_kickoff 小时（按来源分；football-data 是 24，当作 T-24 版的赔率）。
- 赛程（谁和谁、哪天踢）：一直能知道。

只有日期、没有开球时间的比赛（football-data 2019/20 以前全是这样）用保守规则，两头都往“更不容易偷看”的方向估：
- 最早开球：当天 00:00（英国时间）。判断“是不是还在开球前”用它——宁可认为比赛开得早，免得开球后还当作赛前。
- 最晚开球：当天 24:00（英国时间，即第二天 00:00）。判断“什么时候知道结果”用它——宁可认为结果出来得晚。
  例如 2010-08-14 只有日期的比赛，比分要到 2010-08-15 00:00 英国时间（= 2010-08-14 23:00 UTC，夏令时）+ 3 小时才能用，
  所以同一天晚些时候的其他比赛，用不到它的比分（实际上同一天的比赛本来就几乎同时开球）。
有开球时间的比赛，最早、最晚开球都等于真实开球时间。

只知道“这一轮从哪天开始”的杯赛（matches.date_precision = 'round'，维基百科早年的英足总杯、法国杯等），
match_date 是这一轮的第一天，比赛可能是这一轮里的任何一天：
- 最早开球：这一轮第一天 00:00（同上）。
- 最晚开球：这一轮第 round_date_window_days 天（默认第 7 天）的 24:00。比分要到那以后才能用，宁可晚知道，不偷看。

所有时间都是 UTC，写成文字 'YYYY-MM-DD HH:MM:SS'。这种写法按文字排序就是按时间排序，数据库里可以直接比大小。
"""
import pandas as pd

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def to_utc_text(t) -> str:
    """把截止时间统一成 UTC 文字。

    可以给：文字 '2024-03-01 15:00'（当作 UTC）、带时区的文字 '2024-03-01T16:00+01:00'、datetime、pandas.Timestamp。
    没写时区的当作 UTC（项目约定内部时间一律 UTC）；写了时区的换算成 UTC。
    秒以下的部分舍掉（舍掉只会让截止时间更早，不会多看到东西）。
    """
    if t is None:
        raise ValueError("截止时间不能为空：取数据必须说清楚“站在哪个时间点”")
    ts = pd.Timestamp(t)
    if pd.isna(ts):
        raise ValueError(f"看不懂的截止时间：{t!r}")
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime(TIME_FORMAT)


def _local_midnight_utc(dates: pd.Series, tz: str, ambiguous_early: bool) -> pd.Series:
    """某个时区某一天的 00:00，换成 UTC。

    夏令时切换那一刻如果正好是 00:00（英国不会，切换在凌晨 1、2 点），重复的时间取早的或晚的，不存在的时间往后推。
    """
    local = pd.to_datetime(dates, format="%Y-%m-%d")
    out = local.dt.tz_localize(tz, ambiguous=pd.Series(ambiguous_early, index=local.index).to_numpy(),
                               nonexistent="shift_forward")
    return out.dt.tz_convert("UTC").dt.tz_localize(None)


def compute_known_times(matches: pd.DataFrame, feature_settings: dict) -> pd.DataFrame:
    """给每场比赛算出几个关键时间点。

    输入：matches 表的 match_id、match_date（当地日期）、kickoff_utc（可能为空），
         可以带 date_precision（'day' / 'round'；没有这一列时都当 'day'）。
    输出（每场一行，都是 UTC 文字）：
      kickoff_earliest_utc  最早可能的开球时间 —— “开球前”的判断、赛前赔率的时间都从它算
      kickoff_latest_utc    最晚可能的开球时间 —— “比赛已经踢了”、收盘赔率从它算
      result_known_utc      比分、统计能知道的时间 = 最晚开球 + 缓冲
      close_known_utc       收盘赔率能知道的时间 = 最晚开球
    算 T-24 / T-6 / T-1 这些预测时点时，以最早开球为基准（宁早勿晚）。
    """
    tz = feature_settings["date_only_timezone"]
    buffer = pd.Timedelta(hours=float(feature_settings["result_available_after_kickoff_hours"]))
    kick = pd.to_datetime(matches["kickoff_utc"], errors="coerce")  # 空的变成 NaT
    day_start = _local_midnight_utc(matches["match_date"], tz, ambiguous_early=True)
    next_day = (pd.to_datetime(matches["match_date"], format="%Y-%m-%d") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    day_end = _local_midnight_utc(next_day, tz, ambiguous_early=False)
    if "date_precision" in matches.columns:
        # 只知道轮次开始日的：最晚开球再往后推（第一天已经算在 day_end 里，所以推 窗口 - 1 天）
        extra_days = float(feature_settings.get("round_date_window_days", 7)) - 1
        is_round = (matches["date_precision"] == "round").to_numpy()
        day_end = day_end.where(~is_round, day_end + pd.Timedelta(days=extra_days))
    earliest = kick.fillna(day_start)
    latest = kick.fillna(day_end)

    def fmt(s: pd.Series) -> pd.Series:
        return s.dt.strftime(TIME_FORMAT)

    return pd.DataFrame({
        "match_id": matches["match_id"].to_numpy(),
        "kickoff_earliest_utc": fmt(earliest).to_numpy(),
        "kickoff_latest_utc": fmt(latest).to_numpy(),
        "result_known_utc": fmt(latest + buffer).to_numpy(),
        "close_known_utc": fmt(latest).to_numpy(),
    })
