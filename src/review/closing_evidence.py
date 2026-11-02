"""可信收盘字段与赛前最后观察价的证据审查；只读，不采集、不打分。

football-data 的 close 是已登记的来源语义，下载时间可以晚于开球，但不能晚于复盘截止。
懂球帝的最后一条赛前记录只证明那个价格被看到过，不能证明它是开球时的最后价格。
本模块的 known_at_utc 是本地证据最早可用时间；不改变其他取数函数的历史时间政策。
"""
from collections import Counter
from math import isfinite

import pandas as pd

from src.features.known_time import to_utc_text
from src.features.store import MARKETS, FeatureStore


# 这里只登记已有数据字典和导入器明确约定的来源语义；新来源写一个 close 字样不能自证。
REGISTERED_CLOSE_SOURCES = frozenset({"football-data"})
PRICES = ("line", "home", "draw", "away", "over", "under")
COLUMNS = [
    "match_id", "source", "source_key", "bookmaker", "bookmaker_raw", "market", "timing", *PRICES,
    "kickoff_utc", "kickoff_earliest_utc", "close_known_utc", "changed_at_utc", "fetched_at_utc",
    "recorded_at_utc", "known_at_utc", "raw_file", "candidate_table", "candidate_id", "provenance_count",
    "qualified", "qualification", "reason", "reasons", "evidence_note",
]
DECLARED_NOTE = "可信来源的收盘字段，并非准点现场捕捉；fetched_at_utc 是下载时间。"
OBSERVED_NOTE = "仅为赛前观察价格，不能证明是收盘；不进入可信收盘标杆。"


def _values(values, default=None):
    if values is None:
        return default
    return list(dict.fromkeys([values] if isinstance(values, str) else values))


def _missing(value):
    return value is None or bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())


def _time(row, name, reasons, required=False):
    value = row.get(name)
    if _missing(value):
        row[name] = None
        if required:
            reasons.append(f"missing_{name}")
        return None
    try:
        row[name] = to_utc_text(value)
        return row[name]
    except (ValueError, TypeError, OverflowError):
        row[name] = None
        reasons.append(f"invalid_{name}")
        return None


def _valid_odds(row):
    market = row.get("market")
    names = {"1x2": ("home", "draw", "away"), "ou": ("over", "under"), "ah": ("home", "away")}
    try:
        prices = [float(row[name]) for name in names[market]]
        if not all(isfinite(price) and price > 1 for price in prices):
            return False
        if market in ("ou", "ah") and not isfinite(float(row["line"])):
            return False
        return market != "ou" or float(row["line"]) > 0
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def _classify(candidate, cutoff, missing_tables):
    row = {name: candidate.get(name) for name in COLUMNS}
    reasons = []
    fetched = _time(row, "fetched_at_utc", reasons, required=True)
    kickoff = _time(row, "kickoff_earliest_utc", reasons, required=True)
    closed = _time(row, "close_known_utc", reasons, required=True)
    _time(row, "kickoff_utc", reasons)
    if _missing(row.get("raw_file")):
        reasons.append("missing_raw_file")
    if not _valid_odds(row):
        reasons.append("invalid_odds")
    if fetched and fetched > cutoff:
        reasons.append("fetched_after_cutoff")
    row.update(qualified=False, qualification="excluded", recorded_at_utc=None, known_at_utc=None)
    if row["candidate_table"] == "odds":
        if "match_sources" in missing_tables:
            reasons.append("missing_table:match_sources")
        count = row.get("provenance_count")
        if _missing(count) or count == 0:
            reasons.append("missing_provenance")
        elif count > 1:
            reasons.append("ambiguous_provenance")
        if _missing(row.get("source_key")):
            reasons.append("missing_source_key")
        if row["timing"] != "close":
            reasons.append("not_close")
        elif row["source"] not in REGISTERED_CLOSE_SOURCES:
            reasons.append("unregistered_close_source")
        if closed and closed > cutoff:
            reasons.append("not_yet_closed")
        row["known_at_utc"] = max(fetched, closed) if fetched and closed else None
        row["evidence_note"] = DECLARED_NOTE if row["source"] in REGISTERED_CLOSE_SOURCES else "来源未登记可信收盘语义。"
        if not reasons:
            row.update(qualified=True, qualification="source_declared_close")
            reasons = ["registered_source_declared_close"]
    else:
        changed = _time(row, "changed_at_utc", reasons, required=True)
        row["known_at_utc"] = max(changed, fetched) if changed and fetched else None
        row["evidence_note"] = OBSERVED_NOTE
        if changed and kickoff and changed >= kickoff:
            reasons.append("in_play")
        if changed and fetched and changed > fetched:
            reasons.append("changed_after_fetched")
        if changed and changed > cutoff:
            reasons.append("changed_after_cutoff")
        if not reasons:
            # “观察到”必须在最早可能开球前；赛后下载的历史价格另列，不能伪装成准点采集。
            row["qualification"] = "observed_pregame" if fetched < kickoff else "historical_pregame_price"
            reasons = ["not_verified_close"]
            if fetched >= kickoff:
                reasons.append("not_observed_pregame")
    row["reasons"] = tuple(dict.fromkeys(reasons))
    row["reason"] = row["reasons"][0]
    return row


def collect_closing_evidence(fs: FeatureStore, match_ids, cutoff, usage="复盘", bookmakers=None,
                             markets=("1x2",)) -> pd.DataFrame:
    """只审查明确给定的一组比赛，返回每场/公司/玩法的候选和缺失原因。

    qualified=True 才是可信的来源收盘字段。其余行仅用于解释缺失，绝不应参与收盘评分。
    bookmakers 明确给出时会补齐无记录的组合；不给时仅列本批实际出现的公司，完全无数据时
    公司为空。recorded_at_utc 留空：旧表没有逐条入库时间，不能把下载时间虚构成入库时间。
    """
    mids = list(dict.fromkeys(int(mid) for mid in match_ids))
    books, markets = _values(bookmakers), _values(markets, ["1x2"])
    if not markets or set(markets) - MARKETS:
        raise ValueError(f"玩法必须是 {sorted(MARKETS)} 中至少一种")
    if books == []:
        raise ValueError("bookmakers 不能是空列表；不筛选请用 None")
    cutoff = to_utc_text(cutoff)
    candidates = fs.closing_evidence_candidates(mids, cutoff, usage, bookmaker=books, market=markets)
    missing_tables = candidates.attrs["missing_tables"]
    contexts = {r["match_id"]: r for r in candidates.attrs["match_context"]}
    rows = [_classify(row, cutoff, missing_tables) for row in candidates.to_dict("records")]
    # 每个来源/公司/玩法只标记一条“最后实际观察”；历史赛前价和滚球不会抢占这个位置。
    last = {}
    for row in rows:
        if row["qualification"] == "observed_pregame":
            key = (row["match_id"], row["source"], row["bookmaker"], row["market"])
            order = (row["changed_at_utc"], row["fetched_at_utc"], row["candidate_id"])
            if key not in last or order > last[key][0]:
                last[key] = order, row
    for _, row in last.values():
        row["qualification"] = "last_observed_pregame"
    expected_books = books if books is not None else sorted({r["bookmaker"] for r in rows}) or [None]
    present = {(r["match_id"], r["bookmaker"], r["market"]) for r in rows}
    for mid in mids:
        for book in expected_books:
            for market in markets:
                if (mid, book, market) not in present:
                    reasons = ("no_record", *(f"missing_table:{t}" for t in missing_tables))
                    row = dict.fromkeys(COLUMNS)
                    row.update(contexts[mid], bookmaker=book, market=market, qualified=False,
                               qualification="absent", reason="no_record", reasons=reasons,
                               evidence_note="本次指定比赛、公司和玩法没有可审查记录。")
                    rows.append(row)
    result = pd.DataFrame(rows, columns=COLUMNS)
    result["qualified"] = result["qualified"].astype(bool)
    result.attrs.update(requested_match_ids=mids, cutoff_utc=cutoff, missing_tables=missing_tables)
    return result.sort_values(["match_id", "bookmaker", "market", "source", "timing"], na_position="last").reset_index(drop=True)


def coverage_summary(evidence: pd.DataFrame, match_ids, bookmakers=None, markets=("1x2",)) -> dict:
    """覆盖率分母只取指定比赛集合；同场多来源、多条价格不重复算场数。

    每家公司/玩法独立统计；总计是任一指定公司/玩法有合格证据的比赛数，并不表示全部玩法齐全。
    1x2 有证据不要求同场还要有大小球。coverage 为 None 表示分母为空，
    0 表示有待查比赛但没有合格收盘证据，不能从它推导模型是否胜过市场。
    """
    mids = list(dict.fromkeys(int(mid) for mid in match_ids))
    books, markets = _values(bookmakers), _values(markets, ["1x2"])
    if not markets or set(markets) - MARKETS:
        raise ValueError(f"玩法必须是 {sorted(MARKETS)} 中至少一种")
    if books == []:
        raise ValueError("bookmakers 不能是空列表；不筛选请用 None")
    part = evidence[evidence["match_id"].isin(mids) & evidence["market"].isin(markets)]
    if books is not None:
        part = part[part["bookmaker"].isin(books)]
    expected_books = books if books is not None else sorted(part["bookmaker"].dropna().unique()) or [None]
    qualified = part[part["qualified"] == True]  # 只认显式布尔真值，不让非空原因文字变成真。
    groups = []
    for book in expected_books:
        for market in markets:
            selected = part[(part["bookmaker"].isna() if book is None else part["bookmaker"] == book)
                            & (part["market"] == market)]
            covered = set(selected.loc[selected["qualified"] == True, "match_id"])
            reasons = Counter()
            for mid in mids:
                if mid in covered:
                    continue
                failed = selected[selected["match_id"] == mid]
                why = {r for values in failed["reasons"] for r in values} or {"no_record"}
                reasons.update(why)
            groups.append({"bookmaker": book, "market": market, "match_count": len(mids),
                           "qualified_match_count": len(covered),
                           "coverage": len(covered) / len(mids) if mids else None,
                           "absence_reasons": dict(sorted(reasons.items()))})
    count = int(qualified["match_id"].nunique())
    return {"match_count": len(mids), "qualified_match_count": count,
            "coverage": count / len(mids) if mids else None, "requested_match_ids": mids,
            "by_bookmaker_market": groups}


def closing_coverage(evidence: pd.DataFrame, match_ids, bookmakers=None, markets=("1x2",)) -> dict:
    """复盘集成入口；分母始终是调用方明确给出的比赛集合。"""
    return coverage_summary(evidence, match_ids, bookmakers=bookmakers, markets=markets)
