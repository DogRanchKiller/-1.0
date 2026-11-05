'''第 11 阶段：国际赛开发回测（需求文档第 4 节“第 11 阶段：国际赛进数据库和开发回测”的“模型和研究（INC）”）。

要回答的两个问题（事先写下，不临时加）：
  1. 主比较：国家队 Elo（intl_elo，换成胜平负）和懂球帝“多家平均”收盘去水差多少。
     比赛：2021–2023 的正式比赛（不含友谊赛）里有多家平均收盘的（INB 只给比分和懂球帝一样的比赛导赔率）。
  2. 中立场比较：intl_elo（中立场不加主场优势）和 intl_elo_home（一律加主场优势，其余一样）谁准。
     比赛：2010–2023 的全部正式比赛（不用赔率）。
  年份都写在 settings.yaml 的 intl_study 段。

怎么走（和第 8 阶段欧战研究 euro_study.py 一样，只往前走，不偷看）：
  - 时间按周切成格子（intl_study.refit_days = 7 天一格），格子从固定的一天（intl_study.grid_anchor）算起，
    所以不管从哪一年开始跑，格子都落在同样的日子上，同一场比赛的预测一模一样。
  - 每场要预测的比赛先算 T-24：截止时间 = 最早可能的开球 − 24 小时（取数层 fs.cutoff_before）。
    这场比赛归到“T-24 之前最近的那个格子”，就用那个格子那一刻的评分和参数出预测（最多早 7 天，只会更保守）。
  - 走到一个格子：只从取数层取“这一刻已经能知道比分”的 INT 比赛（用途“研究实验”、开发模式），
    还没算进评分的按比赛日期顺序一场一场更新两套 Elo（增量更新，不每步从 1872 年重算）；
    再用这一刻以前 8 年的正式比赛重新拟合两版的有序 logit，给归到这个格子的比赛出预测。
    没有要预测的比赛的格子直接跳过：下一次取数会把中间的比赛一起按日期顺序补上，结果和一周一周走完全一样
    （比分能知道的时间按比赛日期排，所以一批批补和一场场补的顺序一样；程序里也数了“日期顺序不对的场数”，正常是 0）。
  - 全部走完以后，才站在开球后 eval_after_kickoff_hours 小时拿答案（比分）和标杆（多家平均收盘赔率，用途“评估”）打分。
  - 开发模式下取到 2024 年（期末考试赛季）的比赛，取数层直接报错；--last 写 2024 以后程序自己先报错。

判断（compare 段：按周成块 bootstrap 2,000 次、95%、固定种子）：差 = 前 − 后（胜平负 log loss，越小越好）。
  区间整个 > 0 “更差”，整个 < 0 “更准”，跨过 0 “看不出差别”。合计少于 min_total_n 场只列数、不下结论；
  一年少于 min_year_n 场标“样本少”。Brier、命中率、按赛事分组只看看。

结果：报告 intl_study.output（data/reports/国际赛_开发回测对比.md），逐场预测 intl_study.predictions（predictions.csv），
data/ 不进 Git。
用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.intl_study
  快速试跑：--first 2021 --last 2021（只评这几年；评分照样从 1872 年算起）
'''
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.common.odds import devig
from src.evaluate import metrics
from src.evaluate.compare import block_ids, bootstrap_mean_diff, verdict
from src.features import FeatureStore, LookAheadError, open_readonly
from src.features.known_time import to_utc_text
from src.models.intl_elo import IntlElo, UnknownTournamentError

SECTION = 'intl_study'
EVAL_USAGE = '评估'                  # 赛后拿答案、收盘赔率打分（和 backtest 一样）
SMALL_TOTAL = '样本少，只列数，不下结论'
SMALL_YEAR = '样本少'
LOOK = '只看看'
VERDICT_TEXT = {'更准': '更准', '没差别': '看不出差别', '更差': '更差'}   # compare.verdict 的说法换成需求文档的说法
P1, P0, PB = ['pH', 'pD', 'pA'], ['hH', 'hD', 'hA'], ['bH', 'bD', 'bA']   # intl_elo、intl_elo_home、多家平均 的三列
NAMES = {'intl_elo': P1, 'intl_elo_home': P0, '多家平均': PB}


def _text(ts) -> str:
    return pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def years_of(cfg: dict, key: str) -> tuple[int, int]:
    a, b = cfg[key]
    return int(a), int(b)


def check_k_table(names, cfg: dict) -> None:
    '''数据库里出现的赛事名，每一个都要在 k_by_tournament 里写了 K；没写的列出来一起报错（不猜）。'''
    known = {str(n) for v in cfg['k_by_tournament'].values() for n in (v or [])}
    missing = sorted({str(n) for n in names} - known)
    if missing:
        raise UnknownTournamentError(
            f'这些赛事名没写进 settings.yaml 的 intl_study.k_by_tournament（共 {len(missing)} 个），不知道 K 是哪一档：'
            + '、'.join(missing[:30]) + ('……' if len(missing) > 30 else ''))


def tournament_group(name: str, cfg: dict) -> str:
    '''按赛事分组（只看看）：settings 的 tournament_groups，没写进去的算“其他”。'''
    for group, names in cfg.get('tournament_groups', {}).items():
        if name in (names or []):
            return group
    return '其他'


def venues(fs: FeatureStore, cutoff: str, cfg: dict) -> pd.DataFrame:
    '''每场 INT 比赛的中立场和赛事名（取数层 match_venues：赛程，赛前就能知道），只要 venue_source 那个来源。'''
    if not hasattr(fs, 'match_venues'):
        raise RuntimeError('取数层还没有 match_venues（工作树 INA 加的），合并 INA 以后再跑')
    v = fs.match_venues(cutoff, cfg['usage'], competition=cfg['competition'])
    v = v[v['source'] == cfg['venue_source']]
    if v['match_id'].duplicated().any():
        raise ValueError('match_venue 里同一场比赛同一个来源有两行')
    return v[['match_id', 'neutral', 'tournament']].assign(neutral=lambda d: d['neutral'].astype(int).astype(bool))


# ======================= 滚动：一步一步往前走 =======================
def run(settings: dict | None = None, db_path=None, first: int | None = None, last: int | None = None,
        store: FeatureStore | None = None) -> dict:
    s = copy.deepcopy(settings or load_settings())
    cfg = s[SECTION]
    comp, usage = cfg['competition'], cfg['usage']
    dev_last = int(s['seasons']['dev_last'])
    main_y, neu_y = years_of(cfg, 'main_years'), years_of(cfg, 'neutral_years')
    first = min(main_y[0], neu_y[0]) if first is None else int(first)
    last = max(main_y[1], neu_y[1]) if last is None else int(last)
    if last > dev_last:
        raise ValueError(f'开发模式只能评到 {dev_last} 年；{dev_last + 1} 年起是期末考试赛季，不能碰（收到 --last {last}）')
    if first > last:
        raise ValueError(f'--first {first} 比 --last {last} 晚')
    t0 = time.perf_counter()
    fs = store or FeatureStore(open_readonly(db_path), '开发', s)
    if comp not in fs.leagues:
        raise RuntimeError(f'settings 的 leagues 里还没有 {comp}（工作树 INA 加 intl 组），合并 INA 以后再跑')
    horizon = float(cfg['horizon_hours'])
    after = pd.Timedelta(hours=float(s['backtest']['eval_after_kickoff_hours']))
    # 两个终点：
    #   tgt_end：要预测的名单取到这一刻（按最早开球）。day_world 的最早开球 = 前一天 10:00 UTC，
    #            所以 last 年 12 月 31 日的比赛（最早开球 12 月 30 日 10:00）在里面，下一年 1 月 1 日的（12 月 31 日 10:00）不在，
    #            开发模式下不会碰到 2024 年的比赛。
    #   end：拿答案的时刻。last 年最后一天的比分能知道是下一年 1 月 1 日 15:00 UTC，下一年 1 月 1 日的比分要到 1 月 2 日 15:00，
    #        所以站在下一年 1 月 2 日 00:00 正好拿到 last 年全部比分、拿不到下一年的。
    tgt_end = f'{last}-12-31 00:00:00'
    end = f'{last + 1}-01-02 00:00:00'

    # ---- 走之前：赛事名、中立场（赛程，赛前就知道）；要预测的名单（只用赛前就知道的信息） ----
    ven = venues(fs, tgt_end, cfg)
    check_k_table(ven['tournament'].unique(), cfg)
    tgt = fs.prediction_targets(f'{first - 1}-12-01 00:00:00', tgt_end, usage, competition=comp)
    tgt = tgt[(tgt['season'] >= first) & (tgt['season'] <= last)]
    tgt = tgt.merge(ven, on='match_id', how='left')
    if tgt['tournament'].isna().any():
        raise ValueError(f'{int(tgt["tournament"].isna().sum())} 场要预测的比赛在 match_venue 里没有 {cfg["venue_source"]} 的行')
    n_friendly = int((tgt['tournament'] == cfg['friendly_tournament']).sum())
    tgt = tgt[tgt['tournament'] != cfg['friendly_tournament']].copy()         # 友谊赛不打分（照样进 Elo 更新）
    tgt['t24'] = [fs.cutoff_before(int(m), horizon, usage) for m in tgt['match_id']]

    # ---- 格子：从固定的 grid_anchor 起每 refit_days 天一格；每场归到 T-24 之前最近的格子 ----
    step = pd.Timedelta(days=int(cfg['refit_days']))
    anchor = pd.Timestamp(cfg['grid_anchor'])
    t_ns = pd.to_datetime(tgt['t24'])
    tgt['step'] = ((t_ns - anchor) // step).astype(int)
    tgt['step_cutoff'] = [_text(anchor + k * step) for k in tgt['step']]
    by_step = {int(k): g for k, g in tgt.groupby('step')}

    elo = IntlElo.from_settings(cfg, True, 'intl_elo')
    elo_h = IntlElo.from_settings(cfg, False, 'intl_elo_home')
    seen: set = set()
    since = None
    margin = pd.Timedelta(days=10)       # 往回多取 10 天（day_world 的比分开球后约 1 天多才知道），已经算过的按 match_id 跳过
    dropped_no_score = 0
    rows = []
    for i, k in enumerate(sorted(by_step)):
        c = _text(anchor + k * step)
        new = fs.historical_matches(c, usage, competition=comp, data=['比分'], since=since, source=cfg['score_source'])
        new = new[~new['match_id'].isin(seen)]
        if (new['known_at_utc'] > c).any():
            raise LookAheadError(f'取数层返回了截止时间 {c} 之后才能知道的比分，研究停止')
        seen.update(int(m) for m in new['match_id'])
        since = _text(pd.Timestamp(c) - margin)
        ok = new['home_goals'].notna() & new['away_goals'].notna()
        dropped_no_score += int((~ok).sum())
        new = new[ok].merge(ven, on='match_id', how='left')
        if new['tournament'].isna().any():
            raise ValueError(f'{int(new["tournament"].isna().sum())} 场已赛的比赛在 match_venue 里没有 {cfg["venue_source"]} 的行')
        elo.update_frame(new)
        elo_h.update_frame(new)
        elo.fit(c)
        elo_h.fit(c)
        for r in by_step[k].to_dict('records'):
            h, a, nt = r['home_team'], r['away_team'], bool(r['neutral'])
            p1, p0 = elo.predict(h, a, nt), elo_h.predict(h, a, nt)
            row = {'match_id': r['match_id'], 'season': int(r['season']), 'match_date': r['match_date'],
                   'home_team': h, 'away_team': a, 'tournament': r['tournament'], 'group': tournament_group(r['tournament'], cfg),
                   'neutral': nt, 'kickoff_earliest_utc': r['kickoff_earliest_utc'], 't24': r['t24'], 'step_cutoff': c,
                   'elo_home': elo.rating(h), 'elo_away': elo.rating(a), 'diff': elo.rating_diff(h, a, nt),
                   'elo_home_h': elo_h.rating(h), 'elo_away_h': elo_h.rating(a), 'diff_h': elo_h.rating_diff(h, a, nt),
                   'fit_n': elo.fit_n, 'fit_n_h': elo_h.fit_n}
            for cols, p in ((P1, p1), (P0, p0)):
                row.update({col: (np.nan if p is None else float(v)) for col, v in zip(cols, p if p is not None else [0, 0, 0])})
            rows.append(row)
        if (i + 1) % 100 == 0:
            print(f'  走到第 {i + 1}/{len(by_step)} 步（{c[:10]}），用时 {time.perf_counter() - t0:.0f} 秒', flush=True)
    pred = pd.DataFrame(rows)
    t_roll = time.perf_counter() - t0

    # ---- 全部走完：拿答案、标杆 ----
    pred = attach_answers(fs, pred, end, cfg)
    pred = attach_benchmark(fs, pred, after, cfg, s)
    return {'settings': s, 'pred': pred, 'first': first, 'last': last, 'n_targets': len(tgt), 'n_friendly_skipped': n_friendly,
            'n_steps': len(by_step), 'n_updates': elo.n_updates, 'out_of_order': (elo.out_of_order, elo_h.out_of_order),
            'dropped_no_score': dropped_no_score, 'elo': elo, 'elo_h': elo_h,
            'seconds_roll': t_roll, 'seconds': time.perf_counter() - t0}


def attach_answers(fs: FeatureStore, pred: pd.DataFrame, end: str, cfg: dict) -> pd.DataFrame:
    '''全部预测做完以后才取答案（比分、胜平负）。没有答案的（没踢、没比分）不打分。'''
    if pred.empty:
        return pred.assign(home_goals=pd.Series(dtype=float), away_goals=pd.Series(dtype=float), result=pd.Series(dtype=object))
    since = f'{int(pred["season"].min()) - 1}-12-01 00:00:00'
    ans = fs.historical_matches(end, cfg['usage'], competition=cfg['competition'], data=['比分'], since=since,
                                source=cfg['score_source'])
    ans = ans[['match_id', 'home_goals', 'away_goals', 'result']].dropna()
    return pred.merge(ans, on='match_id', how='left')


def attach_benchmark(fs: FeatureStore, pred: pd.DataFrame, after: pd.Timedelta, cfg: dict, s: dict) -> pd.DataFrame:
    '''主比较那几年的比赛，站在开球后 eval_after_kickoff_hours 小时取“多家平均”收盘胜平负（用途“评估”），
    按 backtest.benchmark_devig_method 去水。只认 benchmark_source 这个来源（懂球帝国际赛）。'''
    pred = pred.copy()
    for col in PB:
        pred[col] = np.nan
    if pred.empty:
        return pred
    y0, y1 = years_of(cfg, 'main_years')
    method = s['backtest']['benchmark_devig_method']
    for i in pred.index[(pred['season'] >= y0) & (pred['season'] <= y1)]:
        r = pred.loc[i]
        when = to_utc_text(pd.Timestamp(r['kickoff_earliest_utc']) + after)
        o = fs.closing_odds(int(r['match_id']), when, EVAL_USAGE, bookmaker=cfg['benchmark_bookmaker'], market='1x2')
        o = o[o['source'] == cfg['benchmark_source']].dropna(subset=['home', 'draw', 'away'])
        if len(o) > 1:
            raise ValueError(f'比赛 {r["match_id"]} 有 {len(o)} 行多家平均收盘胜平负')
        if len(o) == 1:
            pred.loc[i, PB] = devig(o.iloc[0][['home', 'draw', 'away']].to_numpy(float), method)
    return pred


# ======================= 打分、比较 =======================
def scored(pred: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    '''一组概率的每场分数：log loss、Brier、命中（概率最大的那项是不是真的发生了）。'''
    p = pred[cols].to_numpy(float)
    y = metrics.encode_results(pred['result'])
    return pd.DataFrame({'log_loss': metrics.log_loss_per_match(p, y), 'brier': metrics.brier_per_match(p, y),
                         'hit': (p.argmax(axis=1) == y).astype(float)}, index=pred.index)


def comparison_rows(pred: pd.DataFrame, a: str, b: str, years: tuple[int, int]) -> pd.DataFrame:
    '''比较 a 和 b 用哪些比赛：这几年的正式比赛（名单里已经没有友谊赛）、有答案、两边都有概率。'''
    if pred.empty:
        return pred
    m = (pred['season'] >= years[0]) & (pred['season'] <= years[1]) & pred['result'].notna()
    m &= pred[NAMES[a]].notna().all(axis=1) & pred[NAMES[b]].notna().all(axis=1)
    return pred[m]


def compare_pair(pred: pd.DataFrame, a: str, b: str, years: tuple[int, int], s: dict) -> dict:
    '''差 = a − b。三个分数都算区间；只有 log loss 下结论（compare.primary_metric），Brier、命中率只看看。
    合计少于 min_total_n 场：区间照算，结论写“样本少，只列数，不下结论”。'''
    cfg, c = s[SECTION], s['compare']
    p = comparison_rows(pred, a, b, years)
    out = {'前': a, '后': b, '年份': f'{years[0]}–{years[1]}', '场数': len(p), 'rows': p}
    if p.empty:
        out['结论'] = '没有可比的比赛'
        return out
    sa, sb = scored(p, NAMES[a]), scored(p, NAMES[b])
    blocks = block_ids(p['match_date'], c['block'])
    out['块数'] = int(len(np.unique(blocks)))
    for m in ('log_loss', 'brier', 'hit'):
        mean, lo, hi = bootstrap_mean_diff(sa[m] - sb[m], blocks, c['n_boot'], c['confidence'], c['seed'])
        out[m] = {'前': float(sa[m].mean()), '后': float(sb[m].mean()), '差': mean, '下限': lo, '上限': hi}
    lo, hi = out[c['primary_metric']]['下限'], out[c['primary_metric']]['上限']
    out['结论'] = SMALL_TOTAL if len(p) < int(cfg['min_total_n']) else VERDICT_TEXT[verdict(lo, hi)]
    return out


def by_group(rows: pd.DataFrame, a: str, b: str, col: str, cfg: dict) -> pd.DataFrame:
    '''按某一列（年份、赛事分组、场地）分开的场数、两边 log loss 和命中率、差（只看看，不算区间）。
    按年份分时，一年少于 min_year_n 场标“样本少”。'''
    if rows.empty:
        return pd.DataFrame()
    sa, sb = scored(rows, NAMES[a]), scored(rows, NAMES[b])
    t = pd.DataFrame({'组': rows[col].to_numpy(), 'a': sa['log_loss'].to_numpy(), 'b': sb['log_loss'].to_numpy(),
                      'ha': sa['hit'].to_numpy(), 'hb': sb['hit'].to_numpy()})
    o = t.groupby('组').agg(场数=('a', 'size'), 前=('a', 'mean'), 后=('b', 'mean'), 前命中=('ha', 'mean'),
                           后命中=('hb', 'mean')).reset_index()
    o['差'] = o['前'] - o['后']
    o = o.rename(columns={'前': f'{a} log loss', '后': f'{b} log loss', '前命中': f'{a} 命中', '后命中': f'{b} 命中'})
    if col == 'season':
        o['备注'] = np.where(o['场数'] < int(cfg['min_year_n']), SMALL_YEAR, '')
    return o


def analyze(out: dict) -> dict:
    '''两个事先写下的比较（年份按 --first / --last 截一下，快速试跑也能出表）。'''
    s, cfg = out['settings'], out['settings'][SECTION]

    def clip(key):
        a, b = years_of(cfg, key)
        return max(a, out['first']), min(b, out['last'])
    out['main'] = compare_pair(out['pred'], 'intl_elo', '多家平均', clip('main_years'), s)
    out['neutral'] = compare_pair(out['pred'], 'intl_elo', 'intl_elo_home', clip('neutral_years'), s)
    return out


# ======================= 报告 =======================
def _signed(v: float, nd: int = 4) -> str:
    return f'{v:+.{nd}f}'


def md_table(df: pd.DataFrame, nd: int = 4) -> str:
    '''pandas 表 → markdown 表（小数保留 nd 位）。'''
    if df.empty:
        return '（没有）'
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        lines.append('| ' + ' | '.join(f'{v:.{nd}f}' if isinstance(v, float) else str(v) for v in r) + ' |')
    return '\n'.join(lines)


def pair_section(res: dict, title: str, cfg: dict) -> list[str]:
    '''一个事先写下的比较：合计表（log loss 下结论，Brier、命中率只看看）+ 分年、分赛事（只看看）。'''
    L = [f'## {title}', '',
         f'比赛：{res["年份"]} 的正式比赛 {res["场数"]} 场' + (f'（{res["块数"]} 周）' if '块数' in res else '')
         + f'。差 = {res["前"]} − {res["后"]}，越小越好（负数 = {res["前"]} 更准）。', '']
    if res['场数'] == 0:
        return L + ['没有可比的比赛。', '']
    L += [f'| 分数 | {res["前"]} | {res["后"]} | 差 | 95% 区间 | 结论 |', '|---|---|---|---|---|---|']
    for m, name in (('log_loss', '胜平负 log loss'), ('brier', 'Brier'), ('hit', '命中率')):
        r = res[m]
        concl = res['结论'] if m == 'log_loss' else LOOK
        L.append(f'| {name} | {r["前"]:.4f} | {r["后"]:.4f} | {_signed(r["差"])} | '
                 f'{_signed(r["下限"])} 到 {_signed(r["上限"])} | {concl} |')
    L += ['', '命中率越大越好（差 > 0 是前者命中多）；Brier、命中率只看看，不下结论。', '',
          f'分年（只看看；一年少于 {int(cfg["min_year_n"])} 场标“样本少”）：', '',
          md_table(by_group(res['rows'], res['前'], res['后'], 'season', cfg)), '',
          '分赛事（只看看）：', '', md_table(by_group(res['rows'], res['前'], res['后'], 'group', cfg)), '']
    return L


def build_report(out: dict) -> str:
    s, cfg, pred = out['settings'], out['settings'][SECTION], out['pred']
    n_pred = int(pred[P1].notna().all(axis=1).sum()) if not pred.empty else 0
    n_ans = int(pred['result'].notna().sum()) if not pred.empty else 0
    L = ['# 国际赛开发回测对比（第 11 阶段，工作树 INC）', '',
         '只是研究：不上线，不接实时预测、定时任务、网站、推送，不改任何上线的模型和设置。开发模式，只用 2023 年及以前的比赛。', '',
         f'- 评的年份：{out["first"]}–{out["last"]}；T-{float(cfg["horizon_hours"]):g}；格子每 {cfg["refit_days"]} 天一格，'
         f'走了 {out["n_steps"]} 个有比赛要预测的格子；滚动 {out["seconds_roll"]:.0f} 秒，合计 {out["seconds"]:.0f} 秒。',
         f'- Elo 更新了 {out["n_updates"]} 场（含友谊赛，从最早一场算起）；比分空着跳过 {out["dropped_no_score"]} 场；'
         f'日期顺序不对的场数（应该是 0）：intl_elo {out["out_of_order"][0]}、intl_elo_home {out["out_of_order"][1]}。',
         f'- 要预测的正式比赛 {out["n_targets"]} 场（另有友谊赛 {out["n_friendly_skipped"]} 场不打分）；出了预测 {n_pred} 场；有答案 {n_ans} 场。',
         f'- 设置：主场优势 {cfg["home_advantage"]}、起始分 {cfg["start_rating"]}、P 四舍五入到整数 {cfg.get("round_points", True)}；'
         f'有序 logit 用截止前 {cfg["fit_years"]} 年的正式比赛，少于 {cfg.get("min_fit_n", 200)} 场不出预测；'
         f'标杆：{cfg["benchmark_bookmaker"]}（来源 {cfg["benchmark_source"]}）收盘胜平负，去水 {s["backtest"]["benchmark_devig_method"]}。',
         f'- 判断：按周成块 bootstrap {s["compare"]["n_boot"]} 次、{float(s["compare"]["confidence"]):.0%}、种子 {s["compare"]["seed"]}；'
         f'合计少于 {cfg["min_total_n"]} 场只列数、不下结论。']
    if not pred.empty:
        y0, y1 = years_of(cfg, 'main_years')
        mm = (pred['season'] >= y0) & (pred['season'] <= y1) & pred['result'].notna()
        L.append(f'- 主比较那几年有答案的正式比赛 {int(mm.sum())} 场，其中有多家平均收盘的 '
                 f'{int((mm & pred[PB].notna().all(axis=1)).sum())} 场（没有的：懂球帝没有这场、比分和懂球帝不一样、收盘价不合格）。')
    L.append('')
    L += pair_section(out['main'], '一、主比较：intl_elo vs 懂球帝多家平均收盘（去水）', cfg)
    L += pair_section(out['neutral'], '二、中立场比较：intl_elo（中立场不加主场优势）vs intl_elo_home（一律加）', cfg)
    nr = out['neutral'].get('rows')
    if nr is not None and not nr.empty:
        L += ['中立场比较按“是不是中立场”分开（只看看）：', '',
              md_table(by_group(nr.assign(场地=np.where(nr['neutral'], '中立场', '非中立场')),
                                'intl_elo', 'intl_elo_home', '场地', cfg)), '']
    L += ['## 三、最后一步的参数（给人看）', '']
    for e in (out['elo'], out['elo_h']):
        L.append(f'- {e.name}：{e.params()}')
    L += [f'- intl_elo 评分最高的 10 队：{out["elo"].top(10)}', '']
    return '\n'.join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description='第 11 阶段：国际赛开发回测（国家队 Elo vs 多家平均收盘、中立场比较）')
    ap.add_argument('--first', type=int, default=None, help='从这一年起评（快速试跑用；评分照样从最早一场算起）')
    ap.add_argument('--last', type=int, default=None, help='评到这一年（不能超过开发赛季最后一年）')
    ap.add_argument('--db', default=None, help='数据库文件（不写用 settings 的 paths.database）')
    ap.add_argument('--report', default=None, help='报告放哪（不写用 intl_study.output）')
    ap.add_argument('--predictions', default=None, help='逐场预测放哪（不写用 intl_study.predictions）')
    a = ap.parse_args()
    out = analyze(run(db_path=a.db, first=a.first, last=a.last))
    cfg = out['settings'][SECTION]
    rep = Path(resolve_path(a.report or cfg['output']))
    prd = Path(resolve_path(a.predictions or cfg['predictions']))
    for p in (rep, prd):
        p.parent.mkdir(parents=True, exist_ok=True)
    out['pred'].to_csv(prd, index=False, encoding='utf-8-sig')
    text = build_report(out)
    rep.write_text(text, encoding='utf-8')
    print(text)
    print(f'\n报告：{rep}\n逐场预测：{prd}')


if __name__ == '__main__':
    main()
