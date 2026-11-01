'''第 10 阶段：角球、黄牌研究（需求文档第 4 节“第 10 阶段：角球、黄牌怎么做”，规则是跑之前写下的，不临时加）。

这个程序只做研究、只出报告：不改任何已有的模型、融合权重 w、倾向门槛、信心等级，不碰期末考试赛季（2024、2025）。

要回答的问题（详见需求文档）：
  1. 两队合计的角球数、黄牌数，能不能用“球队倾向 + 泊松”预测得比“同联赛最近场均”更准？
  2. 要不要负二项（角球、黄牌是不是比泊松更分散）？
  3. 英超的黄牌，知道裁判以后能不能预测得更准？
  没有角球、黄牌盘口赔率，所以不能和市场比，也不能说“能赚钱”。

四组模型（都预测“两队合计”，评分用计数 log loss，越小越准）：
  c0_league   基线：同联赛截止前最近 baseline_matches 场的场均数，当泊松；
  c1_team     球队倾向 + 泊松：主队均值 = exp(截距 + 主场项 + 主队自己的倾向 + 客队让对手拿到的倾向)，客队同理，
              用截止前最近 train_matches 场、按时间指数衰减、带岭惩罚（把每队的倾向往 0 拉，防止小样本乱跳）；
              没有历史的新球队倾向取 0；两队合计 = 两个均值相加（两个独立泊松相加还是泊松）；
  c2_team_nb  均值和 c1 一样，合计数用负二项（方差 = 均值 + α × 均值的平方，α 用训练部分的矩估计）；
  c3_ref      只在英超、只做黄牌：c1 加一个裁判项（两队共用，岭惩罚收缩）。
  计数 log loss = −log(这场实际总数在模型分布下的概率)，所有场平均。

不偷看未来，三道关（和回测一样）：
  1. 所有数据都经过取数层 FeatureStore（开发模式，用途“研究实验”）：只拿截止时间之前已经能知道的比赛，
     用途、赛季不允许就报错；每次拿到的历史再核对一遍 known_at_utc ≤ 截止时间，要预测的比赛绝不会出现在历史里；
  2. 每 refit_days 天为一块：这一块里最早一场开球前 24 小时是截止时间，整块用同一组参数
     （块里晚一些的比赛看不到块里早场的结果，比逐场 T-24 更保守，宁早勿晚）；
  3. 本场的角球、黄牌（答案）和本场裁判，只在全部预测做完以后，用途“复盘”取来打分、接裁判。

两个阶段（需求文档“怎么比”）：
  阶段一：select_seasons 里用 ridge_grid 三档岭惩罚各跑一遍 c1，选计数 log loss 最小的一档，角球、黄牌分别选；
  阶段二：eval_seasons 里只用选好的那一档跑 c0 / c1 / c2 / c3，按事先定的 5 个主比较出结论；
  主比较用按周成块的 bootstrap（src/evaluate/compare.py），95% 和 99% 两个区间都整个在“更准”一边才写“更准”，
  只有 95% 满足的写“可能更准，也可能是巧合”。其余数字（分联赛、分赛季、校准、离散、主场项、裁判项大小）只报告，不下结论。

用法（在项目文件夹里，Claude 运行）：
  .venv\\Scripts\\python -m src.evaluate.corners_cards_study
  试跑（只为量时间，结果放到 *_trial 文件夹，不算正式结果）：--leagues E0 --select 2005 2005 --eval 2014 2014
结果在 data/reports/corners_cards/（data/ 不进 Git）。
'''
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.common.config import load_settings, resolve_path, season_role
from src.evaluate.compare import BETTER, SAME, WORSE, block_ids, bootstrap_mean_diff, verdict
from src.features import FeatureStore, LookAheadError, open_readonly

log = logging.getLogger('corners_cards_study')

CORNERS_DATA = '角球、黄牌（研究）'   # 数据字典里的数据类名
REFEREE_DATA = '裁判（研究）'
SOURCE_OK = 'football-data'          # 只用这个来源：懂球帝来源和杯赛的角球、黄牌是 0，不是真的 0 个而是没有数据
TARGETS = {'corners': ('home_corners', 'away_corners'), 'yellow': ('home_yellow', 'away_yellow')}
TARGET_NAMES = {'corners': '角球', 'yellow': '黄牌'}
M_BASE, M_TEAM, M_NB, M_REF = 'c0_league', 'c1_team', 'c2_team_nb', 'c3_ref'
ALPHA_MIN = 1e-6                      # 离散参数小于这个就当泊松
# 事先定的 5 个主比较：(目标, 旧, 新, 只看哪个联赛)；差 = 新 − 旧，负数 = 新的更准
MAIN_COMPARISONS = [
    ('corners', M_BASE, M_TEAM, None),
    ('yellow', M_BASE, M_TEAM, None),
    ('corners', M_TEAM, M_NB, None),
    ('yellow', M_TEAM, M_NB, None),
    ('yellow', M_TEAM, M_REF, 'E0'),
]


# ======================= 计数分布和拟合 =======================
def count_nll(y, mu, alpha) -> np.ndarray:
    '''每场的计数 log loss：−log P(总数 = y)。alpha ≤ ALPHA_MIN 用泊松，否则用负二项（方差 = mu + alpha × mu²）。'''
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), y.shape)
    pois = mu - y * np.log(mu) + gammaln(y + 1)
    r = 1.0 / np.where(alpha > ALPHA_MIN, alpha, 1.0)
    nb = -(gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
    return np.where(alpha > ALPHA_MIN, nb, pois)


def nb_alpha(y, mu, w) -> float:
    '''矩估计负二项的离散参数：方差 − 均值 = alpha × 均值²，按权重加总。小于 0（比泊松还集中）就记 0。'''
    y, mu, w = (np.asarray(a, dtype=float) for a in (y, mu, w))
    return float(max(0.0, np.sum(w * ((y - mu) ** 2 - mu)) / np.sum(w * mu ** 2)))


def _fit_poisson(hi, ai, yh, ya, w, n_teams, lam, ref_idx, n_refs, lam_ref):
    '''带权重、带岭惩罚的泊松回归（球队的进攻倾向、被对手拿到的倾向、可选的裁判项），用 L-BFGS 解。

    参数顺序：[截距, 主场项, 每队倾向 × T, 每队让对手拿到的倾向 × T, 每个裁判 × n_refs]。
    ref_idx 里 −1 表示这场没有裁判项（裁判太少场或没记录），按 0 算。
    '''
    T = n_teams
    ridx = np.where(ref_idx < 0, n_refs, ref_idx)

    def fun(p):
        c, h, att, de, ref = p[0], p[1], p[2:2 + T], p[2 + T:2 + 2 * T], p[2 + 2 * T:]
        r = np.append(ref, 0.0)[ridx]
        eh = c + h + att[hi] + de[ai] + r
        ea = c + att[ai] + de[hi] + r
        mh, ma = np.exp(eh), np.exp(ea)
        f = np.sum(w * (mh - yh * eh)) + np.sum(w * (ma - ya * ea))
        f += 0.5 * lam * (att @ att + de @ de) + 0.5 * lam_ref * (ref @ ref)
        gh, ga = w * (mh - yh), w * (ma - ya)
        g = np.empty_like(p)
        g[0] = gh.sum() + ga.sum()
        g[1] = gh.sum()
        g[2:2 + T] = np.bincount(hi, gh, T) + np.bincount(ai, ga, T) + lam * att
        g[2 + T:2 + 2 * T] = np.bincount(ai, gh, T) + np.bincount(hi, ga, T) + lam * de
        if n_refs:
            g[2 + 2 * T:] = np.bincount(ridx, gh + ga, n_refs + 1)[:n_refs] + lam_ref * ref
        return f, g

    p0 = np.zeros(2 + 2 * T + n_refs)
    p0[0] = np.log(max(float(np.average((yh + ya) / 2.0, weights=w)), 0.05))
    res = minimize(fun, p0, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
    if not np.isfinite(res.x).all():
        raise ValueError('泊松回归没有解出有限的数')
    p = res.x
    return float(p[0]), float(p[1]), p[2:2 + T].copy(), p[2 + T:2 + 2 * T].copy(), p[2 + 2 * T:].copy()


@dataclass
class Fit:
    '''一个联赛、一个目标（角球或黄牌）、一次拟合的结果。'''
    teams: pd.Index
    refs: pd.Index
    c: float
    h: float
    att: np.ndarray
    de: np.ndarray
    ref: np.ndarray
    ref_n: np.ndarray
    alpha: float


def fit_league(train: pd.DataFrame, target: str, cutoff_ts: pd.Timestamp, cfg: dict, lam: float,
               use_ref: bool = False) -> Fit:
    '''用 train（这个联赛截止前最近的有数据的比赛）拟合。权重按开球离截止时间的天数指数衰减（半衰期 half_life_days）。

    use_ref：加裁判项。训练窗口里不到 ref_min_matches 场的裁判不给参数（裁判项取 0）。
    '''
    hc, ac = TARGETS[target]
    home, away = train['home_team'].to_numpy(), train['away_team'].to_numpy()
    teams = pd.Index(sorted(set(home) | set(away)))
    hi, ai = teams.get_indexer(home), teams.get_indexer(away)
    age = (cutoff_ts - pd.to_datetime(train['kickoff_earliest_utc'])).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (np.clip(age, 0.0, None) / float(cfg['half_life_days']))
    yh, ya = train[hc].to_numpy(float), train[ac].to_numpy(float)
    refs = pd.Index([], dtype=object)
    ref_idx = np.full(len(train), -1)
    ref_n = np.zeros(0)
    if use_ref:
        counts = train['referee'].value_counts()
        keep = counts[counts >= int(cfg['ref_min_matches'])]
        refs = pd.Index(sorted(keep.index), dtype=object)
        ref_idx = refs.get_indexer(train['referee'].astype(object).to_numpy())
        ref_n = keep.loc[list(refs)].to_numpy(float) if len(refs) else np.zeros(0)
    lam_ref = float(cfg['ref_ridge']) if use_ref else 0.0
    c, h, att, de, ref = _fit_poisson(hi, ai, yh, ya, w, len(teams), float(lam), ref_idx, len(refs), lam_ref)
    # 训练行的拟合均值，用来估负二项的离散参数
    r = np.append(ref, 0.0)[np.where(ref_idx < 0, len(refs), ref_idx)]
    mu = np.exp(c + h + att[hi] + de[ai] + r) + np.exp(c + att[ai] + de[hi] + r)
    return Fit(teams, refs, c, h, att, de, ref, ref_n, nb_alpha(yh + ya, mu, w))


def predict_means(fit: Fit, home, away, referee=None) -> np.ndarray:
    '''预测两队合计的均值。训练里没出现过的球队（新球队）倾向取 0；没有裁判项的裁判取 0。'''
    T = len(fit.teams)
    hi, ai = fit.teams.get_indexer(home), fit.teams.get_indexer(away)
    hi, ai = np.where(hi < 0, T, hi), np.where(ai < 0, T, ai)
    att, de = np.append(fit.att, 0.0), np.append(fit.de, 0.0)
    r = 0.0
    if referee is not None and len(fit.refs):
        ri = fit.refs.get_indexer(np.asarray(referee, dtype=object))
        r = np.append(fit.ref, 0.0)[np.where(ri < 0, len(fit.refs), ri)]
    mh = np.exp(fit.c + fit.h + att[hi] + de[ai] + r)
    ma = np.exp(fit.c + att[ai] + de[hi] + r)
    return mh + ma


# ======================= 取数和滚动预测 =======================
def season_targets(fs: FeatureStore, leagues: list[str], first: int, last: int, usage: str) -> pd.DataFrame:
    '''first–last 赛季（含两头）这些联赛的全部比赛，当作预测目标（和回测 season_targets 一样的窗口，但用途是研究实验）。
    开发模式下赛季范围里有考试赛季，取数层会直接报错。'''
    b = fs.settings['backtest']
    ws, we = b['window_start_md'], b['window_end_md']
    t = fs.prediction_targets(f'{first}-{ws}', f'{last + 1}-{we}', usage=usage, competition=leagues)
    t = t[(t['season'] >= first) & (t['season'] <= last)]
    return t.reset_index(drop=True)


@dataclass
class WalkResult:
    preds: pd.DataFrame                 # 每场 × 目标 × 模型一行：match_id、target、model、ridge、mu、alpha
    infos: pd.DataFrame                 # 每次拟合一行：联赛、目标、岭惩罚、主场倍数、离散参数
    pending: pd.DataFrame               # c3 要等答案取回来、接上本场裁判以后才能算：match_id、主客队、对应哪次拟合
    ref_fits: dict = field(default_factory=dict)
    ref_last: Fit | None = None         # 最后一次英超黄牌裁判拟合（报告里列裁判项大小）
    n_blocks: int = 0
    seconds: float = 0.0
    n_old_source_rows: int = 0          # 历史里来源不是 football-data 的行数（不用，报告里写出来）


def walk(fs: FeatureStore, cfg: dict, targets: pd.DataFrame, ridges: dict, with_nb: bool, with_ref: bool) -> WalkResult:
    '''按时间一块一块往前走：每块用截止时间之前的历史拟合，给这一块里的比赛出预测。ridges：目标 → 要跑的岭惩罚列表。'''
    t0 = time.time()
    usage = cfg['usage']
    data = [CORNERS_DATA] + ([REFEREE_DATA] if with_ref else [])
    leagues = sorted(targets['competition'].unique())
    t = targets.sort_values(['kickoff_earliest_utc', 'match_id']).reset_index(drop=True)
    kick = pd.to_datetime(t['kickoff_earliest_utc'])
    t['block'] = ((kick - kick.min().normalize()).dt.days // int(cfg['refit_days'])).astype(int)
    hist, prev_cutoff = None, None
    preds, infos, pending, ref_fits = [], [], [], {}
    ref_last = None
    n_blocks = 0
    for b, blk in t.groupby('block', sort=True):
        n_blocks += 1
        # 这一块里最早的一场开球前 horizon_hours 小时 = 整块的截止时间
        cutoff = fs.cutoff_before(int(blk.iloc[0]['match_id']), cfg['horizon_hours'], usage)
        since = None if prev_cutoff is None else pd.Timestamp(prev_cutoff) - pd.Timedelta(days=9)   # 增量取，往回多取 9 天
        new = fs.historical_matches(cutoff, usage, competition=leagues, data=data, since=since)
        if (new['known_at_utc'] > cutoff).any():
            raise LookAheadError(f'取到了截止时间 {cutoff} 之后才能知道的比赛，不允许')
        hist = new if hist is None else pd.concat([hist, new], ignore_index=True)
        hist = (hist.drop_duplicates('match_id', keep='last')
                .sort_values(['kickoff_earliest_utc', 'match_id']).reset_index(drop=True))
        if hist['match_id'].isin(blk['match_id']).any():
            raise LookAheadError('要预测的比赛出现在历史里了，不允许')
        prev_cutoff = cutoff
        cutoff_ts = pd.Timestamp(cutoff)
        for lg, mt in blk.groupby('competition'):
            home, away = mt['home_team'].to_numpy(), mt['away_team'].to_numpy()
            for target, (hc, ac) in TARGETS.items():
                rows = hist[(hist['competition'] == lg) & (hist['stats_source'] == SOURCE_OK)
                            & hist[hc].notna() & hist[ac].notna()]
                if len(rows) < int(cfg['min_train_matches']):
                    continue          # 这个联赛有数的历史不够，不出预测
                base = rows.tail(int(cfg['baseline_matches']))
                mu0 = float((base[hc] + base[ac]).mean())
                for mid in mt['match_id']:
                    preds.append({'match_id': int(mid), 'target': target, 'model': M_BASE, 'ridge': np.nan,
                                  'mu': mu0, 'alpha': 0.0})
                train = rows.tail(int(cfg['train_matches']))
                for lam in ridges[target]:
                    fit = fit_league(train, target, cutoff_ts, cfg, lam)
                    mu = predict_means(fit, home, away)
                    infos.append({'block': int(b), 'competition': lg, 'target': target, 'ridge': float(lam),
                                  'home_mult': float(np.exp(fit.h)), 'alpha': fit.alpha})
                    for mid, m in zip(mt['match_id'], mu):
                        preds.append({'match_id': int(mid), 'target': target, 'model': M_TEAM, 'ridge': float(lam),
                                      'mu': float(m), 'alpha': 0.0})
                        if with_nb:
                            preds.append({'match_id': int(mid), 'target': target, 'model': M_NB, 'ridge': float(lam),
                                          'mu': float(m), 'alpha': fit.alpha})
                    if with_ref and target == 'yellow' and lg == cfg['referee_league']:
                        fit_r = fit_league(train, target, cutoff_ts, cfg, lam, use_ref=True)
                        key = f'{b}|{lg}'
                        ref_fits[key] = fit_r
                        ref_last = fit_r
                        for mid, h_, a_ in zip(mt['match_id'], home, away):
                            pending.append({'match_id': int(mid), 'home': h_, 'away': a_, 'key': key,
                                            'ridge': float(lam)})
        if n_blocks % 50 == 0:
            log.info('已走 %d 块，到 %s，用时 %.0f 秒', n_blocks, cutoff, time.time() - t0)
    n_old = int((hist['stats_source'] != SOURCE_OK).sum()) if hist is not None and 'stats_source' in hist else 0
    return WalkResult(pd.DataFrame(preds), pd.DataFrame(infos), pd.DataFrame(pending), ref_fits, ref_last,
                      n_blocks, time.time() - t0, n_old)


def fetch_outcomes(fs: FeatureStore, cfg: dict, targets: pd.DataFrame, with_ref: bool = False) -> pd.DataFrame:
    '''全部预测做完以后，站在最后一场开球后 eval_after_kickoff_hours 小时，用途“复盘”取本场的角球、黄牌（答案）和裁判。'''
    hours = float(fs.settings['backtest']['eval_after_kickoff_hours'])
    kick = pd.to_datetime(targets['kickoff_earliest_utc'])
    cutoff = kick.max() + pd.Timedelta(hours=hours)
    since = kick.min() - pd.Timedelta(days=2)
    data = [CORNERS_DATA] + ([REFEREE_DATA] if with_ref else [])
    leagues = sorted(targets['competition'].unique())
    return fs.historical_matches(cutoff, '复盘', competition=leagues, data=data, since=since)


def add_ref_predictions(w: WalkResult, outcomes: pd.DataFrame) -> pd.DataFrame:
    '''c3：用当时拟合好的裁判项，接上本场裁判（答案取回来以后才有，需求文档假设 1：假设开球前已经知道本场裁判）。'''
    if w.pending.empty or 'referee' not in outcomes.columns:
        return w.preds
    ref = outcomes.drop_duplicates('match_id').set_index('match_id')['referee']
    rows = []
    for r in w.pending.itertuples(index=False):
        referee = ref.get(r.match_id, None)
        fit = w.ref_fits[r.key]
        mu = predict_means(fit, np.array([r.home], dtype=object), np.array([r.away], dtype=object),
                           np.array([referee], dtype=object))[0]
        rows.append({'match_id': r.match_id, 'target': 'yellow', 'model': M_REF, 'ridge': r.ridge,
                     'mu': float(mu), 'alpha': 0.0})
    return pd.concat([w.preds, pd.DataFrame(rows)], ignore_index=True)


# ======================= 打分和比较 =======================
def score(preds: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    '''把预测和答案接起来，算每场的计数 log loss。返回（打分表，每个目标的排除情况）。
    答案只用 football-data 来源、两队都有数的比赛；没有答案的比赛被排除（数量写进报告）。'''
    parts = []
    for tg, (hc, ac) in TARGETS.items():
        o = outcomes[(outcomes['stats_source'] == SOURCE_OK) & outcomes[hc].notna() & outcomes[ac].notna()]
        parts.append(pd.DataFrame({'match_id': o['match_id'], 'target': tg, 'y': (o[hc] + o[ac]).astype(float),
                                   'competition': o['competition'], 'season': o['season'],
                                   'match_date': o['match_date']}))
    y = pd.concat(parts, ignore_index=True)
    merged = preds.merge(y, on=['match_id', 'target'], how='inner')
    merged['nll'] = count_nll(merged['y'], merged['mu'], merged['alpha'])
    n_pred = preds.groupby('target')['match_id'].nunique()
    n_scored = merged.groupby('target')['match_id'].nunique()
    excluded = {tg: {'有预测的场数': int(n_pred.get(tg, 0)), '有答案、打了分的场数': int(n_scored.get(tg, 0)),
                     '没有答案被排除的场数': int(n_pred.get(tg, 0) - n_scored.get(tg, 0))} for tg in TARGETS}
    return merged, excluded


def choose_ridge(scored: pd.DataFrame, grid: list[float]) -> tuple[dict, pd.DataFrame]:
    '''选岭惩罚：每个目标里，c1 在三档上都有预测的同一批比赛，平均计数 log loss 最小的一档（并列取小的）。'''
    chosen, rows = {}, []
    for tg in TARGETS:
        s = scored[(scored['target'] == tg) & (scored['model'] == M_TEAM)]
        if s.empty:
            raise ValueError(f'选岭惩罚的赛季里没有「{TARGET_NAMES[tg]}」的预测和答案，没法选')
        pv = s.pivot_table(index='match_id', columns='ridge', values='nll').dropna()
        means = pv.mean()
        chosen[tg] = float(min(means.index, key=lambda k: (round(float(means[k]), 12), float(k))))
        for k in sorted(means.index):
            rows.append({'target': tg, 'ridge': float(k), 'n': len(pv), 'mean_nll': float(means[k]),
                         'chosen': float(k) == chosen[tg]})
    return chosen, pd.DataFrame(rows)


def final_label(v95: str, v99: str) -> str:
    '''按事先定的判定规则：95% 和 99% 区间都在同一边才下结论，只有 95% 的写成可能是巧合。'''
    if v95 == BETTER and v99 == BETTER:
        return '更准'
    if v95 == BETTER:
        return '可能更准，也可能是巧合'
    if v95 == WORSE and v99 == WORSE:
        return '更差'
    if v95 == WORSE:
        return '可能更差，也可能是巧合'
    return '没差别'


def pair_frame(scored: pd.DataFrame, target: str, base: str, new: str, competition: str | None = None) -> pd.DataFrame:
    '''两个模型都有分数的比赛，每场一行：差 = 新 − 旧。'''
    s = scored[scored['target'] == target]
    if competition is not None:
        s = s[s['competition'] == competition]
    a = s[s['model'] == base]
    b = s[s['model'] == new]
    if a['match_id'].duplicated().any() or b['match_id'].duplicated().any():
        raise ValueError('同一场同一个模型有两行分数（岭惩罚只能留一档）')
    a, b = a.set_index('match_id'), b.set_index('match_id')
    common = a.index.intersection(b.index)
    return pd.DataFrame({'match_id': common, 'competition': a.loc[common, 'competition'].to_numpy(),
                         'season': a.loc[common, 'season'].to_numpy(), 'match_date': a.loc[common, 'match_date'].to_numpy(),
                         'base_nll': a.loc[common, 'nll'].to_numpy(), 'new_nll': b.loc[common, 'nll'].to_numpy(),
                         'diff': (b.loc[common, 'nll'] - a.loc[common, 'nll']).to_numpy()})


def compare_pair(scored: pd.DataFrame, target: str, base: str, new: str, settings: dict, cfg: dict,
                 competition: str | None = None) -> dict:
    '''一个比较：平均差、95% 和 99% 的成块 bootstrap 区间、事先定的结论。'''
    p = pair_frame(scored, target, base, new, competition)
    if p.empty:
        raise ValueError(f'没有可比较的比赛：{target} {base} 对 {new} {competition or ""}')
    c = settings['compare']
    blocks = block_ids(p['match_date'], c['block'])
    row = {'target': target, 'base': base, 'new': new, 'competition': competition or '全部', 'n': int(len(p)),
           'blocks': int(len(np.unique(blocks))), 'base_mean': float(p['base_nll'].mean()),
           'new_mean': float(p['new_nll'].mean()), 'diff': float(p['diff'].mean())}
    verdicts = []
    for level in cfg['confidence_levels']:
        _, lo, hi = bootstrap_mean_diff(p['diff'], blocks, int(c['n_boot']), float(level), int(c['seed']))
        tag = int(round(float(level) * 100))
        row[f'ci{tag}_low'], row[f'ci{tag}_high'], row[f'verdict{tag}'] = lo, hi, verdict(lo, hi)
        verdicts.append(row[f'verdict{tag}'])
    row['结论'] = final_label(verdicts[0], verdicts[1])
    return row


def calibration_table(scored: pd.DataFrame, target: str, model: str, bins: int) -> pd.DataFrame:
    '''校准：按预测均值从小到大分成几档，每档的平均预测和实际平均。'''
    s = scored[(scored['target'] == target) & (scored['model'] == model)]
    q = pd.qcut(s['mu'], int(bins), duplicates='drop')
    g = s.groupby(q, observed=True).agg(n=('y', 'size'), 平均预测=('mu', 'mean'), 实际平均=('y', 'mean'))
    g.insert(0, 'target', target)
    return g.reset_index(drop=True)


def dispersion_table(scored: pd.DataFrame, model: str = M_TEAM) -> pd.DataFrame:
    '''加了球队倾向以后剩下的离散：皮尔逊残差平方的平均（泊松应该约等于 1）、矩估计的 alpha、实际总数自己的方差 / 均值。'''
    rows = []
    for tg in TARGETS:
        s = scored[(scored['target'] == tg) & (scored['model'] == model)]
        if s.empty:
            continue
        y, mu = s['y'].to_numpy(), s['mu'].to_numpy()
        rows.append({'target': tg, 'n': len(s), '实际均值': float(y.mean()), '实际方差除以均值': float(y.var(ddof=1) / y.mean()),
                     '皮尔逊残差平方平均': float(np.mean((y - mu) ** 2 / mu)),
                     'alpha矩估计': nb_alpha(y, mu, np.ones(len(y)))})
    return pd.DataFrame(rows)


def analyze(scored: pd.DataFrame, w: WalkResult, settings: dict, cfg: dict) -> dict:
    '''事先定的 5 个主比较，加只报告的分联赛、分赛季、校准、离散、主场项、裁判项。'''
    main = pd.DataFrame([compare_pair(scored, tg, b, n, settings, cfg, comp)
                         for tg, b, n, comp in MAIN_COMPARISONS
                         if not pair_frame(scored, tg, b, n, comp).empty])
    by_league, by_season = [], []
    for tg in TARGETS:
        for lg in sorted(scored['competition'].unique()):
            if not pair_frame(scored, tg, M_BASE, M_TEAM, lg).empty:
                by_league.append(compare_pair(scored, tg, M_BASE, M_TEAM, settings, cfg, lg))
        p = pair_frame(scored, tg, M_BASE, M_TEAM)
        g = p.groupby('season').agg(n=('diff', 'size'), diff=('diff', 'mean')).reset_index()
        g.insert(0, 'target', tg)
        by_season.append(g)
    calib = pd.concat([calibration_table(scored, tg, M_TEAM, cfg['calibration_bins']) for tg in TARGETS
                       if not scored[(scored['target'] == tg) & (scored['model'] == M_TEAM)].empty], ignore_index=True)
    home = (w.infos.groupby(['competition', 'target'])[['home_mult', 'alpha']].mean().reset_index()
            if not w.infos.empty else pd.DataFrame())
    ref_tab = pd.DataFrame()
    if w.ref_last is not None and len(w.ref_last.refs):
        f = w.ref_last
        ref_tab = pd.DataFrame({'裁判': list(f.refs), '裁判项倍数': np.exp(f.ref), '训练场数': f.ref_n}).sort_values('裁判项倍数')
    return {'main': main, 'by_league': pd.DataFrame(by_league), 'by_season': pd.concat(by_season, ignore_index=True),
            'calibration': calib, 'dispersion': dispersion_table(scored), 'home_effect': home, 'referees': ref_tab}


# ======================= 总流程和报告 =======================
def check_seasons(settings: dict, select: list[int], evaluate: list[int]) -> None:
    '''选的赛季和评的赛季都必须是开发赛季（settings 的 seasons），而且选在评之前。期末考试赛季一律不碰。'''
    for first, last in (select, evaluate):
        if first > last or season_role(int(first), settings) != 'dev' or season_role(int(last), settings) != 'dev':
            raise ValueError(f'赛季范围 {first}–{last} 不是开发赛季（只能用 {settings["seasons"]["dev_last"]} 及以前）')
    if select[1] >= evaluate[0]:
        raise ValueError('选岭惩罚的赛季必须在评的赛季之前')


def run_study(fs: FeatureStore, cfg: dict | None = None, leagues: list[str] | None = None,
              select: list[int] | None = None, evaluate: list[int] | None = None) -> dict:
    '''跑完两个阶段，返回全部结果（表都是 DataFrame）。'''
    settings = fs.settings
    cfg = dict(cfg or settings['corners_cards'])
    leagues = list(leagues or cfg['leagues'])
    select = list(select or cfg['select_seasons'])
    evaluate = list(evaluate or cfg['eval_seasons'])
    check_seasons(settings, select, evaluate)
    usage = cfg['usage']

    log.info('阶段一：%s–%s 赛季选岭惩罚（%s）', select[0], select[1], cfg['ridge_grid'])
    t1 = season_targets(fs, leagues, select[0], select[1], usage)
    w1 = walk(fs, cfg, t1, {tg: list(cfg['ridge_grid']) for tg in TARGETS}, with_nb=False, with_ref=False)
    s1, _ = score(w1.preds, fetch_outcomes(fs, cfg, t1))
    chosen, sel_table = choose_ridge(s1, cfg['ridge_grid'])
    log.info('选定的岭惩罚：%s', chosen)

    log.info('阶段二：%s–%s 赛季评', evaluate[0], evaluate[1])
    t2 = season_targets(fs, leagues, evaluate[0], evaluate[1], usage)
    with_ref = cfg['referee_league'] in leagues
    w2 = walk(fs, cfg, t2, {tg: [chosen[tg]] for tg in TARGETS}, with_nb=True, with_ref=with_ref)
    out2 = fetch_outcomes(fs, cfg, t2, with_ref=with_ref)
    s2, excluded = score(add_ref_predictions(w2, out2), out2)
    res = analyze(s2, w2, settings, cfg)
    res.update({'selection': sel_table, 'chosen_ridge': chosen, 'excluded': excluded, 'scored': s2,
                'meta': {'leagues': leagues, 'select_seasons': select, 'eval_seasons': evaluate,
                         '阶段一块数': w1.n_blocks, '阶段二块数': w2.n_blocks,
                         '用时秒': round(w1.seconds + w2.seconds, 1), '历史里来源不是football-data的行数': w2.n_old_source_rows}})
    return res


def format_report(res: dict) -> str:
    '''把主要结果写成一段中文，给不会编程的人看。'''
    m = res['meta']
    lines = ['第 10 阶段：角球、黄牌研究结果', '',
             f'联赛 {m["leagues"]}；选岭惩罚 {m["select_seasons"][0]}–{m["select_seasons"][1]}；评 {m["eval_seasons"][0]}–{m["eval_seasons"][1]}。',
             f'选定的岭惩罚：角球 {res["chosen_ridge"]["corners"]}，黄牌 {res["chosen_ridge"]["yellow"]}。', '',
             '事先定的 5 个主比较（差 = 新 − 旧，负数 = 新的更准；计数 log loss）：']
    for r in res['main'].itertuples(index=False):
        lines.append(f'- {TARGET_NAMES[r.target]}（{r.competition}）{r.new} 对 {r.base}：n={r.n}，差 {r.diff:+.4f}，'
                     f'95% 区间 {r.ci95_low:+.4f} 到 {r.ci95_high:+.4f}，99% 区间 {r.ci99_low:+.4f} 到 {r.ci99_high:+.4f} → {r.结论}')
    lines += ['', '被排除的场数（没有答案）：']
    for tg, d in res['excluded'].items():
        lines.append(f'- {TARGET_NAMES[tg]}：{d}')
    lines += ['', '以下只报告，不下结论（重复比较会碰巧选中，数字只当线索）：',
              '分联赛：' + '；'.join(f'{TARGET_NAMES[r.target]} {r.competition} 差 {r.diff:+.4f}（95% {r.ci95_low:+.4f} 到 {r.ci95_high:+.4f}）'
                                    for r in res['by_league'].itertuples(index=False)) if len(res['by_league']) else '分联赛：无']
    lines += ['', '离散程度：', res['dispersion'].to_string(index=False), '', '主场项倍数、离散（各联赛平均）：',
              res['home_effect'].to_string(index=False) if len(res['home_effect']) else '无']
    if len(res['referees']):
        r = res['referees']
        lines += ['', '裁判项最小的 5 位和最大的 5 位（最后一次拟合，带训练场数，样本小，只当线索）：',
                  pd.concat([r.head(5), r.tail(5)]).to_string(index=False)]
    lines += ['', '用时和块数：' + json.dumps(m, ensure_ascii=False)]
    return '\n'.join(lines)


def write_outputs(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ('main', 'by_league', 'by_season', 'calibration', 'dispersion', 'home_effect', 'referees', 'selection'):
        res[name].to_csv(out_dir / f'{name}.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    res['scored'].to_csv(out_dir / 'scored.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    (out_dir / 'report.txt').write_text(format_report(res), encoding='utf-8')
    (out_dir / 'numbers.json').write_text(json.dumps({'chosen_ridge': res['chosen_ridge'], 'excluded': res['excluded'],
                                                      'meta': res['meta']}, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='第 10 阶段：角球、黄牌研究（只用开发赛季，只出报告）')
    ap.add_argument('--leagues', nargs='+', help='只试跑这几个联赛（结果放 *_trial 文件夹）')
    ap.add_argument('--select', nargs=2, type=int, help='试跑：选岭惩罚的赛季范围')
    ap.add_argument('--eval', nargs=2, type=int, help='试跑：评的赛季范围')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    settings = load_settings()
    cfg = settings['corners_cards']
    fs = FeatureStore(open_readonly(), mode='开发', settings=settings)
    res = run_study(fs, cfg, a.leagues, a.select, a.eval)
    trial = bool(a.leagues or a.select or a.eval)
    out_dir = resolve_path(cfg['output_dir'] + ('_trial' if trial else ''))
    write_outputs(res, out_dir)
    print(format_report(res))
    print(f'{chr(10)}存在 {out_dir}')


if __name__ == '__main__':
    main()
