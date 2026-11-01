'''第 8 阶段：欧战实力表的比较和报告（需求文档第 4 节“第 8 阶段怎么做”和“第 8 阶段事先定的比较”）。

做什么：假装回到过去，每 refit_days（7）天走一步，和回测一样只往前走：
  1. 五大联赛的比分交给 dyn（settings 里 euro.dyn_model，没写就用 dyn_q1e5_b01）；
  2. 欧战比分（src/features/euro_matches.py 整理后）按当时 dyn 的评分记快照，重新估两版欧战实力表：
     加 δ 的版本，和 δ 全为 0 的对照版本（其余完全一样）；
  3. 开球前 24 小时（T-24）落在这一步和下一步之间的欧战比赛，用这一步的两版实力表出胜平负预测（只存档，不上线）；
  4. 五大联赛的比赛同样用这一步的 dyn 出预测，取预期净胜球，给“残差研究”用。
全部走完以后才拿答案（比分）打分。

数据全经取数层（用途“研究实验”、开发模式、阶段 8 只在研究程序自己手里的一份设置里）：
  - 每一步只取这一步截止时间之前已经能知道的联赛比分、欧战比分（取数层再按 since 取一段）；
  - 要预测的名单和答案，在走之前用同一个取数层一次取好（只拿来分步骤、事后打分，预测时不用）；
  - 开发模式取到 2024 赛季以后的数据会直接报错，所以不会碰期末考试赛季。
欧战评分的近似：欧战比赛在比分知道以后的那一步才记进表，评分用那一步 dyn 的评分，不是开球那一刻的（说明见 src/models/euro_strength.py）。

结果在 data/euro_study/（data/ 不进 Git）：euro_predictions.csv、residual_rows.csv、deltas.csv、report.txt。
用法（在项目文件夹里）：
  .venv\\Scripts\\python -m src.evaluate.euro_study
  快速试跑：--last 2010 --eval-first 2008
'''
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import resolve_path
from src.evaluate import metrics
from src.evaluate.backtest import season_targets
from src.evaluate.compare import block_ids, bootstrap_mean_diff, verdict
from src.features import FeatureStore, LookAheadError, open_readonly
from src.features.euro_matches import USAGE, fetch_euro, stage8_settings
from src.models.base import window_start
from src.models.euro_strength import OUTSIDE, EuroStrength, dyn_team_table
from src.models.registry import build_model

DEFAULT_OUT = 'data/euro_study'
DAY_NS = 86400.0 * 1e9


def _text(ts) -> str:
    return pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def leg_kind(rnd, leg) -> str:
    '''回合类型：小组赛 / 两回合（淘汰赛、资格赛的分回合） / 单场。'''
    if str(rnd).startswith('G'):
        return '小组赛'
    return '两回合' if int(leg) > 0 else '单场'


def window_freq(es: EuroStrength, cutoff: str) -> np.ndarray:
    '''欧战频率基线：窗口里欧战比赛的 主胜 / 平 / 客胜 比例（不分中立场、不加权）。一场都没有就是 1/3。'''
    start = window_start(cutoff, es.window_days)
    n = np.zeros(3)
    for r in es.rows:
        if start <= r[0] <= cutoff:
            n[0 if r[7] > r[8] else (1 if r[7] == r[8] else 2)] += 1
    return n / n.sum() if n.sum() > 0 else np.full(3, 1 / 3)


# ======================= 滚动：一步一步往前走 =======================
def run(settings=None, db_path=None, last=None, first_eval=None) -> dict:
    s = stage8_settings(settings)
    cfg = s['euro']
    leagues = list(cfg['rating_leagues'])
    dev_last = int(s['seasons']['dev_last'])
    last = dev_last if last is None else min(int(last), dev_last)
    eval_first = int(cfg['eval_first']) if first_eval is None else int(first_eval)
    first_season = int(cfg['first_season'])
    horizon = float(cfg['horizon_hours'])
    step_days = int(cfg['refit_days'])
    md = cfg['dev_end_month_day']
    t0 = time.perf_counter()
    fs = FeatureStore(open_readonly(db_path), '开发', s)
    end = f'{last + 1}-{md} 00:00:00'

    # ---- 走之前：名单和答案（取数层一次取好；预测时不用） ----
    full, drop_report = fetch_euro(fs, end)
    tgt = full[(full['season'] >= eval_first) & (full['season'] <= last)].copy()
    tgt['t24'] = [fs.cutoff_before(int(m), horizon, USAGE) for m in tgt['match_id']]
    lt = season_targets(fs, leagues, eval_first, last)[['match_id', 'competition', 'season', 'home_team', 'away_team']]
    ans = fs.historical_matches(end, USAGE, competition=leagues, data=['比分'], since=f'{eval_first}-07-01')
    lt = lt.merge(ans[['match_id', 'home_goals', 'away_goals']], on='match_id').dropna(subset=['home_goals', 'away_goals'])
    lt['t24'] = [fs.cutoff_before(int(m), horizon, 'A版输入') for m in lt['match_id']]

    grid = pd.date_range(f'{first_season}-08-01', pd.Timestamp(end), freq=f'{step_days}D')
    texts = [_text(g) for g in grid]
    grid_ns = grid.values.astype('datetime64[ns]')
    for df in (tgt, lt):
        t_ns = pd.to_datetime(df['t24']).values.astype('datetime64[ns]')
        df['step'] = np.searchsorted(grid_ns, t_ns, side='right') - 1
    tgt_by = {int(k): g for k, g in tgt[tgt['step'] >= 0].groupby('step')}
    lt_by = {int(k): g for k, g in lt[lt['step'] >= 0].groupby('step')}

    f = s['features']
    margin = pd.Timedelta(days=float(f['round_date_window_days']) + 2, hours=float(f['result_available_after_kickoff_hours']))
    back = pd.Timedelta(days=60)          # 欧战每步往回取 60 天（同一轮的重复行一定都在里面），已经记过的按 match_id 跳过
    dyn = build_model(cfg.get('dyn_model', 'dyn_q1e5_b01'), s)
    es = EuroStrength.from_settings(s, True)
    es0 = EuroStrength.from_settings(s, False)

    seen_l: set = set()
    since_l = f'{first_season}-07-01'
    pred_rows, res_rows, delta_log, dups = [], [], [], []
    skipped_pred = 0
    diff_m = None
    for k, (g, c) in enumerate(zip(grid, texts)):
        new = fs.historical_matches(c, USAGE, competition=leagues, data=['比分'], since=since_l)
        new = new[~new['match_id'].isin(seen_l)]
        if (new['known_at_utc'] > c).any():
            raise LookAheadError(f'取数层返回了截止时间 {c} 之后才能知道的联赛比赛，研究停止')
        seen_l.update(int(m) for m in new['match_id'])
        since_l = _text(g - margin)
        dyn.fit(new, c)
        table = dyn_team_table(dyn, leagues)

        raw, rep = fetch_euro(fs, c, USAGE, since=_text(g - back))
        if (raw['known_at_utc'] > c).any():
            raise LookAheadError(f'取数层返回了截止时间 {c} 之后才能知道的欧战比赛，研究停止')
        if not rep.empty:
            dups.append(rep)
        es.add(raw, table)
        es0.add(raw, table)
        es.fit(c)
        es0.fit(c)
        delta_log.append((c, es.deltas(), es.ready))

        grp = tgt_by.get(k)
        if grp is not None and es.ready and es0.ready:
            fq = window_freq(es, c)
            for r in grp.to_dict('records'):
                h, a, nt = r['home_team'], r['away_team'], bool(r['neutral'])
                p1 = es.result_probs(h, a, nt, table)
                p0 = es0.result_probs(h, a, nt, table)
                if p1 is None or p0 is None:
                    skipped_pred += 1
                    continue
                gh = table[h][0] if h in table else OUTSIDE
                ga = table[a][0] if a in table else OUTSIDE
                pred_rows.append({
                    'match_id': r['match_id'], 'competition': r['competition'], 'season': r['season'], 'round': r['round'],
                    'leg': r['leg'], 'kind': leg_kind(r['round'], r['leg']), 'neutral': nt, 'match_date': r['match_date'],
                    'home_team': h, 'away_team': a, 'g_home': gh, 'g_away': ga, 'cross': gh != ga,
                    'kickoff_earliest_utc': r['kickoff_earliest_utc'], 'known_at_utc': r['known_at_utc'], 't24': r['t24'],
                    'step_cutoff': c, 'home_goals': int(r['home_goals']), 'away_goals': int(r['away_goals']), 'result': r['result'],
                    'pH': p1[0], 'pD': p1[1], 'pA': p1[2], 'qH': p0[0], 'qD': p0[1], 'qA': p0[2],
                    'fH': fq[0], 'fD': fq[1], 'fA': fq[2],
                    'margin': es.expected_margin(h, a, nt, table), 'margin0': es0.expected_margin(h, a, nt, table)})

        lgrp = lt_by.get(k)
        if lgrp is not None:
            mats = dyn.predict(lgrp[['competition', 'season', 'home_team', 'away_team']], c)
            for r, m in zip(lgrp.to_dict('records'), mats):
                if m is None:
                    continue
                m = np.asarray(m)
                if diff_m is None:
                    diff_m = np.subtract.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
                res_rows.append({'match_id': r['match_id'], 'competition': r['competition'], 'season': r['season'],
                                 'home_team': r['home_team'], 'away_team': r['away_team'], 'home_goals': int(r['home_goals']),
                                 'away_goals': int(r['away_goals']), 't24': r['t24'], 'exp_margin': float((m * diff_m).sum())})
        if (k + 1) % 100 == 0:
            print(f'  走到第 {k + 1}/{len(grid)} 步（{c[:10]}），用时 {time.perf_counter() - t0:.0f} 秒', flush=True)

    dup_all = pd.concat(dups, ignore_index=True).drop_duplicates(['match_id', 'drop_reason']) if dups else pd.DataFrame()
    return {'settings': s, 'pred': pd.DataFrame(pred_rows), 'res': pd.DataFrame(res_rows), 'deltas': delta_log,
            'drop_report': drop_report, 'rolling_drops': dup_all, 'skipped_pred': skipped_pred, 'n_steps': len(grid),
            'eval_first': eval_first, 'last': last, 'es': es, 'es0': es0, 'n_euro_all': len(full),
            'n_targets': len(tgt), 'n_league_targets': len(lt), 'seconds': time.perf_counter() - t0}


# ======================= 主比较和分组 =======================
def main_compare(pred: pd.DataFrame, s: dict) -> pd.DataFrame:
    '''加 δ 对比 δ 全 0（差 = 加 δ − 对照，越小越好）。按周成块 bootstrap，区间整个在 0 的一边才算更准 / 更差。'''
    if pred.empty:
        return pd.DataFrame()
    c = s['compare']
    y = metrics.encode_results(pred['result'])
    new = pred[['pH', 'pD', 'pA']].to_numpy()
    base = pred[['qH', 'qD', 'qA']].to_numpy()
    blocks = block_ids(pred['match_date'], c['block'])
    rows = []
    for name, fn in (('log_loss', metrics.log_loss_per_match), ('brier', metrics.brier_per_match), ('rps', metrics.rps_per_match)):
        a, b = fn(base, y), fn(new, y)
        mean, lo, hi = bootstrap_mean_diff(b - a, blocks, c['n_boot'], c['confidence'], c['seed'])
        rows.append({'分数': name, '场数': len(pred), '块数': len(np.unique(blocks)), '对照平均': float(a.mean()),
                     '加δ平均': float(b.mean()), '差': mean, '区间下限': lo, '区间上限': hi, '结论': verdict(lo, hi),
                     '主要': name == c['primary_metric']})
    return pd.DataFrame(rows)


def breakdown(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    '''按某一列分组，只报平均 log loss 和差（不做区间：分组多了就是重复比较，只看看方向）。'''
    y = metrics.encode_results(pred['result'])
    base = metrics.log_loss_per_match(pred[['qH', 'qD', 'qA']].to_numpy(), y)
    new = metrics.log_loss_per_match(pred[['pH', 'pD', 'pA']].to_numpy(), y)
    t = pd.DataFrame({'组': pred[col].to_numpy(), 'base': base, 'new': new})
    o = t.groupby('组').agg(场数=('base', 'size'), 对照=('base', 'mean'), 加δ=('new', 'mean')).reset_index()
    o['差'] = o['加δ'] - o['对照']
    return o


def yearly_deltas(delta_log: list, years) -> tuple[pd.DataFrame, list]:
    '''每个赛季结束时（下一年 6 月 30 日前最后一步）的 δ，和相邻两年 δ 的相关系数（6 个数，只作参考）。'''
    rows = []
    for y in years:
        cut = f'{y + 1}-06-30 00:00:00'
        ok = [(c, d) for c, d, ready in delta_log if ready and c <= cut]
        if ok:
            rows.append({'年': y, **{k: round(v, 4) for k, v in ok[-1][1].items()}})
    df = pd.DataFrame(rows)
    corr = []
    if len(df) > 1:
        cols = list(df.columns[1:])
        for i in range(1, len(df)):
            a, b = df.iloc[i - 1][cols].to_numpy(float), df.iloc[i][cols].to_numpy(float)
            corr.append((int(df.iloc[i]['年']), float(np.corrcoef(a, b)[0, 1])))
    return df, corr


# ======================= 残差研究 =======================
def residual_rows(res: pd.DataFrame, pred: pd.DataFrame, windows, after_days: float, horizon_hours: float) -> pd.DataFrame:
    '''五大联赛每场每队一行：
      y = 实际净胜球 − dyn 预期净胜球（这队的角度）；
      x(w) = 这队 T-24 之前 w 天内欧战比赛“超预期净胜球”的平均（实际 − 欧战实力表赛前预期，这队的角度）；
      过去 max(w) 天没有欧战的行不进研究；after_euro = 这场联赛开球前 after_days 天内这队踢过欧战。
    只用 T-24 之前已经知道比分的欧战（known_at_utc ≤ T-24）。'''
    if res.empty or pred.empty:
        return pd.DataFrame()
    eu: dict = {}
    for r in pred.to_dict('records'):
        over = float(r['home_goals'] - r['away_goals']) - r['margin']
        ko = float(pd.Timestamp(r['kickoff_earliest_utc']).value)
        kn = float(pd.Timestamp(r['known_at_utc']).value)
        for team, v in ((r['home_team'], over), (r['away_team'], -over)):
            eu.setdefault(team, []).append((ko, kn, v))
    eu = {t: np.array(v, dtype=float) for t, v in eu.items()}
    wmax = max(windows)
    out = []
    for r in res.to_dict('records'):
        t = float(pd.Timestamp(r['t24']).value)
        ko_league = t + horizon_hours * 3600e9
        d = float(r['home_goals'] - r['away_goals']) - r['exp_margin']
        for team, yy in ((r['home_team'], d), (r['away_team'], -d)):
            e = eu.get(team)
            if e is None:
                continue
            known = (e[:, 0] <= t) & (e[:, 1] <= t) & (e[:, 0] > t - wmax * DAY_NS)
            if not known.any():
                continue
            row = {'match_id': r['match_id'], 'team': team, 'season': r['season'], 'date': str(r['t24'])[:10], 'y': yy}
            for w in windows:
                m = known & (e[:, 0] > t - w * DAY_NS)
                row[f'x{w}'] = float(e[m, 2].mean()) if m.any() else np.nan
            row['after_euro'] = bool((known & (e[:, 0] >= ko_league - after_days * DAY_NS)).any())
            out.append(row)
    return pd.DataFrame(out)


def slope_boot(x, y, blocks, n_boot: int, confidence: float, seed: int) -> tuple[float, float, float, float]:
    '''y = 截距 + b·x 的斜率 b，和按块 bootstrap 的区间。返回 (b, 截距, 下限, 上限)。'''
    x, y = np.asarray(x, float), np.asarray(y, float)
    _, inv = np.unique(np.asarray(blocks), return_inverse=True)
    k = int(inv.max()) + 1
    st = np.column_stack([np.bincount(inv, minlength=k).astype(float), np.bincount(inv, weights=x, minlength=k),
                          np.bincount(inv, weights=y, minlength=k), np.bincount(inv, weights=x * x, minlength=k),
                          np.bincount(inv, weights=x * y, minlength=k)])

    def slope(t):
        n, sx, sy, sxx, sxy = (t[..., i] for i in range(5))
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)

    b = float(slope(st.sum(axis=0)))
    a = float(y.mean() - b * x.mean())
    rng = np.random.default_rng(seed)
    boots = []
    for i in range(0, n_boot, 500):
        idx = rng.integers(0, k, size=(min(500, n_boot - i), k))
        boots.append(slope(st[idx].sum(axis=1)))
    boots = np.concatenate(boots)
    al = (1 - confidence) / 2
    lo, hi = np.nanquantile(boots, [al, 1 - al])
    return b, a, float(lo), float(hi)


def residual_summary(rows: pd.DataFrame, windows, primary: int, s: dict) -> pd.DataFrame:
    '''残差回归：主 = 过去 primary 天；另外分“欧战后 4 天内”和其余、窗口换成其他天数。
    区间整个在 0 的一边只算“有迹象”，不是证明（样本小、重复比较、相关不是因果）。'''
    if rows.empty:
        return pd.DataFrame()
    c = s['compare']
    allm = np.ones(len(rows), bool)
    ae = rows['after_euro'].to_numpy()
    items = [(f'过去{primary}天（主）', f'x{primary}', allm), (f'过去{primary}天，欧战后 4 天内', f'x{primary}', ae),
             (f'过去{primary}天，其余', f'x{primary}', ~ae)]
    items += [(f'过去{w}天', f'x{w}', allm) for w in windows if w != primary]
    out = []
    for label, col, mask in items:
        sub = rows[mask & rows[col].notna().to_numpy()]
        if len(sub) < 30:
            out.append({'设定': label, '行数': len(sub), '块数': 0, 'b': np.nan, '截距': np.nan, '下限': np.nan, '上限': np.nan, '结论': '行数太少，不报'})
            continue
        blocks = block_ids(sub['date'], c['block'])
        b, a, lo, hi = slope_boot(sub[col], sub['y'], blocks, c['n_boot'], c['confidence'], c['seed'])
        tag = '有迹象（b>0）' if lo > 0 else ('有迹象（b<0）' if hi < 0 else '没有迹象')
        out.append({'设定': label, '行数': len(sub), '块数': len(np.unique(blocks)), 'b': b, '截距': a, '下限': lo, '上限': hi, '结论': tag})
    return pd.DataFrame(out)


def analyze(out: dict) -> None:
    s, cfg = out['settings'], out['settings']['euro']
    windows = sorted({int(cfg['residual_days']), *[int(w) for w in cfg['residual_days_extra']]})
    out['windows'] = windows
    out['resid_rows'] = residual_rows(out['res'], out['pred'], windows, float(s['rest_days']['euro_within_days']), float(cfg['horizon_hours']))
    out['resid_summary'] = residual_summary(out['resid_rows'], windows, int(cfg['residual_days']), s)


# ======================= 报告 =======================
def build_report(out: dict) -> str:
    s, cfg, pred = out['settings'], out['settings']['euro'], out['pred']
    L: list = []
    add = L.append

    def table(df: pd.DataFrame) -> str:
        return df.round(4).to_string(index=False)

    add('第 8 阶段 欧战实力表研究报告（只在开发赛季；结果不上线，不能据此改 dyn、gb_l7、融合权重、倾向门槛、信心等级或页面）')
    add(f'估计从 {cfg["first_season"]}/{(cfg["first_season"] + 1) % 100:02d} 起，评 {out["eval_first"]}/{(out["eval_first"] + 1) % 100:02d} 到 {out["last"]}/{(out["last"] + 1) % 100:02d} 的欧战，T-24；走了 {out["n_steps"]} 步，用时 {out["seconds"]:.0f} 秒')
    add(f'取数层给出整理后的欧战比赛 {out["n_euro_all"]} 场（含评估赛季以前）；评估赛季名单 {out["n_targets"]} 场，出了预测 {len(pred)} 场，评分或实力表拿不到而跳过 {out["skipped_pred"]} 场')
    add('整理时去掉的行（没有比分、疑似重复）：')
    add(table(out['drop_report']) if not out['drop_report'].empty else '  无')
    add(f'滚动中（各步取数时）实际去掉的疑似重复行数：{int((out["rolling_drops"]["dropped"] == True).sum()) if not out["rolling_drops"].empty else 0}（去掉的行在上面表里逐对核对）')
    add('')
    if pred.empty:
        add('没有可比较的欧战比赛。')
        return chr(10).join(L)
    add('一、主比较：欧战胜平负，加 δ 对比 δ 全 0（差 = 加δ − 对照；区间整个小于 0 才算更准）')
    add(table(main_compare(pred, s)))
    y = metrics.encode_results(pred['result'])
    fl = float(metrics.log_loss_per_match(pred[['fH', 'fD', 'fA']].to_numpy(), y).mean())
    add(f'欧战频率基线（窗口里的主胜/平/客胜比例）log loss = {fl:.4f}')
    add(f'平局：实际占 {float((pred["result"] == "D").mean()):.4f}，加δ 平均预测 {float(pred["pD"].mean()):.4f}，对照平均预测 {float(pred["qD"].mean()):.4f}')
    add('')
    add('二、另外只报告（分组多了就是重复比较，只看方向，不下结论）')
    for title, col in (('分赛事', 'competition'), ('分回合类型', 'kind'), ('分赛季', 'season'),
                       ('两队是不是同一组（δ 只在不同组时起作用）', 'cross'), ('是不是推断的中立场', 'neutral')):
        add(title)
        add(table(breakdown(pred, col)))
    cr = pred[pred['cross']]
    if not cr.empty:
        add('分联赛（只看两队不同组的比赛，涉及这个联赛的算一次，所以一场会算两次）')
        add(table(breakdown(pd.concat([cr.assign(grp=cr['g_home']), cr.assign(grp=cr['g_away'])]), 'grp')))
    nn = pred[~pred['neutral']]
    if not nn.empty:
        add('去掉推断中立场后的主比较')
        add(table(main_compare(nn, s)))
    add('')
    add('三、δ（最后一步的估计，已减去五个联赛的平均；联赛外是比五联赛平均高多少）')
    add(str(out['es'].params()))
    dy, corr = yearly_deltas(out['deltas'], range(out['eval_first'] - 1, out['last'] + 1))
    if not dy.empty:
        add('每个赛季结束时的 δ：')
        add(table(dy))
        add('相邻赛季 δ 的相关系数（6 个数算出来，只作参考）：' + '、'.join(f'{y0}：{v:.2f}' for y0, v in corr))
    add('')
    add('四、残差研究（只报告：欧战超预期净胜球，和接下来联赛里的净胜球残差有没有关系）')
    rr = out.get('resid_rows')
    add(f'进入研究的行数：{0 if rr is None else len(rr)}（五大联赛每场每队一行，过去 {max(out["windows"])} 天没有欧战的行不进）')
    if rr is not None and not rr.empty:
        add(table(out['resid_summary']))
    add('')
    add('说明：判定只看第一、四部分的主比较和主残差；其余都是另外报告，不是判定。区间整个在一边也只是“有迹象”，样本小、做过重复比较、相关不是因果。')
    return chr(10).join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description='第 8 阶段：欧战实力表的比较和报告（需求文档第 4 节）')
    ap.add_argument('--out', default=DEFAULT_OUT, help='结果放哪（data/ 不进 Git）')
    ap.add_argument('--last', type=int, default=None, help='只跑到这个赛季（快速试跑用）')
    ap.add_argument('--eval-first', type=int, default=None, help='从这个赛季起评（快速试跑用）')
    a = ap.parse_args()
    out = run(last=a.last, first_eval=a.eval_first)
    analyze(out)
    text = build_report(out)
    d = Path(resolve_path(a.out))
    d.mkdir(parents=True, exist_ok=True)
    out['pred'].to_csv(d / 'euro_predictions.csv', index=False, encoding='utf-8-sig')
    if out['resid_rows'] is not None and not out['resid_rows'].empty:
        out['resid_rows'].to_csv(d / 'residual_rows.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'cutoff': c, 'ready': r, **dl} for c, dl, r in out['deltas']]).to_csv(d / 'deltas.csv', index=False, encoding='utf-8-sig')
    (d / 'report.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
