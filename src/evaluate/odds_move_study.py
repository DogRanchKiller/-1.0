'''赔率走向研究（需求文档第 14 节“赔率走向研究怎么做”：规则先写下，再写这个程序，再跑）。

一、要回答什么
  下注时间要看“赔率从现在到开球前会往哪走”（第 14 节）。这里先不做模型，只回答三件事：
  1. 价格往哪走：T-24 / T-6 / T-1 那一刻的价格，到开球前最后一个价格，热门的概率平均怎么变（ΔF）。
  2. 能不能事先看出来：T-24 时，（B）这家公司的价格比同场别家高多少，（C）热门有多强，能不能预测后来怎么走。
  3. 数据够不够：每个时点有多少场能用、价格多旧、“最后价格”离 football-data 的真收盘多远。
  判断标准在需求文档里事先写好，这里照做；不写下注建议，不改任何模型参数。

二、几个名词
  - 去水：胜平负三个赔率按比例去水（backtest.benchmark_devig_method），变成概率。
  - 共识概率：同一场、同一个时点，所有“T 时和最后都有价格”的公司，去水概率取平均（至少 min_books 家）。
  - 热门：T 时共识概率最大的那一边。ΔF = 最后价格的热门概率 − T 时的热门概率。
    ΔF > 0：热门的价格后来变差（早下划算）；ΔF < 0：变好（等划算）。
  - β = Σ(x·y) ÷ Σ(x·x)；区间用按周成块 bootstrap（和 compare.py、market_checks.py 一样）。

三、数据（全部经取数层）
  开发模式、用途“机器人下注”，只拿截止时间之前的变盘记录；考试赛季取不到（取数层会直接报错）。
  懂球帝回补的历史比赛每家公司只有“初盘”和“最新”两个价格，所以“T 时的价格”多半是初盘，结果要这样读。

用法：
  .venv/Scripts/python -m src.evaluate.odds_move_study
结果放在 data/backtest/odds_move/：
  coverage.csv 数据够不够；move.csv 价格往哪走；sides.csv 各边概率变化；beta.csv 能不能事先看出来；
  verdict.csv 三个主要结论的判断；close_check.csv 最后价格离真收盘多远；per_match.csv 每场每个时点一行；run_info.json
'''
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from src.common.config import load_settings, resolve_path
from src.common.odds import devig
from src.evaluate.compare import block_ids, bootstrap_mean_diff
from src.evaluate.devig_compare import dev_cutoff
from src.evaluate.market_checks import ratio_ci
from src.features import FeatureStore, open_readonly

ODDS = ['home', 'draw', 'away']
SIDES = ['主胜', '平', '客胜']
LONG_COLS = ['match_id', 'horizon', 'bookmaker', 't_home', 't_draw', 't_away', 't_at',
             'f_home', 'f_draw', 'f_away', 'f_at', 'first_at']
T_NAMES = {'home': 't_home', 'draw': 't_draw', 'away': 't_away', 'changed_at_utc': 't_at'}
F_NAMES = {'home': 'f_home', 'draw': 'f_draw', 'away': 'f_away', 'changed_at_utc': 'f_at'}
BETA_COLS = ['指标', '哪一份', '行数', 'β', '区间下限', '区间上限']
B_LABEL = 'B：这家公司比别家高多少 → 自己的热门价格后来怎么走'
C_LABEL = 'C：热门有多强 → 热门价格后来怎么走'
TITLES = {'coverage': '数据够不够', 'verdict': '三个主要结论', 'beta': '能不能事先看出来（β）',
          'move': '价格往哪走（ΔF）', 'sides': '各边概率变化', 'close_check': '最后价格离 football-data 真收盘多远'}


def trusted_books(settings):
    '''公司对照表里“依据”写核对过的公司名；名字不确定（推测）的公司不用。'''
    c = settings['odds_move_study']
    with open(resolve_path(c['bookmaker_file']), encoding='utf-8') as fh:
        raw = yaml.safe_load(fh)
    return sorted({v['名字'] for v in raw['公司'].values() if v.get('依据') == c['trusted_basis']})


def _valid(df):
    '''赔率缺一个、或不大于 1 的当作没有。'''
    df = df.dropna(subset=ODDS)
    return df[(df[ODDS] > 1).all(axis=1)]


# ======================= 取数 =======================
def collect(fs, settings, matches, books):
    '''每场、每个时点、每家公司一行：T 时的价格和开球前最后一个价格。

    全部经取数层：T 时的价格 = latest_odds（截止时间 = 开球前 h 小时）；最后价格 = 开球前 last_minutes 分钟的最后一个价格。
    取数层只给变盘时间早于开球的记录，开球后的价格拿不到；考试赛季的比赛在这里直接报错。
    '''
    c = settings['odds_move_study']
    usage = c['usage']
    parts = []
    for mid in matches['match_id']:
        mid = int(mid)
        last_cut = fs.cutoff_before(mid, c['last_minutes'] / 60, usage)
        ch = fs.odds_changes(mid, last_cut, usage, bookmaker=books, market='1x2')
        if ch.empty:
            continue
        by = ch.groupby('bookmaker')
        first_at = by['changed_at_utc'].min()          # 这家公司第一个价格的时间
        last = _valid(by.tail(1)).rename(columns=F_NAMES)[['bookmaker', 'f_home', 'f_draw', 'f_away', 'f_at']]
        for h in c['horizons']:
            cut = fs.cutoff_before(mid, h, usage)
            t = _valid(fs.latest_odds(mid, cut, usage, bookmaker=books, market='1x2'))
            t = t.rename(columns=T_NAMES)[['bookmaker', 't_home', 't_draw', 't_away', 't_at']]
            j = t.merge(last, on='bookmaker')
            if j.empty:
                continue
            j['match_id'] = mid
            j['horizon'] = int(h)
            j['first_at'] = j['bookmaker'].map(first_at)
            parts.append(j[LONG_COLS])
    if not parts:
        return pd.DataFrame(columns=LONG_COLS)
    return pd.concat(parts, ignore_index=True)


def consensus(long, matches, settings):
    '''返回 (公司级表, 场级表)。

    公司级：每家公司的去水概率、x1 = 这家公司的热门概率 − 同场别家的平均、y1 = 这家公司自己后来的变化。
    场级：共识概率（ct_ 是 T 时，cf_ 是最后价格）、热门 fav（0 主胜 / 1 平 / 2 客胜）、ΔF、各边变化 d_。
    同一场同一时点少于 min_books 家公司的不要。
    '''
    c = settings['odds_move_study']
    method = settings['backtest']['benchmark_devig_method']
    key = ['match_id', 'horizon']
    if long.empty:
        return long.copy(), pd.DataFrame()
    b = long.copy()
    for tag in ('t', 'f'):
        p = devig(b[[f'{tag}_home', f'{tag}_draw', f'{tag}_away']].to_numpy(float), method)
        for i, s in enumerate(ODDS):
            b[f'p{tag}_{s}'] = p[:, i]
    b = b[b.groupby(key)['bookmaker'].transform('size') >= c['min_books']].copy()
    if b.empty:
        return b, pd.DataFrame()
    pcols = [f'p{tag}_{s}' for tag in ('t', 'f') for s in ODDS]
    g = b.groupby(key)
    m = g[pcols].mean()
    m['n_books'] = g.size()
    m = m.reset_index().rename(columns={col: 'c' + col[1:] for col in pcols})
    ct = m[['ct_' + s for s in ODDS]].to_numpy(float)
    cf = m[['cf_' + s for s in ODDS]].to_numpy(float)
    fav = ct.argmax(axis=1)
    idx = np.arange(len(m))
    m['fav'] = fav
    m['ct_fav'] = ct[idx, fav]
    m['cf_fav'] = cf[idx, fav]
    m['dF'] = m['cf_fav'] - m['ct_fav']
    for i, s in enumerate(ODDS):
        m['d_' + s] = cf[:, i] - ct[:, i]
    info = matches[['match_id', 'season', 'competition', 'match_date', 'kickoff_earliest_utc']]
    m = m.merge(info, on='match_id', how='left')
    b = b.merge(m[key + ['fav', 'n_books']], on=key)
    pt = b[['pt_' + s for s in ODDS]].to_numpy(float)
    pf = b[['pf_' + s for s in ODDS]].to_numpy(float)
    fav_b = b['fav'].to_numpy()
    rows = np.arange(len(b))
    b['pt_fav'] = pt[rows, fav_b]
    b['pf_fav'] = pf[rows, fav_b]
    total = b.groupby(key)['pt_fav'].transform('sum')
    b['x1'] = (b['n_books'] * b['pt_fav'] - total) / (b['n_books'] - 1)   # 一场里各家的 x1 加起来是 0
    b['y1'] = b['pf_fav'] - b['pt_fav']
    b = b.merge(info[['match_id', 'season', 'match_date']], on='match_id', how='left')
    return b, m


# ======================= 统计 =======================
def boot_mean(x, dates, settings):
    '''平均数和 95% 区间（按周成块 bootstrap，设置和 compare.py 一样）。'''
    c = settings['compare']
    return bootstrap_mean_diff(np.asarray(x, float), block_ids(pd.Series(dates), c['block']),
                               c['n_boot'], c['confidence'], c['seed'])


def move_table(m, settings):
    '''问题 1：每个时点 × 分法（全部 / 联赛 / 赛季 / 热门是哪边）：ΔF 平均、区间、动了的比例、平均 |ΔF|。'''
    thr = settings['odds_move_study']['moved_threshold']
    rows = []
    for h, mh in m.groupby('horizon'):
        parts = [('全部', '全部', mh)]
        parts += [('联赛', k, s) for k, s in mh.groupby('competition')]
        parts += [('赛季', str(k), s) for k, s in mh.groupby('season')]
        parts += [('热门是哪边', SIDES[int(k)], s) for k, s in mh.groupby('fav')]
        for kind, name, s in parts:
            mean, lo, hi = boot_mean(s['dF'], s['match_date'], settings)
            rows.append({'时点': 'T-' + str(int(h)), '分法': kind, '哪一份': name, '场数': len(s),
                         'ΔF平均': mean, '区间下限': lo, '区间上限': hi,
                         '动了的比例': float((s['dF'].abs() >= thr).mean()), '平均|ΔF|': float(s['dF'].abs().mean())})
    return pd.DataFrame(rows)


def side_table(m, settings):
    '''问题 1 补充：主胜、平、客胜各自的共识概率平均变了多少（全部比赛）。'''
    rows = []
    for h, mh in m.groupby('horizon'):
        for i, s in enumerate(ODDS):
            mean, lo, hi = boot_mean(mh['d_' + s], mh['match_date'], settings)
            rows.append({'时点': 'T-' + str(int(h)), '哪一边': SIDES[i], '场数': len(mh),
                         '概率平均变化': mean, '区间下限': lo, '区间上限': hi})
    return pd.DataFrame(rows)


def beta_rows(label, x, y, dates, seasons, settings):
    '''β = Σ(x·y) ÷ Σ(x·x)：合并一行，每个赛季一行。区间按周成块 bootstrap（market_checks.ratio_ci）。'''
    df = pd.DataFrame({'x': np.asarray(x, float), 'y': np.asarray(y, float),
                       'date': np.asarray(dates), 'season': np.asarray(seasons)})
    parts = [('合并', df)] + [(str(k), s) for k, s in df.groupby('season')]
    rows = []
    for name, s in parts:
        den = (s['x'] ** 2).to_numpy()
        if den.sum() <= 0:
            continue
        beta, lo, hi = ratio_ci((s['x'] * s['y']).to_numpy(), den, pd.Series(s['date'].to_numpy()), settings)
        rows.append({'指标': label, '哪一份': name, '行数': len(s), 'β': beta, '区间下限': lo, '区间上限': hi})
    return pd.DataFrame(rows, columns=BETA_COLS)


def beta_table(b, m, settings):
    '''问题 2（只看主时点，默认 T-24）：B = 公司对别家的分歧 x1 → y1；C = 热门强度（居中）→ ΔF。'''
    h0 = settings['odds_move_study']['main_horizon']
    mh, bh = m[m['horizon'] == h0], b[b['horizon'] == h0]
    x2 = mh['ct_fav'] - mh['ct_fav'].mean()
    rb = beta_rows(B_LABEL, bh['x1'], bh['y1'], bh['match_date'], bh['season'], settings)
    rc = beta_rows(C_LABEL, x2, mh['dF'], mh['match_date'], mh['season'], settings)
    return pd.concat([rb, rc], ignore_index=True)


def judge(label, mean, lo, hi, by_season, counts, c, need_abs):
    '''判断标准（需求文档第 14 节，事先定）：
    数据够不够：够样本（场数 ≥ min_season_matches）的赛季不少于 min_seasons 个，而且合计 ≥ min_total_matches。
    看到规律要同时满足：① 合并区间整个在 0 的一边；② 够样本的每个赛季的值和合并的同号；③（只对 A）|平均| ≥ min_abs_mean。'''
    big = [s for s, n in counts.items() if n >= c['min_season_matches']]
    total = int(sum(counts.values()))
    out = {'结论项': label, '合并值': mean, '区间下限': lo, '区间上限': hi, '够样本的赛季数': len(big),
           '场数合计': total, '条件1区间不跨0': None, '条件2够样本赛季同号': None, '条件3够大': None}
    if len(big) < c['min_seasons'] or total < c['min_total_matches']:
        out['判断'] = '数据不够判断'
        return out
    ok1 = bool(lo > 0 or hi < 0)
    ok2 = bool(all(by_season.get(s, 0.0) * mean > 0 for s in big))
    ok3 = bool(abs(mean) >= c['min_abs_mean']) if need_abs else True
    out.update({'条件1区间不跨0': ok1, '条件2够样本赛季同号': ok2, '条件3够大': ok3})
    if ok1 and ok2 and ok3:
        out['判断'] = '看到规律（' + ('正' if mean > 0 else '负') + '向）'
    else:
        miss = [n for n, ok in (('①区间跨过 0', ok1), ('②有赛季方向相反', ok2), ('③平均太小', ok3)) if not ok]
        out['判断'] = '没看到可靠规律（没满足：' + '、'.join(miss) + '）'
    return out


def verdict_table(m, beta, settings):
    '''三个主要结论（A：T-24 的 ΔF；B、C：两个 β）各给一个判断。其余的表只是看看，不下结论。'''
    c = settings['odds_move_study']
    mh = m[m['horizon'] == c['main_horizon']]
    if mh.empty:
        return pd.DataFrame()
    counts = mh.groupby('season').size().to_dict()
    mean, lo, hi = boot_mean(mh['dF'], mh['match_date'], settings)
    rows = [judge('A：T-' + str(int(c['main_horizon'])) + ' 的 ΔF（热门价格后来往哪走）', mean, lo, hi,
                  mh.groupby('season')['dF'].mean().to_dict(), counts, c, True)]
    labels = beta['指标'].unique() if not beta.empty else []
    for label in labels:
        s = beta[beta['指标'] == label]
        whole, per = s[s['哪一份'] == '合并'], s[s['哪一份'] != '合并']
        if whole.empty:
            continue
        r = whole.iloc[0]
        by = {int(k): v for k, v in zip(per['哪一份'], per['β'])}
        rows.append(judge(label, r['β'], r['区间下限'], r['区间上限'], by, counts, c, False))
    return pd.DataFrame(rows)


def coverage_table(matches, long, m, settings):
    '''问题 3：每个时点 × 赛季，有多少场能用、T 时的价格是不是就是第一个价格、价格有多旧。'''
    c = settings['odds_move_study']
    kick = pd.to_datetime(matches.set_index('match_id')['kickoff_earliest_utc'])
    rows = []
    for h in c['horizons']:
        lh = long[long['horizon'] == h]
        used = set(m.loc[m['horizon'] == h, 'match_id']) if not m.empty else set()
        for season, ms in matches.groupby('season'):
            ids = set(ms['match_id'])
            l = lh[lh['match_id'].isin(ids)]
            lu = l[l['match_id'].isin(used)]
            if lu.empty:
                first = age_t = age_f = np.nan
            else:
                ko = lu['match_id'].map(kick)
                first = float((lu['t_at'] == lu['first_at']).mean())
                age_t = float(((ko - pd.to_datetime(lu['t_at'])).dt.total_seconds() / 3600).median())
                age_f = float(((ko - pd.to_datetime(lu['f_at'])).dt.total_seconds() / 3600).median())
            rows.append({'时点': 'T-' + str(int(h)), '赛季': int(season), '联赛比赛数': len(ms),
                         '至少一家有价格的场数': int(l['match_id'].nunique()), '能用的场数': len(ids & used),
                         'T时价格就是第一个价格的比例': first, 'T时价格的年龄中位数(小时)': age_t,
                         '最后价格的年龄中位数(小时)': age_f})
    return pd.DataFrame(rows)


def closing_check(fs, settings, long, matches, books):
    '''最后价格和 football-data 同名公司的收盘赔率比（用途“评估”）：三个赔率都差不超过 same_odds_tol 算一样。'''
    c = settings['odds_move_study']
    method = settings['backtest']['benchmark_devig_method']
    if long.empty:
        return pd.DataFrame()
    last = long.drop_duplicates(['match_id', 'bookmaker'])[['match_id', 'bookmaker', 'f_home', 'f_draw', 'f_away']]
    cutoff = dev_cutoff(settings)
    parts = []
    for mid, g in last.groupby('match_id'):
        close = _valid(fs.closing_odds(int(mid), cutoff, '评估', bookmaker=books, market='1x2'))
        if close.empty:
            continue
        parts.append(g.merge(close[['bookmaker', 'home', 'draw', 'away']], on='bookmaker'))
    if not parts:
        return pd.DataFrame()
    j = pd.concat(parts, ignore_index=True)
    fo = j[['f_home', 'f_draw', 'f_away']].to_numpy(float)
    co = j[ODDS].to_numpy(float)
    j['same'] = (np.abs(fo - co) <= c['same_odds_tol']).all(axis=1)
    j['max_diff'] = np.abs(devig(fo, method) - devig(co, method)).max(axis=1)
    j = j.merge(matches[['match_id', 'season']], on='match_id')
    out = j.groupby('season').agg(公司场数=('same', 'size'), 一样的比例=('same', 'mean'),
                                 概率最大差平均=('max_diff', 'mean')).reset_index()
    total = pd.DataFrame([{'season': '全部', '公司场数': len(j), '一样的比例': float(j['same'].mean()),
                           '概率最大差平均': float(j['max_diff'].mean())}])
    out = pd.concat([out.assign(season=out['season'].astype(str)), total], ignore_index=True)
    return out.rename(columns={'season': '赛季'})


# ======================= 总流程 =======================
def study(fs, settings, books=None):
    '''整套研究。fs 必须是开发模式的取数层。返回一组表（字典）。'''
    c = settings['odds_move_study']
    books = books or trusted_books(settings)
    first = c['first_season']
    matches = fs.prediction_targets(f'{first}-06-01 00:00:00', dev_cutoff(settings), c['usage'],
                                    competition=c['leagues'])
    keep = ((matches['season'] >= first) & (matches['season'] <= settings['seasons']['dev_last'])
            & matches['kickoff_utc'].notna())
    matches = matches[keep].reset_index(drop=True)
    long = collect(fs, settings, matches, books)
    b, m = consensus(long, matches, settings)
    empty = pd.DataFrame()
    beta = beta_table(b, m, settings) if not m.empty else empty
    return {
        'coverage': coverage_table(matches, long, m, settings),
        'verdict': verdict_table(m, beta, settings) if not m.empty else empty,
        'beta': beta,
        'move': move_table(m, settings) if not m.empty else empty,
        'sides': side_table(m, settings) if not m.empty else empty,
        'close_check': closing_check(fs, settings, long, matches, books),
        'per_match': m,
    }


def main():
    ap = argparse.ArgumentParser(description='赔率走向研究（需求文档第 14 节）')
    ap.add_argument('--db', default=None, help='数据库文件（不写就用 settings.yaml 的 paths.database）')
    ap.add_argument('--out', default=None, help='结果文件夹（不写就用 settings.yaml 的 odds_move_study.output_dir）')
    a = ap.parse_args()
    settings = load_settings()
    c = settings['odds_move_study']
    books = trusted_books(settings)
    fs = FeatureStore(open_readonly(a.db), '开发', settings)
    tables = study(fs, settings, books)
    out_dir = resolve_path(a.out or c['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if not df.empty:
            df.to_csv(out_dir / (name + '.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
    n_ci = int(sum(df['区间下限'].notna().sum() for df in tables.values() if '区间下限' in df.columns))
    info = {'公司': books, '联赛': c['leagues'], '第一个赛季': c['first_season'], '用途': c['usage'],
            '去水方法': settings['backtest']['benchmark_devig_method'], '算出的区间个数': n_ci,
            '用到的比赛场次(场x时点)': int(len(tables['per_match'])), '设置': c}
    (out_dir / 'run_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 300)
    for name, title in TITLES.items():
        df = tables[name]
        print('==== ' + title + ' ====')
        print('(没有数据)' if df.empty else df.to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('共算了 ' + str(n_ci) + ' 个 95% 区间；每 20 个约有 1 个碰巧不含 0，分联赛、分赛季的更容易。')
    print('结果存在 ' + str(out_dir))


if __name__ == '__main__':
    main()
