"""新闻资料库：从本地文件导入（需求文档 7.1“新闻资料库怎么做”）。

这个程序不联网、不抓任何网站，只读 settings.yaml 的 news.inbox_dir（默认 data/news_inbox）里的本地文件：
  news_*.jsonl        新闻条目，一行一条 JSON，字段：
                        source（来源名，必填）、url（必填）、title（必填）、
                        fetched_at（抓到时间，必填，要带时间，例如 2026-11-05 10:00:00，没写时区的当 UTC）、
                        published_at（发布时间，选填，只有日期也行）、language（选填）、
                        teams（选填，球队名列表，来源自带的关联）、match_ids（选填，比赛编号列表）
  news_facts_*.jsonl  结构化信息，一行一件事，字段：
                        team（球队名，必填）、fact_type（coach_change / long_injury，必填）、
                        effective_date（生效日 yyyy-mm-dd，必填）、recorded_at（录入时间，必填，要带时间）、
                        person、note（选填）、method（人工 / 规则，默认人工）、
                        news_source + news_url（选填，两个一起写：这件事是从哪条已导入的新闻来的）
注意：news_facts_ 开头的文件也符合 news_ 开头，程序先判断 news_facts_。

三条硬规则（防偷看未来，需求文档 7.1）：
  1. 来源必须先在 settings.yaml 的 news.sources 登记为“允许”，没登记、待定、不用的来源一行都不导入；
  2. 抓到时间 / 录入时间必须有，而且要带时间；比现在还晚（超过容忍分钟数）的拒绝，不猜、不补；
  3. 重复的新闻只留先导入的那一行，不改它（抓到时间也不往早改，往早改会让它在更早的截止时间出现）。

可以重复运行，结果不变。导入的文件原样压缩存一份到原始层（只增不改，内容一样就不重复存）。
用法：.venv/Scripts/python -m src.clean.news_import
"""
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.clean.news_schema import create_news_tables
from src.common.config import PROJECT_ROOT, load_settings, resolve_path
from src.common.db import connect
from src.features.known_time import to_utc_text

MASK = chr(0)   # 标题里认出一个队名后，把那几个字盖成这个符号，免得短名字又在里面认一次
NL = chr(10)
HAS_TIME = re.compile(r'[0-9]{1,2}:[0-9]{2}')
DATE_ONLY = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
METHODS = ('人工', '规则')
MAX_REJECTS_IN_REPORT = 50


class RowError(Exception):
    """一行有问题（缺字段、时间看不懂等）：这一行不导入，原因写进报告。"""


def cell(row: dict, key: str) -> str:
    """取一个字段，去掉前后空格；没有或不是文字就当空。"""
    v = row.get(key)
    return v.strip() if isinstance(v, str) else ''


def parse_time(value, what: str, date_only_ok: bool = False):
    """把文件里的时间统一成 UTC 文字。没写（None 或空）返回 None；看不懂就报 RowError。

    没写时区的当 UTC，写了时区的换算成 UTC。只有日期没有时间的：date_only_ok=True（发布时间）时按第二天 00:00 算，
    宁晚勿早；否则（抓到时间、录入时间）拒绝，不猜。
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise RowError(f'{what}要写成文字，例如 2026-11-05 10:00:00')
    text = value.strip()
    try:
        if not HAS_TIME.search(text):
            if not date_only_ok:
                raise RowError(f'{what}只有日期没有时间，不猜')
            day = pd.Timestamp(text)
            return to_utc_text(day.normalize() + pd.Timedelta(days=1))
        return to_utc_text(text)
    except RowError:
        raise
    except (ValueError, TypeError):
        raise RowError(f'{what}看不懂：{text}')


class TeamResolver:
    """球队名 → 统一球队编号。认得：teams 表的编号、中文名（name_zh）、team_names 表里各来源的写法。

    同一个名字对到两支球队的（有歧义），一律不认，免得认错。
    """

    def __init__(self, con, min_chars: int, min_chars_ascii: int):
        owner: dict[str, str] = {}
        ambiguous: set[str] = set()

        def add(name, team_id):
            name = (name or '').strip()
            if not name:
                return
            if name in owner and owner[name] != team_id:
                ambiguous.add(name)
            else:
                owner[name] = team_id

        for team_id, name_zh in con.execute('select team_id, name_zh from teams'):
            add(team_id, team_id)
            add(name_zh, team_id)
        for name, team_id in con.execute('select name, team_id from team_names'):
            add(name, team_id)
        for name in ambiguous:
            owner.pop(name, None)
        self.owner = owner
        # 标题里认球队：先认长的（“巴黎圣日耳曼”先于“巴黎”）；英文名按单词边界认，不区分大小写
        names = [n for n in owner if len(n) >= (min_chars_ascii if n.isascii() else min_chars)]
        self.title_names = sorted(names, key=lambda n: (-len(n), n))
        self.ascii_patterns = {n: re.compile(r'(?<![A-Za-z])' + re.escape(n) + r'(?![A-Za-z])', re.I)
                               for n in self.title_names if n.isascii()}

    def exact(self, name: str):
        """整个名字对得上就返回球队编号，否则 None。"""
        return self.owner.get((name or '').strip())

    def find_in_title(self, title: str) -> list[str]:
        """标题里提到了哪些球队（去重，按出现的先后不保证）。认到的字盖掉，同一段字只算一次。"""
        rest, found = title, []
        for n in self.title_names:
            if n in self.ascii_patterns:
                pat = self.ascii_patterns[n]
                if pat.search(rest):
                    rest = pat.sub(lambda m: MASK * len(m.group()), rest)
                    found.append(self.owner[n])
            elif n in rest:
                rest = rest.replace(n, MASK * len(n))
                found.append(self.owner[n])
        return list(dict.fromkeys(found))


def archive_file(path: Path, data: bytes, raw_root: Path, subdir: str, now: datetime) -> str:
    """导入的文件原样压缩存进原始层：<raw>/<subdir>/<导入日期>/<文件名>_<内容指纹前 12 位>.gz。

    只增不改：内容一样（指纹一样）已经存过就不再存，返回已有的位置。
    返回写进数据库的位置（在项目文件夹里的写相对路径）。
    """
    sha = hashlib.sha256(data).hexdigest()[:12]
    base = raw_root / subdir
    existing = sorted(base.rglob(f'*_{sha}.gz')) if base.is_dir() else []
    if existing:
        target = existing[0]
    else:
        target = base / now.strftime('%Y-%m-%d') / f'{path.stem}_{sha}.gz'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(data, mtime=0))
    try:
        return target.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(target)


class Importer:
    """一次导入的全部状态：登记的来源、球队名对照、各种计数、被拒绝的行。"""

    def __init__(self, con, settings: dict, raw_root: Path, now: datetime):
        self.con = con
        self.cfg = settings['news']
        self.raw_root = raw_root
        self.now = now
        self.now_text = to_utc_text(now)
        # 抓到时间、录入时间最晚能到几点：现在 + 容忍分钟数
        self.latest_ok = to_utc_text(now + timedelta(minutes=float(self.cfg['fetched_future_tolerance_minutes'])))
        self.sources = self.cfg.get('sources') or {}
        self.fact_types = list(self.cfg['fact_types'])
        self.teams = TeamResolver(con, int(self.cfg['min_team_name_chars']), int(self.cfg['min_team_name_chars_ascii']))
        self.stats = Counter()
        self.rejects: list[tuple[str, int, str]] = []
        self.unknown_teams: Counter = Counter()

    # ---------- 小工具 ----------
    def reject(self, kind: str, file: str, lineno: int, reason: str) -> None:
        self.stats[f'{kind}_rejected'] += 1
        self.rejects.append((file, lineno, reason))

    def check_recorded_time(self, value, what: str) -> str:
        """抓到时间 / 录入时间：必填、要带时间、不能比现在晚太多。"""
        t = parse_time(value, what)
        if t is None:
            raise RowError(f'没有{what}（必填，不猜）')
        if t > self.latest_ok:
            raise RowError(f'{what} {t} 比现在 {self.now_text} 还晚（超过容忍时间），多半是时钟或时区写错了')
        return t

    # ---------- 一个文件 ----------
    def file(self, path: Path) -> None:
        name = path.name
        if name.startswith('news_facts_'):
            kind = 'facts'
        elif name.startswith('news_'):
            kind = 'items'
        else:
            self.stats['skipped_files'] += 1
            return
        data = path.read_bytes()
        raw_file = archive_file(path, data, self.raw_root, self.cfg['raw_subdir'], self.now)
        self.stats['files'] += 1
        for lineno, line in enumerate(data.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            short = 'items' if kind == 'items' else 'facts'
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RowError('这一行不是一个 JSON 对象')
                if kind == 'items':
                    self.item(row, name, lineno, raw_file)
                else:
                    self.fact(row, name, lineno)
            except json.JSONDecodeError:
                self.reject(short, name, lineno, '不是合法的 JSON')
            except RowError as e:
                self.reject(short, name, lineno, str(e))

    # ---------- 一条新闻 ----------
    def item(self, row: dict, file: str, lineno: int, raw_file: str) -> None:
        source, url, title = cell(row, 'source'), cell(row, 'url'), cell(row, 'title')
        if not source or not url or not title:
            raise RowError('source、url、title 都要写')
        status = (self.sources.get(source) or {}).get('status')
        if status != '允许':
            raise RowError(f'来源「{source}」没有登记为允许（现在是：{status or "没登记"}）')
        fetched = self.check_recorded_time(row.get('fetched_at'), '抓到时间')
        published = parse_time(row.get('published_at'), '发布时间', date_only_ok=True)
        teams_given, match_ids = row.get('teams') or [], row.get('match_ids') or []
        if not isinstance(teams_given, list) or not isinstance(match_ids, list):
            raise RowError('teams、match_ids 要写成列表')

        old = self.con.execute('select news_id from news_items where source = ? and url = ? and title = ?',
                               (source, url, title)).fetchone()
        if old is not None:
            self.stats['items_dup'] += 1   # 先导入的那一行不改，关联也不补
            return
        cur = self.con.execute(
            'insert into news_items (source, url, title, published_at_utc, fetched_at_utc, language, raw_file, imported_at_utc)'
            ' values (?, ?, ?, ?, ?, ?, ?, ?)',
            (source, url, title, published, fetched, cell(row, 'language') or None, raw_file, self.now_text))
        news_id = cur.lastrowid
        self.stats['items_new'] += 1

        linked: set[str] = set()
        for name in teams_given:
            team_id = self.teams.exact(name) if isinstance(name, str) else None
            if team_id is None:
                self.unknown_teams[str(name)] += 1
            elif team_id not in linked:
                self.link(news_id, team_id, 0, '来源自带')
                linked.add(team_id)
        for team_id in self.teams.find_in_title(title):
            if team_id not in linked:
                self.link(news_id, team_id, 0, '标题认出')
                linked.add(team_id)
        for mid in match_ids:
            ok = isinstance(mid, int) and self.con.execute('select 1 from matches where match_id = ?', (mid,)).fetchone()
            if ok:
                self.link(news_id, '', mid, '来源自带')
            else:
                self.stats['bad_match_ids'] += 1
                self.rejects.append((file, lineno, f'比赛编号 {mid} 在数据库里没有，没关联（新闻本身导入了）'))

    def link(self, news_id: int, team_id: str, match_id: int, method: str) -> None:
        cur = self.con.execute('insert or ignore into news_links (news_id, team_id, match_id, method) values (?, ?, ?, ?)',
                               (news_id, team_id, match_id, method))
        self.stats['links_new'] += cur.rowcount

    # ---------- 一件结构化信息 ----------
    def fact(self, row: dict, file: str, lineno: int) -> None:
        team_name = cell(row, 'team')
        team_id = self.teams.exact(team_name)
        if team_id is None:
            self.unknown_teams[team_name or '(空)'] += 1
            raise RowError(f'球队「{team_name}」认不出，进“待确认”')
        fact_type = cell(row, 'fact_type')
        if fact_type not in self.fact_types:
            raise RowError(f'fact_type「{fact_type}」不在允许的种类里：{self.fact_types}')
        eff = cell(row, 'effective_date')
        try:
            if not DATE_ONLY.match(eff):
                raise ValueError(eff)
            datetime.strptime(eff, '%Y-%m-%d')
        except ValueError:
            raise RowError(f'effective_date 要写成 yyyy-mm-dd，收到：{eff!r}')
        recorded = self.check_recorded_time(row.get('recorded_at'), '录入时间')
        method = cell(row, 'method') or '人工'
        if method not in METHODS:
            raise RowError(f'method 只能是 {METHODS}')
        news_source, news_url = cell(row, 'news_source'), cell(row, 'news_url')
        news_id = None
        if news_source or news_url:
            if not (news_source and news_url):
                raise RowError('news_source 和 news_url 要一起写')
            news_id = self.con.execute('select min(news_id) from news_items where source = ? and url = ?',
                                       (news_source, news_url)).fetchone()[0]
            if news_id is None:
                raise RowError('对不到已导入的来源新闻（先导入新闻，再导入结构化信息）')
        person = cell(row, 'person')
        cur = self.con.execute(
            'insert or ignore into news_facts (team_id, fact_type, effective_date, person, note, news_id, method,'
            ' recorded_at_utc, imported_at_utc) values (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (team_id, fact_type, eff, person, cell(row, 'note') or None, news_id, method, recorded, self.now_text))
        # 重复的（同一队、同一种、同一生效日、同一个人）只留先录入的那一行，不改它的录入时间
        self.stats['facts_new' if cur.rowcount else 'facts_dup'] += 1

    def summary(self) -> dict:
        s = self.stats
        return {'files': s['files'], 'skipped_files': s['skipped_files'],
                'items_new': s['items_new'], 'items_dup': s['items_dup'], 'items_rejected': s['items_rejected'],
                'links_new': s['links_new'], 'bad_match_ids': s['bad_match_ids'],
                'facts_new': s['facts_new'], 'facts_dup': s['facts_dup'], 'facts_rejected': s['facts_rejected'],
                'unknown_teams': dict(self.unknown_teams), 'rejects': list(self.rejects)}


def write_report(path: Path, summary: dict, inbox: Path, now: datetime) -> None:
    """导入报告（能重新生成，放在 data/reports/，不进 Git）。"""
    s = summary
    lines = [f'# 新闻导入报告（{to_utc_text(now)} UTC）', '', f'- 读的文件夹：{inbox}',
             f'- 处理的文件：{s["files"]} 个；不认识的文件（不是 news_ / news_facts_ 开头）：{s["skipped_files"]} 个',
             f'- 新闻条目：新导入 {s["items_new"]}，重复 {s["items_dup"]}，拒绝 {s["items_rejected"]}',
             f'- 球队、比赛关联：新建 {s["links_new"]}；对不上的比赛编号 {s["bad_match_ids"]}',
             f'- 结构化信息：新导入 {s["facts_new"]}，重复 {s["facts_dup"]}，拒绝 {s["facts_rejected"]}', '',
             '## 待确认的球队名（认不出，没关联）', '']
    if s['unknown_teams']:
        lines += [f'- {name}：{n} 次' for name, n in sorted(s['unknown_teams'].items())]
    else:
        lines.append('- 没有')
    lines += ['', f'## 被拒绝或没关联的行（最多列 {MAX_REJECTS_IN_REPORT} 行，共 {len(s["rejects"])} 行）', '']
    if s['rejects']:
        lines += [f'- {f} 第 {n} 行：{why}' for f, n, why in s['rejects'][:MAX_REJECTS_IN_REPORT]]
    else:
        lines.append('- 没有')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NL.join(lines) + NL, encoding='utf-8')


def run(con, settings: dict | None = None, inbox: Path | None = None, raw_root: Path | None = None,
        report_path: Path | None = None, now: datetime | None = None) -> dict:
    """导入 inbox 里的全部本地文件。可以重复运行，结果不变。返回计数字典（见 Importer.summary）。

    出错时整批不算（回滚），已经存进原始层的文件副本留着（只增不改，没有坏处）。
    """
    settings = settings or load_settings()
    cfg = settings['news']
    now = now or datetime.now(timezone.utc)
    inbox = Path(inbox) if inbox else resolve_path(cfg['inbox_dir'])
    raw_root = Path(raw_root) if raw_root else resolve_path(settings['paths']['raw'])
    report_path = Path(report_path) if report_path else resolve_path(settings['paths']['reports']) / cfg['report']
    create_news_tables(con)
    imp = Importer(con, settings, raw_root, now)
    try:
        files = sorted(inbox.glob('*.jsonl')) if inbox.is_dir() else []
        # 先导入新闻，再导入结构化信息：信息可以指向同一批里的新闻
        for path in sorted(files, key=lambda p: (p.name.startswith('news_facts_'), p.name)):
            imp.file(path)
        con.commit()
    except Exception:
        con.rollback()
        raise
    summary = imp.summary()
    write_report(report_path, summary, inbox, now)
    return summary


def main() -> None:
    settings = load_settings()
    con = connect()
    try:
        s = run(con, settings)
    finally:
        con.close()
    print(f'新闻 新导入 {s["items_new"]}、重复 {s["items_dup"]}、拒绝 {s["items_rejected"]}；'
          f'结构化信息 新导入 {s["facts_new"]}、重复 {s["facts_dup"]}、拒绝 {s["facts_rejected"]}')
    print('报告：', resolve_path(settings['paths']['reports']) / settings['news']['report'])


if __name__ == '__main__':
    main()
