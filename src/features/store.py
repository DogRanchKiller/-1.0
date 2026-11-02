"""取数层：以后所有模型和回测都从这里拿数据（需求文档第 17 节：所有模型和回测都必须经过 features 模块取数据）。

用法例子：
    from src.features import FeatureStore, open_readonly
    fs = FeatureStore(open_readonly(), mode="开发")                    # 运行模式：开发 / 考试 / 实时
    cutoff = fs.cutoff_before(match_id, hours=24, usage="A版输入")    # 这场开球前 24 小时（T-24）
    hist = fs.historical_matches(cutoff, usage="A版输入", competition="E0")  # 那时已经踢完的英超比赛和比分
    odds = fs.pre_match_odds(match_id, cutoff, usage="B版输入")        # 那时已经能看到的本场赛前赔率

每个取数函数都要给两样东西：
- 截止时间（UTC）：只返回那时已经能知道的数据（每类数据什么时候能知道，见 known_time.py）；
- 用途：按数据用途字典 config/data_usage.yaml 检查这类数据能不能这样用，不能就报错（见 usage.py）。
另外按运行模式检查赛季：例如开发模式碰到期末考试赛季的比赛就报错。
返回的表都带一列 known_at_utc（这一行数据什么时候能知道），自己可以核对“确实都 ≤ 截止时间”。

和数据库的关系：
- 只读，不改数据库里的任何表。构造时在这个连接上建几张临时表（temp：只在这个连接里、关掉就没了，不写进数据库文件），
  存每场比赛“各类数据什么时候能知道”，后面的查询直接拿来比大小。
  数据库里的比赛有变动（例如实时模式新导入了比分）后，调用 refresh() 重算。
- 一律用 match_id 认比赛，不依赖“一对球队一个赛季只有一场”（杯赛会碰两次）。
  matches 表以后加的列（例如杯赛的 round 轮次、leg 第几回合）会原样带出来，没有这些列也照样运行。
- 联赛 / 杯赛按 competition 字段分：settings.yaml 的 leagues 里列出的是联赛，其他都当杯赛、欧战。
"""
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path, season_role
from src.features.errors import LookAheadError
from src.features.known_time import TIME_FORMAT, compute_known_times, to_utc_text
from src.features.usage import PAST, THIS, DataUsage, normalize_mode, season_label

MARKETS = {"1x2", "ou", "ah"}  # 玩法：胜平负 / 大小球 / 亚盘（和 odds 表的写法一样）
ODDS_COLUMNS = "o.match_id, o.source, o.bookmaker, o.market, o.timing, o.line, o.home, o.draw, o.away, o.over, o.under"


def open_readonly(db_path: Path | str | None = None) -> sqlite3.Connection:
    """只读打开数据库。取数层只读不写；文件不存在时直接报错（不会像 src.common.db.connect 那样新建一个空库）。"""
    path = Path(db_path) if db_path else resolve_path(load_settings()["paths"]["database"])
    if not path.exists():
        raise FileNotFoundError(f"数据库文件不存在：{path}")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _as_list(x) -> list | None:
    """'E0' → ['E0']；['E0', 'D1'] 原样；None 还是 None（表示不筛）。"""
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)
class FeatureStore:
    """按“截止时间 + 用途 + 运行模式”取数据。

    conn：数据库连接（建议用 open_readonly() 只读打开；测试用内存数据库）。
    mode：运行模式，'开发' / '考试' / '实时'（也认 'dev' / 'exam' / 'live'）。
    settings：配置，不给就读 config/settings.yaml（测试里可以改 current_stage 等）。
    """

    def __init__(self, conn: sqlite3.Connection, mode: str, settings: dict | None = None):
        self.conn = conn
        self.mode = normalize_mode(mode)
        self.settings = settings or load_settings()
        self.fs = self.settings["features"]
        self.usage = DataUsage(self.settings, mode=self.mode)
        self._files: dict[str, pd.DataFrame] = {}   # 不在数据库里的数据（data_usage.yaml 里写了“文件”的），读一次就记住
        # 联赛代码：顶级 + 次级（从 settings.yaml 读，不写死）
        self.leagues = [code for group in self.settings["leagues"].values() for code in group]
        # 临时表名带一个随机标签：同一个连接上建两个 FeatureStore（例如一个开发模式、一个考试模式）也互不干扰
        tag = uuid.uuid4().hex[:8]
        self._kt = f"fs_known_{tag}"      # 每场比赛的几个关键时间点
        self._lead = f"fs_odds_lead_{tag}"  # 赛前赔率：每个来源开球前几小时算能知道
        self._prio = f"fs_stats_prio_{tag}"  # 同一场有几个来源的比分时，先用哪个
        self.refresh()

    # ================= 准备 =================
    def refresh(self) -> None:
        """重算每场比赛“什么时候能知道”，放进临时表。数据库里的比赛有变动后要调用一次。"""
        cols = {r[1] for r in self.conn.execute("pragma table_info(matches)")}
        extra = ", date_precision" if "date_precision" in cols else ""   # 旧版比赛表没有这一列
        m = pd.read_sql(f"select match_id, season, match_date, kickoff_utc{extra} from matches", self.conn)
        kt = compute_known_times(m, self.fs)
        roles = {s: season_role(int(s), self.settings) for s in m["season"].unique()}
        kt["season_role"] = m["season"].map(roles).to_numpy()
        c = self.conn
        for t in (self._kt, self._lead, self._prio):
            c.execute(f"drop table if exists temp.{t}")
        c.execute(f"""create temp table {self._kt} (
            match_id integer primary key, season_role text not null,
            kickoff_earliest_utc text not null, kickoff_latest_utc text not null,
            result_known_utc text not null, close_known_utc text not null)""")
        cols = ["match_id", "season_role", "kickoff_earliest_utc", "kickoff_latest_utc",
                "result_known_utc", "close_known_utc"]
        rows = [(int(r[0]), *r[1:]) for r in kt[cols].itertuples(index=False, name=None)]
        c.executemany(f"insert into temp.{self._kt} values (?, ?, ?, ?, ?, ?)", rows)
        # 赛前赔率的提前量写成 SQLite 能直接用的写法，例如 '-24.0 hours'
        c.execute(f"create temp table {self._lead} (source text primary key, modifier text not null)")
        c.executemany(f"insert into temp.{self._lead} values (?, ?)",
                      [(src, f"-{float(h)} hours") for src, h in self.fs["pre_odds_hours_before_kickoff"].items()])
        c.execute(f"create temp table {self._prio} (source text primary key, rank integer not null)")
        c.executemany(f"insert into temp.{self._prio} values (?, ?)",
                      [(src, i) for i, src in enumerate(self.fs["stats_source_priority"])])
        c.commit()

    # ================= 内部检查 =================
    def _competition_filter(self, competition, league: bool) -> tuple[str, list]:
        """按 competition 字段筛选。league=True：只要联赛；False：只要杯赛、欧战。

        competition 给了就检查它真的是联赛（或真的是杯赛）：联赛函数里传杯赛代码，说明调用方拿错了函数，直接报错。
        """
        wanted = _as_list(competition)
        if wanted is not None:
            wrong = [c for c in wanted if (c in self.leagues) != league]
            if wrong:
                kind = "联赛" if league else "杯赛、欧战"
                raise ValueError(f"{wrong} 不是{kind}代码（联赛代码在 settings.yaml 的 leagues）。"
                                 f"{'杯赛、欧战用 cup_results / team_schedule' if league else '联赛用 historical_matches'}")
            return f"m.competition in ({','.join('?' * len(wanted))})", wanted
        ph = ",".join("?" * len(self.leagues))
        return (f"m.competition in ({ph})" if league else f"m.competition not in ({ph})"), list(self.leagues)

    def _check_rows_mode(self, df: pd.DataFrame, usage: str | None) -> pd.DataFrame:
        """取到的每一行都按运行模式检查赛季；有不允许的就报错（不悄悄删掉）。检查完把内部用的 season_role 列去掉。"""
        for role, part in df.groupby("season_role"):
            seasons = "、".join(season_label(int(s)) for s in sorted(part["season"].unique()))
            self.usage.check_mode(self.mode, role, usage, f"取到的数据里有 {seasons} 赛季的 {len(part)} 行")
        return df.drop(columns="season_role")

    def _target(self, match_id: int, cutoff: str, usage: str, data_types: list[str]) -> dict:
        """针对“本场”取数据前的检查：比赛存在、用途允许、运行模式允许这个赛季、赛前用途的截止时间早于开球。"""
        for d in data_types:
            self.usage.check(d, THIS, usage)
        row = self.conn.execute(
            f"select m.match_id, m.season, m.competition, m.home_team, m.away_team, k.season_role, "
            f"k.kickoff_earliest_utc from matches m join temp.{self._kt} k using (match_id) where m.match_id = ?",
            (int(match_id),)).fetchone()
        if row is None:
            raise ValueError(f"没有 match_id = {match_id} 的比赛（刚导入的比赛要先调用 refresh()）")
        mid, season, comp, home, away, role, kickoff = row
        self.usage.check_mode(self.mode, role, usage,
                              f"比赛 {mid}（{comp} {season_label(season)} {home} 对 {away}）")
        if self.usage.is_pre_match(usage) and cutoff >= kickoff:
            raise LookAheadError(
                f"不允许：「{usage}」是赛前用途，截止时间必须早于本场开球。\n"
                f"  比赛 {mid}（{home} 对 {away}）最早可能的开球时间是 {kickoff} UTC，截止时间却是 {cutoff} UTC。\n"
                f"  开球之后才去取本场的数据做赛前预测，就是偷看未来。只有日期没有开球时间的比赛，按当天 00:00（英国时间）开球算。")
        return {"season": season, "kickoff_earliest_utc": kickoff}
    def _stats_groups(self, data_types: list[str]) -> list[tuple[str, list[str]]]:
        """这几类数据在 match_stats 里对应的列，按类分组：[(类名, [列, ...]), ...]。
        一列只放在第一个要它的类里；数据库里还没有的列跳过；一列都没有的类不要。"""
        existing = {r[1] for r in self.conn.execute("pragma table_info(match_stats)")}
        seen: set[str] = set()
        groups = []
        for d in data_types:
            item = self.usage.item(d)
            if not str(item.get("表", "")).startswith("match_stats"):
                raise ValueError(f"「{d}」不在 match_stats 表里，不能用这个函数取（赔率用 historical_odds，杯赛赛程用 team_schedule）")
            cols = [c for c in self.usage.columns(d) if c not in seen and c in existing]
            seen.update(cols)
            if cols:
                groups.append((d, cols))
        return groups

    def _stats_joins(self, groups: list[tuple[str, list[str]]], source: str | None = None) -> tuple[str, str, list]:
        """每一类数据各挑一个来源（match_stats 的主键是 match_id + source，直接 join 的话一场会变成几行）。

        挑法：这类数据的列填得最全的来源先；一样全的，按 settings.yaml 的 stats_source_priority 的顺序。
        例子：2025/26 的比赛 football-data 有比分、射门，没有 xG；懂球帝详情页有比分、射门、xG →
        比分用 football-data（一样全，按顺序），射门和 xG 用懂球帝（它更全）。
        source 给了：先把候选限制到这个来源，再挑完整度，不能先挑其他来源再在外面过滤。
        找不到这个来源时统计列为空，不回退到别的来源；比赛行仍保留，运行模式和赛季检查照常执行。
        返回（要加进 select 的列, join 语句, join 的参数）。第一类的来源叫 stats_source，后面每类的来源叫 stats_source_<类名>。
        """
        if not groups:   # 什么列都不要时，也照“比分”挑一个来源，好给出 stats_source
            groups = [("比分", [])]
        select, joins, params = [], [], []
        source_filter = " and s2.source = ?" if source is not None else ""
        for i, (name, cols) in enumerate(groups):
            a = f"s{i}"
            filled = " + ".join(f"(s2.{c} is not null)" for c in cols) or "null"   # 不能写 0：order by 0 会被当成“按第 0 列排”而报错
            joins.append(f"""
            left join match_stats {a}
              on {a}.match_id = m.match_id
             and {a}.source = (select s2.source from match_stats s2
                               left join temp.{self._prio} p on p.source = s2.source
                               where s2.match_id = m.match_id{source_filter}
                               order by {filled} desc, coalesce(p.rank, 999), s2.source limit 1)""")
            if source is not None:
                params.append(source)
            alias = "stats_source" if i == 0 else f"stats_source_{name}"
            select.append(f'{a}.source as "{alias}"')   # 类名是中文、带顿号，要加引号
            select += [f"{a}.{c}" for c in cols]
        return ", " + ", ".join(select), "".join(joins), params

    def _query_past(self, cutoff: str, groups: list[tuple[str, list[str]]], league: bool, competition=None,
                    team: str | None = None, since=None, newest_first_limit: int | None = None,
                    source: str | None = None) -> pd.DataFrame:
        """“以前的比赛”：已经踢完（finished），而且比分在截止时间之前已经能知道（result_known_utc ≤ 截止时间）。

        同一场比赛有几个来源的统计时，每类数据各挑一个来源（见 _stats_joins）。
        """
        comp_sql, comp_params = self._competition_filter(competition, league)
        where = ["m.status = 'finished'", "k.result_known_utc <= ?", comp_sql]
        params: list = [cutoff, *comp_params]
        if team is not None:
            where.append("(m.home_team = ? or m.away_team = ?)")
            params += [team, team]
        if since is not None:
            where.append("k.kickoff_earliest_utc >= ?")
            params.append(to_utc_text(since))
        stat_sql, join_sql, join_params = self._stats_joins(groups, source)
        order = "desc" if newest_first_limit else "asc"
        sql = f"""
            select m.*, k.season_role, k.kickoff_earliest_utc{stat_sql},
                   k.result_known_utc as known_at_utc
            from matches m
            join temp.{self._kt} k using (match_id){join_sql}
            where {' and '.join(where)}
            order by k.kickoff_earliest_utc {order}, m.match_id {order}"""
        if newest_first_limit:
            sql += " limit ?"
            params.append(int(newest_first_limit))
        df = pd.read_sql(sql, self.conn, params=[*join_params, *params])
        if newest_first_limit:
            df = df.iloc[::-1].reset_index(drop=True)  # 取最近 n 场后，再按时间从早到晚排好
        return df

    # ================= 对外的取数函数 =================
    def cutoff_before(self, match_id: int, hours: float, usage: str | None = None) -> str:
        """某场比赛开球前 hours 小时的截止时间（UTC 文字），例如 hours=24 就是 T-24。

        只有日期没有开球时间的比赛，从当天 00:00（英国时间）往前算，宁早勿晚。
        也按运行模式检查这场比赛的赛季：开发模式下拿考试赛季的比赛当预测目标，在这一步就报错。
        """
        if hours <= 0:
            raise LookAheadError(f"预测时点必须在开球之前（hours 要大于 0），收到 {hours}")
        row = self.conn.execute(
            f"select m.season, k.season_role, k.kickoff_earliest_utc from matches m "
            f"join temp.{self._kt} k using (match_id) where m.match_id = ?", (int(match_id),)).fetchone()
        if row is None:
            raise ValueError(f"没有 match_id = {match_id} 的比赛（刚导入的比赛要先调用 refresh()）")
        season, role, kickoff = row
        if usage is not None:
            self.usage.check_usage_name(usage)
        self.usage.check_mode(self.mode, role, usage, f"比赛 {match_id}（{season_label(season)} 赛季）")
        return to_utc_text(pd.Timestamp(kickoff) - pd.Timedelta(hours=float(hours)))

    def historical_matches(self, cutoff, usage: str, competition=None, data=("比分",), since=None,
                           source: str | None = None) -> pd.DataFrame:
        """截止时间之前已经踢完、结果已经能知道的联赛比赛，每场一行。

        competition：联赛代码，例如 'E0' 或 ['E0', 'E1']；不给 = settings.yaml 里的全部联赛（不含杯赛）。
        data：要哪几类数据，名字按 data_usage.yaml，例如 ('比分', '射门、xG')。每一类都按用途检查。
        since：只要这个时间以后开球的（省时间用，不影响防偷看）。
        source：只从这个 match_stats 来源取统计，每类都在比较完整度前限制候选；None 保持原来的选来源规则。
        指定来源没有记录时保留比赛、统计列为空，不回退其他来源；不影响文件数据、截止时间和赛季限制。
        返回 matches 表的全部列 + 要的统计列 + stats_source（第一类数据来自哪个来源）+ known_at_utc（比分能知道的时间）。
        要了几类数据时，每类各挑一个来源，第二类起的来源在 stats_source_<类名> 列，例如 stats_source_射门、xG。
        """
        cutoff = to_utc_text(cutoff)
        data = list(data)
        for d in ["比赛信息", *data]:
            self.usage.check(d, PAST, usage)
        files = [d for d in data if "文件" in self.usage.item(d)]
        df = self._query_past(cutoff, self._stats_groups([d for d in data if d not in files]), league=True,
                              competition=competition, since=since, source=source)
        for d in files:
            df = self._join_file(df, d, cutoff)
        return self._check_rows_mode(df, usage)

    def attach_pre_match(self, targets: pd.DataFrame, cutoff, usage: str, data) -> pd.DataFrame:
        """给要预测的比赛（本场）接上文件里的赛前数据（例如「伤停（研究）」），返回 targets 加上这些列。

        每一场都按“本场”检查：用途允许、运行模式允许这个赛季、赛前用途的截止时间早于开球（和取别的本场数据一样，见 _target）。
        文件每行有自己的快照时间（as_of_utc）；截止时间早于快照时间的，那一行的数是空值（那时还不知道）。
        现在只支持“什么时候能知道: 快照时间”的文件数据；数据库里的本场数据有各自的函数（例如 pre_match_odds）。
        """
        cutoff = to_utc_text(cutoff)
        data = list(data)
        for d in data:
            if self.usage.item(d).get("什么时候能知道") != "快照时间" or "文件" not in self.usage.item(d):
                raise ValueError(f"「{d}」不是按快照时间记的文件数据，不能用 attach_pre_match 取")
        for mid in targets["match_id"]:
            self._target(int(mid), cutoff, usage, data)
        df = targets
        for d in data:
            df = self._join_file(df, d, cutoff)
        return df

    def _join_file(self, df: pd.DataFrame, data_type: str, cutoff: str) -> pd.DataFrame:
        """把不在数据库里的数据（例如 Understat xG（研究））按 match_id 接到已经取出来的比赛上，文件里没有的场次是空值。

        为什么不会偷看未来，看 data_usage.yaml 写的“什么时候能知道”：
          赛后：只接到“截止时间之前已经能知道比分”的那些行上，这些数据和比分一样是赛后才有的，比分能知道时它也能知道。
                本场的这类数据一律不给（本场: []）。
          快照时间：文件每行有一列 as_of_utc（这一行的数是哪个时间点的情况）。截止时间早于 as_of_utc 的行，数一律变成空值；
                接完把 as_of_utc 去掉。
        文件里的列名是 home_xg 这种，接上时换成 data_usage.yaml 写的列名（例如 us_home_xg），免得和数据库的列混在一起。
        """
        item = self.usage.item(data_type)
        rule = item.get("什么时候能知道")
        if rule not in ("赛后", "快照时间"):
            raise ValueError(f"「{data_type}」：文件里的数据现在只支持“赛后”“快照时间”能知道的")
        if data_type not in self._files:
            path = resolve_path(self.settings["paths"][item["文件"]])
            if not path.exists():
                raise FileNotFoundError(f"「{data_type}」的文件不存在：{path}（先运行生成它的程序，见 data_usage.yaml 的来源）")
            raw = pd.read_csv(path)
            prefix = item.get("列前缀", "")
            cols = {c: c[len(prefix):] if c.startswith(prefix) else c for c in self.usage.columns(data_type)}
            keep = ["match_id", *cols.values()] + (["as_of_utc"] if rule == "快照时间" else [])
            missing = [f for f in keep if f not in raw.columns]
            if missing or raw["match_id"].duplicated().any():
                raise ValueError(f"「{data_type}」的文件 {path.name} 格式不对：缺列 {missing}，或者同一场出现了两次")
            table = raw[keep].rename(columns={v: k for k, v in cols.items()})
            if rule == "快照时间":   # 统一成和截止时间一样的写法（UTC，到秒），才能直接比较文字
                # utc=True：没写时区的当作 UTC；写了时区的（例如 +01:00）先换成 UTC 再写成文字，免得差一小时
                table["as_of_utc"] = pd.to_datetime(table["as_of_utc"], errors="coerce", utc=True).dt.strftime(TIME_FORMAT)
            self._files[data_type] = table
        table = self._files[data_type]
        clash = [c for c in table.columns if c not in ("match_id", "as_of_utc") and c in df.columns]
        if clash:
            raise ValueError(f"「{data_type}」的列 {clash} 和别的数据重名")
        out = df.merge(table, on="match_id", how="left")
        if rule == "快照时间":
            # 快照时间晚于截止时间（或者没写快照时间）的行：那时还不知道，数变成空值
            unknown = ~(out["as_of_utc"].astype("string") <= cutoff).fillna(False).astype(bool)
            value_cols = [c for c in table.columns if c not in ("match_id", "as_of_utc")]
            out.loc[unknown, value_cols] = np.nan
            out = out.drop(columns="as_of_utc")
        return out

    def team_recent_matches(self, team: str, cutoff, usage: str, n: int, competition=None,
                            data=("比分",)) -> pd.DataFrame:
        """某队截止时间之前最近 n 场已经踢完的联赛，从早到晚排。

        team 是统一的球队编号（teams 表，例如 'Man United'）。
        除了 historical_matches 的列，还加上这队视角的列：is_home（是不是主场）、opponent（对手）、
        各项统计的 _for / _against（例如 goals_for 进球、goals_against 失球、shots_for 射门），
        有比分时加 team_result（W 胜 / D 平 / L 负）。
        """
        cutoff = to_utc_text(cutoff)
        data = list(data)
        for d in ["比赛信息", *data]:
            self.usage.check(d, PAST, usage)
        if int(n) <= 0:
            raise ValueError("n 要大于 0")
        if self.conn.execute("select 1 from matches where home_team = ? or away_team = ? limit 1",
                             (team, team)).fetchone() is None:
            raise ValueError(f"数据库里没有球队「{team}」（球队编号见 config/teams.yaml）")
        df = self._query_past(cutoff, self._stats_groups(data), league=True, competition=competition,
                              team=team, newest_first_limit=int(n))
        df = self._check_rows_mode(df, usage)
        return _team_view(df, team)
    def _odds_filters(self, bookmaker, market) -> tuple[list[str], list]:
        """赔率的公司、玩法筛选条件。玩法写错直接报错（写错了会悄悄返回空表，以为没有赔率）。"""
        where, params = [], []
        books = _as_list(bookmaker)
        if books:
            where.append(f"o.bookmaker in ({','.join('?' * len(books))})")
            params += books
        markets = _as_list(market)
        if markets:
            bad = set(markets) - MARKETS
            if bad:
                raise ValueError(f"玩法只能是 {sorted(MARKETS)}（胜平负 / 大小球 / 亚盘），收到 {sorted(bad)}")
            where.append(f"o.market in ({','.join('?' * len(markets))})")
            params += markets
        return where, params

    def pre_match_odds(self, match_id: int, cutoff, usage: str, bookmaker=None, market=None) -> pd.DataFrame:
        """本场截止时间之前已经能看到的赛前赔率，一行是“一家公司 × 一种玩法”。

        football-data 的赛前赔率按开球前 24 小时能知道算（settings.yaml 的 pre_odds_hours_before_kickoff），
        所以截止时间在 T-24 之后才拿得到；T-6、T-1 拿到的也是同一组（football-data 只有一个赛前时点）。
        bookmaker：公司名，按 odds 表的写法，例如 '平博 Pinnacle'、'多家平均'；market：'1x2' / 'ou' / 'ah'。
        """
        cutoff = to_utc_text(cutoff)
        self._target(match_id, cutoff, usage, ["赛前赔率"])
        where, params = self._odds_filters(bookmaker, market)
        # 能知道的时间 = 最早开球 − 这个来源的提前量；没登记提前量的来源 join 不上，一律拿不到
        known = "datetime(k.kickoff_earliest_utc, l.modifier)"
        sql = f"""
            select {ODDS_COLUMNS}, {known} as known_at_utc
            from odds o
            join temp.{self._kt} k on k.match_id = o.match_id
            join temp.{self._lead} l on l.source = o.source
            where o.match_id = ? and o.timing = 'pre' and {known} <= ? {''.join(' and ' + w for w in where)}
            order by o.bookmaker, o.market"""
        return pd.read_sql(sql, self.conn, params=[int(match_id), cutoff, *params])

    def closing_odds(self, match_id: int, cutoff, usage: str, bookmaker=None, market=None) -> pd.DataFrame:
        """本场的收盘赔率。只能赛后用于评估、结算、复盘、显示；开球前（含赛前任何用途）都拿不到。

        能知道的时间 = 开球那一刻（只有日期时 = 当天 24:00 英国时间）。
        """
        cutoff = to_utc_text(cutoff)
        self._target(match_id, cutoff, usage, ["收盘赔率"])
        where, params = self._odds_filters(bookmaker, market)
        sql = f"""
            select {ODDS_COLUMNS}, k.close_known_utc as known_at_utc
            from odds o join temp.{self._kt} k on k.match_id = o.match_id
            where o.match_id = ? and o.timing = 'close' and k.close_known_utc <= ? {''.join(' and ' + w for w in where)}
            order by o.bookmaker, o.market"""
        return pd.read_sql(sql, self.conn, params=[int(match_id), cutoff, *params])

    def odds_changes(self, match_id: int, cutoff, usage: str, bookmaker=None, market=None) -> pd.DataFrame:
        """本场的变盘记录（odds_changes 表，懂球帝），截止时间之前已经变过的每一组价格，按时间从早到晚排。

        能知道的时间 = 网站显示的变盘时间（changed_at_utc）。另外一律只给开球之前的价格：
        懂球帝的“最新”价格在比赛踢完后可能是开球后（滚球）的价格，不是赛前的东西；
        只有日期的比赛按当天 00:00（英国时间）开球算，宁早勿晚。
        bookmaker：公司名，按 config/bookmakers_dongqiudi.yaml 换过的写法，例如 'Bet365'、'平博 Pinnacle'。
        """
        cutoff = to_utc_text(cutoff)
        self._target(match_id, cutoff, usage, ["变盘记录"])
        where, params = self._odds_filters(bookmaker, market)
        sql = f"""
            select o.match_id, o.source, o.bookmaker, o.bookmaker_raw, o.market, o.line, o.home, o.draw, o.away,
                   o.over, o.under, o.changed_at_utc, o.fetched_at_utc, o.changed_at_utc as known_at_utc
            from odds_changes o join temp.{self._kt} k on k.match_id = o.match_id
            where o.match_id = ? and o.changed_at_utc <= ? and o.changed_at_utc < k.kickoff_earliest_utc
            {''.join(' and ' + w for w in where)}
            order by o.changed_at_utc, o.rowid"""
        return pd.read_sql(sql, self.conn, params=[int(match_id), cutoff, *params])

    def latest_odds(self, match_id: int, cutoff, usage: str, bookmaker=None, market=None) -> pd.DataFrame:
        """截止时间那一刻每家公司每种玩法的“当时价格”：odds_changes 里最后一次变盘的那组。

        让球、大小球同一家公司有时同时开几条盘口，这里只取最后变的那一条（一般就是主盘口）。
        """
        df = self.odds_changes(match_id, cutoff, usage, bookmaker, market)
        # 同一时间有两行（初盘和最新的时间一样）时，后导入的是“最新”，取后面那行
        return df.groupby(["bookmaker", "market"], sort=True).tail(1).sort_values(["bookmaker", "market"]).reset_index(drop=True)

    def lottery_odds(self, match_id: int, cutoff, usage: str, product=None) -> pd.DataFrame:
        """本场的彩票赔率（竞彩胜平负、竞彩让球、北单），只能显示。

        懂球帝的彩票赔率没有变盘时间，能知道的时间 = 我们第一次抓到的时间（fetched_at_utc）。
        product：'竞彩胜平负' / '竞彩让球' / '北单'；不给 = 全部。
        """
        cutoff = to_utc_text(cutoff)
        self._target(match_id, cutoff, usage, ["竞彩赔率"])
        where, params = ["o.match_id = ?", "o.fetched_at_utc <= ?"], [int(match_id), cutoff]
        products = _as_list(product)
        if products:
            where.append(f"o.product in ({','.join('?' * len(products))})")
            params += products
        sql = f"""
            select o.match_id, o.source, o.product, o.handicap, o.home, o.draw, o.away, o.fetched_at_utc,
                   o.fetched_at_utc as known_at_utc
            from lottery_odds o where {' and '.join(where)}
            order by o.product, o.handicap, o.fetched_at_utc, o.rowid"""
        return pd.read_sql(sql, self.conn, params=params)

    def historical_odds(self, cutoff, usage: str, timing: str = "pre", competition=None, bookmaker=None,
                        market=None, since=None) -> pd.DataFrame:
        """以前比赛的赔率（截止时间之前已经踢完、比分已经能知道的联赛比赛）。B 版训练、回测里选融合权重用。

        timing：'pre' 赛前 / 'close' 收盘。A 版用途在这里直接报错（A 版不用赔率）。
        数据量大（全部约 189 万行），建议用 competition、bookmaker、market、since 缩小范围。
        """
        cutoff = to_utc_text(cutoff)
        if timing not in ("pre", "close"):
            raise ValueError("timing 只能是 'pre'（赛前）或 'close'（收盘）")
        self.usage.check("比赛信息", PAST, usage)
        self.usage.check("赛前赔率" if timing == "pre" else "收盘赔率", PAST, usage)
        comp_sql, comp_params = self._competition_filter(competition, league=True)
        where = ["m.status = 'finished'", "k.result_known_utc <= ?", comp_sql, "o.timing = ?"]
        params: list = [cutoff, *comp_params, timing]
        w2, p2 = self._odds_filters(bookmaker, market)
        where += w2
        params += p2
        if since is not None:
            where.append("k.kickoff_earliest_utc >= ?")
            params.append(to_utc_text(since))
        sql = f"""
            select m.competition, m.season, m.match_date, k.kickoff_earliest_utc, m.home_team, m.away_team,
                   k.season_role, {ODDS_COLUMNS}, k.result_known_utc as known_at_utc
            from odds o
            join matches m on m.match_id = o.match_id
            join temp.{self._kt} k on k.match_id = o.match_id
            where {' and '.join(where)}
            order by k.kickoff_earliest_utc, o.match_id, o.bookmaker, o.market"""
        df = pd.read_sql(sql, self.conn, params=params)
        return self._check_rows_mode(df, usage)

    def match_result(self, match_id: int, cutoff, usage: str) -> pd.DataFrame:
        """本场的比分（答案）。只能用于评估、机器人结算、复盘；截止时间之前比分还不能知道时，返回空表。"""
        cutoff = to_utc_text(cutoff)
        self._target(match_id, cutoff, usage, ["比分"])
        stat_sql, join_sql, join_params = self._stats_joins(self._stats_groups(["比分"]))
        sql = f"""
            select m.match_id{stat_sql}, k.result_known_utc as known_at_utc
            from matches m
            join temp.{self._kt} k using (match_id){join_sql}
            where m.match_id = ? and m.status = 'finished' and k.result_known_utc <= ? and s0.source is not null"""
        return pd.read_sql(sql, self.conn, params=[*join_params, int(match_id), cutoff])
    def prediction_targets(self, start, end, usage: str, competition=None) -> pd.DataFrame:
        """开球时间在 [start, end) 之间的联赛比赛，当作要预测的目标（回测时一场一场拿出来预测）。

        只返回赛前就知道的比赛信息，不返回 status：一场比赛后来是踢了、改期还是取消，赛前不知道，
        不能拿它来挑预测目标（否则回测里就“碰巧”避开了所有改期的比赛）。
        开发模式下，目标里有期末考试赛季的比赛就报错。
        """
        start, end = to_utc_text(start), to_utc_text(end)
        self.usage.check("比赛信息", THIS, usage)
        comp_sql, comp_params = self._competition_filter(competition, league=True)
        sql = f"""
            select m.*, k.season_role, k.kickoff_earliest_utc
            from matches m join temp.{self._kt} k using (match_id)
            where k.kickoff_earliest_utc >= ? and k.kickoff_earliest_utc < ? and {comp_sql}
            order by k.kickoff_earliest_utc, m.match_id"""
        df = pd.read_sql(sql, self.conn, params=[start, end, *comp_params])
        return self._check_rows_mode(df, usage).drop(columns="status")

    def team_schedule(self, team: str, cutoff, usage: str, include_cups: bool = True, since=None) -> pd.DataFrame:
        """某队截止时间之前已经开球的比赛日程（联赛 + 杯赛、欧战），只有日期、对手、赛事，没有比分。算休息天数、“欧战后”标签用。

        杯赛、欧战的赛程按数据用途字典只能用于「休息天数和欧战后标签」和「显示」：
        A 版要用，先在这里用「休息天数和欧战后标签」取出来算成休息天数，再把休息天数交给模型。
        能知道的时间 = 最晚可能的开球时间（只有日期的杯赛按当天 24:00 英国时间算，宁晚勿早）。
        """
        return self._schedule(cutoff, usage, team=team, include_cups=include_cups, since=since)

    def schedule(self, cutoff, usage: str, since=None) -> pd.DataFrame:
        """所有队截止时间之前已经开球的比赛日程（全部联赛 + 杯赛、欧战），规则和 team_schedule 一样，只是不按队筛。
        回测给每场比赛算休息天数用（src/features/rest_days.py）：一次取一批，比一队一队取快得多。"""
        return self._schedule(cutoff, usage, team=None, include_cups=True, since=since)

    def _schedule(self, cutoff, usage: str, team: str | None, include_cups: bool, since) -> pd.DataFrame:
        cutoff = to_utc_text(cutoff)
        self.usage.check("比赛信息", PAST, usage)
        if include_cups:
            self.usage.check("杯赛欧战赛程", PAST, usage)
        where = ["k.kickoff_latest_utc <= ?", "m.status not in ('postponed', 'cancelled')"]
        params: list = [cutoff]
        if team is not None:
            where.insert(0, "(m.home_team = ? or m.away_team = ?)")
            params = [team, team, *params]
        if not include_cups:
            comp_sql, comp_params = self._competition_filter(None, league=True)
            where.append(comp_sql)
            params += comp_params
        if since is not None:
            where.append("k.kickoff_earliest_utc >= ?")
            params.append(to_utc_text(since))
        ph = ",".join("?" * len(self.leagues))
        sql = f"""
            select m.*, k.season_role, k.kickoff_earliest_utc, k.kickoff_latest_utc,
                   (m.competition not in ({ph})) as is_cup, k.kickoff_latest_utc as known_at_utc
            from matches m join temp.{self._kt} k using (match_id)
            where {' and '.join(where)}
            order by k.kickoff_earliest_utc, m.match_id"""
        df = pd.read_sql(sql, self.conn, params=[*self.leagues, *params])
        df["is_cup"] = df["is_cup"].astype(bool)
        return self._check_rows_mode(df, usage)

    def cup_results(self, cutoff, usage: str, competition=None, team: str | None = None, since=None) -> pd.DataFrame:
        """以前的杯赛、欧战比分（预留位置）。按数据用途字典，第 8 阶段前不能当模型输入，调用会报错。"""
        cutoff = to_utc_text(cutoff)
        self.usage.check("杯赛欧战比分", PAST, usage)
        df = self._query_past(cutoff, self._stats_groups(["杯赛欧战比分"]), league=False,
                              competition=competition, team=team, since=since)
        return self._check_rows_mode(df, usage)

    def news_for_match(self, match_id: int, cutoff, usage: str, days=None) -> pd.DataFrame:
        """和这场比赛有关的新闻标题（第 9 阶段，只能复盘、显示）。规则见 src/features/news.py 和需求文档 7.1。"""
        from src.features.news import news_for_match
        return news_for_match(self, match_id, cutoff, usage, days)

    def news_facts_for_match(self, match_id: int, cutoff, usage: str, fact_type=None) -> pd.DataFrame:
        """主队、客队的主帅更换、核心长期伤缺等结构化信息。规则见 src/features/news.py 和需求文档 7.1。"""
        from src.features.news import news_facts_for_match
        return news_facts_for_match(self, match_id, cutoff, usage, fact_type)


def _team_view(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """把“主队 / 客队”的表换成“这支队 / 对手”的视角。

    例子：home_goals、away_goals → 这队是主队时 goals_for = home_goals，是客队时 goals_for = away_goals。
    """
    df = df.copy()
    home = df["home_team"] == team
    df["is_home"] = home
    df["opponent"] = df["away_team"].where(home, df["home_team"])
    for col in [c for c in df.columns if c.startswith("home_")]:
        base = col[len("home_"):]
        if f"away_{base}" not in df.columns or base == "team":
            continue
        df[f"{base}_for"] = df[col].where(home, df[f"away_{base}"])
        df[f"{base}_against"] = df[f"away_{base}"].where(home, df[col])
    if "result" in df.columns:
        mine = df["result"].map({"H": "W", "A": "L", "D": "D"})
        df["team_result"] = mine.where(home, df["result"].map({"H": "L", "A": "W", "D": "D"}))
    return df
