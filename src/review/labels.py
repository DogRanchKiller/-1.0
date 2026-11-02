"""复盘自动标签（需求文档第 12、15 节“标签和备注”）：德比战、争冠战、保级战、欧战后的联赛。

“欧战后的联赛”是 2026-11-01（工作树 Q1）加的，杯赛、欧战赛程进库以后才能做（待定问题 5 已定）。定义见下面第 3 条。

四个标签的定义（德比、争冠、保级的参数在 settings.yaml 的 labels 段，欧战后的参数在 rest_days 段）：

1. 德比战：两队是 config/derbies.yaml 名单里的一对（固定名单，同城德比、同地区德比、传统宿敌）。
   名单只看对阵，不看比赛重不重要；同一对球队踢的所有联赛比赛都算。

2. 争冠战、保级战：按“这场开球前”的积分榜（src/features/standings.py），对两队分别判断“在不在争”，
   至少一队在争，这场就算争冠战 / 保级战。另外记下几队在争（1 或 2，2 队都在争的就是“直接对话”）。
   一支队“在争冠”要同时满足：
     a. 赛季末段：这队剩余场次（含本场）≤ race.last_rounds（默认 10）；
     b. 冠军还没定、它还有机会：
        - 它是第 1 名：有别的队落后它不超过 gap_points 分，而且那队剩余场次全胜能追上（落后分 ≤ 3 × 那队剩余场次）；
        - 它不是第 1 名：落后第 1 名不超过 gap_points 分，而且它剩余场次全胜能追上（落后分 ≤ 3 × 它的剩余场次）。
   “在保级”一样的思路，关键名次换成“降级区的上沿”：降级区 = 最后 降级 + 降级附加赛 个名次（每个联赛、赛季不同，
   见 config/league_rules.yaml；踢附加赛也算没保住）。
     - 它在降级区里：离安全区最后一名（降级区上面一名）不超过 gap_points 分，而且剩余场次全胜能追上；
     - 它在降级区外：降级区里有队落后它不超过 gap_points 分，而且那队剩余场次全胜能追上它。
   “能追上”按积分相同就算（同分时谁在前要看净胜球等，赛前说不准，按“还有机会”算）。
   简化的地方：只看分差和剩余场次，不看两队之间还剩没剩直接交手，所以“数学上已经没机会”可能晚一两轮才认出来。
   次级联赛也算“争冠”（争第 1 名）；争升级名额（第 2、3 名、附加赛）先不做。

3. 欧战后的联赛：主队、客队分开标（home_after_euro、away_after_euro，True / False）。
   一个队“欧战后”= 这场开球前，它的上一场正式比赛（联赛、杯赛、欧战都算）是欧战（rest_days.euro_competitions），
   而且两场日期相差不超过 rest_days.euro_within_days 天（现在 4 天，例如周四欧联 → 周日联赛是 3 天）。
   判断不在这里另写，直接用梯度提升树特征的同一段程序 src/features/rest_days.py 的 RestDays（设置也是同一份），
   这样标签和模型里的“欧战后”那一列口径一样：以后按这个标签分组看模型准不准，看到的就是模型那一列对应的比赛。
   赛程经过取数层 FeatureStore.schedule 取，用途「休息天数和欧战后标签」（数据用途字典里杯赛欧战赛程只能这么用和显示）；
   只用日期和是哪个赛事，不用杯赛、欧战的对手和比分。

不偷看未来：
   积分榜的截止时间 = 本场“最晚可能的开球时间” − labels.cutoff_hours_before_kickoff 小时。
   有开球时间的比赛，就是开球那一刻；只有日期的比赛（2019/20 以前），是当天 24:00 英国时间，
   但同一天其他比赛的比分要到当天 24:00 + 3 小时才算知道（取数层的规则），所以同一天的比赛互相看不到，
   前一天以前的比分都能看到——和“赛前看积分榜”的意思一样。
   比分全部经过取数层 FeatureStore 取，用途“复盘”；一个联赛赛季只取一次，再在内存里按时间切。
   欧战后标签用同一个截止时间：赛程里只有“最晚可能的开球时间”不晚于截止时间的比赛才算进来
   （只知道轮次开始日的杯赛，最晚开球按这一轮第 7 天算，所以可能还不算）。赛程也是一个联赛赛季只取一次，
   再按截止时间从早到晚，一批一批交给 RestDays——每场比赛只看得到它自己截止时间之前已经开球的比赛。

运行（在项目文件夹里，约几分钟）：
    .venv\\Scripts\\python -m src.review.labels          全部联赛全部赛季，结果写到 data/labels/
    .venv\\Scripts\\python -m src.review.labels --comps E0 --seasons 2022-2023
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.common.config import load_settings, resolve_path, season_role
from src.features import FeatureStore, open_readonly
from src.features.known_time import compute_known_times, to_utc_text
from src.features.rest_days import RestDays
from src.features.standings import LeagueRules, SeasonTable, load_deductions, load_rules, season_schedule, season_table
from src.features.usage import season_label

USAGE = "复盘"
# 每类赛季用哪个运行模式算标签（需求文档第 17 节三种运行模式；数据用途字典里“复盘”在这几种组合下是允许的）：
# 开发赛季 → 开发模式；期末考试赛季 → 考试模式（开发模式连读都不能读）；上线后 → 实时模式。
MODE_FOR_ROLE = {"dev": "开发", "exam": "考试", "live": "实时"}


# ================= 德比名单 =================
def load_derbies(settings: dict | None = None, path: Path | None = None) -> dict[frozenset, dict]:
    """读 config/derbies.yaml，返回 {frozenset({队1, 队2}): {name, name_en, type, country}}。

    队名必须是 config/teams.yaml 里的“英文”（球队编号），写错直接报错；同一对写两次也报错。
    一条可以写 2 队以上（例如同城几支队），名单里任意两队都算这场德比。
    """
    settings = settings or load_settings()
    path = path or resolve_path(settings["labels"]["derbies"])
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    with open(resolve_path(settings["paths"]["teams"]), encoding="utf-8") as f:
        known = {t["英文"]: t["国家"] for t in yaml.safe_load(f)["球队"]}
    out: dict[frozenset, dict] = {}
    for d in raw["德比"]:
        teams = list(d["球队"])
        bad = [t for t in teams if t not in known]
        if bad:
            raise ValueError(f"derbies.yaml「{d['名字']}」：球队编号 {bad} 不在 config/teams.yaml 里")
        for i, a in enumerate(teams):
            for b in teams[i + 1:]:
                key = frozenset((a, b))
                if key in out:
                    raise ValueError(f"derbies.yaml：{a} 对 {b} 写了两次（「{out[key]['name']}」和「{d['名字']}」）")
                out[key] = {"name": d["名字"], "name_en": d.get("英文名", ""), "type": d["类型"],
                            "country": known[a]}
    return out


# ================= 争冠、保级 =================
def race_status(table: pd.DataFrame, rules: LeagueRules, last_rounds: int, gap_points: int) -> pd.DataFrame:
    """按一张赛前积分榜，判断每支队“在不在争冠”“在不在保级”（定义见文件开头）。

    table：standings.py 算出来的积分榜（按名次排好，有 team, position, points, remaining）。
    返回每队一行：team, title_race, relegation_race（True / False）。
    remaining 是这场开球前的剩余场次（含本场），所以本场还没踢。
    """
    t = table.reset_index(drop=True)
    n = len(t)
    pts = t["points"].to_numpy()
    rem = t["remaining"].to_numpy()
    w = rules.win_points
    late = rem <= last_rounds

    def can_reach(behind: int, ahead: int) -> bool:
        """名次靠后的 behind 能不能在“分差不超过 gap_points、剩余场次全胜追得上”的范围内够到 ahead。"""
        gap = pts[ahead] - pts[behind]
        return gap <= gap_points and gap <= w * rem[behind]

    title = [False] * n
    for i in range(n):
        if not late[i]:
            continue
        if i == 0:
            title[i] = any(can_reach(j, 0) for j in range(1, n))
        else:
            title[i] = can_reach(i, 0)

    releg = [False] * n
    m = rules.danger_spots
    if 0 < m < n:
        top_bad = n - m          # 降级区第一名的位置（从 0 数）
        last_safe = top_bad - 1  # 安全区最后一名
        for i in range(n):
            if not late[i]:
                continue
            if i >= top_bad:
                releg[i] = can_reach(i, last_safe)
            else:
                releg[i] = any(can_reach(j, i) for j in range(top_bad, n))
    return pd.DataFrame({"team": t["team"], "title_race": title, "relegation_race": releg})


# ================= 欧战后的联赛 =================
EURO_USAGE = "休息天数和欧战后标签"   # 数据用途字典里，杯赛欧战赛程对“以前的比赛”只允许这个用途和“显示”


def after_euro_flags(fs: FeatureStore, sched: pd.DataFrame, settings: dict) -> dict[int, tuple[bool, bool]]:
    """每场联赛比赛：(主队欧战后, 客队欧战后)。定义见文件开头第 3 条。

    sched：season_schedule 的结果，已经加上 cutoff_utc（每场的截止时间）。
    做法：这个联赛赛季的赛程（联赛 + 杯赛、欧战）经过取数层只取一次，截止时间取这季最大的那个；
    再按截止时间从早到晚一批一批交给 RestDays：处理某个截止时间的比赛时，
    交给 RestDays 的只有“最晚可能的开球时间”不晚于这个截止时间的比赛，所以每场只看得到自己截止时间之前的赛程。
    取赛程的起点 = 这季最早开球往前 cap_days 天：再早的比赛隔得比 cap_days 还久，
    在 RestDays 里和“找不到上一场”结果一样（都不算欧战后），不影响结果，也省得把几十年的赛程全取出来。
    """
    rest = RestDays.from_settings(settings)
    since = pd.to_datetime(sched["kickoff_earliest_utc"]).min() - pd.Timedelta(days=rest.cap)
    games = fs.schedule(sched["cutoff_utc"].max(), EURO_USAGE, since=since)
    games = games.assign(_known=pd.to_datetime(games["known_at_utc"])).sort_values("_known", kind="stable")
    games = games.reset_index(drop=True)
    known = games["_known"].to_numpy()
    out: dict[int, tuple[bool, bool]] = {}
    done = 0
    for cutoff, part in sched.groupby("cutoff_utc", sort=True):
        n = int(known.searchsorted(pd.Timestamp(cutoff).to_datetime64(), side="right"))
        rest.add(games.iloc[done:n])
        done = n
        for m in part.itertuples(index=False):
            out[int(m.match_id)] = (bool(rest.team(m.home_team, m.match_date)[1]),
                                    bool(rest.team(m.away_team, m.match_date)[1]))
    return out


# ================= 一个联赛赛季 =================
def label_season(fs: FeatureStore, competition: str, season: int, derbies: dict, settings: dict,
                 deductions: pd.DataFrame) -> pd.DataFrame:
    """给一个联赛赛季的每场比赛算标签。比分只经过 fs 取一次（截止时间 = 这季最后一个要算的时间点）。"""
    cfg = settings["labels"]
    race = cfg["race"]
    hours = float(cfg["cutoff_hours_before_kickoff"])
    sched = season_schedule(fs, competition, season, USAGE)
    if sched.empty:
        return pd.DataFrame()
    # 每场的截止时间 = 最晚可能的开球时间 − 提前量（为什么用“最晚”见文件开头）
    kt = compute_known_times(sched[["match_id", "season", "match_date", "kickoff_utc"]], settings["features"])
    latest = pd.to_datetime(kt["kickoff_latest_utc"].to_numpy())
    sched["cutoff_utc"] = (latest - pd.Timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    st = season_table(fs, competition, season, sched["cutoff_utc"].max(), USAGE, settings, schedule=sched,
                      deductions=deductions)
    rules = st.rules
    euro = after_euro_flags(fs, sched, settings)
    rows = []
    race_cache: dict[str, pd.DataFrame] = {}
    for m in sched.itertuples(index=False):
        if m.cutoff_utc not in race_cache:
            table = st.at(m.cutoff_utc)
            rs = race_status(table, rules, int(race["last_rounds"]), int(race["gap_points"]))
            race_cache[m.cutoff_utc] = table.merge(rs, on="team").set_index("team")
        tb = race_cache[m.cutoff_utc]
        h, a = tb.loc[m.home_team], tb.loc[m.away_team]
        d = derbies.get(frozenset((m.home_team, m.away_team)))
        n_title = int(h["title_race"]) + int(a["title_race"])
        n_rel = int(h["relegation_race"]) + int(a["relegation_race"])
        rows.append({
            "match_id": int(m.match_id), "competition": competition, "season": season,
            "season_role": season_role(season, settings), "match_date": m.match_date,
            "home_team": m.home_team, "away_team": m.away_team, "standings_cutoff_utc": m.cutoff_utc,
            "is_derby": d is not None, "derby_name": d["name"] if d else "", "derby_type": d["type"] if d else "",
            "is_title_race": n_title > 0, "title_race_teams": n_title,
            "is_relegation_race": n_rel > 0, "relegation_race_teams": n_rel,
            "home_position": int(h["position"]), "away_position": int(a["position"]),
            "home_points": int(h["points"]), "away_points": int(a["points"]),
            "home_remaining": int(h["remaining"]), "away_remaining": int(a["remaining"]),
            "teams_in_league": len(tb), "relegation_spots": rules.danger_spots,
            "tie_unresolved": bool(h["tie_unresolved"] or a["tie_unresolved"]),
            "home_after_euro": euro[int(m.match_id)][0], "away_after_euro": euro[int(m.match_id)][1],
        })
    return pd.DataFrame(rows)


# ================= 全部 =================
def all_seasons(fs: FeatureStore, competition: str, settings: dict) -> list[int]:
    """这个联赛数据库里有哪些赛季（从赛程里看，经过取数层；赛程是赛前就知道的）。"""
    df = fs.prediction_targets("1900-01-01", "2999-01-01", "显示", competition=competition)
    return sorted(int(s) for s in df["season"].unique())


def label_all(settings: dict | None = None, comps: list[str] | None = None, seasons: list[int] | None = None,
              conn=None, verbose: bool = True) -> pd.DataFrame:
    """给全部（或指定的）联赛赛季算标签，返回一张表（每场一行）。

    每类赛季用对应的运行模式建一个 FeatureStore（MODE_FOR_ROLE），所以考试赛季只在考试模式下读。
    conn：数据库连接，不给就只读打开 settings 里的数据库。
    """
    settings = settings or load_settings()
    conn = conn or open_readonly(resolve_path(settings["paths"]["database"]))
    comps = comps or [c for g in settings["leagues"].values() for c in g]
    derbies = load_derbies(settings)
    deductions = load_deductions(settings)
    stores: dict[str, FeatureStore] = {}

    def store(role: str) -> FeatureStore:
        if role not in stores:
            stores[role] = FeatureStore(conn, MODE_FOR_ROLE[role], settings)
        return stores[role]

    # “实时”模式能看到全部赛季的赛程，用它列出有哪些赛季（只取赛程，用途“显示”）
    out = []
    for comp in comps:
        t0 = time.time()
        have = all_seasons(store("live"), comp, settings)
        for s in (seasons or have):
            if s not in have:
                continue
            df = label_season(store(season_role(s, settings)), comp, s, derbies, settings, deductions)
            out.append(df)
        if verbose:
            print(f"{comp}: {sum(len(d) for d in out if len(d) and d['competition'].iat[0] == comp)} 场，"
                  f"{time.time() - t0:.1f} 秒")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def save_labels(df: pd.DataFrame, settings: dict | None = None) -> list[Path]:
    """存成 data/labels/联赛.csv（每个联赛一个文件；带 BOM 的 UTF-8，Excel 打开中文不乱码）。"""
    settings = settings or load_settings()
    folder = resolve_path(settings["labels"]["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for comp, part in df.groupby("competition"):
        p = folder / f"{comp}.csv"
        part.sort_values(["season", "match_date", "match_id"]).to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(p)
    return paths


def load_labels(mode: str, usage: str = USAGE, settings: dict | None = None, comps: list[str] | None = None,
                seasons: tuple[int, int] | None = None) -> pd.DataFrame:
    """读算好的标签，并按运行模式检查赛季（和取数层一样：例如开发模式读到期末考试赛季的标签就报错）。

    标签是从比分算出来的，考试赛季的标签也带着考试赛季的信息，所以同样要锁住。
    usage：读标签干什么。现在数据用途字典里还没有“自动标签”这一类（见报告），先按“比分 / 以前的比赛”的规则查用途。
    seasons：(起, 止) 赛季开始年份，只读这几个赛季，例如开发模式用 (1993, 2023)。不写 = 文件里全部赛季
             （开发模式下全部赛季里有考试赛季，会报错——这是故意的）。
    """
    from src.features.usage import DataUsage, PAST, normalize_mode
    settings = settings or load_settings()
    du = DataUsage(settings)
    du.check("比分", PAST, usage)
    m = normalize_mode(mode)
    folder = resolve_path(settings["labels"]["output_dir"])
    files = sorted(folder.glob("*.csv")) if comps is None else [folder / f"{c}.csv" for c in comps]
    df = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in files], ignore_index=True)
    if seasons is not None:
        # 先按赛季挑出要的行，再检查——挑掉的行不会被读到后面去
        df = df[df["season"].between(seasons[0], seasons[1])].reset_index(drop=True)
    # 赛季用途按赛季年份重新算一遍，不信文件里存的 season_role（文件可能是旧设置下算的）
    df["season_role"] = [season_role(int(x), settings) for x in df["season"]]
    for role, part in df.groupby("season_role"):
        seasons = "、".join(season_label(int(s)) for s in sorted(part["season"].unique()))
        du.check_mode(m, role, usage, f"标签文件里 {seasons} 赛季的 {len(part)} 场")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="复盘自动标签：德比、争冠、保级、欧战后的联赛")
    p.add_argument("--comps", default=None, help="联赛代码，逗号分隔；不写 = 全部")
    p.add_argument("--seasons", default=None, help="赛季范围，例如 2022-2023；不写 = 全部")
    a = p.parse_args()
    settings = load_settings()
    seasons = None
    if a.seasons:
        lo, hi = a.seasons.split("-")
        seasons = list(range(int(lo), int(hi) + 1))
    t0 = time.time()
    df = label_all(settings, a.comps.split(",") if a.comps else None, seasons)
    paths = save_labels(df, settings)
    print(f"共 {len(df)} 场，用时 {time.time() - t0:.0f} 秒；写到 {', '.join(str(x) for x in paths)}")


if __name__ == "__main__":
    main()
