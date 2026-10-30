"""football-data 下载程序的自动检查。用一个假网站代替真网站，不联网。"""
from datetime import date

import pytest

from src.collect.football_data import BASE_URL, Downloader, current_season, looks_like_csv, season_code
from src.common.db import connect

CSV = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,16/08/24,Man United,Fulham,1,0\n"


@pytest.mark.parametrize("year, code", [(2024, "2425"), (1999, "9900"), (2000, "0001"), (1993, "9394")])
def test_season_code(year, code):
    assert season_code(year) == code


@pytest.mark.parametrize("day, season", [
    (date(2026, 10, 30), 2026), (date(2027, 3, 1), 2026),   # 2026/27 赛季中
    (date(2027, 6, 30), 2026), (date(2027, 7, 1), 2027),    # 7 月起算新赛季
])
def test_current_season(day, season):
    assert current_season(day) == season


def test_looks_like_csv():
    assert looks_like_csv(CSV)
    assert looks_like_csv(b"\xef\xbb\xbf" + CSV)          # 文件开头带 BOM 也认
    assert not looks_like_csv(b"<html>Not found</html>")


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content


class FakeSite:
    """假网站：pages 里有的网址返回内容，没有的返回 404。记下每次请求。"""
    def __init__(self, pages):
        self.pages, self.requests, self.headers = pages, [], {}

    def get(self, url, timeout):
        self.requests.append(url)
        return FakeResponse(200, self.pages[url]) if url in self.pages else FakeResponse(404)


@pytest.fixture
def setup(tmp_path):
    con = connect(tmp_path / "test.db")
    def make(pages):
        site = FakeSite(pages)
        return site, Downloader(con, tmp_path / "raw", min_interval=0, session=site, sleep=lambda s: None)
    return con, tmp_path / "raw" / "football-data", make


def url(code, lg="E0"):
    return f"{BASE_URL}/mmz4281/{code}/{lg}.csv"


def test_finished_season_downloaded_once(setup):
    con, out, make = setup
    site, d = make({url("2425"): CSV})
    d.run(["E0"], date(2026, 10, 30))
    assert (out / "E0" / "2425.csv").read_bytes() == CSV        # 原样保存
    assert (out / "E0" / "9394.csv").exists() is False          # 网站没有的不建文件
    site2, d2 = make({url("2425"): CSV})
    d2.run(["E0"], date(2026, 10, 31))
    assert url("2425") not in site2.requests                    # 第二次运行不再请求已有的赛季
    assert url("9394") not in site2.requests                    # 也不再请求“网站没有”的


def test_current_season_snapshots(setup):
    con, out, make = setup
    _, d = make({url("2627"): CSV})
    d.run(["E0"], date(2026, 10, 30))
    _, d = make({url("2627"): CSV})
    d.run(["E0"], date(2026, 10, 31))                           # 内容没变：不存新文件
    new = CSV + b"E0,01/11/26,Arsenal,Chelsea,2,2\n"
    _, d = make({url("2627"): new})
    d.run(["E0"], date(2026, 11, 2))                            # 内容变了：存一份新快照
    files = sorted(p.name for p in (out / "E0" / "2627").glob("*.csv"))
    assert files == ["2026-10-30.csv", "2026-11-02.csv"]
    statuses = [r[0] for r in con.execute(
        "select status from fetch_log where url = ? order by id", (url("2627"),))]
    assert statuses == ["saved", "unchanged", "saved"]


def test_error_page_marked_suspicious(setup):
    con, out, make = setup
    _, d = make({url("2425"): b"<html>error</html>"})
    d.run(["E0"], date(2026, 10, 30))
    assert not (out / "E0" / "2425.csv").exists()               # 不当成正常数据保存
    assert (out / "E0" / "2425.suspicious.html").exists()
    assert con.execute("select count(*) from fetch_log where status = 'suspicious'").fetchone()[0] == 1


def test_every_request_logged(setup):
    con, out, make = setup
    site, d = make({url("2425"): CSV})
    d.run(["E0"], date(2026, 10, 30))
    n_logged = con.execute("select count(*) from fetch_log").fetchone()[0]
    assert n_logged == len(site.requests)                       # 每次请求都有记录，包括 404


def test_http_300_counts_as_missing(setup):
    """真网站对不存在的早期赛季（如意乙 1993/94）回 HTTP 300，要当成“网站没有”，下次不再请求。"""
    con, out, make = setup
    site, d = make({url("2425"): CSV})
    site.get = lambda u, timeout: (site.requests.append(u), FakeResponse(300) if "9394" in u else FakeResponse(404))[1]
    d.fetch(url("9394"), "league_csv", out / "E0" / "9394.csv")
    assert con.execute("select status, http_status from fetch_log").fetchone() == ("not_found", 300)
    assert d.fetch(url("9394"), "league_csv", out / "E0" / "9394.csv") == "skipped"
    assert site.requests.count(url("9394")) == 1
