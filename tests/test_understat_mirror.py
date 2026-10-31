"""Understat 射门明细镜像下载程序的自动检查。用假网站，不联网。"""
import pytest

from src.collect.understat_mirror import MirrorDownloader, Stopped, file_url, season_tag
from src.common.db import connect

CFG = {"repo": "someone/repo", "commit": "abc123", "first_season": 2014, "last_season": 2015,
       "leagues": {"E0": "epl"}}
SHOTS = b"id,minute,result,X,Y,xG,player,h_a,match_id\n1,3,Goal,0.9,0.5,0.4,A,h,100\n"


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content


class FakeSite:
    """pages：网址 → (状态码, 内容)；没有的网址返回 404。"""
    def __init__(self, pages):
        self.pages, self.requests, self.headers = pages, [], {}

    def get(self, url, timeout):
        self.requests.append(url)
        return FakeResponse(*self.pages.get(url, (404, b"")))


@pytest.fixture
def make(tmp_path):
    con = connect(tmp_path / "test.db")
    def build(pages):
        site = FakeSite(pages)
        return con, site, MirrorDownloader(con, tmp_path / "raw", min_interval=0, session=site, sleep=lambda s: None)
    return build


def test_url_uses_fixed_commit():
    assert season_tag(2014) == "14-15"
    assert file_url(CFG, "epl", 2021) == \
        "https://raw.githubusercontent.com/someone/repo/abc123/datasets/epl/shots_epl_21-22.csv"


def test_download_once_and_record(make, tmp_path):
    u14, u15 = file_url(CFG, "epl", 2014), file_url(CFG, "epl", 2015)
    con, site, d = make({u14: (200, SHOTS), u15: (200, b"<html>error</html>")})
    d.run_mirror(CFG)
    out = tmp_path / "raw" / "understat_mirror" / "E0"
    assert (out / "2014.csv").read_bytes() == SHOTS          # 原样保存
    assert (out / "2015.suspicious.txt").exists()            # 不像射门明细：单独存，不当正式文件
    rows = con.execute("select source, data_type, status from fetch_log order by id").fetchall()
    assert rows == [("understat-mirror", "understat_shots", "saved"),
                    ("understat-mirror", "understat_shots", "suspicious")]
    d.run_mirror(CFG)                                        # 第二次：2014 下载过就不再请求
    assert site.requests.count(u14) == 1


@pytest.mark.parametrize("code", [403, 429])
def test_stop_on_refusal(make, code):
    u14 = file_url(CFG, "epl", 2014)
    con, site, d = make({u14: (code, b"")})
    with pytest.raises(Stopped):
        d.run_mirror(CFG)
    assert site.requests == [u14]                            # 停下，不再请求下一个
    assert con.execute("select status, http_status from fetch_log").fetchall() == [("failed", code)]
