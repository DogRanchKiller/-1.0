"""Transfermarkt 球员数据镜像（salimt）下载程序的自动检查。用假网站，不联网。"""
import pytest

from src.collect.salimt_mirror import MirrorDownloader, Stopped, file_url
from src.common.config import load_settings
from src.common.db import connect

CFG = {"repo": "someone/repo", "commit": "abc123",
       "files": {"injuries": {"path": "dl/inj.csv", "header": "player_id,from_date"},
                 "transfers": {"path": "dl/tr.csv", "lfs": True, "header": "player_id,transfer_date"}}}
INJ = b"player_id,from_date,end_date\n1,2020-01-01,2020-02-01\n"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n"


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


def test_url_uses_fixed_commit_and_lfs_host():
    assert file_url(CFG, CFG["files"]["injuries"]) == "https://raw.githubusercontent.com/someone/repo/abc123/dl/inj.csv"
    assert file_url(CFG, CFG["files"]["transfers"]) == \
        "https://media.githubusercontent.com/media/someone/repo/abc123/dl/tr.csv"


def test_settings_pin_a_commit():
    cfg = load_settings()["salimt_mirror"]
    assert len(cfg["commit"]) == 40                                  # 写死完整的提交编号，不是分支名
    assert cfg["files"]["transfers"].get("lfs") is True


def test_download_once_and_record(make, tmp_path):
    ui, ut = (file_url(CFG, CFG["files"][k]) for k in ("injuries", "transfers"))
    con, site, d = make({ui: (200, INJ), ut: (200, LFS_POINTER)})
    d.run_mirror(CFG)
    out = tmp_path / "raw" / "salimt"
    assert (out / "injuries.csv").read_bytes() == INJ                # 原样保存
    assert (out / "transfers.suspicious.txt").exists()               # 拿到 LFS 指针小文件：单独存，不当正式文件
    assert not (out / "transfers.csv").exists()
    rows = con.execute("select source, data_type, status from fetch_log order by id").fetchall()
    assert rows == [("salimt-mirror", "transfermarkt_csv", "saved"),
                    ("salimt-mirror", "transfermarkt_csv", "suspicious")]
    d.run_mirror(CFG)                                                # 第二次：下载过的不再请求
    assert site.requests.count(ui) == 1


@pytest.mark.parametrize("code", [403, 429])
def test_stop_on_refusal(make, code):
    ui = file_url(CFG, CFG["files"]["injuries"])
    con, site, d = make({ui: (code, b"")})
    with pytest.raises(Stopped):
        d.run_mirror(CFG)
    assert site.requests == [ui]                                     # 停下，不再请求下一个
    assert con.execute("select status, http_status from fetch_log").fetchall() == [("failed", code)]
