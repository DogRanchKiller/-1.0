"""局域网网站的账号密码、局域网地址（需求文档第 8 节“局域网网站”、第 16 节“安全”）。

这里管四件事：
1. 自动生成随机密码，手机上好输入：4 组、每组 4 个小写字母或数字，例如 “k7mp-3xq9-…”这种样子（这里只是举格式）。
2. 密码只存“加盐哈希”：把密码和一串随机数（盐）一起反复算 60 万次 SHA256（PBKDF2），存下结果。
   登录时把输入的密码同样算一遍，结果一样才放行。哈希不能倒推出密码，所以 credentials.json 被人看到也拿不到密码。
3. 算出这台电脑在局域网里的地址（例如 192.168.1.23），写进 data/site_server/网站地址和密码.txt 给你看。
   密码明文只写在这个 txt 里（data/ 不进 Git），对话、日志、屏幕上都不显示。
4. 双击用的两个小命令：
       .venv\\Scripts\\python -m src.web.serve_auth show     “查看网站地址和密码.bat”：没有密码就生成，刷新地址，打开 txt
       .venv\\Scripts\\python -m src.web.serve_auth change   “修改网站密码.bat”：输入新密码（不显示），或回车随机生成
   另外 firewall-info 给 PowerShell 脚本读端口、规则名、任务名（开启/关闭局域网访问.bat 用）。
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import re
import secrets
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.common.config import load_settings, resolve_path

CRED_FILE = "credentials.json"        # 密码哈希
SESSIONS_FILE = "sessions.json"        # 登录会话（只存会话的哈希）
INFO_FILE = "网站地址和密码.txt"        # 给你看的地址和密码
SERVER_MARK = "football-lan-site"      # 服务器在每个响应里带 X-Site-Server: 这个值，用来认出“端口上跑的是不是我们自己的服务器”

# 生成密码用的字符：小写字母 + 数字，去掉容易看错的 0 o 1 l i。一共 31 个字符，
# 16 个字符的密码约有 31^16 ≈ 2^79 种可能，加上 60 万次哈希和输错锁定，猜中是不现实的
ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
ALGORITHM = "pbkdf2_sha256"
BEIJING = ZoneInfo("Asia/Shanghai")


class CredentialsError(Exception):
    """credentials.json 坏了（格式不对、字段缺了）。不自动覆盖，免得把你的密码弄丢。"""


# ---------- 文件位置 ----------

def state_dir(settings: dict) -> Path:
    return resolve_path(settings["site_server"]["state_dir"])


def cred_path(settings: dict) -> Path:
    return state_dir(settings) / CRED_FILE


def info_path(settings: dict) -> Path:
    return state_dir(settings) / INFO_FILE


def write_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """先写到旁边的临时文件再换上去：写到一半断电，原来的文件还在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


# ---------- 密码 ----------

def generate_password(groups: int = 4, group_len: int = 4) -> str:
    """生成随机密码：groups 组、每组 group_len 个字符，用短横线连起来。用 secrets（专门做密码的随机数），不用 random。"""
    return "-".join("".join(secrets.choice(ALPHABET) for _ in range(group_len)) for _ in range(groups))


def normalize(password: str, generated: bool) -> str:
    """自动生成的密码：不分大小写，短横线和空格可以不输（手机上输入方便）。自己设的密码原样比较。"""
    if not generated:
        return password
    return re.sub(r"[\s\-]", "", password).lower()


def make_record(username: str, password: str, iterations: int, generated: bool, version: int) -> dict:
    """算出要存进 credentials.json 的内容：账号、盐、哈希、迭代次数。不含密码本身。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", normalize(password, generated).encode("utf-8"), salt, iterations)
    return {
        "username": username,
        "algorithm": ALGORITHM,
        "iterations": iterations,          # 次数跟着存：以后 settings 改了次数，旧密码照样能验证
        "salt": salt.hex(),
        "hash": digest.hex(),
        "generated": generated,            # 是不是自动生成的（决定比较时忽略大小写和短横线）
        "version": version,                # 每改一次密码加 1；会话记着登录时的版本，版本不对就要重新登录
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def verify_password(password: str, record: dict) -> bool:
    """输入的密码对不对。比较用 hmac.compare_digest：不管错在第几位，花的时间都一样，别人没法靠计时一位位猜。"""
    try:
        if record.get("algorithm") != ALGORITHM:
            return False
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
        iterations = int(record["iterations"])
    except (KeyError, TypeError, ValueError):
        return False
    if iterations < 1 or len(expected) != 32 or not isinstance(password, str):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", normalize(password, bool(record.get("generated"))).encode("utf-8"),
                                 salt, iterations)
    return hmac.compare_digest(digest, expected)


# ---------- 操作按钮的防伪口令（CSRF，需求文档第 16 节“安全”“网站上的操作按钮”） ----------
# CSRF：别的网站在你登录着的时候，偷偷让你的浏览器往本网站发一个“点按钮”的请求。
# 防法之一：本网站自己的页面里放一个口令，提交时带回来；别的网站看不到本网站的页面，就拿不到这个口令。
# 口令 = HMAC-SHA256(服务器启动时随机生成的密钥, "actions:" + 这次登录的会话 cookie)：
#   - 和登录绑定：换了登录（别的设备、退出再登）口令就不一样，一个人的口令给别人用不了；
#   - 服务器不用另外存口令，每次现算；密钥只在内存里（new_csrf_key），服务器重启后旧口令作废，刷新页面就有新的。

CSRF_FIELD = "csrf"                    # 表单里那个隐藏字段的名字


def new_csrf_key() -> bytes:
    """服务器启动时生成一次：32 字节随机数，只放内存，不写文件、不进日志。"""
    return secrets.token_bytes(32)


def csrf_token(key: bytes, session_token: str) -> str:
    """这次登录对应的防伪口令（64 个十六进制字符）。"""
    return hmac.new(key, ("actions:" + session_token).encode("ascii"), hashlib.sha256).hexdigest()


def csrf_valid(key: bytes, session_token: str | None, given: str | None) -> bool:
    """提交来的口令对不对。没登录、没带、格式不对都算不对；比较用 hmac.compare_digest（花的时间和错在第几位无关）。"""
    if not session_token or not isinstance(given, str) or not re.fullmatch(r"[0-9a-f]{64}", given):
        return False
    return hmac.compare_digest(csrf_token(key, session_token), given)


def load_credentials(path: Path) -> dict | None:
    """读密码哈希。文件不存在返回 None；文件坏了抛 CredentialsError（不当作“没有密码”，免得被自动覆盖）。"""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CredentialsError(f"{path.name} 读不出来：{type(e).__name__}") from None
    need = {"username": str, "algorithm": str, "iterations": int, "salt": str, "hash": str, "version": int}
    if not isinstance(rec, dict) or any(not isinstance(rec.get(k), t) for k, t in need.items()):
        raise CredentialsError(f"{path.name} 的内容不完整")
    return rec


def save_credentials(path: Path, record: dict) -> None:
    write_atomic(path, json.dumps(record, ensure_ascii=False, indent=2))


def ensure_credentials(settings: dict) -> tuple[dict, str | None]:
    """没有密码就生成一个。返回 (哈希记录, 新生成的明文密码；原来就有密码时是 None)。文件坏了照样抛错，不覆盖。"""
    ss = settings["site_server"]
    path = cred_path(settings)
    rec = load_credentials(path)
    if rec is not None:
        return rec, None
    plain = generate_password(ss["password_groups"], ss["password_group_len"])
    rec = make_record(ss["username"], plain, ss["pbkdf2_iterations"], generated=True, version=1)
    save_credentials(path, rec)
    return rec, plain


def set_password(settings: dict, plain: str, generated: bool) -> dict:
    """换密码：版本号加 1（已经登录的设备都要重新登录），写哈希，再刷新 txt。"""
    ss = settings["site_server"]
    path = cred_path(settings)
    try:
        old = load_credentials(path)
    except CredentialsError:
        old = None                         # 原来的文件坏了：你主动改密码，就直接换成新的
    rec = make_record(ss["username"], plain, ss["pbkdf2_iterations"], generated,
                      version=(old["version"] + 1) if old else 1)
    save_credentials(path, rec)
    write_info_file(settings, plain=plain)
    return rec


# ---------- 局域网地址 ----------

def _usable(ip: str) -> bool:
    """能给手机用的 IPv4：去掉本机回环（127.x）、自动分配失败时的 169.254.x、组播和 0.0.0.0。"""
    try:
        a = ipaddress.IPv4Address(ip)
    except ValueError:
        return False
    return not (a.is_loopback or a.is_link_local or a.is_multicast or a.is_unspecified)


def _private(ip: str) -> bool:
    """家里路由器常发的那几段：10.x、172.16–31.x、192.168.x。"""
    a = ipaddress.IPv4Address(ip)
    return any(a in ipaddress.IPv4Network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


def lan_addresses() -> tuple[str | None, list[str]]:
    """算出这台电脑在局域网里的 IPv4 地址。返回 (最可能的那个, 其余的)。

    最可能的那个：让系统“假装”往外连一下（UDP 不真的发数据），看它选哪块网卡出去，那块网卡的地址就是 Wi-Fi / 网线的地址。
    其余的：电脑名查出来的所有地址（虚拟机、VPN 的虚拟网卡也会在里面，手机一般连不上它们）。
    """
    primary = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))   # 文档专用地址，不会真的发出去；只是让系统选一块网卡
            ip = s.getsockname()[0]
            if _usable(ip):
                primary = ip
    except OSError:
        pass
    others: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if _usable(ip):
                others.add(ip)
    except OSError:
        pass
    others.discard(primary)
    # 192.168 / 10 / 172.16 这些家用网段排前面
    ordered = sorted(others, key=lambda ip: (not _private(ip), tuple(int(x) for x in ip.split("."))))
    if primary is None and ordered:
        primary = ordered.pop(0)
    return primary, ordered


def mdns_name() -> str | None:
    """电脑名.local（局域网里按名字找电脑）。电脑名里有中文、空格等时不能这样用，返回 None。"""
    name = socket.gethostname().strip().lower()
    if not name or not re.fullmatch(r"[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?", name):
        return None
    return f"{name}.local"


def probe_server(port: int, timeout: float = 2.0) -> bool:
    """本机这个端口上跑的是不是我们自己的网站服务器（看响应头 X-Site-Server）。"""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("HEAD", "/login")
        resp = conn.getresponse()
        ok = resp.getheader("X-Site-Server") == SERVER_MARK
        conn.close()
        return ok
    except (OSError, http.client.HTTPException):
        return False


# ---------- 给你看的 txt ----------

_PW_LINE = re.compile(r"^密码：(\S+)\s*$", re.M)


def _old_plain(path: Path, rec: dict) -> str | None:
    """旧 txt 里写的密码。只有它和现在的哈希对得上才用（你改过密码、txt 过时了就不用）。"""
    try:
        m = _PW_LINE.search(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError):
        return None
    if m and verify_password(m.group(1), rec):
        return m.group(1)
    return None


def write_info_file(settings: dict, plain: str | None = None, port: int | None = None,
                    running: bool | None = None, addresses: tuple[str | None, list[str]] | None = None) -> Path:
    """写 data/site_server/网站地址和密码.txt：网址（每次重新算，IP 会变）、账号、密码、怎么用。

    plain：刚生成或刚改的密码；不给就从旧 txt 里找（对得上哈希才用）。
    running：服务器在不在运行（None = 不写这一行）。addresses：测试时给假的地址。
    """
    ss = settings["site_server"]
    path = info_path(settings)
    rec = load_credentials(cred_path(settings))
    port = port or ss["port"]
    primary, others = addresses if addresses is not None else lan_addresses()
    if plain is None and rec is not None:
        plain = _old_plain(path, rec)
    now = datetime.now(timezone.utc).astimezone(BEIJING).strftime("%Y-%m-%d %H:%M")

    lines = ["足球预测 局域网网站", f"（{now} 北京时间更新。路由器重启后地址可能变，再双击“查看网站地址和密码.bat”看最新的）", ""]
    lines.append("【手机怎么打开】手机连上和电脑同一个 Wi-Fi，打开浏览器，在地址栏输入下面的网址：")
    if primary:
        lines.append(f"    http://{primary}:{port}        ← 先试这个")
    else:
        lines.append("    （现在没找到这台电脑的局域网地址：电脑可能没连上 Wi-Fi 或网线）")
    for ip in others:
        lines.append(f"    http://{ip}:{port}        （上面那个打不开再试这个，可能是虚拟网卡）")
    name = mdns_name()
    if name:
        lines.append(f"    http://{name}:{port}        （用电脑名打开：苹果手机一般可以，有的安卓手机不认这种）")
    lines.append(f"    这台电脑自己打开：http://127.0.0.1:{port}")
    lines.append("")
    if rec is None:
        lines += ["账号、密码：还没设置。双击“查看网站地址和密码.bat”会自动生成。"]
    else:
        lines.append(f"账号：{rec['username']}")
        if plain:
            lines.append(f"密码：{plain}")
            if rec.get("generated"):
                lines.append("      （输入时不分大小写，短横线可以不输）")
        else:
            lines.append("密码：（这里没有记录。可能是改过密码；忘了就双击“修改网站密码.bat”换一个）")
    lines.append("")
    if running is not None:
        lines.append("网站服务器：" + ("正在运行" if running else
                     "现在没在运行（注销再登录 Windows，或等几分钟它会自动启动；还不行就告诉 Claude）"))
        lines.append("")
    lines += [
        "【注意】",
        "1. 电脑要开着、网站服务器要在运行（登记了定时任务后，登录 Windows 时自动启动）。",
        "2. 第一次用之前要双击一次“开启局域网访问.bat”（防火墙放行），手机才连得进来。",
        "3. 只在家里自己的 Wi-Fi 用。网站是 http（不加密），咖啡店、酒店这类公共 Wi-Fi 不要用。",
        "4. 登录一次 30 天内不用再登；点页面顶上的“退出登录”可以退出。",
        "5. 这个文件里有密码，不要发给别人、不要截图发群里。",
    ]
    # 带 BOM 的 UTF-8：用 Windows 记事本打开不会乱码
    write_atomic(path, "\r\n".join(lines) + "\r\n", encoding="utf-8-sig")
    return path


# ---------- 双击用的小命令 ----------

def _say(text: str = "") -> None:
    """往屏幕打一行字。没有黑窗口（pythonw）时什么都不做。"""
    if sys.stdout is not None:
        print(text)


def cmd_show(settings: dict, open_file: bool = True) -> int:
    """查看网站地址和密码：没有密码就生成一个，刷新地址，打开 txt。"""
    try:
        _rec, new_plain = ensure_credentials(settings)
    except CredentialsError as e:
        _say(f"密码文件有问题：{e}。没有改它。双击“修改网站密码.bat”可以重新设一个。")
        return 1
    running = probe_server(settings["site_server"]["port"])
    # 刚生成的密码要直接写进去（这时旧 txt 里还没有它）
    path = write_info_file(settings, plain=new_plain, running=running)
    _say(f"地址和密码写在：{path}")
    if open_file and hasattr(os, "startfile"):
        os.startfile(str(path))
    return 0


def cmd_change(settings: dict, ask=getpass.getpass) -> int:
    """修改网站密码：输入两次新密码（屏幕上不显示）；直接回车就随机生成一个。ask 是读输入的函数（测试时换成假的）。"""
    ss = settings["site_server"]
    _say("修改网站密码")
    _say(f"输入新密码后按回车（为了安全，屏幕上不显示你输入的字）。至少 {ss['min_password_length']} 位。")
    _say("什么都不输、直接按回车：电脑随机生成一个好输入的密码。")
    first = ask("新密码：")
    if first == "":
        plain, generated = generate_password(ss["password_groups"], ss["password_group_len"]), True
    else:
        if any(ch.isspace() for ch in first):
            # 带空格的密码手机上很容易输错，而且“网站地址和密码.txt”按一整段不带空格的字读回密码
            _say("没有修改：密码里有空格，手机上很容易输错。请重新双击再试。")
            return 1
        if len(first) < ss["min_password_length"]:
            _say(f"没有修改：只有 {len(first)} 位，至少要 {ss['min_password_length']} 位。请重新双击再试。")
            return 1
        if ask("再输一次：") != first:
            _say("没有修改：两次输入不一样。请重新双击再试。")
            return 1
        plain, generated = first, False
    set_password(settings, plain, generated)
    _say("改好了。新密码写在 data/site_server/网站地址和密码.txt（双击“查看网站地址和密码.bat”打开）。")
    _say("已经登录的手机、电脑都要用新密码重新登录。")
    return 0


def cmd_firewall_info(settings: dict) -> int:
    """给 PowerShell 脚本用：端口、防火墙规则名、任务名（JSON，中文转成 \\uXXXX，避免控制台编码问题）。"""
    ss = settings["site_server"]
    print(json.dumps({"port": ss["port"], "rule": ss["firewall_rule"], "task": ss["task_name"]}, ensure_ascii=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()
    cmds = {"show": cmd_show, "change": cmd_change, "firewall-info": cmd_firewall_info}
    if len(argv) != 1 or argv[0] not in cmds:
        _say("用法：python -m src.web.serve_auth show | change | firewall-info")
        return 2
    return cmds[argv[0]](settings)


if __name__ == "__main__":
    sys.exit(main())
