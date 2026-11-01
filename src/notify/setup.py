"""设置微信推送的口令（“设置微信推送.bat”运行的就是这个）。

做什么：让你粘贴 pushplus 网站上的 token（输入时屏幕上不显示，防止被旁人看到或截图），
写进 config/secrets.yaml 的 pushplus.token。secrets.yaml 里原来的其他内容保留不动
（注意：文件里原来的 # 注释行会丢掉，内容不会丢）。secrets.yaml 不进 Git，不会传到 GitHub。

屏幕上只显示口令有几个字符，不显示口令本身。

运行：.venv\\Scripts\\python -m src.notify.setup
"""
import getpass
import os
from pathlib import Path

import yaml

from src.common.config import CONFIG_DIR


def save_token(token: str, path: Path | None = None) -> None:
    """把口令写进 secrets.yaml 的 pushplus.token，其他键原样保留。文件不存在就新建。"""
    path = Path(path or CONFIG_DIR / "secrets.yaml")
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            # 不认识的格式不覆盖，免得把里面别的密钥弄丢
            raise ValueError(f"{path.name} 的格式看不懂（最外层不是“键: 值”），没有改它。请告诉 Claude 看一下")
    pp = data.get("pushplus")
    data["pushplus"] = {**(pp if isinstance(pp, dict) else {}), "token": token}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def check_token(token: str) -> str | None:
    """简单检查一下粘贴的内容；有问题返回中文说明，没问题返回 None。"""
    if not token:
        return "什么都没粘贴进来"
    if any(c.isspace() for c in token):
        return "里面有空格或换行，可能多复制了别的字"
    if len(token) < 16:
        return f"只有 {len(token)} 个字符，pushplus 的 token 一般要长得多，可能没复制全"
    return None


def main(ask=getpass.getpass, path: Path | None = None) -> bool:
    """问口令、检查、保存。成功返回 True。ask 是读输入的函数（测试时换成假的）。"""
    print("设置微信推送（pushplus 推送加）")
    print("在 pushplus.plus 网站登录后，复制“一对一推送”里的 token，")
    print("然后在下面右键粘贴（或按 Ctrl+V），按回车。为了安全，粘贴后屏幕上不会显示任何字。")
    token = ask("token：").strip()
    problem = check_token(token)
    if problem:
        print(f"没有保存：{problem}。请重新双击“设置微信推送.bat”再试。")
        return False
    try:
        save_token(token, path)
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"没有保存：{str(e).replace(token, '***')}")
        return False
    print(f"已保存（{len(token)} 个字符），写在 config/secrets.yaml，不会上传 GitHub。")
    print("下一步：双击“发一条测试推送.bat”，看微信能不能收到。")
    return True


if __name__ == "__main__":
    main()
