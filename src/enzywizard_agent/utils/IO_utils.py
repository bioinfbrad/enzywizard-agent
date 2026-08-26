from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def normalize_path(value: str | Path | None) -> str | None:
    """把可选用户路径转换为去空白、展开用户目录后的绝对路径。"""

    # None 表示该可选输入没有提供，应原样保留。
    if value is None:
        return None

    # 同时兼容 argparse 传入的字符串和内部调用传入的 Path。
    value_text = str(value).strip()

    # 空字符串按未提供处理，避免解析为当前工作目录。
    if not value_text:
        return None

    # resolve 在这里规范化路径；文件是否存在由相应工具或后续校验负责。
    return str(Path(value_text).expanduser().resolve())


def create_session_id() -> str:
    """生成无连字符的随机 UUID，作为一次 Agent 会话的稳定标识。"""

    return uuid4().hex
