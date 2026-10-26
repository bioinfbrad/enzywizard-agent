from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;]+)"
)
_API_KEY_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*[\"']?)([^\s,;\"']+)"
)
_COMMON_KEY_PATTERN = re.compile(r"\b(?:sk|dashscope|AIza)[-_A-Za-z0-9]{12,}\b")
_SENSITIVE_MAPPING_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "bearer_token",
    "client_secret",
    "password",
    "passphrase",
    "private_key",
    "secret",
    "token",
}


def _is_sensitive_mapping_key(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return normalized in _SENSITIVE_MAPPING_KEYS or normalized.endswith(
        (
            "_api_key",
            "_authorization",
            "_token",
            "_secret",
            "_password",
            "_passphrase",
            "_private_key",
        )
    )


def redact_text(value: Any, *, api_key: str | None = None) -> str:
    """移除日志和 manifest 错误文本中的认证信息。"""

    text = str(value)
    if api_key:
        text = text.replace(api_key, "[REDACTED_API_KEY]")
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "$HOME")
    text = _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", text)
    text = _API_KEY_PATTERN.sub(r"\1[REDACTED]", text)
    return _COMMON_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)


def redact_mapping(value: Any, *, api_key: str | None = None) -> Any:
    """递归脱敏用户显式传入的扩展参数，不修改原始配置。"""

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_mapping_key(key):
                output[str(key)] = "[REDACTED]"
            else:
                output[str(key)] = redact_mapping(item, api_key=api_key)
        return output
    if isinstance(value, list):
        return [redact_mapping(item, api_key=api_key) for item in value]
    if isinstance(value, str):
        return redact_text(value, api_key=api_key)
    return value


def redact_url(value: str, *, api_key: str | None = None) -> str:
    """保留服务端地址用于复现，同时移除用户信息、查询参数和片段。"""

    redacted = redact_text(value, api_key=api_key)
    try:
        parts = urlsplit(redacted)
        hostname = parts.hostname or ""
        if parts.port is not None:
            hostname = f"{hostname}:{parts.port}"
        return urlunsplit((parts.scheme, hostname, parts.path, "", ""))
    except (TypeError, ValueError):
        return redacted


def privacy_safe_file_record(path_value: str | Path, session_dir: Path) -> dict[str, Any]:
    """生成不暴露外部绝对目录的文件标识；校验和由 manifest 层补充。"""

    path = Path(path_value).expanduser()
    try:
        relative_path = path.resolve().relative_to(session_dir.resolve())
        display_path = str(relative_path)
    except (OSError, ValueError):
        display_path = path.name
    return {"path": display_path, "exists": path.is_file()}
