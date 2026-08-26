from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """保存一次 Agent 会话内不应变化的运行配置。

    frozen=True 防止节点在运行过程中意外修改模型连接信息或工作目录。
    """

    # OpenAI 兼容接口的认证和连接配置。
    api_key: str
    base_url: str
    model: str

    # Agent 产生文件时使用的顶层工作目录。
    workdir: Path

    # 允许的最大会话轮数，目前保留给后续循环控制逻辑使用。
    max_turns: int = 3
