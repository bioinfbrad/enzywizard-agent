from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class AgentConfig:
    """保存一次 Agent 会话内不应变化的运行配置。

    frozen=True 防止节点在运行过程中意外修改模型连接信息或工作目录。
    """

    # OpenAI 兼容接口的认证和连接配置。
    api_key: str = field(repr=False)
    base_url: str
    model: str

    # Agent 产生文件时使用的顶层工作目录。
    workdir: Path

    # 允许的最大会话轮数；Analysis 在最后一轮回答后结束状态图。
    max_turns: int = 3

    # 单次模型请求的最长等待时间，防止 CLI 因远端请求长期无响应而卡住。
    llm_timeout_seconds: float = 60.0

    # OpenAI 兼容客户端仅对超时、限流和服务端错误执行有限重试；结构化语义
    # 错误仍由各阶段的编译与校验逻辑处理，不通过重复调用掩盖。
    llm_max_retries: int = 2

    # OpenAI Chat Completions 标准参数。None 表示完全不向服务端发送该参数，
    # 由具体模型采用自己的默认行为。
    temperature: float | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None

    # 供应商扩展只允许由用户显式提供；核心代码不再写死百炼或其他服务参数。
    llm_extra_body: dict[str, Any] | None = None

    # 启动时使用 Router、History Selection 和 Planning Structure 三个代表性
    # 严格 JSON Schema 请求验证模型契约；不支持原生协议时拒绝启动。
    check_model_capabilities: bool = True

    # FAIR terminology enrichment is enabled by default. Retrieval uses only the
    # versioned local YAML knowledge base and never accesses the network.
    ontology_mode: Literal["off", "retrieve", "control"] = "retrieve"

    # Tool recovery remains bounded by deterministic policy. Hybrid diagnosis may
    # explain an unknown failure and propose an identical retry, but it cannot run
    # arbitrary commands or change scientific inputs without user confirmation.
    tool_recovery_mode: Literal["off", "safe"] = "safe"
    max_tool_recovery_attempts: int = 1
    failure_diagnosis_mode: Literal["deterministic", "hybrid"] = "hybrid"

    # 会话资源预算。None 表示该维度不设上限；默认值足以覆盖正常三轮会话，
    # 同时防止失控循环或长期挂起。
    max_llm_calls: int | None = 64
    max_input_tokens: int | None = 500_000
    max_output_tokens: int | None = 128_000
    session_timeout_seconds: float | None = 21_600.0
    tool_timeout_seconds: float | None = 3_600.0

    def __post_init__(self) -> None:
        """拒绝明显非法配置，并复制可变扩展参数。"""

        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be greater than zero")
        if self.llm_max_retries < 0:
            raise ValueError("llm_max_retries must not be negative")
        for name in (
            "max_completion_tokens",
            "max_llm_calls",
            "max_input_tokens",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be greater than zero when provided")
        for name in ("session_timeout_seconds", "tool_timeout_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero when provided")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.ontology_mode not in {"off", "retrieve", "control"}:
            raise ValueError("ontology_mode must be 'off', 'retrieve', or 'control'")
        if self.tool_recovery_mode not in {"off", "safe"}:
            raise ValueError("tool_recovery_mode must be 'off' or 'safe'")
        if self.max_tool_recovery_attempts < 0:
            raise ValueError("max_tool_recovery_attempts must not be negative")
        if self.failure_diagnosis_mode not in {"deterministic", "hybrid"}:
            raise ValueError(
                "failure_diagnosis_mode must be 'deterministic' or 'hybrid'"
            )
        if self.llm_extra_body is not None:
            object.__setattr__(self, "llm_extra_body", dict(self.llm_extra_body))
