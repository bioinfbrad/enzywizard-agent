from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# =============================================================================
# Enum types
# =============================================================================

KnowledgeQueryType = Literal[
    # Planning 查询 EnzyWizard 工具规格。
    "tool_spec",
    # Analysis 查询分析模板或参考案例。
    "analysis_template",
    "analysis_example",
    # 查询已知失败模式或 GitHub 问题。
    "failure_case",
    "github_issue",
]


# =============================================================================
# Output schemas
# =============================================================================


# =============================================================================
# State data models
# =============================================================================


class KnowledgeQuery(BaseModel):
    """记录一次知识查询的输入、来源和返回摘要，便于后续追踪。"""

    # 查询发生的会话轮次及查询用途。
    turn_id: int
    query_type: KnowledgeQueryType

    # 提交给知识源的实际查询文本。
    query: str

    # 知识库或远程来源的逻辑名称。
    knowledge_source: str

    # 命中的知识条目标识列表，不直接保存大段原文。
    retrieved_knowledge_items: list[str]

    # 对命中知识的可选摘要，供后续 Prompt 使用。
    retrieved_knowledge_summary: str | None

    # 本地知识库文件或远程结果落盘文件的来源路径。
    knowledge_source_path: str
