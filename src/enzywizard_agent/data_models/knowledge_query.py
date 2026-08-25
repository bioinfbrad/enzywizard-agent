from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# =============================================================================
# Enum types
# =============================================================================

KnowledgeQueryType = Literal[
    "tool_spec",
    "analysis_template",
    "analysis_example",
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
    turn_id: int
    query_type: KnowledgeQueryType
    query: str
    knowledge_source: str
    retrieved_knowledge_items: list[str]
    retrieved_knowledge_summary: str | None
    knowledge_source_path: str
