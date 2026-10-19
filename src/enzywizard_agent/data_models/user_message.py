from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


# =============================================================================
# Enum types
# =============================================================================


# =============================================================================
# Output schemas
# =============================================================================


# =============================================================================
# State data models
# =============================================================================


class UserMessage(BaseModel):
    """保存一轮用户原始消息及其轮次编号。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_id: int
    content: str

    @model_validator(mode="after")
    def validate_message(self) -> "UserMessage":
        if self.turn_id < 1:
            raise ValueError("turn_id must be greater than or equal to 1")
        if not self.content:
            raise ValueError("content must not be empty")
        return self
