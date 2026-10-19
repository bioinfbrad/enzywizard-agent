from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class UserTurnInput(BaseModel):
    """user_input_interrupt 恢复时接受的下一轮用户输入。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_id: int
    message: str

    @field_validator("turn_id")
    @classmethod
    def validate_turn_id(cls, value: int) -> int:
        if value < 2:
            raise ValueError("next turn_id must be greater than or equal to 2")
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value:
            raise ValueError("message must not be empty")
        return value
