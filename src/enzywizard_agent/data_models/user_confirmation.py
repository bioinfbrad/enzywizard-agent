from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


# =============================================================================
# Enum types
# =============================================================================

ConfirmationDecision = Literal[
    # 用户批准当前计划，下一步可以执行工具。
    "approved",
    # 用户要求修改当前计划，下一步返回 Planning 生成新 plan_id。
    "revision_requested",
    # 用户拒绝当前计划，跳过工具执行并等待下一轮输入。
    "rejected",
]


# =============================================================================
# Output schemas
# =============================================================================


# =============================================================================
# State data models
# =============================================================================


class UserConfirmation(BaseModel):
    """保存用户针对一次具体工具计划提交的确认结果。"""

    # 中断恢复数据来自外部 API，因此禁止未知字段并清理字符串首尾空白。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # turn_id 和 plan_id 将用户决定绑定到当前计划，防止旧页面批准新计划。
    turn_id: int
    plan_id: int
    decision: ConfirmationDecision

    # 仅 revision_requested 允许携带修改意见；其他决定必须显式传入 None。
    revision_request: str | None

    @model_validator(mode="after")
    def validate_confirmation(self) -> "UserConfirmation":
        """校验计划标识以及决定和修改意见之间的业务关系。"""

        if self.turn_id < 1:
            raise ValueError("turn_id must be greater than or equal to 1")
        if self.plan_id < 1:
            raise ValueError("plan_id must be greater than or equal to 1")

        if self.decision == "revision_requested":
            if not self.revision_request:
                raise ValueError(
                    "revision_requested requires a non-empty revision_request"
                )
        elif self.revision_request is not None:
            raise ValueError(
                "approved and rejected confirmations require revision_request=None"
            )

        return self
