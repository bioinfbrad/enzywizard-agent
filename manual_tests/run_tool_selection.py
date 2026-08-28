"""通过真实模型接口手工验证“意图识别到工具选择”的连续流程。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import CleanedFiles
from enzywizard_agent.data_models.planning import PlanningResult
from enzywizard_agent.services.planning_service import (
    classify_intent,
    select_single_protein_tools,
    select_wildtype_mutant_tools,
)
from enzywizard_agent.utils.IO_utils import normalize_path
from enzywizard_agent.utils.planning_utils import purpose_needs_tool


def build_parser() -> argparse.ArgumentParser:
    """创建连续 Planning 手工测试所需的参数解析器。"""

    parser = argparse.ArgumentParser(
        description="Run EnzyWizard-Agent intent recognition and tool selection."
    )
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--structure_file")
    parser.add_argument("--msa_file")
    parser.add_argument("--mut_structure_file")
    parser.add_argument("--mut_msa_file")
    parser.add_argument("--substrate_name")
    parser.add_argument("--amino_acid_substitution")
    return parser


def main() -> None:
    """构造最小状态并连续调用意图识别与工具选择服务。"""

    args = build_parser().parse_args()

    # 该状态模拟 agent_service 完成参数处理后的首轮状态，但不构建 LangGraph。
    state = {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            workdir=Path(args.workdir).expanduser().resolve(),
            max_turns=3,
        ),
        "current_turn_id": 1,
        "current_plan_id": 1,
        "current_user_message": args.message,
        "agent_inputs": AgentInputs(
            structure_file=normalize_path(args.structure_file),
            msa_file=normalize_path(args.msa_file),
            mut_structure_file=normalize_path(args.mut_structure_file),
            mut_msa_file=normalize_path(args.mut_msa_file),
            substrate_name=(
                args.substrate_name.strip() if args.substrate_name else None
            ),
            amino_acid_substitution=(
                args.amino_acid_substitution.strip().upper()
                if args.amino_acid_substitution
                else None
            ),
        ),
        "user_message_summary": "",
        "planning_results": [],
        # 首轮没有可信清理结果，因此程序应根据原始输入决定 clean/mut-clean。
        "cleaned_files": CleanedFiles(
            **{field_name: None for field_name in CleanedFiles.model_fields}
        ),

        # 历史 artifact 当前不参与工具选择，仍保留空状态以贴近正式 AgentState。
        "artifacts": [],
        "artifact_summary": "",
    }

    # 先运行真实意图识别，再构造工具选择 Builder 所需的中间 PlanningResult。
    intent = classify_intent(state)
    planning_result = PlanningResult(
        turn_id=state["current_turn_id"],
        plan_id=state["current_plan_id"],
        planning_route="tool_unused",
        planning_flag="intent_recognized",
        needs_tool=purpose_needs_tool(intent.purpose_type),
        summary=None,
        purpose_type=intent.purpose_type,
        purpose_reason=intent.purpose_reason,
        tool_selection=None,
        parameter_extraction=None,
        tool_plan=None,
        error_messages=[],
    )

    # 两种工具工作流分别调用正式服务；other 不进入工具选择。
    tool_selection = None
    if intent.purpose_type == "single_protein_analysis":
        tool_selection = select_single_protein_tools(state, planning_result)
    elif intent.purpose_type == "wildtype_mutant_comparison":
        tool_selection = select_wildtype_mutant_tools(state, planning_result)

    # 输出最终规范化结果，便于检查动态候选名、程序补充工具和失败字段。
    output = {
        "intent": intent.model_dump(),
        "needs_tool": planning_result.needs_tool,
        "tool_selection": (
            tool_selection.model_dump() if tool_selection is not None else None
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
