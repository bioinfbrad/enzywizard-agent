"""通过真实模型接口手工验证完整 Planning 节点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import CleanedFiles
from enzywizard_agent.nodes.planning import planning_node
from enzywizard_agent.utils.IO_utils import create_session_id, normalize_path


def build_parser() -> argparse.ArgumentParser:
    """创建完整 Planning 手工测试所需的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Run EnzyWizard-Agent through complete Planning."
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
    """构造首轮状态并输出完整 PlanningResult JSON。"""

    args = build_parser().parse_args()
    workdir = Path(args.workdir).expanduser().resolve()

    # 手工测试不经过 agent_service，因此在这里构造与正式入口相同的核心状态。
    state = {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            workdir=workdir,
            max_turns=3,
        ),
        "session_id": create_session_id(),
        "current_turn_id": 1,
        "current_plan_id": 0,
        "current_user_message": args.message.strip(),
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
        "user_messages": [],
        "user_message_summary": "",
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "cleaned_files": CleanedFiles(
            **{field_name: None for field_name in CleanedFiles.model_fields}
        ),
        "current_plan_artifacts": [],
        "artifacts": [],
        "artifact_summary": "",
    }

    # Planning 节点依次执行意图识别、工具选择、默认计划和参数提取。
    final_state = planning_node(state)
    planning_result = final_state["current_planning_result"]
    output = {
        "route": final_state["route"],
        "planning_result": planning_result.model_dump(mode="json"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
