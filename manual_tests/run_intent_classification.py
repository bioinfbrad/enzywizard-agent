"""通过真实模型接口手工验证意图识别结构化输出。"""

from __future__ import annotations

import argparse
from pathlib import Path

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.services.planning_service import classify_intent
from enzywizard_agent.utils.IO_utils import normalize_path


def build_parser() -> argparse.ArgumentParser:
    """创建仅供意图识别手工测试使用的参数解析器。"""

    # 参数名称与正式 CLI 保持一致，便于直接复用真实运行命令。
    parser = argparse.ArgumentParser(
        description="Run EnzyWizard-Agent intent classification only."
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
    """构造意图识别所需最小状态，调用服务并打印格式化 JSON。"""

    args = build_parser().parse_args()

    # 手工测试不经过 agent_service，因此在这里直接建立模型配置和当前输入。
    state = {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            workdir=Path(args.workdir).expanduser().resolve(),
            max_turns=3,
        ),
        # 意图 builder 当前只需要轮次、消息、输入和可选历史摘要。
        "current_turn_id": 1,
        "current_plan_id": 0,
        "current_user_message": args.message,
        "agent_inputs": AgentInputs(
            structure_file=normalize_path(args.structure_file),
            msa_file=normalize_path(args.msa_file),
            mut_structure_file=normalize_path(args.mut_structure_file),
            mut_msa_file=normalize_path(args.mut_msa_file),
            substrate_name=args.substrate_name,
            amino_acid_substitution=args.amino_acid_substitution,
        ),
        "user_message_summary": "",
    }

    # 直接调用意图服务，确保该脚本不会继续执行工具选择。
    intent = classify_intent(state)

    # model_dump_json 展示经过 Pydantic 规范化后的最终字段。
    print(intent.model_dump_json(indent=2))


# 直接运行脚本时进入测试；被测试框架导入时不会自动调用模型。
if __name__ == "__main__":
    main()
