from __future__ import annotations

from argparse import ArgumentParser, Namespace
import sys

from ..services.agent_service import run_agent_service


def add_agent_parser(parser: ArgumentParser) -> None:
    """向顶层解析器注册 Agent 运行所需的命令行参数。"""

    # 模型连接参数和当前轮用户请求均为必填项。
    parser.add_argument("--api_key", required=True, help="OpenAI-compatible API key.")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--workdir", required=True, help="Path to a working directory for EnzyWizard-Agent runs and artifacts.")
    parser.add_argument("--message", required=True, help="User message for EnzyWizard-Agent.")

    # 文件和生物学输入均为可选；组合依赖由 agent_service 统一校验。
    parser.add_argument("--structure_file", default=None, help="Optional protein structure file (.cif/.pdb).")
    parser.add_argument("--msa_file", default=None, help="Optional MSA file (.sto/.a3m/.fasta). Requires --structure_file.")
    parser.add_argument("--mut_structure_file", default=None, help="Optional mutant protein structure file (.cif/.pdb). Requires --structure_file.")
    parser.add_argument("--mut_msa_file", default=None, help="Optional mutant MSA file. Requires --structure_file.")
    parser.add_argument("--substrate_name", default=None, help="Optional single substrate name or SMILES string. Requires --structure_file.")
    parser.add_argument("--amino_acid_substitution", default=None, help="Optional single amino acid substitution such as A123V. Requires --structure_file and --mut_structure_file.")
    parser.add_argument("--max_turns", type=int, default=3, help="Maximum agent turns (default: 3).")

    # argparse 完成解析后，cli.main 会调用这里绑定的 run_agent。
    parser.set_defaults(func=run_agent)


def run_agent(args: Namespace) -> None:
    """把 argparse Namespace 转交给服务层，并按服务结果设置退出状态。"""

    # 命令层只负责参数转发，不在这里实现业务校验或图执行逻辑。
    success = run_agent_service(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        workdir=args.workdir,
        message=args.message,
        structure_file=args.structure_file,
        msa_file=args.msa_file,
        mut_structure_file=args.mut_structure_file,
        mut_msa_file=args.mut_msa_file,
        substrate_name=args.substrate_name,
        amino_acid_substitution=args.amino_acid_substitution,
        max_turns=args.max_turns,
    )

    # 服务返回 False 表示初始化或参数校验失败，CLI 使用非零状态退出。
    if not success:
        sys.exit(1)
