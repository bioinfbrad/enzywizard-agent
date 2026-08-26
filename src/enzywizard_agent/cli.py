from __future__ import annotations

import argparse

from .commands.agent import add_agent_parser


def build_parser() -> argparse.ArgumentParser:
    """创建顶层参数解析器，并注册 Agent 命令所需的全部参数。"""

    # 顶层解析器负责程序名称和总体说明，具体参数由 commands 包集中维护。
    parser = argparse.ArgumentParser(
        prog="enzywizard-agent",
        description="EnzyWizard-Agent: Plan and orchestrate EnzyWizard command-line workflows."
    )
    add_agent_parser(parser)
    return parser


def main() -> None:
    """解析命令行参数，并调用子命令注册的处理函数。"""

    parser = build_parser()
    args = parser.parse_args()

    # add_agent_parser 会把实际入口函数保存到 args.func。
    args.func(args)
