from __future__ import annotations

import argparse

from .commands.agent import add_agent_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzywizard-agent",
        description="EnzyWizard-Agent: Plan and orchestrate EnzyWizard command-line workflows."
    )
    add_agent_parser(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
