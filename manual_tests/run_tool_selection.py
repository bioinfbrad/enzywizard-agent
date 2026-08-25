from __future__ import annotations

import argparse
from pathlib import Path

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import CleanedFiles
from enzywizard_agent.nodes.planning import planning_node
from enzywizard_agent.utils.IO_utils import normalize_path


def build_parser() -> argparse.ArgumentParser:
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
    args = build_parser().parse_args()

    state = {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            workdir=Path(args.workdir).expanduser().resolve(),
            max_turns=3,
        ),
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
        "planning_results": [],
        "cleaned_files": CleanedFiles(
            **{field_name: None for field_name in CleanedFiles.model_fields}
        ),
        "artifacts": [],
        "artifact_summary": "",
    }

    planning_node(state)


if __name__ == "__main__":
    main()
