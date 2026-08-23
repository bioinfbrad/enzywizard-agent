from __future__ import annotations

from pathlib import Path

from ..conf import AgentConfig
from ..data_models import CleanedFiles, InputFiles
from ..state import AgentState
from ..utils.IO_utils import normalize_path


def run_agent_service(
    api_key: str,
    base_url: str,
    model: str,
    workdir: str | Path,
    message: str,
    structure_file: str | Path | None = None,
    msa_file: str | Path | None = None,
    mut_structure_file: str | Path | None = None,
    mut_msa_file: str | Path | None = None,
    max_turns: int = 3,
) -> bool:
    try:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Failed to initialize agent workdir: {workdir}. Reason: {e}")
        return False

    print(f"[INFO] Agent parameter validation started: workdir={workdir}")

    if not str(api_key).strip():
        print("[ERROR] Invalid api_key: empty value is not allowed.")
        return False

    if not str(base_url).strip():
        print("[ERROR] Invalid base_url: empty value is not allowed.")
        return False

    if not str(model).strip():
        print("[ERROR] Invalid model: empty value is not allowed.")
        return False

    if not str(message).strip():
        print("[ERROR] Invalid message: empty value is not allowed.")
        return False

    if max_turns < 1:
        print(f"[ERROR] Invalid max_turns value: {max_turns}. Must be greater than or equal to 1.")
        return False

    if structure_file is None:
        if msa_file is not None:
            print("[ERROR] Invalid input parameter combination: msa_file was provided without structure_file. MSA analysis needs a reference protein structure to fetch the sequence for residue mapping.")
            return False
        if mut_structure_file is not None:
            print("[ERROR] Invalid input parameter combination: mut_structure_file was provided without structure_file. Mutant analysis needs a wild-type reference structure.")
            return False
        if mut_msa_file is not None:
            print("[ERROR] Invalid input parameter combination: mut_msa_file was provided without structure_file. Mutant MSA analysis needs a wild-type reference structure.")
            return False

    if mut_msa_file is not None and mut_structure_file is None:
        print("[ERROR] Invalid input parameter combination: mut_msa_file was provided without mut_structure_file. Mutant MSA analysis needs a reference mutant protein structure to fetch the sequence for residue mapping.")
        return False

    config = AgentConfig(
        api_key=str(api_key).strip(),
        base_url=str(base_url).strip(),
        model=str(model).strip(),
        workdir=workdir,
        max_turns=max_turns,
    )
    input_files = InputFiles(
        structure_file=normalize_path(structure_file),
        msa_file=normalize_path(msa_file),
        mut_structure_file=normalize_path(mut_structure_file),
        mut_msa_file=normalize_path(mut_msa_file),
    )
    state: AgentState = {
        "config": config,
        "turn_count": 0,
        "input_files": input_files,
        "current_user_message": str(message).strip(),
        "user_messages": [],
        "user_message_summary": "",
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "tool_calls": [],
        "tool_calling_summary": "",
        "analysis_results": [],
        "analysis_summary": "",
        "cleaned_files": CleanedFiles(),
        "artifacts": [],
        "artifact_summary": "",
    }

    print("[INFO] Agent parameter validation finished")
    print("[INFO] Agent workflow execution is not implemented yet")
    return True
