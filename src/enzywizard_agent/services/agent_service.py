from __future__ import annotations

from pathlib import Path

from ..conf import AgentConfig
from ..data_models.artifact import CleanedFiles
from ..data_models.agent_inputs import AgentInputs
from ..graph import build_graph
from ..state import AgentState
from ..utils.IO_utils import create_session_id, normalize_path
from ..validators.input_validators import validate_single_amino_acid_substitution


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
    substrate_name: str | None = None,
    amino_acid_substitution: str | None = None,
    max_turns: int = 3,
) -> bool:
    try:
        workdir = Path(workdir).expanduser().resolve()
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

    if substrate_name is not None:
        substrate_name = str(substrate_name).strip()
        if not substrate_name:
            print("[ERROR] Invalid substrate_name: empty value is not allowed.")
            return False
        if ";" in substrate_name:
            print("[ERROR] Invalid substrate_name: multiple substrates are not supported yet. Provide one substrate name or SMILES string.")
            return False
        if structure_file is None:
            print("[ERROR] Invalid input parameter combination: substrate_name was provided without structure_file. Substrate analysis needs a reference protein structure.")
            return False

    if amino_acid_substitution is not None:
        amino_acid_substitution = str(amino_acid_substitution).strip().upper()
        if not amino_acid_substitution:
            print("[ERROR] Invalid amino_acid_substitution: empty value is not allowed.")
            return False
        if structure_file is None:
            print("[ERROR] Invalid input parameter combination: amino_acid_substitution was provided without structure_file. Mutation analysis needs a wild-type reference structure.")
            return False
        if mut_structure_file is None:
            print("[ERROR] Invalid input parameter combination: amino_acid_substitution was provided without mut_structure_file. The amino acid substitution must describe the residue replacement between the wild-type and mutant structures.")
            return False
        error_message = validate_single_amino_acid_substitution(amino_acid_substitution)
        if error_message is not None:
            print(f"[ERROR] {error_message}")
            return False

    if mut_structure_file is not None and amino_acid_substitution is None:
        print("[ERROR] Invalid input parameter combination: mut_structure_file was provided without amino_acid_substitution. Mutation comparison needs the residue replacement, such as A123V, to align biological meaning between wild-type and mutant structures.")
        return False

    config = AgentConfig(
        api_key=str(api_key).strip(),
        base_url=str(base_url).strip(),
        model=str(model).strip(),
        workdir=workdir,
        max_turns=max_turns,
    )
    agent_inputs = AgentInputs(
        structure_file=normalize_path(structure_file),
        msa_file=normalize_path(msa_file),
        mut_structure_file=normalize_path(mut_structure_file),
        mut_msa_file=normalize_path(mut_msa_file),
        substrate_name=substrate_name,
        amino_acid_substitution=amino_acid_substitution,
    )
    state: AgentState = {
        "config": config,
        "session_id": create_session_id(),
        "current_turn_id": 1,
        "current_plan_id": 0,
        "agent_inputs": agent_inputs,
        "current_user_message": str(message).strip(),
        "user_messages": [],
        "user_message_summary": "",
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "tool_executions": [],
        "tool_calling_summary": "",
        "analysis_results": [],
        "analysis_summary": "",
        "cleaned_files": CleanedFiles(
            single_clean_output_dir=None,
            single_cleaned_structure_file=None,
            single_cleaned_cif_file=None,
            single_cleaned_pdb_file=None,
            single_cleaned_fasta_file=None,
            single_clean_report_file=None,
            mut_clean_output_dir=None,
            wt_cleaned_structure_file=None,
            wt_cleaned_cif_file=None,
            wt_cleaned_pdb_file=None,
            wt_cleaned_fasta_file=None,
            mut_cleaned_structure_file=None,
            mut_cleaned_cif_file=None,
            mut_cleaned_pdb_file=None,
            mut_cleaned_fasta_file=None,
            mut_clean_report_file=None,
            cleaned_amino_acid_substitution=None,
        ),
        "current_plan_artifacts": [],
        "artifacts": [],
        "artifact_summary": "",
    }

    print("[INFO] Agent parameter validation finished")
    build_graph()
    return True
