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
    """校验 CLI 输入、建立初始 AgentState 并编译状态图。

    返回 True 表示入口参数和初始化成功；返回 False 表示已经输出用户可读错误。
    当前函数只 build_graph，尚未正式调用 graph.invoke。
    """

    # 工作目录是唯一在参数校验开始前需要落盘的目录，同时转换为绝对路径。
    try:
        workdir = Path(workdir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Failed to initialize agent workdir: {workdir}. Reason: {e}")
        return False

    print(f"[INFO] Agent parameter validation started: workdir={workdir}")

    # 模型连接参数和用户消息不能为空；strip 防止纯空白字符串通过校验。
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

    # 至少允许一轮，否则状态图没有可执行的用户会话。
    if max_turns < 1:
        print(f"[ERROR] Invalid max_turns value: {max_turns}. Must be greater than or equal to 1.")
        return False

    # structure_file 是所有结构相关输入的基础：MSA 需要结构序列进行残基映射，
    # MUT 输入需要 WT 结构作为比较基准。
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

    # 即使已有 WT 结构，MUT MSA 仍必须有 MUT 结构作为该侧序列参考。
    if mut_msa_file is not None and mut_structure_file is None:
        print("[ERROR] Invalid input parameter combination: mut_msa_file was provided without mut_structure_file. Mutant MSA analysis needs a reference mutant protein structure to fetch the sequence for residue mapping.")
        return False

    # 当前只支持一个底物。分号被视为用户尝试提交多个底物的明确分隔符。
    if substrate_name is not None:
        substrate_name = str(substrate_name).strip()
        if not substrate_name:
            print("[ERROR] Invalid substrate_name: empty value is not allowed.")
            return False
        if ";" in substrate_name:
            print("[ERROR] Invalid substrate_name: multiple substrates are not supported yet. Provide one substrate name or SMILES string.")
            return False
        # 底物分析最终需要与蛋白结构关联，因此不能脱离主结构单独输入。
        if structure_file is None:
            print("[ERROR] Invalid input parameter combination: substrate_name was provided without structure_file. Substrate analysis needs a reference protein structure.")
            return False

    # 突变替换统一转为大写，再检查结构依赖和 A123V 格式。
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
        # 具体格式和标准氨基酸检查放在 validators 中，服务层只处理返回错误。
        error_message = validate_single_amino_acid_substitution(amino_acid_substitution)
        if error_message is not None:
            print(f"[ERROR] {error_message}")
            return False

    # MUT 结构和替换信息互相依赖，防止出现无法解释生物学意义的突变比较。
    if mut_structure_file is not None and amino_acid_substitution is None:
        print("[ERROR] Invalid input parameter combination: mut_structure_file was provided without amino_acid_substitution. Mutation comparison needs the residue replacement, such as A123V, to align biological meaning between wild-type and mutant structures.")
        return False

    # 所有校验完成后才创建不可变配置，避免无效参数进入状态。
    config = AgentConfig(
        api_key=str(api_key).strip(),
        base_url=str(base_url).strip(),
        model=str(model).strip(),
        workdir=workdir,
        max_turns=max_turns,
    )
    # 所有文件路径在进入 AgentState 前统一转为绝对路径。
    agent_inputs = AgentInputs(
        structure_file=normalize_path(structure_file),
        msa_file=normalize_path(msa_file),
        mut_structure_file=normalize_path(mut_structure_file),
        mut_msa_file=normalize_path(mut_msa_file),
        substrate_name=substrate_name,
        amino_acid_substitution=amino_acid_substitution,
    )
    # 初始状态只保存标识、当前输入和各阶段的空历史容器。
    state: AgentState = {
        "config": config,

        # session_id 标识一次完整对话；首轮从 turn_id=1 开始，尚未生成计划。
        "session_id": create_session_id(),
        "current_turn_id": 1,
        "current_plan_id": 0,
        "agent_inputs": agent_inputs,
        # 当前消息单独保存，历史列表会在后续用户输入节点中维护。
        "current_user_message": str(message).strip(),
        "user_messages": [],
        "user_message_summary": "",
        # Planning、确认、工具执行和 Analysis 的历史均从空列表开始。
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "tool_executions": [],
        "tool_calling_summary": "",
        "analysis_results": [],
        "analysis_summary": "",
        # 初始没有可信清理结果；clean/mut-clean 执行后再逐字段更新。
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
        # 当前计划产物与全会话产物分开追踪，但全历史保持扁平列表。
        "current_plan_artifacts": [],
        "artifacts": [],
        "artifact_summary": "",
    }

    print("[INFO] Agent parameter validation finished")

    # Planning 已可生成完整 ToolPlan；正式 graph.invoke 仍需等待用户确认 interrupt
    # 和 Tool Calling 节点实现后接入，避免当前空节点被误当成可执行流程。
    build_graph()
    return True
