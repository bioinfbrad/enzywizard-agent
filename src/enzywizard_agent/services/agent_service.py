from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ..conf import AgentConfig
from ..data_models.artifact import CleanedFiles
from ..data_models.runtime import RuntimeSession
from ..data_models.agent_inputs import AgentInputs
from ..graph import build_graph
from ..state import AgentState
from ..utils.checkpoint_utils import create_in_memory_checkpointer
from ..utils.IO_utils import create_session_id, normalize_path
from ..utils.security_utils import redact_text
from ..validators.input_validators import validate_single_amino_acid_substitution
from .llm_service import check_model_capabilities
from .manifest_service import write_session_manifest
from .runtime_service import (
    enforce_session_deadline,
    is_runtime_limit_error,
    record_runtime_error,
    utc_now,
)


_TERMINAL_RUNTIME_STATUSES = {
    "failed",
    "cancelled",
    "timed_out",
    "budget_exceeded",
}


def _print_startup_error(
    message: str,
    *,
    category: str = "input_error",
) -> None:
    """Report pre-graph failures with the same resolution language as Agent turns."""

    print(
        f"[ERROR:{category}] Stage: startup_validation\n"
        "Current session: not started; restart is required.\n"
        f"Reason: {message}\n"
        "Required action: correct the command-line input or configuration, then "
        "restart EnzyWizard-Agent."
    )


def _write_graph_result_manifest(result: AgentState) -> None:
    """根据图状态写 manifest，保留预算、超时和取消等终态。"""

    if result.get("config") is None:
        return
    runtime = result.get("runtime_session")
    if runtime is not None and runtime.status in _TERMINAL_RUNTIME_STATUSES:
        write_session_manifest(result)
        return
    analysis = result.get("current_analysis_result")
    guidance = getattr(analysis, "error_guidance", None)
    if guidance is not None and guidance.restart_required:
        write_session_manifest(
            result,
            status="failed",
            stop_reason=f"{guidance.error_type}:restart_required",
        )
        return
    interrupted = bool(result.get("__interrupt__"))
    write_session_manifest(
        result,
        status="running" if interrupted else "completed",
        stop_reason=None if interrupted else "graph_completed",
    )


def execute_agent_graph(
    graph: CompiledStateGraph,
    state: AgentState,
) -> dict[str, Any]:
    """使用状态中的 session_id 首次执行一个已编译状态图。"""

    session_id = state.get("session_id")
    if session_id is None or not session_id.strip():
        raise ValueError("Agent graph execution requires a non-empty session_id.")

    # thread_id 将本次 invoke 与后续 Command(resume=...) 绑定到同一 checkpoint。
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = graph.invoke(state, config=config)
        _write_graph_result_manifest(result)
        return result
    except KeyboardInterrupt:
        record_runtime_error(
            state,
            stage="agent_graph",
            error="Agent execution was cancelled by the user.",
            category="internal",
        )
        if state.get("config") is not None:
            write_session_manifest(state, status="cancelled", stop_reason="user_interrupt")
        raise
    except Exception as error:
        record_runtime_error(state, stage="agent_graph", error=error)
        runtime = state.get("runtime_session")
        status = (
            runtime.status
            if runtime is not None and runtime.status in {"timed_out", "budget_exceeded"}
            else "failed"
        )
        if state.get("config") is not None:
            write_session_manifest(
                state,
                status=status,
                stop_reason=(
                    runtime.stop_reason
                    if runtime is not None and runtime.stop_reason
                    else "graph_error"
                ),
            )
        raise


def resume_agent_graph(
    graph: CompiledStateGraph,
    session_id: str,
    resume_value: Any,
) -> dict[str, Any]:
    """恢复一个正在等待 interrupt 输入的 CLI 会话。"""

    session_id = str(session_id).strip()
    if not session_id:
        raise ValueError("Agent graph resume requires a non-empty session_id.")

    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)

    # 不允许把 resume 数据发送给不存在、已经完成或当前没有中断的会话。
    active_interrupts = [
        graph_interrupt
        for task in snapshot.tasks
        for graph_interrupt in task.interrupts
    ]
    if not active_interrupts:
        raise ValueError(
            "The requested agent session is not waiting for interrupt input."
        )

    state = snapshot.values
    try:
        enforce_session_deadline(state, "agent_graph_resume")
        result = graph.invoke(Command(resume=resume_value), config=config)
        _write_graph_result_manifest(result)
        return result
    except KeyboardInterrupt:
        record_runtime_error(
            state,
            stage="agent_graph_resume",
            error="Agent execution was cancelled by the user.",
            category="internal",
        )
        if state.get("config") is not None:
            write_session_manifest(state, status="cancelled", stop_reason="user_interrupt")
        raise
    except Exception as error:
        record_runtime_error(state, stage="agent_graph_resume", error=error)
        runtime = state.get("runtime_session")
        status = (
            runtime.status
            if runtime is not None and runtime.status in {"timed_out", "budget_exceeded"}
            else "failed"
        )
        if state.get("config") is not None:
            write_session_manifest(
                state,
                status=status,
                stop_reason=(
                    runtime.stop_reason
                    if runtime is not None and runtime.stop_reason
                    else "graph_error"
                ),
            )
        raise


def cancel_agent_graph(graph: CompiledStateGraph, session_id: str) -> None:
    """在 CLI 等待输入期间被取消时，登记状态并保存已有结果清单。"""

    session_id = str(session_id).strip()
    if not session_id:
        return
    snapshot = graph.get_state({"configurable": {"thread_id": session_id}})
    state = snapshot.values
    if not state or state.get("config") is None:
        return
    record_runtime_error(
        state,
        stage="cli_input",
        error="Agent input was cancelled by the user.",
        category="internal",
    )
    write_session_manifest(state, status="cancelled", stop_reason="user_interrupt")


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
    llm_timeout_seconds: float = 60.0,
    llm_max_retries: int = 2,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
    llm_extra_body: dict[str, Any] | None = None,
    check_model_capabilities: bool = True,
    ontology_mode: Literal["off", "retrieve", "control"] = "retrieve",
    tool_recovery_mode: Literal["off", "safe"] = "safe",
    max_tool_recovery_attempts: int = 1,
    failure_diagnosis_mode: Literal["deterministic", "hybrid"] = "hybrid",
    max_llm_calls: int | None = 64,
    max_input_tokens: int | None = 500_000,
    max_output_tokens: int | None = 128_000,
    session_timeout_seconds: float | None = 21_600.0,
    tool_timeout_seconds: float | None = 3_600.0,
) -> tuple[CompiledStateGraph, dict[str, Any]] | None:
    """校验 CLI 输入、建立初始状态并运行到完成或首个中断。

    成功时返回同一 CLI 进程后续恢复所需的编译图和当前执行结果；参数校验
    失败时输出用户可读错误并返回 None。
    """

    # 工作目录是唯一在参数校验开始前需要落盘的目录，同时转换为绝对路径。
    try:
        workdir = Path(workdir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _print_startup_error(
            f"Failed to initialize agent workdir: {redact_text(e)}",
            category="configuration_error",
        )
        return None

    print(f"[INFO] Agent parameter validation started: workdir={redact_text(workdir)}")

    # 模型连接参数和用户消息不能为空；strip 防止纯空白字符串通过校验。
    if not str(api_key).strip():
        _print_startup_error("Invalid api_key: empty value is not allowed.")
        return None

    if not str(base_url).strip():
        _print_startup_error("Invalid base_url: empty value is not allowed.")
        return None

    if not str(model).strip():
        _print_startup_error("Invalid model: empty value is not allowed.")
        return None

    if not str(message).strip():
        _print_startup_error("Invalid message: empty value is not allowed.")
        return None

    # 至少允许一轮，否则状态图没有可执行的用户会话。
    if max_turns < 1:
        _print_startup_error(
            f"Invalid max_turns value: {max_turns}. Must be greater than or equal to 1."
        )
        return None

    # structure_file 是所有结构相关输入的基础：MSA 需要结构序列进行残基映射，
    # MUT 输入需要 WT 结构作为比较基准。
    if structure_file is None:
        if msa_file is not None:
            _print_startup_error("Invalid input parameter combination: msa_file was provided without structure_file. MSA analysis needs a reference protein structure to fetch the sequence for residue mapping.")
            return None
        if mut_structure_file is not None:
            _print_startup_error("Invalid input parameter combination: mut_structure_file was provided without structure_file. Mutant analysis needs a wild-type reference structure.")
            return None
        if mut_msa_file is not None:
            _print_startup_error("Invalid input parameter combination: mut_msa_file was provided without structure_file. Mutant MSA analysis needs a wild-type reference structure.")
            return None

    # 即使已有 WT 结构，MUT MSA 仍必须有 MUT 结构作为该侧序列参考。
    if mut_msa_file is not None and mut_structure_file is None:
        _print_startup_error("Invalid input parameter combination: mut_msa_file was provided without mut_structure_file. Mutant MSA analysis needs a reference mutant protein structure to fetch the sequence for residue mapping.")
        return None

    # 当前只支持一个底物。分号被视为用户尝试提交多个底物的明确分隔符。
    if substrate_name is not None:
        substrate_name = str(substrate_name).strip()
        if not substrate_name:
            _print_startup_error("Invalid substrate_name: empty value is not allowed.")
            return None
        if ";" in substrate_name:
            _print_startup_error("Invalid substrate_name: multiple substrates are not supported yet. Provide one substrate name or SMILES string.")
            return None
        # 底物分析最终需要与蛋白结构关联，因此不能脱离主结构单独输入。
        if structure_file is None:
            _print_startup_error("Invalid input parameter combination: substrate_name was provided without structure_file. Substrate analysis needs a reference protein structure.")
            return None

    # 突变替换统一转为大写，再检查结构依赖和 A123V 格式。
    if amino_acid_substitution is not None:
        amino_acid_substitution = str(amino_acid_substitution).strip().upper()
        if not amino_acid_substitution:
            _print_startup_error("Invalid amino_acid_substitution: empty value is not allowed.")
            return None
        if structure_file is None:
            _print_startup_error("Invalid input parameter combination: amino_acid_substitution was provided without structure_file. Mutation analysis needs a wild-type reference structure.")
            return None
        if mut_structure_file is None:
            _print_startup_error("Invalid input parameter combination: amino_acid_substitution was provided without mut_structure_file. The amino acid substitution must describe the residue replacement between the wild-type and mutant structures.")
            return None
        # 具体格式和标准氨基酸检查放在 validators 中，服务层只处理返回错误。
        error_message = validate_single_amino_acid_substitution(amino_acid_substitution)
        if error_message is not None:
            _print_startup_error(error_message)
            return None

    # MUT 结构和替换信息互相依赖，防止出现无法解释生物学意义的突变比较。
    if mut_structure_file is not None and amino_acid_substitution is None:
        _print_startup_error("Invalid input parameter combination: mut_structure_file was provided without amino_acid_substitution. Mutation comparison needs the residue replacement, such as A123V, to align biological meaning between wild-type and mutant structures.")
        return None

    # 所有校验完成后才创建不可变配置，避免无效参数进入状态。
    try:
        config = AgentConfig(
            api_key=str(api_key).strip(),
            base_url=str(base_url).strip(),
            model=str(model).strip(),
            workdir=workdir,
            max_turns=max_turns,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_max_retries=llm_max_retries,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            llm_extra_body=llm_extra_body,
            check_model_capabilities=check_model_capabilities,
            ontology_mode=ontology_mode,
            tool_recovery_mode=tool_recovery_mode,
            max_tool_recovery_attempts=max_tool_recovery_attempts,
            failure_diagnosis_mode=failure_diagnosis_mode,
            max_llm_calls=max_llm_calls,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            session_timeout_seconds=session_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
        )
    except ValueError as error:
        _print_startup_error(
            f"Invalid Agent configuration: {error}",
            category="configuration_error",
        )
        return None
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
    session_id = create_session_id()
    state: AgentState = {
        "config": config,

        # session_id 标识一次完整对话；首轮从 turn_id=1 开始，尚未生成计划。
        "session_id": session_id,
        "runtime_session": RuntimeSession(
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
        ),
        "current_turn_id": 1,
        "current_plan_id": 0,
        "agent_inputs": agent_inputs,
        # 当前消息单独保存，历史列表会在后续用户输入节点中维护。
        "current_user_message": str(message).strip(),
        "user_messages": [],
        "user_message_summary": "",
        # 每轮先经过 Request Router；历史查询只在 Router 标记需要时执行。
        "request_routing_results": [],
        "request_routing_summary": "",
        "history_query_results": [],
        "history_query_summary": "",
        "current_history_artifacts": [],
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

    # 模型检查在状态图启动前完成；失败时不会生成计划或执行任何科学工具。
    write_session_manifest(state)
    try:
        capability = check_model_capabilities(state)
        write_session_manifest(state)
        print(
            "[INFO] Model capability check: "
            f"status={capability.status}, strict_structured_output="
            f"{capability.strict_structured_output}, token_usage="
            f"{capability.token_usage_available}"
        )
    except KeyboardInterrupt:
        record_runtime_error(
            state,
            stage="model_capability_check",
            error="Model capability check was cancelled by the user.",
            category="internal",
        )
        write_session_manifest(
            state,
            status="cancelled",
            stop_reason="user_interrupt",
        )
        raise
    except Exception as error:
        if is_runtime_limit_error(error):
            write_session_manifest(state)
        else:
            write_session_manifest(
                state,
                status="failed",
                stop_reason="model_capability_check_failed",
            )
        raise

    # 一个 CLI 进程只创建一个内存 checkpointer 和编译图。图对象会连同首次
    # 执行结果返回给命令层，并在后续所有中断恢复中持续复用。
    checkpointer = create_in_memory_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    result = execute_agent_graph(graph, state)
    return graph, result
