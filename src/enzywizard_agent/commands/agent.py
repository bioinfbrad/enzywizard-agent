from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
import sys

from ..data_models.runtime import AgentRuntimeError
from ..services.agent_service import (
    cancel_agent_graph,
    resume_agent_graph,
    run_agent_service,
)
from ..utils.CLI_utils import (
    display_final_analysis_result,
    display_tool_plan_confirmation,
    display_user_input_request,
    get_single_interrupt_payload,
    read_next_user_input,
    read_tool_plan_confirmation,
)
from ..utils.security_utils import redact_text


def _runtime_error_exit_code(error: BaseException) -> int:
    """把统一运行时错误类别映射为稳定 CLI 退出码。"""

    category = getattr(error, "category", "internal")
    if category == "user_input":
        return 2
    if category == "budget":
        return 3
    if category == "timeout":
        return 124
    return 1


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

    # 标准模型参数默认为 None，因此不会向不支持它们的服务端发送。
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature. Omitted by default.")
    parser.add_argument("--max_completion_tokens", type=int, default=None, help="Optional maximum completion tokens per model call.")
    parser.add_argument("--reasoning_effort", default=None, help="Optional OpenAI-compatible reasoning effort.")
    parser.add_argument("--llm_extra_body_json", default=None, help="Optional provider-specific JSON object passed as extra_body.")
    parser.add_argument("--llm_timeout_seconds", type=float, default=60.0, help="Timeout for one model request (default: 60).")
    parser.add_argument("--llm_max_retries", type=int, default=2, help="Maximum transport retries per model request (default: 2).")
    parser.add_argument("--skip_model_check", action="store_true", help="Skip the startup OpenAI-compatible structured-output capability check.")
    parser.add_argument(
        "--ontology_mode",
        choices=("off", "retrieve", "control"),
        default="retrieve",
        help="Local FAIR terminology enrichment mode (default: retrieve).",
    )
    parser.add_argument(
        "--tool_recovery_mode",
        choices=("off", "safe"),
        default="safe",
        help="Bounded deterministic tool recovery mode (default: safe).",
    )
    parser.add_argument(
        "--max_tool_recovery_attempts",
        type=int,
        default=1,
        help="Maximum safe retries for one failed tool call (default: 1).",
    )
    parser.add_argument(
        "--failure_diagnosis_mode",
        choices=("deterministic", "hybrid"),
        default="hybrid",
        help=(
            "Failure diagnosis mode: deterministic rules only, or hybrid rules "
            "plus policy-reviewed LLM diagnosis (default: hybrid)."
        ),
    )

    # 这些上限约束整个会话；传入 0 表示禁用对应预算上限。
    parser.add_argument("--max_llm_calls", type=int, default=64, help="Maximum LLM calls per session; 0 disables the limit (default: 64).")
    parser.add_argument("--max_input_tokens", type=int, default=500000, help="Maximum cumulative input tokens; 0 disables the limit.")
    parser.add_argument("--max_output_tokens", type=int, default=128000, help="Maximum cumulative output tokens; 0 disables the limit.")
    parser.add_argument("--session_timeout_seconds", type=float, default=21600.0, help="Whole-session runtime limit; 0 disables the limit (default: 21600).")
    parser.add_argument("--tool_timeout_seconds", type=float, default=3600.0, help="Timeout for each tool subprocess; 0 disables the limit (default: 3600).")

    # argparse 完成解析后，cli.main 会调用这里绑定的 run_agent。
    parser.set_defaults(func=run_agent)


def run_agent(args: Namespace) -> None:
    """启动状态图，并在同一 CLI 进程内处理全部外部中断。"""

    # 命令层只转发入口参数；输入校验、状态构造和首次 invoke 均由服务层完成。
    try:
        extra_body = None
        if args.llm_extra_body_json is not None:
            extra_body = json.loads(args.llm_extra_body_json)
            if not isinstance(extra_body, dict):
                raise ValueError("--llm_extra_body_json must contain a JSON object.")
        execution = run_agent_service(
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
            llm_timeout_seconds=args.llm_timeout_seconds,
            llm_max_retries=args.llm_max_retries,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
            llm_extra_body=extra_body,
            check_model_capabilities=not args.skip_model_check,
            ontology_mode=args.ontology_mode,
            tool_recovery_mode=args.tool_recovery_mode,
            max_tool_recovery_attempts=args.max_tool_recovery_attempts,
            failure_diagnosis_mode=args.failure_diagnosis_mode,
            max_llm_calls=args.max_llm_calls or None,
            max_input_tokens=args.max_input_tokens or None,
            max_output_tokens=args.max_output_tokens or None,
            session_timeout_seconds=args.session_timeout_seconds or None,
            tool_timeout_seconds=args.tool_timeout_seconds or None,
        )
    except KeyboardInterrupt:
        print("\n[ERROR] Agent execution was cancelled by the user.")
        sys.exit(130)
    except json.JSONDecodeError as error:
        print(
            f"[ERROR:configuration_error] Invalid --llm_extra_body_json: {error}\n"
            "Required action: correct the JSON value and restart EnzyWizard-Agent."
        )
        sys.exit(2)
    except ValueError as error:
        print(
            "[ERROR:configuration_error] Agent configuration failed: "
            f"{redact_text(error, api_key=args.api_key)}\n"
            "Required action: correct the command-line configuration and restart "
            "EnzyWizard-Agent."
        )
        sys.exit(2)
    except AgentRuntimeError as error:
        print(
            f"[ERROR:{error.category}] Agent execution failed: "
            f"{redact_text(error, api_key=args.api_key)}\n"
            "Current session: cannot continue. Required action: address the "
            "reported cause and restart EnzyWizard-Agent."
        )
        sys.exit(_runtime_error_exit_code(error))
    except Exception as e:
        print(
            "[ERROR:internal] Agent graph execution failed: "
            f"{redact_text(e, api_key=args.api_key)}\n"
            "Current session: cannot continue. Inspect the session manifest and "
            "restart EnzyWizard-Agent."
        )
        sys.exit(1)

    # None 表示服务层已经输出具体的入口参数错误。
    if execution is None:
        sys.exit(2)

    graph, result = execution
    session_id = str(result.get("session_id", "")).strip()
    if not session_id:
        print("[ERROR] Agent graph result is missing session_id.")
        sys.exit(1)

    # while 只负责处理图主动返回的外部中断，不参与节点路由或业务循环。
    while result.get("__interrupt__"):
        try:
            interrupt_payload = get_single_interrupt_payload(result)
            interrupt_type = interrupt_payload.get("interrupt_type")
            if interrupt_type == "tool_plan_confirmation":
                display_tool_plan_confirmation(interrupt_payload)
                resume_value = read_tool_plan_confirmation(interrupt_payload)
            elif interrupt_type == "user_input":
                display_user_input_request(interrupt_payload)
                resume_value = read_next_user_input(interrupt_payload)
            else:
                raise ValueError(
                    f"Unsupported interrupt_type: {interrupt_type!r}."
                )

            result = resume_agent_graph(graph, session_id, resume_value)
        except (EOFError, KeyboardInterrupt):
            cancel_agent_graph(graph, session_id)
            print("\n[ERROR] Agent input was cancelled by the user.")
            sys.exit(130)
        except AgentRuntimeError as error:
            print(
                f"[ERROR:{error.category}] Agent graph resume failed: "
                f"{redact_text(error, api_key=args.api_key)}\n"
                "Current session: cannot continue. Address the reported cause and "
                "restart EnzyWizard-Agent."
            )
            sys.exit(_runtime_error_exit_code(error))
        except Exception as e:
            print(
                "[ERROR:internal] Agent graph resume failed: "
                f"{redact_text(e, api_key=args.api_key)}\n"
                "Current session: cannot continue. Inspect the session manifest and "
                "restart EnzyWizard-Agent."
            )
            sys.exit(1)

    display_final_analysis_result(result)
    analysis = result.get("current_analysis_result")
    guidance = getattr(analysis, "error_guidance", None)
    if guidance is not None and guidance.restart_required:
        print(
            f"[ERROR:{guidance.error_type}] The current session ended because "
            "the required correction cannot be applied through a later chat message."
        )
        sys.exit(2 if guidance.error_type == "input_error" else 1)
    runtime = result.get("runtime_session")
    if runtime is not None and runtime.status == "budget_exceeded":
        print(f"[WARNING] Agent session stopped at the configured budget: {runtime.stop_reason}")
        sys.exit(3)
    if runtime is not None and runtime.status == "timed_out":
        print(f"[ERROR:timeout] Agent session timed out: {runtime.stop_reason}")
        sys.exit(124)
    if runtime is not None and runtime.status in {"failed", "cancelled"}:
        print(f"[ERROR:internal] Agent session ended with status={runtime.status}.")
        sys.exit(1)
    print("[INFO] Agent graph execution completed")
