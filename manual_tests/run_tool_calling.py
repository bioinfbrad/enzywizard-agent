"""不调用 LLM，使用轻量假命令验证完整 Tool Calling 执行流程。"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import CleanedFiles
from enzywizard_agent.data_models.planning import (
    PlanningResult,
    ReportStagingRule,
    ToolCall,
    ToolOutputArtifact,
    ToolOutputReference,
    ToolPlan,
)
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.nodes.tool_calling import tool_calling_node
import enzywizard_agent.services.request_routing_service as routing_service
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.checkpoint_utils import create_in_memory_checkpointer
from enzywizard_agent.utils.tool_plan_utils import (
    audit_tool_plan,
    build_single_protein_tool_plan,
    build_wildtype_mutant_tool_plan,
)


FAKE_TOOL_SCRIPT = r'''#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path


tool_name = Path(sys.argv[0]).name
should_fail = os.environ.get("FAKE_FAIL_TOOL") == tool_name
should_skip_output = os.environ.get("FAKE_SKIP_OUTPUT_TOOL") == tool_name
printed_secret = os.environ.get("FAKE_PRINT_SECRET", "")
if os.environ.get("FAKE_SLEEP_TOOL") == tool_name:
    time.sleep(float(os.environ.get("FAKE_SLEEP_SECONDS", "1.0")))

parser = argparse.ArgumentParser()
if tool_name == "enzywizard-clean":
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_add_H", action="store_false", dest="add_H")
    parser.add_argument("--pH", type=float, default=7.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("cif", "pdb"):
        (output_dir / f"cleaned_input.{suffix}").write_text("STRUCTURE")
    (output_dir / "cleaned_input.fasta").write_text(">input\nAAAA\n")
    (output_dir / "clean_report_input.json").write_text(
        json.dumps({"cleaned": True, "test_min_volume": 50.0})
    )
elif tool_name == "enzywizard-pocket":
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_rad", type=float)
    parser.add_argument("--max_rad", type=float)
    parser.add_argument("--min_volume", type=float)
    args = parser.parse_args()
    if not Path(args.input_path).is_file():
        raise SystemExit(8)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not should_skip_output:
        (output_dir / "pocket_report_input.json").write_text(
            json.dumps({"pockets": [1]})
        )
elif tool_name == "enzywizard-integrate":
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    input_reports = list(Path(args.input_dir).glob("*.json"))
    if len(input_reports) != 2:
        print(f"Expected two staged reports, found {len(input_reports)}", file=sys.stderr)
        raise SystemExit(9)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "integrate_report_input.json").write_text("{}")
    (output_dir / "integrate_nodes_input.json").write_text("[]")
    (output_dir / "integrate_edges_input.json").write_text("[]")
elif tool_name == "enzywizard-substrate":
    parser.add_argument("--substrate_names", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.substrate_names != "C(CO)O":
        raise SystemExit(11)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "substrate_report_smiles1.json").write_text(
        json.dumps({"substrates": [{"substrate_name": "smiles1"}]})
    )
    for conformer_index in range(1, 4):
        (output_dir / f"smiles1_{conformer_index}.sdf").write_text("SDF")
elif tool_name == "enzywizard-dock":
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--substrate_names", required=True)
    parser.add_argument("--substrate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    substrate_files = sorted(Path(args.substrate_dir).glob("smiles1_*.sdf"))
    if args.substrate_names != "smiles1" or len(substrate_files) != 3:
        raise SystemExit(12)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dock_report_input_smiles1.json").write_text("{}")
    (output_dir / "docked_smiles1.sdf").write_text("DOCKED_SDF")
elif tool_name == "enzywizard-interaction":
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--substrate_names", required=False)
    parser.add_argument("--substrate_dir", required=False)
    args = parser.parse_args()
    docked_sdf = Path(args.substrate_dir) / f"{args.substrate_names}.sdf"
    if args.substrate_names != "docked_smiles1" or not docked_sdf.is_file():
        raise SystemExit(13)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "interaction_report_input_docked_smiles1.json").write_text("{}")
else:
    raise SystemExit(10)

if should_fail:
    (Path(args.output_dir) / "log.txt").write_text(
        "y" * 2400 + " LOG_TAIL_MARKER\n"
    )
    print("x" * 2400 + " FAILURE_TAIL_MARKER", file=sys.stderr)
    raise SystemExit(7)

if os.environ.get("FAKE_TRANSIENT_TOOL") == tool_name:
    transient_marker = Path(args.output_dir).parent / ".transient_failure_seen"
    if not transient_marker.exists():
        (Path(args.output_dir) / "failed_attempt_only.tmp").write_text("stale")
        transient_marker.write_text("failed once")
        print("Service temporarily unavailable", file=sys.stderr)
        raise SystemExit(75)

if os.environ.get("FAKE_SCIENTIFIC_ERROR_TOOL") == tool_name:
    print("Invalid parameter: scientific cutoff requires review", file=sys.stderr)
    raise SystemExit(64)

(Path(args.output_dir) / "log.txt").write_text(
    f"{tool_name} completed {printed_secret}\n"
)
print(f"{tool_name} stdout {printed_secret}")
'''


def create_fake_tool_commands(fake_bin_dir: Path) -> None:
    """创建同名假 CLI，使测试覆盖真实 subprocess 参数列表调用。"""

    script_path = fake_bin_dir / "fake_enzywizard_tool.py"
    script_path.write_text(FAKE_TOOL_SCRIPT, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    for tool_name in (
        "enzywizard-clean",
        "enzywizard-pocket",
        "enzywizard-integrate",
        "enzywizard-substrate",
        "enzywizard-dock",
        "enzywizard-interaction",
    ):
        (fake_bin_dir / tool_name).symlink_to(script_path)


def output_artifact(
    execution_dir: Path,
    artifact_type: str,
    filename_pattern: str,
) -> ToolOutputArtifact:
    """为手工 ToolPlan 创建一个必需输出声明。"""

    return ToolOutputArtifact(
        artifact_type=artifact_type,
        path_pattern=str(execution_dir / filename_pattern),
        side="single",
        required=True,
    )


def build_test_tool_plan(workdir: Path) -> ToolPlan:
    """构造 clean -> pocket -> integrate 的完整三步测试计划。"""

    clean_dir = workdir / "01_clean"
    pocket_dir = workdir / "02_pocket"
    integrate_dir = workdir / "03_integrate"
    staged_reports_dir = integrate_dir / "input_reports"

    clean_call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=1,
        tool_name="enzywizard-clean",
        command_name="enzywizard-clean",
        side="single",
        tool_execution_dir=str(clean_dir),
        arguments={
            "input_path": str(workdir / "input.pdb"),
            "output_dir": str(clean_dir),
            "add_H": False,
            "pH": 7.5,
        },
        output_artifacts=[
            output_artifact(clean_dir, "cleaned_structure_cif", "cleaned_input.cif"),
            output_artifact(clean_dir, "cleaned_structure_pdb", "cleaned_input.pdb"),
            output_artifact(clean_dir, "cleaned_sequence_fasta", "cleaned_input.fasta"),
            output_artifact(clean_dir, "clean_report_json", "clean_report_input.json"),
            output_artifact(clean_dir, "log_file", "log.txt"),
        ],
        staging_rules=[],
        call_reason="Prepare the test structure.",
    )
    cleaned_structure_reference = ToolOutputReference(
        source_step_index=1,
        artifact_type="cleaned_structure_cif",
        expected_path_pattern=str(clean_dir / "cleaned_input.cif"),
        json_field=None,
    )
    clean_report_value_reference = ToolOutputReference(
        source_step_index=1,
        artifact_type="clean_report_json",
        expected_path_pattern=str(clean_dir / "clean_report_input.json"),
        json_field="test_min_volume",
    )
    pocket_call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=2,
        tool_name="enzywizard-pocket",
        command_name="enzywizard-pocket",
        side="single",
        tool_execution_dir=str(pocket_dir),
        arguments={
            "input_path": cleaned_structure_reference,
            "output_dir": str(pocket_dir),
            "min_rad": 1.8,
            "max_rad": 6.2,
            "min_volume": clean_report_value_reference,
        },
        output_artifacts=[
            output_artifact(pocket_dir, "pocket_report_json", "pocket_report_*.json"),
            output_artifact(pocket_dir, "log_file", "log.txt"),
        ],
        staging_rules=[],
        call_reason="Find test pockets.",
    )
    integrate_call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=3,
        tool_name="enzywizard-integrate",
        command_name="enzywizard-integrate",
        side="single",
        tool_execution_dir=str(integrate_dir),
        arguments={
            "input_dir": str(staged_reports_dir),
            "output_dir": str(integrate_dir),
            "strict": False,
        },
        output_artifacts=[
            output_artifact(
                integrate_dir,
                "integrate_report_json",
                "integrate_report_*.json",
            ),
            output_artifact(
                integrate_dir,
                "integrate_nodes_json",
                "integrate_nodes_*.json",
            ),
            output_artifact(
                integrate_dir,
                "integrate_edges_json",
                "integrate_edges_*.json",
            ),
            output_artifact(integrate_dir, "log_file", "log.txt"),
        ],
        staging_rules=[
            ReportStagingRule(
                source_step_index=1,
                source_path_pattern=str(clean_dir / "*.json"),
                destination_dir=str(staged_reports_dir),
                side="single",
            ),
            ReportStagingRule(
                source_step_index=2,
                source_path_pattern=str(pocket_dir / "*.json"),
                destination_dir=str(staged_reports_dir),
                side="single",
            ),
        ],
        call_reason="Integrate current-plan reports.",
    )
    return ToolPlan(
        turn_id=1,
        plan_id=1,
        purpose_type="single_protein_analysis",
        summary="Test tool execution.",
        plan_reason="Exercise references, staging, and output tracking.",
        calls=[clean_call, pocket_call, integrate_call],
    )


def build_smiles_reference_tool_plan(workdir: Path) -> ToolPlan:
    """构造多构象 Substrate -> Dock -> Interaction 引用测试计划。"""

    substrate_dir = workdir / "01_substrate"
    dock_dir = workdir / "02_dock"
    interaction_dir = workdir / "03_interaction"

    substrate_call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=1,
        tool_name="enzywizard-substrate",
        command_name="enzywizard-substrate",
        side="single",
        tool_execution_dir=str(substrate_dir),
        arguments={
            "substrate_names": "C(CO)O",
            "output_dir": str(substrate_dir),
        },
        output_artifacts=[
            output_artifact(
                substrate_dir,
                "substrate_report_json",
                "substrate_report_*.json",
            ),
            output_artifact(substrate_dir, "substrate_sdf", "*.sdf"),
            output_artifact(substrate_dir, "log_file", "log.txt"),
        ],
        staging_rules=[],
        call_reason="Resolve the direct SMILES and generate multiple conformers.",
    )
    resolved_substrate_name = ToolOutputReference(
        source_step_index=1,
        artifact_type="substrate_report_json",
        expected_path_pattern=str(substrate_dir / "substrate_report_*.json"),
        json_path=["substrates", 0, "substrate_name"],
    )
    dock_call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=2,
        tool_name="enzywizard-dock",
        command_name="enzywizard-dock",
        side="single",
        tool_execution_dir=str(dock_dir),
        arguments={
            "input_path": str(workdir / "protein.cif"),
            "substrate_names": resolved_substrate_name,
            "substrate_dir": str(substrate_dir),
            "output_dir": str(dock_dir),
        },
        output_artifacts=[
            output_artifact(dock_dir, "dock_report_json", "dock_report_*.json"),
            output_artifact(
                dock_dir,
                "docked_substrate_sdf",
                "docked_*.sdf",
            ),
            output_artifact(dock_dir, "log_file", "log.txt"),
        ],
        staging_rules=[],
        call_reason="Dock every conformer associated with the resolved substrate name.",
    )
    docked_substrate_name = ToolOutputReference(
        source_step_index=2,
        artifact_type="docked_substrate_sdf",
        expected_path_pattern=str(dock_dir / "docked_*.sdf"),
        path_selector="stem",
    )
    interaction_call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=3,
        tool_name="enzywizard-interaction",
        command_name="enzywizard-interaction",
        side="single",
        tool_execution_dir=str(interaction_dir),
        arguments={
            "input_path": str(workdir / "protein.cif"),
            "output_dir": str(interaction_dir),
            "substrate_names": docked_substrate_name,
            "substrate_dir": str(dock_dir),
        },
        output_artifacts=[
            output_artifact(
                interaction_dir,
                "interaction_report_json",
                "interaction_report_*.json",
            ),
            output_artifact(interaction_dir, "log_file", "log.txt"),
        ],
        staging_rules=[],
        call_reason="Read the actual docked SDF by its file stem.",
    )
    return ToolPlan(
        turn_id=1,
        plan_id=1,
        purpose_type="single_protein_analysis",
        summary="Test resolved substrate and docked SDF references.",
        plan_reason="Exercise nested JSON and artifact stem references.",
        calls=[substrate_call, dock_call, interaction_call],
    )


def build_test_state(tool_plan: ToolPlan) -> AgentState:
    """构造已批准计划对应的 Tool Calling 最小状态。"""

    planning_result = PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary=tool_plan.summary,
        purpose_type="single_protein_analysis",
        purpose_reason="Test execution.",
        tool_selection=None,
        parameter_extraction=None,
        tool_plan=tool_plan,
        error_messages=[],
    )
    return {
        "current_planning_result": planning_result,
        "current_user_confirmation": UserConfirmation(
            turn_id=1,
            plan_id=1,
            decision="approved",
            revision_request=None,
        ),
        "tool_executions": [],
        "cleaned_files": CleanedFiles(
            **{field_name: None for field_name in CleanedFiles.model_fields}
        ),
        "current_plan_artifacts": [],
        "artifacts": [],
    }


def empty_cleaned_files() -> CleanedFiles:
    """创建尚无任何可信清理产物的初始状态。"""

    return CleanedFiles(
        **{field_name: None for field_name in CleanedFiles.model_fields}
    )


def validate_generated_plan_references(workdir: Path) -> dict[str, object]:
    """验证单蛋白和 WT/MUT 计划均生成正确的底物产物引用。"""

    config = AgentConfig(
        api_key="test",
        base_url="https://example.invalid/v1",
        model="test-model",
        workdir=workdir,
    )
    single_state: AgentState = {
        "config": config,
        "session_id": "single-reference-test",
        "agent_inputs": AgentInputs(
            structure_file=str(workdir / "single.pdb"),
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name="C(CO)O",
            amino_acid_substitution=None,
        ),
        "cleaned_files": empty_cleaned_files(),
    }
    single_plan = build_single_protein_tool_plan(
        single_state,
        turn_id=1,
        plan_id=1,
        selected_tool_names=[
            "enzywizard-clean",
            "enzywizard-substrate",
            "enzywizard-dock",
            "enzywizard-interaction",
            "enzywizard-integrate",
        ],
        plan_reason="Test single-protein substrate references.",
    )
    single_validation = audit_tool_plan(single_plan, workdir)
    if not single_validation.passed:
        raise AssertionError(single_validation.model_dump_json(indent=2))
    single_substrate = next(
        call for call in single_plan.calls if call.tool_name == "enzywizard-substrate"
    )
    single_dock = next(
        call for call in single_plan.calls if call.tool_name == "enzywizard-dock"
    )
    single_interaction = next(
        call for call in single_plan.calls if call.tool_name == "enzywizard-interaction"
    )
    single_dock_name = single_dock.arguments["substrate_names"]
    single_interaction_name = single_interaction.arguments["substrate_names"]
    if not isinstance(single_dock_name, ToolOutputReference) or (
        single_dock_name.source_step_index != single_substrate.step_index
        or single_dock_name.json_path != ["substrates", 0, "substrate_name"]
    ):
        raise AssertionError(
            "Single-protein Dock does not reference the substrate report."
        )
    if not isinstance(single_interaction_name, ToolOutputReference) or (
        single_interaction_name.source_step_index != single_dock.step_index
        or single_interaction_name.artifact_type != "docked_substrate_sdf"
        or single_interaction_name.path_selector != "stem"
    ):
        raise AssertionError(
            "Single-protein Interaction does not reference Dock SDF stem."
        )

    mutant_state: AgentState = {
        "config": config,
        "session_id": "mutant-reference-test",
        "agent_inputs": AgentInputs(
            structure_file=str(workdir / "wt.pdb"),
            msa_file=None,
            mut_structure_file=str(workdir / "mut.pdb"),
            mut_msa_file=None,
            substrate_name="C(CO)O",
            amino_acid_substitution="S70G",
        ),
        "cleaned_files": empty_cleaned_files(),
    }
    mutant_plan = build_wildtype_mutant_tool_plan(
        mutant_state,
        turn_id=1,
        plan_id=1,
        selected_tool_names=[
            "enzywizard-mut-clean",
            "enzywizard-substrate",
            "enzywizard-dock",
            "enzywizard-interaction",
            "enzywizard-mut-integrate",
        ],
        plan_reason="Test WT/MUT substrate references.",
    )
    mutant_validation = audit_tool_plan(mutant_plan, workdir)
    if not mutant_validation.passed:
        raise AssertionError(mutant_validation.model_dump_json(indent=2))
    shared_substrate = next(
        call
        for call in mutant_plan.calls
        if call.tool_name == "enzywizard-substrate"
    )
    paired_reference_count = 0
    for side in ("wt", "mut"):
        dock_call = next(
            call
            for call in mutant_plan.calls
            if call.tool_name == "enzywizard-dock" and call.side == side
        )
        interaction_call = next(
            call
            for call in mutant_plan.calls
            if call.tool_name == "enzywizard-interaction" and call.side == side
        )
        dock_name = dock_call.arguments["substrate_names"]
        interaction_name = interaction_call.arguments["substrate_names"]
        if not isinstance(dock_name, ToolOutputReference) or (
            dock_name.source_step_index != shared_substrate.step_index
            or dock_name.json_path != ["substrates", 0, "substrate_name"]
        ):
            raise AssertionError(f"{side.upper()} Dock has the wrong substrate reference.")
        if not isinstance(interaction_name, ToolOutputReference) or (
            interaction_name.source_step_index != dock_call.step_index
            or interaction_name.artifact_type != "docked_substrate_sdf"
            or interaction_name.path_selector != "stem"
        ):
            raise AssertionError(
                f"{side.upper()} Interaction has the wrong Dock SDF reference."
            )
        paired_reference_count += 1

    serializer = create_in_memory_checkpointer().serde
    planning_result = PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary=single_plan.summary,
        purpose_type="single_protein_analysis",
        purpose_reason="Test checkpoint serialization.",
        tool_selection=None,
        parameter_extraction=None,
        tool_plan=single_plan,
        error_messages=[],
    )
    restored_planning_result = serializer.loads_typed(
        serializer.dumps_typed(planning_result)
    )
    if restored_planning_result != planning_result:
        raise AssertionError(
            "PlanningResult checkpoint round trip changed output references."
        )

    return {
        "single_reference_chain": True,
        "paired_reference_count": paired_reference_count,
    }


def validate_successful_execution(workdir: Path) -> dict[str, object]:
    """验证完整三步链、延迟引用、报告复制和状态更新。"""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_test_tool_plan(workdir)
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    state = tool_calling_node(build_test_state(tool_plan))
    execution = state["current_tool_execution"]

    if execution.execution_status != "succeeded":
        raise AssertionError(execution.model_dump_json(indent=2))
    if len(execution.call_executions) != 3:
        raise AssertionError("All three tool calls must execute.")
    clean_command = execution.call_executions[0].command
    if "--no_add_H" not in clean_command or clean_command[-2:] != ["--pH", "7.5"]:
        raise AssertionError(f"Boolean or scalar CLI arguments are wrong: {clean_command}")
    resolved_pocket_input = execution.call_executions[1].resolved_arguments[
        "input_path"
    ]
    if resolved_pocket_input != str((workdir / "01_clean/cleaned_input.cif").resolve()):
        raise AssertionError("ToolOutputReference did not resolve to the real artifact.")
    if execution.call_executions[1].resolved_arguments["min_volume"] != 50.0:
        raise AssertionError("ToolOutputReference did not resolve the JSON field value.")
    staged_reports = sorted((workdir / "03_integrate/input_reports").glob("*.json"))
    if len(staged_reports) != 2:
        raise AssertionError("Integrate did not receive both staged reports.")
    if state["cleaned_files"].single_cleaned_cif_file is None:
        raise AssertionError("CleanedFiles was not updated after successful clean.")
    if not all(
        Path(call_execution.stdout_file).is_file()
        and Path(call_execution.stderr_file).is_file()
        for call_execution in execution.call_executions
    ):
        raise AssertionError("Full stdout/stderr files were not preserved.")

    # ToolExecution 包含嵌套 ToolCallExecution 和 Artifact，必须能被当前安全
    # checkpoint 白名单完整序列化和恢复，图中断后才能继续交给 Analysis。
    serializer = create_in_memory_checkpointer().serde
    restored_execution = serializer.loads_typed(
        serializer.dumps_typed(execution)
    )
    if restored_execution != execution:
        raise AssertionError("ToolExecution checkpoint round trip changed the result.")

    return {
        "execution_status": execution.execution_status,
        "call_sequence": [
            call_execution.tool_name
            for call_execution in execution.call_executions
        ],
        "artifact_count": len(state["current_plan_artifacts"]),
        "cleaned_cif": state["cleaned_files"].single_cleaned_cif_file,
    }


def validate_smiles_reference_execution(workdir: Path) -> dict[str, object]:
    """验证多构象 SMILES 名称能沿两级真实产物引用传递。"""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_smiles_reference_tool_plan(workdir)
    (workdir / "protein.cif").write_text("STRUCTURE", encoding="utf-8")
    state = tool_calling_node(build_test_state(tool_plan))
    execution = state["current_tool_execution"]

    if execution.execution_status != "succeeded":
        raise AssertionError(execution.model_dump_json(indent=2))
    substrate_artifacts = [
        artifact
        for artifact in execution.call_executions[0].artifacts
        if artifact.artifact_type == "substrate_sdf"
    ]
    if len(substrate_artifacts) != 3:
        raise AssertionError(
            "All generated substrate conformers must remain tracked as artifacts."
        )
    dock_substrate_name = execution.call_executions[1].resolved_arguments[
        "substrate_names"
    ]
    if dock_substrate_name != "smiles1":
        raise AssertionError(
            f"Dock did not receive the resolved SMILES name: {dock_substrate_name!r}."
        )
    interaction_substrate_name = execution.call_executions[2].resolved_arguments[
        "substrate_names"
    ]
    if interaction_substrate_name != "docked_smiles1":
        raise AssertionError(
            "Interaction did not receive the actual docked SDF stem: "
            f"{interaction_substrate_name!r}."
        )

    return {
        "execution_status": execution.execution_status,
        "substrate_conformer_count": len(substrate_artifacts),
        "dock_substrate_name": dock_substrate_name,
        "interaction_substrate_name": interaction_substrate_name,
    }


def validate_failed_execution(workdir: Path) -> dict[str, object]:
    """验证非零退出码会保留已有结果、截取日志尾部并停止后续调用。"""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_test_tool_plan(workdir)
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    os.environ["FAKE_FAIL_TOOL"] = "enzywizard-pocket"
    try:
        state = tool_calling_node(build_test_state(tool_plan))
    finally:
        os.environ.pop("FAKE_FAIL_TOOL", None)

    execution = state["current_tool_execution"]
    if execution.execution_status != "failed" or execution.failed_step_index != 2:
        raise AssertionError(execution.model_dump_json(indent=2))
    if len(execution.call_executions) != 2:
        raise AssertionError("Execution must stop before the integrate call.")
    failed_call = execution.call_executions[-1]
    if failed_call.exit_code != 7:
        raise AssertionError("The subprocess exit code was not retained.")
    if "FAILURE_TAIL_MARKER" not in failed_call.stderr_excerpt:
        raise AssertionError("stderr excerpt did not retain the log tail.")
    if "LOG_TAIL_MARKER" not in failed_call.log_excerpt:
        raise AssertionError("Tool log excerpt did not retain the log tail.")
    if not failed_call.log_files or not all(
        Path(log_file).is_file() for log_file in failed_call.log_files
    ):
        raise AssertionError("The complete tool log was not preserved.")
    if len(failed_call.stderr_excerpt) > 2100:
        raise AssertionError("stderr excerpt exceeds the intended state size.")
    if state["cleaned_files"].single_cleaned_cif_file is None:
        raise AssertionError("Successful clean output was lost after a later failure.")
    failed_scientific_artifacts = [
        artifact
        for artifact in failed_call.artifacts
        if artifact.artifact_type == "pocket_report_json"
    ]
    if len(failed_scientific_artifacts) != 1:
        raise AssertionError("The fixture did not produce the expected partial report.")
    global_paths = {artifact.path for artifact in state["artifacts"]}
    if failed_scientific_artifacts[0].path in global_paths:
        raise AssertionError("A failed scientific output entered global artifact state.")
    if not any(
        artifact.artifact_type in {"agent_stdout_log", "agent_stderr_log", "log_file"}
        and artifact.source_tool == "enzywizard-pocket"
        for artifact in state["artifacts"]
    ):
        raise AssertionError("Failed-call audit logs were not retained globally.")

    # Simulate loading a checkpoint written before failed-output isolation existed.
    state["artifacts"].extend(failed_scientific_artifacts)
    state["current_turn_id"] = 2
    catalog_paths = {
        entry.artifact.path for entry in routing_service._history_catalog(state)
    }
    if failed_scientific_artifacts[0].path in catalog_paths:
        raise AssertionError("A legacy failed output entered the history catalog.")

    return {
        "execution_status": execution.execution_status,
        "failed_step_index": execution.failed_step_index,
        "exit_code": failed_call.exit_code,
        "executed_call_count": len(execution.call_executions),
        "preserved_artifact_count": len(state["current_plan_artifacts"]),
        "failed_scientific_output_isolated": True,
    }


def validate_missing_required_output(workdir: Path) -> dict[str, object]:
    """验证退出码为零但缺少必需产物时仍会停止计划并明确失败。"""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_test_tool_plan(workdir)
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    os.environ["FAKE_SKIP_OUTPUT_TOOL"] = "enzywizard-pocket"
    try:
        state = tool_calling_node(build_test_state(tool_plan))
    finally:
        os.environ.pop("FAKE_SKIP_OUTPUT_TOOL", None)

    execution = state["current_tool_execution"]
    failed_call = execution.call_executions[-1]
    if execution.execution_status != "failed" or execution.failed_step_index != 2:
        raise AssertionError(execution.model_dump_json(indent=2))
    if failed_call.exit_code != 0:
        raise AssertionError("Missing-output failure should retain exit_code=0.")
    if "pocket_report_json" not in (failed_call.error_message or ""):
        raise AssertionError("Missing artifact type was not included in the error.")

    return {
        "execution_status": execution.execution_status,
        "failed_step_index": execution.failed_step_index,
        "exit_code": failed_call.exit_code,
        "error_message": failed_call.error_message,
    }


def validate_bounded_transient_recovery(workdir: Path) -> dict[str, object]:
    """Verify one classified transient failure is retried once and audited."""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_test_tool_plan(workdir)
    tool_plan.calls = tool_plan.calls[:1]
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    state = build_test_state(tool_plan)
    state["config"] = AgentConfig(
        api_key="recovery-test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        workdir=workdir,
        tool_recovery_mode="safe",
        max_tool_recovery_attempts=1,
    )
    state["session_id"] = "bounded-recovery-test"
    os.environ["FAKE_TRANSIENT_TOOL"] = "enzywizard-clean"
    try:
        state = tool_calling_node(state)
    finally:
        os.environ.pop("FAKE_TRANSIENT_TOOL", None)

    execution = state["current_tool_execution"]
    if execution.execution_status != "succeeded":
        raise AssertionError(execution.model_dump_json(indent=2))
    if len(execution.recovery_attempts) != 1:
        raise AssertionError("Transient failure was not retried exactly once")
    attempt = execution.recovery_attempts[0]
    if attempt.trigger_category != "transient_external":
        raise AssertionError("Transient failure received the wrong diagnosis")
    if attempt.execution_status != "succeeded":
        raise AssertionError("The bounded retry did not succeed")
    decisions = [item.decision for item in execution.workflow_decisions]
    if decisions != ["retry", "continue"]:
        raise AssertionError(f"Unexpected recovery decisions: {decisions}")
    runtime_errors = state["runtime_session"].errors
    if len(runtime_errors) != 1 or not runtime_errors[0].recoverable:
        raise AssertionError("Recovered error was not retained as recoverable")
    clean_dir = Path(tool_plan.calls[0].tool_execution_dir)
    if (clean_dir / "failed_attempt_only.tmp").exists():
        raise AssertionError("Failed-attempt output was not cleared before retry")
    if not (clean_dir / "stderr.txt").is_file():
        raise AssertionError("The first attempt's audit log was not preserved")
    if not (clean_dir / "stderr_attempt_2.txt").is_file():
        raise AssertionError("The retry audit log was not preserved")

    manifest_path = workdir / "session_bounded-recovery-test/session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_execution = manifest["tool_executions"][0]
    if len(manifest_execution["recovery_attempts"]) != 1:
        raise AssertionError("Manifest omitted the recovery attempt")
    return {
        "execution_status": execution.execution_status,
        "recovery_attempt_count": len(execution.recovery_attempts),
        "workflow_decisions": decisions,
        "manifest_schema_version": manifest["manifest_schema_version"],
    }


def validate_scientific_change_requires_confirmation(
    workdir: Path,
) -> dict[str, object]:
    """Verify scientific parameter changes are never applied automatically."""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_test_tool_plan(workdir)
    tool_plan.calls = tool_plan.calls[:1]
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    state = build_test_state(tool_plan)
    state["config"] = AgentConfig(
        api_key="confirmation-test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        workdir=workdir,
        tool_recovery_mode="safe",
        max_tool_recovery_attempts=2,
    )
    state["session_id"] = "scientific-confirmation-test"
    os.environ["FAKE_SCIENTIFIC_ERROR_TOOL"] = "enzywizard-clean"
    try:
        state = tool_calling_node(state)
    finally:
        os.environ.pop("FAKE_SCIENTIFIC_ERROR_TOOL", None)

    execution = state["current_tool_execution"]
    failed_call = execution.call_executions[-1]
    diagnosis = failed_call.failure_diagnosis
    if diagnosis is None or diagnosis.category != "scientific_configuration":
        raise AssertionError("Scientific configuration failure was misclassified")
    if diagnosis.recoverability != "confirmation_required":
        raise AssertionError("Scientific configuration change did not require confirmation")
    if execution.recovery_attempts:
        raise AssertionError("A scientific parameter change was retried automatically")
    if execution.workflow_decisions[-1].decision != "request_confirmation":
        raise AssertionError("The workflow did not request confirmation")
    return {
        "execution_status": execution.execution_status,
        "failure_category": diagnosis.category,
        "recoverability": diagnosis.recoverability,
        "recovery_attempt_count": len(execution.recovery_attempts),
    }


def validate_unresolved_reference(workdir: Path) -> dict[str, object]:
    """验证无法定位上游产物时不会启动当前命令或继续后续步骤。"""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_test_tool_plan(workdir)
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    tool_plan.calls[1].arguments["input_path"] = ToolOutputReference(
        source_step_index=1,
        artifact_type="missing_structure",
        expected_path_pattern=str(workdir / "01_clean/missing.cif"),
        json_field=None,
    )
    state = tool_calling_node(build_test_state(tool_plan))

    execution = state["current_tool_execution"]
    failed_call = execution.call_executions[-1]
    if execution.execution_status != "failed" or execution.failed_step_index != 2:
        raise AssertionError(execution.model_dump_json(indent=2))
    if failed_call.exit_code is not None or failed_call.command:
        raise AssertionError("An unresolved reference must fail before subprocess starts.")
    if "missing_structure" not in (failed_call.error_message or ""):
        raise AssertionError("Reference failure did not identify the missing artifact.")

    return {
        "execution_status": execution.execution_status,
        "failed_step_index": execution.failed_step_index,
        "exit_code": failed_call.exit_code,
        "command_started": bool(failed_call.command),
    }


def validate_tool_timeout(workdir: Path) -> dict[str, object]:
    """验证超时会终止工具进程、保留前序产物并写入终态 manifest。"""

    workdir.mkdir(parents=True, exist_ok=True)
    tool_plan = build_test_tool_plan(workdir)
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    state = build_test_state(tool_plan)
    state["config"] = AgentConfig(
        api_key="timeout-test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        workdir=workdir,
        tool_timeout_seconds=0.15,
    )
    state["session_id"] = "tool-timeout-test"

    os.environ["FAKE_SLEEP_TOOL"] = "enzywizard-pocket"
    os.environ["FAKE_SLEEP_SECONDS"] = "2.0"
    try:
        state = tool_calling_node(state)
    finally:
        os.environ.pop("FAKE_SLEEP_TOOL", None)
        os.environ.pop("FAKE_SLEEP_SECONDS", None)

    execution = state["current_tool_execution"]
    runtime = state["runtime_session"]
    if execution.execution_status != "failed" or execution.failed_step_index != 2:
        raise AssertionError(execution.model_dump_json(indent=2))
    if runtime.status != "timed_out" or runtime.stop_reason != "tool_timeout":
        raise AssertionError("Tool timeout did not update the runtime terminal state")
    if state["cleaned_files"].single_cleaned_cif_file is None:
        raise AssertionError("Tool timeout discarded a successful preceding output")
    manifest_path = workdir / "session_tool-timeout-test/session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "timed_out":
        raise AssertionError("Timeout manifest did not preserve timed_out status")
    if manifest["errors"][-1]["category"] != "timeout":
        raise AssertionError("Tool timeout was not assigned the timeout error category")
    if not manifest["tool_plans"]:
        raise AssertionError("Manifest did not record the approved tool parameters")
    recorded_input = manifest["tool_plans"][0]["calls"][0]["arguments"]["input_path"]
    if str(recorded_input).startswith("/"):
        raise AssertionError("Manifest exposed an absolute tool input path")

    return {
        "execution_status": execution.execution_status,
        "failed_step_index": execution.failed_step_index,
        "runtime_status": runtime.status,
        "preserved_artifact_count": len(state["current_plan_artifacts"]),
    }


def validate_log_redaction(workdir: Path) -> dict[str, object]:
    """验证认证信息不会进入 Agent 日志、状态摘录或 manifest。"""

    workdir.mkdir(parents=True, exist_ok=True)
    secret = "tool-log-private-key-123456789"
    tool_plan = build_test_tool_plan(workdir)
    tool_plan.calls = tool_plan.calls[:1]
    (workdir / "input.pdb").write_text("RAW", encoding="utf-8")
    state = build_test_state(tool_plan)
    state["config"] = AgentConfig(
        api_key=secret,
        base_url="https://example.invalid/v1",
        model="test-model",
        workdir=workdir,
    )
    state["session_id"] = "log-redaction-test"
    os.environ["FAKE_PRINT_SECRET"] = secret
    try:
        state = tool_calling_node(state)
    finally:
        os.environ.pop("FAKE_PRINT_SECRET", None)

    call = state["current_tool_execution"].call_executions[0]
    manifest_path = workdir / "session_log-redaction-test/session_manifest.json"
    protected_texts = [
        call.stdout_excerpt,
        call.stderr_excerpt,
        call.log_excerpt,
        Path(call.stdout_file).read_text(encoding="utf-8"),
        Path(call.stderr_file).read_text(encoding="utf-8"),
        manifest_path.read_text(encoding="utf-8"),
    ]
    if any(secret in text for text in protected_texts):
        raise AssertionError("Agent-managed logs or manifest retained an API key")
    return {
        "execution_status": state["current_tool_execution"].execution_status,
        "stdout_redacted": "[REDACTED_API_KEY]" in call.stdout_excerpt,
        "log_excerpt_redacted": "[REDACTED_API_KEY]" in call.log_excerpt,
    }


def main() -> None:
    """运行成功和失败两类 Tool Calling 测试并打印精简结果。"""

    with tempfile.TemporaryDirectory(prefix="enzywizard-agent-tool-calling-") as root:
        root_path = Path(root)
        fake_bin_dir = root_path / "bin"
        fake_bin_dir.mkdir()
        create_fake_tool_commands(fake_bin_dir)

        original_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{fake_bin_dir}{os.pathsep}{original_path}"
        try:
            results = {
                "generated_plan_reference_case": validate_generated_plan_references(
                    root_path / "generated_plans"
                ),
                "success_case": validate_successful_execution(root_path / "success"),
                "smiles_reference_case": validate_smiles_reference_execution(
                    root_path / "smiles_reference"
                ),
                "failure_case": validate_failed_execution(root_path / "failure"),
                "missing_output_case": validate_missing_required_output(
                    root_path / "missing_output"
                ),
                "bounded_transient_recovery_case": (
                    validate_bounded_transient_recovery(
                        root_path / "bounded_transient_recovery"
                    )
                ),
                "scientific_confirmation_case": (
                    validate_scientific_change_requires_confirmation(
                        root_path / "scientific_confirmation"
                    )
                ),
                "unresolved_reference_case": validate_unresolved_reference(
                    root_path / "unresolved_reference"
                ),
                "timeout_case": validate_tool_timeout(root_path / "timeout"),
                "log_redaction_case": validate_log_redaction(
                    root_path / "log_redaction"
                ),
            }
        finally:
            os.environ["PATH"] = original_path

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
