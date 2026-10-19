"""直接调用当前环境中的 enzywizard-mut-clean，批量验证 1.0.4 行为。"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WT_INPUT = PROJECT_ROOT / "test_inputs" / "1ZG4_WT.pdb"
DEFAULT_MUT_INPUT = PROJECT_ROOT / "test_inputs" / "1ZG6_S70G.pdb"
EXPECTED_PACKAGE_VERSION = "1.0.4"


@dataclass(frozen=True)
class CliTestCase:
    """描述一个可独立运行的 mut-clean CLI 测试案例。"""

    case_id: str
    mutation: str | None
    expected_exit_code: int
    expected_text: str
    validate_success_outputs: bool = False
    pH: float = 7.0


CLI_TEST_CASES = [
    CliTestCase(
        case_id="valid_s70g",
        mutation="S70G",
        expected_exit_code=0,
        expected_text="Mutclean processing finished",
        validate_success_outputs=True,
    ),
    # 1ZG4/1ZG6 有 263 个残基，但原始编号范围是 26-290。这个案例专门
    # 防止 1.0.2 中“用残基数量校验原始编号”的问题再次出现。
    CliTestCase(
        case_id="valid_original_residue_id_280",
        mutation="A280V",
        expected_exit_code=0,
        expected_text="Mutclean processing finished",
        validate_success_outputs=True,
    ),
    CliTestCase(
        case_id="missing_residue_below_range",
        mutation="A25V",
        expected_exit_code=1,
        expected_text="Mutation position not found in wild-type structure",
    ),
    CliTestCase(
        case_id="missing_residue_above_range",
        mutation="A291V",
        expected_exit_code=1,
        expected_text="Mutation position not found in wild-type structure",
    ),
    CliTestCase(
        case_id="invalid_mutation_format",
        mutation="S70",
        expected_exit_code=1,
        expected_text="Invalid amino acid substitution format",
    ),
    CliTestCase(
        case_id="unchanged_amino_acid",
        mutation="S70S",
        expected_exit_code=1,
        expected_text="No amino acid mutated",
    ),
    CliTestCase(
        case_id="duplicate_mutation_position",
        mutation="S70G;A70V",
        expected_exit_code=1,
        expected_text="Duplicate mutation position detected",
    ),
    CliTestCase(
        case_id="ph_below_range",
        mutation="S70G",
        expected_exit_code=1,
        expected_text="Invalid pH value",
        pH=-0.1,
    ),
    CliTestCase(
        case_id="ph_above_range",
        mutation="S70G",
        expected_exit_code=1,
        expected_text="Invalid pH value",
        pH=14.1,
    ),
    # mutation=None 表示故意不提交必需参数，验证 argparse 的退出码和提示。
    CliTestCase(
        case_id="missing_required_mutation_argument",
        mutation=None,
        expected_exit_code=2,
        expected_text="the following arguments are required",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    """创建批量 CLI 测试参数。"""

    parser = argparse.ArgumentParser(
        description="Run real enzywizard-mut-clean 1.0.4 CLI compatibility tests."
    )
    parser.add_argument("--executable", default="enzywizard-mut-clean")
    parser.add_argument("--wt_input", default=str(DEFAULT_WT_INPUT))
    parser.add_argument("--mut_input", default=str(DEFAULT_MUT_INPUT))
    parser.add_argument(
        "--workdir",
        default="/tmp/enzywizard-mut-clean-cli-tests",
        help="Parent directory for timestamped test runs.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON result path. Defaults to <run_dir>/results.json.",
    )
    parser.add_argument(
        "--case_ids",
        nargs="*",
        help="Run only the listed case IDs. By default all cases are run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess output for every case.",
    )
    return parser


def create_run_directory(parent: Path) -> Path:
    """创建不会覆盖之前结果的独立运行目录。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = parent.expanduser().resolve() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_installed_package_version() -> str:
    """读取安装元数据版本，确保测试的确针对当前环境中的包。"""

    try:
        return version("enzywizard-mut-clean")
    except PackageNotFoundError as error:
        raise RuntimeError("enzywizard-mut-clean is not installed.") from error


def get_ca_coordinates(pdb_path: Path) -> np.ndarray:
    """按结构顺序读取单模型 PDB 中的蛋白 CA 坐标。"""

    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, pdb_path)
    model = next(structure.get_models())
    coordinates = [residue["CA"].coord for residue in model.get_residues() if "CA" in residue]
    if not coordinates:
        raise AssertionError(f"No CA atoms found in {pdb_path}.")
    return np.asarray(coordinates, dtype=float)


def validate_saved_alignment(
    wt_pdb_path: Path,
    mut_pdb_path: Path,
    log_text: str,
) -> dict[str, float | int]:
    """确认保存后的 MUT 坐标确实处于 WT 对齐坐标系。"""

    wt_coordinates = get_ca_coordinates(wt_pdb_path)
    mut_coordinates = get_ca_coordinates(mut_pdb_path)
    if wt_coordinates.shape != mut_coordinates.shape:
        raise AssertionError(
            "Cleaned WT/MUT CA coordinate arrays have different shapes: "
            f"{wt_coordinates.shape} != {mut_coordinates.shape}."
        )

    # PDB 坐标单位为埃；工具日志中的 RMSD 单位为纳米。
    direct_rmsd_angstrom = math.sqrt(
        float(np.mean(np.sum((wt_coordinates - mut_coordinates) ** 2, axis=1)))
    )
    direct_rmsd_nm = direct_rmsd_angstrom / 10.0
    match = re.search(r"CA RMSD:\s*([0-9]+(?:\.[0-9]+)?)\s*nm", log_text)
    if match is None:
        raise AssertionError("The tool log does not contain the CA alignment RMSD.")
    logged_rmsd_nm = float(match.group(1))

    # 日志保留四位小数，PDB 坐标保留三位小数，因此允许小量序列化误差。
    if not math.isclose(direct_rmsd_nm, logged_rmsd_nm, abs_tol=0.002):
        raise AssertionError(
            "Saved-coordinate RMSD does not match the alignment log: "
            f"{direct_rmsd_nm:.6f} nm != {logged_rmsd_nm:.6f} nm."
        )
    return {
        "ca_atom_count": int(wt_coordinates.shape[0]),
        "saved_coordinate_rmsd_nm": direct_rmsd_nm,
        "logged_rmsd_nm": logged_rmsd_nm,
    }


def validate_success_outputs(
    output_dir: Path,
    mutation: str,
) -> dict[str, Any]:
    """校验 1.0.4 的完整文件契约、报告字段和坐标对齐结果。"""

    expected_paths = {
        "wt_cif": output_dir / "cleaned_1ZG4_WT.cif",
        "wt_pdb": output_dir / "cleaned_1ZG4_WT.pdb",
        "wt_fasta": output_dir / "cleaned_1ZG4_WT.fasta",
        "mut_cif": output_dir / "cleaned_1ZG6_S70G.cif",
        "mut_pdb": output_dir / "cleaned_1ZG6_S70G.pdb",
        "mut_fasta": output_dir / "cleaned_1ZG6_S70G.fasta",
        "report": output_dir / "mut_clean_report_1ZG4_WT_to_1ZG6_S70G.json",
        "log": output_dir / "log.txt",
    }
    missing_paths = [str(path) for path in expected_paths.values() if not path.is_file()]
    if missing_paths:
        raise AssertionError(f"Missing expected output files: {missing_paths}.")

    report = json.loads(expected_paths["report"].read_text(encoding="utf-8"))
    required_report_fields = {
        "report_type",
        "amino_acid_substitution",
        "cleaned_amino_acid_substitution",
        "wild_type_residue_mapping_old_to_new",
        "wild_type_clean_statistics",
        "mutant_residue_mapping_old_to_new",
        "mutant_clean_statistics",
    }
    missing_fields = sorted(required_report_fields - set(report))
    if missing_fields:
        raise AssertionError(f"Mut-clean report is missing fields: {missing_fields}.")
    if report["amino_acid_substitution"] != mutation:
        raise AssertionError(
            "The report did not preserve the submitted mutation: "
            f"{report['amino_acid_substitution']!r}."
        )
    cleaned_mutation = report["cleaned_amino_acid_substitution"]
    if not isinstance(cleaned_mutation, str) or re.fullmatch(
        r"[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]",
        cleaned_mutation,
    ) is None:
        raise AssertionError(
            "The report contains an invalid cleaned_amino_acid_substitution: "
            f"{cleaned_mutation!r}."
        )

    log_text = expected_paths["log"].read_text(encoding="utf-8", errors="replace")
    alignment = validate_saved_alignment(
        expected_paths["wt_pdb"],
        expected_paths["mut_pdb"],
        log_text,
    )
    return {
        "cleaned_amino_acid_substitution": cleaned_mutation,
        "output_files": [str(path) for path in expected_paths.values()],
        "alignment": alignment,
    }


def build_command(
    executable: str,
    wt_input: Path,
    mut_input: Path,
    case: CliTestCase,
    output_dir: Path,
) -> list[str]:
    """使用参数列表构造真实 CLI 命令，测试时始终禁用加氢以缩短耗时。"""

    command = [
        executable,
        "--wt_input_path",
        str(wt_input),
        "--mut_input_path",
        str(mut_input),
        "--output_dir",
        str(output_dir),
        "--no_add_H",
        "--pH",
        str(case.pH),
    ]
    if case.mutation is not None:
        command.extend(["--mutation", case.mutation])
    return command


def run_case(
    executable: str,
    wt_input: Path,
    mut_input: Path,
    case: CliTestCase,
    run_dir: Path,
) -> dict[str, Any]:
    """执行并判定一个 CLI 案例，保留完整 stdout、stderr 和工具输出。"""

    case_dir = run_dir / case.case_id
    output_dir = case_dir / "tool_output"
    case_dir.mkdir(parents=True, exist_ok=False)
    command = build_command(executable, wt_input, mut_input, case, output_dir)
    started_at = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=case_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    duration_seconds = time.monotonic() - started_at
    (case_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (case_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")

    log_path = output_dir / "log.txt"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    combined_text = "\n".join((completed.stdout or "", completed.stderr or "", log_text))
    failures: list[str] = []
    details: dict[str, Any] = {}

    if completed.returncode != case.expected_exit_code:
        failures.append(
            f"Expected exit code {case.expected_exit_code}, got {completed.returncode}."
        )
    if case.expected_text not in combined_text:
        failures.append(f"Expected output text was not found: {case.expected_text!r}.")
    if case.validate_success_outputs and case.mutation is not None:
        try:
            details = validate_success_outputs(output_dir, case.mutation)
        except (AssertionError, OSError, json.JSONDecodeError) as error:
            failures.append(str(error))

    return {
        "case_id": case.case_id,
        "status": "passed" if not failures else "failed",
        "command": command,
        "exit_code": completed.returncode,
        "duration_seconds": duration_seconds,
        "output_dir": str(output_dir),
        "failures": failures,
        "details": details,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "log": log_text,
    }


def select_cases(case_ids: list[str] | None) -> list[CliTestCase]:
    """按 case_id 筛选案例，并拒绝拼写错误的标识。"""

    if not case_ids:
        return CLI_TEST_CASES
    available = {case.case_id: case for case in CLI_TEST_CASES}
    unknown = sorted(set(case_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}.")
    return [available[case_id] for case_id in case_ids]


def main() -> int:
    """运行选定案例，打印摘要并保存机器可读结果。"""

    args = build_parser().parse_args()
    executable_path = shutil.which(args.executable)
    if executable_path is None:
        print(f"[ERROR] Executable not found on PATH: {args.executable}", file=sys.stderr)
        return 2

    wt_input = Path(args.wt_input).expanduser().resolve()
    mut_input = Path(args.mut_input).expanduser().resolve()
    for input_path in (wt_input, mut_input):
        if not input_path.is_file():
            print(f"[ERROR] Test input not found: {input_path}", file=sys.stderr)
            return 2

    installed_version = get_installed_package_version()
    if installed_version != EXPECTED_PACKAGE_VERSION:
        print(
            "[ERROR] Expected enzywizard-mut-clean "
            f"{EXPECTED_PACKAGE_VERSION}, found {installed_version}.",
            file=sys.stderr,
        )
        return 2

    try:
        selected_cases = select_cases(args.case_ids)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    run_dir = create_run_directory(Path(args.workdir))
    results: list[dict[str, Any]] = []
    for case in selected_cases:
        result = run_case(
            executable_path,
            wt_input,
            mut_input,
            case,
            run_dir,
        )
        results.append(result)
        print(
            f"[{'PASS' if result['status'] == 'passed' else 'FAIL'}] "
            f"{case.case_id} ({result['duration_seconds']:.2f}s)"
        )
        for failure in result["failures"]:
            print(f"  - {failure}")
        if args.verbose:
            print(result["stdout"], end="")
            print(result["stderr"], end="", file=sys.stderr)

    passed_count = sum(result["status"] == "passed" for result in results)
    failed_count = len(results) - passed_count
    payload = {
        "tested_package_version": installed_version,
        "executable": executable_path,
        "wt_input": str(wt_input),
        "mut_input": str(mut_input),
        "run_dir": str(run_dir),
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
        },
        "results": results,
    }
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary: total={len(results)}, passed={passed_count}, failed={failed_count}")
    print(f"Run directory: {run_dir}")
    print(f"Full results: {output_path}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
