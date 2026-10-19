"""对至少两个真实 OpenAI-compatible 服务执行启动能力检查。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.security_utils import redact_text, redact_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the strict EnzyWizard-Agent model contract against multiple services."
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        nargs=4,
        metavar=("NAME", "BASE_URL", "MODEL", "API_KEY"),
        required=True,
        help="Repeat for each OpenAI-compatible service; at least two are required.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=None)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--llm_extra_body_json", default=None)
    parser.add_argument("--llm_timeout_seconds", type=float, default=60.0)
    parser.add_argument("--llm_max_retries", type=int, default=1)
    parser.add_argument(
        "--output",
        default="model_compatibility_results.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.endpoint) < 2:
        raise SystemExit("[ERROR] Provide at least two --endpoint groups.")
    extra_body = None
    if args.llm_extra_body_json is not None:
        extra_body = json.loads(args.llm_extra_body_json)
        if not isinstance(extra_body, dict):
            raise SystemExit("[ERROR] --llm_extra_body_json must be a JSON object.")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="enzywizard-model-compatibility-") as root:
        for index, (name, base_url, model, api_key) in enumerate(args.endpoint, start=1):
            print(f"[端点 {index}/{len(args.endpoint)}] 开始：{name} ({model})", flush=True)
            state: AgentState = {
                "config": AgentConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    workdir=Path(root) / name,
                    llm_timeout_seconds=args.llm_timeout_seconds,
                    llm_max_retries=args.llm_max_retries,
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                    reasoning_effort=args.reasoning_effort,
                    llm_extra_body=extra_body,
                    check_model_capabilities=True,
                    max_llm_calls=2,
                ),
                "session_id": f"compatibility-{index}",
                "runtime_session": RuntimeSession(
                    started_at=utc_now(),
                    started_monotonic=time.monotonic(),
                ),
            }
            try:
                capability = check_model_capabilities(state)
                status = "passed"
                detail = capability.detail
                print("[端点结果] 通过：Chat Completions + strict JSON Schema", flush=True)
            except Exception as error:
                capability = state["runtime_session"].model_capability
                status = "failed"
                detail = redact_text(error, api_key=api_key)
                print(f"[端点结果] 失败：{detail}", flush=True)
            results.append(
                {
                    "name": name,
                    "base_url": redact_url(base_url, api_key=api_key),
                    "model": model,
                    "status": status,
                    "detail": redact_text(detail, api_key=api_key),
                    "capability": capability.model_dump(mode="json") if capability else None,
                }
            )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(item["status"] == "passed" for item in results)
    output.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[全部完成] 总数={len(results)}，通过={passed}，失败={len(results) - passed}")
    print(f"[完整结果] {output}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
