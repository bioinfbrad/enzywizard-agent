# Comprehensive Agent Testing

The test project is split into layers so a failure can be assigned to request
routing, Planning, confirmation, execution, Analysis, or cross-turn state.

## Coverage

### 1. Deterministic contracts

No external model or real EnzyWizard command is used.

- strict Request Router and History Query schemas
- workflow-scope extraction contracts and deterministic analysis-type-to-tool compilation
- graph routes and interrupt/resume contracts
- confirmation approval, revision, and rejection
- safe subprocess command construction
- output reference resolution and required artifact validation
- tool failure capture and log-tail preservation
- real subprocess timeout termination and partial-artifact preservation
- centralized OpenAI-compatible model factory and strict JSON Schema capability contract
- LLM call, token, session-time, and tool-timeout budgets
- privacy-safe session manifests with versions, parameters, and file checksums
- report parsing, evidence generation, WT/MUT comparisons, cross-report consistency,
  and state reset
- historical clean, substrate, Dock, and report reuse compatibility

### 2. Live request semantics

`request_routing_cases.json` contains 100 cases evaluated with `qwen3.7-plus`:

- new single-protein and WT/MUT workflows
- explicit and relative historical report analysis
- artifact, path, graph, and log queries
- required and optional historical output reuse
- mixed history, execution, help, and listing actions
- software help and conceptual scientific questions
- conversation, out-of-scope, clarification, and exit requests
- recomputation and prior-approved-plan references
- structured History Query turn, side, operation, and artifact selection
- Analysis response mode, termination route, and English-only final answers
- 50 holdout/paraphrase cases for current-input versus historical-result boundaries,
  plan-context versus artifact reuse, exact report ontology, and multi-action requests

This layer creates a controlled three-turn artifact registry and does not execute
scientific tools.

### 3. Live Planning

`planning_cases.json` contains 50 cases covering:

- all 17 registered EnzyWizard commands
- single-protein and WT/MUT plans
- individual and batch workflows
- dependency insertion and call order
- profiles and explicit parameters
- missing inputs and unsupported overrides

### 4. Workflow integration

`full_workflow_cases.json` contains 25 workflows. They now enter through the
Request Router and cover Planning, confirmation, repeated revisions, rejection,
Tool Calling, output discovery, staging, and artifact registration.

Without `--execute_tools`, final approved plans are validated without running
scientific tools. With `--execute_tools`, all 25 workflows execute real commands.

### 5. Analysis and cross-turn regression

`run_analysis_module_tests.py` reads a successful real-workflow run and verifies
all 15 report types, evidence compression, WT/MUT calculations, failure guidance,
cross-report consistency warnings, historical queries, reuse compatibility, graph
resume, and session termination.

### 6. Multi-service model compatibility

`run_model_compatibility_tests.py` sends the same minimal strict JSON Schema request
to at least two user-supplied OpenAI-compatible endpoints. It records model names and
capability results, but never writes API keys to the result file.

## Recommended commands

Set the key without placing it in shell history:

```bash
export ENZYWIZARD_API_KEY='your-key'
```

Run all deterministic and live-model tests, but do not execute scientific tools:

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u manual_tests/run_comprehensive_agent_tests.py \
  --model qwen3.7-plus
```

Run only the 100 Router, History Query, and Analysis semantic cases:

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u manual_tests/run_request_routing_batch.py \
  --model qwen3.7-plus \
  --include_analysis
```

Run the complete suite with all 25 real scientific workflows:

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u manual_tests/run_comprehensive_agent_tests.py \
  --model qwen3.7-plus \
  --execute_tools
```

Check two real OpenAI-compatible services with the same model contract:

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u manual_tests/run_model_compatibility_tests.py \
  --endpoint qwen https://example-qwen.invalid/v1 qwen-model QWEN_KEY \
  --endpoint gemini https://generativelanguage.googleapis.com/v1beta/openai/ gemini-model GEMINI_KEY \
  --output ~/enzywizard-model-compatibility-results.json
```

Provider-specific request fields are never inferred. Pass them explicitly with
`--llm_extra_body_json` only when every endpoint in that compatibility run supports
the same extension.

Before a comprehensive run starts, the master runner verifies that the 25 workflow
case expectations cover every command in the tool registry. With `--execute_tools`,
the master runner automatically passes the generated
workflow directory to the deterministic Analysis and cross-turn regression. No
second command or manually supplied run directory is required.

For model stability measurement, add `--repeat_semantic 3`. This repeats the
100 routing cases and 50 Planning cases but does not repeat real tool execution.

Every runner writes results incrementally. A failed case or suite is recorded and
the remaining work continues unless an explicit stop-on-failure option is used.
An approved plan that fails its semantic expectations is not executed with real
scientific tools. The dependent Analysis regression is marked blocked when the real
workflow suite is incomplete, because partial workflow artifacts are not a valid
regression fixture.
