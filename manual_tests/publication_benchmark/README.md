# EnzyWizard-Agent publication benchmark

This directory contains the frozen, publication-oriented benchmark. It replaces the
earlier 100+50+25 development suites for formal evaluation; those suites remain useful
for regression testing but are not mixed into these scores.

## Fixed case set

- Test 1: 72 executable cases for scientific-goal translation and dynamic workflow
  construction: 66 ordinary or revised workflows and six workflows that reuse or
  interpret historical results before new computation.
- Test 2: 48 frozen-report cases for single-enzyme, WT/MUT, enzyme-substrate,
  supporting, conflicting, incomplete, and historical evidence scenarios. Every
  analysis subtype contains at least six independent questions.
- Test 3: 40 controlled cases covering missing startup inputs and ten execution-failure
  subtypes. Each broader failure family is represented by multiple independent faults.
- Total: 160 fixed cases. The loaders reject a wrong count or duplicate case ID.

Every experiment should freeze the code revision, case-file hashes, tool versions,
model settings, and input reports. The runners record the Git revision and SHA-256 of
their case file. The default repeat count is three.

## Programs

```text
run_publication_test_1_workflows.py
run_publication_test_2_analysis.py
run_publication_test_3_recovery.py
run_publication_test_4_mcp_placeholder.py
run_publication_test_5_fair_ablation.py
```

Test 4 deliberately returns `not_implemented`. A fair MCP comparison needs a frozen
platform protocol and export format; the placeholder cannot be reported as a score.

Test 5 runs the same selected Test 1-3 cases under `off`, `retrieve`, and `control`. Results are
paired by test, case ID, and repetition. The control condition injects matched-volume
irrelevant terminology through the Agent's existing control retrieval mode.

## Scoring

Test 1 reports five primary metrics for every formal case: tool-selection accuracy,
parameter correctness and validity, workflow-order validity, required-artifact
completeness, and end-to-end task success. Planning-only and representative execution
modes are development modes; required-artifact and end-to-end scores are only defined
for executed cases.

Test 2 reports factual precision, key-fact recall, integrated-interpretation score, and
evidence-attribution F1. The first three are produced by a blinded structured judge;
evidence-attribution F1 is calculated from validated finding-to-evidence links. Report
parsing and provenance-contract checks are retained as technical diagnostics rather than
additional scientific-quality metrics. The judge prompt does not disclose model or FAIR
condition. For a manuscript, use a fixed independent judge model and validate a
stratified subset with blinded domain experts.

Test 3 invokes the Agent's real input-validation and hybrid failure-recovery paths
against controlled faults. It reports strict diagnosis accuracy, recovery-action
accuracy, recovery success rate, and error-recovery trace completeness. Deterministic
safety policy remains authoritative; the configured model supplies bounded diagnosis
and action advice only where the policy does not already authorize a same-call retry.

### Metric definitions

| Test | Metric | Per-run calculation |
|---|---|---|
| 1 | Tool-selection accuracy | 1 when the selected tool set exactly matches an accepted expert plan and contains no forbidden tool; otherwise 0. |
| 1 | Parameter correctness and validity | 1 when the plan satisfies its parameter contract and every explicitly requested value; otherwise 0. |
| 1 | Workflow-order validity | 1 when step indexes, upstream references, staging dependencies, and expert order constraints are valid; otherwise 0. |
| 1 | Required-artifact completeness | Registered required artifacts divided by declared required artifacts. |
| 1 | End-to-end task success | 1 when planning is correct, execution follows the approved plan, all calls succeed, and artifact completeness is 1; otherwise 0. |
| 2 | Factual precision | Supported candidate scientific claims divided by all candidate scientific claims, assigned by the blinded judge. |
| 2 | Key-fact recall | Expert-required facts recovered divided by all expert-required facts, assigned by the blinded judge. |
| 2 | Integrated-interpretation score | Continuous blinded-judge score for correct cross-report synthesis, scientific boundaries, and limitation handling. |
| 2 | Evidence-attribution F1 | Harmonic mean of valid evidence-reference precision and evidence-linked substantive-finding recall. |
| 3 | Strict diagnosis accuracy | 1 only when category, failure stage, recoverability, and scientific-change boundary all match the expert oracle. |
| 3 | Recovery-action accuracy | 1 only when the policy selects an accepted decision/action, obeys confirmation boundaries, and uses the expected retry count. |
| 3 | Recovery success rate | 1 when the controlled case reaches its expected safe terminal outcome; otherwise 0. |
| 3 | Error-recovery trace completeness | Present mandatory diagnosis, decision, retry, and AI-assessment trace fields divided by applicable mandatory fields. |

All scalar metrics include mean, sample standard deviation, and observation count.
JSON contains complete traces; CSV contains flat analysis rows.

## Profiles

All three runners support a reproducible stratified development profile and the fixed
formal profile:

- `--profile quick`: draw a subtype-balanced sample. Omit `--seed` to draw a new
  sample on each run; the resolved seed is stored in result metadata. Use
  `--quick_case_count N` for an exact total or `--sample_per_subtype N` for the
  legacy per-subtype size. Pass the recorded `--seed` to reproduce a sample.
- `--profile full`: run every frozen case. Use this profile for reported results.
- `--case_ids ...`: override profile sampling with an explicit debugging selection.

## Example

```bash
export ENZYWIZARD_API_KEY="YOUR_API_KEY"
PYTHONPATH=/path/to/enzywizard-agent/src:/path/to/enzywizard-agent/manual_tests \
python /path/to/enzywizard-agent/manual_tests/run_publication_test_5_fair_ablation.py \
  --base_url https://api.openai.com/v1 \
  --model gpt-6-luna \
  --reasoning_effort medium \
  --repeat 3 \
  --profile full \
  --judge_model gpt-6-luna \
  --judge_reasoning_effort medium \
  --workdir ~/enzywizard-publication-fair-ablation
```

Use `--execute_representative` for low-cost execution debugging. Use `--execute_all`
with `--profile full` for the formal Test 1 end-to-end benchmark. Both options require the
real tool environment and sufficient runtime budget. Without either option, Test 1 is a
planning-only development run.
