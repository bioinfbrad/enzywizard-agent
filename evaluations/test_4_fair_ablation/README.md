# Test 4: FAIR-Aligned Terminology Metadata Ablation

Test 4 reuses the 60 scientific interpretation cases from Test 2 and excludes
the eight boundary or unrelated requests. It evaluates the same four Test 2
metrics (`FP`, `KFR`, `IIS`, and `EA_F1`) plus terminology semantic
interpretation accuracy (`TSIA`).

## Conditions

Run the same cases under exactly one of these conditions:

- `--ontology_mode off`: no terminology metadata retrieval.
- `--ontology_mode retrieve`: retrieve relevant FAIR-aligned terminology metadata.
- `--ontology_mode control`: inject the same number of deterministic but irrelevant
  terminology records.

For a paired ablation, keep the model, model settings, Judge, case IDs, seed,
repeat count, reports, and bootstrap settings identical. Only `ontology_mode`
and the output directory may differ.

## TSIA

Each case receives source-neutral terminology criteria derived from its fixed
report types and the actual WT/MUT report sides. Reviewers assign credit only
when the final answer makes an identifiable, semantically correct statement
about the concept and avoids every predefined forbidden interpretation. A
missing concept, label-only mention, semantic error, or forbidden interpretation
is scored false. Exact labels, ontology identifiers, source citations, and
explicit definitions are neither required nor rewarded.

The case score is the number of correctly understood applicable terms divided
by the number of applicable terms. Overall TSIA is the macro-average across
cases. The same gold criteria and review prompt are used in all three
conditions.

The Agent's saved answer retains its `Terminology sources` appendix. The human
or LLM Judge receives the scientific answer body with that appendix removed, so
all three conditions have the same review format and source labels cannot reveal
the condition.

## Ablation audit

Each case saves `retrieval_audit.json` containing the mode, injected terms and
URIs, relevant-term count, injected character count, token estimate, and FAIR
knowledge-base SHA-256. A deterministic preflight rejects a retrieve run when a
case cannot retrieve any corresponding gold concept. The condition comparator
also rejects unmatched cases or settings, off-condition injection, relevant
control terms, unequal control/retrieve injection counts, and knowledge-base
hash mismatches.

## Validation

```bash
PYTHONPATH=src \
/home/wmk/miniconda3/envs/enzywizard-agent/bin/python -B \
evaluations/test_4_fair_ablation/run_test_4.py --validate_only
```

## Example

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
python -u evaluations/test_4_fair_ablation/run_test_4.py \
  --base_url https://api.openai.com/v1 \
  --model gpt-6-luna \
  --reasoning_effort medium \
  --ontology_mode retrieve \
  --profile full \
  --repeat 3 \
  --evaluation_mode full_review \
  --iis_evaluator llm \
  --judge_model gpt-6-luna \
  --bootstrap_iterations 5000 \
  --workdir evaluations/test_4_fair_ablation/results/gpt-6-luna-retrieve
```

After all three matched runs finish, compare them with:

```bash
python evaluations/test_4_fair_ablation/compare_conditions.py \
  --off PATH_TO_OFF_RESULTS.json \
  --control PATH_TO_CONTROL_RESULTS.json \
  --retrieve PATH_TO_RETRIEVE_RESULTS.json \
  --bootstrap_iterations 5000 \
  --output evaluations/test_4_fair_ablation/results/comparison.json
```
