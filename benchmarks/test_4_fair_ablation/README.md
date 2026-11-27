# Test 4 — Domain Concept Definition Evaluation

This standalone benchmark reuses the 60 fixed report tasks and 45 reports
from the rebuilt interpretation benchmark. Its local `cases/` and
`resources/` directories contain the inputs. Every case fixes one ontology
term, its reference definition, and the English follow-up question.

Each repetition starts a fresh Agent session. It runs the original report
task, including the original second turn where present, then sends
“What is the definition of ‘TERM’?” through the Agent's normal routing and
answer nodes. The original turns are archived in the same session. The
specified `--ontology_mode` applies throughout the conversation.

## Scoring

Concept Definition Accuracy (CDA) is the number of correct final definitions
divided by the number of applicable cases. Claude Haiku 5.5 gives a binary
judgment for the final definition. Equivalent wording earns full credit
when it conveys the reference's complete core meaning without a substantive
error. A missing definition scores zero. The evaluator retries an invalid Judge
decision once on the same answer. If a review is still unavailable, the run
reports CDA over scored answers and marks review coverage as incomplete. The
interval samples the same case IDs across repetitions and skips draws with no
scored answer in any repetition. The three-condition comparison requires
complete paired scores and valid retrieval audits. Original report answers
supply conversation history.

The task follows contextual definition generation and semantic answer
equivalence. [Generationary](https://doi.org/10.18653/v1/2020.emnlp-main.585)
and [Answer Equivalence](https://doi.org/10.18653/v1/2022.emnlp-main.20).

Report CDA per repetition, then its mean and sample standard deviation.
The 95% percentile bootstrap samples case IDs with replacement and retains
all repetitions of each sampled case. The condition comparison uses the
same case clusters for both CDA differences.

## Run

Set `ENZYWIZARD_API_KEY` for the Agent model and `ANTHROPIC_API_KEY` for
Claude. Run from the Agent checkout.

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u benchmarks/test_4_fair_ablation/run_test_4.py --validate_only

PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u benchmarks/test_4_fair_ablation/run_test_4.py \
  --model gpt-6-luna --ontology_mode retrieve --profile full --repeat 3 \
  --bootstrap_iterations 10000 \
  --workdir "$HOME/enzywizard_benchmark_results/test_4_fair_ablation/gpt-6-luna/retrieve"
```

Use `--case_ids` for selected cases. Run `off`, `control`, and `retrieve`
separately with matching cases, model settings, Judge and repetition count.
Compare the three finished runs with:

```bash
python benchmarks/test_4_fair_ablation/compare_conditions.py \
  --off /path/to/off/results.json --control /path/to/control/results.json \
  --retrieve /path/to/retrieve/results.json --bootstrap_iterations 10000
```

The comparator reports CDA and its paired differences. Audit metadata records
matched terms and source hashes by turn, target injection coverage, and the
control count relative to retrieval for that condition's input. Comparison
rejects a control run if a target term was injected or its count is invalid.
A missed retrieval stays in the evaluation denominator.

## Blind review

Each run writes `results.json`, `results.csv`, per-case conversation and
retrieval records, and `human_review.json` and `judge_review.json`. The
blind sample uses 20% of first-repetition case IDs, rounded up. Equal
seeds and selected IDs yield the same sample across conditions. Give
the expert only the human file. Fill in each `score` with 0 or 1.

```bash
python benchmarks/test_4_fair_ablation/review_agreement.py /path/to/run_directory
```

The command writes exact agreement and Cohen's κ to `review_agreement.json`.
