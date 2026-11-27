# Test 2 — Scientific Result Interpretation and Evidence Traceability

The benchmark contains the 60 report cases from the previous Test 2. The
questions, fixed reports, answer criteria, acceptable alternatives, Agent
inputs, and report evidence sent for judgment are preserved. The fixture
categories are single-report interpretation, multi-report integration,
multi-report comparison, multi-turn follow-up, and evidence localization.

The benchmark is self-contained under this directory and calls the Agent's
existing analysis and citation pipeline. Review a case in `cases/`. Its
mandatory facts and acceptable answers are in `case_catalog.yaml`, with
supporting atomic facts and field paths in `gold_report_facts.json`.

## Run

```bash
PYTHONPATH=src python benchmarks/test_2_result_interpretation/run_test_2.py --validate_only

PYTHONPATH=src python benchmarks/test_2_result_interpretation/run_test_2.py \
  --model gpt-6-luna --profile full --repeat 3 --bootstrap_iterations 1000
```

Set `ENZYWIZARD_API_KEY` for the model under test and `ANTHROPIC_API_KEY` for the
judge. Use `--case_ids` for a targeted run or `--profile quick` for a seeded
subset. Each case receives one structured request to Claude Haiku 5.5
(`claude-haiku-5-5`) per repetition. The program records judge token usage and
reports requests above Anthropic's 100,000-input-token pricing threshold.
Report summaries and evidence records are sent to the judge. Large raw JSON
reports remain in the fixture directory.

The default output root is
`~/enzywizard-benchmark-results/test_2_result_interpretation`. Each run records
the responses, case materials, original structured Judge replies, item-level decisions,
and metrics in the run directory. If a review fails, the run records the error
and available estimates with incomplete review coverage. An invalid citation
review preserves usable claim judgments and is recorded separately from a case
with no applicable citations. Check
`complete_reviews` and `scored_or_inapplicable_reviews` before reporting the
aggregate. The evaluator retries an invalid Judge output once on the same
answer. If an assessment is still missing, the interval uses available
assessments and retains the same sampled case IDs across repetitions. A draw
without an applicable score in any repetition is skipped. For a single
repetition, sample SD is `null`.

## Scoring

- **Claim F1.** Judge support for every distinct verifiable scientific claim and
  coverage for every predeclared required claim. An acceptance criterion with
  several valid alternatives counts once. A case with required claims and an
  empty answer contributes zero precision and zero recall. Cases without
  applicable required claims are omitted.
- **Citation F1.** For each written verifiable claim, match its distinct actual
  citations to the applicable reference evidence using stable report and
  metric identifiers. Equivalent locations form one evidence unit. When the
  question allows derived comparison evidence, its verified WT and mutant
  sources count as the same evidence units as direct citations to those
  sources. Duplicate citations to a unit count once. General limitations and
  non-scientific explanatory text are excluded from citation scoring. Missing citations
  give zero precision. A claim with no reference evidence has zero recall.
  Omitted claims affect Claim Recall. Cases with no written verifiable claims
  are omitted from Citation F1.

Citation precision and recall are averaged over written claims within each
case. For both metrics, case precision and recall are separately macro-averaged
within each repetition, then combined by harmonic mean. Report the mean and
sample SD across repetitions. Bootstrap draws sample case IDs with replacement
and retain all repetitions of each sampled case. The same procedure produces
one F1 per repetition and averages those scores in every draw. When a draw has
no applicable cases in any repetition, that draw is undefined for the metric.

The evaluation adapts Claim F1 and Citation F1 from
[WebCiteS](https://doi.org/10.18653/v1/2024.acl-long.806).

## Human blind review

After a run completes, it writes two indented JSON files in the run directory.
The sample is drawn by case ID with the run seed from the first repetition
only. A full run samples 12 of 60 cases. A shorter run uses the same 20% rate
rounded up.

- `human_review.json` groups the question, prior conversation when relevant,
  answer, required facts, evidence, and empty 0/1 judgments by case. Complete
  this file without opening the companion judge file. Add notes on missing or
  incorrect claim splits in `reviewer_notes`.
- `judge_review.json` stores the corresponding labels, reasons, and case
  scores. Both files share the same item IDs. Agreement measures labeling on
  these shared items. Both files identify unavailable reviews or unavailable
  citation judgments.

After completing the human labels, run:

```bash
python benchmarks/test_2_result_interpretation/review_agreement.py /path/to/run_directory
```

The script prints the item counts, agreement, and Cohen's kappa for claim
support, required-claim coverage, citation correctness, and reference-evidence
coverage, and saves `review_agreement.json`. Kappa is null when it is undefined.
