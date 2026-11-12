# Test 2: Result Interpretation and Evidence Provenance

This benchmark contains 68 English-language cases:

- 12 single-report interpretations;
- 12 multi-report integrations;
- 12 multi-report comparisons;
- 12 multi-turn explanation follow-ups and revisions;
- 12 report evidence localization and provenance cases;
- 8 boundary or unrelated analysis requests.

Each of the first five categories assigns four distinct tasks to each representative
enzyme system: 1MXT/T244M, 1IU4/S72N, and 1TCA/G107A. The JSON fixtures are fixed
reports produced by EnzyWizard tools. Cases reference these immutable reports through
`case_catalog.yaml`; WT/MUT side and conversational-turn metadata are assigned by the
runner rather than encoded by modifying report content.

The runner uses fixed reports and does not execute EnzyWizard scientific tools. The
Agent analyzes the reports, after which the four declared metrics are calculated:

- factual precision (FP);
- key-fact recall (KFR);
- integrated interpretation score (IIS);
- evidence-attribution F1 (EA-F1).

FP, KFR, and EA-F1 apply only to the 60 report-analysis cases. The eight boundary or
unrelated requests have no supplied report evidence or predefined report facts, so
those three metrics are recorded as N/A and excluded from their aggregate denominators.
Boundary cases are evaluated only with their category-specific IIS dimensions.

FP, KFR, and EA-F1 are evaluated automatically against the source fact pool in
`gold_report_facts.json`. It contains 289 representative atomic facts from the 45 fixed
reports, including report identity, metric name, field path, expected value, numeric
tolerance, inference level, and accepted aliases. Each case selects three to six
question-relevant key facts: three for one report, up to three per report for two
reports, or up to two per report for three reports. Paired mutation cases also include
the recorded substitution. Comparable WT/MUT report pairs include one explicit
directional comparison derived from, and linked to, both source reports. Validation
checks source existence, case-level fact count, report coverage, identifier uniqueness,
side, field path, evidence derivation, and value agreement. It does not use a second
language model.

IIS is the only subjective metric. By default, a human reviewer assigns 0/1/2 scores
using the fixed rubric. Optional `llm` mode supplies the exact same evaluation packet
and rubric to a structured IIS reviewer; it cannot alter FP, KFR, or EA-F1.

The IIS review packet deliberately separates narrative requirements from the atomic
gold facts used by the automatic metrics. Case-specific `must_address` items and
explicit limitations are requirements; the remaining report metrics are available
evidence rather than a checklist. Reviewers score relevance and scientific quality,
not verbosity, and do not penalize omission of optional fields.

Two metric-evaluation modes are available:

- `--evaluation_mode hybrid` (default): FP, KFR, and EA-F1 use deterministic structured
  gold-standard checks; only IIS is reviewed by `--reviewer human|llm`.
- `--evaluation_mode full_review`: the selected reviewer makes item-level judgments for
  every verifiable atomic fact, predefined key fact, generated claim-evidence link,
  required-fact evidence link, and IIS dimension. The program then applies the same
  published FP, KFR, EA-F1, and IIS formulas; the reviewer never supplies final metric
  percentages directly.

In full review, an evidence association is one atomic conclusion linked to one distinct
evidence source or field. A conclusion citing two independent sources contributes two
associations, as do two conclusions supported by one source. Identical links repeated in
the narrative and evidence-summary block are counted once. Merely listing a report
without linking it to a conclusion does not create an association.

The automatic metrics intentionally evaluate the Agent's structured scientific output,
while IIS evaluates the quality of the user-facing narrative. The metric formulas are
unchanged; only the source of the item-level judgments differs from the earlier
all-human/LLM-judge implementation.

Run the static fixture audit with:

```bash
PYTHONPATH=src python evaluations/test_2_result_interpretation/run_test_2.py \
  --validate_only
```

The audit performs no LLM calls and does not execute scientific tools.

Run a quick interactive human evaluation with:

```bash
PYTHONPATH=src python evaluations/test_2_result_interpretation/run_test_2.py \
  --model gpt-6-luna \
  --reasoning_effort medium \
  --ontology_mode retrieve \
  --iis_evaluator human \
  --profile quick \
  --quick_case_count 8 \
  --repeat 1 \
  --bootstrap_iterations 0
```

Use an LLM to simulate only the human IIS rubric with:

```bash
  --iis_evaluator llm \
  --judge_api_key JUDGE_API_KEY \
  --judge_model MODEL_NAME \
  --judge_base_url BASE_URL \
  --judge_reasoning_effort REASONING_EFFORT
```

Run a complete item-level LLM review of all four metrics with:

```bash
PYTHONPATH=src python evaluations/test_2_result_interpretation/run_test_2.py \
  --evaluation_mode full_review \
  --reviewer llm \
  --judge_model gpt-6-luna \
  --profile full \
  --repeat 3
```

Replace `--reviewer llm` with `--reviewer human` for a fully interactive human
review. Human full review asks for the atomic factual claims and generated
claim-evidence links, then presents every predefined key fact and IIS dimension for an
item-level decision. This mode is intentionally substantially slower than `hybrid`.

The IIS reviewer defaults to the fixed `gpt-6-luna` model and the OpenAI API endpoint,
independently of the model under test. `--judge_api_key` or the
`ENZYWIZARD_JUDGE_API_KEY` environment variable can provide a separate credential; if
neither is set, the tested model's API key is reused. Formal human IIS scoring remains
the default. Test 2 uses a 180-second per-request timeout for both the tested model and
the reviewer by default because full item-level review can be substantially longer than
ordinary Agent responses; both limits can be changed independently with
`--llm_timeout_seconds` and `--judge_timeout_seconds`.

Every completed case is written immediately to `results.json`; detailed analysis,
evaluation-packet, and review records are retained under each case/repetition directory.
The summary reports every repetition, the mean and standard deviation across
repetitions, category-level metrics, and optional case-cluster bootstrap confidence
intervals.
