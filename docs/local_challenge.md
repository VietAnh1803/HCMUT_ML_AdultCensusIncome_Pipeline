# Adult Income probability scoring: a local demonstration

This component was added in October 2026 after the original coursework. It is a
small personal demonstration of problem formulation and automated evaluation.
It is not a Project Eris submission and has no SHiP-D infrastructure integration.

## Task

Predict the probability that a record's annual income exceeds $50,000. Use the
14 documented UCI Adult features: age, workclass, fnlwgt, education,
education-num, marital-status, occupation, relationship, race, sex,
capital-gain, capital-loss, hours-per-week and native-country.

`prepare` reads only `adult.data`, applies the experiment's documented
first-occurrence feature policy and stratified seed-42 80/20 split, then writes:

- `train.csv`: row_id, the 14 features in the order above, and binary target.
- `evaluation_features.csv`: row_id and the 14 features, without target.
- `evaluation_labels.csv`: row_id,target, kept separate but publicly recoverable.
- `sample_submission.csv`: row_id,probability, with a constant 0.5 example.
- `manifest.json`: split counts and the explicit public-label warning.

There are 26,028 permitted training records and 6,508 evaluation records. Fit
preprocessing and models on `train.csv` only. Do not use evaluation labels to
produce predictions or change the scoring metric. A Python 3.12 CPU reference
uses logistic regression with training-only preprocessing and seed 42.

The evaluation partition is the same validation data used by the original ML
experiments. Its labels are public; this is an evaluator demonstration, not a
blind competition or a new independent holdout. `adult.test` is never read by
the demo. There is no private-label protection or server sandbox.

## Submission and scoring contract

Submit UTF-8 CSV with exactly these columns in this order:

```csv
row_id,probability
validation-42,0.73
```

The example ID illustrates the format, not a required first row. Every supplied
ID must match the prepared feature file exactly. File order is irrelevant.
The scorer rejects missing, duplicate, blank, null or unexpected IDs, extra or
missing columns, wrong row counts, nonnumeric probabilities, NaN, infinity, and
probabilities outside [0,1]. It also validates both binary ground-truth classes.
Invalid CLI inputs return exit code 2 and an actionable error, without writing a
new score file. Use a fresh output path so an old successful score cannot be
mistaken for a rejected submission.

Primary metric: **average precision**, higher is better, using probability
ranking. Income above $50K is the minority class; AP is more informative than
majority-class accuracy for this task. It is not trapezoidal PR-AUC. Secondary
metrics are ROC-AUC, log loss and threshold-0.5 accuracy/precision/recall/F1 and
confusion matrix. The threshold is fixed by the contract; changing it does not
change AP. This is one scoring contract, not a weighted multi-metric rubric.

## Run

From the repository root, after installing `requirements-improved.txt`:

```bash
python src/local_evaluation.py prepare
python src/local_evaluation.py reference --train .cache/local-evaluation/train.csv --features .cache/local-evaluation/evaluation_features.csv --output .cache/local-evaluation/reference_predictions.csv
python src/local_evaluation.py score --labels .cache/local-evaluation/evaluation_labels.csv --predictions .cache/local-evaluation/reference_predictions.csv --output .cache/local-evaluation/reference_score.json
python -m unittest discover -s tests -v
python tools/run_tests.py
```

The reference command never opens `evaluation_labels.csv`. The scorer joins
predictions to labels by row_id and writes strict machine-readable JSON.
Generated CSVs stay in ignored `.cache/`; dataset attribution and license are
documented in the README. The recorded demo score is in
`reports/verification/local_reference_score.json`, distinct from official-test
model results.
