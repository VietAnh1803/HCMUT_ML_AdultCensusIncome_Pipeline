# Verified Adult Income results

Run UTC: `2026-10-12T07:53:58.423419+00:00`. Python 3.12, CPU, four threads.

## Status and scope

A full unchanged-source rerun reproduced all prior headline scores. A full revised-source run reproduced the same accuracy, AP, ROC-AUC, F1 and thresholds. The changes improve engineering/evaluation checks; they do not claim new model-performance gains. No configuration was chosen after reviewing final-test scores in this audit.

## Dataset

[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), Becker/Kohavi (1996), 1994 census extraction, DOI 10.24432/C5XW20, CC BY 4.0. Target: income >$50K. Fourteen features: six numerical and eight categorical.

Official adult.data: 32,561 rows; adult.test: 16,281. After the preserved policy: 26,028 training, 6,508 validation, 16,249 final-test rows. Stratified 80/20 split of official training, seed 42. Final test contains 12,410 negative and 3,839 positive rows.

Keep the first raw feature occurrence before splitting; exclude test feature matches with official training. Equal features are not proof of duplicate people. One conflicting-label group exists in each official partition and is resolved by file order. This preserves comparability but loses observation weighting and one label in each conflicting group. The unfiltered official partitions are retained in the cache.

- official_train: `{"rows": 32561, "features": 14, "duplicate_feature_rows": 25, "duplicate_observation_rows": 24, "conflicting_label_feature_groups": 1, "positive_rows": 7841, "missing_by_column": {"workclass": 1836, "occupation": 1843, "native-country": 583}}`
- official_test: `{"rows": 16281, "features": 14, "duplicate_feature_rows": 6, "duplicate_observation_rows": 5, "conflicting_label_feature_groups": 1, "positive_rows": 3846, "missing_by_column": {"workclass": 963, "occupation": 966, "native-country": 274}}`
- cross_partition: `{"matching_feature_pairs": 28, "unique_matching_feature_tuples": 26, "same_label_pairs": 25, "different_label_pairs": 3, "conflicting_feature_tuples": 3, "interpretation": "Exact feature equality; no person identifiers establish duplicate individuals."}`
- Removed test feature matches: 26.
- Archive SHA-256: `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`; fresh UCI download matched.
- Split ID SHA-256: `8b65a245597770268e00fc818f7705fd6b958b02679fd5cfe630bef94fb425e5`.
- adult.data SHA-256: `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`
- adult.test SHA-256: `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05`

## Methodology

Imputation/scaling/one-hot encoding fit on training only. Unseen categories are ignored; categorical missing values use Missing, numerical imputation uses the training median. Ten validation configurations include seven classical pipelines and three MLPs. The dummy and legacy controls are excluded from model selection. Validation AP selects the non-control classical model and MLP; validation positive F1 chooses their thresholds. Dummy/legacy thresholds are 0.5. The lock precedes test-label parsing; predeclared dummy and logistic baselines are additionally graded on test.

LightGBM uses fixed estimator counts, not early stopping. The selected MLP has 128/64 ReLU layers, dropout 0.15, BCE, AdamW (lr 0.001, weight decay 0.01), batch size 256, maximum 40 epochs, validation-AP patience 7 and best-checkpoint restoration. Seeds 42/43/44 are all reported; no test-based seed selection. Exact classical configs and legacy recipes are stored in metrics.json.

## Filtered official-test results

| Model | Accuracy | AP | ROC-AUC | Positive F1 | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| dummy_prior | 0.7637 | 0.2363 | 0.5000 | 0.0000 | 0.5000 |
| logistic | 0.8295 | 0.7630 | 0.9055 | 0.6854 | 0.3023 |
| legacy_lightgbm | 0.8429 | 0.8213 | 0.9254 | 0.7154 | 0.5000 |
| legacy_mlp | 0.8130 | 0.7719 | 0.9089 | 0.6822 | 0.5000 |
| lightgbm_small_leaves | 0.8673 | 0.8261 | 0.9276 | 0.7222 | 0.4116 |
| mlp_128_64 | 0.8409 | 0.7762 | 0.9103 | 0.6936 | 0.3121 |

AP is sklearn average precision, not trapezoidal PR-AUC. Positive precision/recall, confusion counts and class distribution are in metrics.json.

## MLP seeds

| Seed | Accuracy | AP | ROC-AUC | Epoch |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.8409 | 0.7762 | 0.9103 | 8 |
| 43 | 0.8450 | 0.7773 | 0.9109 | 8 |
| 44 | 0.8427 | 0.7763 | 0.9103 | 7 |

Three-seed aggregates (mean and sample standard deviation): `{"accuracy": {"mean": 0.8428416107657907, "sample_std": 0.0020654126368731696}, "average_precision": {"mean": 0.7765757710460237, "sample_std": 0.0006105037134422992}, "roc_auc": {"mean": 0.910511665584638, "sample_std": 0.00034233592186275734}}`.

## Old, replaced and unverified results

- legacy_lightgbm: old/new accuracy 0.842944, AP 0.821314, ROC-AUC 0.925387, unchanged. Log-loss change 0 reflects float64 probability validation.
- legacy_mlp: old/new accuracy 0.813035, AP 0.771856, ROC-AUC 0.908852, unchanged. Log-loss change 9.97e-11 reflects float64 probability validation.
- lightgbm_small_leaves: old/new accuracy 0.867253, AP 0.826091, ROC-AUC 0.927633, unchanged. Log-loss change 0 reflects float64 probability validation.
- mlp_128_64: old/new accuracy 0.840852, AP 0.776163, ROC-AUC 0.910312, unchanged. Log-loss change -6.58e-10 reflects float64 probability validation.
- Coursework test-selected historical scores are retained in the original report; not treated as an untouched test.
- New dummy/logistic results are measured baselines, not replacements for historical results.
- No improvement over the October 11 implementation is claimed. Matched-control accuracy deltas include threshold changes.

## Limits and local demonstration

The coursework and extension have already exposed test results. This is a frozen reproducibility check, not a pristine holdout. The historical census sample does not establish present-day generalization, fairness or suitability for decisions about people. One validation split and three seeds offer limited robustness evidence. Row-bootstrap intervals assume independent records; no person IDs establish that assumption.

The local scorer uses the original 6,508-row public validation partition and never reads adult.test. Its logistic reference AP is 0.782419 at a fixed threshold of 0.5. Labels remain public and previously used for selection. This is not a hidden challenge, an Eris submission or SHiP-D-compatible infrastructure. See docs/local_challenge.md and reports/verification/local_reference_score.json.

## Executed verification commands

```bash
python -m unittest discover -s tests -v
python tools/run_tests.py
python src/adult_benchmark.py --output .cache/audit-baseline --cache .cache/adult  # original source only
python src/adult_benchmark.py --output .cache/fitting-verified-run --cache .cache/adult
python src/local_evaluation.py prepare
python src/local_evaluation.py reference --train .cache/local-evaluation/train.csv --features .cache/local-evaluation/evaluation_features.csv --output .cache/local-evaluation/reference_predictions.csv
python src/local_evaluation.py score --labels .cache/local-evaluation/evaluation_labels.csv --predictions .cache/local-evaluation/reference_predictions.csv --output .cache/local-evaluation/reference_score.json
python tools/build_delivery.py --input .cache/fitting-verified-run --baseline .cache/audit-baseline/metrics.json
```

The delivery tool copies completed outputs to reports/improved, produces actual plots and executes a five-cell recorded-results walkthrough. That execution is not a retraining run. Test, experiment and CLI execution results are recorded in reports/verification/verification.json.

## Automated tests and fitting diagnostics

All 50 automated tests pass, including actual early-stopping/checkpoint-weight checks and local CLI integration cases. See test_coverage.md and reports/verification/test_results.json for the separately executed cases.

The unchanged full run also records training AP/log-loss for every validation candidate. Selected LightGBM train/validation AP is 0.8607/0.8432; selected MLP AP is 0.8031/0.8006. Seed 42 stops at epoch 15 and restores epoch 8. Dropout, weight decay and validation early stopping mitigate overfitting; these gaps and baseline comparisons do not rule out underfitting. See fitting_diagnostics.md for the complete measured evidence.
