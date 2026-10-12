# Adult Income: ML and Deep Learning Improvement

Verified by a full CPU rerun on 12 October 2026. This is a post-coursework extension with AI assistance.

## Evaluation protocol

The original notebook compared configurations on its test split and then chose the best one. That test was therefore used for model selection. Historical scores remain in the original report; they are not treated as an untouched test or used to calculate the improvements below.

This extension uses the official UCI `adult.data` for training and validation and the separate `adult.test` for final evaluation. Exact feature duplicates are removed before splitting. Test duplicates and feature overlap with training are excluded. Labels are normalized, including the trailing period in the official test file; missing values are imputed from training.

Training: 26,028; validation: 6,508; final test: 16,249. See the split manifest for counts, raw hashes and split hash.

Ten configurations were evaluated on validation only: seven classical pipelines (including a prior-only dummy) and three PyTorch MLPs. Validation AP chooses one classical model and one MLP; validation positive-class F1 chooses their classification thresholds. The selection lock is written before parsing test labels. The selected models, two predeclared legacy controls, dummy and logistic baseline are evaluated on test.

The controls use StandardScaler without PCA. LightGBM retains SMOTE and class balancing; the MLP retains SMOTE, weighted cross entropy, BatchNorm, dropout 0.3 and 12 epochs. They reproduce representative recipes, not every original preprocessing search or the unseeded historical DL run.

## Measured results

- **dummy_prior**: accuracy 0.7637, AP 0.2363, ROC-AUC 0.5000, positive F1 0.0000, macro-F1 0.4330, log loss 0.5468.
- **logistic**: accuracy 0.8295, AP 0.7630, ROC-AUC 0.9055, positive F1 0.6854, macro-F1 0.7842, log loss 0.3176.
- **legacy_lightgbm**: accuracy 0.8429, AP 0.8213, ROC-AUC 0.9254, positive F1 0.7154, macro-F1 0.8035, log loss 0.3165.
- **legacy_mlp**: accuracy 0.8130, AP 0.7719, ROC-AUC 0.9089, positive F1 0.6822, macro-F1 0.7749, log loss 0.3733.
- **lightgbm_small_leaves**: accuracy 0.8673, AP 0.8261, ROC-AUC 0.9276, positive F1 0.7222, macro-F1 0.8175, log loss 0.2763.
- **mlp_128_64**: accuracy 0.8409, AP 0.7762, ROC-AUC 0.9103, positive F1 0.6936, macro-F1 0.7931, log loss 0.3109.

Against the matched-protocol controls, classical accuracy improved by 2.43 percentage points; MLP accuracy improved by 2.78 percentage points. These changes combine model/training changes and validation-selected thresholds. AP measures ranking independently of those thresholds.

![Measured test comparison](improved/comparison.png)

## What changed

- Classical ML: no synthetic one-hot interpolations or automatic double balancing in the new candidates; bounded LightGBM search with smaller leaves and regularization.
- DL: train-only preprocessing, unweighted binary cross entropy on natural class proportions, AdamW, reduced dropout, deterministic seeds, validation-AP early stopping and checkpoint restoration.
- Thresholds are selected on validation, with no test tuning. All preprocessing fits on training only.

## Stability and limits

The selected MLP was repeated with seeds 42, 43 and 44 without searching on test: accuracy 0.8409–0.8450; AP 0.7762–0.7773.

- lightgbm_small_leaves: paired row-bootstrap 95% AP-difference interval [0.00236, 0.00764] (500 draws).
- mlp_128_64: paired row-bootstrap 95% AP-difference interval [0.00128, 0.00732] (500 draws).

The MLP remains behind LightGBM. Small AP gains should be interpreted with their intervals. Row bootstrap assumes independent records; this dataset has no person identifiers to verify that assumption. The historical public dataset and one fixed validation split do not establish generalization to current populations. The official test has already been observed. This rerun freezes the existing configurations; it is not a pristine holdout. Exact feature duplicates include conflicting labels resolved by file order. See docs/verified_results.md for the audit.

## Reproduce and inspect

Run `python src/adult_benchmark.py`, then `python tools/build_delivery.py`. Use `notebooks/adult_income_improved.ipynb` for the readable walkthrough. The notebook explicitly states when it reuses a recorded run; set `REUSE_RECORDED_RUN = False` to retrain.

Evidence: `improved/metrics.json`, `split_manifest.json`, `selection_lock.json`, `training_curves.json`, `paired_bootstrap.json`, and the executed notebook.

Dataset: Becker, B. & Kohavi, R. (1996). [Adult, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). DOI: 10.24432/C5XW20; CC BY 4.0. Raw data and local prediction files are excluded from the repository.
