# Adult Income audit before changes

Audited on 12 October 2026, before implementation changes. Source checkout:
`9137c2a`, branch `main`, remote `VietAnh1803/HCMUT_ML_AdultCensusIncome_Pipeline`.
The portfolio checkout is `../portfolio`; the editable one-page CV is
`../cv.eris-draft.tex`, exported as `../output/pdf/nguyenvietanh_resume.pdf`.
No user changes were present in the Adult checkout. No publication is authorized.

## Evidence available

- `src/adult_benchmark.py` implements the nine validation candidates, separate
  official test evaluation, and MLP seeds 42, 43 and 44.
- `reports/improved/{metrics,split_manifest,selection_lock,training_curves}.json`
  support the reported 86.73% LightGBM accuracy, 0.9276 ROC-AUC and MLP
  84.09–84.50% range. These are recorded results, pending an independent rerun.
- The existing five protocol tests pass. A full unchanged run is in progress in
  `.cache/audit-baseline`; this is not a smoke run.
- The cached UCI archive matches a fresh download byte for byte. Official data:
  32,561 training rows, 16,281 test rows, 14 features, target income above $50K.
  UCI attributes the data to Becker/Kohavi and licenses it under CC BY 4.0.
- Training-only imputation, scaling and one-hot encoding are implemented.
  Model selection uses validation AP; non-control thresholds use validation
  positive-class F1. The lock is written before parsing test labels.

## Findings

1. The original coursework selected configurations using test scores. Its scores
   are not independent final-test evidence. The later extension also has already
   reported its official-test results; rerunning it cannot make that public test
   pristine again. Freeze configurations and disclose this limitation.
2. Deduplication silently retains the first feature row. Official training has
   25 duplicate feature rows but only 24 duplicate labeled observations, with one
   conflicting-label feature group. Official test has 6 versus 5, also one
   conflicting group. There are 26 test feature matches with official training.
   The current filtered test has 16,249 rows. Feature equality is not proof that
   two records refer to the same person. Preserve the recorded policy for
   comparability, count these cases, and document its tradeoff.
3. The loader does not explicitly reject malformed column counts or invalid
   numerical features. Scoring does not validate shape, finite probabilities,
   probability bounds, binary targets, or thresholds.
4. Precision/recall, per-seed aggregate statistics, model configurations, archive
   checksum validation and repeatable prediction artifacts are incomplete.
5. Logistic regression already provides a useful baseline. A prior-only dummy
   model would expose the imbalance without adding another tuned model.
6. LightGBM uses fixed estimator counts, not early stopping. PyTorch uses CPU,
   AdamW, deterministic seeds, validation-AP early stopping and checkpoint restore.
   Do not claim LightGBM early stopping or GPU training.
7. `tools/build_delivery.py` runs on import, hardcodes a report date, and embeds a
   Colab-specific path. Its executed notebook reads saved results by default;
   execution is not evidence of retraining.
8. There is no independent submission evaluator. Add a small explicitly local,
   public-label demonstration using the existing validation partition. It must
   not reuse final-test labels for development or imply hidden-test security.

## Planned bounded changes

Keep the existing model search, deduplication policy and architecture. Add input
validation, explicit dataset-integrity outputs, scoring tests and a dummy baseline.
Implement a small ID-matched probability scorer and logistic reference solution;
then rerun the full frozen experiment, generate evidence, and revise documentation.
Only then update the CV and portfolio. Formal education status and employment have
been clarified by the user: program completed, graduation ceremony November 2026;
Data Engineer at OSAS via AI-MED since June 2025.
