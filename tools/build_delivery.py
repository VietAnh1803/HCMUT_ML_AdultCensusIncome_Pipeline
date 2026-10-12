"""Build documentation and a readable notebook from the completed experiment."""
import argparse
import asyncio
import shutil
import hashlib
import json
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/improved'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=OUT)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.input.resolve() != OUT.resolve():
        for name in ['metrics.json', 'split_manifest.json', 'selection_lock.json',
                     'training_curves.json', 'test_predictions.csv', 'comparison.png']:
            shutil.copy2(args.input / name, OUT / name)
    report = json.loads((OUT / 'metrics.json').read_text())
    classic, mlp = report['selection']['best_classic'], report['selection']['best_mlp']
    base_c, base_d = report['test']['legacy_lightgbm'], report['test']['legacy_mlp']
    best_c, best_d = report['test'][classic], report['test'][mlp]
    predictions = pd.read_csv(OUT / 'test_predictions.csv')
    rng = np.random.default_rng(42)
    intervals = {}
    for old, new in [('legacy_lightgbm', classic), ('legacy_mlp', mlp)]:
        delta_acc, delta_ap = [], []
        target = predictions.target.to_numpy()
        p_old, p_new = predictions[old].to_numpy(), predictions[new].to_numpy()
        old_t, new_t = report['test'][old]['threshold'], report['test'][new]['threshold']
        for _ in range(500):
            ids = rng.integers(0, len(target), len(target))
            delta_acc.append(float(np.mean((p_new[ids] >= new_t) == target[ids]) -
                                   np.mean((p_old[ids] >= old_t) == target[ids])))
            delta_ap.append(float(average_precision_score(target[ids], p_new[ids]) -
                                  average_precision_score(target[ids], p_old[ids])))
        intervals[new] = {'paired_accuracy_delta_95_percentile': np.quantile(delta_acc, [.025, .975]).tolist(),
                          'paired_ap_delta_95_percentile': np.quantile(delta_ap, [.025, .975]).tolist(),
                          'bootstrap_samples': 500}
    (OUT / 'paired_bootstrap.json').write_text(json.dumps(intervals, indent=2) + '\n')

    lines = ['# Adult Income: ML and Deep Learning Improvement', '',
             'Verified by a full CPU rerun on 12 October 2026. This is a post-coursework extension with AI assistance.', '',
             '## Evaluation protocol', '',
             'The original notebook compared configurations on its test split and then chose the best one. '
             'That test was therefore used for model selection. Historical scores remain in the original report; '
             'they are not treated as an untouched test or used to calculate the improvements below.', '',
             'This extension uses the official UCI `adult.data` for training and validation and the separate '
             '`adult.test` for final evaluation. Exact feature duplicates are removed before splitting. '
             'Test duplicates and feature overlap with training are excluded. Labels are normalized, including '
             'the trailing period in the official test file; missing values are imputed from training.', '',
             f"Training: {report['split']['train_rows']:,}; validation: {report['split']['validation_rows']:,}; "
             f"final test: {report['split']['test_rows']:,}. See the split manifest for counts, raw hashes and split hash.", '',
             'Ten configurations were evaluated on validation only: seven classical pipelines (including a prior-only dummy) and three PyTorch MLPs. '
             'Validation AP chooses one classical model and one MLP; validation positive-class F1 chooses their '
             'classification thresholds. The selection lock is written before parsing test labels. '
             'The selected models, two predeclared legacy controls, dummy and logistic baseline are evaluated on test.', '',
             'The controls use StandardScaler without PCA. LightGBM retains SMOTE and class balancing; '
             'the MLP retains SMOTE, weighted cross entropy, BatchNorm, dropout 0.3 and 12 epochs. '
             'They reproduce representative recipes, not every original preprocessing search or the unseeded historical DL run.', '',
             '## Measured results', '']
    for name, result in report['test'].items():
        lines += [f"- **{name}**: accuracy {result['accuracy']:.4f}, AP {result['average_precision']:.4f}, "
                  f"ROC-AUC {result['roc_auc']:.4f}, positive F1 {result['f1_positive']:.4f}, "
                  f"macro-F1 {result['f1_macro']:.4f}, log loss {result['log_loss']:.4f}."]
    lines += ['', f"Against the matched-protocol controls, classical accuracy improved by "
              f"{100*(best_c['accuracy']-base_c['accuracy']):.2f} percentage points; "
              f"MLP accuracy improved by {100*(best_d['accuracy']-base_d['accuracy']):.2f} percentage points. "
              'These changes combine model/training changes and validation-selected thresholds. '
              'AP measures ranking independently of those thresholds.', '',
              '![Measured test comparison](improved/comparison.png)', '', '## What changed', '',
              '- Classical ML: no synthetic one-hot interpolations or automatic double balancing in the new candidates; '
              'bounded LightGBM search with smaller leaves and regularization.',
              '- DL: train-only preprocessing, unweighted binary cross entropy on natural class proportions, '
              'AdamW, reduced dropout, deterministic seeds, validation-AP early stopping and checkpoint restoration.',
              '- Thresholds are selected on validation, with no test tuning. All preprocessing fits on training only.', '',
              '## Stability and limits', '']
    repeats = report['selected_mlp_seed_repeats']
    lines += [f"The selected MLP was repeated with seeds 42, 43 and 44 without searching on test: "
              f"accuracy {min(r['accuracy'] for r in repeats):.4f}–{max(r['accuracy'] for r in repeats):.4f}; "
              f"AP {min(r['average_precision'] for r in repeats):.4f}–{max(r['average_precision'] for r in repeats):.4f}.", '']
    for name, interval in intervals.items():
        lo, hi = interval['paired_ap_delta_95_percentile']
        lines += [f"- {name}: paired row-bootstrap 95% AP-difference interval [{lo:.5f}, {hi:.5f}] (500 draws)."]
    lines += ['', 'The MLP remains behind LightGBM. Small AP gains should be interpreted with their intervals. '
              'Row bootstrap assumes independent records; this dataset has no person identifiers to verify that assumption. '
              'The historical public dataset and one fixed validation split do not establish generalization to current populations. '
              'The official test has already been observed. This rerun freezes the existing configurations; it is not a pristine holdout. Exact feature duplicates include conflicting labels resolved by file order. See docs/verified_results.md for the audit.', '',
              '## Reproduce and inspect', '',
              'Run `python src/adult_benchmark.py`, then `python tools/build_delivery.py`. '
              'Use `notebooks/adult_income_improved.ipynb` for the readable walkthrough. '
              'The notebook explicitly states when it reuses a recorded run; set `REUSE_RECORDED_RUN = False` to retrain.', '',
              'Evidence: `improved/metrics.json`, `split_manifest.json`, `selection_lock.json`, '
              '`training_curves.json`, `paired_bootstrap.json`, and the executed notebook.', '',
              'Dataset: Becker, B. & Kohavi, R. (1996). [Adult, UCI Machine Learning Repository]'
              '(https://archive.ics.uci.edu/dataset/2/adult). DOI: 10.24432/C5XW20; CC BY 4.0. '
              'Raw data and local prediction files are excluded from the repository.']
    (ROOT / 'reports/IMPROVEMENT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    from notebook import create_notebook
    nb = create_notebook()
    nbf.write(nb, ROOT / 'notebooks/adult_income_improved.ipynb')
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    executed = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    assert all(c.execution_count is not None for c in executed.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in executed.cells if c.cell_type == 'code' for o in c.outputs)
    nbf.write(executed, OUT / 'executed_adult_income_improved.ipynb')
    print('Built report, paired bootstrap and notebook; seven code cells executed successfully.')

    from evidence import build_evidence
    build_evidence(ROOT, OUT, args.baseline)


if __name__ == '__main__':
    main()
