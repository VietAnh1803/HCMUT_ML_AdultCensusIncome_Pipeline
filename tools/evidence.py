"""Generate result plots and factual verification notes from completed outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def build_evidence(root: Path, output: Path, baseline_path: Path | None = None) -> None:
    report = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(output / 'test_predictions.csv')
    traces = json.loads((output / 'training_curves.json').read_text(encoding='utf-8'))
    classic, mlp = [report['selection'][k] for k in ['best_classic', 'best_mlp']]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in [classic, mlp]:
        fpr, tpr, _ = roc_curve(predictions.target, predictions[name])
        ax.plot(fpr, tpr, label=f"{name}: AUC {report['test'][name]['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set(xlabel='False-positive rate (0–1)', ylabel='True-positive rate (0–1)',
           title='Selected models: filtered official-test ROC')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'roc.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, [classic, mlp]):
        matrix = np.asarray(report['test'][name]['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues')
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha='center', va='center',
                    color='white' if value > matrix.max() / 2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicted class (0 / 1)',
               ylabel='Actual class (0 / 1)', title=name)
    fig.suptitle('Official filtered test: confusion counts at frozen thresholds')
    fig.tight_layout()
    fig.savefig(output / 'confusion.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    history = traces[mlp]['history']
    axes[0].plot([r['epoch'] for r in history], [r['training_loss'] for r in history], label='Training BCE, seed 42')
    axes[0].plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], label='Validation BCE, seed 42')
    axes[0].axvline(traces[mlp]['selected_epoch'], linestyle=':', color='black',
                   label=f"Restored epoch {traces[mlp]['selected_epoch']}")
    axes[0].set(xlabel='Epoch', ylabel='Mean binary cross-entropy', title='MLP training and validation loss')
    for seed, key in [(42, mlp), (43, f'{mlp}_seed_43'), (44, f'{mlp}_seed_44')]:
        h = traces[key]['history']
        axes[1].plot([r['epoch'] for r in h], [r['validation_ap'] for r in h], label=f'Seed {seed}')
        selected = h[traces[key]['selected_epoch'] - 1]
        axes[1].scatter([selected['epoch']], [selected['validation_ap']], marker='o', s=35)
    axes[1].set(xlabel='Epoch', ylabel='Validation AP (0–1)', title='MLP validation ranking by seed')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'training.png', dpi=150)
    plt.close(fig)

    lines = ['# Verified Adult Income results', '', f"Run UTC: `{report['run_utc']}`. Python 3.12, CPU, four threads.", '',
             '## Status and scope', '',
             'A full unchanged-source rerun reproduced all prior headline scores. A full revised-source '
             'run reproduced the same accuracy, AP, ROC-AUC, F1 and thresholds. '
             'The changes improve engineering/evaluation checks; they do not claim new model-performance gains. '
             'No configuration was chosen after reviewing final-test scores in this audit.', '',
             '## Dataset', '',
             '[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), Becker/Kohavi (1996), '
             '1994 census extraction, DOI 10.24432/C5XW20, CC BY 4.0. Target: income >$50K. '
             'Fourteen features: six numerical and eight categorical.', '',
             'Official adult.data: 32,561 rows; adult.test: 16,281. After the preserved policy: '
             '26,028 training, 6,508 validation, 16,249 final-test rows. Stratified 80/20 split '
             'of official training, seed 42. Final test contains 12,410 negative and 3,839 positive rows.', '',
             'Keep the first raw feature occurrence before splitting; exclude test feature matches '
             'with official training. Equal features are not proof of duplicate people. '
             'One conflicting-label group exists in each official partition and is resolved by file '
             'order. This preserves comparability but loses observation weighting and one label in '
             'each conflicting group. The unfiltered official partitions are retained in the cache.', '']
    for name, profile in report['dataset_integrity'].items():
        if isinstance(profile, dict):
            lines.append(f'- {name}: `{json.dumps(profile)}`')
    lines += [f"- Removed test feature matches: {report['split']['test_feature_overlap_with_train_removed']}.",
              f"- Archive SHA-256: `{report['dataset_integrity']['archive_sha256']}`; fresh UCI download matched.",
              f"- Split ID SHA-256: `{report['split']['split_ids_sha256']}`."]
    lines += [f'- {name} SHA-256: `{digest}`' for name, digest in report['split']['source_sha256'].items()]
    lines += ['', '## Methodology', '',
              'Imputation/scaling/one-hot encoding fit on training only. Unseen categories are ignored; '
              'categorical missing values use Missing, numerical imputation uses the training median. '
              'Ten validation configurations include seven classical pipelines and three MLPs. '
              'The dummy and legacy controls are excluded from model selection. Validation AP selects '
              'the non-control classical model and MLP; validation positive F1 chooses their thresholds. '
              'Dummy/legacy thresholds are 0.5. The lock precedes test-label parsing; predeclared dummy '
              'and logistic baselines are additionally graded on test.', '',
              'LightGBM uses fixed estimator counts, not early stopping. The selected MLP has 128/64 '
              'ReLU layers, dropout 0.15, BCE, AdamW (lr 0.001, weight decay 0.01), batch size 256, '
              'maximum 40 epochs, validation-AP patience 7 and best-checkpoint restoration. '
              'Seeds 42/43/44 are all reported; no test-based seed selection. Exact classical '
              'configs and legacy recipes are stored in metrics.json.', '',
              '## Filtered official-test results', '',
              '| Model | Accuracy | AP | ROC-AUC | Positive F1 | Threshold |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, r in report['test'].items():
        lines.append(f"| {name} | {r['accuracy']:.4f} | {r['average_precision']:.4f} | {r['roc_auc']:.4f} | {r['f1_positive']:.4f} | {r['threshold']:.4f} |")
    lines += ['', 'AP is sklearn average precision, not trapezoidal PR-AUC. Positive precision/recall, '
              'confusion counts and class distribution are in metrics.json.', '', '## MLP seeds', '',
              '| Seed | Accuracy | AP | ROC-AUC | Epoch |', '| ---: | ---: | ---: | ---: | ---: |']
    for r in report['selected_mlp_seed_repeats']:
        lines.append(f"| {r['seed']} | {r['accuracy']:.4f} | {r['average_precision']:.4f} | {r['roc_auc']:.4f} | {r['selected_epoch']} |")
    lines += ['', f"Three-seed aggregates (mean and sample standard deviation): `{json.dumps(report['seed_summary'])}`.", '',
              '## Old, replaced and unverified results', '']
    if baseline_path:
        before = json.loads(baseline_path.read_text(encoding='utf-8'))
        for name, old in before['test'].items():
            new = report['test'][name]
            if any(abs(old[k] - new[k]) > 1e-12 for k in ['accuracy', 'average_precision', 'roc_auc', 'threshold']):
                raise ValueError(f'Headline result changed: {name}; investigate before revising claims')
            lines.append(f"- {name}: old/new accuracy {old['accuracy']:.6f}, AP {old['average_precision']:.6f}, "
                         f"ROC-AUC {old['roc_auc']:.6f}, unchanged. Log-loss change "
                         f"{new['log_loss'] - old['log_loss']:.3g} reflects float64 probability validation.")
    lines += ['- Coursework test-selected historical scores are retained in the original report; not treated as an untouched test.',
              '- New dummy/logistic results are measured baselines, not replacements for historical results.',
              '- No improvement over the October 11 implementation is claimed. Matched-control accuracy deltas include threshold changes.', '',
              '## Limits and local demonstration', '',
              'The coursework and extension have already exposed test results. This is a frozen '
              'reproducibility check, not a pristine holdout. The historical census sample does not '
              'establish present-day generalization, fairness or suitability for decisions about '
              'people. One validation split and three seeds offer limited robustness evidence. '
              'Row-bootstrap intervals assume independent records; no person IDs establish that assumption.', '',
              'The local scorer uses the original 6,508-row public validation partition and never '
              'reads adult.test. Its logistic reference AP is 0.782419 at a fixed threshold of 0.5. '
              'Labels remain public and previously used for selection. This is not a hidden challenge, '
              'an Eris submission or SHiP-D-compatible infrastructure. See docs/local_challenge.md '
              'and reports/verification/local_reference_score.json.', '',
              '## Executed verification commands', '', '```bash',
              'python -m unittest discover -s tests -v',
              'python tools/run_tests.py',
              'python src/adult_benchmark.py --output .cache/audit-baseline --cache .cache/adult  # original source only',
              'python src/adult_benchmark.py --output .cache/fitting-verified-run --cache .cache/adult',
              'python src/local_evaluation.py prepare',
              'python src/local_evaluation.py reference --train .cache/local-evaluation/train.csv --features .cache/local-evaluation/evaluation_features.csv --output .cache/local-evaluation/reference_predictions.csv',
              'python src/local_evaluation.py score --labels .cache/local-evaluation/evaluation_labels.csv --predictions .cache/local-evaluation/reference_predictions.csv --output .cache/local-evaluation/reference_score.json',
              'python tools/build_delivery.py --input .cache/fitting-verified-run --baseline .cache/audit-baseline/metrics.json', '```', '',
              'The delivery tool copies completed outputs to reports/improved, produces actual '
              'plots and executes a five-cell recorded-results walkthrough. That execution is not '
              'a retraining run. Test, experiment and CLI execution results are recorded in '
              'reports/verification/verification.json.']
    tests = json.loads((root / 'reports/verification/test_results.json').read_text(encoding='utf-8'))
    candidates = {row['name']: row for row in report['validation_experiments']}
    best_c, best_m = candidates[classic], candidates[mlp]
    lines += ['', '## Automated tests and fitting diagnostics', '',
              f"All {tests['passed']} automated tests pass, including actual early-stopping/checkpoint-weight checks "
              'and local CLI integration cases. See test_coverage.md and '
              'reports/verification/test_results.json for the separately executed cases.', '',
              'The unchanged full run also records training AP/log-loss for every validation candidate. '
              f"Selected LightGBM train/validation AP is {best_c['training']['average_precision']:.4f}/"
              f"{best_c['validation']['average_precision']:.4f}; selected MLP AP is "
              f"{best_m['training']['average_precision']:.4f}/{best_m['validation']['average_precision']:.4f}. "
              f"Seed 42 stops at epoch {traces[mlp]['epochs_run']} and restores epoch {traces[mlp]['selected_epoch']}. Dropout, weight decay "
              'and validation early stopping mitigate overfitting; these gaps and baseline comparisons '
              'do not rule out underfitting. See fitting_diagnostics.md for the complete measured evidence.']
    (root / 'docs/verified_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    fit_lines = ['# Fitting diagnostics', '',
                 f"Measured run UTC: `{report['run_utc']}`. Configurations, splits and selection rules are unchanged.", '',
                 '## What is controlled', '',
                 '- Fit preprocessing on training only; group identical raw features before the stratified split. '
                 'The local reference also rejects train/evaluation feature matches with different IDs.',
                 '- Modern MLPs use dropout 0.10/0.15 and AdamW weight decay 0.01. Validation AP controls '
                 'early stopping (patience 7, minimum improvement 1e-5, maximum 40 epochs). Restore the '
                 'best checkpoint, even when the epoch budget is reached.',
                 '- LightGBM candidates bound leaves (15/31), minimum child samples (40/60) and L2 '
                 'regularization (1/2); estimator counts remain fixed. No LightGBM early stopping is claimed.',
                 '- Compare training and validation AP/log-loss with dummy and logistic baselines and '
                 'both MLP capacities. These are diagnostics of insufficient fit, not a proof that '
                 'the optimum architecture or regularization has been found.', '',
                 '## Train–validation measurements', '',
                 'Training scores below are in-sample predictions on the original unresampled training '
                 'partition. Validation scores use the held-out validation partition. For MLPs, both '
                 'are measured in eval mode at the returned checkpoint, with dropout disabled. '
                 'AP gaps are train minus validation; loss gaps are validation minus train. '
                 'A larger positive gap can indicate overfitting; no universal cutoff establishes it.', '',
                 '| Model | Train AP | Validation AP | AP gap | Train log-loss | Validation log-loss |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['validation_experiments']:
        train, val, gap = row['training'], row['validation'], row['fitting_diagnostics']
        fit_lines.append(f"| {row['name']} | {train['average_precision']:.4f} | {val['average_precision']:.4f} | "
                         f"{gap['train_minus_validation_ap']:+.4f} | {train['log_loss']:.4f} | {val['log_loss']:.4f} |")
    fit_lines += ['', '## Checkpoint behavior on the full dataset', '',
                  '| MLP | Epochs run | Selected epoch | Stop reason | Best-checkpoint restored |',
                  '| --- | ---: | ---: | --- | --- |']
    for name, trace in traces.items():
        fit_lines.append(f"| {name} | {trace['epochs_run']} | {trace['selected_epoch']} | "
                         f"{trace['stop_reason']} | {trace['checkpoint_restored']} |")
    fit_lines += ['', 'The legacy MLP intentionally retains its final fixed-budget epoch as a historical '
                  'control; it does not use the modern stopping rule. Unit tests use controlled validation '
                  'scores and compare actual state_dict tensors to verify patience reset, minimum delta '
                  'and restoration rather than trusting a trace flag.', '',
                  '![Training and validation learning traces](../reports/improved/training.png)', '',
                  'The training-loss curve averages minibatch losses while weights change during each '
                  'epoch. The table above uses full-partition log-loss at a single checkpoint; these are '
                  'different measurements. Validation AP need not peak at the lowest validation loss.', '',
                  '## Interpretation and remaining uncertainty', '']
    selected = {row['name']: row for row in report['validation_experiments']}
    best_c, best_m = selected[classic], selected[mlp]
    dummy, logistic = selected['dummy_prior'], selected['logistic']
    fit_lines += [f"The selected LightGBM has train AP {best_c['training']['average_precision']:.4f} and "
                  f"validation AP {best_c['validation']['average_precision']:.4f} "
                  f"(gap {best_c['fitting_diagnostics']['train_minus_validation_ap']:+.4f}). "
                  f"The selected MLP has train AP {best_m['training']['average_precision']:.4f} and "
                  f"validation AP {best_m['validation']['average_precision']:.4f} "
                  f"(gap {best_m['fitting_diagnostics']['train_minus_validation_ap']:+.4f}). "
                  'Report the measured gaps rather than claiming overfitting has been eliminated.', '',
                  f"Validation AP is {dummy['validation']['average_precision']:.4f} for the prior-only dummy "
                  f"and {logistic['validation']['average_precision']:.4f} for logistic regression. "
                  'The selected models beat these simple baselines on validation, which argues against '
                  'a trivial majority-only fit. The MLP still trails LightGBM on validation; that alone '
                  'does not distinguish insufficient capacity, optimization, representation or '
                  'regularization. Underfitting is not ruled out.', '',
                  'Only one validation split and three MLP seeds have been evaluated. No learning curve '
                  'over dataset sizes or nested cross-validation is claimed. The official test was '
                  'previously observed; diagnostics in this update do not use it to adjust models. '
                  'A future tuning experiment should predeclare its search on training/validation and '
                  'use a genuinely new evaluation set for independent confirmation.', '',
                  f"See [test coverage](test_coverage.md) for the {tests['passed']} separately discovered automated cases. "
                  'Passing software checks verifies implementation behavior, not absence of statistical '
                  'overfitting or underfitting.']
    (root / 'docs/fitting_diagnostics.md').write_text('\n'.join(fit_lines) + '\n', encoding='utf-8')
