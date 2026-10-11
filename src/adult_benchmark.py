"""Validation-selected classical ML and PyTorch baselines for UCI Adult."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier

COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']
NUMERIC = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL = [c for c in COLUMNS[:-1] if c not in NUMERIC]
URL = 'https://archive.ics.uci.edu/static/public/2/adult.zip'


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fetch_data(cache):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'adult.zip'
    if not archive.exists():
        with urlopen(URL, timeout=90) as response:
            archive.write_bytes(response.read())
    with ZipFile(archive) as source:
        for name in ['adult.data', 'adult.test', 'adult.names']:
            (cache / name).write_bytes(source.read(name))


def read_adult(path):
    frame = pd.read_csv(path, names=COLUMNS, comment='|', skipinitialspace=True,
                        na_values='?', keep_default_na=True)
    for column in CATEGORICAL:
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        ).replace('?', np.nan).astype(object)
    labels = frame.pop('income').str.strip().str.rstrip('.')
    if not labels.isin(['<=50K', '>50K']).all():
        raise ValueError('Invalid or missing income label')
    return frame, (labels == '>50K').astype(np.int64)


def row_keys(frame):
    return pd.util.hash_pandas_object(frame, index=False)


def preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL),
    ])


def f1_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(scores))])


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    return dict(accuracy=float(accuracy_score(y, predicted)),
                average_precision=float(average_precision_score(y, probability)),
                roc_auc=float(roc_auc_score(y, probability)),
                f1_positive=float(f1_score(y, predicted)),
                f1_macro=float(f1_score(y, predicted, average='macro')),
                log_loss=float(log_loss(y, probability)), threshold=float(threshold),
                confusion_matrix=confusion_matrix(y, predicted).tolist(), n=len(y))


class MLP(nn.Module):
    def __init__(self, width, hidden, dropout, legacy=False):
        super().__init__()
        layers = []
        for size in hidden:
            layers.append(nn.Linear(width, size))
            if legacy:
                layers.append(nn.BatchNorm1d(size))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            width = size
        layers.append(nn.Linear(width, 2 if legacy else 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def predict_mlp(model, x, legacy=False):
    model.eval()
    with torch.no_grad():
        values = model(torch.as_tensor(x, dtype=torch.float32))
        return (values.softmax(1)[:, 1] if legacy else values.sigmoid().ravel()).numpy()


def train_mlp(x_train, y_train, x_val, y_val, *, hidden, dropout, seed, legacy=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if legacy:
        x_train, y_train = SMOTE(sampling_strategy=0.5, random_state=42).fit_resample(x_train, y_train)
    model = MLP(x_train.shape[1], hidden, dropout, legacy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0 if legacy else 0.01)
    # The legacy control reproduces SMOTE + weighted cross entropy; AdamW with
    # zero decay is Adam's update for this recipe.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 2.])) if legacy else nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.as_tensor(x_train, dtype=torch.float32),
                            torch.as_tensor(np.asarray(y_train), dtype=torch.long if legacy else torch.float32))
    loader = DataLoader(dataset, batch_size=128 if legacy else 256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_ap, best_state, best_epoch, stale, history = -1., None, 0, 0, []
    start = time.perf_counter()
    for epoch in range(1, (12 if legacy else 40) + 1):
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits if legacy else logits.ravel(), targets)
            loss.backward()
            optimizer.step()
        val_ap = float(average_precision_score(y_val, predict_mlp(model, x_val, legacy)))
        history.append({'epoch': epoch, 'validation_ap': val_ap})
        if val_ap > best_ap + 1e-5:
            best_ap, best_state, best_epoch, stale = val_ap, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if not legacy and stale >= 7:
            break
    if not legacy:
        model.load_state_dict(best_state)
    return model, {'seed': seed, 'epochs_run': epoch,
                   'selected_epoch': 12 if legacy else best_epoch,
                   'fit_seconds': time.perf_counter() - start, 'history': history}


def main(output, cache):
    output.mkdir(parents=True, exist_ok=True)
    fetch_data(cache)
    x_all, y_all = read_adult(cache / 'adult.data')
    # Exact feature duplicates share a split. Retain the first occurrence.
    keep = ~row_keys(x_all).duplicated()
    duplicates_removed = int((~keep).sum())
    x_all, y_all = x_all.loc[keep], y_all.loc[keep]
    train_ids, val_ids = train_test_split(x_all.index, test_size=0.2, random_state=42, stratify=y_all)
    x_train, x_val = x_all.loc[train_ids], x_all.loc[val_ids]
    y_train, y_val = y_all.loc[train_ids], y_all.loc[val_ids]
    assert set(row_keys(x_train)).isdisjoint(set(row_keys(x_val)))
    assert set(train_ids).isdisjoint(set(val_ids))
    manifest = {'seed': 42, 'train_rows': len(x_train), 'validation_rows': len(x_val),
                'original_train_rows': len(keep), 'train_feature_duplicates_removed': duplicates_removed,
                'split_ids_sha256': hashlib.sha256(np.asarray(train_ids, dtype=np.int64).tobytes() +
                                                   np.asarray(val_ids, dtype=np.int64).tobytes()).hexdigest(),
                'source': URL, 'source_sha256': {n: hashlib.sha256((cache / n).read_bytes()).hexdigest()
                                               for n in ['adult.data', 'adult.test']}}
    save_json(output / 'split_manifest.json', manifest)
    print(f'Train {len(x_train):,}; validation {len(x_val):,}; CPU, four threads.', flush=True)
    controls = ['legacy_lightgbm', 'legacy_mlp']
    model_specs = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'linear_svm': CalibratedClassifierCV(LinearSVC(dual=False, max_iter=3000, random_state=42), cv=3),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=4, n_jobs=4, random_state=42),
        'legacy_lightgbm': LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=8,
                                         class_weight='balanced', random_state=42, n_jobs=4, verbosity=-1),
        'lightgbm_regularized': LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31,
                                               min_child_samples=60, reg_lambda=2., random_state=42,
                                               n_jobs=4, verbosity=-1),
        'lightgbm_small_leaves': LGBMClassifier(n_estimators=500, learning_rate=0.04, num_leaves=15,
                                               min_child_samples=40, reg_lambda=1., random_state=42,
                                               n_jobs=4, verbosity=-1),
    }
    rows, models, traces = [], {}, {}
    for name, estimator in model_specs.items():
        steps = [('preprocessor', preprocessor())]
        if name == 'legacy_lightgbm':
            steps.append(('smote', SMOTE(sampling_strategy=0.5, random_state=42)))
        steps.append(('classifier', clone(estimator)))
        model = ImbPipeline(steps)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_val)[:, 1]
        threshold = 0.5 if name in controls else f1_threshold(y_val, probability)
        row = {'name': name, 'family': 'classic', 'fit_seconds': time.perf_counter()-start,
               'validation': metrics(y_val, probability, threshold)}
        rows.append(row)
        models[name] = model
        print(name, 'validation AP', round(row['validation']['average_precision'], 6), flush=True)
    processor = preprocessor()
    xt = processor.fit_transform(x_train).astype(np.float32)
    xv = processor.transform(x_val).astype(np.float32)
    mlp_specs = {'legacy_mlp': {'hidden': [128, 64], 'dropout': .3, 'legacy': True},
                 'mlp_64_32': {'hidden': [64, 32], 'dropout': .1},
                 'mlp_128_64': {'hidden': [128, 64], 'dropout': .15}}
    for name, configuration in mlp_specs.items():
        model, trace = train_mlp(xt, y_train, xv, y_val, seed=42, **configuration)
        traces[name] = trace
        probability = predict_mlp(model, xv, name == 'legacy_mlp')
        threshold = 0.5 if name in controls else f1_threshold(y_val, probability)
        row = {'name': name, 'family': 'deep_learning', 'fit_seconds': trace['fit_seconds'],
               'validation': metrics(y_val, probability, threshold)}
        rows.append(row)
        models[name] = model
        print(name, 'validation AP', round(row['validation']['average_precision'], 6), flush=True)
    # Freeze model, checkpoint and threshold before parsing official test labels.
    best_classic = max((r for r in rows if r['family'] == 'classic' and r['name'] not in controls),
                       key=lambda r: r['validation']['average_precision'])['name']
    best_mlp = max((r for r in rows if r['family'] == 'deep_learning' and r['name'] not in controls),
                   key=lambda r: r['validation']['average_precision'])['name']
    thresholds = {r['name']: r['validation']['threshold'] for r in rows}
    lock = {'criterion': 'validation average precision', 'best_classic': best_classic,
            'best_mlp': best_mlp, 'test_models': controls + [best_classic, best_mlp],
            'thresholds': thresholds, 'extra_mlp_seeds': [43, 44],
            'split_ids_sha256': manifest['split_ids_sha256']}
    save_json(output / 'selection_lock.json', lock)
    extras = []
    for seed in [43, 44]:
        model, trace = train_mlp(xt, y_train, xv, y_val, seed=seed, **mlp_specs[best_mlp])
        threshold = f1_threshold(y_val, predict_mlp(model, xv))
        extras.append((seed, model, threshold, trace))
        print('MLP repeat seed', seed, 'selected epoch', trace['selected_epoch'], flush=True)
    x_test, y_test = read_adult(cache / 'adult.test')
    original_test_rows = len(x_test)
    unique = ~row_keys(x_test).duplicated()
    overlap = row_keys(x_test).isin(set(row_keys(x_all)))
    x_test, y_test = x_test.loc[unique & ~overlap], y_test.loc[unique & ~overlap]
    assert set(row_keys(x_test)).isdisjoint(set(row_keys(x_all)))
    manifest.update(test_rows=len(x_test), original_test_rows=original_test_rows,
                    test_duplicates_removed=int((~unique).sum()),
                    test_feature_overlap_with_train_removed=int((unique & overlap).sum()))
    save_json(output / 'split_manifest.json', manifest)
    test_results, predictions = {}, {'source_row': x_test.index, 'target': y_test.values}
    xd = processor.transform(x_test).astype(np.float32)
    for name in lock['test_models']:
        probability = (models[name].predict_proba(x_test)[:, 1] if name in model_specs else
                       predict_mlp(models[name], xd, name == 'legacy_mlp'))
        test_results[name] = metrics(y_test, probability, thresholds[name])
        predictions[name] = probability
    repeats = [{'seed': 42, **test_results[best_mlp]}]
    for seed, model, threshold, trace in extras:
        probability = predict_mlp(model, xd)
        repeats.append({'seed': seed, **metrics(y_test, probability, threshold),
                        'selected_epoch': trace['selected_epoch']})
        traces[f'{best_mlp}_seed_{seed}'] = trace
    report = {'protocol': 'Official UCI train -> deduplicated 80/20 train/validation; separate deduplicated official test',
              'validation_experiments': rows, 'selection': lock, 'test': test_results,
              'selected_mlp_seed_repeats': repeats, 'split': manifest,
              'environment': {'python': os.sys.version, 'torch_device': 'cpu', 'threads': 4,
                              'packages': {name: importlib.metadata.version(name) for name in
                                           ['torch', 'scikit-learn', 'lightgbm', 'imbalanced-learn', 'numpy', 'pandas']}},
              'checks': {'train_validation_feature_disjoint': True, 'train_test_feature_disjoint': True,
                         'preprocessing_fit_on_train_only': True, 'selection_saved_before_test_parsing': True,
                         'test_probabilities_finite': all(np.isfinite(v).all() for k,v in predictions.items()
                                                         if k not in ['source_row', 'target'])}}
    save_json(output / 'metrics.json', report)
    save_json(output / 'training_curves.json', traces)
    pd.DataFrame(predictions).to_csv(output / 'test_predictions.csv', index=False)
    print(json.dumps({'test': test_results, 'mlp_repeats': repeats}, indent=2), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(test_results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['accuracy', 'average_precision']):
        values = [test_results[n][metric] for n in names]
        ax.bar(range(len(names)), values, color=['#a1a1aa', '#a1a1aa', '#0e7490', '#2563eb'])
        ax.set_xticks(range(len(names)), names, rotation=20, ha='right')
        ax.set_ylim(0, 1)
        ax.set_title(metric.replace('_', ' ').title() + ' — official test')
        for i, value in enumerate(values):
            ax.text(i, value+.015, f'{value:.3f}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    print('ADULT_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/improved'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/adult'))
    args = parser.parse_args()
    main(args.output, args.cache)
