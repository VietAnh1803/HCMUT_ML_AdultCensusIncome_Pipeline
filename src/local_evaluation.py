"""Public-label local scoring demo, separate from the official test experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from adult_benchmark import (COLUMNS, fetch_data, metrics, preprocessor, read_adult,
                             save_json, training_partition, validate_features)


def validate_ids(frame: pd.DataFrame) -> None:
    if frame.empty or 'row_id' not in frame:
        raise ValueError('Expected at least one row and a row_id column')
    ids = frame.row_id
    if ids.isna().any() or ids.astype(str).str.strip().eq('').any():
        raise ValueError('row_id must not be missing or empty')
    if ids.astype(str).duplicated().any():
        raise ValueError('Duplicate row_id values are not allowed')


def score_submission(labels: pd.DataFrame, predictions: pd.DataFrame,
                     threshold: float = 0.5) -> dict:
    """Join by ID, never by file order, and reject every incomplete submission."""
    if list(labels.columns) != ['row_id', 'target']:
        raise ValueError('Labels require exactly row_id,target columns')
    if list(predictions.columns) != ['row_id', 'probability']:
        raise ValueError('Predictions require exactly row_id,probability columns')
    validate_ids(labels)
    validate_ids(predictions)
    labels, predictions = labels.copy(), predictions.copy()
    labels['row_id'] = labels.row_id.astype(str)
    predictions['row_id'] = predictions.row_id.astype(str)
    expected, supplied = set(labels.row_id), set(predictions.row_id)
    if expected != supplied:
        raise ValueError(f'ID mismatch: {len(expected - supplied)} missing, {len(supplied - expected)} unexpected')
    try:
        target = pd.to_numeric(labels.target, errors='raise').to_numpy()
        probability = pd.to_numeric(predictions.set_index('row_id').loc[labels.row_id, 'probability'],
                                    errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Targets and probabilities must be numeric') from exc
    result = metrics(target, probability, threshold)
    return {'primary_metric': 'average_precision', 'higher_is_better': True,
            'evaluation_scope': 'local public validation demonstration', **result}


def prepare(cache: Path, output: Path) -> None:
    fetch_data(cache)
    x, y = read_adult(cache / 'adult.data')
    x, y, train_ids, eval_ids = training_partition(x, y)
    output.mkdir(parents=True, exist_ok=True)
    train = x.loc[train_ids].copy()
    train.insert(0, 'row_id', [f'train-{i}' for i in train_ids])
    train['target'] = y.loc[train_ids].to_numpy()
    evaluation = x.loc[eval_ids].copy()
    evaluation.insert(0, 'row_id', [f'validation-{i}' for i in eval_ids])
    labels = pd.DataFrame({'row_id': evaluation.row_id, 'target': y.loc[eval_ids].to_numpy()})
    train.to_csv(output / 'train.csv', index=False)
    evaluation.to_csv(output / 'evaluation_features.csv', index=False)
    labels.to_csv(output / 'evaluation_labels.csv', index=False)
    pd.DataFrame({'row_id': evaluation.row_id, 'probability': 0.5}).to_csv(
        output / 'sample_submission.csv', index=False)
    save_json(output / 'manifest.json', {
        'source': 'official adult.data only; adult.test is not read', 'seed': 42,
        'train_rows': len(train), 'evaluation_rows': len(evaluation), 'features': len(COLUMNS) - 1,
        'labels_are_public': True, 'primary_metric': 'average_precision',
        'warning': 'This is the original validation partition, already used for model selection; not a new holdout.'})


def reference(train_path: Path, features_path: Path, output: Path) -> None:
    """Fit on train.csv and predict features only; no evaluation labels are read."""
    train = pd.read_csv(train_path, dtype={'row_id': str})
    evaluation = pd.read_csv(features_path, dtype={'row_id': str})
    validate_ids(train)
    validate_ids(evaluation)
    if list(train.columns) != ['row_id', *COLUMNS[:-1], 'target']:
        raise ValueError('Training file does not match the documented schema')
    if list(evaluation.columns) != ['row_id', *COLUMNS[:-1]]:
        raise ValueError('Evaluation feature file must not contain target or extra columns')
    x_train, x_eval = train[COLUMNS[:-1]], evaluation[COLUMNS[:-1]]
    validate_features(x_train)
    validate_features(x_eval)
    if not np.isin(train.target.to_numpy(), [0, 1]).all() or train.target.nunique() != 2:
        raise ValueError('Training targets must contain both binary classes')
    if set(train.row_id) & set(evaluation.row_id):
        raise ValueError('Training and evaluation IDs overlap')
    # Compare values across files: hashing separately can miss equal numeric
    # values when CSV inference makes one partition int and the other float.
    if not x_eval.merge(x_train.drop_duplicates(), on=COLUMNS[:-1], how='inner').empty:
        raise ValueError('Training and evaluation features overlap even with different IDs')
    model = Pipeline([('preprocessor', preprocessor()),
                      ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    model.fit(x_train, train.target)
    probability = model.predict_proba(x_eval)[:, 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row_id': evaluation.row_id, 'probability': probability}).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--cache', type=Path, default=Path('.cache/adult'))
    prep.add_argument('--output', type=Path, default=Path('.cache/local-evaluation'))
    ref = commands.add_parser('reference')
    ref.add_argument('--train', type=Path, required=True)
    ref.add_argument('--features', type=Path, required=True)
    ref.add_argument('--output', type=Path, required=True)
    score = commands.add_parser('score')
    score.add_argument('--labels', type=Path, required=True)
    score.add_argument('--predictions', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.cache, args.output)
        elif args.command == 'reference':
            reference(args.train, args.features, args.output)
        else:
            result = score_submission(pd.read_csv(args.labels, dtype={'row_id': str}),
                                      pd.read_csv(args.predictions, dtype={'row_id': str}), args.threshold)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
