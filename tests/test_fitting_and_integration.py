"""Behavioral checks for data boundaries, checkpoint selection and the local CLI."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import adult_benchmark as benchmark
import local_evaluation as evaluation


def feature_frame(n=40, age_start=20):
    values = [39, 'Private', 100, 'Bachelors', 13, 'Never-married',
              'Other-service', 'Own-child', 'White', 'Male', 0, 0, 40, 'Vietnam']
    frame = pd.DataFrame([values] * n, columns=benchmark.COLUMNS[:-1])
    frame['age'] = np.arange(age_start, age_start + n)
    return frame


class DataBoundaryTests(unittest.TestCase):
    def test_loader_rejects_missing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            feature_frame(1).assign(income='?').to_csv(path, header=False, index=False)
            with self.assertRaisesRegex(ValueError, 'income label'):
                benchmark.read_adult(path)

    def test_loader_rejects_missing_numeric_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            feature_frame(1).assign(age='?', income='<=50K').to_csv(path, header=False, index=False)
            with self.assertRaisesRegex(ValueError, 'Numeric feature age'):
                benchmark.read_adult(path)

    def test_feature_column_order_is_enforced(self):
        with self.assertRaisesRegex(ValueError, 'documented order'):
            benchmark.validate_features(feature_frame().iloc[:, ::-1])

    def test_empty_feature_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.validate_features(feature_frame(0))

    def test_categorical_missing_values_have_explicit_encoding(self):
        frame = feature_frame(2)
        frame.loc[0, 'workclass'] = np.nan
        processor = benchmark.preprocessor().fit(frame)
        encoder = processor.named_transformers_['categorical'].named_steps['encode']
        self.assertIn('Missing', encoder.categories_[0])
        self.assertTrue(np.isfinite(processor.transform(frame)).all())

    def test_numeric_imputation_uses_training_median(self):
        # Direct preprocessor coverage; the raw-data loader rejects numeric missingness.
        frame = feature_frame(3).assign(age=[20., np.nan, 40.])
        processor = benchmark.preprocessor().fit(frame)
        imputer = processor.named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(imputer.statistics_[0], 30.)
        output = processor.transform(frame)
        self.assertAlmostEqual(output[1, 0], 0.)

    def test_stratified_split_preserves_minority_proportion(self):
        frame = feature_frame(100)
        labels = pd.Series([0] * 80 + [1] * 20)
        _, clean_labels, train, validation = benchmark.training_partition(frame, labels)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(validation), 20)
        self.assertEqual(int(clean_labels.loc[train].sum()), 16)
        self.assertEqual(int(clean_labels.loc[validation].sum()), 4)

    def test_conflicting_duplicate_keeps_first_label(self):
        frame = feature_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        labels = pd.Series([i % 2 for i in range(40)] + [1])
        clean, clean_labels, _, _ = benchmark.training_partition(frame, labels)
        self.assertEqual(len(clean), 40)
        self.assertEqual(clean_labels.loc[0], 0)

    def test_missing_category_feature_matches_are_detected(self):
        frame = feature_frame(1).assign(workclass=np.nan)
        profile = benchmark.cross_partition_profile(frame, [0], frame.copy(), [1])
        self.assertEqual(profile['matching_feature_pairs'], 1)
        self.assertEqual(profile['conflicting_feature_tuples'], 1)


class FittingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.train = rng.normal(size=(32, 3)).astype(np.float32)
        self.labels = np.array([0, 1] * 16)
        self.validation = rng.normal(size=(8, 3)).astype(np.float32)
        self.val_labels = np.array([0, 1] * 4)

    def controlled_train(self, scores, **options):
        snapshots = []
        actual_predict = benchmark.predict_mlp

        def capture(model, features, legacy=False):
            snapshots.append({key: value.clone() for key, value in model.state_dict().items()})
            return actual_predict(model, features, legacy)

        arguments = dict(hidden=[4], dropout=.15, seed=42, max_epochs=len(scores), patience=2)
        arguments.update(options)
        with patch.object(benchmark, 'average_precision_score', side_effect=scores), \
                patch.object(benchmark, 'predict_mlp', side_effect=capture):
            model, trace = benchmark.train_mlp(self.train, self.labels, self.validation,
                                              self.val_labels, **arguments)
        return model, trace, snapshots

    def assert_state_equal(self, model, state):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_flat_validation_stops_after_patience(self):
        _, trace, _ = self.controlled_train([.6] * 8)
        self.assertEqual(trace['epochs_run'], 3)
        self.assertEqual(trace['selected_epoch'], 1)
        self.assertEqual(trace['stop_reason'], 'patience_exhausted')

    def test_best_checkpoint_weights_are_restored(self):
        model, trace, states = self.controlled_train([.6, .8, .7, .65, .64])
        self.assertEqual(trace['selected_epoch'], 2)
        self.assertTrue(trace['checkpoint_restored'])
        self.assert_state_equal(model, states[1])
        self.assertTrue(any(not torch.equal(value, states[-1][key])
                            for key, value in model.state_dict().items()))

    def test_validation_improvement_resets_patience(self):
        _, trace, _ = self.controlled_train([.6, .59, .7, .69, .68, .67])
        self.assertEqual(trace['epochs_run'], 5)
        self.assertEqual(trace['selected_epoch'], 3)

    def test_tiny_validation_changes_do_not_reset_patience(self):
        _, trace, _ = self.controlled_train([.6, .600001, .600002, .7])
        self.assertEqual(trace['epochs_run'], 3)
        self.assertEqual(trace['selected_epoch'], 1)

    def test_min_delta_requires_strict_improvement(self):
        _, trace, _ = self.controlled_train([.5, .625, .625, .9], min_delta=.125)
        self.assertEqual(trace['epochs_run'], 3)
        self.assertEqual(trace['best_validation_ap'], .5)

    def test_epoch_budget_still_restores_best_checkpoint(self):
        model, trace, states = self.controlled_train([.6, .8, .7], patience=10)
        self.assertEqual(trace['stop_reason'], 'epoch_budget')
        self.assertEqual(trace['epochs_run'], 3)
        self.assert_state_equal(model, states[1])

    def test_training_same_seed_reproduces_weights_and_history(self):
        arguments = dict(hidden=[4], dropout=.15, seed=42, max_epochs=3)
        first, trace1 = benchmark.train_mlp(self.train, self.labels, self.validation, self.val_labels, **arguments)
        second, trace2 = benchmark.train_mlp(self.train, self.labels, self.validation, self.val_labels, **arguments)
        self.assert_state_equal(first, second.state_dict())
        self.assertEqual(trace1['history'], trace2['history'])
        self.assertEqual(trace1['selected_epoch'], trace2['selected_epoch'])

    def test_modern_optimizer_applies_weight_decay(self):
        with patch.object(benchmark.torch.optim, 'AdamW', wraps=torch.optim.AdamW) as optimizer:
            self.controlled_train([.6])
        self.assertEqual(optimizer.call_args.kwargs['weight_decay'], .01)
        self.assertEqual(optimizer.call_args.kwargs['lr'], .001)

    def test_legacy_control_retains_final_epoch_without_early_stopping(self):
        # Enough minority rows for the historical SMOTE k=5 recipe.
        self.labels = np.array([0] * 24 + [1] * 8)
        with patch.object(benchmark.torch.optim, 'AdamW', wraps=torch.optim.AdamW) as optimizer:
            model, trace, states = self.controlled_train([.8, .7, .6, .5], legacy=True)
        self.assertEqual(trace['epochs_run'], 4)
        self.assertEqual(trace['selected_epoch'], 4)
        self.assertEqual(trace['best_epoch'], 1)
        self.assertFalse(trace['checkpoint_restored'])
        self.assertEqual(trace['stop_reason'], 'legacy_fixed_budget')
        self.assertEqual(optimizer.call_args.kwargs['weight_decay'], 0)
        self.assert_state_equal(model, states[-1])

    def test_invalid_training_configuration_fails_before_fit(self):
        cases = [{'max_epochs': 0}, {'max_epochs': True}, {'patience': 0}, {'patience': 1.5},
                 {'min_delta': -1}, {'min_delta': np.nan}, {'dropout': 1}, {'dropout': -.1},
                 {'hidden': []}, {'hidden': [0]}, {'hidden': [1.5]}]
        for invalid in cases:
            arguments = dict(hidden=[4], dropout=.1, seed=42, max_epochs=1)
            arguments.update(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                benchmark.train_mlp(self.train, self.labels, self.validation, self.val_labels, **arguments)

    def test_invalid_training_arrays_fail_before_fit(self):
        cases = [(self.train[:0], self.labels[:0], self.validation, self.val_labels),
                 (self.train, self.labels[:-1], self.validation, self.val_labels),
                 (self.train, np.zeros(32), self.validation, self.val_labels),
                 (self.train, self.labels, self.validation[:, :2], self.val_labels),
                 (self.train, self.labels, np.full_like(self.validation, np.nan), self.val_labels)]
        for arrays in cases:
            with self.subTest(), self.assertRaises(ValueError):
                benchmark.train_mlp(*arrays, hidden=[4], dropout=.1, seed=42, max_epochs=1)

    def test_mlp_inference_rejects_empty_vector_and_nonfinite_inputs(self):
        model = benchmark.MLP(3, [4], .1)
        for features in [np.empty((0, 3)), np.ones(3), np.full((2, 3), np.inf)]:
            with self.subTest(), self.assertRaises(ValueError):
                benchmark.predict_mlp(model, features)

    def test_legacy_inference_returns_positive_softmax_probability(self):
        model = benchmark.MLP(3, [4], .1, legacy=True)
        actual = benchmark.predict_mlp(model, self.validation, legacy=True)
        with torch.no_grad():
            expected = model(torch.as_tensor(self.validation)).softmax(1)[:, 1].numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (8,))


class LocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.train = feature_frame(40)
        self.train.insert(0, 'row_id', [f'train-{i}' for i in range(40)])
        self.train['target'] = [i % 2 for i in range(40)]
        self.features = feature_frame(8, age_start=70)
        self.features.insert(0, 'row_id', [f'eval-{i}' for i in range(8)])
        self.train_path = self.root / 'train.csv'
        self.features_path = self.root / 'features.csv'
        self.output = self.root / 'nested' / 'predictions.csv'

    def run_reference(self):
        self.train.to_csv(self.train_path, index=False)
        self.features.to_csv(self.features_path, index=False)
        evaluation.reference(self.train_path, self.features_path, self.output)

    def run_prepare(self, name):
        cache = self.root / 'cache'
        cache.mkdir(exist_ok=True)
        feature_frame(40).assign(income=['<=50K', '>50K'] * 20).to_csv(
            cache / 'adult.data', header=False, index=False)
        destination = self.root / name
        with patch.object(evaluation, 'fetch_data'), \
                patch.object(evaluation, 'read_adult', wraps=benchmark.read_adult) as reader:
            evaluation.prepare(cache, destination)
        self.assertEqual(reader.call_args_list[0].args[0].name, 'adult.data')
        self.assertEqual(reader.call_count, 1)
        return destination

    def test_prepare_separates_public_labels_and_never_reads_test(self):
        destination = self.run_prepare('prepared')
        train = pd.read_csv(destination / 'train.csv')
        features = pd.read_csv(destination / 'evaluation_features.csv')
        labels = pd.read_csv(destination / 'evaluation_labels.csv')
        self.assertNotIn('target', features)
        self.assertEqual(list(labels.columns), ['row_id', 'target'])
        self.assertEqual(len(train), 32)
        self.assertEqual(len(features), 8)
        self.assertTrue(set(benchmark.row_keys(train[benchmark.COLUMNS[:-1]])).isdisjoint(
            benchmark.row_keys(features[benchmark.COLUMNS[:-1]])))
        manifest = json.loads((destination / 'manifest.json').read_text())
        self.assertTrue(manifest['labels_are_public'])

    def test_prepare_is_byte_reproducible(self):
        first, second = self.run_prepare('first'), self.run_prepare('second')
        for filename in ['train.csv', 'evaluation_features.csv', 'evaluation_labels.csv',
                         'sample_submission.csv', 'manifest.json']:
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_reference_reads_only_train_and_feature_files(self):
        with patch.object(evaluation.pd, 'read_csv', wraps=pd.read_csv) as reader:
            self.run_reference()
        self.assertEqual([call.args[0] for call in reader.call_args_list],
                         [self.train_path, self.features_path])
        predictions = pd.read_csv(self.output)
        self.assertEqual(predictions.row_id.tolist(), self.features.row_id.tolist())
        self.assertTrue(predictions.probability.between(0, 1).all())
        self.assertEqual(list(predictions.columns), ['row_id', 'probability'])

    def test_reference_rejects_label_leakage_in_features(self):
        self.features['target'] = 0
        with self.assertRaisesRegex(ValueError, 'must not contain target'):
            self.run_reference()
        self.assertFalse(self.output.exists())

    def test_reference_rejects_invalid_training_labels(self):
        for targets in [np.zeros(40), [2] + self.train.target.tolist()[1:]]:
            self.train['target'] = targets
            with self.subTest(), self.assertRaisesRegex(ValueError, 'both binary classes'):
                self.run_reference()
        self.assertFalse(self.output.exists())

    def test_reference_rejects_id_overlap(self):
        self.features.loc[0, 'row_id'] = 'train-0'
        with self.assertRaisesRegex(ValueError, 'IDs overlap'):
            self.run_reference()

    def test_reference_rejects_feature_overlap_under_different_ids(self):
        self.features.loc[0, benchmark.COLUMNS[:-1]] = self.train.loc[0, benchmark.COLUMNS[:-1]]
        for numeric_dtype in ['int64', 'float64']:
            self.features[benchmark.NUMERIC] = self.features[benchmark.NUMERIC].astype(numeric_dtype)
            with self.subTest(numeric_dtype=numeric_dtype), self.assertRaisesRegex(ValueError, 'features overlap'):
                self.run_reference()
        self.assertFalse(self.output.exists())

    def test_cli_scores_csv_and_writes_finite_json(self):
        labels = self.root / 'labels.csv'
        predictions = self.root / 'submission.csv'
        output = self.root / 'score.json'
        pd.DataFrame({'row_id': ['a', 'b'], 'target': [0, 1]}).to_csv(labels, index=False)
        pd.DataFrame({'row_id': ['b', 'a'], 'probability': [1., 0.]}).to_csv(predictions, index=False)
        result = subprocess.run([sys.executable, str(ROOT / 'src/local_evaluation.py'), 'score',
                                 '--labels', str(labels), '--predictions', str(predictions),
                                 '--output', str(output)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        score = json.loads(output.read_text())
        self.assertEqual(score['average_precision'], 1.)
        self.assertEqual(score['accuracy'], 1.)
        self.assertTrue(np.isfinite(score['log_loss']))

    def test_cli_invalid_submission_exits_two_without_new_score(self):
        labels = self.root / 'labels.csv'
        predictions = self.root / 'submission.csv'
        output = self.root / 'score.json'
        pd.DataFrame({'row_id': ['a', 'b'], 'target': [0, 1]}).to_csv(labels, index=False)
        pd.DataFrame({'row_id': ['a'], 'probability': [.2]}).to_csv(predictions, index=False)
        result = subprocess.run([sys.executable, str(ROOT / 'src/local_evaluation.py'), 'score',
                                 '--labels', str(labels), '--predictions', str(predictions),
                                 '--output', str(output)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn('ID mismatch', result.stderr)
        self.assertFalse(output.exists())

    def test_id_string_normalization_rejects_collisions(self):
        labels = pd.DataFrame({'row_id': [1, '1'], 'target': [0, 1]})
        predictions = pd.DataFrame({'row_id': ['1', '2'], 'probability': [.1, .9]})
        with self.assertRaisesRegex(ValueError, 'Duplicate row_id'):
            evaluation.score_submission(labels, predictions)


if __name__ == '__main__':
    unittest.main()
