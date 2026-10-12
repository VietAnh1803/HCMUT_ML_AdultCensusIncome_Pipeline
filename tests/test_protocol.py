import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adult_benchmark import (COLUMNS, MLP, duplicate_profile, f1_threshold, metrics,
                             predict_mlp, preprocessor, read_adult, row_keys,
                             training_partition, validate_features)
from adult_benchmark import fetch_data, cross_partition_profile
import torch


class ProtocolTests(unittest.TestCase):
    def test_loader_rejects_missing_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            for text in ['1,2,3\n', ','.join(['1'] * 16) + '\n']:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_adult(path)
            (Path(directory) / 'adult.zip').write_bytes(b'corrupt archive')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                fetch_data(Path(directory))

    def test_features_require_schema_and_finite_numerics(self):
        values = [39, 'Private', 100, 'Bachelors', 13, 'Never-married',
                  'Other-service', 'Own-child', 'White', 'Male', 0, 0, 40, 'Vietnam']
        frame = pd.DataFrame([values], columns=COLUMNS[:-1])
        for bad in [frame.drop(columns='age'), frame.assign(age='bad'), frame.assign(age=np.inf)]:
            with self.subTest(), self.assertRaises(ValueError):
                validate_features(bad)

    def test_duplicate_observations_are_distinct_from_label_conflicts(self):
        x = pd.DataFrame({'age': [25, 25, 25, 30]})
        result = duplicate_profile(x, [0, 0, 1, 0])
        self.assertEqual(result['duplicate_feature_rows'], 2)
        self.assertEqual(result['duplicate_observation_rows'], 1)
        self.assertEqual(result['conflicting_label_feature_groups'], 1)
        cross = cross_partition_profile(pd.DataFrame({'age': [25, 25]}), [0, 1],
                                        pd.DataFrame({'age': [25]}), [0])
        self.assertEqual(cross['matching_feature_pairs'], 2)
        self.assertEqual(cross['different_label_pairs'], 1)
        self.assertEqual(cross['unique_matching_feature_tuples'], 1)

    def test_split_determinism_and_feature_disjointness(self):
        x = pd.DataFrame({'age': list(range(40)) + [0]})
        y = pd.Series([i % 2 for i in range(40)] + [0])
        clean, _, train, validation = training_partition(x, y)
        _, _, train_again, val_again = training_partition(x, y)
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(validation, val_again)
        self.assertTrue(set(row_keys(clean.loc[train])).isdisjoint(row_keys(clean.loc[validation])))
        self.assertEqual(len(clean), 40)

    def test_mlp_inference_is_deterministic_and_bounded(self):
        torch.manual_seed(42)
        model = MLP(3, [4], .5)
        x = np.ones((5, 3), dtype=np.float32)
        first, second = predict_mlp(model, x), predict_mlp(model, x)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (5,))
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        self.assertFalse(model.training)

    def test_scorer_rejects_invalid_shapes_and_probabilities(self):
        for p in [[.1], [[.1], [.9]], [.1, np.nan], [-.1, .9], [.1, 1.1]]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                metrics([0, 1], p, .5)
    def test_official_label_normalization_and_missing_value(self):
        row = '39, ?, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, >50K.\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adult.test'
            path.write_text('| comment\n' + row, encoding='utf-8')
            x, y = read_adult(path)
        self.assertEqual(y.tolist(), [1])
        self.assertTrue(pd.isna(x.workclass.iloc[0]))
        self.assertNotIn('income', x)

    def test_bad_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data'
            for label in ['invalid', '1']:
                path.write_text(f'39, Private, 1, Bachelors, 13, Never-married, Other-service, Own-child, White, Male, 0, 0, 40, Vietnam, {label}\n')
                with self.subTest(label=label), self.assertRaises(ValueError):
                    read_adult(path)

    def test_keys_ignore_row_index_and_find_duplicate_features(self):
        frame = pd.DataFrame({'age': [25, 25, 30], 'country': ['Vietnam', 'Vietnam', 'Other']}, index=[4, 7, 8])
        self.assertEqual(row_keys(frame).iloc[0], row_keys(frame).iloc[1])
        self.assertEqual(int(row_keys(frame).duplicated().sum()), 1)

    def test_validation_threshold_maximizes_positive_f1(self):
        y = np.array([0, 0, 1, 1, 0])
        p = np.array([.05, .3, .4, .8, .1])
        threshold = f1_threshold(y, p)
        result = metrics(y, p, threshold)
        self.assertEqual(result['f1_positive'], 1.)
        self.assertEqual(result['confusion_matrix'], [[3, 0], [0, 2]])

    def test_transform_does_not_learn_test_categories_or_statistics(self):
        values = dict(zip(COLUMNS[:-1], [39, 'Private', 100, 'Bachelors', 13, 'Never-married',
                                        'Other-service', 'Own-child', 'White', 'Male', 0, 0, 40, 'Vietnam']))
        train = pd.DataFrame([values, {**values, 'age': 41}])
        test = pd.DataFrame([{**values, 'age': 10000, 'native-country': 'Unknown'}])
        fitted = preprocessor().fit(train)
        scale = fitted.named_transformers_['numeric'].named_steps['scale']
        before = scale.mean_.copy()
        output = fitted.transform(test)
        self.assertEqual(output.shape, (1, fitted.transform(train).shape[1]))
        np.testing.assert_array_equal(scale.mean_, before)
        self.assertEqual(scale.mean_[0], 40.)
        self.assertTrue(np.isfinite(output).all())
        categories = fitted.named_transformers_['categorical'].named_steps['encode'].categories_
        self.assertNotIn('Unknown', categories[-1])


if __name__ == '__main__':
    unittest.main()
