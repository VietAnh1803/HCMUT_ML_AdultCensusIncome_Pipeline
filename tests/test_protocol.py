import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adult_benchmark import COLUMNS, f1_threshold, metrics, preprocessor, read_adult, row_keys


class ProtocolTests(unittest.TestCase):
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
            path.write_text('39, Private, 1, Bachelors, 13, Never-married, Other-service, Own-child, White, Male, 0, 0, 40, Vietnam, invalid\n')
            with self.assertRaises(ValueError):
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
        np.testing.assert_array_equal(scale.mean_, before)
        self.assertEqual(scale.mean_[0], 40.)
        self.assertTrue(np.isfinite(output).all())
        categories = fitted.named_transformers_['categorical'].named_steps['encode'].categories_
        self.assertNotIn('Unknown', categories[-1])


if __name__ == '__main__':
    unittest.main()
