import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from local_evaluation import score_submission


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = pd.DataFrame({'row_id': ['a', 'b', 'c', 'd'], 'target': [0, 1, 0, 1]})
        self.predictions = pd.DataFrame({'row_id': ['a', 'b', 'c', 'd'], 'probability': [.1, .9, .2, .8]})

    def test_perfect_score_and_order_independent_join(self):
        result = score_submission(self.labels, self.predictions.iloc[::-1])
        self.assertEqual(result['average_precision'], 1.0)
        self.assertEqual(result['confusion_matrix'], [[2, 0], [0, 2]])
        self.assertEqual(result['precision_positive'], 1.0)
        self.assertEqual(result['recall_positive'], 1.0)

    def test_wrong_ranking_known_average_precision(self):
        wrong = self.predictions.copy()
        wrong['probability'] = [.9, .1, .8, .2]
        self.assertAlmostEqual(score_submission(self.labels, wrong)['average_precision'], 5 / 12)

    def test_constant_probability_baseline(self):
        constant = self.predictions.assign(probability=.5)
        self.assertEqual(score_submission(self.labels, constant)['average_precision'], .5)

    def test_missing_extra_duplicate_and_blank_ids(self):
        cases = [self.predictions.iloc[:-1],
                 pd.concat([self.predictions, pd.DataFrame({'row_id': ['x'], 'probability': [.5]})]),
                 self.predictions.assign(row_id=['a', 'a', 'c', 'd']),
                 self.predictions.assign(row_id=['a', '', 'c', 'd']),
                 self.predictions.assign(row_id=['a', None, 'c', 'd'])]
        for frame in cases:
            with self.subTest(frame=frame.to_dict()), self.assertRaises(ValueError):
                score_submission(self.labels, frame)

    def test_nan_infinite_out_of_range_and_nonnumeric_predictions(self):
        for invalid in [np.nan, np.inf, -.1, 1.1, 'bad']:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                score_submission(self.labels, self.predictions.assign(probability=[invalid, .9, .2, .8]))

    def test_schema_and_target_errors(self):
        cases = [(self.labels, self.predictions.rename(columns={'probability': 'prediction'})),
                 (self.labels, self.predictions.assign(extra=1)),
                 (self.labels.assign(target=0), self.predictions),
                 (self.labels.assign(target=[0, 2, 0, 1]), self.predictions),
                 (self.labels.assign(row_id=['a', 'a', 'c', 'd']), self.predictions)]
        for labels, predictions in cases:
            with self.subTest(), self.assertRaises(ValueError):
                score_submission(labels, predictions)

    def test_invalid_threshold(self):
        for threshold in [-1, 2, np.nan]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                score_submission(self.labels, self.predictions, threshold)


if __name__ == '__main__':
    unittest.main()
