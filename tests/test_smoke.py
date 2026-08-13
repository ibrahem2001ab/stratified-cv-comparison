"""Small offline checks for the reproducibility pipeline."""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold

from reproduce_experiment import DATASETS, N_REPEATS, N_SPLITS, models


class PipelineSmokeTests(unittest.TestCase):
    def test_declared_fit_count(self):
        self.assertEqual(len(DATASETS) * 2 * 2 * N_REPEATS * N_SPLITS, 2600)


    def test_models_fit_mixed_dataframe(self):
        X_num, y = make_classification(
            n_samples=80,
            n_features=4,
            weights=[0.75, 0.25],
            random_state=235975,
        )
        X = pd.DataFrame(X_num, columns=["x1", "x2", "x3", "x4"])
        X["category"] = np.where(X["x1"] > 0, "high", "low")
        X.loc[0, "x1"] = np.nan
        X.loc[1, "category"] = None
        target = pd.Series(y.astype(str))

        for estimator in models(235975).values():
            estimator.fit(X.iloc[:60], target.iloc[:60])
            prediction = estimator.predict(X.iloc[60:])
            self.assertEqual(prediction.shape, (20,))


    def test_stratification_reduces_class_share_variation(self):
        y = np.array([0] * 91 + [1] * 9)
        X = np.arange(len(y)).reshape(-1, 1)
        regular = KFold(n_splits=5, shuffle=True, random_state=235975)
        stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=235975)

        def minority_sd(splitter):
            shares = [np.mean(y[test] == 1) for _, test in splitter.split(X, y)]
            return np.std(shares, ddof=1)

        self.assertLess(minority_sd(stratified), minority_sd(regular))


if __name__ == "__main__":
    unittest.main()
