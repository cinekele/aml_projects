from unittest import TestCase
import numpy as np
import pandas as pd
from irls import LogisticRegression, IRLS, Optimizer


class TestLogisticRegression(TestCase):
    def test__include_interactions(self):
        X1 = np.zeros((10, 2))
        X2 = np.ones((10, 2))
        X = np.hstack([X1, X2])
        interactions = np.array([
            [1, 2],
            [2, 3]
        ])
        lr = LogisticRegression()
        lr._set_interactions(interactions)
        X_new = lr._include_interactions(X)
        self.assertEqual(X_new.shape, (10, 6))
        X_expected = np.hstack([X1, X2, np.zeros((10, 1)), np.ones((10, 1))])
        self.assertTrue((X_new == X_expected).all())

    def test__include_interactions_pandas(self):
        X1 = np.zeros((10, 2))
        X2 = np.ones((10, 2))
        X = pd.DataFrame(np.hstack([X1, X2]))
        X.columns = ['a', 'b', 'c', 'd']
        interactions = np.array([
            ['b', 'c'],
            ['c', 'd']
        ])
        lr = LogisticRegression()
        lr._set_interactions(interactions)
        X_new = lr._include_interactions(X)
        self.assertEqual(X_new.shape, (10, 6))
        X_expected = np.hstack([X1, X2, np.zeros((10, 1)), np.ones((10, 1))])
        self.assertTrue((X_new == X_expected).all())

    def test__bad_initalization_optimzer(self):
        with self.assertRaises(Exception) as ex:
            LogisticRegression(optimizer="NotOptimizer")
            self.assertEqual("This optimizer doesn't exist", str(ex.exception))

    def test__proper_initalization_optimzer(self):
        lr = LogisticRegression(optimizer=IRLS(1e-6))
        self.assertTrue(issubclass(lr.optimizer.__class__, Optimizer))
