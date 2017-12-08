import unittest
import numpy as np
import pandas as pd
from knn import KNN
from scipy.spatial.distance import euclidean
import pandas.util.testing as pdt
import numpy.testing as npt


class TestKNN(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame([
            [1, 0, 0, 2, 1],
            [0, 10, 0, 0, 0],
            [0, 0, 10, 0, 0],
            [0, 3, 0, 30, 0],
            [0, 0, 1, 0, 90],
            [0, 0, 3, 2, 30],
            [0, 0, 3, 1, 30]
        ])
        self.y = pd.DataFrame([
            ['a', 1],
            ['a', 2],
            ['a', 3],
            ['b', 4],
            ['b', 5],
            ['b', 6],
            ['b', 7]
        ])

    def test_knn(self):
        model = KNN(10, weighted=True)

    def test_fit(self):
        model = KNN(3)
        model.fit(self.X, self.y)

    def test_query(self):
        knn = KNN(3)
        model = knn.fit(self.X, self.y)
        gen = model.query(self.X.iloc[4:])
        dist, md = next(gen)
        exp_dist = np.array([
            0,
            euclidean(self.X.iloc[4],
                      self.X.iloc[6]),
            euclidean(self.X.iloc[4],
                      self.X.iloc[5])
        ])
        exp_md = pd.DataFrame([
            ['b', 5],
            ['b', 7],
            ['b', 6]],
        index=[4, 6, 5])
        npt.assert_allclose(exp_dist, dist)
        pdt.assert_frame_equal(exp_md, md)

    def test_predict(self):
        knn = KNN(3)
        model = knn.fit(self.X, self.y)
        md = model.predict(self.X.iloc[4:])
        exp_md = pd.DataFrame({
            4: ['b', 6],
            5: ['b', 6],
            6: ['b', 6]
        }, index=[0, 1]).T
        pdt.assert_frame_equal(exp_md, md)


if __name__ == "__main__":
    unittest.main()
