from unittest import TestCase

from forecast_box._util import *


class TestCompute_ar_order(TestCase):
    def test_varying_train_pct(self):
        results = [compute_ar_orders(N=5, forward_steps=[1], train_pct=pct,
                                     min_order=1, min_nrows=1)[0]
                   for pct in [0.25, 0.5, 0.75]]
        self.assertListEqual([4, 3, 2], results)


class TestBuild_y_train(TestCase):
    def setUp(self):
        self.time_series = pd.Series(data=[1, 2, 3, 4, 5],
                                     index=pd.date_range('2000-01-01',
                                                         periods=5))

    def test_type(self):
        self.assertIsInstance(
            build_y_train(self.time_series, forward_step=1, ar_order=1),
            pd.Series)

    def test_shape(self):
        self.assertEqual(build_y_train(self.time_series, forward_step=1,
                                       ar_order=1).shape, (4,))

    def test_values(self):
        self.assertListEqual(build_y_train(self.time_series, forward_step=1,
                                           ar_order=1).values.tolist(),
                             [2, 3, 4, 5])
        self.assertListEqual(build_y_train(self.time_series, forward_step=1,
                                           ar_order=2).values.tolist(),
                             [3, 4, 5])
        self.assertListEqual(build_y_train(self.time_series, forward_step=2,
                                           ar_order=1).values.tolist(),
                             [3, 4, 5])
        self.assertListEqual(build_y_train(self.time_series, forward_step=2,
                                           ar_order=2).values.tolist(),
                             [4, 5])


class TestBuild_X_train(TestCase):
    def setUp(self):
        self.time_series = pd.Series(data=[1, 2, 3, 4, 5],
                                     index=pd.date_range('2000-01-01',
                                                         periods=5))

    def test_type(self):
        self.assertIsInstance(
            build_X_train(self.time_series, forward_step=1, ar_order=1),
            pd.DataFrame)

    def test_shape(self):
        self.assertEqual(build_X_train(self.time_series, forward_step=1,
                                       ar_order=1).shape, (4, 1))
        self.assertEqual(build_X_train(self.time_series, forward_step=1,
                                       ar_order=2).shape, (3, 2))

    def test_values(self):
        self.assertListEqual(build_X_train(self.time_series, forward_step=1,
                                           ar_order=1).values.tolist(),
                             [[1], [2], [3], [4]])
        self.assertListEqual(build_X_train(self.time_series, forward_step=1,
                                           ar_order=2).values.tolist(),
                             [[2, 1], [3, 2], [4, 3]])
        self.assertListEqual(build_X_train(self.time_series, forward_step=2,
                                           ar_order=1).values.tolist(),
                             [[1], [2], [3]])
        self.assertListEqual(build_X_train(self.time_series, forward_step=2,
                                           ar_order=2).values.tolist(),
                             [[2, 1], [3, 2]])

