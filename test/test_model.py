"""

Testing model.py module

"""

from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_index_equal, \
    assert_frame_equal
from forecast_box.model import *


# TODO: Check forward steps <= 0
class ExampleModel(Model):
    def __init__(self, forward_steps, ar_order, **kwargs):
        Model.__init__(self, forward_steps, ar_order, **kwargs)

    def _train_once(self, y_train, X_train):
        return {'theta': 123}

    def _predict_once(self, X_test, forward_step):
        return pd.Series(data=[9, 9, 9],
                         index=pd.date_range('2000-01-03', periods=3))


class TestModel(TestCase):
    def setUp(self):
        self.time_series = pd.Series(data=[10, 9, 8, 7, 6],
                                     index=pd.date_range('2000-01-01',
                                                         periods=5))
        self.model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        fake_metric_fun = lambda x, y: 555
        self.model.train(self.time_series, metric_fun=fake_metric_fun)

    def test_create_valid_input(self):
        for name, type in [('last_value', LastValue),
                           ('mean', Mean),
                           ('linear_regression', LinearRegression),
                           ('random_forest', RandomForest)]:
            model = Model.create(name, {'forward_steps': [2, 3],
                                        'ar_order': 1,
                                        'add_day_of_week': True})
            self.assertIsInstance(model, type)
            self.assertListEqual(model.fixed_params['forward_steps'], [2, 3])
            self.assertEqual(model.fixed_params['ar_order'], 1)
            self.assertEqual(model.name, name)
            self.assertEqual(model.fixed_params['min_size'], 4)
            self.assertEqual(model.fixed_params['add_day_of_week'], True)

    def test_create_invalid_input(self):
        self.assertRaises(Exception, Model.create, 'blabla',
                          {'forward_steps': [1, 2, 3], 'ar_order': 1})

    def test_kwargs_default(self):
        model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        self.assertEqual(model.fixed_params['add_day_of_week'], False)

    def test_train_trained_params(self):
        self.assertDictEqual(self.model.trained_params, {2: {'theta': 123},
                                                         3: {'theta': 123}})

    def test_train_fitted_values(self):
        expected = pd.Series(data=[9, 9, 9],
                             index=pd.date_range('2000-01-03', periods=3))
        assert_frame_equal(self.model.fitted_values,
                           pd.DataFrame({2: expected,
                                         3: expected}))

    def test_train_residuals(self):
        expected_2 = pd.Series(data=[-1, -2, -3],
                               index=pd.date_range('2000-01-03', periods=3))
        expected_3 = pd.Series(data=[np.nan, -2.0, -3.0],
                               index=pd.date_range('2000-01-03', periods=3))
        assert_frame_equal(self.model.residuals,
                           pd.DataFrame({2: expected_2, 3: expected_3}))

    def test_train_metric(self):
        self.assertDictEqual(self.model.metric, {2: 555, 3: 555})

    def test_train_small_data(self):
        time_series = pd.Series(data=[10],
                                index=pd.date_range('2000-01-01', periods=1))
        model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        self.assertRaises(Exception, model.train, time_series)

    def test_forecast_results(self):
        expected = pd.Series(data=[9, 9, 9, 9, 9, 9],
                             index=pd.date_range('2000-01-03',
                                                 periods=3).append(
                                 pd.date_range('2000-01-03', periods=3)))
        assert_series_equal(self.model.forecast(self.time_series), expected)

    def test_forecast_small_data(self):
        time_series = pd.Series(data=[10],
                                index=pd.date_range('2000-01-01', periods=1))
        model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        self.assertRaises(Exception, model.forecast, time_series)

    def test_summarize(self):
        pass

    def test_plot(self):
        pass


class TestLastValue(TestCase):
    def setUp(self):
        self.time_series = pd.Series(data=[1, 2, 3, 4, 5],
                                     index=pd.date_range('2000-01-01',
                                                         periods=5))
        self.model = LastValue(forward_steps=[1], ar_order=2)
        self.model.train(self.time_series)
        self.forecasted_values = self.model.forecast(self.time_series)

    def test_predicted_values(self):
        self.assertListEqual(self.model.fitted_values[1].values.tolist(),
                             [2.0, 3.0, 4.0])
        assert_index_equal(self.model.fitted_values[1].index,
                           pd.date_range('2000-01-03', periods=3))
        self.assertListEqual(self.forecasted_values.values.tolist(), [5.0])
        assert_index_equal(self.forecasted_values.index,
                           pd.date_range('2000-01-06', periods=1))


class TestMean(TestCase):
    def setUp(self):
        self.time_series = pd.Series(data=[1, 2, 3, 4, 5],
                                     index=pd.date_range('2000-01-01',
                                                         periods=5))
        self.model = Mean(forward_steps=[1], ar_order=2)
        self.model.train(self.time_series)
        self.forecasted_values = self.model.forecast(self.time_series)

    def test_predicted_values(self):
        self.assertListEqual(self.model.fitted_values[1].values.tolist(),
                             [1.5, 2.5, 3.5])
        assert_index_equal(self.model.fitted_values[1].index,
                           pd.date_range('2000-01-03', periods=3))
        self.assertListEqual(self.forecasted_values.values.tolist(), [4.5])
        assert_index_equal(self.forecasted_values.index,
                           pd.date_range('2000-01-06', periods=1))


class TestLinearRegression(TestCase):
    def setUp(self):
        self.time_series = pd.Series(data=[1, 2, 3, 4, 5],
                                     index=pd.date_range('2000-01-01',
                                                         periods=5))
        self.model = LinearRegression(forward_steps=[1], ar_order=2)
        self.model.train(self.time_series)
        self.forecasted_values = self.model.forecast(self.time_series)

    def test_trained_params(self):
        np.testing.assert_almost_equal(
            self.model.trained_params[1]['model'].intercept_, 1.5)
        np.testing.assert_array_almost_equal(
            self.model.trained_params[1]['model'].coef_, [0.5, 0.5])

    def test_predicted_values(self):
        np.testing.assert_array_almost_equal(
            self.model.fitted_values[1].values, [3.0, 4.0, 5.0])
        assert_index_equal(self.model.fitted_values[1].index,
                           pd.date_range('2000-01-03', periods=3))
        np.testing.assert_array_almost_equal(
            self.forecasted_values.values.tolist(), [6.0])
        assert_index_equal(self.forecasted_values.index,
                           pd.date_range('2000-01-06', periods=1))


# TODO: Need to mock out the sklearn random forest call
class TestRandomForest(TestCase):
    def setUp(self):
        self.time_series = pd.Series(data=[1, 2, 3, 4, 5],
                                     index=pd.date_range('2000-01-01',
                                                         periods=5))
        self.model = LinearRegression(forward_steps=[1], ar_order=2)
        self.model.train(self.time_series)
        self.forecasted_values = self.model.forecast(self.time_series)

    def test_trained_params(self):
        pass

    def test_predicted_values(self):
        pass

