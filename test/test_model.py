"""

Testing model.py module

"""

from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_index_equal
from forecast_box.model import *


class ExampleModel(Model):
    def _train_once(self, y_train, X_train):
        return {'theta': 123}

    def _predict_once(self, X_test, forward_step):
        return pd.Series(data=[1, 2, 3])


class TestModel(TestCase):
    def test_create_valid_input(self):
        for name, type in [('last_value', LastValue),
                           ('mean', Mean),
                           ('linear_regression', LinearRegression)]:
            model = Model.create(name, {'forward_steps': [1, 2, 3],
                                        'ar_order': 1})
            self.assertIsInstance(model, type)
            self.assertListEqual(model.fixed_params['forward_steps'],
                                 [1, 2, 3])
            self.assertEqual(model.fixed_params['ar_order'], 1)
            self.assertEqual(model.name, name)
            self.assertEqual(model.fixed_params['min_size'], 4)

    def test_create_invalid_input(self):
        self.assertRaises(Exception, Model.create, 'blabla',
                          {'forward_steps': [1, 2, 3], 'ar_order': 1})

    def test_train_results(self):
        time_series = pd.Series(data=[10, 9, 8, 7, 6],
                                index=pd.date_range('2000-01-01', periods=5))
        model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        model.train(y_raw=time_series)
        self.assertEqual(len(model.trained_params), 2)
        self.assertDictEqual(model.trained_params[2], {'theta': 123})
        self.assertDictEqual(model.trained_params[3], {'theta': 123})
        self.assertEqual(len(model.fitted_values), 2)
        assert_series_equal(model.fitted_values[2], pd.Series(data=[1, 2, 3]))
        assert_series_equal(model.fitted_values[3], pd.Series(data=[1, 2, 3]))

    def test_train_small_data(self):
        time_series = pd.Series(data=[1, 2],
                                index=pd.date_range('2000-01-01', periods=2))
        model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        self.assertRaises(Exception, model.train, time_series)

    def test_forecast_results(self):
        time_series = pd.Series(data=[10, 9, 8, 7, 6],
                                index=pd.date_range('2000-01-01', periods=5))
        model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        assert_series_equal(model.forecast(y_raw=time_series),
                            pd.Series(data=[1, 2, 3, 1, 2, 3],
                                      index=[0, 1, 2, 0, 1, 2]))

    def test_forecast_small_data(self):
        time_series = pd.Series(data=[1, 2],
                                index=pd.date_range('2000-01-01', periods=2))
        model = ExampleModel(forward_steps=[2, 3], ar_order=1)
        self.assertRaises(Exception, model.forecast, time_series)


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


