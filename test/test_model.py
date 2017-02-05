from unittest import TestCase
import mock

import numpy as np
import pandas as pd
from forecast_box.model import *


class TestModel(TestCase):
    def test_create_valid(self):
        fixed_params = {
            'forward_steps': [1],
            'ar_orders': [1]
        }
        self.assertIsInstance(Model.create('last_value', fixed_params), LastValue)
        self.assertIsInstance(Model.create('mean', fixed_params), Mean)
        self.assertIsInstance(Model.create('linear_regression', fixed_params),LinearRegression)

    def test_create_invalid(self):
        fixed_params = {
            'forward_steps': [1],
            'ar_orders': [1]
        }
        self.assertRaises(Exception, Model.create, 'blabla', fixed_params)

    #TODO: Mock out _train_once() and _predict_once()
    def test_train(self):
        pass

    def test_forecast(self):
        pass


class TestLastValue(TestCase):

    def test_fitted_values(self):
        time_series = pd.Series(
            data=[1, 2, 3, 4, 5],
            index=pd.date_range('2000-01-01', periods=5))

        model = LastValue(forward_steps=[1], ar_orders=[1])
        model.train(time_series)
        self.assertListEqual(model.fitted_values[1].values.tolist(),
                             [1, 2, 3, 4])

    def test_predict_once(self):
        pass

