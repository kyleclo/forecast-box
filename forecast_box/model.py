"""

Model class

"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from _util import *


# TODO: add time-specific features to X matrix for train() and forecast()
# TODO: incorporate X_raw into X matrix for train() and forecast()
# TODO: allow passing in a function that automatically selects ar_order
# TODO: extend train() and forecast() to allow for multiresponse models
class Model(object):
    """Class used for training and predicting with time series models

    Parameters
    ----------
    forward_steps:
        list of ints such that Model predicts the value(s) at
        time = N + forward_steps given a time_series of length N.

    ar_order:
        int representing number of autoregressive terms when training the
        model.  Determines number of columns in X_train or X_forecast.

    Members
    -------
    name:

    fixed_params:
        dict storing parameters, including 'forward_steps', 'ar_orders',
        and 'min_size' (which is minimum length of time_series for which
        training is feasible).
        See specific Model implementations for any additional params.

    data:
        dict storing input data passed to train() with keys 'y' and 'X'.
        Values are initialized to None.

    is_trained:
        boolean defaults to False and set to True after running train().

    trained_params:
        dict where key = forward_step, and value = trained parameters.
        Values are initialized to None.
        See specific Model implementations for keys.

    fitted_values:
        dict where key = forward_step, and value = fitted time_series.
        Values are initialized to None.

    Methods
    -------
    @staticmethod
    create (name, params):
        Instantiates Model object based on name (str) and a dict of params
        specific to named Model.

    train (y_raw, X_raw=None):
        Fits model parameters using response vector 'y' and (optional)
        feature matrix 'X'.  Inputs should be pd.Series / pd.DataFrame,
        respectively, both indexed by pd.DatetimeIndex.
        Input data, learned parameters, and fitted values for each
        'forward_step' are saved as members.

    forecast (y_raw, X_raw=None):
        Returns pd.Series of forecasted values looking forward from
        last time index of 'y_raw' and 'X_raw' inputs.
    """

    @staticmethod
    def create(name, params):
        possible_models = {
            'last_value': LastValue,
            'mean': Mean,
            'linear_regression': LinearRegression
        }

        if possible_models.get(name) is None:
            raise Exception(name + '-class of Model doesnt exist.')
        else:
            model = possible_models[name](**params)

        return model

    def __init__(self, forward_steps, ar_order):
        self.name = None
        self.fixed_params = {'forward_steps': forward_steps,
                             'ar_order': ar_order,
                             'min_size': max(forward_steps) + ar_order}
        self.data = {'y_raw': None, 'X_raw': None}
        self.is_trained = False
        self.trained_params = {s: None for s in forward_steps}
        self.fitted_values = {s: None for s in forward_steps}

    def train(self, y_raw, X_raw=None):
        self._check_data_size(y_raw)
        self.data['y'], self.data['X'] = y_raw, X_raw

        for s in self.fixed_params['forward_steps']:
            y_train = build_y_train(y_raw, s, self.fixed_params['ar_order'])
            X_train = build_X_train(y_raw, s, self.fixed_params['ar_order'], add_day_of_week=False)
            self.trained_params[s] = self._train_once(y_train, X_train)
            self.fitted_values[s] = self._predict_once(X_train, s)

        self.is_trained = True

    def forecast(self, y_raw, X_raw=None):
        self._check_data_size(y_raw)

        forecasted_values = []
        for s in self.fixed_params['forward_steps']:
            X_forecast = build_X_forecast(y_raw, s, self.fixed_params['ar_order'], add_day_of_week=False)
            forecasted_values.append(self._predict_once(X_forecast, s))

        return pd.concat(forecasted_values, axis=0)

    def _train_once(self, y_train, X_train):
        raise NotImplementedError

    def _predict_once(self, X_test, forward_step):
        raise NotImplementedError

    def _check_data_size(self, time_series):
        if time_series.size < self.fixed_params['min_size']:
            raise Exception('Not enough data. See min_size.')


class LastValue(Model):
    """Forecasts future value is equal to the last observed value

        e.g.  If today is the 10th and forward_steps = [2, 4] then
              I predict the values for the 12th and 14th to be equal to the
              value observed today.
    """

    def __init__(self, forward_steps, ar_order):
        Model.__init__(self, forward_steps, ar_order)
        self.name = 'last_value'

    def _train_once(self, y_train, X_train):
        return {}

    def _predict_once(self, X_test, forward_step):
        return X_test.iloc[:,0]


class Mean(Model):
    """Forecasts future value is equal to the average of past values"""

    def __init__(self, forward_steps, ar_order):
        Model.__init__(self, forward_steps, ar_order)
        self.name = 'mean'

    def _train_once(self, y_train, X_train):
        return {}

    def _predict_once(self, X_test, forward_step):
        return X_test.mean(axis=1)


# TODO: expand constructor to take kwargs into fixed_params
class LinearRegression(Model):
    """Forecasts future value by fitting linear regression model on AR terms"""

    def __init__(self, forward_steps, ar_order):
        Model.__init__(self, forward_steps, ar_order)
        self.name = 'linear_regression'

    def _train_once(self, y_train, X_train):
        model = linear_model.LinearRegression().fit(y=y_train,
                                                    X=X_train)
        return {'model': model}

    def _predict_once(self, X_test, forward_step):
        model = self.trained_params[forward_step]['model']
        return pd.Series(model.predict(X_test), index=X_test.index)

