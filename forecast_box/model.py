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

    def __init__(self, forward_steps, ar_order, **kwargs):
        self.name = None
        self.fixed_params = {'forward_steps': forward_steps,
                             'ar_order': ar_order,
                             'min_size': max(forward_steps) + ar_order,
                             'add_day_of_week': kwargs.get('add_day_of_week',
                                                           False)}
        self.data = {'y_raw': None, 'X_raw': None}
        self.is_trained = False
        self.trained_params = {s: None for s in forward_steps}
        self.fitted_values = {s: None for s in forward_steps}

    def train(self, y_raw, X_raw=None):
        self._check_inputs(y_raw, X_raw)
        self.data['y'], self.data['X'] = y_raw, X_raw
        for s in self.fixed_params['forward_steps']:
            y_train = self._build_y_train(y_raw, s)
            X_train = self._build_X_train(y_raw, s)
            self.trained_params[s] = self._train_once(y_train, X_train)
            self.fitted_values[s] = self._predict_once(X_train, s)
        self.is_trained = True

    def forecast(self, y_raw, X_raw=None):
        self._check_inputs(y_raw, X_raw)
        forecasted_values = []
        for s in self.fixed_params['forward_steps']:
            X_forecast = self._build_X_forecast(y_raw, s)
            forecasted_values.append(self._predict_once(X_forecast, s))
        return pd.concat(forecasted_values, axis=0)

    def _train_once(self, y_train, X_train):
        raise NotImplementedError

    def _predict_once(self, X_test, forward_step):
        raise NotImplementedError

    # TODO: Check index are DateTime
    def _check_inputs(self, y_raw, X_raw):
        if not isinstance(y_raw, pd.Series):
            raise Exception('y_raw should be pd.Series.')
        if y_raw.size < self.fixed_params['min_size']:
            raise Exception('Not enough data. See min_size.')

        if X_raw is not None and not isinstance(X_raw, pd.DataFrame):
            raise Exception('X_raw should be pd.DataFrame.')
        if X_raw is not None and X_raw.size != y_raw.size:
            raise Exception('X_raw not same size as y_raw.')

    def _build_y_train(self, time_series, forward_step):
        ar_order = self.fixed_params['ar_order']
        return time_series.tail(time_series.size - forward_step - ar_order + 1)

    def _build_X_train(self, time_series, forward_step):
        ar_order = self.fixed_params['ar_order']

        X = pd.concat([time_series.shift(forward_step + lag - 1) for lag in
                       range(1, ar_order + 1)], axis=1).dropna(axis=0)

        if self.fixed_params['add_day_of_week']:
            one_hot = np.stack(
                [np.float64(time_series.index.dayofweek == index_day) for
                 index_day in range(7)], axis=1)
            X = pd.merge(X, pd.DataFrame(data=one_hot,
                                         index=time_series.index),
                         how='inner', left_index=True, right_index=True)

        X.columns = np.arange(len(X.columns))
        return X

    def _build_X_forecast(self, time_series, forward_step):
        ar_order = self.fixed_params['ar_order']
        target_date = time_series.index[-1] + forward_step

        X = pd.DataFrame(data=time_series.tail(ar_order)[::-1].reshape(1, -1),
                         index=[target_date])

        if self.fixed_params['add_day_of_week']:
            X = pd.merge(X, pd.DataFrame(
                data=np.float64(np.arange(7) == target_date).reshape(-1, 7),
                index=[target_date]),
                         how='inner', left_index=True, right_index=True)

        X.columns = np.arange(len(X.columns))
        return X


class LastValue(Model):
    """Forecasts future value is equal to the last observed value

        e.g.  If today is the 10th and forward_steps = [2, 4] then
              I predict the values for the 12th and 14th to be equal to the
              value observed today.
    """

    def __init__(self, forward_steps, ar_order, **kwargs):
        Model.__init__(self, forward_steps, ar_order, **kwargs)
        self.name = 'last_value'

    def _train_once(self, y_train, X_train):
        return {}

    def _predict_once(self, X_test, forward_step):
        return X_test.iloc[:, 0]


class Mean(Model):
    """Forecasts future value is equal to the average of past values"""

    def __init__(self, forward_steps, ar_order, **kwargs):
        Model.__init__(self, forward_steps, ar_order, **kwargs)
        self.name = 'mean'

    def _train_once(self, y_train, X_train):
        return {}

    def _predict_once(self, X_test, forward_step):
        return X_test.mean(axis=1)


# TODO: pop LinearRegression-specific kwargs before passing to Model.__init__
class LinearRegression(Model):
    """Forecasts future value by fitting linear regression model on AR terms"""

    def __init__(self, forward_steps, ar_order, **kwargs):
        Model.__init__(self, forward_steps, ar_order, **kwargs)
        self.name = 'linear_regression'

    def _train_once(self, y_train, X_train):
        model = linear_model.LinearRegression().fit(y=y_train,
                                                    X=X_train)
        return {'model': model}

    def _predict_once(self, X_test, forward_step):
        model = self.trained_params[forward_step]['model']
        return pd.Series(model.predict(X_test), index=X_test.index)

