"""

Model class

"""

import numpy as np
import pandas as pd
from collections import namedtuple
import statsmodels.api as sm


ModelSpec = namedtuple('ModelSpec', ['name', 'params'])


class Model:
    """Class used for training and predicting with time series models

    Members
    ----------
    forward_steps:
        Integer s.t. Model predicts the value at t = N + forward_steps
        given a time_series of length N.  Defaults to forward_steps = 1.

    predicted_values:
        Predicted value at time t = N + forward_steps.  Has None value until
        train() method is used.

    Methods
    -------
    @staticmethod
    create (name, params):
        Instantiates Model object based on name (str).
        Optionally, can provide a dict of params specific to named Model.

    train (time_series):
        Takes pd.Series object indexed by pd.DatetimeIndex and fits model
        parameters.  Learned parameters are saved in Model object.

    predict ():
        Returns predicted value (float) for trained time_series.

        TODO 1:  Allow return of pd.Series with pd.DatetimeIndex index
        TODO 2:  Allow input of alternative pd.Series object (test set)

    """

    @staticmethod
    def create(name, params):
        if name == 'last_value':
            return LastValue(**params)
        if name == 'seasonal_mean':
            return SeasonalMean(**params)
        if name == 'classical_decomposition':
            return ClassicalDecomposition(**params)
        if name == 'arima':
            return Arima(**params)
        else:
            raise Exception(name + '-class of Model doesnt exist.')

    def __init__(self, forward_steps=1):
        self.forward_steps = forward_steps
        self.predicted_values = None

    def train(self, time_series):
        raise NotImplementedError

    def predict(self):
        return self.predicted_values


# class LastValue(Model):
#     def __init__(self, forward_steps):
#         Model.__init__(self, forward_steps)
#
#     def train(self, time_series):
#         #self.model = lambda x: x[-1]
#         self.predicted_values = time_series[-1]


class LastValue(Model):
    def __init__(self, forward_steps):
        Model.__init__(self, forward_steps)

    def train(self, time_series):
        self.model = lambda x: x[-1]
        self.predicted_values = time_series[-1]


class SeasonalMean(Model):
    def __init__(self, forward_steps, period=7):
        Model.__init__(self, forward_steps)
        self.period = period
        self.seasonal_means = None

    def train(self, time_series):
        values_by_season = [[]] * self.period

        i = 0
        while i < time_series.size:
            values_by_season[i % self.period].append(time_series[i])
            i += 1

        self.seasonal_means = np.mean(values_by_season, axis=0)
        self.predicted_values = self.seasonal_means[
            (i + self.forward_steps - 1) % self.period]


class ClassicalDecomposition(Model):
    def train(self, time_series):
        return None

    def predict(self):
        return None


class Arima(Model):
    def train(self, time_series):
        return None

    def predict(self):
        return None
