"""

Model class

"""

import numpy as np
import pandas as pd
from collections import namedtuple
import statsmodels.api as sm


ModelSpec = namedtuple('ModelSpec', ['name', 'forward_steps', 'params'])


class Model:
    """Abstract class of Models, which are used to train time series models

    Static Method
    -------------
    create (name, forward_steps, params):
        Looks up a type of Model input name (str), and instantiates
        the object by passing forward_steps (int) and params (dict).

    Inherited by Models
    -------------------

        Parameters
        ----------
        forward_steps:
            Models predict the value at t = N + forward_steps given
            time_series of length N.  Most commonly, forward_steps = 1.

        Methods
        -------
        train (time_series):
            Takes pd.Series object indexed by pd.tslib.Timestamp and fits model
            parameters. Learned parameters are saved in Model object.

        predict ():
            Returns predicted value (float) for trained time_series.

            TODO 1:  Allow return of pd.Series with pd.tslib.Timestamp index
            TODO 2:  Allow input of alternative pd.Series object (test set)

    """

    @staticmethod
    def create(name, forward_steps, params):
        if name == 'last_value':
            return LastValue(forward_steps)
        if name == 'seasonal_mean':
            return SeasonalMean(forward_steps, **params)
        if name == 'classical_decomposition':
            return ClassicalDecomposition(forward_steps, **params)
        if name == 'arima':
            return Arima(forward_steps, **params)
        else:
            raise Exception('Model class doesnt exist.')

    def __init__(self, forward_steps):
        self.forward_steps = forward_steps
        self.predicted_values = None

    def train(self, time_series):
        raise NotImplementedError

    def predict(self):
        return self.predicted_values


class LastValue(Model):
    def __init__(self, forward_steps):
        Model.__init__(self, forward_steps)

    def train(self, time_series):
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
