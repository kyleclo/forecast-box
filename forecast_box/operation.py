"""

Operation-related classes

"""

import numpy as np
import pandas as pd
from collections import namedtuple
from model import *

OpSpec = namedtuple('OpSpec', ['name', 'params'])


class Operation(object):
    """Class containing transformations, filters, etc. applied to time_series

    Members
    -------
    next_operation:
        Reference to the next Operation to be executed after current one.
        Defaults to None if unspecified.

    Methods
    -------
    @staticmethod
    create (name, params):
        Instantiates Operation object based on name (str) and a dict of params
        specific to named Operation.

    apply (time_series):
        Takes pd.Series object indexed by pd.DatetimeIndex, performs a
        transformation, and passes modified series to next Operation to
        apply. Finally returns series after completing recursion stack.
    """

    @staticmethod
    def create(name, params):
        possible_operations = {
            'forecast': Forecast,
            'smoothing_filter': SmoothingFilter,
            'stabilize_variance': StabilizeVariance,
            'normalization': Normalization
        }

        if possible_operations.get(name) is None:
            raise Exception(name + '-type of Operation doesnt exist.')
        else:
            operation = possible_operations[name](**params)

        return operation

    def __init__(self):
        self.next_operation = None

    def apply(self, time_series):
        raise NotImplementedError

    def _continue(self, time_series):
        if self.next_operation is None:
            return time_series
        else:
            return self.next_operation.apply(time_series)


class Forecast(Operation):
    """Operation that fits a specified Model and forecasts future value(s)

    Parameters
    ----------
    name:
        str for Model.create()

    params:
        dict of parameters to pass into Model.create()
    """

    def __init__(self, model):
        Operation.__init__(self)
        self.model = model

    def apply(self, time_series):
        print 'Forecasting future value(s)...'

        if not self.model.is_trained:
            self.model.train(time_series)
        return self._continue(self.model.forecast(time_series))


class SmoothingFilter(Operation):
    """Operation that applies a moving average/median filter to time_series

    Parameters
    ----------
    method:
        String indicating 'mean' or 'median' filter.  Defaults to 'mean'.

    window:
        Integer indicating size of window.  Defaults to 3.

    center:
        Boolean indicating whether window is centered.  Defaults to True.

    Note
    ----
    Regarding edge values of time_series, these filters will take as
    many values as possible as opposed to returning NaN.

    """

    def __init__(self, method='mean', window=3, center=True):
        Operation.__init__(self)
        self.method = method
        self.window = window
        self.center = center

    def apply(self, time_series):
        print 'Applying smoothing filter...'

        if self.method == 'mean':
            time_series = time_series.rolling(window=self.window,
                                              center=self.center,
                                              min_periods=1).mean()
        elif self.method == 'median':
            time_series = time_series.rolling(window=self.window,
                                              center=self.center,
                                              min_periods=1).median()
        else:
            raise Exception('Invalid method. Select mean or median.')

        return self._continue(time_series)


class StabilizeVariance(Operation):
    def __init__(self, family=None):
        Operation.__init__(self)
        self.family = family

    def apply(self, time_series):
        print 'Applying variance-stabilizing transform...'

        if self.family == 'poisson':
            return self._continue(np.sqrt(time_series)) ** 2

        elif self.family == 'binomial':
            # http://blog.as.uky.edu/sta695/wp-content/uploads/2013/01/stabilization.pdf
            raise NotImplementedError

        else:
            raise NotImplementedError


class Normalization(Operation):
    def __init__(self, method='01', target_min=0.1, target_max=0.9):
        Operation.__init__(self)
        self.method = method
        self.target_min = target_min
        self.target_max = target_max

    def apply(self, time_series):
        print 'Normalizing...'

        # Useful if data needs to be in [0,1] (e.g. neural network activation)
        # http://ajbasweb.com/old/ajbas/2011/june-2011/570-580.pdf
        if self.method == '01':
            observed_min = np.min(time_series)
            observed_max = np.max(time_series)
            return self._rescale(x=self._continue(
                self._rescale(x=time_series,
                              old_min=observed_min, old_max=observed_max,
                              new_min=self.target_min,
                              new_max=self.target_max)),
                old_min=self.target_min, old_max=self.target_max,
                new_min=observed_min, new_max=observed_max
            )

        else:
            raise NotImplementedError

    def _rescale(self, x, old_min, old_max, new_min, new_max):
        return (x - old_min) / (old_max - old_min) * (
            new_max - new_min) + new_min


class Differencing(Operation):
    def __init__(self, period=1):
        Operation.__init__(self)
        self.period = period

    # Note: No difference whether first or seasonal differences performed first
    #       but recommended seasonal first since maybe stationarity achieved
    # https://www.otexts.org/fpp/8/1
    def apply(self, time_series):
        print 'Differencing...'
        original = time_series
        deltas = time_series.diff(self.period).dropna()

        return original.head(self.period).append(
            self._continue(deltas) + original.shift(self.period)).dropna()
