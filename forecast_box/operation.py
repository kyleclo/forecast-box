"""

Operation-related classes

"""

import numpy as np
import pandas as pd
from collections import namedtuple
from model import *

OpSpec = namedtuple('OpSpec', ['name', 'params'])


class Operation:
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
        Instantiates Operation object based on name (str).
        Optionally, can provide a dict of params specific to named Model.

    apply (time_series):
        Takes pd.Series object indexed by pd.DatetimeIndex and passes
        modified version to next Operation.  Finally returns a pd.Series
        object after completing recursion stack.
    """

    @staticmethod
    def create(name, params):
        if name == 'predict':
            return Predict(**params)
        if name == 'changepoint_truncate':
            return ChangepointTruncate(**params)
        if name == 'smoothing_filter':
            return SmoothingFilter(**params)
        if name == 'stabilize_variance':
            return StabilizeVariance(**params)
        if name == 'normalization':
            return Normalization(**params)
        else:
            raise Exception(name + '-type of Operation doesnt exist.')

    def __init__(self):
        self.next_operation = None

    def apply(self, time_series):
        print 'Not implemented yet. Skipping...'
        return self._continue(time_series)

    def _continue(self, time_series):
        if self.next_operation is None:
            return time_series
        else:
            return self.next_operation.apply(time_series)


class Predict(Operation):
    """Operation that fits a specified Model and predicts a future value

    Parameters
    ----------
    model_spec:
        OpSpec namedtuple with 'name' (str) and 'params' (dict) fields

    """

    def __init__(self, model_spec):
        Operation.__init__(self)
        self.model_name = model_spec.name
        self.model_params = model_spec.params

    def apply(self, time_series):
        print 'Predicting future value(s)...'

        model = Model.create(self.model_name, self.model_params)
        model.train(time_series)
        time_series = model.predict()

        return self._continue(time_series)


class ChangepointTruncate(Operation):
    def __init__(self, min_length=100):
        Operation.__init__(self)
        self.min_length = min_length


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
            print 'Not implemented yet. Skipping...'
            return self._continue(time_series)

        else:
            return self._continue(time_series)


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
                              new_min=self.target_min, new_max=self.target_max)),
                old_min=self.target_min, old_max=self.target_max,
                new_min=observed_min, new_max=observed_max
            )

        else:
            return self._continue(time_series)

    def _rescale(self, x, old_min, old_max, new_min, new_max):
        return (x - old_min) / (old_max - old_min) * (
        new_max - new_min) + new_min


class Differencing(Operation):
    def __init__(self, method='first', lag=1):
        Operation.__init__(self)
        self.method = method
        self.lag = lag

    # Note: No difference whether first or seasonal differences performed first
    #       but recommended seasonal first since maybe stationarity achieved
    # https://www.otexts.org/fpp/8/1
    def apply(self, time_series):

        if self.method == 'first':

            return self._continue(time_series)


# y1 y2 y3 y4 y5
# y2-y1 y3-y2 y4-y3 y5-y4  (store y1, y2, y3, y4)

# predict y6-y5
# add back y5 to get y6

# predict y2-y1 y3-y2 y4-y3 y5-y4
# add back y1, y2, y3, y4 to get y2, y3, y4, y5.
# no value for y1.



