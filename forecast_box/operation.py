"""

Operation-related classes

"""

import numpy as np
import pandas as pd
from collections import namedtuple
from model import *

OpSpec = namedtuple('OpSpec', ['name', 'params'])
OpTuple = namedtuple('OpTuple', ['name', 'operation'])


# TODO: name and auto print message in superclass
# TODO: enforce minimum size
# TODO: make apply() wrap _apply() (see Models) to maintain common structure like min_size checking
# TODO: collect all params in dict (see Models)
class Operation(object):
    """Class containing transformations, filters, etc. applied to time_series

    Members
    -------
    next_operation:
        reference to the next Operation to be executed after current one.
        Defaults to None if unspecified.

    Methods
    -------
    @staticmethod
    create (name, params):
        instantiates Operation object based on name (str) and a dict of params
        specific to named Operation.

    apply (time_series):
        takes pd.Series object indexed by pd.DatetimeIndex, performs a
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
        self.min_size = 1

    def apply(self, time_series):
        self._check_data_size(time_series)
        return self._apply(time_series)

    def _apply(self, time_series):
        raise NotImplementedError

    def _continue(self, time_series):
        if self.next_operation is None:
            return time_series
        else:
            return self.next_operation.apply(time_series)

    def _check_data_size(self, time_series):
        if time_series.size < self.min_size:
            raise Exception('Not enough data. See min_size.')

# TODO: check model.is_trained to skip training if want to reuse trained params
# TODO: additional parameter to force re-training of model params
class Forecast(Operation):
    """Operation that trains a provided Model and forecasts future value(s)

    Parameters
    ----------
    model:
        Model object
    """

    def __init__(self, model):
        Operation.__init__(self)
        self.model = model
        self.forward_steps = model.fixed_params['forward_steps']
        self.ar_orders = model.fixed_params['ar_orders']
        self.min_size = model.fixed_params['min_size']


    def _apply(self, time_series):
        print 'Forecasting future value(s)...'
        self.model.train(time_series)
        return self._continue(self.model.forecast(time_series))


# TODO: time lag parameter to allow for seasonal smoothing
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

    def _apply(self, time_series):
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
            raise Exception('Invalid method. Try mean or median.')
        return self._continue(time_series)


class StabilizeVariance(Operation):
    def __init__(self, family='poisson'):
        Operation.__init__(self)
        self.family = family

    def _apply(self, time_series):
        print 'Applying variance-stabilizing transform...'
        if self.family == 'poisson':
            return self._continue(np.sqrt(time_series)) ** 2
        elif self.family == 'binomial':
            # http://blog.as.uky.edu/sta695/wp-content/uploads/2013/01/stabilization.pdf
            raise NotImplementedError
        else:
            raise Exception('Invalid family. Try poisson or binomial.')


class Normalization(Operation):
    def __init__(self, method='01', target_min=0.1, target_max=0.9):
        Operation.__init__(self)
        self.min_size = 2
        self.method = method
        self.target_min = target_min
        self.target_max = target_max

    def _apply(self, time_series):
        print 'Normalizing...'

        # Useful if data needs to be in [0,1] (e.g. neural network activation)
        # http://ajbasweb.com/old/ajbas/2011/june-2011/570-580.pdf
        if self.method == '01':
            observed_min = np.min(time_series)
            observed_max = np.max(time_series)
            return rescale(x=self._continue(rescale(x=time_series,
                                                    new_min=self.target_min,
                                                    new_max=self.target_max)),
                           new_min=observed_min,
                           new_max=observed_max)
        else:
            raise Exception('Invalid method. Try 01.')


class Differencing(Operation):
    def __init__(self, period=1):
        Operation.__init__(self)
        self.min_size = 2
        self.period = period

    # Note: No difference whether first or seasonal differences performed first
    #       but recommended seasonal first since maybe stationarity achieved
    # https://www.otexts.org/fpp/8/1
    def _apply(self, time_series):
        print 'Differencing...'

        original = time_series
        deltas = time_series.diff(self.period).dropna()

        return original.head(self.period).append(
            self._continue(deltas) + original.shift(self.period)).dropna()
