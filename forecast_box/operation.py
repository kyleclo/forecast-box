"""

Operation-related classes

"""

import numpy as np
import pandas as pd
from collections import namedtuple
from model import *

OpSpec = namedtuple('OpSpec', ['name', 'params'])


class Operation:
    @staticmethod
    def create(name, params):
        if name == 'changepoint_truncate':
            return ChangepointTruncate(**params)
        if name == 'smoothing_filter':
            return SmoothingFilter(**params)
        if name == 'stabilize_variance':
            return StabilizeVariance(**params)
        if name == 'predict':
            return Predict(**params)

    def set_next_operation(self, operation):
        self.next_operation = operation

    def apply(self, time_series):
        print 'Not implemented yet. Skipping...'
        return self._continue(time_series)

    def _continue(self, time_series):
        if hasattr(self, 'next_operation') and self.next_operation is not None:
            time_series = self.next_operation.apply(time_series)
        return time_series


class ChangepointTruncate(Operation):
    def __init__(self, min_length=100):
        self.min_length = min_length


class SmoothingFilter(Operation):
    def __init__(self, method='mean', window=3, center=True):
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
        self.family = family

    def apply(self, time_series):
        print 'Applying variance-stabilizing transform...'

        if self.family == 'poisson':
            return self._continue(np.sqrt(time_series)) ** 2

        elif self.family == 'binomial':
            # http://blog.as.uky.edu/sta695/wp-content/uploads/2013/01/stabilization.pdf
            raise NotImplementedError

        else:
            return self._continue(time_series)


class Predict(Operation):
    def __init__(self, model_name, forward_steps, model_params):
        self.forward_steps = forward_steps
        self.model_name = model_name
        self.model_params = model_params

    def apply(self, time_series):
        print 'Predicting future value(s)...'

        model = Model.create(self.model_name,
                             self.forward_steps,
                             self.model_params)
        model.train(time_series)
        time_series = model.predict()

        return self._continue(time_series)
