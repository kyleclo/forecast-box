"""

Forecaster class

"""

from operation import *


class Forecaster(object):
    """Core class that produces forecasted values given time series data

    Members
    -------
        start_operation:
            Reference to the first Operation that begins forecast chain

    Methods
    -------
        @staticmethod
        build (operation_specs):
            Takes a list of OpSpec objects and returns a Forecaster object
            that can execute specified Operations in order.
            Use this instead of Constructor.

        forecast (time_series):
            Takes a pd.Series object indexed by pd.DatetimeIndex
            and returns a pd.Series object containing forecasted value(s)
            resulting from built-in sequence of Operations.
    """

    @staticmethod
    def build(operation_specs):
        previous_operation = None
        for spec in reversed(operation_specs):
            new_operation = Operation.create(spec.name, spec.params)
            new_operation.next_operation = previous_operation
            previous_operation = new_operation

        return Forecaster(previous_operation)

    def __init__(self, start_operation):
        self.start_operation = start_operation

    def forecast(self, time_series):
        return self.start_operation.apply(time_series)
