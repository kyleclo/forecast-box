"""

Forecaster class

"""

from operation import *


class Forecaster(object):
    """Core class that produces forecasted values given time series data

    Members
    -------
        operations:
            list of OpTuple (name, Operation) in order from start to finish

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
            Final Operation should be of type Forecast.
    """

    @staticmethod
    def build(operation_specs):
        if operation_specs[-1].name != 'forecast':
            raise Exception('Final operation should be a Forecast.')

        operations_list = []
        min_size = 1

        previous_operation = None
        for spec in reversed(operation_specs):
            new_operation = Operation.create(spec.name, spec.params)
            new_operation.next_operation = previous_operation
            operations_list.append(OpTuple(spec.name, new_operation))
            min_size = max(min_size, new_operation.min_size)
            previous_operation = new_operation

        forecaster = Forecaster(operations_list[::-1],
                                operations_list[0].operation.forward_steps,
                                min_size)
        return forecaster

    def __init__(self, operations_list, forward_steps, min_size):
        self.operations_list = operations_list
        self.forward_steps = forward_steps
        self.min_size = min_size

    def forecast(self, time_series):
        return self.operations_list[0].operation.apply(time_series)

