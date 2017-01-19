"""

ForecastBox-related classes

"""

from operation import *


class ForecasterFactory:
    """Factory used to instantiate ForecastBox objects

    Methods
    -------
        build (operation_specs):
            Takes a list of OpSpec objects and returns a Forecaster object
            that can execute specified Operations in order.
    """

    def build(self, operation_specs):
        new_forecast_box = Forecaster()

        previous_operation = None
        for spec in reversed(operation_specs):
            new_operation = Operation.create(spec.name, spec.params)
            new_operation.set_next_operation(previous_operation)
            previous_operation = new_operation
        new_forecast_box.start_operation = previous_operation


        return new_forecast_box


class Forecaster:
    """Core class that produces forecasted values given time series data

    Members
    -------
        start_operation:
            Reference to the first Operation that begins forecast chain

    Methods
    -------
        forecast (time_series):
            Takes a pd.Series object indexed by pd.tslib.Timestamp (subclass
            of datetime.datetime) and returns a pd.Series object containing
            forecasted value(s) resulting from built-in Operations.
    """
    def __init__(self):
        self.start_operation = None

    def forecast(self, time_series):
        if self.start_operation is None:
            raise Exception('Needs operations. Use a ForecasterFactory build() method.')

        #TODO: check if pandas Series object
        return self.start_operation.apply(time_series)
