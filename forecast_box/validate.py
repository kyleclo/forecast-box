"""

Validation

"""

import numpy as np
import pandas as pd


def validate_forecaster(forecaster, time_series, performance_fun):
    """Applies a forecaster to a time series to evaluate performance"""

    performance = []
    min_size = forecaster.min_size
    max_size = time_series.size - max(forecaster.forward_steps)
    for n in range(min_size, max_size + 1):
        sub_time_series = time_series.head(n)
        forecasted_values = forecaster.forecast(sub_time_series)
        actual_values = time_series[forecasted_values.index]
        performance.append(performance_fun(actual_values, forecasted_values))

    return pd.Series(data=performance,
                     index=time_series.index[min_size - 1:max_size])

