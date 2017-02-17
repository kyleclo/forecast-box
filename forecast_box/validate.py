"""

Validation

"""

import numpy as np
import pandas as pd

from model import Model


# TODO: different versions with resampling or subsampling
def validate_model(name, params, time_series, metric_fun):
    """Evaluates performance of Model forecast method on time series"""

    min_size = max(params['forward_steps']) + params['ar_order']
    max_size = time_series.size - max(params['forward_steps'])
    metric = []
    for n in range(min_size, max_size + 1):
        print 'Simulating forecasts for ' + str(time_series.index[n - 1])
        sub_time_series = time_series.head(n)
        model = Model.create(name, params)
        model.train(sub_time_series)
        forecasted_values = model.forecast(sub_time_series)
        actual_values = time_series[forecasted_values.index]
        metric.append(metric_fun(actual_values, forecasted_values))
    return pd.Series(data=metric,
                     index=time_series.index[(min_size - 1):max_size])


# def validate_forecaster(forecaster, time_series, performance_fun):
#     """Applies a forecaster to a time series to evaluate performance"""
#
#     performance = []
#     min_size = forecaster.min_size
#     max_size = time_series.size - max(forecaster.forward_steps)
#     for n in range(min_size, max_size + 1):
#         print 'Simulating forecaster for ' + str(time_series.index[n - 1])
#         sub_time_series = time_series.head(n)
#         forecasted_values = forecaster.forecast(sub_time_series)
#         actual_values = time_series[forecasted_values.index]
#         performance.append(performance_fun(actual_values, forecasted_values))
#
#     return pd.Series(data=performance,
#                      index=time_series.index[min_size - 1:max_size])

