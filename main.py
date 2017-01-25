"""

Test

"""

import numpy as np
import pandas as pd
from datetime import date
from forecast_box import *

if __name__ == '__main__':

    forecaster = Forecaster.build(
        operation_specs=[
            OpSpec('smoothing_filter',
                   {'method': 'mean', 'window': 3, 'center': False}),
            OpSpec('changepoint_truncate',
                   {'min_length': 100}),
            OpSpec('stabilize_variance',
                   {'family': 'poisson'}),
            OpSpec('predict',
                   {'model_spec': ModelSpec('seasonal_mean',
                                            {'forward_steps': 1, 'period': 7})})
        ])

    time_series = pd.Series(data=np.random.poisson(lam=10.0, size=100),
                            index=pd.date_range('2017-01-01', periods=100, freq='D'))

    # target_date = date(*map(int, '2017-04-01'.split('-')))

    N = time_series.size
    index_start = N / 3
    yhat = pd.Series(
        data=[forecaster.forecast(time_series[:i]) for i in range(index_start, N)],
        index=pd.date_range(time_series.index[index_start],
                            periods=N - index_start,
                            freq='D', name='yhat'))

    res = time_series[index_start:] - yhat

    print time_series.values
    print yhat.values
    print res.abs().mean()


    # get correct number of k


