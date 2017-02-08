"""

Example setting up and using a simple forecaster

"""

import numpy as np
import pandas as pd

from forecast_box import *
from forecast_box.validate import validate_forecaster
from sklearn.metrics import mean_squared_error

#################
#
# generate data
#
#################

N = 50
time_series = pd.Series(data=np.float64(np.random.poisson(lam=10, size=N)),
                        index=pd.date_range('2000-01-01', periods=N))

###########################
#
# set up (untrained) model
#
###########################

model_params = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_orders': [30, 30, 30, 30, 30]
}
model = Model.create('linear_regression', model_params)

####################
#
# set up forecaster
#
####################

forecaster = Forecaster.build(
    operation_specs=[
        OpSpec('smoothing_filter',
               {'method': 'mean', 'window': 3, 'center': False}),
        OpSpec('stabilize_variance',
               {'family': 'poisson'}),
        OpSpec('forecast',
               {'model': model})
    ])
print forecaster.forecast(time_series)


########################
#
# validate performance
#
########################

performance = validate_forecaster(forecaster, time_series, mean_squared_error)
performance.plot()









#
# N = time_series.size
# index_start = N / 3
# yhat = pd.Series(
#     data=[forecaster.forecast(time_series[:i]) for i in range(index_start, N)],
#     index=pd.date_range(time_series.index[index_start],
#                         periods=N - index_start))
#
# res = time_series[index_start:] - yhat
#
# print time_series.values
# print yhat.values
# print res.abs().mean()
