"""

Example setting up and using a simple forecaster

"""

import numpy as np
import pandas as pd

from forecast_box import *



#################
#
# generate data
#
#################

N = 100
time_series = pd.Series(data=np.float64(np.random.poisson(lam=10, size=N)),
                        index=pd.date_range('2000-01-01', periods=N))



###########################
#
# set up (untrained) model
#
###########################

model_params = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_orders': compute_ar_orders(N=time_series.size,
                                   forward_steps=[1, 2, 3, 4, 5],
                                   train_pct=0.8,
                                   min_order=10,
                                   min_nrows=40)
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

forecaster.forecast(time_series)



#############################################################
#
# forecast a different series with previously trained model
#
#############################################################

N_new = 50
new_time_series = pd.Series(data=np.float64(np.random.poisson(lam=10, size=N_new)),
                        index=pd.date_range('2001-01-01', periods=N_new))

forecaster.forecast(new_time_series)







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