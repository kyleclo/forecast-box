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

model = Model.create('linear_regression',
                     params={'forward_steps': [1, 2, 3, 4, 5],
                             'ar_order': 10})

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

rmse_over_time = np.sqrt(validate_forecaster(forecaster,
                                             time_series,
                                             mean_squared_error))
overall_rmse = rmse_over_time.mean(axis=0)

###############################################
#
# compare performance with another forecaster
#
###############################################

forecaster2 = Forecaster.build(
    operation_specs=[
        OpSpec('forecast',
               {'model': Model.create('mean',
                     params={'forward_steps': [1, 2, 3, 4, 5],
                             'ar_order': 10})})
    ])

rmse_over_time2 = np.sqrt(validate_forecaster(forecaster2,
                                             time_series,
                                             mean_squared_error))
overall_rmse2 = rmse_over_time2.mean(axis=0)

pd.concat([rmse_over_time.rename('linear_regression'),
           rmse_over_time2.rename('mean')], axis=1).plot()


