"""

Example training and forecasting with models

"""

import numpy as np
import pandas as pd

from forecast_box.model import *
from forecast_box.validate import validate_model

#################
#
# generate data
#
#################

N = 100
np.random.seed(123)
time_series = pd.Series(data=np.float64(np.random.poisson(lam=10, size=N)),
                        index=pd.date_range('2000-01-01', periods=N))

########################################
#
# MODEL 1: Naive - Last observed value
#
########################################

# instantiate and train model for 5 different forecast horizons
fixed_params_lv = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_order': 1
}
model_lv = LastValue(**fixed_params_lv)
model_lv.train(time_series)
model_lv.summarize()
model_lv.plot()

##################################
#
# MODEL 2: Rolling 30 day average
#
##################################

fixed_params_m = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_order': 30
}
model_m = Mean(**fixed_params_m)
model_m.train(time_series)
model_m.summarize()
model_m.plot()

#########################################################
#
# MODEL 3: Linear Regression with day of week indicators
#
#########################################################

fixed_params_lr = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_order': 7,
    'add_day_of_week': True
}
model_lr = LinearRegression(**fixed_params_lr)
model_lr.train(time_series)
model_lr.summarize()
model_lr.plot()

###############################
#
# Forecasting with the models
#
###############################

print model_lv.forecast(time_series)
print model_m.forecast(time_series)
print model_lr.forecast(time_series)

#########################################
#
# Validating model forecasting abilities
#
#########################################

mse_lv = validate_model('last_value', fixed_params_lv,
                        time_series, mean_squared_error)
mse_m = validate_model('mean', fixed_params_m,
                       time_series, mean_squared_error)
mse_lr = validate_model('linear_regression', fixed_params_lr,
                        time_series, mean_squared_error)

pd.concat([mse_lv.rename('last_value'),
           mse_m.rename('mean'),
           mse_lr.rename('linear_regression')],
          axis=1).plot()

print mse_lv.mean()
print mse_m.mean()
print mse_lr.mean()

##########################################################################
#
# CONCLUSION:
#   - LastValue is terrible
#   - LinearRegression seems good on fitted, but overfits for forecasting
#   - Mean has best forecast performance (expected since data is iid Poisson)
#
##########################################################################
