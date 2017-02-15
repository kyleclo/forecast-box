"""

Example with three simple models

"""

import numpy as np
import pandas as pd

from forecast_box.model import *

#################
#
# generate data
#
#################

N = 100
time_series = pd.Series(data=np.float64(np.random.poisson(lam=10, size=N)),
                        index=pd.date_range('2000-01-01', periods=N))

########################################
#
# MODEL 1: Naive - Last observed value
#
########################################

# 1. instantiate model for 5 different forecast horizons
fixed_params_lv = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_order': 1
}
model_lv = LastValue(**fixed_params_lv)

# 2. training produces 5 sets of fitted values
model_lv.train(time_series)
fitted_values_lv = model_lv.fitted_values
pd.concat([fitted_values_lv[s] for s in fixed_params_lv['forward_steps']],
          axis=1).head(10)

# quick and dirty visualization of model fit
pd.concat([time_series] + [fitted_values_lv[s] for s in
                           fixed_params_lv['forward_steps']], axis=1).plot()

# 3. for each forward_step, get the RMSE of our fitted values
rmse_lv = {}
for s in fixed_params_lv['forward_steps']:
    residuals = time_series - model_lv.fitted_values[s]
    rmse_lv.update({s: np.sqrt(np.mean(residuals ** 2))})

# 4. forecast values at N + forward_steps
forecasted_values_lv = model_lv.forecast(time_series)

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
fitted_values_m = model_m.fitted_values
pd.concat([fitted_values_m[s] for s in fixed_params_m['forward_steps']],
          axis=1).head(10)

pd.concat([time_series] + [fitted_values_m[s] for s in
                           fixed_params_m['forward_steps']], axis=1).plot()

rmse_m = {}
for s in fixed_params_m['forward_steps']:
    residuals = time_series - model_m.fitted_values[s]
    rmse_m.update({s: np.sqrt(np.mean(residuals ** 2))})

forecasted_values_m = model_m.forecast(time_series)

#########################################################
#
# MODEL 3: Linear Regression with day of week indicators
#
#########################################################

fixed_params_lr = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_order': 30,
    'add_day_of_week': True
}
model_lr = LinearRegression(**fixed_params_lr)

model_lr.train(time_series)
fitted_values_lr = model_lr.fitted_values
pd.concat([fitted_values_lr[s] for s in fixed_params_lr['forward_steps']],
          axis=1).head(10)

pd.concat([time_series] + [fitted_values_lr[s] for s in
                           fixed_params_lr['forward_steps']], axis=1).plot()

rmse_lr = {}
for s in fixed_params_lr['forward_steps']:
    residuals = time_series - model_lr.fitted_values[s]
    rmse_lr.update({s: np.sqrt(np.mean(residuals ** 2))})

forecasted_values_lr = model_lr.forecast(time_series)

