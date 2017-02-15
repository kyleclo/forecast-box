"""

Example training and forecasting with models

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
    'ar_order': 30,
    'add_day_of_week': True
}
model_lr = LinearRegression(**fixed_params_lr)
model_lr.train(time_series)
model_lr.summarize()
model_lr.plot()

