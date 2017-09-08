"""

Example training and forecasting with more complex model

"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

from forecast_box.model import *

#################
#
# generate data
#
#################

N = 1000
pd.Series(data=sm.tsa.arma_generate_sample(ar=[1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75],
                                           ma=[1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5], nsample=N,
                                           distrvs=lambda n: np.random.poisson(
                                               lam=1.0, size=n)),
          index=pd.date_range('2000-01-01', periods=N)).plot()
time_series = pd.Series(data=np.float64(np.random.poisson(lam=10, size=N)),
                        index=pd.date_range('2000-01-01', periods=N))

#################################################
#
# Random Forests with day of week indicators
#
#################################################

fixed_params = {
    'forward_steps': [1, 2, 3, 4, 5],
    'ar_order': 30,
    'add_day_of_week': True
}
model = RandomForest(**fixed_params)

#################################################
#
# Train linear model with day of week indicators
#
#################################################

model.train(time_series, metric_fun=mean_absolute_error)

###############
#
# View results
#
###############

model.summarize()
model.plot()

###############################
#
# Forecasting with the model
#
###############################

print model.forecast(time_series)
