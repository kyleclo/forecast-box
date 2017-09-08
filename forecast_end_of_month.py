"""

Forecast end of month

"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from forecast_box import *

#################
#
# Generate data
#
#################
N = 100
np.random.seed(123)
time_series = pd.Series(data=np.float64(np.random.poisson(lam=10, size=N)),
                        index=pd.date_range(start='2000-01-01', periods=N,
                                            name='date'),
                        name='y')


###############################
#
# Forecast end of month total
#
###############################

name = 'linear_regression'
params = {
    'ar_order': 7,
    'add_day_of_week': True
}

def forecast_end_of_month_total(time_series, name, params):
    """Forecasts total at end of current month given time series so far"""

    last_observed_day = time_series.index[-1]
    observed_values = time_series.tail(last_observed_day.day)

    days_to_forecast = last_observed_day.days_in_month - last_observed_day.day
    if days_to_forecast > 0:
        params['forward_steps'] = range(1, days_to_forecast + 1)
        model = Model.create(name, params)
        model.train(time_series)
        forecasted_values = model.forecast(time_series)
    else:
        forecasted_values = np.array(0)

    return observed_values.sum() + forecasted_values.sum()

#print forecast_end_of_month_total(time_series, name, params)

##############################
#
# Validation
#
##############################

def validate_end_of_month_forecast(time_series, name, params):
    """asdf"""

    df = time_series.reset_index()
    df = df.assign(index_month=time_series.index.is_month_start.cumsum())
    df = df.assign(max_days_in_month=time_series.index.days_in_month)
    df = pd.merge(
        df.groupby('index_month')['y'].agg(['count', 'sum']).reset_index(),
        df, how='inner', on='index_month')
    df = df.set_index('date')

    forecasted_totals = [None] * time_series.size
    for i in range(time_series.size):
        print 'Simulating forecasts for ' + str(time_series.index[i])
        sub_time_series = time_series.head(i + 1)
        try:
            forecasted_totals[i] = forecast_end_of_month_total(sub_time_series,
                                                               name, params)
        except:
            forecasted_totals[i] = np.nan
    forecasted_totals = pd.Series(data=forecasted_totals,
                                  index=time_series.index)

    df = pd.merge(df, forecasted_totals.to_frame('forecast'), how='outer',
                  left_index=True, right_index=True)

    df = df[df['count'] == df['max_days_in_month']].dropna()

    return df

df = validate_end_of_month_forecast(time_series, name, params)

##############################
#
# Results
#
##############################

df[['sum', 'forecast']].plot()
print (mean_absolute_error(df['sum'], df['forecast']) / df['sum']).drop_duplicates()

