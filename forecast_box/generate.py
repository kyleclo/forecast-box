"""

Randomly generate time series data

"""

import numpy as np
import pandas as pd

def generate_time_series(N, family, params):
    if family == 'poisson':
        return np.random.poisson(lam=params['lambda'], size=N)


def generate_arma(N, family, params):
    if family == 'poisson':
        errors = np.random.poisson(lam=params['lambda'], size=N)


def add_trend():
    pass

def add_outliers():
    pass

def add_seasonality():
    pass

def add_changepoints():
    pass


generate_time_series(N=100, family='poisson', params={'lambda': 10})
generate_time_series(N=100, family='poisson', params={'lambda': 10,
                                                      'ar_weights': [0.5, 0.2],
                                                      'ma_weights': [0.3, 0.2]})
