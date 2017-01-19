"""

Useful functions that are deemed unsafe.
Require refactoring or self-implementation.

"""

import numpy as np
import pandas as pd
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial
from scipy.signal import argrelextrema


# time_series = pd.Series(data.reshape(-1), index=pd.date_range('2016-01-01', periods=data.size, freq='D', name='y'))

def find_changepoint(time_series):
    N = len(time_series)
    _, _, Pcp_full = offcd.offline_changepoint_detection(time_series,
                                                         partial(
                                                             offcd.const_prior,
                                                             l=N + 1),
                                                         offcd.fullcov_obs_log_likelihood,
                                                         truncate=-50)
    cpt_sample_freqs = pd.Series(np.exp(Pcp_full).sum(0), index=time_series.index[1:N+1])
    pd.Series(cpt_sample_freqs, index=pd.date_range('2017-01-01', periods=N, freq='D', name='y')).rolling(window=3, center=True).mean().dropna()
