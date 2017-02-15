"""

Utility functions for transforming data

"""

# TODO: Deciding whether to keep these functions here or as class methods

import numpy as np
import pandas as pd


# def build_y_train(time_series, forward_step, ar_order):
#     """Convert time_series to y Series for training"""
#     if forward_step + ar_order > time_series.size:
#         raise Exception('Not enough data. Decrease forward_step or ar_order.')
#
#     return time_series.tail(time_series.size - forward_step - ar_order + 1)
#
#
# def build_X_train(time_series, forward_step, ar_order, add_day_of_week=True):
#     """Convert time_series to X DataFrame for training"""
#     if time_series.size <= forward_step + ar_order:
#         raise Exception('Not enough data. Decrease forward_step or ar_order.')
#
#     X = pd.concat([time_series.shift(forward_step + lag - 1) for lag in
#                    range(1, ar_order + 1)], axis=1).dropna(axis=0)
#
#     if add_day_of_week == True:
#         X = pd.merge(X, pd.DataFrame(data=np.stack(
#             [np.float64(time_series.index.dayofweek == index_day) for index_day
#              in range(7)], axis=1), index=time_series.index), how='inner',
#                      left_index=True, right_index=True)
#
#     X.columns = np.arange(len(X.columns))
#     return X
#
#
# def build_X_forecast(time_series, forward_step, ar_order, add_day_of_week=True):
#     """Convert time_series to X DataFrame for forecasting"""
#     if forward_step + ar_order > time_series.size:
#         raise Exception('Not enough data. Decrease ar_order.')
#
#     target_date = time_series.index[-1] + forward_step
#     X = pd.DataFrame(data=time_series.tail(ar_order)[::-1].reshape(1, -1),
#                      index=[target_date])
#
#     if add_day_of_week == True:
#         X = pd.merge(X, pd.DataFrame(
#             data=np.float64(np.arange(7) == target_date).reshape(-1, 7),
#             index=[target_date]), how='inner', left_index=True,
#                      right_index=True)
#
#     X.columns = np.arange(len(X.columns))
#     return X


def rescale(x, new_min, new_max):
    """Rescales values in array to have new min and max"""
    if len(x) < 2:
        raise Exception('Not enough data.')
    old_min, old_max = np.min(x), np.max(x)
    y = (np.float64(x) - old_min) / (old_max - old_min) * (
        new_max - new_min) + new_min
    return y


# TODO: Rethink what this function should do. Maybe train_pct should be min_train_pct.
def compute_ar_orders(N, forward_steps, train_pct, min_order, min_nrows):
    """Computes recommended number of autoregressive terms for modeling

    Parameters
    ----------
    N:
        int; length of time_series

    forward_steps:
        list of ints

    train_pct:
        float between 0.0 and 1.0 indicating desired ratio of number rows
        in training set to number observations in time series.
        As train_pct increases, you get more observations in a training set
        but fewer autoregressive terms as features.
        At most, a time series of length N can produce a training set with
        N - 1 rows (i.e. train_pct = 1.0).

    min_order:
        int > 1 indicating minimum number of autoregressive terms as features.

    min_nrows:
        int > 1 indicating minimum number of observations in training set.

    Returns
    -------
    ar_orders:
        list of int indicating recommended number of autoregressive terms to
        include as features for each forward_step
    """

    def _compute_ar_order(forward_step):
        def _compute_nrows(ar_order):
            return N - forward_step - ar_order + 1

        candidate_orders = np.arange(min_order, N)
        candidate_orders = candidate_orders[
            _compute_nrows(candidate_orders) >= min_nrows]

        if candidate_orders.size == 0:
            raise Exception(
                'Not enough data. Decrease min_order or min_nrows.')

        obs_pct = 1.0 * _compute_nrows(candidate_orders) / (N - 1)
        abs_loss = np.abs(obs_pct - train_pct)

        return candidate_orders[np.argmin(abs_loss)]

    return [_compute_ar_order(s) for s in forward_steps]
