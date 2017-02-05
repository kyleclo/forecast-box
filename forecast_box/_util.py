"""

Utility functions for transforming data

"""

import numpy as np
import pandas as pd


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
        candidate_orders = candidate_orders[_compute_nrows(candidate_orders) >= min_nrows]

        if candidate_orders.size == 0:
            raise Exception('Not enough data. Decrease min_order or min_nrows.')

        obs_pct = 1.0 * _compute_nrows(candidate_orders) / (N - 1)
        abs_loss = np.abs(obs_pct - train_pct)

        return candidate_orders[np.argmin(abs_loss)]

    return [_compute_ar_order(s) for s in forward_steps]


def build_y_train(time_series, forward_step, ar_order):
    """Convert time_series to y Series for training"""

    return time_series.tail(time_series.size - forward_step - ar_order + 1)


def build_X_train(time_series, forward_step, ar_order):
    """Convert time_series to X DataFrame for training"""

    def build_lagged_matrix(time_series, num_columns):
        def _add_lagged_column(index_column):
            if index_column == 0:
                return time_series.rename(index_column).to_frame()
            else:
                return pd.concat([_add_lagged_column(index_column - 1),
                                  time_series.shift(index_column).rename(
                                      index_column)],
                                 axis=1)

        return _add_lagged_column(num_columns - 1)

    X = build_lagged_matrix(time_series=time_series.shift(forward_step),
                            num_columns=ar_order)
    return X.dropna(axis=0).rename(columns=lambda j: j + forward_step)


def build_X_forecast(time_series, forward_step, ar_order):
    """Convert time_series to X DataFrame for forecasting"""

    return pd.DataFrame(data=time_series.tail(ar_order)[::-1].reshape(1, -1),
                      index=[time_series.index[-1] + forward_step])


# def get_seasonal_index(seasonal_period, start_date, target_date):
#     """Get the seasonal index of a target date"""
#
#     target_date - start_date