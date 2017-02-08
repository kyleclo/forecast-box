from unittest import TestCase
import mock

import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
from forecast_box.operation import *

class TestDifferencing(TestCase):

    def test_returns_original(self):
        time_series = pd.Series(
            data=np.arange(5.0),
            index=pd.date_range('2000-01-01', periods=5))

        operation = Differencing(period=1)
        assert_series_equal(time_series, operation.apply(time_series))

    def test_differenced_values(self):
        time_series = pd.Series(
            data=np.arange(5.0),
            index=pd.date_range('2000-01-01', periods=5))

        expected = pd.Series(
            data=np.repeat(1.0, repeats=4),
            index=pd.date_range('2000-01-02', periods=4)
        )
        operation = Differencing(period=1)

        def mock_continue(x):
            assert_series_equal(x, expected)
            return x

        operation._continue = mock_continue
        operation.apply(time_series)

