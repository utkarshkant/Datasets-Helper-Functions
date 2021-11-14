# # UDF for shape and size of data
# import pandas as pd

# def get_df_shape(data):
#   print(f"# rows: {data.shape[0]}")
#   print(f"# cols: {data.shape[1]}")

###############################################################################################################

# UDF to plot acf & pacf as subplots
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(y, lags=30, zero=False, figsize=(15,6)):
    """
    Plots the ACF and PACF plots together in a subplot.

    Parameters:
    - `y`: The time series for which you want to plot the ACF & PACF correlograms.
    - `lags`: Number of lags that you wish to plot. Default is `30`.
    - `zero`: Default is set to `False`, that removes the correlation at lag=0. Assign `True` to plot correlation at lag=0.
    - `figsize`: Plot size. Default is set to `(15,6)`.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plot_acf(y, zero=zero, auto_ylims=True, lags=lags, ax=ax[0]);
    plot_pacf(y, zero=zero, auto_ylims=True, lags=lags, ax=ax[1]);

###############################################################################################################

# UDF for ADF Test
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def adf_tester(series, alpha=0.05):
    """
    Determines whether the given time series is stationary or not.

    Parameter
    - `series`: Time series under test.
    - `alpha`: significant value for p-value test. Default as 0.05.
    """
    series = pd.Series(series)
    adf_results = adfuller(series.dropna())

    print(f"ADF Test Statistic: {adf_results[0]}")
    print(f"p-value: {adf_results[1]}")
    print(f"Significant Value: {alpha}\n")
    if adf_results[1] <= alpha:
        print("No unit root was detected in the time series.")
        print("|| Data is Stationary ||")
    else:
        print("Unit root has been detected in the time series.")
        print("|| Data is Not Stationary ||")

###############################################################################################################

# UDF for KPSS Test for Stationarity

from statsmodels.tsa.stattools import kpss

def kpss_tester(series,regression='c'):
  """
  Determines whether the given time series is stationary or not.
  
  Parameters:
  - series: Pandas series (1-d array). The time series that is to be tested
  - regression : str{"c", "ct"}
    The null hypothesis for the KPSS test.
      "c" : The data is stationary around a constant (default).
      "ct" : The data is stationary around a trend.
  """
  kpss_result = kpss(series.dropna(), regression=regression)
  
  if kpss_result[1] <= 0.05:
    print(f"p-value: {kpss_result[1]}")
    print("Result: Data is Not Stationary")
  else:
    print(f"p-value: {kpss_result[1]}")
    print("Result: Data is Stationary")
