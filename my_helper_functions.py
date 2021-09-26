# # UDF for shape and size of data
# import pandas as pd

# def get_df_shape(data):
#   print(f"# rows: {data.shape[0]}")
#   print(f"# cols: {data.shape[1]}")

# UDF for Augmented Dickey Fuller Test

from statsmodels.tsa.stattools import adfuller

def adf_tester(series):
  """
  Determines whether the given time series is stationary or not.
  """
  adf_result = adfuller(series.dropna(), autolag="AIC")

  if adf_result[1] <= 0.05:
    print(f"p-value: {adf_result[1]}")
    print("Result: Data is Stationary!")
  else:
    print(f"p-value: {adf_result[1]}")
    print("Data is NOT Stationary!")

############################################

# KPSS Test for Stationarity
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
