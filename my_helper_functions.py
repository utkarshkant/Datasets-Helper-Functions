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

  if adf_result[2] <= 0.05:
    print(f"p-value: {adf_result[1]}")
    print("Result: Data is Stationary!")
  else:
    print(f"p-value: {adf_result[1]}")
    print("Data is NOT Stationary!")
