# UDF for shape and size of data
def get_shape(df):
  """
  Represents the shape of a dataset in a more clear format.
  """
  print(f"# rows: {df.shape[0]})
  print(f"# cols: {df.shape[1]})

# UDF for Augmented Dickey Fuller Test

from statsmodels.tsa.stattools import adfuller

def adf_tester(series):
  """
  Determines whether the given time series is stationary or not.
  """
  adf_result = adfuller(series.dropna(), autolag="AIC")

  if adf_result[2] <= 0.05:
    print("Result: Data is Stationary!")
  else:
    print("Data is NOT Stationary!")
