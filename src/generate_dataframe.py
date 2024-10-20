import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA


def make_predictions_prophet(df: pd.DataFrame, periods: int = 1) -> (pd.DataFrame, pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df.set_index('date', inplace=True)
    
    # Prepare data for Prophet
    product_sales = df.groupby(df.index)['unit_sales'].sum().reset_index()  # Summing unit_sales per day
    prophet_df = product_sales.rename(columns={'date': 'ds', 'unit_sales': 'y'})  # Rename columns for Prophet

    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(prophet_df)

    # Create future DataFrame with daily frequency
    future = model.make_future_dataframe(periods=periods, freq='D')

    # Predict future values
    forecast = model.predict(future)

    # Find the last date in the original data
    last_date = prophet_df['ds'].max()

    # Separate given (historical) and predicted values
    given_data = forecast[forecast['ds'] <= last_date][['ds', 'yhat']]
    predicted_values = forecast[forecast['ds'] > last_date][['ds', 'yhat']]
    
    return given_data, predicted_values


def make_predictions_arima(train: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    train['date'] = pd.to_datetime(train['date'], format='%m/%d/%Y')
    train.set_index('date', inplace=True)
    train = train.groupby(train.index).sum()
    train = train.asfreq('ME')

    model = ARIMA(train['unit_sales'], order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=periods)

    return forecast
