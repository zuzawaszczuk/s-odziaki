import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA


def make_predictions_prophet(df: pd.DataFrame, periods: int = 1) -> (pd.DataFrame, pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df.set_index('date', inplace=True)
    product_sales = df.groupby('item_nbr')['unit_sales'].resample('ME').sum()
    product_data = product_sales.reset_index()
    prophet_df = product_data[['date', 'unit_sales']].rename(columns={'date': 'ds', 'unit_sales': 'y'})

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)

    last_date = prophet_df['ds'].max()

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
