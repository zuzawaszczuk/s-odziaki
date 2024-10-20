import pandas as pd
import numpy as np
import torch
from src.generate_dataframe import make_predictions_prophet, make_predictions_arima


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def prepare_data_for_testing(datapath: str) -> (pd.DataFrame, pd.DataFrame):
    df = pd.read_csv(datapath)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df.set_index('date', inplace=True)
    product_sales = df.groupby('item_nbr')['unit_sales'].resample('ME').sum()
    product_data = product_sales.reset_index()

    train = product_data[product_data['date'] <= '2017-01-01']
    test = product_data[product_data['date'] > '2017-01-01']

    return train, test


def test_prophet(periods: int) -> (np.array, np.array):
    train, test = prepare_data_for_testing('data/train_115850.csv')

    old, preds = make_predictions_prophet(train, periods)

    return test['unit_sales'].to_numpy(), preds['yhat'].to_numpy()


def test_arima(periods: int) -> (np.array, np.array):
    train, test = prepare_data_for_testing('data/train_115850.csv')

    preds = make_predictions_arima(train, periods)

    return test['unit_sales'].to_numpy(), preds.to_numpy()


if __name__ == '__main__':
    real_data, forecast = test_prophet(8)
    rmsle_value = rmsle(real_data, forecast)
    print(f'RMSLE for prophet: {rmsle_value}')

    real_data_2, forecast_2 = test_arima(8)
    rmsle_value_2 = rmsle(real_data_2, forecast_2)
    print(f'RMSLE for ARIMA: {rmsle_value_2}')
