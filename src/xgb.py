import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import process_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

def xgb_predict():
        # Get data
        X_train, y_train, X_test, y_test = process_data("data/train_bread.csv")

        # Define model
        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                        n_estimators=1000,
                        early_stopping_rounds=50,
                        max_depth=3,
                        learning_rate=0.01,
                        eval_metric='rmsle')

        # Train model
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100)

        future_dates = pd.date_range(start='2017-11-01', periods=4, freq='D')
        future_data = pd.DataFrame({'date': future_dates})
        future_data['dayofweek'] = future_data['date'].dt.dayofweek
        future_data['month'] = future_data['date'].dt.month
        future_data['quarter'] = future_data['date'].dt.quarter
        future_data['year'] = future_data['date'].dt.year
        future_data['dayofyear'] = future_data['date'].dt.dayofyear
        future_data['dayofmonth'] = future_data['date'].dt.day
        future_data['weekofyear'] = future_data['date'].dt.isocalendar().week

        X_future = future_data.drop(columns=['date'])
        future_predictions = reg.predict(X_future)
        df = pd.DataFrame(data=X_future, columns=['ds'])
        df['yhat'] = future_predictions

        return df, df