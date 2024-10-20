import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import process_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error

import xgboost as xgb

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmsle_value = np.sqrt(mean_squared_log_error(y_test, predictions))
    return predictions, model

X_train, y_train, X_test, y_test = process_data("data/train_bread.csv")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Decision Tree': DecisionTreeRegressor()
}

predictions_df = pd.DataFrame(y_test, columns=['unit_sales'])

# Evaluate each model
for name, model in models.items():
    predictions, fitted_model = evaluate_model(model, X_train, y_train, X_test, y_test)
    predictions_df[name] = predictions

plt.figure(figsize=(15, 5))
for name in models.keys():
    plt.plot(predictions_df[name], label=f'{name} Predictions', linestyle='--', marker='.')

plt.plot(predictions_df['unit_sales'], label='Truth Data', color='black')
plt.legend()
plt.title('Raw Data and Predictions from Different Models')
plt.xlabel('Date')
plt.ylabel('Unit Sales')
plt.savefig('predictions_comparison.png')  # Save the figure
plt.close()  # Close the figure to avoid displaying it