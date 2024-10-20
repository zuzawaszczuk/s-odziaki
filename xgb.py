import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import process_data
import xgboost as xgb


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

results = reg.evals_result()
epochs = len(results['validation_1']['rmsle'])
x_axis = range(0, epochs)

plt.figure()
plt.plot(x_axis, results['validation_0']['rmsle'], label='Train RMSLE')
plt.plot(x_axis, results['validation_1']['rmsle'], label='Test RMSLE')
plt.legend()
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('RMSLE')
plt.title('XGBoost RMSLE Performance')
plt.savefig('xgboost_rmsle_performance.png')  # Save the figure
plt.close()

predictions = reg.predict(X_test)
df = pd.DataFrame(data=y_test, columns=['unit_sales'])
df['prediction'] = predictions

ax = df.plot(figsize=(15, 6))
df['prediction'].plot(ax=ax, style='.', color='red')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.ylabel('Unit Sales')
plt.savefig('xgboost_prediction_comparision.png')
plt.close()
