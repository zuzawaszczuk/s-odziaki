import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate sample historical sales data
def generate_sample_data():
    data = {
        'date': pd.date_range(start='2022-01-01', periods=365, freq='D'),
        'product_id': np.random.choice([1, 2, 3, 4, 5], size=365),
        'sales_quantity': np.random.poisson(lam=20, size=365)  # Poisson distribution for sales
    }
    return pd.DataFrame(data)

# Load your historical data
df = generate_sample_data()

# Preprocess the data
df['day_of_year'] = df['date'].dt.dayofyear
df['product_id'] = df['product_id'].astype('category').cat.codes  # Convert product_id to numeric

# Features and target variable
X = df[['day_of_year', 'product_id']]
y = df['sales_quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Function to predict future demand
def predict_demand(model, future_dates, product_id):
    day_of_year = [date.dayofyear for date in future_dates]
    product_id_encoded = [product_id] * len(future_dates)
    future_X = pd.DataFrame({'day_of_year': day_of_year, 'product_id': product_id_encoded})
    return model.predict(future_X)

# Example of predicting demand for the next 30 days for product_id = 1
future_dates = pd.date_range(start='2023-01-01', periods=30)
future_demand = predict_demand(model, future_dates, product_id=1)
print(f"Predicted demand for product_id 1 for the next 30 days: {future_demand}")