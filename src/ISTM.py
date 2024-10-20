import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler


def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback])
    return torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float()


class SalesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=2, batch_first=True)
        self.linear = nn.Linear(100, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


if __name__ == '__main__':
    df = pd.read_csv('data/train_bread.csv')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df.set_index('date', inplace=True)
    df = df.groupby('item_nbr')['unit_sales'].resample('ME').sum()
    df = df.reset_index()
    timeseries = df[['unit_sales']].values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    timeseries_scaled = scaler.fit_transform(timeseries)

    train_size = int(len(timeseries_scaled) * 0.67)
    test_size = len(timeseries_scaled) - train_size
    train, test = timeseries_scaled[:train_size], timeseries_scaled[train_size:]

    lookback = 8
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    model = SalesModel()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    # train_rmse_arr = []

    n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    with torch.no_grad():
        train_plot = np.ones(timeseries.shape[0]) * np.nan
        y_pred_train = model(X_train).squeeze().numpy()
        y_pred_train = y_pred_train.reshape(-1, 1)
        train_plot[lookback:train_size] = scaler.inverse_transform(y_pred_train).reshape(-1)

        test_plot = np.ones(timeseries.shape[0]) * np.nan
        y_pred_test = model(X_test).squeeze().numpy()
        y_pred_test = y_pred_test.reshape(-1, 1)
        test_plot[train_size + lookback:len(timeseries)] = scaler.inverse_transform(y_pred_test).reshape(-1)

    plt.plot(scaler.inverse_transform(timeseries_scaled), c='b', label='Actual')
    plt.plot(train_plot, c='r', label='Train Predictions')
    plt.plot(test_plot, c='g', label='Test Predictions')
    plt.legend()
    # plt.savefig('sine_wave.png', format='png')
    plt.show()
