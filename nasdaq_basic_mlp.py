import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from helpers import load_stock_data

class NasdaqBasicMLPRunner:
    def __init__(
        self,
        start_train="1990-01-01",
        end_train="2022-12-31",
        start_test="2023-01-01",
        end_test="2023-06-30",
        lr=0.3,
        epochs=1000,
        log_every_epochs=100,
    ):
        self.lr = lr
        self.epochs = epochs
        self.log_every_epochs = log_every_epochs
        self._setup_data(start_train, end_train, start_test, end_test)

        self.model = nn.Sequential(
            nn.Linear(self.x_train.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _setup_data(self, start_train, end_train, start_test, end_test):
        self.data = load_stock_data("nasdaq_19900101_20230630")
        self.data_train = self.data.loc[start_train:end_train].dropna()
        self.data_test = self.data.loc[start_test:end_test].dropna()
        self.scaler = StandardScaler()
        self.x_train = self.data_train.drop("close", axis=1).values
        self.y_train = self.data_train["close"].values
        self.x_test = self.data_test.drop("close", axis=1).values
        self.y_test = self.data_test["close"].values
        self.x_scaled_train = self.scaler.fit_transform(self.x_train)
        self.x_scaled_test = self.scaler.transform(self.x_test)
        self.x_train_torch = torch.from_numpy(self.x_scaled_train.astype(np.float32))
        self.y_train_torch = torch.from_numpy(self.y_train.reshape(-1, 1).astype(np.float32))
        self.x_test_torch = torch.from_numpy(self.x_scaled_test.astype(np.float32))
        self.y_test_torch = torch.from_numpy(self.y_test.reshape(-1, 1).astype(np.float32))

    def train_step(self):
        self.model.train()
        pred_train = self.model(self.x_train_torch)
        loss = self.loss_fn(pred_train, self.y_train_torch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            loss = self.train_step()
            if epoch % self.log_every_epochs == 0:
                print(f"Epoch {epoch} Loss: {loss}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def test(self):
        pred = self.predict(self.x_test_torch).detach().numpy()[:, 0]
        return {
            "mse": mean_squared_error(self.y_test, pred),
            "mae": mean_absolute_error(self.y_test, pred),
            "r2": r2_score(self.y_test, pred),
        }

if __name__ == "__main__":
    runner = NasdaqBasicMLPRunner(epochs=3000)
    runner.train()
    print(runner.test())
