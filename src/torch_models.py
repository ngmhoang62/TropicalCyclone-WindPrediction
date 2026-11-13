"""Optional CPU-reproducible PyTorch models with fixed training budgets."""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SequenceNetwork(nn.Module):
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.recurrent = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, value):
        output, _ = self.recurrent(value)
        return self.output(output[:, -1]).squeeze(-1)


class TorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, kind="lstm", hidden_size=32, n_steps=4, epochs=35, seed=42):
        self.kind = kind
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.epochs = epochs
        self.seed = seed

    def _tensor(self, X):
        value = np.asarray(X, dtype=np.float32)
        if self.kind == "lstm":
            # Feature builder orders blocks newest-to-oldest; LSTM sees oldest first.
            value = value.reshape(len(value), self.n_steps, -1)[:, ::-1, :].copy()
        return torch.from_numpy(value)

    def fit(self, X, y):
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.use_deterministic_algorithms(True)
        values = self._tensor(X)
        self.n_features_in_ = np.asarray(X).shape[1]
        labels = np.asarray(y, dtype=np.float32)
        self.target_mean_ = float(labels.mean())
        self.target_scale_ = max(float(labels.std()), 1e-6)
        labels = torch.from_numpy((labels - self.target_mean_) / self.target_scale_)
        if self.kind == "lstm":
            self.network_ = SequenceNetwork(values.shape[-1], self.hidden_size)
        elif self.kind == "mlp":
            self.network_ = nn.Sequential(
                nn.Linear(values.shape[-1], self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 1),
                nn.Flatten(0),
            )
        else:
            raise ValueError(self.kind)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=0.001, weight_decay=0.01)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(values, labels),
            batch_size=128,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        self.training_loss_ = []
        self.network_.train()
        # Epoch budget is frozen before the holdout; no test-driven early stopping.
        for _ in range(self.epochs):
            total = 0.0
            for batch, target in loader:
                optimizer.zero_grad()
                output = self.network_(batch)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            self.training_loss_.append(total / len(values))
        self.network_.eval()
        return self

    def predict(self, X):
        self.network_.eval()
        with torch.inference_mode():
            result = self.network_(self._tensor(X)).numpy()
        return result * self.target_scale_ + self.target_mean_
