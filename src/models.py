"""Small, explicit model families and training-only preprocessing."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET


class Persistence(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return X[TARGET].to_numpy()


class LinearTrend(Persistence):
    def __init__(self, horizon_hours=6):
        self.horizon_hours = horizon_hours

    def predict(self, X):
        return X[TARGET].to_numpy() + self.horizon_hours / 6 * X["wind_change_6h"].to_numpy()


def candidate_parameters(name, config):
    grids = {
        "persistence": {},
        "trend": {},
        "elasticnet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        "random_forest": {
            "n_estimators": [150, 250],
            "max_depth": [8, 14, None],
            "min_samples_leaf": [3, 8, 15],
            "max_features": [0.7, 1.0],
        },
        "hist_gradient_boosting": {
            "max_iter": [100, 180],
            "learning_rate": [0.04, 0.08],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [1.0, 5.0],
        },
        "torch_lstm": {"hidden_size": [32]},
        "torch_mlp": {"hidden_size": [64]},
    }
    if name not in grids:
        raise ValueError(f"Unknown model {name}; available: {list(grids)}")
    grid = grids[name]
    if not grid:
        return [{}]
    count = int(np.prod([len(v) for v in grid.values()]))
    return list(
        ParameterSampler(
            grid, n_iter=min(config["search_trials"], count), random_state=config["seed"]
        )
    )


def build_model(name, parameters, config, horizon_hours, sequence_columns):
    seed = config["seed"]
    if name == "persistence":
        return Persistence()
    if name == "trend":
        return LinearTrend(horizon_hours)
    if name == "elasticnet":
        estimator = ElasticNet(**parameters, max_iter=10000, random_state=seed)
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            **parameters, n_jobs=config.get("n_jobs", 2), random_state=seed
        )
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            **parameters, early_stopping=False, random_state=seed
        )
    elif name in {"torch_lstm", "torch_mlp"}:
        from .torch_models import TorchRegressor

        estimator = TorchRegressor(
            kind=name.removeprefix("torch_"),
            **parameters,
            n_steps=len(config["history_hours"]),
            epochs=config.get("torch_epochs", 35),
            seed=seed,
        )
    else:
        raise ValueError(name)
    # Imputer and scaler are refit in EVERY training fold.
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator
    )


def model_columns(name, all_columns, sequence_columns):
    return sequence_columns if name == "torch_lstm" else all_columns


def predict(model, X):
    result = np.asarray(model.predict(X), dtype=float).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model predictions")
    return np.maximum(0, result)
