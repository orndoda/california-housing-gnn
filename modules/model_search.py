from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # must be imported before HalvingGridSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer
from sklearn.base import clone
import warnings

warnings.filterwarnings("ignore")


def _make_model_grids(random_state: Optional[int] = None
                     ) -> List[Tuple[str, Pipeline, Dict[str, List[Any]]]]:
    """Return list of (name, pipeline, param_grid). Parameter keys use 'model__' prefix."""

    knn_pipe = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])
    rf_pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=random_state))])
    enet_pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet())])
    svr_pipe = Pipeline([("scaler", StandardScaler()), ("model", SVR())])

    knn_grid = {
        "model__n_neighbors": list(range(1, 32, 2)),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
        "model__leaf_size": [10, 20, 30, 40, 50],
        "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    }

    rf_grid = {
        "model__n_estimators": [100, 500, 1000],
        "model__max_depth": [None, 5, 10, 20, 50],
        "model__min_samples_split": [2, 5, 10, 15, 20],
        "model__min_samples_leaf": [1, 2, 4, 6, 8],
        "model__max_features": ["auto", "sqrt", "log2"],
        "model__bootstrap": [True, False],
    }

    enet_grid = {
        "model__alpha": list(np.logspace(-6, 2, 20)),
        "model__l1_ratio": list(np.linspace(0.0, 1.0, 21)),
        "model__tol": [1e-8, 1e-6, 1e-5, 1e-4, 1e-3],
    }

    return [
        ("KNeighborsRegressor", knn_pipe, knn_grid),
        ("RandomForestRegressor", rf_pipe, rf_grid),
        ("ElasticNet", enet_pipe, enet_grid),
    ]


def halving_grid_search_models_with_test(
    X_train,
    X_test,
    y_train,
    y_test,
    cv=5,
    verbose: int = 0,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    factor: int = 3,
    aggressive_elimination: bool = False,
) -> pd.DataFrame:
    """
    Run HalvingGridSearchCV using a single scoring string 'neg_mean_squared_error'.
    Returns a DataFrame with CV best (by neg MSE) and test-set evaluation.
    """
    models = _make_model_grids(random_state=random_state)
    scoring = "neg_mean_squared_error"  # single-string scoring required by user

    rows: List[Dict[str, Any]] = []

    for model_name, pipeline, param_grid in models:
        search = HalvingGridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit=False,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            factor=factor,
            aggressive_elimination=aggressive_elimination,
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_

        # Refit best params on full training set and evaluate on test set
        best_pipeline = clone(pipeline)
        best_pipeline.set_params(**best_params)
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)

        test_mse = float(mean_squared_error(y_test, y_pred))
        test_mape = float(mean_absolute_percentage_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        rows.append({
            "model": model_name,
            "metric": "mean_squared_error",
            "best_params": best_params,
            "test_mse": test_mse,
            "test_mape": test_mape,
            "test_r2": test_r2,
        })

    df = pd.DataFrame(rows)
    cols = ["model", "metric", "cv_mean_mse", "cv_std_mse", "best_params", "test_mse", "test_mape", "test_r2"]
    df = df[cols].sort_values(["model"], ascending=[True]).reset_index(drop=True)
    return df