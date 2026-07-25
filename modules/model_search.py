from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

from typing import Optional, List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def _make_model_grids(random_state: Optional[int] = None
                     ) -> List[Tuple[str, Pipeline, Dict[str, List[Any]]]]:
    """Return list of (name, pipeline, param_grid). Parameter keys use 'model__' prefix."""

    knn_pipe = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])
    rf_pipe  = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=random_state))])
    enet_pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet())])
    svr_pipe = Pipeline([("scaler", StandardScaler()), ("model", SVR())])

    # Much larger KNN grid
    knn_grid = {
        "model__n_neighbors": list(range(1, 32, 2)),        # 1,3,5,...,31
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],                                 # Manhattan, Euclidean
        "model__leaf_size": [10, 20, 30, 40, 50],
        "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    }

    # Much larger RandomForest grid
    rf_grid = {
        "model__n_estimators": [100, 200, 500, 800, 1000],
        "model__max_depth": [None, 5, 10, 20, 30, 50],
        "model__min_samples_split": [2, 5, 10, 15, 20],
        "model__min_samples_leaf": [1, 2, 4, 6, 8],
        "model__max_features": ["auto", "sqrt", "log2", 0.1, 0.2, 0.5],
        "model__bootstrap": [True, False],
    }

    # ElasticNet (wide range)
    enet_grid = {
        # alpha on a log scale from very small to large
        "model__alpha": list(np.logspace(-6, 2, 20)),            # 1e-6 ... 1e2
        # l1_ratio from pure ridge (0) to pure lasso (1)
        "model__l1_ratio": list(np.linspace(0.0, 1.0, 21)),      # 0.0, 0.05, ..., 1.0
        "model__fit_intercept": [True, False],
        "model__max_iter": [500, 1000, 5000, 10000],
        "model__tol": [1e-8, 1e-6, 1e-5, 1e-4, 1e-3],
        "model__positive": [False, True],                        # constrain coefficients to be non-negative
        "model__warm_start": [False, True],
    }

    # Much larger SVR grid
    svr_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100, 1000],
        "model__kernel": ["rbf", "linear", "poly", "sigmoid"],
        "model__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0],
        "model__epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5],
        "model__degree": [2, 3, 4],         # used when kernel='poly'
        "model__coef0": [0.0, 0.1, 0.5],    # used for poly/sigmoid
    }

    return [
        ("KNeighborsRegressor", knn_pipe, knn_grid),
        ("RandomForestRegressor", rf_pipe, rf_grid),
        ("ElasticNet", enet_pipe, enet_grid),
        ("SVR", svr_pipe, svr_grid),
    ]

def _scoring_dict() -> Dict[str, Any]:
    """
    Scoring dict for GridSearchCV. Use sklearn's negative-error scorers so higher is better.
    Keys are the names that will appear in cv_results_ as mean_test_<key>.
    """
    return {
        "neg_mean_squared_error": make_scorer(mean_squared_error, greater_is_better=False),
        "neg_mean_absolute_percentage_error": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "r2": make_scorer(r2_score, greater_is_better=True),
    }


def grid_search_models_with_test(
    X_train,
    X_test,
    y_train,
    y_test,
    cv=5,
    verbose: int = 0,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Run grid search for multiple models and evaluate best configurations on the provided test set.

    Args:
        X_train, X_test: feature matrices (numpy arrays or pandas DataFrames).
        y_train, y_test: target arrays/Series.
        cv: int or CV splitter for GridSearchCV.
        verbose: verbosity passed to GridSearchCV.
        n_jobs: jobs for GridSearchCV.
        random_state: seed used for models that accept it (e.g., RandomForest).

    Returns:
        pandas.DataFrame with one row per (model, metric) containing:
            - model: model name
            - metric: metric used to select best params (friendly name)
            - best_params: dict of best hyperparameters found for that metric
            - cv_mean_score: mean CV score (converted to positive for error metrics)
            - cv_std_score: std CV score (converted to positive for error metrics)
            - test_mse: MSE on the provided test set (computed with mean_squared_error)
            - test_mape: MAPE on the provided test set (computed with mean_absolute_percentage_error)
            - test_r2: R2 on the provided test set (computed with r2_score)
    """
    models = _make_model_grids(random_state=random_state)
    scoring = _scoring_dict()
    scoring_keys = list(scoring.keys())  # e.g. ['neg_mean_squared_error', 'neg_mean_absolute_percentage_error', 'r2']

    rows = []

    for model_name, pipeline, param_grid in models:
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit=False,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            return_train_score=False,
        )

        gs.fit(X_train, y_train)
        cv_results = gs.cv_results_

        for score_key in scoring_keys:
            mean_key = f"mean_test_{score_key}"
            std_key = f"std_test_{score_key}"

            # Choose best index (argmax because scoring uses negative errors for error metrics)
            best_idx = int(np.argmax(cv_results[mean_key]))
            best_params = cv_results["params"][best_idx]
            raw_mean = cv_results[mean_key][best_idx]
            raw_std = cv_results[std_key][best_idx]

            # Convert raw negative metrics to positive error values for readability
            if score_key.startswith("neg_"):
                display_metric = score_key.replace("neg_", "")
                cv_mean = -raw_mean
                cv_std = abs(raw_std)
            else:
                display_metric = score_key
                cv_mean = raw_mean
                cv_std = raw_std

            # Build a fresh pipeline, set best params, fit on full training data, evaluate on test set
            best_pipeline = clone(pipeline)
            # set_params expects keys like 'model__param'
            best_pipeline.set_params(**best_params)
            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)

            test_mse = float(mean_squared_error(y_test, y_pred))
            test_mape = float(mean_absolute_percentage_error(y_test, y_pred))
            test_r2 = float(r2_score(y_test, y_pred))

            rows.append({
                "model": model_name,
                "metric": display_metric,
                "best_params": best_params,
                "cv_mean_score": float(cv_mean),
                "cv_std_score": float(cv_std),
                "test_mse": test_mse,
                "test_mape": test_mape,
                "test_r2": test_r2,
            })

    df = pd.DataFrame(rows)
    # Order columns for readability
    cols = ["model", "metric", "cv_mean_score", "cv_std_score", "best_params", "test_mse", "test_mape", "test_r2"]
    df = df[cols]
    # Sort by model then metric
    df = df.sort_values(["model", "metric"], ascending=[True, True]).reset_index(drop=True)
    return df
