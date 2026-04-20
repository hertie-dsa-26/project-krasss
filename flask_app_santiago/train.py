# train.py

# This script trains and evaluates a Kernel Ridge Regression model for each
# health variable independently. It is run once to generate the fitted models
# which are then loaded by the Flask app for predictions.
#
# These are the steps of the script and the corresponding modules that it calls:

#   (1) Prepares the data and creates time series splits       - splitter.py

#   (2) Runs a grid search using CrossValidator to find
#       the best hyperparameters                               - cross_validator.py
#                                                                 preprocessing.py
#                                                                 krr.py

#   (3) Fits the final model on the full 2013-2022
#       training data                                          - krr.py
#                                                              - preprocessing.py

#   (4) Evaluates on the held-out 2023 test set               - assessment.py

#   (5) Saves the fitted model and preprocessor to disk
#       as a .pkl file                                         - pickle (built-in)


# This file defines:
#   save_model()        — saves a fitted model and preprocessor to disk
#   load_model()        — loads a fitted model and preprocessor from disk
#   tune_and_evaluate() — runs the full pipeline for one target variable
#   print_summary()     — prints a formatted results table
#   main()              — entry point, loops over all target variables
#
# Output: one .pkl file per target variable saved to MODELS_DIR
# These files are loaded by the Flask app via load_model()

import numpy as np
import pandas as pd
import pickle
import os

from splitter import Splitter, prepare_data          # data splitting
from preprocessing import Preprocessor, detect_categorical_columns  # data cleaning
from krr import KernelRidgeRegression           # the ML model
from assessment import Assessment                       # MSE and R² scoring
from cross_validator import CrossValidator                   # CV loop

# ── GLOBAL VARIABLES ─────────────────────────────────────────────────────────
DATA_PATH = "data/merged_final_transformed.csv"
MODELS_DIR = "models"
TARGETS = ['BPHIGH', 'CASTHMA', 'COPD', 'MHLTH', 'PHLTH', 'STROKE', 'SLEEP']
LAMB_GRID = [1e-2, 1e-1, 1.0, 10.0]
SIGMA2_GRID = [50.0, 75.0, 100.0, 150.0, 200.0]


# ── DEFINED IN THIS FILE ──────────────────────────────────────────────────────

def save_model(target, krr, preprocessor):
    """
    Saves the fitted KRR model and preprocessor as a .pkl file.
    Both are saved together so the Flask app can load everything it needs
    in one call without having to refit anything.

    Args:
        target (str):                 Health variable name e.g. 'COPD'
        krr (KernelRidgeRegression):  Fitted model — from krr.py
        preprocessor (Preprocessor):  Fitted preprocessor — from preprocessing.py
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{target}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"krr": krr, "preprocessor": preprocessor}, f)
    print(f"  Saved model → {path}")


def load_model(target):
    """
    Loads a fitted KRR model and preprocessor from disk.
    Used by the Flask app to load pre-trained models without retraining.

    Args:
        target (str): Health variable name e.g. 'COPD'

    Returns:
        krr (KernelRidgeRegression): Fitted model
        preprocessor (Preprocessor): Fitted preprocessor
    """
    path = os.path.join(MODELS_DIR, f"{target}.pkl")
    assert os.path.exists(path), f"No saved model found for {target} at {path}"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["krr"], bundle["preprocessor"]


def tune_and_evaluate(df, target, lamb_grid, sigma2_grid):
    """
    Runs the full training pipeline for a single target variable.
    Calls prepare_data() from splitter.py, CrossValidator from cross_validator.py,
    Preprocessor from preprocessing.py, KernelRidgeRegression from krr.py,
    and Assessment from assessment.py.

    Args:
        df (pd.DataFrame):  Full dataset
        target (str):       Target health variable e.g. 'COPD'
        lamb_grid (list):   Regularisation values to search
        sigma2_grid (list): Kernel bandwidth values to search

    Returns:
        dict: best_params, CV_MSE, CV_R2, Test_MSE, Test_R2
    """
    # ── Data preparation — splitter.py ────────────────────────────
    # prepare_data() from splitter.py
    X, y, years = prepare_data(df, target)
    cat_col_indices = detect_categorical_columns(X)  # from preprocessing.py
    # Splitter from splitter.py
    splitter = Splitter(X.to_numpy(), y, years)

    # ── Determine start_fold ──────────────────────────────────────
    # Always use the last 3 stable folds for hyperparameter selection.
    # Early folds have small training sets and produce unreliable scores.
    n_folds = len(splitter.time_series_splits())
    start_fold = max(0, n_folds - 3)
    print(f"\n  Folds: {n_folds} | Scoring from fold: {start_fold}")

    # ── Grid search — cross_validator.py + krr.py + preprocessing.py ─
    # CrossValidator handles the CV loop and preprocessing per fold.
    # For each combination of lamb and sigma2 it fits the model on each
    # training fold and scores it on the validation fold.
    # Assessment is only used at the end for final test scoring.

    best_mse = np.inf
    best_r2_cv = None
    best_params = {}
    # CrossValidator from cross_validator.py
    cv = CrossValidator(start_fold=start_fold)

    for lamb in lamb_grid:
        for sigma2 in sigma2_grid:
            preprocessor = Preprocessor(
                cat_col_indices=cat_col_indices)  # from preprocessing.py
            krr = KernelRidgeRegression(
                lamb=lamb, sigma2=sigma2)  # from krr.py
            scores = cv.cross_val_score(krr, splitter, preprocessor)

            if len(scores["mse"]) == 0:
                continue

            mean_mse = np.mean(scores["mse"])
            if mean_mse < best_mse and not np.isnan(mean_mse):
                best_mse = mean_mse
                best_r2_cv = np.mean(scores["r2"])
                best_params = {"lamb": lamb, "sigma2": sigma2}

    print(
        f"  Best params: {best_params} | CV MSE: {best_mse:.4f} | CV R²: {best_r2_cv:.4f}")

    # ── Final fit — krr.py + preprocessing.py ────────────────────
    # Refit on the full 2013-2022 training set using the best hyperparameters.
    # This is the model that gets saved and used for Flask predictions.
    X_train, X_test, y_train, y_test = splitter.get_test_split()  # from splitter.py

    preprocessor = Preprocessor(
        cat_col_indices=cat_col_indices)  # from preprocessing.py
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    krr = KernelRidgeRegression(**best_params)  # from krr.py
    krr.fit(X_train_scaled, y_train)
    y_pred = krr.predict(X_test_scaled)

    # ── Save — pickle ─────────────────────────────────────────────
    save_model(target, krr, preprocessor)

    # ── Final evaluation — assessment.py ─────────────────────────
    # Assessment is used here only for final scoring — not for CV
    # from assessment.py
    assessment = Assessment()
    mse = assessment.mean_squared_error(y_test, y_pred)
    r2 = assessment.r2_score(y_test, y_pred)

    print(f"  Test MSE: {mse:.4f} | Test R²: {r2:.4f}")
    print(f"  y_test=[{y_test.min():.2f}, {y_test.max():.2f}] "
          f"y_pred=[{y_pred.min():.2f}, {y_pred.max():.2f}]")

    return {
        "best_params": best_params,
        "CV_MSE":      round(best_mse, 4),
        "CV_R2":       round(best_r2_cv, 4),
        "Test_MSE":    round(mse, 4),
        "Test_R2":     round(r2, 4)
    }


def print_summary(results):
    """
    Prints a formatted summary table of results across all targets.
    """
    print(f"\n{'='*85}")
    print(f"{'Target':<10} {'lamb':>8} {'sigma2':>8} {'CV MSE':>10} {'CV R²':>10} {'Test MSE':>10} {'Test R²':>10}")
    print(f"{'-'*85}")
    for target, res in results.items():
        print(f"{target:<10} "
              f"{res['best_params']['lamb']:>8} "
              f"{res['best_params']['sigma2']:>8} "
              f"{res['CV_MSE']:>10} "
              f"{res['CV_R2']:>10} "
              f"{res['Test_MSE']:>10} "
              f"{res['Test_R2']:>10}")
    print(f"{'-'*85}")
    print(f"{'Mean':<10} {'':>8} {'':>8} "
          f"{np.mean([v['CV_MSE'] for v in results.values()]):>10.4f} "
          f"{np.mean([v['CV_R2'] for v in results.values()]):>10.4f} "
          f"{np.mean([v['Test_MSE'] for v in results.values()]):>10.4f} "
          f"{np.mean([v['Test_R2'] for v in results.values()]):>10.4f}")
    print(f"{'='*85}")


def main():
    # ── Load data ─────────────────────────────────────────────────
    print("Loading data")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Pipeline for each target ──────────────────────────────

    # tune_and_evaluate() is defined in this file and calls all other modules
    results = {}
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")
        results[target] = tune_and_evaluate(df, target, LAMB_GRID, SIGMA2_GRID)

    # ── Print summary ─────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
