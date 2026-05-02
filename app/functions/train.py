# train.py

# This script trains and evaluates a Kernel Ridge Regression model for each
# health variable independently. It is run once to generate the fitted models
# which are then loaded by the Flask app for predictions.
#
# These are the steps of the script and the corresponding modules that it calls:

#   (1) Prepares the data and creates time series splits       - splitter.py

#   (2) Runs a grid search using CrossValidator to find
#       the best hyperparameters                               - cross_validator.py
#                                                              - preprocessing.py
#                                                              - random_fourier_features.py
#                                                              - xgboost_wrapper.py   

#   (3) Fits the final model on the full 2013-2022
#       training data                                          - random_fourier_features.py
#                                                              - xgboost_wrapper.py 
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
# from random_fourier_features import RFFRidgeRegression

from splitter import Splitter, prepare_data          # data splitting
from preprocessing import Preprocessor, detect_categorical_columns  # data cleaning
from assessment import Assessment                       # MSE and R² scoring
from cross_validator import CrossValidator                   # CV loop
from random_fourier_features import RFFRidgeRegression    # ML model for all targets except SLEEP
from xgboost_wrapper import XGBoostWrapper              # ML model for SLEEP

# ── GLOBAL VARIABLES ─────────────────────────────────────────────────────────
DATA_PATH = "../../data/merged_final_transformed.csv"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TARGETS = ['CASTHMA', 'MHLTH', 'PHLTH', 'STROKE', 'SLEEP']
XGB_TARGETS = {"SLEEP"}
RFF_R_GRID     = [200, 500, 1000]
RFF_SIGMA_GRID = [0.1, 0.5, 1.0, 2.0]
RFF_LAMB_GRID  = [1e-5, 1e-3, 1e-1]
XGB_N_ESTIMATORS_GRID  = [100, 300]
XGB_MAX_DEPTH_GRID     = [4, 6, 8]
XGB_LEARNING_RATE_GRID = [0.01, 0.05, 0.1]


# ── DEFINED IN THIS FILE ──────────────────────────────────────────────────────

def save_model(target: str, model, preprocessor: Preprocessor) -> None:
    """
    Saves the fitted model and preprocessor as a .pkl file.
    Both are saved together so the Flask app can load everything it needs
    in one call without having to refit anything.

    Args:
        target (str):                 Health variable name e.g. 'COPD'
        model:                        Fitted model — either RFF or xgboost
        preprocessor (Preprocessor):  Fitted preprocessor — from preprocessing.py
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{target}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "preprocessor": preprocessor}, f)
    print(f"  Saved model → {path}")


def load_model(target: str) -> tuple:
    """
    Loads a fitted model and preprocessor from disk.
    Used by the Flask app to load pre-trained models without retraining.

    Args:
        target (str): Health variable name e.g. 'COPD'

    Returns:
        model: Fitted model
        preprocessor (Preprocessor): Fitted preprocessor
    """
    path = os.path.join(MODELS_DIR, f"{target}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found for {target} at {path}. "
            f"Run train.py first to generate the model files."
        )

    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["preprocessor"]


def tune_and_evaluate(df: pd.DataFrame, target: str) -> dict:
    """
    Runs the full training pipeline for a single target variable.
    Calls prepare_data() from splitter.py, CrossValidator from cross_validator.py,
    Preprocessor from preprocessing.py, RFFRegression from random_fourier_features.py,
    XGBoostWrapper from xgboost_wrapper.py, and Assessment from assessment.py.

    Args:
        df (pd.DataFrame):  Full dataset
        target (str):       Target health variable e.g. 'COPD'

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
    best_model_type = None
    # CrossValidator from cross_validator.py
    cv = CrossValidator(start_fold=start_fold)

    if target in XGB_TARGETS:
        # ── XGBoost grid search ────────────────────────────────────
        for n_est in XGB_N_ESTIMATORS_GRID:
            for depth in XGB_MAX_DEPTH_GRID:
                for lr in XGB_LEARNING_RATE_GRID:
                    preprocessor = Preprocessor(cat_col_indices=cat_col_indices)
                    model        = XGBoostWrapper(n_estimators=n_est, max_depth=depth, learning_rate=lr)
                    scores       = cv.cross_val_score(model, splitter, preprocessor)

                    if len(scores["mse"]) == 0:
                        continue

                    mean_mse = np.mean(scores["mse"])
                    if mean_mse < best_mse and not np.isnan(mean_mse):
                        best_mse        = mean_mse
                        best_r2_cv      = np.mean(scores["r2"])
                        best_params     = {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}
                        best_model_type = "xgboost"

    else:
        # ── RFF grid search ────────────────────────────────────────
        for sigma in RFF_SIGMA_GRID:
            for lamb in RFF_LAMB_GRID:
                for R in RFF_R_GRID:
                    preprocessor = Preprocessor(cat_col_indices=cat_col_indices)
                    model        = RFFRidgeRegression(sigma=sigma, lamb=lamb, R=R)
                    scores       = cv.cross_val_score(model, splitter, preprocessor)

                    if len(scores["mse"]) == 0:
                        continue

                    mean_mse = np.mean(scores["mse"])
                    if mean_mse < best_mse and not np.isnan(mean_mse):
                        best_mse        = mean_mse
                        best_r2_cv      = np.mean(scores["r2"])
                        best_params     = {"sigma": sigma, "lamb": lamb, "R": R}
                        best_model_type = "rff"

    print(f"  Model: {best_model_type} | Best params: {best_params} | CV MSE: {best_mse:.4f} | CV R²: {best_r2_cv:.4f}")

    # ── Final fit ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = splitter.get_test_split()

    preprocessor   = Preprocessor(cat_col_indices=cat_col_indices)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled  = preprocessor.transform(X_test)

    # ── Calibration split — hold out last full calendar year ──────
    # splitter.train_years = [2013..2022], already computed in Splitter.__init__
    cal_year  = splitter.train_years[-1]       # 2022
    fit_years = splitter.train_years[:-1]      # 2013–2021

    # Get the year for every row in X_train using the train mask
    train_mask  = np.isin(splitter.years, splitter.train_years)
    train_years = splitter.years[train_mask]   # aligned to X_train rows

    fit_mask = train_years <= fit_years[-1]
    cal_mask = train_years == cal_year

    X_fit, y_fit = X_train_scaled[fit_mask], y_train[fit_mask]
    X_cal, y_cal = X_train_scaled[cal_mask], y_train[cal_mask]

    print(f"  Fit years: {fit_years[0]}–{fit_years[-1]} ({fit_mask.sum()} rows) | "
          f"Cal year: {cal_year} ({cal_mask.sum()} rows)")

    # Rebuild best model and fit
    if best_model_type == "xgboost":
        model = XGBoostWrapper(**best_params)
    else:
        model = RFFRidgeRegression(**best_params)

    model.fit(X_fit, y_fit)
    model.calibrate(X_cal, y_cal) 
    y_pred = model.predict(X_test_scaled)
    
    # ── Save ──────────────────────────────────────────────────────
    save_model(target, model, preprocessor)

    # ── Final evaluation — assessment.py ─────────────────────────
    assessment = Assessment()
    mse = assessment.mean_squared_error(y_test, y_pred)
    r2  = assessment.r2_score(y_test, y_pred)

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


def print_summary(results: dict) -> None:
    print(f"\n{'='*95}")
    print(f"{'Target':<10} {'Best Params':<40} {'CV MSE':>10} {'CV R²':>10} {'Test MSE':>10} {'Test R²':>10}")
    print(f"{'-'*95}")
    for target, res in results.items():
        params_str = " | ".join(f"{k}={v}" for k, v in res["best_params"].items())
        print(f"{target:<10} "
              f"{params_str:<40} "
              f"{res['CV_MSE']:>10} "
              f"{res['CV_R2']:>10} "
              f"{res['Test_MSE']:>10} "
              f"{res['Test_R2']:>10}")
    print(f"{'-'*95}")
    print(f"{'Mean':<10} {'':40} "
          f"{np.mean([v['CV_MSE'] for v in results.values()]):>10.4f} "
          f"{np.mean([v['CV_R2'] for v in results.values()]):>10.4f} "
          f"{np.mean([v['Test_MSE'] for v in results.values()]):>10.4f} "
          f"{np.mean([v['Test_R2'] for v in results.values()]):>10.4f}")
    print(f"{'='*95}")


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
        results[target] = tune_and_evaluate(df, target)

    # ── Print summary ─────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()