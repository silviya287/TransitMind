"""
TransitMind - Regression comparison

Target : scheduled_trips_in_period  (number of GTFS-scheduled trips starting in a
         route's time period).  This is SCHEDULED SERVICE ACTIVITY, not passenger
         demand - the GTFS feed contains no passenger counts.
Inputs : route structure (stops, duration, shapes, ...) + time period.
Split  : 75/25, whole routes kept together (no route appears in both sets).
Output : data/processed/regression_results.csv, models/regression_models.joblib
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.feature_engineering import (  # noqa: E402
    DATA_PROCESSED, FEATURES, MODELS_DIR, NUMERIC_FEATURES, RANDOM_STATE,
    REGRESSION_TARGET, group_cv, group_split, load_ml_dataset, make_pipeline)

RESULTS_FILE = DATA_PROCESSED / "regression_results.csv"
MODEL_FILE = MODELS_DIR / "regression_models.joblib"


def best_single_feature(train):
    """Simple linear regression needs ONE predictor: pick the most correlated on TRAIN only."""
    corr = train[NUMERIC_FEATURES].corrwith(train[REGRESSION_TARGET]).abs().fillna(0)
    return corr.idxmax()


def build_models(train):
    single = best_single_feature(train)
    poly_numeric = Pipeline([("scale", StandardScaler()),
                             ("poly", PolynomialFeatures(degree=2, include_bias=False))])
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    poly_prep = ColumnTransformer([
        ("num", poly_numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["time_period"])])
    simple_prep = ColumnTransformer([("one", StandardScaler(), [single])])
    return {
        f"Linear Regression (1 feature: {single})":
            Pipeline([("prep", simple_prep), ("model", LinearRegression())]),
        "Multiple Linear Regression": make_pipeline(LinearRegression()),
        "Polynomial Regression (deg 2, Ridge)":
            Pipeline([("prep", poly_prep), ("model", Ridge(alpha=10.0))]),
        "Decision Tree": make_pipeline(
            DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE),
            scale=False),
        "Random Forest": make_pipeline(
            RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3,
                                  random_state=RANDOM_STATE, n_jobs=1), scale=False),
        "SVR (RBF)": make_pipeline(TransformedTargetRegressor(
            regressor=SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
            transformer=StandardScaler())),
        "Ridge": make_pipeline(RidgeCV(alphas=np.logspace(-2, 3, 12))),
        "Lasso": make_pipeline(LassoCV(alphas=np.logspace(-3, 1, 12), cv=5,
                                       random_state=RANDOM_STATE, max_iter=20000)),
    }


def main():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print(f"TransitMind - Regression (target: {REGRESSION_TARGET})")
    print("=" * 60)
    df = load_ml_dataset()
    if df[REGRESSION_TARGET].nunique() < 5:
        sys.exit("ERROR: regression target has too few distinct values.")
    train, test = group_split(df)
    print(f"Train rows: {len(train):,} ({train.route_id.nunique()} routes) | "
          f"Test rows: {len(test):,} ({test.route_id.nunique()} routes)")
    X_tr, y_tr = train[FEATURES], train[REGRESSION_TARGET]
    X_te, y_te = test[FEATURES], test[REGRESSION_TARGET]
    cv = group_cv(train)

    rows, fitted = [], {}
    for name, model in build_models(train).items():
        t0 = time.time()
        try:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="r2").mean()
        except Exception as exc:                       # keep going, report clearly
            print(f"  ! {name} failed: {exc}")
            continue
        mse = mean_squared_error(y_te, pred)
        rows.append({"model": name, "MAE": mean_absolute_error(y_te, pred), "MSE": mse,
                     "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_te, pred),
                     "CV_R2_mean": cv_r2, "train_seconds": round(time.time() - t0, 2),
                     "target": REGRESSION_TARGET})
        fitted[name] = model
        print(f"  {name:<48} MAE={rows[-1]['MAE']:.2f}  RMSE={rows[-1]['RMSE']:.2f}  "
              f"R2={rows[-1]['R2']:.3f}")
    if not rows:
        sys.exit("ERROR: no regression model could be trained.")

    res = pd.DataFrame(rows).sort_values("R2", ascending=False).round(4)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_FILE, index=False)
    best = res.iloc[0]["model"]
    joblib.dump({"models": fitted, "best": best, "target": REGRESSION_TARGET,
                 "features": FEATURES}, MODEL_FILE)
    print(f"\nBest model by test R2: {best}")
    print(f"Saved {RESULTS_FILE}\nSaved {MODEL_FILE}")


if __name__ == "__main__":
    main()
