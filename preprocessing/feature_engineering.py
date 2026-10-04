"""
TransitMind - shared paths, feature definitions and helpers.

Every model script imports from here so that the feature set, the train/test
split and the preprocessing are identical everywhere (no copy-paste drift).

Leakage rules
-------------
* The regression target (scheduled_trips_in_period) and the classification
  label (derived from trips_per_hour) are NOT allowed to appear as features,
  nor are other trip-count columns that add up to them.
* Rows of the same route are always kept on the same side of a split
  (GroupShuffleSplit / GroupKFold on route_id), because all route-structure
  features are identical across the four time periods of one route.
* Scaling / one-hot encoding happens inside sklearn Pipelines, so it is fitted
  on training data only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

OPERATIONAL_FILE = DATA_PROCESSED / "pmpml_operational_data.csv"
ROUTE_SUMMARY_FILE = DATA_PROCESSED / "pmpml_route_summary.csv"
REVENUE_FILE = DATA_PROCESSED / "pmpml_revenue_data.csv"
ML_DATASET_FILE = DATA_PROCESSED / "pmpml_ml_dataset.csv"

RANDOM_STATE = 42
TIME_PERIODS = ["Morning Peak", "Midday", "Evening Peak", "Off Peak"]
# Hours covered by each period (matches preprocessing/build_dataset.py)
PERIOD_HOURS = {"Morning Peak": 4, "Midday": 6, "Evening Peak": 4, "Off Peak": 14}
CLASS_LABELS = ["Low", "Medium", "High"]

# ---- model inputs -----------------------------------------------------------
NUMERIC_FEATURES = [
    "average_stops",            # stops per trip
    "average_trip_duration",    # minutes
    "unique_stops",             # distinct stops served by the route
    "max_stops_per_trip",
    "shape_count",              # distinct route shapes (variants / branches)
    "direction_count",
    "stops_per_minute",         # stop density along the trip
    "route_complexity",         # unique_stops / average_stops (variant spread)
    "period_hours",
    "mean_start_hour",
]
CATEGORICAL_FEATURES = ["time_period"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REGRESSION_TARGET = "scheduled_trips_in_period"
CLASSIFICATION_SOURCE = "trips_per_hour"      # label is a tercile of this
CLASSIFICATION_TARGET = "activity_class"


def check_file(path, hint):
    if not Path(path).exists():
        sys.exit(f"ERROR: missing file {path}\n       {hint}")


def load_ml_dataset():
    check_file(ML_DATASET_FILE, "Run: python preprocessing\\create_ml_dataset.py")
    df = pd.read_csv(ML_DATASET_FILE, dtype={"route_id": str, "route_short_name": str})
    missing = [c for c in FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET, "route_id"]
               if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: ML dataset lacks columns {missing}. Re-run create_ml_dataset.py")
    if len(df) < 30 or df["route_id"].nunique() < 10:
        sys.exit(f"ERROR: only {len(df)} rows / {df['route_id'].nunique()} routes - "
                 "too little data for a train/test split.")
    return df


def group_split(df, test_size=0.25):
    """Train/test split that keeps every route entirely in one side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["route_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def group_cv(train_df, n_splits=5):
    """List of (train, valid) index pairs, grouped by route. Reusable by sklearn."""
    n_splits = min(n_splits, train_df["route_id"].nunique())
    return list(GroupKFold(n_splits=n_splits).split(train_df, groups=train_df["route_id"]))


def make_preprocessor(scale=True, numeric=None):
    numeric = numeric or NUMERIC_FEATURES
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def make_pipeline(model, scale=True, numeric=None):
    return Pipeline([("prep", make_preprocessor(scale, numeric)), ("model", model)])


def add_derived_features(df):
    """Derived transport features (used by the dataset builder and the dashboard)."""
    out = df.copy()
    out["stops_per_minute"] = out["average_stops"] / out["average_trip_duration"].clip(lower=1)
    out["route_complexity"] = out["unique_stops"] / out["average_stops"].clip(lower=1)
    out["period_hours"] = out["time_period"].map(PERIOD_HOURS)
    return out


def class_from_terciles(values, edges=None):
    """Transparent labels: Low / Medium / High from the 33rd and 67th percentiles."""
    values = pd.Series(values)
    if edges is None:
        edges = (values.quantile(1 / 3), values.quantile(2 / 3))
    lab = np.where(values <= edges[0], "Low", np.where(values <= edges[1], "Medium", "High"))
    return pd.Series(lab, index=values.index), edges
