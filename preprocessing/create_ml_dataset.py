"""
TransitMind - Build the ML-ready dataset (unit = route + time period)

Input : data/processed/pmpml_operational_data.csv  (GTFS stop-times, 627k rows)
        data/processed/pmpml_revenue_data.csv       (optional, 2019, routes 2 and 27 only)
Output: data/processed/pmpml_ml_dataset.csv

The 627k stop-time rows are first collapsed to ONE row per trip (~15k) and
then to one row per route x time period (<= 4 x 314).  Only the columns that
are needed are read, so peak memory stays well below 1 GB.

What is real and what is a proxy
--------------------------------
* trips, stops, durations, hours  -> scheduled GTFS data (real, but *schedule*, not ridership)
* scheduled_trips_in_period       -> number of scheduled trips starting in that period
* trips_per_hour                  -> trips / hours in the period (service frequency)
* buses_required_proxy            -> trips_per_hour x average duration / 60
                                     (vehicles needed to run the schedule; Little's law)
* hist_* revenue columns          -> 2019 revenue for routes 2 and 27 ONLY. It is a
                                     historical route/time-period signal, NOT a join on
                                     service date and NOT passenger demand.  Because it
                                     covers 2 of ~314 routes it is kept for the dashboard
                                     and Q-learning context but is NOT a model feature.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.feature_engineering import (  # noqa: E402
    CLASSIFICATION_SOURCE, CLASSIFICATION_TARGET, ML_DATASET_FILE, OPERATIONAL_FILE,
    PERIOD_HOURS, REVENUE_FILE, TIME_PERIODS, add_derived_features, check_file,
    class_from_terciles)

USECOLS = ["trip_id", "route_id", "route_short_name", "shape_id", "direction_id",
           "stop_id", "stop_sequence", "arrival_time", "arrival_hour", "time_period"]


def build_trip_table():
    check_file(OPERATIONAL_FILE, "Run: python preprocessing\\build_dataset.py")
    head = pd.read_csv(OPERATIONAL_FILE, nrows=1)
    missing = [c for c in USECOLS if c not in head.columns]
    if missing:
        sys.exit(f"ERROR: operational data lacks columns {missing}")

    df = pd.read_csv(
        OPERATIONAL_FILE, usecols=USECOLS, low_memory=False,
        dtype={"trip_id": str, "route_id": str, "route_short_name": str,
               "time_period": str})
    n_rows = len(df)
    df["arrival_time"] = pd.to_timedelta(df["arrival_time"], errors="coerce")
    bad = df["arrival_time"].isna() | df["stop_sequence"].isna() | df["route_id"].isna()
    if bad.any():
        print(f"Warning: dropping {int(bad.sum()):,} stop-times with invalid time/route/sequence")
        df = df[~bad]
    if df.empty:
        sys.exit("ERROR: no valid stop-time rows.")

    df = df.sort_values(["trip_id", "stop_sequence"])
    g = df.groupby("trip_id", sort=False)
    trips = g.agg(
        route_id=("route_id", "first"),
        route_short_name=("route_short_name", "first"),
        shape_id=("shape_id", "first"),
        direction_id=("direction_id", "first"),
        time_period=("time_period", "first"),          # period of the first stop
        start_hour=("arrival_hour", "first"),
        stop_count=("stop_id", "nunique"),
        first_time=("arrival_time", "min"),
        last_time=("arrival_time", "max"),
    ).reset_index()
    trips["duration_min"] = (trips["last_time"] - trips["first_time"]).dt.total_seconds() / 60
    trips = trips[trips["duration_min"] >= 0]
    # unique stops per route (needs the stop-level table once)
    route_stops = df.groupby("route_id")["stop_id"].nunique().rename("unique_stops")
    del df
    print(f"Stop-times read: {n_rows:,} -> trips: {len(trips):,}")
    return trips, route_stops


def revenue_features():
    """Historical (2019) revenue per route number and time period, if available."""
    if not REVENUE_FILE.exists():
        print("Note: revenue file not found; hist_* columns will be empty.")
        return None
    rev = pd.read_csv(REVENUE_FILE)
    if not {"Route_No", "Hour", "Revenue"}.issubset(rev.columns):
        print("Note: revenue file lacks Route_No/Hour/Revenue; skipping revenue features.")
        return None

    def period(h):
        return ("Morning Peak" if 6 <= h < 10 else "Midday" if 10 <= h < 16
                else "Evening Peak" if 16 <= h < 20 else "Off Peak")

    rev["time_period"] = rev["Hour"].astype(int).map(period)
    rev["route_short_name"] = rev["Route_No"].astype(int).astype(str)
    out = (rev.groupby(["route_short_name", "time_period"])["Revenue"]
              .agg(hist_revenue_records="size", hist_avg_revenue_per_trip="mean",
                   hist_total_revenue="sum").reset_index())
    return out


def main():
    print("=" * 60)
    print("TransitMind - ML dataset (route x time period)")
    print("=" * 60)
    trips, route_stops = build_trip_table()

    # ---- route-level structure -------------------------------------------
    route = trips.groupby("route_id").agg(
        route_short_name=("route_short_name", "first"),
        total_trips=("trip_id", "nunique"),
        average_stops=("stop_count", "mean"),
        max_stops_per_trip=("stop_count", "max"),
        average_trip_duration=("duration_min", "mean"),
        shape_count=("shape_id", "nunique"),
        direction_count=("direction_id", "nunique"),
    ).join(route_stops).reset_index()

    # ---- period-level activity -------------------------------------------
    per = trips.groupby(["route_id", "time_period"]).agg(
        scheduled_trips_in_period=("trip_id", "nunique"),
        mean_start_hour=("start_hour", "mean"),
    ).reset_index()

    # Every route x period combination, so "no trips in this period" is an
    # explicit 0 instead of a silently missing row.
    grid = pd.MultiIndex.from_product([route["route_id"], TIME_PERIODS],
                                      names=["route_id", "time_period"]).to_frame(index=False)
    ds = grid.merge(per, how="left", on=["route_id", "time_period"])
    ds["scheduled_trips_in_period"] = ds["scheduled_trips_in_period"].fillna(0).astype(int)
    fallback_hour = ds["time_period"].map({"Morning Peak": 8, "Midday": 13,
                                           "Evening Peak": 18, "Off Peak": 22})
    ds["mean_start_hour"] = ds["mean_start_hour"].fillna(fallback_hour)
    ds = ds.merge(route, on="route_id", how="left")

    # trips per period as route-level descriptive columns (NOT model features)
    pivot = ds.pivot(index="route_id", columns="time_period",
                     values="scheduled_trips_in_period")
    pivot = pivot.rename(columns={"Morning Peak": "morning_trip_count",
                                  "Midday": "midday_trip_count",
                                  "Evening Peak": "evening_trip_count",
                                  "Off Peak": "offpeak_trip_count"}).reset_index()
    ds = ds.merge(pivot, on="route_id", how="left")

    ds = add_derived_features(ds)
    ds["trips_per_hour"] = ds["scheduled_trips_in_period"] / ds["period_hours"]
    ds["buses_required_proxy"] = ds["trips_per_hour"] * ds["average_trip_duration"] / 60
    ds["operational_intensity"] = ds["buses_required_proxy"]   # same quantity, readable name

    # Classification label: Low / Medium / High by terciles of trips_per_hour,
    # computed only on rows that actually have service.
    active = ds["trips_per_hour"] > 0
    _, edges = class_from_terciles(ds.loc[active, CLASSIFICATION_SOURCE])
    ds[CLASSIFICATION_TARGET], _ = class_from_terciles(ds[CLASSIFICATION_SOURCE], edges)
    print(f"Activity class edges (trips/hour): Low <= {edges[0]:.3f} < Medium <= {edges[1]:.3f} < High")

    # ---- historical revenue signal ---------------------------------------
    rev = revenue_features()
    if rev is not None:
        ds = ds.merge(rev, how="left", on=["route_short_name", "time_period"])
    else:
        for c in ("hist_revenue_records", "hist_avg_revenue_per_trip", "hist_total_revenue"):
            ds[c] = float("nan")
    ds["hist_revenue_records"] = ds["hist_revenue_records"].fillna(0).astype(int)
    ds["has_revenue_history"] = (ds["hist_revenue_records"] > 0).astype(int)

    # ---- validation -------------------------------------------------------
    ds = ds.replace([float("inf"), float("-inf")], float("nan"))
    key_cols = ["average_stops", "average_trip_duration", "unique_stops", "trips_per_hour"]
    n_before = len(ds)
    ds = ds.dropna(subset=key_cols)
    if len(ds) < n_before:
        print(f"Warning: dropped {n_before - len(ds)} rows with missing key features")
    if ds.empty:
        sys.exit("ERROR: ML dataset is empty.")

    ds = ds.round(4).sort_values(["route_id", "time_period"]).reset_index(drop=True)
    ML_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(ML_DATASET_FILE, index=False)

    print(f"\nSaved {ML_DATASET_FILE}")
    print(f"Rows: {len(ds):,} | routes: {ds['route_id'].nunique()} | columns: {ds.shape[1]}")
    print("Class balance:", ds[CLASSIFICATION_TARGET].value_counts().to_dict())
    print(f"Rows with 2019 revenue history: {int(ds['has_revenue_history'].sum())} "
          f"(routes: {sorted(ds.loc[ds.has_revenue_history == 1, 'route_short_name'].unique())})")
    print("\nSample:")
    print(ds.head(6).T.to_string())


if __name__ == "__main__":
    main()
