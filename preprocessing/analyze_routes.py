"""
TransitMind - Route Analysis

Purpose:
    Analyze PMPML operational GTFS data and create a route-level summary.

Input:
    data/processed/pmpml_operational_data.csv

Output:
    data/processed/pmpml_route_summary.csv

The analysis is performed at the TRIP level where appropriate,
so that time-period trip counts are not inflated by stop counts.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pmpml_operational_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pmpml_route_summary.csv"
)


# ============================================================
# 2. CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


print("=" * 60)
print("TransitMind - Route Analysis")
print("=" * 60)

print("\nLoading data from:")
print(INPUT_FILE)


# ============================================================
# 3. LOAD ONLY REQUIRED COLUMNS
# ============================================================

columns = [
    "trip_id",
    "route_id",
    "route_short_name",
    "direction_id",
    "stop_id",
    "stop_sequence",
    "arrival_time",
    "arrival_hour",
    "time_period",
]

df = pd.read_csv(
    INPUT_FILE,
    usecols=columns,
    low_memory=False
)

print(f"\nLoaded {len(df):,} stop-time records.")


# ============================================================
# 4. FIX DATA TYPES
# ============================================================

print("\nConverting data types...")


# Convert arrival time back into timedelta.
df["arrival_time"] = pd.to_timedelta(
    df["arrival_time"],
    errors="coerce"
)


# Convert numeric columns.
df["direction_id"] = pd.to_numeric(
    df["direction_id"],
    errors="coerce"
)

df["arrival_hour"] = pd.to_numeric(
    df["arrival_hour"],
    errors="coerce"
)

df["stop_sequence"] = pd.to_numeric(
    df["stop_sequence"],
    errors="coerce"
)


# ============================================================
# 5. REMOVE INVALID RECORDS
# ============================================================

invalid_times = df["arrival_time"].isna().sum()

if invalid_times > 0:
    print(
        f"\nWarning: {invalid_times:,} records have "
        "invalid arrival times."
    )

    df = df.dropna(
        subset=["arrival_time"]
    )


print(
    f"Valid stop-time records remaining: {len(df):,}"
)


# ============================================================
# 6. SORT DATA
# ============================================================

print("\nSorting records...")

df = df.sort_values(
    [
        "trip_id",
        "stop_sequence"
    ]
)


# ============================================================
# 7. CREATE TRIP-LEVEL DATASET
# ============================================================

print("\nCreating trip-level summary...")


"""
A GTFS trip contains multiple stop-time records.

Example:

Trip A
    Stop 1
    Stop 2
    Stop 3
    Stop 4

If we count the rows directly, Trip A would be counted
four times.

Instead, we create ONE record per trip.
"""


trip_summary = (
    df.groupby(
        "trip_id",
        dropna=False
    )
    .agg(
        route_id=("route_id", "first"),
        route_short_name=("route_short_name", "first"),
        direction_id=("direction_id", "first"),
        time_period=("time_period", "first"),

        stop_count=("stop_id", "nunique"),

        first_stop_time=("arrival_time", "min"),

        last_stop_time=("arrival_time", "max")
    )
    .reset_index()
)


print(
    f"Unique trips identified: "
    f"{len(trip_summary):,}"
)


# ============================================================
# 8. CALCULATE TRIP DURATION
# ============================================================

print("\nCalculating trip durations...")


trip_summary["trip_duration_minutes"] = (
    trip_summary["last_stop_time"]
    - trip_summary["first_stop_time"]
).dt.total_seconds() / 60


# ============================================================
# 9. REMOVE INVALID DURATIONS
# ============================================================

invalid_duration = (
    trip_summary["trip_duration_minutes"].isna()
    |
    (trip_summary["trip_duration_minutes"] < 0)
)


invalid_count = invalid_duration.sum()


if invalid_count > 0:
    print(
        f"Warning: Removing "
        f"{invalid_count:,} invalid trip durations."
    )

    trip_summary = trip_summary[
        ~invalid_duration
    ]


# ============================================================
# 10. CREATE ROUTE-LEVEL SUMMARY
# ============================================================

print("\nCreating route-level summary...")


route_summary = (
    trip_summary
    .groupby(
        [
            "route_id",
            "route_short_name"
        ],
        dropna=False
    )
    .agg(

        # Number of unique scheduled trips
        total_trips=(
            "trip_id",
            "nunique"
        ),

        # Average number of stops in a trip
        average_stops_per_trip=(
            "stop_count",
            "mean"
        ),

        # Average scheduled duration
        average_trip_duration_minutes=(
            "trip_duration_minutes",
            "mean"
        ),

        # Minimum scheduled duration
        minimum_trip_duration_minutes=(
            "trip_duration_minutes",
            "min"
        ),

        # Maximum scheduled duration
        maximum_trip_duration_minutes=(
            "trip_duration_minutes",
            "max"
        )
    )
    .reset_index()
)


# ============================================================
# 11. COUNT UNIQUE STOPS PER ROUTE
# ============================================================

print("Calculating unique stops per route...")


route_stops = (
    df
    .groupby(
        "route_id"
    )["stop_id"]
    .nunique()
    .reset_index(
        name="unique_stops"
    )
)


route_summary = route_summary.merge(
    route_stops,
    on="route_id",
    how="left"
)


# ============================================================
# 12. TIME-PERIOD TRIP COUNTS
# ============================================================

print(
    "Calculating time-period trip counts..."
)


"""
IMPORTANT:

We use trip_summary here instead of df.

df contains one row per stop.

trip_summary contains one row per trip.

Therefore each trip is counted exactly once.
"""


time_period_summary = pd.crosstab(
    trip_summary["route_id"],
    trip_summary["time_period"]
)


# Rename columns.
time_period_summary.columns = [
    f"trips_{str(column).lower().replace(' ', '_')}"
    for column in time_period_summary.columns
]


time_period_summary = (
    time_period_summary
    .reset_index()
)


# ============================================================
# 13. MERGE TIME-PERIOD COUNTS
# ============================================================

route_summary = route_summary.merge(
    time_period_summary,
    on="route_id",
    how="left"
)


# ============================================================
# 14. MAKE SURE ALL EXPECTED TIME PERIODS EXIST
# ============================================================

expected_period_columns = [
    "trips_morning_peak",
    "trips_midday",
    "trips_evening_peak",
    "trips_off_peak",
]


for column in expected_period_columns:

    if column not in route_summary.columns:

        route_summary[column] = 0


# Convert to integers.
for column in expected_period_columns:

    route_summary[column] = (
        route_summary[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# 15. ROUND NUMERIC VALUES
# ============================================================

numeric_columns = [
    "average_stops_per_trip",
    "average_trip_duration_minutes",
    "minimum_trip_duration_minutes",
    "maximum_trip_duration_minutes",
]


for column in numeric_columns:

    if column in route_summary.columns:

        route_summary[column] = (
            route_summary[column]
            .round(2)
        )


# ============================================================
# 16. VERIFY TRIP COUNTS
# ============================================================

print("\nChecking time-period trip counts...")


route_summary["period_trip_total"] = (
    route_summary["trips_morning_peak"]
    + route_summary["trips_midday"]
    + route_summary["trips_evening_peak"]
    + route_summary["trips_off_peak"]
)


mismatch = (
    route_summary["period_trip_total"]
    != route_summary["total_trips"]
)


mismatch_count = mismatch.sum()


if mismatch_count > 0:

    print(
        f"Warning: {mismatch_count} routes have "
        "a mismatch between total trips and "
        "time-period trip counts."
    )

else:

    print(
        "✓ Time-period trip counts match "
        "total trip counts for all routes."
    )


# We don't need this helper column in the final CSV.
route_summary = route_summary.drop(
    columns=["period_trip_total"]
)


# ============================================================
# 17. SORT ROUTES
# ============================================================

route_summary = route_summary.sort_values(
    "total_trips",
    ascending=False
)


# ============================================================
# 18. SAVE OUTPUT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


route_summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ROUTE ANALYSIS COMPLETE")
print("=" * 60)


print(
    f"\nRoutes analyzed: "
    f"{len(route_summary):,}"
)


print(
    f"\nOutput saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 20. DISPLAY TOP 10 ROUTES
# ============================================================

print(
    "\nTop 10 routes by number of scheduled trips:"
)

print("-" * 60)


display_columns = [
    "route_id",
    "route_short_name",
    "total_trips",
    "unique_stops",
    "average_stops_per_trip",
    "average_trip_duration_minutes",
    "trips_morning_peak",
    "trips_midday",
    "trips_evening_peak",
    "trips_off_peak",
]


available_columns = [
    column
    for column in display_columns
    if column in route_summary.columns
]


print(
    route_summary[
        available_columns
    ]
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("Done.")
print("=" * 60)