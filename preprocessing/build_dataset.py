import pandas as pd
from pathlib import Path


# ============================================================
# TransitMind - Build Operational Dataset
# ============================================================

DATA_DIR = Path("data/raw/mdb-3137-202608070123")

print("=" * 60)
print("TRANSITMIND - BUILDING OPERATIONAL DATASET")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load GTFS files
# ------------------------------------------------------------

print("\nLoading GTFS files...")

routes = pd.read_csv(DATA_DIR / "routes.txt")
trips = pd.read_csv(DATA_DIR / "trips.txt")
stop_times = pd.read_csv(DATA_DIR / "stop_times.txt")
stops = pd.read_csv(DATA_DIR / "stops.txt")


print(f"Routes     : {len(routes):,}")
print(f"Trips      : {len(trips):,}")
print(f"Stop times : {len(stop_times):,}")
print(f"Stops      : {len(stops):,}")


# ------------------------------------------------------------
# 2. Connect trips with routes
# ------------------------------------------------------------

print("\nConnecting trips with routes...")

trip_data = trips.merge(
    routes[
        [
            "route_id",
            "route_short_name",
            "route_long_name",
            "route_type"
        ]
    ],
    on="route_id",
    how="left"
)


# ------------------------------------------------------------
# 3. Connect stop times with trip information
# ------------------------------------------------------------

print("Connecting stop times with trips...")

operational_data = stop_times.merge(
    trip_data[
        [
            "trip_id",
            "route_id",
            "route_short_name",
            "route_long_name",
            "direction_id",
            "shape_id"
        ]
    ],
    on="trip_id",
    how="left"
)


# ------------------------------------------------------------
# 4. Connect stop information
# ------------------------------------------------------------

print("Connecting stop information...")

operational_data = operational_data.merge(
    stops[
        [
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon"
        ]
    ],
    on="stop_id",
    how="left"
)


# ------------------------------------------------------------
# 5. Create time-based features
# ------------------------------------------------------------

print("Creating time features...")

operational_data["arrival_time"] = pd.to_timedelta(
    operational_data["arrival_time"]
)

operational_data["departure_time"] = pd.to_timedelta(
    operational_data["departure_time"]
)

operational_data["arrival_hour"] = (
    operational_data["arrival_time"].dt.total_seconds() / 3600
).astype(int) % 24


def classify_period(hour):
    if 6 <= hour < 10:
        return "Morning Peak"
    elif 10 <= hour < 16:
        return "Midday"
    elif 16 <= hour < 20:
        return "Evening Peak"
    else:
        return "Off Peak"


operational_data["time_period"] = operational_data[
    "arrival_hour"
].apply(classify_period)


# ------------------------------------------------------------
# 6. Sort the dataset
# ------------------------------------------------------------

operational_data = operational_data.sort_values(
    [
        "route_id",
        "trip_id",
        "stop_sequence"
    ]
)


# ------------------------------------------------------------
# 7. Save processed dataset
# ------------------------------------------------------------

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "pmpml_operational_data.csv"

operational_data.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 8. Display summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nOutput file:")
print(output_file)

print(f"\nRows    : {len(operational_data):,}")
print(f"Columns : {len(operational_data.columns)}")

print("\nColumns:")
for column in operational_data.columns:
    print(f"  - {column}")

print("\nSample:")
print(
    operational_data.head(5).to_string(index=False)
)

print("\nDone.")