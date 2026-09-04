import pandas as pd
from pathlib import Path

# Location of the extracted PMPML GTFS data
DATA_DIR = Path("data/raw/mdb-3137-202608070123")

files = [
    "agency.txt",
    "routes.txt",
    "stops.txt",
    "trips.txt",
    "stop_times.txt",
    "shapes.txt",
    "calendar.txt",
]

print("=" * 60)
print("TRANSITMIND - PMPML GTFS DATA INSPECTION")
print("=" * 60)

for filename in files:
    filepath = DATA_DIR / filename

    print(f"\n{'-' * 60}")
    print(f"FILE: {filename}")
    print(f"{'-' * 60}")

    if not filepath.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(filepath)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))