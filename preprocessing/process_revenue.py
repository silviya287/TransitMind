"""
TransitMind - Revenue preprocessing

Input : data/raw/PMPML_OneDay_Data.xlsx  (Sheet1 + Sheet2; Sheet3 is empty and ignored)
Output: data/processed/pmpml_revenue_data.csv

Known source problem
--------------------
Sheet2 has the right *values* under the wrong *headers*.  Its logical column
order is   Date | Time | Bus_No | Revenue | Route_No
(the header row says Date | Bus_No | Time | ...).  Sheet2 is therefore read
by position.  A time may also be stored as an Excel day-fraction (0.534722
means 12:50), which is converted explicitly.  Bus numbers such as 1076 or
R265 are identifiers and are never interpreted as times.

Only rows that are completely unusable (no valid date, time, revenue or
route) are dropped, and the number of dropped rows is reported.
"""
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "PMPML_OneDay_Data.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "pmpml_revenue_data.csv"

FINAL_COLUMNS = ["Date", "Time", "Bus_No", "Revenue", "Route_No"]


def clean_time(value):
    """Return a datetime.time or None for any supported Excel/text time."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, datetime):          # check before time (datetime is not time)
        return value.time().replace(microsecond=0)
    if isinstance(value, time):
        return value.replace(microsecond=0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Excel stores a time of day as a fraction of 24 h (0 <= x < 1).
        if not 0 <= float(value) < 1:
            return None
        seconds = int(round(float(value) * 86400)) % 86400
        return time(seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    text = str(value).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    try:                                      # a fraction stored as text, e.g. "0.53"
        return clean_time(float(text))
    except ValueError:
        return None


def clean_date(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).normalize()
    return pd.to_datetime(str(value).strip(), dayfirst=True, errors="coerce")


def load_sheets(path):
    if not path.exists():
        raise FileNotFoundError(f"Revenue workbook not found: {path}")
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    except Exception as exc:                  # corrupted file, missing openpyxl, ...
        raise RuntimeError(f"Could not read {path.name}: {exc}") from exc
    for needed in ("Sheet1", "Sheet2"):
        if needed not in sheets:
            raise KeyError(f"Sheet '{needed}' missing. Found: {list(sheets)}")
    if "Sheet3" in sheets and not sheets["Sheet3"].dropna(how="all").empty:
        print("Note: Sheet3 is not empty but is ignored by design.")

    s1, s2 = sheets["Sheet1"], sheets["Sheet2"]
    for name, df in (("Sheet1", s1), ("Sheet2", s2)):
        if df.shape[1] < 5:
            raise ValueError(f"{name} has {df.shape[1]} columns, expected 5.")

    # Sheet1 is already  Date | Bus_No | Time | Revenue | Route_No
    s1 = s1.iloc[:, :5].copy()
    s1.columns = ["Date", "Bus_No", "Time", "Revenue", "Route_No"]
    # Sheet2 values are  Date | Time | Bus_No | Revenue | Route_No  (headers are wrong)
    s2 = s2.iloc[:, :5].copy()
    s2.columns = ["Date", "Time", "Bus_No", "Revenue", "Route_No"]
    return s1[FINAL_COLUMNS], s2[FINAL_COLUMNS]


def main():
    print("=" * 60)
    print("TransitMind - Revenue preprocessing")
    print("=" * 60)
    s1, s2 = load_sheets(INPUT_FILE)
    print(f"Sheet1 rows: {len(s1)} | Sheet2 rows: {len(s2)}")
    df = pd.concat([s1.assign(Source="Sheet1"), s2.assign(Source="Sheet2")], ignore_index=True)

    df["Date"] = df["Date"].map(clean_date)
    df["Time"] = df["Time"].map(clean_time)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Route_No"] = pd.to_numeric(df["Route_No"], errors="coerce")
    df["Bus_No"] = (df["Bus_No"].astype(str).str.strip()
                    .str.replace(r"\.0$", "", regex=True))

    invalid_times = int(df["Time"].isna().sum())
    invalid_revenue = int(df["Revenue"].isna().sum())
    invalid_dates = int(df["Date"].isna().sum())
    invalid_routes = int(df["Route_No"].isna().sum())
    negative_revenue = int((df["Revenue"] < 0).sum())

    bad = df[["Date", "Time", "Revenue", "Route_No"]].isna().any(axis=1) | (df["Revenue"] < 0)
    if bad.any():
        print("\nDropping completely invalid rows:")
        print(df[bad].to_string(index=False))
    df = df[~bad].copy()
    if df.empty:
        raise ValueError("No valid revenue records left after cleaning.")

    df["Route_No"] = df["Route_No"].astype(int)
    df["Revenue"] = df["Revenue"].astype(float)
    df["Hour"] = df["Time"].map(lambda t: t.hour)
    df["Time"] = df["Time"].map(lambda t: t.strftime("%H:%M"))
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["Route_No", "Date", "Time", "Bus_No"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nShape            :", df.shape)
    print("Columns          :", df.columns.tolist())
    print("Routes           :", [int(r) for r in sorted(df["Route_No"].unique())])
    print("Dates            :", sorted(df["Date"].unique()))
    print(f"Invalid times    : {invalid_times}")
    print(f"Invalid revenue  : {invalid_revenue}  (negative: {negative_revenue})")
    print(f"Invalid dates    : {invalid_dates} | invalid routes: {invalid_routes}")
    print(f"Rows dropped     : {int(bad.sum())}")
    print("\nFirst 10 records:")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as err:
        print(f"ERROR: {err}")
        sys.exit(1)
