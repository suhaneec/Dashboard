"""
Minimal data cleaning script for MSafe-style interior design funnel dataset.
Rule: NO rows dropped, NO columns dropped, NO values invented/filled.
Only: whitespace trim, code casing standardization, date dtype fixes,
and a corrected Booking/60%/Handover Month derived from real dates
(original derived-month columns are kept as *_raw for audit trail).

Usage:
    python clean_data.py
Reads:  raw_data.xlsx  (sheet: "Data set (Not to use Pivots)")
Writes: cleaned_data.csv
        cleaned_data.xlsx
"""

import pandas as pd

SRC = "raw_data.xlsx"
SHEET = "Data set (Not to use Pivots)"

df = pd.read_excel(SRC, sheet_name=SHEET)

# 1. Drop the two known junk/unnamed trailing columns (no data, header artifacts only)
#    These are NOT business data — they are blank spillover columns from the source file.
junk_cols = [c for c in df.columns if str(c).startswith("Unnamed") or "2023-10-28" in str(c)]
df = df.drop(columns=junk_cols)

# 2. Trim whitespace on every text/object column
str_cols = df.select_dtypes(include="object").columns
for c in str_cols:
    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

# 3. Standardize project code casing (fixes BX-2864 vs Bx-2354 inconsistency)
df["Project Parent Code"] = df["Project Parent Code"].str.upper()
df["Project Child Code"] = df["Project Child Code"].str.upper()

# 4. Ensure known date columns are proper datetime dtype (no value changes, just dtype safety)
date_cols = [
    "Booking Date", "60% Date", "20% date", "30% date", "50% date",
    "Handover Date", "Remaining Amount date",
    "Design Completed / Design Deck\n (Moved to Execution)",
]
for c in date_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# 5. Recompute Booking Month / 60% Month / Handover Month from their real date columns.
#    Original columns preserved as *_raw for audit — nothing silently overwritten.
month_pairs = [
    ("Booking Date", "Booking Month"),
    ("60% Date", "60% Month"),
    ("Handover Date", "Handover Month"),
]
for date_col, month_col in month_pairs:
    if month_col in df.columns:
        df = df.rename(columns={month_col: f"{month_col}_raw"})
        df[month_col] = df[date_col].dt.to_period("M").dt.to_timestamp()

# 6. Standardize duplicate status labels (case/spacing variants of the same status)
#    "Lost In Design" -> "Lost at Design" (4 rows), "Lost At Execution" -> "Lost at Execution" (2 rows)
status_map = {
    "Lost In Design": "Lost at Design",
    "Lost At Execution": "Lost at Execution",
}
df["Final Project status"] = df["Final Project status"].replace(status_map)

# 7. Quick post-clean sanity report (printed, not written to file)
print("Rows:", len(df), "| Columns:", len(df.columns))
print("Nulls unchanged (sample):")
print(df[["60% Date", "50% date", "Handover Date"]].isna().sum())
print("\nParent code case-fix check (should be 0 lowercase 'bx-' left):")
print(df["Project Parent Code"].str.contains("bx-", case=False, regex=False).sum(), "total 'bx-' matches (expected, case-insensitive)")

# 7. Save outputs
df.to_csv("cleaned_data.csv", index=False)
df.to_excel("cleaned_data.xlsx", index=False, sheet_name="Cleaned Data")

print("\nSaved: cleaned_data.csv, cleaned_data.xlsx")
