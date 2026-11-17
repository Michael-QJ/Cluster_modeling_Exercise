"""Inspect gym_members.csv without modifying the source data."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "gym_members.csv"
NON_ID_COLUMNS = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
# Gym membership ages outside this range are treated as implausible.
MIN_PLAUSIBLE_AGE = 13
MAX_PLAUSIBLE_AGE = 100


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    section("1. Shape, column names, and data types")
    print(f"Row count: {len(df)}")
    print(f"Column count: {df.shape[1]}")
    print("Column names:")
    for name in df.columns:
        print(f"  - {name}")
    print("Data types:")
    print(df.dtypes.to_string())

    section("2. Missing values in each column")
    missing = df.isna().sum()
    missing_report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percent": (missing / len(df) * 100).round(2),
        }
    )
    print(missing_report.to_string())
    print(f"Total missing cells: {int(missing.sum())}")

    section("3. Duplicate rows and duplicate member_id values")
    duplicate_row_mask = df.duplicated(keep=False)
    print(f"Duplicate rows (all copies): {int(duplicate_row_mask.sum())}")
    print(f"Extra duplicate rows (beyond first copy): {int(df.duplicated().sum())}")
    if duplicate_row_mask.any():
        print(df.loc[duplicate_row_mask].sort_values(list(df.columns)).to_string(index=False))

    duplicate_id_mask = df.duplicated(subset=["member_id"], keep=False)
    n_duplicate_ids = df.loc[duplicate_id_mask, "member_id"].nunique()
    print(f"member_id values that appear more than once: {int(n_duplicate_ids)}")
    print(f"Rows involved in duplicate member_id values: {int(duplicate_id_mask.sum())}")
    if duplicate_id_mask.any():
        print(
            df.loc[duplicate_id_mask]
            .sort_values(["member_id"])
            .to_string(index=False)
        )

    section("4. Summary statistics for the six non-ID columns")
    print(df[NON_ID_COLUMNS].describe().to_string())

    section("5. Mean monthly_extras_spend")
    mean_spend = df["monthly_extras_spend"].mean()
    print(f"mean monthly_extras_spend: {mean_spend:.4f}")

    section("6. Members by monthly_extras_spend threshold")
    below = int((df["monthly_extras_spend"] < 20).sum())
    at_or_above = int((df["monthly_extras_spend"] >= 20).sum())
    missing_spend = int(df["monthly_extras_spend"].isna().sum())
    print(f"monthly_extras_spend < 20:  {below}")
    print(f"monthly_extras_spend >= 20: {at_or_above}")
    print(f"monthly_extras_spend missing (not counted in either group): {missing_spend}")

    section("7. Potentially invalid values")
    checks = [
        ("negative age", df["age"] < 0),
        (
            f"implausible age (under {MIN_PLAUSIBLE_AGE} or over {MAX_PLAUSIBLE_AGE})",
            (df["age"] < MIN_PLAUSIBLE_AGE) | (df["age"] > MAX_PLAUSIBLE_AGE),
        ),
        ("negative visits_per_month", df["visits_per_month"] < 0),
        ("negative avg_visit_minutes", df["avg_visit_minutes"] < 0),
        ("negative classes_per_month", df["classes_per_month"] < 0),
        ("negative months_member", df["months_member"] < 0),
        ("negative monthly_extras_spend", df["monthly_extras_spend"] < 0),
        (
            "avg_visit_minutes above 240 (more than 4 hours)",
            df["avg_visit_minutes"] > 240,
        ),
        (
            "visits_per_month above 60",
            df["visits_per_month"] > 60,
        ),
    ]

    any_flagged = False
    for label, mask in checks:
        count = int(mask.fillna(False).sum())
        print(f"{label}: {count}")
        if count:
            any_flagged = True
            print(df.loc[mask.fillna(False)].to_string(index=False))
            print()

    if not any_flagged:
        print("No rows matched the invalid-value checks above.")


if __name__ == "__main__":
    main()
