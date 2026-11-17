"""Score the $5 / -$10 extras credit on training members only.

Each member contributes +5 when monthly_extras_spend is below 20 and -10
otherwise. Cluster totals are the sum of those member contributions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "gym_members.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SEGMENTS_PATH = OUTPUT_DIR / "member_segments.csv"
ID_COLUMN = "member_id"
SPEND_COLUMN = "monthly_extras_spend"
SPEND_THRESHOLD = 20
GAIN_BELOW_THRESHOLD = 5
LOSS_AT_OR_ABOVE_THRESHOLD = -10


def main() -> None:
    members = pd.read_csv(DATA_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    merged = members.merge(segments, on=ID_COLUMN, how="inner", validate="one_to_one")
    if len(merged) != len(members) or len(merged) != len(segments):
        raise ValueError("member_id merge did not match every row on both sides")

    training = merged.loc[merged["split"] == "train"].copy()
    below = training[SPEND_COLUMN] < SPEND_THRESHOLD
    training["incremental_net"] = np.where(
        below,
        GAIN_BELOW_THRESHOLD,
        LOSS_AT_OR_ABOVE_THRESHOLD,
    )

    analysis = (
        training.groupby("cluster", sort=True)
        .agg(
            train_members=(ID_COLUMN, "size"),
            below_20=("incremental_net", lambda values: int((values == GAIN_BELOW_THRESHOLD).sum())),
            at_or_above_20=(
                "incremental_net",
                lambda values: int((values == LOSS_AT_OR_ABOVE_THRESHOLD).sum()),
            ),
            total_monthly_incremental_net=("incremental_net", "sum"),
        )
        .reset_index()
    )
    analysis["pct_below_20"] = 100 * analysis["below_20"] / analysis["train_members"]
    analysis["net_per_member"] = (
        analysis["total_monthly_incremental_net"] / analysis["train_members"]
    )
    analysis = analysis[
        [
            "cluster",
            "train_members",
            "below_20",
            "at_or_above_20",
            "pct_below_20",
            "total_monthly_incremental_net",
            "net_per_member",
        ]
    ]

    print("Training-only credit analysis (member-level +5 below $20, -10 at or above $20)")
    print(f"Training members scored: {len(training)}")
    print(
        analysis.to_string(
            index=False,
            formatters={
                "cluster": "{:.0f}".format,
                "train_members": "{:.0f}".format,
                "below_20": "{:.0f}".format,
                "at_or_above_20": "{:.0f}".format,
                "pct_below_20": "{:.2f}".format,
                "total_monthly_incremental_net": "{:.2f}".format,
                "net_per_member": "{:.2f}".format,
            },
        )
    )

    output_path = OUTPUT_DIR / "training_credit_analysis.csv"
    analysis.to_csv(output_path, index=False)
    print(f"Saved analysis: {output_path}")


if __name__ == "__main__":
    main()
