"""Evaluate the fixed cluster-0 credit policy on the existing 330 members.

Baseline gives the credit to every member. V1 gives it only when
decide_credit assigns cluster 0. The new-member example is reported
separately and is not included in these totals.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from credit_policy import FEATURE_ORDER, decide_credit

DATA_PATH = Path(__file__).resolve().parent / "gym_members.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SEGMENTS_PATH = OUTPUT_DIR / "member_segments.csv"
ID_COLUMN = "member_id"
SPEND_COLUMN = "monthly_extras_spend"
SPEND_THRESHOLD = 20
GAIN_BELOW_THRESHOLD = 5
LOSS_AT_OR_ABOVE_THRESHOLD = -10

NEW_MEMBER = {
    "age": 30,
    "visits_per_month": 4,
    "avg_visit_minutes": 42,
    "classes_per_month": 1,
    "months_member": 5,
    "monthly_extras_spend": 10,
}

EXPECTED_FULL_BASELINE = -1200
EXPECTED_TRAIN_BASELINE = -855
EXPECTED_TRAIN_V1 = 110
EXPECTED_TRAIN_IMPROVEMENT = 965


def main() -> None:
    members = pd.read_csv(DATA_PATH)
    segments = pd.read_csv(SEGMENTS_PATH)
    if len(members) != 330:
        raise ValueError(f"Expected 330 members, found {len(members)}")

    decisions = [decide_credit(row) for row in members[FEATURE_ORDER].to_dict(orient="records")]
    results = members[[ID_COLUMN, SPEND_COLUMN]].copy()
    results["cluster"] = [decision.cluster_id for decision in decisions]
    results["gets_credit"] = [decision.gets_credit for decision in decisions]
    results = results.merge(
        segments[[ID_COLUMN, "split", "cluster"]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_saved"),
    )
    if len(results) != len(members):
        raise ValueError("member_id merge did not keep all 330 members")
    mismatched = results.loc[results["cluster"] != results["cluster_saved"], ID_COLUMN]
    if not mismatched.empty:
        raise ValueError(f"Cluster assignments differ from member_segments.csv: {mismatched.tolist()}")
    if not results["gets_credit"].eq(results["cluster"] == 0).all():
        raise ValueError("gets_credit must be true only for cluster 0")

    below = results[SPEND_COLUMN] < SPEND_THRESHOLD
    results["baseline_contribution"] = np.where(below, GAIN_BELOW_THRESHOLD, LOSS_AT_OR_ABOVE_THRESHOLD)
    results["v1_contribution"] = np.where(results["gets_credit"], results["baseline_contribution"], 0)
    results = results[
        [
            ID_COLUMN,
            "split",
            "cluster",
            "gets_credit",
            SPEND_COLUMN,
            "baseline_contribution",
            "v1_contribution",
        ]
    ]

    summary_rows = []
    for group_name, group in (
        ("train", results.loc[results["split"] == "train"]),
        ("test", results.loc[results["split"] == "test"]),
        ("all", results),
    ):
        baseline_net = int(group["baseline_contribution"].sum())
        v1_net = int(group["v1_contribution"].sum())
        summary_rows.append(
            {
                "group": group_name,
                "n_members": int(len(group)),
                "credit_recipients": int(group["gets_credit"].sum()),
                "baseline_net": baseline_net,
                "v1_net": v1_net,
                "improvement": v1_net - baseline_net,
            }
        )
    summary = pd.DataFrame(summary_rows)
    by_group = summary.set_index("group")
    if int(by_group.loc["all", "baseline_net"]) != EXPECTED_FULL_BASELINE:
        raise ValueError("Full-data baseline net is not -1200")
    train_row = by_group.loc["train"]
    if (
        int(train_row["baseline_net"]) != EXPECTED_TRAIN_BASELINE
        or int(train_row["v1_net"]) != EXPECTED_TRAIN_V1
        or int(train_row["improvement"]) != EXPECTED_TRAIN_IMPROVEMENT
    ):
        raise ValueError("Training policy totals do not match the training-only analysis")

    print("Policy value by split")
    print(
        summary.to_string(
            index=False,
            formatters={
                "n_members": "{:.0f}".format,
                "credit_recipients": "{:.0f}".format,
                "baseline_net": "{:.0f}".format,
                "v1_net": "{:.0f}".format,
                "improvement": "{:.0f}".format,
            },
        )
    )
    print("Verified full-data baseline = -1200")
    print("Verified training baseline = -855, v1 = +110, improvement = +965")

    member_path = OUTPUT_DIR / "member_credit_results.csv"
    summary_path = OUTPUT_DIR / "policy_value_summary.csv"
    results.to_csv(member_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved member results: {member_path}")
    print(f"Saved summary: {summary_path}")

    new_decision = decide_credit(NEW_MEMBER)
    print()
    print("New member, excluded from the 330-member totals")
    print(
        "age=30, visits_per_month=4, avg_visit_minutes=42, "
        "classes_per_month=1, months_member=5, monthly_extras_spend=10"
    )
    print(f"Assigned cluster: {new_decision.cluster_id}")
    print(f"Receives credit: {new_decision.gets_credit}")


if __name__ == "__main__":
    main()
