"""Describe the saved k=4 segments without refitting or resplitting."""

from pathlib import Path

import joblib
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "gym_members.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "kmeans_candidates.joblib"
SPLIT_PATH = OUTPUT_DIR / "data_split.csv"
CHOSEN_K = 4
ID_COLUMN = "member_id"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    feature_order = list(bundle["feature_order"])
    model = bundle["models"][CHOSEN_K]

    members = pd.read_csv(DATA_PATH)
    split = pd.read_csv(SPLIT_PATH)
    if set(split[ID_COLUMN]) != set(members[ID_COLUMN]):
        raise ValueError("data_split.csv member_id values do not match gym_members.csv")
    if len(split) != len(members) or split[ID_COLUMN].duplicated().any():
        raise ValueError("data_split.csv must contain each member exactly once")

    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=feature_order,
    )
    centers.insert(0, "cluster_id", range(len(centers)))

    print("1. Cluster centers in original units")
    print(centers.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    scaled = scaler.transform(members[feature_order])
    assigned = members[[ID_COLUMN]].copy()
    assigned["cluster"] = model.predict(scaled)
    assigned = assigned.merge(split, on=ID_COLUMN, how="left", validate="one_to_one")
    assigned = assigned[[ID_COLUMN, "split", "cluster"]]

    counts = (
        assigned.groupby("cluster")["split"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["train", "test"], fill_value=0)
        .reindex(centers["cluster_id"], fill_value=0)
    )
    profiles = centers.copy()
    profiles.insert(1, "train_members", counts["train"].to_numpy())
    profiles.insert(2, "test_members", counts["test"].to_numpy())
    profiles.insert(3, "total_members", profiles["train_members"] + profiles["test_members"])

    print()
    print("2-3. Cluster sizes and original-unit centroids")
    print(
        profiles.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
            formatters={
                "cluster_id": "{:.0f}".format,
                "train_members": "{:.0f}".format,
                "test_members": "{:.0f}".format,
                "total_members": "{:.0f}".format,
            },
        )
    )

    profiles_path = OUTPUT_DIR / "segment_profiles.csv"
    members_path = OUTPUT_DIR / "member_segments.csv"
    profiles.to_csv(profiles_path, index=False)
    assigned.to_csv(members_path, index=False)
    print()
    print(f"Saved profiles: {profiles_path}")
    print(f"Saved member assignments: {members_path}")
    print(f"Assigned members: {len(assigned)}")


if __name__ == "__main__":
    main()
