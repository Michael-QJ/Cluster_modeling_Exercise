"""Fit candidate KMeans models on a held-out gym-member split.

member_id is kept for tracking and is never used as a feature.
The scaler and every KMeans model are fit on the training split only.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "gym_members.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_DIR = Path(__file__).resolve().parent / "models"

FEATURE_COLUMNS = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
ID_COLUMN = "member_id"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
K_VALUES = range(1, 9)


def sum_squared_distance(features: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared Euclidean distances from each row to its assigned center."""
    assigned = centers[labels]
    return float(np.sum((features - assigned) ** 2))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    members = pd.read_csv(DATA_PATH)
    features = members[FEATURE_COLUMNS]
    member_ids = members[ID_COLUMN]

    # 0.7 * 330 is 230.999... in floating point, and train_test_split floors
    # a float train_size, which would yield 230/100 instead of 70/30.
    n_train = int(round(TRAIN_FRACTION * len(members)))
    n_test = len(members) - n_train
    train_features, test_features, train_ids, test_ids = train_test_split(
        features,
        member_ids,
        train_size=n_train,
        test_size=n_test,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("1. Train/test split")
    print(f"Training rows: {len(train_features)}")
    print(f"Test rows: {len(test_features)}")
    print(f"Total rows: {len(train_features) + len(test_features)}")

    split_table = pd.concat(
        [
            pd.DataFrame({ID_COLUMN: train_ids.to_numpy(), "split": "train"}),
            pd.DataFrame({ID_COLUMN: test_ids.to_numpy(), "split": "test"}),
        ],
        ignore_index=True,
    )
    split_path = OUTPUT_DIR / "data_split.csv"
    split_table.to_csv(split_path, index=False)
    print(f"Saved split: {split_path}")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    print()
    print("2. StandardScaler fit on training features only")
    for name, mean, scale in zip(FEATURE_COLUMNS, scaler.mean_, scaler.scale_, strict=True):
        print(f"  {name}: mean={mean:.6f}, scale={scale:.6f}")

    models = {}
    metric_rows = []
    print()
    print("3-4. KMeans fit on scaled training data (k = 1..8)")
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(train_scaled)

        train_labels = model.labels_
        test_labels = model.predict(test_scaled)
        train_sse = sum_squared_distance(train_scaled, model.cluster_centers_, train_labels)
        test_sse = sum_squared_distance(test_scaled, model.cluster_centers_, test_labels)

        models[k] = model
        metric_rows.append(
            {
                "k": k,
                "train_sse": train_sse,
                "test_sse": test_sse,
                "train_sse_per_member": train_sse / len(train_scaled),
                "test_sse_per_member": test_sse / len(test_scaled),
                "n_train": len(train_scaled),
                "n_test": len(test_scaled),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = OUTPUT_DIR / "fit_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved metrics: {metrics_path}")
    print()
    print(
        "Training and test samples have different sizes "
        f"({len(train_scaled)} vs {len(test_scaled)}). "
        "Total training SSE and total test SSE should not be compared "
        "directly as measures of cluster quality. SSE per member puts both "
        "curves on a common per-person scale."
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    k_axis = metrics["k"].to_numpy()

    axes[0].plot(k_axis, metrics["train_sse"], marker="o", label=f"Training (n={len(train_scaled)})")
    axes[0].plot(k_axis, metrics["test_sse"], marker="o", label=f"Test (n={len(test_scaled)})")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Total SSE")
    axes[0].set_title("Total SSE by k")
    axes[0].set_xticks(list(K_VALUES))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        k_axis,
        metrics["train_sse_per_member"],
        marker="o",
        label=f"Training (n={len(train_scaled)})",
    )
    axes[1].plot(
        k_axis,
        metrics["test_sse_per_member"],
        marker="o",
        label=f"Test (n={len(test_scaled)})",
    )
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("SSE per member")
    axes[1].set_title("SSE per member by k")
    axes[1].set_xticks(list(K_VALUES))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(
        "Elbow curves from training-fitted KMeans\n"
        f"Samples differ in size (train n={len(train_scaled)}, test n={len(test_scaled)}), "
        "so total training and test SSE should not be compared directly as quality measures.",
        fontsize=11,
    )
    figure.tight_layout()
    elbow_path = OUTPUT_DIR / "elbow.png"
    figure.savefig(elbow_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved elbow plot: {elbow_path}")

    bundle_path = MODEL_DIR / "kmeans_candidates.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "feature_order": FEATURE_COLUMNS,
            "models": models,
            "random_state": RANDOM_STATE,
            "train_size": TRAIN_FRACTION,
        },
        bundle_path,
    )
    print(f"Saved fitted scaler and models: {bundle_path}")


if __name__ == "__main__":
    main()
