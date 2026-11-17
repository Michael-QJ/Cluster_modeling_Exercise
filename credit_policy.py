"""Fixed credit policy: offer the credit only to cluster 0.

The cluster comes from the saved training-fitted scaler and k=4 model.
Eligibility is the assigned cluster, not an individual spend cutoff.
"""

from pathlib import Path
from typing import NamedTuple

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "kmeans_candidates.joblib"
CHOSEN_K = 4
CREDIT_CLUSTER = 0

_bundle = joblib.load(MODEL_PATH)
SCALER = _bundle["scaler"]
FEATURE_ORDER = list(_bundle["feature_order"])
MODEL = _bundle["models"][CHOSEN_K]


class CreditDecision(NamedTuple):
    cluster_id: int
    gets_credit: bool


def _feature_frame(member_features) -> pd.DataFrame:
    """Return one row of original-unit features in the saved feature order."""
    if isinstance(member_features, pd.DataFrame):
        frame = member_features
    elif isinstance(member_features, pd.Series):
        frame = member_features.to_frame().T
    elif isinstance(member_features, dict):
        frame = pd.DataFrame([member_features])
    else:
        values = list(member_features)
        if len(values) != len(FEATURE_ORDER):
            raise ValueError(
                f"Expected {len(FEATURE_ORDER)} features in order {FEATURE_ORDER}, got {len(values)}"
            )
        frame = pd.DataFrame([values], columns=FEATURE_ORDER)

    missing = [name for name in FEATURE_ORDER if name not in frame.columns]
    if missing:
        raise ValueError(f"member_features is missing {missing}")
    if len(frame) != 1:
        raise ValueError("decide_credit accepts one member at a time")
    return frame.loc[:, FEATURE_ORDER]


def decide_credit(member_features) -> CreditDecision:
    """Assign one member with the saved model and apply the fixed cluster-0 policy."""
    scaled = SCALER.transform(_feature_frame(member_features))
    cluster_id = int(MODEL.predict(scaled)[0])
    return CreditDecision(cluster_id=cluster_id, gets_credit=cluster_id == CREDIT_CLUSTER)
