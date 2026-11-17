# Gym member clustering

Cluster gym members from the included `gym_members.csv`, choose k from training and test elbow curves, and value a fixed extras-spend credit policy. The recorded choice is **k = 4**, and the credit is offered **only to cluster 0**. Those choices came from the training analysis and are not revised from test or full-data results.

## Data

The analysis uses `gym_members.csv` in the project root. It has 330 members and these columns:

`member_id`, `age`, `visits_per_month`, `avg_visit_minutes`, `classes_per_month`, `months_member`, `monthly_extras_spend`

`member_id` is kept for tracking and is excluded from the scaler and KMeans. No separate download is required.

## Setup

The scripts were run with Python 3.14.7. Create a virtual environment and install the pinned packages:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

## Reproduce the analysis

Run the scripts in this order from the project root. Each later script reads files written by the earlier ones. `fit_clusters.py` refits the scaler and KMeans models and overwrites `models/kmeans_candidates.joblib` and the split, metrics, and elbow plot.

```bash
.venv\Scripts\python.exe inspect_data.py
.venv\Scripts\python.exe fit_clusters.py
.venv\Scripts\python.exe describe_segments.py
.venv\Scripts\python.exe inspect_credit_policy.py
.venv\Scripts\python.exe evaluate_policy.py
```

| Order | Script | What it does | Writes |
|---|---|---|---|
| 1 | `inspect_data.py` | Reports shape, types, missing values, duplicates, summary statistics, and invalid-value checks. | Nothing |
| 2 | `fit_clusters.py` | Splits 231/99 with `random_state=42`, fits `StandardScaler` and KMeans for k = 1..8 on the scaled training data only, and records train and test SSE. | `outputs/data_split.csv`, `outputs/fit_metrics.csv`, `outputs/elbow.png`, `models/kmeans_candidates.joblib` |
| 3 | `describe_segments.py` | Loads the saved scaler and k = 4 model, inverse-transforms the centers, and assigns all 330 members. | `outputs/segment_profiles.csv`, `outputs/member_segments.csv` |
| 4 | `inspect_credit_policy.py` | Scores the +$5 / -$10 credit on training members only, by cluster. | `outputs/training_credit_analysis.csv` |
| 5 | `evaluate_policy.py` | Applies `credit_policy.decide_credit` to the 330 members and to one new member who is not added to the totals. | `outputs/member_credit_results.csv`, `outputs/policy_value_summary.csv` |

`credit_policy.py` is imported by `evaluate_policy.py`. It loads the saved scaler, feature order, and k = 4 model. `decide_credit` returns that model's cluster id and whether the member receives the credit. Eligibility is cluster 0 only.

The saved model and `outputs/member_segments.csv` are in the repository, so `describe_segments.py`, `inspect_credit_policy.py`, and `evaluate_policy.py` can be run without calling `fit_clusters.py`. Use the same scikit-learn version as `requirements.txt`; a joblib model fitted with another version may not load.

`evaluate_policy.py` checks three recorded totals: full-data baseline net -$1,200, and training baseline -$855, v1 +$110, improvement +$965.
