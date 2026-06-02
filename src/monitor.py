import pandas as pd
import numpy as np
from scipy import stats
import joblib
import os
import sys
import json

sys.path.append(os.path.dirname(__file__))
from preprocess import load_and_preprocess

def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold=0.05):
    drift_results = {}
    drifted_columns = []

    for col in reference.columns:
        # KS test checks if two distributions are different
        stat, p_value = stats.ks_2samp(reference[col], current[col])
        drift_detected = p_value < threshold
        drift_results[col] = {
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(drift_detected)
        }
        if drift_detected:
            drifted_columns.append(col)

    return drift_results, drifted_columns

def monitor():
    X_train, X_test, y_train, y_test = load_and_preprocess(
        "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    print("Running drift detection...\n")
    drift_results, drifted_columns = detect_drift(X_train, X_test)

    print(f"Total features checked : {len(drift_results)}")
    print(f"Features with drift    : {len(drifted_columns)}")

    if drifted_columns:
        print(f"\nDrifted features: {drifted_columns}")
    else:
        print("\nNo significant drift detected.")

    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w") as f:
        json.dump(drift_results, f, indent=2)

    print("\nDrift report saved to reports/drift_report.json")

if __name__ == "__main__":
    monitor()