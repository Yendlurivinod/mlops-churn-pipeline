import pandas as pd
import numpy as np
from scipy import stats
import json
import os

FEATURES = ['recency', 'frequency', 'monetary', 'avg_order_value',
            'high_value', 'recency_score']
DRIFT_THRESHOLD = 0.05  # p-value threshold for KS test

def load_datasets(reference_path, current_path):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    print(f"✓ Loaded reference ({len(reference)} rows) and current ({len(current)} rows)")
    return reference, current

def run_ks_test(reference, current):
    results = {}
    drift_detected = False

    print("\n--- Feature Drift Report ---")
    for feature in FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            print(f"⚠ Feature '{feature}' not found — skipping")
            continue

        ks_stat, p_value = stats.ks_2samp(
            reference[feature].dropna(),
            current[feature].dropna()
        )

        drifted = p_value < DRIFT_THRESHOLD
        if drifted:
            drift_detected = True

        results[feature] = {
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": bool(drifted)
        }

        status = "⚠ DRIFT" if drifted else "✓ OK"
        print(f"{status} | {feature:<20} | KS={ks_stat:.4f} | p={p_value:.4f}")

    return results, drift_detected

def check_population_shift(reference, current):
    ref_size = len(reference)
    cur_size = len(current)
    shift = abs(ref_size - cur_size) / ref_size
    if shift > 0.2:
        print(f"⚠ Population size shift detected — {shift:.1%} change")
        return True
    print(f"✓ Population size stable — {shift:.1%} change")
    return False

def save_drift_report(results, drift_detected, path="data/processed/drift_report.json"):
    report = {
        "drift_detected": bool(drift_detected),
        "features": results
    }
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Drift report saved to {path}")

def detect_drift(reference_path="data/processed/reference.csv",
                 current_path="data/processed/features.csv"):
    print("\n--- Running Drift Detection ---")

    if not os.path.exists(reference_path):
        print("⚠ No reference dataset found — skipping drift detection")
        return False

    reference, current = load_datasets(reference_path, current_path)
    results, feature_drift = run_ks_test(reference, current)
    population_drift = check_population_shift(reference, current)
    drift_detected = feature_drift or population_drift

    save_drift_report(results, drift_detected)

    print(f"\n{'⚠ DRIFT DETECTED — triggering retraining' if drift_detected else '✓ No drift detected — model stable'}")
    return drift_detected

if __name__ == "__main__":
    drift = detect_drift()
    exit(1 if drift else 0)
