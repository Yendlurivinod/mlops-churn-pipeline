import pandas as pd
import os
import json
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

FEATURES = ['recency', 'frequency', 'monetary', 'avg_order_value',
            'high_value', 'recency_score']
TARGET = 'churned'
MODEL_THRESHOLD = 0.75

def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]
    print(f"✓ Data loaded — {len(df)} records, {len(FEATURES)} features")
    return X, y

def train_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"✓ Model trained")
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "f1_score": round(f1_score(y_test, preds), 4)
    }
    print(f"✓ Evaluation — Accuracy: {metrics['accuracy']} | AUC: {metrics['roc_auc']} | F1: {metrics['f1_score']}")
    return metrics

def passes_threshold(metrics):
    if metrics['roc_auc'] >= MODEL_THRESHOLD:
        print(f"✓ Model passed deployment threshold (AUC {metrics['roc_auc']} >= {MODEL_THRESHOLD})")
        return True
    print(f"✗ Model failed threshold (AUC {metrics['roc_auc']} < {MODEL_THRESHOLD}) — skipping deploy")
    return False

def save_metrics(metrics, path="data/processed/latest_metrics.json"):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {path}")

def log_experiment(params, metrics, run_id):
    log = {"run_id": run_id, "params": params, "metrics": metrics}
    os.makedirs("mlruns", exist_ok=True)
    with open(f"mlruns/run_{run_id}.json", 'w') as f:
        json.dump(log, f, indent=2)
    print(f"✓ Experiment logged to mlruns/run_{run_id}.json")

def run_training(data_path="data/processed/features.csv"):
    print("\n--- Running Training Pipeline ---")
    import time
    run_id = str(int(time.time()))

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8
    }
    log_experiment(params, metrics, run_id)
    save_metrics(metrics)

    if passes_threshold(metrics):
        os.makedirs("models", exist_ok=True)
        with open("models/churn_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("✓ Model saved to models/churn_model.pkl")
        return model, metrics

    return None, metrics

if __name__ == "__main__":
    run_training()
