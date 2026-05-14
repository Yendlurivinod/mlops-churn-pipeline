from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import time

app = Flask(__name__)

MODEL_PATH = "models/churn_model.pkl"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠ No model found at {MODEL_PATH} — run train.py first")

load_model()

FEATURES = ['recency', 'frequency', 'monetary', 'avg_order_value',
            'high_value', 'recency_score']

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    start = time.time()
    try:
        data = request.json
        df = pd.DataFrame([data])
        missing = set(FEATURES) - set(df.columns)
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400
        prediction = int(model.predict(df[FEATURES])[0])
        probability = round(float(model.predict_proba(df[FEATURES])[0][1]), 4)
        latency_ms = round((time.time() - start) * 1000, 2)
        return jsonify({
            "customer_id": data.get("customer_id", "unknown"),
            "churn_prediction": prediction,
            "churn_probability": probability,
            "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW",
            "latency_ms": latency_ms
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        data = request.json
        df = pd.DataFrame(data)
        predictions = model.predict(df[FEATURES]).tolist()
        probabilities = model.predict_proba(df[FEATURES])[:, 1].tolist()
        results = []
        for i, record in enumerate(data):
            results.append({
                "customer_id": record.get("customer_id", f"record_{i}"),
                "churn_prediction": int(predictions[i]),
                "churn_probability": round(probabilities[i], 4),
                "risk_level": "HIGH" if probabilities[i] > 0.7 else "MEDIUM" if probabilities[i] > 0.4 else "LOW"
            })
        return jsonify({
            "total_records": len(results),
            "high_risk_count": sum(1 for r in results if r["risk_level"] == "HIGH"),
            "predictions": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({"model_path": MODEL_PATH, "features": FEATURES, "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
