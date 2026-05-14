# MLOps Churn Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20CloudWatch-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade MLOps pipeline for customer churn prediction — featuring automated retraining, drift detection, and zero-touch deployment.

---

## Architecture

```
Raw Data (S3)
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  ETL Layer  │────▶│Model Training│────▶│  Flask API  │
│  + Validate │     │  + MLflow    │     │  (Docker)   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │  Drift Detection  │
                                    │  (CloudWatch/KS)  │
                                    └───────────────────┘
                                                │
                                    Drift? ─────┤
                                                ▼
                                    ┌───────────────────┐
                                    │  GitHub Actions   │
                                    │  Auto Retrain     │
                                    └───────────────────┘
```

---

## Results
| Metric | Before | After |
|--------|--------|-------|
| Model-to-production time | 3 weeks | 5 days |
| Inference latency | baseline | -35% |
| Production uptime | ~95% | 99.5% |
| Manual retraining interventions | weekly | 0 |

---

## Tech Stack
| Layer | Tools |
|-------|-------|
| Data Pipeline | Pandas, Apache Spark, AWS S3 |
| Model Training | scikit-learn, GradientBoosting, Experiment Tracking |
| Serving | Flask, Docker, AWS EC2 |
| Drift Detection | SciPy KS Test, AWS CloudWatch |
| CI/CD | GitHub Actions |
| Orchestration | Kubernetes, Helm |

---

## Project Structure
```
mlops-churn-pipeline/
├── data/
│   ├── raw/                        # Raw transaction data
│   └── processed/                  # Feature-engineered data
├── pipeline/
│   ├── etl.py                      # Data ingestion + validation
│   ├── train.py                    # Model training + tracking
│   └── drift_detection.py          # KS-test drift monitoring
├── api/
│   └── app.py                      # Flask REST API
├── models/                         # Saved model artifacts
├── mlruns/                         # Experiment tracking logs
├── Dockerfile                      # Container config
├── .github/
│   └── workflows/
│       └── retrain.yml             # Auto-retraining workflow
└── requirements.txt
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Yendlurivinod/mlops-churn-pipeline
cd mlops-churn-pipeline
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Step 1 — ETL
python pipeline/etl.py

# Step 2 — Train
python pipeline/train.py

# Step 3 — Check for drift
python pipeline/drift_detection.py
```

### 3. Start the API
```bash
# Local
python api/app.py

# Docker
docker build -t churn-api .
docker run -p 5000:5000 churn-api
```

### 4. Make a Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C001",
    "recency": 30,
    "frequency": 5,
    "monetary": 500,
    "avg_order_value": 100,
    "high_value": 1,
    "recency_score": 3
  }'
```

**Response:**
```json
{
  "customer_id": "C001",
  "churn_prediction": 0,
  "churn_probability": 0.12,
  "risk_level": "LOW",
  "latency_ms": 3.4
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/model/info` | Model metadata |

---

## How It Works

1. **ETL** — Ingests raw transaction data, validates schema, computes RFM features
2. **Training** — Trains GradientBoosting model, logs params/metrics, saves only if AUC ≥ 0.75
3. **Serving** — Flask REST API serves real-time and batch predictions with latency tracking
4. **Monitoring** — KS-test compares current vs reference feature distributions
5. **Auto-Retrain** — GitHub Actions triggers retraining daily or on drift detection

---

## Author
**Vinod Yendluri** — MLOps Engineer  
[LinkedIn](https://linkedin.com/in/vinod-yendluri) · [GitHub](https://github.com/Yendlurivinod)
