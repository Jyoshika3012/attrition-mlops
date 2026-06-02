# Employee Attrition MLOps Pipeline

An end-to-end production-grade ML system that predicts employee attrition, built with MLflow, FastAPI, Docker, and GitHub Actions CI/CD.

---

## Architecture

```
Raw Data → Preprocessing → Model Training → Experiment Tracking
                                                    ↓
Drift Monitoring ← Live API (FastAPI) ← Model Registry (MLflow)
                                                    ↓
                              CI/CD Pipeline (GitHub Actions)
```

---

## Features

- **ML Pipeline** — Data preprocessing, feature engineering, Random Forest classifier with class imbalance handling
- **Experiment Tracking** — MLflow logs every run with parameters, metrics, and artifacts
- **REST API** — FastAPI serves predictions with automatic Swagger documentation
- **Containerization** — Docker packages the entire app for consistent deployment
- **CI/CD** — GitHub Actions auto-retrains the model on every push
- **Drift Monitoring** — KS-test based feature drift detection alerts when data distribution shifts

---

## Tech Stack

| Component | Tool |
|---|---|
| ML Framework | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API Serving | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Drift Detection | SciPy (KS Test) |
| Data | IBM HR Analytics (Kaggle) |

---

## Project Structure

```
attrition-mlops/
├── src/
│   ├── preprocess.py      # Data cleaning and encoding
│   ├── train.py           # Model training with MLflow tracking
│   ├── predict.py         # Prediction utilities
│   └── monitor.py         # Data drift detection
├── api/
│   └── app.py             # FastAPI prediction endpoint
├── .github/workflows/
│   └── retrain.yml        # CI/CD pipeline
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/Jyoshika3012/attrition-mlops.git
cd attrition-mlops
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```

### 3. View experiments
```bash
mlflow ui --port 5001
# Open http://127.0.0.1:5001
```

### 4. Run the API
```bash
uvicorn api.app:app --reload
# Open http://127.0.0.1:8000/docs
```

### 5. Run with Docker
```bash
docker build -t attrition-mlops .
docker run -p 8000:8000 attrition-mlops
```

### 6. Check drift
```bash
python src/monitor.py
```

---

## API Usage

**POST** `/predict`

```json
{
  "Age": 35,
  "Department": "Sales",
  "JobSatisfaction": 2,
  "OverTime": "Yes",
  "MonthlyIncome": 4000
}
```

**Response**
```json
{
  "attrition": true,
  "probability": 0.5335,
  "risk": "High"
}
```

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 84.69% |
| F1 Score | 0.2857 |
| Precision | 0.3750 |
| Recall | 0.2308 |

*Note: Class imbalance handled using `class_weight='balanced'`. Dataset is 84% negative class.*

---

## Key Engineering Decisions

- **Class imbalance** — Used `class_weight='balanced'` instead of oversampling to avoid data leakage
- **Drift detection** — Kolmogorov-Smirnov test chosen over chi-square for continuous features
- **API design** — Pydantic models enforce strict input validation at the API layer
- **CI/CD** — Training pipeline runs on every push to ensure model is always up to date
