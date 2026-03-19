# 🛡️ Industry-Level Credit Card Fraud Detection Platform

A production-ready Machine Learning platform built with a modular, stage-based architecture. This project demonstrates end-to-end MLOps capabilities, from automated data ingestion and class imbalance handling (SMOTE) to real-time REST API serving and interactive visualization.

## 🚀 Key Features
*   **Modular Pipeline**: Decoupled stages for Data Ingestion, Validation, Transformation, Training, and Evaluation.
*   **MLOps Tracking (MLflow)**: Full experiment lifecycle tracking with hyperparameter tuning logged to an MLflow server.
*   **Imbalance Handling**: Integrated SMOTE (Synthetic Minority Over-sampling Technique) to handle highly skewed fraud data.
*   **FastAPI Backend**: High-performance REST API for real-time fraud prediction.
*   **Recruiter Dashboard**: Interactive Streamlit UI with Plotly visualizations for data storytelling and live model testing.
*   **Self-Initializing**: Runs with a single command; automated environment setup via `template.py`.

## 🛠️ Architecture
The system follows a "Clean Architecture" pattern, separating raw logic (components) from orchestration (pipelines).

```mermaid
graph LR
    A[Data Ingestion] --> B[Validation]
    B --> C[Transformation]
    C --> D[MLflow Training]
    D --> E[Evaluation]
    E --> F[FastAPI Service]
    F --> G[Streamlit UI]
```

## 📋 Getting Started

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
Executes ingestion, SMOTE, training, and evaluation in one go.
```bash
python main.py
```

### 3. Start the Services
**Launch API (Backend):**
```bash
python app.py
```
**Launch Dashboard (Frontend):**
```bash
streamlit run dashboard/app.py
```

### 4. View MLOps Tracking
```bash
mlflow ui
```

## 🧪 Tech Stack
- **Languages**: Python
- **ML**: Scikit-Learn, Imbalanced-Learn, SMOTE
- **MLOps**: MLflow
- **Backend**: FastAPI, Uvicorn
- **UI**: Streamlit, Plotly
- **DevOps**: Docker (Template ready)

---
*Created as a high-quality portfolio piece for Machine Learning Engineer / Data Scientist roles.*
