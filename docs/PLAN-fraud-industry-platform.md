# PLAN: Fraud Detection Industry Platform

## Overview
High-quality portfolio project transforming a notebook-based fraud detection model into a production-ready ML platform. Includes modular pipelines, experiment tracking, a REST API, and an interactive recruiter-facing dashboard.

## Project Type
**BACKEND + WEB (Hybrid)**

## Success Criteria
*   [ ] **Modular Code**: Logic split into `src/`, `api/`, and `dashboard/`.
*   [ ] **MLOps**: MLflow integration for hyperparameter tuning and model logging.
*   [ ] **REST API**: FastAPI endpoints for single and batch fraud predictions.
*   [ ] **Dashboard**: Streamlit UI with integrated EDA (interactive charts) and prediction interface.
*   [ ] **Clean Code**: Adherence to industry standards (typing, documentation, config management).

## Tech Stack
*   **Language**: Python 3.x
*   **Data**: Pandas, Numpy, Scikit-learn, Imbalanced-learn (SMOTE)
*   **Reporting**: MLflow (Tracking Server)
*   **Backend**: FastAPI, Uvicorn
*   **Frontend**: Streamlit, Plotly (Interactive Viz)
*   **Environment**: Python-dotenv

## File Structure
```
.
├── api/                # FastAPI logic
│   ├── main.py
│   └── schemas.py
├── dashboard/          # Streamlit UI
│   ├── app.py
│   └── visualizations.py
├── data/               # CSV storage
├── docs/               # Planning & docs
├── models/             # Saved .pkl files
├── src/                # Core ML Library
│   ├── config.py       # Configuration
│   ├── data_loader.py  # ingestion & EDA helpers
│   ├── processing.py   # SMOTE & Scaling
│   └── trainer.py      # MLflow Tuning logic
├── requirements.txt
└── .env
```

## Task Breakdown

### Phase 1: Foundation & Data Processing
*   **task_1**: Initialize modular structure and `requirements.txt`.
*   **task_2**: Implement `src/data_loader.py` with outlier detection and `src/processing.py` with SMOTE logic.

### Phase 2: MLOps & Training
*   **task_3**: Implement `src/trainer.py` with MLflow tracking.
*   **task_4**: Run hyperparameter tuning for Random Forest/XGBoost and log top 5 models to MLflow.

### Phase 3: Backend API Development
*   **task_5**: Build FastAPI app in `api/main.py`.
*   **task_6**: Integrate model loading logic and `/predict` endpoint.

### Phase 4: Recruiter Dashboard (EDA + UI)
*   **task_7**: Build `dashboard/visualizations.py` with Plotly (Distributions, Correlation, Outliers).
*   **task_8**: Finish `dashboard/app.py` connecting the Viz + API inference.

## Phase X: Verification Matrix
*   [ ] No purple/violet hex codes used in charts (UI-UX protocol).
*   [ ] FastAPI docs (`/docs`) accessible and functional.
*   [ ] MLflow dashboard shows at least 5 tuning iterations.
*   [ ] `python .agent/scripts/verify_all.py` passes all critical checks.
