# Project Walkthrough: Fraud Detection Platform

Congratulations! You now have an industry-level Machine Learning platform. This guide will help you run and showcase your project to recruiters.

## 1. Setup Your Credentials
Open the `.env` file in the root directory and fill in your DagsHub or MLflow credentials:
*   `MLFLOW_TRACKING_URI`: Found in your DagsHub repository settings.
*   `MLFLOW_TRACKING_USERNAME`: Your DagsHub username.
*   `MLFLOW_TRACKING_PASSWORD`: Your DagsHub access token.

## 2. Execute the Automated Pipeline
Run the following command to process your data, handle the class imbalance (SMOTE), and train your model with hyperparameter tuning.
```bash
python main.py
```
*   **What happens?** The system will create an `artifacts/` folder and log every tuning iteration to MLflow.

## 3. Launch the Backend API
Start the FastAPI server to handle real-time prediction requests.
```bash
python app.py
```
*   The API will be available at `http://localhost:8080`.

## 4. Launch the Recruiter Dashboard
Open a new terminal and launch the interactive UI.
```bash
streamlit run dashboard/app.py
```
*   **EDA Tab**: Interact with Plotly charts of your transaction data.
*   **Prediction Tab**: Test the model live by entering transaction details.

## 5. View MLOps Insights
Enter `mlflow ui` in your terminal to view the detailed charts of your hyperparameter tuning and model performance metrics.

---
**Recruiter Showcase Tips:**
- Highlight the **Modular Architecture** (how `src/` is separate from `api/`).
- Show the **SMOTE** logic in `src/fraud_detection/components/data_transformation.py` as your solution for imbalanced data.
- Demonstrate the **MLflow tuning runs** to show you are data-driven.
