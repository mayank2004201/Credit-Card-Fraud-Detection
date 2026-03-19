import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from fraud_detection.entity import ModelEvaluationConfig
from fraud_detection.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return accuracy, rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Integrated with environment variables
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run(run_name="Evaluation_Final_Model"):
            predictions = model.predict(test_x)
            (accuracy, rmse, mae, r2) = self.eval_metrics(test_y, predictions)
            
            # Saving metrics as local json
            scores = {"accuracy": accuracy, "rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model registry does not work with local file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestModel")
            else:
                mlflow.sklearn.log_model(model, "model")
