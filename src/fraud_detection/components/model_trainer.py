import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from fraud_detection import logger
from fraud_detection.entity import ModelTrainerConfig
import mlflow
from urllib.parse import urlparse

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            
            X_train = train_data.drop(['is_fraud'], axis=1)
            y_train = train_data['is_fraud']

            # Hyperparameter Tuning Loop (Simplified Grid Search for MLflow Demo)
            n_estimators_list = [50, 100, 150]
            max_depth_list = [5, 10, 15]

            for n_est in n_estimators_list:
                for depth in max_depth_list:
                    with mlflow.start_run(run_name=f"RF_est{n_est}_depth{depth}"):
                        rf = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_split=self.config.min_samples_split,
                            criterion=self.config.criterion,
                            random_state=42
                        )
                        rf.fit(X_train, y_train)

                        # Log Params and Metrics
                        mlflow.log_param("n_estimators", n_est)
                        mlflow.log_param("max_depth", depth)
                        
                        score = rf.score(X_train, y_train)
                        mlflow.log_metric("train_accuracy", score)
                        
                        logger.info(f"Model trained with n_est={n_est}, depth={depth}, Score={score}")

            # Save the final model (Using the config params as the 'production' choice)
            final_rf = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                criterion=self.config.criterion,
                random_state=42
            )
            final_rf.fit(X_train, y_train)
            
            joblib.dump(final_rf, os.path.join(self.config.root_dir, self.config.model_name))
            logger.info(f"Final model saved at {self.config.root_dir}")

        except Exception as e:
            raise e
