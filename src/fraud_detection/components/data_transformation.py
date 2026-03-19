import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from fraud_detection import logger
from fraud_detection.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformation(self):
        try:
            data = pd.read_csv(self.config.data_path)

            # Use the V1-V28 and Amount columns from the dataset
            v_cols = [f'V{i}' for i in range(1, 29)]
            numeric_cols = v_cols + ['Amount']
            X = data[numeric_cols]
            y = data[self.config.target_column]

            # Train Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            logger.info("Train test split completed")

            # SMOTE for imbalanced class
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE completed. Resampled shape: {X_train_res.shape}")

            # Scaling
            scaler = StandardScaler()
            X_train_res = scaler.fit_transform(X_train_res)
            X_test = scaler.transform(X_test)
            logger.info("Scaling completed")

            # Combine back to dataframe for saving
            train_df = pd.DataFrame(X_train_res, columns=numeric_cols)
            train_df["Class"] = y_train_res
            
            test_df = pd.DataFrame(X_test, columns=numeric_cols)
            test_df["Class"] = y_test.values

            train_df.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

            logger.info("Saved transformed data to artifacts")

        except Exception as e:
            raise e
