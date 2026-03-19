import os
import pandas as pd
from fraud_detection import logger
from fraud_detection.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            # Define expected columns (simplified for brevity)
            expected_cols = ["amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long", "is_fraud"]

            for col in expected_cols:
                if col not in all_cols:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
