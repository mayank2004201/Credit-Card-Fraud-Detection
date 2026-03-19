import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def predict(self, data):
        # We expect data to be a dataframe with V1-V28 and Amount
        v_cols = [f'V{i}' for i in range(1, 29)]
        cols = v_cols + ['Amount']
        data = data[cols] 
        prediction = self.model.predict(data)
        return prediction
