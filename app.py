from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.responses import Response
from fraud_detection.pipeline.prediction import PredictionPipeline
import pandas as pd

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(data: dict):
    try:
        # Example data input: { "amt": 100, "zip": 12345, ... }
        df = pd.DataFrame([data])
        obj = PredictionPipeline()
        predict = obj.predict(df)
        return {"prediction": int(predict[0])}
    except Exception as e:
        raise e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
