import streamlit as st
import pandas as pd
import requests
import os
from dashboard.visualizations import plot_fraud_distribution, plot_amt_distribution, plot_correlation_heatmap

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Credit Card Fraud Detection Platform")
st.markdown("---")

# Load Sample Data for EDA
@st.cache_data
def load_data():
    if os.path.exists("data/fraudTrain.csv"):
        return pd.read_csv("data/fraudTrain.csv").head(1000)
    return None

df = load_data()

tabs = st.tabs(["📊 Interactive EDA", "🔍 Real-Time Prediction", "⚙️ MLOps Tracking"])

with tabs[0]:
    st.header("Exploratory Data Analysis")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_fraud_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_amt_distribution(df), use_container_width=True)
        
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
    else:
        st.warning("Data not found. Please run the pipeline first.")

with tabs[1]:
    st.header("Test the Model")
    st.write("Enter transaction details to predict fraud probability.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("Transaction Amount ($)", value=100.0)
            zip_code = st.number_input("Zip Code", value=12345)
            city_pop = st.number_input("City Population", value=50000)
        with col2:
            unix_time = st.number_input("Unix Time", value=1371251923)
            lat = st.number_input("Latitude", value=40.0)
            long = st.number_input("Longitude", value=-74.0)
            
        submit = st.form_submit_button("Predict")
        
        if submit:
            payload = {
                "amt": amt, "zip": zip_code, "lat": lat, "long": long,
                "city_pop": city_pop, "unix_time": unix_time,
                "merch_lat": lat + 0.01, "merch_long": long + 0.01
            }
            try:
                response = requests.post("http://localhost:8080/predict", json=payload)
                result = response.json()
                if result["prediction"] == 1:
                    st.error("🚨 ALERT: Fraudulent Transaction Detected!")
                else:
                    st.success("✅ Transaction is Legitimate.")
            except Exception as e:
                st.error(f"API Error: {e}. Is the FastAPI server running?")

with tabs[2]:
    st.header("MLOps Lifecycle")
    st.write("This project uses **MLflow** for experiment tracking and model registry.")
    st.info("To view the full experiment history, run `mlflow ui` in your terminal.")
    st.markdown("""
    **Key Metrics Tracked:**
    - Accuracy / Recall / F1-Score
    - Hyperparameter tuning results
    - Training duration
    """)
