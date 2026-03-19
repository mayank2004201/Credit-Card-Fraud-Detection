import streamlit as st
import pandas as pd
import requests
import random
from visualizations import plot_fraud_distribution, plot_amount_distribution, plot_correlation_heatmap, plot_v_features

st.set_page_config(page_title="FraudShield AI", layout="wide", page_icon="🛡️")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #e74c3c; color: white; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ FraudShield AI: Detection Platform")
st.markdown("---")

tabs = st.tabs(["📊 Analytics Dashboard", "🔍 Real-Time Prediction", "⚙️ MLOps Tracking"])

# Load sample data for viz
@st.cache_data
def load_data():
    try:
        return pd.read_csv("artifacts/data_ingestion/fraudTrain.csv").head(5000)
    except:
        return None

df = load_data()

with tabs[0]:
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_fraud_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_amount_distribution(df), use_container_width=True)
        
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        
        st.subheader("Deep Dive into PCA Features")
        v_select = st.selectbox("Select Feature to Inspect:", [f"V{i}" for i in range(1, 29)])
        v_num = int(v_select[1:])
        st.plotly_chart(plot_v_features(df, v_num), use_container_width=True)
    else:
        st.warning("Please run the training pipeline (main.py) to generate data for visualization.")

with tabs[1]:
    st.subheader("Transaction Inference Engine")
    
    col_a, col_b = st.columns([1, 3])
    
    with col_a:
        if st.button("🎲 Generate Random Sample"):
            if df is not None:
                sample = df.sample(1).iloc[0]
                st.session_state['sample_data'] = sample.to_dict()
            else:
                st.error("No training data available to sample from.")

    # Form with 29 fields is too much, so we'll use a mix or a JSON area for "advanced" users,
    # and a simplified view with a random sample populate.
    
    if 'sample_data' in st.session_state:
        current_data = st.session_state['sample_data']
    else:
        current_data = {f"V{i}": 0.0 for i in range(1, 29)}
        current_data["Amount"] = 100.0

    with st.form("predict_form"):
        st.info("The model expects 28 PCA-transformed features (V1-V28) and the Transaction Amount.")
        
        # Display major features or just a subset for the UI, but send all to the API
        input_data = {}
        
        c1, c2, c3 = st.columns(3)
        input_data["Amount"] = c1.number_input("Amount ($)", value=float(current_data.get("Amount", 0.0)))
        input_data["V1"] = c2.number_input("V1 (Principal Component)", value=float(current_data.get("V1", 0.0)))
        input_data["V2"] = c3.number_input("V2 (Principal Component)", value=float(current_data.get("V2", 0.0)))
        
        with st.expander("Adjust all 28 PCA Features"):
            cols = st.columns(4)
            for i in range(1, 29):
                key = f"V{i}"
                input_data[key] = cols[(i-1)%4].number_input(key, value=float(current_data.get(key, 0.0)), format="%.4f")

        submit = st.form_submit_button("🛡️ ANALYZE TRANSACTION")
        
        if submit:
            try:
                # Call FastAPI backend
                response = requests.post("http://localhost:8080/predict", json=input_data)
                result = response.json()
                
                if result['prediction'] == 1:
                    st.error("🚨 FRAUD DETECTED! High sensitivity alert.")
                else:
                    st.success("✅ TRANSACTION SECURE. No fraud patterns detected.")
            except Exception as e:
                st.error(f"Prediction failed. Ensure API is running at localhost:8080. Error: {e}")

with tabs[2]:
    st.subheader("MLOps Health Monitoring")
    st.info("Check DagsHub for full experiment tracking results.")
    st.metric("Model Version", "v1.0.0", "Production")
    st.metric("Active Run Host", "DagsHub / MLflow")
    
    if st.button("Open MLflow Dashboard"):
        st.info("Visit your DagsHub repository to view the MLflow UI.")
