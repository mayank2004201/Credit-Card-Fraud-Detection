import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_fraud_distribution(df):
    target = 'Class' if 'Class' in df.columns else 'is_fraud'
    fig = px.pie(df, names=target, title="Fraud vs Non-Fraud Distribution",
                 color_discrete_sequence=['#2ecc71', '#e74c3c'])
    return fig

def plot_amount_distribution(df):
    col = 'Amount' if 'Amount' in df.columns else 'amt'
    fig = px.histogram(df, x=col, color='Class' if 'Class' in df.columns else 'is_fraud', 
                       title="Transaction Amount Distribution",
                       marginal="box", nbins=50)
    return fig

def plot_correlation_heatmap(df):
    # Only plot V columns and Amount/Class
    cols = [f'V{i}' for i in range(1, 29)] + (['Amount', 'Class'] if 'Class' in df.columns else ['amt', 'is_fraud'])
    existing_cols = [c for c in cols if c in df.columns]
    corr = df[existing_cols].corr()
    fig = px.imshow(corr, title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r')
    return fig

def plot_v_features(df, v_num):
    col = f'V{v_num}'
    fig = px.violin(df, x='Class', y=col, color='Class', 
                     title=f"Distribution of {col} by Class", box=True)
    return fig
