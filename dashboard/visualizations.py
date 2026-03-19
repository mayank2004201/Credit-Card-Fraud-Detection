import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_fraud_distribution(df):
    fig = px.pie(df, names='is_fraud', title='Fraud vs Non-Fraud Distribution', color_discrete_sequence=['#00CC96', '#EF553B'])
    return fig

def plot_amt_distribution(df):
    fig = px.histogram(df, x='amt', color='is_fraud', barmode='overlay', title='Transaction Amount Distribution', nbins=50)
    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, title='Feature Correlation Heatmap')
    return fig
