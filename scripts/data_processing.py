
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


# -------------------------
# Load raw data
# -------------------------
df = pd.read_csv("../data/raw/data.csv")
# -------------------------
# Feature engineering
# -------------------------
def feature_engineering(df):
    # Convert datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Extract datetime features
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year

    # Aggregate customer-level features
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Transaction_Hour': 'nunique'
    }).reset_index()

    agg_df.columns = ['CustomerId', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Num_Transactions', 'Unique_Hours']
    df = pd.merge(df, agg_df, on='CustomerId', how='left')

    return df


# -------------------------
# Compute RFM metrics
# -------------------------
def compute_rfm(df, snapshot_date):
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'nunique',
        'Amount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm


# -------------------------
# Label high-risk customers using KMeans
# -------------------------
def label_high_risk_customers(rfm_df, n_clusters=3):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # High-risk = customers with highest Recency (least recent activity)
    cluster_stats = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats['Recency'].idxmax()

    rfm_df['is_high_risk'] = rfm_df['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
    return rfm_df[['CustomerId', 'is_high_risk']]


# -------------------------
# Merge target variable
# -------------------------
def add_target_variable(df, rfm_target_df):
    return df.merge(rfm_target_df, on='CustomerId', how='left')


# -------------------------
# Save processed data
# -------------------------
def save_processed_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.csv")
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")

    df = pd.read_csv("../data/raw/data.csv")
    df = feature_engineering(df)

    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = compute_rfm(df, snapshot_date)
    rfm_target_df = label_high_risk_customers(rfm)
    df = add_target_variable(df, rfm_target_df)

    print("DataFrame shape before saving:", df.shape)
    save_processed_data(df, path=PROCESSED_PATH)
    print("✅ Data processed and saved to:", PROCESSED_PATH)
