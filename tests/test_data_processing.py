import pandas as pd

def process_test_data(df):
    """
    Processes raw credit risk data to match features used during training.
    Returns a DataFrame ready for prediction.
    """

    # Convert transaction time to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

    # Extract time-based features
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year

    # Handle missing time values if any
    df['Transaction_Hour'].fillna(0, inplace=True)
    df['Transaction_Day'].fillna(0, inplace=True)
    df['Transaction_Month'].fillna(0, inplace=True)
    df['Transaction_Year'].fillna(0, inplace=True)

    # Group-level aggregations (per CountryCode)
    agg = df.groupby('CountryCode').agg({
        'Amount': ['sum', 'mean', 'std'],
        'Transaction_Hour': pd.Series.nunique,
        'TransactionId': 'count'
    }).reset_index()

    # Rename columns
    agg.columns = [
        'CountryCode', 'Total_Amount', 'Avg_Amount', 'Std_Amount',
        'Unique_Hours', 'Num_Transactions'
    ]

    # Select base columns (assumes each CountryCode appears at least once)
    base = df[['CountryCode', 'Amount', 'Value', 'PricingStrategy', 'FraudResult',
               'Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year']
             ].drop_duplicates(subset='CountryCode')

    # Merge aggregate features into base
    merged = pd.merge(base, agg, on='CountryCode', how='left')

    # Final column order
    final_columns = [
        'CountryCode',
        'Amount',
        'Value',
        'PricingStrategy',
        'FraudResult',
        'Transaction_Hour',
        'Transaction_Day',
        'Transaction_Month',
        'Transaction_Year',
        'Total_Amount',
        'Avg_Amount',
        'Std_Amount',
        'Num_Transactions',
        'Unique_Hours'
    ]

    return merged[final_columns]
