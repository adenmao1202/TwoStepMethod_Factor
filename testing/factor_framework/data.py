# Done 


import pandas as pd
import numpy as np

def load_data(filename, date_column='open_time'):
    
    df = pd.read_csv(filename)
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    return df

def calculate_forward_returns(df, price_col='close', periods=[1, 6, 24]):
    
    returns = {}
    for period in periods:
        returns[f'forward_return_{period}'] = (
            df[price_col].shift(-period) / df[price_col] - 1
        )
    return pd.DataFrame(returns, index=df.index)

def calculate_vwap(df, price_col='close', volume_col='volume', window=6):
    
    df = df.copy()
    df['vwap'] = ((df[price_col] * df[volume_col]).rolling(window=window).sum() / 
                 df[volume_col].rolling(window=window).sum())
    return df

def split_train_test(df, test_ratio=0.3):
    
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test

def resample_data(df, freq='1H'):
    
    return df.resample(freq).last()

# diff missing values handler 
def handle_missing_values(df, method='ffill'):
    
    if method == 'ffill':
        return df.fillna(method='ffill')
    elif method == 'bfill':
        return df.fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    else:
        return df.dropna()
