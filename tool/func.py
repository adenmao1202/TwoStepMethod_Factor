# factor calculation func 

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def ts_mean(series, window):
    # Calculate time series mean over a rolling window using only past data
    return series.rolling(window=window).mean()

def calculate_vwap(df, price_col='close', volume_col='volume', window=1):
    # Calculate Volume Weighted Average Price (VWAP)
    df = df.copy()
    # VWAP = sum(price * volume) / sum(volume)
    df['vwap'] = ((df[price_col] * df[volume_col]).rolling(window=window).sum() / 
                  df[volume_col].rolling(window=window).sum())
    return df

def calculate_price_to_vwap_alpha(df, price_col='close', vwap_mean_window=3, lag=1):
    # Calculate alpha based on the difference between price and VWAP moving average
    df = df.copy()
    
    # Calculate VWAP
    df = calculate_vwap(df, price_col=price_col)
    
    # Shift prices and VWAP to ensure we only use past data
    price_lag = df[price_col].shift(lag)
    vwap_lag = df['vwap'].shift(lag)
    
    # Calculate ts_mean using lagged VWAP data
    vwap_mean = ts_mean(vwap_lag, vwap_mean_window)
    
    # Calculate the alpha
    alpha = -(price_lag - vwap_mean)
    
    return alpha

def calculate_forward_returns(df, price_col='close', periods=[1, 5, 10]):
    # Calculate forward returns for multiple periods
    returns = {}
    for period in periods:
        returns[f'forward_return_{period}'] = (
            df[price_col].shift(-period) / df[price_col] - 1
        )
    return pd.DataFrame(returns, index=df.index)

def calculate_rolling_ic(signals, returns, window=252):
    # Calculate rolling Information Coefficient between signals and returns
    rolling_ic = pd.Series(index=signals.index, dtype=float)
    
    for i in range(window, len(signals)):
        if i % 1000 == 0:  # Progress indicator
            print(f"Processing IC calculation: {i}/{len(signals)}", end='\r')
            
        s_window = signals.iloc[i-window:i]
        r_window = returns.iloc[i-window:i]
        valid_data = ~(s_window.isna() | r_window.isna())
        if valid_data.sum() > 0:
            rolling_ic.iloc[i] = stats.spearmanr(
                s_window[valid_data], 
                r_window[valid_data]
            )[0]
            
    return rolling_ic

def analyze_alpha(df, alpha_func=None, alpha_col=None, periods=[1, 5, 10], 
                  price_col='close', window=252, **alpha_kwargs):
    # Comprehensive alpha analysis with look-ahead bias prevention
    df = df.copy()
    results = {}
    
    # Calculate alpha first (using only past data)
    print("Calculating alpha...")
    if alpha_func is not None:
        df['alpha'] = alpha_func(df, **alpha_kwargs)
    elif alpha_col is not None:
        if alpha_col in df.columns:
            df['alpha'] = df[alpha_col]
        else:
            raise ValueError(f"Alpha column '{alpha_col}' not found in DataFrame")
    else:
        raise ValueError("Either alpha_func or alpha_col must be provided")
    
    # Calculate forward returns (our targets)
    print("Calculating forward returns...")
    forward_returns = calculate_forward_returns(df, price_col=price_col, periods=periods)
    df = pd.concat([df, forward_returns], axis=1)
    
    # Shift alpha to ensure we're using past signal for future returns
    df['alpha_lag'] = df['alpha'].shift(1)
    
    # Remove NaN values
    df = df.dropna()
    
    print("Running period analysis...")
    for period in periods:
        print(f"\nAnalyzing period {period}...")
        return_col = f'forward_return_{period}'
        period_results = {}
        
        # Calculate IC using lagged alpha
        ic = stats.spearmanr(df['alpha_lag'], df[return_col])[0]
        period_results['IC'] = ic
        
        # Calculate rolling IC
        print(f"Calculating rolling IC for period {period}...")
        rolling_ic = calculate_rolling_ic(df['alpha_lag'], df[return_col], window=window)
        valid_ic = rolling_ic.dropna()
        if len(valid_ic) > 0:
            icir = valid_ic.mean() / valid_ic.std()
            period_results['ICIR'] = icir
        else:
            period_results['ICIR'] = np.nan
        
        # Calculate turnover
        period_results['turnover'] = (
            df['alpha_lag'].diff().abs().mean() / df['alpha_lag'].abs().mean()
        )
        
        # Calculate quintile returns
        try:
            quintiles = pd.qcut(df['alpha_lag'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            quintile_returns = df.groupby(quintiles)[return_col].mean()
            period_results['quintile_returns'] = quintile_returns
            
            # Calculate spread (Q5-Q1)
            period_results['spread'] = quintile_returns['Q5'] - quintile_returns['Q1']
            
            # Calculate t-stat of spread
            spread_series = df[df['alpha_lag'] >= df['alpha_lag'].quantile(0.8)][return_col] - \
                           df[df['alpha_lag'] <= df['alpha_lag'].quantile(0.2)][return_col]
            t_stat = np.sqrt(len(spread_series)) * spread_series.mean() / spread_series.std()
            period_results['t_stat'] = t_stat
            
        except Exception as e:
            print(f"Error in quintile calculation for period {period}: {e}")
            continue
            
        results[period] = period_results
    
    return results, df

def plot_quintile_analysis(results, period, title_prefix="Alpha"):
    # Plot quintile analysis results
    plt.figure(figsize=(10, 6))
    quintile_returns = results[period]['quintile_returns']
    
    bars = plt.bar(range(len(quintile_returns)), quintile_returns * 1e5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title(f'Quintile Returns Analysis for {title_prefix} (Period: {period})\nUsing Past Data Only')
    plt.xlabel('Quintiles')
    plt.ylabel('Average Return (1e-5)')
    plt.xticks(range(len(quintile_returns)), quintile_returns.index)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def print_analysis_results(results):
    # Print comprehensive analysis results
    for period, period_results in results.items():
        print(f"\nPeriod {period} Analysis (Using Past Data Only):")
        print("-" * 50)
        print(f"Information Coefficient (IC): {period_results['IC']:.4f}")
        print(f"IC-IR: {period_results['ICIR']:.4f}")
        print(f"Turnover: {period_results['turnover']:.4f}")
        print(f"Q5-Q1 Spread: {period_results['spread']*1e5:.4f} (1e-5)")
        print(f"T-Statistic: {period_results['t_stat']:.4f}")
        print("\nQuintile Returns (1e-5):")
        print(period_results['quintile_returns'] * 1e5)

def aggregate_ohlcv(df, freq='1H', price_col='close', volume_col='volume', 
                   open_col='open', high_col='high', low_col='low', 
                   time_col='open_time'):
    # Aggregate OHLCV data to a specified frequency
    # Ensure timestamp is index
    if time_col in df.columns:
        df = df.set_index(time_col)
    
    # Define aggregation functions
    agg_dict = {
        open_col: 'first',
        high_col: 'max',
        low_col: 'min',
        price_col: 'last',
        volume_col: 'sum'
    }
    
    # Only include columns that exist in the DataFrame
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    # Resample and aggregate
    resampled = df.resample(freq).agg(agg_dict)
    
    return resampled