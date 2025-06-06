"""
因子評估模組 - 評估因子預測能力的工具
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .data import calculate_forward_returns

def analyze_factor(df, factor, periods=[1, 5, 10], price_col='close'):
    """
    Comprehensive single-factor analysis with improved error handling
    and robust quantile calculation
    """
    results = {}
    
    # Ensure factor values are DataFrame or Series
    if not isinstance(factor, (pd.DataFrame, pd.Series)):
        factor = pd.Series(factor, index=df.index)
    
    # Calculate future returns
    forward_returns = calculate_forward_returns(df, price_col=price_col, periods=periods)
    
    # Shift factor to ensure using past data to predict future
    factor_lag = factor.shift(1)
    
    # Remove NaN values
    valid_data = pd.concat([factor_lag, forward_returns], axis=1).dropna()
    factor_lag = valid_data.iloc[:, 0]  # More robust than using name
    
    # Analyze each period
    for period in periods:
        period_results = {}
        returns_col = f'forward_return_{period}'
        
        if returns_col not in valid_data.columns:
            print(f"Warning: Return column '{returns_col}' not found for period {period}")
            # Create default empty results
            period_results['IC'] = 0
            period_results['t_stat'] = 0
            period_results['quintile_returns'] = pd.Series()
            period_results['spread'] = 0
            results[period] = period_results
            continue
        
        # Information coefficient (IC)
        try:
            ic = stats.spearmanr(factor_lag, valid_data[returns_col])[0]
            if np.isnan(ic):
                ic = 0
                print(f"Warning: IC is NaN for period {period}, setting to 0")
            period_results['IC'] = ic
        except Exception as e:
            print(f"Error calculating IC for period {period}: {e}")
            period_results['IC'] = 0
        
        # Quantile analysis
        try:
            # Robust quintile calculation
            # First check if there's enough unique values
            unique_values = factor_lag.dropna().unique()
            
            if len(unique_values) < 5:
                print(f"Warning: Not enough unique values for quantile analysis (period {period})")
                # Fake quintile returns for consistency
                quintile_returns = pd.Series(
                    [0, 0, 0, 0, 0],
                    index=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
                )
            else:
                # Use qcut with duplicate handling
                quintiles = pd.qcut(
                    factor_lag, 
                    5, 
                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                    duplicates='drop'  # Handle duplicate bin edges
                )
                quintile_returns = valid_data.groupby(quintiles, observed=True)[returns_col].mean()
                
                # Ensure all quintiles are present
                for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    if q not in quintile_returns.index:
                        quintile_returns.loc[q] = 0
                        
                quintile_returns = quintile_returns.sort_index()
            
            period_results['quintile_returns'] = quintile_returns
            
            # Q5-Q1 return spread
            if 'Q5' in quintile_returns.index and 'Q1' in quintile_returns.index:
                period_results['spread'] = quintile_returns['Q5'] - quintile_returns['Q1']
            else:
                period_results['spread'] = 0
            
            # t-statistic - with robust calculation
            try:
                top_quintile = valid_data[factor_lag >= factor_lag.quantile(0.8)][returns_col]
                bottom_quintile = valid_data[factor_lag <= factor_lag.quantile(0.2)][returns_col]
                
                if len(top_quintile) > 0 and len(bottom_quintile) > 0:
                    spread_series = top_quintile - bottom_quintile.mean()  # More robust
                    
                    if len(spread_series) > 0 and spread_series.std() > 0:
                        t_stat = np.sqrt(len(spread_series)) * spread_series.mean() / spread_series.std()
                    else:
                        t_stat = 0
                else:
                    t_stat = 0
                    
                if np.isnan(t_stat) or np.isinf(t_stat):
                    t_stat = 0
                    
                period_results['t_stat'] = t_stat
            except Exception as e:
                print(f"Error calculating t-statistic for period {period}: {e}")
                period_results['t_stat'] = 0
                
        except Exception as e:
            print(f"Error in quintile analysis for period {period}: {e}")
            # Create default quintile returns
            period_results['quintile_returns'] = pd.Series(
                [0, 0, 0, 0, 0],
                index=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            )
            period_results['spread'] = 0
            period_results['t_stat'] = 0
            
        results[period] = period_results
    
    return results

def plot_quintile_returns(results, period, title=None):
    """Plot quintile returns analysis chart"""
    plt.figure(figsize=(10, 6))
    
    if period not in results or 'quintile_returns' not in results[period]:
        print(f"Quintile results for period {period} not available")
        return plt
        
    quintile_returns = results[period]['quintile_returns']
    
    if len(quintile_returns) == 0:
        print(f"No quintile returns data for period {period}")
        plt.text(0.5, 0.5, "No data available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        return plt
    
    bars = plt.bar(range(len(quintile_returns)), quintile_returns * 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.title(title or f'Period {period} Quintile Returns Analysis')
    plt.xlabel('Quintile')
    plt.ylabel('Average Return (%)')
    plt.xticks(range(len(quintile_returns)), quintile_returns.index)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def calculate_ic_series(df, factor, period=5, price_col='close', window=24*21):
    """Calculate rolling information coefficient time series"""
    try:
        # Ensure factor values are DataFrame or Series
        if not isinstance(factor, (pd.DataFrame, pd.Series)):
            factor = pd.Series(factor, index=df.index)
        
        # Calculate future returns
        returns = df[price_col].pct_change(period).shift(-period)
        
        # Shift factor to ensure using past data to predict future
        factor_lag = factor.shift(1)
        
        # Remove NaN values
        valid_data = pd.concat([factor_lag, returns], axis=1).dropna()
        
        if len(valid_data) <= window:
            print(f"Warning: Not enough data points for IC series calculation (need > {window})")
            return pd.Series()
            
        # Initialize IC series
        ic_series = pd.Series(index=valid_data.index)
        
        # Calculate rolling IC
        for i in range(len(valid_data) - window + 1):
            window_data = valid_data.iloc[i:i+window]
            try:
                ic = stats.spearmanr(window_data.iloc[:, 0], window_data.iloc[:, 1])[0]
                if np.isnan(ic):
                    ic = 0
                ic_series.iloc[i+window-1] = ic
            except:
                ic_series.iloc[i+window-1] = 0
        
        return ic_series
    except Exception as e:
        print(f"Error calculating IC series: {e}")
        return pd.Series()

def plot_ic_series(ic_series, title=None):
    """Plot information coefficient time series chart"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(ic_series.index, ic_series.values)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=ic_series.mean(), color='g', linestyle='--', 
               label=f'Average IC: {ic_series.mean():.4f}')
    
    plt.title(title or 'Factor Information Coefficient (IC) Time Series')
    plt.xlabel('Date')
    plt.ylabel('Information Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def calculate_turnover(factor, window=1):
    """Calculate factor turnover rate"""
    # Standardize factor values
    factor_std = factor.rank(pct=True)
    
    # Calculate rank changes between adjacent time points
    rank_diff = factor_std.diff(window).abs()
    
    # Calculate average turnover rate
    turnover = rank_diff.mean() / 2  # Divide by 2 because each rank change affects two assets
    
    return turnover

def factor_stability(factor, window=20):
    """Calculate factor stability (autocorrelation coefficient)"""
    # Calculate rolling autocorrelation coefficient
    autocorr = factor.rolling(window=window).corr(factor.shift(1))
    
    return autocorr.mean()
