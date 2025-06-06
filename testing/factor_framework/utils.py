# done 

"""
Utility helper functions - Providing various helper functionalities
"""
import pandas as pd
import numpy as np
from scipy import stats
import time
import functools


def normalize_factor(factor):
    """Normalize factor values (mean=0, std=1)"""
    return (factor - factor.mean()) / factor.std()



# remove auto corr 
def neutralize_factor(factor, neutralizers):
    
    # Ensure factor is pandas Series
    if not isinstance(factor, pd.Series):
        if isinstance(factor, np.ma.MaskedArray):
            factor = pd.Series(factor.data, index=neutralizers.index)
        else:
            factor = pd.Series(factor, index=neutralizers.index)
    
    # Ensure neutralizers is DataFrame
    if not isinstance(neutralizers, pd.DataFrame):
        if isinstance(neutralizers, pd.Series):
            neutralizers = pd.DataFrame(neutralizers)
        else:
            neutralizers = pd.DataFrame(neutralizers, index=factor.index)
    
    # Remove missing values
    valid_data = pd.concat([factor, neutralizers], axis=1).dropna()
    
    # If too few data, return original factor
    if len(valid_data) < 10:
        return factor
    
    # Extract factor and neutralizing variables
    y = valid_data.iloc[:, 0]
    X = valid_data.iloc[:, 1:]
    
    # Add constant term
    X = pd.concat([pd.Series(1, index=X.index), X], axis=1)
    
    try:
        # Calculate regression coefficients
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        
        # Calculate residuals (neutralized factor)
        residual = y - X @ beta
        
        # Create Series with original index
        neutralized = pd.Series(np.nan, index=factor.index)
        neutralized.loc[residual.index] = residual
        
        return neutralized
    except Exception as e:
        print(f"Neutralization failed, returning original factor")
        return factor


# cut outliers 
def winsorize_factor(factor, limits=(0.001, 0.001)):
    """Winsorize factor values, limiting the influence of extreme values"""
    if not isinstance(factor, pd.Series):
        try:
            factor = pd.Series(factor)
        except:
            return factor
    
    try:
        # Drop NaN values for winsorization
        valid_data = factor.dropna()
        
        # Calculate percentiles
        lower = valid_data.quantile(limits[0])
        upper = valid_data.quantile(1 - limits[1])
        
        # Apply winsorization
        winsorized = factor.copy()
        winsorized[factor < lower] = lower
        winsorized[factor > upper] = upper
        
        return winsorized
    except Exception as e:
        # If winsorization fails, return original factor
        return factor

def rank_transform_factor(factor):
    """Apply rank transformation to factor"""
    if not isinstance(factor, pd.Series):
        try:
            factor = pd.Series(factor)
        except:
            return factor
    
    # Apply rank transform
    return factor.rank(method='first', pct=True)


# 標準化 
def standardize_factor(factor):
    """Apply Z-score transformation to factor"""
    if not isinstance(factor, pd.Series):
        try:
            factor = pd.Series(factor)
        except:
            return factor
    
    # Apply standardization (z-score)
    mean = factor.mean()
    std = factor.std()
    
    if std == 0:
        return factor - mean
    
    return (factor - mean) / std

def sigmoid_transform(factor):
    """Apply Sigmoid transformation to factor"""
    if not isinstance(factor, pd.Series):
        try:
            factor = pd.Series(factor)
        except:
            return factor
    
    # Apply sigmoid transformation
    return 1 / (1 + np.exp(-factor))



# ---------------------- tool --------------------------------
# 計時 
def time_it(func):
    """Decorator to measure execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper



def calculate_ic(factor, returns):
    """Calculate Information Coefficient (IC)"""
    # Ensure both are Series
    if not isinstance(factor, pd.Series):
        factor = pd.Series(factor)
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Align indices
    aligned_data = pd.concat([factor, returns], axis=1).dropna()
    
    if len(aligned_data) < 5:
        return {'IC': 0, 't_stat': 0, 'p_value': 1}
    
    # Calculate IC (Spearman rank correlation)
    ic, p_value = stats.spearmanr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
    
    # Calculate t-statistic
    t_stat = ic * np.sqrt((len(aligned_data) - 2) / (1 - ic**2))
    
    return {'IC': ic, 't_stat': t_stat, 'p_value': p_value}


# 合併多期回報 
def merge_periods(df, periods):
    """Merge multi-period returns into a single DataFrame"""
    merged = {}
    
    for period in periods:
        col_name = f'return_{period}'
        col = df[col_name] if col_name in df.columns else None
        
        if col is None:
            # Try alternative naming
            col_name = f'future_return_{period}'
            col = df[col_name] if col_name in df.columns else None
        
        if col is not None:
            merged[period] = col
    
    return pd.DataFrame(merged)

# 確保資料是 pandas Series 或 DataFrame 
def ensure_series(data, index=None):
    """Ensure data is a pandas Series"""
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    elif hasattr(data, 'mask') and hasattr(data, 'data'):
        # Handle MaskedArray
        series = pd.Series(data.data, index=index)
        if hasattr(data, 'mask'):
            mask = data.mask if isinstance(data.mask, np.ndarray) else np.array(data.mask)
            series[mask] = np.nan
        return series
    else:
        try:
            return pd.Series(data, index=index)
        except:
            # Last resort
            return pd.Series(np.nan, index=index)

# 確保資料是 pandas DataFrame 
def ensure_dataframe(data, index=None):
    """Ensure data is a pandas DataFrame"""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pd.Series):
        return pd.DataFrame(data)
    else:
        try:
            return pd.DataFrame(data, index=index)
        except:
            # Last resort
            return pd.DataFrame(index=index)
