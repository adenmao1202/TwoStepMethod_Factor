# Watch this one !!! 


import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Union, Callable, Any
from tqdm.auto import tqdm
from .analysis import analyze_factor
from .weights import combine_factors
from .utils import normalize_factor, time_it


# 測試單一因子 
@time_it
def test_factor(
    df: pd.DataFrame,
    factor: Union[pd.Series, pd.DataFrame],
    periods: List[int] = [1, 6, 24],
    price_col: str = 'close',
    verbose: bool = True
) -> Dict[int, Dict[str, Any]]:
    
    # Timing
    start_time = time.time()
    
    # Normalize factor
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    
    normalized_factor = normalize_factor(factor)
    
    # Analyze factor
    results = {}
    for period in periods:
        period_results = analyze_factor(df, normalized_factor, periods=[period], price_col=price_col)
        results[period] = period_results[period]
    
    # Output execution time
    if verbose:
        print(f"test_factor execution time: {time.time() - start_time:.4f} seconds")
    
    return results


# 測試多因子組合 
@time_it
def test_factor_combination(
    df: pd.DataFrame,
    factors: List[Tuple[str, Union[pd.Series, pd.DataFrame]]],
    weights: np.ndarray = None,
    periods: List[int] = [1, 6, 24],
    price_col: str = 'close',
    normalize: bool = True
) -> Dict[int, Dict[str, Any]]:
    # Combine factors
    combined_factor = combine_factors(factors, weights, normalize)
    
    # Test combined factor
    results = test_factor(df, combined_factor, periods, price_col)
    
    return results


# 看 this one !!! 
def backtest_factor(
    df: pd.DataFrame,
    factor: Union[pd.Series, pd.DataFrame],
    holding_period: int = 24,
    n_groups: int = 5,
    price_col: str = 'close',
    transaction_cost: float = 0.001
) -> pd.DataFrame:
    
    # Standardize factor
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    
    # Normalize factor
    print("Normalizing factor...")
    normalized_factor = normalize_factor(factor)
    
    # Shift factor to ensure using past data to predict future
    factor_lag = normalized_factor.shift(1)
    
    # Create a working copy with reset index to handle multi-symbol data
    print("Preparing data for backtest...")
    df_reset = df.reset_index()
    
    # If there's no 'symbol' column but there are duplicate timestamps,
    # we'll create a dummy symbol column for proper grouping
    if 'symbol' not in df_reset.columns and df_reset.iloc[:, 0].duplicated().any():
        print("Warning: Duplicate timestamps detected. Adding dummy symbol column.")
        df_reset['symbol'] = 'default'
    
    # Create a working DataFrame with necessary data
    factor_reset = factor_lag.reset_index()
    
    # Merge data with factor
    print("Merging factor data with price data...")
    backtest_df = pd.merge(
        df_reset,
        factor_reset,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Rename the factor column if it got a generic name
    if 0 in backtest_df.columns:
        backtest_df.rename(columns={0: 'factor'}, inplace=True)
    
    # Drop rows with NaN factor values
    backtest_df = backtest_df.dropna(subset=['factor'])
    print(f"Processing {len(backtest_df)} valid data points...")
    
    # Calculate future returns
    results_data = []
    
    # Group by date (first column after reset_index) and symbol if it exists
    group_cols = [backtest_df.columns[0]]
    if 'symbol' in backtest_df.columns:
        group_cols.append('symbol')
    
    # Get groups for tqdm
    groups = list(backtest_df.groupby(group_cols))
    print(f"Processing {len(groups)} groups...")
    
    # Process each group with progress bar
    for _, group in tqdm(groups, desc="Calculating returns", unit="group"):
        try:
            # Get current timestamp and price
            current_timestamp = group.iloc[0, 0]  # First column is timestamp
            current_price = group[price_col].iloc[0]
            current_factor = group['factor'].iloc[0]
            
            # Get current symbol if available
            current_symbol = group['symbol'].iloc[0] if 'symbol' in group.columns else None
            
            # Find future data point
            future_timestamp = current_timestamp + pd.Timedelta(hours=holding_period)
            
            # Find matching future data
            if current_symbol is not None:
                future_data = df_reset[(df_reset.iloc[:, 0] >= future_timestamp) & 
                                      (df_reset['symbol'] == current_symbol)]
            else:
                future_data = df_reset[df_reset.iloc[:, 0] >= future_timestamp]
            
            # Skip if no future data found
            if future_data.empty:
                continue
                
            # Get the closest future data point
            future_data = future_data.iloc[0]
            future_price = future_data[price_col]
            
            # Calculate return
            future_return = future_price / current_price - 1
            future_return -= transaction_cost * 2  # Apply transaction costs
            
            # Store result
            result = {
                'timestamp': current_timestamp,
                'factor': current_factor,
                'return': future_return
            }
            
            # Add symbol if available
            if current_symbol is not None:
                result['symbol'] = current_symbol
                
            results_data.append(result)
            
        except Exception as e:
            print(f"Error processing group: {e}")
            continue
    
    # If no results, return empty DataFrame
    if not results_data:
        print("Warning: No valid backtest results generated.")
        return pd.DataFrame()
    
    print(f"Successfully processed {len(results_data)} data points.")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Assign quantiles
    if not results_df.empty:
        if n_groups > 1:
            try:
                results_df['group'] = pd.qcut(results_df['factor'], n_groups, labels=False)
            except Exception as e:
                print(f"Error creating quantiles: {e}")
                # Fallback to equal groups if qcut fails
                try:
                    results_df['group'] = pd.cut(results_df['factor'], n_groups, labels=False)
                except:
                    print("Error creating equal-width bins. Assigning group 0 to all rows.")
                    results_df['group'] = 0
        else:
            # Single group case
            results_df['group'] = 0
            
        # Create group return time series
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(results_df['timestamp']):
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            
        # Set timestamp as index
        results_df = results_df.set_index('timestamp')
        
        # Calculate average return by group and timestamp
        pivot_results = results_df.pivot_table(
            index=results_df.index,
            columns='group',
            values='return',
            aggfunc='mean'
        )
        
        # Add long-short spread
        if n_groups > 1:
            # High minus low (top minus bottom group)
            pivot_results['long_short'] = pivot_results[n_groups-1] - pivot_results[0]
            
        return pivot_results
    else:
        return pd.DataFrame()




# 看 this one !!! 
def walk_forward_test(
    df: pd.DataFrame,
    factor_func: Callable,
    train_window: int = 252,
    test_window: int = 63,
    periods: List[int] = [1, 5, 10],
    price_col: str = 'close',
    **factor_kwargs
) -> Dict[str, Any]:
    
    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: DataFrame index is not DatetimeIndex. Attempting to convert...")
        try:
            df = df.set_index(pd.DatetimeIndex(df.index))
        except:
            raise ValueError("Cannot convert index to DatetimeIndex. Please ensure DataFrame has datetime index.")
    
    # Sort by date
    df = df.sort_index()
    
    # Calculate total length and number of folds
    total_days = (df.index.max() - df.index.min()).days
    fold_days = train_window + test_window
    n_folds = max(1, total_days // fold_days)
    
    print(f"Performing walk-forward test with {n_folds} folds")
    print(f"Total data span: {df.index.min()} to {df.index.max()} ({total_days} days)")
    print(f"Training window: {train_window} days, Testing window: {test_window} days")
    
    # Initialize results
    results = {
        'ic_by_fold': [],
        'returns_by_fold': [],
        'factor_values': []
    }
    
    # Loop through folds
    for fold in range(n_folds):
        print(f"\nProcessing fold {fold+1}/{n_folds}")
        
        # Calculate fold start and end dates
        start_date = df.index.min() + pd.Timedelta(days=fold*fold_days)
        train_end_date = start_date + pd.Timedelta(days=train_window)
        test_end_date = train_end_date + pd.Timedelta(days=test_window)
        
        print(f"Train period: {start_date} to {train_end_date}")
        print(f"Test period: {train_end_date} to {test_end_date}")
        
        # Get train and test data
        train_data = df[(df.index >= start_date) & (df.index < train_end_date)]
        test_data = df[(df.index >= train_end_date) & (df.index < test_end_date)]
        
        if len(train_data) < 10 or len(test_data) < 5:
            print(f"Skipping fold {fold+1} due to insufficient data")
            continue
            
        # Generate factor on training data
        print("Generating factor on training data...")
        try:
            factor = factor_func(train_data, **factor_kwargs)
        except Exception as e:
            print(f"Error generating factor: {e}")
            continue
            
        # Test factor on test data
        print("Testing factor on test data...")
        try:
            # Apply factor to test data
            test_factor = factor_func(test_data, **factor_kwargs)
            
            # Run factor tests
            test_results = test_factor(test_data, test_factor, periods=periods, price_col=price_col)
            
            # Store results
            results['ic_by_fold'].append({
                'fold': fold,
                'train_start': start_date,
                'train_end': train_end_date,
                'test_start': train_end_date,
                'test_end': test_end_date,
                'results': test_results
            })
            
            # Run backtest
            backtest_results = backtest_factor(
                test_data, 
                test_factor,
                holding_period=max(periods),
                price_col=price_col
            )
            
            if not backtest_results.empty:
                results['returns_by_fold'].append({
                    'fold': fold,
                    'returns': backtest_results
                })
                
            # Store factor values
            results['factor_values'].append({
                'fold': fold,
                'factor': test_factor
            })
            
        except Exception as e:
            print(f"Error testing factor: {e}")
            continue
    
    # Calculate aggregate results
    print("\nCalculating aggregate results...")
    
    # Aggregate IC statistics
    ic_stats = {}
    for period in periods:
        period_ics = [fold['results'][period]['IC'] for fold in results['ic_by_fold'] 
                     if period in fold['results'] and 'IC' in fold['results'][period]]
        
        if period_ics:
            ic_stats[period] = {
                'mean_ic': np.mean(period_ics),
                'median_ic': np.median(period_ics),
                'ic_t_stat': np.mean(period_ics) / (np.std(period_ics) / np.sqrt(len(period_ics))) if len(period_ics) > 1 else 0,
                'positive_ic_rate': sum(1 for ic in period_ics if ic > 0) / len(period_ics)
            }
    
    results['ic_stats'] = ic_stats
    
    # Aggregate return statistics
    if results['returns_by_fold']:
        # Combine return series
        all_returns = {}
        for fold_result in results['returns_by_fold']:
            for col in fold_result['returns'].columns:
                if col not in all_returns:
                    all_returns[col] = []
                all_returns[col].extend(fold_result['returns'][col].tolist())
        
        # Calculate performance metrics
        return_stats = {}
        for group, returns in all_returns.items():
            returns_array = np.array(returns)
            return_stats[group] = {
                'mean_return': np.mean(returns_array),
                'sharpe': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
                'win_rate': np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
            }
            
        results['return_stats'] = return_stats
        
    print("Walk-forward testing complete!")
    return results
