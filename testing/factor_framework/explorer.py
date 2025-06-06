"""
Explorer module - Provides a complete factor exploration workflow
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Callable, Any
import time
from .data import load_data, calculate_forward_returns
from .factors import price_to_vwap_factor, momentum_factor, volume_factor, volatility_factor
from .analysis import analyze_factor, plot_quintile_returns
from .combinations import generate_factor_combinations
from .weights import create_weight_schemes, combine_factors
from .testing import test_factor, test_factor_combination
from .ranking import rank_factors, filter_top_factors, rank_combinations
from .optimization import optimize_weights, maximize_ic, maximize_combined_metric
from .utils import time_it

@time_it
def run_factor_exploration(
    data: pd.DataFrame,
    base_factors: Dict[str, Union[pd.Series, pd.DataFrame]],
    periods: List[int] = [1, 6, 24],
    max_combination_size: int = 3,
    target_period: int = 24,
    top_n: int = 5,
    price_col: str = 'close',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a complete factor exploration process
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data
    base_factors : Dict[str, Union[pd.Series, pd.DataFrame]]
        Base factor dictionary, keys are factor names, values are factor data
    periods : List[int], default [1, 6, 24]
        Time periods to test (hours)
    max_combination_size : int, default 3
        Maximum combination size
    target_period : int, default 24
        Target evaluation period (hours)
    top_n : int, default 5
        Select top N best results
    price_col : str, default 'close'
        Price column name
    verbose : bool, default True
        Whether to output detailed information
        
    Returns:
    --------
    Dict[str, Any]
        Exploration results
    """
    results = {}
    
    if verbose:
        print("Starting factor exploration process...")
    start_time = time.time()
    
    # 1. Test single factors
    if verbose:
        print("\n1. Testing single factors")
    single_factor_results = {}
    for name, factor in base_factors.items():
        if verbose:
            print(f"  Testing factor: {name}")
        single_factor_results[name] = test_factor(data, factor, periods, price_col)
    
    # 2. Rank single factors
    if verbose:
        print("\n2. Ranking single factors")
    ranked_factors = rank_factors(single_factor_results, 'IC', target_period)
    if verbose:
        print(ranked_factors)
    results['ranked_single_factors'] = ranked_factors
    
    # 3. Generate factor combinations
    if verbose:
        print("\n3. Generating factor combinations")
    factor_combinations = generate_factor_combinations(base_factors, max_combination_size)
    if verbose:
        print(f"  Generated {len(factor_combinations)} combinations")
    
    # 4. Create weight schemes
    if verbose:
        print("\n4. Creating weight schemes")
    weight_schemes = {}
    for combo_name, combo_factors in factor_combinations.items():
        n_factors = len(combo_factors)
        if n_factors > 1:  # Single factors don't need weight schemes
            weight_schemes[combo_name] = create_weight_schemes(n_factors)
    
    # 5. Test factor combinations
    if verbose:
        print("\n5. Testing factor combinations")
    combination_results = {}
    for combo_name, combo_factors in factor_combinations.items():
        if verbose:
            print(f"  Testing combination: {combo_name}")
        
        # Single factors use previous results
        if len(combo_factors) == 1:
            factor_name = combo_factors[0][0]
            combination_results[combo_name] = {'equal': single_factor_results[factor_name]}
            continue
        
        # Multi-factor tests with different weight schemes
        combo_schemes = weight_schemes[combo_name]
        combo_results = {}
        
        for scheme_name, weights in combo_schemes.items():
            if verbose:
                print(f"    Weight scheme: {scheme_name}")
            combo_results[scheme_name] = test_factor_combination(
                data, combo_factors, weights, periods, price_col
            )
        
        combination_results[combo_name] = combo_results
    
    # 6. Rank combination results
    if verbose:
        print("\n6. Ranking combination results")
    ranked_combinations = rank_combinations(combination_results, 'IC', target_period)
    if verbose:
        print(ranked_combinations.head(top_n))
    results['ranked_combinations'] = ranked_combinations
    
    # 7. Select best combination for weight optimization
    if verbose:
        print("\n7. Selecting best combination for weight optimization")
    if not ranked_combinations.empty:
        best_combo = ranked_combinations.iloc[0]['combination']
        best_combo_factors = factor_combinations[best_combo]
        
        if verbose:
            print(f"  Optimizing combination: {best_combo}")
        
        # Only multi-factor combinations need optimization
        if len(best_combo_factors) > 1:
            if verbose:
                print("  Using combined metric for optimization")
            optimal_weights, optimal_value = optimize_weights(
                data, best_combo_factors, maximize_combined_metric, target_period, price_col
            )
            
            if verbose:
                print(f"  Optimal weights: {optimal_weights}")
                print(f"  Optimized metric value: {optimal_value:.6f}")
            
            # Test with optimized weights
            optimized_results = test_factor_combination(
                data, best_combo_factors, optimal_weights, periods, price_col
            )
            
            # Add to results
            combination_results[best_combo]['optimized'] = optimized_results
            results['optimal_weights'] = {
                'combination': best_combo,
                'weights': optimal_weights,
                'value': optimal_value
            }
    
    # 8. Generate results summary
    if verbose:
        print("\n8. Generating results summary")
    summary = {
        'best_single_factor': ranked_factors.iloc[0]['factor'] if not ranked_factors.empty else None,
        'best_combination': ranked_combinations.iloc[0]['combination'] if not ranked_combinations.empty else None,
        'best_weight_scheme': ranked_combinations.iloc[0]['weights'] if not ranked_combinations.empty else None,
        'exploration_time': time.time() - start_time
    }
    
    results['summary'] = summary
    if verbose:
        print(f"\nExploration completed, time taken: {summary['exploration_time']:.2f} seconds")
    
    return results

def visualize_exploration_results(results: Dict[str, Any], period: int = 24) -> None:
    """
    Visualize factor exploration results
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Factor exploration results
    period : int, default 24
        Target evaluation period
    """
    # 1. Plot single factor IC comparison
    if 'ranked_single_factors' in results and not results['ranked_single_factors'].empty:
        plt.figure(figsize=(12, 6))
        
        df = results['ranked_single_factors']
        bars = plt.bar(df['factor'], df['IC'])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        plt.title(f'Single Factor Information Coefficient (IC) Comparison - Period {period}')
        plt.xlabel('Factor')
        plt.ylabel('Information Coefficient (IC)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 2. Plot best combination comparison
    if 'ranked_combinations' in results and not results['ranked_combinations'].empty:
        plt.figure(figsize=(14, 7))
        
        df = results['ranked_combinations'].head(10)  # Show top 10
        
        # Create combination+weight labels
        labels = [f"{row['combination']}\n({row['weights']})" for _, row in df.iterrows()]
        
        bars = plt.bar(range(len(labels)), df['IC'])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        plt.title(f'Best Factor Combinations Information Coefficient (IC) Comparison - Period {period}')
        plt.xlabel('Factor Combination')
        plt.ylabel('Information Coefficient (IC)')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 3. If optimized weights exist, plot weight distribution
    if 'optimal_weights' in results:
        plt.figure(figsize=(10, 6))
        
        combo = results['optimal_weights']['combination']
        weights = results['optimal_weights']['weights']
        
        # Extract factor names
        factor_names = combo.split('+')
        
        # Plot weight distribution
        bars = plt.bar(factor_names, weights)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.title(f'Optimized Weight Distribution - {combo}')
        plt.xlabel('Factor')
        plt.ylabel('Weight')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def create_factor_report(
    data: pd.DataFrame, 
    factor: pd.Series, 
    factor_name: str, 
    periods: List[int] = [1, 6, 24], 
    price_col: str = 'close',
    output_file: str = None,
    verbose: bool = True
) -> str:
    """
    Create a comprehensive factor analysis report
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data
    factor : pd.Series
        Factor values
    factor_name : str
        Name of the factor
    periods : List[int], default [1, 6, 24]
        Time periods to analyze
    price_col : str, default 'close'
        Price column name
    output_file : str, optional
        Output file path for the report
    verbose : bool, default True
        Whether to output detailed information
        
    Returns:
    --------
    str
        Report content
    """
    # Ensure factor is a Series
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    
    # Rename factor for consistency
    factor = factor.rename('factor')
    
    # Create report content
    report = f"===== {factor_name} Factor Report =====\n\n"
    
    # 1. Basic factor statistics
    report += "1. Basic Factor Statistics:\n"
    report += str(factor.describe()) + "\n\n"
    
    # 2. Factor analysis
    report += "2. Factor Analysis:\n"
    results = analyze_factor(data, factor, periods=periods, price_col=price_col)
    
    for period in periods:
        period_results = results[period]
        report += f"\nPeriod {period} Analysis Results:\n"
        
        # Information coefficient
        ic = period_results.get('ic', period_results.get('IC', float('nan')))
        report += f"  Information Coefficient (IC): {ic:.6f}\n"
        
        # Return spread
        spread = period_results.get('spread', float('nan'))
        report += f"  Return Spread (Q5-Q1): {spread:.6f}\n"
        
        # t-statistic
        t_stat = period_results.get('t_stat', float('nan'))
        report += f"  t-statistic: {t_stat:.6f}\n\n"
        
        # Quintile returns
        quintile_returns = period_results.get('quintile_returns', None)
        if quintile_returns is not None:
            report += "  Quintile Returns:\n"
            for quintile, ret in quintile_returns.items():
                report += f"    {quintile}: {ret:.6f}\n"
            report += "\n"
    
    # 3. Quintile returns charts
    report += "3. Quintile Returns Charts:\n"
    
    # Add charts if output file is specified
    if output_file:
        # Create charts and save them
        pass
    
    report += "\n===== Report End =====\n"
    
    # Print or save report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        if verbose:
            print(f"Report saved to {output_file}")
    else:
        if verbose:
            print(report)
    
    return report

def batch_test_factors(
    data: pd.DataFrame,
    factor_functions: Dict[str, Callable],
    periods: List[int] = [1, 6, 24],
    price_col: str = 'close',
    **kwargs
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Batch test multiple factor functions
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data
    factor_functions : Dict[str, Callable]
        Factor function dictionary, keys are function names, values are function objects
    periods : List[int], default [1, 6, 24]
        Time periods to test (hours)
    price_col : str, default 'close'
        Price column name
    **kwargs
        Other parameters to pass to factor functions
        
    Returns:
    --------
    Dict[str, Dict[int, Dict[str, Any]]]
        Test results dictionary
    """
    results = {}
    
    print("Starting batch testing of factors...")
    
    for name, func in factor_functions.items():
        print(f"Calculating factor: {name}")
        try:
            # Calculate factor
            factor = func(data, **kwargs)
            
            # Test factor
            print(f"Testing factor: {name}")
            factor_results = test_factor(data, factor, periods, price_col)
            
            # Store results
            results[name] = factor_results
            
        except Exception as e:
            print(f"Error processing factor {name}: {e}")
            continue
    
    print("Batch testing completed")
    return results
