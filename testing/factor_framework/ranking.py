"""
結果排序與過濾 - 對測試結果進行排序和過濾
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any

def rank_factors(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    metric: str = 'IC',
    period: int = 5,
    ascending: bool = False
) -> pd.DataFrame:
   
    \
    data = []
    for factor_name, factor_results in results.items():
        if period in factor_results and metric in factor_results[period]:
            data.append({
                'factor': factor_name,
                metric: factor_results[period][metric]
            })
    
    # 創建DataFrame並排序
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by=metric, ascending=ascending)
    
    return df



def filter_top_factors(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    metric: str = 'IC',
    period: int = 5,
    top_n: int = 5,
    min_threshold: float = None
) -> Dict[str, Dict[int, Dict[str, Any]]]:
   
    
    # 對因子進行排序
    ranked_factors = rank_factors(results, metric, period)
    
    # 應用最小閾值篩選
    if min_threshold is not None:
        ranked_factors = ranked_factors[ranked_factors[metric] > min_threshold]
    
    # 選擇前N個因子
    top_factors = ranked_factors.head(top_n)['factor'].tolist()
    
    # 提取這些因子的結果
    filtered_results = {factor: results[factor] for factor in top_factors if factor in results}
    
    return filtered_results


def extract_metrics(
    results: Dict[int, Dict[str, Any]], 
    combo_name: str,
    weight_scheme: str = 'equal'
) -> pd.DataFrame:
    
    metrics = []
    
    for period, period_results in results.items():
        if 'IC' in period_results:
            metrics.append({
                'combination': combo_name,
                'weights': weight_scheme,
                'period': period,
                'IC': period_results['IC'],
                'spread': period_results.get('spread', np.nan),
                't_stat': period_results.get('t_stat', np.nan)
            })
    
    return pd.DataFrame(metrics)


def rank_combinations(
    all_results: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    metric: str = 'IC',
    period: int = 5
) -> pd.DataFrame:
    
    all_metrics = []
    
    # 提取所有組合的指標
    for combo_name, weight_results in all_results.items():
        for weight_scheme, results in weight_results.items():
            metrics_df = extract_metrics(results, combo_name, weight_scheme)
            all_metrics.append(metrics_df)
    
    if not all_metrics:
        return pd.DataFrame()
    
    # 合併所有指標
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # 篩選指定週期的結果
    period_metrics = combined_metrics[combined_metrics['period'] == period]
    
    # 按指定指標排序
    ranked_combinations = period_metrics.sort_values(by=metric, ascending=False)
    
    return ranked_combinations

def calculate_factor_contribution(
    df: pd.DataFrame,
    factors: Dict[str, Union[pd.Series, pd.DataFrame]],
    combined_factor: pd.Series
) -> pd.DataFrame:
    """
    計算每個因子對組合因子的貢獻度
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始數據
    factors : Dict[str, Union[pd.Series, pd.DataFrame]]
        因子字典，鍵為因子名稱，值為因子數據
    combined_factor : pd.Series
        組合因子
        
    Returns:
    --------
    pd.DataFrame
        因子貢獻度DataFrame
    """
    # 計算每個因子與組合因子的相關性
    correlations = {}
    for name, factor in factors.items():
        if isinstance(factor, pd.DataFrame):
            factor = factor.iloc[:, 0]
        
        # 計算相關係數
        correlation = factor.corr(combined_factor)
        correlations[name] = correlation
    
    # 計算貢獻度（相關係數的絕對值）
    contributions = {name: abs(corr) for name, corr in correlations.items()}
    
    # 標準化貢獻度，使總和為1
    total = sum(contributions.values())
    if total > 0:
        normalized_contributions = {name: contrib / total for name, contrib in contributions.items()}
    else:
        normalized_contributions = {name: 1.0 / len(contributions) for name in contributions}
    
    # 創建DataFrame
    result = pd.DataFrame({
        'factor': list(correlations.keys()),
        'correlation': list(correlations.values()),
        'contribution': list(normalized_contributions.values())
    })
    
    # 按貢獻度排序
    result = result.sort_values(by='contribution', ascending=False)
    
    return result
