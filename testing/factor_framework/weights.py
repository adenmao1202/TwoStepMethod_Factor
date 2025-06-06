"""
權重生成與管理 - 生成不同的權重方案並組合因子
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Callable
from .utils import normalize_factor

def create_weight_schemes(n_factors: int) -> Dict[str, np.ndarray]:
    """
    創建多種權重方案
    
    Parameters:
    -----------
    n_factors : int
        因子數量
        
    Returns:
    --------
    Dict[str, np.ndarray]
        權重方案字典，鍵為方案名稱，值為權重數組
    """
    schemes = {}
    
    # 等權重
    schemes['equal'] = np.ones(n_factors) / n_factors
    
    # 線性遞減權重
    linear_weights = np.arange(n_factors, 0, -1)
    schemes['linear'] = linear_weights / linear_weights.sum()
    
    # 指數遞減權重
    exp_weights = np.exp(-np.arange(n_factors))
    schemes['exponential'] = exp_weights / exp_weights.sum()
    
    # 二次方遞減權重
    quad_weights = np.arange(n_factors, 0, -1) ** 2
    schemes['quadratic'] = quad_weights / quad_weights.sum()
    
    return schemes

def combine_factors(
    factors: List[Tuple[str, Union[pd.Series, pd.DataFrame]]], 
    weights: np.ndarray = None,
    normalize: bool = True
) -> pd.Series:
    """
    按權重組合多個因子
    
    Parameters:
    -----------
    factors : List[Tuple[str, Union[pd.Series, pd.DataFrame]]]
        因子列表，每個元素為(因子名稱, 因子數據)元組
    weights : np.ndarray, optional
        權重數組，默認為等權重
    normalize : bool, default True
        是否對每個因子進行標準化
        
    Returns:
    --------
    pd.Series
        組合後的因子
    """
    # 默認使用等權重
    if weights is None:
        weights = np.ones(len(factors)) / len(factors)
    
    # 確保權重數組長度與因子數量相同
    assert len(weights) == len(factors), "權重數量必須與因子數量相同"
    
    # 提取因子數據
    factor_data = []
    for _, factor in factors:
        # 如果是DataFrame，取第一列
        if isinstance(factor, pd.DataFrame):
            factor = factor.iloc[:, 0]
        
        # 標準化因子
        if normalize:
            factor = normalize_factor(factor)
            
        factor_data.append(factor)
    
    # 將因子數據轉換為DataFrame
    factor_df = pd.concat(factor_data, axis=1)
    factor_df.columns = [name for name, _ in factors]
    
    # 應用權重
    combined_factor = factor_df.dot(weights)
    combined_factor.name = 'combined_factor'
    
    return combined_factor

def dynamic_weight_adjustment(
    factors: List[Tuple[str, Union[pd.Series, pd.DataFrame]]], 
    performance_func: Callable,
    lookback_window: int = 60,
    min_weight: float = 0.05
) -> pd.DataFrame:
    """
    基於歷史表現動態調整因子權重
    
    Parameters:
    -----------
    factors : List[Tuple[str, Union[pd.Series, pd.DataFrame]]]
        因子列表，每個元素為(因子名稱, 因子數據)元組
    performance_func : Callable
        評估因子表現的函數，接受因子數據和時間窗口，返回表現分數
    lookback_window : int, default 60
        回顧窗口大小
    min_weight : float, default 0.05
        最小權重限制
        
    Returns:
    --------
    pd.DataFrame
        動態權重DataFrame，索引為時間，列為因子名稱
    """
    # 提取因子名稱和數據
    factor_names = [name for name, _ in factors]
    factor_data = [data for _, data in factors]
    
    # 確保所有因子數據是Series
    for i in range(len(factor_data)):
        if isinstance(factor_data[i], pd.DataFrame):
            factor_data[i] = factor_data[i].iloc[:, 0]
    
    # 創建因子DataFrame
    factor_df = pd.concat(factor_data, axis=1)
    factor_df.columns = factor_names
    
    # 初始化權重DataFrame
    weights_df = pd.DataFrame(index=factor_df.index, columns=factor_names)
    
    # 對每個時間點計算動態權重
    for i in range(lookback_window, len(factor_df)):
        # 獲取回顧窗口數據
        window_data = factor_df.iloc[i-lookback_window:i]
        
        # 計算每個因子的表現分數
        performance_scores = []
        for name in factor_names:
            score = performance_func(window_data[name], window_data)
            performance_scores.append(max(score, 0))  # 確保分數非負
        
        # 如果所有分數都為零，使用等權重
        if sum(performance_scores) == 0:
            weights = np.ones(len(factor_names)) / len(factor_names)
        else:
            # 基於表現分數計算權重
            weights = np.array(performance_scores) / sum(performance_scores)
            
            # 應用最小權重限制
            if min_weight > 0:
                # 調整低於最小權重的因子
                low_weight_indices = weights < min_weight
                if np.any(low_weight_indices):
                    # 需要調整的總權重
                    total_adjustment = min_weight * sum(low_weight_indices) - sum(weights[low_weight_indices])
                    # 從其他因子中扣除
                    high_weight_indices = ~low_weight_indices
                    if np.any(high_weight_indices):
                        weights[high_weight_indices] -= total_adjustment * weights[high_weight_indices] / sum(weights[high_weight_indices])
                        weights[low_weight_indices] = min_weight
        
        # 存儲權重
        weights_df.iloc[i] = weights
    
    return weights_df.dropna()
