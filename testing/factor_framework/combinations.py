
import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Callable, Union, Tuple

def generate_factor_combinations(
    base_factors: Dict[str, Union[pd.Series, pd.DataFrame]], 
    max_size: int = 3
) -> Dict[str, List[Tuple[str, Union[pd.Series, pd.DataFrame]]]]:
    """
    生成因子組合
    
    Parameters:
    -----------
    base_factors : Dict[str, Union[pd.Series, pd.DataFrame]]
        基礎因子字典，鍵為因子名稱，值為因子數據
    max_size : int, default 3
        最大組合大小
        
    Returns:
    --------
    Dict[str, List[Tuple[str, Union[pd.Series, pd.DataFrame]]]]
        因子組合字典，鍵為組合名稱，值為(因子名稱, 因子數據)元組列表
    """
    combinations = {}
    
    # 添加單一因子
    for name, factor in base_factors.items():
        combinations[name] = [(name, factor)]
    
    # 生成多因子組合
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(base_factors.items(), size):
            # 組合名稱
            combo_name = '+'.join([item[0] for item in combo])
            # 組合內容
            combinations[combo_name] = list(combo)
    
    return combinations

def create_orthogonal_factors(
    factors: Dict[str, pd.Series], 
    method: str = 'gram_schmidt'
) -> Dict[str, pd.Series]:
    """
    
    正交化方法，可選 'gram_schmidt' 或 'pca'
        
    """
    # 將因子轉換為DataFrame
    factor_df = pd.DataFrame({name: factor for name, factor in factors.items()})
    factor_df = factor_df.dropna()
    
    if method == 'gram_schmidt':
        # Gram-Schmidt正交化
        orthogonal_factors = {}
        remaining_factors = list(factors.keys())
        
        # 第一個因子保持不變
        first_factor = remaining_factors[0]
        orthogonal_factors[first_factor] = factor_df[first_factor]
        
        # 對剩餘因子進行正交化
        for factor_name in remaining_factors[1:]:
            # 複製當前因子
            current = factor_df[factor_name].copy()
            
            # 減去與已正交化因子的投影
            for ortho_name, ortho_factor in orthogonal_factors.items():
                projection = (current * ortho_factor).sum() / (ortho_factor * ortho_factor).sum()
                current = current - projection * ortho_factor
            
            # 添加到正交化因子字典
            orthogonal_factors[f"ortho_{factor_name}"] = current
            
        return orthogonal_factors
    
    elif method == 'pca':
        from sklearn.decomposition import PCA
        
        # 標準化數據
        standardized_df = (factor_df - factor_df.mean()) / factor_df.std()
        
        # 應用PCA
        pca = PCA(n_components=len(factors))
        principal_components = pca.fit_transform(standardized_df)
        
        # 創建正交因子
        orthogonal_factors = {}
        for i in range(len(factors)):
            component = pd.Series(principal_components[:, i], index=factor_df.index)
            orthogonal_factors[f"pc_{i+1}"] = component
            
        return orthogonal_factors
    
    else:
        raise ValueError(f"不支持的正交化方法: {method}")

def generate_interaction_factors(
    factors: Dict[str, pd.Series], 
    operations: List[Callable] = None
) -> Dict[str, pd.Series]:
    """
    生成因子交互項
    
    Parameters:
    -----------
    factors : Dict[str, pd.Series]
        原始因子字典
    operations : List[Callable], optional
        交互操作列表，默認為乘法和加法
        
    Returns:
    --------
    Dict[str, pd.Series]
        交互因子字典
    """
    if operations is None:
        operations = [
            (lambda x, y: x * y, '*'),
            (lambda x, y: x + y, '+'),
            (lambda x, y: x - y, '-'),
            (lambda x, y: x / y, '/')
        ]
    
    interaction_factors = {}
    
    # 生成所有因子對的交互項
    for (name1, factor1), (name2, factor2) in itertools.combinations(factors.items(), 2):
        for op_func, op_symbol in operations:
            try:
                # 應用操作
                interaction = op_func(factor1, factor2)
                # 創建交互因子名稱
                interaction_name = f"{name1}{op_symbol}{name2}"
                # 添加到交互因子字典
                interaction_factors[interaction_name] = interaction
            except Exception as e:
                print(f"生成交互因子 {name1}{op_symbol}{name2} 時出錯: {e}")
                continue
    
    return interaction_factors
