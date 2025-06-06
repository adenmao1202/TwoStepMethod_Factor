"""
權重優化算法 - 實現不同的權重優化方法
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Callable, Any
from scipy.optimize import minimize
import time
from .weights import combine_factors
from .analysis import analyze_factor

def optimize_weights(
    df: pd.DataFrame,
    factors: List[Tuple[str, Union[pd.Series, pd.DataFrame]]],
    objective_func: Callable,
    period: int = 24,
    price_col: str = 'close',
    constraints: List[Dict] = None,
    bounds: List[Tuple[float, float]] = None,
    method: str = 'SLSQP',
    max_iter: int = 100
) -> Tuple[np.ndarray, float]:
    """
    優化因子權重以最大化目標函數
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始數據
    factors : List[Tuple[str, Union[pd.Series, pd.DataFrame]]]
        因子列表，每個元素為(因子名稱, 因子數據)元組
    objective_func : Callable
        目標函數，接受因子分析結果，返回要最大化的值
    period : int, default 24
        評估週期（小時）
    price_col : str, default 'close'
        價格列名
    constraints : List[Dict], optional
        優化約束條件列表
    bounds : List[Tuple[float, float]], optional
        權重邊界列表
    method : str, default 'SLSQP'
        優化方法
    max_iter : int, default 100
        最大迭代次數
        
    Returns:
    --------
    Tuple[np.ndarray, float]
        優化後的權重數組和目標函數值
    """
    n_factors = len(factors)
    
    # 默認約束：權重總和為1
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    # 默認邊界：每個權重在[0, 1]之間
    if bounds is None:
        bounds = [(0.0, 1.0) for _ in range(n_factors)]
    
    # 初始猜測：等權重
    initial_weights = np.ones(n_factors) / n_factors
    
    # 定義目標函數（取負值是因為scipy.optimize.minimize默認最小化）
    def objective(weights):
        # 組合因子
        combined = combine_factors(factors, weights)
        
        # 分析組合因子
        results = analyze_factor(df, combined, periods=[period], price_col=price_col)
        
        # 計算目標值
        if period in results:
            objective_value = objective_func(results[period])
            return -objective_value  # 取負值以進行最大化
        else:
            return 0.0
    
    # 優化
    start_time = time.time()
    result = minimize(
        objective,
        initial_weights,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter}
    )
    end_time = time.time()
    
    print(f"優化完成，耗時: {end_time - start_time:.4f} 秒")
    print(f"優化狀態: {result.success}, 迭代次數: {result.nit}")
    
    # 返回優化後的權重和目標函數值（取負號轉回正值）
    return result.x, -result.fun

def maximize_ic(results):
    """最大化信息係數"""
    return results.get('IC', 0.0)

def maximize_spread(results):
    """最大化收益差"""
    return results.get('spread', 0.0)

def maximize_t_stat(results):
    """最大化t統計量"""
    return results.get('t_stat', 0.0)

def maximize_combined_metric(results, ic_weight=0.5, spread_weight=0.3, t_stat_weight=0.2):
    """最大化綜合指標"""
    ic = results.get('IC', 0.0)
    spread = results.get('spread', 0.0)
    t_stat = results.get('t_stat', 0.0)
    
    # 標準化指標
    ic_norm = ic if abs(ic) <= 1.0 else np.sign(ic)
    spread_norm = min(abs(spread), 0.1) * np.sign(spread) / 0.1
    t_stat_norm = min(abs(t_stat), 3.0) * np.sign(t_stat) / 3.0
    
    # 計算綜合指標
    combined = (ic_weight * ic_norm + 
                spread_weight * spread_norm + 
                t_stat_weight * t_stat_norm)
    
    return combined

def grid_search_weights(
    df: pd.DataFrame,
    factors: List[Tuple[str, Union[pd.Series, pd.DataFrame]]],
    objective_func: Callable,
    period: int = 5,
    price_col: str = 'close',
    grid_size: int = 5
) -> Tuple[np.ndarray, float]:
    """
    使用網格搜索優化因子權重
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始數據
    factors : List[Tuple[str, Union[pd.Series, pd.DataFrame]]]
        因子列表，每個元素為(因子名稱, 因子數據)元組
    objective_func : Callable
        目標函數，接受因子分析結果，返回要最大化的值
    period : int, default 5
        評估週期
    price_col : str, default 'close'
        價格列名
    grid_size : int, default 5
        每個維度的網格數量
        
    Returns:
    --------
    Tuple[np.ndarray, float]
        最佳權重數組和目標函數值
    """
    n_factors = len(factors)
    
    if n_factors > 3:
        print(f"警告: 網格搜索對於 {n_factors} 個因子可能非常耗時，建議使用其他優化方法")
    
    # 創建網格點
    if n_factors == 2:
        # 二維情況，權重總和為1
        grid_points = []
        for i in range(grid_size + 1):
            w1 = i / grid_size
            w2 = 1 - w1
            grid_points.append([w1, w2])
    else:
        # 多維情況，使用隨機採樣
        np.random.seed(42)
        grid_points = []
        for _ in range(grid_size ** min(n_factors, 3)):  # 限制點數
            weights = np.random.rand(n_factors)
            weights = weights / weights.sum()  # 標準化使總和為1
            grid_points.append(weights)
    
    # 評估每個網格點
    best_objective = -np.inf
    best_weights = None
    
    start_time = time.time()
    for weights in grid_points:
        # 組合因子
        combined = combine_factors(factors, weights)
        
        # 分析組合因子
        results = analyze_factor(df, combined, periods=[period], price_col=price_col)
        
        # 計算目標值
        if period in results:
            objective_value = objective_func(results[period])
            
            # 更新最佳結果
            if objective_value > best_objective:
                best_objective = objective_value
                best_weights = weights
    
    end_time = time.time()
    print(f"網格搜索完成，耗時: {end_time - start_time:.4f} 秒")
    
    return np.array(best_weights), best_objective

def genetic_algorithm_weights(
    df: pd.DataFrame,
    factors: List[Tuple[str, Union[pd.Series, pd.DataFrame]]],
    objective_func: Callable,
    period: int = 5,
    price_col: str = 'close',
    population_size: int = 50,
    generations: int = 20,
    mutation_rate: float = 0.1
) -> Tuple[np.ndarray, float]:
    """
    使用遺傳算法優化因子權重
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始數據
    factors : List[Tuple[str, Union[pd.Series, pd.DataFrame]]]
        因子列表，每個元素為(因子名稱, 因子數據)元組
    objective_func : Callable
        目標函數，接受因子分析結果，返回要最大化的值
    period : int, default 5
        評估週期
    price_col : str, default 'close'
        價格列名
    population_size : int, default 50
        種群大小
    generations : int, default 20
        迭代代數
    mutation_rate : float, default 0.1
        變異率
        
    Returns:
    --------
    Tuple[np.ndarray, float]
        最佳權重數組和目標函數值
    """
    n_factors = len(factors)
    
    # 評估適應度函數
    def fitness(weights):
        # 標準化權重使總和為1
        weights = np.array(weights) / np.sum(weights)
        
        # 組合因子
        combined = combine_factors(factors, weights)
        
        # 分析組合因子
        results = analyze_factor(df, combined, periods=[period], price_col=price_col)
        
        # 計算目標值
        if period in results:
            return objective_func(results[period])
        else:
            return 0.0
    
    # 初始化種群
    np.random.seed(42)
    population = []
    for _ in range(population_size):
        # 生成隨機權重
        weights = np.random.rand(n_factors)
        weights = weights / weights.sum()  # 標準化使總和為1
        population.append(weights)
    
    # 遺傳算法迭代
    start_time = time.time()
    for generation in range(generations):
        # 計算每個個體的適應度
        fitness_values = [fitness(ind) for ind in population]
        
        # 選擇父代（輪盤賭選擇）
        fitness_sum = sum(max(0, f) for f in fitness_values)
        if fitness_sum <= 0:
            selection_probs = [1/len(fitness_values) for _ in fitness_values]
        else:
            selection_probs = [max(0, f)/fitness_sum for f in fitness_values]
        
        # 創建新一代
        new_population = []
        for _ in range(population_size):
            # 選擇兩個父代
            parent1 = population[np.random.choice(len(population), p=selection_probs)]
            parent2 = population[np.random.choice(len(population), p=selection_probs)]
            
            # 交叉
            crossover_point = np.random.randint(1, n_factors)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # 變異
            for i in range(n_factors):
                if np.random.random() < mutation_rate:
                    child[i] = np.random.random()
            
            # 標準化權重
            child = child / child.sum()
            
            new_population.append(child)
        
        # 更新種群
        population = new_population
        
        # 打印進度
        if (generation + 1) % 5 == 0:
            best_fitness = max(fitness_values)
            print(f"代數 {generation + 1}/{generations}, 最佳適應度: {best_fitness:.6f}")
    
    # 找出最佳個體
    fitness_values = [fitness(ind) for ind in population]
    best_idx = np.argmax(fitness_values)
    best_weights = population[best_idx]
    best_fitness = fitness_values[best_idx]
    
    end_time = time.time()
    print(f"遺傳算法完成，耗時: {end_time - start_time:.4f} 秒")
    
    return best_weights, best_fitness
