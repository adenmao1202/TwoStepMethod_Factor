"""
Lightweight Factor Testing Framework - Single Responsibility Design
"""

"""
This framework supports automated combination and testing of investment factors,
evaluating their predictive power for future returns.
"""

# Import core components
from .data import load_data, calculate_vwap, calculate_forward_returns
from .factors import (
    # Basic technical factors
    price_to_vwap_factor, momentum_factor, volume_factor, volatility_factor,
    rsi_factor, bollinger_band_factor, macd_factor,
    
    # Cryptocurrency specific factors
    buy_sell_pressure_factor, trade_activity_factor, price_impact_factor,
    liquidity_imbalance_factor, trading_efficiency_factor,
    volume_divergence_factor, vpin_factor, hl_volume_position_factor,
    trade_size_change_factor, volume_surge_factor,
    
    # Combined factors
    market_balance_factor, smart_momentum_factor, trading_quality_factor,
    liquidity_risk_factor
)
from .utils import (normalize_factor, neutralize_factor, winsorize_factor, 
                   rank_transform_factor, standardize_factor, sigmoid_transform)
from .testing import test_factor, test_factor_combination, backtest_factor
from .analysis import analyze_factor
from .weights import (create_weight_schemes, combine_factors, 
                     dynamic_weight_adjustment)
from .combinations import generate_factor_combinations, generate_interaction_factors
from .backtesting import FactorBacktester

# Export the main API
__all__ = [
    # Data functions
    'load_data', 'calculate_vwap', 'calculate_forward_returns',
    
    # Factors
    'price_to_vwap_factor', 'momentum_factor', 'volume_factor', 'volatility_factor',
    'rsi_factor', 'bollinger_band_factor', 'macd_factor',
    
    # Cryptocurrency specific factors
    'buy_sell_pressure_factor', 'trade_activity_factor', 'price_impact_factor',
    'liquidity_imbalance_factor', 'trading_efficiency_factor',
    'volume_divergence_factor', 'vpin_factor', 'hl_volume_position_factor',
    'trade_size_change_factor', 'volume_surge_factor',
    
    # Combined factors
    'market_balance_factor', 'smart_momentum_factor', 'trading_quality_factor',
    'liquidity_risk_factor',
    
    # Utility functions
    'normalize_factor', 'neutralize_factor', 'winsorize_factor',
    'rank_transform_factor', 'standardize_factor', 'sigmoid_transform',
    
    # Testing functions
    'test_factor', 'test_factor_combination', 'backtest_factor',
    'analyze_factor',
    
    # Combinations
    'create_weight_schemes', 'combine_factors', 'dynamic_weight_adjustment',
    'generate_factor_combinations', 'generate_interaction_factors',
    
    # Classes
    'FactorBacktester'
]
