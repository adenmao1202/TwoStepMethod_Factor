# Done 


import pandas as pd
import numpy as np
from .data import calculate_vwap

# 通用時間序列函數
def ts_mean(series, window):
    return series.rolling(window=window).mean()

def ts_std(series, window):
    return series.rolling(window=window).std()

def ts_sum(series, window):
    return series.rolling(window=window).sum()

def ts_max(series, window):
    return series.rolling(window=window).max()

def ts_min(series, window):
    return series.rolling(window=window).min()

def ts_median(series, window):
    return series.rolling(window=window).median()

def ts_quantile(series, window, q):
    return series.rolling(window=window).quantile(q)

def ts_mode(series, window):
    return series.rolling(window=window).mode()

def ts_range(series, window):
    return series.rolling(window=window).max() - series.rolling(window=window).min()

def ts_skewness(series, window):
    return series.rolling(window=window).skew()

def ts_kurtosis(series, window):
    return series.rolling(window=window).kurt()

def ts_mean_abs_deviation(series, window):
    return series.rolling(window=window).apply(lambda x: np.abs(x - x.mean()))

def ts_median_abs_deviation(series, window):
    return series.rolling(window=window).apply(lambda x: np.abs(x - x.median()))

def ts_log_return(series, window):
    return np.log(series).diff(window)

def ts_log_return_2(series, window):
    return np.log(series).diff(2*window)


# ----- 基礎技術因子 ----- #

def price_to_vwap_factor(df, vwap_mean_window=3, lag=1, price_col='close'):
    """價格與VWAP比較因子"""
    df = df.copy()
    
    # 計算VWAP
    df = calculate_vwap(df, price_col=price_col)
    
    # 移動價格和VWAP以確保只使用過去數據
    price_lag = df[price_col].shift(lag)
    vwap_lag = df['vwap'].shift(lag)
    
    # 計算VWAP移動平均
    vwap_mean = ts_mean(vwap_lag, vwap_mean_window)
    
    # 計算因子
    alpha = -(price_lag - vwap_mean)
    
    return alpha

def momentum_factor(df, window=24, lag=1, price_col='close'):
    """動量因子 - 過去窗口價格變化"""
    return df[price_col].pct_change(window).shift(lag)

def volume_factor(df, window=24, lag=1, volume_col='volume'):
    """成交量變化因子"""
    return df[volume_col].pct_change(window).shift(lag)

def volatility_factor(df, window=24*5, lag=1, price_col='close'):
    """波動率因子 - 負號表示低波動性更好"""
    return -df[price_col].pct_change().rolling(window).std().shift(lag)

def rsi_factor(df, window=24*3, lag=1, price_col='close'):
    """相對強弱指標(RSI)因子"""
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.shift(lag)

def bollinger_band_factor(df, window=24*5, std_dev=2, lag=1, price_col='close'):
    """布林帶因子 - 返回價格在帶中的位置和帶寬"""
    rolling_mean = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    # 計算價格相對於布林帶的位置 (0-1之間)
    bb_position = (df[price_col] - lower_band) / (upper_band - lower_band)
    
    # 布林帶寬度
    bb_width = (upper_band - lower_band) / rolling_mean
    
    return pd.DataFrame({
        'bb_position': bb_position.shift(lag),
        'bb_width': bb_width.shift(lag)
    }, index=df.index)

def macd_factor(df, fast=12*6, slow=26*6, signal=9*6, lag=1, price_col='close'):
    """MACD因子 - 返回MACD線、信號線和柱狀圖"""
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd_line': macd_line.shift(lag),
        'signal_line': signal_line.shift(lag),
        'macd_histogram': macd_histogram.shift(lag)
    }, index=df.index)

# ----- 加密貨幣特定因子 ----- #

def buy_sell_pressure_factor(df, window=24, lag=1):
    """
    计算买卖压力因子，衡量市场买卖力量的不平衡程度
    买方力量占比 = taker_buy_base_asset_volume / volume
    """
    buy_ratio = df['taker_buy_base_asset_volume'] / df['volume']
    
    # 计算相对于过去窗口的买入压力
    mean_buy_ratio = buy_ratio.rolling(window=window).mean()
    buy_pressure = buy_ratio / mean_buy_ratio - 1
    
    return buy_pressure.shift(lag)

def trade_activity_factor(df, window=24, lag=1):
    """
    基于交易次数的交易活跃度因子
    衡量当前交易活跃度相对于历史水平
    """
    # 计算每单位成交量的交易次数
    trades_per_volume = df['number_of_trades'] / df['volume']
    
    # 计算相对于过去窗口的活跃度变化
    mean_activity = trades_per_volume.rolling(window=window).mean()
    activity_change = trades_per_volume / mean_activity - 1
    
    return activity_change.shift(lag)

def price_impact_factor(df, window=24, lag=1):
    """
    衡量单位成交量对价格的影响力
    较小的成交量导致较大的价格变动意味着市场深度不足
    """
    # 计算价格变化幅度（高低价差）
    price_range = (df['high'] - df['low']) / df['low']
    
    # 计算单位成交量的价格影响
    impact = price_range / df['volume']
    
    # 相对于过去窗口的价格影响变化
    mean_impact = impact.rolling(window=window).mean()
    impact_ratio = impact / mean_impact
    
    return impact_ratio.shift(lag)

def liquidity_imbalance_factor(df, window=24, lag=1):
    """
    衡量买卖双方提供的流动性不平衡程度
    基于买卖报价成交量的比例
    """
    # 买方流动性
    buy_liquidity = df['taker_buy_quote_asset_volume']
    
    # 卖方流动性 (总报价资产成交量 - 买方报价资产成交量)
    sell_liquidity = df['quote_asset_volume'] - df['taker_buy_quote_asset_volume']
    
    # 计算不平衡比率
    imbalance = (buy_liquidity - sell_liquidity) / (buy_liquidity + sell_liquidity)
    
    # 相对于过去窗口的不平衡变化
    imbalance_ma = imbalance.rolling(window=window).mean()
    imbalance_change = imbalance - imbalance_ma
    
    return imbalance_change.shift(lag)

def trading_efficiency_factor(df, window=24, lag=1):
    """
    计算价格变动与成交量的效率比
    高效率表示较少的成交量带来较大的价格变动
    """
    # 绝对价格变动
    price_change = abs(df['close'] - df['open'])
    
    # 成交量标准化（相对于窗口内的平均成交量）
    vol_ma = df['volume'].rolling(window=window).mean()
    norm_volume = df['volume'] / vol_ma
    
    # 效率比：价格变动/标准化成交量
    efficiency = price_change / norm_volume
    
    # 相对效率（与窗口内的平均效率相比）
    efficiency_ma = efficiency.rolling(window=window).mean()
    relative_efficiency = efficiency / efficiency_ma - 1
    
    return relative_efficiency.shift(lag)

def volume_divergence_factor(df, window=24, lag=1):
    """
    计算成交量与价格趋势的差异
    当价格上涨但成交量下降时，可能表示上涨势头减弱
    """
    # 价格变动方向
    price_direction = np.sign(df['close'].pct_change())
    
    # 成交量变动
    volume_change = df['volume'].pct_change()
    
    # 成交量与价格方向的一致性（正值表示一致，负值表示不一致）
    consistency = price_direction * volume_change
    
    # 相对于窗口内的平均一致性
    consistency_ma = consistency.rolling(window=window).mean()
    divergence = consistency - consistency_ma
    
    return divergence.shift(lag)

def vpin_factor(df, window=24, buckets=50, lag=1):
    """
    计算基于成交量同步化的知情交易概率指标
    高VPIN表示可能有更多的知情交易者在市场中
    """
    # 计算每个时段的成交量桶
    df_copy = df.copy()
    df_copy['bucket_volume'] = df['volume'].rolling(window=buckets).sum() / buckets
    
    # 估计买卖不平衡
    df_copy['imbalance'] = abs(df['taker_buy_base_asset_volume'] - (df['volume'] - df['taker_buy_base_asset_volume']))
    
    # 计算VPIN
    vpin = df_copy['imbalance'].rolling(window=buckets).sum() / (df_copy['bucket_volume'] * buckets)
    
    # 标准化VPIN (相对于窗口内的均值和标准差)
    vpin_ma = vpin.rolling(window=window).mean()
    vpin_std = vpin.rolling(window=window).std().replace(0, 1e-8)  # 避免除以零
    normalized_vpin = (vpin - vpin_ma) / vpin_std
    
    return normalized_vpin.shift(lag)

def hl_volume_position_factor(df, window=24, lag=1):
    """
    计算加权高低价位置因子
    考虑在高价区和低价区的成交量分布
    """
    # 当前价格在当日高低价区间中的位置 (0-1)
    hl_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-8)  # 避免除以零
    
    # 成交量加权的高低价位置
    # 当成交量高于平均时增加权重
    volume_ma = df['volume'].rolling(window=window).mean()
    volume_weight = df['volume'] / volume_ma
    
    # 应用成交量权重（高成交量放大位置信号）
    weighted_position = hl_position * volume_weight
    
    # 相对于窗口内的平均加权位置
    position_ma = weighted_position.rolling(window=window).mean()
    relative_position = weighted_position - position_ma
    
    return relative_position.shift(lag)

def trade_size_change_factor(df, window=24, lag=1):
    """
    计算平均交易规模的变化
    平均交易规模 = 成交量 / 交易次数
    """
    # 计算平均交易规模
    avg_trade_size = df['volume'] / df['number_of_trades'].replace(0, 1e-8)  # 避免除以零
    
    # 计算相对于过去窗口的交易规模变化
    size_ma = avg_trade_size.rolling(window=window).mean()
    size_change = avg_trade_size / size_ma - 1
    
    return size_change.shift(lag)

def volume_surge_factor(df, window=24, surge_threshold=2, lag=1):
    """
    检测成交量激增的因子
    成交量显著高于过去窗口的平均值
    """
    # 计算成交量相对于移动平均的比率
    volume_ma = df['volume'].rolling(window=window).mean()
    volume_ratio = df['volume'] / volume_ma
    
    # 计算成交量激增信号（二值化后平滑处理）
    surge_signal = (volume_ratio > surge_threshold).astype(float)
    smoothed_signal = surge_signal.rolling(window=5).mean()
    
    return smoothed_signal.shift(lag) 

# ----- 组合因子 ----- #

def market_balance_factor(df, window=24, lag=1):
    """
    市场平衡因子 - 结合买卖压力和流动性不平衡
    """
    buy_sell = buy_sell_pressure_factor(df, window, lag)
    liquidity = liquidity_imbalance_factor(df, window, lag)
    
    # 组合买卖压力和流动性不平衡
    return 0.5 * buy_sell + 0.5 * liquidity

def smart_momentum_factor(df, window=24, volume_weight=0.3, lag=1):
    """
    智能动量因子 - 结合价格动量和成交量变化
    """
    price_momentum = momentum_factor(df, window, lag)
    vol_change = volume_factor(df, window, lag)
    
    # 当成交量增加时增强动量信号
    return price_momentum * (1 + volume_weight * vol_change)

def trading_quality_factor(df, window=24, lag=1):
    """
    交易质量因子 - 结合交易效率和交易规模变化
    """
    efficiency = trading_efficiency_factor(df, window, lag)
    size_change = trade_size_change_factor(df, window, lag)
    
    # 高效率和适当的交易规模变化表示高质量交易
    return 0.6 * efficiency + 0.4 * size_change

def liquidity_risk_factor(df, window=24, lag=1):
    """
    流动性风险因子 - 结合价格影响和成交量激增
    """
    impact = price_impact_factor(df, window, lag)
    surge = volume_surge_factor(df, window, 2, lag)
    
    # 高价格影响和成交量急剧变化表示流动性风险
    return 0.7 * impact - 0.3 * surge 