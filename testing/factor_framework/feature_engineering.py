import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from .factors import (
    ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_range,
    ts_skewness, ts_kurtosis, rsi_factor, bollinger_band_factor, macd_factor
)
from .utils import normalize_factor, winsorize_factor
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    特征工程类，用于数据预处理和特征构建
    """
    
    def __init__(self, 
                 scaling_method: str = 'robust',
                 winsorize_limits: Tuple[float, float] = (0.005, 0.995),
                 fill_method: str = 'ffill'):
        """
        初始化特征工程器
        
        参数:
        ----
        scaling_method: 标准化方法 ('standard', 'robust', 或 None)
        winsorize_limits: 去极值范围 (下限百分比, 上限百分比)
        fill_method: 填充缺失值方法 ('ffill', 'bfill', 'zero', 'mean')
        """
        self.scaling_method = scaling_method
        self.winsorize_limits = winsorize_limits
        self.fill_method = fill_method
        
        # 初始化缩放器
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None
    
    def prepare_target_variables(self, 
                               df: pd.DataFrame, 
                               price_col: str = 'close',
                               horizons: List[int] = [1, 4, 24]) -> pd.DataFrame:
        """
        准备目标变量 - 计算未来N小时的价格变动百分比
        
        参数:
        ----
        df: 数据框，包含价格数据
        price_col: 价格列名
        horizons: 预测时间窗口列表（小时）
        
        返回:
        ----
        添加了目标变量的数据框
        """
        result_df = df.copy()
        
        # 计算每个时间窗口的未来收益率
        for horizon in horizons:
            # 未来N小时的价格变动百分比
            future_return = df[price_col].pct_change(periods=horizon).shift(-horizon)
            
            # 添加到结果数据框
            result_df[f'future_return_{horizon}h'] = future_return
            
            # 可选：添加未来N小时的方向（1=上涨，0=下跌）
            result_df[f'future_direction_{horizon}h'] = (future_return > 0).astype(int)
        
        return result_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        df_filled = df.copy()
        
        for col in df.columns:
            # 跳过目标变量列（未来收益率）
            if col.startswith('future_'):
                continue
                
            # 根据指定方法填充缺失值
            if self.fill_method == 'ffill':
                df_filled[col] = df_filled[col].fillna(method='ffill')
                # 对开始的缺失值使用后向填充
                df_filled[col] = df_filled[col].fillna(method='bfill')
            elif self.fill_method == 'bfill':
                df_filled[col] = df_filled[col].fillna(method='bfill')
                # 对结束的缺失值使用前向填充
                df_filled[col] = df_filled[col].fillna(method='ffill')
            elif self.fill_method == 'zero':
                df_filled[col] = df_filled[col].fillna(0)
            elif self.fill_method == 'mean':
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
        return df_filled
    
    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """去极值处理"""
        df_winsorized = df.copy()
        
        for col in df.columns:
            # 跳过目标变量列和分类变量
            if col.startswith('future_') or df[col].dtype == 'object':
                continue
                
            # 计算分位数界限
            lower_limit = df[col].quantile(self.winsorize_limits[0])
            upper_limit = df[col].quantile(self.winsorize_limits[1])
            
            # 去极值
            df_winsorized[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        return df_winsorized
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        if self.scaler is None:
            return df
            
        df_scaled = df.copy()
        feature_cols = [col for col in df.columns if not col.startswith('future_') and df[col].dtype != 'object']
        
        if feature_cols:
            # 拟合并转换特征
            scaled_features = self.scaler.fit_transform(df[feature_cols])
            
            # 将标准化后的特征放回数据框
            df_scaled[feature_cols] = scaled_features
        
        return df_scaled
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理主函数
        
        参数:
        ----
        df: 原始数据框
        
        返回:
        ----
        预处理后的数据框
        """
        # 处理缺失值
        df_processed = self._handle_missing_values(df)
        
        # 去极值
        df_processed = self._winsorize(df_processed)
        
        # 标准化特征
        df_processed = self._scale_features(df_processed)
        
        return df_processed
    
    def generate_price_features(self, 
                              df: pd.DataFrame, 
                              price_cols: List[str] = ['open', 'high', 'low', 'close'],
                              windows: List[int] = [6, 12, 24, 72, 144]) -> pd.DataFrame:
        """
        生成价格相关特征
        
        参数:
        ----
        df: 数据框
        price_cols: 价格相关列
        windows: 窗口大小列表（小时）
        
        返回:
        ----
        添加了价格特征的数据框
        """
        result_df = df.copy()
        
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 高低价差比
            result_df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            
            # 开盘收盘价差比
            result_df['oc_ratio'] = (df['open'] - df['close']) / df['close']
        
        # 移动平均特征
        for window in windows:
            for col in price_cols:
                if col in df.columns:
                    # 移动平均
                    result_df[f'{col}_ma_{window}h'] = ts_mean(df[col], window)
                    
                    # 价格相对于移动平均的位置
                    result_df[f'{col}_relative_to_ma_{window}h'] = df[col] / result_df[f'{col}_ma_{window}h'] - 1
                    
        # 波动率特征 
        for window in windows:
            if 'close' in df.columns:
                # 计算收益率
                returns = df['close'].pct_change()
                
                # 波动率 (标准差)
                result_df[f'volatility_{window}h'] = ts_std(returns, window)
                
                # 归一化的价格范围
                result_df[f'norm_range_{window}h'] = ts_range(df['close'], window) / ts_mean(df['close'], window)
                
                # 偏度和峰度
                result_df[f'skewness_{window}h'] = ts_skewness(returns, window)
                result_df[f'kurtosis_{window}h'] = ts_kurtosis(returns, window)
        
        return result_df
    
    def generate_volume_features(self,
                              df: pd.DataFrame,
                              volume_col: str = 'volume',
                              price_col: str = 'close',
                              windows: List[int] = [6, 12, 24, 72, 144]) -> pd.DataFrame:
        """
        生成成交量相关特征
        
        参数:
        ----
        df: 数据框
        volume_col: 成交量列名
        price_col: 价格列名
        windows: 窗口大小列表（小时）
        
        返回:
        ----
        添加了成交量特征的数据框
        """
        result_df = df.copy()
        
        # 基本成交量特征
        if volume_col in df.columns:
            # 对数成交量
            result_df[f'log_{volume_col}'] = np.log1p(df[volume_col])
            
            # 成交量变化率
            result_df[f'{volume_col}_change'] = df[volume_col].pct_change()
            
            # 成交量移动平均
            for window in windows:
                result_df[f'{volume_col}_ma_{window}h'] = ts_mean(df[volume_col], window)
                
                # 当前成交量相对于平均水平
                result_df[f'{volume_col}_relative_{window}h'] = df[volume_col] / result_df[f'{volume_col}_ma_{window}h']
                
                # 对数成交量相对变化
                log_vol = result_df[f'log_{volume_col}']
                log_vol_ma = ts_mean(log_vol, window)
                result_df[f'log_vol_rel_{window}h'] = log_vol - log_vol_ma
        
        # 价格-成交量关系特征
        if volume_col in df.columns and price_col in df.columns:
            # 成交量加权价格
            result_df['volume_weighted_price'] = df[price_col] * df[volume_col]
            
            # 价格-成交量相关性
            for window in windows:
                price_returns = df[price_col].pct_change()
                vol_changes = df[volume_col].pct_change()
                
                # 计算移动相关性 (这里使用协方差代替，因为相关性计算较复杂)
                price_vol_cov = (price_returns * vol_changes).rolling(window=window).mean()
                result_df[f'price_vol_cov_{window}h'] = price_vol_cov
        
        return result_df
    
    def generate_technical_indicators(self,
                                  df: pd.DataFrame,
                                  ohlcv_cols: List[str] = ['open', 'high', 'low', 'close', 'volume']) -> pd.DataFrame:
        """
        生成技术指标
        
        参数:
        ----
        df: 数据框
        ohlcv_cols: OHLCV列名

        返回:
        ----
        添加了技术指标的数据框
        """
        result_df = df.copy()
        
        # 确保所有必要的列都存在
        required_cols = {
            'open': ohlcv_cols[0] if len(ohlcv_cols) > 0 else None,
            'high': ohlcv_cols[1] if len(ohlcv_cols) > 1 else None,
            'low': ohlcv_cols[2] if len(ohlcv_cols) > 2 else None,
            'close': ohlcv_cols[3] if len(ohlcv_cols) > 3 else None,
            'volume': ohlcv_cols[4] if len(ohlcv_cols) > 4 else None
        }
        
        # 检查必要的价格列是否存在
        price_cols_exist = all(required_cols[col] is not None and required_cols[col] in df.columns for col in ['close', 'high', 'low'])
        
        if not price_cols_exist:
            print("警告: 缺少必要的价格列(close, high, low), 无法计算部分技术指标")
            return result_df
            
        # 使用factors.py中的纯Python实现替代TA-Lib
        
        # 1. RSI - 相对强弱指标
        close_col = required_cols['close']
        # 使用3个不同的窗口计算RSI
        for window in [6, 14, 24]:
            # 使用纯Python实现
            rsi = rsi_factor(df, window=window, price_col=close_col)
            result_df[f'rsi_{window}'] = rsi
        
        # 2. 布林带
        for window in [20, 50]:
            bb_result = bollinger_band_factor(df, window=window, price_col=close_col)
            result_df[f'bb_upper_{window}'] = bb_result['bb_position'] + 1
            result_df[f'bb_lower_{window}'] = 1 - bb_result['bb_position']
            result_df[f'bb_width_{window}'] = bb_result['bb_width']
        
        # 3. MACD
        macd_result = macd_factor(df, fast=12, slow=26, signal=9, price_col=close_col)
        result_df['macd'] = macd_result['macd_line']
        result_df['macd_signal'] = macd_result['signal_line']
        result_df['macd_hist'] = macd_result['macd_histogram']
        
        # 4. 动量指标
        for window in [10, 20, 60]:
            # 简单动量：当前收盘价与n天前收盘价的比率
            result_df[f'momentum_{window}'] = df[close_col] / df[close_col].shift(window) - 1
            
            # 累积收益率，表示过去n天的累积收益
            result_df[f'cumulative_return_{window}'] = df[close_col].pct_change().rolling(window=window).apply(lambda x: (1 + x).prod() - 1)
        
        # 5. 趋势强度指标 (自定义)
        for window in [14, 30]:
            # 上涨天数比例
            up_days = (df[close_col].diff() > 0).rolling(window=window).sum() / window
            result_df[f'up_days_ratio_{window}'] = up_days
            
            # 趋势强度：用正收益和负收益的比率表示
            pos_returns = df[close_col].diff().clip(lower=0).rolling(window=window).sum()
            neg_returns = df[close_col].diff().clip(upper=0).abs().rolling(window=window).sum()
            result_df[f'trend_strength_{window}'] = pos_returns / (neg_returns + 1e-8)  # 避免除以零
        
        # 6. 价格变化率
        for window in [1, 5, 20]:
            result_df[f'price_change_{window}d'] = df[close_col].pct_change(periods=window)
            
        # 7. 交易量变化率
        if required_cols['volume'] is not None and required_cols['volume'] in df.columns:
            volume_col = required_cols['volume']
            for window in [1, 5, 20]:
                result_df[f'volume_change_{window}d'] = df[volume_col].pct_change(periods=window)
                
            # 8. 成交量趋势指标
            for window in [10, 20]:
                # 成交量相对变化
                vol_ma = df[volume_col].rolling(window=window).mean()
                result_df[f'volume_trend_{window}'] = df[volume_col] / vol_ma - 1
                
                # 价格体积趋势
                price_vol = df[close_col] * df[volume_col]
                price_vol_ma = price_vol.rolling(window=window).mean()
                result_df[f'price_volume_trend_{window}'] = price_vol / price_vol_ma - 1
        
        return result_df
    
    def generate_crypto_specific_features(self,
                                          df: pd.DataFrame,
                                          price_col: str = 'close',
                                          volume_col: str = 'volume',
                                          windows: List[int] = [6, 12, 24, 72]) -> pd.DataFrame:
        """
        生成加密货币特有的特征
        
        参数:
        ----
        df: 数据框
        price_col: 价格列名
        volume_col: 成交量列名
        windows: 窗口大小列表（小时）
        
        返回:
        ----
        添加了加密货币特有特征的数据框
        """
        result_df = df.copy()
        
        # 1. 买卖压力
        if 'taker_buy_base_asset_volume' in df.columns and volume_col in df.columns:
            # 买方占比
            result_df['buy_ratio'] = df['taker_buy_base_asset_volume'] / df[volume_col]
            
            # 不同窗口的买方占比移动平均
            for window in windows:
                result_df[f'buy_ratio_ma_{window}h'] = ts_mean(result_df['buy_ratio'], window)
                
                # 买方压力相对变化
                result_df[f'buy_pressure_{window}h'] = result_df['buy_ratio'] / result_df[f'buy_ratio_ma_{window}h'] - 1
                
        # 2. 流动性指标
        if 'number_of_trades' in df.columns and volume_col in df.columns:
            # 平均交易规模
            result_df['avg_trade_size'] = df[volume_col] / df['number_of_trades']
            
            # 交易密度
            for window in windows:
                # 每小时平均交易数量
                result_df[f'trades_per_hour_{window}h'] = ts_mean(df['number_of_trades'], window)
                
                # 交易规模变化
                avg_size_ma = ts_mean(result_df['avg_trade_size'], window)
                result_df[f'trade_size_change_{window}h'] = result_df['avg_trade_size'] / avg_size_ma - 1
                
        # 3. 价格影响因子
        if all(col in df.columns for col in ['high', 'low', volume_col]):
            # 价格范围
            result_df['price_range'] = (df['high'] - df['low']) / df['low']
            
            # 单位成交量的价格影响
            result_df['price_impact'] = result_df['price_range'] / df[volume_col]
            
            for window in windows:
                # 价格影响的移动平均
                result_df[f'price_impact_ma_{window}h'] = ts_mean(result_df['price_impact'], window)
                
                # 相对价格影响
                result_df[f'relative_impact_{window}h'] = result_df['price_impact'] / result_df[f'price_impact_ma_{window}h']
                
        # 4. 波动率相关指标
        if price_col in df.columns:
            returns = df[price_col].pct_change()
            
            for window in windows:
                # 已经在price_features中计算了基本波动率，这里添加额外指标
                
                # 上行波动率 vs 下行波动率
                pos_returns = returns.clip(lower=0)
                neg_returns = returns.clip(upper=0).abs()
                
                result_df[f'upside_volatility_{window}h'] = ts_std(pos_returns, window) 
                result_df[f'downside_volatility_{window}h'] = ts_std(neg_returns, window)
                
                # 波动率比率 (上行/下行)
                up_vol = result_df[f'upside_volatility_{window}h']
                down_vol = result_df[f'downside_volatility_{window}h']
                result_df[f'volatility_ratio_{window}h'] = up_vol / (down_vol + 1e-8)  # 避免除以零
                
        # 5. 价格-成交量趋势背离
        if price_col in df.columns and volume_col in df.columns:
            for window in windows:
                # 价格趋势
                price_ma = ts_mean(df[price_col], window)
                price_trend = df[price_col] / price_ma - 1
                
                # 成交量趋势
                vol_ma = ts_mean(df[volume_col], window)
                vol_trend = df[volume_col] / vol_ma - 1
                
                # 趋势背离指标
                result_df[f'trend_divergence_{window}h'] = price_trend - vol_trend
                
        # 6. 修改后的VPIN (Volume-Synchronized Probability of Informed Trading)
        if volume_col in df.columns and 'taker_buy_base_asset_volume' in df.columns:
            for window in [12, 24, 48]:
                # 计算成交不平衡
                imbalance = (df['taker_buy_base_asset_volume'] - (df[volume_col] - df['taker_buy_base_asset_volume'])).abs()
                
                # 标准化不平衡
                vpin = (imbalance / df[volume_col]).rolling(window=window).mean()
                result_df[f'vpin_{window}h'] = vpin
                
        # 7. 高低价在成交量权重下的位置
        if all(col in df.columns for col in ['high', 'low', 'close', volume_col]):
            # 日内高低价位置
            for window in [24, 48]:  # 假设是日内或两日数据
                # 高低价范围
                high_max = ts_max(df['high'], window)
                low_min = ts_min(df['low'], window)
                
                # 当前收盘价在范围中的位置
                result_df[f'close_position_{window}h'] = (df['close'] - low_min) / (high_max - low_min + 1e-8)
                
                # 添加成交量权重
                vol_sum = ts_sum(df[volume_col], window)
                vol_weighted_close = (df['close'] * df[volume_col]).rolling(window=window).sum() / (vol_sum + 1e-8)
                vol_weighted_position = (vol_weighted_close - low_min) / (high_max - low_min + 1e-8)
                result_df[f'vol_weighted_position_{window}h'] = vol_weighted_position
                
        return result_df
    
    def create_all_features(self, 
                          df: pd.DataFrame, 
                          price_cols: List[str] = ['open', 'high', 'low', 'close'],
                          volume_col: str = 'volume',
                          horizons: List[int] = [1, 4, 24]) -> pd.DataFrame:
        """
        一次性创建所有特征
        
        参数:
        ----
        df: 数据框
        price_cols: 价格相关列
        volume_col: 成交量列
        horizons: 预测时间窗口列表（小时）
        
        返回:
        ----
        包含所有特征和目标变量的预处理数据框
        """
        # 克隆数据框
        result_df = df.copy()
        
        # 1. 准备目标变量
        result_df = self.prepare_target_variables(result_df, price_col='close', horizons=horizons)
        
        # 2. 生成价格特征
        result_df = self.generate_price_features(result_df, price_cols=price_cols)
        
        # 3. 生成成交量特征
        if volume_col in df.columns:
            result_df = self.generate_volume_features(result_df, volume_col=volume_col)
        
        # 4. 生成技术指标
        ohlcv_cols = price_cols + [volume_col] if volume_col in df.columns else price_cols
        result_df = self.generate_technical_indicators(result_df, ohlcv_cols=ohlcv_cols)
        
        # 5. 生成加密货币特有特征
        result_df = self.generate_crypto_specific_features(result_df)
        
        # 6. 数据预处理
        result_df = self.preprocess_data(result_df)
        
        return result_df 