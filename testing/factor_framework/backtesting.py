import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class FactorBacktester:
    """
    因子回测类，用于验证特征选择的有效性
    """
    
    def __init__(self,
                initial_capital: float = 10000,
                fee_rate: float = 0.001,
                n_groups: int = 5):
        """
        初始化回测器
        
        参数:
        ----
        initial_capital: 初始资金
        fee_rate: 交易费率
        n_groups: 分组数量
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.n_groups = n_groups
        self.results = {}
        
    def evaluate_factor_ic(self,
                          df: pd.DataFrame,
                          factor_values: pd.Series,
                          target_col: str,
                          rolling_window: Optional[int] = None) -> Dict:
        """
        评估因子的IC值
        
        参数:
        ----
        df: 数据框
        factor_values: 因子值Series
        target_col: 目标变量列名
        rolling_window: 滚动窗口大小，None表示计算整体IC
        
        返回:
        ----
        IC评估结果字典
        """
        # 确保因子值和目标变量有相同的索引
        common_index = factor_values.index.intersection(df.index)
        factor = factor_values.loc[common_index]
        target = df.loc[common_index, target_col]
        
        # 移除缺失值
        valid_data = pd.DataFrame({'factor': factor, 'target': target}).dropna()
        factor = valid_data['factor']
        target = valid_data['target']
        
        results = {}
        
        # 计算整体IC
        if len(factor) > 0 and len(target) > 0:
            ic, p_value = spearmanr(factor, target)
            results['overall_ic'] = ic
            results['overall_p_value'] = p_value
        else:
            results['overall_ic'] = np.nan
            results['overall_p_value'] = np.nan
        
        # 计算滚动IC
        if rolling_window is not None:
            rolling_ic = []
            dates = []
            
            for i in range(rolling_window, len(factor)):
                start_idx = i - rolling_window
                end_idx = i
                
                if start_idx >= 0 and end_idx < len(factor):
                    f_window = factor.iloc[start_idx:end_idx]
                    t_window = target.iloc[start_idx:end_idx]
                    
                    # 确保窗口内有足够的数据
                    if len(f_window) > 10 and len(t_window) > 10:
                        try:
                            window_ic, _ = spearmanr(f_window, t_window)
                            rolling_ic.append(window_ic)
                            dates.append(factor.index[end_idx - 1])
                        except:
                            rolling_ic.append(np.nan)
                            dates.append(factor.index[end_idx - 1])
            
            # 创建滚动IC Series
            if rolling_ic:
                results['rolling_ic'] = pd.Series(rolling_ic, index=dates)
            else:
                results['rolling_ic'] = pd.Series()
        
        return results
    
    def run_group_backtest(self,
                         df: pd.DataFrame,
                         factor_values: pd.Series,
                         target_col: str,
                         holding_period: int = 1) -> Dict:
        """
        运行分组回测
        
        参数:
        ----
        df: 数据框
        factor_values: 因子值Series
        target_col: 目标变量列名
        holding_period: 持有期（时间单位同df索引）
        
        返回:
        ----
        回测结果字典
        """
        # 确保因子值和目标变量有相同的索引
        common_index = factor_values.index.intersection(df.index)
        factor = factor_values.loc[common_index]
        
        # 创建分组标签
        groups = pd.qcut(factor, self.n_groups, labels=False, duplicates='drop')
        groups = pd.Series(groups, index=factor.index)
        
        # 初始化分组收益率
        group_returns = {i: [] for i in range(self.n_groups)}
        group_dates = {i: [] for i in range(self.n_groups)}
        
        # 对每个日期进行回测
        for date in sorted(groups.index):
            # 获取当前日期的分组
            current_group = groups.loc[date]
            
            # 如果没有分组信息，跳过
            if pd.isna(current_group):
                continue
                
            # 计算未来持有期收益率
            future_date_idx = df.index.get_indexer([date])[0] + holding_period
            
            # 如果未来日期超出范围，跳过
            if future_date_idx >= len(df.index):
                continue
                
            future_date = df.index[future_date_idx]
            
            # 计算收益率
            if date in df.index and future_date in df.index:
                future_return = df.loc[future_date, target_col]
                
                # 添加到相应分组
                group_returns[current_group].append(future_return)
                group_dates[current_group].append(future_date)
        
        # 转换为DataFrame
        returns_df = pd.DataFrame()
        
        for group in range(self.n_groups):
            if group_returns[group] and group_dates[group]:
                group_series = pd.Series(group_returns[group], index=group_dates[group])
                group_series = group_series.sort_index()
                returns_df[f'Group_{group}'] = group_series
        
        # 计算多空组合（最高分组减最低分组）
        if f'Group_0' in returns_df.columns and f'Group_{self.n_groups-1}' in returns_df.columns:
            common_dates = returns_df[f'Group_0'].index.intersection(returns_df[f'Group_{self.n_groups-1}'].index)
            returns_df['Long_Short'] = returns_df.loc[common_dates, f'Group_{self.n_groups-1}'] - returns_df.loc[common_dates, f'Group_0']
        
        # 计算性能指标
        performance = self._calculate_performance(returns_df)
        
        results = {
            'returns': returns_df,
            'performance': performance,
            'groups': groups
        }
        
        return results
    
    def _calculate_performance(self, returns_df: pd.DataFrame) -> Dict:
        """计算回测性能指标"""
        performance = {}
        
        # 对每个分组计算性能指标
        for col in returns_df.columns:
            returns = returns_df[col].dropna()
            
            if len(returns) == 0:
                continue
                
            # 累积收益率
            cumulative_return = (1 + returns).cumprod().iloc[-1] - 1
            
            # 年化收益率 (假设252个交易日)
            annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
            
            # 年化波动率
            annualized_volatility = returns.std() * np.sqrt(252)
            
            # 夏普比率
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            max_drawdown = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()
            
            # 胜率
            win_rate = (returns > 0).sum() / len(returns)
            
            # 存储结果
            performance[col] = {
                'cumulative_return': cumulative_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(returns)
            }
        
        return performance
    
    def combine_factors(self,
                      factor_dict: Dict[str, pd.Series],
                      weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        组合多个因子
        
        参数:
        ----
        factor_dict: 因子字典 {因子名: 因子值Series}
        weights: 权重字典 {因子名: 权重}，如果为None则使用等权重
        
        返回:
        ----
        组合后的因子值Series
        """
        if not factor_dict:
            raise ValueError("因子字典不能为空")
            
        # 如果没有提供权重，使用等权重
        if weights is None:
            weights = {name: 1.0 / len(factor_dict) for name in factor_dict.keys()}
        
        # 检查所有因子和权重
        if set(weights.keys()) != set(factor_dict.keys()):
            raise ValueError("权重字典与因子字典的键不匹配")
            
        # 找出所有因子的共同索引
        common_index = None
        for factor in factor_dict.values():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        # 初始化组合因子
        combined_factor = pd.Series(0.0, index=common_index)
        
        # 标准化并加权组合因子
        for name, factor in factor_dict.items():
            # 提取共同索引的数据
            aligned_factor = factor.loc[common_index]
            
            # 标准化因子 (z-score)
            mean = aligned_factor.mean()
            std = aligned_factor.std()
            if std > 0:
                normalized_factor = (aligned_factor - mean) / std
            else:
                normalized_factor = aligned_factor - mean
                
            # 加权求和
            combined_factor += normalized_factor * weights[name]
        
        return combined_factor
    
    def plot_returns(self,
                   returns_df: pd.DataFrame,
                   title: str = 'Group Returns',
                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制分组回测收益率
        
        参数:
        ----
        returns_df: 收益率数据框
        title: 图表标题
        figsize: 图形大小
        
        返回:
        ----
        matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算累积收益率
        cumulative_returns = (1 + returns_df).cumprod()
        
        # 设置颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(returns_df.columns)))
        
        # 绘制每个分组的累积收益率
        for i, col in enumerate(returns_df.columns):
            # 如果是多空组合，使用不同的样式
            if col == 'Long_Short':
                ax.plot(cumulative_returns.index, cumulative_returns[col], 
                       label=col, color='black', linestyle='--', linewidth=2)
            else:
                ax.plot(cumulative_returns.index, cumulative_returns[col], 
                       label=col, color=colors[i])
        
        # 设置图表
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('累积收益率', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_ic_timeseries(self,
                         rolling_ic: pd.Series,
                         title: str = 'Rolling Information Coefficient',
                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制滚动IC时间序列
        
        参数:
        ----
        rolling_ic: 滚动IC值Series
        title: 图表标题
        figsize: 图形大小
        
        返回:
        ----
        matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制IC时间序列
        ax.plot(rolling_ic.index, rolling_ic.values, color='blue', alpha=0.7)
        
        # 添加0线
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # 添加平均IC线
        mean_ic = rolling_ic.mean()
        ax.axhline(y=mean_ic, color='g', linestyle='--', 
                  label=f'Mean IC: {mean_ic:.4f}')
        
        # 设置图表
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('IC值', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self,
                      backtest_results: Dict,
                      factor_name: str,
                      target_col: str,
                      output_path: Optional[str] = None) -> str:
        """
        生成回测报告
        
        参数:
        ----
        backtest_results: 回测结果字典
        factor_name: 因子名称
        target_col: 目标变量列名
        output_path: 报告输出路径，如果为None则只返回报告文本
        
        返回:
        ----
        报告文本
        """
        if 'returns' not in backtest_results or 'performance' not in backtest_results:
            raise ValueError("回测结果格式不正确")
            
        returns_df = backtest_results['returns']
        performance = backtest_results['performance']
        
        # 生成报告内容
        report = []
        report.append(f"# {factor_name} 因子回测报告")
        report.append(f"\n## 1. 回测概览")
        report.append(f"- 因子名称: {factor_name}")
        report.append(f"- 目标变量: {target_col}")
        report.append(f"- 分组数量: {self.n_groups}")
        report.append(f"- 回测周期: {returns_df.index[0]} 至 {returns_df.index[-1]}")
        report.append(f"- 交易次数: {sum(perf['num_trades'] for perf in performance.values())}")
        
        # IC统计
        if 'ic_results' in backtest_results:
            ic_results = backtest_results['ic_results']
            report.append(f"\n## 2. 信息系数 (IC) 分析")
            report.append(f"- 整体IC: {ic_results.get('overall_ic', 'N/A'):.4f}")
            report.append(f"- IC显著性 (p值): {ic_results.get('overall_p_value', 'N/A'):.4f}")
            if 'rolling_ic' in ic_results:
                rolling_ic = ic_results['rolling_ic']
                report.append(f"- 平均滚动IC: {rolling_ic.mean():.4f}")
                report.append(f"- 滚动IC波动率: {rolling_ic.std():.4f}")
                report.append(f"- 滚动IC正值比例: {(rolling_ic > 0).mean():.2%}")
        
        # 分组绩效
        report.append(f"\n## 3. 分组绩效")
        for group, perf in performance.items():
            report.append(f"\n### {group}")
            report.append(f"- 累积收益率: {perf['cumulative_return']:.2%}")
            report.append(f"- 年化收益率: {perf['annualized_return']:.2%}")
            report.append(f"- 年化波动率: {perf['annualized_volatility']:.2%}")
            report.append(f"- 夏普比率: {perf['sharpe_ratio']:.2f}")
            report.append(f"- 最大回撤: {perf['max_drawdown']:.2%}")
            report.append(f"- 胜率: {perf['win_rate']:.2%}")
            report.append(f"- 交易次数: {perf['num_trades']}")
        
        # 多空组合分析
        if 'Long_Short' in performance:
            report.append(f"\n## 4. 多空组合分析")
            ls_perf = performance['Long_Short']
            report.append(f"- 累积收益率: {ls_perf['cumulative_return']:.2%}")
            report.append(f"- 年化收益率: {ls_perf['annualized_return']:.2%}")
            report.append(f"- 年化波动率: {ls_perf['annualized_volatility']:.2%}")
            report.append(f"- 夏普比率: {ls_perf['sharpe_ratio']:.2f}")
            report.append(f"- 最大回撤: {ls_perf['max_drawdown']:.2%}")
            report.append(f"- 胜率: {ls_perf['win_rate']:.2%}")
        
        # 总结与结论
        report.append(f"\n## 5. 结论")
        
        # 判断因子是否有效
        if 'Long_Short' in performance:
            ls_perf = performance['Long_Short']
            if ls_perf['sharpe_ratio'] > 1.0:
                report.append(f"- 该因子表现良好，多空组合夏普比率为 {ls_perf['sharpe_ratio']:.2f}")
            elif ls_perf['sharpe_ratio'] > 0.5:
                report.append(f"- 该因子表现一般，多空组合夏普比率为 {ls_perf['sharpe_ratio']:.2f}")
            else:
                report.append(f"- 该因子表现较差，多空组合夏普比率为 {ls_perf['sharpe_ratio']:.2f}")
        
        if 'ic_results' in backtest_results:
            ic_results = backtest_results['ic_results']
            if abs(ic_results.get('overall_ic', 0)) > 0.05:
                report.append(f"- 因子与目标变量具有统计显著的相关性，IC为 {ic_results.get('overall_ic', 0):.4f}")
            else:
                report.append(f"- 因子与目标变量相关性较弱，IC为 {ic_results.get('overall_ic', 0):.4f}")
        
        # 合并成最终报告文本
        report_text = "\n".join(report)
        
        # 如果指定了输出路径，保存报告
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"回测报告已保存至: {output_path}")
        
        return report_text 