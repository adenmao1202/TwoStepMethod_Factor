import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .feature_engineering import FeatureEngineer
from .feature_selection import FactorSelector
from .backtesting import FactorBacktester

class FactorResearchPipeline:

    
    def __init__(self,
                data_path: str,
                output_dir: str = "output",
                config: Optional[Dict] = None):
        """
        初始化因子研究流水线
        
        参数:
        ----
        data_path: 数据文件路径
        output_dir: 输出目录路径
        config: 配置参数字典
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.config = config or self._default_config()
        
        # 加载模块
        self.feature_engineer = FeatureEngineer(
            scaling_method=self.config.get('scaling_method', 'robust'),
            winsorize_limits=(self.config.get('winsorize_lower', 0.005), self.config.get('winsorize_upper', 0.995)),
            fill_method=self.config.get('fill_method', 'ffill')
        )
        
        self.factor_selector = FactorSelector(
            prediction_horizons=self.config.get('prediction_horizons', [1, 4, 24]),
            cv_folds=self.config.get('cv_folds', 5),
            random_state=self.config.get('random_state', 42)
        )
        
        self.backtester = FactorBacktester(
            initial_capital=self.config.get('initial_capital', 10000),
            fee_rate=self.config.get('fee_rate', 0.001),
            n_groups=self.config.get('n_groups', 5)
        )
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 记录时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 存储数据和结果
        self.df = None
        self.features_df = None
        self.selected_features = {}
        self.backtest_results = {}
        self.execution_time = {}
        
    def _default_config(self) -> Dict:
        return {
            # 数据处理配置
            'scaling_method': 'robust',  # 标准化方法
            'winsorize_lower': 0.005,    # 去极值下界
            'winsorize_upper': 0.995,    # 去极值上界
            'fill_method': 'ffill',      # 填充缺失值方法
            
            # 特征选择配置
            'prediction_horizons': [1, 4, 24],  # 预测时间窗口
            'cv_folds': 5,               # 交叉验证折数
            'random_state': 42,          # 随机种子
            'top_n_lasso': 50,           # Lasso选择的特征数量
            'top_n_final': 20,           # 最终选择的特征数量
            
            # 回测配置
            'initial_capital': 10000,    # 初始资金
            'fee_rate': 0.001,           # 交易费率
            'n_groups': 5,               # 分组数量
            'rolling_window': 60,        # 滚动窗口大小
            
            # 数据列名配置
            'price_cols': ['open', 'high', 'low', 'close'],  # 价格列名
            'volume_col': 'volume',      # 交易量列名
            'timestamp_col': None,       # 时间戳列名，None表示使用索引
            
            # 流程控制
            'run_feature_engineering': True,  # 是否运行特征工程
            'run_feature_selection': True,    # 是否运行特征选择
            'run_backtest': True,             # 是否运行回测
            'save_intermediate': True,        # 是否保存中间结果
            'verbose': True                   # 是否输出详细信息
        }
    
    def load_data(self) -> pd.DataFrame:
        """
        加载数据
        
        返回:
        ----
        加载的数据框
        """
        print("1. 加载数据...")
        start_time = time.time()
        
        # 确定文件类型并加载
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.parquet'):
            self.df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.pkl') or self.data_path.endswith('.pickle'):
            self.df = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path}")
        
        # 如果没有指定时间戳列，尝试将索引转换为日期时间
        if self.config['timestamp_col'] is None:
            try:
                if not isinstance(self.df.index, pd.DatetimeIndex):
                    self.df.index = pd.to_datetime(self.df.index)
            except:
                print("警告: 无法将索引转换为DatetimeIndex")
        else:
            # 使用指定的时间戳列作为索引
            timestamp_col = self.config['timestamp_col']
            if timestamp_col in self.df.columns:
                self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
                self.df.set_index(timestamp_col, inplace=True)
        
        # 确保索引已排序
        self.df = self.df.sort_index()
        
        # 打印数据信息
        print(f"  数据形状: {self.df.shape}")
        print(f"  时间范围: {self.df.index.min()} - {self.df.index.max()}")
        print(f"  缺失值: {self.df.isna().sum().sum()} ({self.df.isna().sum().sum() / self.df.size:.2%})")
        
        # 记录执行时间
        self.execution_time['load_data'] = time.time() - start_time
        
        # 如果配置为保存中间结果，保存原始数据
        if self.config['save_intermediate']:
            raw_data_path = os.path.join(self.output_dir, f"raw_data_{self.timestamp}.parquet")
            self.df.to_parquet(raw_data_path)
            print(f"  原始数据已保存至: {raw_data_path}")
        
        return self.df
    
    def generate_features(self) -> pd.DataFrame:
        """
        生成特征
        
        返回:
        ----
        包含特征的数据框
        """
        if not self.config['run_feature_engineering']:
            print("跳过特征工程阶段")
            return self.df
            
        print("\n2. 生成特征...")
        start_time = time.time()
        
        # 如果数据尚未加载，先加载数据
        if self.df is None:
            self.load_data()
        
        # 提取配置参数
        price_cols = self.config['price_cols']
        volume_col = self.config['volume_col']
        prediction_horizons = self.config['prediction_horizons']
        
        # 检查所需列是否存在
        for col in price_cols + [volume_col]:
            if col not in self.df.columns:
                print(f"  警告: 列 {col} 不存在于数据框中")
        
        # 生成特征
        self.features_df = self.feature_engineer.create_all_features(
            df=self.df,
            price_cols=price_cols,
            volume_col=volume_col,
            horizons=prediction_horizons
        )
        
        # 打印特征信息
        feature_cols = [col for col in self.features_df.columns if not col.startswith('future_')]
        print(f"  生成了 {len(feature_cols)} 个特征")
        print(f"  数据形状: {self.features_df.shape}")
        
        # 记录执行时间
        self.execution_time['generate_features'] = time.time() - start_time
        
        # 如果配置为保存中间结果，保存特征数据
        if self.config['save_intermediate']:
            features_path = os.path.join(self.output_dir, f"features_{self.timestamp}.parquet")
            self.features_df.to_parquet(features_path)
            print(f"  特征数据已保存至: {features_path}")
        
        return self.features_df
    
    def select_features(self) -> Dict:
        """
        特征选择
        
        返回:
        ----
        特征选择结果字典
        """
        if not self.config['run_feature_selection']:
            print("跳过特征选择阶段")
            return {}
            
        print("\n3. 特征选择...")
        start_time = time.time()
        
        # 如果特征尚未生成，先生成特征
        if self.features_df is None:
            self.generate_features()
        
        # 提取特征列（排除目标变量列）
        feature_cols = [col for col in self.features_df.columns if not col.startswith('future_')]
        
        # 运行两阶段特征选择
        selection_results = self.factor_selector.run_selection_pipeline(
            df=self.features_df,
            features=feature_cols,
            top_n_lasso=self.config['top_n_lasso'],
            top_n_final=self.config['top_n_final']
        )
        
        # 保存选择的特征
        self.selected_features = selection_results['final_selected']
        
        # 打印选择结果
        print("\n特征选择结果:")
        for horizon, features in self.selected_features.items():
            print(f"  {horizon}小时窗口: 选择了 {len(features)} 个特征")
            print(f"    前5个特征: {', '.join(features[:5])}")
        
        # 记录执行时间
        self.execution_time['select_features'] = time.time() - start_time
        
        # 生成特征选择报告
        report_path = os.path.join(self.output_dir, f"feature_selection_report_{self.timestamp}.md")
        self.factor_selector.generate_report(output_path=report_path)
        print(f"  特征选择报告已保存至: {report_path}")
        
        # 绘制特征重要性图
        for horizon in self.config['prediction_horizons']:
            # XGBoost特征重要性
            fig = self.factor_selector.plot_feature_importance(horizon=horizon, plot_type='xgb')
            fig_path = os.path.join(self.output_dir, f"xgb_importance_{horizon}h_{self.timestamp}.png")
            fig.savefig(fig_path)
            plt.close(fig)
        
        # 绘制不同时间窗口比较图
        fig = self.factor_selector.plot_horizon_comparison()
        fig_path = os.path.join(self.output_dir, f"horizon_comparison_{self.timestamp}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        
        return selection_results
    
    def run_backtest(self) -> Dict:
        """
        运行回测
        
        返回:
        ----
        回测结果字典
        """
        if not self.config['run_backtest']:
            print("跳过回测阶段")
            return {}
            
        print("\n4. 回测验证...")
        start_time = time.time()
        
        # 如果特征尚未选择，先选择特征
        if not self.selected_features:
            self.select_features()
        
        # 如果特征尚未生成，先生成特征
        if self.features_df is None:
            self.generate_features()
        
        backtest_results = {}
        
        # 对每个预测时间窗口运行回测
        for horizon, features in self.selected_features.items():
            print(f"\n回测 {horizon}小时窗口...")
            target_col = f'future_return_{horizon}h'
            
            # 为选定的特征创建组合因子
            factor_dict = {}
            for feature in features:
                if feature in self.features_df.columns:
                    factor_dict[feature] = self.features_df[feature]
            
            # 组合因子
            combined_factor = self.backtester.combine_factors(factor_dict)
            
            # 评估因子IC
            ic_results = self.backtester.evaluate_factor_ic(
                df=self.features_df,
                factor_values=combined_factor,
                target_col=target_col,
                rolling_window=self.config['rolling_window']
            )
            
            # 运行分组回测
            group_results = self.backtester.run_group_backtest(
                df=self.features_df,
                factor_values=combined_factor,
                target_col=target_col,
                holding_period=horizon
            )
            
            # 合并结果
            results = {
                'factor': combined_factor,
                'ic_results': ic_results,
                'returns': group_results['returns'],
                'performance': group_results['performance'],
                'groups': group_results['groups']
            }
            
            backtest_results[horizon] = results
            
            # 打印回测结果
            print(f"  因子IC: {ic_results['overall_ic']:.4f} (p值: {ic_results['overall_p_value']:.4f})")
            if 'Long_Short' in group_results['performance']:
                ls_perf = group_results['performance']['Long_Short']
                print(f"  多空组合夏普比率: {ls_perf['sharpe_ratio']:.2f}")
                print(f"  多空组合年化收益率: {ls_perf['annualized_return']:.2%}")
            
            # 绘制回测图表
            fig = self.backtester.plot_returns(
                group_results['returns'],
                title=f'{horizon}小时窗口回测结果'
            )
            fig_path = os.path.join(self.output_dir, f"backtest_{horizon}h_{self.timestamp}.png")
            fig.savefig(fig_path)
            plt.close(fig)
            
            # 绘制IC时间序列
            if 'rolling_ic' in ic_results and not ic_results['rolling_ic'].empty:
                fig = self.backtester.plot_ic_timeseries(
                    ic_results['rolling_ic'],
                    title=f'{horizon}小时窗口滚动IC'
                )
                fig_path = os.path.join(self.output_dir, f"rolling_ic_{horizon}h_{self.timestamp}.png")
                fig.savefig(fig_path)
                plt.close(fig)
            
            # 生成回测报告
            report_path = os.path.join(self.output_dir, f"backtest_report_{horizon}h_{self.timestamp}.md")
            self.backtester.generate_report(
                backtest_results=results,
                factor_name=f"组合因子 ({horizon}小时)",
                target_col=target_col,
                output_path=report_path
            )
            print(f"  回测报告已保存至: {report_path}")
        
        # 保存回测结果
        self.backtest_results = backtest_results
        
        # 记录执行时间
        self.execution_time['run_backtest'] = time.time() - start_time
        
        return backtest_results
    
    def run_pipeline(self) -> Dict:
        """
        运行完整流水线
        
        返回:
        ----
        流水线结果字典
        """
        print(f"开始因子研究流水线 - {self.timestamp}")
        total_start_time = time.time()
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 生成特征
        self.generate_features()
        
        # 3. 特征选择
        self.select_features()
        
        # 4. 回测
        self.run_backtest()
        
        # 记录总执行时间
        total_time = time.time() - total_start_time
        self.execution_time['total'] = total_time
        
        # 生成执行时间报告
        print("\n执行时间统计:")
        for stage, seconds in self.execution_time.items():
            print(f"  {stage}: {seconds:.2f}秒 ({seconds/60:.2f}分钟)")
        
        # 保存执行时间报告
        time_report_path = os.path.join(self.output_dir, f"execution_time_{self.timestamp}.txt")
        with open(time_report_path, 'w') as f:
            f.write(f"因子研究流水线执行时间报告 - {self.timestamp}\n\n")
            for stage, seconds in self.execution_time.items():
                f.write(f"{stage}: {seconds:.2f}秒 ({seconds/60:.2f}分钟)\n")
        
        # 生成总体结果摘要
        summary_path = os.path.join(self.output_dir, f"results_summary_{self.timestamp}.md")
        self._generate_summary_report(summary_path)
        
        print(f"\n流水线完成! 总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"结果已保存至: {self.output_dir}")
        
        return {
            'data': self.df,
            'features': self.features_df,
            'selected_features': self.selected_features,
            'backtest_results': self.backtest_results,
            'execution_time': self.execution_time
        }
    
    def _generate_summary_report(self, output_path: str) -> str:
        """
        生成总体结果摘要报告
        
        参数:
        ----
        output_path: 输出路径
        
        返回:
        ----
        报告文本
        """
        report = []
        report.append(f"# 因子研究结果摘要 - {self.timestamp}")
        report.append("\n## 1. 数据概览")
        report.append(f"- 数据形状: {self.df.shape}")
        report.append(f"- 时间范围: {self.df.index.min()} - {self.df.index.max()}")
        
        report.append("\n## 2. 特征工程")
        feature_cols = [col for col in self.features_df.columns if not col.startswith('future_')]
        report.append(f"- 生成特征数量: {len(feature_cols)}")
        
        report.append("\n## 3. 特征选择结果")
        for horizon, features in self.selected_features.items():
            report.append(f"\n### {horizon}小时窗口")
            report.append(f"- 选择特征数量: {len(features)}")
            report.append("- 前10个特征:")
            for i, feature in enumerate(features[:10]):
                report.append(f"  {i+1}. {feature}")
        
        report.append("\n## 4. 回测性能")
        for horizon, results in self.backtest_results.items():
            report.append(f"\n### {horizon}小时窗口")
            
            # IC统计
            ic_results = results.get('ic_results', {})
            report.append(f"- 因子IC: {ic_results.get('overall_ic', 'N/A')}")
            
            # 多空组合性能
            if 'Long_Short' in results.get('performance', {}):
                ls_perf = results['performance']['Long_Short']
                report.append(f"- 多空组合夏普比率: {ls_perf['sharpe_ratio']:.2f}")
                report.append(f"- 多空组合年化收益率: {ls_perf['annualized_return']:.2%}")
                report.append(f"- 多空组合最大回撤: {ls_perf['max_drawdown']:.2%}")
        
        report.append("\n## 5. 执行时间")
        for stage, seconds in self.execution_time.items():
            report.append(f"- {stage}: {seconds:.2f}秒 ({seconds/60:.2f}分钟)")
        
        # 保存报告
        report_text = "\n".join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        return report_text 