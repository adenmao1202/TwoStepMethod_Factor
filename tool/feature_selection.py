import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class FactorSelector:
    """基于Lasso和XGBoost的两阶段因子筛选框架"""
    
    def __init__(self, 
                 prediction_horizons: List[int] = [1, 4, 24],
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        初始化因子选择器
        
        参数:
        ----
        prediction_horizons: 预测周期列表（小时）
        cv_folds: 时间序列交叉验证折数
        random_state: 随机种子
        """
        self.prediction_horizons = prediction_horizons
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.lasso_results = {}
        self.xgb_results = {}
        self.ensemble_results = {}
        self.tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    def prepare_target(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """为特定预测周期准备目标变量（未来N小时收益率）"""
        # 假设df中价格列为'close'
        if 'future_return_' + str(horizon) in df.columns:
            # 已有目标列
            return df['future_return_' + str(horizon)]
        else:
            # 计算未来收益率
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            return future_return
    
    def run_lasso_selection(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           alphas: List[float] = None,
                           max_iter: int = 10000,
                           normalize: bool = True) -> Dict:
        """
        执行Lasso线性模型的因子筛选
        
        参数:
        ----
        X: 特征矩阵
        y: 目标变量
        alphas: 正则化系数列表
        max_iter: 最大迭代次数
        normalize: 是否标准化特征
        
        返回:
        ----
        包含Lasso选择结果的字典
        """
        # 设置默认的alpha搜索范围
        if alphas is None:
            alphas = np.logspace(-6, 0, 100)
        
        # 训练LassoCV模型
        lasso_model = LassoCV(
            cv=self.tscv,
            alphas=alphas,
            max_iter=max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 移除缺失值
        mask = ~pd.isna(y)
        X_valid = X[mask].copy()
        y_valid = y[mask].copy()
        
        # 转换类别变量（如果有）
        X_processed = pd.get_dummies(X_valid, drop_first=True)
        
        # 训练模型
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_model.fit(X_processed, y_valid)
        
        # 获取最佳alpha和系数
        best_alpha = lasso_model.alpha_
        coef = pd.Series(lasso_model.coef_, index=X_processed.columns)
        
        # 获取非零系数（筛选出的特征）
        nonzero_coef = coef[coef != 0].sort_values(key=abs, ascending=False)
        
        # 计算训练和交叉验证的MAE
        y_pred = lasso_model.predict(X_processed)
        train_mae = mean_absolute_error(y_valid, y_pred)
        
        # 准备交叉验证MAE
        cv_scores = []
        for train_idx, test_idx in self.tscv.split(X_processed):
            X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
            y_train, y_test = y_valid.iloc[train_idx], y_valid.iloc[test_idx]
            
            model = lasso_model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores.append(mae)
        
        # 返回结果
        return {
            'model': lasso_model,
            'best_alpha': best_alpha,
            'coefficients': coef,
            'selected_features': nonzero_coef,
            'train_mae': train_mae,
            'cv_mae': np.mean(cv_scores),
            'cv_mae_std': np.std(cv_scores)
        }
    
    def run_xgboost_selection(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             preselected_features: List[str] = None,
                             params: Dict = None) -> Dict:
        """
        执行XGBoost树模型的因子筛选
        
        参数:
        ----
        X: 特征矩阵
        y: 目标变量
        preselected_features: 预先选中的特征列表（来自Lasso）
        params: XGBoost参数字典
        
        返回:
        ----
        包含XGBoost选择结果的字典
        """
        # 设置默认XGBoost参数
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_state
            }
        
        # 使用预选特征（如果有）
        if preselected_features is not None and len(preselected_features) > 0:
            X_selected = X[preselected_features].copy()
        else:
            X_selected = X.copy()
        
        # 移除缺失值
        mask = ~pd.isna(y)
        X_valid = X_selected[mask].copy()
        y_valid = y[mask].copy()
        
        # 转换类别变量（如果有）
        X_processed = pd.get_dummies(X_valid, drop_first=True)
        
        # 训练XGBoost模型
        dtrain = xgb.DMatrix(X_processed, label=y_valid)
        
        # 准备交叉验证结果
        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            folds=self.tscv,
            metrics='mae',
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # 获取最佳迭代次数
        best_rounds = cv_results.shape[0]
        cv_mae = cv_results.iloc[-1]['test-mae-mean']
        cv_mae_std = cv_results.iloc[-1]['test-mae-std']
        
        # 使用全部数据训练最终模型
        final_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=best_rounds
        )
        
        # 计算特征重要性
        importance_gain = pd.Series(final_model.get_score(importance_type='gain'), 
                                    name='gain').sort_values(ascending=False)
        importance_weight = pd.Series(final_model.get_score(importance_type='weight'), 
                                      name='weight').sort_values(ascending=False)
        importance_cover = pd.Series(final_model.get_score(importance_type='cover'), 
                                     name='cover').sort_values(ascending=False)
        
        # 计算排列重要性
        perm_importance = permutation_importance(
            estimator=lambda X: final_model.predict(xgb.DMatrix(X)),
            X=X_processed.values,
            y=y_valid.values,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        perm_imp = pd.Series(
            perm_importance.importances_mean,
            index=X_processed.columns,
            name='permutation'
        ).sort_values(ascending=False)
        
        # 合并所有重要性指标
        all_importance = pd.DataFrame({
            'gain': importance_gain,
            'weight': importance_weight,
            'cover': importance_cover,
            'permutation': perm_imp
        })
        
        # 标准化各列并计算综合得分
        for col in all_importance.columns:
            if all_importance[col].sum() > 0:
                all_importance[col] = all_importance[col] / all_importance[col].sum()
        
        # 填充缺失值为0
        all_importance = all_importance.fillna(0)
        
        # 计算综合得分 (加权平均)
        weights = {'gain': 0.4, 'weight': 0.1, 'cover': 0.1, 'permutation': 0.4}
        all_importance['composite_score'] = sum(all_importance[col] * weight 
                                               for col, weight in weights.items())
        
        # 按综合得分排序
        all_importance = all_importance.sort_values('composite_score', ascending=False)
        
        # 返回结果
        return {
            'model': final_model,
            'feature_importance': all_importance,
            'best_iterations': best_rounds,
            'cv_mae': cv_mae,
            'cv_mae_std': cv_mae_std,
            'importance_gain': importance_gain,
            'importance_weight': importance_weight,
            'importance_cover': importance_cover,
            'importance_permutation': perm_imp
        }
    
    def run_selection_pipeline(self, 
                              df: pd.DataFrame, 
                              features: List[str],
                              top_n_lasso: int = None,
                              top_n_final: int = 10) -> Dict:
        """
        运行完整的两阶段选择流程
        
        参数:
        ----
        df: 数据框，包含特征和价格数据
        features: 待选特征列表
        top_n_lasso: Lasso阶段选择的特征数量，None表示自动选择
        top_n_final: 最终选择的特征数量
        
        返回:
        ----
        包含选择结果的字典
        """
        results = {}
        
        # 确保特征在数据框中
        valid_features = [f for f in features if f in df.columns]
        if len(valid_features) == 0:
            raise ValueError("没有有效的特征列")
        
        X = df[valid_features]
        
        # 对每个预测周期执行选择
        for horizon in self.prediction_horizons:
            print(f"\n执行预测周期 {horizon} 小时的特征选择:")
            
            # 准备目标变量
            y = self.prepare_target(df, horizon)
            
            # 第一阶段: Lasso选择
            print(f"阶段1: 运行Lasso线性模型筛选...")
            lasso_result = self.run_lasso_selection(X, y)
            
            # 显示Lasso结果摘要
            selected_count = (lasso_result['coefficients'] != 0).sum()
            print(f"  - Lasso选择了 {selected_count} 个特征 (最佳alpha: {lasso_result['best_alpha']:.6f})")
            print(f"  - 交叉验证MAE: {lasso_result['cv_mae']:.6f} ±{lasso_result['cv_mae_std']:.6f}")
            
            # 确定要传给XGBoost的特征数量
            if top_n_lasso is not None:
                n_features = min(top_n_lasso, len(lasso_result['selected_features']))
            else:
                # 自动选择特征数量
                n_features = max(min(len(lasso_result['selected_features']), len(X.columns) // 3), 10)
            
            selected_features = lasso_result['selected_features'].index[:n_features].tolist()
            
            # 第二阶段: XGBoost选择
            print(f"阶段2: 运行XGBoost树模型筛选 (使用 {len(selected_features)} 个Lasso预选特征)...")
            xgb_result = self.run_xgboost_selection(X, y, selected_features)
            
            # 显示XGBoost结果摘要
            print(f"  - XGBoost最佳迭代次数: {xgb_result['best_iterations']}")
            print(f"  - 交叉验证MAE: {xgb_result['cv_mae']:.6f} ±{xgb_result['cv_mae_std']:.6f}")
            
            # 集成评估: 合并Lasso和XGBoost结果
            print("执行集成评估...")
            
            # 准备最终排名
            final_ranking = self._calculate_ensemble_ranking(lasso_result, xgb_result)
            
            # 选择最终特征
            final_features = final_ranking.index[:top_n_final].tolist()
            
            # 保存结果
            self.lasso_results[horizon] = lasso_result
            self.xgb_results[horizon] = xgb_result
            self.ensemble_results[horizon] = {
                'final_ranking': final_ranking,
                'selected_features': final_features,
                'lasso_cv_mae': lasso_result['cv_mae'],
                'xgb_cv_mae': xgb_result['cv_mae']
            }
            
            results[horizon] = {
                'selected_features': final_features,
                'feature_ranking': final_ranking
            }
            
            print(f"选择的前 {top_n_final} 个特征:")
            for i, feature in enumerate(final_features, 1):
                score = final_ranking.loc[feature]
                print(f"  {i}. {feature} (得分: {score:.4f})")
        
        return results
    
    def _calculate_ensemble_ranking(self, lasso_result: Dict, xgb_result: Dict) -> pd.Series:
        """计算集成排名"""
        # 从Lasso获取特征系数
        lasso_coef = lasso_result['coefficients'].copy()
        # 标准化系数
        if not (lasso_coef == 0).all():
            lasso_coef = lasso_coef.abs() / lasso_coef.abs().sum()
        
        # 从XGBoost获取特征重要性
        xgb_importance = xgb_result['feature_importance']['composite_score'].copy()
        
        # 合并结果
        all_features = set(lasso_coef.index) | set(xgb_importance.index)
        ensemble_scores = pd.Series(0, index=all_features)
        
        # 为Lasso系数分配40%权重
        for feature in lasso_coef.index:
            if feature in ensemble_scores.index:
                ensemble_scores[feature] += 0.4 * lasso_coef.get(feature, 0)
        
        # 为XGBoost重要性分配60%权重
        for feature in xgb_importance.index:
            if feature in ensemble_scores.index:
                ensemble_scores[feature] += 0.6 * xgb_importance.get(feature, 0)
        
        # 按综合得分排序
        return ensemble_scores.sort_values(ascending=False)
    
    def plot_feature_importance(self, horizon: int = None, top_n: int = 20):
        """绘制特征重要性图"""
        if horizon is None:
            horizon = self.prediction_horizons[0]
        
        if horizon not in self.ensemble_results:
            raise ValueError(f"没有 {horizon} 小时周期的结果。请先运行选择流程。")
        
        # 获取结果
        final_ranking = self.ensemble_results[horizon]['final_ranking']
        
        # 选择前N个特征
        top_features = final_ranking.nlargest(top_n)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        ax = top_features.sort_values().plot(kind='barh')
        
        # 添加标签和标题
        plt.xlabel('重要性得分')
        plt.ylabel('特征名称')
        plt.title(f'前 {top_n} 个特征重要性 (预测周期: {horizon}小时)')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # 添加数值标签
        for i, v in enumerate(top_features.sort_values()):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_horizon_comparison(self, top_n: int = 10):
        """比较不同预测周期的特征重要性"""
        if not self.ensemble_results:
            raise ValueError("没有结果。请先运行选择流程。")
        
        # 收集每个周期的前N个特征
        all_top_features = set()
        for horizon in self.prediction_horizons:
            if horizon in self.ensemble_results:
                top_features = self.ensemble_results[horizon]['final_ranking'].nlargest(top_n).index
                all_top_features.update(top_features)
        
        # 创建比较dataframe
        comparison_df = pd.DataFrame(index=sorted(all_top_features))
        
        # 填充每个周期的重要性得分
        for horizon in self.prediction_horizons:
            if horizon in self.ensemble_results:
                ranking = self.ensemble_results[horizon]['final_ranking']
                comparison_df[f'{horizon}h'] = ranking.reindex(comparison_df.index).fillna(0)
        
        # 标准化每列
        for col in comparison_df.columns:
            if comparison_df[col].sum() > 0:
                comparison_df[col] = comparison_df[col] / comparison_df[col].sum()
        
        # 按总重要性排序
        comparison_df['总分'] = comparison_df.sum(axis=1)
        comparison_df = comparison_df.sort_values('总分', ascending=False).drop('总分', axis=1)
        
        # 绘制热图
        plt.figure(figsize=(10, 12))
        sns.heatmap(comparison_df, cmap='YlGnBu', annot=True, fmt='.3f')
        plt.title('不同预测周期的特征重要性比较')
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_report(self, output_path: str = None):
        """生成特征选择结果报告"""
        if not self.ensemble_results:
            raise ValueError("没有结果。请先运行选择流程。")
        
        report = []
        report.append("# 因子选择结果报告")
        report.append("\n## 1. 概述")
        report.append(f"- 评估了 {len(self.prediction_horizons)} 个预测周期: {', '.join(map(str, self.prediction_horizons))}小时")
        report.append(f"- 使用了 {self.cv_folds} 折时间序列交叉验证")
        report.append(f"- 采用Lasso线性模型和XGBoost树模型两阶段选择")
        
        # 添加每个周期的结果
        report.append("\n## 2. 各预测周期结果")
        
        for horizon in self.prediction_horizons:
            if horizon in self.ensemble_results:
                result = self.ensemble_results[horizon]
                report.append(f"\n### 2.{horizon}. {horizon}小时预测周期")
                
                # Lasso结果
                lasso_result = self.lasso_results[horizon]
                report.append("\n#### Lasso线性模型结果")
                report.append(f"- 最佳alpha: {lasso_result['best_alpha']:.6f}")
                report.append(f"- 交叉验证MAE: {lasso_result['cv_mae']:.6f} ±{lasso_result['cv_mae_std']:.6f}")
                nonzero_count = (lasso_result['coefficients'] != 0).sum()
                report.append(f"- 选择了 {nonzero_count} 个非零系数特征")
                
                report.append("\n前10个非零系数特征:")
                for i, (feature, coef) in enumerate(lasso_result['selected_features'].items()[:10], 1):
                    report.append(f"  {i}. {feature}: {coef:.6f}")
                
                # XGBoost结果
                xgb_result = self.xgb_results[horizon]
                report.append("\n#### XGBoost树模型结果")
                report.append(f"- 最佳迭代次数: {xgb_result['best_iterations']}")
                report.append(f"- 交叉验证MAE: {xgb_result['cv_mae']:.6f} ±{xgb_result['cv_mae_std']:.6f}")
                
                report.append("\n前10个综合重要性特征:")
                top_features = xgb_result['feature_importance'].head(10)
                for i, (feature, row) in enumerate(top_features.iterrows(), 1):
                    report.append(f"  {i}. {feature}: {row['composite_score']:.6f}")
                
                # 集成结果
                report.append("\n#### 集成评估结果")
                final_features = result['selected_features']
                report.append(f"- 最终选择 {len(final_features)} 个特征")
                
                report.append("\n最终选择的特征:")
                final_ranking = result['final_ranking']
                for i, feature in enumerate(final_features, 1):
                    score = final_ranking.loc[feature]
                    report.append(f"  {i}. {feature}: {score:.6f}")
        
        # 添加跨周期比较
        report.append("\n## 3. 预测周期比较")
        report.append("\n不同预测周期共有的重要特征:")
        
        # 找出在所有周期中都重要的特征
        common_features = set()
        for horizon in self.prediction_horizons:
            if horizon in self.ensemble_results:
                if not common_features:
                    common_features = set(self.ensemble_results[horizon]['selected_features'])
                else:
                    common_features &= set(self.ensemble_results[horizon]['selected_features'])
        
        for feature in sorted(common_features):
            report.append(f"- {feature}")
        
        # 合并报告
        report_text = "\n".join(report)
        
        # 保存或返回报告
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"报告已保存至 {output_path}")
        
        return report_text 