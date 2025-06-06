#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试两阶段特征选择模块
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import time

from feature_selection import LassoFeatureSelector, XGBoostFeatureSelector, TwoStageFeatureSelector

def create_synthetic_data(n_samples=1000, n_features=50, n_informative=10, random_state=42):
    """
    创建合成数据用于测试
    
    参数:
    ----
    n_samples: 样本数量
    n_features: 特征总数
    n_informative: 有信息量的特征数
    random_state: 随机种子
    
    返回:
    ----
    X: 特征矩阵
    y: 目标变量
    """
    print("生成合成数据中...")
    np.random.seed(random_state)
    
    # 创建时间索引
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_samples)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # 创建特征矩阵
    X = pd.DataFrame(np.random.randn(n_samples, n_features), index=dates)
    
    # 命名特征
    feature_names = []
    for i in tqdm(range(n_features), desc="生成特征名称"):
        if i < n_informative:
            feature_names.append(f"important_feature_{i+1}")
        else:
            feature_names.append(f"noise_feature_{i-n_informative+1}")
    
    X.columns = feature_names
    
    # 创建目标变量 - 只与重要特征相关
    y = pd.Series(0, index=dates)
    
    for i in tqdm(range(n_informative), desc="生成目标变量"):
        # 添加线性关系
        if i < n_informative // 2:
            coef = np.random.uniform(0.5, 2.0)
            y += coef * X[f"important_feature_{i+1}"]
        # 添加非线性关系
        else:
            y += np.sin(X[f"important_feature_{i+1}"]) + X[f"important_feature_{i+1}"]**2 / 5
    
    # 添加噪声
    y += np.random.randn(n_samples) * 0.5
    print("合成数据生成完成！")
    
    return X, y

def test_lasso_selector():
    """测试Lasso特征选择器"""
    print("\n===== 测试 LassoFeatureSelector =====")
    start_time = time.time()
    
    # 创建合成数据
    X, y = create_synthetic_data(n_samples=500, n_features=50, n_informative=10)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 创建Lasso选择器
    lasso_selector = LassoFeatureSelector(cv_folds=3, random_state=42)
    
    # 拟合模型
    print("拟合Lasso模型中，请稍候...")
    lasso_selector.fit(X, y, top_n=15)
    
    # 获取选择的特征
    selected_features = lasso_selector.selected_features_
    print(f"\nLasso选择的特征 ({len(selected_features)}):")
    for feature in selected_features:
        print(f"   {feature}")
    
    # 获取特征重要性
    importances = lasso_selector.get_support()
    print("\n特征重要性 (前5个):")
    for feature, importance in importances.head(5).items():
        print(f"   {feature}: {importance:.4f}")
    
    # 绘制特征重要性
    print("绘制Lasso特征重要性图...")
    plt.figure(figsize=(10, 6))
    lasso_selector.plot_feature_importance(top_n=10)
    plt.tight_layout()
    
    # 保存结果
    os.makedirs("test_results", exist_ok=True)
    plt.savefig("test_results/lasso_importance.png")
    plt.close()
    
    end_time = time.time()
    print(f"Lasso测试完成，用时: {end_time - start_time:.2f}秒")
    
    return lasso_selector

def test_xgboost_selector():
    """测试XGBoost特征选择器"""
    print("\n===== 测试 XGBoostFeatureSelector =====")
    start_time = time.time()
    
    # 创建合成数据
    X, y = create_synthetic_data(n_samples=500, n_features=50, n_informative=10)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 创建XGBoost选择器
    xgb_selector = XGBoostFeatureSelector(cv_folds=3, random_state=42)
    
    # 拟合模型
    print("拟合XGBoost模型中，请稍候...")
    xgb_selector.fit(X, y, top_n=15)
    
    # 获取选择的特征
    selected_features = xgb_selector.selected_features_
    print(f"\nXGBoost选择的特征 ({len(selected_features)}):")
    for feature in selected_features:
        print(f"   {feature}")
    
    # 获取特征重要性
    importances = xgb_selector.get_support()
    print("\n特征重要性 (前5个):")
    for feature, importance in importances.head(5).items():
        print(f"   {feature}: {importance:.4f}")
    
    # 绘制特征重要性
    print("绘制XGBoost特征重要性图...")
    plt.figure(figsize=(10, 6))
    xgb_selector.plot_feature_importance(top_n=10)
    plt.tight_layout()
    
    # 保存结果
    os.makedirs("test_results", exist_ok=True)
    plt.savefig("test_results/xgb_importance.png")
    plt.close()
    
    # 绘制不同指标的比较
    print("绘制特征指标比较图...")
    plt.figure(figsize=(12, 8))
    xgb_selector.plot_metrics_comparison(top_n=10)
    plt.tight_layout()
    plt.savefig("test_results/xgb_metrics_comparison.png")
    plt.close()
    
    end_time = time.time()
    print(f"XGBoost测试完成，用时: {end_time - start_time:.2f}秒")
    
    return xgb_selector

def test_two_stage_selector():
    """测试两阶段特征选择器"""
    print("\n===== 测试 TwoStageFeatureSelector =====")
    start_time = time.time()
    
    # 创建合成数据
    X, y = create_synthetic_data(n_samples=500, n_features=50, n_informative=10)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 创建两阶段选择器
    two_stage_selector = TwoStageFeatureSelector(cv_folds=3, random_state=42)
    
    # 拟合模型
    print("拟合两阶段选择器中，请稍候...")
    print("第一阶段: Lasso筛选...")
    two_stage_selector.fit(X, y, top_n_lasso=20, top_n_final=10)
    
    # 获取选择的特征
    lasso_selected = two_stage_selector.lasso_selected_features_
    final_selected = two_stage_selector.final_selected_features_
    
    print(f"\nLasso阶段选择的特征 ({len(lasso_selected)}):")
    for feature in lasso_selected:
        print(f"   {feature}")
    
    print(f"\n最终选择的特征 ({len(final_selected)}):")
    for feature in final_selected:
        print(f"   {feature}")
    
    # 绘制特征重要性比较
    print("绘制特征重要性比较图...")
    plt.figure(figsize=(12, 8))
    two_stage_selector.plot_importance_comparison(top_n=10)
    plt.tight_layout()
    
    # 保存结果
    os.makedirs("test_results", exist_ok=True)
    plt.savefig("test_results/two_stage_comparison.png")
    plt.close()
    
    # 绘制Lasso和XGBoost各自的特征重要性
    print("绘制各模型特征重要性图...")
    plt.figure(figsize=(10, 6))
    two_stage_selector.plot_lasso_importance(top_n=10)
    plt.tight_layout()
    plt.savefig("test_results/two_stage_lasso.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    two_stage_selector.plot_xgb_importance(top_n=10)
    plt.tight_layout()
    plt.savefig("test_results/two_stage_xgb.png")
    plt.close()
    
    end_time = time.time()
    print(f"两阶段特征选择测试完成，用时: {end_time - start_time:.2f}秒")
    
    return two_stage_selector

def main():
    """主测试函数"""
    print("开始测试两阶段特征选择模块...")
    total_start_time = time.time()
    
    # 显示测试进度
    test_steps = ["Lasso特征选择", "XGBoost特征选择", "两阶段特征选择"]
    
    # 测试Lasso选择器
    print(f"\n[1/3] 测试 {test_steps[0]}")
    lasso_selector = test_lasso_selector()
    
    # 测试XGBoost选择器
    print(f"\n[2/3] 测试 {test_steps[1]}")
    xgb_selector = test_xgboost_selector()
    
    # 测试两阶段选择器
    print(f"\n[3/3] 测试 {test_steps[2]}")
    two_stage_selector = test_two_stage_selector()
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\n所有测试完成！总用时: {total_time:.2f}秒")
    print(f"结果保存在 test_results 目录")

if __name__ == "__main__":
    main() 