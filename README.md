# Project Pipeline

## 1. 數據預處理階段

### 資料清理
- 處理 missing values 通過**前向填充 (Forward-Fill)** 方式
- 使用 **Robust Scaling** 方法標準化數據
- **Winsorizing** 處理極端值，將超出指定百分位的數據縮減到閾值範圍內

### 品質檢查
顯示數據形狀、缺失值比例、時間範圍等基本統計信息

### 資料視覺化
生成價格和交易量的時間序列圖表，直觀展示數據特性

---

## 2. 因子生成階段

系統性生成多種類型的因子，每種因子使用不同的時間週期參數，包括：

### 價格相關因子
- **Price-to-VWAP 因子**：比較當前價格與加權平均價格的關係

### 技術指標因子
- **Momentum 因子** (8小時、12小時、24小時)：捕捉價格趨勢動量
- **RSI 因子** (8小時、12小時、24小時)：測量超買超賣狀態
- **MACD 因子**：包含 MACD Line、Signal Line 和 Histogram
- **Volatility 因子** (8小時、12小時、24小時)：衡量價格波動程度

### 交易量相關因子
- **Volume 因子** (8小時、12小時、24小時)：分析交易量變化
- **Volume Divergence 因子**：檢測價格與交易量的背離情況

### 市場微結構因子
- **Buy-Sell Pressure 因子**：分析買賣壓力不平衡
- **Trade Activity 因子**：衡量市場活躍度
- **Price Impact 因子**：評估訂單對價格的影響程度
- **Liquidity Imbalance 因子**：檢測買賣盤流動性不平衡
- **Trading Efficiency 因子**：測量市場交易效率
- **VPIN 因子**：估計知情交易概率
- **High-Low Volume Position 因子**：分析高低點交易量分布
- **Trade Size Change 因子**：監測交易規模變化

### 複合因子
- **Market Balance 因子**：綜合評估市場買賣力量平衡
- **Smart Momentum 因子**：結合價格動量和交易量權重
- **Trading Quality 因子**：綜合評估交易質量

---

## 3. 因子處理與特徵工程

### 特徵處理
使用 `FeatureEngineer` 類對原始因子進行處理：
- 處理極端值 **(Winsorization)**
- **標準化 (Robust Scaling)**：使用median替代mean，對異常值較不敏感
- **缺失值填充 (Forward-Fill)**
- 生成多個預測時間範圍的目標變量 (`future_return_Xh`)

### 特徵可視化
- 生成因子**相關性熱圖**，分析因子間關係
- 繪製因子與**未來收益率的散點圖**
- 顯示**因子分布圖**，檢查偏斜度和異常值

---

## 4. 因子選擇階段

使用 `FactorSelector` 類實現**兩階段式特徵選擇流程**：
> **注意**：所有CV都有基於**Time Series Split**進行交叉處理

### 第一階段 (Lasso 選擇)
- 針對每個預測時間範圍 **(1小時、4小時、24小時)** 分別使用 `LassoCV` 算法
- 根據 **L1 正則化係數**選出最重要的前 **N 個特徵** (N設為15個)
- 輸出每個時間範圍的 **Lasso 特徵重要性排序**

### 第二階段 (XGBoost 選擇)
- 使用第一階段選出的特徵，應用 **XGBoost** 進行進一步篩選
- 利用多種重要性指標：
  - `gain`
  - `weight`
  - `cover`
  - `total_gain`
  - `total_cover`
- **綜合評分**選出最終的前 **M 個特徵** (預設為5個)
- 輸出每個時間範圍的 **XGBoost 特徵重要性排序**和視覺化結果

### 特徵重要性
生成不同預測時間範圍的**特徵重要性比較圖**，分析哪些因子在不同時間範圍內更重要

---

## 5. 回測評估階段

使用 `FactorBacktester` 類進行回測：

### IC (Information Coefficient) 分析
- 計算因子與未來收益率的 **Spearman Rank 相關係數**
- 生成**滾動 IC 時間序列圖**，評估因子預測能力隨時間變化
- 計算 **IC 統計指標**：
  - IC Mean
  - IC t-stat
- 比較**IC和ML方法選擇的重疊性**

### 分組回測
- 根據因子值將資產分為 **N 組** (N 設為5組)
- 計算每組的**累積收益率**
- 構建 **Long-Short 策略** (買入最高分組、賣空最低分組)
- 計算**策略表現指標**：
  - 年化收益率
  - Sharpe 比率
  - 最大回撤等

> **框架特色**：由於不想使用現有因子回測框架如alphalens等，因此自建了一個完整框架

---

## 框架結構簡介

### 1. FactorResearchPipeline 主類
**功能**：整合整個研究流程，從數據加載到最終報告生成

**主要組件**：
- `load_data()`: 數據加載與預處理
- `generate_features()`: 特徵生成
- `select_features()`: 特徵選擇
- `run_backtest()`: 回測評估
- `run_pipeline()`: 執行完整流程

### 2. FeatureEngineer 特徵工程類
**功能**：生成和處理各種因子

**主要方法**：
- `create_all_features()`: 生成所有特徵
- `process_features()`: 處理和標準化特徵
- 各種因子計算方法 (`price_factors`, `momentum_factors` 等)

### 3. FactorSelector 因子選擇類
**功能**：執行兩階段特徵選擇流程

**主要方法**：
- `run_selection_pipeline()`: 執行完整選擇流程
- `run_lasso_selection()`: 執行 Lasso 特徵選擇
- `run_xgboost_selection()`: 執行 XGBoost 特徵選擇
- `plot_feature_importance()`: 視覺化特徵重要性

### 4. FactorBacktester 回測類
**功能**：評估選定因子的預測能力和實際表現

**主要方法**：
- `evaluate_factor_ic()`: 計算因子信息系數
- `run_group_backtest()`: 執行分組回測
- `calculate_performance()`: 計算績效指標
- `generate_report()`: 生成回測報告
