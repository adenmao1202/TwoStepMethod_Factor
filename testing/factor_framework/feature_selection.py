import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class FactorSelector:
    
    def __init__(self,
                prediction_horizons: List[int] = [1, 4, 24],
                cv_folds: int = 5,
                random_state: int = 42):
        
        self.prediction_horizons = prediction_horizons
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        self.feature_importances = {}
        self.lasso_models = {}
        self.xgb_models = {}
        self.tscv = TimeSeriesSplit(n_splits=cv_folds)
        
    # 看 this one !!! 
    def run_selection_pipeline(self,
                             df: pd.DataFrame,
                             features: List[str],
                             top_n_lasso: int = 50,
                             top_n_final: int = 20,
                             lasso_params: Optional[Dict] = None,
                             xgb_params: Optional[Dict] = None) -> Dict:
        """
        Run complete two-stage feature selection pipeline
        
        Parameters:
        ----
        df: DataFrame containing features and target variables
        features: List of feature column names
        top_n_lasso: Number of features to select in first stage
        top_n_final: Number of features to select in final selection
        lasso_params: Lasso model parameters dictionary
        xgb_params: XGBoost model parameters dictionary
        
        Returns:
        ----
        Dictionary containing selection results
        """
        results = {}
        
        # Ensure all necessary target variables exist
        for horizon in self.prediction_horizons:
            target_col = f'future_return_{horizon}h'
            if target_col not in df.columns:
                raise ValueError(f"Target variable column {target_col} does not exist in the DataFrame")
        
        print("Stage 1: Lasso Feature Selection")
        lasso_selected = {}
        
        # Run Lasso selection for each prediction time window
        for horizon in self.prediction_horizons:
            print(f"\nAnalyzing {horizon} hour prediction window...")
            target_col = f'future_return_{horizon}h'
            
            # Run Lasso feature selection
            selected_features, importances = self._run_lasso_selection(
                df=df,
                features=features,
                target_col=target_col,
                top_n=top_n_lasso,
                params=lasso_params
            )
            
            # Save results
            lasso_selected[horizon] = selected_features
            self.feature_importances[f'lasso_{horizon}h'] = importances
            
            # Output results
            print(f"Lasso selected {len(selected_features)} features for {horizon} hour window")
            print("Top 10 important features:")
            for i, (feature, importance) in enumerate(importances.head(10).items()):
                print(f"  {i+1}. {feature}: {importance:.6f}")
        
        
        print("\nStage 2: XGBoost Feature Selection")
        xgb_selected = {}
        final_selected = {}
        
        # Run XGBoost selection for each prediction time window
        for horizon in self.prediction_horizons:
            print(f"\nAnalyzing {horizon} hour prediction window...")
            target_col = f'future_return_{horizon}h'
            
            # Use Lasso filtered features to run XGBoost
            selected_features, importances = self._run_xgboost_selection(
                df=df,
                features=lasso_selected[horizon],
                target_col=target_col,
                top_n=top_n_final,
                params=xgb_params
            )
            
            # Save results
            xgb_selected[horizon] = selected_features
            self.feature_importances[f'xgb_{horizon}h'] = importances
            final_selected[horizon] = selected_features
            
            # Output results
            print(f"XGBoost selected {len(selected_features)} features for {horizon} hour window")
            print("Top 10 important features:")
            for i, (feature, importance) in enumerate(importances.head(10).items()):
                print(f"  {i+1}. {feature}: {importance:.6f}")
        
        # Perform ensemble evaluation
        print("\nEnsemble Evaluation Stage")
        ensemble_results = self._ensemble_evaluation(df, final_selected)
        
        # Results summary
        results = {
            'lasso_selected': lasso_selected,
            'xgb_selected': xgb_selected,
            'final_selected': final_selected,
            'feature_importances': self.feature_importances,
            'ensemble_results': ensemble_results
        }
        
        self.results = results
        return results
    
    # 看 this one !!! 
    def _run_lasso_selection(self,
                           df: pd.DataFrame,
                           features: List[str],
                           target_col: str,
                           top_n: int = 50,
                           params: Optional[Dict] = None) -> Tuple[List[str], pd.Series]:
        """
        Run Lasso feature selection
        
        Parameters:
        ----
        df: DataFrame
        features: List of feature column names
        target_col: Target variable column name
        top_n: Number of features to select
        params: Lasso model parameters
        
        Returns:
        ----
        (List of selected features, Feature importance Series)
        """
        # Prepare data
        X = df[features].copy()
        y = df[target_col].copy()
        
        # Filter out rows with NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Set default parameters
        default_params = {
            'cv': self.tscv,
            'random_state': self.random_state,
            'n_jobs': -1,
            'max_iter': 10000,
            'tol': 1e-4
        }
        
        # Update default parameters
        if params:
            default_params.update(params)
        
        # Create Lasso model
        lasso = LassoCV(**default_params)
        
        # Train model
        lasso.fit(X, y)
        
        # Save model
        horizon = int(target_col.split('_')[-1][:-1])  # Extract time window
        self.lasso_models[horizon] = lasso
        
        # Get feature importance
        importances = pd.Series(np.abs(lasso.coef_), index=features)
        importances = importances.sort_values(ascending=False)
        
        # Select top features
        selected_features = importances.head(top_n).index.tolist()
        
        return selected_features, importances

    # 看 this one !!! 
    def _run_xgboost_selection(self,
                             df: pd.DataFrame,
                             features: List[str],
                             target_col: str,
                             top_n: int = 20,
                             params: Optional[Dict] = None) -> Tuple[List[str], pd.Series]:
        
        # Prepare data
        X = df[features].copy()
        y = df[target_col].copy()
        
        # Filter out rows with NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Set default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state
        }
        
        # Update default parameters
        if params:
            default_params.update(params)
        
        # Train model
        model = xgb.XGBRegressor(**default_params)
        model.fit(X, y)
        
        # Save model
        horizon = int(target_col.split('_')[-1][:-1])  # Extract time window
        self.xgb_models[horizon] = model
        
        # Get feature importance (weight)
        weight_importance = model.get_booster().get_score(importance_type='weight')
        weight_importance = {features[int(k.replace('f', ''))]: v for k, v in weight_importance.items()}
        weight_series = pd.Series(weight_importance).fillna(0)
        
        # Get feature importance (gain)
        gain_importance = model.get_booster().get_score(importance_type='gain')
        gain_importance = {features[int(k.replace('f', ''))]: v for k, v in gain_importance.items()}
        gain_series = pd.Series(gain_importance).fillna(0)
        
        # Get feature importance (permutation)
        try:
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=10, 
                random_state=self.random_state
            )
            perm_series = pd.Series(perm_importance.importances_mean, index=features)
        except Exception as e:
            print(f"Warning: Permutation importance calculation failed: {e}")
            perm_series = pd.Series(0, index=features)
        
        # Create combined importance
        # Normalize each importance measure
        weight_norm = (weight_series - weight_series.min()) / (weight_series.max() - weight_series.min()) if weight_series.max() > weight_series.min() else weight_series
        gain_norm = (gain_series - gain_series.min()) / (gain_series.max() - gain_series.min()) if gain_series.max() > gain_series.min() else gain_series
        perm_norm = (perm_series - perm_series.min()) / (perm_series.max() - perm_series.min()) if perm_series.max() > perm_series.min() else perm_series
        
        # Combine with equal weights
        combined_importance = (weight_norm + gain_norm + perm_norm) / 3
        
        # Sort by combined importance
        combined_importance = combined_importance.sort_values(ascending=False)
        
        # Select top features
        selected_features = combined_importance.head(top_n).index.tolist()
        
        return selected_features, combined_importance

    # 看 this one !!! 
    def _ensemble_evaluation(self,
                           df: pd.DataFrame,
                           selected_features: Dict[int, List[str]]) -> Dict:
        """
        Perform ensemble evaluation of selected features
        
        Parameters:
        ----
        df: DataFrame
        selected_features: Dictionary mapping prediction horizons to selected feature lists
        
        Returns:
        ----
        Dictionary containing evaluation results
        """
        results = {}
        
        # Find common features across different horizons
        all_features = []
        for horizon, features in selected_features.items():
            all_features.extend(features)
        
        # Count feature frequency
        feature_counts = {}
        for feature in all_features:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Evaluate based on feature frequency
        results['feature_frequency'] = {feature: count for feature, count in sorted_features}
        
        # Create unified feature set across horizons (union)
        unified_features = list(set(all_features))
        results['unified_features'] = unified_features
        
        # Evaluate based on feature interactions
        interaction_scores = {}
        
        # If we have enough features, analyze pairwise correlations
        if len(unified_features) > 1:
            feature_data = df[unified_features].copy()
            feature_data = feature_data.dropna()
            
            if len(feature_data) > 10:
                # Calculate correlation matrix
                corr_matrix = feature_data.corr()
                
                # Find highly correlated pairs
                high_corr_pairs = []
                for i in range(len(unified_features)):
                    for j in range(i+1, len(unified_features)):
                        f1, f2 = unified_features[i], unified_features[j]
                        corr = corr_matrix.loc[f1, f2]
                        if abs(corr) > 0.7:  # Correlation threshold
                            high_corr_pairs.append((f1, f2, corr))
                
                # Sort by absolute correlation
                high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Store results
                results['high_correlation_pairs'] = high_corr_pairs
        
        # Recommend final feature set
        if sorted_features:
            # Option 1: Take most frequent features
            top_frequent = [feature for feature, _ in sorted_features[:min(10, len(sorted_features))]]
            
            # Option 2: Take highest importance from XGBoost
            top_importance = []
            for horizon in self.prediction_horizons:
                if horizon in selected_features:
                    top_importance.extend(selected_features[horizon][:3])  # Take top 3 from each horizon
            top_importance = list(dict.fromkeys(top_importance))  # Remove duplicates while preserving order
            
            # Option 3: Balance frequency and importance
            # Filter to features that appear in at least 2 horizons
            frequent_features = [feature for feature, count in sorted_features if count > 1]
            
            results['recommendations'] = {
                'by_frequency': top_frequent,
                'by_importance': top_importance,
                'balanced': frequent_features
            }
        
        return results
    
    def plot_feature_importance(self,
                              horizon: int,
                              top_n: int = 20,
                              plot_type: str = 'xgb',  # 'lasso', 'xgb', 'both'
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'lasso' or plot_type == 'both':
            # Get Lasso importances
            lasso_key = f'lasso_{horizon}h'
            if lasso_key in self.feature_importances:
                lasso_imp = self.feature_importances[lasso_key].head(top_n)
                if plot_type == 'lasso':
                    lasso_imp.sort_values().plot(kind='barh', ax=ax)
                    ax.set_title(f'Lasso Feature Importance (Horizon: {horizon}h)')
                else:
                    # Store for later use in 'both' plot
                    lasso_data = lasso_imp
            else:
                print(f"No Lasso importance data for horizon {horizon}h")
                lasso_data = pd.Series()
        
        if plot_type == 'xgb' or plot_type == 'both':
            # Get XGBoost importances
            xgb_key = f'xgb_{horizon}h'
            if xgb_key in self.feature_importances:
                xgb_imp = self.feature_importances[xgb_key].head(top_n)
                if plot_type == 'xgb':
                    xgb_imp.sort_values().plot(kind='barh', ax=ax)
                    ax.set_title(f'XGBoost Feature Importance (Horizon: {horizon}h)')
                else:
                    # Store for later use in 'both' plot
                    xgb_data = xgb_imp
            else:
                print(f"No XGBoost importance data for horizon {horizon}h")
                xgb_data = pd.Series()
        
        if plot_type == 'both':
            # Plot both importances on the same plot
            if not lasso_data.empty and not xgb_data.empty:
                # Use top features from either method
                all_features = set(lasso_data.index) | set(xgb_data.index)
                top_features = sorted(all_features, key=lambda x: 
                                     (lasso_data.get(x, 0) + xgb_data.get(x, 0)), 
                                     reverse=True)[:top_n]
                
                # Prepare data for plotting
                plot_data = pd.DataFrame(0, index=top_features, columns=['Lasso', 'XGBoost'])
                for feature in top_features:
                    if feature in lasso_data.index:
                        plot_data.loc[feature, 'Lasso'] = lasso_data[feature]
                    if feature in xgb_data.index:
                        plot_data.loc[feature, 'XGBoost'] = xgb_data[feature]
                
                # Normalize
                for col in plot_data.columns:
                    if plot_data[col].max() > 0:
                        plot_data[col] = plot_data[col] / plot_data[col].max()
                
                # Plot
                plot_data.sort_values('XGBoost').plot(kind='barh', ax=ax)
                ax.set_title(f'Feature Importance Comparison (Horizon: {horizon}h)')
                ax.set_xlabel('Normalized Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_horizon_comparison(self,
                              top_n: int = 15,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        
        fig, axes = plt.subplots(1, len(self.prediction_horizons), figsize=figsize, sharey=True)
        
        # If only one horizon, convert axes to list
        if len(self.prediction_horizons) == 1:
            axes = [axes]
        
        # Collect all important features across horizons
        all_important = []
        for i, horizon in enumerate(self.prediction_horizons):
            xgb_key = f'xgb_{horizon}h'
            if xgb_key in self.feature_importances:
                top_features = self.feature_importances[xgb_key].head(top_n).index.tolist()
                all_important.extend(top_features)
        
        # Get unique features
        unique_features = list(set(all_important))
        
        # Plot for each horizon
        for i, horizon in enumerate(self.prediction_horizons):
            xgb_key = f'xgb_{horizon}h'
            if xgb_key in self.feature_importances:
                # Get importance values for all features
                imp_values = []
                for feature in unique_features:
                    if feature in self.feature_importances[xgb_key]:
                        imp_values.append(self.feature_importances[xgb_key][feature])
                    else:
                        imp_values.append(0)
                
                # Plot
                subset_df = pd.DataFrame({
                    'feature': unique_features,
                    'importance': imp_values
                })
                subset_df = subset_df.sort_values('importance', ascending=False).head(top_n)
                subset_df.plot(x='feature', y='importance', kind='bar', ax=axes[i], legend=False)
                axes[i].set_title(f'Horizon: {horizon}h')
                axes[i].set_xlabel('')
                axes[i].tick_params(axis='x', rotation=90)
        
        fig.suptitle('Feature Importance by Prediction Horizon', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        return fig
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        
        if not self.results:
            return "No results available. Run selection_pipeline first."
        
        report = []
        report.append("# Feature Selection Report")
        report.append("\n## 1. Overview")
        report.append(f"- Number of prediction horizons analyzed: {len(self.prediction_horizons)}")
        report.append(f"- Prediction horizons: {', '.join(map(str, self.prediction_horizons))}")
        
        # Summary of features selected
        report.append("\n## 2. Feature Selection Summary")
        for horizon in self.prediction_horizons:
            report.append(f"\n### Horizon: {horizon}h")
            
            # Lasso selection
            if horizon in self.results['lasso_selected']:
                lasso_features = self.results['lasso_selected'][horizon]
                report.append(f"- Lasso selected {len(lasso_features)} features")
                if lasso_features:
                    report.append("- Top 10 Lasso features:")
                    lasso_imp = self.feature_importances.get(f'lasso_{horizon}h', pd.Series())
                    if not lasso_imp.empty:
                        for i, (feature, imp) in enumerate(lasso_imp.head(10).items()):
                            report.append(f"  {i+1}. {feature}: {imp:.6f}")
            
            # XGBoost selection
            if horizon in self.results['xgb_selected']:
                xgb_features = self.results['xgb_selected'][horizon]
                report.append(f"- XGBoost selected {len(xgb_features)} features")
                if xgb_features:
                    report.append("- Top 10 XGBoost features:")
                    xgb_imp = self.feature_importances.get(f'xgb_{horizon}h', pd.Series())
                    if not xgb_imp.empty:
                        for i, (feature, imp) in enumerate(xgb_imp.head(10).items()):
                            report.append(f"  {i+1}. {feature}: {imp:.6f}")
        
        # Feature frequency analysis
        report.append("\n## 3. Feature Frequency Analysis")
        if 'ensemble_results' in self.results and 'feature_frequency' in self.results['ensemble_results']:
            freq = self.results['ensemble_results']['feature_frequency']
            report.append("- Features sorted by frequency across horizons:")
            for i, (feature, count) in enumerate(freq.items()):
                if i < 20:  # Show top 20
                    report.append(f"  {i+1}. {feature}: appears in {count} horizons")
        
        # Correlation analysis
        report.append("\n## 4. Feature Correlation Analysis")
        if ('ensemble_results' in self.results and 
            'high_correlation_pairs' in self.results['ensemble_results']):
            corr_pairs = self.results['ensemble_results']['high_correlation_pairs']
            if corr_pairs:
                report.append("- Highly correlated feature pairs:")
                for i, (f1, f2, corr) in enumerate(corr_pairs[:10]):  # Show top 10
                    report.append(f"  {i+1}. {f1} & {f2}: correlation = {corr:.4f}")
            else:
                report.append("- No highly correlated feature pairs found")
        
        # Recommendations
        report.append("\n## 5. Recommendations")
        if ('ensemble_results' in self.results and 
            'recommendations' in self.results['ensemble_results']):
            recs = self.results['ensemble_results']['recommendations']
            
            report.append("### By Frequency:")
            if 'by_frequency' in recs:
                for feature in recs['by_frequency']:
                    report.append(f"- {feature}")
            
            report.append("\n### By Importance:")
            if 'by_importance' in recs:
                for feature in recs['by_importance']:
                    report.append(f"- {feature}")
            
            report.append("\n### Balanced Approach (Frequency + Importance):")
            if 'balanced' in recs:
                for feature in recs['balanced']:
                    report.append(f"- {feature}")
        
        # Format as string
        report_str = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            return f"Report saved to {output_path}"
        
        return report_str

class LassoFeatureSelector:
    
    
    def __init__(self, 
                cv_folds: int = 5, 
                random_state: int = 42,
                max_iter: int = 10000,
                tol: float = 1e-4,
                n_jobs: int = -1):
        
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.model = None
        self.feature_names = None
        self.feature_importances_ = None
        self.selected_features_ = None
        self.tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, top_n: int = 50) -> 'LassoFeatureSelector':
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Create and fit model
        self.model = LassoCV(
            cv=self.tscv,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            n_jobs=self.n_jobs
        )
        
        # Fit model
        self.model.fit(X, y)
        
        # Get feature importances (absolute coefficients)
        self.feature_importances_ = pd.Series(
            np.abs(self.model.coef_),
            index=self.feature_names
        )
        
        # Sort by importance
        self.feature_importances_ = self.feature_importances_.sort_values(ascending=False)
        
        # Select top features
        self.selected_features_ = self.feature_importances_.head(top_n).index.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to keep only selected features
        
        Parameters:
        ----
        X: Feature matrix
        
        Returns:
        ----
        Transformed data with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        return X[self.selected_features_].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, top_n: int = 50) -> pd.DataFrame:
        """
        Fit model and transform data
        
        Parameters:
        ----
        X: Feature matrix
        y: Target variable
        top_n: Number of top features to select
        
        Returns:
        ----
        Transformed data with only selected features
        """
        self.fit(X, y, top_n)
        return self.transform(X)
    
    def get_support(self) -> pd.Series:
        """
        Get feature importance values
        
        Returns:
        ----
        Series with feature importance values
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        return self.feature_importances_
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance
        
        Parameters:
        ----
        top_n: Number of top features to show
        figsize: Figure size
        
        Returns:
        ----
        Matplotlib Figure object
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        top_features = self.feature_importances_.head(top_n)
        top_features.sort_values().plot(kind='barh', ax=ax)
        ax.set_title('Lasso Feature Importance')
        plt.tight_layout()
        
        return fig

# 看 this one !!! 
class XGBoostFeatureSelector:
    """
    Feature selector based on XGBoost model
    """
    
    def __init__(self,
                cv_folds: int = 5,
                random_state: int = 42,
                xgb_params: Optional[Dict] = None):
        """
        Initialize XGBoost feature selector
        
        Parameters:
        ----
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        xgb_params: XGBoost parameters dictionary
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.xgb_params = xgb_params or {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        self.xgb_params['random_state'] = random_state
        
        self.model = None
        self.feature_names = None
        self.selected_features_ = None
        self.importance_weight_ = None
        self.importance_gain_ = None
        self.importance_perm_ = None
        self.importance_combined_ = None
        self.final_selected_features_ = None
        self.tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> 'XGBoostFeatureSelector':
        """
        Fit XGBoost model to select features
        
        Parameters:
        ----
        X: Feature matrix
        y: Target variable
        top_n: Number of top features to select
        
        Returns:
        ----
        Self instance
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Create and fit model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        
        # Fit model
        self.model.fit(X, y)
        
        # Calculate importances
        self._calculate_importances(X, y)
        
        # Combine importance measures and select top features
        self._combine_importances(top_n)
        
        return self
    
    def _calculate_importances(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculate different feature importance measures
        
        Parameters:
        ----
        X: Feature matrix
        y: Target variable
        """
        # Get weight importance
        weight_importance = self.model.get_booster().get_score(importance_type='weight')
        
        # 修正特徵索引轉換邏輯 - 使用直接映射而不是嘗試轉換為整數
        # 創建特徵名稱與索引的映射
        feature_map = {f'f{i}': name for i, name in enumerate(self.feature_names)}
        
        # 轉換特徵名稱 - 使用映射
        weight_importance = {feature_map.get(k, k): v for k, v in weight_importance.items()}
        
        # Create Series and fill missing values with 0
        self.importance_weight_ = pd.Series(weight_importance).reindex(
            self.feature_names, fill_value=0
        )
        
        # Get gain importance
        gain_importance = self.model.get_booster().get_score(importance_type='gain')
        
        # 使用同樣的映射方法轉換特徵名稱
        gain_importance = {feature_map.get(k, k): v for k, v in gain_importance.items()}
        
        # Create Series and fill missing values with 0
        self.importance_gain_ = pd.Series(gain_importance).reindex(
            self.feature_names, fill_value=0
        )
        
        # Calculate permutation importance
        try:
            perm_importance = permutation_importance(
                self.model, X, y, 
                n_repeats=10, 
                random_state=self.random_state
            )
            self.importance_perm_ = pd.Series(
                perm_importance.importances_mean, 
                index=self.feature_names
            )
        except Exception as e:
            print(f"Warning: Permutation importance calculation failed: {e}")
            self.importance_perm_ = pd.Series(0, index=self.feature_names)
    
    def _combine_importances(self, top_n: int = 20) -> None:
        """
        Combine different importance measures and select top features
        
        Parameters:
        ----
        top_n: Number of top features to select
        """
        # Normalize each importance measure to [0, 1] range
        def normalize_series(s: pd.Series) -> pd.Series:
            if s.max() == s.min():
                return s
            return (s - s.min()) / (s.max() - s.min())
        
        weight_norm = normalize_series(self.importance_weight_)
        gain_norm = normalize_series(self.importance_gain_)
        perm_norm = normalize_series(self.importance_perm_)
        
        # Create combined importance score
        self.importance_combined_ = (weight_norm + gain_norm + perm_norm) / 3
        
        # Sort by combined importance
        self.importance_combined_ = self.importance_combined_.sort_values(ascending=False)
        
        # Select top features
        self.final_selected_features_ = self.importance_combined_.head(top_n).index.tolist()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to keep only selected features
        
        Parameters:
        ----
        X: Feature matrix
        
        Returns:
        ----
        Transformed data with only selected features
        """
        if self.final_selected_features_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        return X[self.final_selected_features_].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> pd.DataFrame:
        """
        Fit model and transform data
        
        Parameters:
        ----
        X: Feature matrix
        y: Target variable
        top_n: Number of top features to select
        
        Returns:
        ----
        Transformed data with only selected features
        """
        self.fit(X, y, top_n)
        return self.transform(X)
    
    def get_support(self) -> pd.Series:
        """
        Get combined feature importance values
        
        Returns:
        ----
        Series with feature importance values
        """
        if self.importance_combined_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        return self.importance_combined_
    
    def get_metric_importance(self, metric: str) -> pd.Series:
        """
        Get specific importance metric values
        
        Parameters:
        ----
        metric: Importance metric ('weight', 'gain', 'perm', or 'combined')
        
        Returns:
        ----
        Series with feature importance values for the specified metric
        """
        if metric == 'weight':
            return self.importance_weight_
        elif metric == 'gain':
            return self.importance_gain_
        elif metric == 'perm':
            return self.importance_perm_
        elif metric == 'combined':
            return self.importance_combined_
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'weight', 'gain', 'perm', or 'combined'")
    
    def plot_feature_importance(self, 
                              top_n: int = 20, 
                              metric: str = 'combined', 
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot feature importance
        
        Parameters:
        ----
        top_n: Number of top features to show
        metric: Importance metric to use ('weight', 'gain', 'perm', or 'combined')
        figsize: Figure size
        
        Returns:
        ----
        Matplotlib Figure object
        """
        # Get importance values
        importance = self.get_metric_importance(metric)
        
        if importance is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        top_features = importance.head(top_n)
        top_features.sort_values().plot(kind='barh', ax=ax)
        
        metric_names = {
            'weight': 'Weight',
            'gain': 'Gain',
            'perm': 'Permutation',
            'combined': 'Combined'
        }
        
        ax.set_title(f'XGBoost Feature Importance ({metric_names.get(metric, metric)})')
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_comparison(self, top_n: int = 15, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comparison of different importance metrics
        
        Parameters:
        ----
        top_n: Number of top features to show
        figsize: Figure size
        
        Returns:
        ----
        Matplotlib Figure object
        """
        if self.importance_combined_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        # Get top features based on combined importance
        top_features = self.importance_combined_.head(top_n).index.tolist()
        
        # Create dataframe with all metrics
        plot_data = pd.DataFrame({
            'Weight': self.importance_weight_[top_features],
            'Gain': self.importance_gain_[top_features],
            'Permutation': self.importance_perm_[top_features],
            'Combined': self.importance_combined_[top_features]
        })
        
        # Normalize each column
        for col in plot_data.columns:
            if plot_data[col].max() > 0:
                plot_data[col] = plot_data[col] / plot_data[col].max()
        
        # Sort by combined importance
        plot_data = plot_data.sort_values('Combined', ascending=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        plot_data.plot(kind='barh', ax=ax)
        
        ax.set_title('XGBoost Feature Importance Metrics Comparison')
        ax.set_xlabel('Normalized Importance')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        return fig

class TwoStageFeatureSelector:
    """
    Two-stage feature selector combining Lasso and XGBoost
    """
    
    def __init__(self,
               cv_folds: int = 5,
               random_state: int = 42,
               lasso_params: Optional[Dict] = None,
               xgb_params: Optional[Dict] = None):
        
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.lasso_params = lasso_params or {}
        self.xgb_params = xgb_params or {}
        
        # Create individual selectors
        self.lasso_selector = LassoFeatureSelector(
            cv_folds=cv_folds,
            random_state=random_state,
            **lasso_params
        )
        
        self.xgb_selector = XGBoostFeatureSelector(
            cv_folds=cv_folds,
            random_state=random_state,
            xgb_params=xgb_params
        )
        
        self.lasso_selected_features_ = None
        self.final_selected_features_ = None
    
    def fit(self, 
          X: pd.DataFrame, 
          y: pd.Series, 
          top_n_lasso: int = 50, 
          top_n_final: int = 20) -> 'TwoStageFeatureSelector':
        """
        Fit two-stage feature selection model
        
        Parameters:
        ----
        X: Feature matrix
        y: Target variable
        top_n_lasso: Number of features to select in first stage (Lasso)
        top_n_final: Number of features to select in final stage (XGBoost)
        
        Returns:
        ----
        Self instance
        """
        # Handle special cases
        if X.shape[1] <= top_n_final:
            print(f"Warning: Number of features ({X.shape[1]}) is less than or equal to top_n_final ({top_n_final}). Skipping selection.")
            self.lasso_selected_features_ = X.columns.tolist()
            self.final_selected_features_ = X.columns.tolist()
            return self
        
        # Stage 1: Lasso selection
        print(f"Stage 1: Selecting top {top_n_lasso} features with Lasso")
        self.lasso_selector.fit(X, y, top_n=top_n_lasso)
        self.lasso_selected_features_ = self.lasso_selector.selected_features_
        
        # If Lasso didn't find enough features, use all
        if len(self.lasso_selected_features_) < 2:
            print("Warning: Lasso selected too few features. Using all features for XGBoost.")
            self.lasso_selected_features_ = X.columns.tolist()
        
        # Stage 2: XGBoost selection from Lasso-selected features
        print(f"Stage 2: Selecting top {top_n_final} features with XGBoost from {len(self.lasso_selected_features_)} Lasso-selected features")
        X_lasso = X[self.lasso_selected_features_]
        self.xgb_selector.fit(X_lasso, y, top_n=min(top_n_final, len(self.lasso_selected_features_)))
        self.final_selected_features_ = self.xgb_selector.final_selected_features_
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to keep only selected features
        
        Parameters:
        ----
        X: Feature matrix
        
        Returns:
        ----
        Transformed data with only selected features
        """
        if self.final_selected_features_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        return X[self.final_selected_features_].copy()
    
    def fit_transform(self, 
                    X: pd.DataFrame, 
                    y: pd.Series, 
                    top_n_lasso: int = 50, 
                    top_n_final: int = 20) -> pd.DataFrame:
        """
        Fit model and transform data
        
        Parameters:
        ----
        X: Feature matrix
        y: Target variable
        top_n_lasso: Number of features to select in first stage (Lasso)
        top_n_final: Number of features to select in final stage (XGBoost)
        
        Returns:
        ----
        Transformed data with only selected features
        """
        self.fit(X, y, top_n_lasso, top_n_final)
        return self.transform(X)
    
    def get_lasso_support(self) -> pd.Series:
        """
        Get Lasso feature importance values
        
        Returns:
        ----
        Series with Lasso feature importance values
        """
        return self.lasso_selector.get_support()
    
    def get_xgb_support(self) -> pd.Series:
        """
        Get XGBoost feature importance values
        
        Returns:
        ----
        Series with XGBoost feature importance values
        """
        return self.xgb_selector.get_support()
    
    def plot_lasso_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot Lasso feature importance
        
        Parameters:
        ----
        top_n: Number of top features to show
        figsize: Figure size
        
        Returns:
        ----
        Matplotlib Figure object
        """
        return self.lasso_selector.plot_feature_importance(top_n, figsize)
    
    def plot_xgb_importance(self, 
                          top_n: int = 20, 
                          metric: str = 'combined', 
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot XGBoost feature importance
        
        Parameters:
        ----
        top_n: Number of top features to show
        metric: Importance metric to use ('weight', 'gain', 'perm', or 'combined')
        figsize: Figure size
        
        Returns:
        ----
        Matplotlib Figure object
        """
        return self.xgb_selector.plot_feature_importance(top_n, metric, figsize)
    
    def plot_importance_comparison(self, top_n: int = 15, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comparison of Lasso and XGBoost feature importance
        
        Parameters:
        ----
        top_n: Number of top features to show
        figsize: Figure size
        
        Returns:
        ----
        Matplotlib Figure object
        """
        if self.final_selected_features_ is None:
            raise ValueError("Model has not been fit yet. Call fit first.")
        
        # Get importances
        lasso_imp = self.get_lasso_support()
        xgb_imp = self.get_xgb_support()
        
        # Get union of top features from both methods
        lasso_top = lasso_imp.head(top_n).index.tolist()
        xgb_top = xgb_imp.head(top_n).index.tolist()
        all_top = list(set(lasso_top) | set(xgb_top))
        
        # Sort by combined importance
        comb_imp = {}
        for feature in all_top:
            l_imp = lasso_imp.get(feature, 0)
            x_imp = xgb_imp.get(feature, 0)
            
            # Normalize within each method
            l_norm = l_imp / lasso_imp.max() if lasso_imp.max() > 0 else 0
            x_norm = x_imp / xgb_imp.max() if xgb_imp.max() > 0 else 0
            
            comb_imp[feature] = (l_norm + x_norm) / 2
        
        # Sort by combined importance
        sorted_features = sorted(comb_imp.items(), key=lambda x: x[1], reverse=True)
        top_features = [f for f, _ in sorted_features[:top_n]]
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame(index=top_features, columns=['Lasso', 'XGBoost'])
        
        for feature in top_features:
            plot_data.loc[feature, 'Lasso'] = lasso_imp.get(feature, 0) / lasso_imp.max() if lasso_imp.max() > 0 else 0
            plot_data.loc[feature, 'XGBoost'] = xgb_imp.get(feature, 0) / xgb_imp.max() if xgb_imp.max() > 0 else 0
        
        # Sort by average importance
        plot_data['Average'] = (plot_data['Lasso'] + plot_data['XGBoost']) / 2
        plot_data = plot_data.sort_values('Average', ascending=True)
        plot_data = plot_data.drop('Average', axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        plot_data.plot(kind='barh', ax=ax)
        
        ax.set_title('Lasso vs XGBoost Feature Importance Comparison')
        ax.set_xlabel('Normalized Importance')
        
        plt.tight_layout()
        return fig 