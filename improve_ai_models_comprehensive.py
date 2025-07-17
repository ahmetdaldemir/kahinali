# -*- coding: utf-8 -*-
"""
Kahin Ultima - Comprehensive AI Model Improvement Script
Enhanced feature engineering, feature selection, and hyperparameter optimization
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_improvement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveAIImprovement:
    def __init__(self):
        self.data_collector = DataCollector()
        self.technical_analysis = TechnicalAnalysis()
        self.ai_model = AIModel()

    def load_training_data(self, coin_list_file='valid_coins.txt', timeframe='1h', months=6):
        logger.info("Loading training data...")
        with open(coin_list_file, 'r') as f:
            coins = [line.strip() for line in f.readlines() if line.strip()]
        all_data = []
        for coin in coins[:50]:
            data_file = f"data/{coin}_{timeframe.upper()}_{months}months.csv"
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                df['symbol'] = coin
                all_data.append(df)
                logger.info(f"Loaded {coin}: {len(df)} rows")
            else:
                logger.warning(f"Data file not found: {data_file}")
        if not all_data:
            logger.error("No training data loaded!")
            return None
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined training data: {len(combined_data)} rows")
        return combined_data

    def enhance_feature_engineering(self, df):
        logger.info("Starting enhanced feature engineering...")
        if df is None or df.empty:
            logger.error("Input DataFrame is empty")
            return df
        df = df.copy()
        df['price_change'] = df['close'].pct_change()
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            if 'volume' in df.columns:
                df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
        df['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 0
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek if hasattr(df.index, 'dayofweek') else 0
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        logger.info(f"Enhanced feature engineering completed. Total features: {len(df.columns)}")
        return df

    def create_advanced_labels(self, df):
        logger.info("Creating advanced labels...")
        if df is None or df.empty:
            return df
        df = df.copy()
        for horizon in [5, 10, 20, 30]:
            df[f'future_close_{horizon}'] = df['close'].shift(-horizon)
            df[f'return_{horizon}'] = (df[f'future_close_{horizon}'] - df['close']) / df['close']
            df[f'volatility_{horizon}'] = df['close'].pct_change().rolling(horizon).std()
            df[f'dynamic_threshold_{horizon}'] = df[f'volatility_{horizon}'] * 1.5
            df[f'label_{horizon}'] = (df[f'return_{horizon}'] > df[f'dynamic_threshold_{horizon}']).astype(int)
        logger.info("Advanced labels created successfully")
        return df

    def perform_feature_selection(self, X, y):
        logger.info("Performing feature selection...")
        kbest = SelectKBest(score_func=f_classif, k=min(80, X.shape[1]))
        kbest.fit(X, y)
        kbest_features = X.columns[kbest.get_support()].tolist()
        rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=80)
        rf_selector.fit(X, y)
        rf_features = X.columns[rf_selector.get_support()].tolist()
        selected_features = list(set(kbest_features + rf_features))
        if len(selected_features) > 125:
            selected_features = selected_features[:125]
        logger.info(f"Feature selection completed. Selected {len(selected_features)} features.")
        return X[selected_features], selected_features

    def optimize_hyperparameters(self, X, y):
        logger.info("Optimizing hyperparameters...")
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', None]
        }
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
        rf_grid.fit(X, y)
        best_rf = rf_grid.best_params_
        logger.info(f"Best RF params: {best_rf}")
        return best_rf

    def train_and_evaluate(self, X, y, best_rf):
        logger.info("Training and evaluating model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(**best_rf, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logger.info(f"Results: {results}")
        return rf, results

    def save_model(self, model, features, params, results):
        logger.info("Saving model and metadata...")
        joblib.dump(model, os.path.join(Config.MODELS_DIR, 'rf_improved.pkl'))
        with open(os.path.join(Config.MODELS_DIR, 'improved_feature_cols.pkl'), 'wb') as f:
            pickle.dump(features, f)
        with open(os.path.join(Config.MODELS_DIR, 'improved_hyperparameters.pkl'), 'wb') as f:
            pickle.dump(params, f)
        with open(os.path.join(Config.MODELS_DIR, 'improvement_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        logger.info("Model and metadata saved.")

    def run(self):
        logger.info("Starting comprehensive AI model improvement...")
        data = self.load_training_data()
        if data is None:
            logger.error("No data loaded. Exiting.")
            return
        ta_data = self.technical_analysis.calculate_all_indicators(data)
        features = self.enhance_feature_engineering(ta_data)
        labeled = self.create_advanced_labels(features)
        target_col = 'label_10'
        if target_col not in labeled.columns:
            logger.error(f"Target column {target_col} not found.")
            return
        exclude_cols = ['label_5', 'label_20', 'label_30', 'label_dynamic', 'trend_label', 'momentum_label',
                        'future_close_5', 'future_close_10', 'future_close_20', 'future_close_30',
                        'return_5', 'return_10', 'return_20', 'return_30']
        feature_cols = [col for col in labeled.columns if col not in exclude_cols and col != target_col]
        X = labeled[feature_cols].fillna(0)
        y = labeled[target_col].fillna(0)
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        X_selected, selected_features = self.perform_feature_selection(X, y)
        best_rf = self.optimize_hyperparameters(X_selected, y)
        model, results = self.train_and_evaluate(X_selected, y, best_rf)
        self.save_model(model, selected_features, best_rf, results)
        logger.info("Comprehensive AI model improvement completed!")
        print("\nAI MODEL IMPROVEMENT COMPLETED SUCCESSFULLY!\n")

def main():
    improvement = ComprehensiveAIImprovement()
    improvement.run()

if __name__ == "__main__":
    main() 