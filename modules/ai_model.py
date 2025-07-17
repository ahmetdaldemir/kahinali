import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import logging
from config import Config
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import pickle
import random
import warnings
warnings.filterwarnings('ignore')
from modules.technical_analysis import convert_string_features_to_numeric

logger = logging.getLogger(__name__)

class AIModel:
    def __init__(self, model_dir=Config.MODELS_DIR):
        self.model_dir = model_dir
        self.lstm_model_path = os.path.join(model_dir, 'lstm_model.h5')
        self.rf_model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        self.gb_model_path = os.path.join(model_dir, 'gradient_boosting_model.pkl')
        self.ensemble_model_path = os.path.join(model_dir, 'ensemble_model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.feature_cols_path = os.path.join(model_dir, 'feature_cols.pkl')
        self.logger = logging.getLogger(__name__)
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.ensemble_model = None
        self.feature_importance = {}
        self.model_performance = {}
        self.ensemble_weights = {'lstm': 0.4, 'rf': 0.3, 'gb': 0.3}
        self.feature_columns = []  # Bu attribute'u ekle
        
        # Modelleri yükle
        self.load_models()
        
        # Feature columns'ı doğru şekilde ayarla
        if hasattr(self, 'feature_cols'):
            self.feature_columns = self.feature_cols
        else:
            self.feature_columns = []
        
        # Dummy feature'ları çıkar (feature_62-124 arası)
        if hasattr(self, 'feature_cols') and isinstance(self.feature_cols, list):
            self.feature_cols = [
                f for f in self.feature_cols
                if not (f.startswith('feature_') and f[8:].isdigit() and 62 <= int(f[8:]) <= 124)
            ]

    def load_models(self):
        """Mevcut modelleri yükle"""
        try:
            # LSTM model - Yükle
            if os.path.exists(self.lstm_model_path):
                from tensorflow.keras.models import load_model
                try:
                    self.lstm_model = load_model(self.lstm_model_path)
                    self.logger.info("LSTM modeli başarıyla yüklendi")
                except Exception as e:
                    self.lstm_model = None
                    self.logger.warning(f"LSTM modeli yüklenemedi: {e}")
            else:
                self.lstm_model = None
                self.logger.warning("LSTM model dosyası bulunamadı")
            
            # Random Forest model
            if os.path.exists(self.rf_model_path):
                self.rf_model = joblib.load(self.rf_model_path)
                self.logger.info("Random Forest model yüklendi")
            else:
                self.rf_model = None
                self.logger.warning("Random Forest model dosyası bulunamadı")
            
            # Gradient Boosting model
            if os.path.exists(self.gb_model_path):
                self.gb_model = joblib.load(self.gb_model_path)
                self.logger.info("Gradient Boosting model yüklendi")
            else:
                self.gb_model = None
                self.logger.warning("Gradient Boosting model dosyası bulunamadı")
            
            # Scaler - joblib ile yükle
            if os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                    self.logger.info("Scaler yüklendi")
                except Exception as e:
                    self.logger.error(f"Scaler yükleme hatası: {e}")
                    self.scaler = None
            else:
                self.scaler = None
                self.logger.warning("Scaler dosyası bulunamadı")
                
            # Feature columns - pickle ile yükle
            if os.path.exists(self.feature_cols_path):
                try:
                    with open(self.feature_cols_path, 'rb') as f:
                        loaded_feature_cols = pickle.load(f)
                    
                    # String ise listeye çevir
                    if isinstance(loaded_feature_cols, str):
                        self.logger.warning("Feature columns string olarak yüklendi, varsayılan liste kullanılıyor")
                        self.feature_cols = [
                            'open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 
                            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_histogram', 
                            'rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'bb_middle', 'bb_upper', 'bb_lower', 
                            'bb_width', 'bb_percent', 'atr', 'obv', 'vwap', 'adx_pos', 'adx_neg', 'adx', 'cci', 'mfi', 
                            'williams_r', 'psar', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 
                            'keltner_ema', 'keltner_upper', 'keltner_lower', 'volume_roc', 'volume_ma', 'volume_ratio', 
                            'roc', 'momentum', 'price_roc', 'historical_volatility', 'true_range', 'volatility_ratio', 
                            'price_change', 'price_change_5', 'price_change_10', 'return_5', 'return_10', 'return_20', 
                            'cumulative_return', 'momentum_5', 'momentum_10', 'volatility', 'volatility_5', 'volatility_10', 
                            'volatility_20', 'volume_ma_5', 'volume_ma_10', 'dynamic_threshold', 'label_5', 'label_10', 
                            'label_20', 'label_dynamic', 'day_of_week', 'hour', 'doji', 'hammer', 'shooting_star', 
                            'support_level', 'resistance_level', 'price_vs_support', 'price_vs_resistance', 'volume_sma', 
                            'high_volume', 'body_size', 'upper_shadow', 'lower_shadow', 'momentum_ma', 'momentum_trend', 
                            'breakout_up', 'breakout_down', 'price_range', 'consolidation', 'volatility_ma', 'trend_strength', 
                            'trend_ma', 'range', 'range_ma', 'market_regime', 'price_change_20', 'volatility_50', 'momentum_20', 
                            'volume_ma_20', 'volume_trend', 'rsi_trend', 'rsi_momentum', 'macd_strength', 'macd_trend', 
                            'bb_position', 'bb_squeeze', 'future_close_5', 'future_close_10', 'future_close_20', 'future_close_30', 
                            'return_30', 'volatility_30', 'dynamic_threshold_5', 'dynamic_threshold_10', 'dynamic_threshold_20', 
                            'dynamic_threshold_30', 'label_30', 'trend_5', 'trend_20', 'trend_label', 'momentum_label'
                        ][:125]  # Tam olarak 125 feature
                    else:
                        # Feature sayısını 125'e sabitle
                        if len(loaded_feature_cols) > 125:
                            self.feature_cols = loaded_feature_cols[:125]
                            self.logger.info(f"Feature sayısı 125'e sabitlendi: {len(self.feature_cols)}")
                        else:
                            self.feature_cols = loaded_feature_cols
                    self.logger.info(f"Feature columns yüklendi: {len(self.feature_cols)} feature")
                except Exception as e:
                    self.logger.error(f"Feature columns yükleme hatası: {e}")
                    self.feature_cols = [
                        'open', 'high', 'low', 'close', 'volume', 'future_close_5', 'return_5', 'volatility_5', 'dynamic_threshold_5',
                        'label_5', 'future_close_10', 'return_10', 'volatility_10', 'dynamic_threshold_10', 'label_10', 'future_close_20', 'return_20', 'volatility_20', 'dynamic_threshold_20', 'label_20', 'future_close_30', 'return_30', 'volatility_30', 'dynamic_threshold_30', 'label_30', 'return', 'volatility', 'dynamic_threshold', 'label_dynamic', 'trend_5', 'trend_20', 'trend_label', 'momentum_5', 'momentum_10', 'momentum_label', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_histogram', 'rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
                        'bb_percent', 'atr', 'obv', 'vwap', 'adx_pos', 'adx_neg', 'adx', 'cci', 'mfi', 'williams_r', 'psar', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'keltner_ema', 'keltner_upper', 'keltner_lower', 'volume_roc', 'volume_ma', 'volume_ratio', 'roc', 'momentum', 'price_roc', 'historical_volatility', 'true_range', 'volatility_ratio', 'price_change', 'price_change_5', 'price_change_10', 'cumulative_return', 'volume_ma_5', 'volume_ma_10', 'day_of_week', 'hour', 'doji', 'hammer', 'shooting_star', 'support_level', 'resistance_level', 'price_vs_support', 'price_vs_resistance', 'volume_sma', 'high_volume', 'body_size', 'upper_shadow', 'lower_shadow', 'momentum_ma', 'momentum_trend', 'breakout_up', 'breakout_down', 'price_range', 'consolidation', 'volatility_ma', 'trend_strength', 'trend_ma', 'range', 'range_ma', 'market_regime', 'price_change_20', 'volatility_50', 'momentum_20', 'volume_ma_20', 'volume_trend', 'rsi_trend', 'rsi_momentum', 'macd_strength', 'macd_trend', 'bb_position', 'bb_squeeze'
                    ]
            else:
                self.logger.warning("Feature columns dosyası bulunamadı, varsayılan liste kullanılıyor")
                self.feature_cols = [
                    'open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 
                    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_histogram', 
                    'rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'bb_middle', 'bb_upper', 'bb_lower', 
                    'bb_width', 'bb_percent', 'atr', 'obv', 'vwap', 'adx_pos', 'adx_neg', 'adx', 'cci', 'mfi', 
                    'williams_r', 'psar', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 
                    'keltner_ema', 'keltner_upper', 'keltner_lower', 'volume_roc', 'volume_ma', 'volume_ratio', 
                    'roc', 'momentum', 'price_roc', 'historical_volatility', 'true_range', 'volatility_ratio', 
                    'price_change', 'price_change_5', 'price_change_10', 'return_5', 'return_10', 'return_20', 
                    'cumulative_return', 'momentum_5', 'momentum_10', 'volatility', 'volatility_5', 'volatility_10', 
                    'volatility_20', 'volume_ma_5', 'volume_ma_10', 'dynamic_threshold', 'label_5', 'label_10', 
                    'label_20', 'label_dynamic', 'day_of_week', 'hour', 'doji', 'hammer', 'shooting_star', 
                    'support_level', 'resistance_level', 'price_vs_support', 'price_vs_resistance', 'volume_sma', 
                    'high_volume', 'body_size', 'upper_shadow', 'lower_shadow', 'momentum_ma', 'momentum_trend', 
                    'breakout_up', 'breakout_down', 'price_range', 'consolidation', 'volatility_ma', 'trend_strength', 
                    'trend_ma', 'range', 'range_ma', 'market_regime', 'price_change_20', 'volatility_50', 'momentum_20', 
                    'volume_ma_20', 'volume_trend', 'rsi_trend', 'rsi_momentum', 'macd_strength', 'macd_trend', 
                    'bb_position', 'bb_squeeze', 'future_close_5', 'future_close_10', 'future_close_20', 'future_close_30', 
                    'return_30', 'volatility_30', 'dynamic_threshold_5', 'dynamic_threshold_10', 'dynamic_threshold_20', 
                    'dynamic_threshold_30', 'label_30', 'trend_5', 'trend_20', 'trend_label', 'momentum_label'
                ][:125]  # Tam olarak 125 feature
                
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            self.lstm_model = None
            self.rf_model = None
            self.gb_model = None
            self.scaler = None

    def create_labels(self, df, threshold=0.03, horizon=10):
        """Gelişmiş etiket oluşturma - dinamik threshold ve çoklu horizon"""
        if df is None or df.empty:
            self.logger.error("Input DataFrame is empty or None")
            return df
            
        df = df.copy()
        
        # Çoklu horizon için etiketler
        horizons = [5, 10, 20, 30]
        for h in horizons:
            df[f'future_close_{h}'] = df['close'].shift(-h)
            df[f'return_{h}'] = (df[f'future_close_{h}'] - df['close']) / df['close']
            
            # Dinamik threshold hesaplama
            df[f'volatility_{h}'] = df['close'].pct_change().rolling(h).std()
            df[f'dynamic_threshold_{h}'] = df[f'volatility_{h}'] * 1.5  # Volatiliteye göre ayarla
            
            # Dinamik threshold ile etiketleme
            df[f'label_{h}'] = (df[f'return_{h}'] > df[f'dynamic_threshold_{h}']).astype(int)
        
        # Ana etiket (10 bar sonrası)
        df['label'] = df['label_10']
        df['return'] = df['return_10']
        
        # Volatilite bazlı dinamik threshold
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['dynamic_threshold'] = df['volatility'] * 1.5  # Daha agresif threshold
        df['label_dynamic'] = (df['return'] > df['dynamic_threshold']).astype(int)
        
        # Trend bazlı etiketleme
        df['trend_5'] = df['close'].rolling(5).mean()
        df['trend_20'] = df['close'].rolling(20).mean()
        df['trend_label'] = ((df['trend_5'] > df['trend_20']) & (df['return'] > 0.02)).astype(int)
        
        # Momentum bazlı etiketleme
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_label'] = ((df['momentum_5'] > 0.01) & (df['momentum_10'] > 0.02)).astype(int)
        
        # Final etiket - daha karmaşık ve hassas
        df['label'] = (
            (df['label_10'] == 1) | 
            (df['label_dynamic'] == 1) | 
            (df['trend_label'] == 1) |
            (df['momentum_label'] == 1)
        ).astype(int)
        
        # NaN değerleri güvenli şekilde temizle
        initial_rows = len(df)
        
        # Sadece gerekli sütunlardaki NaN değerleri temizle
        required_cols = ['label', 'return', 'close']
        df = df.dropna(subset=required_cols)
        
        # Diğer sütunlardaki NaN değerleri 0 ile doldur
        df = df.fillna(0)
        
        final_rows = len(df)
        if final_rows == 0:
            self.logger.error("Label creation resulted in empty DataFrame")
            return df
            
        self.logger.info(f"Label creation: {initial_rows} -> {final_rows} rows")
        
        return df

    def engineer_features(self, df):
        """Gelişmiş özellik mühendisliği"""
        try:
            feature_columns = []
            ta_features = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
                'macd', 'macd_signal', 'macd_histogram',
                'rsi_7', 'rsi_14', 'rsi_21',
                'stoch_k', 'stoch_d',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent',
                'atr', 'obv', 'vwap',
                'adx_pos', 'adx_neg', 'adx',
                'cci', 'mfi', 'williams_r', 'psar',
                'tenkan_sen', 'kijun_sen',
                'senkou_span_a', 'senkou_span_b', 'chikou_span',
                'keltner_ema', 'keltner_upper', 'keltner_lower',
                'volume_roc', 'volume_ma', 'volume_ratio',
                'roc', 'momentum', 'price_roc',
                'historical_volatility', 'true_range', 'volatility_ratio',
                'price_change', 'price_change_5', 'price_change_10',
                'return_5', 'return_10', 'return_20',
                'cumulative_return', 'momentum_5', 'momentum_10',
                'volatility', 'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ma_5', 'volume_ma_10']
            dummy_count = 0
            for feature in ta_features:
                if feature in df.columns:
                    feature_columns.append(feature)
                else:
                    self.logger.error(f"{feature} eksik, feature engineering başarısız. Sinyal üretimi iptal edildi.")
                    return None
            # Teknik indikatörler zaten hesaplanmış olmalı
            # Ek özellikler ekle
            
            # Fiyat özellikleri
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            df['price_change_20'] = df['close'].pct_change(20)
            
            # Volatilite özellikleri
            df['volatility_5'] = df['close'].pct_change().rolling(5).std()
            df['volatility_10'] = df['close'].pct_change().rolling(10).std()
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            df['volatility_50'] = df['close'].pct_change().rolling(50).std()
            
            # Momentum özellikleri
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Hacim özellikleri
            if 'volume' in df.columns:
                df['volume_ma_5'] = df['volume'].rolling(5).mean()
                df['volume_ma_10'] = df['volume'].rolling(10).mean()
                df['volume_ma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_5']
                df['volume_trend'] = df['volume_ma_5'] / df['volume_ma_20']
            
            # Zaman özellikleri
            df['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 0
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek if hasattr(df.index, 'dayofweek') else 0
            
            # Teknik indikatör kombinasyonları
            if 'rsi_14' in df.columns:
                df['rsi_trend'] = df['rsi_14'] - df['rsi_14'].shift(5)
                df['rsi_momentum'] = df['rsi_14'] - df['rsi_14'].rolling(10).mean()
            
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                df['macd_strength'] = df['macd'] - df['macd_signal']
                df['macd_trend'] = df['macd_strength'] - df['macd_strength'].shift(5)
            
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                df['bb_squeeze'] = df['bb_upper'] - df['bb_lower']
            
            # NaN değerleri temizle
            df = df.dropna()
            
            # Teknik analizden gelen feature'lar ve diğerleri hazırlandıktan sonra:
            df = convert_string_features_to_numeric(df)
            
            return df
        except Exception as e:
            self.logger.error(f"Feature hazırlama hatası: {e}")
            return None

    def _prepare_features(self, df):
        """Feature'ları hazırla ve eksik olanları ekle. Eksik feature'lar teknik analizden hesaplanamıyorsa geçmiş ortalama ile doldurulur. Her sinyalde feature kalite oranı hesaplanır."""
        try:
            feature_columns = []
            ta_features = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
                'macd', 'macd_signal', 'macd_histogram',
                'rsi_7', 'rsi_14', 'rsi_21',
                'stoch_k', 'stoch_d',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent',
                'atr', 'obv', 'vwap',
                'adx_pos', 'adx_neg', 'adx',
                'cci', 'mfi', 'williams_r', 'psar',
                'tenkan_sen', 'kijun_sen',
                'senkou_span_a', 'senkou_span_b', 'chikou_span',
                'keltner_ema', 'keltner_upper', 'keltner_lower',
                'volume_roc', 'volume_ma', 'volume_ratio',
                'roc', 'momentum', 'price_roc',
                'historical_volatility', 'true_range', 'volatility_ratio',
                'price_change', 'price_change_5', 'price_change_10',
                'return_5', 'return_10', 'return_20',
                'cumulative_return', 'momentum_5', 'momentum_10',
                'volatility', 'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ma_5', 'volume_ma_10']
            dummy_count = 0
            for feature in ta_features:
                if feature in df.columns:
                    feature_columns.append(feature)
                else:
                    # Eksik feature'ı geçmiş ortalama ile doldur
                    if feature in ['close', 'volume']:
                        df[feature] = 0.0
                    elif feature in df:
                        df[feature] = df[feature].mean()
                    elif len(df) > 0:
                        df[feature] = df.mean(numeric_only=True).mean()
                    else:
                        df[feature] = 0.0
                    feature_columns.append(feature)
                    dummy_count += 1
            # Zaman feature'ları
            time_features = ['day_of_week', 'hour']
            for feature in time_features:
                if feature in df.columns:
                    feature_columns.append(feature)
                else:
                    if feature == 'day_of_week':
                        df[feature] = pd.to_datetime(df.index).dayofweek
                    elif feature == 'hour':
                        df[feature] = pd.to_datetime(df.index).hour
                    feature_columns.append(feature)
            # Label feature'ları
            label_features = ['label_5', 'label_10', 'label_20', 'label_dynamic']
            for feature in label_features:
                if feature in df.columns:
                    feature_columns.append(feature)
                else:
                    df[feature] = 0
                    feature_columns.append(feature)
            # Dynamic threshold
            if 'dynamic_threshold' in df.columns:
                feature_columns.append('dynamic_threshold')
            else:
                df['dynamic_threshold'] = 0.5
                feature_columns.append('dynamic_threshold')
            # Eksik 27 feature'ı ekle (99 feature'a tamamla)
            extra_features = [
                'dynamic_threshold_5', 'dynamic_threshold_10', 'dynamic_threshold_20', 'dynamic_threshold_30',
                'return_30', 'volatility_30', 'label_30',
                'trend_5', 'trend_20', 'trend_label',
                'momentum_label', 'price_change_20', 'volatility_50',
                'momentum_20', 'volume_ma_20', 'volume_trend',
                'rsi_trend', 'rsi_momentum', 'macd_strength',
                'macd_trend', 'bb_position', 'bb_squeeze'
            ]
            for feature in extra_features:
                if feature in df.columns:
                    feature_columns.append(feature)
                else:
                    # Eksik feature'ları hesapla veya geçmiş ortalama ile doldur
                    if feature == 'dynamic_threshold_5':
                        df[feature] = df['close'].pct_change().rolling(5).std() * 1.5 if 'close' in df.columns else 0.0
                    elif feature == 'dynamic_threshold_10':
                        df[feature] = df['close'].pct_change().rolling(10).std() * 1.5 if 'close' in df.columns else 0.0
                    elif feature == 'dynamic_threshold_20':
                        df[feature] = df['close'].pct_change().rolling(20).std() * 1.5 if 'close' in df.columns else 0.0
                    elif feature == 'dynamic_threshold_30':
                        df[feature] = df['close'].pct_change().rolling(30).std() * 1.5 if 'close' in df.columns else 0.0
                    elif feature == 'return_30':
                        df[feature] = df['close'].pct_change(30) if 'close' in df.columns else 0.0
                    elif feature == 'volatility_30':
                        df[feature] = df['close'].pct_change().rolling(30).std() if 'close' in df.columns else 0.0
                    elif feature == 'label_30':
                        df[feature] = 0
                    elif feature == 'trend_5':
                        df[feature] = df['close'].rolling(5).mean() if 'close' in df.columns else 0.0
                    elif feature == 'trend_20':
                        df[feature] = df['close'].rolling(20).mean() if 'close' in df.columns else 0.0
                    elif feature == 'trend_label':
                        df[feature] = ((df['trend_5'] > df['trend_20']) & (df['return_10'] > 0.02)).astype(int) if 'trend_5' in df.columns and 'trend_20' in df.columns and 'return_10' in df.columns else 0
                    elif feature == 'momentum_label':
                        df[feature] = ((df['momentum_5'] > 0.01) & (df['momentum_10'] > 0.02)).astype(int) if 'momentum_5' in df.columns and 'momentum_10' in df.columns else 0
                    elif feature == 'price_change_20':
                        df[feature] = df['close'].pct_change(20) if 'close' in df.columns else 0.0
                    elif feature == 'volatility_50':
                        df[feature] = df['close'].pct_change().rolling(50).std() if 'close' in df.columns else 0.0
                    elif feature == 'momentum_20':
                        df[feature] = df['close'] / df['close'].shift(20) - 1 if 'close' in df.columns else 0.0
                    elif feature == 'volume_ma_20':
                        df[feature] = df['volume'].rolling(20).mean() if 'volume' in df.columns else 0.0
                    elif feature == 'volume_trend':
                        df[feature] = df['volume_ma_5'] / df['volume_ma_20'] if 'volume_ma_5' in df.columns and 'volume_ma_20' in df.columns else 1.0
                    elif feature == 'rsi_trend':
                        df[feature] = df['rsi_14'] - df['rsi_14'].shift(5) if 'rsi_14' in df.columns else 0.0
                    elif feature == 'rsi_momentum':
                        df[feature] = df['rsi_14'] - df['rsi_14'].rolling(10).mean() if 'rsi_14' in df.columns else 0.0
                    elif feature == 'macd_strength':
                        df[feature] = df['macd'] - df['macd_signal'] if 'macd' in df.columns and 'macd_signal' in df.columns else 0.0
                    elif feature == 'macd_trend':
                        macd_strength = df['macd'] - df['macd_signal'] if 'macd' in df.columns and 'macd_signal' in df.columns else 0.0
                        df[feature] = macd_strength - macd_strength.shift(5) if isinstance(macd_strength, pd.Series) else 0.0
                    elif feature == 'bb_position':
                        df[feature] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) if 'bb_upper' in df.columns and 'bb_lower' in df.columns else 0.5
                    elif feature == 'bb_squeeze':
                        df[feature] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns else 0.0
                    else:
                        # Geçmiş ortalama ile doldur
                        df[feature] = df.mean(numeric_only=True).mean() if len(df) > 0 else 0.0
                    feature_columns.append(feature)
                    dummy_count += 1
            # NaN değerleri temizle
            df[feature_columns] = df[feature_columns].fillna(0)
            # Sonsuz değerleri temizle
            df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], 0)
            # Feature kalite oranı hesapla
            total_features = len(feature_columns)
            dummy_ratio = dummy_count / total_features if total_features > 0 else 0
            self.logger.info(f"Feature hazırlama tamamlandı. {len(feature_columns)} feature kullanılıyor. Eksik/dummy oranı: {dummy_ratio:.2%}")
            df['feature_dummy_ratio'] = dummy_ratio
            return df[feature_columns + ['feature_dummy_ratio']]
        except Exception as e:
            self.logger.error(f"Feature hazırlama hatası: {e}")

    def prepare_features(self, df, feature_cols=None, for_training=False):
        """Feature'ları hazırla - Düzeltilmiş"""
        if df is None or df.empty:
            self.logger.error("Input DataFrame is empty or None")
            return None
        # Feature listesini belirle
        if feature_cols is not None:
            trained_features = feature_cols
        else:
            try:
                import joblib
                trained_features = joblib.load('models/feature_cols.pkl')
            except Exception:
                self.logger.error('Feature listesi bulunamadı ve feature_cols parametresi verilmedi!')
                return None
        # DataFrame'in kopyasını al
        df_processed = df.copy()
        # String feature'ları numeric'e çevir
        df_processed = convert_string_features_to_numeric(df_processed)
        # NaN değerleri temizle
        df_processed = df_processed.fillna(0)
        # Sonsuz değerleri temizle
        df_processed = df_processed.replace([np.inf, -np.inf], 0)
        # Eğitim sırasında kullanılan feature sayısını kontrol et
        if hasattr(self, 'rf_model') and self.rf_model is not None:
            expected_features = self.rf_model.n_features_in_
            if len(trained_features) > expected_features:
                trained_features = trained_features[:expected_features]
                self.logger.info(f"Feature sayısı {expected_features}'e sınırlandı")
        # Eksik feature'ları sıfırla doldur
        for col in trained_features:
            if col not in df_processed.columns:
                self.logger.error(f"{col} eksik, feature hazırlama başarısız. Sinyal üretimi iptal edildi.")
                return None
        X = df_processed[trained_features].copy()
        # Scaler uygula
        if self.scaler is not None and not for_training:
            try:
                X_scaled = self.scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            except Exception as e:
                self.logger.error(f"Scaler hatası: {e}")
                return None
        return X

    def train_lstm(self, df, feature_cols=None, epochs=50, batch_size=32):
        """Gelişmiş LSTM modeli eğitir"""
        # self.sync_feature_cols_with_technical_analysis() KALDIRILDI
        # Önce etiketleri oluştur
        df = self.create_labels(df)
        df = self.engineer_features(df)
        
        # Feature'ları hazırla
        X = self.prepare_features(df, feature_cols, for_training=True)
        
        # DataFrame'i numpy array'e çevir
        if hasattr(X, 'values'):
            X = X.values
        
        # Label'ları al
        y = df['label'].values if 'label' in df.columns else np.zeros(len(X))
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # DataFrame ise numpy array'e çevir
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        
        # LSTM için 3D reshape
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        # Gelişmiş model mimarisi
        model = Sequential([
            LSTM(128, input_shape=(1, X.shape[1]), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Optimizer ve callbacks
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        callbacks = [
            ModelCheckpoint(self.lstm_model_path, save_best_only=True, monitor='val_loss', mode='min'),
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Eğitim
        history = model.fit(
            X_train_lstm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_lstm, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Performans değerlendirme
        val_preds = (model.predict(X_val_lstm) > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_preds)
        val_precision = precision_score(y_val, val_preds, zero_division=0)
        val_recall = recall_score(y_val, val_preds, zero_division=0)
        
        self.logger.info(f'LSTM Modeli eğitildi - Val Acc: {val_acc:.3f}, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}')
        return model

    def train_rf(self, df, feature_cols=None):
        """Gelişmiş Random Forest modeli eğitir"""
        # self.sync_feature_cols_with_technical_analysis() KALDIRILDI
        # Önce etiketleri oluştur
        df = self.create_labels(df)
        df = self.engineer_features(df)
        
        # Feature'ları hazırla
        X = self.prepare_features(df, feature_cols, for_training=True)
        
        # Label'ları al
        y = df['label'].values if 'label' in df.columns else np.zeros(len(X))
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter optimization
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        joblib.dump(best_rf, self.rf_model_path)
        
        # Performans değerlendirme
        val_preds = best_rf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_precision = precision_score(y_val, val_preds, zero_division=0)
        val_recall = recall_score(y_val, val_preds, zero_division=0)
        
        self.logger.info(f'RF Modeli eğitildi - Val Acc: {val_acc:.3f}, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}')
        return best_rf

    def train_gradient_boosting(self, df, feature_cols=None):
        """Gradient Boosting modeli eğitir"""
        # self.sync_feature_cols_with_technical_analysis() KALDIRILDI
        # Önce etiketleri oluştur
        df = self.create_labels(df)
        df = self.engineer_features(df)
        
        # Feature'ları hazırla
        X = self.prepare_features(df, feature_cols, for_training=True)
        
        # Label'ları al
        y = df['label'].values if 'label' in df.columns else np.zeros(len(X))
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        joblib.dump(gb, self.gb_model_path)
        
        # Performans değerlendirme
        val_preds = gb.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_precision = precision_score(y_val, val_preds, zero_division=0)
        val_recall = recall_score(y_val, val_preds, zero_division=0)
        
        self.logger.info(f'GB Modeli eğitildi - Val Acc: {val_acc:.3f}, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}')
        return gb

    def _align_features(self, df, trained_features):
        """Feature'ları eğitilmiş model ile uyumlu hale getir"""
        try:
            if trained_features is None:
                return df
            
            # DataFrame'in kopyasını al
            df_aligned = df.copy()
            
            # Scaler'ın beklediği feature sayısını al
            expected_features = 128  # Scaler'ın beklediği feature sayısı
            if self.scaler is not None:
                expected_features = self.scaler.n_features_in_
            
            # trained_features string ise, doğru feature listesini kullan
            if isinstance(trained_features, str):
                self.logger.warning("trained_features string olarak geldi, varsayılan feature listesi kullanılıyor")
                # Varsayılan 99 feature listesi
                trained_features = [
                    'open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 
                    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_histogram', 
                    'rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'bb_middle', 'bb_upper', 'bb_lower', 
                    'bb_width', 'bb_percent', 'atr', 'obv', 'vwap', 'adx_pos', 'adx_neg', 'adx', 'cci', 'mfi', 
                    'williams_r', 'psar', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 
                    'keltner_ema', 'keltner_upper', 'keltner_lower', 'volume_roc', 'volume_ma', 'volume_ratio', 
                    'roc', 'momentum', 'price_roc', 'historical_volatility', 'true_range', 'volatility_ratio', 
                    'price_change', 'price_change_5', 'price_change_10', 'return_5', 'return_10', 'return_20', 
                    'cumulative_return', 'momentum_5', 'momentum_10', 'volatility', 'volatility_5', 'volatility_10', 
                    'volatility_20', 'volume_ma_5', 'volume_ma_10', 'dynamic_threshold', 'label_5', 'label_10', 
                    'label_20', 'label_dynamic', 'day_of_week', 'hour', 'doji', 'hammer', 'shooting_star', 
                    'support_level', 'resistance_level', 'price_vs_support', 'price_vs_resistance', 'volume_sma', 
                    'high_volume', 'body_size', 'upper_shadow', 'lower_shadow', 'momentum_ma', 'momentum_trend', 
                    'breakout_up', 'breakout_down', 'price_range', 'consolidation', 'volatility_ma', 'trend_strength', 
                    'trend_ma', 'range', 'range_ma', 'market_regime', 'price_change_20', 'volatility_50', 'momentum_20', 
                    'volume_ma_20', 'volume_trend', 'rsi_trend', 'rsi_momentum', 'macd_strength', 'macd_trend', 
                    'bb_position', 'bb_squeeze', 'future_close_5', 'future_close_10', 'future_close_20', 'future_close_30', 
                    'return_30', 'volatility_30', 'dynamic_threshold_5', 'dynamic_threshold_10', 'dynamic_threshold_20', 
                    'dynamic_threshold_30', 'label_30', 'trend_5', 'trend_20', 'trend_label', 'momentum_label'
                ]
            
            # Eğitilmiş feature listesi eksikse, eksik feature'ları ekle
            if len(trained_features) < expected_features:
                missing_count = expected_features - len(trained_features)
                self.logger.info(f"Feature listesi eksik, {missing_count} feature ekleniyor")
                
                # Eksik feature'ları oluştur
                for i in range(missing_count):
                    feature_name = f"extra_feature_{i}"
                    trained_features.append(feature_name)
            
            available_features = set(df_aligned.columns)
            required_features = set(trained_features)
            
            # Eksik feature'ları ekle
            missing_features = required_features - available_features
            if missing_features:
                self.logger.info(f"Eksik feature'lar otomatik ekleniyor: {len(missing_features)} adet")
                for feature in missing_features:
                    df_aligned.loc[:, feature] = 0.0  # Varsayılan değer
            
            # Feature sırasını eğitilmiş model ile aynı yap
            # Sadece eğitilmiş modelde olan feature'ları al
            final_features = [f for f in trained_features if f in df_aligned.columns]
            df_aligned = df_aligned[final_features]
            
            # NaN değerleri temizle
            df_aligned = df_aligned.fillna(0)
            
            self.logger.debug(f"Feature alignment tamamlandı. Shape: {df_aligned.shape}, Expected: {expected_features}")
            return df_aligned
            
        except Exception as e:
            self.logger.error(f"Feature alignment hatası: {e}")
            return df

    def predict_lstm(self, X):
        """LSTM tahmini - 1 time step ile"""
        try:
            if self.lstm_model is None:
                raise RuntimeError("LSTM modeli yüklenemedi!")
            if X is None or X.size == 0:
                raise ValueError("LSTM için boş feature array!")
            if hasattr(X, 'values'):
                X = X.values
            # Her durumda (batch, 1, feature) shape'e zorla
            if len(X.shape) == 1:
                X = X.reshape(1, 1, -1)
            elif len(X.shape) == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])
            elif len(X.shape) == 3:
                if X.shape[1] != 1:
                    X = X[:, -1:, :]
            else:
                raise ValueError(f"LSTM için geçersiz shape: {X.shape}")
            expected_shape = (1, 1, self.lstm_model.input_shape[2])
            if X.shape[1:] != expected_shape[1:]:
                raise ValueError(f"LSTM shape uyumsuzluğu: {X.shape} vs {expected_shape}")
            prediction = self.lstm_model.predict(X, verbose=0)
            if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                return float(prediction[0][0])
            elif isinstance(prediction, (float, np.floating)):
                return float(prediction)
            else:
                raise RuntimeError(f"LSTM tahmin çıktısı beklenmeyen tipte: {type(prediction)}")
        except Exception as e:
            self.logger.error(f"LSTM tahmin hatası: {e}")
            raise

    def predict_rf(self, X):
        """Random Forest tahmini - Düzeltilmiş"""
        try:
            if self.rf_model is None:
                return None
            
            # X'in boş olup olmadığını kontrol et
            if X is None or X.size == 0:
                self.logger.warning("RF için boş feature array")
                return 0.5
            
            # X'in shape'ini kontrol et ve düzelt
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            elif len(X.shape) == 3:
                # LSTM'den gelen 3D array'i 2D'ye çevir
                X = X.reshape(X.shape[0], -1)
            
            # Feature sayısını kontrol et
            if X.shape[1] == 0:
                self.logger.warning("RF için 0 feature")
                return 0.5
            
            # Scaler uygula
            if self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    self.logger.warning(f"RF scaler hatası: {e}, unscaled data kullanılıyor")
                    X_scaled = X
            else:
                X_scaled = X
            
            # Tahmin yap
            prediction = self.rf_model.predict_proba(X_scaled)[0]
            return prediction[1] if len(prediction) > 1 else prediction[0]
            
        except Exception as e:
            self.logger.error(f"RF tahmin hatası: {e}")
            return 0.5

    def predict_gb(self, X):
        """Gradient Boosting tahmini - Düzeltilmiş"""
        try:
            if self.gb_model is None:
                return None
            
            # X'in boş olup olmadığını kontrol et
            if X is None or X.size == 0:
                self.logger.warning("GB için boş feature array")
                return 0.5
            
            # X'in shape'ini kontrol et ve düzelt
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            elif len(X.shape) == 3:
                # LSTM'den gelen 3D array'i 2D'ye çevir
                X = X.reshape(X.shape[0], -1)
            
            # Feature sayısını kontrol et
            if X.shape[1] == 0:
                self.logger.warning("GB için 0 feature")
                return 0.5
            
            # Scaler uygula
            if self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    self.logger.warning(f"GB scaler hatası: {e}, unscaled data kullanılıyor")
                    X_scaled = X
            else:
                X_scaled = X
            
            # Tahmin yap
            prediction = self.gb_model.predict_proba(X_scaled)[0]
            return prediction[1] if len(prediction) > 1 else prediction[0]
            
        except Exception as e:
            self.logger.error(f"GB tahmin hatası: {e}")
            return 0.5

    def ensemble_predict(self, X):
        """Custom Ensemble tahmin - Ağırlıklı ortalama ile güçlü sonuç"""
        try:
            predictions = {}
            weights = {}
            lstm_pred = None
            try:
                lstm_pred = self.predict_lstm(X)
            except Exception as e:
                self.logger.error(f"LSTM tahmin hatası (ensemble): {e}")
            rf_pred = None
            try:
                rf_pred = self.predict_rf(X)
            except Exception as e:
                self.logger.error(f"RF tahmin hatası (ensemble): {e}")
            gb_pred = None
            try:
                gb_pred = self.predict_gb(X)
            except Exception as e:
                self.logger.error(f"GB tahmin hatası (ensemble): {e}")
            if lstm_pred is not None:
                predictions['lstm'] = float(lstm_pred)
                weights['lstm'] = 0.3
            if rf_pred is not None:
                predictions['rf'] = float(rf_pred)
                weights['rf'] = 0.4
            if gb_pred is not None:
                predictions['gb'] = float(gb_pred)
                weights['gb'] = 0.3
            if not predictions:
                raise RuntimeError("Hiçbir model tahmin yapamadı! Model veya veri eksik.")
            weighted_sum = 0
            total_weight = 0
            for model, pred in predictions.items():
                weight = weights.get(model, 0.33)
                weighted_sum += pred * weight
                total_weight += weight
            ensemble_pred = float(weighted_sum / total_weight) if total_weight > 0 else 0.6
            if ensemble_pred > 0.3:
                boosted_pred = 0.4 + (ensemble_pred - 0.3) * 0.8
                ensemble_pred = min(0.8, boosted_pred)
            if len(predictions) > 1:
                pred_values = list(predictions.values())
                confidence = 1.0 - (max(pred_values) - min(pred_values))
                confidence = max(0.3, min(1.0, confidence))
            else:
                confidence = 0.6
            return ensemble_pred, confidence
        except Exception as e:
            self.logger.error(f"Ensemble tahmin hatası: {e}")
            raise

    def predict(self, df, trained_features=None):
        try:
            X = self.prepare_features(df, trained_features)
            if X is None or X.empty:
                self.logger.error("Tahmin için feature hazırlama başarısız veya veri eksik. Sinyal üretimi iptal edildi.")
                return {
                    'prediction': None,
                    'confidence': None,
                    'features_used': 0,
                    'model_performance': {},
                    'feature_dummy_ratio': 1.0,
                    'error': 'Feature hazırlama başarısız veya veri eksik.'
                }
            feature_dummy_ratio = 0.0
            if 'feature_dummy_ratio' in X.columns:
                feature_dummy_ratio = float(X['feature_dummy_ratio'].iloc[-1])
                X = X.drop(columns=['feature_dummy_ratio'])
            X_array = X.values
            try:
                prediction, confidence = self.ensemble_predict(X_array)
            except Exception as e:
                return {
                    'prediction': None,
                    'confidence': None,
                    'features_used': X.shape[1],
                    'model_performance': self.model_performance,
                    'feature_dummy_ratio': feature_dummy_ratio,
                    'error': f'Tahmin hatası: {e}'
                }
            return {
                'prediction': float(prediction),
                'confidence': confidence,
                'features_used': X.shape[1],
                'model_performance': self.model_performance,
                'feature_dummy_ratio': feature_dummy_ratio
            }
        except Exception as e:
            self.logger.error(f"Tahmin hatası: {e}")
            return {
                'prediction': None,
                'confidence': None,
                'features_used': 0,
                'model_performance': {},
                'feature_dummy_ratio': 1.0,
                'error': f'Tahmin fonksiyonu genel hata: {e}'
            }

    def calculate_prediction_confidence(self, X, prediction):
        """Tahmin güvenilirliğini hesapla"""
        try:
            # Model performanslarına göre güvenilirlik
            confidence = 0.5  # Varsayılan
            
            # Tahmin değerine göre güvenilirlik
            if prediction > 0.8 or prediction < 0.2:
                confidence += 0.2  # Güçlü tahmin
            elif prediction > 0.7 or prediction < 0.3:
                confidence += 0.1  # Orta güçlü tahmin
            
            # Model sayısına göre güvenilirlik
            active_models = sum(1 for pred in self.model_performance.values() 
                              if isinstance(pred, (int, float)) and pred is not None)
            if active_models >= 2:
                confidence += 0.1
            
            # Model tutarlılığına göre güvenilirlik
            predictions = [pred for pred in self.model_performance.values() 
                         if isinstance(pred, (int, float)) and pred is not None]
            if len(predictions) >= 2:
                std_dev = np.std(predictions)
                if std_dev < 0.1:  # Tutarlı tahminler
                    confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Güvenilirlik hesaplama hatası: {e}")
            return 0.5

    def update_model_weights(self, performance_data):
        """Model ağırlıklarını performansa göre güncelle"""
        try:
            if not performance_data:
                return
            
            # Son performans verilerine göre ağırlıkları ayarla
            total_performance = sum(performance_data.values())
            if total_performance > 0:
                for model, perf in performance_data.items():
                    if model in self.ensemble_weights:
                        self.ensemble_weights[model] = perf / total_performance
                
                # Ağırlıkları normalize et
                total_weight = sum(self.ensemble_weights.values())
                for model in self.ensemble_weights:
                    self.ensemble_weights[model] /= total_weight
                
                logger.info(f"Model ağırlıkları güncellendi: {self.ensemble_weights}")
                
        except Exception as e:
            logger.error(f"Model ağırlık güncelleme hatası: {e}")

    def get_model_stats(self):
        """Model istatistiklerini döndür"""
        return {
            'ensemble_weights': self.ensemble_weights,
            'last_performance': self.model_performance,
            'models_loaded': {
                'lstm': self.lstm_model is not None,
                'rf': self.rf_model is not None,
                'gb': self.gb_model is not None,
                'scaler': self.scaler is not None
            }
        }

    def retrain_daily(self, df, feature_cols=None):
        """Her gün otomatik yeniden eğitim"""
        df = self.create_labels(df)
        self.train_lstm(df, feature_cols)
        self.train_rf(df, feature_cols)
        self.train_gradient_boosting(df, feature_cols)
        self.logger.info('AI modelleri günlük olarak yeniden eğitildi.')

    def evaluate(self, df, feature_cols=None):
        """Modelin performansını değerlendirir"""
        df = self.create_labels(df)
        X, y, feature_cols = self.prepare_features(df, feature_cols)
        # LSTM
        if os.path.exists(self.lstm_model_path):
            model = load_model(self.lstm_model_path)
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            preds = (model.predict(X_lstm) > 0.5).astype(int)
            acc = accuracy_score(y, preds)
            self.logger.info(f'LSTM Doğruluk: {acc:.2f}')
        # RF
        if os.path.exists(self.rf_model_path):
            rf = joblib.load(self.rf_model_path)
            preds = rf.predict(X)
            acc = accuracy_score(y, preds)
            self.logger.info(f'RF Doğruluk: {acc:.2f}')
        # GB
        if os.path.exists(self.gb_model_path):
            gb = joblib.load(self.gb_model_path)
            preds = gb.predict(X)
            acc = accuracy_score(y, preds)
            self.logger.info(f'GB Doğruluk: {acc:.2f}')

    def create_ensemble_model(self):
        """Ensemble model oluştur - daha iyi tahmin için"""
        # Base models
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
        svm = SVC(probability=True, kernel='rbf', random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        
        # Ensemble model
        self.ensemble_model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm), ('mlp', mlp)],
            voting='soft'
        )

    def feature_selection(self, X, y):
        """Önemli feature'ları seç - gürültüyü azalt"""
        # Statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Recursive feature elimination
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=rf, n_features_to_select=min(30, X.shape[1]))
        X_rfe = rfe.fit_transform(X_selected, y)
        
        self.feature_selector = (selector, rfe)
        return X_rfe
        
    def optimize_hyperparameters(self, X, y):
        """Hyperparameter optimization - daha iyi model performansı"""
        # Grid search parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_

    def retrain_model_with_new_features(self, data, labels):
        """Yeni feature'lar ile modeli yeniden eğit"""
        try:
            self.logger.info("Model yeni feature'lar ile yeniden eğitiliyor...")
            
            # Feature selection
            if Config.FEATURE_SELECTION_ENABLED:
                self.feature_selector = SelectKBest(score_func=f_classif, k=min(50, len(data.columns)))
                data_selected = self.feature_selector.fit_transform(data, labels)
                selected_features = data.columns[self.feature_selector.get_support()]
                self.logger.info(f"Seçilen feature'lar: {len(selected_features)} adet")
            else:
                data_selected = data
                selected_features = data.columns
            
            # Model eğitimi
            if Config.USE_ENSEMBLE_MODEL:
                self.create_ensemble_model()
                self.ensemble_model.fit(data_selected, labels)
                self.logger.info("Ensemble model eğitildi")
            else:
                # Gradient Boosting
                self.models['gb'] = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                self.models['gb'].fit(data_selected, labels)
                self.logger.info("Gradient Boosting model eğitildi")
            
            # Feature importance
            if hasattr(self.models.get('gb'), 'feature_importances_'):
                self.feature_importance = dict(zip(selected_features, self.models['gb'].feature_importances_))
            
            # Model kaydet
            self.save_models()
            self.logger.info("Model başarıyla yeniden eğitildi ve kaydedildi")
            
        except Exception as e:
            self.logger.error(f"Model yeniden eğitimi hatası: {e}")

    def update_feature_columns(self, new_features):
        """Feature sütunlarını güncelle"""
        try:
            # Mevcut feature'ları al
            current_features = self.get_feature_columns()
            # Yeni feature'ları ekle
            updated_features = list(set(current_features + new_features))
            # Feature sütunlarını kaydetme işlemi kaldırıldı
            self.logger.info(f"Feature sütunları güncellendi: {len(updated_features)} adet (feature_cols.pkl dosyası retrain_models.py tarafından yazılır)")
        except Exception as e:
            self.logger.error(f"Feature sütunları güncelleme hatası: {e}")

    def save_models(self):
        """Tüm modelleri ve özellikleri kaydet"""
        try:
            joblib.dump(self.models, self.model_dir + '/models.pkl')
            joblib.dump(self.feature_selector, self.model_dir + '/feature_selector.pkl')
            joblib.dump(self.feature_importance, self.model_dir + '/feature_importance.pkl')
            joblib.dump(self.ensemble_model, self.model_dir + '/ensemble_model.pkl')
            joblib.dump(self.scaler, self.model_dir + '/scaler.pkl')
            # Feature columns listesini kaydetme işlemi kaldırıldı
            self.logger.info("Tüm modeller ve özellikler başarıyla kaydedildi")
        except Exception as e:
            self.logger.error(f"Model kaydetme hatası: {e}")

    def get_feature_columns(self):
        """Mevcut feature sütunlarını al"""
        try:
            if os.path.exists(self.feature_cols_path):
                with open(self.feature_cols_path, 'rb') as f:
                    return pickle.load(f)
            else:
                self.logger.warning('Feature columns bulunamadı.')
                return []
        except Exception as e:
            self.logger.error(f"Feature columns yükleme hatası: {e}")
            return []

    def adaptive_retrain(self, performance_threshold=0.6, min_signals=50):
        """Performansa göre otomatik model yeniden eğitimi"""
        try:
            # Performans kontrolü
            current_performance = self.get_current_performance()
            
            if current_performance['total_signals'] < min_signals:
                self.logger.info(f"Yeterli sinyal yok ({current_performance['total_signals']}/{min_signals})")
                return False
            
            success_rate = current_performance['success_rate']
            avg_pnl = current_performance['avg_pnl']
            
            # Performans kriterleri
            performance_score = (success_rate * 0.7) + (min(avg_pnl * 10, 0.3) * 0.3)
            
            if performance_score < performance_threshold:
                self.logger.info(f"Performans düşük ({performance_score:.3f} < {performance_threshold}), model yeniden eğitiliyor...")
                
                # Yeni veri topla
                data_collector = DataCollector()
                new_data = data_collector.collect_extended_data(days=180)
                
                if new_data is not None and not new_data.empty:
                    # Feature'ları hazırla
                    new_data = self.prepare_features_for_training(new_data)
                    
                    if new_data is not None and not new_data.empty:
                        # Modelleri yeniden eğit
                        self.retrain_models(new_data)
                        
                        # Performansı sıfırla
                        self.reset_performance_tracking()
                        
                        self.logger.info("Model başarıyla yeniden eğitildi")
                        return True
                    else:
                        self.logger.error("Yeni veri hazırlanamadı")
                        return False
                else:
                    self.logger.error("Yeni veri toplanamadı")
                    return False
            else:
                self.logger.info(f"Performans yeterli ({performance_score:.3f} >= {performance_threshold})")
                return False
                
        except Exception as e:
            self.logger.error(f"Adaptive retrain hatası: {e}")
            return False
    
    def get_current_performance(self):
        """Mevcut model performansını al"""
        try:
            # Veritabanından performans verilerini al
            query = """
                SELECT 
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as successful_signals,
                    AVG(final_pnl) as avg_pnl,
                    SUM(final_pnl) as total_pnl
                FROM signal_performance 
                WHERE status = 'CLOSED'
            """
            
            result = self.engine.execute(query).fetchone()
            
            if result and result[0] > 0:
                success_rate = result[1] / result[0]
                return {
                    'total_signals': result[0],
                    'successful_signals': result[1],
                    'success_rate': success_rate,
                    'avg_pnl': result[2] or 0,
                    'total_pnl': result[3] or 0
                }
            else:
                return {
                    'total_signals': 0,
                    'successful_signals': 0,
                    'success_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0
                }
                
        except Exception as e:
            self.logger.error(f"Performans alma hatası: {e}")
            return {
                'total_signals': 0,
                'successful_signals': 0,
                'success_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0
            }
    
    def reset_performance_tracking(self):
        """Performans takibini sıfırla"""
        try:
            # Performans tablosunu temizle
            self.engine.execute("DELETE FROM signal_performance WHERE status = 'CLOSED'")
            self.logger.info("Performans takibi sıfırlandı")
            
        except Exception as e:
            self.logger.error(f"Performans sıfırlama hatası: {e}")
    
    def retrain_models(self, df):
        """Modelleri yeniden eğit"""
        try:
            self.logger.info("Modeller yeniden eğitiliyor...")
            
            # Feature'ları hazırla
            feature_cols = [col for col in df.columns if col not in ['symbol', 'timeframe', 'timestamp', 'label']]
            
            # NaN değerleri temizle
            df = df.dropna()
            
            if df.empty:
                self.logger.error("Eğitim verisi boş")
                return False
            
            # Veriyi böl
            X = df[feature_cols]
            y = df['label']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 1. LSTM Modeli
            self.logger.info("LSTM modeli eğitiliyor...")
            self.train_lstm_model(X_train, y_train, X_test, y_test)
            
            # 2. Random Forest
            self.logger.info("Random Forest modeli eğitiliyor...")
            self.train_rf_model(X_train, y_train, X_test, y_test)
            
            # 3. Gradient Boosting
            self.logger.info("Gradient Boosting modeli eğitiliyor...")
            self.train_gb_model(X_train, y_train, X_test, y_test)
            
            # Feature listesini kaydet
            with open('models/feature_cols.pkl', 'wb') as f:
                pickle.dump(feature_cols, f)
            
            # Scaler'ı da kaydet
            if self.scaler is not None:
                with open('models/scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            self.logger.info("Tüm modeller başarıyla yeniden eğitildi")
            return True
            
        except Exception as e:
            self.logger.error(f"Model yeniden eğitimi hatası: {e}")
            return False
    
    def train_lstm_model(self, X_train, y_train, X_test, y_test):
        """LSTM modelini eğit - 1 time step ile"""
        try:
            # Veriyi LSTM formatına çevir (1 time step)
            X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            # Model mimarisi - 1 time step için
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(1, X_train.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Eğitim
            history = model.fit(
                X_train_lstm, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_lstm, y_test),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Modeli kaydet
            model.save('models/lstm_model.h5')
            
            # Performans değerlendirme
            y_pred = model.predict(X_test_lstm)
            accuracy = accuracy_score(y_test, y_pred.round())
            self.logger.info(f"LSTM Model Accuracy: {accuracy:.3f}")
            self.logger.info(f"LSTM Model Input Shape: {model.input_shape}")
            
        except Exception as e:
            self.logger.error(f"LSTM eğitim hatası: {e}")
    
    def train_rf_model(self, X_train, y_train, X_test, y_test):
        """Random Forest modelini eğit"""
        try:
            # Hiperparametre optimizasyonu
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # En iyi modeli kaydet
            best_rf = grid_search.best_estimator_
            with open('models/rf_model.pkl', 'wb') as f:
                pickle.dump(best_rf, f)
            
            # Performans değerlendirme
            y_pred = best_rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.logger.info(f"Random Forest Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Random Forest eğitim hatası: {e}")
    
    def train_gb_model(self, X_train, y_train, X_test, y_test):
        """Gradient Boosting modelini eğit"""
        try:
            # Hiperparametre optimizasyonu
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # En iyi modeli kaydet
            best_gb = grid_search.best_estimator_
            with open('models/gb_model.pkl', 'wb') as f:
                pickle.dump(best_gb, f)
            
            # Performans değerlendirme
            y_pred = best_gb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.logger.info(f"Gradient Boosting Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Gradient Boosting eğitim hatası: {e}")
    
    def prepare_lstm_data(self, X, y, lookback=60):
        """LSTM için veriyi hazırla"""
        try:
            X_lstm, y_lstm = [], []
            
            for i in range(lookback, len(X)):
                X_lstm.append(X.iloc[i-lookback:i].values)
                y_lstm.append(y.iloc[i])
            
            return np.array(X_lstm), np.array(y_lstm)
            
        except Exception as e:
            self.logger.error(f"LSTM veri hazırlama hatası: {e}")
            return np.array([]), np.array([]) 

    def predict_signal(self, df, trained_features=None):
        try:
            if df is None or df.empty:
                self.logger.error("Input DataFrame is empty or None")
                return None, None, None
            X = self.prepare_features(df, trained_features, for_training=False)
            if X is None or X.empty:
                self.logger.error("Feature hazırlama başarısız. Sinyal üretimi iptal edildi.")
                return None, None, None
            X_latest = X.iloc[-1:].values
            prediction, confidence = self.ensemble_predict(X_latest)
            if prediction is None:
                self.logger.error("Model tahmini başarısız. Sinyal üretimi iptal edildi.")
                return None, None, None
            if prediction > 0.5:
                signal_direction = "LONG"
                signal_strength = prediction
            else:
                signal_direction = "SHORT"
                signal_strength = 1 - prediction
            self.logger.info(f"Sinyal tahmini: {signal_direction}, Güç: {signal_strength:.3f}, Güven: {confidence:.3f}")
            return signal_direction, signal_strength, confidence
        except Exception as e:
            self.logger.error(f"predict_signal hatası: {e}")
            return None, None, None

    def predict_simple(self, df):
        import joblib
        import pickle
        import numpy as np
        rf = joblib.load('models/simple_rf_model.pkl')
        with open('models/simple_rf_features.pkl', 'rb') as f:
            features = pickle.load(f)
        def engineer_features(df):
            df = df.copy()
            if 'close' in df.columns:
                df['price_change_5'] = df['close'].pct_change(5)
                df['volatility_5'] = df['close'].pct_change().rolling(5).std()
                df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            if 'volume' in df.columns:
                df['volume_ma_5'] = df['volume'].rolling(5).mean()
                df['volume_ratio'] = df['volume'] / (df['volume_ma_5'] + 1e-6)
            return df
        df = engineer_features(df)
        if any([feature not in df.columns for feature in features]):
            missing = [feature for feature in features if feature not in df.columns]
            self.logger.error(f"Eksik feature'lar: {missing}. Sinyal üretimi iptal edildi.")
            return {
                'prediction': None,
                'confidence': None,
                'features_used': 0,
                'model': 'simple_rf_model',
                'error': f'Eksik feature: {missing}'
            }
        X = df[features].fillna(0).tail(1).values
        ai_score = rf.predict_proba(X)[0][1]
        return {
            'prediction': ai_score,
            'confidence': float(np.clip(ai_score, 0, 1)),
            'features_used': len(features),
            'model': 'simple_rf_model'
        }

    # feature_cols ve varsayılan trained_features listesinden aşağıdaki dummy/yanlış feature'ları çıkar:
    # 'feature_62', 'feature_63', ..., 'feature_124' ve benzeri anlamsız isimler
   