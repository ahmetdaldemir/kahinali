#!/usr/bin/env python3
"""
Kahin Ultima - 6 Aylık Verilerle AI Model Yeniden Eğitimi
Toplanan 6 aylık verilerle AI modellerini güncelleyerek performansı artırır
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_retraining_6months.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self):
        self.data_dir = 'data'
        self.models_dir = 'models'
        self.timeframes = ['1h', '4h', '1d']
        self.feature_columns = None
        self.scaler = None
        self.feature_selector = None
        
        # Model parametreleri
        self.lstm_units = 64  # Daha küçük model
        self.lstm_dropout = 0.2
        self.lstm_learning_rate = 0.001
        self.lstm_epochs = 50  # Daha az epoch
        self.lstm_batch_size = 64  # Daha büyük batch size
        
        # Feature selection
        self.n_features = 50  # Daha az feature seç
        
        # Labeling parametreleri
        self.future_periods = 24  # 24 saat sonrası için tahmin
        self.profit_threshold = 0.02  # %2 kar eşiği
        self.loss_threshold = -0.015  # %1.5 zarar eşiği
        
        # Directory'leri oluştur
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_6months_data(self):
        """6 aylık verileri yükle ve birleştir"""
        logger.info("6 aylık veriler yükleniyor...")
        
        all_data = []
        coin_count = 0
        
        # Tüm 6 aylık veri dosyalarını tara
        for file in os.listdir(self.data_dir):
            if file.endswith('_6months.csv'):
                coin_name = file.replace('_1h_6months.csv', '').replace('_4h_6months.csv', '').replace('_1d_6months.csv', '')
                timeframe = file.split('_')[-2] if '_' in file else '1h'
                
                filepath = os.path.join(self.data_dir, file)
                try:
                    df = pd.read_csv(filepath)
                    df['coin'] = coin_name
                    df['timeframe'] = timeframe
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Veri kalitesi kontrolü
                    if len(df) >= 100:  # Minimum 100 satır
                        all_data.append(df)
                        coin_count += 1
                        
                except Exception as e:
                    logger.warning(f"{file} yüklenirken hata: {e}")
        
        if not all_data:
            logger.error("Hiç veri yüklenemedi!")
            return None
        
        # Tüm verileri birleştir
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Toplam {len(combined_df)} satır veri yüklendi ({coin_count} coin)")
        
        return combined_df
    
    def create_features(self, df):
        """Teknik analiz özellikleri oluştur"""
        logger.info("Teknik analiz özellikleri oluşturuluyor...")
        
        features_df = df.copy()
        
        # Temel özellikler
        features_df['price_change'] = features_df['close'].pct_change()
        features_df['volume_change'] = features_df['volume'].pct_change()
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        features_df['close_open_ratio'] = features_df['close'] / features_df['open']
        
        # Hareketli ortalamalar
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = features_df['close'].rolling(period).mean()
            features_df[f'ema_{period}'] = features_df['close'].ewm(span=period).mean()
            features_df[f'price_sma_{period}_ratio'] = features_df['close'] / features_df[f'sma_{period}']
        
        # RSI
        delta = features_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = features_df['close'].ewm(span=12).mean()
        ema26 = features_df['close'].ewm(span=26).mean()
        features_df['macd'] = ema12 - ema26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # Bollinger Bands
        features_df['bb_middle'] = features_df['close'].rolling(20).mean()
        bb_std = features_df['close'].rolling(20).std()
        features_df['bb_upper'] = features_df['bb_middle'] + (bb_std * 2)
        features_df['bb_lower'] = features_df['bb_middle'] - (bb_std * 2)
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Volatilite
        features_df['volatility'] = features_df['close'].rolling(20).std()
        features_df['atr'] = self.calculate_atr(features_df)
        
        # Momentum göstergeleri
        for period in [5, 10, 20]:
            features_df[f'momentum_{period}'] = features_df['close'] - features_df['close'].shift(period)
            features_df[f'roc_{period}'] = features_df['close'].pct_change(period)
        
        # Volume göstergeleri
        features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
        
        # Zaman özellikleri
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['month'] = features_df['timestamp'].dt.month
        
        # NaN değerleri temizle
        features_df = features_df.dropna()
        
        # Veri temizleme - sonsuz değerleri temizle
        logger.info("Özellik verilerinde sonsuz değerler temizleniyor...")
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Tekrar NaN temizleme
        features_df = features_df.dropna()
        
        logger.info(f"Özellik oluşturma tamamlandı. Toplam {len(features_df)} satır")
        return features_df
    
    def calculate_atr(self, df, period=14):
        """Average True Range hesapla"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def create_labels(self, df):
        """Gelecek fiyat hareketlerine göre etiketler oluştur"""
        logger.info("Etiketler oluşturuluyor...")
        
        # Gelecek fiyat değişimlerini hesapla
        future_returns = []
        for i in range(len(df) - self.future_periods):
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + self.future_periods]['close']
            return_pct = (future_price - current_price) / current_price
            future_returns.append(return_pct)
        
        # Eksik değerleri doldur
        future_returns.extend([0] * self.future_periods)
        
        # Etiketleri oluştur
        labels = []
        for ret in future_returns:
            if ret > self.profit_threshold:
                labels.append(1)  # Al sinyali
            elif ret < self.loss_threshold:
                labels.append(-1)  # Sat sinyali
            else:
                labels.append(0)  # Nötr
        
        df['label'] = labels
        df['future_return'] = future_returns
        
        # Etiket dağılımını logla
        label_counts = df['label'].value_counts()
        logger.info(f"Etiket dağılımı: {label_counts.to_dict()}")
        
        return df
    
    def select_features(self, df):
        """En önemli özellikleri seç"""
        logger.info("Özellik seçimi yapılıyor...")
        
        # Sayısal özellikleri seç
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_columns if col not in ['label', 'future_return']]
        
        X = df[feature_cols].copy()
        
        # Veri temizleme - sonsuz değerleri ve çok büyük değerleri temizle
        logger.info("Veri temizleme yapılıyor...")
        
        # Sonsuz değerleri NaN ile değiştir
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Çok büyük değerleri (outlier) temizle
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32']:
                # 99.9% quantile'ı aşan değerleri temizle
                q99 = X[col].quantile(0.999)
                q01 = X[col].quantile(0.001)
                X[col] = X[col].clip(lower=q01, upper=q99)
        
        # NaN değerleri 0 ile doldur
        X = X.fillna(0)
        
        # Veri kalitesi kontrolü
        logger.info(f"Veri temizleme sonrası: {X.shape}")
        logger.info(f"NaN değerler: {X.isna().sum().sum()}")
        logger.info(f"Sonsuz değerler: {np.isinf(X.values).sum()}")
        
        y = df['label']
        
        # Feature selection - mevcut özellik sayısına göre ayarla
        n_available_features = X.shape[1]
        k_features = min(self.n_features, n_available_features)
        
        logger.info(f"Mevcut özellik sayısı: {n_available_features}, seçilecek özellik sayısı: {k_features}")
        
        if k_features == n_available_features:
            # Tüm özellikleri kullan
            X_selected = X.values
            selected_features = X.columns.tolist()
        else:
            # Feature selection yap
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        self.feature_columns = selected_features
        
        logger.info(f"En önemli {len(selected_features)} özellik seçildi")
        
        return X_selected, selected_features
    
    def prepare_lstm_data(self, df, feature_cols):
        """LSTM için veri hazırla"""
        logger.info("LSTM verisi hazırlanıyor...")
        
        # Veri boyutunu sınırla (bellek sorunu için)
        max_samples = 100000  # Maksimum 100K örnek
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.info(f"Veri boyutu {max_samples} örneğe sınırlandı")
        
        # Özellik verilerini hazırla
        X = df[feature_cols].values
        y = df['label'].values
        
        # Normalize et
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # LSTM için sequence oluştur (son 30 veri noktası - daha kısa)
        sequence_length = 30
        X_lstm, y_lstm = [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_lstm.append(X_scaled[i-sequence_length:i])
            y_lstm.append(y[i])
        
        X_lstm = np.array(X_lstm, dtype=np.float32)  # float32 kullan (daha az bellek)
        y_lstm = np.array(y_lstm, dtype=np.float32)
        
        logger.info(f"LSTM veri shape: X={X_lstm.shape}, y={y_lstm.shape}")
        logger.info(f"LSTM veri boyutu: {X_lstm.nbytes / 1024**3:.2f} GB")
        
        return X_lstm, y_lstm
    
    def train_models(self, X, y, X_lstm, y_lstm):
        """Tüm modelleri eğit"""
        logger.info("Modeller eğitiliyor...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
        
        # 1. Random Forest
        logger.info("Random Forest eğitiliyor...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # 2. Gradient Boosting
        logger.info("Gradient Boosting eğitiliyor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        
        # 3. LSTM
        logger.info("LSTM eğitiliyor...")
        lstm_model = self.create_lstm_model(X_lstm.shape[1], X_lstm.shape[2])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # LSTM eğitimi
        lstm_history = lstm_model.fit(
            X_lstm_train, y_lstm_train,
            epochs=self.lstm_epochs,
            batch_size=self.lstm_batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Model performanslarını değerlendir
        self.evaluate_models(rf_model, gb_model, lstm_model, X_test, y_test, X_lstm_test, y_lstm_test)
        
        # Modelleri kaydet
        self.save_models(rf_model, gb_model, lstm_model)
        
        return rf_model, gb_model, lstm_model
    
    def create_lstm_model(self, n_features, sequence_length):
        """LSTM model oluştur"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(self.lstm_dropout),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.lstm_dropout),
            Dense(self.lstm_units // 4, activation='relu'),
            Dropout(self.lstm_dropout),
            Dense(1, activation='tanh')  # -1 ile 1 arası çıktı
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.lstm_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def evaluate_models(self, rf_model, gb_model, lstm_model, X_test, y_test, X_lstm_test, y_lstm_test):
        """Model performanslarını değerlendir"""
        logger.info("Model performansları değerlendiriliyor...")
        
        # Random Forest
        rf_pred = rf_model.predict(X_test)
        rf_mse = np.mean((rf_pred - y_test) ** 2)
        rf_mae = np.mean(np.abs(rf_pred - y_test))
        
        # Gradient Boosting
        gb_pred = gb_model.predict(X_test)
        gb_mse = np.mean((gb_pred - y_test) ** 2)
        gb_mae = np.mean(np.abs(gb_pred - y_test))
        
        # LSTM
        lstm_pred = lstm_model.predict(X_lstm_test).flatten()
        lstm_mse = np.mean((lstm_pred - y_lstm_test) ** 2)
        lstm_mae = np.mean(np.abs(lstm_pred - y_lstm_test))
        
        logger.info(f"Random Forest - MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")
        logger.info(f"Gradient Boosting - MSE: {gb_mse:.4f}, MAE: {gb_mae:.4f}")
        logger.info(f"LSTM - MSE: {lstm_mse:.4f}, MAE: {lstm_mae:.4f}")
        
        # Feature importance (Random Forest)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("En önemli 10 özellik:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def save_models(self, rf_model, gb_model, lstm_model):
        """Modelleri kaydet"""
        logger.info("Modeller kaydediliyor...")
        
        # Backup mevcut modelleri
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Yeni modelleri kaydet
        joblib.dump(rf_model, os.path.join(self.models_dir, 'random_forest_model.pkl'))
        joblib.dump(gb_model, os.path.join(self.models_dir, 'gradient_boosting_model.pkl'))
        lstm_model.save(os.path.join(self.models_dir, 'lstm_model.h5'))
        
        # Feature bilgilerini kaydet
        joblib.dump(self.feature_columns, os.path.join(self.models_dir, 'feature_columns.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        joblib.dump(self.feature_selector, os.path.join(self.models_dir, 'feature_selector.pkl'))
        
        # Ensemble model oluştur
        ensemble_model = {
            'rf_model': rf_model,
            'gb_model': gb_model,
            'lstm_model': lstm_model,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector
        }
        
        joblib.dump(ensemble_model, os.path.join(self.models_dir, 'ensemble_model.pkl'))
        
        logger.info("Tüm modeller başarıyla kaydedildi!")
    
    def retrain_all(self):
        """Tüm yeniden eğitim sürecini çalıştır"""
        logger.info("=== 6 AYLIK VERİLERLE MODEL YENİDEN EĞİTİMİ BAŞLADI ===")
        start_time = datetime.now()
        
        try:
            # 1. Veri yükle
            df = self.load_6months_data()
            if df is None:
                return
            
            # 2. Özellik oluştur
            df = self.create_features(df)
            
            # 3. Etiket oluştur
            df = self.create_labels(df)
            
            # 4. Özellik seç
            X, feature_cols = self.select_features(df)
            
            # 5. LSTM verisi hazırla
            X_lstm, y_lstm = self.prepare_lstm_data(df, feature_cols)
            
            # 6. Modelleri eğit
            rf_model, gb_model, lstm_model = self.train_models(X, df['label'], X_lstm, y_lstm)
            
            # 7. Sonuç raporu
            total_time = datetime.now() - start_time
            logger.info("=== MODEL YENİDEN EĞİTİMİ TAMAMLANDI ===")
            logger.info(f"Toplam süre: {total_time}")
            logger.info(f"Kullanılan veri: {len(df)} satır")
            logger.info(f"Özellik sayısı: {len(feature_cols)}")
            logger.info("Modeller başarıyla güncellendi!")
            
        except Exception as e:
            logger.error(f"Yeniden eğitim sırasında hata: {e}")
            raise

def main():
    """Ana fonksiyon"""
    try:
        retrainer = ModelRetrainer()
        retrainer.retrain_all()
        
        logger.info("Model yeniden eğitimi tamamlandı!")
        
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}")
        raise

if __name__ == "__main__":
    main() 