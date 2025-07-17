#!/usr/bin/env python3
"""
Kahin Ultima - Hızlı Model Yeniden Eğitimi
6 aylık verilerle hızlı ve optimize edilmiş model eğitimi
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quick_retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuickModelRetrainer:
    def __init__(self):
        self.data_dir = 'data'
        self.models_dir = 'models'
        self.max_samples = 50000  # Daha küçük veri seti
        self.n_features = 30  # Daha az feature
        
        # Model parametreleri
        self.rf_n_estimators = 50  # Daha az ağaç
        self.gb_n_estimators = 50
        
        # Labeling parametreleri
        self.future_periods = 12  # 12 saat sonrası
        self.profit_threshold = 0.015  # %1.5 kar
        self.loss_threshold = -0.01  # %1 zarar
        
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_sample_data(self):
        """Örnek veri yükle"""
        logger.info("Örnek veri yükleniyor...")
        
        all_data = []
        sample_coins = 0
        
        # İlk 100 coin'den örnek al
        for file in os.listdir(self.data_dir):
            if file.endswith('_1h_6months.csv') and sample_coins < 100:
                coin_name = file.replace('_1h_6months.csv', '')
                filepath = os.path.join(self.data_dir, file)
                
                try:
                    df = pd.read_csv(filepath)
                    df['coin'] = coin_name
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    if len(df) >= 100:
                        all_data.append(df)
                        sample_coins += 1
                        
                except Exception as e:
                    logger.warning(f"{file} yüklenirken hata: {e}")
        
        if not all_data:
            logger.error("Hiç veri yüklenemedi!")
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Toplam {len(combined_df)} satır veri yüklendi ({sample_coins} coin)")
        
        return combined_df
    
    def create_basic_features(self, df):
        """Temel özellikler oluştur"""
        logger.info("Temel özellikler oluşturuluyor...")
        
        features_df = df.copy()
        
        # Temel özellikler
        features_df['price_change'] = features_df['close'].pct_change()
        features_df['volume_change'] = features_df['volume'].pct_change()
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        features_df['close_open_ratio'] = features_df['close'] / features_df['open']
        
        # Hareketli ortalamalar
        for period in [5, 10, 20]:
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
        
        # Bollinger Bands
        features_df['bb_middle'] = features_df['close'].rolling(20).mean()
        bb_std = features_df['close'].rolling(20).std()
        features_df['bb_upper'] = features_df['bb_middle'] + (bb_std * 2)
        features_df['bb_lower'] = features_df['bb_middle'] - (bb_std * 2)
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Volatilite
        features_df['volatility'] = features_df['close'].rolling(20).std()
        
        # Momentum
        for period in [5, 10]:
            features_df[f'momentum_{period}'] = features_df['close'] - features_df['close'].shift(period)
            features_df[f'roc_{period}'] = features_df['close'].pct_change(period)
        
        # Volume
        features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
        
        # Zaman
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        
        # Veri temizleme
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        
        # Veri boyutunu sınırla
        if len(features_df) > self.max_samples:
            features_df = features_df.sample(n=self.max_samples, random_state=42)
            logger.info(f"Veri boyutu {self.max_samples} örneğe sınırlandı")
        
        logger.info(f"Özellik oluşturma tamamlandı. Toplam {len(features_df)} satır")
        return features_df
    
    def create_labels(self, df):
        """Etiketler oluştur"""
        logger.info("Etiketler oluşturuluyor...")
        
        future_returns = []
        for i in range(len(df) - self.future_periods):
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + self.future_periods]['close']
            return_pct = (future_price - current_price) / current_price
            future_returns.append(return_pct)
        
        future_returns.extend([0] * self.future_periods)
        
        labels = []
        for ret in future_returns:
            if ret > self.profit_threshold:
                labels.append(1)
            elif ret < self.loss_threshold:
                labels.append(-1)
            else:
                labels.append(0)
        
        df['label'] = labels
        df['future_return'] = future_returns
        
        label_counts = df['label'].value_counts()
        logger.info(f"Etiket dağılımı: {label_counts.to_dict()}")
        
        return df
    
    def select_features(self, df):
        """Özellik seçimi"""
        logger.info("Özellik seçimi yapılıyor...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_columns if col not in ['label', 'future_return']]
        
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # En önemli özellikleri seç
        if len(feature_cols) > self.n_features:
            # Basit korelasyon bazlı seçim
            correlations = []
            for col in feature_cols:
                corr = abs(df[col].corr(df['label']))
                correlations.append((col, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_features = [col for col, _ in correlations[:self.n_features]]
        else:
            selected_features = feature_cols
        
        self.feature_columns = selected_features
        X_selected = X[selected_features].values
        
        logger.info(f"{len(selected_features)} özellik seçildi")
        return X_selected, selected_features
    
    def train_models(self, X, y):
        """Modelleri eğit"""
        logger.info("Modeller eğitiliyor...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Random Forest
        logger.info("Random Forest eğitiliyor...")
        rf_model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10  # Derinliği sınırla
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # 2. Gradient Boosting
        logger.info("Gradient Boosting eğitiliyor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=self.gb_n_estimators, 
            random_state=42,
            max_depth=6  # Derinliği sınırla
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # Performans değerlendir
        self.evaluate_models(rf_model, gb_model, X_test_scaled, y_test)
        
        # Modelleri kaydet
        self.save_models(rf_model, gb_model)
        
        return rf_model, gb_model
    
    def evaluate_models(self, rf_model, gb_model, X_test, y_test):
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
        
        logger.info(f"Random Forest - MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")
        logger.info(f"Gradient Boosting - MSE: {gb_mse:.4f}, MAE: {gb_mae:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("En önemli 10 özellik:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def save_models(self, rf_model, gb_model):
        """Modelleri kaydet"""
        logger.info("Modeller kaydediliyor...")
        
        # Yeni modelleri kaydet
        joblib.dump(rf_model, os.path.join(self.models_dir, 'random_forest_model.pkl'))
        joblib.dump(gb_model, os.path.join(self.models_dir, 'gradient_boosting_model.pkl'))
        
        # Feature bilgilerini kaydet
        joblib.dump(self.feature_columns, os.path.join(self.models_dir, 'feature_columns.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        
        # Ensemble model oluştur
        ensemble_model = {
            'rf_model': rf_model,
            'gb_model': gb_model,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        
        joblib.dump(ensemble_model, os.path.join(self.models_dir, 'ensemble_model.pkl'))
        
        logger.info("Modeller başarıyla kaydedildi!")
    
    def retrain_all(self):
        """Tüm süreci çalıştır"""
        logger.info("=== HIZLI MODEL YENİDEN EĞİTİMİ BAŞLADI ===")
        start_time = datetime.now()
        
        try:
            # 1. Veri yükle
            df = self.load_sample_data()
            if df is None:
                return
            
            # 2. Özellik oluştur
            df = self.create_basic_features(df)
            
            # 3. Etiket oluştur
            df = self.create_labels(df)
            
            # 4. Özellik seç
            X, feature_cols = self.select_features(df)
            
            # 5. Modelleri eğit
            rf_model, gb_model = self.train_models(X, df['label'])
            
            # 6. Sonuç raporu
            total_time = datetime.now() - start_time
            logger.info("=== HIZLI MODEL YENİDEN EĞİTİMİ TAMAMLANDI ===")
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
        retrainer = QuickModelRetrainer()
        retrainer.retrain_all()
        
        logger.info("Hızlı model yeniden eğitimi tamamlandı!")
        
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}")
        raise

if __name__ == "__main__":
    main() 