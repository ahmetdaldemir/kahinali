#!/usr/bin/env python3
"""
KAHİN ULTIMA - Kapsamlı AI Model Eğitimi
Tüm modelleri (LSTM, Random Forest, Gradient Boosting) eksiksiz eğitir
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import gc
import joblib

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.ai_model import AIModel
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_ai_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteAITraining:
    """Kapsamlı AI Model Eğitimi"""
    
    def __init__(self):
        self.logger = logger
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model dosya yolları
        self.lstm_model_path = os.path.join(self.models_dir, 'lstm_model.h5')
        self.rf_model_path = os.path.join(self.models_dir, 'random_forest_model.pkl')
        self.gb_model_path = os.path.join(self.models_dir, 'gradient_boosting_model.pkl')
        self.ensemble_model_path = os.path.join(self.models_dir, 'ensemble_model.pkl')
        self.feature_cols_path = os.path.join(self.models_dir, 'feature_cols.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        
    def prepare_training_data(self):
        """Eğitim verisini hazırla"""
        self.logger.info("📊 Eğitim verisi hazırlanıyor...")
        
        try:
            # Veri dosyasını kontrol et
            data_file = 'train_data_with_target.csv'
            if not os.path.exists(data_file):
                self.logger.error(f"Veri dosyası bulunamadı: {data_file}")
                return None
            
            # Veriyi yükle
            df = pd.read_csv(data_file)
            self.logger.info(f"Veri yüklendi: {df.shape}")
            
            # NaN değerleri temizle
            df = df.dropna()
            self.logger.info(f"NaN temizleme sonrası: {df.shape}")
            
            # Sonsuz değerleri temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            self.logger.info(f"Sonsuz değer temizleme sonrası: {df.shape}")
            
            # Target kolonunu ayır
            if 'target' not in df.columns:
                self.logger.error("Target kolonu bulunamadı!")
                return None
            
            X = df.drop(['target'], axis=1)
            y = df['target']
            
            # Feature listesini kaydet
            feature_cols = X.columns.tolist()
            joblib.dump(feature_cols, self.feature_cols_path)
            self.logger.info(f"Feature listesi kaydedildi: {len(feature_cols)} feature")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scaler oluştur ve kaydet
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            joblib.dump(scaler, self.scaler_path)
            self.logger.info("Scaler kaydedildi")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'feature_cols': feature_cols
            }
            
        except Exception as e:
            self.logger.error(f"Veri hazırlama hatası: {e}")
            return None
    
    def train_random_forest(self, data):
        """Random Forest modelini eğit"""
        self.logger.info("🌲 Random Forest eğitiliyor...")
        
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(data['X_train'], data['y_train'])
            
            # Tahmin
            y_pred = rf_model.predict(data['X_test'])
            accuracy = accuracy_score(data['y_test'], y_pred)
            
            # Modeli kaydet
            joblib.dump(rf_model, self.rf_model_path)
            
            self.logger.info(f"✓ Random Forest eğitildi - Doğruluk: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Random Forest eğitim hatası: {e}")
            return None
    
    def train_gradient_boosting(self, data):
        """Gradient Boosting modelini eğit"""
        self.logger.info("🚀 Gradient Boosting eğitiliyor...")
        
        try:
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            gb_model.fit(data['X_train'], data['y_train'])
            
            # Tahmin
            y_pred = gb_model.predict(data['X_test'])
            accuracy = accuracy_score(data['y_test'], y_pred)
            
            # Modeli kaydet
            joblib.dump(gb_model, self.gb_model_path)
            
            self.logger.info(f"✓ Gradient Boosting eğitildi - Doğruluk: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Gradient Boosting eğitim hatası: {e}")
            return None
    
    def train_lstm(self, data):
        """LSTM modelini eğit"""
        self.logger.info("🧠 LSTM modeli eğitiliyor...")
        
        try:
            # LSTM için veriyi yeniden şekillendir
            X_train_lstm = data['X_train_scaled'].reshape((data['X_train_scaled'].shape[0], 1, data['X_train_scaled'].shape[1]))
            X_test_lstm = data['X_test_scaled'].reshape((data['X_test_scaled'].shape[0], 1, data['X_test_scaled'].shape[1]))
            
            # LSTM modeli oluştur
            lstm_model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(1, data['X_train_scaled'].shape[1])),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            # Modeli derle
            lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Modeli eğit
            history = lstm_model.fit(
                X_train_lstm, data['y_train'],
                validation_data=(X_test_lstm, data['y_test']),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Tahmin
            y_pred_proba = lstm_model.predict(X_test_lstm)
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(data['y_test'], y_pred)
            
            # Modeli kaydet
            lstm_model.save(self.lstm_model_path)
            
            self.logger.info(f"✓ LSTM modeli eğitildi - Doğruluk: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"LSTM eğitim hatası: {e}")
            return None
    
    def create_ensemble(self, data):
        """Ensemble modeli oluştur"""
        self.logger.info("🎯 Ensemble modeli oluşturuluyor...")
        
        try:
            # Modelleri yükle
            rf_model = joblib.load(self.rf_model_path)
            gb_model = joblib.load(self.gb_model_path)
            lstm_model = tf.keras.models.load_model(self.lstm_model_path)
            
            # Ensemble tahminleri
            rf_pred = rf_model.predict_proba(data['X_test'])[:, 1]
            gb_pred = gb_model.predict_proba(data['X_test'])[:, 1]
            
            # LSTM tahmini
            X_test_lstm = data['X_test_scaled'].reshape((data['X_test_scaled'].shape[0], 1, data['X_test_scaled'].shape[1]))
            lstm_pred = lstm_model.predict(X_test_lstm).flatten()
            
            # Ağırlıklı ensemble
            ensemble_pred = (0.4 * rf_pred + 0.3 * gb_pred + 0.3 * lstm_pred)
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(data['y_test'], ensemble_pred_binary)
            
            # Ensemble modeli kaydet
            ensemble_data = {
                'rf_model_path': self.rf_model_path,
                'gb_model_path': self.gb_model_path,
                'lstm_model_path': self.lstm_model_path,
                'weights': {'rf': 0.4, 'gb': 0.3, 'lstm': 0.3}
            }
            
            joblib.dump(ensemble_data, self.ensemble_model_path)
            
            self.logger.info(f"✓ Ensemble modeli oluşturuldu - Doğruluk: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Ensemble oluşturma hatası: {e}")
            return None
    
    def test_models(self):
        """Tüm modelleri test et"""
        self.logger.info("🧪 Modeller test ediliyor...")
        
        try:
            # AI model modülünü başlat
            ai_model = AIModel()
            
            # Test verisi al
            collector = DataCollector()
            df = collector.get_historical_data('BTC/USDT', '1h', 100)
            
            if df is None or df.empty:
                self.logger.error("Test verisi alınamadı")
                return False
            
            # Teknik analiz
            ta = TechnicalAnalysis()
            df_with_indicators = ta.calculate_all_indicators(df)
            
            if df_with_indicators is None or df_with_indicators.empty:
                self.logger.error("Teknik analiz başarısız")
                return False
            
            # AI tahmin
            result = ai_model.predict(df_with_indicators)
            
            if isinstance(result, tuple) and len(result) == 2:
                prediction, confidence = result
                if prediction is not None and confidence is not None:
                    self.logger.info(f"✓ AI tahmin başarılı: {prediction:.4f}, Güven: {confidence:.4f}")
                    return True
                else:
                    self.logger.error("AI tahmin başarısız")
                    return False
            else:
                self.logger.error(f"AI tahmin format hatası: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Model test hatası: {e}")
            return False
    
    def run_complete_training(self):
        """Kapsamlı eğitim sürecini çalıştır"""
        self.logger.info("🚀 KAPSAMLI AI MODEL EĞİTİMİ BAŞLATIYOR")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1. Veri hazırlama
            data = self.prepare_training_data()
            if data is None:
                return False
            
            # 2. Random Forest eğitimi
            rf_accuracy = self.train_random_forest(data)
            if rf_accuracy is None:
                return False
            
            # 3. Gradient Boosting eğitimi
            gb_accuracy = self.train_gradient_boosting(data)
            if gb_accuracy is None:
                return False
            
            # 4. LSTM eğitimi
            lstm_accuracy = self.train_lstm(data)
            if lstm_accuracy is None:
                return False
            
            # 5. Ensemble oluşturma
            ensemble_accuracy = self.create_ensemble(data)
            if ensemble_accuracy is None:
                return False
            
            # 6. Model testi
            test_success = self.test_models()
            
            # 7. Sonuç raporu
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("🎯 EĞİTİM SONUÇLARI")
            self.logger.info("=" * 60)
            self.logger.info(f"⏱️  Süre: {duration}")
            self.logger.info(f"🌲 Random Forest: {rf_accuracy:.4f}")
            self.logger.info(f"🚀 Gradient Boosting: {gb_accuracy:.4f}")
            self.logger.info(f"🧠 LSTM: {lstm_accuracy:.4f}")
            self.logger.info(f"🎯 Ensemble: {ensemble_accuracy:.4f}")
            self.logger.info(f"🧪 Test: {'✓ Başarılı' if test_success else '❌ Başarısız'}")
            
            # Ortalama doğruluk
            avg_accuracy = (rf_accuracy + gb_accuracy + lstm_accuracy) / 3
            self.logger.info(f"📊 Ortalama Doğruluk: {avg_accuracy:.4f}")
            
            if test_success:
                self.logger.info("🎉 KAPSAMLI EĞİTİM BAŞARIYLA TAMAMLANDI!")
                return True
            else:
                self.logger.error("❌ Model testi başarısız!")
                return False
                
        except Exception as e:
            self.logger.error(f"Eğitim süreci hatası: {e}")
            return False

def main():
    """Ana fonksiyon"""
    trainer = CompleteAITraining()
    success = trainer.run_complete_training()
    
    if success:
        print("\n✅ KAPSAMLI AI EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
        print("📁 Modeller 'models/' klasöründe kaydedildi")
        print("🧪 Sistem test edildi ve çalışıyor")
    else:
        print("\n❌ Eğitim sürecinde hata oluştu!")
    
    return success

if __name__ == "__main__":
    main() 