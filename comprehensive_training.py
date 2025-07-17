#!/usr/bin/env python3
"""
Kapsamlı AI Model Eğitimi
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle
import joblib

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from modules.ai_model import AIModel
from modules.technical_analysis import TechnicalAnalysis

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveTrainer:
    """Kapsamlı model eğitimi sınıfı"""
    
    def __init__(self):
        self.ai_model = AIModel()
        self.ta = TechnicalAnalysis()
        self.scaler = None
        self.feature_cols = None
        
    def load_and_prepare_data(self):
        """Eğitim verisini yükle ve hazırla"""
        logger.info("Eğitim verisi yükleniyor...")
        
        try:
            # Veriyi yükle
            df = pd.read_csv('data/processed_training_data.csv')
            logger.info(f"Veri yüklendi: {df.shape}")
            
            # Label dağılımını kontrol et
            label_counts = df['label_dynamic'].value_counts()
            logger.info(f"Label dağılımı: {label_counts.to_dict()}")
            
            # Feature'ları belirle
            exclude_cols = ['label_dynamic', 'label_5', 'label_10', 'label_20']
            self.feature_cols = [col for col in df.columns if col not in exclude_cols]
            logger.info(f"Feature sayısı: {len(self.feature_cols)}")
            
            # NaN değerleri temizle
            df = df.dropna()
            logger.info(f"NaN temizleme sonrası: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {e}")
            return None
    
    def prepare_features(self, df):
        """Feature'ları hazırla"""
        logger.info("Feature'lar hazırlanıyor...")
        
        try:
            # X ve y hazırla
            X = df[self.feature_cols]
            y = df['label_dynamic']
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scaler oluştur
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Scaler'ı kaydet
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Feature listesini kaydet
            with open('models/feature_cols.pkl', 'wb') as f:
                pickle.dump(self.feature_cols, f)
            
            logger.info("Feature'lar hazırlandı ve kaydedildi")
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Feature hazırlama hatası: {e}")
            return None, None, None, None
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Random Forest modelini eğit"""
        logger.info("Random Forest eğitiliyor...")
        
        try:
            # Hiperparametre optimizasyonu
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_rf = grid_search.best_estimator_
            
            # Performans değerlendirme
            y_pred = best_rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"RF - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Modeli kaydet
            with open('models/rf_model.pkl', 'wb') as f:
                pickle.dump(best_rf, f)
            
            logger.info("Random Forest eğitildi ve kaydedildi")
            return best_rf
            
        except Exception as e:
            logger.error(f"Random Forest eğitim hatası: {e}")
            return None
    
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        """Gradient Boosting modelini eğit"""
        logger.info("Gradient Boosting eğitiliyor...")
        
        try:
            # Hiperparametre optimizasyonu
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_gb = grid_search.best_estimator_
            
            # Performans değerlendirme
            y_pred = best_gb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"GB - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Modeli kaydet
            with open('models/gb_model.pkl', 'wb') as f:
                pickle.dump(best_gb, f)
            
            logger.info("Gradient Boosting eğitildi ve kaydedildi")
            return best_gb
            
        except Exception as e:
            logger.error(f"Gradient Boosting eğitim hatası: {e}")
            return None
    
    def train_lstm(self, X_train, X_test, y_train, y_test):
        """LSTM modelini eğit"""
        logger.info("LSTM eğitiliyor...")
        
        try:
            # LSTM için 3D reshape
            X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            
            # Gelişmiş LSTM modeli
            model = Sequential([
                LSTM(128, input_shape=(1, X_train.shape[1]), return_sequences=True),
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
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
                ModelCheckpoint('models/lstm_model.h5', save_best_only=True, monitor='val_accuracy')
            ]
            
            # Eğitim
            history = model.fit(
                X_train_lstm, y_train,
                validation_data=(X_test_lstm, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Performans değerlendirme
            test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test, verbose=0)
            logger.info(f"LSTM - Test Accuracy: {test_accuracy:.4f}")
            
            logger.info("LSTM eğitildi ve kaydedildi")
            return model
            
        except Exception as e:
            logger.error(f"LSTM eğitim hatası: {e}")
            return None
    
    def create_ensemble(self, rf_model, gb_model, lstm_model, X_test, y_test):
        """Ensemble model oluştur"""
        logger.info("Ensemble model oluşturuluyor...")
        
        try:
            # Tahminler
            rf_pred = rf_model.predict_proba(X_test)[:, 1]
            gb_pred = gb_model.predict_proba(X_test)[:, 1]
            
            # LSTM tahmin
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            lstm_pred = lstm_model.predict(X_test_lstm).flatten()
            
            # Ensemble tahmin (ağırlıklı ortalama)
            ensemble_pred = (0.4 * rf_pred + 0.4 * gb_pred + 0.2 * lstm_pred)
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            # Performans değerlendirme
            accuracy = accuracy_score(y_test, ensemble_pred_binary)
            precision = precision_score(y_test, ensemble_pred_binary)
            recall = recall_score(y_test, ensemble_pred_binary)
            f1 = f1_score(y_test, ensemble_pred_binary)
            
            logger.info(f"Ensemble - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Ensemble modeli kaydet
            ensemble_data = {
                'rf_model': rf_model,
                'gb_model': gb_model,
                'lstm_model': lstm_model,
                'weights': [0.4, 0.4, 0.2]
            }
            
            with open('models/ensemble_model.pkl', 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            logger.info("Ensemble model oluşturuldu ve kaydedildi")
            return ensemble_data
            
        except Exception as e:
            logger.error(f"Ensemble oluşturma hatası: {e}")
            return None
    
    def run_comprehensive_training(self):
        """Kapsamlı eğitim sürecini çalıştır"""
        logger.info("=== KAPSAMLI AI MODEL EĞİTİMİ BAŞLADI ===")
        start_time = datetime.now()
        
        try:
            # 1. Veri yükle ve hazırla
            df = self.load_and_prepare_data()
            if df is None:
                return False
            
            # 2. Feature'ları hazırla
            X_train, X_test, y_train, y_test = self.prepare_features(df)
            if X_train is None:
                return False
            
            # 3. Random Forest eğit
            rf_model = self.train_random_forest(X_train, X_test, y_train, y_test)
            if rf_model is None:
                return False
            
            # 4. Gradient Boosting eğit
            gb_model = self.train_gradient_boosting(X_train, X_test, y_train, y_test)
            if gb_model is None:
                return False
            
            # 5. LSTM eğit
            lstm_model = self.train_lstm(X_train, X_test, y_train, y_test)
            if lstm_model is None:
                return False
            
            # 6. Ensemble oluştur
            ensemble = self.create_ensemble(rf_model, gb_model, lstm_model, X_test, y_test)
            if ensemble is None:
                return False
            
            # 7. Sonuç raporu
            total_time = datetime.now() - start_time
            logger.info("=== KAPSAMLI EĞİTİM TAMAMLANDI ===")
            logger.info(f"Toplam süre: {total_time}")
            logger.info(f"Kullanılan veri: {len(df)} satır")
            logger.info(f"Feature sayısı: {len(self.feature_cols)}")
            logger.info("Tüm modeller başarıyla eğitildi!")
            
            return True
            
        except Exception as e:
            logger.error(f"Eğitim sırasında hata: {e}")
            return False

def main():
    """Ana fonksiyon"""
    trainer = ComprehensiveTrainer()
    success = trainer.run_comprehensive_training()
    
    if success:
        print("✅ Kapsamlı eğitim başarıyla tamamlandı!")
    else:
        print("❌ Eğitim sırasında hata oluştu!")

if __name__ == "__main__":
    main() 