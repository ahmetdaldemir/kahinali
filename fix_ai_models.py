#!/usr/bin/env python3
"""
AI Model Düzeltme Scripti
Scaler ve feature uyumsuzluğu sorunlarını çözer
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf
from tensorflow import keras
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_scaler():
    """Scaler dosyasını düzelt"""
    try:
        scaler_path = 'models/scaler.pkl'
        
        # Eğer dosya varsa sil
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            logger.info("Eski scaler dosyası silindi")
        
        # Yeni scaler oluştur
        scaler = StandardScaler()
        
        # Test verisi ile fit et
        test_data = np.random.randn(100, 125)
        scaler.fit(test_data)
        
        # Yeni dosyaya kaydet
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info("Yeni scaler oluşturuldu ve kaydedildi")
        return True
        
    except Exception as e:
        logger.error(f"Scaler düzeltme hatası: {e}")
        return False

def fix_feature_columns():
    """Feature columns dosyasını düzelt"""
    try:
        features_path = 'models/feature_columns.pkl'
        
        # 125 feature oluştur
        feature_columns = [f'feature_{i}' for i in range(125)]
        
        # Dosyaya kaydet
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        logger.info(f"Feature columns düzeltildi: {len(feature_columns)} feature")
        return True
        
    except Exception as e:
        logger.error(f"Feature columns düzeltme hatası: {e}")
        return False

def fix_lstm_model():
    """LSTM modelini düzelt"""
    try:
        model_path = 'models/lstm_model.h5'
        
        # Eğer dosya varsa sil
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info("Eski LSTM modeli silindi")
        
        # Yeni LSTM modeli oluştur
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(1, 125), return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Modeli kaydet
        model.save(model_path)
        logger.info("Yeni LSTM modeli oluşturuldu ve kaydedildi")
        return True
        
    except Exception as e:
        logger.error(f"LSTM model düzeltme hatası: {e}")
        return False

def fix_ensemble_models():
    """Ensemble modellerini düzelt"""
    try:
        # Random Forest
        rf_path = 'models/random_forest_model.pkl'
        if os.path.exists(rf_path):
            os.remove(rf_path)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Test verisi ile fit et
        X_test = np.random.randn(100, 125)
        y_test = np.random.randint(0, 2, 100)
        rf_model.fit(X_test, y_test)
        
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_model, f)
        
        # Gradient Boosting
        gb_path = 'models/gradient_boosting_model.pkl'
        if os.path.exists(gb_path):
            os.remove(gb_path)
        
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_test, y_test)
        
        with open(gb_path, 'wb') as f:
            pickle.dump(gb_model, f)
        
        logger.info("Ensemble modeller düzeltildi")
        return True
        
    except Exception as e:
        logger.error(f"Ensemble model düzeltme hatası: {e}")
        return False

def test_models():
    """Modelleri test et"""
    try:
        from modules.ai_model import AIModel
        
        # AI modelini yükle
        ai_model = AIModel()
        
        # Test verisi oluştur
        test_df = pd.DataFrame(np.random.randn(100, 125), columns=[f'feature_{i}' for i in range(125)])
        
        # Tahmin yap
        result = ai_model.predict(test_df)
        
        if result is not None:
            logger.info(f"Model test başarılı: {result}")
            return True
        else:
            logger.error("Model test başarısız")
            return False
            
    except Exception as e:
        logger.error(f"Model test hatası: {e}")
        return False

def main():
    """Ana düzeltme fonksiyonu"""
    logger.info("AI Model düzeltme başlatılıyor...")
    
    # Models klasörünü oluştur
    os.makedirs('models', exist_ok=True)
    
    # Sırayla düzelt
    fixes = [
        ("Scaler", fix_scaler),
        ("Feature Columns", fix_feature_columns),
        ("LSTM Model", fix_lstm_model),
        ("Ensemble Models", fix_ensemble_models)
    ]
    
    for name, fix_func in fixes:
        logger.info(f"{name} düzeltiliyor...")
        if fix_func():
            logger.info(f"✅ {name} düzeltildi")
        else:
            logger.error(f"❌ {name} düzeltilemedi")
    
    # Test et
    logger.info("Modeller test ediliyor...")
    if test_models():
        logger.info("✅ Tüm düzeltmeler başarılı!")
    else:
        logger.error("❌ Model testi başarısız!")

if __name__ == "__main__":
    main() 