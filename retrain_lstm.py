#!/usr/bin/env python3
"""
LSTM modelini yeni 128 feature ile yeniden eğit
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import joblib

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def retrain_lstm():
    """LSTM modelini yeni feature sayısı ile yeniden eğit"""
    try:
        print("=== LSTM Modeli Yeniden Eğitiliyor ===")
        
        # Eğitim verisini yükle
        print("Eğitim verisi yükleniyor...")
        data_path = os.path.join(Config.DATA_DIR, 'processed_training_data.csv')
        
        if not os.path.exists(data_path):
            print(f"❌ Eğitim verisi bulunamadı: {data_path}")
            return False
            
        df = pd.read_csv(data_path)
        print(f"✅ Veri yüklendi: {df.shape}")
        
        # Feature columns yükle
        feature_cols = joblib.load(os.path.join(Config.MODELS_DIR, 'feature_cols.pkl'))
        print(f"✅ Feature columns yüklendi: {len(feature_cols)} feature")
        
        # X ve y hazırla
        X = df[feature_cols]
        y = df['label_dynamic']
        
        print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        
        # NaN değerleri temizle ve veri tiplerini düzelt
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = X.astype(np.float32)  # float32'ye dönüştür
        y = y.astype(np.float32)  # float32'ye dönüştür
        print(f"✅ NaN temizleme ve veri tipi düzeltme sonrası: {X.shape}")
        print(f"✅ X dtypes: {X.dtypes.iloc[0]}, y dtype: {y.dtype}")
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
        
        # LSTM için veriyi reshape et (samples, timesteps, features)
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])).astype(np.float32)
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])).astype(np.float32)
        
        print(f"✅ LSTM Train shape: {X_train_lstm.shape}, dtype: {X_train_lstm.dtype}")
        print(f"✅ LSTM Test shape: {X_test_lstm.shape}, dtype: {X_test_lstm.dtype}")
        
        # Veri tiplerini kontrol et
        print(f"✅ y_train dtype: {y_train.dtype}")
        print(f"✅ y_test dtype: {y_test.dtype}")
        
        # NaN kontrolü
        print(f"✅ X_train_lstm NaN sayısı: {np.isnan(X_train_lstm).sum()}")
        print(f"✅ X_test_lstm NaN sayısı: {np.isnan(X_test_lstm).sum()}")
        print(f"✅ y_train NaN sayısı: {np.isnan(y_train).sum()}")
        print(f"✅ y_test NaN sayısı: {np.isnan(y_test).sum()}")
        
        # LSTM modeli oluştur
        print("LSTM modeli oluşturuluyor...")
        model = Sequential([
            LSTM(128, input_shape=(1, len(feature_cols)), return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model oluşturuldu. Input shape: {model.input_shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Modeli eğit
        print("LSTM eğitiliyor...")
        history = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_test_lstm, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Test
        y_pred_proba = model.predict(X_test_lstm)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n=== LSTM Sonuçları ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Modeli kaydet
        model_path = os.path.join(Config.MODELS_DIR, 'lstm_model.h5')
        model.save(model_path)
        print(f"✅ LSTM modeli kaydedildi: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = retrain_lstm()
    if success:
        print("\n🎉 LSTM modeli başarıyla yeniden eğitildi!")
    else:
        print("\n❌ LSTM modeli yeniden eğitilemedi!") 