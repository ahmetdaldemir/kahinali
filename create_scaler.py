#!/usr/bin/env python3
"""
Eksik scaler'ı oluştur
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def create_scaler():
    """StandardScaler oluştur ve kaydet"""
    try:
        print("=== StandardScaler Oluşturuluyor ===")
        
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

        # Sadece eğitim verisinde olan feature'ları kullan
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"⚠️ Uyarı: Eğitim verisinde olmayan feature'lar atlandı: {missing_features}")
        if not available_features:
            print("❌ Hiçbir feature eğitim verisinde bulunamadı!")
            return False

        # X hazırla
        X = df[available_features]
        print(f"✅ X shape: {X.shape}")
        
        # NaN değerleri temizle
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"✅ NaN temizleme sonrası: {X.shape}")
        
        # StandardScaler oluştur ve fit et
        print("StandardScaler eğitiliyor...")
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Scaler'ı kaydet
        scaler_path = os.path.join(Config.MODELS_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler kaydedildi: {scaler_path}")
        
        # Test et
        X_scaled = scaler.transform(X[:100])  # İlk 100 satır
        print(f"✅ Scaler test edildi: {X_scaled.shape}")
        print(f"✅ Scaled data mean: {X_scaled.mean():.6f}")
        print(f"✅ Scaled data std: {X_scaled.std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = create_scaler()
    if success:
        print("\n🎉 StandardScaler başarıyla oluşturuldu!")
    else:
        print("\n❌ StandardScaler oluşturulamadı!") 