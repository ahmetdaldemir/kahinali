#!/usr/bin/env python3
"""
LSTM model test scripti
"""

import numpy as np
from modules.ai_model import AIModel

def test_lstm():
    try:
        print("AI Model yükleniyor...")
        model = AIModel()
        
        print(f"Feature sayısı: {len(model.feature_cols)}")
        
        if model.lstm_model is None:
            print("LSTM model yüklenemedi!")
            return
        
        print("Test data oluşturuluyor...")
        test_data = np.random.random((1, 1, 128))
        print(f"Test data shape: {test_data.shape}")
        
        print("LSTM tahmin yapılıyor...")
        prediction = model.lstm_model.predict(test_data)
        print(f"LSTM tahmin başarılı! Shape: {prediction.shape}")
        print(f"Tahmin değeri: {prediction[0][0]}")
        
        print("✅ LSTM model testi başarılı!")
        
    except Exception as e:
        print(f"❌ LSTM test hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

if __name__ == "__main__":
    test_lstm() 