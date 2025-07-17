import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import gc

def final_lstm_fix():
    """Final LSTM fix with correct 125 features"""
    print("🔧 FINAL LSTM FIX - 125 features ile yeniden oluşturuyorum...")
    
    # 1. Memory temizle
    gc.collect()
    
    # 2. Veri yükle
    df = pd.read_csv('data/processed_training_data.csv')
    print(f"✅ Veri yüklendi: {df.shape}")
    
    # 3. Feature columns belirle (exclude target columns)
    exclude_cols = ['label_dynamic', 'label_5', 'label_10', 'label_20']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"✅ Feature sayısı: {len(feature_cols)}")
    print(f"✅ İlk 10 feature: {feature_cols[:10]}")
    
    # 4. Feature columns kaydet
    with open('models/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("✅ Feature columns kaydedildi")
    
    # 5. Scaler oluştur ve kaydet
    scaler = MinMaxScaler()
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler kaydedildi")
    
    # 6. LSTM için veri hazırla
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y = df['label_dynamic'].values
    
    print(f"✅ LSTM input shape: {X_lstm.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    # 7. LSTM model oluştur
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(1, len(feature_cols))),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # 8. Model input shape kontrol et
    print(f"✅ Model input shape: {model.input_shape}")
    print(f"✅ Beklenen feature sayısı: {len(feature_cols)}")
    
    # 9. Model eğit
    print("🔄 LSTM modeli eğitiliyor...")
    model.fit(X_lstm, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
    
    # 10. Model kaydet
    model.save('models/lstm_model.h5')
    print("✅ LSTM modeli kaydedildi")
    
    # 11. Final kontrol
    print(f"✅ Final model input shape: {model.input_shape}")
    print(f"✅ Feature sayısı: {len(feature_cols)}")
    
    # 12. Test prediction
    test_input = X_lstm[:1]  # İlk satır
    prediction = model.predict(test_input)
    print(f"✅ Test prediction başarılı: {prediction[0][0]:.4f}")
    
    return True

if __name__ == "__main__":
    final_lstm_fix() 