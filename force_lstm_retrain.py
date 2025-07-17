import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

def force_lstm_retrain():
    """Force LSTM retraining with correct features"""
    print("🔄 LSTM modelini zorla yeniden eğitiyorum...")
    
    # 1. Veri yükle
    df = pd.read_csv('data/processed_training_data.csv')
    print(f"✅ Veri yüklendi: {df.shape}")
    
    # 2. Feature columns belirle
    exclude_cols = ['label_dynamic', 'label_5', 'label_10', 'label_20']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"✅ Feature sayısı: {len(feature_cols)}")
    
    # 3. Feature columns kaydet
    with open('models/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("✅ Feature columns kaydedildi")
    
    # 4. Scaler oluştur ve kaydet
    scaler = MinMaxScaler()
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler kaydedildi")
    
    # 5. LSTM için veri hazırla
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y = df['label_dynamic'].values
    
    # 6. LSTM model oluştur
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(1, len(feature_cols))),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # 7. Model eğit
    print("🔄 LSTM modeli eğitiliyor...")
    model.fit(X_lstm, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    # 8. Model kaydet
    model.save('models/lstm_model.h5')
    print("✅ LSTM modeli kaydedildi")
    
    # 9. Model input shape kontrol et
    print(f"✅ LSTM input shape: {model.input_shape}")
    print(f"✅ Feature sayısı: {len(feature_cols)}")
    
    return True

if __name__ == "__main__":
    force_lstm_retrain() 