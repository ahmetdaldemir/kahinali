import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_and_retrain_lstm():
    """Fix feature columns and retrain LSTM model"""
    try:
        # Load training data
        print("📊 Loading training data...")
        df = pd.read_csv('data/processed_training_data.csv')
        print(f"✅ Data loaded: {df.shape}")
        
        # Get all feature columns (exclude target and non-feature columns)
        exclude_cols = ['label_dynamic', 'label_5', 'label_10', 'label_20']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"✅ Feature count: {len(feature_cols)}")
        print(f"✅ Features: {feature_cols[:10]}...")
        
        # Save correct feature columns
        with open('models/feature_cols.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        print("✅ Feature columns saved!")
        
        # Prepare features
        X = df[feature_cols].values
        y = df['label_dynamic'].values
        
        print(f"✅ X shape: {X.shape}")
        print(f"✅ y shape: {y.shape}")
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Scaler saved!")
        
        # Reshape for LSTM (samples, timesteps, features)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        print(f"✅ X reshaped: {X_reshaped.shape}")
        
        # Build LSTM model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(1, len(feature_cols))),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("🚀 Training LSTM model...")
        history = model.fit(
            X_reshaped, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model
        model.save('models/lstm_model.h5')
        print("✅ LSTM model saved successfully!")
        
        # Test prediction
        test_sample = X_reshaped[:1]
        prediction = model.predict(test_sample)
        print(f"✅ Test prediction shape: {prediction.shape}")
        print(f"✅ Test prediction value: {prediction[0][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Fixing features and retraining LSTM...")
    success = fix_and_retrain_lstm()
    if success:
        print("✅ LSTM retraining completed successfully!")
    else:
        print("❌ LSTM retraining failed!") 