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

def check_features():
    """Check current feature columns"""
    try:
        with open('models/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        print(f"✅ Feature count: {len(feature_cols)}")
        print(f"✅ Features: {feature_cols[:10]}...")
        return feature_cols
    except Exception as e:
        print(f"❌ Error loading feature_cols.pkl: {e}")
        return None

def retrain_lstm():
    """Retrain LSTM model with current features"""
    try:
        # Load feature columns
        feature_cols = check_features()
        if not feature_cols:
            return False
            
        # Load training data
        print("📊 Loading training data...")
        df = pd.read_csv('data/processed_training_data.csv')
        print(f"✅ Data loaded: {df.shape}")
        
        # Prepare features
        X = df[feature_cols].values
        y = df['label_dynamic'].values
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reshape for LSTM (samples, timesteps, features)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        print(f"✅ X shape: {X_reshaped.shape}")
        print(f"✅ y shape: {y.shape}")
        
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
        print(f"❌ Error retraining LSTM: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Checking features and retraining LSTM...")
    success = retrain_lstm()
    if success:
        print("✅ LSTM retraining completed successfully!")
    else:
        print("❌ LSTM retraining failed!") 