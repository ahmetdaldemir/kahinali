import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Eski model dosyalarını sil
model_files = [
    'models/ensemble_model.pkl',
    'models/feature_cols.pkl',
    'models/feature_importance.pkl',
    'models/feature_selector.pkl',
    'models/gb_model.pkl',
    'models/lstm_model.h5',
    'models/models.pkl',
    'models/rf_model.pkl',
    'models/scaler.pkl'
]
for f in model_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"Silindi: {f}")

# 2. Feature listesi ve scaler oluştur
print("Veri yükleniyor...")
df = pd.read_csv('data/processed_training_data.csv')
exclude_cols = ['label_dynamic', 'label_5', 'label_10', 'label_20']
feature_cols = [col for col in df.columns if col not in exclude_cols]
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"Feature cols kaydedildi: {len(feature_cols)} feature")

X = df[feature_cols].values
y = df['label_dynamic'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler kaydedildi.")

# 3. LSTM Modeli Eğit
print("LSTM modeli eğitiliyor...")
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, len(feature_cols))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_lstm, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
lstm_model.save('models/lstm_model.h5')
print("LSTM modeli kaydedildi.")

# 4. RF ve GB Modeli Eğit
print("Random Forest eğitiliyor...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("RF modeli kaydedildi.")

print("Gradient Boosting eğitiliyor...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_scaled, y)
with open('models/gb_model.pkl', 'wb') as f:
    pickle.dump(gb, f)
print("GB modeli kaydedildi.")

print("Tüm modeller ve feature/scaler dosyaları güncellendi!") 