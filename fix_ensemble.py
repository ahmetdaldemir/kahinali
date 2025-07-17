import pandas as pd
import pickle
import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

print("🔧 Ensemble Model Düzeltiliyor...")
print("=" * 50)

# 1. Veriyi yükle
print("📊 Veri yükleniyor...")
df = pd.read_csv('data/processed_training_data.csv')
feature_cols = joblib.load('models/feature_cols.pkl')
X = df[feature_cols].values
y = df['label_dynamic'].values
print(f"✓ Veri yüklendi: {X.shape}")

# 2. Yeni modeller oluştur ve eğit
print("🤖 Yeni modeller eğitiliyor...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

rf.fit(X, y)
gb.fit(X, y)
print("✓ RF ve GB modelleri eğitildi")

# 3. Ensemble oluştur ve fit et
print("🎯 Ensemble model oluşturuluyor...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)
ensemble.fit(X, y)
print("✓ Ensemble model fit edildi")

# 4. Tüm modelleri kaydet
print("💾 Modeller kaydediliyor...")
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(gb, 'models/gb_model.pkl')

with open('models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

# 5. Feature importance güncelle
feature_importance = {
    'rf_importance': rf.feature_importances_,
    'gb_importance': gb.feature_importances_
}
with open('models/feature_importance.pkl', 'wb') as f:
    pickle.dump(feature_importance, f)

# 6. Models dict güncelle
models_dict = {
    'ensemble': ensemble,
    'rf': rf,
    'gb': gb,
    'lstm': 'models/lstm_model.h5'
}
with open('models/models.pkl', 'wb') as f:
    pickle.dump(models_dict, f)

print("✅ Tüm modeller başarıyla güncellendi!")
print("=" * 50) 