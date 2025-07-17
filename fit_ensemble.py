import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier

# Veriyi yükle
print('Veri yükleniyor...')
df = pd.read_csv('data/processed_training_data.csv')
X = df[joblib.load('models/feature_cols.pkl')].values
y = df['label_dynamic'].values

# Modelleri yükle
print('Modeller yükleniyor...')
rf = joblib.load('models/rf_model.pkl')
gb = joblib.load('models/gb_model.pkl')

# Ensemble oluştur
ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
print('Ensemble fit ediliyor...')
ensemble.fit(X, y)

# Joblib ile kaydet
joblib.dump(ensemble, 'models/ensemble_model.pkl')
print('Ensemble model joblib ile kaydedildi.')

# Hemen yükleyip test et
ensemble_loaded = joblib.load('models/ensemble_model.pkl')
print('Yeniden yüklenen ensemble predict testi:', ensemble_loaded.predict(X[:5])) 