import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

class AIPipeline:
    def __init__(self, data_path='data/processed_training_data.csv'):
        self.data_path = data_path
        self.scaler = None
        self.feature_cols = None
        self.models = {}
        self.feature_importance = {}
        
    def load_and_prepare_data(self):
        """Veriyi yükle ve hazırla"""
        print("📊 Veri yükleniyor...")
        
        # Veriyi yükle
        df = pd.read_csv(self.data_path)
        print(f"✓ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        
        # Label sütunlarını bul
        label_cols = [col for col in df.columns if col.startswith('label')]
        print(f"✓ Label sütunları bulundu: {label_cols}")
        
        # Feature sütunlarını belirle (label ve target sütunları hariç)
        exclude_cols = label_cols + ['future_close_5', 'future_close_10', 'future_close_20', 'future_close_30']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"✓ Feature sayısı: {len(self.feature_cols)}")
        print(f"✓ İlk 10 feature: {self.feature_cols[:10]}")
        
        # NaN değerleri temizle
        df = df.dropna()
        print(f"✓ NaN temizlendi: {df.shape[0]} satır kaldı")
        
        # --- OUTLIER TEMİZLİĞİ ---
        before = df.shape[0]
        for col in self.feature_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower = Q1 - 2 * IQR
                upper = Q3 + 2 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = df.shape[0]
        print(f"✓ Outlier temizliği: {before-after} satır çıkarıldı, {after} satır kaldı")
        
        # Feature ve target verilerini ayır
        X = df[self.feature_cols].values
        y = df['label_dynamic'].values  # Ana label
        
        print(f"✓ X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, df
    
    def create_scaler(self, X):
        """Scaler oluştur ve kaydet"""
        print("🔧 Scaler oluşturuluyor...")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Scaler'ı kaydet
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Feature listesini kaydet
        with open('models/feature_cols.pkl', 'wb') as f:
            pickle.dump(self.feature_cols, f)
        
        print(f"✓ Scaler kaydedildi: models/scaler.pkl")
        print(f"✓ Feature listesi kaydedildi: models/feature_cols.pkl")
        
        return X_scaled
    
    def train_ml_models(self, X_scaled, y):
        """ML modellerini eğit"""
        print("🤖 ML modelleri eğitiliyor...")
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        # Random Forest
        print("  - Random Forest eğitiliyor...")
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        print(f"    ✓ RF Accuracy: {rf_score:.4f}")
        # Gradient Boosting
        print("  - Gradient Boosting eğitiliyor...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_score = gb_model.score(X_test, y_test)
        print(f"    ✓ GB Accuracy: {gb_score:.4f}")
        # XGBoost
        print("  - XGBoost eğitiliyor...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_test, y_test)
        print(f"    ✓ XGB Accuracy: {xgb_score:.4f}")
        # LightGBM
        print("  - LightGBM eğitiliyor...")
        lgbm_model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_score = lgbm_model.score(X_test, y_test)
        print(f"    ✓ LGBM Accuracy: {lgbm_score:.4f}")
        # Modelleri kaydet
        with open('models/rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        with open('models/gb_model.pkl', 'wb') as f:
            pickle.dump(gb_model, f)
        with open('models/xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        with open('models/lgbm_model.pkl', 'wb') as f:
            pickle.dump(lgbm_model, f)
        # Feature importance kaydet
        self.feature_importance = {
            'rf_importance': rf_model.feature_importances_,
            'gb_importance': gb_model.feature_importances_,
            'xgb_importance': xgb_model.feature_importances_,
            'lgbm_importance': lgbm_model.feature_importances_
        }
        with open('models/feature_importance.pkl', 'wb') as f:
            pickle.dump(self.feature_importance, f)
        print("✓ ML modelleri kaydedildi")
        return rf_model, gb_model, xgb_model, lgbm_model
    
    def optimize_hyperparameters(self, X, y):
        """Random Forest ve XGBoost için GridSearch ile en iyi parametreleri bul"""
        print("🔬 Hiperparametre optimizasyonu başlatılıyor (GridSearch)...")
        # Random Forest parametreleri
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
        rf_grid.fit(X, y)
        print(f"  ✓ RF en iyi parametreler: {rf_grid.best_params_}")
        # XGBoost parametreleri
        from xgboost import XGBClassifier
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
        xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='f1', n_jobs=-1)
        xgb_grid.fit(X, y)
        print(f"  ✓ XGB en iyi parametreler: {xgb_grid.best_params_}")
        return rf_grid.best_estimator_, xgb_grid.best_estimator_
    
    def train_lstm_model(self, X_scaled, y):
        """LSTM modelini eğit"""
        print("🧠 LSTM modeli eğitiliyor...")
        
        # LSTM için veriyi yeniden şekillendir (samples, timesteps, features)
        # Burada tek timestep kullanıyoruz
        X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X_lstm, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # LSTM modeli oluştur
        model = Sequential([
            LSTM(64, input_shape=(1, X_scaled.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Modeli eğit
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Test accuracy
        lstm_score = model.evaluate(X_test, y_test, verbose=0)[1]
        print(f"✓ LSTM Accuracy: {lstm_score:.4f}")
        
        # Modeli kaydet
        model.save('models/lstm_model.h5')
        print("✓ LSTM modeli kaydedildi: models/lstm_model.h5")
        
        return model
    
    def create_ensemble_model(self, rf_model, gb_model, xgb_model, lgbm_model):
        """Ensemble model oluştur"""
        print("🎯 Ensemble model oluşturuluyor...")
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('xgb', xgb_model),
                ('lgbm', lgbm_model)
            ],
            voting='soft'
        )
        # Ensemble'i kaydet
        with open('models/ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble, f)
        print("✓ Ensemble model kaydedildi")
        return ensemble
    
    def create_models_dict(self, rf_model, gb_model, xgb_model, lgbm_model, ensemble):
        """Models dict oluştur"""
        print("📁 Models dict oluşturuluyor...")
        models_dict = {
            'ensemble': ensemble,
            'rf': rf_model,
            'gb': gb_model,
            'xgb': xgb_model,
            'lgbm': lgbm_model,
            'lstm': 'models/lstm_model.h5'
        }
        with open('models/models.pkl', 'wb') as f:
            pickle.dump(models_dict, f)
        print("✓ Models dict kaydedildi")
    
    def create_feature_selector(self, threshold=0.01):
        """Feature selector oluştur: Önemli feature'ları otomatik seç"""
        print("🎛️ Feature selector (otomatik) oluşturuluyor...")
        # Feature importance yükle
        rf_importance = self.feature_importance.get('rf_importance', np.zeros(len(self.feature_cols)))
        gb_importance = self.feature_importance.get('gb_importance', np.zeros(len(self.feature_cols)))
        xgb_importance = self.feature_importance.get('xgb_importance', np.zeros(len(self.feature_cols)))
        lgbm_importance = self.feature_importance.get('lgbm_importance', np.zeros(len(self.feature_cols)))
        # Ortalama importance
        avg_importance = (rf_importance + gb_importance + xgb_importance + lgbm_importance) / 4
        # Seçim: importance toplamı threshold'dan büyük olanlar
        feature_selector = avg_importance > threshold
        print(f"✓ {feature_selector.sum()} önemli feature seçildi (toplam {len(feature_selector)})")
        # Seçili feature isimlerini kaydet
        selected_features = [f for f, sel in zip(self.feature_cols, feature_selector) if sel]
        with open('models/selected_features.pkl', 'wb') as f:
            pickle.dump(selected_features, f)
        with open('models/feature_selector.pkl', 'wb') as f:
            pickle.dump(feature_selector, f)
        print("✓ Feature selector ve seçili feature listesi kaydedildi")
        return feature_selector
    
    def run_backtest(self, model, X, y, entry_price, commission=0.001, slippage=0.001):
        """Model ile geçmiş veri üzerinde backtest yap, komisyon ve slippage dahil net PnL ve başarı oranı hesapla"""
        print("🧪 Backtest başlatılıyor...")
        preds = model.predict(X)
        # Sinyal: 1 ise pozisyon aç, 0 ise açma
        n_signals = (preds == 1).sum()
        n_correct = ((preds == 1) & (y == 1)).sum()
        # Basit PnL hesabı: Her sinyalde %X kar/zarar, komisyon ve slippage düşülerek
        avg_pnl = 0.05  # Örnek: ortalama hedef %5
        avg_loss = -0.03  # Örnek: ortalama stop %3
        total_pnl = 0
        for pred, true in zip(preds, y):
            if pred == 1:
                if true == 1:
                    total_pnl += avg_pnl - commission - slippage
                else:
                    total_pnl += avg_loss - commission - slippage
        success_rate = n_correct / n_signals if n_signals > 0 else 0
        print(f"  ✓ Sinyal sayısı: {n_signals}, Başarı oranı: {success_rate:.2%}, Net PnL: {total_pnl:.2f}")
        return success_rate, total_pnl
    
    def run_full_pipeline(self, auto_retrain=False, retrain_period_days=7, optimize_hp=False, run_backtest=False):
        """Tam pipeline'ı çalıştır"""
        print("🚀 AI Pipeline başlatılıyor...")
        print("=" * 50)
        try:
            # 1. Veri hazırlama
            X, y, df = self.load_and_prepare_data()
            # 2. Scaler oluşturma
            X_scaled = self.create_scaler(X)
            # --- Hiperparametre optimizasyonu ---
            if optimize_hp:
                print("\n🔬 Hiperparametre optimizasyonu aktif!")
                best_rf, best_xgb = self.optimize_hyperparameters(X_scaled, y)
                print("  ✓ Optimize modeller kaydediliyor...")
                with open('models/rf_model_optimized.pkl', 'wb') as f:
                    pickle.dump(best_rf, f)
                with open('models/xgb_model_optimized.pkl', 'wb') as f:
                    pickle.dump(best_xgb, f)
            # 3. ML modelleri eğitme
            rf_model, gb_model, xgb_model, lgbm_model = self.train_ml_models(X_scaled, y)
            # 4. LSTM modeli eğitme
            lstm_model = self.train_lstm_model(X_scaled, y)
            # 5. Ensemble model oluşturma
            ensemble = self.create_ensemble_model(rf_model, gb_model, xgb_model, lgbm_model)
            # 6. Models dict oluşturma
            self.create_models_dict(rf_model, gb_model, xgb_model, lgbm_model, ensemble)
            # 7. Feature selector oluşturma (otomatik)
            self.create_feature_selector()
            # --- BACKTEST ---
            if run_backtest:
                print("\n🧪 Backtest ve simülasyon başlatılıyor!")
                entry_price = df['close'].values if 'close' in df.columns else np.ones(len(y))
                self.run_backtest(rf_model, X_scaled, y, entry_price, commission=0.001, slippage=0.001)
                self.run_backtest(xgb_model, X_scaled, y, entry_price, commission=0.001, slippage=0.001)
                self.run_backtest(ensemble, X_scaled, y, entry_price, commission=0.001, slippage=0.001)
            print("=" * 50)
            print("🎉 AI Pipeline başarıyla tamamlandı!")
            print("📊 Oluşturulan dosyalar:")
            print("  - models/scaler.pkl")
            print("  - models/feature_cols.pkl")
            print("  - models/rf_model.pkl")
            print("  - models/gb_model.pkl")
            print("  - models/xgb_model.pkl")
            print("  - models/lgbm_model.pkl")
            print("  - models/lstm_model.h5")
            print("  - models/ensemble_model.pkl")
            print("  - models/feature_importance.pkl")
            print("  - models/feature_selector.pkl")
            print("  - models/models.pkl")
            print("\n✅ Sistem canlıya hazır!")
            # --- ONLINE/INCREMENTAL LEARNING ---
            if auto_retrain:
                print("\n🔄 Otomatik model güncelleme (adaptive retrain) kontrol ediliyor...")
                from modules.ai_model import AIModel
                ai_model = AIModel()
                retrain_result = ai_model.adaptive_retrain(performance_threshold=0.6, min_signals=50)
                if retrain_result:
                    print("✅ Otomatik model güncelleme başarıyla tetiklendi!")
                else:
                    print("ℹ️ Model güncelleme gerekmedi veya yeterli veri yok.")
        except Exception as e:
            print(f"❌ Hata oluştu: {str(e)}")
            raise

if __name__ == "__main__":
    # Models klasörünü oluştur
    os.makedirs('models', exist_ok=True)
    
    # Pipeline'ı çalıştır
    pipeline = AIPipeline()
    pipeline.run_full_pipeline() 

# Optuna ile hiperparametre optimizasyonu için örnek fonksiyon (isteğe bağlı)
def optuna_hyperparameter_search(X, y):
    import optuna
    from xgboost import XGBClassifier
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', verbosity=0)
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print('Optuna en iyi parametreler:', study.best_params)
    return study.best_params 