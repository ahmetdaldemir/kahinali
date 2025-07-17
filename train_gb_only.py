#!/usr/bin/env python3
"""
Sadece Gradient Boosting modelini eğit
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def train_gradient_boosting():
    """Sadece Gradient Boosting modelini eğit"""
    try:
        print("=== Gradient Boosting Model Eğitimi ===")
        
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
        
        # X ve y hazırla
        X = df[feature_cols]
        y = df['label_dynamic']  # Ana label
        
        print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Gradient Boosting modeli
        print("Gradient Boosting eğitiliyor...")
        gb = GradientBoostingClassifier(
            n_estimators=50,  # Daha az estimator (hızlı eğitim)
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=1  # İlerleme göster
        )
        
        gb.fit(X_train, y_train)
        
        # Test
        y_pred = gb.predict(X_test)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n=== Gradient Boosting Sonuçları ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Modeli kaydet
        model_path = os.path.join(Config.MODELS_DIR, 'gb_model.pkl')
        joblib.dump(gb, model_path)
        print(f"✅ Model kaydedildi: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = train_gradient_boosting()
    if success:
        print("\n🎉 Gradient Boosting eğitimi başarıyla tamamlandı!")
    else:
        print("\n❌ Gradient Boosting eğitimi başarısız!") 