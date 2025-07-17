#!/usr/bin/env python3
"""
KAHİN ULTIMA - Hafif AI Model Eğitimi
Bellek sorunlarını önlemek için optimize edilmiş eğitim scripti
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import gc

warnings.filterwarnings('ignore')

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.ai_model import AIModel
from modules.data_collector import DataCollector

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lightweight_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def clean_infinite_values(df):
    """Sonsuz değerleri temizle"""
    logger.info("Sonsuz değerler temizleniyor...")
    
    # Sonsuz değerleri NaN'a çevir
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # NaN değerleri medyan ile doldur
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    logger.info(f"Temizleme sonrası veri boyutu: {df.shape}")
    return df

def prepare_training_data():
    """Eğitim verisini hazırla"""
    logger.info("=== HAFİF AI MODEL EĞİTİMİ BAŞLADI ===")
    
    try:
        # Veri dosyasını kontrol et
        data_file = 'train_data_with_target.csv'
        if not os.path.exists(data_file):
            logger.error(f"Veri dosyası bulunamadı: {data_file}")
            return None
        
        # Veriyi yükle
        logger.info("Eğitim verisi yükleniyor...")
        df = pd.read_csv(data_file)
        logger.info(f"Veri yüklendi: {df.shape}")
        
        # Label dağılımını kontrol et
        if 'target' in df.columns:
            label_dist = df['target'].value_counts()
            logger.info(f"Label dağılımı: {dict(label_dist)}")
        
        # Sonsuz değerleri temizle
        df = clean_infinite_values(df)
        
        # Feature'ları hazırla
        feature_columns = [col for col in df.columns if col not in ['target', 'timestamp', 'symbol']]
        logger.info(f"Feature sayısı: {len(feature_columns)}")
        
        # Veriyi böl
        X = df[feature_columns]
        y = df['target'] if 'target' in df.columns else None
        
        if y is None:
            logger.error("Target kolonu bulunamadı")
            return None
        
        # Veriyi eğitim ve test olarak böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Eğitim seti: {X_train.shape}")
        logger.info(f"Test seti: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
        
    except Exception as e:
        logger.error(f"Veri hazırlama hatası: {e}")
        return None

def train_random_forest(X_train, X_test, y_train, y_test, feature_columns):
    """Random Forest modelini eğit"""
    logger.info("Random Forest eğitiliyor...")
    
    try:
        # Daha küçük parametrelerle model oluştur
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Daha az ağaç
            max_depth=10,       # Daha sığ ağaçlar
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1  # Tek thread kullan
        )
        
        # Modeli eğit
        rf_model.fit(X_train, y_train)
        
        # Tahmin yap
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Random Forest doğruluk: {accuracy:.4f}")
        
        # Modeli kaydet
        model_path = 'models/random_forest_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        logger.info(f"Random Forest modeli kaydedildi: {model_path}")
        
        # Feature importance'ları kaydet
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = 'models/feature_importance.pkl'
        with open(importance_path, 'wb') as f:
            pickle.dump(feature_importance, f)
        logger.info(f"Feature importance kaydedildi: {importance_path}")
        
        return rf_model, accuracy
        
    except Exception as e:
        logger.error(f"Random Forest eğitim hatası: {e}")
        return None, 0

def train_gradient_boosting(X_train, X_test, y_train, y_test, feature_columns):
    """Gradient Boosting modelini eğit"""
    logger.info("Gradient Boosting eğitiliyor...")
    
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Daha küçük parametrelerle model oluştur
        gb_model = GradientBoostingClassifier(
            n_estimators=50,    # Daha az ağaç
            max_depth=6,         # Daha sığ ağaçlar
            learning_rate=0.1,
            random_state=42
        )
        
        # Modeli eğit
        gb_model.fit(X_train, y_train)
        
        # Tahmin yap
        y_pred = gb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Gradient Boosting doğruluk: {accuracy:.4f}")
        
        # Modeli kaydet
        model_path = 'models/gradient_boosting_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(gb_model, f)
        logger.info(f"Gradient Boosting modeli kaydedildi: {model_path}")
        
        return gb_model, accuracy
        
    except Exception as e:
        logger.error(f"Gradient Boosting eğitim hatası: {e}")
        return None, 0

def create_ensemble_model(rf_model, gb_model, feature_columns):
    """Ensemble model oluştur"""
    logger.info("Ensemble model oluşturuluyor...")
    
    try:
        ensemble = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'feature_columns': feature_columns,
            'created_at': datetime.now().isoformat()
        }
        
        # Ensemble modeli kaydet
        ensemble_path = 'models/ensemble_model.pkl'
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble, f)
        logger.info(f"Ensemble model kaydedildi: {ensemble_path}")
        
        return ensemble
        
    except Exception as e:
        logger.error(f"Ensemble model oluşturma hatası: {e}")
        return None

def main():
    """Ana eğitim fonksiyonu"""
    logger.info("🚀 HAFİF AI MODEL EĞİTİMİ BAŞLATIYOR")
    
    try:
        # Belleği temizle
        gc.collect()
        
        # Veriyi hazırla
        data_result = prepare_training_data()
        if data_result is None:
            logger.error("Veri hazırlama başarısız")
            return False
        
        X_train, X_test, y_train, y_test, feature_columns = data_result
        
        # Belleği temizle
        gc.collect()
        
        # Random Forest eğit
        rf_model, rf_accuracy = train_random_forest(X_train, X_test, y_train, y_test, feature_columns)
        
        # Belleği temizle
        gc.collect()
        
        # Gradient Boosting eğit
        gb_model, gb_accuracy = train_gradient_boosting(X_train, X_test, y_train, y_test, feature_columns)
        
        # Belleği temizle
        gc.collect()
        
        # Ensemble model oluştur
        if rf_model is not None and gb_model is not None:
            ensemble = create_ensemble_model(rf_model, gb_model, feature_columns)
            
            # Sonuç raporu
            logger.info("=" * 50)
            logger.info("📊 EĞİTİM SONUÇLARI")
            logger.info("=" * 50)
            logger.info(f"Random Forest Doğruluk: {rf_accuracy:.4f}")
            logger.info(f"Gradient Boosting Doğruluk: {gb_accuracy:.4f}")
            logger.info(f"Ortalama Doğruluk: {(rf_accuracy + gb_accuracy) / 2:.4f}")
            logger.info("✅ Eğitim başarıyla tamamlandı!")
            
            return True
        else:
            logger.error("Model eğitimi başarısız")
            return False
            
    except Exception as e:
        logger.error(f"Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Hafif eğitim başarıyla tamamlandı!")
    else:
        print("❌ Eğitim başarısız oldu!") 