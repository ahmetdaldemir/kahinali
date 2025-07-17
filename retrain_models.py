#!/usr/bin/env python3
"""
AI Model Retraining Script
Bu script mevcut feature seti ile AI modellerini yeniden eğitir
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import glob

# Project root'u path'e ekle
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from modules.ai_model import AIModel
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis, FIXED_FEATURE_LIST

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_training_data():
    """Eğitim için veri topla"""
    try:
        logger.info("Eğitim verisi toplanıyor...")
        
        # Data collector
        collector = DataCollector()
        
        # Popüler coinler
        popular_coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        
        all_data = []
        
        for symbol in popular_coins:
            try:
                logger.info(f"{symbol} verisi toplanıyor...")
                
                # Farklı timeframe'ler için veri topla
                for timeframe in ['1h', '4h', '1d']:
                    try:
                        # Son 1000 veri
                        data = collector.get_historical_data(symbol, timeframe, limit=1000)
                        if data is not None and not data.empty:
                            data['symbol'] = symbol
                            data['timeframe'] = timeframe
                            all_data.append(data)
                            logger.info(f"{symbol} {timeframe}: {len(data)} satır")
                    except Exception as e:
                        logger.error(f"{symbol} {timeframe} veri toplama hatası: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"{symbol} veri toplama hatası: {e}")
                continue
        
        if not all_data:
            logger.error("Hiç veri toplanamadı!")
            return None
        
        # Tüm verileri birleştir
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Toplam {len(combined_data)} satır veri toplandı")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Veri toplama hatası: {e}")
        return None

def prepare_training_data(data):
    """Eğitim verisini hazırla"""
    try:
        logger.info("Eğitim verisi hazırlanıyor...")
        
        # Technical analysis
        ta = TechnicalAnalysis()
        
        # Her symbol için teknik analiz yap
        processed_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 100:  # Minimum veri kontrolü
                continue
                
            try:
                # Teknik analiz uygula
                symbol_data = ta.calculate_all_indicators(symbol_data)
                
                # NaN değerleri temizle
                symbol_data = symbol_data.dropna()
                
                if len(symbol_data) > 50:  # Yeterli veri kontrolü
                    processed_data.append(symbol_data)
                    logger.info(f"{symbol}: {len(symbol_data)} satır işlendi")
                    
            except Exception as e:
                logger.error(f"{symbol} teknik analiz hatası: {e}")
                continue
        
        if not processed_data:
            logger.error("İşlenmiş veri bulunamadı!")
            return None
        
        # Tüm verileri birleştir
        final_data = pd.concat(processed_data, ignore_index=True)
        
        # Son kontrol
        final_data = final_data.dropna()
        
        # FIXED_FEATURE_LIST ile karşılaştır
        expected_features = set(FIXED_FEATURE_LIST)
        actual_features = set(final_data.columns)
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        logger.info(f"Beklenen feature sayısı: {len(expected_features)}")
        logger.info(f"Gerçek feature sayısı: {len(actual_features)}")
        logger.info(f"Eksik feature'lar: {missing_features}")
        logger.info(f"Ekstra feature'lar: {extra_features}")
        
        # Eksik feature'ları 0 ile doldur
        for feature in missing_features:
            final_data[feature] = 0
            logger.info(f"Eksik feature dolduruldu: {feature}")
        
        logger.info(f"Final eğitim verisi: {len(final_data)} satır, {len(final_data.columns)} sütun")
        logger.info(f"Sütunlar: {list(final_data.columns)}")
        
        return final_data
        
    except Exception as e:
        logger.error(f"Veri hazırlama hatası: {e}")
        return None

def retrain_models(training_data):
    """AI modellerini yeniden eğit"""
    try:
        logger.info("AI modelleri yeniden eğitiliyor...")
        
        # --- MODELLERİ VE FEATURE DOSYALARINI OTOMATİK SİL ---
        model_files = [
            'lstm_model.h5',
            'rf_model.pkl',
            'gb_model.pkl',
            'scaler.pkl',
            'feature_cols.pkl'
        ]
        for f in model_files:
            fpath = os.path.join('models', f)
            if os.path.exists(fpath):
                os.remove(fpath)
                logger.info(f"Silindi: {fpath}")
        
        # Teknik analiz ile feature engineering
        ta = TechnicalAnalysis()
        training_data = ta.calculate_all_indicators(training_data)

        # Sadece FIXED_FEATURE_LIST'i kullan
        feature_cols = [col for col in FIXED_FEATURE_LIST if col in training_data.columns]
        training_data = training_data[feature_cols + ['label']] if 'label' in training_data.columns else training_data[feature_cols]

        # Feature listesi dosyasını kaydet
        import joblib
        joblib.dump(feature_cols, 'models/feature_cols.pkl')
        logger.info(f"Eğitimde kullanılan feature sayısı: {len(feature_cols)}")
        logger.info(f"Eğitimde feature isimleri: {feature_cols}")
        print(f"Eğitimde feature sayısı: {len(feature_cols)}")
        print(f"Eğitimde feature isimleri: {feature_cols}")
        
        # Sadece bu feature'ları kullan
        training_data = training_data[feature_cols]
        
        # --- EĞİTİM VERİSİNİ CSV'YE KAYDET (test için) ---
        training_data.to_csv('train_data_for_test.csv', index=False)
        
        # NaN/inf tespiti ve loglama
        nan_cols = training_data.columns[training_data.isnull().any()].tolist()
        inf_cols = training_data.columns[np.isinf(training_data.values).any(axis=0)].tolist()
        if nan_cols:
            logger.warning(f"NaN üreten feature'lar: {nan_cols}")
        if inf_cols:
            logger.warning(f"Inf üreten feature'lar: {inf_cols}")
        if not nan_cols and not inf_cols:
            logger.info("Hiçbir feature NaN veya inf üretmiyor.")
        
        # Temizlik
        training_data = training_data.replace([np.inf, -np.inf], 0)
        training_data = training_data.fillna(0)
        
        # AI model
        ai_model = AIModel()
        
        # Eğitim verisi kontrolü
        if training_data is None or training_data.empty:
            logger.error("Eğitim verisi yok!")
            return False
        
        # Minimum veri kontrolü
        if len(training_data) < 1000:
            logger.warning(f"Az veri: {len(training_data)} satır. En az 1000 satır önerilir.")
        
        # Feature sayısını kontrol et
        feature_count = len(feature_cols)
        logger.info(f"Feature sayısı: {feature_count}")
        
        # Kaydedilen feature_cols.pkl dosyasının içeriğini logla
        try:
            loaded_cols = joblib.load(feature_cols_path)
            logger.info(f"feature_cols.pkl içeriği: {loaded_cols}")
            print(f"feature_cols.pkl içeriği: {loaded_cols}")
        except Exception as e:
            logger.error(f"feature_cols.pkl okunamadı: {e}")
        
        # Modelleri eğit
        logger.info("LSTM modeli eğitiliyor...")
        lstm_model = ai_model.train_lstm(training_data, epochs=30, batch_size=32)
        
        logger.info("Random Forest modeli eğitiliyor...")
        rf_model = ai_model.train_rf(training_data)
        
        logger.info("Gradient Boosting modeli eğitiliyor...")
        gb_model = ai_model.train_gradient_boosting(training_data)
        
        # Modelleri kaydet
        ai_model.save_models()
        
        # Model dosyalarının boyutlarını kontrol et
        model_files_to_check = [
            'lstm_model.h5',
            'rf_model.pkl', 
            'gb_model.pkl',
            'scaler.pkl',
            'feature_cols.pkl'
        ]
        
        for model_file in model_files_to_check:
            file_path = os.path.join('models', model_file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"{model_file} boyutu: {file_size} bytes")
                if file_size < 1000:  # 1KB'dan küçükse uyarı
                    logger.warning(f"{model_file} çok küçük! ({file_size} bytes)")
            else:
                logger.error(f"{model_file} bulunamadı!")
        
        logger.info("Tüm modeller başarıyla eğitildi ve kaydedildi!")
        return True
        
    except Exception as e:
        logger.error(f"Model eğitimi hatası: {e}")
        return False


def test_models():
    """Eğitilen modelleri test et"""
    try:
        logger.info("Modeller test ediliyor...")
        
        # AI model
        ai_model = AIModel()
        
        # Eğitimde kullanılan feature listesini yükle
        import joblib
        feature_cols_path = os.path.join('models', 'feature_cols.pkl')
        if os.path.exists(feature_cols_path):
            feature_cols = joblib.load(feature_cols_path)
        else:
            logger.error("feature_cols.pkl bulunamadı!")
            return False
        
        # --- TEST VERİSİNİ YÜKLE ---
        if os.path.exists('train_data_for_test.csv'):
            test_data = pd.read_csv('train_data_for_test.csv')
        else:
            logger.error("train_data_for_test.csv bulunamadı!")
            return False
        
        # Teknik analiz ile feature engineering
        ta = TechnicalAnalysis()
        test_data = ta.calculate_all_indicators(test_data)
        
        # --- 0 ile doldurulan feature'ları logla ---
        eksikler = [col for col in feature_cols if col not in test_data.columns]
        if eksikler:
            logger.warning(f"0 ile doldurulan feature'lar: {eksikler}")
        for col in eksikler:
            test_data[col] = 0
        
        # --- NaN içeren feature'ları logla ---
        nan_olanlar = test_data.columns[test_data.isna().any()].tolist()
        if nan_olanlar:
            logger.warning(f"NaN içeren feature'lar: {nan_olanlar}")
        
        # Sütun isimlerini ve sayısını logla ve ekrana yaz
        logger.info(f"Testte kullanılan feature sayısı: {len(test_data.columns)}")
        logger.info(f"Testte feature isimleri: {list(test_data.columns)}")
        print(f"Testte feature sayısı: {len(test_data.columns)}")
        print(f"Testte feature isimleri: {list(test_data.columns)}")
        
        # Test tahminleri
        logger.info("LSTM test tahmini...")
        try:
            lstm_pred = ai_model.predict_lstm(test_data)
            logger.info(f"LSTM: {lstm_pred}")
        except Exception as e:
            logger.error(f"LSTM test hatası: {e}")
        
        logger.info("RF test tahmini...")
        try:
            rf_pred = ai_model.predict_rf(test_data)
            logger.info(f"RF: {rf_pred}")
        except Exception as e:
            logger.error(f"RF test hatası: {e}")
        
        logger.info("GB test tahmini...")
        try:
            gb_pred = ai_model.predict_gb(test_data)
            logger.info(f"GB: {gb_pred}")
        except Exception as e:
            logger.error(f"GB test hatası: {e}")
        
        logger.info("Model testleri tamamlandı!")
        return True
        
    except Exception as e:
        logger.error(f"Model test hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    try:
        logger.info("AI Model Retraining başlatılıyor...")
        
        # Eğitim verisini topla
        data = collect_training_data()
        if data is None or data.empty:
            logger.error("Eğitim verisi toplanamadı!")
            return
        # --- HIZLI MOD PATCH KALDIRILDI ---
        # data = data.head(100)  # Bu satır silindi, tüm veri kullanılacak
        # Eğitim verisini hazırla
        training_data = prepare_training_data(data)
        if training_data is None or training_data.empty:
            logger.error("Eğitim verisi hazırlanamadı!")
            return
        # Modelleri yeniden eğit
        retrain_models(training_data)
        # Test fonksiyonu varsa çalıştır
        try:
            test_models()
        except Exception as e:
            logger.error(f"Test fonksiyonu hatası: {e}")
        
        logger.info("AI Model Retraining tamamlandı!")
        return True
        
    except Exception as e:
        logger.error(f"Retraining hatası: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ AI Model Retraining başarılı!")
    else:
        print("❌ AI Model Retraining başarısız!")
        sys.exit(1) 