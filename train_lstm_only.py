#!/usr/bin/env python3
"""
Sadece LSTM Modelini Eğitme Scripti
Bu script sadece LSTM modelini eğitir ve test eder.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Project root'u path'e ekle
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from modules.ai_model import AIModel
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_simple_data():
    """Basit eğitim verisi topla"""
    try:
        logger.info("Basit eğitim verisi toplanıyor...")
        
        # Data collector
        collector = DataCollector()
        
        # Sadece BTC/USDT için veri topla
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        logger.info(f"{symbol} {timeframe} verisi toplanıyor...")
        data = collector.get_historical_data(symbol, timeframe, limit=1000)
        
        if data is None or data.empty:
            logger.error("Veri toplanamadı!")
            return None
        
        logger.info(f"Toplam {len(data)} satır veri toplandı")
        return data
        
    except Exception as e:
        logger.error(f"Veri toplama hatası: {e}")
        return None

def prepare_data_for_lstm(data):
    """LSTM için veriyi hazırla"""
    try:
        logger.info("LSTM için veri hazırlanıyor...")
        
        # Technical analysis
        ta = TechnicalAnalysis()
        
        # Teknik analiz uygula
        data = ta.calculate_all_indicators(data)
        
        # NaN değerleri temizle
        data = data.dropna()
        
        logger.info(f"İşlenmiş veri: {len(data)} satır, {len(data.columns)} sütun")
        logger.info(f"Sütunlar: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        logger.error(f"Veri hazırlama hatası: {e}")
        return None

def train_lstm_model(data):
    """LSTM modelini eğit"""
    try:
        logger.info("LSTM modeli eğitiliyor...")
        
        # AI model
        ai_model = AIModel()
        
        # Eğitim verisi kontrolü
        if data is None or data.empty:
            logger.error("Eğitim verisi yok!")
            return False
        
        # Minimum veri kontrolü
        if len(data) < 100:
            logger.error(f"Çok az veri: {len(data)} satır. En az 100 satır gerekli.")
            return False
        
        logger.info(f"Eğitim verisi: {len(data)} satır")
        
        # LSTM modelini eğit
        lstm_model = ai_model.train_lstm(data, epochs=20, batch_size=32)
        
        if lstm_model is not None:
            logger.info("LSTM modeli başarıyla eğitildi!")
            return True
        else:
            logger.error("LSTM modeli eğitilemedi!")
            return False
        
    except Exception as e:
        logger.error(f"LSTM eğitimi hatası: {e}")
        return False

def test_lstm_model():
    """Eğitilen LSTM modelini test et"""
    try:
        logger.info("LSTM modeli test ediliyor...")
        
        # AI model
        ai_model = AIModel()
        
        # Test verisi oluştur
        test_data = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50500, 50600, 50700, 50800, 50900],
            'low': [49500, 49600, 49700, 49800, 49900],
            'close': [50000, 50100, 50200, 50300, 50400],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'rsi_14': [50, 55, 60, 65, 70],
            'macd': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sma_20': [50000, 50050, 50100, 50150, 50200],
            'bb_upper': [51000, 51100, 51200, 51300, 51400],
            'bb_lower': [49000, 49100, 49200, 49300, 49400],
            'atr': [100, 110, 120, 130, 140],
            'obv': [1000000, 1100000, 1200000, 1300000, 1400000],
            'vwap': [50000, 50050, 50100, 50150, 50200],
            'adx': [25, 30, 35, 40, 45],
            'cci': [0, 50, 100, 150, 200],
            'mfi': [50, 55, 60, 65, 70],
            'williams_r': [-20, -15, -10, -5, 0],
            'psar': [49000, 49100, 49200, 49300, 49400],
            'tenkan_sen': [50000, 50050, 50100, 50150, 50200],
            'kijun_sen': [50000, 50025, 50050, 50075, 50100],
            'senkou_span_a': [50000, 50050, 50100, 50150, 50200],
            'senkou_span_b': [50000, 50025, 50050, 50075, 50100],
            'chikou_span': [50000, 50050, 50100, 50150, 50200],
            'keltner_ema': [50000, 50050, 50100, 50150, 50200],
            'keltner_upper': [51000, 51100, 51200, 51300, 51400],
            'keltner_lower': [49000, 49100, 49200, 49300, 49400],
            'volume_roc': [0.1, 0.2, 0.3, 0.4, 0.5],
            'volume_ma': [1000, 1100, 1200, 1300, 1400],
            'volume_ratio': [1.0, 1.1, 1.2, 1.3, 1.4],
            'roc': [0.01, 0.02, 0.03, 0.04, 0.05],
            'momentum': [0.01, 0.02, 0.03, 0.04, 0.05],
            'price_roc': [0.01, 0.02, 0.03, 0.04, 0.05],
            'historical_volatility': [0.02, 0.025, 0.03, 0.035, 0.04],
            'true_range': [100, 110, 120, 130, 140],
            'volatility_ratio': [1.0, 1.1, 1.2, 1.3, 1.4],
            'price_change': [0.01, 0.02, 0.03, 0.04, 0.05],
            'price_change_5': [0.05, 0.06, 0.07, 0.08, 0.09],
            'price_change_10': [0.1, 0.11, 0.12, 0.13, 0.14],
            'return_5': [0.05, 0.06, 0.07, 0.08, 0.09],
            'return_10': [0.1, 0.11, 0.12, 0.13, 0.14],
            'return_20': [0.15, 0.16, 0.17, 0.18, 0.19],
            'cumulative_return': [0.1, 0.12, 0.14, 0.16, 0.18],
            'momentum_5': [0.01, 0.02, 0.03, 0.04, 0.05],
            'momentum_10': [0.02, 0.04, 0.06, 0.08, 0.1],
            'volatility': [0.02, 0.025, 0.03, 0.035, 0.04],
            'volatility_5': [0.02, 0.025, 0.03, 0.035, 0.04],
            'volatility_10': [0.02, 0.025, 0.03, 0.035, 0.04],
            'volatility_20': [0.02, 0.025, 0.03, 0.035, 0.04],
            'volume_ma_5': [1000, 1100, 1200, 1300, 1400],
            'volume_ma_10': [1000, 1100, 1200, 1300, 1400],
            'dynamic_threshold': [0.03, 0.035, 0.04, 0.045, 0.05],
            'label_5': [0, 1, 0, 1, 0],
            'label_10': [0, 1, 0, 1, 0],
            'label_20': [0, 1, 0, 1, 0],
            'label_dynamic': [0, 1, 0, 1, 0],
            'day_of_week': [0, 1, 2, 3, 4],
            'hour': [0, 6, 12, 18, 23]
        })
        
        # Label ekle
        test_data['label'] = [0, 1, 0, 1, 0]
        
        # Test tahmini
        logger.info("LSTM test tahmini...")
        result = ai_model.predict(test_data)
        
        if result is not None:
            logger.info(f"LSTM tahmin sonucu: {result}")
            logger.info("LSTM modeli test başarılı!")
            return True
        else:
            logger.error("LSTM tahmin başarısız!")
            return False
        
    except Exception as e:
        logger.error(f"LSTM test hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    try:
        logger.info("LSTM Model Eğitimi başlatılıyor...")
        
        # 1. Veri topla
        raw_data = collect_simple_data()
        if raw_data is None:
            logger.error("Veri toplanamadı!")
            return False
        
        # 2. Veriyi hazırla
        training_data = prepare_data_for_lstm(raw_data)
        if training_data is None:
            logger.error("Veri hazırlanamadı!")
            return False
        
        # 3. LSTM modelini eğit
        success = train_lstm_model(training_data)
        if not success:
            logger.error("LSTM eğitimi başarısız!")
            return False
        
        # 4. Modeli test et
        test_success = test_lstm_model()
        if not test_success:
            logger.warning("LSTM testi başarısız!")
        
        logger.info("LSTM Model Eğitimi tamamlandı!")
        return True
        
    except Exception as e:
        logger.error(f"LSTM eğitimi hatası: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ LSTM Model Eğitimi başarılı!")
    else:
        print("❌ LSTM Model Eğitimi başarısız!")
        sys.exit(1) 