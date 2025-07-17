#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from modules.whale_tracker import WhaleTracker
from modules.news_analysis import NewsAnalysis
from modules.signal_manager import SignalManager
from modules.telegram_bot import TelegramBot
from config import Config
import logging

# Logging ayarla
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def safe_format(val):
    try:
        return f"{float(val):.2f}"
    except Exception:
        return "N/A"

def test_signal_generation():
    """Sinyal üretimini test et"""
    try:
        # Modülleri başlat
        data_collector = DataCollector()
        ta = TechnicalAnalysis()
        ai_model = AIModel()
        whale_tracker = WhaleTracker()
        news_analysis = NewsAnalysis()
        signal_manager = SignalManager()
        telegram_bot = TelegramBot()
        
        # Test için tek bir coin al
        pairs = data_collector.get_usdt_pairs(max_pairs=1)
        if not pairs:
            logger.error("Test için coin bulunamadı")
            return
            
        pair = pairs[0]
        logger.info(f"Test edilen coin: {pair}")
        
        # Veri topla
        df = data_collector.get_historical_data(pair, '1h', limit=100)
        if df.empty:
            logger.error("Veri toplanamadı")
            return
            
        logger.info(f"Veri toplandı: {len(df)} satır")
        
        # Teknik analiz
        df = ta.calculate_all_indicators(df)
        if df.empty:
            logger.error("Teknik analiz başarısız")
            return
            
        ta_signals = ta.generate_signals(df)
        ta_strength = ta.calculate_signal_strength(ta_signals)
        logger.info(f"TA Strength: {ta_strength}")
        
        # AI tahminleri
        try:
            ai_direction, ai_score = ai_model.predict_lstm(df)
            logger.info(f"LSTM: {ai_direction}, {ai_score}")
        except Exception as e:
            logger.error(f"LSTM hatası: {e}")
            ai_direction, ai_score = "NEUTRAL", 0.5
            
        try:
            rf_direction, rf_score = ai_model.predict_rf(df)
            logger.info(f"RF: {rf_direction}, {rf_score}")
        except Exception as e:
            logger.error(f"RF hatası: {e}")
            rf_direction, rf_score = "NEUTRAL", 0.5
        
        # Whale etkisi
        try:
            whale_score = whale_tracker.get_whale_signal(pair)
            logger.info(f"Whale: {whale_score}")
        except Exception as e:
            logger.error(f"Whale hatası: {e}")
            whale_score = 0.5
        
        # News etkisi
        try:
            coin = pair.split('/')[0]
            news_impact, _ = news_analysis.get_news_impact([coin])
            news_score = news_impact.get(coin, {}).get('impact_score', 0.5) if news_impact.get(coin) else 0.5
            logger.info(f"News: {news_score}")
        except Exception as e:
            logger.error(f"News hatası: {e}")
            news_score = 0.5
        
        # Tüm değerleri kontrol et
        logger.info("=== DEĞER KONTROLLERİ ===")
        logger.info(f"ai_direction: {ai_direction} (type: {type(ai_direction)})")
        logger.info(f"rf_direction: {rf_direction} (type: {type(rf_direction)})")
        logger.info(f"ai_score: {safe_format(ai_score)} (type: {type(ai_score)})")
        logger.info(f"rf_score: {safe_format(rf_score)} (type: {type(rf_score)})")
        logger.info(f"ta_strength: {safe_format(ta_strength)} (type: {type(ta_strength)})")
        logger.info(f"whale_score: {safe_format(whale_score)} (type: {type(whale_score)})")
        logger.info(f"news_score: {safe_format(news_score)} (type: {type(news_score)})")
        
        # String concatenation testi
        logger.info("=== STRING CONCATENATION TEST ===")
        try:
            test_str = f"Test: {ai_direction} - {rf_direction}"
            logger.info(f"String concatenation başarılı: {test_str}")
        except Exception as e:
            logger.error(f"String concatenation hatası: {e}")
        
        # Toplam skor hesapla
        try:
            total_score = (
                float(ai_score) * 0.4 +
                float(rf_score) * 0.3 +
                float(ta_strength) * 0.2 +
                float(whale_score) * 0.05 +
                0.5 * 0.025 +  # social_score
                float(news_score) * 0.025
            )
            logger.info(f"Toplam skor: {total_score}")
        except Exception as e:
            logger.error(f"Skor hesaplama hatası: {e}")
        
        # Ana yön belirleme
        try:
            main_direction = ai_direction if ai_score > rf_score else rf_direction
            logger.info(f"Ana yön: {main_direction}")
        except Exception as e:
            logger.error(f"Yön belirleme hatası: {e}")
        
        logger.info("Test tamamlandı!")
        
    except Exception as e:
        logger.error(f"Genel test hatası: {e}")
        import traceback
        logger.error(f"Hata detayı: {traceback.format_exc()}")

if __name__ == "__main__":
    test_signal_generation() 