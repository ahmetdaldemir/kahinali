#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create logs directory
if not os.path.exists('logs'):
    os.makedirs('logs')

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kahin_ultima.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    from config import Config
    logger.info("Config imported successfully")
except ImportError as e:
    logger.error(f"Failed to import config: {e}")
    sys.exit(1)

try:
    from modules.data_collector import DataCollector
    from modules.technical_analysis import TechnicalAnalysis
    from modules.signal_manager import SignalManager
    from modules.telegram_bot import TelegramBot
    from modules.market_analysis import MarketAnalysis
    from modules.performance import PerformanceAnalyzer
    from modules.dynamic_strictness import DynamicStrictness
    from modules.signal_tracker import SignalTracker
    from modules.ai_model import AIModel
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

# Hatalı coin ve hata mesajlarını merkezi olarak tutan global bir liste
FAILED_COINS = []

def main():
    """Ana fonksiyon"""
    try:
        logger.info("Kahin Ultima baslatiliyor...")
        
        # Config directories oluştur
        Config.create_directories()
        
        # Data collector instance
        data_collector = DataCollector()
        
        # Market analysis instance
        market_analyzer = MarketAnalysis()
        
        # Performance analyzer
        performance_analyzer = PerformanceAnalyzer()
        
        # PostgreSQL tabloları oluştur
        signal_manager = SignalManager()
        
        # SignalTracker instance
        signal_tracker = SignalTracker()
        
        logger.info("Tüm modüller başarıyla yüklendi")
        
        # Basit zamanlayıcı - son sinyal uretim zamanını takip et
        last_signal_time = datetime.now()
        last_signal_close_check = datetime.now()
        signal_interval = 60  # 1 dakika
        close_check_interval = 600  # 10 dakika
        adapt_interval = 1800  # 30 dakika
        last_adapt_check = datetime.now()
        
        logger.info("Ana döngü başlatılıyor...")
        
        # Ana döngü - basit zamanlayıcı ile
        while True:
            try:
                current_time = datetime.now()
                time_since_last_signal = (current_time - last_signal_time).total_seconds()
                time_since_last_close_check = (current_time - last_signal_close_check).total_seconds()
                time_since_last_adapt = (current_time - last_adapt_check).total_seconds()
                
                # Her dakika sinyal üretimi
                if time_since_last_signal >= signal_interval:
                    logger.info("[TIMER] Sinyal uretimi zamanı geldi, fonksiyon cagriliyor...")
                    try:
                        generate_signals_wrapper(market_analyzer, performance_analyzer)
                        last_signal_time = current_time
                        logger.info("[TIMER] Sinyal uretimi tamamlandı, sonraki kontrol: 1 dakika sonra")
                    except Exception as e:
                        logger.error(f"[TIMER] Sinyal uretimi hatasi: {e}")

                # Her 10 dakikada bir açık sinyalleri kontrol et ve kapat
                if time_since_last_close_check >= close_check_interval:
                    logger.info("[TIMER] Açık sinyaller kontrol ediliyor...")
                    try:
                        signal_tracker.track_open_signals()
                        last_signal_close_check = current_time
                        logger.info("[TIMER] Açık sinyal kontrolü tamamlandı, sonraki kontrol: 10 dakika sonra")
                    except Exception as e:
                        logger.error(f"[TIMER] Açık sinyal kontrolü hatası: {e}")

                # Her 30 dakikada bir yeni coin ve piyasa adaptasyonu
                if time_since_last_adapt >= adapt_interval:
                    logger.info("[TIMER] Yeni coin ve piyasa adaptasyonu kontrol ediliyor...")
                    try:
                        data_collector.adapt_to_market_conditions()
                        last_adapt_check = current_time
                        logger.info("[TIMER] Adaptasyon tamamlandı, sonraki kontrol: 30 dakika sonra")
                    except Exception as e:
                        logger.error(f"[TIMER] Adaptasyon hatası: {e}")
                
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Program kullanıcı tarafından durduruldu")
                break
            except Exception as e:
                logger.error(f"Ana döngüde hata: {e}")
                time.sleep(60)  # 1 dakika bekle ve devam et
                
    except Exception as e:
        logger.error(f"Program baslatma hatasi: {e}")
        raise

def generate_signals_wrapper(market_analyzer, performance_analyzer):
    """Sinyal üretimi wrapper fonksiyonu"""
    try:
        logger.info("Sinyal uretimi baslatiliyor...")
        
        # Modülleri başlat
        data_collector = DataCollector()
        technical_analyzer = TechnicalAnalysis()
        signal_manager = SignalManager()
        telegram_bot = TelegramBot()
        ai_model = AIModel()
        
        # Dinamik sıkılık sistemi başlat
        dynamic_strictness = DynamicStrictness()
        logger.info("✅ Dinamik sıkılık sistemi başlatıldı")
        
        # Popüler coinleri al
        popular_coins = data_collector.get_popular_usdt_pairs(max_pairs=Config.MAX_COINS_TO_TRACK)
        logger.info(f"Toplam {len(popular_coins)} coin işlenecek")
        
        # İlk 10 coin'i test et
        test_coins = popular_coins[:10]
        signals_generated = 0
        
        for coin in test_coins:
            try:
                logger.info(f"İşleniyor: {coin}")
                
                # Veri topla
                df = data_collector.get_historical_data(coin, '1h', limit=100)
                if df.empty:
                    logger.warning(f"❌ {coin}: Veri alinamadi, atlaniyor")
                    continue
                
                logger.info(f"✅ {coin}: {len(df)} satir veri alindı")
                
                # Teknik analiz
                df = technical_analyzer.calculate_all_indicators(df)
                if df.empty:
                    logger.warning(f"❌ {coin}: Teknik analiz basarisiz, atlaniyor")
                    continue
                
                logger.info(f"✅ {coin}: Teknik analiz tamamlandı")
                
                # AI analizi
                try:
                    logger.info(f"🤖 {coin}: AI analizi basliyor...")
                    ai_result = ai_model.predict(df)
                    
                    if ai_result is None:
                        logger.warning(f"❌ {coin}: AI sonucu None, atlanıyor")
                        continue
                    
                    logger.info(f"✅ {coin}: AI tahmini tamamlandi")
                    signals_generated += 1
                    
                except Exception as e:
                    logger.error(f"❌ {coin}: AI analizi hatasi: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"{coin}: [EXCEPTION] Coin işlenirken beklenmeyen hata: {str(e)}")
                continue
        
        logger.info(f"🎯 Toplam {signals_generated} sinyal başarıyla işlendi.")
        
    except Exception as e:
        logger.error(f"Sinyal uretimi hatasi: {e}")
        import traceback
        logger.error(f"Hata detayi: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 