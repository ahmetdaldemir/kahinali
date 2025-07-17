#!/usr/bin/env python3
"""
KAHİN Ultima - Tam Sistem Entegrasyon Testi
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database():
    """Veritabanı bağlantısını test et"""
    print("🔍 Veritabanı test ediliyor...")
    try:
        from modules.signal_manager import SignalManager
        sm = SignalManager()
        print("✅ Veritabanı bağlantısı başarılı")
        return True
    except Exception as e:
        print(f"❌ Veritabanı hatası: {e}")
        return False

def test_data_collector():
    """Veri toplama modülünü test et"""
    print("🔍 Veri toplama modülü test ediliyor...")
    try:
        from modules.data_collector import DataCollector
        dc = DataCollector()
        print("✅ Veri toplama modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ Veri toplama hatası: {e}")
        return False

def test_technical_analysis():
    """Teknik analiz modülünü test et"""
    print("🔍 Teknik analiz modülü test ediliyor...")
    try:
        from modules.technical_analysis import TechnicalAnalysis
        ta = TechnicalAnalysis()
        print("✅ Teknik analiz modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ Teknik analiz hatası: {e}")
        return False

def test_ai_model():
    """AI model modülünü test et"""
    print("🔍 AI model modülü test ediliyor...")
    try:
        from modules.ai_model import AIModel
        ai = AIModel()
        print("✅ AI model modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ AI model hatası: {e}")
        return False

def test_social_media():
    """Sosyal medya modülünü test et"""
    print("🔍 Sosyal medya modülü test ediliyor...")
    try:
        from modules.social_media import SocialMediaSentiment
        sma = SocialMediaSentiment()
        print("✅ Sosyal medya modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ Sosyal medya hatası: {e}")
        return False

def test_news_analysis():
    """Haber analizi modülünü test et"""
    print("🔍 Haber analizi modülü test ediliyor...")
    try:
        from modules.news_analysis import NewsAnalysis
        na = NewsAnalysis()
        print("✅ Haber analizi modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ Haber analizi hatası: {e}")
        return False

def test_whale_tracker():
    """Whale tracking modülünü test et"""
    print("🔍 Whale tracking modülü test ediliyor...")
    try:
        from modules.whale_tracker import WhaleTracker
        wt = WhaleTracker()
        print("✅ Whale tracking modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ Whale tracking hatası: {e}")
        return False

def test_telegram_bot():
    """Telegram bot modülünü test et"""
    print("🔍 Telegram bot modülü test ediliyor...")
    try:
        from modules.telegram_bot import TelegramBot
        tb = TelegramBot()
        print("✅ Telegram bot modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ Telegram bot hatası: {e}")
        return False

def test_performance():
    """Performans analizi modülünü test et"""
    print("🔍 Performans analizi modülü test ediliyor...")
    try:
        from modules.performance import PerformanceAnalyzer
        pa = PerformanceAnalyzer()
        print("✅ Performans analizi modülü yüklendi")
        return True
    except Exception as e:
        print(f"❌ Performans analizi hatası: {e}")
        return False

def test_web_panel():
    """Web panel test et"""
    print("🔍 Web panel test ediliyor...")
    try:
        import requests
        response = requests.get("http://localhost:5000/api/signals", timeout=5)
        if response.status_code == 200:
            print("✅ Web panel API çalışıyor")
            return True
        else:
            print(f"❌ Web panel API hatası: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web panel hatası: {e}")
        return False

def test_config():
    """Konfigürasyon test et"""
    print("🔍 Konfigürasyon test ediliyor...")
    try:
        from config import Config
        print(f"✅ Konfigürasyon yüklendi")
        print(f"   - PostgreSQL: {Config.DB_HOST}:{Config.DB_PORT}")
        print(f"   - Database: {Config.DB_NAME}")
        print(f"   - Flask: {Config.FLASK_HOST}:{Config.FLASK_PORT}")
        return True
    except Exception as e:
        print(f"❌ Konfigürasyon hatası: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 KAHİN Ultima - Tam Sistem Entegrasyon Testi")
    print("=" * 50)
    
    tests = [
        ("Konfigürasyon", test_config),
        ("Veritabanı", test_database),
        ("Veri Toplama", test_data_collector),
        ("Teknik Analiz", test_technical_analysis),
        ("AI Model", test_ai_model),
        ("Sosyal Medya", test_social_media),
        ("Haber Analizi", test_news_analysis),
        ("Whale Tracking", test_whale_tracker),
        ("Telegram Bot", test_telegram_bot),
        ("Performans Analizi", test_performance),
        ("Web Panel", test_web_panel),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test hatası: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 TEST SONUÇLARI:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Toplam: {total} test, Başarılı: {passed}, Başarısız: {total - passed}")
    
    if passed == total:
        print("🎉 TÜM MODÜLLER BAŞARIYLA ÇALIŞIYOR!")
    else:
        print("⚠️  BAZI MODÜLLERDE SORUN VAR!")
        print("\n🔧 Öneriler:")
        print("1. API anahtarlarını .env dosyasına ekleyin")
        print("2. Web panelini başlatın: python app/web.py")
        print("3. Eksik modülleri kontrol edin")
    
    return passed == total

if __name__ == "__main__":
    main() 