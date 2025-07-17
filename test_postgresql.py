#!/usr/bin/env python3
"""
PostgreSQL Bağlantı Test Scripti
"""

from modules.signal_manager import SignalManager
import datetime

def test_postgresql():
    print("🔍 PostgreSQL bağlantısı test ediliyor...")
    
    try:
        # Signal Manager'ı başlat
        sm = SignalManager()
        print("✅ Signal Manager başlatıldı")
        
        # Test sinyali oluştur
        test_signal = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'direction': 'BUY',
            'ai_score': 0.85,
            'ta_strength': 0.78,
            'whale_score': 0.72,
            'social_score': 0.68,
            'news_score': 0.75,
            'timestamp': str(datetime.datetime.now()),
            'predicted_gain': 5.2,
            'predicted_duration': '2h'
        }
        
        # Sinyali kaydet
        sm.save_signal_db(test_signal)
        print("✅ Test sinyali PostgreSQL'e kaydedildi")
        
        # Sinyalleri yükle
        signals = sm.get_latest_signals(5)
        print(f"✅ {len(signals)} sinyal yüklendi")
        
        # Performans istatistikleri
        stats = sm.get_performance_stats()
        print(f"✅ Performans istatistikleri: {stats}")
        
        print("\n🎉 PostgreSQL bağlantısı başarılı!")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

if __name__ == "__main__":
    test_postgresql() 