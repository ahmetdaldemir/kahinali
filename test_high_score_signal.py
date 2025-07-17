#!/usr/bin/env python3
"""
Yüksek skorlu test sinyali oluşturma scripti
"""

import os
import sys
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.signal_manager import SignalManager
from config import Config

def create_high_score_test_signal():
    """Yüksek skorlu test sinyali oluştur ve kaydet"""
    try:
        print("=== Yüksek Skorlu Test Sinyali Oluşturuluyor ===")
        
        # SignalManager'ı başlat
        signal_manager = SignalManager()
        
        # Yüksek skorlu test verisi
        test_analysis_data = {
            'ai_score': 0.85,  # Yüksek AI skoru
            'ta_strength': 0.80,  # Yüksek TA gücü
            'whale_score': 0.75,  # Yüksek whale skoru
            'social_score': 0.70,  # Yüksek social skoru
            'news_score': 0.65,  # Yüksek news skoru
            'breakout_probability': 0.75,  # Yüksek breakout olasılığı
            'close': 50000.0,  # BTC fiyatı
            'current_price': 50000.0,
            'atr': 2500.0,  # ATR değeri
            'volatility': 0.08,  # Volatilite
            'volume_ratio': 1.5,  # Hacim oranı
            'trend_strength': 0.85,  # Trend gücü
            'market_cap': 1000000000,  # Market cap
            'volume_24h': 50000000,  # 24h hacim
            'price_change_24h': 0.05,  # 24h fiyat değişimi
            'timeframe': '1h'
        }
        
        # Yüksek confidence ile sinyal oluştur
        signal = signal_manager.create_signal(
            symbol='BTC/USDT',
            direction='LONG',
            confidence=0.85,  # Yüksek confidence
            analysis_data=test_analysis_data
        )
        
        if signal:
            print(f"✅ Test sinyali oluşturuldu:")
            print(f"   Symbol: {signal.get('symbol')}")
            print(f"   Direction: {signal.get('direction')}")
            print(f"   AI Score: {signal.get('ai_score')}")
            print(f"   TA Strength: {signal.get('ta_strength')}")
            print(f"   Confidence: {signal.get('confidence')}")
            print(f"   Predicted Breakout Threshold: {signal.get('predicted_breakout_threshold')}")
            print(f"   Predicted Breakout Time: {signal.get('predicted_breakout_time_hours')}")
            
            # Sinyali kaydet
            print("\n=== Sinyal Kaydediliyor ===")
            signal_manager.save_signal_db(signal)
            print("✅ Test sinyali veritabanına kaydedildi!")
            
            return True
        else:
            print("❌ Test sinyali oluşturulamadı!")
            return False
            
    except Exception as e:
        print(f"❌ Test sinyali oluşturma hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    create_high_score_test_signal() 