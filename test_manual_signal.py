#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.signal_manager import SignalManager

def create_test_signal():
    """Test için manuel sinyal oluştur"""
    
    # Test sinyal verisi - Tüm metrikler doğru şekilde ayarlandı
    test_signal = {
        "symbol": "TEST/USDT",
        "timeframe": "1h",
        "direction": "BUY",
        "ai_score": 0.75,
        "ta_strength": 0.80,
        "whale_score": 0.65,
        "social_score": 0.45,
        "news_score": 0.30,
        "current_price": 1.50,
        "entry_price": 1.50,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "predicted_gain": 8.5,
        "predicted_duration": "6-12 saat",
        "quality_score": 0.68,
        "quality_checks_passed": True,
        "confidence": 0.75,
        "feature_dummy_ratio": 0.0,
        "feature_quality_warning": False,
        "dynamic_strictness": 0.65,
        "strictness_level": "ORTA SIKILIK",
        "dynamic_min_confidence": 0.65,
        "dynamic_min_ai_score": 0.55,
        "dynamic_min_ta_strength": 0.60,
        "strictness_recommendation": "Dengeli sinyal - Orta sıkılık, güvenli fırsatlar",
        
        # Panelde gösterilmek istenen metrikler - Sayısal değerler
        "volume_score": 0.72,
        "momentum_score": 0.68,
        "pattern_score": 0.75,
        "whale_direction_score": 0.65,
        "order_book_imbalance": 0.58,
        "top_bid_walls": "1.48:50000, 1.47:75000, 1.46:100000",
        "top_ask_walls": "1.52:45000, 1.53:60000, 1.54:80000",
        
        # Diğer metrikler
        "breakout_probability": 0.70,
        "risk_reward_ratio": 2.5,
        "signal_strength": 0.73,
        "market_sentiment": 0.65,
        "take_profit": 1.62,
        "stop_loss": 1.44,
        "support_level": 1.47,
        "resistance_level": 1.53,
        "target_time_hours": 8.0,
        "predicted_breakout_threshold": 0.025,
        "predicted_breakout_time_hours": 6.0,
        
        # Success metrics
        "success_metrics": {
            "quality_score": 0.68,
            "volume_score": 0.72,
            "momentum_score": 0.68,
            "pattern_score": 0.75,
            "breakout_probability": 0.70,
            "risk_reward_ratio": 2.5,
            "confidence_level": 0.75,
            "signal_strength": 0.73,
            "market_sentiment": 0.65
        }
    }
    
    try:
        print("=== Test Sinyali Oluşturuluyor ===")
        
        # SignalManager'ı başlat
        signal_manager = SignalManager()
        
        # Sinyali kaydet
        print("Sinyal JSON kaydediliyor...")
        signal_manager.save_signal_json(test_signal)
        
        print("Sinyal CSV kaydediliyor...")
        signal_manager.save_signal_csv(test_signal)
        
        print("Sinyal veritabanına kaydediliyor...")
        signal_manager.save_signal_db(test_signal)
        
        print("✅ Test sinyali başarıyla oluşturuldu ve kaydedildi!")
        print(f"Symbol: {test_signal['symbol']}")
        print(f"Direction: {test_signal['direction']}")
        print(f"AI Score: {test_signal['ai_score']}")
        print(f"Volume Score: {test_signal['volume_score']}")
        print(f"Momentum Score: {test_signal['momentum_score']}")
        print(f"Pattern Score: {test_signal['pattern_score']}")
        print(f"Whale Direction: {test_signal['whale_direction_score']}")
        print(f"Order Book Imbalance: {test_signal['order_book_imbalance']}")
        print(f"Bid Walls: {test_signal['top_bid_walls']}")
        print(f"Ask Walls: {test_signal['top_ask_walls']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test sinyali oluşturulurken hata: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    create_test_signal() 