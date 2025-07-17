#!/usr/bin/env python3
"""
Sinyal Filtreleme Test Scripti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.signal_manager import SignalManager
from config import Config

def test_signal_filtering():
    """Sinyal filtreleme sistemini test et"""
    print("🧪 SİNYAL FİLTRELEME TESTİ")
    print("=" * 50)
    
    # SignalManager'ı başlat
    signal_manager = SignalManager()
    
    # Test sinyali oluştur (loglardan alınan değerler)
    test_signal = {
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'direction': 'LONG',
        'ai_score': 0.2672,
        'ta_strength': 0.1333,
        'whale_score': 0.5,
        'social_score': 0.5,
        'news_score': 0.5,
        'current_price': 50000,
        'entry_price': 50000,
        'timestamp': '2025-07-02 17:32:02',
        'predicted_gain': 5.0,
        'predicted_duration': '4-8 saat',
        'quality_score': 0.3152,
        'confidence': 0.7000
    }
    
    print(f"Test sinyali:")
    print(f"  Symbol: {test_signal['symbol']}")
    print(f"  AI Score: {test_signal['ai_score']}")
    print(f"  TA Strength: {test_signal['ta_strength']}")
    print(f"  Quality Score: {test_signal['quality_score']}")
    print(f"  Confidence: {test_signal['confidence']}")
    
    # 1. Sinyal yönünü belirle
    direction = signal_manager.determine_signal_direction(test_signal)
    print(f"\n1. Sinyal yönü: {direction}")
    
    # 2. Sinyal skorunu hesapla
    signal_score = signal_manager.calculate_signal_score(test_signal)
    print(f"2. Hesaplanan sinyal skoru: {signal_score}")
    
    # 3. Filtreleme testi
    print(f"\n3. FİLTRELEME TESTİ")
    print(f"   Config.MIN_SIGNAL_CONFIDENCE: {Config.MIN_SIGNAL_CONFIDENCE}")
    print(f"   Config.MIN_AI_SCORE: {Config.MIN_AI_SCORE}")
    print(f"   Config.MIN_TA_STRENGTH: {Config.MIN_TA_STRENGTH}")
    print(f"   Config.SIGNAL_QUALITY_THRESHOLD: {Config.SIGNAL_QUALITY_THRESHOLD}")
    print(f"   Config.MIN_CONFIDENCE_THRESHOLD: {Config.MIN_CONFIDENCE_THRESHOLD}")
    
    # İlk filtreleme testi
    filtered_signals = signal_manager.filter_signals(
        [test_signal],
        min_confidence=Config.MIN_SIGNAL_CONFIDENCE,
        max_signals=5
    )
    
    print(f"\n   İlk filtreleme sonucu: {len(filtered_signals)} sinyal geçti")
    
    if filtered_signals:
        print(f"   ✅ Sinyal filtrelemeden geçti!")
        for signal in filtered_signals:
            print(f"   Final score: {signal.get('final_score', 'N/A')}")
    else:
        print(f"   ❌ Sinyal filtrelemeden geçemedi!")
        print(f"   Sebep: signal_score ({signal_score}) < min_confidence ({Config.MIN_SIGNAL_CONFIDENCE})")
    
    # Manuel filtreleme testi
    print(f"\n4. MANUEL FİLTRELEME TESTİ")
    
    ai_passed = test_signal['ai_score'] >= Config.MIN_AI_SCORE
    ta_passed = test_signal['ta_strength'] >= Config.MIN_TA_STRENGTH
    quality_passed = test_signal['quality_score'] >= Config.SIGNAL_QUALITY_THRESHOLD
    confidence_passed = test_signal['confidence'] >= Config.MIN_CONFIDENCE_THRESHOLD
    
    print(f"   AI Score: {test_signal['ai_score']} >= {Config.MIN_AI_SCORE} = {ai_passed}")
    print(f"   TA Strength: {test_signal['ta_strength']} >= {Config.MIN_TA_STRENGTH} = {ta_passed}")
    print(f"   Quality Score: {test_signal['quality_score']} >= {Config.SIGNAL_QUALITY_THRESHOLD} = {quality_passed}")
    print(f"   Confidence: {test_signal['confidence']} >= {Config.MIN_CONFIDENCE_THRESHOLD} = {confidence_passed}")
    
    all_passed = ai_passed and ta_passed and quality_passed and confidence_passed
    print(f"\n   Tüm filtreler geçildi mi: {all_passed}")
    
    if all_passed:
        print(f"   ✅ Manuel filtrelemeden geçti!")
    else:
        print(f"   ❌ Manuel filtrelemeden geçemedi!")
    
    return all_passed

if __name__ == "__main__":
    test_signal_filtering() 