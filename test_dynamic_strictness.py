#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dinamik Sıkılık Sistemi Testi
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from modules.dynamic_strictness import dynamic_strictness
import numpy as np
from datetime import datetime

def test_dynamic_strictness():
    """Dinamik sıkılık sistemini test et"""
    print("=" * 60)
    print("🤖 DİNAMİK SIKILIK SİSTEMİ TESTİ")
    print("=" * 60)
    
    # Test 1: Başlangıç durumu
    print("\n📊 TEST 1: Başlangıç Durumu")
    print("-" * 40)
    initial_status = dynamic_strictness.get_current_status()
    print(f"Mevcut sıkılık: {initial_status['current_strictness']:.2f}")
    print(f"Sıkılık seviyesi: {initial_status['strictness_level']}")
    print(f"Öneri: {initial_status['recommendation']}")
    
    # Test 2: Güçlü yükseliş senaryosu
    print("\n📈 TEST 2: Güçlü Yükseliş Senaryosu")
    print("-" * 40)
    bullish_market = {
        'price_data': [100, 102, 105, 108, 112, 115, 118, 122, 125, 128, 130],
        'technical_indicators': {
            'rsi': 75,  # Aşırı alım
            'macd': 0.8  # Güçlü yükseliş
        },
        'volume_data': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200, 3500],
        'sentiment_data': {
            'overall_sentiment': 0.8  # Çok pozitif
        },
        'ai_predictions': {
            'confidence': 0.9  # Yüksek güven
        }
    }
    
    bullish_result = dynamic_strictness.update_strictness(bullish_market)
    print(f"Yeni sıkılık: {bullish_result['current_strictness']:.2f}")
    print(f"Sebep: {bullish_result['strictness_level']}")
    print(f"Öneri: {bullish_result['recommendation']}")
    
    # Test 3: Düşüş senaryosu
    print("\n📉 TEST 3: Düşüş Senaryosu")
    print("-" * 40)
    bearish_market = {
        'price_data': [100, 98, 95, 92, 89, 86, 83, 80, 77, 74, 70],
        'technical_indicators': {
            'rsi': 25,  # Aşırı satım
            'macd': -0.6  # Güçlü düşüş
        },
        'volume_data': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        'sentiment_data': {
            'overall_sentiment': 0.2  # Çok negatif
        },
        'ai_predictions': {
            'confidence': 0.3  # Düşük güven
        }
    }
    
    bearish_result = dynamic_strictness.update_strictness(bearish_market)
    print(f"Yeni sıkılık: {bearish_result['current_strictness']:.2f}")
    print(f"Sebep: {bearish_result['strictness_level']}")
    print(f"Öneri: {bearish_result['recommendation']}")
    
    # Test 4: Sideways senaryosu
    print("\n🔄 TEST 4: Sideways Senaryosu")
    print("-" * 40)
    sideways_market = {
        'price_data': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 100],
        'technical_indicators': {
            'rsi': 50,  # Nötr
            'macd': 0.1  # Zayıf sinyal
        },
        'volume_data': [1000, 1050, 1100, 1050, 1000, 1050, 1100, 1050, 1000, 1050, 1100],
        'sentiment_data': {
            'overall_sentiment': 0.5  # Nötr
        },
        'ai_predictions': {
            'confidence': 0.6  # Orta güven
        }
    }
    
    sideways_result = dynamic_strictness.update_strictness(sideways_market)
    print(f"Yeni sıkılık: {sideways_result['current_strictness']:.2f}")
    print(f"Sebep: {sideways_result['strictness_level']}")
    print(f"Öneri: {sideways_result['recommendation']}")
    
    # Test 5: Geçmiş kontrolü
    print("\n📈 TEST 5: Geçmiş Kontrolü")
    print("-" * 40)
    history = dynamic_strictness.strictness_history
    print(f"Toplam geçmiş kayıt: {len(history)}")
    
    if history:
        print("\nSon 5 değişiklik:")
        for i, record in enumerate(history[-5:], 1):
            change = record['new_strictness'] - record['old_strictness']
            change_text = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
            print(f"{i}. {record['timestamp'][:19]} | {record['old_strictness']:.2f} → {record['new_strictness']:.2f} ({change_text})")
            print(f"   Sebep: {record['reason']}")
    
    # Test 6: Durum kaydetme/yükleme
    print("\n💾 TEST 6: Durum Kaydetme/Yükleme")
    print("-" * 40)
    dynamic_strictness.save_status("test_strictness.json")
    print("✅ Durum kaydedildi")
    
    # Yeni instance oluştur ve yükle
    from modules.dynamic_strictness import DynamicStrictness
    new_instance = DynamicStrictness()
    new_instance.load_status("test_strictness.json")
    print(f"✅ Durum yüklendi: {new_instance.current_strictness:.2f}")
    
    # Test dosyasını temizle
    if os.path.exists("test_strictness.json"):
        os.remove("test_strictness.json")
        print("✅ Test dosyası temizlendi")
    
    print("\n" + "=" * 60)
    print("✅ DİNAMİK SIKILIK SİSTEMİ TESTİ TAMAMLANDI")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_dynamic_strictness() 