#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performans Takibi Testi
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from modules.dynamic_strictness import dynamic_strictness
from datetime import datetime, timedelta

def test_performance_tracking():
    """Performans takibi sistemini test et"""
    print("=" * 60)
    print("📊 PERFORMANS TAKİBİ SİSTEMİ TESTİ")
    print("=" * 60)
    
    # Test 1: Başlangıç durumu
    print("\n📊 TEST 1: Başlangıç Durumu")
    print("-" * 40)
    initial_performance = dynamic_strictness.get_performance_summary()
    print(f"Başarı oranı: {initial_performance['success_rate']:.1%}")
    print(f"Toplam sinyal: {initial_performance['total_signals']}")
    print(f"Otomatik ayarlama: {initial_performance['auto_adjustment_status']}")
    
    # Test 2: Düşük performans senaryosu
    print("\n📉 TEST 2: Düşük Performans Senaryosu")
    print("-" * 40)
    low_performance_results = {
        'total_signals': 10,
        'successful_signals': 3,  # %30 başarı
        'failed_signals': 7,
        'timestamp': datetime.now().isoformat()
    }
    
    dynamic_strictness.update_performance_metrics(low_performance_results)
    performance_after_low = dynamic_strictness.get_performance_summary()
    print(f"Başarı oranı: {performance_after_low['success_rate']:.1%}")
    print(f"Otomatik ayarlama: {performance_after_low['auto_adjustment_status']}")
    
    # Test 3: Yüksek performans senaryosu
    print("\n📈 TEST 3: Yüksek Performans Senaryosu")
    print("-" * 40)
    high_performance_results = {
        'total_signals': 15,
        'successful_signals': 13,  # %87 başarı
        'failed_signals': 2,
        'timestamp': datetime.now().isoformat()
    }
    
    dynamic_strictness.update_performance_metrics(high_performance_results)
    performance_after_high = dynamic_strictness.get_performance_summary()
    print(f"Başarı oranı: {performance_after_high['success_rate']:.1%}")
    print(f"Otomatik ayarlama: {performance_after_high['auto_adjustment_status']}")
    
    # Test 4: Optimal performans senaryosu
    print("\n⚖️ TEST 4: Optimal Performans Senaryosu")
    print("-" * 40)
    optimal_performance_results = {
        'total_signals': 20,
        'successful_signals': 14,  # %70 başarı
        'failed_signals': 6,
        'timestamp': datetime.now().isoformat()
    }
    
    dynamic_strictness.update_performance_metrics(optimal_performance_results)
    performance_after_optimal = dynamic_strictness.get_performance_summary()
    print(f"Başarı oranı: {performance_after_optimal['success_rate']:.1%}")
    print(f"Otomatik ayarlama: {performance_after_optimal['auto_adjustment_status']}")
    
    # Test 5: Geçmiş kontrolü
    print("\n📈 TEST 5: Performans Geçmişi")
    print("-" * 40)
    history = dynamic_strictness.strictness_history
    print(f"Toplam geçmiş kayıt: {len(history)}")
    
    if history:
        print("\nSon 5 performans ayarlaması:")
        for i, record in enumerate(history[-5:], 1):
            timestamp = record.get('timestamp', 'Unknown')
            old_strictness = record.get('old_strictness', 0)
            new_strictness = record.get('new_strictness', 0)
            reason = record.get('reason', 'Unknown')
            adjustment_type = record.get('adjustment_type', 'manual')
            
            print(f"{i}. {timestamp} | {old_strictness:.3f} → {new_strictness:.3f} ({adjustment_type})")
            print(f"   Sebep: {reason}")
    
    # Test 6: Otomatik ayarlama durumu
    print("\n🤖 TEST 6: Otomatik Ayar Durumu")
    print("-" * 40)
    current_status = dynamic_strictness.get_current_status()
    auto_adjustment = current_status.get('auto_adjustment', {})
    
    print(f"Otomatik ayarlama aktif: {auto_adjustment.get('enabled', False)}")
    print(f"Mevcut başarı oranı: {auto_adjustment.get('success_rate', 0):.1%}")
    print(f"Hedef başarı oranı: {auto_adjustment.get('min_target', 0):.1%} - {auto_adjustment.get('max_target', 0):.1%}")
    
    print("\n" + "=" * 60)
    print("✅ PERFORMANS TAKİBİ SİSTEMİ TESTİ TAMAMLANDI")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_performance_tracking() 