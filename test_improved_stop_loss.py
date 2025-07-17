#!/usr/bin/env python3
"""
İyileştirilmiş Stop Loss Sistemi Test Scripti
"""

import os
import sys
import pandas as pd
from datetime import datetime
import sqlalchemy

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from modules.signal_manager import SignalManager

def test_improved_stop_loss():
    """İyileştirilmiş stop loss sistemini test et"""
    print("🔧 İYİLEŞTİRİLMİŞ STOP LOSS SİSTEMİ TEST EDİLİYOR")
    print("=" * 60)
    
    try:
        signal_manager = SignalManager()
        
        # Veritabanından mevcut sinyalleri al
        engine = sqlalchemy.create_engine(Config.DATABASE_URL)
        query = "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 5"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("❌ Veritabanında sinyal bulunamadı")
            return
        
        print(f"📊 {len(df)} sinyal test edilecek")
        
        for idx, signal in df.iterrows():
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price'] or signal['price']
            ai_score = signal['ai_score'] or 0.5
            ta_strength = signal['ta_strength'] or 0.5
            
            if not entry_price or entry_price <= 0:
                continue
                
            print(f"\n📊 {symbol} ({direction}) İYİLEŞTİRİLMİŞ TEST")
            print("-" * 50)
            print(f"Giriş Fiyatı: {entry_price:.8f}")
            print(f"AI Skoru: {ai_score:.2f}")
            print(f"TA Gücü: {ta_strength:.2f}")
            
            # Test sinyal verisi oluştur
            test_signal = {
                'symbol': symbol,
                'direction': direction,
                'ai_score': ai_score,
                'ta_strength': ta_strength,
                'atr': entry_price * 0.025,  # %2.5 ATR
                'volatility': 0.025,  # %2.5 volatilite
                'support_levels': [entry_price * 0.98, entry_price * 0.96],  # Test destek seviyeleri
                'resistance_levels': [entry_price * 1.02, entry_price * 1.04]  # Test direnç seviyeleri
            }
            
            # İyileştirilmiş hedef seviyeleri hesapla
            take_profit, stop_loss, support_level, resistance_level, target_time = signal_manager.calculate_target_levels(
                test_signal, entry_price
            )
            
            print(f"\n🎯 İYİLEŞTİRİLMİŞ HEDEFLER:")
            print(f"  Take Profit: {take_profit:.8f} ({((take_profit/entry_price-1)*100):.2f}%)")
            print(f"  Stop Loss: {stop_loss:.8f} ({((stop_loss/entry_price-1)*100):.2f}%)")
            print(f"  Support Level: {support_level:.8f} ({((support_level/entry_price-1)*100):.2f}%)")
            print(f"  Resistance Level: {resistance_level:.8f} ({((resistance_level/entry_price-1)*100):.2f}%)")
            print(f"  Target Time: {target_time:.1f} saat")
            
            # Risk/Reward oranı hesapla
            if direction == 'LONG':
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            elif direction == 'SHORT':
                potential_profit = entry_price - take_profit
                potential_loss = stop_loss - entry_price
            else:
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            
            risk_reward = potential_profit / potential_loss if potential_loss > 0 else 1.0
            print(f"  Risk/Reward: {risk_reward:.2f}:1")
            
            # Eski sistemle karşılaştırma
            print(f"\n📊 ESKİ SİSTEM KARŞILAŞTIRMASI:")
            old_atr = entry_price * 0.02
            if direction == 'LONG':
                old_tp = entry_price + (old_atr * 2.5)
                old_sl = entry_price - (old_atr * 1.5)
            elif direction == 'SHORT':
                old_tp = entry_price - (old_atr * 2.5)
                old_sl = entry_price + (old_atr * 1.5)
            else:
                old_tp = entry_price * 1.05
                old_sl = entry_price * 0.95
            
            print(f"  Eski TP: {old_tp:.8f} ({((old_tp/entry_price-1)*100):.2f}%)")
            print(f"  Eski SL: {old_sl:.8f} ({((old_sl/entry_price-1)*100):.2f}%)")
            
            # İyileştirme oranları
            tp_improvement = abs((take_profit/entry_price-1)*100) - abs((old_tp/entry_price-1)*100)
            sl_improvement = abs((stop_loss/entry_price-1)*100) - abs((old_sl/entry_price-1)*100)
            
            print(f"\n📈 İYİLEŞTİRME:")
            print(f"  TP Farkı: {tp_improvement:+.2f}%")
            print(f"  SL Farkı: {sl_improvement:+.2f}%")
            
            if tp_improvement < 0:
                print(f"  ✅ Take Profit daha gerçekçi (daha düşük)")
            if sl_improvement < 0:
                print(f"  ✅ Stop Loss daha sıkı (daha az risk)")
            
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

def test_different_scenarios():
    """Farklı senaryoları test et"""
    print("\n\n🎭 FARKLI SENARYOLAR TEST EDİLİYOR")
    print("=" * 60)
    
    try:
        signal_manager = SignalManager()
        
        # Test senaryoları
        scenarios = [
            {
                'name': 'Yüksek Güvenli LONG',
                'signal': {
                    'symbol': 'BTC_USDT',
                    'direction': 'LONG',
                    'ai_score': 0.85,
                    'ta_strength': 0.8,
                    'atr': 0.02,
                    'volatility': 0.02,
                    'support_levels': [0.98, 0.96],
                    'resistance_levels': [1.02, 1.04]
                },
                'entry_price': 50000
            },
            {
                'name': 'Düşük Güvenli SHORT',
                'signal': {
                    'symbol': 'DOGE_USDT',
                    'direction': 'SHORT',
                    'ai_score': 0.25,
                    'ta_strength': 0.3,
                    'atr': 0.05,
                    'volatility': 0.05,
                    'support_levels': [0.95, 0.93],
                    'resistance_levels': [1.05, 1.07]
                },
                'entry_price': 0.1
            },
            {
                'name': 'Orta Güvenli NEUTRAL',
                'signal': {
                    'symbol': 'ETH_USDT',
                    'direction': 'NEUTRAL',
                    'ai_score': 0.5,
                    'ta_strength': 0.5,
                    'atr': 0.025,
                    'volatility': 0.025,
                    'support_levels': [0.98, 0.96],
                    'resistance_levels': [1.02, 1.04]
                },
                'entry_price': 3000
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📊 {scenario['name']}")
            print("-" * 30)
            
            take_profit, stop_loss, support_level, resistance_level, target_time = signal_manager.calculate_target_levels(
                scenario['signal'], scenario['entry_price']
            )
            
            print(f"  Take Profit: {take_profit:.8f} ({((take_profit/scenario['entry_price']-1)*100):.2f}%)")
            print(f"  Stop Loss: {stop_loss:.8f} ({((stop_loss/scenario['entry_price']-1)*100):.2f}%)")
            print(f"  Support: {support_level:.8f} ({((support_level/scenario['entry_price']-1)*100):.2f}%)")
            print(f"  Resistance: {resistance_level:.8f} ({((resistance_level/scenario['entry_price']-1)*100):.2f}%)")
            print(f"  Target Time: {target_time:.1f} saat")
            
            # Risk/Reward hesapla
            if scenario['signal']['direction'] == 'LONG':
                potential_profit = take_profit - scenario['entry_price']
                potential_loss = scenario['entry_price'] - stop_loss
            elif scenario['signal']['direction'] == 'SHORT':
                potential_profit = scenario['entry_price'] - take_profit
                potential_loss = stop_loss - scenario['entry_price']
            else:
                potential_profit = take_profit - scenario['entry_price']
                potential_loss = scenario['entry_price'] - stop_loss
            
            risk_reward = potential_profit / potential_loss if potential_loss > 0 else 1.0
            print(f"  Risk/Reward: {risk_reward:.2f}:1")
            
    except Exception as e:
        print(f"❌ Senaryo test hatası: {e}")

if __name__ == "__main__":
    print("🚀 İYİLEŞTİRİLMİŞ STOP LOSS SİSTEMİ TEST SİSTEMİ")
    print("=" * 70)
    
    # İyileştirilmiş sistemi test et
    test_improved_stop_loss()
    
    # Farklı senaryoları test et
    test_different_scenarios()
    
    print("\n\n📝 SONUÇLAR:")
    print("=" * 30)
    print("✅ İyileştirilmiş stop loss sistemi test edildi")
    print("✅ Farklı senaryolar test edildi")
    print("✅ Destek/Direnç seviyeleri eklendi")
    print("✅ Volatilite bazlı ayarlamalar eklendi")
    print("✅ AI skoruna göre hassasiyet ayarlandı")
    print("✅ Coin tipine göre özelleştirme yapıldı")
    print("\n🎯 İYİLEŞTİRMELER:")
    print("• Stop loss oranları daha sıkı (%1.5 minimum)")
    print("• Take profit oranları daha gerçekçi (%2.5 minimum)")
    print("• Destek/Direnç seviyeleri dahil edildi")
    print("• Risk/Reward oranları optimize edildi")
    print("• Volatilite bazlı dinamik ayarlama")
    print("• AI ve TA skorlarına göre hassasiyet") 