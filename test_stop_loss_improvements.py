#!/usr/bin/env python3
"""
Stop Loss ve Destek/Direnç Seviyeleri Test Scripti
"""

import os
import sys
import pandas as pd
from datetime import datetime
import json

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from modules.signal_manager import SignalManager
from modules.technical_analysis import TechnicalAnalysis
from modules.data_collector import DataCollector

def test_current_stop_loss_calculation():
    """Mevcut stop loss hesaplamalarını test et"""
    print("🔍 MEVCUT STOP LOSS HESAPLAMALARI TEST EDİLİYOR")
    print("=" * 60)
    
    try:
        signal_manager = SignalManager()
        data_collector = DataCollector()
        ta = TechnicalAnalysis()
        
        # Test coinleri
        test_symbols = ['BTC_USDT', 'ETH_USDT', 'ADA_USDT', 'SOL_USDT']
        
        for symbol in test_symbols:
            print(f"\n📊 {symbol} TEST EDİLİYOR")
            print("-" * 40)
            
            # Veri al
            df = data_collector.get_historical_data(symbol, '1h', 100)
            if df is None or df.empty:
                print(f"❌ {symbol} için veri alınamadı")
                continue
            
            # Teknik analiz ekle
            df = ta.add_all_indicators(df)
            
            current_price = df['close'].iloc[-1]
            print(f"Güncel Fiyat: {current_price:.8f}")
            
            # ATR değeri
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
            print(f"ATR: {atr:.8f} ({atr/current_price*100:.2f}%)")
            
            # Destek/Direnç seviyeleri
            support_levels = ta.find_dynamic_support(df)
            resistance_levels = ta.find_dynamic_resistance(df)
            
            print(f"Destek Seviyeleri: {[f'{s:.8f}' for s in support_levels[:3]]}")
            print(f"Direnç Seviyeleri: {[f'{r:.8f}' for r in resistance_levels[:3]]}")
            
            # LONG pozisyon testi
            print(f"\n🟢 LONG Pozisyon Testi:")
            long_tp = current_price + (atr * 2.5)
            long_sl = current_price - (atr * 1.5)
            long_support = current_price - (atr * 1.0)
            long_resistance = long_tp + (atr * 0.5)
            
            print(f"  Take Profit: {long_tp:.8f} (+{((long_tp/current_price-1)*100):.2f}%)")
            print(f"  Stop Loss: {long_sl:.8f} ({((long_sl/current_price-1)*100):.2f}%)")
            print(f"  Support: {long_support:.8f} ({((long_support/current_price-1)*100):.2f}%)")
            print(f"  Resistance: {long_resistance:.8f} (+{((long_resistance/current_price-1)*100):.2f}%)")
            
            # SHORT pozisyon testi
            print(f"\n🔴 SHORT Pozisyon Testi:")
            short_tp = current_price - (atr * 2.5)
            short_sl = current_price + (atr * 1.5)
            short_support = short_tp - (atr * 0.5)
            short_resistance = current_price + (atr * 1.0)
            
            print(f"  Take Profit: {short_tp:.8f} ({((short_tp/current_price-1)*100):.2f}%)")
            print(f"  Stop Loss: {short_sl:.8f} (+{((short_sl/current_price-1)*100):.2f}%)")
            print(f"  Support: {short_support:.8f} ({((short_support/current_price-1)*100):.2f}%)")
            print(f"  Resistance: {short_resistance:.8f} (+{((short_resistance/current_price-1)*100):.2f}%)")
            
            # Risk/Reward oranları
            long_rr = (long_tp - current_price) / (current_price - long_sl)
            short_rr = (current_price - short_tp) / (short_sl - current_price)
            
            print(f"\n📈 Risk/Reward Oranları:")
            print(f"  LONG: {long_rr:.2f}:1")
            print(f"  SHORT: {short_rr:.2f}:1")
            
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

def test_improved_stop_loss_calculation():
    """İyileştirilmiş stop loss hesaplamalarını test et"""
    print("\n\n🔧 İYİLEŞTİRİLMİŞ STOP LOSS HESAPLAMALARI")
    print("=" * 60)
    
    try:
        signal_manager = SignalManager()
        data_collector = DataCollector()
        ta = TechnicalAnalysis()
        
        # Test coinleri
        test_symbols = ['BTC_USDT', 'ETH_USDT', 'ADA_USDT', 'SOL_USDT']
        
        for symbol in test_symbols:
            print(f"\n📊 {symbol} İYİLEŞTİRİLMİŞ TEST")
            print("-" * 40)
            
            # Veri al
            df = data_collector.get_historical_data(symbol, '1h', 100)
            if df is None or df.empty:
                print(f"❌ {symbol} için veri alınamadı")
                continue
            
            # Teknik analiz ekle
            df = ta.add_all_indicators(df)
            
            current_price = df['close'].iloc[-1]
            print(f"Güncel Fiyat: {current_price:.8f}")
            
            # Gelişmiş destek/direnç analizi
            support_levels = ta.find_dynamic_support(df)
            resistance_levels = ta.find_dynamic_resistance(df)
            
            # ATR hesaplama
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
            
            # Volatilite analizi
            volatility = df['close'].pct_change().std() * 100
            print(f"Volatilite: {volatility:.2f}%")
            
            # Trend gücü
            ema_20 = df['ema_20'].iloc[-1] if 'ema_20' in df.columns else current_price
            ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else current_price
            trend_strength = abs(ema_20 - ema_50) / current_price
            
            # İyileştirilmiş hesaplamalar
            print(f"\n🟢 İYİLEŞTİRİLMİŞ LONG Pozisyon:")
            
            # Destek bazlı stop loss
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=0)
                if nearest_support > 0:
                    long_sl_support = nearest_support * 0.995  # %0.5 altında
                else:
                    long_sl_support = current_price * (1 - 0.03)  # %3 altında
            else:
                long_sl_support = current_price * (1 - 0.03)
            
            # ATR bazlı stop loss
            long_sl_atr = current_price - (atr * 1.5)
            
            # Volatilite bazlı stop loss
            volatility_stop = current_price * (1 - volatility/100 * 0.5)
            
            # En iyi stop loss seçimi
            long_sl = max(long_sl_support, long_sl_atr, volatility_stop)
            
            # Take profit hesaplama
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
                if nearest_resistance != float('inf'):
                    long_tp_resistance = nearest_resistance * 1.005  # %0.5 üstünde
                else:
                    long_tp_resistance = current_price * (1 + 0.06)  # %6 üstünde
            else:
                long_tp_resistance = current_price * (1 + 0.06)
            
            # ATR bazlı take profit
            long_tp_atr = current_price + (atr * 2.5)
            
            # En iyi take profit seçimi
            long_tp = min(long_tp_resistance, long_tp_atr)
            
            print(f"  Take Profit: {long_tp:.8f} (+{((long_tp/current_price-1)*100):.2f}%)")
            print(f"  Stop Loss: {long_sl:.8f} ({((long_sl/current_price-1)*100):.2f}%)")
            print(f"  Risk/Reward: {((long_tp-current_price)/(current_price-long_sl)):.2f}:1")
            
            # SHORT pozisyon
            print(f"\n🔴 İYİLEŞTİRİLMİŞ SHORT Pozisyon:")
            
            # Direnç bazlı stop loss
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
                if nearest_resistance != float('inf'):
                    short_sl_resistance = nearest_resistance * 1.005  # %0.5 üstünde
                else:
                    short_sl_resistance = current_price * (1 + 0.03)  # %3 üstünde
            else:
                short_sl_resistance = current_price * (1 + 0.03)
            
            # ATR bazlı stop loss
            short_sl_atr = current_price + (atr * 1.5)
            
            # Volatilite bazlı stop loss
            short_volatility_stop = current_price * (1 + volatility/100 * 0.5)
            
            # En iyi stop loss seçimi
            short_sl = min(short_sl_resistance, short_sl_atr, short_volatility_stop)
            
            # Take profit hesaplama
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=0)
                if nearest_support > 0:
                    short_tp_support = nearest_support * 0.995  # %0.5 altında
                else:
                    short_tp_support = current_price * (1 - 0.06)  # %6 altında
            else:
                short_tp_support = current_price * (1 - 0.06)
            
            # ATR bazlı take profit
            short_tp_atr = current_price - (atr * 2.5)
            
            # En iyi take profit seçimi
            short_tp = max(short_tp_support, short_tp_atr)
            
            print(f"  Take Profit: {short_tp:.8f} ({((short_tp/current_price-1)*100):.2f}%)")
            print(f"  Stop Loss: {short_sl:.8f} (+{((short_sl/current_price-1)*100):.2f}%)")
            print(f"  Risk/Reward: {((current_price-short_tp)/(short_sl-current_price)):.2f}:1")
            
    except Exception as e:
        print(f"❌ İyileştirilmiş test hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

def create_improved_stop_loss_function():
    """İyileştirilmiş stop loss hesaplama fonksiyonu oluştur"""
    print("\n\n🔧 İYİLEŞTİRİLMİŞ STOP LOSS FONKSİYONU")
    print("=" * 60)
    
    improved_function = '''
def calculate_improved_target_levels(self, signal, entry_price):
    """İyileştirilmiş hedef seviyeler hesaplama"""
    try:
        direction = signal.get('direction', 'LONG')
        ai_score = signal.get('ai_score', 0.5)
        ta_strength = signal.get('ta_strength', 0.5)
        symbol = signal.get('symbol', '')
        
        # ATR hesaplama
        atr = signal.get('atr', entry_price * 0.02)
        
        # Volatilite analizi
        volatility = signal.get('volatility', 0.02)
        
        # Destek/Direnç seviyeleri
        support_levels = signal.get('support_levels', [])
        resistance_levels = signal.get('resistance_levels', [])
        
        # AI skoruna göre ATR çarpanını ayarla
        if ai_score > 0.8:
            atr_multiplier = 1.2  # Yüksek güven - daha sıkı
        elif ai_score < 0.3:
            atr_multiplier = 2.0  # Düşük güven - daha geniş
        else:
            atr_multiplier = 1.5  # Orta güven
        
        # TA gücüne göre ayarlama
        if ta_strength > 0.7:
            atr_multiplier *= 0.9  # Güçlü TA - daha sıkı
        elif ta_strength < 0.3:
            atr_multiplier *= 1.2  # Zayıf TA - daha geniş
        
        # Coin volatilitesine göre ayarlama
        volatile_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF']
        if any(coin in symbol.upper() for coin in volatile_coins):
            atr_multiplier *= 1.3  # Volatil coinler - daha geniş
        
        # Volatilite bazlı ayarlama
        if volatility > 0.05:  # %5'ten yüksek volatilite
            atr_multiplier *= 1.2
        elif volatility < 0.01:  # %1'den düşük volatilite
            atr_multiplier *= 0.8
        
        if direction == 'LONG':
            # LONG pozisyon için iyileştirilmiş hesaplama
            
            # Stop Loss - En iyi seçenek
            sl_candidates = []
            
            # 1. Destek bazlı stop loss
            if support_levels:
                nearest_support = max([s for s in support_levels if s < entry_price], default=0)
                if nearest_support > 0:
                    sl_candidates.append(nearest_support * 0.995)  # %0.5 altında
            
            # 2. ATR bazlı stop loss
            sl_candidates.append(entry_price - (atr * atr_multiplier))
            
            # 3. Volatilite bazlı stop loss
            sl_candidates.append(entry_price * (1 - volatility * 0.5))
            
            # 4. Minimum stop loss (%2)
            sl_candidates.append(entry_price * 0.98)
            
            # En iyi stop loss seçimi (en yüksek)
            stop_loss = max(sl_candidates)
            
            # Take Profit - En iyi seçenek
            tp_candidates = []
            
            # 1. Direnç bazlı take profit
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > entry_price], default=float('inf'))
                if nearest_resistance != float('inf'):
                    tp_candidates.append(nearest_resistance * 1.005)  # %0.5 üstünde
            
            # 2. ATR bazlı take profit (2:1 risk/reward)
            tp_candidates.append(entry_price + (atr * atr_multiplier * 2))
            
            # 3. Volatilite bazlı take profit
            tp_candidates.append(entry_price * (1 + volatility * 1.5))
            
            # 4. Minimum take profit (%3)
            tp_candidates.append(entry_price * 1.03)
            
            # En iyi take profit seçimi (en düşük)
            take_profit = min(tp_candidates)
            
            # Destek/Direnç seviyeleri
            support_level = stop_loss * 0.995  # Stop loss'un altında
            resistance_level = take_profit * 1.005  # Take profit'in üstünde
            
        elif direction == 'SHORT':
            # SHORT pozisyon için iyileştirilmiş hesaplama
            
            # Stop Loss - En iyi seçenek
            sl_candidates = []
            
            # 1. Direnç bazlı stop loss
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > entry_price], default=float('inf'))
                if nearest_resistance != float('inf'):
                    sl_candidates.append(nearest_resistance * 1.005)  # %0.5 üstünde
            
            # 2. ATR bazlı stop loss
            sl_candidates.append(entry_price + (atr * atr_multiplier))
            
            # 3. Volatilite bazlı stop loss
            sl_candidates.append(entry_price * (1 + volatility * 0.5))
            
            # 4. Minimum stop loss (%2)
            sl_candidates.append(entry_price * 1.02)
            
            # En iyi stop loss seçimi (en düşük)
            stop_loss = min(sl_candidates)
            
            # Take Profit - En iyi seçenek
            tp_candidates = []
            
            # 1. Destek bazlı take profit
            if support_levels:
                nearest_support = max([s for s in support_levels if s < entry_price], default=0)
                if nearest_support > 0:
                    tp_candidates.append(nearest_support * 0.995)  # %0.5 altında
            
            # 2. ATR bazlı take profit (2:1 risk/reward)
            tp_candidates.append(entry_price - (atr * atr_multiplier * 2))
            
            # 3. Volatilite bazlı take profit
            tp_candidates.append(entry_price * (1 - volatility * 1.5))
            
            # 4. Minimum take profit (%3)
            tp_candidates.append(entry_price * 0.97)
            
            # En iyi take profit seçimi (en yüksek)
            take_profit = max(tp_candidates)
            
            # Destek/Direnç seviyeleri
            support_level = take_profit * 0.995  # Take profit'in altında
            resistance_level = stop_loss * 1.005  # Stop loss'un üstünde
            
        else:  # NEUTRAL
            take_profit = entry_price * 1.02
            stop_loss = entry_price * 0.98
            support_level = entry_price * 0.99
            resistance_level = entry_price * 1.01
        
        # Risk/Ödül oranı hesapla
        if direction == 'LONG':
            potential_profit = take_profit - entry_price
            potential_loss = entry_price - stop_loss
        elif direction == 'SHORT':
            potential_profit = entry_price - take_profit
            potential_loss = stop_loss - entry_price
        else:
            potential_profit = take_profit - entry_price
            potential_loss = entry_price - stop_loss
        
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 1.0
        
        # Hedef süreyi hesapla
        target_time = 24.0  # Varsayılan 24 saat
        
        # AI skoruna göre süre ayarlaması
        if ai_score > 0.8:
            target_time *= 0.8  # Daha hızlı
        elif ai_score < 0.3:
            target_time *= 1.2  # Daha yavaş
        
        # Coin tipine göre süre ayarlaması
        volatile_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF']
        if any(coin in symbol.upper() for coin in volatile_coins):
            target_time *= 0.7  # Volatil coinler daha hızlı
        
        return take_profit, stop_loss, support_level, resistance_level, target_time
        
    except Exception as e:
        self.logger.error(f"İyileştirilmiş hedef seviyeler hesaplama hatası: {e}")
        # Varsayılan değerler
        if direction == 'LONG':
            return entry_price * 1.05, entry_price * 0.95, entry_price * 0.95, entry_price * 1.05, 24.0
        elif direction == 'SHORT':
            return entry_price * 0.95, entry_price * 1.05, entry_price * 0.95, entry_price * 1.05, 24.0
        else:
            return entry_price * 1.02, entry_price * 0.98, entry_price * 0.99, entry_price * 1.01, 24.0
'''
    
    print("✅ İyileştirilmiş stop loss fonksiyonu hazırlandı")
    print("Bu fonksiyon şu özellikleri içerir:")
    print("• Destek/Direnç seviyelerine dayalı stop loss")
    print("• ATR bazlı dinamik hesaplama")
    print("• Volatilite bazlı ayarlama")
    print("• AI skoruna göre hassasiyet")
    print("• Coin tipine göre özelleştirme")
    print("• Minimum/Maksimum sınırlar")
    print("• Risk/Reward optimizasyonu")
    
    return improved_function

if __name__ == "__main__":
    print("🚀 STOP LOSS VE DESTEK/DİRENÇ SEVİYELERİ TEST SİSTEMİ")
    print("=" * 70)
    
    # Mevcut durumu test et
    test_current_stop_loss_calculation()
    
    # İyileştirilmiş durumu test et
    test_improved_stop_loss_calculation()
    
    # İyileştirilmiş fonksiyon oluştur
    improved_function = create_improved_stop_loss_function()
    
    print("\n\n📝 SONUÇLAR:")
    print("=" * 30)
    print("✅ Mevcut sistem test edildi")
    print("✅ İyileştirilmiş sistem test edildi")
    print("✅ Yeni fonksiyon hazırlandı")
    print("\n🔧 ÖNERİLER:")
    print("1. SignalManager'a yeni calculate_improved_target_levels fonksiyonu ekle")
    print("2. Destek/Direnç seviyelerini daha iyi hesapla")
    print("3. Volatilite bazlı ayarlamalar ekle")
    print("4. AI skoruna göre hassasiyet ayarla")
    print("5. Coin tipine göre özelleştirme yap") 