#!/usr/bin/env python3
"""
KAHİN ULTIMA - Teknik Analiz Test Scripti
Teknik analiz modülünün tüm fonksiyonlarını test eder
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.technical_analysis import TechnicalAnalysis
from modules.data_collector import DataCollector

def test_technical_analysis():
    """Teknik analiz modülünü test et"""
    print("🔧 KAHİN ULTIMA - TEKNİK ANALİZ TESTİ BAŞLATIYOR")
    print("=" * 60)
    
    try:
        # 1. Modülleri başlat
        print("1. Modüller başlatılıyor...")
        ta = TechnicalAnalysis()
        collector = DataCollector()
        print("✓ Modüller başlatıldı")
        
        # 2. Test verisi al
        print("2. Test verisi alınıyor...")
        symbol = 'BTC/USDT'
        df = collector.get_historical_data(symbol, '1h', 500)
        
        if df is None or df.empty:
            print("❌ Test verisi alınamadı")
            return False
            
        print(f"✓ {len(df)} satır veri alındı")
        print(f"✓ Veri aralığı: {df.index[0]} - {df.index[-1]}")
        
        # 3. Temel teknik analiz
        print("3. Temel teknik analiz yapılıyor...")
        df_with_indicators = ta.calculate_all_indicators(df)
        
        if df_with_indicators is None or df_with_indicators.empty:
            print("❌ Teknik analiz başarısız")
            return False
            
        print(f"✓ Teknik analiz tamamlandı")
        print(f"✓ Toplam kolon sayısı: {df_with_indicators.shape[1]}")
        
        # 4. Sinyal üretimi
        print("4. Sinyal üretimi test ediliyor...")
        signals = ta.generate_signals(df_with_indicators)
        
        if signals:
            print("✓ Sinyal üretimi başarılı")
            for signal_name, signal_data in signals.items():
                if isinstance(signal_data, pd.Series):
                    signal_count = signal_data.sum()
                    print(f"  - {signal_name}: {signal_count} sinyal")
        else:
            print("⚠ Sinyal üretimi başarısız")
        
        # 5. Sinyal gücü hesaplama
        print("5. Sinyal gücü hesaplanıyor...")
        signal_strength = ta.calculate_signal_strength(signals)
        print(f"✓ Sinyal gücü: {signal_strength:.3f}")
        
        # 6. Trend analizi
        print("6. Trend analizi yapılıyor...")
        trend_direction = ta.get_trend_direction(df_with_indicators)
        trend_strength = ta.calculate_trend_strength(df_with_indicators)
        print(f"✓ Trend yönü: {trend_direction}")
        print(f"✓ Trend gücü: {trend_strength:.3f}")
        
        # 7. Destek/Direnç seviyeleri
        print("7. Destek/Direnç seviyeleri hesaplanıyor...")
        support_resistance = ta.get_support_resistance(df_with_indicators)
        print(f"✓ Destek: {support_resistance['support']:.2f}")
        print(f"✓ Direnç: {support_resistance['resistance']:.2f}")
        
        # 8. Momentum analizi
        print("8. Momentum analizi yapılıyor...")
        momentum_score = ta.calculate_momentum_score(df_with_indicators)
        print(f"✓ Momentum skoru: {momentum_score:.3f}")
        
        # 9. Market rejimi analizi
        print("9. Market rejimi analizi yapılıyor...")
        market_regime = ta.get_market_regime(df_with_indicators)
        print(f"✓ Market rejimi: {market_regime}")
        
        # 10. Gelişmiş teknik analiz
        print("10. Gelişmiş teknik analiz yapılıyor...")
        advanced_analysis = ta.analyze_technical_signals(df_with_indicators)
        
        if advanced_analysis and 'error' not in advanced_analysis:
            print("✓ Gelişmiş analiz başarılı")
            
            # RSI değerleri
            if 'rsi' in advanced_analysis:
                rsi_values = advanced_analysis['rsi']
                for period, value in rsi_values.items():
                    print(f"  - RSI {period}: {value:.2f}")
            
            # MACD değerleri
            if 'macd' in advanced_analysis:
                macd_data = advanced_analysis['macd']
                print(f"  - MACD: {macd_data.get('macd', 0):.4f}")
                print(f"  - MACD Signal: {macd_data.get('signal', 0):.4f}")
                print(f"  - MACD Histogram: {macd_data.get('histogram', 0):.4f}")
            
            # Bollinger Bands
            if 'bollinger_bands' in advanced_analysis:
                bb_data = advanced_analysis['bollinger_bands']
                print(f"  - BB Upper: {bb_data.get('upper', 0):.2f}")
                print(f"  - BB Middle: {bb_data.get('middle', 0):.2f}")
                print(f"  - BB Lower: {bb_data.get('lower', 0):.2f}")
                print(f"  - BB Width: {bb_data.get('width', 0):.4f}")
                print(f"  - BB Percent: {bb_data.get('percent', 0):.2f}")
        else:
            print("⚠ Gelişmiş analiz başarısız")
            if advanced_analysis and 'error' in advanced_analysis:
                print(f"  Hata: {advanced_analysis['error']}")
        
        # 11. Pattern analizi
        print("11. Pattern analizi yapılıyor...")
        df_with_patterns = ta.analyze_patterns(df_with_indicators)
        
        if df_with_patterns is not None and not df_with_patterns.empty:
            print("✓ Pattern analizi başarılı")
            
            # Son satırdaki pattern'ları kontrol et
            last_row = df_with_patterns.iloc[-1]
            pattern_columns = [col for col in df_with_patterns.columns if 'pattern' in col.lower() or 'doji' in col.lower() or 'hammer' in col.lower()]
            
            for pattern_col in pattern_columns:
                if pattern_col in last_row and last_row[pattern_col]:
                    print(f"  - {pattern_col}: TESPİT EDİLDİ")
        else:
            print("⚠ Pattern analizi başarısız")
        
        # 12. Sonuç raporu
        print("\n" + "=" * 60)
        print("📊 TEKNİK ANALİZ TEST SONUÇLARI")
        print("=" * 60)
        
        # Son değerler
        latest = df_with_indicators.iloc[-1]
        current_price = latest['close']
        
        print(f"📈 Sembol: {symbol}")
        print(f"💰 Güncel Fiyat: {current_price:.2f}")
        print(f"📅 Son Güncelleme: {df_with_indicators.index[-1]}")
        
        # Teknik göstergeler
        print(f"\n🔧 TEKNİK GÖSTERGELER:")
        print(f"  - RSI (14): {latest.get('rsi_14', 'N/A'):.2f}")
        print(f"  - MACD: {latest.get('macd', 'N/A'):.4f}")
        print(f"  - EMA (20): {latest.get('ema_20', 'N/A'):.2f}")
        print(f"  - SMA (50): {latest.get('sma_50', 'N/A'):.2f}")
        print(f"  - ATR: {latest.get('atr', 'N/A'):.4f}")
        print(f"  - Volume: {latest.get('volume', 'N/A'):.0f}")
        
        # Sinyal durumu
        print(f"\n🚦 SİNYAL DURUMU:")
        print(f"  - Sinyal Gücü: {signal_strength:.3f}")
        print(f"  - Trend Yönü: {trend_direction}")
        print(f"  - Trend Gücü: {trend_strength:.3f}")
        print(f"  - Momentum: {momentum_score:.3f}")
        print(f"  - Market Rejimi: {market_regime}")
        
        print("\n✅ Teknik analiz testi başarıyla tamamlandı!")
        return True
        
    except Exception as e:
        print(f"❌ Teknik analiz testi hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_symbols():
    """Birden fazla sembol için teknik analiz testi"""
    print("\n🔧 ÇOKLU SEMBOL TEKNİK ANALİZ TESTİ")
    print("=" * 60)
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    ta = TechnicalAnalysis()
    collector = DataCollector()
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 {symbol} analiz ediliyor...")
        
        try:
            # Veri al
            df = collector.get_historical_data(symbol, '1h', 200)
            
            if df is None or df.empty:
                print(f"❌ {symbol} için veri alınamadı")
                continue
            
            # Teknik analiz
            df_with_indicators = ta.calculate_all_indicators(df)
            
            if df_with_indicators is None or df_with_indicators.empty:
                print(f"❌ {symbol} teknik analiz başarısız")
                continue
            
            # Son değerler
            latest = df_with_indicators.iloc[-1]
            
            # Sinyal analizi
            signals = ta.generate_signals(df_with_indicators)
            signal_strength = ta.calculate_signal_strength(signals)
            trend_direction = ta.get_trend_direction(df_with_indicators)
            
            results[symbol] = {
                'price': latest['close'],
                'rsi': latest.get('rsi_14', 50),
                'macd': latest.get('macd', 0),
                'signal_strength': signal_strength,
                'trend': trend_direction,
                'volume': latest.get('volume', 0)
            }
            
            print(f"✓ {symbol} analizi tamamlandı")
            print(f"  - Fiyat: {latest['close']:.2f}")
            print(f"  - RSI: {latest.get('rsi_14', 50):.2f}")
            print(f"  - Sinyal Gücü: {signal_strength:.3f}")
            print(f"  - Trend: {trend_direction}")
            
        except Exception as e:
            print(f"❌ {symbol} analiz hatası: {e}")
            continue
    
    # Özet rapor
    print(f"\n📋 ÖZET RAPOR:")
    print(f"{'Sembol':<12} {'Fiyat':<10} {'RSI':<8} {'Sinyal':<8} {'Trend':<10}")
    print("-" * 50)
    
    for symbol, data in results.items():
        print(f"{symbol:<12} {data['price']:<10.2f} {data['rsi']:<8.2f} {data['signal_strength']:<8.3f} {data['trend']:<10}")
    
    return len(results) > 0

def main():
    """Ana test fonksiyonu"""
    print("🚀 KAHİN ULTIMA - TEKNİK ANALİZ TEST SİSTEMİ")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Tek sembol detaylı analiz
    if test_technical_analysis():
        success_count += 1
    
    # Test 2: Çoklu sembol analizi
    if test_multiple_symbols():
        success_count += 1
    
    # Final rapor
    print(f"\n🎯 TEST SONUÇLARI:")
    print(f"✓ Başarılı testler: {success_count}/{total_tests}")
    print(f"✓ Başarı oranı: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 Tüm testler başarıyla tamamlandı!")
    else:
        print("⚠ Bazı testler başarısız oldu, lütfen hataları kontrol edin.")
    
    return success_count == total_tests

if __name__ == "__main__":
    main() 