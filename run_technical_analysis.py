#!/usr/bin/env python3
"""
KAHİN ULTIMA - Teknik Analiz Çalıştırıcı
Belirtilen semboller için detaylı teknik analiz yapar
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

def analyze_symbol_detailed(symbol, timeframe='1h', limit=500):
    """Belirtilen sembol için detaylı teknik analiz"""
    print(f"🔧 {symbol} DETAYLI TEKNİK ANALİZ")
    print("=" * 60)
    
    try:
        # Modülleri başlat
        ta = TechnicalAnalysis()
        collector = DataCollector()
        
        # Veri topla
        print(f"📊 {symbol} için {timeframe} verisi alınıyor...")
        df = collector.get_historical_data(symbol, timeframe, limit)
        
        if df is None or df.empty:
            print(f"❌ {symbol} için veri alınamadı")
            return None
            
        print(f"✓ {len(df)} satır veri alındı")
        print(f"✓ Veri aralığı: {df.index[0]} - {df.index[-1]}")
        
        # Teknik analiz
        print("🔧 Teknik analiz yapılıyor...")
        df_with_indicators = ta.calculate_all_indicators(df)
        
        if df_with_indicators is None or df_with_indicators.empty:
            print("❌ Teknik analiz başarısız")
            return None
            
        print(f"✓ Teknik analiz tamamlandı ({df_with_indicators.shape[1]} kolon)")
        
        # Gelişmiş analiz
        print("🔬 Gelişmiş analiz yapılıyor...")
        advanced_analysis = ta.analyze_technical_signals(df_with_indicators)
        
        # Pattern analizi
        print("📈 Pattern analizi yapılıyor...")
        df_with_patterns = ta.analyze_patterns(df_with_indicators)
        
        # Sinyal analizi
        signals = ta.generate_signals(df_with_indicators)
        signal_strength = ta.calculate_signal_strength(signals)
        trend_direction = ta.get_trend_direction(df_with_indicators)
        trend_strength = ta.calculate_trend_strength(df_with_indicators)
        momentum_score = ta.calculate_momentum_score(df_with_indicators)
        market_regime = ta.get_market_regime(df_with_indicators)
        
        # Son değerler
        latest = df_with_indicators.iloc[-1]
        
        # Rapor oluştur
        report = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points': len(df),
            'current_price': latest['close'],
            'price_change_24h': df['close'].pct_change().iloc[-1] * 100,
            'volume_24h': latest.get('volume', 0),
            'technical_indicators': {
                'rsi_14': latest.get('rsi_14', 50),
                'rsi_7': latest.get('rsi_7', 50),
                'rsi_21': latest.get('rsi_21', 50),
                'macd': latest.get('macd', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'macd_histogram': latest.get('macd_histogram', 0),
                'ema_20': latest.get('ema_20', 0),
                'ema_50': latest.get('ema_50', 0),
                'sma_50': latest.get('sma_50', 0),
                'sma_200': latest.get('sma_200', 0),
                'bb_upper': latest.get('bb_upper', 0),
                'bb_middle': latest.get('bb_middle', 0),
                'bb_lower': latest.get('bb_lower', 0),
                'bb_width': latest.get('bb_width', 0),
                'bb_percent': latest.get('bb_percent', 0.5),
                'atr': latest.get('atr', 0),
                'stoch_k': latest.get('stoch_k', 50),
                'stoch_d': latest.get('stoch_d', 50),
                'cci': latest.get('cci', 0),
                'mfi': latest.get('mfi', 50),
                'williams_r': latest.get('williams_r', -50),
                'obv': latest.get('obv', 0),
                'vwap': latest.get('vwap', 0)
            },
            'signals': {
                'signal_strength': signal_strength,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'market_regime': market_regime
            },
            'advanced_analysis': advanced_analysis,
            'support_resistance': ta.get_support_resistance(df_with_indicators),
            'patterns': {}
        }
        
        # Pattern'ları kontrol et
        if df_with_patterns is not None and not df_with_patterns.empty:
            last_row = df_with_patterns.iloc[-1]
            pattern_columns = [col for col in df_with_patterns.columns 
                             if any(keyword in col.lower() for keyword in 
                                   ['pattern', 'doji', 'hammer', 'shooting', 'engulfing', 
                                    'morning', 'evening', 'double', 'head', 'triangle', 'flag'])]
            
            for pattern_col in pattern_columns:
                if pattern_col in last_row and last_row[pattern_col]:
                    report['patterns'][pattern_col] = True
                else:
                    report['patterns'][pattern_col] = False
        
        return report
        
    except Exception as e:
        print(f"❌ {symbol} analiz hatası: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_detailed_report(report):
    """Detaylı raporu yazdır"""
    if not report:
        print("❌ Rapor bulunamadı")
        return
    
    print("\n" + "=" * 80)
    print(f"📊 {report['symbol']} DETAYLI TEKNİK ANALİZ RAPORU")
    print("=" * 80)
    print(f"📅 Tarih: {report['timestamp']}")
    print(f"⏰ Zaman Dilimi: {report['timeframe']}")
    print(f"📈 Veri Noktası: {report['data_points']}")
    print()
    
    # Fiyat bilgileri
    print("💰 FİYAT BİLGİLERİ:")
    print(f"  Güncel Fiyat: {report['current_price']:.4f}")
    print(f"  24s Değişim: {report['price_change_24h']:.2f}%")
    print(f"  24s Hacim: {report['volume_24h']:,.0f}")
    print()
    
    # Temel göstergeler
    indicators = report['technical_indicators']
    print("🔧 TEMEL GÖSTERGELER:")
    print(f"  RSI (14): {indicators['rsi_14']:.2f}")
    print(f"  RSI (7): {indicators['rsi_7']:.2f}")
    print(f"  RSI (21): {indicators['rsi_21']:.2f}")
    print(f"  MACD: {indicators['macd']:.4f}")
    print(f"  MACD Signal: {indicators['macd_signal']:.4f}")
    print(f"  MACD Histogram: {indicators['macd_histogram']:.4f}")
    print(f"  EMA (20): {indicators['ema_20']:.4f}")
    print(f"  EMA (50): {indicators['ema_50']:.4f}")
    print(f"  SMA (50): {indicators['sma_50']:.4f}")
    print(f"  SMA (200): {indicators['sma_200']:.4f}")
    print()
    
    # Bollinger Bands
    print("📊 BOLLINGER BANDS:")
    print(f"  Üst Bant: {indicators['bb_upper']:.4f}")
    print(f"  Orta Bant: {indicators['bb_middle']:.4f}")
    print(f"  Alt Bant: {indicators['bb_lower']:.4f}")
    print(f"  Genişlik: {indicators['bb_width']:.4f}")
    print(f"  Pozisyon: {indicators['bb_percent']:.2f}")
    print()
    
    # Diğer göstergeler
    print("📈 DİĞER GÖSTERGELER:")
    print(f"  ATR: {indicators['atr']:.4f}")
    print(f"  Stochastic K: {indicators['stoch_k']:.2f}")
    print(f"  Stochastic D: {indicators['stoch_d']:.2f}")
    print(f"  CCI: {indicators['cci']:.2f}")
    print(f"  MFI: {indicators['mfi']:.2f}")
    print(f"  Williams %R: {indicators['williams_r']:.2f}")
    print(f"  OBV: {indicators['obv']:,.0f}")
    print(f"  VWAP: {indicators['vwap']:.4f}")
    print()
    
    # Sinyal durumu
    signals = report['signals']
    print("🚦 SİNYAL DURUMU:")
    print(f"  Sinyal Gücü: {signals['signal_strength']:.3f}")
    print(f"  Trend Yönü: {signals['trend_direction']}")
    print(f"  Trend Gücü: {signals['trend_strength']:.3f}")
    print(f"  Momentum Skoru: {signals['momentum_score']:.3f}")
    print(f"  Market Rejimi: {signals['market_regime']}")
    print()
    
    # Destek/Direnç
    sr = report['support_resistance']
    print("🎯 DESTEK/DİRENÇ:")
    print(f"  Destek: {sr['support']:.4f}")
    print(f"  Direnç: {sr['resistance']:.4f}")
    print()
    
    # Pattern'lar
    if report['patterns']:
        print("📊 TESPİT EDİLEN PATTERN'LAR:")
        for pattern, detected in report['patterns'].items():
            status = "✅ TESPİT EDİLDİ" if detected else "❌ TESPİT EDİLMEDİ"
            print(f"  {pattern}: {status}")
        print()
    
    # Gelişmiş analiz
    if report['advanced_analysis'] and 'error' not in report['advanced_analysis']:
        adv = report['advanced_analysis']
        print("🔬 GELİŞMİŞ ANALİZ:")
        
        if 'rsi' in adv:
            print("  RSI Değerleri:")
            for period, value in adv['rsi'].items():
                print(f"    {period}: {value:.2f}")
        
        if 'macd' in adv:
            macd = adv['macd']
            print("  MACD Analizi:")
            print(f"    MACD: {macd.get('macd', 0):.4f}")
            print(f"    Signal: {macd.get('signal', 0):.4f}")
            print(f"    Histogram: {macd.get('histogram', 0):.4f}")
            print(f"    Bullish Cross: {'Evet' if macd.get('bullish_cross', False) else 'Hayır'}")
            print(f"    Bearish Cross: {'Evet' if macd.get('bearish_cross', False) else 'Hayır'}")
        
        if 'bollinger_bands' in adv:
            bb = adv['bollinger_bands']
            print("  Bollinger Bands:")
            print(f"    Üst: {bb.get('upper', 0):.4f}")
            print(f"    Orta: {bb.get('middle', 0):.4f}")
            print(f"    Alt: {bb.get('lower', 0):.4f}")
            print(f"    Genişlik: {bb.get('width', 0):.4f}")
            print(f"    Pozisyon: {bb.get('percent', 0):.2f}")
        
        if 'patterns' in adv:
            patterns = adv['patterns']
            print("  Pattern Durumu:")
            for pattern, detected in patterns.items():
                status = "✅" if detected else "❌"
                print(f"    {pattern}: {status}")
    
    print("=" * 80)

def analyze_multiple_symbols(symbols, timeframe='1h', limit=500):
    """Birden fazla sembol için analiz"""
    print(f"🔧 ÇOKLU SEMBOL ANALİZİ ({len(symbols)} sembol)")
    print("=" * 60)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 {symbol} analiz ediliyor...")
        report = analyze_symbol_detailed(symbol, timeframe, limit)
        
        if report:
            results[symbol] = report
            print(f"✓ {symbol} analizi tamamlandı")
        else:
            print(f"❌ {symbol} analizi başarısız")
    
    # Özet rapor
    if results:
        print(f"\n📋 ÖZET RAPOR ({len(results)} sembol):")
        print(f"{'Sembol':<12} {'Fiyat':<12} {'RSI':<8} {'MACD':<12} {'Trend':<12} {'Sinyal':<8}")
        print("-" * 70)
        
        for symbol, report in results.items():
            indicators = report['technical_indicators']
            signals = report['signals']
            
            print(f"{symbol:<12} {report['current_price']:<12.4f} "
                  f"{indicators['rsi_14']:<8.2f} {indicators['macd']:<12.4f} "
                  f"{signals['trend_direction']:<12} {signals['signal_strength']:<8.3f}")
    
    return results

def main():
    """Ana fonksiyon"""
    print("🚀 KAHİN ULTIMA - TEKNİK ANALİZ ÇALIŞTIRICI")
    print("=" * 60)
    
    # Test sembolleri
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT']
    
    # Çoklu sembol analizi
    results = analyze_multiple_symbols(symbols, '1h', 500)
    
    # Detaylı raporlar
    print(f"\n📊 DETAYLI RAPORLAR:")
    for symbol, report in results.items():
        print_detailed_report(report)
    
    # Sonuç
    print(f"\n🎯 ANALİZ TAMAMLANDI:")
    print(f"✓ Toplam analiz edilen sembol: {len(results)}")
    print(f"✓ Başarılı analiz: {len(results)}")
    print(f"✓ Başarı oranı: {(len(results)/len(symbols))*100:.1f}%")
    
    return len(results) > 0

if __name__ == "__main__":
    main() 