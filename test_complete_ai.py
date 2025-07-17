#!/usr/bin/env python3
"""
KAHİN ULTIMA - Kapsamlı AI Modelleri Test
Tüm modelleri (LSTM, RF, GB) test eder
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.ai_model import AIModel
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis

def test_complete_ai():
    """Kapsamlı AI modellerini test et"""
    print("🤖 KAPSAMLI AI MODELLERİ TEST EDİLİYOR")
    print("=" * 60)
    
    try:
        # 1. AI model modülünü başlat
        print("1. AI modelleri yükleniyor...")
        ai_model = AIModel()
        print("✓ AI modelleri yüklendi")
        
        # 2. Test verisi al
        print("2. Test verisi alınıyor...")
        collector = DataCollector()
        df = collector.get_historical_data('BTC/USDT', '1h', 100)
        
        if df is None or df.empty:
            print("❌ Test verisi alınamadı")
            return False
            
        print(f"✓ {len(df)} satır veri alındı")
        
        # 3. Teknik analiz
        print("3. Teknik analiz yapılıyor...")
        ta = TechnicalAnalysis()
        df_with_indicators = ta.calculate_all_indicators(df)
        
        if df_with_indicators is None or df_with_indicators.empty:
            print("❌ Teknik analiz başarısız")
            return False
            
        print(f"✓ Teknik analiz tamamlandı")
        print(f"✓ Toplam kolon sayısı: {df_with_indicators.shape[1]}")
        
        # 4. AI tahmin
        print("4. AI tahmin yapılıyor...")
        result = ai_model.predict(df_with_indicators)
        
        if isinstance(result, dict) and 'prediction' in result and 'confidence' in result:
            prediction = result['prediction']
            confidence = result['confidence']
            if prediction is not None and confidence is not None:
                print(f"✓ AI tahmin başarılı!")
                print(f"  - Tahmin: {prediction:.4f}")
                print(f"  - Güven: {confidence:.4f}")
                
                # 5. Model durumları
                print("5. Model durumları kontrol ediliyor...")
                
                if ai_model.lstm_model is not None:
                    print("  ✓ LSTM modeli aktif")
                else:
                    print("  ❌ LSTM modeli yok")
                
                if ai_model.rf_model is not None:
                    print("  ✓ Random Forest modeli aktif")
                else:
                    print("  ❌ Random Forest modeli yok")
                
                if ai_model.gb_model is not None:
                    print("  ✓ Gradient Boosting modeli aktif")
                else:
                    print("  ❌ Gradient Boosting modeli yok")
                
                if ai_model.scaler is not None:
                    print("  ✓ Scaler aktif")
                else:
                    print("  ❌ Scaler yok")
                
                # 6. Sonuç raporu
                print("\n" + "=" * 60)
                print("🎯 KAPSAMLI AI TEST SONUÇLARI")
                print("=" * 60)
                
                latest = df_with_indicators.iloc[-1]
                current_price = latest['close']
                
                print(f"📈 Sembol: BTC/USDT")
                print(f"💰 Güncel Fiyat: {current_price:.2f}")
                print(f"🤖 AI Tahmin: {prediction:.4f}")
                print(f"🎯 Güven Skoru: {confidence:.4f}")
                
                # Tahmin yorumu
                if prediction > 0.7:
                    trend = "GÜÇLÜ YÜKSELİŞ"
                elif prediction > 0.6:
                    trend = "YÜKSELİŞ"
                elif prediction > 0.4:
                    trend = "NÖTR"
                elif prediction > 0.3:
                    trend = "DÜŞÜŞ"
                else:
                    trend = "GÜÇLÜ DÜŞÜŞ"
                
                print(f"📊 Trend: {trend}")
                
                if confidence > 0.8:
                    confidence_level = "YÜKSEK"
                elif confidence > 0.6:
                    confidence_level = "ORTA"
                else:
                    confidence_level = "DÜŞÜK"
                
                print(f"🎯 Güven Seviyesi: {confidence_level}")
                
                print("\n✅ KAPSAMLI AI TESTİ BAŞARIYLA TAMAMLANDI!")
                return True
            else:
                print("❌ AI tahmin başarısız")
                return False
        else:
            print(f"❌ AI tahmin format hatası: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_symbols():
    """Birden fazla sembol için AI testi"""
    print("\n🤖 ÇOKLU SEMBOL AI TESTİ")
    print("=" * 60)
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    ai_model = AIModel()
    collector = DataCollector()
    ta = TechnicalAnalysis()
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 {symbol} AI analizi...")
        
        try:
            # Veri al
            df = collector.get_historical_data(symbol, '1h', 100)
            
            if df is None or df.empty:
                print(f"❌ {symbol} için veri alınamadı")
                continue
            
            # Teknik analiz
            df_with_indicators = ta.calculate_all_indicators(df)
            
            if df_with_indicators is None or df_with_indicators.empty:
                print(f"❌ {symbol} teknik analiz başarısız")
                continue
            
            # AI tahmin
            result = ai_model.predict(df_with_indicators)
            
            if isinstance(result, dict) and 'prediction' in result and 'confidence' in result:
                prediction = result['prediction']
                confidence = result['confidence']
                
                if prediction is not None and confidence is not None:
                    latest = df_with_indicators.iloc[-1]
                    
                    results[symbol] = {
                        'price': latest['close'],
                        'prediction': prediction,
                        'confidence': confidence,
                        'trend': 'YÜKSELİŞ' if prediction > 0.5 else 'DÜŞÜŞ'
                    }
                    
                    print(f"✓ {symbol} AI analizi tamamlandı")
                    print(f"  - Fiyat: {latest['close']:.2f}")
                    print(f"  - Tahmin: {prediction:.4f}")
                    print(f"  - Güven: {confidence:.4f}")
                    print(f"  - Trend: {'YÜKSELİŞ' if prediction > 0.5 else 'DÜŞÜŞ'}")
                else:
                    print(f"❌ {symbol} AI tahmin başarısız")
            else:
                print(f"❌ {symbol} AI tahmin format hatası")
                
        except Exception as e:
            print(f"❌ {symbol} analiz hatası: {e}")
            continue
    
    # Özet rapor
    if results:
        print(f"\n📋 AI ÖZET RAPOR:")
        print(f"{'Sembol':<12} {'Fiyat':<10} {'Tahmin':<8} {'Güven':<8} {'Trend':<10}")
        print("-" * 50)
        
        for symbol, data in results.items():
            print(f"{symbol:<12} {data['price']:<10.2f} {data['prediction']:<8.4f} {data['confidence']:<8.4f} {data['trend']:<10}")
    
    return len(results) > 0

def main():
    """Ana test fonksiyonu"""
    print("🚀 KAHİN ULTIMA - KAPSAMLI AI TEST SİSTEMİ")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Tek sembol detaylı AI analizi
    if test_complete_ai():
        success_count += 1
    
    # Test 2: Çoklu sembol AI analizi
    if test_multiple_symbols():
        success_count += 1
    
    # Final rapor
    print(f"\n🎯 TEST SONUÇLARI:")
    print(f"✓ Başarılı testler: {success_count}/{total_tests}")
    print(f"✓ Başarı oranı: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 Tüm AI testleri başarıyla tamamlandı!")
        print("🤖 Sistem tam eksiksiz çalışıyor!")
    else:
        print("⚠ Bazı AI testleri başarısız oldu, lütfen hataları kontrol edin.")
    
    return success_count == total_tests

if __name__ == "__main__":
    main() 