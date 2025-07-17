#!/usr/bin/env python3
"""
Sistem performansını test etmek için script
"""

import time
import pandas as pd
import numpy as np
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from modules.data_collector import DataCollector

def test_system_performance():
    """Sistem performansını test et"""
    print("🔧 Sistem performansı test ediliyor...")
    
    try:
        # Test verisi oluştur
        print("📊 Test verisi oluşturuluyor...")
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=1000, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.randn(1000) * 1000,
            'high': 50000 + np.random.randn(1000) * 1200,
            'low': 50000 + np.random.randn(1000) * 800,
            'close': 50000 + np.random.randn(1000) * 1000,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        # Teknik analiz performans testi
        print("🔍 Teknik analiz performans testi...")
        ta = TechnicalAnalysis()
        
        start_time = time.time()
        ta_result = ta.analyze_technical_signals(data)
        ta_time = time.time() - start_time
        
        print(f"✅ Teknik analiz tamamlandı: {ta_time:.2f} saniye")
        
        # AI model performans testi
        print("🤖 AI model performans testi...")
        ai_model = AIModel()
        
        start_time = time.time()
        ai_result = ai_model.predict(data)
        ai_time = time.time() - start_time
        
        print(f"✅ AI analiz tamamlandı: {ai_time:.2f} saniye")
        
        # Toplam performans
        total_time = ta_time + ai_time
        print(f"\n📊 PERFORMANS SONUÇLARI:")
        print(f"  Teknik Analiz: {ta_time:.2f} saniye")
        print(f"  AI Analiz: {ai_time:.2f} saniye")
        print(f"  Toplam: {total_time:.2f} saniye")
        print(f"  Veri boyutu: {len(data)} satır")
        
        # Sonuç kalitesi kontrolü
        print(f"\n📈 SONUÇ KALİTESİ:")
        print(f"  Teknik indikatör sayısı: {len(ta_result)}")
        
        # AI sonucunu kontrol et
        if isinstance(ai_result, dict):
            print(f"  AI skor: {ai_result.get('prediction', 0):.3f}")
            print(f"  AI güven: {ai_result.get('confidence', 0):.3f}")
        else:
            print(f"  AI skor: {ai_result:.3f}")
            print(f"  AI güven: 0.500")
        
        # Performans değerlendirmesi
        if total_time < 5:
            print("✅ Performans: Mükemmel")
        elif total_time < 10:
            print("✅ Performans: İyi")
        elif total_time < 20:
            print("⚠️ Performans: Orta")
        else:
            print("❌ Performans: Yavaş")
        
        print("\n✅ Sistem performans testi tamamlandı!")
        
    except Exception as e:
        print(f"❌ Performans test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system_performance() 