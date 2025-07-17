#!/usr/bin/env python3
"""
AI Modellerini Test Et
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

def test_ai_models():
    """AI modellerini test et"""
    print("🤖 AI MODELLERİ TEST EDİLİYOR")
    print("=" * 50)
    
    try:
        # Modülleri başlat
        ai_model = AIModel()
        collector = DataCollector()
        ta = TechnicalAnalysis()
        
        # Test verisi al
        symbol = 'BTC/USDT'
        df = collector.get_historical_data(symbol, '1h', 100)
        
        if df is None or df.empty:
            print("❌ Test verisi alınamadı")
            return False
        
        print(f"✓ {len(df)} satır veri alındı")
        
        # Teknik analiz yap
        df_with_indicators = ta.calculate_all_indicators(df)
        
        if df_with_indicators is None or df_with_indicators.empty:
            print("❌ Teknik analiz başarısız")
            return False
        
        print("✓ Teknik analiz tamamlandı")
        
        # AI tahmin yap
        print("AI tahmin yapılıyor...")
        prediction_result = ai_model.predict(df_with_indicators)
        
        if prediction_result and 'error' not in prediction_result:
            print("✓ AI tahmin başarılı")
            print(f"  - Tahmin: {prediction_result.get('prediction', 'N/A')}")
            print(f"  - Güven: {prediction_result.get('confidence', 'N/A')}")
            print(f"  - Kullanılan feature sayısı: {prediction_result.get('features_used', 'N/A')}")
        else:
            print("❌ AI tahmin başarısız")
            if prediction_result and 'error' in prediction_result:
                print(f"  Hata: {prediction_result['error']}")
        
        # Model istatistikleri
        stats = ai_model.get_model_stats()
        print(f"\n📊 MODEL İSTATİSTİKLERİ:")
        print(f"  - RF Model: {'✓' if stats['models_loaded']['rf'] else '❌'}")
        print(f"  - GB Model: {'✓' if stats['models_loaded']['gb'] else '❌'}")
        print(f"  - LSTM Model: {'✓' if stats['models_loaded']['lstm'] else '❌'}")
        print(f"  - Scaler: {'✓' if stats['models_loaded']['scaler'] else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ai_models()
    if success:
        print("\n✅ AI modelleri testi başarılı!")
    else:
        print("\n❌ AI modelleri testi başarısız!") 