import pandas as pd
import numpy as np
from modules.ai_model import AIModel
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from config import Config
from modules.technical_analysis import FIXED_FEATURE_LIST

def test_ai_scores():
    """AI modelinin skorlarını test et"""
    print("🤖 AI Model Skor Testi")
    print("=" * 50)
    
    # Modülleri başlat
    data_collector = DataCollector()
    ta = TechnicalAnalysis()
    ai_model = AIModel()
    
    # Test coinleri
    test_coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    
    for coin in test_coins:
        try:
            print(f"\n📊 {coin} test ediliyor...")
            
            # Veri topla
            df = data_collector.get_historical_data(coin, '1h', limit=1000)
            if df.empty:
                print(f"❌ {coin} için veri alınamadı")
                continue
                
            # Teknik analiz
            df = ta.calculate_all_indicators(df)
            if df.empty:
                print(f"❌ {coin} için teknik analiz başarısız")
                continue

            # Kolonları FIXED_FEATURE_LIST ile sırala ve eksik olanları sıfırla
            for col in FIXED_FEATURE_LIST:
                if col not in df.columns:
                    df[col] = 0
            df = df[FIXED_FEATURE_LIST]

            # AI skoru hesapla
            ai_score = ai_model.predict(df)
            print(f"✅ {coin}: AI Skoru = {ai_score['prediction']:.4f}")
            # Detaylı analiz
            if ai_score['prediction'] > 0.5:
                print(f"   🟢 Yüksek skor! Potansiyel sinyal")
            elif ai_score['prediction'] > 0.3:
                print(f"   🟡 Orta skor")
            else:
                print(f"   🔴 Düşük skor")
                
        except Exception as e:
            print(f"❌ {coin} test edilirken hata: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test tamamlandı!")

if __name__ == "__main__":
    test_ai_scores() 