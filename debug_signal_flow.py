#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from config import Config

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_signal_flow():
    """Sinyal üretim sürecini adım adım debug et"""
    
    print("=" * 60)
    print("🔍 SİNYAL ÜRETİM SÜRECİ DEBUG")
    print("=" * 60)
    
    try:
        # 1. Veri toplama
        print("\n1️⃣ VERİ TOPLAMA:")
        data_collector = DataCollector()
        popular_coins = data_collector.get_popular_usdt_pairs(max_pairs=5)  # Sadece 5 coin
        print(f"   Popüler coinler: {popular_coins}")
        
        if not popular_coins:
            print("   ❌ Hiç coin bulunamadı!")
            return
        
        # 2. Tek coin test et
        test_coin = popular_coins[0]
        print(f"\n2️⃣ TEST COİN: {test_coin}")
        
        # 3. Veri al
        print("\n3️⃣ VERİ ALMA:")
        data = data_collector.get_historical_data(test_coin, '1h', limit=100)
        print(f"   Veri boyutu: {len(data) if data is not None else 'None'}")
        
        if data is None or data.empty:
            print("   ❌ Veri alınamadı!")
            return
        
        # 4. Teknik analiz
        print("\n4️⃣ TEKNİK ANALİZ:")
        ta = TechnicalAnalysis()
        ta_data = ta.calculate_all_indicators(data)
        print(f"   TA veri boyutu: {len(ta_data) if ta_data is not None else 'None'}")
        
        if ta_data is None or ta_data.empty:
            print("   ❌ TA analizi başarısız!")
            return
        
        # 5. AI analizi
        print("\n5️⃣ AI ANALİZİ:")
        ai_model = AIModel()
        ai_result = ai_model.predict(ta_data)
        print(f"   AI sonucu: {ai_result}")
        
        # 6. Skor hesaplama
        print("\n6️⃣ SKOR HESAPLAMA:")
        
        # AI skoru
        ai_score = ai_result.get('prediction', 0.5)
        print(f"   AI skoru: {ai_score:.4f}")
        
        # TA skoru (basit hesaplama)
        ta_strength = 0.5  # Basit değer
        print(f"   TA skoru: {ta_strength:.4f}")
        
        # Toplam skor
        total_score = (ai_score * 0.6) + (ta_strength * 0.4)
        print(f"   Toplam skor: {total_score:.4f}")
        
        # 7. Eşik kontrolü
        print("\n7️⃣ EŞİK KONTROLÜ:")
        print(f"   Minimum eşik: {Config.MIN_SIGNAL_CONFIDENCE}")
        print(f"   AI eşik: {Config.MIN_AI_SCORE}")
        print(f"   TA eşik: {Config.MIN_TA_STRENGTH}")
        
        # 8. Sonuç
        print("\n8️⃣ SONUÇ:")
        if total_score >= Config.MIN_SIGNAL_CONFIDENCE:
            print(f"   ✅ SİNYAL ÜRETİLECEK! Skor: {total_score:.4f}")
        else:
            print(f"   ❌ SİNYAL ÜRETİLMEYECEK! Skor: {total_score:.4f} < {Config.MIN_SIGNAL_CONFIDENCE}")
        
        if ai_score >= Config.MIN_AI_SCORE:
            print(f"   ✅ AI skoru yeterli: {ai_score:.4f}")
        else:
            print(f"   ❌ AI skoru yetersiz: {ai_score:.4f} < {Config.MIN_AI_SCORE}")
            
        if ta_strength >= Config.MIN_TA_STRENGTH:
            print(f"   ✅ TA skoru yeterli: {ta_strength:.4f}")
        else:
            print(f"   ❌ TA skoru yetersiz: {ta_strength:.4f} < {Config.MIN_TA_STRENGTH}")
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_signal_flow() 