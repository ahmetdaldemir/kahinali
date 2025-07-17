#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dinamik Sıkılık Entegrasyon Testi
Sinyal üretim sürecinde dinamik sıkılık sisteminin çalışmasını test eder
"""

import os
import sys
import logging
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.dynamic_strictness import DynamicStrictness
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from config import Config

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dynamic_strictness_integration():
    """Dinamik sıkılık entegrasyonunu test et"""
    print("🧪 DİNAMİK SIKILIK ENTEGRASYON TESTİ")
    print("=" * 60)
    
    try:
        # Modülleri başlat
        dynamic_strictness = DynamicStrictness()
        data_collector = DataCollector()
        technical_analyzer = TechnicalAnalysis()
        ai_model = AIModel()
        
        print("✅ Modüller başlatıldı")
        
        # Popüler coinleri al
        popular_coins = data_collector.get_popular_usdt_pairs(max_pairs=10)
        print(f"📊 {len(popular_coins)} coin bulundu")
        
        # Market verisi topla
        print("\n📈 Market verisi toplanıyor...")
        market_data = {
            'price_data': [],
            'technical_indicators': {},
            'volume_data': [],
            'sentiment_data': {'overall_sentiment': 0.5},
            'ai_predictions': {'confidence': 0.5}
        }
        
        # İlk 5 coin'den veri topla
        for coin in popular_coins[:5]:
            try:
                data = data_collector.get_historical_data(coin, '1h', 50)
                if not data.empty:
                    # Fiyat verisi
                    market_data['price_data'].extend(data['close'].tolist())
                    
                    # Teknik analiz
                    ta_data = technical_analyzer.calculate_all_indicators(data)
                    if 'rsi_14' in ta_data.columns:
                        market_data['technical_indicators']['rsi'] = ta_data['rsi_14'].iloc[-1]
                    if 'macd' in ta_data.columns:
                        market_data['technical_indicators']['macd'] = ta_data['macd'].iloc[-1]
                    
                    # Hacim verisi
                    if 'volume' in data.columns:
                        market_data['volume_data'].extend(data['volume'].tolist())
                    
                    # AI tahmini
                    ai_result = ai_model.predict(ta_data)
                    if ai_result:
                        market_data['ai_predictions']['confidence'] = ai_result.get('confidence', 0.5)
                        
                    print(f"   ✅ {coin}: Veri toplandı")
                        
            except Exception as e:
                print(f"   ❌ {coin}: Veri toplama hatası - {e}")
                continue
        
        # Dinamik sıkılığı güncelle
        print("\n🔧 Dinamik sıkılık hesaplanıyor...")
        strictness_status = dynamic_strictness.update_strictness(market_data)
        
        current_strictness = strictness_status['current_strictness']
        strictness_level = strictness_status['strictness_level']
        recommendation = strictness_status['recommendation']
        
        print(f"   Sıkılık değeri: {current_strictness:.3f}")
        print(f"   Sıkılık seviyesi: {strictness_level}")
        print(f"   Öneri: {recommendation}")
        
        # Dinamik eşikleri hesapla
        print("\n⚙️ Dinamik eşikler hesaplanıyor...")
        dynamic_min_confidence = max(0.3, min(0.8, current_strictness))
        dynamic_min_ai_score = max(0.3, min(0.7, current_strictness - 0.1))
        dynamic_min_ta_strength = max(0.2, min(0.6, current_strictness - 0.15))
        
        print(f"   Dinamik Min. Güven: {dynamic_min_confidence:.3f} (sabit: {Config.MIN_SIGNAL_CONFIDENCE})")
        print(f"   Dinamik Min. AI Skoru: {dynamic_min_ai_score:.3f} (sabit: {Config.MIN_AI_SCORE})")
        print(f"   Dinamik Min. TA Gücü: {dynamic_min_ta_strength:.3f} (sabit: {Config.MIN_TA_STRENGTH})")
        
        # Eşik karşılaştırması
        print("\n📊 EŞİK KARŞILAŞTIRMASI:")
        confidence_diff = dynamic_min_confidence - Config.MIN_SIGNAL_CONFIDENCE
        ai_diff = dynamic_min_ai_score - Config.MIN_AI_SCORE
        ta_diff = dynamic_min_ta_strength - Config.MIN_TA_STRENGTH
        
        print(f"   Güven eşiği farkı: {confidence_diff:+.3f}")
        print(f"   AI skoru farkı: {ai_diff:+.3f}")
        print(f"   TA gücü farkı: {ta_diff:+.3f}")
        
        # Sinyal üretim simülasyonu
        print("\n🎯 SİNYAL ÜRETİM SİMÜLASYONU:")
        
        test_signals = 0
        passed_signals = 0
        
        for coin in popular_coins[:10]:
            try:
                # Test verisi oluştur
                test_ai_score = 0.6  # Orta AI skoru
                test_ta_strength = 0.5  # Orta TA gücü
                test_total_score = (test_ai_score * 0.4 + test_ta_strength * 0.25 + 0.5 * 0.35)
                
                # Dinamik eşik kontrolü
                passed_confidence = test_total_score >= dynamic_min_confidence
                passed_ai = test_ai_score >= dynamic_min_ai_score
                passed_ta = test_ta_strength >= dynamic_min_ta_strength
                
                all_passed = passed_confidence and passed_ai and passed_ta
                
                test_signals += 1
                if all_passed:
                    passed_signals += 1
                
                status = "✅ GEÇTİ" if all_passed else "❌ BAŞARISIZ"
                print(f"   {coin}: {status} (AI: {test_ai_score:.2f}, TA: {test_ta_strength:.2f}, Total: {test_total_score:.2f})")
                
            except Exception as e:
                print(f"   {coin}: Test hatası - {e}")
                continue
        
        # Sonuçlar
        print(f"\n📈 TEST SONUÇLARI:")
        print(f"   Test edilen sinyal: {test_signals}")
        print(f"   Geçen sinyal: {passed_signals}")
        print(f"   Geçme oranı: {passed_signals/test_signals*100:.1f}%")
        
        # Dinamik sıkılık etkisi
        if current_strictness < Config.MIN_SIGNAL_CONFIDENCE:
            print(f"   🟢 Dinamik sıkılık daha gevşek - daha fazla sinyal üretilecek")
        elif current_strictness > Config.MIN_SIGNAL_CONFIDENCE:
            print(f"   🔴 Dinamik sıkılık daha sıkı - daha az sinyal üretilecek")
        else:
            print(f"   🟡 Dinamik sıkılık sabit eşikle aynı")
        
        print("\n✅ Dinamik sıkılık entegrasyon testi başarılı!")
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_dynamic_strictness_integration() 