#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.signal_manager import SignalManager
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel

def test_real_signal_production():
    """Gerçek sinyal üretim sürecini test et"""
    
    print("=== GERÇEK SİNYAL ÜRETİM TESTİ ===")
    
    try:
        # Modülleri başlat
        signal_manager = SignalManager()
        data_collector = DataCollector()
        ta = TechnicalAnalysis()
        ai_model = AIModel()
        
        # Test için BTC/USDT verisi al
        symbol = "BTC/USDT"
        print(f"1. {symbol} için veri alınıyor...")
        
        df = data_collector.get_historical_data(symbol, '1h', limit=200)
        if df.empty:
            print("❌ Veri alınamadı!")
            return False
        
        print(f"✓ {len(df)} satır veri alındı")
        
        # Teknik analiz
        print("2. Teknik analiz hesaplanıyor...")
        df_ta = ta.calculate_all_indicators(df)
        if df_ta.empty:
            print("❌ Teknik analiz başarısız!")
            return False
        
        print(f"✓ Teknik analiz tamamlandı. Kolon sayısı: {len(df_ta.columns)}")
        
        # AI tahmini
        print("3. AI tahmini yapılıyor...")
        ai_result = ai_model.predict_simple(df_ta)
        if ai_result is None:
            print("❌ AI tahmini başarısız!")
            return False
        
        ai_score = ai_result.get('prediction', 0.5)
        confidence = ai_result.get('confidence', 0.5)
        print(f"✓ AI tahmini: {ai_score:.4f}, Confidence: {confidence:.4f}")
        
        # Analysis data hazırla
        print("4. Analysis data hazırlanıyor...")
        analysis_data = df_ta.iloc[-1].to_dict()
        analysis_data['ai_score'] = ai_score
        analysis_data['ta_strength'] = ta.calculate_signal_strength(df_ta)
        analysis_data['whale_score'] = 0.5  # Dummy
        analysis_data['social_score'] = 0.0  # Pasif
        analysis_data['news_score'] = 0.0    # Pasif
        analysis_data['confidence'] = confidence
        analysis_data['trend_alignment'] = ta.get_trend_direction(df_ta)
        analysis_data['timeframe'] = '1h'  # Timeframe ekle
        
        # Metrikleri hesapla
        print("5. Metrikler hesaplanıyor...")
        
        # Volume score
        volume_score = ta.calculate_volume_score(df_ta)
        analysis_data['volume_score'] = volume_score if volume_score != 'Veri Yok' else 0.5
        print(f"   Volume Score: {analysis_data['volume_score']}")
        
        # Momentum score
        momentum_score = ta.calculate_momentum_score(df_ta)
        analysis_data['momentum_score'] = momentum_score if momentum_score != 'Veri Yok' else 0.5
        print(f"   Momentum Score: {analysis_data['momentum_score']}")
        
        # Pattern score
        pattern_score = df_ta['pattern_score'].iloc[-1] if 'pattern_score' in df_ta.columns else 0.5
        analysis_data['pattern_score'] = pattern_score
        print(f"   Pattern Score: {analysis_data['pattern_score']}")
        
        # Whale tracker metrikleri
        from modules.whale_tracker import detect_whale_trades
        whale_data = detect_whale_trades(symbol)
        analysis_data['whale_direction_score'] = whale_data.get('whale_direction_score', 0.0)
        analysis_data['order_book_imbalance'] = whale_data.get('order_book_imbalance', 0.0)
        analysis_data['top_bid_walls'] = whale_data.get('top_bid_walls', [])  # DÜZELTİLDİ: str(...) kaldırıldı
        analysis_data['top_ask_walls'] = whale_data.get('top_ask_walls', [])  # DÜZELTİLDİ: str(...) kaldırıldı
        print(f"   Whale Direction: {analysis_data['whale_direction_score']}")
        print(f"   Order Book Imbalance: {analysis_data['order_book_imbalance']}")
        print(f"   Bid Walls: {analysis_data['top_bid_walls']}")
        print(f"   Ask Walls: {analysis_data['top_ask_walls']}")
        
        # Sinyal oluştur
        print("6. Sinyal oluşturuluyor...")
        direction = 'LONG' if ai_score > 0.5 else 'SHORT'
        
        signal = signal_manager.create_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            analysis_data=analysis_data
        )
        
        if signal is None:
            print("❌ Sinyal oluşturulamadı!")
            return False
        
        print("✓ Sinyal oluşturuldu!")
        print(f"   Symbol: {signal['symbol']}")
        print(f"   Direction: {signal['direction']}")
        print(f"   AI Score: {signal['ai_score']}")
        print(f"   Volume Score: {signal['volume_score']}")
        print(f"   Momentum Score: {signal['momentum_score']}")
        print(f"   Pattern Score: {signal['pattern_score']}")
        print(f"   Whale Direction: {signal['whale_direction_score']}")
        print(f"   Order Book Imbalance: {signal['order_book_imbalance']}")
        print(f"   Bid Walls: {signal['top_bid_walls']}")
        print(f"   Ask Walls: {signal['top_ask_walls']}")
        
        # Sinyali kaydet
        print("7. Sinyal kaydediliyor...")
        signal_manager.save_signal_json(signal)
        signal_manager.save_signal_csv(signal)
        signal_manager.save_signal_db(signal)
        print("✓ Sinyal kaydedildi!")
        
        print("\n✅ GERÇEK SİNYAL ÜRETİM TESTİ BAŞARILI!")
        print("Panelde kontrol edin: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_real_signal_production() 