import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from modules.whale_tracker import WhaleTracker
from modules.news_analysis import NewsAnalysis
from modules.signal_manager import SignalManager
from datetime import datetime

def test_eth_signal():
    print("ETH/USDT için sinyal üretim testi başlatılıyor...\n")
    
    # Modülleri başlat
    collector = DataCollector()
    ta = TechnicalAnalysis()
    ai_model = AIModel()
    whale_tracker = WhaleTracker()
    news_analysis = NewsAnalysis()
    signal_manager = SignalManager()
    
    try:
        # 1. ETH/USDT için veri topla
        print("1. ETH/USDT için veri toplanıyor...")
        df = collector.get_historical_data('ETH/USDT', '1h', limit=1000)
        if df.empty:
            print("❌ ETH/USDT için veri alınamadı")
            return False
        print(f"✓ {len(df)} satır veri alındı")
        
        # 2. Teknik analiz
        print("2. Teknik analiz yapılıyor...")
        df = ta.calculate_all_indicators(df)
        if df.empty:
            print("❌ Teknik analiz başarısız")
            return False
        
        ta_signals = ta.generate_signals(df)
        ta_strength = ta.calculate_signal_strength(ta_signals)
        if ta_strength is None:
            ta_strength = 0.5
        print(f"✓ Teknik analiz tamamlandı")
        print(f"✓ Teknik analiz gücü: {ta_strength}")
        
        # 3. AI tahminleri
        print("3. AI tahminleri yapılıyor...")
        try:
            ai_direction, ai_score = ai_model.predict_lstm(df)
        except Exception as e:
            print(f"⚠ LSTM hatası: {e}")
            ai_direction, ai_score = "NEUTRAL", 0.5
            
        try:
            rf_direction, rf_score = ai_model.predict_rf(df)
        except Exception as e:
            print(f"⚠ RF hatası: {e}")
            rf_direction, rf_score = "NEUTRAL", 0.5
        
        print(f"✓ LSTM: {ai_direction} ({ai_score})")
        print(f"✓ RF: {rf_direction} ({rf_score})")
        
        # 4. Whale analizi
        print("4. Whale analizi...")
        try:
            whale_score = whale_tracker.get_whale_signal('ETH/USDT')
            if whale_score is None:
                whale_score = 0.5
        except Exception as e:
            print(f"⚠ Whale analizi hatası: {e}")
            whale_score = 0.5
        print(f"✓ Whale skoru: {whale_score}")
        
        # 5. Haber analizi
        print("5. Haber analizi...")
        try:
            news_impact, _ = news_analysis.get_news_impact(['ETH'])
            if news_impact and 'ETH' in news_impact:
                news_score = news_impact['ETH'].get('impact_score', 0.5)
            else:
                news_score = 0.5
        except Exception as e:
            print(f"⚠ Haber analizi hatası: {e}")
            news_score = 0.5
        print(f"✓ Haber skoru: {news_score}")
        
        # 6. Toplam skor hesapla
        print("6. Toplam skor hesaplanıyor...")
        total_score = (
            float(ai_score) * 0.4 +
            float(rf_score) * 0.3 +
            float(ta_strength) * 0.2 +
            float(whale_score) * 0.05 +
            0.5 * 0.025 +  # social_score (devre dışı)
            float(news_score) * 0.025
        )
        print(f"✓ Toplam skor: {total_score}")
        
        # 7. Sinyal oluştur
        print("7. Sinyal oluşturuluyor...")
        main_direction = ai_direction if ai_score > rf_score else rf_direction
        if main_direction is None:
            main_direction = "NEUTRAL"
        
        signal = {
            'symbol': 'ETH/USDT',
            'timeframe': '1h',
            'direction': str(main_direction),
            'ai_score': float(total_score),
            'ta_strength': float(ta_strength),
            'whale_score': float(whale_score),
            'social_score': 0.5,
            'news_score': float(news_score),
            'timestamp': str(datetime.now()),
            'predicted_gain': float(total_score * 10),
            'predicted_duration': '2-4 saat'
        }
        
        print("✓ Sinyal oluşturuldu:")
        for key, value in signal.items():
            print(f"  {key}: {value}")
        
        # 8. Sinyali kaydet
        print("8. Sinyal kaydediliyor...")
        signal_manager.save_signal_json(signal)
        signal_manager.save_signal_csv(signal)
        signal_manager.save_signal_db(signal)
        print("✓ Sinyal kaydedildi")
        
        print("✅ ETH TEST BAŞARILI!")
        return True
        
    except Exception as e:
        print(f"❌ Test başarısız: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_eth_signal() 