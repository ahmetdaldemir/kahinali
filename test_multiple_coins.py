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

def test_multiple_coins():
    coins = ['ADA/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'SOL/USDT']
    
    collector = DataCollector()
    ta = TechnicalAnalysis()
    ai_model = AIModel()
    whale_tracker = WhaleTracker()
    news_analysis = NewsAnalysis()
    signal_manager = SignalManager()
    
    successful_signals = 0
    
    for coin in coins:
        print(f"\n{'='*50}")
        print(f"Testing {coin}")
        print(f"{'='*50}")
        
        try:
            # 1. Veri topla
            print(f"1. {coin} için veri toplanıyor...")
            df = collector.get_historical_data(coin, '1h', limit=1000)
            if df.empty:
                print(f"❌ {coin} için veri alınamadı")
                continue
            print(f"✓ {len(df)} satır veri alındı")
            
            # 2. Teknik analiz
            print(f"2. {coin} için teknik analiz...")
            df = ta.calculate_all_indicators(df)
            if df.empty:
                print(f"❌ {coin} teknik analiz başarısız")
                continue
            
            ta_signals = ta.generate_signals(df)
            ta_strength = ta.calculate_signal_strength(ta_signals)
            if ta_strength is None:
                ta_strength = 0.5
            print(f"✓ Teknik analiz gücü: {ta_strength}")
            
            # 3. AI tahminleri (hata durumunda varsayılan değerler)
            print(f"3. {coin} için AI tahminleri...")
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
            print(f"4. {coin} için whale analizi...")
            try:
                whale_score = whale_tracker.get_whale_signal(coin)
                if whale_score is None:
                    whale_score = 0.5
            except Exception as e:
                print(f"⚠ Whale analizi hatası: {e}")
                whale_score = 0.5
            print(f"✓ Whale skoru: {whale_score}")
            
            # 5. Haber analizi
            print(f"5. {coin} için haber analizi...")
            coin_symbol = coin.split('/')[0]
            try:
                news_impact, _ = news_analysis.get_news_impact([coin_symbol])
                if news_impact and coin_symbol in news_impact:
                    news_score = news_impact[coin_symbol].get('impact_score', 0.5)
                else:
                    news_score = 0.5
            except Exception as e:
                print(f"⚠ Haber analizi hatası: {e}")
                news_score = 0.5
            print(f"✓ Haber skoru: {news_score}")
            
            # 6. Toplam skor hesapla
            print(f"6. {coin} için toplam skor...")
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
            print(f"7. {coin} için sinyal oluşturuluyor...")
            main_direction = ai_direction if ai_score > rf_score else rf_direction
            if main_direction is None:
                main_direction = "NEUTRAL"
            
            signal = {
                'symbol': coin,
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
            
            # 8. Sinyali kaydet
            print(f"8. {coin} sinyali kaydediliyor...")
            signal_manager.save_signal_json(signal)
            signal_manager.save_signal_csv(signal)
            signal_manager.save_signal_db(signal)
            print(f"✓ {coin} sinyali kaydedildi")
            
            successful_signals += 1
            print(f"✅ {coin} TEST BAŞARILI!")
            
        except Exception as e:
            print(f"❌ {coin} test başarısız: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*50}")
    print(f"TEST SONUÇLARI")
    print(f"{'='*50}")
    print(f"Toplam test edilen coin: {len(coins)}")
    print(f"Başarılı sinyal: {successful_signals}")
    print(f"Başarı oranı: {(successful_signals/len(coins)*100):.1f}%")

if __name__ == "__main__":
    test_multiple_coins() 