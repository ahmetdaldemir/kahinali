#!/usr/bin/env python3
import pandas as pd
from sqlalchemy import create_engine
from config import Config

def check_ar_signal():
    try:
        engine = create_engine(Config.DATABASE_URL)
        query = "SELECT * FROM signals WHERE symbol = 'AR/USDT' ORDER BY timestamp DESC LIMIT 1"
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            signal = df.iloc[0]
            print("=== AR/USDT Sinyal Analizi ===")
            print(f"Sembol: {signal['symbol']}")
            print(f"Zaman Dilimi: {signal['timeframe']}")
            print(f"Yön: {signal['direction']}")
            print(f"AI Skoru: {signal['ai_score']:.3f} ({signal['ai_score']*100:.1f}%)")
            print(f"TA Gücü: {signal['ta_strength']:.3f} ({signal['ta_strength']*100:.1f}%)")
            print(f"Whale Skoru: {signal['whale_score']:.3f} ({signal['whale_score']*100:.1f}%)")
            print(f"Sosyal Skor: {signal['social_score']:.3f} ({signal['social_score']*100:.1f}%)")
            print(f"Haber Skoru: {signal['news_score']:.3f}")
            print(f"Giriş Fiyatı: {signal['entry_price']}")
            print(f"Tahmini Kazanç: {signal['predicted_gain']:.2f}%")
            print(f"Tahmini Süre: {signal['predicted_duration']} saat")
            print(f"Zaman: {signal['timestamp']}")
            print(f"Sonuç: {signal['result']}")
            
            # Gelişmiş metrikler
            print("\n=== Gelişmiş Metrikler ===")
            print(f"Volatilite Skoru: {signal.get('volatility_score', 'N/A')}")
            print(f"Trend Gücü: {signal.get('trend_strength', 'N/A')}")
            print(f"Market Rejimi: {signal.get('market_regime', 'N/A')}")
            print(f"Sinyal Kalite Skoru: {signal.get('signal_quality_score', 'N/A')}")
            print(f"Risk/Ödül Oranı: {signal.get('risk_reward_ratio', 'N/A')}")
            
            # Sinyal üretim kriterleri analizi
            print("\n=== Sinyal Üretim Kriterleri Analizi ===")
            ai_score = signal['ai_score']
            ta_strength = signal['ta_strength']
            whale_score = signal['whale_score']
            social_score = signal['social_score']
            news_score = signal['news_score']
            
            print(f"AI Skoru (0.6+ gerekli): {'✅' if ai_score >= 0.6 else '❌'} {ai_score:.3f}")
            print(f"TA Gücü (0.5+ gerekli): {'✅' if ta_strength >= 0.5 else '❌'} {ta_strength:.3f}")
            print(f"Whale Skoru (0.5+ gerekli): {'✅' if whale_score >= 0.5 else '❌'} {whale_score:.3f}")
            print(f"Sosyal Skor (0.5+ gerekli): {'✅' if social_score >= 0.5 else '❌'} {social_score:.3f}")
            print(f"Haber Skoru (5+ gerekli): {'✅' if news_score >= 5 else '❌'} {news_score:.3f}")
            
            # Toplam skor hesaplama
            total_score = (ai_score * 0.35 + ta_strength * 0.25 + whale_score * 0.20 + 
                          social_score * 0.12 + min(news_score/10, 1.0) * 0.08)
            print(f"\nToplam Ağırlıklı Skor: {total_score:.3f} ({total_score*100:.1f}%)")
            print(f"Sinyal Üretildi mi: {'✅ EVET' if total_score >= 0.6 else '❌ HAYIR'}")
            
            # Neden sinyal üretildi/üretilmedi
            print("\n=== Sinyal Üretim Nedeni ===")
            if total_score >= 0.6:
                print("✅ Yeterli toplam skor nedeniyle sinyal üretildi")
                if news_score >= 5:
                    print("✅ Yüksek haber skoru (5+) sinyal üretimini destekledi")
                if ta_strength >= 0.7:
                    print("✅ Güçlü teknik analiz (0.7+) sinyal güvenilirliğini artırdı")
            else:
                print("❌ Toplam skor yetersiz (0.6 altında)")
                if ai_score < 0.6:
                    print("❌ AI skoru düşük")
                if ta_strength < 0.5:
                    print("❌ Teknik analiz gücü düşük")
                if whale_score < 0.5:
                    print("❌ Whale skoru düşük")
            
        else:
            print("AR/USDT sinyali bulunamadı")
            
        # Tüm sinyallerin entry_price değerlerini kontrol et
        print("\n=== Entry Price Kontrolü ===")
        query2 = "SELECT symbol, entry_price, timestamp FROM signals ORDER BY timestamp DESC LIMIT 10"
        df2 = pd.read_sql(query2, engine)
        print(df2.to_string())
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    check_ar_signal() 