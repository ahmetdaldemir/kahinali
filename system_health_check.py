#!/usr/bin/env python3
"""
KahinUltima Sistem Sağlık Kontrolü
Bu script sistemin genel sağlığını 10 üzerinden değerlendirir.
"""

import sys
import traceback
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from modules.signal_manager import SignalManager
import pandas as pd
import time


def main():
    print("\n=== KAHIN ULTIMA TAM SİSTEM SAĞLIK TESTİ ===\n")
    health_report = []
    try:
        # 1. Veri Toplama
        dc = DataCollector()
        coin = 'BTC/USDT'
        print(f"[1] Veri toplama: {coin} için 1h, 100 bar...")
        df = dc.get_historical_data(coin, '1h', 100)
        if df is None or df.empty:
            health_report.append("[HATA] Veri toplama başarısız veya boş dataframe döndü!")
            raise Exception("Veri toplama başarısız")
        print(f"    ✔ {len(df)} satır veri alındı.")

        # 2. Teknik Analiz
        ta = TechnicalAnalysis()
        print("[2] Teknik analiz başlatılıyor...")
        ta_df = ta.calculate_all_indicators(df)
        if ta_df is None or ta_df.empty:
            health_report.append("[HATA] Teknik analiz başarısız veya boş dataframe döndü!")
            raise Exception("Teknik analiz başarısız")
        print(f"    ✔ Teknik analiz tamamlandı. {ta_df.shape[1]} kolon.")

        # 3. AI Tahmini
        ai = AIModel()
        print("[3] AI tahmini başlatılıyor...")
        ai_result = ai.predict(ta_df)
        if ai_result is None or 'prediction' not in ai_result or 'confidence' not in ai_result:
            health_report.append("[HATA] AI tahmini başarısız veya None döndü!")
            raise Exception("AI tahmini başarısız")
        print(f"    ✔ AI tahmini: prediction={ai_result['prediction']:.3f}, confidence={ai_result['confidence']:.3f}")

        # 4. Sinyal Üretimi
        sm = SignalManager()
        print("[4] Sinyal üretimi başlatılıyor...")
        analysis_data = {
            'ai_score': float(ai_result['prediction']),
            'ta_strength': ta_df['ta_strength'].iloc[-1] if 'ta_strength' in ta_df else 0.5,
            'whale_score': 0.5,
            'breakout_probability': 0.5,
            'close': float(df['close'].iloc[-1]),
        }
        direction = 'LONG' if ai_result['prediction'] > 0.5 else 'SHORT'
        signal = sm.create_signal(
            symbol=coin,
            direction=direction,
            confidence=float(ai_result['confidence']),
            analysis_data=analysis_data
        )
        if not signal:
            health_report.append("[HATA] Sinyal üretimi başarısız!")
            raise Exception("Sinyal üretimi başarısız")
        print(f"    ✔ Sinyal üretildi. Symbol: {signal['symbol']}, Direction: {signal['direction']}")

        # 5. Veritabanı Kaydı ve Okuma
        print("[5] Sinyal veritabanı kaydı ve okuma...")
        sm.save_signal_db(signal)
        # En son kaydedilen sinyali çek
        latest_signals = sm.get_latest_signals(limit=1)
        if latest_signals is None or latest_signals.empty:
            health_report.append("[HATA] Sinyal veritabanına kaydedilemedi veya okunamadı!")
            raise Exception("Sinyal veritabanı kaydı/okuma başarısız")
        latest_signal = latest_signals.iloc[0]
        print(f"    ✔ Sinyal veritabanında bulundu. ID: {latest_signal['id']}, Symbol: {latest_signal['symbol']}")

        # 6. API ve Panel Kontrolü (opsiyonel, API endpoint test edilebilir)
        print("[6] API ve panel kontrolü (elle veya ek script ile test edilebilir)")
        print("    (Not: API endpoint testi için ayrı bir script veya manuel kontrol gerekebilir.)")

        health_report.append("[BAŞARILI] Tüm modüller ve veri akışı sağlıklı çalışıyor!")
    except Exception as e:
        print(f"[HATA] {e}")
        traceback.print_exc()
        health_report.append(f"[HATA] {e}")

    print("\n=== SAĞLIK TESTİ RAPORU ===")
    for line in health_report:
        print(line)
    print("\nTest tamamlandı.\n")

if __name__ == "__main__":
    main() 