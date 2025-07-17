#!/usr/bin/env python3
"""
KAHİN ULTIMA - Sinyal Üretim Zinciri Testi
Veri → Teknik Analiz → AI → Sinyal Oluşturma → Filtreleme adımlarını adım adım test eder
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from modules.signal_manager import SignalManager


def test_signal_production(symbol='BTC/USDT', timeframe='1h', limit=200):
    print(f"\n🚦 SİNYAL ÜRETİM ZİNCİRİ TESTİ: {symbol} [{timeframe}]")
    print("=" * 60)
    collector = DataCollector()
    ta = TechnicalAnalysis()
    ai_model = AIModel()
    signal_manager = SignalManager()

    # 1. Veri Toplama
    print("1. Veri çekiliyor...")
    df = collector.get_historical_data(symbol, timeframe, limit)
    if df is None or df.empty:
        print("❌ Veri alınamadı!")
        return
    print(f"✓ {len(df)} satır veri alındı. Son bar: {df.index[-1]}")

    # 2. Teknik Analiz
    print("2. Teknik analiz hesaplanıyor...")
    df_ta = ta.calculate_all_indicators(df)
    if df_ta is None or df_ta.empty:
        print("❌ Teknik analiz başarısız!")
        return
    print(f"✓ Teknik analiz tamamlandı. Kolon sayısı: {df_ta.shape[1]}")

    # 3. AI Skorları
    print("3. AI tahmini yapılıyor...")
    try:
        features = ai_model.prepare_features(df_ta)
        print(f"- Hazırlanan feature shape: {features.shape}")
        print(f"- Feature isimleri: {list(features.columns)}")
        # Eksik/NaN/inf kontrolü
        nan_count = features.isna().sum().sum()
        inf_count = np.isinf(features.values).sum()
        print(f"- NaN sayısı: {nan_count}, inf sayısı: {inf_count}")
        if nan_count > 0 or inf_count > 0:
            print("❌ Feature'larda NaN veya inf var! İlk 5 satır:")
            print(features.head())
            return
        # Scaler ve model feature uyumu
        expected_features = getattr(ai_model, 'feature_columns', None)
        if expected_features is not None:
            print(f"- Modelin beklediği feature sayısı: {len(expected_features)}")
            # Dataframe'i modelin beklediği sıraya ve isimlere göre reindex et
            features = features.reindex(columns=expected_features, fill_value=0)
            if list(features.columns) != list(expected_features):
                print("❌ Feature isimleri modelle uyumsuz (reindex sonrası bile)! Lütfen model ve feature listesi uyumunu kontrol edin.")
                print(f"Beklenen: {expected_features}")
                print(f"Gelen: {list(features.columns)}")
                return
        ai_preds = ai_model.predict(features)
        print(f"- ai_preds tipi: {type(ai_preds)}")
        print(f"- ai_preds içeriği (ilk 5): {str(ai_preds)[:300]}")
        if ai_preds is None:
            print("❌ AI tahmini başarısız! (None)")
            return
        if isinstance(ai_preds, (int, float)) and ai_preds == -1:
            print("❌ AI modeli -1 hata kodu döndürdü! Predict fonksiyonunda hata var.")
            return
        ai_score = None
        if isinstance(ai_preds, dict) and 'prediction' in ai_preds:
            ai_score = ai_preds['prediction']
            print(f"✓ AI tahmini tamamlandı. Tahmin: {ai_score:.4f}")
        elif hasattr(ai_preds, '__len__') and len(ai_preds) > 0:
            ai_score = ai_preds[-1]
            print(f"✓ AI tahmini tamamlandı. Son skor: {ai_score:.4f}")
        else:
            print(f"❌ AI tahmini başarısız! (Boş çıktı)")
            return
    except Exception as e:
        print(f"❌ AI tahmini hatası: {e}")
        import traceback
        print(traceback.format_exc())
        return

    # 4. Sinyal Oluşturma
    print("4. Sinyal oluşturuluyor...")
    analysis_data = df_ta.iloc[-1].to_dict()
    analysis_data['ai_score'] = ai_score
    analysis_data['ta_strength'] = ta.calculate_signal_strength(df_ta)
    analysis_data['whale_score'] = 0.5  # Dummy
    analysis_data['confidence'] = 0.5   # Dummy
    analysis_data['trend_alignment'] = ta.get_trend_direction(df_ta)
    analysis_data['volume_score'] = analysis_data.get('volume', 0) / 1000
    analysis_data['pattern_score'] = 0.5
    analysis_data['momentum_score'] = ta.calculate_momentum_score(df_ta)
    analysis_data['volatility'] = analysis_data.get('atr', 0)

    signal = signal_manager.create_signal(symbol, 'LONG', 0.5, analysis_data)
    if not signal:
        print("❌ Sinyal oluşturulamadı!")
        return
    print(f"✓ Sinyal oluşturuldu. AI: {signal['ai_score']:.3f}, TA: {signal['ta_strength']:.3f}")

    # 5. Filtreleme
    print("5. Sinyal filtrelemesi yapılıyor...")
    filtered = signal_manager.filter_signals([signal], min_confidence=0.45)
    if filtered:
        print(f"✓ Sinyal filtreyi geçti! Fırsat skoru: {filtered[0].get('opportunity_score', 'N/A')}")
    else:
        print("⚠ Sinyal filtreyi geçemedi.")

    print("\nTest tamamlandı.")

if __name__ == "__main__":
    test_signal_production() 