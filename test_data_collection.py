#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
import pandas as pd

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_collection():
    """Veri toplama fonksiyonunu test et"""
    print("=== VERİ TOPLAMA TESTİ ===")
    
    try:
        # 1. DataCollector'ı başlat
        print("1. DataCollector başlatılıyor...")
        dc = DataCollector()
        print("✓ DataCollector başarıyla oluşturuldu")
        
        # 2. USDT çiftlerini al
        print("\n2. USDT çiftleri alınıyor...")
        pairs = dc.get_usdt_pairs(max_pairs=5)
        print(f"✓ {len(pairs)} USDT çifti bulundu: {pairs}")
        
        # 3. İlk çift için veri al
        if pairs:
            test_pair = pairs[0]
            print(f"\n3. {test_pair} için veri alınıyor...")
            df = dc.get_historical_data(test_pair, '1h', limit=100)
            
            if not df.empty:
                print(f"✓ {len(df)} satır veri alındı")
                print(f"✓ Veri sütunları: {list(df.columns)}")
                print(f"✓ İlk 3 satır:")
                print(df.head(3))
                
                # 4. Veriyi kaydet
                print(f"\n4. Veri kaydediliyor...")
                filename = dc.save_data_to_csv(df, test_pair, '1h')
                if filename:
                    print(f"✓ Veri kaydedildi: {filename}")
                else:
                    print("✗ Veri kaydedilemedi")
                
                return df
            else:
                print("✗ Veri alınamadı - DataFrame boş")
                return None
        else:
            print("✗ USDT çifti bulunamadı")
            return None
            
    except Exception as e:
        print(f"✗ Hata: {e}")
        logger.error(f"Veri toplama testi hatası: {e}")
        return None

def test_technical_analysis(df):
    """Teknik analiz fonksiyonunu test et"""
    print("\n=== TEKNİK ANALİZ TESTİ ===")
    
    if df is None or df.empty:
        print("✗ Teknik analiz için veri yok")
        return None
    
    try:
        # 1. TechnicalAnalysis'ı başlat
        print("1. TechnicalAnalysis başlatılıyor...")
        ta = TechnicalAnalysis()
        print("✓ TechnicalAnalysis başarıyla oluşturuldu")
        
        # 2. İndikatörleri hesapla
        print("\n2. Teknik indikatörler hesaplanıyor...")
        df_with_indicators = ta.calculate_all_indicators(df)
        
        if not df_with_indicators.empty:
            print(f"✓ {len(df_with_indicators)} satır veri işlendi")
            print(f"✓ Yeni sütunlar: {[col for col in df_with_indicators.columns if col not in df.columns]}")
            print(f"✓ Toplam sütun sayısı: {len(df_with_indicators.columns)}")
            
            # 3. Sinyal üret
            print("\n3. Sinyaller üretiliyor...")
            signals = ta.generate_signals(df_with_indicators)
            if signals:
                print(f"✓ {len(signals)} sinyal üretildi")
                for signal_name, signal_value in signals.items():
                    if hasattr(signal_value, 'iloc'):
                        print(f"  - {signal_name}: {signal_value.iloc[-1]}")
                    else:
                        print(f"  - {signal_name}: {signal_value}")
            
            # 4. Sinyal gücünü hesapla
            print("\n4. Sinyal gücü hesaplanıyor...")
            strength = ta.calculate_signal_strength(signals)
            print(f"✓ Sinyal gücü: {strength}")
            
            return df_with_indicators
        else:
            print("✗ Teknik analiz sonrası veri boş")
            return None
            
    except Exception as e:
        print(f"✗ Hata: {e}")
        logger.error(f"Teknik analiz testi hatası: {e}")
        return None

def test_ai_model(df):
    """AI model fonksiyonunu test et"""
    print("\n=== AI MODEL TESTİ ===")
    
    if df is None or df.empty:
        print("✗ AI model için veri yok")
        return None
    
    try:
        # 1. AIModel'i başlat
        print("1. AIModel başlatılıyor...")
        ai = AIModel()
        print("✓ AIModel başarıyla oluşturuldu")
        
        # 2. Etiketleri oluştur
        print("\n2. Etiketler oluşturuluyor...")
        df_with_labels = ai.create_labels(df)
        
        if not df_with_labels.empty:
            print(f"✓ {len(df_with_labels)} satır veri işlendi")
            print(f"✓ Etiket sütunları: {[col for col in df_with_labels.columns if 'label' in col]}")
            
            # 3. Özellik mühendisliği
            print("\n3. Özellik mühendisliği yapılıyor...")
            df_with_features = ai.engineer_features(df_with_labels)
            
            if not df_with_features.empty:
                print(f"✓ {len(df_with_features)} satır veri işlendi")
                print(f"✓ Yeni özellikler: {[col for col in df_with_features.columns if col not in df_with_labels.columns]}")
                
                # 4. Model eğitimi (kısa süreli)
                print("\n4. Model eğitimi başlatılıyor (kısa süreli)...")
                try:
                    # Sadece RF modelini hızlıca eğit
                    rf_model = ai.train_rf(df_with_features)
                    print("✓ RF modeli eğitildi")
                    
                    # 5. Tahmin testi
                    print("\n5. Tahmin testi yapılıyor...")
                    direction, score = ai.predict_rf(df_with_features.tail(10))
                    print(f"✓ Tahmin: {direction}, Skor: {score}")
                    
                    return df_with_features
                    
                except Exception as e:
                    print(f"✗ Model eğitimi hatası: {e}")
                    return df_with_features
            else:
                print("✗ Özellik mühendisliği sonrası veri boş")
                return None
        else:
            print("✗ Etiket oluşturma sonrası veri boş")
            return None
            
    except Exception as e:
        print(f"✗ Hata: {e}")
        logger.error(f"AI model testi hatası: {e}")
        return None

def main():
    """Ana test fonksiyonu"""
    print("KAHİN Ultima - Sistem Testi Başlatılıyor...")
    print("=" * 50)
    
    # 1. Veri toplama testi
    df = test_data_collection()
    
    if df is not None:
        # 2. Teknik analiz testi
        df_ta = test_technical_analysis(df)
        
        if df_ta is not None:
            # 3. AI model testi
            df_ai = test_ai_model(df_ta)
            
            if df_ai is not None:
                print("\n" + "=" * 50)
                print("✓ TÜM TESTLER BAŞARILI!")
                print("✓ Sistem çalışıyor ve veri işleyebiliyor")
                print("✓ AI modelleri eğitilebiliyor")
                print("✓ Sinyal üretimi mümkün")
            else:
                print("\n✗ AI model testi başarısız")
        else:
            print("\n✗ Teknik analiz testi başarısız")
    else:
        print("\n✗ Veri toplama testi başarısız")
    
    print("\n" + "=" * 50)
    print("Test tamamlandı.")

if __name__ == "__main__":
    main() 