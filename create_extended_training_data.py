#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KahinUltima - Genişletilmiş Eğitim Verisi Oluşturucu
Geçmiş verileri toplayarak AI modelleri için eğitim verisi hazırlar
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from config import Config

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_extended_training_data():
    """Genişletilmiş eğitim verisi oluştur"""
    
    print("=== GENİŞLETİLMİŞ EĞİTİM VERİSİ OLUŞTURUCU ===")
    
    try:
        # Modülleri başlat
        data_collector = DataCollector()
        technical_analyzer = TechnicalAnalysis()
        ai_model = AIModel()
        
        # Popüler coinler
        popular_coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        timeframes = ['1h', '4h', '1d']
        
        all_data = []
        
        for coin in popular_coins:
            print(f"\n--- {coin} için veri toplanıyor ---")
            
            for timeframe in timeframes:
                try:
                    print(f"  {timeframe} timeframe için veri alınıyor...")
                    
                    # Daha fazla geçmiş veri al (1000 satır)
                    data = data_collector.get_historical_data(
                        symbol=coin, 
                        timeframe=timeframe, 
                        limit=1000
                    )
                    
                    if data is not None and not data.empty:
                        print(f"    ✓ {len(data)} satır veri alındı")
                        
                        # Teknik analiz uygula
                        data_with_ta = technical_analyzer.calculate_all_indicators(data)
                        
                        # Timestamp sütununu tekrar ekle (eğer kaybolmuşsa)
                        if 'timestamp' not in data_with_ta.columns and 'timestamp' in data.columns:
                            data_with_ta['timestamp'] = data['timestamp']
                        
                        # NaN değerleri temizle
                        data_with_ta = data_with_ta.dropna()
                        
                        # Timeframe bilgisi ekle
                        data_with_ta['timeframe'] = timeframe
                        data_with_ta['symbol'] = coin
                        
                        all_data.append(data_with_ta)
                        
                        # Dosyaya kaydet
                        filename = f"data/{coin.replace('/', '_')}_{timeframe}_extended.csv"
                        data_with_ta.to_csv(filename)
                        print(f"    ✓ Veri kaydedildi: {filename}")
                        
                    else:
                        print(f"    ✗ {timeframe} için veri alınamadı")
                        
                except Exception as e:
                    print(f"    ✗ {timeframe} hatası: {e}")
                    continue
        
        # Tüm verileri birleştir
        print("\n--- Tüm veriler birleştiriliyor ---")
        all_data = []
        
        for coin in popular_coins:
            for timeframe in timeframes:
                filename = f"data/{coin.replace('/', '_')}_{timeframe}_extended.csv"
                if os.path.exists(filename):
                    try:
                        df = pd.read_csv(filename)
                        # Timestamp sütununu datetime'a çevir
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                        
                        # Coin ve timeframe bilgisi ekle
                        df['symbol'] = coin
                        df['timeframe'] = timeframe
                        
                        all_data.append(df)
                        print(f"    ✓ {coin} {timeframe} verisi eklendi")
                    except Exception as e:
                        print(f"    ✗ {coin} {timeframe} verisi eklenirken hata: {e}")
        
        if all_data:
            # Tüm verileri birleştir
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()
            
            # NaN değerleri temizle
            combined_data = combined_data.dropna()
            
            # Kaydet
            output_file = "data/extended_training_data.csv"
            combined_data.to_csv(output_file)
            
            print(f"\n✅ Genişletilmiş eğitim verisi oluşturuldu!")
            print(f"📊 Toplam {len(combined_data)} satır veri")
            print(f"📁 Dosya: {output_file}")
            
            # AI model için hazırla
            print(f"\n--- AI Model için veri hazırlanıyor ---")
            
            # Etiketler oluştur
            labeled_data = ai_model.create_labels(combined_data)
            
            if labeled_data is not None and not labeled_data.empty:
                # Feature engineering
                engineered_data = ai_model.engineer_features(labeled_data)
                
                if engineered_data is not None and not engineered_data.empty:
                    # Eğitim verisi olarak kaydet
                    engineered_data.to_csv('data/processed_training_data.csv', index=False)
                    print("✓ İşlenmiş eğitim verisi kaydedildi: data/processed_training_data.csv")
                    
                    # Model eğitimi başlat
                    print(f"\n--- Model Eğitimi Başlatılıyor ---")
                    
                    try:
                        # LSTM eğitimi
                        print("LSTM model eğitiliyor...")
                        ai_model.train_lstm(engineered_data, epochs=10)
                        
                        # Random Forest eğitimi
                        print("Random Forest model eğitiliyor...")
                        ai_model.train_rf(engineered_data)
                        
                        # Gradient Boosting eğitimi
                        print("Gradient Boosting model eğitiliyor...")
                        ai_model.train_gradient_boosting(engineered_data)
                        
                        # Modelleri kaydet
                        ai_model.save_models()
                        print("✓ Tüm modeller eğitildi ve kaydedildi")
                        
                    except Exception as e:
                        print(f"✗ Model eğitimi hatası: {e}")
                else:
                    print("✗ Feature engineering başarısız")
            else:
                print("✗ Etiket oluşturma başarısız")
            
            return combined_data
            
        else:
            print("❌ Birleştirilecek veri bulunamadı!")
            return None
            
    except Exception as e:
        print(f"✗ Genel hata: {e}")
        return None

if __name__ == "__main__":
    print("KahinUltima - Genişletilmiş Eğitim Verisi Oluşturucu")
    print("=" * 60)
    
    success = create_extended_training_data()
    
    if success is not None:
        print("\n" + "=" * 60)
        print("✓ GENİŞLETİLMİŞ EĞİTİM VERİSİ BAŞARIYLA OLUŞTURULDU!")
        print("✓ AI modelleri eğitildi ve kaydedildi")
        print("✓ Sistem artık daha fazla geçmiş veri ile çalışabilir")
    else:
        print("\n" + "=" * 60)
        print("✗ EĞİTİM VERİSİ OLUŞTURMA BAŞARISIZ!")
        print("✗ Lütfen hataları kontrol edin ve tekrar deneyin") 