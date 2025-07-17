#!/usr/bin/env python3
"""
Sonsuz değerleri temizle
"""

import pandas as pd
import numpy as np
import os

def fix_infinity_values():
    """Sonsuz değerleri temizle"""
    print("🔧 Sonsuz değerler temizleniyor...")
    
    try:
        # Veriyi yükle
        df = pd.read_csv('data/processed_training_data.csv')
        print(f"✅ Veri yüklendi: {df.shape}")
        
        # Sonsuz değerleri kontrol et
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        print(f"⚠️ Sonsuz değerler: {inf_counts[inf_counts > 0].to_dict()}")
        
        # Sonsuz değerleri NaN ile değiştir
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # NaN değerleri temizle
        df = df.dropna()
        print(f"✅ Temizleme sonrası: {df.shape}")
        
        # Veriyi kaydet
        df.to_csv('data/processed_training_data_clean.csv', index=False)
        print("✅ Temizlenmiş veri kaydedildi: data/processed_training_data_clean.csv")
        
        # Orijinal dosyayı yedekle ve yenisini kopyala
        if os.path.exists('data/processed_training_data.csv'):
            os.rename('data/processed_training_data.csv', 'data/processed_training_data_backup.csv')
            print("✅ Orijinal dosya yedeklendi")
        
        os.rename('data/processed_training_data_clean.csv', 'data/processed_training_data.csv')
        print("✅ Temizlenmiş veri ana dosya olarak ayarlandı")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

if __name__ == "__main__":
    fix_infinity_values() 