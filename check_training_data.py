#!/usr/bin/env python3
"""
Eğitim verisini kontrol et
"""

import pandas as pd
import os

def check_training_data():
    """Eğitim verisini kontrol et"""
    try:
        # Veri dosyasını kontrol et
        data_path = 'data/processed_training_data.csv'
        if not os.path.exists(data_path):
            print("❌ Eğitim verisi bulunamadı!")
            return False
            
        # Veriyi yükle
        df = pd.read_csv(data_path)
        print(f"✅ Veri boyutu: {df.shape}")
        
        # Label dağılımını kontrol et
        if 'label_dynamic' in df.columns:
            label_counts = df['label_dynamic'].value_counts().to_dict()
            print(f"✅ Label dağılımı: {label_counts}")
        else:
            print("❌ label_dynamic sütunu bulunamadı!")
            print(f"Mevcut sütunlar: {list(df.columns)}")
            
        # Feature sayısını kontrol et
        feature_cols = [col for col in df.columns if col not in ['label_dynamic', 'label_5', 'label_10', 'label_20']]
        print(f"✅ Feature sayısı: {len(feature_cols)}")
        
        # NaN değerleri kontrol et
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️ NaN değerler: {nan_counts[nan_counts > 0].to_dict()}")
        else:
            print("✅ NaN değer yok")
            
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

if __name__ == "__main__":
    check_training_data() 