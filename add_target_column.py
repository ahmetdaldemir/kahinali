#!/usr/bin/env python3
"""
Eğitim verisine target kolonu ekle
"""

import pandas as pd
import numpy as np

def add_target_column():
    """Eğitim verisine target kolonu ekle"""
    print("Eğitim verisine target kolonu ekleniyor...")
    
    # Veriyi yükle
    df = pd.read_csv('train_data_for_test.csv')
    print(f"Orijinal veri boyutu: {df.shape}")
    
    # Target kolonu oluştur (gelecek fiyat artışı)
    # 5 periyot sonraki fiyat artışı
    df['target'] = 0
    
    # Fiyat artışı hesapla
    for i in range(len(df) - 5):
        current_price = df.iloc[i]['close']
        future_price = df.iloc[i + 5]['close']
        
        # %2'den fazla artış varsa 1, yoksa 0
        if (future_price - current_price) / current_price > 0.02:
            df.iloc[i, df.columns.get_loc('target')] = 1
    
    print(f"Target kolonu eklendi. Yeni boyut: {df.shape}")
    print(f"Target dağılımı: {df['target'].value_counts()}")
    
    # Veriyi kaydet
    df.to_csv('train_data_with_target.csv', index=False)
    print("Veri kaydedildi: train_data_with_target.csv")
    
    return df

if __name__ == "__main__":
    add_target_column() 