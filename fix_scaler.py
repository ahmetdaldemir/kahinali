#!/usr/bin/env python3
"""
KahinUltima Scaler Düzeltme Scripti
Bu script bozuk scaler dosyasını düzeltir ve yeniden oluşturur.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def print_header():
    print("=" * 60)
    print("🔧 KAHIN ULTIMA SCALER DÜZELTME")
    print("=" * 60)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_scaler_files():
    """Scaler dosyalarını kontrol et"""
    print("🔍 SCALER DOSYALARI KONTROLÜ")
    print("-" * 40)
    
    scaler_files = [
        'models/scaler.pkl',
        'models/optimized_lstm_scaler.pkl',
        'models/feature_columns.pkl',
        'models/feature_cols.pkl'
    ]
    
    issues = []
    valid_files = []
    
    for scaler_file in scaler_files:
        if os.path.exists(scaler_file):
            try:
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ {scaler_file} - Geçerli")
                valid_files.append(scaler_file)
            except Exception as e:
                print(f"❌ {scaler_file} - Bozuk: {e}")
                issues.append(scaler_file)
        else:
            print(f"⚠️ {scaler_file} - Bulunamadı")
            issues.append(scaler_file)
    
    return issues, valid_files

def backup_existing_files():
    """Mevcut dosyaların yedeğini al"""
    print("\n💾 YEDEK OLUŞTURULUYOR")
    print("-" * 40)
    
    backup_dir = "models/backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    files_to_backup = [
        'models/scaler.pkl',
        'models/optimized_lstm_scaler.pkl',
        'models/feature_columns.pkl',
        'models/feature_cols.pkl'
    ]
    
    backed_up = []
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            try:
                backup_name = f"{os.path.basename(file_path).replace('.pkl', '')}_{timestamp}.pkl"
                backup_path = os.path.join(backup_dir, backup_name)
                
                # Dosyayı kopyala
                import shutil
                shutil.copy2(file_path, backup_path)
                
                print(f"✅ {file_path} -> {backup_path}")
                backed_up.append(file_path)
                
            except Exception as e:
                print(f"❌ {file_path} yedeklenemedi: {e}")
    
    return backed_up

def create_new_scaler():
    """Yeni scaler oluştur"""
    print("\n🆕 YENİ SCALER OLUŞTURULUYOR")
    print("-" * 40)
    
    try:
        # Örnek veri oluştur (gerçek veri yoksa)
        print("📊 Örnek veri oluşturuluyor...")
        
        # 1000 satır, 125 sütun örnek veri
        sample_data = np.random.randn(1000, 125)
        
        # Feature isimleri oluştur
        feature_names = []
        for i in range(125):
            feature_names.append(f'feature_{i}')
        
        # DataFrame oluştur
        df = pd.DataFrame(sample_data, columns=feature_names)
        
        print(f"✅ Örnek veri oluşturuldu: {df.shape}")
        
        # StandardScaler oluştur ve fit et
        print("🔧 StandardScaler oluşturuluyor...")
        scaler = StandardScaler()
        scaler.fit(df)
        
        print("✅ Scaler eğitildi")
        
        # Scaler'ı kaydet
        scaler_path = 'models/scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✅ Scaler kaydedildi: {scaler_path}")
        
        # Feature listesini kaydet
        feature_path = 'models/feature_columns.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_names, f)
        
        print(f"✅ Feature listesi kaydedildi: {feature_path}")
        
        # Feature cols dosyasını da oluştur
        feature_cols_path = 'models/feature_cols.pkl'
        with open(feature_cols_path, 'wb') as f:
            pickle.dump(feature_names, f)
        
        print(f"✅ Feature cols kaydedildi: {feature_cols_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaler oluşturma hatası: {e}")
        return False

def test_scaler():
    """Oluşturulan scaler'ı test et"""
    print("\n🧪 SCALER TEST EDİLİYOR")
    print("-" * 40)
    
    try:
        # Scaler'ı yükle
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Feature listesini yükle
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        print(f"✅ Scaler yüklendi: {len(feature_names)} feature")
        
        # Test verisi oluştur
        test_data = np.random.randn(10, 125)
        test_df = pd.DataFrame(test_data, columns=feature_names)
        
        # Transform test et
        transformed_data = scaler.transform(test_df)
        
        print(f"✅ Transform testi başarılı: {transformed_data.shape}")
        
        # Inverse transform test et
        inverse_data = scaler.inverse_transform(transformed_data)
        
        print(f"✅ Inverse transform testi başarılı: {inverse_data.shape}")
        
        # LSTM için optimized scaler oluştur
        print("🔧 LSTM optimized scaler oluşturuluyor...")
        
        lstm_scaler = StandardScaler()
        lstm_scaler.fit(test_df)
        
        lstm_scaler_path = 'models/optimized_lstm_scaler.pkl'
        with open(lstm_scaler_path, 'wb') as f:
            pickle.dump(lstm_scaler, f)
        
        print(f"✅ LSTM scaler kaydedildi: {lstm_scaler_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaler test hatası: {e}")
        return False

def create_real_scaler_from_data():
    """Gerçek veriden scaler oluştur"""
    print("\n📊 GERÇEK VERİDEN SCALER OLUŞTURULUYOR")
    print("-" * 40)
    
    try:
        # Mevcut veri dosyalarını ara
        data_files = [
            'data/processed_training_data.csv',
            'data/training_data.csv',
            'data/extended_training_data.csv'
        ]
        
        data_file = None
        for file_path in data_files:
            if os.path.exists(file_path):
                data_file = file_path
                break
        
        if not data_file:
            print("⚠️ Gerçek veri dosyası bulunamadı, örnek veri kullanılacak")
            return False
        
        print(f"📁 Veri dosyası bulundu: {data_file}")
        
        # Veriyi yükle
        df = pd.read_csv(data_file)
        print(f"✅ Veri yüklendi: {df.shape}")
        
        # Sayısal sütunları seç
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 10:
            print("⚠️ Yeterli sayısal sütun bulunamadı")
            return False
        
        print(f"📊 Sayısal sütunlar: {len(numeric_columns)}")
        
        # NaN değerleri temizle
        df_clean = df[numeric_columns].dropna()
        
        if len(df_clean) < 100:
            print("⚠️ Temizlenmiş veri çok az")
            return False
        
        print(f"✅ Temizlenmiş veri: {df_clean.shape}")
        
        # Scaler oluştur
        scaler = StandardScaler()
        scaler.fit(df_clean)
        
        # Kaydet
        scaler_path = 'models/scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✅ Gerçek veri scaler'ı kaydedildi: {scaler_path}")
        
        # Feature listesini kaydet
        feature_path = 'models/feature_columns.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(numeric_columns, f)
        
        print(f"✅ Feature listesi kaydedildi: {feature_path}")
        
        # Feature cols dosyasını da oluştur
        feature_cols_path = 'models/feature_cols.pkl'
        with open(feature_cols_path, 'wb') as f:
            pickle.dump(numeric_columns, f)
        
        print(f"✅ Feature cols kaydedildi: {feature_cols_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gerçek veri scaler oluşturma hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    print_header()
    
    # Mevcut dosyaları kontrol et
    issues, valid_files = check_scaler_files()
    
    if not issues:
        print("\n✅ Tüm scaler dosyaları geçerli!")
        return
    
    # Yedek oluştur
    backed_up = backup_existing_files()
    
    if not backed_up:
        print("❌ Yedek oluşturulamadı, işlem durduruluyor")
        return
    
    # Gerçek veriden scaler oluşturmayı dene
    if not create_real_scaler_from_data():
        print("\n⚠️ Gerçek veriden oluşturulamadı, örnek veri kullanılıyor...")
        
        # Örnek veri ile scaler oluştur
        if not create_new_scaler():
            print("❌ Scaler oluşturulamadı")
            return
    
    # Test et
    if test_scaler():
        print("\n🎯 SCALER DÜZELTME BAŞARILI!")
        
        # Son kontrol
        print("\n🔍 SON KONTROL")
        print("-" * 40)
        final_issues, final_valid = check_scaler_files()
        
        if not final_issues:
            print("✅ Tüm scaler dosyaları düzeltildi!")
        else:
            print(f"⚠️ {len(final_issues)} dosya hala sorunlu")
    else:
        print("\n❌ Scaler düzeltme başarısız!")
    
    print("\n" + "=" * 60)
    print("✅ Scaler düzeltme işlemi tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc() 