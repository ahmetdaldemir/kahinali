#!/usr/bin/env python3
"""
KahinUltima LSTM Shape Uyumsuzluğu Düzeltme Scripti
Bu script LSTM modelindeki shape uyumsuzluklarını düzeltir.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

def print_header():
    print("=" * 60)
    print("🔧 KAHIN ULTIMA LSTM SHAPE DÜZELTME")
    print("=" * 60)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_lstm_model():
    """LSTM modelini kontrol et"""
    print("🔍 LSTM MODEL KONTROLÜ")
    print("-" * 40)
    
    model_files = [
        'models/lstm_model.h5',
        'models/optimized_lstm_model.h5'
    ]
    
    issues = []
    valid_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = keras.models.load_model(model_file)
                input_shape = model.input_shape
                output_shape = model.output_shape
                
                print(f"✅ {model_file}")
                print(f"   Input shape: {input_shape}")
                print(f"   Output shape: {output_shape}")
                
                # Shape kontrolü
                if len(input_shape) == 3 and input_shape[2] == 125:
                    print(f"   ✅ Input shape uygun")
                    valid_models.append(model_file)
                else:
                    print(f"   ⚠️ Input shape uyumsuz: {input_shape[2]} != 125")
                    issues.append(f"{model_file}: Input shape {input_shape[2]} != 125")
                
            except Exception as e:
                print(f"❌ {model_file} - Hatalı: {e}")
                issues.append(f"{model_file}: {e}")
        else:
            print(f"⚠️ {model_file} - Bulunamadı")
            issues.append(f"{model_file}: Bulunamadı")
    
    return issues, valid_models

def check_feature_consistency():
    """Feature tutarlılığını kontrol et"""
    print("\n📊 FEATURE TUTARLILIK KONTROLÜ")
    print("-" * 40)
    
    feature_files = [
        'models/feature_columns.pkl',
        'models/feature_cols.pkl'
    ]
    
    feature_counts = {}
    
    for feature_file in feature_files:
        if os.path.exists(feature_file):
            try:
                with open(feature_file, 'rb') as f:
                    features = pickle.load(f)
                
                feature_counts[feature_file] = len(features)
                print(f"✅ {feature_file}: {len(features)} feature")
                
            except Exception as e:
                print(f"❌ {feature_file}: {e}")
        else:
            print(f"⚠️ {feature_file}: Bulunamadı")
    
    # Tutarlılık kontrolü
    if len(set(feature_counts.values())) == 1:
        print(f"✅ Tüm feature dosyaları tutarlı: {list(feature_counts.values())[0]} feature")
        return True
    else:
        print(f"⚠️ Feature sayıları tutarsız: {feature_counts}")
        return False

def create_consistent_features():
    """Tutarlı feature listesi oluştur"""
    print("\n🆕 TUTARLI FEATURE LİSTESİ OLUŞTURULUYOR")
    print("-" * 40)
    
    try:
        # 125 feature oluştur (LSTM için standart)
        feature_names = []
        
        # Teknik göstergeler
        for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle']:
            for period in [5, 10, 20, 50, 100]:
                feature_names.append(f'{indicator}_{period}')
        
        # Fiyat özellikleri
        price_features = ['open', 'high', 'low', 'close', 'volume']
        for feature in price_features:
            feature_names.append(f'{feature}_normalized')
            feature_names.append(f'{feature}_pct_change')
        
        # Momentum göstergeleri
        momentum_features = ['momentum', 'roc', 'williams_r', 'cci', 'adx']
        for feature in momentum_features:
            feature_names.append(f'{feature}_5')
            feature_names.append(f'{feature}_14')
        
        # Volatilite göstergeleri
        volatility_features = ['atr', 'natr', 'trange']
        for feature in volatility_features:
            feature_names.append(f'{feature}_14')
        
        # Hacim göstergeleri
        volume_features = ['obv', 'ad', 'cmf', 'mfi']
        for feature in volume_features:
            feature_names.append(f'{feature}_14')
        
        # Eksik feature'ları tamamla
        while len(feature_names) < 125:
            feature_names.append(f'feature_{len(feature_names)}')
        
        # İlk 125 feature'ı al
        feature_names = feature_names[:125]
        
        print(f"✅ {len(feature_names)} feature oluşturuldu")
        
        # Feature dosyalarını güncelle
        feature_files = [
            'models/feature_columns.pkl',
            'models/feature_cols.pkl'
        ]
        
        for feature_file in feature_files:
            with open(feature_file, 'wb') as f:
                pickle.dump(feature_names, f)
            print(f"✅ {feature_file} güncellendi")
        
        return feature_names
        
    except Exception as e:
        print(f"❌ Feature oluşturma hatası: {e}")
        return None

def create_new_lstm_model(feature_count=125):
    """Yeni LSTM modeli oluştur"""
    print(f"\n🆕 YENİ LSTM MODELİ OLUŞTURULUYOR ({feature_count} feature)")
    print("-" * 40)
    
    try:
        # LSTM modeli oluştur
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=(1, feature_count)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Model derle
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ LSTM modeli oluşturuldu")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Modeli kaydet
        model_path = 'models/lstm_model.h5'
        model.save(model_path)
        print(f"✅ Model kaydedildi: {model_path}")
        
        # Optimized model de oluştur
        optimized_model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(1, feature_count)),
            keras.layers.Dropout(0.1),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimized_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        optimized_path = 'models/optimized_lstm_model.h5'
        optimized_model.save(optimized_path)
        print(f"✅ Optimized model kaydedildi: {optimized_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM model oluşturma hatası: {e}")
        return False

def test_lstm_prediction():
    """LSTM tahmin testi yap"""
    print("\n🧪 LSTM TAHMİN TESTİ")
    print("-" * 40)
    
    try:
        # Model yükle
        model = keras.models.load_model('models/lstm_model.h5')
        
        # Feature listesini yükle
        with open('models/feature_columns.pkl', 'rb') as f:
            features = pickle.load(f)
        
        # Test verisi oluştur
        test_data = np.random.randn(1, 1, len(features))
        
        # Tahmin yap
        prediction = model.predict(test_data)
        
        print(f"✅ Test tahmini başarılı")
        print(f"   Input shape: {test_data.shape}")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Tahmin değeri: {prediction[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM test hatası: {e}")
        return False

def backup_existing_models():
    """Mevcut modellerin yedeğini al"""
    print("\n💾 MODEL YEDEKLERİ OLUŞTURULUYOR")
    print("-" * 40)
    
    backup_dir = "models/backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_files = [
        'models/lstm_model.h5',
        'models/optimized_lstm_model.h5'
    ]
    
    backed_up = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                backup_name = f"{os.path.basename(model_file).replace('.h5', '')}_{timestamp}.h5"
                backup_path = os.path.join(backup_dir, backup_name)
                
                import shutil
                shutil.copy2(model_file, backup_path)
                
                print(f"✅ {model_file} -> {backup_path}")
                backed_up.append(model_file)
                
            except Exception as e:
                print(f"❌ {model_file} yedeklenemedi: {e}")
    
    return backed_up

def main():
    """Ana fonksiyon"""
    print_header()
    
    # Mevcut modelleri kontrol et
    issues, valid_models = check_lstm_model()
    
    # Feature tutarlılığını kontrol et
    feature_consistent = check_feature_consistency()
    
    if not issues and feature_consistent:
        print("\n✅ LSTM modelleri ve feature'lar tutarlı!")
        return
    
    # Yedek oluştur
    backed_up = backup_existing_models()
    
    if not backed_up and issues:
        print("❌ Yedek oluşturulamadı, işlem durduruluyor")
        return
    
    # Tutarlı feature'lar oluştur
    features = create_consistent_features()
    
    if not features:
        print("❌ Feature oluşturulamadı")
        return
    
    # Yeni LSTM modeli oluştur
    if not create_new_lstm_model(len(features)):
        print("❌ LSTM modeli oluşturulamadı")
        return
    
    # Test et
    if test_lstm_prediction():
        print("\n🎯 LSTM SHAPE DÜZELTME BAŞARILI!")
        
        # Son kontrol
        print("\n🔍 SON KONTROL")
        print("-" * 40)
        final_issues, final_valid = check_lstm_model()
        
        if not final_issues:
            print("✅ Tüm LSTM modelleri düzeltildi!")
        else:
            print(f"⚠️ {len(final_issues)} model hala sorunlu")
    else:
        print("\n❌ LSTM düzeltme başarısız!")
    
    print("\n" + "=" * 60)
    print("✅ LSTM shape düzeltme işlemi tamamlandı!")
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