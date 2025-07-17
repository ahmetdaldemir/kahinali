#!/usr/bin/env python3
"""
Model Durumu Kontrol Scripti
"""

import sys
import os
import pickle
import logging
from datetime import datetime

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.ai_model import AIModel
    from modules.data_collector import DataCollector
    from config import Config
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_files():
    """Model dosyalarının durumunu kontrol et"""
    print("🔍 Model dosyaları kontrol ediliyor...")
    
    model_files = {
        'lstm_model.h5': 'models/lstm_model.h5',
        'feature_cols.pkl': 'models/feature_cols.pkl'
    }
    
    status = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            modified = datetime.fromtimestamp(os.path.getmtime(path))
            status[name] = {
                'exists': True,
                'size_mb': round(size, 2),
                'modified': modified.strftime('%Y-%m-%d %H:%M:%S'),
                'path': path
            }
        else:
            status[name] = {
                'exists': False,
                'size_mb': 0,
                'modified': None,
                'path': path
            }
    
    return status

def check_feature_columns():
    """Feature kolonlarını kontrol et"""
    print("🔍 Feature kolonları kontrol ediliyor...")
    
    try:
        if os.path.exists('models/feature_cols.pkl'):
            with open('models/feature_cols.pkl', 'rb') as f:
                feature_cols = pickle.load(f)
            
            return {
                'exists': True,
                'count': len(feature_cols),
                'columns': feature_cols[:10] + ['...'] if len(feature_cols) > 10 else feature_cols
            }
        else:
            return {'exists': False, 'count': 0, 'columns': []}
    except Exception as e:
        return {'exists': False, 'error': str(e), 'count': 0, 'columns': []}

def test_model_prediction():
    """Model tahmin testi"""
    print("🧪 Model tahmin testi yapılıyor...")
    
    try:
        ai_model = AIModel()
        
        # Test verisi oluştur
        test_data = {
            'close': [50000, 50100, 50200, 50300, 50400],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'rsi': [50, 55, 60, 65, 70],
            'macd': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        # Tahmin yap
        prediction = ai_model.predict_signal('BTC/USDT', test_data)
        
        return {
            'success': True,
            'prediction': prediction,
            'model_loaded': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model_loaded': False
        }

def check_data_availability():
    """Veri erişilebilirliğini kontrol et"""
    print("🔍 Veri erişilebilirliği kontrol ediliyor...")
    
    try:
        collector = DataCollector()
        
        # Binance bağlantısını test et
        exchange = collector.get_exchange()
        ticker = exchange.fetch_ticker('BTC/USDT')
        
        return {
            'success': True,
            'binance_connected': True,
            'test_symbol': 'BTC/USDT',
            'current_price': ticker['last']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'binance_connected': False
        }

def main():
    print("🤖 KAHİN ULTİMA MODEL DURUM KONTROLÜ")
    print("=" * 50)
    print(f"📅 Kontrol Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Model dosyaları kontrolü
    model_files = check_model_files()
    print("📁 MODEL DOSYALARI")
    print("-" * 30)
    
    all_files_exist = True
    for name, info in model_files.items():
        if info['exists']:
            print(f"   ✅ {name}: {info['size_mb']} MB ({info['modified']})")
        else:
            print(f"   ❌ {name}: EKSİK")
            all_files_exist = False
    
    print()
    
    # 2. Feature kolonları kontrolü
    feature_info = check_feature_columns()
    print("🔧 FEATURE KOLONLARI")
    print("-" * 30)
    
    if feature_info['exists']:
        print(f"   ✅ Feature sayısı: {feature_info['count']}")
        print(f"   📋 İlk 10 kolon: {feature_info['columns']}")
    else:
        print(f"   ❌ Feature dosyası eksik")
        if 'error' in feature_info:
            print(f"      Hata: {feature_info['error']}")
    
    print()
    
    # 3. Model tahmin testi
    prediction_test = test_model_prediction()
    print("🧪 MODEL TAHMİN TESTİ")
    print("-" * 30)
    
    if prediction_test['success']:
        print("   ✅ Model tahmin yapabiliyor")
        print(f"   📊 Test tahmini: {prediction_test['prediction']}")
    else:
        print("   ❌ Model tahmin yapamıyor")
        print(f"      Hata: {prediction_test['error']}")
    
    print()
    
    # 4. Veri erişilebilirliği
    data_test = check_data_availability()
    print("🌐 VERİ ERİŞİLEBİLİRLİĞİ")
    print("-" * 30)
    
    if data_test['success']:
        print("   ✅ Binance API bağlantısı var")
        print(f"   💰 {data_test['test_symbol']} fiyatı: ${data_test['current_price']:,.2f}")
    else:
        print("   ❌ Binance API bağlantısı yok")
        print(f"      Hata: {data_test['error']}")
    
    print()
    
    # 5. Genel durum değerlendirmesi
    print("🎯 GENEL DURUM")
    print("-" * 30)
    
    if all_files_exist and feature_info['exists'] and prediction_test['success'] and data_test['success']:
        print("   ✅ TÜM SİSTEMLER HAZIR!")
        print("   🚀 Model eğitimi yapılabilir")
    elif all_files_exist and feature_info['exists'] and prediction_test['success']:
        print("   ⚠️ Model hazır ama veri erişimi yok")
        print("   🔄 İnternet bağlantısı kontrol edilmeli")
    elif all_files_exist and feature_info['exists']:
        print("   ⚠️ Model dosyaları var ama test başarısız")
        print("   🔄 Model yeniden eğitilmeli")
    else:
        print("   ❌ Model dosyaları eksik")
        print("   🔄 Model eğitimi gerekli")
    
    print()
    print("=" * 50)
    
    return all_files_exist and feature_info['exists'] and prediction_test['success'] and data_test['success']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 