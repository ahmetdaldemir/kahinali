#!/usr/bin/env python3
"""
KahinUltima Sistem Uyumluluk Kontrolü
Bu script sistem bileşenlerinin uyumluluğunu kontrol eder.
"""

import sys
import os
import importlib
import sqlite3
import json
from datetime import datetime

def check_python_version():
    """Python versiyonunu kontrol et"""
    print("=== Python Versiyon Kontrolü ===")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    if sys.version_info >= (3, 8):
        print("✅ Python versiyonu uyumlu (3.8+)")
        return True
    else:
        print("❌ Python versiyonu uyumsuz (3.8+ gerekli)")
        return False

def check_required_packages():
    """Gerekli paketleri kontrol et"""
    print("\n=== Paket Uyumluluk Kontrolü ===")
    
    required_packages = {
        'tensorflow': '2.13.0',
        'keras': '2.13.1',
        'pandas': '2.1.1',
        'numpy': '1.24.3',
        'scikit-learn': '1.3.0',
        'ccxt': '4.0.77',
        'psycopg2-binary': '2.9.7',
        'flask': '2.3.3',
        'requests': '2.31.0',
        'python-dotenv': '1.0.0',
        'schedule': '1.2.0',
        'ta': '0.10.2',
        'yfinance': '0.2.18'
    }
    
    all_compatible = True
    
    for package, expected_version in required_packages.items():
        try:
            module = importlib.import_module(package.replace('-', '_'))
            actual_version = getattr(module, '__version__', 'Unknown')
            print(f"{package}: {actual_version} (Beklenen: {expected_version})")
            
            if actual_version == expected_version:
                print(f"  ✅ {package} versiyonu uyumlu")
            else:
                print(f"  ⚠️  {package} versiyonu farklı ama çalışabilir")
                
        except ImportError:
            print(f"  ❌ {package} yüklü değil")
            all_compatible = False
    
    return all_compatible

def check_database():
    """Veritabanı durumunu kontrol et"""
    print("\n=== Veritabanı Kontrolü ===")
    
    # SQLite kontrolü
    if os.path.exists('kahin_ultima.db'):
        try:
            conn = sqlite3.connect('kahin_ultima.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            print(f"✅ SQLite veritabanı mevcut")
            print(f"   Tablolar: {table_names}")
            print(f"   Dosya boyutu: {os.path.getsize('kahin_ultima.db')} bytes")
            
            conn.close()
            return True
        except Exception as e:
            print(f"❌ SQLite veritabanı hatası: {e}")
            return False
    else:
        print("⚠️  SQLite veritabanı bulunamadı")
        return False

def check_model_files():
    """Model dosyalarını kontrol et"""
    print("\n=== Model Dosyaları Kontrolü ===")
    
    model_files = {
        'models/lstm_model.h5': 'LSTM Model',
        'models/feature_cols.pkl': 'Feature Columns',
        'models/scaler.pkl': 'Scaler',
        'models/rf_model.pkl': 'Random Forest Model',
        'models/gb_model.pkl': 'Gradient Boosting Model'
    }
    
    all_exist = True
    
    for file_path, description in model_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} ({size} bytes)")
        else:
            print(f"❌ {description}: {file_path} bulunamadı")
            all_exist = False
    
    return all_exist

def check_configuration():
    """Konfigürasyon dosyalarını kontrol et"""
    print("\n=== Konfigürasyon Kontrolü ===")
    
    config_files = {
        'config.py': 'Ana Konfigürasyon',
        'requirements.txt': 'Bağımlılıklar',
        '.env': 'Çevre Değişkenleri'
    }
    
    all_exist = True
    
    for file_path, description in config_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} ({size} bytes)")
        else:
            print(f"⚠️  {description}: {file_path} bulunamadı")
            if file_path == '.env':
                print("   Not: .env dosyası opsiyonel, environment variables kullanılabilir")
            else:
                all_exist = False
    
    return all_exist

def check_directories():
    """Gerekli dizinleri kontrol et"""
    print("\n=== Dizin Yapısı Kontrolü ===")
    
    required_dirs = {
        'data': 'Veri Dizini',
        'models': 'Model Dizini',
        'signals': 'Sinyal Dizini',
        'logs': 'Log Dizini',
        'modules': 'Modül Dizini'
    }
    
    all_exist = True
    
    for dir_path, description in required_dirs.items():
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"✅ {description}: {dir_path} ({file_count} dosya)")
        else:
            print(f"❌ {description}: {dir_path} bulunamadı")
            all_exist = False
    
    return all_exist

def check_module_imports():
    """Modül import'larını test et"""
    print("\n=== Modül Import Testi ===")
    
    modules_to_test = [
        'modules.data_collector',
        'modules.technical_analysis',
        'modules.ai_model',
        'modules.signal_manager',
        'modules.telegram_bot',
        'modules.whale_tracker',
        'modules.news_analysis',
        'modules.performance',
        'modules.signal_tracker',
        'modules.market_analysis'
    ]
    
    all_importable = True
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name} başarıyla import edildi")
        except ImportError as e:
            print(f"❌ {module_name} import hatası: {e}")
            all_importable = False
        except Exception as e:
            print(f"⚠️  {module_name} beklenmeyen hata: {e}")
    
    return all_importable

def check_api_keys():
    """API anahtarlarını kontrol et"""
    print("\n=== API Anahtarları Kontrolü ===")
    
    # .env dosyasını yükle
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    api_keys = {
        'BINANCE_API_KEY': 'Binance API',
        'BINANCE_SECRET_KEY': 'Binance Secret',
        'TELEGRAM_BOT_TOKEN': 'Telegram Bot',
        'TELEGRAM_CHAT_ID': 'Telegram Chat ID',
        'NEWS_API_KEY': 'News API'
    }
    
    all_configured = True
    
    for key_name, description in api_keys.items():
        value = os.getenv(key_name)
        if value and value.strip():
            print(f"✅ {description}: Yapılandırılmış")
        else:
            print(f"⚠️  {description}: Yapılandırılmamış (opsiyonel)")
    
    return True

def generate_compatibility_report():
    """Uyumluluk raporu oluştur"""
    print("=" * 60)
    print("KAHIN ULTIMA SİSTEM UYUMLULUK RAPORU")
    print("=" * 60)
    print(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checks = [
        ("Python Versiyonu", check_python_version),
        ("Paket Uyumluluğu", check_required_packages),
        ("Veritabanı", check_database),
        ("Model Dosyaları", check_model_files),
        ("Konfigürasyon", check_configuration),
        ("Dizin Yapısı", check_directories),
        ("Modül Import'ları", check_module_imports),
        ("API Anahtarları", check_api_keys)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} kontrolünde hata: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 60)
    print("ÖZET RAPOR")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{check_name}: {status}")
    
    print(f"\nToplam: {passed}/{total} kontrol başarılı")
    
    if passed == total:
        print("🎉 Sistem tamamen uyumlu!")
    elif passed >= total * 0.8:
        print("⚠️  Sistem büyük ölçüde uyumlu, bazı sorunlar var")
    else:
        print("❌ Sistemde önemli uyumsuzluklar var")
    
    return results

if __name__ == "__main__":
    generate_compatibility_report() 