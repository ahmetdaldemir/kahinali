#!/usr/bin/env python3
"""
KahinUltima Sistem Durumu Kontrol Scripti
Bu script sistemin durumunu kontrol eder ve raporlar.
"""

import os
import sys
import platform
import subprocess
import importlib
import psutil
import shutil
from datetime import datetime
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🔍 KAHIN ULTIMA SİSTEM DURUMU KONTROLÜ")
    print("=" * 60)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Sistem: {platform.system()} {platform.release()}")
    print()

def check_python_environment():
    """Python ortamını kontrol et"""
    print("🐍 PYTHON ORTAMI KONTROLÜ")
    print("-" * 40)
    
    # Python versiyonu
    version = sys.version_info
    print(f"Python Versiyonu: {version.major}.{version.minor}.{version.micro}")
    
    # Sanal ortam kontrolü
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Sanal ortam aktif")
    else:
        print("⚠️  Sanal ortam aktif değil")
    
    # Gerekli paketler
    required_packages = [
        'pandas', 'numpy', 'ccxt', 'requests', 'flask',
        'psycopg2', 'dotenv', 'sklearn', 'tensorflow',
        'matplotlib', 'seaborn', 'tqdm'
    ]
    
    print("\n📦 Gerekli Paketler:")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Kurulu değil")
    
    print()

def check_system_resources():
    """Sistem kaynaklarını kontrol et"""
    print("💻 SİSTEM KAYNAKLARI")
    print("-" * 40)
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu_count} çekirdek, %{cpu_percent} kullanım")
    
    # RAM
    memory = psutil.virtual_memory()
    ram_gb = memory.total // (1024**3)
    ram_used_gb = memory.used // (1024**3)
    ram_percent = memory.percent
    print(f"RAM: {ram_gb}GB toplam, {ram_used_gb}GB kullanım (%{ram_percent})")
    
    # Disk
    disk = psutil.disk_usage('.')
    disk_gb = disk.total // (1024**3)
    disk_free_gb = disk.free // (1024**3)
    disk_percent = (disk.used / disk.total) * 100
    print(f"Disk: {disk_gb}GB toplam, {disk_free_gb}GB boş (%{disk_percent:.1f} kullanım)")
    
    # Network
    try:
        network = psutil.net_io_counters()
        print(f"Network: {network.bytes_sent // (1024**2)}MB gönderilen, {network.bytes_recv // (1024**2)}MB alınan")
    except:
        print("Network: Bilgi alınamadı")
    
    print()

def check_directories():
    """Gerekli klasörleri kontrol et"""
    print("📁 KLASÖR KONTROLÜ")
    print("-" * 40)
    
    required_dirs = [
        'logs', 'data', 'signals', 'models', 'models/backup'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            # Klasör boyutunu hesapla
            try:
                total_size = sum(f.stat().st_size for f in Path(directory).rglob('*') if f.is_file())
                size_mb = total_size // (1024**2)
                print(f"✅ {directory}/ ({size_mb}MB)")
            except:
                print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - Mevcut değil")
    
    print()

def check_config_files():
    """Konfigürasyon dosyalarını kontrol et"""
    print("⚙️ KONFİGÜRASYON DOSYALARI")
    print("-" * 40)
    
    config_files = [
        '.env', 'config.py', 'requirements.txt'
    ]
    
    for file in config_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - Mevcut değil")
    
    print()

def check_database_connection():
    """Veritabanı bağlantısını kontrol et"""
    print("🗄️ VERİTABANI KONTROLÜ")
    print("-" * 40)
    
    try:
        # config.py'den veritabanı bilgilerini al
        sys.path.append('.')
        from config import Config
        
        print(f"Host: {Config.POSTGRES_HOST}")
        print(f"Port: {Config.POSTGRES_PORT}")
        print(f"Database: {Config.POSTGRES_DB}")
        print(f"User: {Config.POSTGRES_USER}")
        
        # Bağlantı testi
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT,
                database=Config.POSTGRES_DB,
                user=Config.POSTGRES_USER,
                password=Config.POSTGRES_PASSWORD
            )
            conn.close()
            print("✅ Veritabanı bağlantısı başarılı")
        except Exception as e:
            print(f"❌ Veritabanı bağlantı hatası: {e}")
        
    except ImportError:
        print("❌ config.py dosyası bulunamadı")
    except Exception as e:
        print(f"❌ Veritabanı kontrol hatası: {e}")
    
    print()

def check_api_connections():
    """API bağlantılarını kontrol et"""
    print("🔗 API BAĞLANTILARI")
    print("-" * 40)
    
    try:
        from config import Config
        
        # Binance API
        if Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY:
            print("✅ Binance API anahtarları mevcut")
        else:
            print("⚠️ Binance API anahtarları eksik")
        
        # Telegram
        if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            print("✅ Telegram bot ayarları mevcut")
        else:
            print("⚠️ Telegram bot ayarları eksik")
        
        # Twitter
        if Config.TWITTER_API_KEY:
            print("✅ Twitter API anahtarı mevcut")
        else:
            print("⚠️ Twitter API anahtarı eksik")
        
        # Reddit
        if Config.REDDIT_CLIENT_ID:
            print("✅ Reddit API ayarları mevcut")
        else:
            print("⚠️ Reddit API ayarları eksik")
        
        # News API
        if Config.NEWS_API_KEY:
            print("✅ News API anahtarı mevcut")
        else:
            print("⚠️ News API anahtarı eksik")
        
    except ImportError:
        print("❌ config.py dosyası bulunamadı")
    except Exception as e:
        print(f"❌ API kontrol hatası: {e}")
    
    print()

def check_log_files():
    """Log dosyalarını kontrol et"""
    print("📝 LOG DOSYALARI")
    print("-" * 40)
    
    log_files = [
        'logs/kahin_ultima.log',
        'logs/model_training.log',
        'logs/model_retraining.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            size_mb = size // (1024**2)
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            print(f"✅ {log_file} ({size_mb}MB, {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"⚠️ {log_file} - Mevcut değil")
    
    print()

def check_data_files():
    """Veri dosyalarını kontrol et"""
    print("📊 VERİ DOSYALARI")
    print("-" * 40)
    
    data_dir = 'data'
    if os.path.exists(data_dir):
        files = list(Path(data_dir).glob('*.csv'))
        if files:
            total_size = sum(f.stat().st_size for f in files)
            total_mb = total_size // (1024**2)
            print(f"✅ {len(files)} CSV dosyası bulundu ({total_mb}MB)")
            
            # En büyük dosyaları listele
            largest_files = sorted(files, key=lambda x: x.stat().st_size, reverse=True)[:5]
            for file in largest_files:
                size_mb = file.stat().st_size // (1024**2)
                print(f"   📄 {file.name} ({size_mb}MB)")
        else:
            print("⚠️ CSV dosyası bulunamadı")
    else:
        print("❌ data/ klasörü mevcut değil")
    
    print()

def check_model_files():
    """Model dosyalarını kontrol et"""
    print("🤖 MODEL DOSYALARI")
    print("-" * 40)
    
    model_dir = 'models'
    if os.path.exists(model_dir):
        model_files = list(Path(model_dir).glob('*.pkl')) + list(Path(model_dir).glob('*.h5'))
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files)
            total_mb = total_size // (1024**2)
            print(f"✅ {len(model_files)} model dosyası bulundu ({total_mb}MB)")
            
            for file in model_files:
                size_mb = file.stat().st_size // (1024**2)
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                print(f"   🤖 {file.name} ({size_mb}MB, {mtime.strftime('%Y-%m-%d')})")
        else:
            print("⚠️ Model dosyası bulunamadı")
    else:
        print("❌ models/ klasörü mevcut değil")
    
    print()

def run_quick_tests():
    """Hızlı testler çalıştır"""
    print("🧪 HIZLI TESTLER")
    print("-" * 40)
    
    test_scripts = [
        'test_system.py',
        'test_api.py',
        'test_db.py'
    ]
    
    for test_script in test_scripts:
        if os.path.exists(test_script):
            try:
                print(f"🔍 {test_script} çalıştırılıyor...")
                result = subprocess.run([sys.executable, test_script], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"   ✅ {test_script} başarılı")
                else:
                    print(f"   ⚠️ {test_script} uyarılar var")
            except subprocess.TimeoutExpired:
                print(f"   ⚠️ {test_script} zaman aşımı")
            except Exception as e:
                print(f"   ❌ {test_script} hatası: {e}")
        else:
            print(f"   ⚠️ {test_script} bulunamadı")
    
    print()

def generate_summary():
    """Özet rapor oluştur"""
    print("📋 SİSTEM ÖZETİ")
    print("-" * 40)
    
    # Sistem durumu
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    print(f"💻 Sistem Durumu:")
    print(f"   • CPU: %{psutil.cpu_percent()} kullanım")
    print(f"   • RAM: %{memory.percent} kullanım")
    print(f"   • Disk: %{(disk.used / disk.total) * 100:.1f} kullanım")
    
    # Dosya sayıları
    data_files = len(list(Path('data').glob('*.csv'))) if os.path.exists('data') else 0
    model_files = len(list(Path('models').glob('*.pkl'))) + len(list(Path('models').glob('*.h5'))) if os.path.exists('models') else 0
    log_files = len(list(Path('logs').glob('*.log'))) if os.path.exists('logs') else 0
    
    print(f"📁 Dosya Durumu:")
    print(f"   • Veri dosyaları: {data_files}")
    print(f"   • Model dosyaları: {model_files}")
    print(f"   • Log dosyaları: {log_files}")
    
    # Öneriler
    print(f"💡 Öneriler:")
    if memory.percent > 80:
        print("   • RAM kullanımı yüksek, sistem performansı etkilenebilir")
    if (disk.used / disk.total) * 100 > 90:
        print("   • Disk alanı kritik seviyede")
    if data_files == 0:
        print("   • Veri dosyası bulunamadı, veri toplama gerekli")
    if model_files == 0:
        print("   • Model dosyası bulunamadı, model eğitimi gerekli")
    
    print()

def main():
    """Ana kontrol fonksiyonu"""
    print_header()
    
    check_python_environment()
    check_system_resources()
    check_directories()
    check_config_files()
    check_database_connection()
    check_api_connections()
    check_log_files()
    check_data_files()
    check_model_files()
    run_quick_tests()
    generate_summary()
    
    print("=" * 60)
    print("✅ Sistem kontrolü tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Kontrol kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Kontrol sırasında hata: {e}")
        import traceback
        traceback.print_exc() 