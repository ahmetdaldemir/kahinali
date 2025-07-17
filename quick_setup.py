#!/usr/bin/env python3
"""
KahinUltima Hızlı Kurulum Scripti
Bu script sistemi yeni bir bilgisayarda hızlıca kurmak için kullanılır.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🚀 KAHIN ULTIMA HIZLI KURULUM SCRIPTI")
    print("=" * 60)
    print()

def check_python_version():
    """Python versiyonunu kontrol et"""
    print("📋 Python versiyonu kontrol ediliyor...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 veya üzeri gerekli!")
        print(f"   Mevcut versiyon: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Uygun")
    return True

def check_system_requirements():
    """Sistem gereksinimlerini kontrol et"""
    print("\n📋 Sistem gereksinimleri kontrol ediliyor...")
    
    # Disk alanı kontrolü
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        print(f"   Disk alanı: {free_gb} GB boş")
        
        if free_gb < 10:
            print("⚠️  Düşük disk alanı! En az 10GB önerilir.")
        else:
            print("✅ Yeterli disk alanı")
    except:
        print("⚠️  Disk alanı kontrol edilemedi")
    
    # RAM kontrolü (Windows için)
    if platform.system() == "Windows":
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total // (1024**3)
            print(f"   RAM: {ram_gb} GB")
            
            if ram_gb < 8:
                print("⚠️  Düşük RAM! En az 8GB önerilir.")
            else:
                print("✅ Yeterli RAM")
        except:
            print("⚠️  RAM kontrol edilemedi")
    
    print("✅ Sistem gereksinimleri kontrol edildi")

def create_directories():
    """Gerekli klasörleri oluştur"""
    print("\n📁 Gerekli klasörler oluşturuluyor...")
    
    directories = [
        'logs',
        'data',
        'signals',
        'models',
        'models/backup'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Klasörler oluşturuldu")

def setup_virtual_environment():
    """Python sanal ortamı oluştur"""
    print("\n🐍 Python sanal ortamı oluşturuluyor...")
    
    if os.path.exists("venv"):
        print("   ⚠️  Sanal ortam zaten mevcut")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("   ✅ Sanal ortam oluşturuldu")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ Sanal ortam oluşturulamadı")
        return False

def install_requirements():
    """Gerekli paketleri kur"""
    print("\n📦 Gerekli paketler kuruluyor...")
    
    # Sanal ortamı aktifleştir
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
    else:
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
    
    try:
        # pip'i güncelle
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("   ✅ pip güncellendi")
        
        # requirements.txt varsa kur
        if os.path.exists("requirements.txt"):
            subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
            print("   ✅ requirements.txt kuruldu")
        else:
            # Temel paketleri kur
            basic_packages = [
                "pandas", "numpy", "ccxt", "requests", "flask",
                "psycopg2-binary", "python-dotenv", "scikit-learn",
                "tensorflow", "matplotlib", "seaborn", "tqdm"
            ]
            
            for package in basic_packages:
                try:
                    subprocess.run([pip_path, "install", package], check=True)
                    print(f"   ✅ {package} kuruldu")
                except:
                    print(f"   ⚠️  {package} kurulamadı")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Paket kurulumu başarısız: {e}")
        return False

def create_env_file():
    """Örnek .env dosyası oluştur"""
    print("\n🔧 .env dosyası oluşturuluyor...")
    
    if os.path.exists(".env"):
        print("   ⚠️  .env dosyası zaten mevcut")
        return True
    
    env_content = """# KahinUltima Çevre Değişkenleri
# Bu dosyayı kendi API anahtarlarınızla doldurun

# API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Twitter API
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=KAHIN_Ultima_Bot/1.0

# News API
NEWS_API_KEY=your_news_api_key_here

# PostgreSQL Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=kahin_ultima
POSTGRES_USER=kahin_user
POSTGRES_PASSWORD=your_password_here

# Flask
FLASK_SECRET_KEY=kahin-ultima-secret-key-2024
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
"""
    
    try:
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_content)
        print("   ✅ .env dosyası oluşturuldu")
        print("   ⚠️  Lütfen API anahtarlarınızı .env dosyasına ekleyin")
        return True
    except Exception as e:
        print(f"   ❌ .env dosyası oluşturulamadı: {e}")
        return False

def run_tests():
    """Temel testleri çalıştır"""
    print("\n🧪 Temel testler çalıştırılıyor...")
    
    test_scripts = [
        "test_system.py",
        "test_api.py", 
        "test_db.py",
        "test_data_collection.py"
    ]
    
    for test_script in test_scripts:
        if os.path.exists(test_script):
            try:
                print(f"   🔍 {test_script} çalıştırılıyor...")
                result = subprocess.run([sys.executable, test_script], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"   ✅ {test_script} başarılı")
                else:
                    print(f"   ⚠️  {test_script} uyarılar var")
            except subprocess.TimeoutExpired:
                print(f"   ⚠️  {test_script} zaman aşımı")
            except Exception as e:
                print(f"   ❌ {test_script} hatası: {e}")
        else:
            print(f"   ⚠️  {test_script} bulunamadı")

def print_next_steps():
    """Sonraki adımları göster"""
    print("\n" + "=" * 60)
    print("🎉 KURULUM TAMAMLANDI!")
    print("=" * 60)
    
    print("\n📋 SONRAKI ADIMLAR:")
    print("1. 🔑 API anahtarlarınızı .env dosyasına ekleyin")
    print("2. 🗄️  PostgreSQL veritabanını kurun ve yapılandırın")
    print("3. 🔧 config.py dosyasını ihtiyaçlarınıza göre düzenleyin")
    print("4. 🚀 Sistemi başlatın: python main.py")
    print("5. 🌐 Web arayüzünü açın: python app/web.py")
    
    print("\n📚 YARDIM:")
    print("• Detaylı kurulum: DEPLOYMENT_GUIDE.md dosyasını okuyun")
    print("• Sorun giderme: Log dosyalarını kontrol edin")
    print("• Test: python test_system.py")
    
    print("\n⚠️  ÖNEMLİ:")
    print("• API anahtarlarınızı güvenli tutun")
    print("• .env dosyasını git'e eklemeyin")
    print("• Düzenli backup alın")

def main():
    """Ana kurulum fonksiyonu"""
    print_header()
    
    # Sistem kontrolü
    if not check_python_version():
        return False
    
    check_system_requirements()
    
    # Kurulum adımları
    create_directories()
    
    if not setup_virtual_environment():
        return False
    
    if not install_requirements():
        return False
    
    if not create_env_file():
        return False
    
    # Testler
    run_tests()
    
    # Sonraki adımlar
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Kurulum başarıyla tamamlandı!")
        else:
            print("\n❌ Kurulum sırasında hatalar oluştu!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Kurulum kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        sys.exit(1) 