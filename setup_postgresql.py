#!/usr/bin/env python3
"""
PostgreSQL Kurulum ve Yapılandırma Scripti
KAHİN Ultima için PostgreSQL veritabanını kurar ve yapılandırır.
"""

import os
import sys
import subprocess
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import Config

def check_postgresql_installed():
    """PostgreSQL'in kurulu olup olmadığını kontrol et"""
    try:
        result = subprocess.run(['psql', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ PostgreSQL kurulu: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PostgreSQL kurulu değil!")
        return False

def install_postgresql():
    """PostgreSQL'i kur (Windows için)"""
    print("📦 PostgreSQL kurulumu başlatılıyor...")
    
    # Windows için PostgreSQL kurulum linki
    download_url = "https://www.postgresql.org/download/windows/"
    print(f"🔗 PostgreSQL'i şu adresten indirin: {download_url}")
    print("📋 Kurulum adımları:")
    print("   1. PostgreSQL installer'ı indirin")
    print("   2. Kurulum sırasında şu ayarları yapın:")
    print(f"      - Port: {Config.POSTGRES_PORT}")
    print(f"      - Superuser: {Config.POSTGRES_USER}")
    print(f"      - Password: {Config.POSTGRES_PASSWORD}")
    print("   3. Kurulum tamamlandıktan sonra bu scripti tekrar çalıştırın")
    
    return False

def create_database():
    """Veritabanını oluştur"""
    try:
        # PostgreSQL'e bağlan
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD,
            database='postgres'  # Varsayılan veritabanı
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Veritabanının var olup olmadığını kontrol et
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (Config.POSTGRES_DB,))
        exists = cursor.fetchone()
        
        if not exists:
            # Veritabanını oluştur
            cursor.execute(f"CREATE DATABASE {Config.POSTGRES_DB}")
            print(f"✅ Veritabanı oluşturuldu: {Config.POSTGRES_DB}")
        else:
            print(f"✅ Veritabanı zaten mevcut: {Config.POSTGRES_DB}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Veritabanı oluşturulamadı: {e}")
        return False

def test_connection():
    """Veritabanı bağlantısını test et"""
    try:
        conn = psycopg2.connect(Config.DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL bağlantısı başarılı!")
        print(f"📊 PostgreSQL versiyonu: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL bağlantısı başarısız: {e}")
        return False

def create_env_file():
    """Örnek .env dosyası oluştur"""
    env_content = f"""# PostgreSQL Configuration
POSTGRES_HOST={Config.POSTGRES_HOST}
POSTGRES_PORT={Config.POSTGRES_PORT}
POSTGRES_DB={Config.POSTGRES_DB}
POSTGRES_USER={Config.POSTGRES_USER}
POSTGRES_PASSWORD={Config.POSTGRES_PASSWORD}
DATABASE_URL={Config.DATABASE_URL}

# API Keys (Bunları kendi API anahtarlarınızla değiştirin)
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

# Flask
FLASK_SECRET_KEY=kahin-ultima-secret-key-2024
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Trading Parameters
MIN_SIGNAL_CONFIDENCE=0.7
MAX_COINS_TO_TRACK=100

# AI Model Parameters
LSTM_LOOKBACK_DAYS=60
MODEL_RETRAIN_INTERVAL_HOURS=24

# Whale Tracking
WHALE_THRESHOLD_USDT=100000

# Social Media
SOCIAL_MEDIA_UPDATE_INTERVAL_MINUTES=15

# News
NEWS_UPDATE_INTERVAL_MINUTES=30

# Performance Tracking
SIGNAL_EXPIRY_HOURS=24
MIN_PROFIT_PERCENTAGE=2.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/kahin_ultima.log
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ .env dosyası oluşturuldu")
    print("📝 Lütfen .env dosyasını düzenleyerek API anahtarlarınızı ekleyin")

def main():
    """Ana kurulum fonksiyonu"""
    print("🚀 KAHİN Ultima PostgreSQL Kurulum Scripti")
    print("=" * 50)
    
    # PostgreSQL kurulumunu kontrol et
    if not check_postgresql_installed():
        if not install_postgresql():
            return
    
    # Veritabanını oluştur
    if not create_database():
        print("❌ Veritabanı oluşturulamadı. Lütfen PostgreSQL kurulumunu tamamlayın.")
        return
    
    # Bağlantıyı test et
    if not test_connection():
        print("❌ Veritabanı bağlantısı başarısız. Lütfen ayarları kontrol edin.")
        return
    
    # .env dosyasını oluştur
    create_env_file()
    
    print("\n🎉 PostgreSQL kurulumu tamamlandı!")
    print("\n📋 Sonraki adımlar:")
    print("   1. .env dosyasını düzenleyerek API anahtarlarınızı ekleyin")
    print("   2. Python sanal ortamınızı aktifleştirin")
    print("   3. Gereksinimleri yükleyin: pip install -r requirements.txt")
    print("   4. Sistemi başlatın: python main.py")

if __name__ == "__main__":
    main() 