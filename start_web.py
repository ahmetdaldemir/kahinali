#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import logging
from config import Config

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/web_dashboard.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Gerekli bağımlılıkları kontrol et"""
    try:
        import flask
        import psycopg2
        import sqlalchemy
        logger.info("✅ Tüm bağımlılıklar mevcut")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        return False

def check_database():
    """Veritabanı bağlantısını kontrol et"""
    try:
        from sqlalchemy import create_engine
        engine = create_engine(Config.DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            logger.info("✅ Veritabanı bağlantısı başarılı")
        return True
    except Exception as e:
        logger.error(f"❌ Veritabanı bağlantı hatası: {e}")
        return False

def start_web_dashboard():
    """Web dashboard'u başlat"""
    try:
        logger.info("🚀 Web Dashboard başlatılıyor...")
        
        # Flask uygulamasını başlat
        from app.web import app
        
        logger.info(f"🌐 Web Dashboard http://{Config.FLASK_HOST}:{Config.FLASK_PORT} adresinde başlatılıyor")
        
        # Production modunda çalıştır
        app.run(
            host=Config.FLASK_HOST,
            port=Config.FLASK_PORT,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Web Dashboard başlatma hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    logger.info("🎯 Kahinali Web Dashboard Başlatıcı")
    
    # 1. Bağımlılıkları kontrol et
    if not check_dependencies():
        logger.error("Bağımlılık kontrolü başarısız!")
        sys.exit(1)
    
    # 2. Veritabanını kontrol et
    if not check_database():
        logger.error("Veritabanı kontrolü başarısız!")
        sys.exit(1)
    
    # 3. Web dashboard'u başlat
    start_web_dashboard()

if __name__ == "__main__":
    main() 