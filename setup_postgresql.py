#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import psycopg2
from sqlalchemy import create_engine, text
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Komut çalıştır ve sonucu logla"""
    try:
        logger.info(f"Çalıştırılıyor: {description}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {description} başarılı")
            if result.stdout:
                logger.info(f"Çıktı: {result.stdout}")
        else:
            logger.error(f"❌ {description} başarısız: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"❌ {description} hatası: {e}")
        return False

def install_postgresql():
    """PostgreSQL kurulumu"""
    commands = [
        ("sudo apt update", "Sistem güncellemesi"),
        ("sudo apt install -y postgresql postgresql-contrib", "PostgreSQL kurulumu"),
        ("sudo systemctl start postgresql", "PostgreSQL servisi başlatma"),
        ("sudo systemctl enable postgresql", "PostgreSQL servisi otomatik başlatma"),
        ("sudo systemctl status postgresql", "PostgreSQL durumu kontrol")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def configure_postgresql():
    """PostgreSQL konfigürasyonu"""
    try:
        # PostgreSQL kullanıcısına geç
        commands = [
            ("sudo -u postgres psql -c \"ALTER USER postgres PASSWORD '3010726904';\"", "PostgreSQL root şifresi ayarlama"),
            ("sudo -u postgres psql -c \"CREATE DATABASE kahin_ultima;\"", "Veritabanı oluşturma"),
            ("sudo -u postgres psql -c \"CREATE USER laravel WITH PASSWORD 'secret';\"", "Laravel kullanıcısı oluşturma"),
            ("sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO laravel;\"", "Laravel kullanıcısına yetki verme"),
            ("sudo -u postgres psql -c \"ALTER USER laravel CREATEDB;\"", "Laravel kullanıcısına DB oluşturma yetkisi"),
            ("sudo -u postgres psql -c \"\\l\"", "Veritabanları listesi")
        ]
        
        for command, description in commands:
            if not run_command(command, description):
                return False
        return True
    except Exception as e:
        logger.error(f"PostgreSQL konfigürasyon hatası: {e}")
        return False

def test_connection():
    """Veritabanı bağlantısını test et"""
    try:
        logger.info("Veritabanı bağlantısı test ediliyor...")
        
        # Config'den bağlantı bilgilerini al
        db_url = Config.DATABASE_URL
        logger.info(f"Bağlantı URL: {db_url}")
        
        # SQLAlchemy ile test
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"✅ PostgreSQL bağlantısı başarılı: {version}")
        
        # psycopg2 ile test
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        conn.close()
        logger.info("✅ psycopg2 bağlantısı başarılı")
        
        return True
    except Exception as e:
        logger.error(f"❌ Veritabanı bağlantı hatası: {e}")
        return False

def create_tables():
    """Gerekli tabloları oluştur"""
    try:
        logger.info("Tablolar oluşturuluyor...")
        engine = create_engine(Config.DATABASE_URL)
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20),
            timeframe VARCHAR(10),
            direction VARCHAR(10),
            ai_score DECIMAL(5,4),
            ta_strength DECIMAL(5,4),
            whale_score DECIMAL(5,4),
            social_score DECIMAL(5,4),
            news_score DECIMAL(5,4),
            timestamp VARCHAR(50),
            predicted_gain DECIMAL(10,4),
            predicted_duration VARCHAR(20),
            entry_price DECIMAL(15,8) DEFAULT NULL,
            exit_price DECIMAL(15,8) DEFAULT NULL,
            result VARCHAR(20) DEFAULT NULL,
            realized_gain DECIMAL(10,4) DEFAULT NULL,
            duration DECIMAL(10,4) DEFAULT NULL,
            take_profit DECIMAL(15,8) DEFAULT NULL,
            stop_loss DECIMAL(15,8) DEFAULT NULL,
            support_level DECIMAL(15,8) DEFAULT NULL,
            resistance_level DECIMAL(15,8) DEFAULT NULL,
            target_time_hours DECIMAL(10,2) DEFAULT NULL,
            max_hold_time_hours DECIMAL(10,2) DEFAULT 24.0,
            predicted_breakout_threshold DECIMAL(10,4) DEFAULT NULL,
            actual_max_gain DECIMAL(10,4) DEFAULT NULL,
            actual_max_loss DECIMAL(10,4) DEFAULT NULL,
            breakout_achieved BOOLEAN DEFAULT FALSE,
            breakout_time_hours DECIMAL(10,4) DEFAULT NULL,
            predicted_breakout_time_hours DECIMAL(10,4) DEFAULT NULL,
            risk_reward_ratio DECIMAL(10,4) DEFAULT NULL,
            actual_risk_reward_ratio DECIMAL(10,4) DEFAULT NULL,
            volatility_score DECIMAL(5,4) DEFAULT NULL,
            trend_strength DECIMAL(5,4) DEFAULT NULL,
            market_regime VARCHAR(20) DEFAULT NULL,
            signal_quality_score DECIMAL(5,4) DEFAULT NULL,
            success_metrics JSONB DEFAULT NULL,
            volume_score DECIMAL(5,4) DEFAULT NULL,
            momentum_score DECIMAL(5,4) DEFAULT NULL,
            pattern_score DECIMAL(5,4) DEFAULT NULL,
            order_book_imbalance DECIMAL(10,4) DEFAULT NULL,
            top_bid_walls TEXT DEFAULT NULL,
            top_ask_walls TEXT DEFAULT NULL,
            whale_direction_score DECIMAL(10,4) DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        
        logger.info("✅ Tablolar başarıyla oluşturuldu")
        return True
    except Exception as e:
        logger.error(f"❌ Tablo oluşturma hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    logger.info("🚀 PostgreSQL Kurulum ve Konfigürasyon Başlatılıyor...")
    
    # 1. PostgreSQL kurulumu
    if not install_postgresql():
        logger.error("PostgreSQL kurulumu başarısız!")
        sys.exit(1)
    
    # 2. PostgreSQL konfigürasyonu
    if not configure_postgresql():
        logger.error("PostgreSQL konfigürasyonu başarısız!")
        sys.exit(1)
    
    # 3. Bağlantı testi
    if not test_connection():
        logger.error("Veritabanı bağlantı testi başarısız!")
        sys.exit(1)
    
    # 4. Tabloları oluştur
    if not create_tables():
        logger.error("Tablo oluşturma başarısız!")
        sys.exit(1)
    
    logger.info("🎉 PostgreSQL kurulum ve konfigürasyon tamamlandı!")

if __name__ == "__main__":
    main() 