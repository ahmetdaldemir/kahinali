#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import psycopg2
from sqlalchemy import create_engine, text, inspect
from config import Config

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/force_create_tables.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_connection():
    """Veritabanı bağlantısını test et"""
    try:
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
        
        # SQLAlchemy ile test
        engine = create_engine(Config.DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✅ SQLAlchemy bağlantısı başarılı")
        
        return True
    except Exception as e:
        logger.error(f"❌ Veritabanı bağlantı hatası: {e}")
        return False

def force_create_signals_table():
    """Signals tablosunu zorla oluştur"""
    try:
        # psycopg2 ile direkt oluştur
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Tablo var mı kontrol et
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'signals'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            logger.info("✅ Signals tablosu zaten mevcut")
            cursor.execute("SELECT COUNT(*) FROM signals")
            count = cursor.fetchone()[0]
            logger.info(f"📊 Signals tablosu: {count} kayıt")
        else:
            logger.info("🔄 Signals tablosu oluşturuluyor...")
            
            create_table_sql = """
            CREATE TABLE signals (
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
            
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info("✅ Signals tablosu başarıyla oluşturuldu")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Signals tablosu oluşturma hatası: {e}")
        import traceback
        logger.error(f"Hata detayı: {traceback.format_exc()}")
        return False

def force_create_performance_table():
    """Performance tablosunu zorla oluştur"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Tablo var mı kontrol et
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'performance'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            logger.info("✅ Performance tablosu zaten mevcut")
        else:
            logger.info("🔄 Performance tablosu oluşturuluyor...")
            
            create_table_sql = """
            CREATE TABLE performance (
                id SERIAL PRIMARY KEY,
                date DATE,
                total_signals INTEGER DEFAULT 0,
                successful_signals INTEGER DEFAULT 0,
                failed_signals INTEGER DEFAULT 0,
                success_rate DECIMAL(5,4) DEFAULT 0.0,
                average_profit DECIMAL(10,4) DEFAULT 0.0,
                total_profit DECIMAL(15,8) DEFAULT 0.0,
                max_drawdown DECIMAL(10,4) DEFAULT 0.0,
                sharpe_ratio DECIMAL(10,4) DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info("✅ Performance tablosu başarıyla oluşturuldu")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance tablosu oluşturma hatası: {e}")
        return False

def force_create_system_logs_table():
    """System logs tablosunu zorla oluştur"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Tablo var mı kontrol et
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'system_logs'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            logger.info("✅ System logs tablosu zaten mevcut")
        else:
            logger.info("🔄 System logs tablosu oluşturuluyor...")
            
            create_table_sql = """
            CREATE TABLE system_logs (
                id SERIAL PRIMARY KEY,
                level VARCHAR(10),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                module VARCHAR(50),
                error_details TEXT
            )
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info("✅ System logs tablosu başarıyla oluşturuldu")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ System logs tablosu oluşturma hatası: {e}")
        return False

def create_indexes():
    """Indexleri oluştur"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction)",
            "CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result)",
            "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Indexler oluşturuldu")
        return True
        
    except Exception as e:
        logger.error(f"❌ Index oluşturma hatası: {e}")
        return False

def test_table_access():
    """Tablo erişimini test et"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Test verisi ekle
        test_insert = """
        INSERT INTO signals (symbol, timeframe, direction, ai_score, ta_strength, whale_score, social_score, news_score, timestamp, entry_price, current_price) 
        VALUES ('TEST/USDT', '1h', 'BUY', 0.75, 0.80, 0.70, 0.65, 0.60, '2024-01-01 12:00:00', 100.00, 100.00)
        ON CONFLICT DO NOTHING
        """
        
        cursor.execute(test_insert)
        conn.commit()
        
        # Test verisini sil
        cursor.execute("DELETE FROM signals WHERE symbol = 'TEST/USDT'")
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Tablo erişim testi başarılı")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tablo erişim testi hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    logger.info("🚀 Zorla Tablo Oluşturma Başlatılıyor...")
    
    # 1. Bağlantı testi
    if not test_connection():
        logger.error("Veritabanı bağlantısı başarısız!")
        sys.exit(1)
    
    # 2. Tabloları zorla oluştur
    tables_created = 0
    
    if force_create_signals_table():
        tables_created += 1
    
    if force_create_performance_table():
        tables_created += 1
    
    if force_create_system_logs_table():
        tables_created += 1
    
    # 3. Indexleri oluştur
    create_indexes()
    
    # 4. Tablo erişimini test et
    test_table_access()
    
    # 5. Tablo durumlarını raporla
    logger.info("📊 Tablo durumları:")
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        for table in ['signals', 'performance', 'system_logs']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"   - {table}: {count} kayıt")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Tablo durumu kontrolü hatası: {e}")
    
    logger.info(f"🎉 Zorla tablo oluşturma tamamlandı! {tables_created} tablo oluşturuldu/mevcut")

if __name__ == "__main__":
    main() 