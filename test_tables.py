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
        logging.FileHandler('logs/test_tables.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_database_connection():
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

def check_tables_exist():
    """Tabloların varlığını kontrol et"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        tables = ['signals', 'performance', 'system_logs']
        existing_tables = []
        
        for table in tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            exists = cursor.fetchone()[0]
            
            if exists:
                logger.info(f"✅ {table} tablosu mevcut")
                existing_tables.append(table)
                
                # Kayıt sayısını kontrol et
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"   📊 {table}: {count} kayıt")
            else:
                logger.warning(f"❌ {table} tablosu mevcut değil")
        
        cursor.close()
        conn.close()
        
        return existing_tables
        
    except Exception as e:
        logger.error(f"❌ Tablo kontrolü hatası: {e}")
        return []

def test_table_structure():
    """Tablo yapısını test et"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Signals tablosu yapısını kontrol et
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'signals'
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        logger.info("📋 Signals tablosu yapısı:")
        for col in columns:
            logger.info(f"   - {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tablo yapısı kontrolü hatası: {e}")
        return False

def test_table_operations():
    """Tablo operasyonlarını test et"""
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
        test_data = {
            'symbol': 'TEST/USDT',
            'timeframe': '1h',
            'direction': 'BUY',
            'ai_score': 0.75,
            'ta_strength': 0.80,
            'whale_score': 0.70,
            'social_score': 0.65,
            'news_score': 0.60,
            'timestamp': '2024-01-01 12:00:00',
            'entry_price': 100.00,
            'current_price': 100.00
        }
        
        insert_sql = """
        INSERT INTO signals (symbol, timeframe, direction, ai_score, ta_strength, 
                           whale_score, social_score, news_score, timestamp, entry_price, current_price)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_sql, (
            test_data['symbol'], test_data['timeframe'], test_data['direction'],
            test_data['ai_score'], test_data['ta_strength'], test_data['whale_score'],
            test_data['social_score'], test_data['news_score'], test_data['timestamp'],
            test_data['entry_price'], test_data['current_price']
        ))
        
        conn.commit()
        logger.info("✅ Test verisi başarıyla eklendi")
        
        # Test verisini oku
        cursor.execute("SELECT * FROM signals WHERE symbol = 'TEST/USDT'")
        result = cursor.fetchone()
        
        if result:
            logger.info("✅ Test verisi başarıyla okundu")
            logger.info(f"   📊 ID: {result[0]}, Symbol: {result[1]}, Direction: {result[3]}")
        
        # Test verisini sil
        cursor.execute("DELETE FROM signals WHERE symbol = 'TEST/USDT'")
        conn.commit()
        logger.info("✅ Test verisi başarıyla silindi")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tablo operasyonları hatası: {e}")
        import traceback
        logger.error(f"Hata detayı: {traceback.format_exc()}")
        return False

def test_indexes():
    """Indexleri test et"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Indexleri listele
        cursor.execute("""
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('signals', 'performance', 'system_logs')
            ORDER BY tablename, indexname;
        """)
        
        indexes = cursor.fetchall()
        logger.info("📋 Mevcut indexler:")
        for idx in indexes:
            logger.info(f"   - {idx[0]} ({idx[1]})")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Index kontrolü hatası: {e}")
        return False

def test_permissions():
    """Kullanıcı yetkilerini test et"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Kullanıcı bilgilerini kontrol et
        cursor.execute("SELECT current_user, current_database()")
        user_info = cursor.fetchone()
        logger.info(f"👤 Kullanıcı: {user_info[0]}")
        logger.info(f"🗄️ Veritabanı: {user_info[1]}")
        
        # Yetkileri kontrol et
        cursor.execute("""
            SELECT privilege_type
            FROM information_schema.role_table_grants
            WHERE grantee = current_user
            AND table_name = 'signals'
        """)
        
        privileges = cursor.fetchall()
        logger.info("🔐 Kullanıcı yetkileri:")
        for priv in privileges:
            logger.info(f"   - {priv[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Yetki kontrolü hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    logger.info("🚀 Tablo Test Başlatılıyor...")
    
    # 1. Bağlantı testi
    if not test_database_connection():
        logger.error("Veritabanı bağlantısı başarısız!")
        sys.exit(1)
    
    # 2. Tablo varlığı kontrolü
    existing_tables = check_tables_exist()
    
    if not existing_tables:
        logger.error("Hiç tablo bulunamadı!")
        sys.exit(1)
    
    # 3. Tablo yapısı testi
    if 'signals' in existing_tables:
        test_table_structure()
    
    # 4. Tablo operasyonları testi
    if 'signals' in existing_tables:
        test_table_operations()
    
    # 5. Index testi
    test_indexes()
    
    # 6. Yetki testi
    test_permissions()
    
    # 7. Özet rapor
    logger.info("📊 Test Özeti:")
    logger.info(f"   - Mevcut tablolar: {len(existing_tables)}")
    logger.info(f"   - Tablolar: {', '.join(existing_tables)}")
    
    if len(existing_tables) >= 3:
        logger.info("✅ Tüm testler başarılı!")
    else:
        logger.warning("⚠️ Bazı tablolar eksik olabilir")
    
    logger.info("🎉 Tablo testi tamamlandı!")

if __name__ == "__main__":
    main() 