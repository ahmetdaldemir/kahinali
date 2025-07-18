#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from sqlalchemy import create_engine, text, inspect
from config import Config

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database_migration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DatabaseMigration:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL)
        self.inspector = inspect(self.engine)
        
    def check_connection(self):
        """Veritabanı bağlantısını kontrol et"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("✅ Veritabanı bağlantısı başarılı")
                return True
        except Exception as e:
            logger.error(f"❌ Veritabanı bağlantı hatası: {e}")
            return False
    
    def table_exists(self, table_name):
        """Tablo var mı kontrol et"""
        try:
            return table_name in self.inspector.get_table_names()
        except Exception as e:
            logger.error(f"Tablo kontrolü hatası: {e}")
            return False
    
    def create_signals_table(self):
        """Signals tablosunu oluştur"""
        try:
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
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            logger.info("✅ Signals tablosu oluşturuldu")
            return True
            
        except Exception as e:
            logger.error(f"❌ Signals tablosu oluşturma hatası: {e}")
            return False
    
    def create_performance_table(self):
        """Performance tablosunu oluştur"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS performance (
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
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            logger.info("✅ Performance tablosu oluşturuldu")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance tablosu oluşturma hatası: {e}")
            return False
    
    def create_system_logs_table(self):
        """System logs tablosunu oluştur"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS system_logs (
                id SERIAL PRIMARY KEY,
                level VARCHAR(10),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                module VARCHAR(50),
                error_details TEXT
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            logger.info("✅ System logs tablosu oluşturuldu")
            return True
            
        except Exception as e:
            logger.error(f"❌ System logs tablosu oluşturma hatası: {e}")
            return False
    
    def create_indexes(self):
        """Performans için indexler oluştur"""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction)",
                "CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result)",
                "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date)"
            ]
            
            with self.engine.connect() as conn:
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                conn.commit()
            
            logger.info("✅ Indexler oluşturuldu")
            return True
            
        except Exception as e:
            logger.error(f"❌ Index oluşturma hatası: {e}")
            return False
    
    def run_migration(self):
        """Tüm migration'ları çalıştır"""
        logger.info("🚀 Veritabanı migration başlatılıyor...")
        
        # 1. Bağlantı kontrolü
        if not self.check_connection():
            return False
        
        # 2. Tabloları oluştur
        tables_created = 0
        
        if not self.table_exists('signals'):
            if self.create_signals_table():
                tables_created += 1
        else:
            logger.info("✅ Signals tablosu zaten mevcut")
            tables_created += 1
        
        if not self.table_exists('performance'):
            if self.create_performance_table():
                tables_created += 1
        else:
            logger.info("✅ Performance tablosu zaten mevcut")
            tables_created += 1
        
        if not self.table_exists('system_logs'):
            if self.create_system_logs_table():
                tables_created += 1
        else:
            logger.info("✅ System logs tablosu zaten mevcut")
            tables_created += 1
        
        # 3. Indexleri oluştur
        self.create_indexes()
        
        # 4. Tablo durumlarını raporla
        logger.info("📊 Tablo durumları:")
        for table in ['signals', 'performance', 'system_logs']:
            if self.table_exists(table):
                with self.engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.fetchone()[0]
                    logger.info(f"   - {table}: {count} kayıt")
            else:
                logger.error(f"   - {table}: TABLO YOK!")
        
        logger.info(f"🎉 Migration tamamlandı! {tables_created} tablo oluşturuldu/mevcut")
        return True

def main():
    """Ana fonksiyon"""
    migration = DatabaseMigration()
    success = migration.run_migration()
    
    if success:
        logger.info("✅ Migration başarılı!")
        sys.exit(0)
    else:
        logger.error("❌ Migration başarısız!")
        sys.exit(1)

if __name__ == "__main__":
    main() 