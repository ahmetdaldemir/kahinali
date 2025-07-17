#!/usr/bin/env python3
"""
Veritabanı şemasını güncelleme scripti
Yeni gelişmiş başarı kriterleri alanlarını ekler
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import json
import sqlalchemy

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from modules.signal_manager import SignalManager

# Ortam değişkeninden veya config.py'den veritabanı bağlantı adresini al
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/kahin_ultima')

# Eklenmesi gereken kolonlar ve tipleri
COLUMNS = [
    ('quality_score', 'NUMERIC'),
    ('market_regime', 'VARCHAR(32)'),
    ('volatility_regime', 'VARCHAR(32)'),
    ('volume_score', 'NUMERIC'),
    ('momentum_score', 'NUMERIC'),
    ('pattern_score', 'NUMERIC'),
    ('breakout_probability', 'NUMERIC'),
    ('risk_reward_ratio', 'NUMERIC'),
    ('confidence_level', 'NUMERIC'),
    ('signal_strength', 'NUMERIC'),
    ('market_sentiment', 'NUMERIC'),
]

def create_performance_tables(engine):
    """Performans takibi için tablolar oluştur"""
    try:
        print("=== Performans Takibi Tabloları Oluşturuluyor ===")
        
        # Sinyal performans tablosu
        signal_performance_query = """
            CREATE TABLE IF NOT EXISTS signal_performance (
                id SERIAL PRIMARY KEY,
                signal_id INTEGER NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                confidence DECIMAL(5,4) NOT NULL,
                entry_price DECIMAL(20,8) NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                status VARCHAR(10) DEFAULT 'OPEN',
                current_price DECIMAL(20,8),
                current_pnl DECIMAL(10,6),
                max_profit DECIMAL(10,6) DEFAULT 0,
                max_loss DECIMAL(10,6) DEFAULT 0,
                exit_price DECIMAL(20,8),
                exit_time TIMESTAMP,
                final_pnl DECIMAL(10,6),
                duration_hours DECIMAL(10,2),
                success BOOLEAN,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        
        # Model performans tablosu
        model_performance_query = """
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                total_signals INTEGER DEFAULT 0,
                successful_signals INTEGER DEFAULT 0,
                total_pnl DECIMAL(10,6) DEFAULT 0,
                success_rate DECIMAL(5,4) DEFAULT 0,
                avg_pnl DECIMAL(10,6) DEFAULT 0,
                direction_performance JSONB,
                confidence_performance JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        
        # Performans analizi tablosu
        performance_analytics_query = """
            CREATE TABLE IF NOT EXISTS performance_analytics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                total_signals INTEGER DEFAULT 0,
                successful_signals INTEGER DEFAULT 0,
                total_pnl DECIMAL(10,6) DEFAULT 0,
                success_rate DECIMAL(5,4) DEFAULT 0,
                avg_pnl DECIMAL(10,6) DEFAULT 0,
                best_period VARCHAR(20),
                worst_period VARCHAR(20),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        
        # Model eğitim geçmişi tablosu
        model_training_history_query = """
            CREATE TABLE IF NOT EXISTS model_training_history (
                id SERIAL PRIMARY KEY,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_type VARCHAR(20) NOT NULL,
                accuracy DECIMAL(5,4),
                precision DECIMAL(5,4),
                recall DECIMAL(5,4),
                f1_score DECIMAL(5,4),
                training_data_size INTEGER,
                feature_count INTEGER,
                training_duration_minutes INTEGER,
                notes TEXT
            );
        """
        
        # Tabloları oluştur
        with engine.connect() as conn:
            conn.execute(text(signal_performance_query))
            conn.execute(text(model_performance_query))
            conn.execute(text(performance_analytics_query))
            conn.execute(text(model_training_history_query))
            conn.commit()
        
        print("✅ Performans takibi tabloları oluşturuldu")
        
    except Exception as e:
        print(f"❌ Performans tabloları oluşturma hatası: {e}")

def create_performance_indexes(engine):
    """Performans için index'ler oluştur"""
    try:
        print("=== Performans Index'leri Oluşturuluyor ===")
        
        # Sinyal performans index'leri
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_signal_performance_signal_id ON signal_performance(signal_id);",
            "CREATE INDEX IF NOT EXISTS idx_signal_performance_symbol ON signal_performance(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_signal_performance_status ON signal_performance(status);",
            "CREATE INDEX IF NOT EXISTS idx_signal_performance_entry_time ON signal_performance(entry_time);",
            "CREATE INDEX IF NOT EXISTS idx_signal_performance_success ON signal_performance(success);",
            
            # Model performans index'leri
            "CREATE INDEX IF NOT EXISTS idx_model_performance_created_at ON model_performance(created_at);",
            
            # Performans analizi index'leri
            "CREATE INDEX IF NOT EXISTS idx_performance_analytics_symbol ON performance_analytics(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_performance_analytics_success_rate ON performance_analytics(success_rate);",
            
            # Model eğitim geçmişi index'leri
            "CREATE INDEX IF NOT EXISTS idx_model_training_history_date ON model_training_history(training_date);",
            "CREATE INDEX IF NOT EXISTS idx_model_training_history_type ON model_training_history(model_type);"
        ]
        
        with engine.connect() as conn:
            for index_query in indexes:
                conn.execute(text(index_query))
            conn.commit()
        
        print("✅ Performans index'leri oluşturuldu")
        
    except Exception as e:
        print(f"❌ Index oluşturma hatası: {e}")

def insert_initial_performance_data(engine):
    """Başlangıç performans verilerini ekle"""
    try:
        print("=== Başlangıç Performans Verileri Ekleniyor ===")
        
        # Model performans başlangıç kaydı
        initial_performance_query = """
            INSERT INTO model_performance (total_signals, successful_signals, total_pnl, success_rate, avg_pnl)
            VALUES (0, 0, 0, 0, 0)
            ON CONFLICT DO NOTHING;
        """
        
        with engine.connect() as conn:
            conn.execute(text(initial_performance_query))
            conn.commit()
        
        print("✅ Başlangıç performans verileri eklendi")
        
    except Exception as e:
        print(f"❌ Başlangıç veri ekleme hatası: {e}")

def update_signals_table(engine):
    """Sinyaller tablosunu günceller"""
    try:
        with engine.connect() as conn:
            # Status sütunu ekle
            conn.execute(text("""
                ALTER TABLE signals 
                ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active'
            """))
            conn.commit()
            print("✓ status VARCHAR(20) DEFAULT 'active' alanı eklendi")
            
            # Mevcut sinyalleri güncelle (12 saat sonra expired)
            conn.execute(text("""
                UPDATE signals 
                SET status = CASE 
                    WHEN exit_time IS NOT NULL THEN 'closed'
                    WHEN timestamp::timestamp < NOW() - INTERVAL '12 hours' THEN 'expired'
                    ELSE 'active'
                END
            """))
            conn.commit()
            print("✓ Mevcut sinyaller status alanı güncellendi")
            
    except Exception as e:
        print(f"❌ Sinyaller tablosu güncellenirken hata: {e}")

def add_columns():
    engine = sqlalchemy.create_engine(DATABASE_URL)
    with engine.connect() as conn:
        for col, typ in COLUMNS:
            try:
                print(f"Adding column {col} ({typ}) if not exists...")
                conn.execute(text(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {col} {typ};"))
            except Exception as e:
                print(f"Hata oluştu: {col}: {e}")
    print("Tüm kolonlar kontrol edildi ve eklendi.")

def update_database_schema():
    """Veritabanı şemasını güncelle"""
    try:
        print("=== Veritabanı Şeması Güncelleniyor ===")
        
        # SignalManager'ı başlat
        signal_manager = SignalManager()
        
        # PostgreSQL bağlantısı
        engine = create_engine(Config.DATABASE_URL)
        
        # Yeni alanları ekle
        alter_queries = [
            # Gelişmiş başarı kriterleri için yeni alanlar
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS predicted_breakout_threshold DECIMAL(10,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS actual_max_gain DECIMAL(10,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS actual_max_loss DECIMAL(10,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS breakout_achieved BOOLEAN DEFAULT FALSE",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS breakout_time_hours DECIMAL(10,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS predicted_breakout_time_hours DECIMAL(10,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS risk_reward_ratio DECIMAL(10,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS actual_risk_reward_ratio DECIMAL(10,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS volatility_score DECIMAL(5,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS trend_strength DECIMAL(5,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS market_regime VARCHAR(20) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS signal_quality_score DECIMAL(5,4) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS success_metrics JSONB DEFAULT NULL",
            
            # Hedef fiyatlar ve seviyeler için yeni alanlar
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS take_profit DECIMAL(15,8) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS stop_loss DECIMAL(15,8) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS support_level DECIMAL(15,8) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS resistance_level DECIMAL(15,8) DEFAULT NULL",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS target_time_hours DECIMAL(10,2) DEFAULT 24.0",
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS max_hold_time_hours DECIMAL(10,2) DEFAULT 24.0"
        ]
        
        with engine.connect() as conn:
            for query in alter_queries:
                try:
                    conn.execute(text(query))
                    print(f"✅ {query.split('ADD COLUMN IF NOT EXISTS')[1].split()[0]} alanı eklendi")
                except Exception as e:
                    print(f"⚠️ {query.split('ADD COLUMN IF NOT EXISTS')[1].split()[0]} alanı zaten mevcut: {e}")
            
            conn.commit()
        
        # Performans takibi tablolarını oluştur
        create_performance_tables(engine)
        
        # Performans index'lerini oluştur
        create_performance_indexes(engine)
        
        # Başlangıç performans verilerini ekle
        insert_initial_performance_data(engine)
        
        print("\n=== Mevcut Sinyalleri Güncelleme ===")
        
        # Mevcut sinyalleri al
        query = "SELECT * FROM signals WHERE success_metrics IS NULL"
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            print(f"📊 {len(df)} sinyal güncellenecek...")
            
            for idx, signal in df.iterrows():
                try:
                    signal_data = signal.to_dict()
                    
                    # Başarı metriklerini hesapla
                    success_metrics = signal_manager.calculate_success_metrics(signal_data)
                    
                    # Güncelleme sorgusu
                    update_query = """
                        UPDATE signals SET
                            predicted_breakout_threshold = :predicted_threshold,
                            predicted_breakout_time_hours = :predicted_time,
                            risk_reward_ratio = :risk_reward,
                            volatility_score = :volatility_score,
                            trend_strength = :trend_strength,
                            market_regime = :market_regime,
                            signal_quality_score = :signal_quality,
                            success_metrics = :success_metrics
                        WHERE id = :signal_id
                    """
                    
                    with engine.connect() as conn:
                        conn.execute(text(update_query), {
                            'predicted_threshold': success_metrics.get('predicted_breakout_threshold', 0.025),
                            'predicted_time': success_metrics.get('predicted_breakout_time', 24.0),
                            'risk_reward': success_metrics.get('risk_reward_ratio', 1.67),
                            'volatility_score': success_metrics.get('volatility_score', 0.5),
                            'trend_strength': success_metrics.get('trend_strength', 0.5),
                            'market_regime': success_metrics.get('market_regime', 'BİLİNMEYEN'),
                            'signal_quality': success_metrics.get('signal_quality', 0.5),
                            'success_metrics': json.dumps(success_metrics),
                            'signal_id': signal['id']
                        })
                        conn.commit()
                    
                    if (idx + 1) % 10 == 0:
                        print(f"📈 {idx + 1}/{len(df)} sinyal güncellendi...")
                        
                except Exception as e:
                    print(f"❌ Sinyal {signal['id']} güncellenirken hata: {e}")
                    continue
            
            print(f"✅ {len(df)} sinyal başarıyla güncellendi!")
        else:
            print("✅ Tüm sinyaller zaten güncel!")
        
        print("\n=== Sinyaller Tablosu Güncelleniyor ===")
        update_signals_table(engine)
        
        print("\n=== Veritabanı Şeması Güncelleme Tamamlandı ===")
        print("🎉 Gelişmiş başarı kriterleri ve performans takibi sistemi aktif!")
        
    except Exception as e:
        print(f"❌ Veritabanı güncelleme hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

if __name__ == "__main__":
    update_database_schema()
    add_columns()
    # Veritabanı bağlantısı
    engine = create_engine(DATABASE_URL)
    update_signals_table(engine) 