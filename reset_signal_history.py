#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinyal geçmişini temizleme scripti - PostgreSQL için
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import sqlalchemy
from sqlalchemy import text

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_signal_history():
    """Tüm sinyal geçmişini temizle - PostgreSQL"""
    
    logger.info("🔧 Sinyal geçmişi temizleme başlıyor...")
    
    # 1. Signals klasöründeki JSON dosyalarını sil
    signals_dir = "signals"
    if os.path.exists(signals_dir):
        json_files = [f for f in os.listdir(signals_dir) if f.endswith('.json')]
        csv_files = [f for f in os.listdir(signals_dir) if f.endswith('.csv')]
        
        deleted_json = 0
        deleted_csv = 0
        
        for file in json_files:
            try:
                os.remove(os.path.join(signals_dir, file))
                deleted_json += 1
            except Exception as e:
                logger.error(f"JSON dosya silme hatası {file}: {e}")
        
        for file in csv_files:
            try:
                os.remove(os.path.join(signals_dir, file))
                deleted_csv += 1
            except Exception as e:
                logger.error(f"CSV dosya silme hatası {file}: {e}")
        
        logger.info(f"✅ {deleted_json} JSON dosya ve {deleted_csv} CSV dosya silindi")
    
    # 2. PostgreSQL'den sinyalleri temizle
    try:
        from config import Config
        
        # PostgreSQL bağlantısı
        engine = sqlalchemy.create_engine(Config.DATABASE_URL)
        
        with engine.connect() as conn:
            # Sinyal tablosunu temizle
            delete_query = "DELETE FROM signals"
            result = conn.execute(text(delete_query))
            deleted_db = result.rowcount
            
            # ID sıralamasını sıfırla (eğer sequence varsa)
            try:
                reset_query = "ALTER SEQUENCE signals_id_seq RESTART WITH 1"
                conn.execute(text(reset_query))
                logger.info("✅ ID sıralaması sıfırlandı")
            except Exception as e:
                logger.warning(f"ID sıralaması sıfırlanamadı: {e}")
            
            # Değişiklikleri kaydet
            conn.commit()
        
        logger.info(f"✅ PostgreSQL'den {deleted_db} sinyal kaydı silindi")
        
    except Exception as e:
        logger.error(f"PostgreSQL temizleme hatası: {e}")
    
    # 3. Log dosyasını temizle
    log_file = "logs/kahin_ultima.log"
    if os.path.exists(log_file):
        try:
            # Log dosyasını boşalt
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Log dosyası {datetime.now()} tarihinde temizlendi\n")
            logger.info("✅ Log dosyası temizlendi")
        except Exception as e:
            logger.error(f"Log temizleme hatası: {e}")
    
    logger.info("🎉 Sinyal geçmişi temizleme tamamlandı!")
    logger.info("📝 Sistem yeniden başlatıldığında yeni sinyaller üretilecek")

if __name__ == "__main__":
    clean_signal_history() 