#!/usr/bin/env python3
"""
Veritabanını Temizleme Scripti
"""

import psycopg2
from config import Config

def clean_database():
    print("🧹 Veritabanı temizleniyor...")
    
    try:
        # PostgreSQL bağlantısı
        conn = psycopg2.connect(Config.DATABASE_URL)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Tüm sinyalleri sil
        cursor.execute("DELETE FROM signals")
        deleted_count = cursor.rowcount
        
        print(f"✅ {deleted_count} sinyal silindi!")
        
        # Tabloyu sıfırla (ID'leri sıfırla)
        cursor.execute("ALTER SEQUENCE signals_id_seq RESTART WITH 1")
        print("✅ ID sıralaması sıfırlandı!")
        
        conn.close()
        print("🎉 Veritabanı temizlendi!")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    clean_database() 