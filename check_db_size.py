#!/usr/bin/env python3
"""
KahinUltima PostgreSQL Veritabanı Boyut Kontrol Scripti
Bu script PostgreSQL veritabanının boyutunu ve detaylarını gösterir.
"""

import os
import sys
import psycopg2
from datetime import datetime
import glob

def print_header():
    print("=" * 60)
    print("📊 KAHIN ULTIMA VERİTABANI BOYUT KONTROLÜ")
    print("=" * 60)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def get_db_connection():
    """PostgreSQL bağlantısı oluştur"""
    try:
        # Config dosyasından bağlantı bilgilerini al
        from config import Config
        
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL bağlantı hatası: {e}")
        return None

def get_database_size():
    """Veritabanı boyutunu al"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        
        # Veritabanı boyutu
        cursor.execute("""
            SELECT 
                pg_size_pretty(pg_database_size(current_database())) as size,
                pg_database_size(current_database()) as size_bytes
        """)
        result = cursor.fetchone()
        
        conn.close()
        return {
            'size_pretty': result[0],
            'size_bytes': result[1],
            'size_mb': result[1] / (1024 * 1024)
        }
    except Exception as e:
        print(f"❌ Veritabanı boyutu alınamadı: {e}")
        return None

def get_table_sizes():
    """Tablo boyutlarını al"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        
        # Tablo boyutları
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size_pretty,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size_pretty,
                pg_relation_size(schemaname||'.'||tablename) as table_size_bytes
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)
        
        tables = []
        for row in cursor.fetchall():
            tables.append({
                'schema': row[0],
                'name': row[1],
                'total_size_pretty': row[2],
                'total_size_bytes': row[3],
                'table_size_pretty': row[4],
                'table_size_bytes': row[5],
                'total_size_mb': row[3] / (1024 * 1024)
            })
        
        conn.close()
        return tables
    except Exception as e:
        print(f"❌ Tablo boyutları alınamadı: {e}")
        return []

def get_table_row_counts():
    """Tablo satır sayılarını al"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        # Tablo listesini al
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        row_counts = {}
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                row_counts[table] = count
            except:
                row_counts[table] = 0
        
        conn.close()
        return row_counts
    except Exception as e:
        print(f"❌ Satır sayıları alınamadı: {e}")
        return {}

def analyze_database():
    """Veritabanını analiz et"""
    print("🔍 POSTGRESQL VERİTABANI ANALİZİ")
    print("-" * 40)
    
    # Veritabanı boyutu
    db_size = get_database_size()
    if not db_size:
        print("❌ Veritabanı boyutu alınamadı")
        return
    
    print(f"📁 Veritabanı: kahin_ultima")
    print(f"   💾 Toplam boyut: {db_size['size_pretty']} ({db_size['size_mb']:.2f} MB)")
    
    # Tablo boyutları
    tables = get_table_sizes()
    row_counts = get_table_row_counts()
    
    if not tables:
        print("   ❌ Tablo bilgisi alınamadı")
        return
    
    print(f"   📊 Tablo sayısı: {len(tables)}")
    
    total_rows = sum(row_counts.values())
    print(f"   📈 Toplam satır: {total_rows:,}")
    
    print(f"\n   📋 TABLO DETAYLARI:")
    total_table_size = 0
    
    for table in tables:
        table_name = table['name']
        row_count = row_counts.get(table_name, 0)
        total_table_size += table['total_size_mb']
        
        print(f"      • {table_name}")
        print(f"        - Satır: {row_count:,}")
        print(f"        - Toplam boyut: {table['total_size_pretty']}")
        print(f"        - Tablo boyutu: {table['table_size_pretty']}")
    
    print(f"\n📊 TOPLAM TABLO BOYUTU: {total_table_size:.2f} MB")
    
    # Boyut kategorileri
    if db_size['size_mb'] < 100:
        print("✅ Veritabanı boyutu normal (100 MB altında)")
    elif db_size['size_mb'] < 1000:
        print("⚠️ Veritabanı boyutu büyük (100 MB - 1 GB)")
    elif db_size['size_mb'] < 10000:
        print("🚨 Veritabanı boyutu çok büyük (1 GB - 10 GB)")
    else:
        print("💥 Veritabanı boyutu kritik (10 GB üzerinde)")
    
    return db_size['size_mb']

def check_disk_space():
    """Disk alanını kontrol et"""
    print("\n💽 DİSK ALANI KONTROLÜ")
    print("-" * 40)
    
    try:
        import shutil
        
        # Mevcut dizinin disk alanı
        total, used, free = shutil.disk_usage('.')
        
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        print(f"💾 Toplam disk alanı: {total_gb:.2f} GB")
        print(f"📊 Kullanılan alan: {used_gb:.2f} GB")
        print(f"🆓 Boş alan: {free_gb:.2f} GB")
        print(f"📈 Kullanım oranı: {(used/total)*100:.1f}%")
        
        if free_gb < 1:
            print("🚨 Dikkat: Disk alanı kritik seviyede!")
        elif free_gb < 5:
            print("⚠️ Dikkat: Disk alanı az!")
        else:
            print("✅ Disk alanı yeterli")
            
    except Exception as e:
        print(f"❌ Disk alanı kontrol edilemedi: {e}")

def show_table_details():
    """Tablo detaylarını göster"""
    print("\n📋 TABLO DETAYLI ANALİZ")
    print("-" * 40)
    
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Tablo listesini al
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            print(f"\n📊 Tablo: {table}")
            
            # Satır sayısı
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            print(f"   📈 Satır sayısı: {row_count:,}")
            
            # Sütun bilgileri
            cursor.execute(f"""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = '{table}' AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            print(f"   📋 Sütun sayısı: {len(columns)}")
            
            # İlk birkaç sütunu göster
            if columns:
                print(f"   📝 Sütunlar:")
                for col in columns[:5]:  # İlk 5 sütunu göster
                    col_name, col_type, col_length = col
                    length_info = f"({col_length})" if col_length else ""
                    print(f"      - {col_name}: {col_type}{length_info}")
                
                if len(columns) > 5:
                    print(f"      ... ve {len(columns) - 5} sütun daha")
            
            # İlk birkaç satırı göster
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_rows = cursor.fetchall()
                
                print(f"   👀 Örnek veriler:")
                for i, row in enumerate(sample_rows, 1):
                    # İlk 3 sütunu göster
                    preview = str(row[:3]) if len(row) >= 3 else str(row)
                    if len(preview) > 100:
                        preview = preview[:100] + "..."
                    print(f"      {i}. {preview}")
            
            # Tablo boyutu
            cursor.execute(f"""
                SELECT 
                    pg_size_pretty(pg_total_relation_size('{table}')) as total_size,
                    pg_size_pretty(pg_relation_size('{table}')) as table_size
            """)
            size_result = cursor.fetchone()
            print(f"   💾 Toplam boyut: {size_result[0]}")
            print(f"   📄 Tablo boyutu: {size_result[1]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Tablo detayları alınamadı: {e}")

def optimize_database():
    """Veritabanını optimize et"""
    print("\n🔧 POSTGRESQL VERİTABANI OPTİMİZASYONU")
    print("-" * 40)
    
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # VACUUM işlemi
        print("🧹 VACUUM işlemi başlatılıyor...")
        cursor.execute("VACUUM ANALYZE")
        print("✅ VACUUM ANALYZE tamamlandı")
        
        # İstatistikleri güncelle
        print("📊 İstatistikler güncelleniyor...")
        cursor.execute("ANALYZE")
        print("✅ ANALYZE tamamlandı")
        
        # İndeksleri yeniden oluştur
        print("🔍 İndeksler yeniden oluşturuluyor...")
        cursor.execute("REINDEX DATABASE kahin_ultima")
        print("✅ REINDEX tamamlandı")
        
        conn.commit()
        conn.close()
        
        print("🎯 PostgreSQL veritabanı optimizasyonu tamamlandı!")
        
    except Exception as e:
        print(f"❌ Optimizasyon hatası: {e}")

def check_postgresql_status():
    """PostgreSQL servis durumunu kontrol et"""
    print("\n🔍 POSTGRESQL SERVİS DURUMU")
    print("-" * 40)
    
    conn = get_db_connection()
    if not conn:
        print("❌ PostgreSQL bağlantısı kurulamadı")
        return False
    
    try:
        cursor = conn.cursor()
        
        # PostgreSQL versiyonu
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"📋 PostgreSQL Versiyonu: {version.split(',')[0]}")
        
        # Aktif bağlantılar
        cursor.execute("SELECT count(*) FROM pg_stat_activity")
        active_connections = cursor.fetchone()[0]
        print(f"🔗 Aktif bağlantılar: {active_connections}")
        
        # Maksimum bağlantılar
        cursor.execute("SHOW max_connections")
        max_connections = cursor.fetchone()[0]
        print(f"📊 Maksimum bağlantılar: {max_connections}")
        
        # Bağlantı kullanım oranı
        usage_ratio = (active_connections / int(max_connections)) * 100
        print(f"📈 Bağlantı kullanım oranı: {usage_ratio:.1f}%")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL durumu kontrol edilemedi: {e}")
        return False

def main():
    """Ana fonksiyon"""
    print_header()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--optimize':
            optimize_database()
        elif sys.argv[1] == '--details':
            show_table_details()
        elif sys.argv[1] == '--disk':
            check_disk_space()
        elif sys.argv[1] == '--status':
            check_postgresql_status()
        else:
            print("❌ Geçersiz parametre")
            print("Kullanım:")
            print("  python check_db_size.py --optimize # Veritabanını optimize et")
            print("  python check_db_size.py --details  # Tablo detaylarını göster")
            print("  python check_db_size.py --disk     # Disk alanını kontrol et")
            print("  python check_db_size.py --status   # PostgreSQL durumunu kontrol et")
    else:
        # Varsayılan analiz
        check_postgresql_status()
        analyze_database()
        check_disk_space()
        show_table_details()
    
    print("\n" + "=" * 60)
    print("✅ PostgreSQL veritabanı analizi tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc() 