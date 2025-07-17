import psycopg2
from config import Config

def check_database():
    try:
        # PostgreSQL bağlantısı
        conn = psycopg2.connect(Config.DATABASE_URL)
        cur = conn.cursor()
        
        # Tabloları listele
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cur.fetchall()
        print("📊 PostgreSQL Veritabanı Tabloları:")
        print("=" * 40)
        
        for table in tables:
            table_name = table[0]
            print(f"✅ {table_name}")
            
            # Her tablonun kayıt sayısını göster
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                print(f"   📈 Kayıt sayısı: {count}")
            except Exception as e:
                print(f"   ❌ Kayıt sayısı alınamadı: {e}")
        
        print("\n" + "=" * 40)
        print(f"🔗 Bağlantı: {Config.DATABASE_URL}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Veritabanı bağlantı hatası: {e}")

if __name__ == "__main__":
    check_database() 