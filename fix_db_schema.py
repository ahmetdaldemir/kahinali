import sqlalchemy
from sqlalchemy import text

# Doğru veritabanı bağlantısı
DATABASE_URL = 'postgresql://laravel:secret@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

# Eklenmesi gereken kolonlar
COLUMNS = [
    ('quality_score', 'NUMERIC'),
    ('volatility_regime', 'VARCHAR(32)'),
    ('volume_score', 'NUMERIC'),
    ('momentum_score', 'NUMERIC'),
    ('pattern_score', 'NUMERIC'),
    ('breakout_probability', 'NUMERIC'),
    ('confidence_level', 'NUMERIC'),
    ('signal_strength', 'NUMERIC'),
    ('market_sentiment', 'NUMERIC'),
]

def add_missing_columns():
    try:
        with engine.connect() as conn:
            for col, typ in COLUMNS:
                try:
                    print(f"Kolon ekleniyor: {col} ({typ})...")
                    conn.execute(text(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {col} {typ};"))
                    conn.commit()
                    print(f"✅ {col} eklendi")
                except Exception as e:
                    print(f"❌ {col} eklenirken hata: {e}")
            
            print("\nTüm kolonlar kontrol edildi!")
            
    except Exception as e:
        print(f"Veritabanı bağlantı hatası: {e}")

def fix_timestamp_column():
    with engine.connect() as conn:
        print("[1] Kolon tipi kontrol ediliyor...")
        # Kolonun mevcut tipini kontrol et
        result = conn.execute(text("""
            SELECT data_type FROM information_schema.columns 
            WHERE table_name = 'signals' AND column_name = 'timestamp'
        """))
        dtype = result.scalar()
        print(f"Mevcut tip: {dtype}")
        if dtype != 'timestamp without time zone':
            print("[2] Kolon tipi dönüştürülüyor...")
            alter_sql = """
                ALTER TABLE signals 
                ALTER COLUMN timestamp TYPE timestamp 
                USING (CASE 
                    WHEN length(timestamp::text) > 10 THEN timestamp::timestamp 
                    ELSE to_timestamp(timestamp::int) 
                END)
            """
            conn.execute(text(alter_sql))
            print("[3] Kolon tipi başarıyla dönüştürüldü!")
        else:
            print("Zaten doğru tipte.")

if __name__ == "__main__":
    add_missing_columns()
    fix_timestamp_column() 