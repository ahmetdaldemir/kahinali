import sqlalchemy
from sqlalchemy import text
from datetime import datetime

# Veritabanı bağlantısı
DATABASE_URL = 'postgresql://postgres:3010726904@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Sinyal 228'in tüm verilerini al
        query = """
            SELECT * FROM signals WHERE id = 228
        """
        
        result = conn.execute(text(query))
        row = result.fetchone()
        
        if row:
            print("Sinyal 228 bulundu:")
            print("-" * 50)
            
            # Tüm sütunları ve değerlerini yazdır
            for column in row._mapping.keys():
                value = row._mapping[column]
                print(f"{column}: {value} (tip: {type(value)})")
        else:
            print("Sinyal 228 bulunamadı!")
            
except Exception as e:
    print(f"Hata: {e}") 