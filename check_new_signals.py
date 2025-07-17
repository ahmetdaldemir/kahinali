import sqlalchemy
from sqlalchemy import text
from datetime import datetime, timedelta

# Veritabanı bağlantısı
DATABASE_URL = 'postgresql://postgres:3010726904@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

# Son 1 saatteki sinyalleri kontrol et
one_hour_ago = datetime.now() - timedelta(hours=1)

try:
    with engine.connect() as conn:
        # Son 1 saatteki sinyalleri al
        query = """
            SELECT id, symbol, direction, timestamp, ai_score, quality_score
            FROM signals 
            WHERE timestamp::timestamp >= :one_hour_ago
            ORDER BY timestamp DESC
            LIMIT 10
        """
        
        result = conn.execute(text(query), {'one_hour_ago': one_hour_ago})
        
        print(f"Son 1 saatteki sinyaller (saat {one_hour_ago.strftime('%H:%M')} sonrası):")
        print("-" * 60)
        
        signals_found = False
        for row in result:
            signals_found = True
            print(f"ID: {row[0]}, Symbol: {row[1]}, Direction: {row[2]}, Time: {row[3]}, AI Score: {row[4]:.3f}, Quality: {row[5]:.3f}")
        
        if not signals_found:
            print("Son 1 saatte sinyal bulunamadı!")
        
        # Toplam sinyal sayısı
        count_query = "SELECT COUNT(*) as total FROM signals"
        count_result = conn.execute(text(count_query))
        total_count = count_result.scalar()
        print(f"\nToplam sinyal sayısı: {total_count}")
        
        # En son sinyal tarihi
        last_query = "SELECT MAX(timestamp) as last_time FROM signals"
        last_result = conn.execute(text(last_query))
        last_time = last_result.scalar()
        print(f"En son sinyal tarihi: {last_time}")
        
except Exception as e:
    print(f"Hata: {e}") 