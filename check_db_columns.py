import sqlalchemy
from sqlalchemy import text

# Veritabanı bağlantısı
DATABASE_URL = 'postgresql://laravel:secret@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Signals tablosundaki kolonları listele
        result = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'signals' 
            ORDER BY ordinal_position
        """))
        
        print("Signals tablosundaki kolonlar:")
        print("-" * 40)
        for row in result:
            print(f"{row[0]} ({row[1]})")
            
        # Eksik kolonları kontrol et
        required_columns = [
            'quality_score', 'market_regime', 'volatility_regime', 
            'volume_score', 'momentum_score', 'pattern_score', 
            'breakout_probability', 'risk_reward_ratio', 'confidence_level', 
            'signal_strength', 'market_sentiment'
        ]
        
        existing_columns = [row[0] for row in conn.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'signals'
        """))]
        
        print("\nEksik kolonlar:")
        print("-" * 20)
        missing = [col for col in required_columns if col not in existing_columns]
        if missing:
            for col in missing:
                print(f"❌ {col}")
        else:
            print("✅ Tüm kolonlar mevcut!")

except Exception as e:
    print(f"Hata: {e}") 