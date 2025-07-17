from sqlalchemy import create_engine, text
from config import Config

ALTERS = [
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS order_book_imbalance DECIMAL(10,4) DEFAULT NULL;",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS top_bid_walls TEXT DEFAULT NULL;",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS top_ask_walls TEXT DEFAULT NULL;",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS whale_direction_score DECIMAL(10,4) DEFAULT NULL;",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS volume_score DECIMAL(5,4) DEFAULT NULL;",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS momentum_score DECIMAL(5,4) DEFAULT NULL;",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pattern_score DECIMAL(5,4) DEFAULT NULL;",
]

def main():
    engine = create_engine(Config.DATABASE_URL)
    with engine.connect() as conn:
        for sql in ALTERS:
            try:
                print(f"Çalıştırılıyor: {sql}")
                conn.execute(text(sql))
                print("Başarılı.")
            except Exception as e:
                print(f"Hata: {e}")
        conn.commit()
    print("Tüm kolonlar kontrol edildi ve eklendi.")

if __name__ == "__main__":
    main() 