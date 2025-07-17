#!/usr/bin/env python3
"""
Sinyal detaylarını kontrol et
"""

import sqlalchemy
from sqlalchemy import text
import pandas as pd
from sqlalchemy import create_engine
from config import Config

# Veritabanı bağlantısı
DATABASE_URL = 'postgresql://postgres:3010726904@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

def check_signal_details():
    """Sinyal detaylarını kontrol et"""
    try:
        with engine.connect() as conn:
            # Son 3 sinyalin detaylarını al
            query = """
                SELECT id, symbol, entry_price, exit_price, take_profit, stop_loss, 
                       support_level, resistance_level, target_time_hours, result,
                       predicted_breakout_threshold, predicted_breakout_time_hours
                FROM signals 
                ORDER BY id DESC
                LIMIT 3
            """
            
            result = conn.execute(text(query))
            
            print("Son 3 sinyalin detayları:")
            print("-" * 80)
            
            for row in result:
                print(f"ID: {row[0]}, Symbol: {row[1]}")
                print(f"  Entry Price: {row[2]}")
                print(f"  Exit Price: {row[3]}")
                print(f"  Take Profit: {row[4]}")
                print(f"  Stop Loss: {row[5]}")
                print(f"  Support: {row[6]}")
                print(f"  Resistance: {row[7]}")
                print(f"  Target Time: {row[8]}")
                print(f"  Breakout Threshold: {row[9]}")
                print(f"  Breakout Time: {row[10]}")
                print(f"  Result: {row[11]}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Hata: {e}")

def check_breakout_threshold():
    """Son 5 sinyalde predicted_breakout_threshold alanını kontrol et"""
    try:
        engine = create_engine(Config.DATABASE_URL)
        query = """
            SELECT id, symbol, predicted_breakout_threshold 
            FROM signals 
            ORDER BY id DESC 
            LIMIT 5
        """
        df = pd.read_sql(query, engine)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    check_signal_details()
    print("\n--- Son 5 sinyalde breakout threshold ---")
    check_breakout_threshold() 