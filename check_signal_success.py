#!/usr/bin/env python3
"""
Kapanan sinyallerin başarı oranını analiz eden script
"""

import sqlalchemy
from sqlalchemy import text
from datetime import datetime, timedelta

DATABASE_URL = 'postgresql://postgres:3010726904@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

def check_signal_success():
    print("📊 Kapanan sinyallerin başarı oranı analiz ediliyor...")
    try:
        with engine.connect() as conn:
            # Son 90 günde kapanan sinyalleri al
            query = """
                SELECT id, symbol, direction, timestamp, result, realized_gain
                FROM signals
                WHERE result IS NOT NULL AND result != '' AND timestamp::timestamp >= :since
                ORDER BY timestamp DESC
            """
            since = datetime.now() - timedelta(days=90)
            result = conn.execute(text(query), {'since': since})
            
            total = 0
            success = 0
            fail = 0
            timeout = 0
            profit_sum = 0
            for row in result:
                total += 1
                if row[4] and str(row[4]).upper() == 'SUCCESS':
                    success += 1
                elif row[4] and str(row[4]).upper() == 'TIMEOUT':
                    timeout += 1
                else:
                    fail += 1
                if row[5]:
                    try:
                        profit_sum += float(row[5])
                    except:
                        pass
            
            print(f"Son 90 günde kapanan toplam sinyal: {total}")
            print(f"  Başarılı (SUCCESS): {success}")
            print(f"  Başarısız (FAIL/STOP): {fail}")
            print(f"  Süresi dolan (TIMEOUT): {timeout}")
            if total > 0:
                print(f"  Başarı oranı: %{(success/total)*100:.1f}")
                print(f"  Ortalama realized gain: {profit_sum/total:.4f}")
            else:
                print("  Kapanan sinyal yok!")
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    check_signal_success() 