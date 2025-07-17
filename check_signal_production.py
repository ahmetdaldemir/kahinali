#!/usr/bin/env python3
"""
Sinyal üretim sürecini detaylı kontrol etmek için script
"""

import sqlalchemy
from sqlalchemy import text
from datetime import datetime, timedelta
import json

# Veritabanı bağlantısı
DATABASE_URL = 'postgresql://postgres:3010726904@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

def check_signal_production():
    """Sinyal üretim sürecini kontrol et"""
    print("🔍 Sinyal üretim süreci kontrol ediliyor...")
    
    try:
        with engine.connect() as conn:
            # Son 2 saatteki sinyalleri al
            two_hours_ago = datetime.now() - timedelta(hours=2)
            
            query = """
                SELECT id, symbol, direction, timestamp, ai_score, quality_score, 
                       confidence, status, entry_price, stop_loss, take_profit
                FROM signals 
                WHERE timestamp::timestamp >= :two_hours_ago
                ORDER BY timestamp DESC
                LIMIT 20
            """
            
            result = conn.execute(text(query), {'two_hours_ago': two_hours_ago})
            
            print(f"Son 2 saatteki sinyaller (saat {two_hours_ago.strftime('%H:%M')} sonrası):")
            print("=" * 80)
            
            signals_found = False
            total_signals = 0
            long_signals = 0
            short_signals = 0
            neutral_signals = 0
            high_ai_signals = 0
            
            for row in result:
                signals_found = True
                total_signals += 1
                
                # Yön sayılarını hesapla
                if row[2] == 'LONG':
                    long_signals += 1
                elif row[2] == 'SHORT':
                    short_signals += 1
                else:
                    neutral_signals += 1
                
                # Yüksek AI skorlu sinyalleri say
                if row[4] and row[4] > 0.9:
                    high_ai_signals += 1
                
                print(f"ID: {row[0]}, Symbol: {row[1]}, Direction: {row[2]}, Time: {row[3]}")
                print(f"  AI Score: {row[4]:.3f}, Quality: {row[5]:.3f}")
                print(f"  Confidence: {row[6]:.3f}, Status: {row[7]}")
                if row[8]:
                    print(f"  Entry: {row[8]:.4f}, SL: {row[9]:.4f}, TP: {row[10]:.4f}")
                print("-" * 40)
            
            if not signals_found:
                print("Son 2 saatte sinyal bulunamadı!")
                return
            
            # İstatistikler
            print(f"\n📊 SİNYAL ÜRETİM İSTATİSTİKLERİ:")
            print(f"  Toplam sinyal: {total_signals}")
            print(f"  LONG sinyaller: {long_signals} ({long_signals/total_signals*100:.1f}%)")
            print(f"  SHORT sinyaller: {short_signals} ({short_signals/total_signals*100:.1f}%)")
            print(f"  NEUTRAL sinyaller: {neutral_signals} ({neutral_signals/total_signals*100:.1f}%)")
            print(f"  Yüksek AI skorlu (>0.9): {high_ai_signals} ({high_ai_signals/total_signals*100:.1f}%)")
            
            # Ortalama skorlar
            avg_query = """
                SELECT AVG(ai_score) as avg_ai, AVG(quality_score) as avg_quality
                FROM signals 
                WHERE timestamp::timestamp >= :two_hours_ago
            """
            
            avg_result = conn.execute(text(avg_query), {'two_hours_ago': two_hours_ago})
            avg_row = avg_result.fetchone()
            
            if avg_row:
                print(f"\n📈 ORTALAMA SKORLAR:")
                print(f"  Ortalama AI Score: {avg_row[0]:.3f}")
                print(f"  Ortalama Quality Score: {avg_row[1]:.3f}")
            
            # Son sinyal zamanı
            last_query = "SELECT MAX(timestamp) as last_time FROM signals"
            last_result = conn.execute(text(last_query))
            last_time = last_result.scalar()
            
            if last_time:
                time_diff = datetime.now() - last_time
                print(f"\n⏰ SON SİNYAL ZAMANI:")
                print(f"  Son sinyal: {last_time}")
                print(f"  Geçen süre: {time_diff}")
                
                if time_diff > timedelta(minutes=30):
                    print("  ⚠️ Son 30 dakikada sinyal üretilmemiş!")
                else:
                    print("  ✅ Sinyal üretimi aktif")
            
            # Toplam sinyal sayısı
            count_query = "SELECT COUNT(*) as total FROM signals"
            count_result = conn.execute(text(count_query))
            total_count = count_result.scalar()
            print(f"\n📊 Toplam sinyal sayısı: {total_count}")
            
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    check_signal_production() 