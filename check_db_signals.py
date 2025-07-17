#!/usr/bin/env python3
"""
Veritabanındaki sinyalleri kontrol et
"""

import sqlalchemy
import pandas as pd
from config import Config

def check_signals():
    try:
        engine = sqlalchemy.create_engine(Config.DATABASE_URL)
        df = pd.read_sql('SELECT * FROM signals ORDER BY timestamp DESC LIMIT 5', engine)
        
        print(f"📊 {len(df)} sinyal bulundu")
        print("=" * 50)
        
        for idx, signal in df.iterrows():
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal['entry_price'] or signal['price']
            ai_score = signal['ai_score'] or 0.5
            ta_strength = signal['ta_strength'] or 0.5
            
            print(f"\n{symbol} ({direction})")
            print(f"  Giriş Fiyatı: {entry_price:.8f}")
            print(f"  AI Skoru: {ai_score:.2f}")
            print(f"  TA Gücü: {ta_strength:.2f}")
            
            if entry_price and entry_price > 0:
                # ATR hesaplama
                atr = entry_price * 0.02
                
                if direction == 'LONG':
                    tp = entry_price + (atr * 2.5)
                    sl = entry_price - (atr * 1.5)
                    print(f"  Take Profit: {tp:.8f} (+{((tp/entry_price-1)*100):.2f}%)")
                    print(f"  Stop Loss: {sl:.8f} ({((sl/entry_price-1)*100):.2f}%)")
                elif direction == 'SHORT':
                    tp = entry_price - (atr * 2.5)
                    sl = entry_price + (atr * 1.5)
                    print(f"  Take Profit: {tp:.8f} ({((tp/entry_price-1)*100):.2f}%)")
                    print(f"  Stop Loss: {sl:.8f} (+{((sl/entry_price-1)*100):.2f}%)")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    check_signals() 