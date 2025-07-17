#!/usr/bin/env python3
"""
Debug script for target price fixing
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
    print("✅ Config loaded")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

try:
    from modules.signal_manager import SignalManager
    print("✅ SignalManager loaded")
except Exception as e:
    print(f"❌ SignalManager error: {e}")
    sys.exit(1)

def debug_target_prices():
    """Debug hedef fiyatları"""
    try:
        print("=== Debug Başlatılıyor ===")
        
        # PostgreSQL bağlantısı
        engine = create_engine(Config.DATABASE_URL)
        print("✅ Database connection created")
        
        # Sinyalleri kontrol et
        query = """
            SELECT id, symbol, direction, entry_price, ai_score, ta_strength, 
                   take_profit, stop_loss, support_level, resistance_level, target_time_hours
            FROM signals 
            ORDER BY created_at DESC 
            LIMIT 5
        """
        
        df = pd.read_sql(query, engine)
        print(f"✅ {len(df)} sinyal bulundu")
        
        if not df.empty:
            print("\nİlk 3 sinyal:")
            for idx, signal in df.head(3).iterrows():
                print(f"\nSinyal {signal['id']} ({signal['symbol']}):")
                print(f"  Entry: {signal['entry_price']}")
                print(f"  TP: {signal['take_profit']}")
                print(f"  SL: {signal['stop_loss']}")
                print(f"  Support: {signal['support_level']}")
                print(f"  Resistance: {signal['resistance_level']}")
                
                # Aynı mı kontrol et
                entry = signal['entry_price']
                tp = signal['take_profit']
                sl = signal['stop_loss']
                
                if tp == entry and sl == entry:
                    print("  ⚠️ Hedef fiyatlar entry_price ile aynı!")
                else:
                    print("  ✅ Hedef fiyatlar farklı")
        
        # SignalManager test
        signal_manager = SignalManager()
        print("✅ SignalManager instance created")
        
        # Test hesaplama
        if not df.empty:
            test_signal = df.iloc[0]
            signal_data = {
                'symbol': test_signal['symbol'],
                'direction': test_signal['direction'],
                'ai_score': test_signal['ai_score'] or 0.5,
                'ta_strength': test_signal['ta_strength'] or 0.5,
                'predicted_duration': '4-8 saat'
            }
            
            entry_price = test_signal['entry_price']
            print(f"\nTest hesaplama için: {signal_data['symbol']} @ {entry_price}")
            
            take_profit, stop_loss, support_level, resistance_level, target_time = signal_manager.calculate_target_levels(
                signal_data, entry_price
            )
            
            print(f"Yeni hesaplanan değerler:")
            print(f"  TP: {take_profit:.8f} ({((take_profit/entry_price-1)*100):.2f}%)")
            print(f"  SL: {stop_loss:.8f} ({((stop_loss/entry_price-1)*100):.2f}%)")
            print(f"  Support: {support_level:.8f} ({((support_level/entry_price-1)*100):.2f}%)")
            print(f"  Resistance: {resistance_level:.8f} ({((resistance_level/entry_price-1)*100):.2f}%)")
            print(f"  Target Time: {target_time:.1f} saat")
        
    except Exception as e:
        print(f"❌ Debug hatası: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_target_prices() 