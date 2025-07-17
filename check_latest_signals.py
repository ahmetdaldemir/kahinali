#!/usr/bin/env python3
"""
En son üretilen sinyalleri kontrol et
"""

from modules.signal_manager import SignalManager
from datetime import datetime

def check_latest_signals():
    try:
        sm = SignalManager()
        signals = sm.get_signals()
        
        print("=== SON SİNYALLER ===")
        print(f"Toplam sinyal sayısı: {len(signals)}")
        print(f"Açık sinyal sayısı: {len(sm.get_open_signals())}")
        
        if not signals.empty:
            print("\nSon 5 sinyal:")
            for i, row in signals.tail(5).iterrows():
                print(f"{row['timestamp']} - {row['symbol']} - {row['direction']} - AI Skor: {row['ai_score']}")
        else:
            print("Hiç sinyal bulunamadı!")
        
        print("\n=== SİSTEM DURUMU ===")
        print(f"Son sinyal zamanı: {signals.iloc[-1]['timestamp'] if not signals.empty else 'Yok'}")
        print(f"Şu anki zaman: {sm.get_signals().iloc[-1]['timestamp'] if not sm.get_signals().empty else 'Yok'}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")

if __name__ == "__main__":
    check_latest_signals() 