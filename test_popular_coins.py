import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from config import Config
import time

def test_popular_coins():
    print("Popüler 400 coin testi başlatılıyor...\n")
    
    collector = DataCollector()
    
    try:
        # Popüler coinleri al
        print("1. Popüler USDT çiftleri alınıyor...")
        start_time = time.time()
        pairs = collector.get_popular_usdt_pairs(max_pairs=400)
        end_time = time.time()
        
        print(f"   Süre: {end_time - start_time:.2f} saniye")
        print(f"   Toplam {len(pairs)} çift bulundu")
        
        if len(pairs) > 0:
            print(f"   İlk 20 çift: {pairs[:20]}")
            print(f"   Son 10 çift: {pairs[-10:]}")
        
        # İlk 5 coin için veri toplama testi
        print("\n2. İlk 5 coin için veri toplama testi...")
        test_pairs = pairs[:5] if len(pairs) >= 5 else pairs
        
        for pair in test_pairs:
            print(f"   {pair} için veri toplanıyor...")
            start_time = time.time()
            df = collector.get_historical_data(pair, '1h', limit=100)
            end_time = time.time()
            
            if not df.empty:
                print(f"     ✓ {len(df)} satır veri alındı ({end_time - start_time:.2f}s)")
                print(f"     Son fiyat: {df['close'].iloc[-1]:.8f}")
            else:
                print(f"     ✗ Veri alınamadı")
        
        print(f"\n✓ Test tamamlandı! {len(pairs)} popüler coin bulundu.")
        
    except Exception as e:
        print(f"✗ Test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_popular_coins() 