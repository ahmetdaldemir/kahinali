#!/usr/bin/env python3
"""
Coin sayısı test scripti
"""

from modules.data_collector import DataCollector
from config import Config

def test_coin_count():
    print("=== COIN SAYISI TESTİ ===")
    
    # DataCollector oluştur
    dc = DataCollector()
    
    # 1. Tüm USDT çiftleri
    print("\n1. Tüm USDT çiftleri:")
    all_pairs = dc.get_usdt_pairs()
    print(f"   Toplam USDT çifti: {len(all_pairs)}")
    
    # 2. Popüler USDT çiftleri
    print("\n2. Popüler USDT çiftleri:")
    popular_pairs = dc.get_popular_usdt_pairs(max_pairs=Config.MAX_COINS_TO_TRACK)
    print(f"   Popüler USDT çifti: {len(popular_pairs)}")
    print(f"   Config.MAX_COINS_TO_TRACK: {Config.MAX_COINS_TO_TRACK}")
    
    # 3. İlk 20 popüler coin
    print("\n3. İlk 20 popüler coin:")
    for i, pair in enumerate(popular_pairs[:20]):
        print(f"   {i+1:2d}. {pair}")
    
    # 4. Son 10 popüler coin
    print("\n4. Son 10 popüler coin:")
    for i, pair in enumerate(popular_pairs[-10:]):
        print(f"   {len(popular_pairs)-9+i:2d}. {pair}")
    
    # 5. Sinyal üretim sistemi
    print("\n5. Sinyal üretim sistemi:")
    print(f"   Sistem şu anda ilk 400 coin'i işliyor")
    print(f"   Test pairs: {popular_pairs[:400]}")
    
    # 6. Potansiyel iyileştirme
    print("\n6. Potansiyel iyileştirme:")
    print(f"   Sistem {len(popular_pairs)} coin'i takip edebilir")
    print(f"   Şu anda 400 coin işliyor")
    print(f"   Tam kapasite kullanılıyor!")
    
    return len(popular_pairs)

if __name__ == "__main__":
    total_coins = test_coin_count()
    print(f"\n=== SONUÇ ===")
    print(f"Toplam {total_coins} coin için sinyal üretebilirsiniz!") 