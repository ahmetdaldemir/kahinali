#!/usr/bin/env python3
"""
KahinUltima Coin Listesi Validasyon Scripti
Bu script coin listenizi Binance'de gerçekten var olan USDT paritelerine göre filtreler.
"""

import requests
import os
from datetime import datetime

def print_header():
    print("=" * 60)
    print("🔎 KAHIN ULTIMA COIN LİSTESİ VALIDASYONU")
    print("=" * 60)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def get_binance_usdt_symbols():
    """Binance'den geçerli USDT paritelerini al"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = set()
        for s in data['symbols']:
            if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
                # Hem BTC/USDT hem de BTCUSDT formatlarını ekle
                symbols.add(s['baseAsset'] + '/USDT')
                symbols.add(s['baseAsset'] + 'USDT')
        return symbols
    except Exception as e:
        print(f"❌ Binance API hatası: {e}")
        return set()

def read_local_coin_list(filename='coin_list.txt'):
    """Yerel coin listesini oku"""
    if not os.path.exists(filename):
        print(f"❌ {filename} bulunamadı. Lütfen coin listenizi bu dosyaya ekleyin.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        coins = [line.strip() for line in f if line.strip()]
    return coins

def write_list(filename, items):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(item + '\n')
    print(f"✅ {filename} kaydedildi. ({len(items)} adet)")

def main():
    print_header()
    
    # Yerel coin listesini oku
    local_coins = read_local_coin_list()
    if not local_coins:
        return
    print(f"📋 Yerel coin sayısı: {len(local_coins)}")
    
    # Binance'den geçerli coinleri al
    binance_coins = get_binance_usdt_symbols()
    print(f"🌐 Binance USDT parite sayısı: {len(binance_coins)}")
    
    # Karşılaştır
    valid_coins = [coin for coin in local_coins if coin in binance_coins]
    invalid_coins = [coin for coin in local_coins if coin not in binance_coins]
    
    print(f"\n✅ Binance'de bulunan coin sayısı: {len(valid_coins)}")
    print(f"❌ Binance'de olmayan coin sayısı: {len(invalid_coins)}")
    
    if invalid_coins:
        print("\n❌ Binance'de olmayan coinler:")
        for coin in invalid_coins:
            print(f"   - {coin}")
    
    # Sonuçları kaydet
    write_list('valid_coins.txt', valid_coins)
    write_list('invalid_coins.txt', invalid_coins)
    
    print("\n🎯 Validasyon tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    main() 