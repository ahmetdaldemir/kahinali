#!/usr/bin/env python3
"""
API bağlantı test scripti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
import requests

def test_binance_api():
    """Binance API bağlantısını test et"""
    print("=== BİNANCE API BAĞLANTI TESTİ ===")
    
    # 1. Basit ping testi
    try:
        response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
        print(f"✅ Ping testi: {response.status_code}")
    except Exception as e:
        print(f"❌ Ping testi hatası: {e}")
        return
    
    # 2. Exchange info testi
    try:
        response = requests.get('https://api.binance.com/api/v3/exchangeInfo', timeout=10)
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
        print(f"✅ Exchange info: {len(symbols)} USDT çifti bulundu")
    except Exception as e:
        print(f"❌ Exchange info hatası: {e}")
        return
    
    # 3. DataCollector testi
    print("\n=== DATACOLLECTOR TESTİ ===")
    try:
        dc = DataCollector()
        
        # 400 coin iste
        pairs = dc.get_popular_usdt_pairs(max_pairs=400)
        print(f"📊 Dönen coin sayısı: {len(pairs)}")
        print(f"📋 İlk 10 coin: {pairs[:10]}")
        print(f"📋 Son 10 coin: {pairs[-10:]}")
        
        if len(pairs) < 400:
            print(f"⚠️ Sadece {len(pairs)} coin döndü, 400 bekleniyordu")
        else:
            print(f"✅ 400 coin başarıyla alındı!")
            
    except Exception as e:
        print(f"❌ DataCollector hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_binance_api() 