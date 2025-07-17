import os
import sys
import time
import ccxt
from datetime import datetime

def fix_binance_api_issues():
    """Binance API hatalarını düzelt"""
    
    print("============================================================")
    print("🔧 BİNANCE API HATALARINI DÜZELTME")
    print("============================================================")
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Config'den API anahtarlarını al
        from config import Config
        
        # Binance API bağlantısını test et
        print("🔍 Binance API bağlantısı test ediliyor...")
        
        # Spot API test
        binance_spot = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET_KEY,
            'timeout': 30000,  # 30 saniye timeout
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        # Spot API test
        try:
            print("📊 Spot API test ediliyor...")
            spot_ticker = binance_spot.fetch_ticker('BTC/USDT')
            print(f"✅ Spot API çalışıyor - BTC fiyatı: ${spot_ticker['last']:,.2f}")
        except Exception as e:
            print(f"❌ Spot API hatası: {e}")
        
        # USDT çiftlerini al
        print("\n🔄 USDT çiftleri alınıyor...")
        try:
            # Spot marketlerden USDT çiftlerini al
            spot_markets = binance_spot.load_markets()
            usdt_pairs = [symbol for symbol in spot_markets.keys() if symbol.endswith('/USDT')]
            
            print(f"✅ {len(usdt_pairs)} USDT çifti bulundu")
            
            # İlk 10 çifti test et
            test_pairs = usdt_pairs[:10]
            print(f"🧪 İlk 10 çift test ediliyor: {test_pairs}")
            
            for pair in test_pairs:
                try:
                    ticker = binance_spot.fetch_ticker(pair)
                    print(f"✅ {pair}: ${ticker['last']:,.4f}")
                    time.sleep(0.1)  # Rate limit
                except Exception as e:
                    print(f"❌ {pair} hatası: {e}")
            
        except Exception as e:
            print(f"❌ USDT çiftleri alınamadı: {e}")
        
        print("\n✅ Binance API düzeltme tamamlandı!")
        return True
        
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        return False

if __name__ == "__main__":
    fix_binance_api_issues() 