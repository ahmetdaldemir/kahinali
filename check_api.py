import requests
import json

def check_api():
    base_url = "http://127.0.0.1:5000"
    
    try:
        # 1. Tüm sinyalleri al
        print("1. Tüm sinyaller:")
        response = requests.get(f"{base_url}/api/signals")
        if response.status_code == 200:
            signals_data = response.json()
            
            # Yeni format kontrolü
            if isinstance(signals_data, dict) and 'signals' in signals_data:
                signals = signals_data['signals']
                pagination = signals_data.get('pagination', {})
                print(f"   Toplam {len(signals)} sinyal bulundu (sayfa {pagination.get('page', 1)})")
            else:
                # Eski format
                signals = signals_data if isinstance(signals_data, list) else []
                print(f"   Toplam {len(signals)} sinyal bulundu")
            
            # Coinlere göre grupla
            coin_counts = {}
            for signal in signals:
                if isinstance(signal, dict):
                    symbol = signal.get('symbol', 'Unknown')
                    coin_counts[symbol] = coin_counts.get(symbol, 0) + 1
            
            print("   Coinlere göre dağılım:")
            for coin, count in coin_counts.items():
                print(f"     {coin}: {count} sinyal")
            
            # Son 5 sinyali göster
            print("   Son 5 sinyal:")
            for signal in signals[:5]:
                if isinstance(signal, dict):
                    symbol = signal.get('symbol', 'Unknown')
                    direction = signal.get('direction', 'Unknown')
                    ai_score = signal.get('ai_score', 0)
                    timestamp = signal.get('timestamp', 'Unknown')
                    print(f"     {symbol} - {direction} - Skor: {ai_score:.2f} - {timestamp}")
        else:
            print(f"   ❌ API hatası: {response.status_code}")
        
        # 2. İstatistikler
        print("\n2. İstatistikler:")
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            if isinstance(stats, dict):
                latest_signals = stats.get('latest_signals', [])
                print(f"   Son sinyaller: {len(latest_signals)}")
                
                performance_summary = stats.get('performance_summary', {})
                if isinstance(performance_summary, dict):
                    print(f"   Performans özeti: {performance_summary}")
                else:
                    print(f"   Performans özeti: {performance_summary}")
            else:
                print(f"   ❌ Beklenmeyen format: {type(stats)}")
        else:
            print(f"   ❌ Stats API hatası: {response.status_code}")
        
        # 3. Performans
        print("\n3. Performans:")
        response = requests.get(f"{base_url}/api/performance")
        if response.status_code == 200:
            performance = response.json()
            if isinstance(performance, dict):
                print(f"   Performans verisi: {performance}")
            else:
                print(f"   Performans verisi: {performance}")
        else:
            print(f"   ❌ Performance API hatası: {response.status_code}")
        
        print("\n✓ API kontrolü tamamlandı!")
        
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_api() 