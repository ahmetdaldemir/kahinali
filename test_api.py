import requests
import json

# API endpoint'lerini test et
base_url = "http://localhost:5000"

# Sinyalleri al
print("=== Sinyaller API Test ===")
response = requests.get(f"{base_url}/api/signals?limit=5&page=1")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Toplam sinyal sayısı: {data.get('total', 'N/A')}")
    print(f"Sayfa: {data.get('page', 'N/A')}")
    print(f"Limit: {data.get('limit', 'N/A')}")
    print(f"Gelen sinyal sayısı: {len(data.get('signals', []))}")
    
    if data.get('signals'):
        print("\nİlk 3 sinyal:")
        for i, signal in enumerate(data['signals'][:3]):
            print(f"{i+1}. {signal.get('symbol', 'N/A')} - {signal.get('direction', 'N/A')} - {signal.get('timestamp', 'N/A')}")
else:
    print(f"Hata: {response.text}")

# Stats API test
print("\n=== Stats API Test ===")
response = requests.get(f"{base_url}/api/stats")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Toplam sinyal: {data.get('total_signals', 'N/A')}")
    print(f"Açık sinyal: {data.get('open_signals', 'N/A')}")
    print(f"Kapalı sinyal: {data.get('closed_signals', 'N/A')}")
else:
    print(f"Hata: {response.text}") 