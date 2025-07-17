import requests
import json

def test_detailed_analysis():
    try:
        # Sinyal 228 için detaylı analiz isteği
        response = requests.get('http://localhost:5000/api/signal/228/detailed-analysis')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Detaylı analiz başarılı!")
            print(f"Signal ID: {data.get('signal_id')}")
            print(f"Symbol: {data.get('signal_info', {}).get('symbol')}")
            print(f"Criteria Analysis Keys: {list(data.get('criteria_analysis', {}).keys())}")
            print(f"Production Reason: {data.get('production_reason', {}).get('total_percentage')}%")
            print(f"Investment Recommendations: {len(data.get('investment_recommendations', []))}")
            print(f"Risk Warnings: {len(data.get('risk_warnings', []))}")
            print(f"Summary: {data.get('summary', {}).get('overall_percentage')}%")
        else:
            print(f"✗ Hata: {response.status_code}")
            print(f"Hata mesajı: {response.text}")
            
    except Exception as e:
        print(f"✗ Bağlantı hatası: {e}")

if __name__ == "__main__":
    test_detailed_analysis() 