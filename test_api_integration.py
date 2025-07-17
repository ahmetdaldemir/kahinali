#!/usr/bin/env python3
"""
API entegrasyonunu test etmek için script
"""

import requests
import json

def test_api_integration():
    """API entegrasyonunu test et"""
    print("🔧 API entegrasyonu test ediliyor...")
    
    try:
        # API'nin çalışıp çalışmadığını kontrol et
        base_url = "http://localhost:5000"
        
        # Stats endpoint'i test et
        print("📊 Stats endpoint test ediliyor...")
        response = requests.get(f"{base_url}/api/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats API çalışıyor - Toplam sinyal: {stats.get('total_signals', 0)}")
        else:
            print(f"❌ Stats API hatası: {response.status_code}")
            return
        
        # Sinyaller endpoint'i test et
        print("📊 Sinyaller endpoint test ediliyor...")
        response = requests.get(f"{base_url}/api/signals?limit=5", timeout=10)
        if response.status_code == 200:
            signals = response.json()
            print(f"✅ Sinyaller API çalışıyor - {len(signals.get('signals', []))} sinyal")
        else:
            print(f"❌ Sinyaller API hatası: {response.status_code}")
            return
        
        # Detaylı analiz endpoint'i test et (yeni teknik analiz ile)
        print("📊 Detaylı analiz endpoint test ediliyor...")
        response = requests.get(f"{base_url}/api/analysis/detailed/ETH", timeout=10)
        if response.status_code == 200:
            analysis = response.json()
            print("✅ Detaylı analiz API çalışıyor")
            
            # Yeni teknik analiz sonuçlarını kontrol et
            ta_indicators = analysis.get('technical_indicators', {})
            if ta_indicators:
                print("✅ Yeni teknik analiz sonuçları mevcut:")
                print(f"  - EMA değerleri: {len(ta_indicators.get('ema', {}))} adet")
                print(f"  - RSI değerleri: {len(ta_indicators.get('rsi', {}))} adet")
                print(f"  - MACD: {'Mevcut' if 'macd' in ta_indicators else 'Yok'}")
                print(f"  - Bollinger: {'Mevcut' if 'bollinger' in ta_indicators else 'Yok'}")
                print(f"  - ADX: {'Mevcut' if 'adx' in ta_indicators else 'Yok'}")
                print(f"  - Patterns: {len(ta_indicators.get('patterns', {}))} adet")
            else:
                print("❌ Teknik analiz sonuçları bulunamadı")
        else:
            print(f"❌ Detaylı analiz API hatası: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return
        
        print("\n✅ Tüm API entegrasyon testleri başarılı!")
        
    except requests.exceptions.ConnectionError:
        print("❌ API sunucusu çalışmıyor. Lütfen 'python app/web.py' komutunu çalıştırın.")
    except Exception as e:
        print(f"❌ API test hatası: {e}")

if __name__ == "__main__":
    test_api_integration() 