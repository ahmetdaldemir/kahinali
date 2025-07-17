#!/usr/bin/env python3
"""
API'den gelen sinyalleri kontrol etme scripti
"""

import requests
import json

def check_api_signals():
    """API'den gelen sinyalleri kontrol et"""
    try:
        response = requests.get('http://127.0.0.1:5000/api/signals?limit=10')
        data = response.json()
        
        print("Son 10 sinyal:")
        for signal in data['signals']:
            print(f"ID: {signal['id']}, Symbol: {signal['symbol']}, Breakout: {signal['predicted_breakout_threshold']}")
            
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    check_api_signals() 