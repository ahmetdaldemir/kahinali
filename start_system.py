#!/usr/bin/env python3
"""
Kahin Ultima Trading System - Başlatma Scripti
"""

import sys
import os
import logging
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    print("🚀 Kahin Ultima Trading System başlatılıyor...")
    print(f"⏰ Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Sistem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"❌ Sistem başlatılırken hata oluştu: {e}")
        logging.error(f"Sistem başlatma hatası: {e}")
        sys.exit(1) 