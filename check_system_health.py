#!/usr/bin/env python3
"""
Sistem Sağlığı Kontrol Scripti
"""

import sys
import os
import logging
from datetime import datetime

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.performance import PerformanceAnalyzer
    from config import Config
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🏥 KAHİN ULTİMA SİSTEM SAĞLIĞI KONTROLÜ")
    print("=" * 50)
    
    try:
        # Performans analizörü oluştur
        analyzer = PerformanceAnalyzer()
        
        # Sistem sağlığı kontrolü
        print("🔍 Sistem sağlığı kontrol ediliyor...")
        health_status = analyzer.system_health_check()
        
        # Sonuçları göster
        overall_status = health_status.get('overall_status', 'UNKNOWN')
        print(f"\n📊 Genel Durum: {overall_status}")
        
        # Detaylı kontroller
        checks = health_status.get('checks', {})
        for check_name, check_result in checks.items():
            status = check_result.get('status', 'UNKNOWN')
            if status == 'HEALTHY':
                print(f"   ✅ {check_name}: İYİ")
            elif status == 'WARNING':
                print(f"   ⚠️ {check_name}: UYARI")
            elif status == 'FAILED':
                print(f"   ❌ {check_name}: HATA")
            
            # Detayları göster
            for key, value in check_result.items():
                if key != 'status':
                    print(f"      - {key}: {value}")
        
        # Öneriler
        recommendations = health_status.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Öneriler:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Sistem optimizasyonu
        print(f"\n🔧 Sistem optimizasyonu başlatılıyor...")
        optimization_result = analyzer.auto_optimize_system()
        
        if optimization_result['success']:
            print("✅ Sistem optimizasyonu tamamlandı")
            for opt in optimization_result.get('optimizations_applied', []):
                print(f"   - {opt}")
        else:
            print("⚠️ Sistem optimizasyonu başarısız")
        
        print("\n" + "=" * 50)
        print("🎯 SİSTEM HAZIR!")
        print(f"🌐 Web arayüzü: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sistem sağlığı kontrolü hatası: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 