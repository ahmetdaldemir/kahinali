#!/usr/bin/env python3
"""
Sistem Durumu Raporu
"""

import requests
import json
from datetime import datetime

def get_system_status():
    """Sistem durumunu al"""
    try:
        # Temel istatistikler
        stats_response = requests.get('http://localhost:5000/api/stats')
        stats = stats_response.json() if stats_response.status_code == 200 else {}
        
        # Sistem sağlığı
        health_response = requests.get('http://localhost:5000/api/system/health')
        health = health_response.json() if health_response.status_code == 200 else {}
        
        # Performans istatistikleri
        perf_response = requests.get('http://localhost:5000/api/performance/advanced-stats')
        performance = perf_response.json() if perf_response.status_code == 200 else {}
        
        return {
            'timestamp': str(datetime.now()),
            'stats': stats,
            'health': health,
            'performance': performance
        }
    except Exception as e:
        return {'error': str(e)}

def print_status_report():
    """Durum raporunu yazdır"""
    print("🎯 KAHİN ULTİMA SİSTEM DURUM RAPORU")
    print("=" * 60)
    print(f"📅 Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    status = get_system_status()
    
    if 'error' in status:
        print(f"❌ Hata: {status['error']}")
        return
    
    # Temel İstatistikler
    stats = status.get('stats', {})
    print("📊 TEMEL İSTATİSTİKLER")
    print("-" * 30)
    print(f"   Toplam Sinyal: {stats.get('total_signals', 0)}")
    print(f"   Açık Sinyal: {stats.get('open_signals', 0)}")
    print(f"   Başarı Oranı: %{stats.get('success_rate', 0):.2f}")
    
    avg_profit = stats.get('avg_profit', 0)
    if avg_profit is not None:
        print(f"   Ortalama Kar: %{avg_profit:.2f}")
    else:
        print(f"   Ortalama Kar: %0.00")
    
    print(f"   Sistem Durumu: {stats.get('system_status', 'unknown')}")
    print()
    
    # Sistem Sağlığı
    health = status.get('health', {})
    overall_status = health.get('overall_status', 'UNKNOWN')
    print("🏥 SİSTEM SAĞLIĞI")
    print("-" * 30)
    print(f"   Genel Durum: {overall_status}")
    
    checks = health.get('checks', {})
    for check_name, check_result in checks.items():
        status_icon = "✅" if check_result.get('status') == 'HEALTHY' else "⚠️" if check_result.get('status') == 'WARNING' else "❌"
        print(f"   {status_icon} {check_name}: {check_result.get('status', 'UNKNOWN')}")
    
    print()
    
    # Performans İstatistikleri
    performance = status.get('performance', {})
    if performance:
        print("📈 PERFORMANS İSTATİSTİKLERİ")
        print("-" * 30)
        print(f"   Toplam İşlem: {performance.get('total_trades', 0)}")
        print(f"   Kazançlı İşlem: {performance.get('profitable_trades', 0)}")
        print(f"   Zararlı İşlem: {performance.get('losing_trades', 0)}")
        print(f"   Net Kar: %{performance.get('net_profit', 0):.2f}")
        print()
    
    # Öneriler
    recommendations = health.get('recommendations', [])
    if recommendations:
        print("💡 ÖNERİLER")
        print("-" * 30)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        print()
    
    # Sistem Durumu
    print("🎯 SİSTEM DURUMU")
    print("-" * 30)
    if overall_status == 'HEALTHY':
        print("   ✅ SİSTEM SAĞLIKLI VE ÇALIŞIYOR")
    elif overall_status == 'WARNING':
        print("   ⚠️ SİSTEM UYARI DURUMUNDA")
    elif overall_status == 'CRITICAL':
        print("   ❌ SİSTEM KRİTİK DURUMDA")
    else:
        print("   ❓ SİSTEM DURUMU BİLİNMEYEN")
    
    print()
    print("🌐 Web Arayüzü: http://localhost:5000")
    print("=" * 60)

if __name__ == "__main__":
    print_status_report() 