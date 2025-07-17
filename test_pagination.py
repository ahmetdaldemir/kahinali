import sys
import os
import time
import math
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from config import Config

def calculate_data_collection_time():
    """2 yıllık veri çekme süresini hesapla"""
    
    print("=== 2 YILLIK VERİ ÇEKME SÜRE HESAPLAMASI ===\n")
    
    # Parametreler
    YEARS = 2
    DAYS_PER_YEAR = 365
    TOTAL_DAYS = YEARS * DAYS_PER_YEAR
    
    # Timeframe'ler ve saatlik veri sayıları
    TIMEFRAMES = {
        '1h': 24,      # Günde 24 saat
        '4h': 6,       # Günde 6 veri
        '1d': 1        # Günde 1 veri
    }
    
    # Rate limit bilgileri (Binance API)
    RATE_LIMITS = {
        'requests_per_second': 10,  # Saniyede 10 istek
        'requests_per_minute': 1200,  # Dakikada 1200 istek
        'requests_per_hour': 72000,   # Saatte 72000 istek
        'delay_between_calls': 0.1,   # Her çağrı arası 0.1 saniye
        'max_per_call': 1000         # Her çağrıda maksimum 1000 veri
    }
    
    # Coin sayısı
    TOTAL_COINS = 400  # Sistemde desteklenen maksimum coin sayısı
    
    print(f"📊 HESAPLAMA PARAMETRELERİ:")
    print(f"   • Yıl sayısı: {YEARS}")
    print(f"   • Toplam gün: {TOTAL_DAYS}")
    print(f"   • Coin sayısı: {TOTAL_COINS}")
    print(f"   • Timeframe'ler: {list(TIMEFRAMES.keys())}")
    print(f"   • Rate limit: {RATE_LIMITS['requests_per_second']} istek/saniye")
    print(f"   • Çağrı arası bekleme: {RATE_LIMITS['delay_between_calls']} saniye")
    print(f"   • Maksimum veri/çağrı: {RATE_LIMITS['max_per_call']}")
    
    print(f"\n📈 VERİ MİKTARI HESAPLAMASI:")
    
    total_data_points = 0
    total_api_calls = 0
    
    for timeframe, daily_count in TIMEFRAMES.items():
        data_points = TOTAL_DAYS * daily_count
        api_calls = math.ceil(data_points / RATE_LIMITS['max_per_call'])
        
        print(f"   • {timeframe}: {data_points:,} veri noktası ({api_calls:,} API çağrısı)")
        total_data_points += data_points
        total_api_calls += api_calls
    
    print(f"   • TOPLAM: {total_data_points:,} veri noktası ({total_api_calls:,} API çağrısı)")
    
    # Tüm coinler için
    total_all_coins = total_api_calls * TOTAL_COINS
    print(f"   • {TOTAL_COINS} coin için: {total_all_coins:,} API çağrısı")
    
    print(f"\n⏱️ SÜRE HESAPLAMASI:")
    
    # Rate limit'e göre süre hesaplama
    calls_per_second = RATE_LIMITS['requests_per_second']
    total_seconds = total_all_coins / calls_per_second
    
    # Bekleme süreleri
    total_delay = total_all_coins * RATE_LIMITS['delay_between_calls']
    total_seconds_with_delay = total_seconds + total_delay
    
    # Zaman formatlarına çevir
    hours = total_seconds_with_delay / 3600
    days = hours / 24
    
    print(f"   • Sadece API çağrıları: {total_seconds/3600:.1f} saat ({total_seconds/86400:.1f} gün)")
    print(f"   • Bekleme süreleri dahil: {hours:.1f} saat ({days:.1f} gün)")
    
    # Gerçekçi tahmin (network latency, hatalar, vs.)
    realistic_multiplier = 1.5  # %50 ek süre
    realistic_hours = hours * realistic_multiplier
    realistic_days = days * realistic_multiplier
    
    print(f"   • Gerçekçi tahmin (hata payı dahil): {realistic_hours:.1f} saat ({realistic_days:.1f} gün)")
    
    print(f"\n🚀 OPTİMİZASYON ÖNERİLERİ:")
    
    # Paralel işleme ile süre azaltma
    parallel_processes = 5  # 5 paralel işlem
    parallel_hours = realistic_hours / parallel_processes
    parallel_days = parallel_hours / 24
    
    print(f"   • {parallel_processes} paralel işlem ile: {parallel_hours:.1f} saat ({parallel_days:.1f} gün)")
    
    # Batch işleme ile süre azaltma
    batch_size = 10  # 10 coin'lik batch'ler
    batch_hours = parallel_hours / batch_size
    batch_days = batch_hours / 24
    
    print(f"   • Batch işleme ({batch_size} coin/batch): {batch_hours:.1f} saat ({batch_days:.1f} gün)")
    
    print(f"\n💾 VERİ DEPOLAMA:")
    
    # Veri boyutu hesaplama (yaklaşık)
    bytes_per_record = 100  # Her veri kaydı için yaklaşık 100 byte
    total_bytes = total_data_points * TOTAL_COINS * bytes_per_record
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024
    
    print(f"   • Toplam veri boyutu: {total_mb:.1f} MB ({total_gb:.2f} GB)")
    
    print(f"\n⚠️ DİKKAT EDİLMESİ GEREKENLER:")
    print(f"   • Binance API rate limit'leri aşılmamalı")
    print(f"   • Network bağlantısı stabil olmalı")
    print(f"   • Yeterli disk alanı olmalı ({total_gb:.1f} GB)")
    print(f"   • Hata durumunda retry mekanizması olmalı")
    print(f"   • Veri bütünlüğü kontrol edilmeli")
    
    print(f"\n📋 ÖZET:")
    print(f"   • En hızlı süre: {batch_days:.1f} gün")
    print(f"   • Ortalama süre: {realistic_days:.1f} gün")
    print(f"   • En yavaş süre: {days:.1f} gün")
    
    return {
        'fastest_days': batch_days,
        'average_days': realistic_days,
        'slowest_days': days,
        'total_data_points': total_data_points * TOTAL_COINS,
        'total_storage_gb': total_gb
    }

def test_single_coin_speed():
    """Tek coin için gerçek veri çekme hızını test et"""
    
    print(f"\n=== TEK COİN HIZ TESTİ ===")
    
    collector = DataCollector()
    
    # Test coin'i
    test_symbol = 'BTC/USDT'
    test_timeframe = '1h'
    test_limit = 100  # 100 veri noktası
    
    print(f"Test ediliyor: {test_symbol} ({test_timeframe}) - {test_limit} veri")
    
    start_time = time.time()
    
    try:
        data = collector.get_historical_data(test_symbol, test_timeframe, test_limit)
        end_time = time.time()
        
        duration = end_time - start_time
        actual_records = len(data) if not data.empty else 0
        
        print(f"✓ Başarılı!")
        print(f"   • Süre: {duration:.2f} saniye")
        print(f"   • Alınan veri: {actual_records} kayıt")
        print(f"   • Hız: {actual_records/duration:.1f} veri/saniye")
        
        # Bu hıza göre 2 yıllık hesaplama
        if actual_records > 0:
            records_per_second = actual_records / duration
            total_records_2y = 2 * 365 * 24  # 2 yıl, 1h timeframe
            estimated_seconds = total_records_2y / records_per_second
            estimated_hours = estimated_seconds / 3600
            
            print(f"   • 2 yıllık tahmin: {estimated_hours:.1f} saat")
        
        return duration, actual_records
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return None, 0

if __name__ == "__main__":
    # Teorik hesaplama
    results = calculate_data_collection_time()
    
    # Gerçek hız testi
    test_single_coin_speed()
    
    print(f"\n🎯 SONUÇ:")
    print(f"2 yıllık veri çekme işlemi yaklaşık {results['average_days']:.1f} gün sürecektir.")
    print(f"Bu süre optimizasyonlarla {results['fastest_days']:.1f} güne düşürülebilir.")
