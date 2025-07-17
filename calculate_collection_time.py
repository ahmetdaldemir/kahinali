import sys
import os
import time
import math
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector

def calculate_detailed_collection_time():
    """400 coin için tüm zaman dilimlerinde 2 yıllık veri çekme süresini hesapla"""
    
    print("=== 400 COİN - TÜM ZAMAN DİLİMLERİ - 2 YILLIK VERİ ÇEKME SÜRE HESAPLAMASI ===\n")
    
    # Parametreler
    YEARS = 2
    DAYS_PER_YEAR = 365
    TOTAL_DAYS = YEARS * DAYS_PER_YEAR
    TOTAL_COINS = 400
    
    # Tüm zaman dilimleri ve günlük veri sayıları
    TIMEFRAMES = {
        '5m': 288,    # 24 * 60 / 5 = 288 veri/gün
        '15m': 96,    # 24 * 60 / 15 = 96 veri/gün
        '1h': 24,     # 24 veri/gün
        '4h': 6,      # 6 veri/gün
        '1d': 1       # 1 veri/gün
    }
    
    # Binance API Rate Limits
    RATE_LIMITS = {
        'requests_per_second': 10,      # Saniyede 10 istek
        'requests_per_minute': 1200,    # Dakikada 1200 istek
        'requests_per_hour': 72000,     # Saatte 72000 istek
        'delay_between_calls': 0.1,     # Her çağrı arası 0.1 saniye
        'max_per_call': 1000,           # Her çağrıda maksimum 1000 veri
        'realistic_delay': 0.2          # Gerçekçi bekleme süresi
    }
    
    print(f"📊 HESAPLAMA PARAMETRELERİ:")
    print(f"   • Yıl sayısı: {YEARS}")
    print(f"   • Toplam gün: {TOTAL_DAYS:,}")
    print(f"   • Coin sayısı: {TOTAL_COINS:,}")
    print(f"   • Zaman dilimleri: {list(TIMEFRAMES.keys())}")
    print(f"   • Rate limit: {RATE_LIMITS['requests_per_second']} istek/saniye")
    print(f"   • Çağrı arası bekleme: {RATE_LIMITS['realistic_delay']} saniye")
    print(f"   • Maksimum veri/çağrı: {RATE_LIMITS['max_per_call']:,}")
    
    print(f"\n📈 VERİ MİKTARI HESAPLAMASI:")
    
    total_data_points = 0
    total_api_calls = 0
    timeframe_details = {}
    
    for timeframe, daily_count in TIMEFRAMES.items():
        # 2 yıl için gerçek veri noktası sayısı
        data_points = TOTAL_DAYS * daily_count
        
        # Bu veriyi almak için kaç API çağrısı gerekiyor?
        # Her çağrıda maksimum 1000 veri alabiliyoruz
        api_calls = math.ceil(data_points / RATE_LIMITS['max_per_call'])
        
        timeframe_details[timeframe] = {
            'data_points': data_points,
            'api_calls': api_calls,
            'daily_count': daily_count,
            'actual_limit_per_call': min(RATE_LIMITS['max_per_call'], data_points)
        }
        
        print(f"   • {timeframe}: {data_points:,} veri noktası ({api_calls:,} API çağrısı)")
        print(f"     - Günde {daily_count} veri")
        print(f"     - 2 yılda {data_points:,} veri")
        print(f"     - {api_calls} API çağrısı gerekiyor")
        
        total_data_points += data_points
        total_api_calls += api_calls
    
    print(f"\n   • TOPLAM: {total_data_points:,} veri noktası ({total_api_calls:,} API çağrısı)")
    
    # Tüm coinler için
    total_all_coins = total_api_calls * TOTAL_COINS
    print(f"   • {TOTAL_COINS:,} coin için: {total_all_coins:,} API çağrısı")
    
    print(f"\n⏱️ SÜRE HESAPLAMASI:")
    
    # 1. Sadece API çağrıları (rate limit'e göre)
    calls_per_second = RATE_LIMITS['requests_per_second']
    api_only_seconds = total_all_coins / calls_per_second
    api_only_hours = api_only_seconds / 3600
    api_only_days = api_only_hours / 24
    
    print(f"   1. Sadece API çağrıları:")
    print(f"      • {api_only_seconds:,.0f} saniye")
    print(f"      • {api_only_hours:.1f} saat")
    print(f"      • {api_only_days:.1f} gün")
    
    # 2. Bekleme süreleri dahil
    total_delay = total_all_coins * RATE_LIMITS['realistic_delay']
    with_delay_seconds = api_only_seconds + total_delay
    with_delay_hours = with_delay_seconds / 3600
    with_delay_days = with_delay_hours / 24
    
    print(f"\n   2. Bekleme süreleri dahil:")
    print(f"      • {with_delay_seconds:,.0f} saniye")
    print(f"      • {with_delay_hours:.1f} saat")
    print(f"      • {with_delay_days:.1f} gün")
    
    # 3. Gerçekçi tahmin (network latency, hatalar, vs.)
    realistic_multiplier = 1.5  # %50 ek süre
    realistic_seconds = with_delay_seconds * realistic_multiplier
    realistic_hours = realistic_seconds / 3600
    realistic_days = realistic_hours / 24
    
    print(f"\n   3. Gerçekçi tahmin (hata payı dahil):")
    print(f"      • {realistic_seconds:,.0f} saniye")
    print(f"      • {realistic_hours:.1f} saat")
    print(f"      • {realistic_days:.1f} gün")
    
    print(f"\n🚀 OPTİMİZASYON SENARYOLARI:")
    
    # Paralel işleme senaryoları
    parallel_scenarios = [1, 2, 5, 10, 20]
    
    for workers in parallel_scenarios:
        parallel_hours = realistic_hours / workers
        parallel_days = parallel_hours / 24
        
        print(f"   • {workers} paralel işlem: {parallel_hours:.1f} saat ({parallel_days:.1f} gün)")
    
    print(f"\n💾 VERİ DEPOLAMA:")
    
    # Veri boyutu hesaplama
    bytes_per_record = 150  # Her veri kaydı için yaklaşık 150 byte (timestamp, OHLCV, vs.)
    total_bytes = total_data_points * TOTAL_COINS * bytes_per_record
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024
    total_tb = total_gb / 1024
    
    print(f"   • Toplam veri boyutu: {total_mb:,.0f} MB ({total_gb:.1f} GB)")
    if total_gb > 1024:
        print(f"   • Büyük veri: {total_tb:.2f} TB")
    
    print(f"\n📊 DETAYLI ZAMAN DİLİMİ ANALİZİ:")
    
    for timeframe, details in timeframe_details.items():
        data_points = details['data_points']
        api_calls = details['api_calls']
        
        # Tek coin için süre
        single_coin_seconds = (api_calls / calls_per_second) + (api_calls * RATE_LIMITS['realistic_delay'])
        single_coin_minutes = single_coin_seconds / 60
        
        # Tüm coinler için süre
        all_coins_seconds = single_coin_seconds * TOTAL_COINS
        all_coins_hours = all_coins_seconds / 3600
        
        print(f"   • {timeframe}:")
        print(f"     - Veri: {data_points:,} nokta")
        print(f"     - API çağrısı: {api_calls:,}")
        print(f"     - Tek coin: {single_coin_minutes:.1f} dakika")
        print(f"     - Tüm coinler: {all_coins_hours:.1f} saat")
    
    print(f"\n⚠️ DİKKAT EDİLMESİ GEREKENLER:")
    print(f"   • Binance API rate limit'leri aşılmamalı")
    print(f"   • Network bağlantısı stabil olmalı")
    print(f"   • Yeterli disk alanı olmalı ({total_gb:.1f} GB)")
    print(f"   • Hata durumunda retry mekanizması olmalı")
    print(f"   • Veri bütünlüğü kontrol edilmeli")
    print(f"   • İşlem kesintisiz devam etmeli")
    
    print(f"\n🎯 ÖNERİLER:")
    print(f"   • 5-10 paralel işlem kullanın")
    print(f"   • Batch işleme yapın (10-20 coin/grup)")
    print(f"   • İlerleme takibi yapın")
    print(f"   • Hata logları tutun")
    print(f"   • Düzenli backup alın")
    
    print(f"\n📋 ÖZET:")
    print(f"   • En hızlı süre (20 paralel): {realistic_hours/20:.1f} saat")
    print(f"   • Ortalama süre (5 paralel): {realistic_hours/5:.1f} saat")
    print(f"   • En yavaş süre (tek işlem): {realistic_hours:.1f} saat")
    print(f"   • Veri boyutu: {total_gb:.1f} GB")
    
    return {
        'fastest_hours': realistic_hours/20,
        'average_hours': realistic_hours/5,
        'slowest_hours': realistic_hours,
        'total_data_points': total_data_points * TOTAL_COINS,
        'total_storage_gb': total_gb,
        'total_api_calls': total_all_coins
    }

def test_real_speed():
    """Gerçek hız testi yap"""
    
    print(f"\n=== GERÇEK HIZ TESTİ ===")
    
    collector = DataCollector()
    
    # Test parametreleri
    test_symbol = 'BTC/USDT'
    test_timeframes = ['1h', '4h', '1d']
    test_limit = 1000  # Maksimum veri al
    
    total_time = 0
    total_records = 0
    
    for timeframe in test_timeframes:
        print(f"\nTest ediliyor: {test_symbol} ({timeframe}) - {test_limit} veri")
        
        start_time = time.time()
        
        try:
            data = collector.get_historical_data(test_symbol, timeframe, test_limit)
            end_time = time.time()
            
            duration = end_time - start_time
            actual_records = len(data) if not data.empty else 0
            
            total_time += duration
            total_records += actual_records
            
            print(f"✓ Başarılı!")
            print(f"   • Süre: {duration:.2f} saniye")
            print(f"   • Alınan veri: {actual_records} kayıt")
            print(f"   • Hız: {actual_records/duration:.1f} veri/saniye")
            
        except Exception as e:
            print(f"✗ Hata: {e}")
    
    if total_records > 0:
        avg_speed = total_records / total_time
        print(f"\n📊 GENEL TEST SONUCU:")
        print(f"   • Toplam süre: {total_time:.2f} saniye")
        print(f"   • Toplam veri: {total_records} kayıt")
        print(f"   • Ortalama hız: {avg_speed:.1f} veri/saniye")
        
        # Bu hıza göre 2 yıllık hesaplama
        TIMEFRAMES = {
            '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1
        }
        
        total_records_2y = 0
        for timeframe in TIMEFRAMES:
            daily_count = TIMEFRAMES[timeframe]
            total_records_2y += 2 * 365 * daily_count
        
        estimated_seconds = total_records_2y / avg_speed
        estimated_hours = estimated_seconds / 3600
        
        print(f"   • 2 yıllık tahmin (tek coin): {estimated_hours:.1f} saat")
        print(f"   • 400 coin tahmin: {estimated_hours * 400 / 3600:.1f} saat")

if __name__ == "__main__":
    # Teorik hesaplama
    results = calculate_detailed_collection_time()
    
    # Gerçek hız testi
    test_real_speed()
    
    print(f"\n🎯 SONUÇ:")
    print(f"400 coin için tüm zaman dilimlerinde 2 yıllık veri çekme işlemi:")
    print(f"• En hızlı: {results['fastest_hours']:.1f} saat")
    print(f"• Ortalama: {results['average_hours']:.1f} saat") 
    print(f"• En yavaş: {results['slowest_hours']:.1f} saat")
    print(f"• Veri boyutu: {results['total_storage_gb']:.1f} GB")
    print(f"• Toplam API çağrısı: {results['total_api_calls']:,}") 