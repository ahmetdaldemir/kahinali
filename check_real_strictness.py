import os
import sys
import time
import psutil
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

def check_real_strictness():
    """Sistemin gerçek sıkılık seviyesini hesapla"""
    
    print("============================================================")
    print("🔒 KAHIN ULTIMA GERÇEK SIKILIK SEVİYESİ ANALİZİ")
    print("============================================================")
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_score = 0
    max_score = 0
    issues = []
    warnings = []
    
    # 1. SİSTEM KAYNAKLARI (25 puan)
    print("💻 SİSTEM KAYNAKLARI (25 puan)")
    print("-" * 40)
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # CPU kullanımı (10 puan)
        if cpu_percent < 50:
            cpu_score = 10
            print(f"✅ CPU kullanımı: {cpu_percent:.1f}% (Mükemmel)")
        elif cpu_percent < 80:
            cpu_score = 7
            print(f"⚠️ CPU kullanımı: {cpu_percent:.1f}% (İyi)")
        else:
            cpu_score = 3
            print(f"❌ CPU kullanımı: {cpu_percent:.1f}% (Yüksek)")
            issues.append(f"CPU kullanımı çok yüksek: {cpu_percent:.1f}%")
        
        # RAM kullanımı (10 puan)
        if memory.percent < 70:
            ram_score = 10
            print(f"✅ RAM kullanımı: {memory.percent:.1f}% (Mükemmel)")
        elif memory.percent < 90:
            ram_score = 7
            print(f"⚠️ RAM kullanımı: {memory.percent:.1f}% (İyi)")
        else:
            ram_score = 3
            print(f"❌ RAM kullanımı: {memory.percent:.1f}% (Kritik)")
            issues.append(f"RAM kullanımı kritik: {memory.percent:.1f}%")
        
        # Disk alanı (5 puan)
        if disk.percent < 80:
            disk_score = 5
            print(f"✅ Disk kullanımı: {disk.percent:.1f}% (Yeterli)")
        else:
            disk_score = 2
            print(f"❌ Disk kullanımı: {disk.percent:.1f}% (Az)")
            issues.append(f"Disk alanı az: {disk.percent:.1f}%")
        
        system_score = cpu_score + ram_score + disk_score
        total_score += system_score
        max_score += 25
        
        print(f"📊 Sistem kaynakları puanı: {system_score}/25\n")
        
    except Exception as e:
        print(f"❌ Sistem kaynakları kontrol edilemedi: {e}")
        issues.append("Sistem kaynakları kontrol edilemedi")
        max_score += 25
    
    # 2. VERİTABANI SAĞLIĞI (20 puan)
    print("🗄️ VERİTABANI SAĞLIĞI (20 puan)")
    print("-" * 40)
    
    try:
        # PostgreSQL kontrolü
        import psycopg2
        from config import Config
        
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            port=Config.DB_PORT
        )
        
        cursor = conn.cursor()
        
        # Tablo sayısı (5 puan)
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        
        if table_count >= 5:
            table_score = 5
            print(f"✅ Tablo sayısı: {table_count} (Yeterli)")
        else:
            table_score = 2
            print(f"⚠️ Tablo sayısı: {table_count} (Az)")
            warnings.append(f"Tablo sayısı az: {table_count}")
        
        # Sinyal sayısı (10 puan)
        cursor.execute("SELECT COUNT(*) FROM signals")
        signal_count = cursor.fetchone()[0]
        
        if signal_count > 0:
            signal_score = 10
            print(f"✅ Sinyal sayısı: {signal_count} (Aktif)")
        else:
            signal_score = 3
            print(f"❌ Sinyal sayısı: 0 (Pasif)")
            issues.append("Veritabanında hiç sinyal yok")
        
        # Veritabanı boyutu (5 puan)
        cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = cursor.fetchone()[0]
        
        if "MB" in db_size and float(db_size.split()[0]) < 50:
            size_score = 5
            print(f"✅ Veritabanı boyutu: {db_size} (Normal)")
        else:
            size_score = 2
            print(f"⚠️ Veritabanı boyutu: {db_size} (Büyük)")
            warnings.append(f"Veritabanı boyutu büyük: {db_size}")
        
        db_score = table_score + signal_score + size_score
        total_score += db_score
        max_score += 20
        
        print(f"📊 Veritabanı puanı: {db_score}/20\n")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Veritabanı kontrol edilemedi: {e}")
        issues.append("Veritabanı bağlantısı başarısız")
        max_score += 20
    
    # 3. DOSYA SİSTEMİ (20 puan)
    print("📁 DOSYA SİSTEMİ (20 puan)")
    print("-" * 40)
    
    # Log boyutu (10 puan)
    log_size = 0
    for log_file in Path("logs").glob("*.log"):
        log_size += log_file.stat().st_size
    
    log_size_mb = log_size / (1024 * 1024)
    
    if log_size_mb < 50:
        log_score = 10
        print(f"✅ Log boyutu: {log_size_mb:.1f} MB (Normal)")
    elif log_size_mb < 200:
        log_score = 7
        print(f"⚠️ Log boyutu: {log_size_mb:.1f} MB (Büyük)")
        warnings.append(f"Log boyutu büyük: {log_size_mb:.1f} MB")
    else:
        log_score = 3
        print(f"❌ Log boyutu: {log_size_mb:.1f} MB (Çok büyük)")
        issues.append(f"Log boyutu çok büyük: {log_size_mb:.1f} MB")
    
    # Model dosyaları (10 puan)
    model_files = list(Path("models").glob("*.pkl")) + list(Path("models").glob("*.h5"))
    
    if len(model_files) >= 10:
        model_score = 10
        print(f"✅ Model dosyaları: {len(model_files)} (Tam)")
    elif len(model_files) >= 5:
        model_score = 7
        print(f"⚠️ Model dosyaları: {len(model_files)} (Eksik)")
        warnings.append(f"Model dosyaları eksik: {len(model_files)}")
    else:
        model_score = 3
        print(f"❌ Model dosyaları: {len(model_files)} (Çok az)")
        issues.append(f"Model dosyaları çok az: {len(model_files)}")
    
    file_score = log_score + model_score
    total_score += file_score
    max_score += 20
    
    print(f"📊 Dosya sistemi puanı: {file_score}/20\n")
    
    # 4. PERFORMANS TESTİ (20 puan)
    print("⚡ PERFORMANS TESTİ (20 puan)")
    print("-" * 40)
    
    try:
        # Basit performans testi
        start_time = time.time()
        
        # Simüle edilmiş işlem
        import numpy as np
        data = np.random.rand(1000, 100)
        result = np.dot(data, data.T)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if duration < 0.1:
            perf_score = 20
            print(f"✅ Performans: {duration:.3f} saniye (Mükemmel)")
        elif duration < 0.5:
            perf_score = 15
            print(f"⚠️ Performans: {duration:.3f} saniye (İyi)")
        elif duration < 1.0:
            perf_score = 10
            print(f"⚠️ Performans: {duration:.3f} saniye (Orta)")
        else:
            perf_score = 5
            print(f"❌ Performans: {duration:.3f} saniye (Yavaş)")
            issues.append(f"Performans yavaş: {duration:.3f} saniye")
        
        total_score += perf_score
        max_score += 20
        
        print(f"📊 Performans puanı: {perf_score}/20\n")
        
    except Exception as e:
        print(f"❌ Performans testi başarısız: {e}")
        issues.append("Performans testi başarısız")
        max_score += 20
    
    # 5. GÜVENLİK VE KONFİGÜRASYON (15 puan)
    print("🔐 GÜVENLİK VE KONFİGÜRASYON (15 puan)")
    print("-" * 40)
    
    # API anahtarları kontrolü
    config_score = 0
    
    try:
        from config import Config
        
        if hasattr(Config, 'BINANCE_API_KEY') and Config.BINANCE_API_KEY:
            config_score += 3
            print("✅ Binance API anahtarı mevcut")
        else:
            print("❌ Binance API anahtarı eksik")
            issues.append("Binance API anahtarı eksik")
        
        if hasattr(Config, 'TELEGRAM_BOT_TOKEN') and Config.TELEGRAM_BOT_TOKEN:
            config_score += 3
            print("✅ Telegram bot token mevcut")
        else:
            print("❌ Telegram bot token eksik")
            issues.append("Telegram bot token eksik")
        
        if hasattr(Config, 'MAX_COINS_TO_TRACK') and Config.MAX_COINS_TO_TRACK >= 400:
            config_score += 3
            print("✅ Coin sayısı yeterli")
        else:
            print("⚠️ Coin sayısı az")
            warnings.append("Coin sayısı az")
        
        if os.path.exists('.env'):
            config_score += 3
            print("✅ .env dosyası mevcut")
        else:
            print("❌ .env dosyası eksik")
            issues.append(".env dosyası eksik")
        
        if os.path.exists('requirements.txt'):
            config_score += 3
            print("✅ requirements.txt mevcut")
        else:
            print("❌ requirements.txt eksik")
            issues.append("requirements.txt eksik")
        
        total_score += config_score
        max_score += 15
        
        print(f"📊 Konfigürasyon puanı: {config_score}/15\n")
        
    except Exception as e:
        print(f"❌ Konfigürasyon kontrol edilemedi: {e}")
        issues.append("Konfigürasyon kontrol edilemedi")
        max_score += 15
    
    # SONUÇ HESAPLAMA
    strictness_percentage = (total_score / max_score) * 100
    strictness_score = (total_score / max_score) * 10
    
    print("============================================================")
    print("📊 GERÇEK SIKILIK SEVİYESİ SONUCU")
    print("============================================================")
    
    print(f"🎯 TOPLAM PUAN: {total_score}/{max_score}")
    print(f"📈 SIKILIK YÜZDESİ: {strictness_percentage:.1f}%")
    print(f"🔒 SIKILIK SEVİYESİ: {strictness_score:.1f}/10")
    
    # Seviye belirleme
    if strictness_score >= 9.0:
        level = "🔴 ÇOK SIKI"
        status = "Mükemmel"
    elif strictness_score >= 7.5:
        level = "🟠 SIKI"
        status = "İyi"
    elif strictness_score >= 6.0:
        level = "🟡 ORTA"
        status = "Kabul edilebilir"
    elif strictness_score >= 4.0:
        level = "🟢 GEVŞEK"
        status = "İyileştirilmeli"
    else:
        level = "🔵 ÇOK GEVŞEK"
        status = "Kritik"
    
    print(f"🏆 SEVİYE: {level}")
    print(f"📋 DURUM: {status}")
    
    print(f"\n⚠️ TESPİT EDİLEN SORUNLAR ({len(issues)} adet):")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    if warnings:
        print(f"\n⚠️ UYARILAR ({len(warnings)} adet):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    print(f"\n💡 ÖNERİLER:")
    if strictness_score < 7.0:
        print("   • Sistem gevşek çalışıyor, optimizasyon gerekli")
        print("   • Yüksek öncelikli sorunları çözün")
    elif strictness_score < 8.5:
        print("   • Sistem orta seviyede sıkı, iyileştirmeler yapılabilir")
        print("   • Uyarıları kontrol edin")
    else:
        print("   • Sistem sıkı çalışıyor, düzenli kontrolleri sürdürün")
    
    print("============================================================")
    
    return {
        'score': strictness_score,
        'percentage': strictness_percentage,
        'level': level,
        'status': status,
        'issues': issues,
        'warnings': warnings
    }

if __name__ == "__main__":
    check_real_strictness() 