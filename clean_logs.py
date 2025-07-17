#!/usr/bin/env python3
"""
Log Dosyalarını Temizleme Scripti
Tüm log dosyalarını temizler ve sistemi yeni başlangıç için hazırlar
"""

import os
import shutil
from datetime import datetime
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_logs():
    """Log dosyalarını temizle"""
    print("📝 Log dosyaları temizleniyor...")
    
    logs_dir = "logs"
    backup_dir = os.path.join(logs_dir, "backup")
    
    try:
        # Ana logs dizinindeki tüm log dosyalarını sil
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
        
        deleted_count = 0
        
        for file in log_files:
            file_path = os.path.join(logs_dir, file)
            try:
                # Dosya boyutunu kontrol et
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"Silindi: {file} ({file_size_mb:.2f} MB)")
            except Exception as e:
                logger.error(f"Log dosyası silinemedi {file}: {e}")
        
        # Backup dizinini temizle (varsa)
        if os.path.exists(backup_dir):
            try:
                backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.log')]
                for file in backup_files:
                    file_path = os.path.join(backup_dir, file)
                    os.remove(file_path)
                    logger.info(f"Backup silindi: {file}")
                
                # Boş backup dizinini sil
                os.rmdir(backup_dir)
                logger.info("Backup dizini silindi")
                
            except Exception as e:
                logger.error(f"Backup dizini temizlenemedi: {e}")
        
        print(f"✅ {deleted_count} log dosyası silindi")
        print("✅ Log sistemi temizlendi!")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Log temizleme sırasında hata: {e}")
        return 0

def create_fresh_logs():
    """Yeni log dosyaları oluştur"""
    print("🆕 Yeni log dosyaları oluşturuluyor...")
    
    logs_dir = "logs"
    
    # Ana log dosyası
    main_log = os.path.join(logs_dir, "kahin_ultima.log")
    with open(main_log, 'w', encoding='utf-8') as f:
        f.write(f"=== KAHİN ULTIMA SİSTEM BAŞLANGIÇ ===\n")
        f.write(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sistem yeniden başlatıldı - Tüm loglar temizlendi\n")
        f.write("=" * 50 + "\n\n")
    
    print("✅ Yeni log dosyaları oluşturuldu")

def verify_log_cleanup():
    """Log temizliğini doğrula"""
    print("🔍 Log temizliği doğrulanıyor...")
    
    logs_dir = "logs"
    remaining_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    
    print(f"📁 Logs dizini: {logs_dir}")
    print(f"📄 Kalan log dosyası sayısı: {len(remaining_files)}")
    
    if remaining_files:
        print("📋 Kalan log dosyaları:")
        for file in remaining_files:
            file_path = os.path.join(logs_dir, file)
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"   - {file} ({file_size_mb:.2f} MB)")
    else:
        print("✅ Log sistemi tamamen temiz")

def main():
    """Ana temizlik fonksiyonu"""
    print("🧹 KAHİN ULTIMA - LOG TEMİZLEME")
    print("=" * 50)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Logları temizle
    deleted_count = clean_logs()
    print()
    
    # Yeni log dosyaları oluştur
    create_fresh_logs()
    print()
    
    # Doğrulama
    verify_log_cleanup()
    print()
    
    print("🎉 Log temizleme tamamlandı!")
    print(f"Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("💡 Sistem artık temiz loglarla çalışmaya hazır!")

if __name__ == "__main__":
    main() 