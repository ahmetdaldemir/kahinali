#!/usr/bin/env python3
"""
Log hata kontrol scripti
"""

import os
import re
from datetime import datetime

def check_log_errors():
    """Log dosyalarında hataları kontrol et"""
    print("=== LOG HATA KONTROLÜ ===")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    log_files = [
        'logs/kahin_ultima.log',
        'logs/model_training.log',
        'logs/model_retraining.log',
        'logs/data_collection_2year.log'
    ]
    
    error_patterns = [
        r'ERROR',
        r'WARNING', 
        r'Exception',
        r'Traceback',
        r'Failed',
        r'Error',
        r'CRITICAL'
    ]
    
    total_errors = 0
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"❌ {log_file} bulunamadı")
            continue
            
        file_size = os.path.getsize(log_file)
        print(f"📄 {log_file} ({file_size:,} bytes)")
        
        if file_size == 0:
            print("   ⚪ Boş dosya")
            continue
            
        errors_found = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                for pattern in error_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        errors_found.append(f"Satır {i}: {line.strip()}")
                        break
                        
            if errors_found:
                print(f"   ❌ {len(errors_found)} hata bulundu:")
                for error in errors_found[-5:]:  # Son 5 hata
                    print(f"      {error}")
                total_errors += len(errors_found)
            else:
                print("   ✅ Hata bulunamadı")
                
        except Exception as e:
            print(f"   ⚠️ Dosya okuma hatası: {e}")
            
        print()
    
    print("=== ÖZET ===")
    print(f"Toplam hata sayısı: {total_errors}")
    
    if total_errors == 0:
        print("🎉 Tüm loglar temiz! Hata yok.")
    else:
        print(f"⚠️ {total_errors} hata bulundu.")
    
    # Ek olarak ilk hata satırını göster
    main_log = 'logs/kahin_ultima.log'
    if os.path.exists(main_log):
        with open(main_log, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                    print(f"\n--- İlk hata satırı ---\nSatır {i}: {line.strip()}\n----------------------")
                    break
    
    return total_errors

if __name__ == "__main__":
    check_log_errors() 