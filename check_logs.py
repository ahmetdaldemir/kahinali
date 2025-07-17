#!/usr/bin/env python3
"""
KahinUltima Log Dosyası Kontrol Scripti
Bu script log dosyalarını kontrol eder ve hataları tespit eder.
"""

import os
import sys
import glob
import re
from datetime import datetime, timedelta
import json

def print_header():
    print("=" * 60)
    print("📋 KAHIN ULTIMA LOG DOSYASI KONTROLÜ")
    print("=" * 60)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def get_log_files():
    """Log dosyalarını bul"""
    log_files = []
    
    # logs klasöründeki tüm .log dosyaları
    if os.path.exists('logs'):
        log_files.extend(glob.glob('logs/*.log'))
    
    # Diğer log dosyaları
    log_patterns = [
        '*.log',
        'logs/*.log',
        'logs/*.txt',
        '*.out',
        '*.err'
    ]
    
    for pattern in log_patterns:
        log_files.extend(glob.glob(pattern))
    
    # Tekrarları kaldır
    log_files = list(set(log_files))
    
    return log_files

def get_file_info(file_path):
    """Dosya bilgilerini al"""
    try:
        stat = os.stat(file_path)
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        return {
            'path': file_path,
            'size_mb': size_mb,
            'modified': mtime,
            'age_days': (datetime.now() - mtime).days
        }
    except Exception as e:
        return None

def check_log_file_health(file_path):
    """Tek bir log dosyasının sağlığını kontrol et"""
    issues = []
    
    try:
        # Dosya boyutu kontrolü
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        if size_mb > 1000:  # 1 GB
            issues.append(f"Dosya çok büyük: {size_mb:.1f} MB")
        elif size_mb > 100:  # 100 MB
            issues.append(f"Dosya büyük: {size_mb:.1f} MB")
        
        # Dosya okunabilirliği kontrolü
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # İlk 1000 satırı kontrol et
                lines = []
                for i, line in enumerate(f):
                    if i >= 1000:
                        break
                    lines.append(line)
                
                if not lines:
                    issues.append("Dosya boş")
                    return issues
                
                # Son satırları kontrol et
                last_lines = lines[-10:] if len(lines) >= 10 else lines
                
                # Hata pattern'lerini kontrol et
                error_patterns = [
                    r'ERROR',
                    r'CRITICAL',
                    r'FATAL',
                    r'Exception',
                    r'Traceback',
                    r'Failed',
                    r'Error',
                    r'Timeout',
                    r'Connection refused',
                    r'Database error',
                    r'API error',
                    r'Rate limit',
                    r'Memory error',
                    r'Disk full',
                    r'Permission denied'
                ]
                
                error_count = 0
                for line in last_lines:
                    for pattern in error_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            error_count += 1
                            break
                
                if error_count > 0:
                    issues.append(f"Son satırlarda {error_count} hata tespit edildi")
                
                # Log format kontrolü
                valid_log_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # Tarih
                    r'\d{2}:\d{2}:\d{2}',  # Saat
                    r'INFO|WARNING|ERROR|DEBUG|CRITICAL'  # Log seviyesi
                ]
                
                format_issues = 0
                for line in last_lines:
                    if not any(re.search(pattern, line) for pattern in valid_log_patterns):
                        format_issues += 1
                
                if format_issues > 5:
                    issues.append("Log formatı tutarsız")
                
        except UnicodeDecodeError:
            issues.append("Dosya encoding hatası")
        except PermissionError:
            issues.append("Dosya okuma izni yok")
        except Exception as e:
            issues.append(f"Dosya okuma hatası: {e}")
    
    except Exception as e:
        issues.append(f"Genel dosya hatası: {e}")
    
    return issues

def analyze_log_content(file_path, max_lines=1000):
    """Log dosyasının içeriğini analiz et"""
    print(f"\n📄 {os.path.basename(file_path)} İÇERİK ANALİZİ")
    print("-" * 40)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.strip())
        
        if not lines:
            print("❌ Dosya boş")
            return
        
        # İstatistikler
        total_lines = len(lines)
        error_lines = 0
        warning_lines = 0
        info_lines = 0
        
        # Hata analizi
        errors = []
        warnings = []
        
        for line in lines:
            if re.search(r'ERROR|CRITICAL|FATAL|Exception|Traceback', line, re.IGNORECASE):
                error_lines += 1
                errors.append(line)
            elif re.search(r'WARNING|WARN', line, re.IGNORECASE):
                warning_lines += 1
                warnings.append(line)
            elif re.search(r'INFO', line, re.IGNORECASE):
                info_lines += 1
        
        print(f"📊 Toplam satır: {total_lines}")
        print(f"❌ Hata satırları: {error_lines}")
        print(f"⚠️ Uyarı satırları: {warning_lines}")
        print(f"ℹ️ Bilgi satırları: {info_lines}")
        
        # Son hataları göster
        if errors:
            print(f"\n🚨 SON HATALAR (son 5):")
            for error in errors[-5:]:
                print(f"   • {error[:100]}...")
        
        # Son uyarıları göster
        if warnings:
            print(f"\n⚠️ SON UYARILAR (son 5):")
            for warning in warnings[-5:]:
                print(f"   • {warning[:100]}...")
        
        # Son satırları göster
        print(f"\n📝 SON SATIRLAR:")
        for line in lines[-5:]:
            print(f"   • {line}")
    
    except Exception as e:
        print(f"❌ İçerik analizi hatası: {e}")

def check_log_rotation():
    """Log rotasyon kontrolü"""
    print("\n🔄 LOG ROTASYON KONTROLÜ")
    print("-" * 40)
    
    log_files = get_log_files()
    
    if not log_files:
        print("✅ Log dosyası bulunamadı")
        return
    
    # Tarih bazlı log dosyalarını kontrol et
    date_pattern = r'\d{4}-\d{2}-\d{2}|\d{8}|\d{6}'
    rotated_logs = []
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        if re.search(date_pattern, filename):
            rotated_logs.append(log_file)
    
    if rotated_logs:
        print(f"✅ {len(rotated_logs)} tarihli log dosyası bulundu")
        for log in rotated_logs[:5]:  # İlk 5'ini göster
            print(f"   📄 {os.path.basename(log)}")
    else:
        print("⚠️ Tarihli log dosyası bulunamadı (rotasyon yapılmamış olabilir)")
    
    # Eski log dosyalarını kontrol et
    old_logs = []
    for log_file in log_files:
        info = get_file_info(log_file)
        if info and info['age_days'] > 30:
            old_logs.append(log_file)
    
    if old_logs:
        print(f"⚠️ {len(old_logs)} eski log dosyası (>30 gün)")
        for log in old_logs[:3]:
            info = get_file_info(log)
            print(f"   📄 {os.path.basename(log)} ({info['age_days']} gün önce)")
    else:
        print("✅ Eski log dosyası yok")

def check_log_permissions():
    """Log dosyası izinlerini kontrol et"""
    print("\n🔐 LOG DOSYASI İZİNLERİ")
    print("-" * 40)
    
    log_files = get_log_files()
    
    if not log_files:
        print("✅ Log dosyası bulunamadı")
        return
    
    permission_issues = []
    
    for log_file in log_files:
        try:
            # Dosya okunabilir mi?
            with open(log_file, 'r') as f:
                f.read(1)
            
            # Dosya yazılabilir mi?
            with open(log_file, 'a') as f:
                pass
            
            print(f"✅ {os.path.basename(log_file)} - İzinler OK")
            
        except PermissionError:
            permission_issues.append(log_file)
            print(f"❌ {os.path.basename(log_file)} - İzin hatası")
        except Exception as e:
            permission_issues.append(log_file)
            print(f"❌ {os.path.basename(log_file)} - {e}")
    
    if permission_issues:
        print(f"\n⚠️ {len(permission_issues)} dosyada izin sorunu var")
    else:
        print("\n✅ Tüm log dosyalarının izinleri uygun")

def main():
    """Ana fonksiyon"""
    print_header()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--content':
            # İçerik analizi
            log_files = get_log_files()
            if log_files:
                analyze_log_content(log_files[0])  # İlk log dosyasını analiz et
            else:
                print("❌ Log dosyası bulunamadı")
        elif sys.argv[1] == '--all':
            # Tüm log dosyalarını analiz et
            log_files = get_log_files()
            for log_file in log_files:
                analyze_log_content(log_file)
        else:
            print("❌ Geçersiz parametre")
            print("Kullanım:")
            print("  python check_logs.py --content # İlk log dosyasını analiz et")
            print("  python check_logs.py --all     # Tüm log dosyalarını analiz et")
    else:
        # Varsayılan kontrol
        log_files = get_log_files()
        
        if not log_files:
            print("❌ Log dosyası bulunamadı")
            return
        
        print("📊 LOG DOSYALARI ÖZETİ")
        print("-" * 40)
        
        total_size = 0
        all_issues = []
        
        for log_file in log_files:
            info = get_file_info(log_file)
            if info:
                total_size += info['size_mb']
                print(f"📄 {info['path']}")
                print(f"   Boyut: {info['size_mb']:.2f} MB")
                print(f"   Son değişiklik: {info['modified'].strftime('%Y-%m-%d %H:%M')} ({info['age_days']} gün önce)")
                
                # Sağlık kontrolü
                issues = check_log_file_health(log_file)
                if issues:
                    all_issues.extend([f"{os.path.basename(log_file)}: {issue}" for issue in issues])
                    print(f"   ⚠️ Sorunlar: {', '.join(issues)}")
                else:
                    print(f"   ✅ Sağlıklı")
                print()
        
        print(f"📊 TOPLAM LOG BOYUTU: {total_size:.2f} MB")
        
        if all_issues:
            print(f"\n⚠️ TESPİT EDİLEN SORUNLAR ({len(all_issues)} adet):")
            for i, issue in enumerate(all_issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("\n✅ Hiç sorun tespit edilmedi")
        
        # Ek kontroller
        check_log_rotation()
        check_log_permissions()
    
    print("\n" + "=" * 60)
    print("✅ Log dosyası kontrolü tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc() 