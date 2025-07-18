#!/bin/bash

# Kahinali Troubleshooting Scripti

echo "🔧 Kahinali Sorun Giderme Başlatılıyor..."

# Renkli çıktı
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# 1. Sistem durumu kontrolü
echo "=== 1. SİSTEM DURUMU ==="
log "Sistem durumu kontrol ediliyor..."
uptime
free -h
df -h

# 2. Python kurulumu kontrolü
echo ""
echo "=== 2. PYTHON KURULUMU ==="
log "Python versiyonları kontrol ediliyor..."
python3 --version
python3.11 --version

# 3. PostgreSQL durumu
echo ""
echo "=== 3. POSTGRESQL DURUMU ==="
log "PostgreSQL durumu kontrol ediliyor..."
systemctl status postgresql --no-pager

# 4. Servis durumları
echo ""
echo "=== 4. SERVİS DURUMLARI ==="
log "Kahinali servisleri kontrol ediliyor..."
systemctl status kahinali.service --no-pager
echo ""
systemctl status kahinali-web.service --no-pager

# 5. Proje dizini kontrolü
echo ""
echo "=== 5. PROJE DİZİNİ ==="
log "Proje dizini kontrol ediliyor..."
cd /var/www/html/kahin
pwd
ls -la

# 6. Virtual environment kontrolü
echo ""
echo "=== 6. VIRTUAL ENVIRONMENT ==="
log "Virtual environment kontrol ediliyor..."
if [ -d "venv" ]; then
    log "Virtual environment mevcut"
    source venv/bin/activate
    python --version
    pip list | head -10
else
    error "Virtual environment bulunamadı!"
fi

# 7. Veritabanı bağlantısı testi
echo ""
echo "=== 7. VERİTABANI BAĞLANTISI ==="
log "Veritabanı bağlantısı test ediliyor..."
if [ -f "setup_postgresql.py" ]; then
    python3 setup_postgresql.py
else
    error "setup_postgresql.py bulunamadı!"
fi

# 8. Log dosyaları kontrolü
echo ""
echo "=== 8. LOG DOSYALARI ==="
log "Log dosyaları kontrol ediliyor..."
if [ -f "logs/kahin_ultima.log" ]; then
    echo "Ana sistem log (son 10 satır):"
    tail -10 logs/kahin_ultima.log
else
    warning "Ana sistem log dosyası bulunamadı"
fi

if [ -f "logs/web_dashboard.log" ]; then
    echo "Web dashboard log (son 10 satır):"
    tail -10 logs/web_dashboard.log
else
    warning "Web dashboard log dosyası bulunamadı"
fi

# 9. Systemd logları
echo ""
echo "=== 9. SYSTEMD LOGLARI ==="
log "Systemd logları kontrol ediliyor..."
echo "Ana sistem son 10 log:"
journalctl -u kahinali.service -n 10 --no-pager
echo ""
echo "Web dashboard son 10 log:"
journalctl -u kahinali-web.service -n 10 --no-pager

# 10. Port kontrolü
echo ""
echo "=== 10. PORT KONTROLÜ ==="
log "Port durumları kontrol ediliyor..."
netstat -tlnp | grep :5000 || echo "Port 5000 dinlenmiyor"
netstat -tlnp | grep :5432 || echo "Port 5432 dinlenmiyor"

# 11. Firewall durumu
echo ""
echo "=== 11. FIREWALL DURUMU ==="
log "Firewall durumu kontrol ediliyor..."
ufw status

# 12. Bağımlılık kontrolü
echo ""
echo "=== 12. BAĞIMLILIK KONTROLÜ ==="
log "Python bağımlılıkları kontrol ediliyor..."
if [ -f "requirements.txt" ]; then
    echo "Requirements.txt mevcut"
    echo "İlk 10 bağımlılık:"
    head -10 requirements.txt
else
    error "requirements.txt bulunamadı!"
fi

# 13. Dosya izinleri
echo ""
echo "=== 13. DOSYA İZİNLERİ ==="
log "Dosya izinleri kontrol ediliyor..."
ls -la main.py
ls -la app/web.py
ls -la config.py

# 14. Çözüm önerileri
echo ""
echo "=== 14. ÇÖZÜM ÖNERİLERİ ==="
log "Sorun giderme önerileri:"

if ! systemctl is-active --quiet kahinali.service; then
    echo "❌ Ana sistem çalışmıyor"
    echo "   Çözüm: systemctl restart kahinali.service"
fi

if ! systemctl is-active --quiet kahinali-web.service; then
    echo "❌ Web dashboard çalışmıyor"
    echo "   Çözüm: systemctl restart kahinali-web.service"
fi

if ! systemctl is-active --quiet postgresql; then
    echo "❌ PostgreSQL çalışmıyor"
    echo "   Çözüm: systemctl start postgresql"
fi

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment yok"
    echo "   Çözüm: python3.11 -m venv venv"
fi

if [ ! -f "logs/kahin_ultima.log" ]; then
    echo "❌ Log dizini yok"
    echo "   Çözüm: mkdir -p logs"
fi

echo ""
log "🎯 Troubleshooting tamamlandı!"
echo ""
echo "📋 Manuel Kontroller:"
echo "   - Web Dashboard: http://185.209.228.189:5000"
echo "   - PostgreSQL: sudo -u postgres psql"
echo "   - Ana sistem log: tail -f logs/kahin_ultima.log"
echo "   - Web log: tail -f logs/web_dashboard.log" 