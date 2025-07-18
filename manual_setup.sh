#!/bin/bash

# Kahinali Manuel Kurulum Scripti
# Sunucu: 185.209.228.189

set -e

echo "🚀 Kahinali Manuel Kurulum Başlatılıyor..."

# Renkli çıktı için
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log fonksiyonu
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# 1. Sistem güncellemesi
log "Sistem güncelleniyor..."
apt update -y
apt upgrade -y

# 2. Python 3.11 ve development tools kurulumu
log "Python 3.11 ve development tools kuruluyor..."
apt install -y python3.11 python3.11-venv python3.11-pip python3.11-dev
apt install -y build-essential gcc g++ make
apt install -y python3-dev python3-pip

# 3. Git kurulumu
log "Git kuruluyor..."
apt install -y git

# 4. PostgreSQL kurulumu
log "PostgreSQL kuruluyor..."
apt install -y postgresql postgresql-contrib
systemctl start postgresql
systemctl enable postgresql

# 5. PostgreSQL konfigürasyonu
log "PostgreSQL konfigürasyonu yapılıyor..."
sudo -u postgres createuser --interactive --pwprompt postgres || true
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '3010726904';"
sudo -u postgres psql -c "CREATE DATABASE kahin_ultima;"
sudo -u postgres psql -c "CREATE USER laravel WITH PASSWORD 'secret';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO laravel;"
sudo -u postgres psql -c "ALTER USER laravel CREATEDB;"

# 6. TA-Lib kurulumu
log "TA-Lib kuruluyor..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install

# 7. Proje dizini oluşturma
log "Proje dizini oluşturuluyor..."
mkdir -p /var/www/html
cd /var/www/html

# Mevcut projeyi sil ve yeniden klonla
rm -rf kahin
git clone https://github.com/ahmetdaldemir/kahinali.git kahin
cd kahin

# 8. Virtual environment oluştur
log "Virtual environment oluşturuluyor..."
python3.11 -m venv venv
source venv/bin/activate

# 9. Python bağımlılıkları yükleniyor
log "Python bağımlılıkları yükleniyor..."
pip install --upgrade pip
pip install wheel setuptools

# TA-Lib'i önce kur
log "TA-Lib kuruluyor..."
pip install TA-Lib

# Diğer bağımlılıkları yükle
log "Diğer bağımlılıklar yükleniyor..."
pip install -r requirements.txt

# 10. Gerekli dizinleri oluştur
log "Gerekli dizinler oluşturuluyor..."
mkdir -p logs signals data models

# 11. Otomatik veritabanı migration'ı çalıştır
log "Veritabanı migration başlatılıyor..."
source venv/bin/activate
python3 database_migration.py

# Migration başarısızsa tekrar dene
if [ $? -ne 0 ]; then
    warning "Migration başarısız, tekrar deneniyor..."
    sleep 5
    source venv/bin/activate
    python3 database_migration.py
fi

# 12. Systemd service dosyalarını kopyala
log "Systemd service dosyaları kopyalanıyor..."
cp kahinali.service /etc/systemd/system/
cp kahinali-web.service /etc/systemd/system/

# 13. Systemd'yi yeniden yükle
log "Systemd yeniden yükleniyor..."
systemctl daemon-reload

# 14. Servisleri etkinleştir
log "Servisler etkinleştiriliyor..."
systemctl enable kahinali.service
systemctl enable kahinali-web.service

# 15. Firewall ayarları
log "Firewall ayarları yapılıyor..."
ufw allow 5000
ufw allow 22
ufw --force enable

# 16. Servisleri başlat
log "Servisler başlatılıyor..."
systemctl start kahinali.service
systemctl start kahinali-web.service

# 17. Servis durumlarını kontrol et
log "Servis durumları kontrol ediliyor..."
echo "=== Kahinali Ana Sistem Durumu ==="
systemctl status kahinali.service --no-pager

echo "=== Kahinali Web Dashboard Durumu ==="
systemctl status kahinali-web.service --no-pager

# 18. Log dosyalarını kontrol et
log "Log dosyaları kontrol ediliyor..."
if [ -f "logs/kahin_ultima.log" ]; then
    echo "=== Son 10 log satırı ==="
    tail -10 logs/kahin_ultima.log
else
    warning "Ana sistem log dosyası bulunamadı"
fi

if [ -f "logs/web_dashboard.log" ]; then
    echo "=== Web Dashboard Son 10 Log ==="
    tail -10 logs/web_dashboard.log
else
    warning "Web dashboard log dosyası bulunamadı"
fi

# 19. Veritabanı bağlantısını test et
log "Veritabanı bağlantısı test ediliyor..."
cd /var/www/html/kahin
source venv/bin/activate
python3 setup_postgresql.py

# 20. Web dashboard'u test et
log "Web dashboard test ediliyor..."
curl -s http://localhost:5000/api/signals > /dev/null && log "✅ Web dashboard erişilebilir" || error "❌ Web dashboard erişilemiyor"

# 21. Veritabanı tablolarını kontrol et
log "Veritabanı tabloları kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "\dt"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM signals;"

echo ""
log "🎉 Manuel kurulum tamamlandı!"
echo ""
echo "📋 Önemli Bilgiler:"
echo "   - Web Dashboard: http://185.209.228.189:5000"
echo "   - Ana Sistem Log: /var/www/html/kahin/logs/kahin_ultima.log"
echo "   - Web Log: /var/www/html/kahin/logs/web_dashboard.log"
echo ""
echo "🔧 Servis Yönetimi:"
echo "   - Ana sistem durumu: systemctl status kahinali.service"
echo "   - Web dashboard durumu: systemctl status kahinali-web.service"
echo "   - Ana sistem yeniden başlat: systemctl restart kahinali.service"
echo "   - Web dashboard yeniden başlat: systemctl restart kahinali-web.service"
echo ""
echo "📊 Log Görüntüleme:"
echo "   - Ana sistem logları: journalctl -u kahinali.service -f"
echo "   - Web dashboard logları: journalctl -u kahinali-web.service -f" 