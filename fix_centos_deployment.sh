#!/bin/bash

# Kahinali CentOS/RHEL Deployment Düzeltme Scripti
# Sunucu: 185.209.228.189

echo "🚀 CentOS/RHEL Deployment Düzeltme Başlatılıyor..."

# Renkli çıktı için
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

# 1. Sistem güncellemesi (CentOS/RHEL için)
log "Sistem güncellemesi yapılıyor..."
yum update -y

# 2. Gerekli paketleri kur (CentOS/RHEL için)
log "Gerekli paketler kuruluyor..."
yum install -y epel-release
yum groupinstall -y "Development Tools"
yum install -y python3 python3-pip python3-devel
yum install -y git wget curl
yum install -y postgresql postgresql-server postgresql-contrib
yum install -y gcc gcc-c++ make

# 3. PostgreSQL'i başlat
log "PostgreSQL başlatılıyor..."
postgresql-setup initdb
systemctl start postgresql
systemctl enable postgresql

# 4. PostgreSQL kullanıcısını oluştur
log "PostgreSQL kullanıcısı oluşturuluyor..."
sudo -u postgres createuser --interactive --pwprompt laravel || true
sudo -u postgres psql -c "ALTER USER laravel PASSWORD 'secret';"

# 5. Proje dizinini oluştur
log "Proje dizini oluşturuluyor..."
mkdir -p /var/www/html
cd /var/www/html

# 6. Mevcut projeyi sil ve yeniden klonla
rm -rf kahin
git clone https://github.com/ahmetdaldemir/kahinali.git kahin
cd kahin

# 7. Python3.9 ile virtual environment oluştur
log "Virtual environment oluşturuluyor..."
python3 -m venv venv
source venv/bin/activate

# 8. Bağımlılıkları yükle
log "Bağımlılıklar yükleniyor..."
pip install --upgrade pip
pip install wheel setuptools

# 9. TA-Lib yerine ta kullan (CentOS'ta TA-Lib sorunlu)
log "TA kütüphanesi kuruluyor..."
pip install ta

# 10. Diğer bağımlılıkları yükle
log "Diğer bağımlılıklar yükleniyor..."
pip install -r requirements.txt

# 11. PostgreSQL konfigürasyonu
log "PostgreSQL konfigürasyonu yapılıyor..."
sudo -u postgres psql -c "CREATE DATABASE kahin_ultima;"
sudo -u postgres psql -c "CREATE USER laravel WITH PASSWORD 'secret';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO laravel;"
sudo -u postgres psql -c "ALTER USER laravel CREATEDB;"

# 12. Gerekli dizinleri oluştur
log "Dizinler oluşturuluyor..."
mkdir -p logs signals data models

# 13. Virtual environment'ı aktifleştir ve migration çalıştır
log "Veritabanı migration başlatılıyor..."
source venv/bin/activate
python3 database_migration.py

# 14. Migration başarısızsa zorla tablo oluştur
if [ $? -ne 0 ]; then
    warning "Migration başarısız, zorla tablo oluşturma deneniyor..."
    python3 force_create_tables.py
fi

# 15. Manuel tablo oluşturma
log "Manuel tablo oluşturma çalıştırılıyor..."
chmod +x manual_create_tables.sh
./manual_create_tables.sh

# 16. Systemd service dosyalarını kopyala
log "Systemd servisleri kuruluyor..."
cp kahinali.service /etc/systemd/system/
cp kahinali-web.service /etc/systemd/system/

# 17. Systemd'yi yeniden yükle
systemctl daemon-reload

# 18. Servisleri etkinleştir ve başlat
log "Servisler başlatılıyor..."
systemctl enable kahinali.service
systemctl enable kahinali-web.service
systemctl start kahinali.service
systemctl start kahinali-web.service

# 19. Firewall ayarları (CentOS/RHEL için)
log "Firewall ayarları yapılıyor..."
systemctl start firewalld
systemctl enable firewalld
firewall-cmd --permanent --add-port=5000/tcp
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --reload

# 20. Servis durumlarını kontrol et
log "Servis durumları kontrol ediliyor..."
systemctl status kahinali.service --no-pager
systemctl status kahinali-web.service --no-pager

# 21. Veritabanı tablolarını kontrol et
log "Veritabanı tabloları kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "\dt"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM signals;"

# 22. Web dashboard'u test et
log "Web dashboard test ediliyor..."
sleep 10
curl -s http://localhost:5000/api/signals > /dev/null && log "✅ Web dashboard erişilebilir" || error "❌ Web dashboard erişilemiyor"

echo ""
log "🎉 CentOS/RHEL Deployment Düzeltme Tamamlandı!"
echo ""
echo "📋 Sistem Bilgileri:"
echo "   - OS: CentOS/RHEL"
echo "   - Python: $(python3 --version)"
echo "   - PostgreSQL: $(psql --version)"
echo ""
echo "📊 Veritabanı Bilgileri:"
echo "   - Host: localhost"
echo "   - Port: 5432"
echo "   - Database: kahin_ultima"
echo "   - User: laravel"
echo "   - Password: secret"
echo ""
echo "🌐 Web Dashboard:"
echo "   - URL: http://185.209.228.189:5000"
echo "   - API: http://185.209.228.189:5000/api/signals"
echo ""
echo "🔧 Kontrol Komutları:"
echo "   - Servis durumu: systemctl status kahinali.service"
echo "   - Web durumu: systemctl status kahinali-web.service"
echo "   - Tablolar: sudo -u postgres psql -d kahin_ultima -c '\dt'"
echo "   - Firewall: firewall-cmd --list-all" 