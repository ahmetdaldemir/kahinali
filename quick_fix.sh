#!/bin/bash

# Kahinali Hızlı Sorun Çözme Scripti
# Sunucu: 185.209.228.189

echo "🔧 Kahinali Hızlı Sorun Çözme Başlatılıyor..."

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

# 1. Proje dizinine git
cd /var/www/html/kahin

# 2. PostgreSQL kullanıcısını düzelt
log "PostgreSQL kullanıcısı düzeltiliyor..."
sudo -u postgres createuser --interactive --pwprompt postgres || true
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '3010726904';" || true

# 3. Veritabanını yeniden oluştur
log "Veritabanı yeniden oluşturuluyor..."
sudo -u postgres psql -c "DROP DATABASE IF EXISTS kahin_ultima;"
sudo -u postgres psql -c "CREATE DATABASE kahin_ultima;"
sudo -u postgres psql -c "CREATE USER laravel WITH PASSWORD 'secret';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO laravel;"
sudo -u postgres psql -c "ALTER USER laravel CREATEDB;"

# 4. Virtual environment'ı yeniden oluştur
log "Virtual environment yeniden oluşturuluyor..."
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate

# 5. Bağımlılıkları yeniden yükle
log "Bağımlılıklar yeniden yükleniyor..."
pip install --upgrade pip
pip install wheel setuptools

# TA-Lib'i atla, sadece ta kütüphanesini kullan
log "TA kütüphanesi yükleniyor..."
pip install ta>=0.10.2

# Diğer bağımlılıkları yükle
log "Diğer bağımlılıklar yükleniyor..."
pip install -r requirements.txt

# 6. Migration'ı çalıştır
log "Veritabanı migration başlatılıyor..."
source venv/bin/activate
python3 database_migration.py

# 7. Servisleri yeniden başlat
log "Servisler yeniden başlatılıyor..."
systemctl restart kahinali.service
systemctl restart kahinali-web.service

# 8. Servis durumlarını kontrol et
log "Servis durumları kontrol ediliyor..."
systemctl status kahinali.service --no-pager
systemctl status kahinali-web.service --no-pager

# 9. Veritabanı tablolarını kontrol et
log "Veritabanı tabloları kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "\dt"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM signals;"

# 10. Web dashboard'u test et
log "Web dashboard test ediliyor..."
sleep 10
curl -s http://localhost:5000/api/signals > /dev/null && log "✅ Web dashboard erişilebilir" || error "❌ Web dashboard erişilemiyor"

echo ""
log "🎉 Hızlı sorun çözme tamamlandı!"
echo ""
echo "📋 Kontrol Komutları:"
echo "   - Ana sistem log: tail -f logs/kahin_ultima.log"
echo "   - Web dashboard log: tail -f logs/web_dashboard.log"
echo "   - Servis durumu: systemctl status kahinali.service"
echo "   - Web dashboard: http://185.209.228.189:5000" 