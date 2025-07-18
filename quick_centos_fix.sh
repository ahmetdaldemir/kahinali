#!/bin/bash

# Kahinali CentOS Hızlı Düzeltme Scripti
# Sunucu: 185.209.228.189

echo "🚀 CentOS Hızlı Düzeltme Başlatılıyor..."

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

# 2. Virtual environment'ı yeniden oluştur
log "Virtual environment yeniden oluşturuluyor..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# 3. Bağımlılıkları yükle
log "Bağımlılıklar yükleniyor..."
pip install --upgrade pip
pip install wheel setuptools

# TA-Lib yerine ta kullan
pip install ta

# Diğer bağımlılıkları yükle
pip install -r requirements.txt

# 4. PostgreSQL'i kontrol et ve başlat
log "PostgreSQL kontrol ediliyor..."
systemctl status postgresql > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log "PostgreSQL başlatılıyor..."
    postgresql-setup initdb
    systemctl start postgresql
    systemctl enable postgresql
fi

# 5. PostgreSQL kullanıcısını oluştur
log "PostgreSQL kullanıcısı oluşturuluyor..."
sudo -u postgres psql -c "CREATE DATABASE kahin_ultima;" 2>/dev/null || true
sudo -u postgres psql -c "CREATE USER laravel WITH PASSWORD 'secret';" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO laravel;" 2>/dev/null || true
sudo -u postgres psql -c "ALTER USER laravel CREATEDB;" 2>/dev/null || true

# 6. Dizinleri oluştur
log "Dizinler oluşturuluyor..."
mkdir -p logs signals data models

# 7. Migration çalıştır
log "Veritabanı migration başlatılıyor..."
source venv/bin/activate
python3 database_migration.py

# Migration başarısızsa zorla tablo oluştur
if [ $? -ne 0 ]; then
    warning "Migration başarısız, zorla tablo oluşturma deneniyor..."
    python3 force_create_tables.py
fi

# 8. Manuel tablo oluşturma
log "Manuel tablo oluşturma çalıştırılıyor..."
chmod +x manual_create_tables.sh
./manual_create_tables.sh

# 9. Systemd servislerini düzelt
log "Systemd servisleri düzeltiliyor..."
cat > /etc/systemd/system/kahinali.service << 'EOF'
[Unit]
Description=Kahinali Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/html/kahin
Environment=PATH=/var/www/html/kahin/venv/bin
ExecStart=/var/www/html/kahin/venv/bin/python3 main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/kahinali-web.service << 'EOF'
[Unit]
Description=Kahinali Web Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/html/kahin
Environment=PATH=/var/www/html/kahin/venv/bin
ExecStart=/var/www/html/kahin/venv/bin/python3 -m flask --app app.web run --host=0.0.0.0 --port=5000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 10. Systemd'yi yeniden yükle
systemctl daemon-reload

# 11. Servisleri başlat
log "Servisler başlatılıyor..."
systemctl enable kahinali.service
systemctl enable kahinali-web.service
systemctl restart kahinali.service
systemctl restart kahinali-web.service

# 12. Firewall ayarları
log "Firewall ayarları yapılıyor..."
systemctl start firewalld
systemctl enable firewalld
firewall-cmd --permanent --add-port=5000/tcp
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --reload

# 13. Servis durumlarını kontrol et
log "Servis durumları kontrol ediliyor..."
systemctl status kahinali.service --no-pager
systemctl status kahinali-web.service --no-pager

# 14. Veritabanı tablolarını kontrol et
log "Veritabanı tabloları kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "\dt"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM signals;"

# 15. Web dashboard'u test et
log "Web dashboard test ediliyor..."
sleep 15
curl -s http://localhost:5000/api/signals > /dev/null && log "✅ Web dashboard erişilebilir" || error "❌ Web dashboard erişilemiyor"

echo ""
log "🎉 CentOS Hızlı Düzeltme Tamamlandı!"
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