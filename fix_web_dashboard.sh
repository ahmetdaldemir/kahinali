#!/bin/bash

# Kahinali Web Dashboard Düzeltme Scripti
# Sunucu: 185.209.228.189

echo "🚀 Web Dashboard Düzeltme Başlatılıyor..."

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

# 2. Virtual environment'ı aktifleştir
source venv/bin/activate

# 3. Web servisini kontrol et
log "Web servisi durumu kontrol ediliyor..."
systemctl status kahinali-web.service --no-pager

# 4. Web servisini durdur
log "Web servisi durduruluyor..."
systemctl stop kahinali-web.service

# 5. Port 5000'i kontrol et
log "Port 5000 kontrol ediliyor..."
netstat -tlnp | grep :5000 || echo "Port 5000 boş"

# 6. Web dashboard dosyalarını kontrol et
log "Web dashboard dosyaları kontrol ediliyor..."
ls -la app/
ls -la app/templates/

# 7. Flask uygulamasını test et
log "Flask uygulaması test ediliyor..."
python3 -c "
from app.web import app
print('Flask app başarıyla import edildi')
"

# 8. Web dashboard'u manuel başlat
log "Web dashboard manuel başlatılıyor..."
cd /var/www/html/kahin
source venv/bin/activate

# Flask uygulamasını arka planda başlat
nohup python3 -m flask --app app.web run --host=0.0.0.0 --port=5000 > logs/web_dashboard.log 2>&1 &
WEB_PID=$!

# PID'i kaydet
echo $WEB_PID > /tmp/web_dashboard.pid
log "Web dashboard PID: $WEB_PID"

# 9. Servisin başlamasını bekle
log "Web dashboard başlaması bekleniyor..."
sleep 10

# 10. Port kontrolü
log "Port kontrolü yapılıyor..."
netstat -tlnp | grep :5000

# 11. Process kontrolü
log "Process kontrolü yapılıyor..."
ps aux | grep flask

# 12. Log dosyasını kontrol et
log "Web dashboard logları kontrol ediliyor..."
tail -20 logs/web_dashboard.log

# 13. Web dashboard'u test et
log "Web dashboard test ediliyor..."
sleep 5

# Localhost test
curl -s http://localhost:5000/api/signals > /dev/null
if [ $? -eq 0 ]; then
    log "✅ Localhost web dashboard erişilebilir"
else
    error "❌ Localhost web dashboard erişilemiyor"
fi

# 0.0.0.0 test
curl -s http://0.0.0.0:5000/api/signals > /dev/null
if [ $? -eq 0 ]; then
    log "✅ 0.0.0.0 web dashboard erişilebilir"
else
    error "❌ 0.0.0.0 web dashboard erişilemiyor"
fi

# 14. Firewall kontrolü
log "Firewall kontrolü yapılıyor..."
ufw status
ufw allow 5000

# 15. Systemd service dosyasını düzelt
log "Systemd service dosyası düzeltiliyor..."
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

# 16. Systemd'yi yeniden yükle
systemctl daemon-reload

# 17. Manuel process'i durdur
if [ -f /tmp/web_dashboard.pid ]; then
    MANUAL_PID=$(cat /tmp/web_dashboard.pid)
    log "Manuel process durduruluyor: $MANUAL_PID"
    kill $MANUAL_PID 2>/dev/null || true
    rm -f /tmp/web_dashboard.pid
fi

# 18. Systemd servisini başlat
log "Systemd web servisi başlatılıyor..."
systemctl enable kahinali-web.service
systemctl start kahinali-web.service

# 19. Servis durumunu kontrol et
log "Web servisi durumu kontrol ediliyor..."
systemctl status kahinali-web.service --no-pager

# 20. Servisin başlamasını bekle
log "Web servisi başlaması bekleniyor..."
sleep 15

# 21. Final test
log "Final web dashboard testi yapılıyor..."
sleep 5

# Localhost test
curl -s http://localhost:5000/api/signals > /dev/null
if [ $? -eq 0 ]; then
    log "✅ Final test: Localhost web dashboard erişilebilir"
else
    error "❌ Final test: Localhost web dashboard erişilemiyor"
fi

# 22. Log dosyasını kontrol et
log "Son log kontrolü..."
tail -10 logs/web_dashboard.log

# 23. Port durumu
log "Final port durumu:"
netstat -tlnp | grep :5000

# 24. Process durumu
log "Final process durumu:"
ps aux | grep flask

echo ""
log "🎉 Web Dashboard Düzeltme Tamamlandı!"
echo ""
echo "📊 Web Dashboard Bilgileri:"
echo "   - URL: http://185.209.228.189:5000"
echo "   - API: http://185.209.228.189:5000/api/signals"
echo "   - Local: http://localhost:5000"
echo ""
echo "🔧 Kontrol Komutları:"
echo "   - Servis durumu: systemctl status kahinali-web.service"
echo "   - Logları görüntüle: journalctl -u kahinali-web.service -f"
echo "   - Port kontrolü: netstat -tlnp | grep :5000"
echo "   - Process kontrolü: ps aux | grep flask"
echo ""
echo "📋 Sorun Giderme:"
echo "   - Servisi yeniden başlat: systemctl restart kahinali-web.service"
echo "   - Logları temizle: journalctl --vacuum-time=1d"
echo "   - Firewall kontrolü: ufw status" 