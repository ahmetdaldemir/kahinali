# Web Dashboard Sorun Çözüm Rehberi

## Sorun
Web dashboard erişilemiyor hatası alınıyor.

## Hızlı Çözüm

### 1. Otomatik Düzeltme (Önerilen)
```bash
cd /var/www/html/kahin
chmod +x fix_web_dashboard.sh
./fix_web_dashboard.sh
```

### 2. Manuel Düzeltme
```bash
cd /var/www/html/kahin
source venv/bin/activate

# Web servisini durdur
systemctl stop kahinali-web.service

# Port 5000'i kontrol et
netstat -tlnp | grep :5000

# Flask uygulamasını manuel başlat
nohup python3 -m flask --app app.web run --host=0.0.0.0 --port=5000 > logs/web_dashboard.log 2>&1 &

# Test et
curl -s http://localhost:5000/api/signals
```

### 3. Systemd Servis Düzeltme
```bash
# Service dosyasını düzelt
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

# Systemd'yi yeniden yükle
systemctl daemon-reload
systemctl enable kahinali-web.service
systemctl start kahinali-web.service
```

## Detaylı Sorun Giderme

### 1. Flask Uygulaması Testi
```bash
cd /var/www/html/kahin
source venv/bin/activate
python3 test_web_dashboard.py
```

### 2. Port Kontrolü
```bash
# Port 5000'i kontrol et
netstat -tlnp | grep :5000

# Firewall kontrolü
ufw status
ufw allow 5000
```

### 3. Process Kontrolü
```bash
# Flask process'lerini bul
ps aux | grep flask

# Process'i durdur
pkill -f flask
```

### 4. Log Kontrolü
```bash
# Systemd logları
journalctl -u kahinali-web.service -f

# Flask logları
tail -f logs/web_dashboard.log
```

### 5. Bağımlılık Kontrolü
```bash
cd /var/www/html/kahin
source venv/bin/activate

# Flask'ı test et
python3 -c "from flask import Flask; print('Flask OK')"

# App'i test et
python3 -c "from app.web import app; print('App OK')"
```

## Yaygın Sorunlar ve Çözümleri

### 1. Port Zaten Kullanımda
```bash
# Port 5000'i kullanan process'i bul
lsof -i :5000

# Process'i durdur
kill -9 <PID>
```

### 2. Virtual Environment Sorunu
```bash
# Virtual environment'ı yeniden oluştur
cd /var/www/html/kahin
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Yetki Sorunu
```bash
# Dizin yetkilerini düzelt
chown -R root:root /var/www/html/kahin
chmod -R 755 /var/www/html/kahin
```

### 4. Flask Import Hatası
```bash
# PYTHONPATH'i ayarla
export PYTHONPATH=/var/www/html/kahin:$PYTHONPATH

# Flask uygulamasını test et
python3 -c "import sys; sys.path.append('/var/www/html/kahin'); from app.web import app; print('OK')"
```

### 5. Database Bağlantı Sorunu
```bash
# Database bağlantısını test et
cd /var/www/html/kahin
source venv/bin/activate
python3 -c "from config import Config; from sqlalchemy import create_engine; engine = create_engine(Config.DATABASE_URL); print('DB OK')"
```

## Test Komutları

### 1. Basit Test
```bash
# Localhost test
curl -s http://localhost:5000/api/signals

# External test
curl -s http://185.209.228.189:5000/api/signals
```

### 2. Detaylı Test
```bash
# Tüm endpoint'leri test et
curl -s http://localhost:5000/ | head -5
curl -s http://localhost:5000/api/signals | jq '.[0:2]'
curl -s http://localhost:5000/api/performance | jq '.'
```

### 3. Performance Test
```bash
# Yük testi
ab -n 100 -c 10 http://localhost:5000/api/signals
```

## Monitoring

### 1. Real-time Monitoring
```bash
# Log takibi
tail -f logs/web_dashboard.log

# Process takibi
watch -n 1 'ps aux | grep flask'

# Port takibi
watch -n 1 'netstat -tlnp | grep :5000'
```

### 2. Health Check
```bash
# Health check scripti
while true; do
    if curl -s http://localhost:5000/api/signals > /dev/null; then
        echo "$(date): Web dashboard OK"
    else
        echo "$(date): Web dashboard ERROR"
        systemctl restart kahinali-web.service
    fi
    sleep 30
done
```

## Web Dashboard Bilgileri

### URLs
- **Ana Sayfa**: http://185.209.228.189:5000
- **API**: http://185.209.228.189:5000/api/signals
- **Performance**: http://185.209.228.189:5000/api/performance
- **Stats**: http://185.209.228.189:5000/api/stats

### API Endpoints
- `GET /api/signals` - Tüm sinyaller
- `GET /api/signals/open` - Açık sinyaller
- `GET /api/signals/closed` - Kapalı sinyaller
- `GET /api/performance` - Performans verileri
- `GET /api/stats` - İstatistikler

### Servis Komutları
```bash
# Servis durumu
systemctl status kahinali-web.service

# Servisi başlat
systemctl start kahinali-web.service

# Servisi durdur
systemctl stop kahinali-web.service

# Servisi yeniden başlat
systemctl restart kahinali-web.service

# Servisi etkinleştir
systemctl enable kahinali-web.service
```

## Sorun Giderme Checklist

- [ ] Virtual environment aktif mi?
- [ ] Flask uygulaması import edilebiliyor mu?
- [ ] Port 5000 boş mu?
- [ ] Firewall port 5000'e izin veriyor mu?
- [ ] Database bağlantısı çalışıyor mu?
- [ ] Systemd servisi aktif mi?
- [ ] Log dosyalarında hata var mı?
- [ ] Process çalışıyor mu?

## Emergency Fix

Eğer hiçbir şey çalışmıyorsa:

```bash
cd /var/www/html/kahin

# Her şeyi durdur
pkill -f flask
systemctl stop kahinali-web.service

# Virtual environment'ı yeniden oluştur
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Flask uygulamasını manuel başlat
nohup python3 -m flask --app app.web run --host=0.0.0.0 --port=5000 > logs/web_dashboard.log 2>&1 &

# Test et
sleep 10
curl -s http://localhost:5000/api/signals
``` 