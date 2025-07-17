# KahinUltima Sistem Deployment Rehberi

## 📋 Sistem Gereksinimleri

### Donanım Gereksinimleri:
- **CPU**: Minimum 4 çekirdek (8 çekirdek önerilen)
- **RAM**: Minimum 8GB (16GB önerilen)
- **Disk**: Minimum 50GB boş alan (SSD önerilen)
- **İnternet**: Stabil ve hızlı bağlantı

### Yazılım Gereksinimleri:
- **İşletim Sistemi**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **Python**: 3.8 veya üzeri
- **PostgreSQL**: 13 veya üzeri
- **Git**: Sürüm kontrolü için

## 🚀 Kurulum Adımları

### 1. Sistem Hazırlığı

#### Windows için:
```bash
# Python kurulumu (python.org'dan indirin)
# PostgreSQL kurulumu (postgresql.org'dan indirin)
# Git kurulumu (git-scm.com'dan indirin)
```

#### Linux (Ubuntu) için:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv postgresql postgresql-contrib git
```

#### macOS için:
```bash
# Homebrew kurulumu
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Gerekli paketler
brew install python3 postgresql git
```

### 2. Proje Kopyalama

```bash
# GitHub'dan klonlama (eğer repo varsa)
git clone https://github.com/kullanici/kahin-ultima.git
cd kahin-ultima

# Veya mevcut dosyaları kopyalama
# Tüm proje klasörünü yeni bilgisayara kopyalayın
```

### 3. Python Sanal Ortam Kurulumu

```bash
# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı aktifleştirme
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Gerekli paketleri kurma
pip install -r requirements.txt
```

### 4. PostgreSQL Veritabanı Kurulumu

#### Veritabanı oluşturma:
```sql
-- PostgreSQL'e bağlanın
sudo -u postgres psql

-- Veritabanı oluşturma
CREATE DATABASE kahin_ultima;

-- Kullanıcı oluşturma
CREATE USER kahin_user WITH PASSWORD 'your_password';

-- Yetkileri verme
GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO kahin_user;

-- Çıkış
\q
```

### 5. Çevre Değişkenleri Ayarlama

`.env` dosyası oluşturun:

```env
# API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Twitter API
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=KAHIN_Ultima_Bot/1.0

# News API
NEWS_API_KEY=your_news_api_key

# PostgreSQL Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=kahin_ultima
POSTGRES_USER=kahin_user
POSTGRES_PASSWORD=your_password

# Flask
FLASK_SECRET_KEY=kahin-ultima-secret-key-2024
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

### 6. Veritabanı Şeması Kurulumu

```bash
# Veritabanı şemasını oluşturma
python setup_postgresql.py

# Veya manuel olarak:
psql -U kahin_user -d kahin_ultima -f database_schema.sql
```

### 7. Sistem Testi

```bash
# Temel testler
python test_system.py
python test_api.py
python test_db.py

# Veri toplama testi
python test_data_collection.py
```

## 🔧 Konfigürasyon Ayarları

### 1. config.py Dosyasını Düzenleme

```python
# Sistem parametrelerini yeni ortama göre ayarlayın
MAX_COINS_TO_TRACK = 400  # İzlenecek coin sayısı
SIGNAL_TIMEFRAMES = ['1h', '4h', '1d']  # Zaman dilimleri
```

### 2. Log Klasörü Oluşturma

```bash
mkdir logs
mkdir data
mkdir signals
mkdir models
```

### 3. Gerekli Klasörlerin İzinleri

```bash
# Linux/macOS için
chmod 755 logs data signals models
```

## 🚀 Sistem Başlatma

### 1. Ana Sistem Başlatma

```bash
# Ana sistemi başlatma
python main.py

# Veya
python start_system.py
```

### 2. Web Arayüzü Başlatma

```bash
# Web sunucusunu başlatma
python app/web.py

# Tarayıcıda açın: http://localhost:5000
```

### 3. Arka Plan İşlemleri

```bash
# Model eğitimi
python train_models.py

# Veri toplama
python collect_2year_data.py
```

## 📊 Sistem İzleme

### 1. Log Dosyaları

```bash
# Sistem logları
tail -f logs/kahin_ultima.log

# Model eğitim logları
tail -f logs/model_training.log
```

### 2. Sistem Durumu Kontrolü

```bash
# Sistem durumu
python system_status_report.py

# API testi
python test_api.py
```

## 🔄 Otomatik Başlatma (Opsiyonel)

### Windows için (Task Scheduler):

1. **Başlat** > **Görev Zamanlayıcısı**
2. **Temel Görev Oluştur**
3. Sistem başlangıcında çalışacak şekilde ayarlayın

### Linux için (systemd):

```bash
# Servis dosyası oluşturma
sudo nano /etc/systemd/system/kahin-ultima.service
```

```ini
[Unit]
Description=KahinUltima Trading System
After=network.target postgresql.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/kahin-ultima
Environment=PATH=/path/to/kahin-ultima/venv/bin
ExecStart=/path/to/kahin-ultima/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Servisi etkinleştirme
sudo systemctl enable kahin-ultima
sudo systemctl start kahin-ultima
```

## 🔒 Güvenlik Ayarları

### 1. Firewall Ayarları

```bash
# Windows Firewall
# Sadece gerekli portları açın (5000, 5432)

# Linux UFW
sudo ufw allow 5000
sudo ufw allow 5432
```

### 2. API Key Güvenliği

- API anahtarlarını güvenli şekilde saklayın
- `.env` dosyasını git'e eklemeyin
- Düzenli olarak API anahtarlarını değiştirin

## 📈 Performans Optimizasyonu

### 1. Sistem Ayarları

```python
# config.py'de performans ayarları
MAX_WORKERS = 5  # Paralel işlem sayısı
BATCH_SIZE = 10  # Batch işleme boyutu
CACHE_DURATION = 300  # Cache süresi (saniye)
```

### 2. Veritabanı Optimizasyonu

```sql
-- PostgreSQL performans ayarları
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

## 🆘 Sorun Giderme

### Yaygın Sorunlar:

1. **PostgreSQL Bağlantı Hatası**
   ```bash
   # PostgreSQL servisini kontrol edin
   sudo systemctl status postgresql
   ```

2. **Python Paket Hatası**
   ```bash
   # Sanal ortamı yeniden oluşturun
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **API Key Hatası**
   ```bash
   # .env dosyasını kontrol edin
   cat .env
   ```

4. **Disk Alanı Sorunu**
   ```bash
   # Disk kullanımını kontrol edin
   df -h
   ```

## 📞 Destek

Sorun yaşarsanız:
1. Log dosyalarını kontrol edin
2. Sistem durumu raporunu çalıştırın
3. Test scriptlerini çalıştırın
4. Gerekirse sistemi yeniden başlatın

## 🔄 Güncelleme

```bash
# Sistemi güncelleme
git pull origin main
pip install -r requirements.txt --upgrade
python setup_postgresql.py
```

---

**Not**: Bu rehber genel kurulum adımlarını içerir. Spesifik ortamınıza göre bazı ayarlamalar gerekebilir. 