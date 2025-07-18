# PostgreSQL Tablo Oluşturma Rehberi

## Sorun
PostgreSQL kullanıcısı ve veritabanı var ama tablolar otomatik olarak oluşturulmuyor.

## Çözüm Yöntemleri

### 1. Otomatik Çözüm (Önerilen)

```bash
# Sunucuda çalıştır
cd /var/www/html/kahin
chmod +x quick_fix.sh
./quick_fix.sh
```

Bu script:
- PostgreSQL bağlantısını test eder
- Virtual environment'ı aktifleştirir
- Migration'ı çalıştırır
- Başarısızsa zorla tablo oluşturur
- Manuel tablo oluşturma scriptini çalıştırır
- Servisleri yeniden başlatır

### 2. Manuel Çözüm

```bash
# Sunucuda çalıştır
cd /var/www/html/kahin
chmod +x manual_create_tables.sh
./manual_create_tables.sh
```

Bu script:
- PostgreSQL bağlantısını test eder
- Tüm tabloları manuel olarak oluşturur
- Indexleri oluşturur
- Test verisi ekler
- Servisleri yeniden başlatır

### 3. Python Script ile Çözüm

```bash
# Sunucuda çalıştır
cd /var/www/html/kahin
source venv/bin/activate
python3 force_create_tables.py
```

Bu script:
- Veritabanı bağlantısını test eder
- Tabloları zorla oluşturur
- Indexleri oluşturur
- Tablo erişimini test eder

### 4. Test Etme

```bash
# Tabloları test et
cd /var/www/html/kahin
source venv/bin/activate
python3 test_tables.py
```

## Tablo Yapısı

### Signals Tablosu
```sql
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    direction VARCHAR(10),
    ai_score DECIMAL(5,4),
    ta_strength DECIMAL(5,4),
    whale_score DECIMAL(5,4),
    social_score DECIMAL(5,4),
    news_score DECIMAL(5,4),
    timestamp VARCHAR(50),
    predicted_gain DECIMAL(10,4),
    predicted_duration VARCHAR(20),
    entry_price DECIMAL(15,8) DEFAULT NULL,
    exit_price DECIMAL(15,8) DEFAULT NULL,
    result VARCHAR(20) DEFAULT NULL,
    realized_gain DECIMAL(10,4) DEFAULT NULL,
    duration DECIMAL(10,4) DEFAULT NULL,
    take_profit DECIMAL(15,8) DEFAULT NULL,
    stop_loss DECIMAL(15,8) DEFAULT NULL,
    support_level DECIMAL(15,8) DEFAULT NULL,
    resistance_level DECIMAL(15,8) DEFAULT NULL,
    target_time_hours DECIMAL(10,2) DEFAULT NULL,
    max_hold_time_hours DECIMAL(10,2) DEFAULT 24.0,
    predicted_breakout_threshold DECIMAL(10,4) DEFAULT NULL,
    actual_max_gain DECIMAL(10,4) DEFAULT NULL,
    actual_max_loss DECIMAL(10,4) DEFAULT NULL,
    breakout_achieved BOOLEAN DEFAULT FALSE,
    breakout_time_hours DECIMAL(10,4) DEFAULT NULL,
    predicted_breakout_time_hours DECIMAL(10,4) DEFAULT NULL,
    risk_reward_ratio DECIMAL(10,4) DEFAULT NULL,
    actual_risk_reward_ratio DECIMAL(10,4) DEFAULT NULL,
    volatility_score DECIMAL(5,4) DEFAULT NULL,
    trend_strength DECIMAL(5,4) DEFAULT NULL,
    market_regime VARCHAR(20) DEFAULT NULL,
    signal_quality_score DECIMAL(5,4) DEFAULT NULL,
    success_metrics JSONB DEFAULT NULL,
    volume_score DECIMAL(5,4) DEFAULT NULL,
    momentum_score DECIMAL(5,4) DEFAULT NULL,
    pattern_score DECIMAL(5,4) DEFAULT NULL,
    order_book_imbalance DECIMAL(10,4) DEFAULT NULL,
    top_bid_walls TEXT DEFAULT NULL,
    top_ask_walls TEXT DEFAULT NULL,
    whale_direction_score DECIMAL(10,4) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Performance Tablosu
```sql
CREATE TABLE performance (
    id SERIAL PRIMARY KEY,
    date DATE,
    total_signals INTEGER DEFAULT 0,
    successful_signals INTEGER DEFAULT 0,
    failed_signals INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 0.0,
    average_profit DECIMAL(10,4) DEFAULT 0.0,
    total_profit DECIMAL(15,8) DEFAULT 0.0,
    max_drawdown DECIMAL(10,4) DEFAULT 0.0,
    sharpe_ratio DECIMAL(10,4) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### System Logs Tablosu
```sql
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(10),
    message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    module VARCHAR(50),
    error_details TEXT
);
```

## Indexler

```sql
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date);
```

## Kontrol Komutları

### Tabloları Listele
```bash
sudo -u postgres psql -d kahin_ultima -c "\dt"
```

### Kayıt Sayılarını Kontrol Et
```bash
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM signals;"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM performance;"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM system_logs;"
```

### Tablo Yapısını Görüntüle
```bash
sudo -u postgres psql -d kahin_ultima -c "\d signals"
```

### Indexleri Listele
```bash
sudo -u postgres psql -d kahin_ultima -c "\di"
```

## Veritabanı Bilgileri

- **Host**: localhost
- **Port**: 5432
- **Database**: kahin_ultima
- **User**: laravel
- **Password**: secret

## Sorun Giderme

### 1. Bağlantı Hatası
```bash
# PostgreSQL servisini kontrol et
systemctl status postgresql

# PostgreSQL'i yeniden başlat
systemctl restart postgresql
```

### 2. Yetki Hatası
```bash
# Kullanıcı yetkilerini kontrol et
sudo -u postgres psql -c "\du"

# Yetkileri düzelt
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO laravel;"
sudo -u postgres psql -c "ALTER USER laravel CREATEDB;"
```

### 3. Tablo Oluşturma Hatası
```bash
# Veritabanını yeniden oluştur
sudo -u postgres psql -c "DROP DATABASE IF EXISTS kahin_ultima;"
sudo -u postgres psql -c "CREATE DATABASE kahin_ultima;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE kahin_ultima TO laravel;"

# Tabloları yeniden oluştur
./manual_create_tables.sh
```

## Log Dosyaları

- `logs/force_create_tables.log` - Zorla tablo oluşturma logları
- `logs/test_tables.log` - Tablo test logları
- `logs/database_migration.log` - Migration logları

## Web Dashboard

- **URL**: http://185.209.228.189:5000
- **API**: http://185.209.228.189:5000/api/signals

## Servis Durumları

```bash
# Servisleri kontrol et
systemctl status kahinali.service
systemctl status kahinali-web.service

# Servisleri yeniden başlat
systemctl restart kahinali.service
systemctl restart kahinali-web.service
``` 