# PostgreSQL Kurulum ve Yapılandırma Rehberi

## 🚀 KAHİN Ultima PostgreSQL Kurulumu

Bu rehber, KAHİN Ultima sistemini PostgreSQL veritabanı ile çalışacak şekilde yapılandırmanızı sağlar.

## 📋 Gereksinimler

- Windows 10/11
- Python 3.8+
- PostgreSQL 13+ (kurulum sırasında otomatik olarak kurulacak)

## 🔧 Kurulum Adımları

### 1. PostgreSQL Kurulumu

#### Otomatik Kurulum (Önerilen)
```bash
python setup_postgresql.py
```

Bu script:
- PostgreSQL'in kurulu olup olmadığını kontrol eder
- Kurulu değilse kurulum talimatlarını gösterir
- Veritabanını oluşturur
- Bağlantıyı test eder
- `.env` dosyasını oluşturur

#### Manuel Kurulum
1. [PostgreSQL İndirme Sayfası](https://www.postgresql.org/download/windows/)
2. PostgreSQL installer'ı indirin ve çalıştırın
3. Kurulum sırasında şu ayarları yapın:
   - Port: `5432`
   - Superuser: `kahin_user`
   - Password: `kahin_password`
   - Database: `kahin_ultima`

### 2. Python Sanal Ortamı

```bash
# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı aktifleştir (Windows)
venv\Scripts\activate

# Gereksinimleri yükle
pip install -r requirements.txt
```

### 3. Veritabanı Yapılandırması

#### Otomatik Yapılandırma
```bash
python setup_postgresql.py
```

#### Manuel Yapılandırma
1. `.env` dosyasını düzenleyin:
```env
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=kahin_ultima
POSTGRES_USER=kahin_user
POSTGRES_PASSWORD=kahin_password
DATABASE_URL=postgresql://kahin_user:kahin_password@localhost:5432/kahin_ultima

# API Keys (Bunları kendi API anahtarlarınızla değiştirin)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
# ... diğer API anahtarları
```

### 4. Veritabanı Bağlantı Testi

```bash
python -c "
from config import Config
import psycopg2
try:
    conn = psycopg2.connect(Config.DATABASE_URL)
    print('✅ PostgreSQL bağlantısı başarılı!')
    conn.close()
except Exception as e:
    print(f'❌ Bağlantı hatası: {e}')
"
```

## 🗄️ Veritabanı Yapısı

### Signals Tablosu
```sql
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    ai_score FLOAT NOT NULL,
    ta_strength FLOAT NOT NULL,
    whale_score FLOAT NOT NULL,
    social_score FLOAT NOT NULL,
    news_score FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    predicted_gain FLOAT NOT NULL,
    predicted_duration VARCHAR(50) NOT NULL,
    result VARCHAR(20),
    realized_gain FLOAT,
    duration FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Index'ler
- `idx_signals_symbol_timeframe` - Symbol ve timeframe için composite index
- `idx_signals_timestamp` - Timestamp için descending index
- `idx_signals_result` - Result için index

## 🔍 Performans Optimizasyonları

### 1. Connection Pooling
```python
# config.py'de zaten yapılandırılmış
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
```

### 2. Query Optimizasyonu
- Prepared statements kullanımı
- Parametrized queries
- Index'lerin doğru kullanımı

### 3. Backup Stratejisi
```bash
# Otomatik backup scripti
pg_dump -h localhost -U kahin_user -d kahin_ultima > backup_$(date +%Y%m%d_%H%M%S).sql
```

## 🚨 Sorun Giderme

### Bağlantı Hatası
```
psycopg2.OperationalError: connection to server at "localhost" (127.0.0.1), port 5432 failed
```

**Çözüm:**
1. PostgreSQL servisinin çalıştığını kontrol edin
2. Windows Services'den "postgresql-x64-13" servisini başlatın
3. Firewall ayarlarını kontrol edin

### Yetki Hatası
```
psycopg2.OperationalError: FATAL: password authentication failed
```

**Çözüm:**
1. `.env` dosyasındaki şifreyi kontrol edin
2. PostgreSQL'de şifreyi sıfırlayın:
```sql
ALTER USER kahin_user PASSWORD 'kahin_password';
```

### Veritabanı Bulunamadı
```
psycopg2.OperationalError: database "kahin_ultima" does not exist
```

**Çözüm:**
```sql
CREATE DATABASE kahin_ultima;
```

## 📊 Monitoring

### Veritabanı İstatistikleri
```sql
-- Tablo boyutları
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE tablename = 'signals';

-- Index kullanımı
SELECT 
    indexrelname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE relname = 'signals';
```

### Performans Metrikleri
- Sinyal sayısı: `SELECT COUNT(*) FROM signals;`
- Başarı oranı: `SELECT AVG(CASE WHEN result = 'profit' THEN 1 ELSE 0 END) * 100 FROM signals WHERE result IS NOT NULL;`
- Ortalama kazanç: `SELECT AVG(realized_gain) FROM signals WHERE result = 'profit';`

## 🔄 Migration (SQLite'dan PostgreSQL'e)

Eğer mevcut SQLite veritabanınız varsa:

```python
import pandas as pd
from sqlalchemy import create_engine

# SQLite'dan veri oku
sqlite_engine = create_engine('sqlite:///kahin_ultima.db')
df = pd.read_sql('SELECT * FROM signals', sqlite_engine)

# PostgreSQL'e yaz
postgres_engine = create_engine('postgresql://kahin_user:kahin_password@localhost:5432/kahin_ultima')
df.to_sql('signals', postgres_engine, if_exists='append', index=False)
```

## ✅ Kurulum Tamamlandı

Kurulum tamamlandıktan sonra:

1. **Sistemi başlatın:**
```bash
python main.py
```

2. **Web panelini açın:**
```
http://localhost:5000
```

3. **Telegram botunu test edin:**
```bash
python -c "from modules.telegram_bot import TelegramBot; bot = TelegramBot(); bot.send_message('Test mesajı')"
```

## 📞 Destek

Sorun yaşarsanız:
1. Log dosyalarını kontrol edin: `logs/kahin_ultima.log`
2. PostgreSQL log'larını kontrol edin
3. Bağlantı ayarlarını doğrulayın
4. API anahtarlarının doğru olduğundan emin olun

---

**🎉 Tebrikler! KAHİN Ultima PostgreSQL ile çalışmaya hazır!** 