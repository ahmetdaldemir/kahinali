#!/bin/bash

# Kahinali Manuel Tablo Oluşturma Scripti
# Sunucu: 185.209.228.189

echo "🚀 Kahinali Manuel Tablo Oluşturma Başlatılıyor..."

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

# 3. PostgreSQL bağlantısını test et
log "PostgreSQL bağlantısı test ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "SELECT 1;" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    log "✅ PostgreSQL bağlantısı başarılı"
else
    error "❌ PostgreSQL bağlantısı başarısız"
    exit 1
fi

# 4. Mevcut tabloları kontrol et
log "Mevcut tablolar kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "\dt"

# 5. Signals tablosunu oluştur
log "Signals tablosu oluşturuluyor..."
sudo -u postgres psql -d kahin_ultima -c "
CREATE TABLE IF NOT EXISTS signals (
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
"

# 6. Performance tablosunu oluştur
log "Performance tablosu oluşturuluyor..."
sudo -u postgres psql -d kahin_ultima -c "
CREATE TABLE IF NOT EXISTS performance (
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
"

# 7. System logs tablosunu oluştur
log "System logs tablosu oluşturuluyor..."
sudo -u postgres psql -d kahin_ultima -c "
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(10),
    message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    module VARCHAR(50),
    error_details TEXT
);
"

# 8. Indexleri oluştur
log "Indexler oluşturuluyor..."
sudo -u postgres psql -d kahin_ultima -c "
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction);
CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date);
"

# 9. Tabloları kontrol et
log "Tablolar kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "\dt"

# 10. Test verisi ekle
log "Test verisi ekleniyor..."
sudo -u postgres psql -d kahin_ultima -c "
INSERT INTO signals (symbol, timeframe, direction, ai_score, ta_strength, whale_score, social_score, news_score, timestamp, entry_price, current_price) 
VALUES ('BTC/USDT', '1h', 'BUY', 0.75, 0.80, 0.70, 0.65, 0.60, '$(date +'%Y-%m-%d %H:%M:%S')', 45000.00, 45000.00)
ON CONFLICT DO NOTHING;
"

# 11. Kayıt sayılarını kontrol et
log "Kayıt sayıları kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM signals;"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM performance;"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM system_logs;"

# 12. Python scriptini çalıştır
log "Python tablo oluşturma scripti çalıştırılıyor..."
python3 force_create_tables.py

# 13. Servisleri yeniden başlat
log "Servisler yeniden başlatılıyor..."
systemctl restart kahinali.service
systemctl restart kahinali-web.service

# 14. Servis durumlarını kontrol et
log "Servis durumları kontrol ediliyor..."
systemctl status kahinali.service --no-pager
systemctl status kahinali-web.service --no-pager

# 15. Web dashboard'u test et
log "Web dashboard test ediliyor..."
sleep 10
curl -s http://localhost:5000/api/signals > /dev/null && log "✅ Web dashboard erişilebilir" || error "❌ Web dashboard erişilemiyor"

echo ""
log "🎉 Manuel tablo oluşturma tamamlandı!"
echo ""
echo "📋 Veritabanı Bilgileri:"
echo "   - Host: localhost"
echo "   - Port: 5432"
echo "   - Database: kahin_ultima"
echo "   - User: laravel"
echo "   - Password: secret"
echo ""
echo "🔧 Kontrol Komutları:"
echo "   - Tabloları listele: sudo -u postgres psql -d kahin_ultima -c '\dt'"
echo "   - Signals kayıt sayısı: sudo -u postgres psql -d kahin_ultima -c 'SELECT COUNT(*) FROM signals;'"
echo "   - Tablo yapısı: sudo -u postgres psql -d kahin_ultima -c '\d signals'"
echo ""
echo "📊 Web Dashboard:"
echo "   - URL: http://185.209.228.189:5000"
echo "   - API: http://185.209.228.189:5000/api/signals" 