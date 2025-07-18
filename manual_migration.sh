#!/bin/bash

# Kahinali Manuel Migration Scripti
# Sunucu: 185.209.228.189

echo "🚀 Kahinali Manuel Migration Başlatılıyor..."

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

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# 1. Proje dizinine git
cd /var/www/html/kahin

# 2. Virtual environment'ı aktifleştir
source venv/bin/activate

# 3. PostgreSQL durumunu kontrol et
log "PostgreSQL durumu kontrol ediliyor..."
systemctl status postgresql --no-pager

# 4. Veritabanı bağlantısını test et
log "Veritabanı bağlantısı test ediliyor..."
python3 setup_postgresql.py

# 5. Migration'ı çalıştır
log "Veritabanı migration başlatılıyor..."
python3 database_migration.py

# 6. Migration sonucunu kontrol et
if [ $? -eq 0 ]; then
    log "✅ Migration başarılı!"
else
    error "❌ Migration başarısız!"
    log "Tekrar deneniyor..."
    sleep 5
    python3 database_migration.py
fi

# 7. Tabloları kontrol et
log "Veritabanı tabloları kontrol ediliyor..."
sudo -u postgres psql -d kahin_ultima -c "\dt"

# 8. Signals tablosu kayıt sayısını kontrol et
log "Signals tablosu kayıt sayısı:"
sudo -u postgres psql -d kahin_ultima -c "SELECT COUNT(*) FROM signals;"

# 9. Tablo yapısını kontrol et
log "Signals tablosu yapısı:"
sudo -u postgres psql -d kahin_ultima -c "\d signals"

# 10. Test verisi ekle (opsiyonel)
read -p "Test verisi eklemek ister misiniz? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Test verisi ekleniyor..."
    sudo -u postgres psql -d kahin_ultima -c "
    INSERT INTO signals (symbol, timeframe, direction, ai_score, ta_strength, whale_score, social_score, news_score, timestamp, entry_price, current_price) 
    VALUES ('BTC/USDT', '1h', 'BUY', 0.75, 0.80, 0.70, 0.65, 0.60, '$(date +'%Y-%m-%d %H:%M:%S')', 45000.00, 45000.00)
    ON CONFLICT DO NOTHING;
    "
    log "Test verisi eklendi!"
fi

echo ""
log "🎉 Manuel migration tamamlandı!"
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
echo "📊 Migration Log:"
echo "   - Log dosyası: logs/database_migration.log" 