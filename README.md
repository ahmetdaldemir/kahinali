# KAHİN Ultima - Yapay Zeka Destekli Kripto Sinyal Sistemi

KAHİN Ultima, gelişmiş yapay zeka ve veri analizi teknolojilerini kullanarak kripto para sinyalleri üreten kapsamlı bir sistemdir.

## 🚀 Özellikler

### 🤖 AI Modülü
- **LSTM + Random Forest** modelleri ile yön tahmini
- Geçmiş fiyat verilerinden etiketli veri üretimi
- Otomatik günlük model yeniden eğitimi
- Tahmin skoru (0-1 arası)

### 📊 Teknik Analiz
- **RSI, MACD, Stochastic, Bollinger Bands**
- **CCI, ADX, EMA, SMA, OBV, VWAP, MFI**
- Tüm indikatörlerin normalize edilmesi
- AI ile birleştirilmiş sinyal üretimi

### 📱 Sosyal Medya Analizi
- **Reddit** ve **Twitter** veri toplama
- NLP ile sentiment analizi
- En popüler coin eşleştirmesi
- Gerçek zamanlı duygu skorları

### 📰 Haber Analizi
- **RSS** ve **News API** entegrasyonu
- Kripto haberlerinin sentiment analizi
- Haber etkisinin sinyal modeline dahil edilmesi

### 🐋 Whale Tracker
- **Binance Order Book** analizi
- Büyük hacimli işlemlerin tespiti
- Whale puanının AI modeline dahil edilmesi

### 🔔 Telegram Entegrasyonu
- Yeni sinyal bildirimleri
- `/signals` komutu ile aktif sinyaller
- `/performance` komutu ile başarı oranı

### 📈 Web Paneli
- **Flask** tabanlı modern arayüz
- Sinyal görselleştirme ve filtreleme
- Geçmiş performans analizi
- Gerçek zamanlı güncellemeler

### 📊 Performans Takibi
- Başarı oranı hesaplama
- Ortalama yükseliş analizi
- Coin bazlı performans metrikleri

## 🛠️ Kurulum

### 1. Gereksinimler
```bash
Python 3.8+
PostgreSQL (opsiyonel)
```

### 2. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 3. Konfigürasyon
```bash
# env_example.txt dosyasını .env olarak kopyalayın
cp env_example.txt .env

# .env dosyasını düzenleyerek API anahtarlarınızı ekleyin
nano .env
```

### 4. Veritabanı Kurulumu (Opsiyonel)
```sql
CREATE DATABASE kahin_ultima;
```

## 🚀 Kullanım

### Ana Sistem Çalıştırma
```bash
python main.py
```

### Web Paneli Çalıştırma
```bash
python app/web.py
```

### Telegram Bot Çalıştırma
```bash
python -c "from modules.telegram_bot import TelegramBot; TelegramBot().run()"
```

## 📁 Proje Yapısı

```
KahinUltima/
├── main.py                 # Ana akış dosyası
├── config.py              # Sistem konfigürasyonu
├── requirements.txt       # Python bağımlılıkları
├── env_example.txt       # Örnek çevre değişkenleri
├── modules/              # Ana modüller
│   ├── __init__.py
│   ├── data_collector.py    # Veri toplama
│   ├── technical_analysis.py # Teknik analiz
│   ├── ai_model.py          # AI modelleri
│   ├── social_media.py      # Sosyal medya analizi
│   ├── news_analysis.py     # Haber analizi
│   ├── whale_tracker.py     # Whale takibi
│   ├── signal_manager.py    # Sinyal yönetimi
│   ├── telegram_bot.py      # Telegram entegrasyonu
│   └── performance.py       # Performans analizi
├── app/                   # Web uygulaması
│   └── web.py            # Flask web paneli
├── data/                 # Ham veriler
├── models/               # Eğitilmiş AI modelleri
├── signals/              # Üretilen sinyaller
└── logs/                 # Sistem logları
```

## ⚙️ Konfigürasyon

### API Anahtarları
- **Binance**: Fiyat verisi ve order book için
- **Telegram**: Bot bildirimleri için
- **Twitter**: Sosyal medya analizi için
- **Reddit**: Sosyal medya analizi için
- **News API**: Haber analizi için

### Sistem Parametreleri
- `MIN_SIGNAL_CONFIDENCE`: Minimum sinyal güven skoru (0.7)
- `MAX_COINS_TO_TRACK`: Takip edilecek maksimum coin sayısı (100)
- `WHALE_THRESHOLD_USDT`: Whale tespiti için eşik değeri (100,000 USDT)

## 📊 Sinyal Üretim Süreci

1. **Veri Toplama**: Binance'den fiyat verileri
2. **Teknik Analiz**: Tüm TA indikatörlerinin hesaplanması
3. **AI Tahminleri**: LSTM ve Random Forest modelleri
4. **Sosyal Analiz**: Reddit/Twitter sentiment
5. **Haber Analizi**: RSS/News API sentiment
6. **Whale Tracking**: Order book analizi
7. **Skor Hesaplama**: Tüm faktörlerin ağırlıklı ortalaması
8. **Sinyal Üretimi**: Yüksek skorlu sinyallerin üretilmesi
9. **Kayıt ve Bildirim**: JSON/CSV/DB kaydı + Telegram

## 🔧 Zamanlanmış Görevler

- **Her 5 dakika**: Sinyal üretimi
- **Her gün 02:00**: AI modellerinin yeniden eğitimi
- **Her 15 dakika**: Sosyal medya güncellemesi
- **Her 30 dakika**: Haber güncellemesi

## 📈 Performans Metrikleri

- **Başarı Oranı**: Sinyallerin doğruluk yüzdesi
- **Ortalama Yükseliş**: Başarılı sinyallerin ortalama kazancı
- **Ortalama Süre**: Sinyal gerçekleşme süresi
- **Coin Bazlı Analiz**: Her coin için ayrı performans

## 🛡️ Güvenlik

- API anahtarları `.env` dosyasında saklanır
- Veritabanı bağlantısı şifrelenir
- Log dosyaları güvenli şekilde tutulur

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 Destek

Sorularınız için issue açabilir veya iletişime geçebilirsiniz.

---

**⚠️ Uyarı**: Bu sistem eğitim amaçlıdır. Gerçek trading kararları için profesyonel danışmanlık alın. 