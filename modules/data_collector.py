import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from config import Config
import requests
import sys
import io
# Konsol çıktısı için UTF-8 encoding ayarı (Windows'ta UnicodeEncodeError önler)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except Exception:
    pass

# Log encoding fix (Windows'ta Unicode hatası önler)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', encoding='utf-8')

class DataCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        api_key = Config.BINANCE_API_KEY.strip() if Config.BINANCE_API_KEY and Config.BINANCE_API_KEY.strip() else None
        secret_key = Config.BINANCE_SECRET_KEY.strip() if Config.BINANCE_SECRET_KEY and Config.BINANCE_SECRET_KEY.strip() else None
        
        # Fiyat verisi için sadece public endpoint kullan - timeout sürelerini artır
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},  # Sadece spot
            'sandbox': False,
            'verbose': False,
            'timeout': 30000,  # 30 saniye timeout
            'rateLimit': 1000,  # Rate limit'i artır
            'session': requests.Session()
        })
        
        # Session timeout ayarları
        self.exchange.session.timeout = 30
        self.exchange.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Ek güvenlik: defaultType'ın yanlışlıkla değişmesini engelle
        if hasattr(self.exchange, 'options'):
            self.exchange.options['defaultType'] = 'spot'
            
        # Eğer API anahtarları varsa, ayrı bir private exchange instance'ı oluştur
        if api_key and secret_key:
            self.private_exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'sandbox': False,
                'verbose': False,
                'timeout': 30000,
                'rateLimit': 1000
            })
            if hasattr(self.private_exchange, 'options'):
                self.private_exchange.options['defaultType'] = 'spot'
            self.logger.info("API anahtarları bulundu, private işlemler için ayrı instance kullanılacak.")
        else:
            self.private_exchange = None
            self.logger.info("API anahtarları yok, sadece public endpoint'ler kullanılacak.")
        
    def get_usdt_pairs(self, max_pairs=None):
        try:
            # Sadece public endpoint kullan
            default_pairs = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
                'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT',
                'XLM/USDT', 'BCH/USDT', 'FIL/USDT', 'TRX/USDT', 'NEAR/USDT',
            ]
            
            try:
                # Markets yükleme işlemini daha güvenli hale getir
                markets = self.exchange.load_markets()
                usdt_pairs = [s for s in markets if s.endswith('/USDT')]
                if usdt_pairs:
                    return usdt_pairs[:max_pairs] if max_pairs else usdt_pairs
            except Exception as e:
                self.logger.warning(f"Markets yüklenirken hata, varsayılan liste kullanılacak: {e}")
            
            return default_pairs[:max_pairs] if max_pairs else default_pairs
            
        except Exception as e:
            self.logger.error(f"USDT çiftleri alınamadı, varsayılan listeye geçiliyor: {e}")
            return default_pairs[:max_pairs] if max_pairs else default_pairs
    
    def get_popular_usdt_pairs(self, max_pairs=400):
        """Sadece valid_coins.txt'deki ilk 400 coin ile API'den dönen aktif USDT çiftlerinin kesişimini döndür"""
        try:
            # valid_coins.txt'den coinleri oku
            with open('valid_coins.txt', 'r', encoding='utf-8') as f:
                valid_coins = [line.strip() for line in f if line.strip()]
            valid_coins = valid_coins[:max_pairs]
            
            # API'den aktif USDT çiftlerini çek - daha güvenli hale getir
            try:
                markets = self.exchange.load_markets()
                # Ek güvenlik: spot olmayan marketleri filtrele
                usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT') and markets[symbol].get('spot', False)]
                active_pairs = [pair for pair in usdt_pairs if markets[pair].get('active', False)]
                # Kesişim: sadece valid_coins.txt'de olanlar
                filtered = [pair for pair in active_pairs if pair in valid_coins]
                if not filtered:
                    self.logger.warning(f"API+valid_coins.txt kesişimi boş! valid_coins.txt fallback kullanılacak.")
                    return valid_coins
                self.logger.info(f"API+valid_coins.txt kesişimi: {len(filtered)} coin")
                return filtered
            except Exception as e:
                self.logger.error(f"API markets yüklenirken hata: {e}. Sadece valid_coins.txt kullanılacak.")
                return valid_coins
                
        except Exception as e:
            self.logger.error(f"API veya valid_coins.txt okunurken hata: {e}. Sadece valid_coins.txt kullanılacak.")
            # Fallback: valid_coins.txt'den ilk 400 coin'i döndür
            try:
                with open('valid_coins.txt', 'r', encoding='utf-8') as f:
                    coins = [line.strip() for line in f if line.strip()]
                if not coins:
                    self.logger.critical("valid_coins.txt boş! Coin listesi döndürülemiyor. Lütfen dosyayı kontrol edin.")
                    # Son çare: en popüler 20 coin'i döndür
                    return [
                        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                        'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
                        'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT',
                        'XLM/USDT', 'BCH/USDT', 'FIL/USDT', 'TRX/USDT', 'NEAR/USDT',
                    ]
                self.logger.info(f"valid_coins.txt fallback: {len(coins[:max_pairs])} coin")
                return coins[:max_pairs]
            except Exception as e2:
                self.logger.critical(f"valid_coins.txt okunamadı: {e2}. Son çare olarak en popüler 20 coin döndürülüyor.")
                return [
                    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                    'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
                    'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT',
                    'XLM/USDT', 'BCH/USDT', 'FIL/USDT', 'TRX/USDT', 'NEAR/USDT',
                ]
    
    def get_historical_data(self, symbol, timeframe, limit=1000):
        """Geçmiş fiyat verilerini al (maksimum veri, temizleme ve outlier filtreleme ile)"""
        try:
            all_ohlcv = []
            since = None
            max_per_call = 1000  # Binance fetch_ohlcv limiti
            total = 0
            
            # Daha uzun geçmiş veri için multiple calls
            while total < limit:
                fetch_limit = min(max_per_call, limit - total)
                try:
                    # Sembolü otomatik olarak düzelt
                    if '/' not in symbol and not symbol.endswith('USDT'):
                        symbol = f"{symbol}/USDT"
                    elif symbol.endswith('USDT') and '/' not in symbol:
                        symbol = symbol[:-4] + '/USDT'
                    # Sadece public exchange kullan - private endpoint'lerden kaçın
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=fetch_limit)
                except Exception as e:
                    self.logger.error(f"{symbol} için fetch_ohlcv çağrısında hata veya timeout: {e}")
                    break
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                total += len(ohlcv)
                if len(ohlcv) < fetch_limit:
                    break
                since = ohlcv[-1][0] + 1  # Son timestamp'ten devam et
                time.sleep(0.1)  # Rate limit - daha hızlı
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Veri temizliği: eksik satırları ve uç değerleri filtrele
            df = df.dropna()
            
            # Outlier detection ve temizleme
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    Q1 = df[col].quantile(0.001)  # Daha hassas outlier detection
                    Q3 = df[col].quantile(0.999)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 2 * IQR  # Daha geniş aralık
                    upper_bound = Q3 + 2 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Duplicate timestamp'leri temizle
            df = df[~df.index.duplicated(keep='first')]
            
            # Sıralama
            df = df.sort_index()
            
            self.logger.info(f"{symbol} {timeframe}: {len(df)} satır veri alındı")
            return df
            
        except Exception as e:
            self.logger.error(f"{symbol} için veri alınırken hata: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol):
        """Mevcut fiyatı al"""
        try:
            # Sadece public exchange kullan
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"{symbol} mevcut fiyat alınırken hata: {e}")
            return None
    
    def get_order_book(self, symbol, limit=100):
        """Order book verilerini al - daha güvenli hale getir"""
        import time
        try:
            symbol_binance = self._to_binance_symbol(symbol)
            retries = 3
            for attempt in range(retries):
                try:
                    order_book = self.exchange.fetch_order_book(symbol_binance, limit)
                    break
                except Exception as e:
                    self.logger.error(f"Order book alınırken hata (ccxt/binance): {symbol_binance} - {type(e).__name__}: {e} (deneme {attempt+1}/{retries})")
                    if attempt < retries - 1:
                        time.sleep(5)  # 5 saniye bekle ve tekrar dene
                    else:
                        return None
            # Bid ve ask'ları ayrı ayrı hesapla
            bids = pd.DataFrame(order_book['bids'], columns=['price', 'amount'])
            asks = pd.DataFrame(order_book['asks'], columns=['price', 'amount'])
            bids['total'] = bids['amount'].cumsum()
            asks['total'] = asks['amount'].cumsum()
            return {
                'bids': bids,
                'asks': asks,
                'bid_volume': bids['amount'].sum(),
                'ask_volume': asks['amount'].sum(),
                'spread': asks['price'].iloc[0] - bids['price'].iloc[0]
            }
        except Exception as e:
            import traceback
            self.logger.error(f"{symbol} order book alınırken genel hata: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return None
    
    def get_24h_stats(self, symbol):
        """24 saatlik istatistikleri al"""
        try:
            # Sadece public exchange kullan
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last_price': ticker['last'],
                'change_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'volume_24h': ticker['baseVolume'],
                'quote_volume_24h': ticker['quoteVolume']
            }
        except Exception as e:
            self.logger.error(f"{symbol} 24h istatistikleri alınırken hata: {e}")
            return None
    
    def get_multiple_prices(self, symbols):
        """Birden fazla sembolün fiyatlarını al"""
        try:
            # Sadece public exchange kullan
            tickers = self.exchange.fetch_tickers(symbols)
            prices = {}
            
            for symbol in symbols:
                if symbol in tickers:
                    prices[symbol] = {
                        'price': tickers[symbol]['last'],
                        'change_24h': tickers[symbol]['percentage'],
                        'volume_24h': tickers[symbol]['baseVolume']
                    }
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Çoklu fiyat alınırken hata: {e}")
            return {}
    
    def save_data_to_csv(self, df, symbol, timeframe):
        """Veriyi CSV dosyasına kaydet"""
        try:
            safe_symbol = str(symbol).replace('/', '_') if symbol else ''
            filename = f"{Config.DATA_DIR}/{safe_symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename)
            self.logger.info(f"Veri kaydedildi: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Veri kaydedilirken hata: {e}")
            return None
    
    def load_data_from_csv(self, symbol, timeframe, date=None):
        """CSV dosyasından veri yükle"""
        try:
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
            safe_symbol = str(symbol).replace('/', '_') if symbol else ''
            filename = f"{Config.DATA_DIR}/{safe_symbol}_{timeframe}_{date}.csv"
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            self.logger.error(f"CSV'den veri yüklenirken hata: {e}")
            return pd.DataFrame()
    
    def get_supported_timeframes(self):
        """Desteklenen tüm timeframeleri döndür"""
        return ['5m', '15m', '1h', '4h', '1d']
    
    def collect_extended_data(self, symbols=None, days=365, timeframes=['1h', '4h', '1d']):
        """Genişletilmiş veri toplama - daha fazla veri ve çoklu timeframe"""
        if symbols is None:
            # Daha fazla popüler coin ekle
            symbols = [
                'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI',
                'ATOM', 'LTC', 'BCH', 'XLM', 'VET', 'TRX', 'FIL', 'THETA', 'XMR', 'NEO',
                'ALGO', 'ICP', 'FTT', 'XTZ', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'SNX', 'YFI',
                'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'JUP', 'PYTH', 'JTO', 'BOME',
                'BOOK', 'GME', 'TRUMP', 'BIDEN', 'PALANTIR', 'NVIDIA', 'TSLA', 'APPLE', 'META'
            ]
        
        all_data = []
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"Veri toplama {i}/{total_symbols}: {symbol}")
                
                for timeframe in timeframes:
                    try:
                        # Daha uzun geçmiş veri al
                        data = self.get_historical_data(symbol, timeframe, days)
                        if data is not None and not data.empty:
                            data['symbol'] = symbol
                            data['timeframe'] = timeframe
                            all_data.append(data)
                            self.logger.info(f"{symbol} {timeframe}: {len(data)} satır")
                        else:
                            self.logger.warning(f"{symbol} {timeframe} için veri alınamadı")
                    except Exception as e:
                        self.logger.error(f"{symbol} {timeframe} veri toplama hatası: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"{symbol} genel veri toplama hatası: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Toplam {len(combined_df)} satır veri toplandı")
            return combined_df
        else:
            self.logger.error("Hiç veri toplanamadı!")
            return None
    
    def get_multi_timeframe_data(self, symbol, timeframes=['5m', '15m', '1h', '4h'], limit=5000):
        """Çoklu timeframe veri toplama"""
        try:
            multi_tf_data = {}
            
            for tf in timeframes:
                try:
                    data = self.get_historical_data(symbol, tf, limit)
                    if not data.empty:
                        multi_tf_data[tf] = data
                        self.logger.info(f"{symbol} {tf}: {len(data)} satır")
                    else:
                        self.logger.warning(f"{symbol} {tf} için veri alınamadı")
                except Exception as e:
                    self.logger.error(f"{symbol} {tf} veri toplama hatası: {e}")
                    continue
                
                time.sleep(0.2)  # Rate limit
            
            return multi_tf_data
            
        except Exception as e:
            self.logger.error(f"Çoklu timeframe veri toplama hatası: {e}")
            return {}
    
    def get_order_book_data(self, symbol, limit=100):
        """Order book verilerini al - daha detaylı"""
        try:
            symbol_binance = self._to_binance_symbol(symbol)
            
            # Markets yükleme işlemini daha güvenli hale getir
            try:
                order_book = self.exchange.fetch_order_book(symbol_binance, limit)
            except Exception as e:
                self.logger.error(f"Order book alınırken hata: {e}")
                return None
            
            # Bid ve ask'ları ayrı ayrı hesapla
            bids = pd.DataFrame(order_book['bids'], columns=['price', 'amount'])
            asks = pd.DataFrame(order_book['asks'], columns=['price', 'amount'])
            
            # Toplam hacimleri hesapla
            bids['total'] = bids['amount'].cumsum()
            asks['total'] = asks['amount'].cumsum()
            
            # Spread analizi
            spread = asks['price'].iloc[0] - bids['price'].iloc[0]
            spread_percentage = (spread / bids['price'].iloc[0]) * 100
            
            # Order book imbalance
            bid_volume = bids['amount'].sum()
            ask_volume = asks['amount'].sum()
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            return {
                'bids': bids,
                'asks': asks,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'imbalance': imbalance,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"{symbol} order book alınırken hata: {e}")
            return None 

    # --- ALTERNATİF BORSALAR ---
    def get_kucoin_order_book(self, symbol, limit=100):
        """KuCoin order book verisi"""
        try:
            kucoin = ccxt.kucoin({'enableRateLimit': True})
            order_book = kucoin.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            self.logger.error(f"KuCoin order book alınırken hata: {e}")
            return None

    def get_okx_order_book(self, symbol, limit=100):
        """OKX order book verisi"""
        try:
            okx = ccxt.okx({'enableRateLimit': True})
            order_book = okx.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            self.logger.error(f"OKX order book alınırken hata: {e}")
            return None

    # --- ZİNCİR ÜSTÜ (ON-CHAIN) VERİ ---
    def get_etherscan_tx_count(self, address, api_key='YOUR_ETHERSCAN_API_KEY'):
        """Etherscan'den bir adresin toplam işlem sayısını al (örnek)"""
        try:
            url = f'https://api.etherscan.io/api?module=proxy&action=eth_getTransactionCount&address={address}&tag=latest&apikey={api_key}'
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return data.get('result', None)
        except Exception as e:
            self.logger.error(f"Etherscan tx count alınırken hata: {e}")
            return None

    def get_bscscan_token_balance(self, address, contract, api_key='YOUR_BSCSCAN_API_KEY'):
        """BscScan'den bir adresin token bakiyesini al (örnek)"""
        try:
            url = f'https://api.bscscan.com/api?module=account&action=tokenbalance&contractaddress={contract}&address={address}&apikey={api_key}'
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return data.get('result', None)
        except Exception as e:
            self.logger.error(f"BscScan token balance alınırken hata: {e}")
            return None

    def get_solana_account_info(self, address):
        """Solana explorer'dan bir adresin temel bilgisini al (örnek)"""
        try:
            url = f'https://public-api.solscan.io/account/{address}'
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return data
        except Exception as e:
            self.logger.error(f"Solana account info alınırken hata: {e}")
            return None 

    def update_valid_coins_with_new_listings(self, max_pairs=400):
        """Yeni çıkan coinleri tespit edip valid_coins.txt'ye otomatik ekle"""
        try:
            markets = self.exchange.load_markets()
            usdt_pairs = [s for s in markets if s.endswith('/USDT')]
            # Mevcut valid_coins.txt'yi oku
            with open('valid_coins.txt', 'r', encoding='utf-8') as f:
                valid_coins = set(line.strip() for line in f if line.strip())
            new_coins = [pair for pair in usdt_pairs if pair not in valid_coins]
            if new_coins:
                with open('valid_coins.txt', 'a', encoding='utf-8') as f:
                    for coin in new_coins:
                        f.write(coin + '\n')
                self.logger.info(f"Yeni coinler valid_coins.txt'ye eklendi: {new_coins}")
            else:
                self.logger.info("Yeni coin bulunamadı.")
            return new_coins
        except Exception as e:
            self.logger.error(f"Yeni coin güncelleme hatası: {e}")
            return []

    def adapt_to_market_conditions(self):
        """Piyasa koşullarına göre otomatik adaptasyon (ör. volatilite, yeni coin)"""
        try:
            # Örnek: BTC/USDT volatilitesine göre sistem parametrelerini ayarla
            df = self.get_historical_data('BTC/USDT', '1h', limit=500)
            if not df.empty:
                volatility = df['close'].pct_change().rolling(24).std().iloc[-1]
                if volatility > 0.05:
                    self.logger.info(f"Yüksek volatilite tespit edildi ({volatility:.2%}), eşikler gevşetilebilir.")
                else:
                    self.logger.info(f"Volatilite normal ({volatility:.2%}), standart eşikler kullanılabilir.")
            # Yeni coinleri güncelle
            self.update_valid_coins_with_new_listings()
        except Exception as e:
            self.logger.error(f"Piyasa adaptasyon hatası: {e}") 

    def get_recent_trades(self, symbol, minutes=2):
        """Son X dakikadaki işlemleri (trades) al"""
        try:
            now = int(pd.Timestamp.utcnow().timestamp() * 1000)
            since = now - minutes * 60 * 1000
            trades = self.exchange.fetch_trades(symbol, since=since)
            filtered = [
                {'price': t['price'], 'amount': t['amount'], 'timestamp': t['timestamp']}
                for t in trades if t['timestamp'] >= since
            ]
            return filtered
        except Exception as e:
            self.logger.error(f"{symbol} trade verisi alınırken hata: {e}")
            return []

    def update_historical_data(self, symbol, timeframe, limit=1000):
        """Mevcut CSV'yi günceller, eksik barları ekler"""
        import os
        import pandas as pd
        from datetime import datetime
        safe_symbol = str(symbol).replace('/', '_') if symbol else ''
        filename = f"{Config.DATA_DIR}/{safe_symbol}_{timeframe}_6months.csv"
        # 1. Mevcut dosyayı oku
        if os.path.exists(filename):
            df_old = pd.read_csv(filename, index_col=0, parse_dates=True)
            if not df_old.empty:
                last_timestamp = df_old.index[-1]
                # 2. API'den yeni barları çek (son barın timestamp'inden itibaren)
                now = pd.Timestamp.utcnow()
                delta = now - last_timestamp
                bars_needed = int(delta.total_seconds() // self._timeframe_to_seconds(timeframe))
                if bars_needed <= 0:
                    self.logger.info(f"{symbol} {timeframe}: Güncel, yeni bar yok.")
                    return filename
                df_new = self.get_historical_data(symbol, timeframe, limit=bars_needed+2)
                df_new = df_new[df_new.index > last_timestamp]
                if not df_new.empty:
                    df_updated = pd.concat([df_old, df_new])
                    df_updated = df_updated[~df_updated.index.duplicated(keep='first')]
                    df_updated = df_updated.sort_index()
                    df_updated.to_csv(filename)
                    self.logger.info(f"{symbol} {timeframe}: {len(df_new)} yeni bar eklendi.")
                else:
                    self.logger.info(f"{symbol} {timeframe}: Yeni bar bulunamadı.")
            else:
                # Dosya boşsa baştan çek
                df_new = self.get_historical_data(symbol, timeframe, limit=limit)
                df_new.to_csv(filename)
                self.logger.info(f"{symbol} {timeframe}: Dosya boştu, baştan çekildi.")
        else:
            # Dosya yoksa baştan çek
            df_new = self.get_historical_data(symbol, timeframe, limit=limit)
            df_new.to_csv(filename)
            self.logger.info(f"{symbol} {timeframe}: Dosya yoktu, baştan çekildi.")
        return filename

    def _timeframe_to_seconds(self, timeframe):
        """Timeframe stringini saniyeye çevirir"""
        mapping = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400, '1d': 86400}
        return mapping.get(timeframe, 3600)

    def _to_binance_symbol(self, symbol):
        """Sembolü Binance API'nin beklediği formata çevir (ör. ADA/USDT -> ADAUSDT)"""
        if '/' in symbol:
            return symbol.replace('/', '')
        return symbol 