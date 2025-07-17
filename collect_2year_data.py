import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_collector import DataCollector
from config import Config

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection_2year.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollectionManager:
    def __init__(self):
        self.data_collector = DataCollector()
        self.lock = threading.Lock()
        self.collected_data = {}
        self.errors = []
        self.stats = {
            'total_coins': 0,
            'successful_coins': 0,
            'failed_coins': 0,
            'total_data_points': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Tüm zaman dilimleri
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        
        # 2 yıl için veri hesaplama
        self.years = 2
        self.days_per_year = 365
        self.total_days = self.years * self.days_per_year
        
        # Her timeframe için günlük veri sayısı
        self.daily_counts = {
            '5m': 288,    # 24 * 60 / 5
            '15m': 96,    # 24 * 60 / 15
            '1h': 24,     # 24 saat
            '4h': 6,      # 24 / 4
            '1d': 1       # 1 gün
        }
        
    def get_all_coins(self):
        """Tüm aktif USDT çiftlerini al"""
        try:
            logger.info("Aktif USDT çiftleri alınıyor...")
            coins = self.data_collector.get_popular_usdt_pairs(max_pairs=400)
            logger.info(f"Toplam {len(coins)} coin bulundu")
            return coins
        except Exception as e:
            logger.error(f"Coin listesi alınırken hata: {e}")
            # Fallback liste
            fallback_coins = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
                'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'BCH/USDT', 'XLM/USDT', 'VET/USDT', 'TRX/USDT', 'FIL/USDT', 'THETA/USDT',
                'XMR/USDT', 'NEO/USDT', 'ALGO/USDT', 'ICP/USDT', 'FTT/USDT', 'XTZ/USDT', 'AAVE/USDT', 'SUSHI/USDT', 'COMP/USDT', 'MKR/USDT',
                'SNX/USDT', 'YFI/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT', 'JUP/USDT', 'PYTH/USDT', 'JTO/USDT',
                'BOME/USDT', 'ETC/USDT', 'ICX/USDT', 'IOTA/USDT', 'ONG/USDT', 'ONT/USDT', 'QTUM/USDT', 'TUSD/USDT', 'USDC/USDT', 'ZIL/USDT',
                'ZRX/USDT', '1INCH/USDT', 'ACH/USDT', 'ALPHA/USDT', 'ANKR/USDT', 'ANT/USDT', 'APE/USDT', 'API3/USDT', 'APT/USDT', 'AR/USDT',
                'ARB/USDT', 'ARK/USDT', 'ASTR/USDT', 'ATA/USDT', 'AUDIO/USDT', 'AXS/USDT', 'BADGER/USDT', 'BAKE/USDT', 'BAL/USDT', 'BAND/USDT',
                'BAT/USDT', 'BICO/USDT', 'BLZ/USDT', 'BNT/USDT', 'BOND/USDT', 'BOSON/USDT', 'BTTC/USDT', 'BTT/USDT', 'BUSD/USDT', 'C98/USDT',
                'CAKE/USDT', 'CELO/USDT', 'CELR/USDT', 'CFX/USDT', 'CHR/USDT', 'CHZ/USDT', 'CKB/USDT', 'CLV/USDT', 'COS/USDT', 'COTI/USDT',
                'CRV/USDT', 'CTSI/USDT', 'CTXC/USDT', 'CVP/USDT', 'CVX/USDT', 'DASH/USDT', 'DATA/USDT', 'DCR/USDT', 'DEGO/USDT', 'DENT/USDT',
                'DGB/USDT', 'DIA/USDT', 'DOCK/USDT', 'DODO/USDT', 'DUSK/USDT', 'DYDX/USDT', 'EGLD/USDT', 'ENJ/USDT', 'ENS/USDT', 'EOS/USDT',
                'FET/USDT', 'FLM/USDT', 'FLOW/USDT', 'FLR/USDT', 'FORTH/USDT', 'FTM/USDT', 'FXS/USDT', 'GALA/USDT', 'GAL/USDT', 'GHST/USDT',
                'GLM/USDT', 'GLMR/USDT', 'GMT/USDT', 'GMX/USDT', 'GODS/USDT', 'GRT/USDT', 'HBAR/USDT', 'HFT/USDT', 'HIVE/USDT', 'HOT/USDT',
                'ID/USDT', 'IDEX/USDT', 'ILV/USDT', 'IMX/USDT', 'INJ/USDT', 'IOTX/USDT', 'IRIS/USDT', 'JASMY/USDT', 'JOE/USDT', 'JST/USDT',
                'KAVA/USDT', 'KDA/USDT', 'KEEP/USDT', 'KEY/USDT', 'KLAY/USDT', 'KNC/USDT', 'KSM/USDT', 'LAZIO/USDT', 'LDO/USDT', 'LINA/USDT',
                'LIT/USDT', 'LOKA/USDT', 'LPT/USDT', 'LQTY/USDT', 'LRC/USDT', 'LSK/USDT', 'LTO/USDT', 'LUNA/USDT', 'MAGIC/USDT', 'MANA/USDT',
                'MASK/USDT', 'MBL/USDT', 'MBOX/USDT', 'MC/USDT', 'MDT/USDT', 'MINA/USDT', 'MLN/USDT', 'MOB/USDT', 'MULTI/USDT', 'NEAR/USDT',
                'NKN/USDT', 'NMR/USDT', 'OCEAN/USDT', 'OGN/USDT', 'OM/USDT', 'OMG/USDT', 'ONE/USDT', 'OP/USDT', 'ORN/USDT', 'OXT/USDT',
                'PAXG/USDT', 'PEOPLE/USDT', 'PERP/USDT', 'PHA/USDT', 'POLS/USDT', 'POLYGON/USDT', 'POND/USDT', 'POWR/USDT', 'PROM/USDT',
                'QNT/USDT', 'QUICK/USDT', 'RAD/USDT', 'RARE/USDT', 'RAY/USDT', 'REEF/USDT', 'REN/USDT', 'REQ/USDT', 'RLC/USDT', 'RNDR/USDT',
                'ROSE/USDT', 'RSR/USDT', 'RUNE/USDT', 'RVN/USDT', 'SAND/USDT', 'SCRT/USDT', 'SFP/USDT', 'SKL/USDT', 'SLP/USDT', 'SPELL/USDT',
                'SRM/USDT', 'STARL/USDT', 'STG/USDT', 'STMX/USDT', 'STORJ/USDT', 'STPT/USDT', 'STRAX/USDT', 'STX/USDT', 'SUPER/USDT', 'SXP/USDT',
                'SYN/USDT', 'SYS/USDT', 'T/USDT', 'TFUEL/USDT', 'TLM/USDT', 'TOKE/USDT', 'TOMO/USDT', 'TRB/USDT', 'TRIBE/USDT', 'TRU/USDT',
                'TVK/USDT', 'TWT/USDT', 'UMA/USDT', 'UNFI/USDT', 'USDP/USDT', 'UTK/USDT', 'VGX/USDT', 'VTHO/USDT', 'WAVES/USDT', 'WAXP/USDT',
                'WBTC/USDT', 'WOO/USDT', 'XEC/USDT', 'XEM/USDT', 'XVG/USDT', 'XVS/USDT', 'YGG/USDT', 'ZEC/USDT', 'ZEN/USDT'
            ]
            return fallback_coins[:400]
    
    def calculate_required_data_points(self, timeframe):
        """Belirli timeframe için gerekli veri noktası sayısını hesapla"""
        daily_count = self.daily_counts.get(timeframe, 1)
        return self.total_days * daily_count
    
    def collect_coin_data(self, coin, timeframe):
        """Tek coin için belirli timeframe'de veri topla"""
        try:
            required_points = self.calculate_required_data_points(timeframe)
            
            # Maksimum veri al (Binance limiti 1000)
            max_limit = min(required_points, 5000)  # Güvenli limit
            
            logger.info(f"{coin} {timeframe}: {max_limit} veri noktası alınıyor...")
            
            data = self.data_collector.get_historical_data(coin, timeframe, max_limit)
            
            if data is not None and not data.empty:
                # Veriyi kaydet
                with self.lock:
                    if coin not in self.collected_data:
                        self.collected_data[coin] = {}
                    self.collected_data[coin][timeframe] = data
                    self.stats['total_data_points'] += len(data)
                
                logger.info(f"✓ {coin} {timeframe}: {len(data)} veri alındı")
                return True, len(data)
            else:
                logger.warning(f"✗ {coin} {timeframe}: Veri alınamadı")
                return False, 0
                
        except Exception as e:
            error_msg = f"{coin} {timeframe}: {str(e)}"
            logger.error(error_msg)
            with self.lock:
                self.errors.append(error_msg)
            return False, 0
    
    def collect_all_data_for_coin(self, coin):
        """Tek coin için tüm timeframe'lerde veri topla"""
        coin_success = True
        total_points = 0
        
        for timeframe in self.timeframes:
            try:
                success, points = self.collect_coin_data(coin, timeframe)
                if not success:
                    coin_success = False
                total_points += points
                
                # Rate limit - her timeframe arası bekle
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"{coin} {timeframe} genel hata: {e}")
                coin_success = False
        
        with self.lock:
            if coin_success:
                self.stats['successful_coins'] += 1
            else:
                self.stats['failed_coins'] += 1
        
        return coin_success, total_points
    
    def save_data_to_files(self):
        """Toplanan verileri dosyalara kaydet"""
        logger.info("Veriler dosyalara kaydediliyor...")
        
        # Data klasörünü oluştur
        data_dir = "data/2year_collection"
        os.makedirs(data_dir, exist_ok=True)
        
        saved_files = []
        
        for coin, timeframes_data in self.collected_data.items():
            coin_name = coin.replace('/', '_')
            
            for timeframe, data in timeframes_data.items():
                if not data.empty:
                    filename = f"{data_dir}/{coin_name}_{timeframe}_2year.csv"
                    try:
                        data.to_csv(filename)
                        saved_files.append(filename)
                        logger.info(f"✓ {filename} kaydedildi")
                    except Exception as e:
                        logger.error(f"✗ {filename} kaydedilemedi: {e}")
        
        # İstatistikleri kaydet
        stats_file = f"{data_dir}/collection_stats.json"
        stats_data = {
            'collection_date': datetime.now().isoformat(),
            'stats': self.stats,
            'errors': self.errors,
            'saved_files': saved_files
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
            logger.info(f"✓ İstatistikler kaydedildi: {stats_file}")
        except Exception as e:
            logger.error(f"✗ İstatistikler kaydedilemedi: {e}")
        
        return saved_files
    
    def run_collection(self, max_workers=5):
        """Ana veri toplama işlemini çalıştır"""
        logger.info("=== 2 YILLIK VERİ TOPLAMA BAŞLATILIYOR ===")
        
        self.stats['start_time'] = datetime.now()
        
        # Tüm coinleri al
        coins = self.get_all_coins()
        self.stats['total_coins'] = len(coins)
        
        logger.info(f"Toplam {len(coins)} coin için veri toplanacak")
        logger.info(f"Timeframe'ler: {self.timeframes}")
        logger.info(f"Paralel işlem sayısı: {max_workers}")
        
        # Progress bar
        total_tasks = len(coins) * len(self.timeframes)
        
        with tqdm(total=total_tasks, desc="Veri toplama") as pbar:
            
            # ThreadPoolExecutor ile paralel işleme
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                
                # Her coin için future oluştur
                future_to_coin = {
                    executor.submit(self.collect_all_data_for_coin, coin): coin 
                    for coin in coins
                }
                
                # Sonuçları topla
                for future in as_completed(future_to_coin):
                    coin = future_to_coin[future]
                    
                    try:
                        success, points = future.result()
                        if success:
                            logger.info(f"✓ {coin} tamamlandı ({points} veri)")
                        else:
                            logger.warning(f"✗ {coin} hatalı")
                        
                        # Progress bar güncelle
                        pbar.update(len(self.timeframes))
                        
                    except Exception as e:
                        logger.error(f"✗ {coin} işleme hatası: {e}")
                        pbar.update(len(self.timeframes))
        
        self.stats['end_time'] = datetime.now()
        
        # Sonuçları göster
        self.print_summary()
        
        # Verileri kaydet
        saved_files = self.save_data_to_files()
        
        return saved_files
    
    def print_summary(self):
        """Toplama özetini yazdır"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "="*60)
        print("📊 VERİ TOPLAMA ÖZETİ")
        print("="*60)
        print(f"🕐 Başlangıç: {self.stats['start_time']}")
        print(f"🕐 Bitiş: {self.stats['end_time']}")
        print(f"⏱️ Toplam süre: {duration}")
        print(f"📈 Toplam coin: {self.stats['total_coins']}")
        print(f"✅ Başarılı: {self.stats['successful_coins']}")
        print(f"❌ Başarısız: {self.stats['failed_coins']}")
        print(f"📊 Toplam veri noktası: {self.stats['total_data_points']:,}")
        print(f"📁 Toplanan coin sayısı: {len(self.collected_data)}")
        
        if self.errors:
            print(f"\n⚠️ Hatalar ({len(self.errors)}):")
            for error in self.errors[:10]:  # İlk 10 hatayı göster
                print(f"   • {error}")
            if len(self.errors) > 10:
                print(f"   ... ve {len(self.errors) - 10} hata daha")
        
        print("="*60)

def main():
    """Ana fonksiyon"""
    try:
        # Data collection manager oluştur
        manager = DataCollectionManager()
        
        # Veri toplama işlemini başlat
        saved_files = manager.run_collection(max_workers=5)
        
        print(f"\n🎯 İşlem tamamlandı!")
        print(f"📁 {len(saved_files)} dosya kaydedildi")
        print(f"💾 Veriler 'data/2year_collection/' klasöründe")
        
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"Ana işlem hatası: {e}")
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    main() 