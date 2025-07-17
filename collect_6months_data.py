#!/usr/bin/env python3
"""
Kahin Ultima - 6 Aylık Veri Toplama Scripti
Binance API rate limitlerini göz önünde bulundurarak güvenli veri çekme
"""

import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from modules.data_collector import DataCollector
from config import Config

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection_6months.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExtendedDataCollector:
    def __init__(self):
        self.data_collector = DataCollector()
        self.data_dir = 'data'
        self.timeframes = ['1h', '4h', '1d']  # 6 aylık analiz için optimal timeframes
        self.days_back = 180  # 6 ay
        
        # Rate limiting ayarları - Binance API limitlerini aşmamak için
        self.request_delay = 0.2  # 200ms between requests (daha güvenli)
        self.batch_size = 5  # Her batch'te 5 coin (daha küçük batch'ler)
        self.batch_delay = 2  # Batch'ler arası 2 saniye (daha uzun bekleme)
        
        # Veri kalitesi ayarları
        self.min_data_points = 100  # Minimum veri noktası
        self.max_retries = 3
        
        # Data directory oluştur
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_all_coins(self):
        """Tüm geçerli coinleri al"""
        try:
            # Önce API'den al
            coins = self.data_collector.get_popular_usdt_pairs(max_pairs=400)
            logger.info(f"API'den {len(coins)} coin alındı")
            return coins
        except Exception as e:
            logger.error(f"API'den coin listesi alınamadı: {e}")
            # Fallback: valid_coins.txt'den oku
            try:
                with open('valid_coins.txt', 'r') as f:
                    coins = [line.strip() for line in f if line.strip()]
                logger.info(f"valid_coins.txt'den {len(coins)} coin okundu")
                return coins
            except Exception as e2:
                logger.error(f"valid_coins.txt okunamadı: {e2}")
                return []
    
    def collect_coin_data(self, symbol, timeframe):
        """Tek bir coin için veri topla"""
        try:
            logger.info(f"{symbol} - {timeframe}: Veri toplama başladı")
            
            # 6 aylık veri için limit hesapla
            if timeframe == '1h':
                limit = self.days_back * 24  # 6 ay * 24 saat
            elif timeframe == '4h':
                limit = self.days_back * 6   # 6 ay * 6 (4 saatlik)
            elif timeframe == '1d':
                limit = self.days_back        # 6 ay
            else:
                limit = 5000
            
            # Veri çek
            df = self.data_collector.get_historical_data(symbol, timeframe, limit)
            
            if df is None or df.empty:
                logger.warning(f"{symbol} - {timeframe}: Veri alınamadı")
                return None
            
            # Veri kalitesi kontrolü
            if len(df) < self.min_data_points:
                logger.warning(f"{symbol} - {timeframe}: Yetersiz veri ({len(df)} nokta)")
                return None
            
            # NaN değerleri temizle
            df = df.dropna()
            
            if len(df) < self.min_data_points:
                logger.warning(f"{symbol} - {timeframe}: NaN temizleme sonrası yetersiz veri")
                return None
            
            # Dosya adı oluştur
            filename = f"{symbol.replace('/', '_')}_{timeframe}_6months.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # CSV'ye kaydet
            df.to_csv(filepath)
            logger.info(f"{symbol} - {timeframe}: {len(df)} satır veri kaydedildi -> {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"{symbol} - {timeframe}: Hata - {e}")
            return None
    
    def collect_batch_data(self, coins_batch):
        """Bir batch coin için veri topla"""
        results = {}
        
        for symbol in coins_batch:
            symbol_results = {}
            
            for timeframe in self.timeframes:
                # Rate limiting
                time.sleep(self.request_delay)
                
                # Retry logic
                for attempt in range(self.max_retries):
                    try:
                        df = self.collect_coin_data(symbol, timeframe)
                        if df is not None:
                            symbol_results[timeframe] = df
                            break
                        else:
                            logger.warning(f"{symbol} - {timeframe}: Deneme {attempt + 1} başarısız")
                            if attempt < self.max_retries - 1:
                                time.sleep(2)  # Retry öncesi bekle
                    except Exception as e:
                        logger.error(f"{symbol} - {timeframe}: Deneme {attempt + 1} hatası - {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(5)  # Hata sonrası daha uzun bekle
            
            if symbol_results:
                results[symbol] = symbol_results
            
            # Coin arası kısa bekleme
            time.sleep(0.2)
        
        return results
    
    def collect_all_data(self):
        """Tüm coinler için 6 aylık veri topla"""
        logger.info("=== 6 AYLIK VERİ TOPLAMA BAŞLADI ===")
        start_time = datetime.now()
        
        # Tüm coinleri al
        all_coins = self.get_all_coins()
        if not all_coins:
            logger.error("Hiç coin bulunamadı!")
            return
        
        logger.info(f"Toplam {len(all_coins)} coin için veri toplanacak")
        
        # Batch'lere böl
        batches = [all_coins[i:i + self.batch_size] for i in range(0, len(all_coins), self.batch_size)]
        logger.info(f"Toplam {len(batches)} batch oluşturuldu")
        
        total_successful = 0
        total_failed = 0
        
        for i, batch in enumerate(batches):
            logger.info(f"=== BATCH {i+1}/{len(batches)} BAŞLADI ===")
            logger.info(f"Batch coinleri: {batch}")
            
            batch_start = datetime.now()
            batch_results = self.collect_batch_data(batch)
            
            # Batch sonuçlarını değerlendir
            batch_successful = len(batch_results)
            batch_failed = len(batch) - batch_successful
            
            total_successful += batch_successful
            total_failed += batch_failed
            
            batch_duration = datetime.now() - batch_start
            logger.info(f"Batch {i+1} tamamlandı: {batch_successful} başarılı, {batch_failed} başarısız")
            logger.info(f"Batch süresi: {batch_duration}")
            
            # Batch'ler arası bekleme
            if i < len(batches) - 1:  # Son batch değilse
                logger.info(f"Sonraki batch için {self.batch_delay} saniye bekleniyor...")
                time.sleep(self.batch_delay)
        
        # Final rapor
        total_duration = datetime.now() - start_time
        logger.info("=== VERİ TOPLAMA TAMAMLANDI ===")
        logger.info(f"Toplam süre: {total_duration}")
        logger.info(f"Başarılı: {total_successful}/{len(all_coins)} coin")
        logger.info(f"Başarısız: {total_failed}/{len(all_coins)} coin")
        logger.info(f"Başarı oranı: {(total_successful/len(all_coins)*100):.1f}%")
        
        # Özet dosyası oluştur
        self.create_summary_report(all_coins, total_successful, total_failed, total_duration)
    
    def create_summary_report(self, all_coins, successful, failed, duration):
        """Özet rapor oluştur"""
        report_file = os.path.join(self.data_dir, 'data_collection_summary.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== KAHIN ULTIMA 6 AYLIK VERİ TOPLAMA RAPORU ===\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Toplam süre: {duration}\n")
            f.write(f"Toplam coin: {len(all_coins)}\n")
            f.write(f"Başarılı: {successful}\n")
            f.write(f"Başarısız: {failed}\n")
            f.write(f"Başarı oranı: {(successful/len(all_coins)*100):.1f}%\n")
            f.write(f"Timeframes: {', '.join(self.timeframes)}\n")
            f.write(f"Veri süresi: {self.days_back} gün\n\n")
            
            f.write("=== BAŞARILI COINLER ===\n")
            for coin in all_coins:
                coin_files = []
                for tf in self.timeframes:
                    filename = f"{coin.replace('/', '_')}_{tf}_6months.csv"
                    filepath = os.path.join(self.data_dir, filename)
                    if os.path.exists(filepath):
                        coin_files.append(tf)
                
                if coin_files:
                    f.write(f"{coin}: {', '.join(coin_files)}\n")
            
            f.write("\n=== BAŞARISIZ COINLER ===\n")
            for coin in all_coins:
                coin_files = []
                for tf in self.timeframes:
                    filename = f"{coin.replace('/', '_')}_{tf}_6months.csv"
                    filepath = os.path.join(self.data_dir, filename)
                    if os.path.exists(filepath):
                        coin_files.append(tf)
                
                if not coin_files:
                    f.write(f"{coin}: Hiç veri yok\n")
        
        logger.info(f"Özet rapor oluşturuldu: {report_file}")

def main():
    """Ana fonksiyon"""
    try:
        collector = ExtendedDataCollector()
        collector.collect_all_data()
        
        logger.info("Veri toplama işlemi tamamlandı!")
        
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}")
        raise

if __name__ == "__main__":
    main() 