#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.signal_manager import SignalManager
from modules.telegram_bot import TelegramBot
from modules.whale_tracker import get_whale_score, detect_whale_trades
# from modules.news_analyzer import NewsAnalyzer  # Devre dışı
from modules.market_analysis import MarketAnalysis
from modules.performance import PerformanceAnalyzer
from modules.dynamic_strictness import DynamicStrictness
from modules.signal_tracker import SignalTracker

# Logging ayarla
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kahin_ultima.log', encoding='utf-8')
    ]
)

import os
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.info('Log testi: Sistem başlatıldı.')

logger = logging.getLogger(__name__)

# Hatalı coin ve hata mesajlarını merkezi olarak tutan global bir liste
FAILED_COINS = []

class KahinUltima:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self.data_collector = DataCollector()
        self.ta = TechnicalAnalysis()
        self.ai_model = AIModel()
        # self.social_media = SocialMediaSentiment()  # Devre dışı
        self.news_analysis = NewsAnalysis()
        self.whale_tracker = WhaleTracker()
        self.signal_manager = SignalManager()
        self.telegram_bot = TelegramBot()
        Config.create_directories()
        self.performance_tracker = PerformanceTracker()
        
        # System status
        self.is_running = False
        self.last_signal_generation = None
        self.last_performance_update = None
        self.system_health = 'HEALTHY'

    def generate_signals(self):
        """Ana sinyal üretim fonksiyonu - TÜM valid_coins.txt taranacak şekilde güncellendi"""
        try:
            self.logger.info("Sinyal uretimi baslatiliyor...")
            # Coin listesini her döngüde valid_coins.txt'den oku
            with open('valid_coins.txt') as f:
                pairs = [line.strip() for line in f if line.strip()]
            if not pairs:
                self.logger.error("valid_coins.txt dosyasından coin alınamadı, sinyal üretimi iptal edildi.")
                self.system_health = 'ERROR'
                return
            signals_generated = 0
            self.logger.info(f"Toplam {len(pairs)} coin islenecek (valid_coins.txt)")
            
            # Konfigurasyon degerlerini logla
            self.logger.info(f"Konfigurasyon degerleri:")
            self.logger.info(f"  MIN_AI_SCORE: {Config.MIN_AI_SCORE}")
            self.logger.info(f"  MIN_TA_STRENGTH: {Config.MIN_TA_STRENGTH}")
            self.logger.info(f"  MIN_WHALE_SCORE: {Config.MIN_WHALE_SCORE}")
            self.logger.info(f"  SIGNAL_QUALITY_THRESHOLD: {Config.SIGNAL_QUALITY_THRESHOLD}")
            self.logger.info(f"  MIN_SIGNAL_CONFIDENCE: {Config.MIN_SIGNAL_CONFIDENCE}")
            
            # Sadece ilk 400 coin'i işle
            test_pairs = pairs[:400]
            self.logger.info(f"Islenecek coin sayisi: {len(test_pairs)}")
            
            # Her coin icin sinyal uret
            for pair in test_pairs:
                try:
                    self.logger.info(f"Isleniyor: {pair}")
                    
                    # Her coin için market_regime ve regime_note default başlat
                    market_regime = 'unknown'
                    regime_note = ''

                    # Son 24 saatte sinyal kontrolü
                    recent_signals = self.signal_manager.get_recent_signals_by_symbol(pair, hours=8)
                    acil_firsat = False
                    if recent_signals and len(recent_signals) > 0:
                        # Acil fırsat kontrolü: Son 2 bar içinde %2 veya daha fazla fiyat değişimi var mı?
                        df = self.data_collector.get_historical_data(pair, '1h', 100) # 100 bar alındığı için
                        if not df.empty and len(df) >= 2:
                            price_now = df['close'].iloc[-1]
                            price_prev = df['close'].iloc[-2]
                            price_change = abs(price_now - price_prev) / price_prev
                            if price_change >= 0.02:
                                acil_firsat = True
                                self.logger.info(f"    ⚡ {pair}: Acil fırsat: Ani hareket tespit edildi (%{price_change*100:.2f}), zaman limiti esnetildi!")
                    if recent_signals and len(recent_signals) > 0 and not acil_firsat:
                        self.logger.info(f"    {pair}: Son 8 saatte zaten sinyal üretilmiş, atlanıyor.")
                        continue
                    
                    # 1. Veri topla
                    df = self.data_collector.get_historical_data(pair, '1h', limit=100)
                    if df.empty:
                        self.logger.warning(f"    ❌ {pair}: Veri alinamadi, atlaniyor")
                        continue
                    
                    self.logger.info(f"    ✅ {pair}: {len(df)} satir veri alindı")
                    
                    # 2. Teknik analiz
                    df = self.ta.calculate_all_indicators(df)
                    if df.empty:
                        self.logger.warning(f"    ❌ {pair}: Teknik analiz basarisiz, atlaniyor")
                        continue
                    
                    self.logger.info(f"    ✅ {pair}: Teknik analiz tamamlandı")
                    
                    # Market regime tespiti
                    market_regime = self.ta.get_market_regime(df)
                    self.logger.info(f"    {pair}: Market regime tespit edildi: {market_regime}")
                    
                    # Market regime'e göre dinamik eşik ve risk ayarı
                    if market_regime == 'trending':
                        min_conf = Config.MIN_SIGNAL_CONFIDENCE * 0.9  # Daha agresif
                        min_ai = Config.MIN_AI_SCORE * 0.9
                        min_ta = Config.MIN_TA_STRENGTH * 0.9
                        regime_note = 'Trend piyasası: Eşikler gevşetildi, daha fazla sinyal.'
                    elif market_regime == 'volatile':
                        min_conf = Config.MIN_SIGNAL_CONFIDENCE * 1.1  # Daha sıkı
                        min_ai = Config.MIN_AI_SCORE * 1.1
                        min_ta = Config.MIN_TA_STRENGTH * 1.1
                        regime_note = 'Volatil piyasa: Eşikler sıkılaştırıldı, risk azaltıldı.'
                    elif market_regime == 'ranging' or market_regime == 'sideways':
                        min_conf = Config.MIN_SIGNAL_CONFIDENCE * 1.2  # Çok daha sıkı
                        min_ai = Config.MIN_AI_SCORE * 1.2
                        min_ta = Config.MIN_TA_STRENGTH * 1.2
                        regime_note = 'Yatay piyasa: Eşikler çok sıkı, az sinyal.'
                    else:
                        min_conf = Config.MIN_SIGNAL_CONFIDENCE
                        min_ai = Config.MIN_AI_SCORE
                        min_ta = Config.MIN_TA_STRENGTH
                        regime_note = 'Normal/karma piyasa: Varsayılan eşikler.'
                    self.logger.info(f"    {pair}: {regime_note}")
                    
                    # 3. AI analizi
                    try:
                        self.logger.info(f"    🤖 {pair}: AI analizi basliyor...")
                        
                        # AI modelini baslat
                        ai_model = AIModel()
                        
                        # AI tahmini yap
                        self.logger.info(f"    🔮 {pair}: AI tahmini yapiliyor...")
                        ai_result = ai_model.predict(df)
                        
                        if ai_result is None:
                            self.logger.warning(f"    ❌ {pair}: AI sonucu None, atlanıyor")
                            continue
                        
                        self.logger.info(f"    ✅ {pair}: AI tahmini tamamlandi: {ai_result}")
                        
                        # AI skorunu al
                        ai_score = ai_result.get('prediction', 0.5)
                        confidence = ai_result.get('confidence', 0.5)
                        feature_dummy_ratio = ai_result.get('feature_dummy_ratio', 1.0)
                        self.logger.info(f"    🧩 {pair}: Feature eksik/dummy oranı: {feature_dummy_ratio:.2%}")
                        if feature_dummy_ratio > 0.10:
                            self.logger.warning(f"    ⚠️ {pair}: Sinyalde feature'ların %{feature_dummy_ratio*100:.1f}'i dummy/eksik! Bu sinyalde veri kalitesi düşük.")
                        
                    except Exception as e:
                        self.logger.error(f"    ❌ {pair}: AI analizi hatasi: {e}")
                        # Dummy skorla devam etmek yerine, hatalı coin ve hata mesajını kaydet
                        FAILED_COINS.append({'coin': pair, 'error': str(e), 'timestamp': str(datetime.now())})
                        continue  # Sadece bu coin atlanır, sistemin kalan kısmı devam eder
                    
                    # 4. Teknik analiz skoru
                    ta_strength = self.calculate_ta_strength(df, df)
                    self.logger.info(f"    📊 {pair}: TA Strength={ta_strength:.3f}")
                    
                    # 5. Whale skoru (gerçek analiz)
                    try:
                        whale_score = get_whale_score(pair)
                        if whale_score is None:
                            whale_score = 0.5
                    except Exception as e:
                        self.logger.error(f"    ❌ {pair}: Whale analizi hatası: {e}")
                        whale_score = 0.5
                    social_score = 0.0  # Pasif
                    news_score = 0.0    # Pasif
                    self.logger.info(f"    🐋 {pair}: Whale Score={whale_score:.3f}")
                    self.logger.info(f"    📰 {pair}: News Score={news_score:.3f} (PASİF)")

                    # 6. Toplam skor hesapla (sadece AI, TA, Whale)
                    total_score = (
                        ai_score * 0.5 +
                        ta_strength * 0.35 +
                        whale_score * 0.15
                    )
                    self.logger.info(f"    📊 {pair}: Total Score={total_score:.3f} | AI Score={ai_score:.3f} | TA Strength={ta_strength:.3f} | Whale Score={whale_score:.3f}")
                    self.logger.info(f"    Eşikler: MIN_CONF={min_conf:.3f} | MIN_AI={min_ai:.3f} | MIN_TA={min_ta:.3f}")
                    print(f"    📊 {pair}: Total Score={total_score:.3f} | AI Score={ai_score:.3f} | TA Strength={ta_strength:.3f} | Whale Score={whale_score:.3f}")
                    print(f"    Eşikler: MIN_CONF={min_conf:.3f} | MIN_AI={min_ai:.3f} | MIN_TA={min_ta:.3f}")
                    # 7. Minimum skor kontrolü - DÜZELTİLMİŞ
                    if total_score < min_conf:
                        self.logger.info(f"    ❌ {pair}: Toplam skor düşük ({total_score:.3f} < {min_conf}) [Market regime etkili] - SİNYAL ÜRETİLMEDİ")
                        print(f"    ❌ {pair}: Toplam skor düşük ({total_score:.3f} < {min_conf}) [Market regime etkili] - SİNYAL ÜRETİLMEDİ")
                        continue
                    if ai_score < min_ai:
                        self.logger.info(f"    ❌ {pair}: AI skoru düşük ({ai_score:.3f} < {min_ai}) [Market regime etkili] - SİNYAL ÜRETİLMEDİ")
                        print(f"    ❌ {pair}: AI skoru düşük ({ai_score:.3f} < {min_ai}) [Market regime etkili] - SİNYAL ÜRETİLMEDİ")
                        continue
                    if ta_strength < min_ta:
                        self.logger.info(f"    ❌ {pair}: TA skoru düşük ({ta_strength:.3f} < {min_ta}) [Market regime etkili] - SİNYAL ÜRETİLMEDİ")
                        print(f"    ❌ {pair}: TA skoru düşük ({ta_strength:.3f} < {min_ta}) [Market regime etkili] - SİNYAL ÜRETİLMEDİ")
                        continue
                    
                    # Çoklu onay filtresi: Whale skoru da kontrol edilecek
                    # if whale_score < Config.MIN_WHALE_SCORE:
                    #     self.logger.info(f"    ❌ {pair}: Whale skoru düşük ({whale_score:.3f} < {Config.MIN_WHALE_SCORE}) - Çoklu onay filtresi")
                    #     continue
                    
                    self.logger.info(f"    ✅ {pair}: Tüm skorlar geçti!")
                    
                    # 8. Sinyal yönü belirle - DÜZELTİLMİŞ
                    if ai_score > 0.6:  # Güçlü alım sinyali
                        direction = 'LONG'
                    elif ai_score < 0.4:  # Güçlü satım sinyali
                        direction = 'SHORT'
                    else:
                        direction = 'NEUTRAL'
                    
                    self.logger.info(f"    {pair}: Direction={direction}")
                    
                    # 9. Sinyal olustur
                    current_price = df['close'].iloc[-1] if not df.empty else None
                    
                    if current_price is None or np.isnan(current_price) or current_price <= 0:
                        self.logger.warning(f"    {pair}: Gecersiz fiyat: {current_price}")
                        continue
                    
                    # Sinyal oluşturma
                    self.logger.info(f"    {pair}: Sinyal oluşturuluyor... [Market regime: {market_regime} | {regime_note}]")
                    
                    # Analysis data'yı hazırla - eksik metrikleri ekle
                    analysis_data = df.tail(1).to_dict('records')[0]
                    
                    # Zorunlu alanları ekle
                    analysis_data['timeframe'] = '1h'
                    analysis_data['social_score'] = 0.0  # Pasif
                    analysis_data['news_score'] = 0.0    # Pasif
                    
                    # Teknik analiz metriklerini hesapla ve ekle
                    try:
                        # Volume score hesapla
                        volume_score = self.ta.calculate_volume_score(df) if hasattr(self.ta, 'calculate_volume_score') else 0.5
                        analysis_data['volume_score'] = volume_score if volume_score != 'Veri Yok' else 0.5
                        
                        # Momentum score hesapla
                        momentum_score = self.ta.calculate_momentum_score(df) if hasattr(self.ta, 'calculate_momentum_score') else 0.5
                        analysis_data['momentum_score'] = momentum_score if momentum_score != 'Veri Yok' else 0.5
                        
                        # Pattern score hesapla (son satırdan al)
                        pattern_score = df['pattern_score'].iloc[-1] if 'pattern_score' in df.columns else 0.5
                        analysis_data['pattern_score'] = pattern_score
                        
                        # Whale tracker metriklerini hesapla
                        whale_data = detect_whale_trades(pair)
                        analysis_data['whale_direction_score'] = whale_data.get('whale_direction_score', 0.0)
                        analysis_data['order_book_imbalance'] = whale_data.get('order_book_imbalance', 0.0)
                        analysis_data['top_bid_walls'] = whale_data.get('top_bid_walls', [])
                        analysis_data['top_ask_walls'] = whale_data.get('top_ask_walls', [])
                        
                        self.logger.info(f"    {pair}: Metrikler hesaplandı - Volume: {analysis_data['volume_score']}, Momentum: {analysis_data['momentum_score']}, Pattern: {analysis_data['pattern_score']}")
                        
                    except Exception as e:
                        self.logger.error(f"    {pair}: Metrik hesaplama hatası: {e}")
                        # Varsayılan değerler
                        analysis_data['volume_score'] = 0.5
                        analysis_data['momentum_score'] = 0.5
                        analysis_data['pattern_score'] = 0.5
                        analysis_data['whale_direction_score'] = 0.0
                        analysis_data['order_book_imbalance'] = 0.0
                        analysis_data['top_bid_walls'] = []
                        analysis_data['top_ask_walls'] = []
                    
                    signal = self.signal_manager.create_signal(
                        symbol=pair,
                        direction=direction,
                        confidence=confidence,
                        analysis_data=analysis_data
                    )
                    
                    if signal is not None:
                        # Sinyal kaydından hemen önce market_regime ve regime_note kesin olarak tanımlı olsun
                        try:
                            market_regime
                        except NameError:
                            market_regime = 'unknown'
                        try:
                            regime_note
                        except NameError:
                            regime_note = ''
                        signal['market_regime'] = market_regime
                        signal['regime_note'] = regime_note
                        try:
                            self.logger.info(f"    {pair}: Sinyal kaydediliyor...")
                            self.signal_manager.save_signal_json(signal)
                            self.signal_manager.save_signal_csv(signal)
                            self.signal_manager.save_signal_db(signal)  # Tek kayıt
                            self.logger.info(f"    {pair}: Sinyal kaydedildi!")
                        except Exception as e:
                            self.logger.error(f"    {pair}: Sinyal kaydetme hatası: {e}")
                    else:
                        self.logger.warning(f"    {pair}: create_signal None döndürdü!")
                    
                    signals_generated += 1
                    self.logger.info(f"    {pair}: Sinyal basariyla uretildi - Skor: {total_score:.2f}")
                
                except Exception as e:
                    self.logger.error(f"    {pair}: Genel hata: {e}")
                    continue
            
            if signals_generated == 0:
                self.logger.warning("Hicbir coin icin sinyal uretilemedi!")
                self.system_health = 'WARNING'
            else:
                self.logger.info(f"Toplam {signals_generated} sinyal uretildi.")
                self.system_health = 'HEALTHY'
                
        except Exception as e:
            self.logger.error(f"Sinyal uretim fonksiyonunda genel hata: {e}")
            self.system_health = 'ERROR'

    def retrain_models(self):
        """AI modellerini yeniden eğit"""
        try:
            self.logger.info("AI modelleri yeniden egitiliyor...")
            # Örnek veri ile egitim (gercek uygulamada daha fazla veri kullanilir)
            pairs = self.data_collector.get_usdt_pairs(max_pairs=10)
            for pair in pairs:
                df = self.data_collector.get_historical_data(pair, '1h', limit=1000)
                if not df.empty:
                    df = self.ta.calculate_all_indicators(df)
                    if not df.empty:
                        self.ai_model.retrain_daily(df)
                        break
            self.logger.info("AI modelleri yeniden egitildi")
        except Exception as e:
            self.logger.error(f"Model yeniden egitiminde hata: {e}")

    def run_scheduler(self):
        """Zamanlanmis gorevleri calistir"""
        # Zamanlanmis gorevler
        schedule.every(30).minutes.do(self.generate_signals)
        schedule.every(10).minutes.do(self.signal_tracker.track_open_signals)  # Açık sinyal kontrolü
        schedule.every(4).hours.do(lambda: self.performance_tracker.real_time_monitoring())
        
        self.logger.info("Zamanlanmis gorevler baslatildi")
        
        # Schedule ayarları
        schedule.every(1).minutes.do(functools.partial(generate_signals_wrapper, market_analyzer, performance_analyzer))
        schedule.every(4).hours.do(lambda: performance_analyzer.real_time_monitoring())
        
        # Ana döngü - sadece schedule çalıştır
        while True:
            try:
                self.logger.info("[LOOP] schedule.run_pending cagriliyor...")
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                self.logger.info("Program kullanıcı tarafından durduruldu")
                break
            except Exception as e:
                self.logger.error(f"Ana döngüde hata: {e}")
                time.sleep(60)  # 1 dakika bekle ve devam et

    def run_performance_monitoring(self):
        """Performans takibi calistir"""
        try:
            # Sinyal takibi
            self.signal_tracker.track_open_signals()
            
            # Performans güncelleme
            self.update_signal_performance()
            
            # Günlük temizlik (günde bir kez)
            current_hour = datetime.now().hour
            if current_hour == 0 and self.last_performance_update != datetime.now().date():
                self.signal_tracker.run_daily_cleanup()
                self.last_performance_update = datetime.now().date()
                
        except Exception as e:
            self.logger.error(f"Performans takibi hatasi: {e}")
    
    def update_signal_performance(self):
        """Açık sinyallerin performansını güncelle"""
        try:
            # Açık sinyalleri al
            open_signals = self.db.get_open_signals()
            
            for signal in open_signals:
                try:
                    signal_id = signal['id']
                    symbol = signal['symbol']
                    
                    # Güncel fiyatı al
                    current_price = self.get_current_price(symbol)
                    if current_price is None:
                        continue
                    
                    # Performansı güncelle
                    self.performance_tracker.update_signal_performance(
                        signal_id, current_price, datetime.now()
                    )
                    
                except Exception as e:
                    self.logger.error(f"Sinyal {signal_id} performans güncelleme hatasi: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Sinyal performans güncelleme hatasi: {e}")
    
    def get_current_price(self, symbol):
        """Sembol icin güncel fiyatı al"""
        try:
            data_collector = DataCollector()
            data = data_collector.get_historical_data(symbol, '1h', 1)
            
            if data is not None and not data.empty:
                return data['close'].iloc[-1]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"{symbol} fiyat alma hatasi: {e}")
            return None

    def analyze_symbol_advanced(self, symbol):
        """Gelişmiş sembol analizi - çoklu timeframe ve pattern recognition"""
        try:
            self.logger.info(f"Gelişmiş analiz baslatiliyor: {symbol}")
            
            # Çoklu timeframe veri toplama
            timeframes = ['5m', '15m', '1h', '4h']
            multi_tf_data = self.data_collector.get_multi_timeframe_data(symbol, timeframes)
            
            if not multi_tf_data:
                self.logger.warning(f"{symbol} için çoklu timeframe veri alınamadı")
                return None
            
            # Ana timeframe (1h) için analiz
            main_data = multi_tf_data.get('1h')
            if main_data is None or main_data.empty:
                self.logger.warning(f"{symbol} 1h verisi bulunamadı")
                return None
            
            # Teknik analiz
            ta_result = self.ta.analyze_technical_signals(main_data)
            if not ta_result or 'error' in ta_result:
                self.logger.warning(f"{symbol} teknik analiz hatası")
                return None
            
            # AI analizi
            ai_result = self.ai_model.predict(main_data)
            
            # Sinyal oluşturma
            signal = self.create_advanced_signal(symbol, main_data, ai_result, multi_tf_data)
            
            if signal:
                # Sinyal kalitesi değerlendirme
                quality_eval = self.signal_manager.evaluate_signal_quality(signal)
                signal['quality_evaluation'] = quality_eval
                
                # Piyasa koşulları kontrolü
                market_conditions = self.signal_manager.check_market_conditions(symbol)
                signal = self.signal_manager.adjust_signal_parameters(signal, market_conditions)
                
                # Minimum kalite skoru kontrolü
                if quality_eval['total_score'] >= 50:  # Minimum 50 puan
                    self.logger.info(f"Yüksek kaliteli sinyal: {symbol} - Skor: {quality_eval['total_score']}")
                    return signal
                else:
                    self.logger.info(f"Düşük kaliteli sinyal reddedildi: {symbol} - Skor: {quality_eval['total_score']}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Gelişmiş analiz hatasi {symbol}: {e}")
            return None
    
    def create_advanced_signal(self, symbol, data, ai_result, multi_tf_data):
        """Gelişmiş sinyal oluşturma"""
        try:
            if data is None or data.empty:
                return None
            
            # Teknik analiz sonucunu al
            ta_result = self.ta.analyze_technical_signals(data)
            if not ta_result or 'error' in ta_result:
                return None
            
            latest = data.iloc[-1]
            
            # Çoklu timeframe trend analizi
            tf_trends = {}
            for tf, tf_data in multi_tf_data.items():
                if not tf_data.empty:
                    tf_latest = tf_data.iloc[-1]
                    tf_trends[tf] = {
                        'trend': 'BULLISH' if tf_latest['close'] > tf_latest.get('sma_20', tf_latest['close']) else 'BEARISH',
                        'strength': abs(tf_latest['close'] - tf_latest.get('sma_20', tf_latest['close'])) / tf_latest['close']
                    }
            
            # Pattern recognition sonuçları
            patterns = []
            patterns_data = ta_result.get('patterns', {})
            for pattern_name, is_detected in patterns_data.items():
                if is_detected:
                    patterns.append(pattern_name.replace('_', ' ').title())
            
            # Sinyal oluştur
            signal = {
                'symbol': symbol,
                'timestamp': str(datetime.now()),
                'price': latest['close'],
                'direction': 'BUY' if ai_result['prediction'] > 0.5 else 'SELL',
                'confidence': ai_result['confidence'],
                'ai_score': ai_result['prediction'],
                'technical_score': ta_result.get('adx', 25),
                'patterns_detected': patterns,
                'multi_timeframe_trends': tf_trends,
                'volume_ratio': ta_result.get('obv', 1.0),
                'trend_strength': ta_result.get('adx', 25),
                'stop_loss': self.calculate_stop_loss_from_ta(ta_result, ai_result['prediction']),
                'take_profit': self.calculate_take_profit_from_ta(ta_result, ai_result['prediction']),
                'risk_reward_ratio': 2.0,  # Sabit risk/reward oranı
                'market_conditions': self.get_market_conditions(data),
                'analysis_details': ta_result
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Sinyal olusturma hatasi {symbol}: {e}")
            return None
    
    def calculate_technical_score(self, latest_data):
        """Teknik analiz skoru hesapla"""
        try:
            score = 0.5  # Başlangıç skoru
            
            # RSI analizi
            rsi = latest_data.get('rsi_14', 50)
            if 30 <= rsi <= 70:
                score += 0.1
            elif rsi < 30 or rsi > 70:
                score += 0.2  # Aşırı alım/satım bölgeleri
            
            # MACD analizi
            macd = latest_data.get('macd', 0)
            macd_signal = latest_data.get('macd_signal', 0)
            if macd > macd_signal:
                score += 0.1
            
            # Bollinger Bands analizi
            bb_pos = latest_data.get('bb_percent', 0.5)
            if bb_pos < 0.2:  # Alt banda yakın
                score += 0.1
            elif bb_pos > 0.8:  # Üst banda yakın
                score -= 0.1
            
            # Hacim analizi
            vol_ratio = latest_data.get('volume_ratio', 1.0)
            if vol_ratio > 1.5:
                score += 0.1
            
            return min(max(score, 0.0), 1.0)  # 0-1 arasında sınırla
            
        except Exception as e:
            self.logger.error(f"Teknik skor hesaplama hatasi: {e}")
            return 0.5
    
    def calculate_stop_loss_from_ta(self, ta_result, ai_prediction):
        """Yeni teknik analiz sonucundan stop loss hesapla"""
        try:
            atr = ta_result.get('atr', 0)
            current_price = ta_result.get('ema', {}).get(5, 0)  # EMA5'i fiyat olarak kullan
            
            if atr == 0 or current_price == 0:
                return current_price * 0.95 if ai_prediction > 0.5 else current_price * 1.05
            
            if ai_prediction > 0.5:  # Long pozisyon
                return current_price - (atr * 2)
            else:  # Short pozisyon
                return current_price + (atr * 2)
        except Exception as e:
            self.logger.error(f"Stop loss hesaplama hatasi: {e}")
            return 0
    
    def calculate_take_profit_from_ta(self, ta_result, ai_prediction):
        """Yeni teknik analiz sonucundan take profit hesapla"""
        try:
            atr = ta_result.get('atr', 0)
            current_price = ta_result.get('ema', {}).get(5, 0)  # EMA5'i fiyat olarak kullan
            
            if atr == 0 or current_price == 0:
                return current_price * 1.10 if ai_prediction > 0.5 else current_price * 0.90
            
            if ai_prediction > 0.5:  # Long pozisyon
                return current_price + (atr * 4)  # 2:1 risk/reward
            else:  # Short pozisyon
                return current_price - (atr * 4)
        except Exception as e:
            self.logger.error(f"Take profit hesaplama hatasi: {e}")
            return 0
    
    def get_market_conditions(self, data):
        """Piyasa koşullarını belirle"""
        try:
            if data is None or data.empty:
                return {'volatility': 'NORMAL', 'trend': 'NEUTRAL'}
            
            latest = data.iloc[-1]
            
            # Volatilite hesaplama
            atr_ratio = latest.get('atr', 0) / latest['close']
            if atr_ratio > 0.03:
                volatility = 'HIGH'
            elif atr_ratio < 0.01:
                volatility = 'LOW'
            else:
                volatility = 'NORMAL'
            
            # Trend hesaplama
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            
            if latest['close'] > sma_20 > sma_50:
                trend = 'BULLISH'
            elif latest['close'] < sma_20 < sma_50:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
            
            return {
                'volatility': volatility,
                'trend': trend,
                'volume_trend': latest.get('volume_trend', 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Piyasa koşulları hesaplama hatasi: {e}")
            return {'volatility': 'NORMAL', 'trend': 'NEUTRAL'}

    def calculate_ta_strength(self, ta_data, historical_data=None):
        """Technical analysis strength hesapla"""
        try:
            # RSI strength
            rsi_strength = 0.0
            if 'rsi_14' in ta_data.columns:
                rsi = ta_data['rsi_14'].iloc[-1]
                if rsi < 30 or rsi > 70:
                    rsi_strength = 0.8
                elif rsi < 40 or rsi > 60:
                    rsi_strength = 0.6
                else:
                    rsi_strength = 0.3
            
            # MACD strength
            macd_strength = 0.0
            if 'macd' in ta_data.columns and 'macd_signal' in ta_data.columns:
                macd = ta_data['macd'].iloc[-1]
                macd_signal = ta_data['macd_signal'].iloc[-1]
                if abs(macd - macd_signal) > 0.001:
                    macd_strength = 0.7
                else:
                    macd_strength = 0.3
            
            # Moving average strength
            ma_strength = 0.0
            if 'sma_20' in ta_data.columns and 'sma_50' in ta_data.columns and historical_data is not None:
                sma_20 = ta_data['sma_20'].iloc[-1]
                sma_50 = ta_data['sma_50'].iloc[-1]
                close = historical_data['close'].iloc[-1] if 'close' in historical_data.columns else 0
                
                if close > sma_20 > sma_50:
                    ma_strength = 0.8
                elif close < sma_20 < sma_50:
                    ma_strength = 0.8
                else:
                    ma_strength = 0.4
            
            # Overall TA strength
            ta_strength = (rsi_strength + macd_strength + ma_strength) / 3
            return min(1.0, max(0.0, ta_strength))
            
        except Exception as e:
            self.logger.error(f"TA strength calculation hatasi: {e}")
            return 0.5

    def calculate_advanced_risk_levels(self, current_price, direction, support_levels, 
                                     resistance_levels, breakout_threshold, multi_timeframe):
        """Gelişmiş stop loss ve take profit hesaplama"""
        try:
            # ATR bazlı volatilite
            atr_multiplier = 2.0  # Varsayılan ATR çarpanı
            
            # Trend gücüne göre ATR çarpanını ayarla
            trend_strength = multi_timeframe.get('1h', {}).get('strength', 0.5)
            if trend_strength > 0.7:
                atr_multiplier = 1.5  # Güçlü trend - daha sıkı
            elif trend_strength < 0.3:
                atr_multiplier = 3.0  # Zayıf trend - daha geniş
            
            # Momentum bazlı ayarlama
            momentum_score = multi_timeframe.get('1h', {}).get('momentum', 0.5)
            if momentum_score > 0.7:
                atr_multiplier *= 0.8  # Yüksek momentum - daha sıkı
            elif momentum_score < 0.3:
                atr_multiplier *= 1.2  # Düşük momentum - daha geniş
            
            # Breakout threshold bazlı ayarlama
            if breakout_threshold < 0.02:  # %2'den az
                atr_multiplier *= 0.7  # Sıkı breakout - daha sıkı stop
            elif breakout_threshold > 0.05:  # %5'ten fazla
                atr_multiplier *= 1.3  # Geniş breakout - daha geniş stop
            
            if direction == 'BUY':
                # Stop loss - En yakın destek seviyesinin altında
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], default=0)
                    if nearest_support > 0:
                        stop_loss = nearest_support * 0.995  # %0.5 altında
                    else:
                        stop_loss = current_price * (1 - 0.03)  # %3 altında
                else:
                    stop_loss = current_price * (1 - 0.03)
                
                # Take profit - En yakın direnç seviyesinin üstünde
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
                    if nearest_resistance != float('inf'):
                        take_profit = nearest_resistance * 1.005  # %0.5 üstünde
                    else:
                        take_profit = current_price * (1 + 0.06)  # %6 üstünde
                else:
                    take_profit = current_price * (1 + 0.06)
            
            else:  # SELL
                # Stop loss - En yakın direnç seviyesinin üstünde
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
                    if nearest_resistance != float('inf'):
                        stop_loss = nearest_resistance * 1.005  # %0.5 üstünde
                    else:
                        stop_loss = current_price * (1 + 0.03)  # %3 üstünde
                else:
                    stop_loss = current_price * (1 + 0.03)
                
                # Take profit - En yakın destek seviyesinin altında
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], default=0)
                    if nearest_support > 0:
                        take_profit = nearest_support * 0.995  # %0.5 altında
                    else:
                        take_profit = current_price * (1 - 0.06)  # %6 altında
                else:
                    take_profit = current_price * (1 - 0.06)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Gelişmiş risk seviyeleri hesaplama hatasi: {e}")
            # Varsayılan degerler
            if direction == 'BUY':
                return current_price * 0.97, current_price * 1.06
            else:
                return current_price * 1.03, current_price * 0.94
    
    def calculate_risk_reward_ratio(self, current_price, stop_loss, take_profit):
        """Risk/ödül oranını hesapla"""
        try:
            if current_price <= 0:
                return 0.0
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk <= 0:
                return 0.0
            result = reward / risk
            try:
                return float(result)
            except Exception:
                return 0.0
        except Exception as e:
            self.logger.error(f"Risk/ödül oranı hesaplama hatasi: {e}")
            return 0.0

def generate_signals_wrapper(market_analyzer, performance_analyzer):
    try:
        logger.info("[SCHEDULE] generate_signals cagriliyor...")
        generate_signals(market_analyzer, performance_analyzer)
        logger.info("[SCHEDULE] generate_signals tamamlandı.")
    except Exception as e:
        logger.error(f"[SCHEDULE] generate_signals hata: {e}")

def main():
    """Ana fonksiyon"""
    try:
        # Config directories oluştur
        Config.create_directories()
        
        # Data collector instance
        data_collector = DataCollector()
        
        # Market analysis instance
        market_analyzer = MarketAnalysis()
        
        # Performance analyzer (not tracker)
        performance_analyzer = PerformanceAnalyzer()
        
        logger.info("Kahin Ultima baslatiliyor...")
        
        # PostgreSQL tabloları oluştur
        signal_manager = SignalManager()
        
        # SignalTracker instance (EKLENDİ)
        signal_tracker = SignalTracker()
        
        # Market conditions analizi
        if Config.MARKET_REGIME_DETECTION:
            logger.info("Market regime detection baslatiliyor...")
            # BTC data al ve market regime detect et
            btc_data = data_collector.get_historical_data('BTC/USDT', '4h', 100)
            if not btc_data.empty:
                market_regime, volatility_regime = market_analyzer.detect_market_regime(btc_data)
                logger.info(f"Market regime: {market_regime}, Volatility: {volatility_regime}")
        
        # Sentiment analysis
        if Config.SENTIMENT_ANALYSIS_ENABLED:
            logger.info("Market sentiment analysis baslatiliyor...")
            sentiment_score = market_analyzer.analyze_market_sentiment()
            logger.info(f"Market sentiment score: {sentiment_score:.2f}")
        
        # Real-time monitoring baslat
        if Config.REAL_TIME_MONITORING:
            logger.info("Real-time monitoring baslatiliyor...")
            monitoring_data = performance_analyzer.real_time_monitoring()
            logger.info(f"System health: {monitoring_data.get('system_health', {}).get('status', 'unknown')}")
        
        # Zamanlanmis gorevler baslat
        logger.info("Zamanlanmis gorevler baslatildi")
        
        # Basit zamanlayıcı - son sinyal uretim zamanını takip et
        last_signal_time = datetime.now()
        last_signal_close_check = datetime.now()
        signal_interval = 60  # 1 dakika
        close_check_interval = 600  # 10 dakika
        adapt_interval = 1800  # 30 dakika
        last_adapt_check = datetime.now()
        
        # Ana döngü - basit zamanlayıcı ile
        while True:
            try:
                current_time = datetime.now()
                time_since_last_signal = (current_time - last_signal_time).total_seconds()
                time_since_last_close_check = (current_time - last_signal_close_check).total_seconds()
                time_since_last_adapt = (current_time - last_adapt_check).total_seconds()
                
                # Her dakika sinyal üretimi
                if time_since_last_signal >= signal_interval:
                    logger.info("[TIMER] Sinyal uretimi zamanı geldi, fonksiyon cagriliyor...")
                    try:
                        generate_signals_wrapper(market_analyzer, performance_analyzer)
                        last_signal_time = current_time
                        logger.info("[TIMER] Sinyal uretimi tamamlandı, sonraki kontrol: 1 dakika sonra")
                    except Exception as e:
                        logger.error(f"[TIMER] Sinyal uretimi hatasi: {e}")

                # Her 10 dakikada bir açık sinyalleri kontrol et ve kapat
                if time_since_last_close_check >= close_check_interval:
                    logger.info("[TIMER] Açık sinyaller kontrol ediliyor...")
                    try:
                        signal_tracker.track_open_signals()
                        last_signal_close_check = current_time
                        logger.info("[TIMER] Açık sinyal kontrolü tamamlandı, sonraki kontrol: 10 dakika sonra")
                    except Exception as e:
                        logger.error(f"[TIMER] Açık sinyal kontrolü hatası: {e}")

                # Her 30 dakikada bir yeni coin ve piyasa adaptasyonu
                if time_since_last_adapt >= adapt_interval:
                    logger.info("[TIMER] Yeni coin ve piyasa adaptasyonu kontrol ediliyor...")
                    try:
                        data_collector.adapt_to_market_conditions()
                        last_adapt_check = current_time
                        logger.info("[TIMER] Adaptasyon tamamlandı, sonraki kontrol: 30 dakika sonra")
                    except Exception as e:
                        logger.error(f"[TIMER] Adaptasyon hatası: {e}")
                
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Program kullanıcı tarafından durduruldu")
                break
            except Exception as e:
                logger.error(f"Ana döngüde hata: {e}")
                time.sleep(60)  # 1 dakika bekle ve devam et
                
    except Exception as e:
        logger.error(f"Program baslatma hatasi: {e}")
        raise

def generate_signals(market_analyzer, performance_analyzer):
    """Sinyal uretimi - DİNAMİK SIKILIK ENTEGRASYONU"""
    try:
        logger.info("Sinyal uretimi baslatiliyor (Dinamik Sıkılık Aktif)...")
        
        # Modülleri başlat
        data_collector = DataCollector()
        technical_analyzer = TechnicalAnalysis()
        signal_manager = SignalManager()
        telegram_bot = TelegramBot()
        from modules.ai_model import AIModel
        ai_model = AIModel()
        
        # Dinamik sıkılık sistemi başlat
        dynamic_strictness = DynamicStrictness()
        logger.info("✅ Dinamik sıkılık sistemi başlatıldı")
        current_status = dynamic_strictness.get_current_status()
        current_strictness = current_status['current_strictness']
        strictness_level = current_status['strictness_level']
        recommendation = current_status['recommendation']
        
        # Popüler coinleri al
        popular_coins = data_collector.get_popular_usdt_pairs(max_pairs=Config.MAX_COINS_TO_TRACK)
        logger.info(f"Toplam {len(popular_coins)} coin işlenecek")
        
        # Market verisi topla ve dinamik sıkılığı güncelle
        logger.info("📈 Market verisi toplanıyor ve dinamik sıkılık hesaplanıyor...")
        
        # Örnek market verisi (gerçek uygulamada daha kapsamlı olacak)
        market_data = {
            'price_data': [],
            'technical_indicators': {},
            'volume_data': [],
            'sentiment_data': {'overall_sentiment': 0.5},
            'ai_predictions': {'confidence': 0.5}
        }
        
        # İlk 30 coin'den ve iki timeframe'den market verisi topla
        sample_coins = popular_coins[:30]
        for coin in sample_coins:
            try:
                for tf in ['1h', '4h']:
                    data = data_collector.get_historical_data(coin, tf, 50)
                    if not data.empty:
                        market_data['price_data'].extend(data['close'].tolist())
                        ta_data = technical_analyzer.calculate_all_indicators(data)
                        if 'rsi_14' in ta_data.columns:
                            market_data['technical_indicators']['rsi'] = ta_data['rsi_14'].iloc[-1]
                        if 'macd' in ta_data.columns:
                            market_data['technical_indicators']['macd'] = ta_data['macd'].iloc[-1]
                        if 'volume' in data.columns:
                            market_data['volume_data'].extend(data['volume'].tolist())
                        ai_result = ai_model.predict(ta_data)
                        if ai_result:
                            market_data['ai_predictions']['confidence'] = ai_result.get('confidence', 0.5)
            except Exception as e:
                logger.warning(f"Market verisi toplama hatası ({coin}): {e}")
                continue
        # Dinamik eşikleri güncelle (daha sıkı alt limit)
        dynamic_min_confidence = max(0.5, min(0.8, current_strictness))
        dynamic_min_ai_score = max(0.45, min(0.7, current_strictness - 0.1))
        dynamic_min_ta_strength = max(0.35, min(0.6, current_strictness - 0.15))
        logger.info(f"⚙️ DİNAMİK EŞİKLER (GÜNCEL):")
        logger.info(f"   MIN_CONFIDENCE: {dynamic_min_confidence:.3f} (sabit: {Config.MIN_SIGNAL_CONFIDENCE})")
        logger.info(f"   MIN_AI_SCORE: {dynamic_min_ai_score:.3f} (sabit: {Config.MIN_AI_SCORE})")
        logger.info(f"   MIN_TA_STRENGTH: {dynamic_min_ta_strength:.3f} (sabit: {Config.MIN_TA_STRENGTH})")
        
        tum_sinyaller = []
        batch_size = 20
        total_batches = (len(popular_coins) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(popular_coins))
            batch_coins = popular_coins[start_idx:end_idx]
            logger.info(f"📦 Batch {batch_num + 1}/{total_batches}: {len(batch_coins)} coin işleniyor...")
            
            batch_signals = 0
            
            for coin in batch_coins:
                try:
                    logger.info(f"İşleniyor: {coin}")
                    
                    # Her coin için market_regime ve regime_note default başlat
                    market_regime = 'unknown'
                    regime_note = ''

                    # Son 24 saatte sinyal kontrolü
                    recent_signals = signal_manager.get_recent_signals_by_symbol(coin, hours=8)
                    acil_firsat = False
                    if recent_signals and len(recent_signals) > 0:
                        # Acil fırsat kontrolü: Son 2 bar içinde %2 veya daha fazla fiyat değişimi var mı?
                        df = data_collector.get_historical_data(coin, '1h', 100) # 100 bar alındığı için
                        if not df.empty and len(df) >= 2:
                            price_now = df['close'].iloc[-1]
                            price_prev = df['close'].iloc[-2]
                            price_change = abs(price_now - price_prev) / price_prev
                            if price_change >= 0.02:
                                acil_firsat = True
                                logger.info(f"{coin}: Acil fırsat: Ani hareket tespit edildi (%{price_change*100:.2f}), zaman limiti esnetildi!")
                    if recent_signals and len(recent_signals) > 0 and not acil_firsat:
                        logger.info(f"{coin}: Son 8 saatte zaten sinyal üretilmiş, atlanıyor.")
                        continue
                    
                    # Veri al
                    historical_data = data_collector.get_historical_data(coin, '1h', 100)
                    if historical_data.empty:
                        logger.info(f"{coin}: Veri alınamadı, atlanıyor.")
                        continue
                    logger.info(f"{coin}: {len(historical_data)} satır veri alındı")
                    
                    # Teknik analiz
                    ta_data = technical_analyzer.calculate_all_indicators(historical_data)
                    
                    # Teknik analizden sonra, skorlar hesaplanmadan hemen önce debug ekle
                    if 'df_ta' in locals():
                        print('[DEBUG][main.py] Teknik analiz df_ta son satır:', df_ta.tail(1))
                        print('[DEBUG][main.py] Teknik analiz df_ta kolonlar:', df_ta.columns)
                    if 'ta_data' in locals():
                        print('[DEBUG][main.py] Teknik analiz ta_data son satır:', ta_data.tail(1))
                        print('[DEBUG][main.py] Teknik analiz ta_data kolonlar:', ta_data.columns)
                    
                    # --- CURRENT PRICE EKLE ---
                    if 'close' in ta_data.columns and not ta_data['close'].empty:
                        current_price = ta_data['close'].iloc[-1]
                        if current_price is None or np.isnan(current_price) or current_price <= 0:
                            logger.warning(f"{coin}: Geçersiz fiyat: {current_price}")
                            continue
                    else:
                        logger.warning(f"{coin}: Teknik analizde fiyat (close) bulunamadı, atlanıyor.")
                        continue
                    
                    # AI prediction
                    ai_result = ai_model.predict(ta_data)
                    if ai_result is None:
                        logger.info(f"{coin}: AI sonucu None, atlanıyor.")
                        continue
                    ai_score = ai_result.get('prediction', 0.5)
                    confidence = ai_result.get('confidence', 0.5)
                    feature_dummy_ratio = ai_result.get('feature_dummy_ratio', 1.0)
                    # --- DIRECTION EKLE ---
                    if ai_score > 0.6:
                        direction = 'LONG'
                    elif ai_score < 0.4:
                        direction = 'SHORT'
                    else:
                        direction = 'NEUTRAL'
                    
                    # TA strength hesapla
                    ta_strength = calculate_ta_strength(ta_data, historical_data)
                    
                    # Whale score hesapla
                    try:
                        whale_score = get_whale_score(coin)
                        if whale_score is None:
                            whale_score = 0.5
                    except Exception:
                        whale_score = 0.5
                    
                    # --- TOTAL SCORE HESAPLA (KESİN DÜZELTME) ---
                    total_score = (
                        ai_score * 0.5 +
                        ta_strength * 0.35 +
                        whale_score * 0.15
                    )

                    logger.info(f"{coin}: [AKIŞ] Teknik analiz ve whale analizinden sonra skor hesaplama başlıyor.")
                    # --- Skorları hesapla ve logla ---
                    logger.info(f"{coin}: AI_SCORE={ai_score:.3f}, TA_STRENGTH={ta_strength:.3f}, WHALE_SCORE={whale_score:.3f}, TOTAL_SCORE={total_score:.3f}, CONFIDENCE={confidence:.3f}")
                    logger.info(f"{coin}: [AKIŞ] Skor hesaplama ve loglama tamamlandı, eşik kontrollerine geçiliyor.")

                    if total_score is None or (hasattr(np, 'isnan') and np.isnan(total_score)):
                        logger.info(f"{coin}: [ELENDİ] Toplam skor hesaplanamadı (None/NaN) - SİNYAL ÜRETİLMEDİ")
                        continue

                    if total_score < dynamic_min_confidence:
                        logger.info(f"{coin}: [ELENDİ] Toplam skor düşük: {total_score:.3f} < {dynamic_min_confidence:.3f} - SİNYAL ÜRETİLMEDİ")
                        continue

                    if ai_score < dynamic_min_ai_score:
                        logger.info(f"{coin}: [ELENDİ] AI skoru düşük: {ai_score:.3f} < {dynamic_min_ai_score:.3f} - SİNYAL ÜRETİLMEDİ")
                        continue

                    if ta_strength < dynamic_min_ta_strength:
                        logger.info(f"{coin}: [ELENDİ] TA skoru düşük: {ta_strength:.3f} < {dynamic_min_ta_strength:.3f} - SİNYAL ÜRETİLMEDİ")
                        continue

                    logger.info(f"{coin}: [GEÇTİ] Tüm eşikler aşıldı, sinyal üretilecek.")
                    
                    # Çoklu onay filtresi: Whale skoru da kontrol edilecek
                    # if whale_score < Config.MIN_WHALE_SCORE:
                    #     logger.info(f"    ❌ {pair}: Whale skoru düşük ({whale_score:.3f} < {Config.MIN_WHALE_SCORE}) - Çoklu onay filtresi")
                    #     continue
                    
                    logger.info(f"    ✅ {coin}: Tüm skorlar geçti!")
                    
                    # 8. Sinyal yönü belirle - DÜZELTİLMİŞ
                    if ai_score > 0.6:  # Güçlü alım sinyali
                        direction = 'LONG'
                    elif ai_score < 0.4:  # Güçlü satım sinyali
                        direction = 'SHORT'
                    else:
                        direction = 'NEUTRAL'
                    
                    logger.info(f"    {coin}: Direction={direction}")
                    
                    # 9. Sinyal olustur
                    current_price = historical_data['close'].iloc[-1] if not historical_data.empty else None
                    
                    if current_price is None or np.isnan(current_price) or current_price <= 0:
                        logger.warning(f"    {coin}: Gecersiz fiyat: {current_price}")
                        continue
                    
                    # Sinyal oluşturma
                    logger.info(f"    {coin}: Sinyal oluşturuluyor... [Market regime: {market_regime} | {regime_note}]")
                    
                    # Analysis data'yı hazırla - eksik metrikleri ekle
                    analysis_data = ta_data.tail(1).to_dict('records')[0] if not ta_data.empty else {}
                    
                    # Zorunlu alanları ekle
                    analysis_data['timeframe'] = '1h'
                    analysis_data['social_score'] = 0.0  # Pasif
                    analysis_data['news_score'] = 0.0    # Pasif
                    
                    # Teknik analiz metriklerini hesapla ve ekle
                    try:
                        # Volume score hesapla
                        volume_score = technical_analyzer.calculate_volume_score(historical_data) if hasattr(technical_analyzer, 'calculate_volume_score') else 0.5
                        analysis_data['volume_score'] = volume_score if volume_score != 'Veri Yok' else 0.5
                        
                        # Momentum score hesapla
                        momentum_score = technical_analyzer.calculate_momentum_score(historical_data) if hasattr(technical_analyzer, 'calculate_momentum_score') else 0.5
                        analysis_data['momentum_score'] = momentum_score if momentum_score != 'Veri Yok' else 0.5
                        
                        # Pattern score hesapla (son satırdan al)
                        pattern_score = historical_data['pattern_score'].iloc[-1] if 'pattern_score' in historical_data.columns else 0.5
                        analysis_data['pattern_score'] = pattern_score
                        
                        # Whale tracker metriklerini hesapla
                        whale_data = detect_whale_trades(coin)
                        analysis_data['whale_direction_score'] = whale_data.get('whale_direction_score', 0.0)
                        analysis_data['order_book_imbalance'] = whale_data.get('order_book_imbalance', 0.0)
                        analysis_data['top_bid_walls'] = whale_data.get('top_bid_walls', [])
                        analysis_data['top_ask_walls'] = whale_data.get('top_ask_walls', [])
                        
                        logger.info(f"    {coin}: Metrikler hesaplandı - Volume: {analysis_data['volume_score']}, Momentum: {analysis_data['momentum_score']}, Pattern: {analysis_data['pattern_score']}")
                        
                    except Exception as e:
                        logger.error(f"    {coin}: Metrik hesaplama hatası: {e}")
                        # Varsayılan değerler
                        analysis_data['volume_score'] = 0.5
                        analysis_data['momentum_score'] = 0.5
                        analysis_data['pattern_score'] = 0.5
                        analysis_data['whale_direction_score'] = 0.0
                        analysis_data['order_book_imbalance'] = 0.0
                        analysis_data['top_bid_walls'] = []
                        analysis_data['top_ask_walls'] = []
                    
                    signal = signal_manager.create_signal(
                        symbol=coin,
                        direction=direction,
                        confidence=confidence,
                        analysis_data=analysis_data
                    )
                    
                    if signal is not None:
                        # Sinyal kaydından hemen önce market_regime ve regime_note kesin olarak tanımlı olsun
                        try:
                            market_regime
                        except NameError:
                            market_regime = 'unknown'
                        try:
                            regime_note
                        except NameError:
                            regime_note = ''
                        signal['market_regime'] = market_regime
                        signal['regime_note'] = regime_note
                        try:
                            logger.info(f"    {coin}: Sinyal kaydediliyor...")
                            signal_manager.save_signal_json(signal)
                            signal_manager.save_signal_csv(signal)
                            try:
                                signal_manager.save_signal_db(signal)
                            except Exception as db_exc:
                                logger.error(f"{coin}: Sinyal veritabanına kaydedilemedi, Telegram'a gönderilmeyecek! Hata: {db_exc}")
                                continue  # DB kaydı başarısızsa Telegram'a gönderme
                            telegram_bot.send_signal_notification(signal)
                            final_signals += 1
                        except Exception as e:
                            logger.error(f"{coin}: Sinyal kaydetme/gönderme hatası: {e}")
                    else:
                        logger.warning(f"{coin}: create_signal None döndürdü!")
                    
                    tum_sinyaller.append(signal)
                    batch_signals += 1
                    
                except Exception as e:
                    logger.error(f"{coin}: [EXCEPTION] Coin işlenirken beklenmeyen hata: {str(e)}")
                    continue
            
            logger.info(f"Batch {batch_num + 1} tamamlandı: {batch_signals} sinyal üretildi")
        
        logger.info(f"TÜM BATCH'LER TAMAMLANDI: Toplam {len(tum_sinyaller)} sinyal üretildi")
        
        if len(tum_sinyaller) == 0:
            logger.warning("❌ HİÇ SİNYAL ÜRETİLMEDİ!")
            return
        
        # DİNAMİK FİLTRELEME
        logger.info(f"🔍 Dinamik filtreleme başlıyor: {len(tum_sinyaller)} sinyal")
        logger.info(f"   Dinamik eşik: {dynamic_min_confidence:.3f} (sabit: {Config.MIN_SIGNAL_CONFIDENCE})")
        
        filtered_signals = signal_manager.filter_signals(
            tum_sinyaller,
            min_confidence=dynamic_min_confidence,  # Dinamik eşik kullan
            max_signals=Config.MAX_SIGNALS_PER_BATCH
        )
        logger.info(f"✅ Dinamik filtreleme sonrası: {len(filtered_signals)} sinyal geçti")
        print(f"[DEBUG] Filtreleme sonrası {len(filtered_signals)} sinyal var.")
        if filtered_signals:
            print(f"[DEBUG] İlk sinyal örneği: {filtered_signals[0]}")
        else:
            print("[DEBUG] Filtreleme sonrası hiç sinyal yok!")
        final_signals = 0
        
        for signal_data in filtered_signals:
            try:
                print(f"[DEBUG] Sinyal kaydediliyor: {signal_data['symbol']}")
                print('[DEBUG][main.py] Kaydedilecek sinyal:', signal_data)
                print('[DEBUG][main.py] Skorlar:', {
                    'signal_strength': signal_data.get('signal_strength'),
                    'ta_strength': signal_data.get('ta_strength'),
                    'volume_score': signal_data.get('volume_score'),
                    'momentum_score': signal_data.get('momentum_score'),
                    'pattern_score': signal_data.get('pattern_score'),
                })
                logger.info(f"Sinyal kaydediliyor: {signal_data['symbol']}")
                signal_manager.save_signal_json(signal_data)
                signal_manager.save_signal_csv(signal_data)
                try:
                    signal_manager.save_signal_db(signal_data)
                except Exception as db_exc:
                    logger.error(f"{signal_data['symbol']}: Sinyal veritabanına kaydedilemedi, Telegram'a gönderilmeyecek! Hata: {db_exc}")
                    continue  # DB kaydı başarısızsa Telegram'a gönderme
                telegram_bot.send_signal_notification(signal_data)
                final_signals += 1
                
            except Exception as e:
                logger.error(f"{signal_data['symbol']}: Sinyal kaydetme/gönderme hatası: {e}")
        
        logger.info(f"🎯 Toplam {final_signals} sinyal başarıyla kaydedildi.")
        
        # PERFORMANS TAKİBİ VE OTOMATİK AYARLAMA
        logger.info("📊 Performans takibi başlatılıyor...")
        
        # Sinyal sonuçlarını hazırla
        signal_results = {
            'total_signals': final_signals,
            'successful_signals': 0,  # Bu değer gerçek zamanlı olarak güncellenecek
            'failed_signals': 0,      # Bu değer gerçek zamanlı olarak güncellenecek
            'timestamp': datetime.now().isoformat()
        }
        
        # Dinamik sıkılık performans metriklerini güncelle
        dynamic_strictness.update_performance_metrics(signal_results)
        
        # Performans özetini al
        performance_summary = dynamic_strictness.get_performance_summary()
        
        logger.info("🔧 Dinamik sikilik ozeti:")
        logger.info(f"   Siklik seviyesi: {strictness_level}")
        logger.info(f"   Siklik degeri: {current_strictness:.3f}")
        logger.info(f"   Oneri: {recommendation}")
        logger.info(f"   Dinamik esikler: CONF={dynamic_min_confidence:.3f}, AI={dynamic_min_ai_score:.3f}, TA={dynamic_min_ta_strength:.3f}")
        
        # Performans bilgilerini logla
        logger.info("📈 PERFORMANS OZETI:")
        logger.info(f"   Basari orani: {performance_summary.get('success_rate', 0):.1%}")
        logger.info(f"   Toplam sinyal: {performance_summary.get('total_signals', 0)}")
        logger.info(f"   Otomatik ayarlama: {performance_summary.get('auto_adjustment_status', 'Unknown')}")
        
        # Son ayarlama bilgisini logla
        last_adjustment = performance_summary.get('last_adjustment')
        if last_adjustment:
            logger.info(f"   Son ayarlama: {last_adjustment.get('reason', 'Unknown')}")
        
        logger.info("✅ Performans takibi tamamlandi")
        
    except Exception as e:
        logger.error(f"Sinyal uretimi hatasi: {e}")
        import traceback
        logger.error(f"Hata detayi: {traceback.format_exc()}")

def safe_format(val):
    try:
        return f"{float(val):.2f}"
    except Exception:
        return "N/A"

def calculate_ta_strength(ta_data, historical_data=None):
    """Technical analysis strength hesapla"""
    try:
        # RSI strength
        rsi_strength = 0.0
        if 'rsi_14' in ta_data.columns:
            rsi = ta_data['rsi_14'].iloc[-1]
            if rsi < 30 or rsi > 70:
                rsi_strength = 0.8
            elif rsi < 40 or rsi > 60:
                rsi_strength = 0.6
            else:
                rsi_strength = 0.3
        
        # MACD strength
        macd_strength = 0.0
        if 'macd' in ta_data.columns and 'macd_signal' in ta_data.columns:
            macd = ta_data['macd'].iloc[-1]
            macd_signal = ta_data['macd_signal'].iloc[-1]
            if abs(macd - macd_signal) > 0.001:
                macd_strength = 0.7
            else:
                macd_strength = 0.3
        
        # Moving average strength
        ma_strength = 0.0
        if 'sma_20' in ta_data.columns and 'sma_50' in ta_data.columns and historical_data is not None:
            sma_20 = ta_data['sma_20'].iloc[-1]
            sma_50 = ta_data['sma_50'].iloc[-1]
            close = historical_data['close'].iloc[-1] if 'close' in historical_data.columns else 0
            
            if close > sma_20 > sma_50:
                ma_strength = 0.8
            elif close < sma_20 < sma_50:
                ma_strength = 0.8
            else:
                ma_strength = 0.4
        
        # Overall TA strength
        ta_strength = (rsi_strength + macd_strength + ma_strength) / 3
        return min(1.0, max(0.0, ta_strength))
        
    except Exception as e:
        logger.error(f"TA strength calculation hatasi: {e}")
        return 0.5

import threading
try:
    from app.web import app
    from config import Config
    def run_flask():
        print("🔮 Kahin Ultima Web Panel Başlatılıyor...")
        print(f"🌐 Web arayüzü: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
        print("📊 API endpoint: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/")
        print("=" * 50)
        app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=False, threaded=True)
    if __name__ == "__main__":
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        main()
except Exception as e:
    print(f"Web panel başlatılamadı: {e}")
    if __name__ == "__main__":
        main()

if __name__ == "__main__":
    main() 