import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from datetime import timezone
from sqlalchemy import create_engine, text
from config import Config
from modules.data_collector import DataCollector
from modules.signal_manager import SignalManager

class SignalTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_url = Config.DATABASE_URL
        self.engine = create_engine(self.db_url)
        self.data_collector = DataCollector()
        self.signal_manager = SignalManager()  # <-- EKLENDİ
        
        # Sinyal takip parametreleri
        self.take_profit_threshold = 0.05  # %5 kar
        self.stop_loss_threshold = -0.03   # %3 zarar
        self.max_hold_hours = 8           # Maksimum 8 saat tut (daha önce 48 idi)
        
    def track_open_signals(self):
        self.logger.info("Açık sinyal takip fonksiyonu BAŞLADI")
        try:
            open_signals = self.get_open_signals()
            if not open_signals:
                self.logger.info("Takip edilecek açık sinyal yok")
                self.logger.info("Açık sinyal takip fonksiyonu BİTTİ")
                return
            self.logger.info(f"{len(open_signals)} açık sinyal takip ediliyor")
            for signal in open_signals:
                try:
                    # Tüm önemli alanları güvenli string olarak set et
                    for key in ['symbol', 'direction', 'timeframe']:
                        val = signal.get(key, '')
                        if val is None:
                            self.logger.warning(f"Sinyal {signal.get('id')} için {key} alanı None!")
                            signal[key] = ''
                        else:
                            signal[key] = str(val)
                    # entry_time ve timestamp güvenli kontrol
                    entry_time = signal.get('entry_time') or signal.get('timestamp')
                    if entry_time is None:
                        self.logger.error(f"Sinyal {signal.get('id')} için entry_time/timestamp alanı None! Sinyal atlanıyor.")
                        continue
                    if isinstance(entry_time, str):
                        try:
                            entry_time = datetime.strptime(entry_time, "%d.%m.%Y %H:%M")
                        except Exception:
                            try:
                                entry_time = pd.to_datetime(entry_time)
                            except Exception:
                                self.logger.error(f"Sinyal {signal.get('id')} için entry_time stringi çözümlenemedi! Sinyal atlanıyor.")
                                continue
                    if not hasattr(entry_time, 'replace'):
                        self.logger.error(f"Sinyal {signal.get('id')} için entry_time nesnesi hatalı! Sinyal atlanıyor.")
                        continue
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                    current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
                    duration = current_time - entry_time
                    duration_hours = duration.total_seconds() / 3600
                    should_exit = duration_hours >= 8
                    self.logger.info(f"Sinyal ID: {signal.get('id')}, entry_time: {entry_time}, now: {current_time}, duration_hours: {duration_hours:.2f}, should_exit: {should_exit}")
                    if should_exit:
                        self.logger.info(f"Kapanma KOŞULU SAĞLANDI: Sinyal ID {signal.get('id')}")
                    self.update_signal_result(signal)
                except Exception as e:
                    self.logger.error(f"Sinyal {signal.get('id')} için takip hatası: {e}")
            self.logger.info("Açık sinyal takip fonksiyonu BİTTİ")
        except Exception as e:
            self.logger.error(f"Açık sinyal takip fonksiyonu genel hata: {e}")
    
    def get_open_signals(self):
        """Açık sinyalleri al"""
        try:
            self.logger.info("[DEBUG] get_open_signals fonksiyonu çağrıldı.")
            query = """
                SELECT id, symbol, direction, entry_price, timestamp, 
                       ai_score, ta_strength, whale_score, confidence
                FROM signals 
                WHERE result IS NULL OR result = 'None' OR result = ''
                ORDER BY timestamp DESC
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                signals = [dict(row._mapping) for row in result.fetchall()]
            self.logger.info(f"[DEBUG] get_open_signals: {len(signals)} açık sinyal bulundu.")
            return signals
            
        except Exception as e:
            self.logger.error(f"Açık sinyaller alınamadı: {e}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")
            return []
    
    def update_signal_result(self, signal):
        """Sinyal sonucunu güncelle"""
        try:
            self.logger.info(f"[DEBUG] update_signal_result çağrıldı. Parametreler: {signal}")
            signal_id = signal['id']
            # Symbol ve direction güvenli string
            symbol = str(signal.get('symbol', '')) if signal.get('symbol', '') else ''
            entry_price = float(signal['entry_price'])
            entry_time = pd.to_datetime(signal['timestamp'])
            current_time = datetime.now()
            # Güncel fiyatı al
            current_price = self.get_current_price(symbol)
            if current_price is None:
                self.logger.warning(f"{symbol} güncel fiyat alınamadı")
                return
            # PnL hesapla (her direction için aynı: fiyat artışı/azalışı)
            pnl_percent = (current_price - entry_price) / entry_price
            # Süre hesapla
            duration_hours = (current_time - entry_time).total_seconds() / 3600
            # Çıkış koşullarını kontrol et
            should_exit, exit_reason = self.check_exit_conditions(
                pnl_percent, duration_hours, signal
            )
            self.logger.info(f"[DEBUG] update_signal_result: should_exit={should_exit}, exit_reason={exit_reason}, pnl_percent={pnl_percent}, duration_hours={duration_hours}")
            if should_exit:
                # Sinyali kapat
                result = 'profit' if pnl_percent > 0 else 'loss'
                realized_gain = pnl_percent
                self.close_signal(signal_id, result, realized_gain, current_price, exit_reason)
                self.logger.info(f"Sinyal kapatıldı: {symbol} - {result.upper()} "
                               f"({pnl_percent:.2%}) - {exit_reason}")
            else:
                # Sinyal hala açık, sadece güncel durumu kaydet
                self.update_signal_status(signal_id, current_price, pnl_percent)
                self.logger.info(f"[DEBUG] update_signal_result: Sinyal hala açık, güncel durum kaydedildi. signal_id={signal_id}")
        except Exception as e:
            self.logger.error(f"Sinyal {signal_id} sonuç güncelleme hatası: {e}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")
    
    def get_current_price(self, symbol):
        """Sembol için güncel fiyatı al"""
        try:
            symbol = str(symbol) if symbol else ''
            # 1 saatlik veri al (son fiyat için)
            data = self.data_collector.get_historical_data(symbol, '1h', 1)
            
            if data is not None and not data.empty:
                return float(data['close'].iloc[-1])
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"{symbol} fiyat alma hatası: {e}")
            return None
    
    def check_exit_conditions(self, pnl_percent, duration_hours, signal):
        """Çıkış koşullarını kontrol et (NEUTRAL dahil tüm sinyaller için geçerli)"""
        try:
            # 1. Take Profit kontrolü
            if pnl_percent >= self.take_profit_threshold:
                return True, f"Take Profit ({pnl_percent:.2%})"
            # 2. Stop Loss kontrolü
            if pnl_percent <= self.stop_loss_threshold:
                return True, f"Stop Loss ({pnl_percent:.2%})"
            # 3. Maksimum tutma süresi kontrolü
            if duration_hours >= self.max_hold_hours:
                return True, f"Max Hold Time ({duration_hours:.1f}h)"
            # 4. Dinamik stop loss (yüksek güvenli sinyaller için)
            confidence = signal.get('confidence', 0)
            if confidence > 0.8 and pnl_percent >= 0.02:  # %2 kar varsa
                return True, f"High Confidence Exit ({pnl_percent:.2%})"
            # 5. AI Score bazlı çıkış
            ai_score = signal.get('ai_score', 0)
            if ai_score < 0.3 and pnl_percent >= 0.01:  # Düşük AI skoru, erken çık
                return True, f"Low AI Score Exit ({pnl_percent:.2%})"
            return False, "Still Open"
        except Exception as e:
            self.logger.error(f"Çıkış koşulları kontrol hatası: {e}")
            return False, "Error"
    
    def close_signal(self, signal_id, result, realized_gain, exit_price, exit_reason):
        """Sinyali kapat"""
        try:
            self.logger.info(f"[DEBUG] close_signal çağrıldı. signal_id={signal_id}, result={result}, realized_gain={realized_gain}, exit_price={exit_price}, exit_reason={exit_reason}")
            current_time = datetime.now()
            
            # Sinyal bilgilerini al
            signal_query = "SELECT timestamp FROM signals WHERE id = :signal_id"
            with self.engine.connect() as conn:
                result_query = conn.execute(text(signal_query), {'signal_id': signal_id})
                signal_data = result_query.fetchone()
                
                if not signal_data:
                    self.logger.error(f"Sinyal {signal_id} bulunamadı")
                    return
                
                entry_time = pd.to_datetime(signal_data.timestamp)
                duration_hours = (current_time - entry_time).total_seconds() / 3600
                
                # Sinyali güncelle
                update_query = """
                    UPDATE signals 
                    SET result = :result,
                        realized_gain = :realized_gain,
                        exit_price = :exit_price,
                        exit_time = :exit_time,
                        duration_hours = :duration_hours,
                        exit_reason = :exit_reason,
                        updated_at = :updated_at
                    WHERE id = :signal_id
                """
                
                update_result = conn.execute(text(update_query), {
                    'result': result,
                    'realized_gain': realized_gain,
                    'exit_price': exit_price,
                    'exit_time': current_time,
                    'duration_hours': duration_hours,
                    'exit_reason': exit_reason,
                    'updated_at': current_time,
                    'signal_id': signal_id
                })
                conn.commit()
                self.logger.info(f"[DEBUG] close_signal: SQL güncellemesi yapıldı. rowcount={getattr(update_result, 'rowcount', 'N/A')}")
                self.logger.info(f"Sinyal kapatıldı: {signal_id}, exit_time: {current_time}, reason: {exit_reason}")
                # --- SignalManager ile hem veritabanı hem JSON güncelle ---
                self.signal_manager.update_signal_result(signal_id, result, realized_gain, duration_hours, exit_price)
        except Exception as e:
            self.logger.error(f"Sinyal {signal_id} kapatma hatası: {e}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")
    
    def update_signal_status(self, signal_id, current_price, pnl_percent):
        """Açık sinyalin durumunu güncelle"""
        try:
            update_query = """
                UPDATE signals 
                SET current_price = :current_price,
                    current_pnl = :current_pnl,
                    updated_at = :updated_at
                WHERE id = :signal_id
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(update_query), {
                    'current_price': current_price,
                    'current_pnl': pnl_percent,
                    'updated_at': datetime.now(),
                    'signal_id': signal_id
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Sinyal {signal_id} durum güncelleme hatası: {e}")
    
    def get_performance_summary(self):
        """Performans özeti al"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN result = 'profit' THEN 1 END) as profitable_signals,
                    COUNT(CASE WHEN result = 'loss' THEN 1 END) as loss_signals,
                    COUNT(CASE WHEN result IS NULL OR result = 'None' THEN 1 END) as open_signals,
                    AVG(CASE WHEN result = 'profit' THEN realized_gain END) as avg_profit,
                    AVG(CASE WHEN result = 'loss' THEN realized_gain END) as avg_loss,
                    AVG(realized_gain) as avg_total_return,
                    AVG(duration_hours) as avg_duration
                FROM signals 
                WHERE timestamp::timestamp >= NOW() - INTERVAL '7 days'
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
                if row:
                    data = dict(row._mapping)
                    
                    # Başarı oranı hesapla
                    total_closed = data['profitable_signals'] + data['loss_signals']
                    if total_closed > 0:
                        success_rate = (data['profitable_signals'] / total_closed) * 100
                    else:
                        success_rate = 0
                    
                    data['success_rate'] = success_rate
                    return data
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Performans özeti alınamadı: {e}")
            return {}
    
    def cleanup_old_signals(self, days=7):
        """Eski sinyalleri temizle"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 7 günden eski açık sinyalleri kapat
            cleanup_query = """
                UPDATE signals 
                SET result = 'expired',
                    realized_gain = 0,
                    exit_reason = 'Auto cleanup - expired',
                    updated_at = :updated_at
                WHERE (result IS NULL OR result = 'None' OR result = '')
                AND timestamp::timestamp < :cutoff_date
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(cleanup_query), {
                    'cutoff_date': cutoff_date,
                    'updated_at': datetime.now()
                })
                conn.commit()
                
                cleaned_count = result.rowcount
                if cleaned_count > 0:
                    self.logger.info(f"{cleaned_count} eski sinyal temizlendi")
                    
        except Exception as e:
            self.logger.error(f"Eski sinyal temizleme hatası: {e}")
    
    def run_daily_cleanup(self):
        """Günlük temizlik işlemleri"""
        try:
            # Eski sinyalleri temizle
            self.cleanup_old_signals(days=7)
            
            # Performans özeti logla
            performance = self.get_performance_summary()
            if performance:
                self.logger.info(f"Günlük performans: {performance.get('success_rate', 0):.1f}% "
                               f"başarı oranı, {performance.get('total_signals', 0)} toplam sinyal")
                
        except Exception as e:
            self.logger.error(f"Günlük temizlik hatası: {e}")
