import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging
from config import Config
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from modules.whale_tracker import get_whale_score, detect_whale_trades
# from modules.news_analyzer import NewsAnalyzer
from improve_ai_models_simple import get_dynamic_threshold
import glob

class SignalManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.signals_dir = Config.SIGNALS_DIR
        os.makedirs(self.signals_dir, exist_ok=True)
        self.db_url = Config.DATABASE_URL
        self.engine = create_engine(self.db_url)
        self._create_tables()

    def _create_tables(self):
        """PostgreSQL tablolarını oluştur"""
        try:
            # PostgreSQL için SERIAL kullan
            create_table_sql = """
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
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            self.logger.info("PostgreSQL veritabanı tabloları oluşturuldu.")
        except Exception as e:
            self.logger.error(f"Tablo oluşturulurken hata: {e}")

    def save_signal_json(self, signal):
        try:
            dt = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol_val = signal.get('symbol', '')
            safe_symbol = str(symbol_val).replace('/', '_') if symbol_val else ''
            for key in ['symbol', 'direction', 'timeframe']:
                val = signal.get(key, '')
                if val is None:
                    self.logger.warning(f"Sinyal {signal.get('id', 'bilinmiyor')} için {key} alanı None!")
                    signal[key] = ''
                else:
                    signal[key] = str(val)
            # Bid/Ask duvarlarını JSON string olarak kaydet
            for wall_key in ['top_bid_walls', 'top_ask_walls']:
                val = signal.get(wall_key, [])
                if isinstance(val, str):
                    try:
                        # Eğer zaten JSON string ise, parse et ve tekrar dump et
                        arr = json.loads(val)
                        signal[wall_key] = json.dumps(arr, ensure_ascii=False)
                    except Exception:
                        # Eğer virgüllü string ise, split ile diziye çevir
                        try:
                            arr = []
                            for item in val.split(','):
                                if ':' in item:
                                    price, amount = item.split(':')
                                    arr.append({'price': float(price), 'amount': float(amount)})
                            signal[wall_key] = json.dumps(arr, ensure_ascii=False)
                        except Exception:
                            signal[wall_key] = json.dumps([], ensure_ascii=False)
                elif isinstance(val, list):
                    signal[wall_key] = json.dumps(val, ensure_ascii=False)
                else:
                    signal[wall_key] = json.dumps([], ensure_ascii=False)
            filename = os.path.join(self.signals_dir, f"signal_{safe_symbol}_{dt}.json")
            safe_signal = {}
            for key, value in signal.items():
                if value is None:
                    safe_signal[key] = ""
                elif isinstance(value, (int, float)):
                    safe_signal[key] = value
                else:
                    safe_signal[key] = str(value)
            safe_signal['symbol'] = str(signal.get('symbol', '')) if signal.get('symbol', '') else ''
            safe_signal['direction'] = str(signal.get('direction', '')) if signal.get('direction', '') else ''
            safe_signal['timeframe'] = str(signal.get('timeframe', '')) if signal.get('timeframe', '') else ''
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(safe_signal, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"Sinyal JSON kaydedildi: {filename}")
        except Exception as e:
            self.logger.error(f"Sinyal JSON kaydedilemedi: {e}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")

    def save_signal_csv(self, signal):
        try:
            dt = datetime.now().strftime('%Y%m%d')
            # Tüm önemli alanları güvenli string olarak set et
            for key in ['symbol', 'direction', 'timeframe']:
                val = signal.get(key, '')
                if val is None:
                    self.logger.warning(f"Sinyal {signal.get('id', 'bilinmiyor')} için {key} alanı None!")
                    signal[key] = ''
                else:
                    signal[key] = str(val)
            filename = os.path.join(self.signals_dir, f"signals_{dt}.csv")
            # Signal verisini güvenli hale getir
            safe_signal = {}
            for key, value in signal.items():
                if value is None:
                    safe_signal[key] = ""
                elif isinstance(value, (int, float)):
                    safe_signal[key] = value
                else:
                    safe_signal[key] = str(value)
            # Ayrıca symbol, direction, timeframe gibi alanları güvenli ata
            safe_signal['symbol'] = str(signal.get('symbol', '')) if signal.get('symbol', '') else ''
            safe_signal['direction'] = str(signal.get('direction', '')) if signal.get('direction', '') else ''
            safe_signal['timeframe'] = str(signal.get('timeframe', '')) if signal.get('timeframe', '') else ''
            df = pd.DataFrame([safe_signal])
            if os.path.exists(filename):
                df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8')
            else:
                df.to_csv(filename, index=False, encoding='utf-8')
            self.logger.info(f"Sinyal CSV kaydedildi: {filename}")
        except Exception as e:
            self.logger.error(f"Sinyal CSV kaydedilemedi: {e}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")

    def save_signal_db(self, signal):
        """Sinyali PostgreSQL veritabanına kaydet - Hataları detaylı logla ve eksik/yanlış alanları kontrol et"""
        try:
            # Gerekli alanları kontrol et
            required_fields = [
                'symbol', 'timeframe', 'direction', 'ai_score', 'ta_strength', 'whale_score',
                'social_score', 'news_score', 'timestamp', 'entry_price', 'current_price'
            ]
            for field in required_fields:
                if field not in signal or signal[field] in [None, '', [], {}]:
                    self.logger.error(f"[DB KAYIT HATASI] Eksik veya boş alan: {field} | Sinyal: {signal}")
                    raise ValueError(f"Eksik veya boş alan: {field}")

            # Tüm önemli alanları güvenli string olarak set et
            for key in ['symbol', 'direction', 'timeframe']:
                val = signal.get(key, '')
                if val is None:
                    self.logger.warning(f"Sinyal {signal.get('id', 'bilinmiyor')} için {key} alanı None!")
                    signal[key] = ''
                else:
                    signal[key] = str(val)
            self.logger.info(f"Veritabanına kayıt başlatılıyor: {signal.get('symbol', 'Unknown')}")
            if not self.engine:
                self.logger.error("Veritabanı engine'i bulunamadı!")
                raise RuntimeError("Veritabanı engine'i bulunamadı!")

            # Sinyal verilerini hazırla
            entry_price = signal.get('entry_price', 0.0)
            current_price = signal.get('current_price', entry_price)

            # Hedef fiyatları hesapla
            take_profit, stop_loss, support_level, resistance_level, target_time = self.calculate_target_levels(
                signal, entry_price
            )

            # Success metrics'i güvenli şekilde al
            success_metrics = signal.get('success_metrics', {})
            self.logger.info(f"Success metrics: {success_metrics}")

            # SQL sorgusunu hazırla
            query = """
                INSERT INTO signals (
                    symbol, timeframe, direction, ai_score, ta_strength, whale_score,        
                    social_score, news_score, timestamp, predicted_gain, predicted_duration, 
                    entry_price, current_price, quality_score, market_regime, volatility_regime,
                    volume_score, momentum_score, pattern_score, breakout_probability,       
                    risk_reward_ratio, confidence_level, signal_strength, market_sentiment,
                    take_profit, stop_loss, support_level, resistance_level, target_time_hours,
                    predicted_breakout_threshold, predicted_breakout_time_hours,
                    order_book_imbalance, top_bid_walls, top_ask_walls, whale_direction_score
                ) VALUES (
                    :symbol, :timeframe, :direction, :ai_score, :ta_strength, 
                    :whale_score, :social_score, :news_score, :timestamp,        
                    :predicted_gain, :predicted_duration, :entry_price, :current_price,
                    :quality_score, :market_regime, :volatility_regime, :volume_score,
                    :momentum_score, :pattern_score, :breakout_probability,
                    :risk_reward_ratio, :confidence_level, :signal_strength, :market_sentiment,
                    :take_profit, :stop_loss, :support_level, :resistance_level, :target_time_hours,
                    :predicted_breakout_threshold, :predicted_breakout_time_hours,
                    :order_book_imbalance, :top_bid_walls, :top_ask_walls, :whale_direction_score
                )
            """

            def safe_numeric(value, min_val=-999.99, max_val=999.99, default=0.0):
                try:
                    if value is None or pd.isna(value):
                        return default
                    num_val = float(value)
                    return max(min_val, min(max_val, num_val))
                except (ValueError, TypeError):
                    return default

            breakout_threshold = signal.get('predicted_breakout_threshold')
            if breakout_threshold is None:
                breakout_threshold = self.calculate_breakout_threshold(signal)
            breakout_time = signal.get('predicted_breakout_time_hours')
            if breakout_time is None:
                breakout_time = self.predict_breakout_time(signal)

            params = {
                'symbol': signal.get('symbol', 'Unknown'),
                'timeframe': signal.get('timeframe', '1h'),
                'direction': signal.get('direction', 'LONG'),
                'ai_score': safe_numeric(signal.get('ai_score', 0.0), 0.0, 1.0, 0.0),
                'ta_strength': safe_numeric(signal.get('ta_strength', 0.0), 0.0, 1.0, 0.0),
                'whale_score': safe_numeric(signal.get('whale_score', 0.0), 0.0, 1.0, 0.0),
                'social_score': safe_numeric(signal.get('social_score', 0.0), 0.0, 1.0, 0.0),
                'news_score': safe_numeric(signal.get('news_score', 0.0), 0.0, 1.0, 0.0),
                'timestamp': signal.get('timestamp', datetime.now()),
                'predicted_gain': safe_numeric(signal.get('predicted_gain', 0.0), -100.0, 1000.0, 0.0),
                'predicted_duration': signal.get('predicted_duration', '4-8 saat'),
                'entry_price': safe_numeric(entry_price, 0.0, 1000000.0, 0.0),
                'current_price': safe_numeric(current_price, 0.0, 1000000.0, 0.0),
                'quality_score': safe_numeric(success_metrics.get('quality_score', 0.0), 0.0, 1.0, 0.0),
                'market_regime': signal.get('market_regime', 'NORMAL'),
                'volatility_regime': signal.get('volatility_regime', 'normal'),
                'volume_score': safe_numeric(signal.get('volume_score', 0.0), 0.0, 1.0, 0.0),
                'momentum_score': safe_numeric(signal.get('momentum_score', 0.0), 0.0, 1.0, 0.0),
                'pattern_score': safe_numeric(signal.get('pattern_score', 0.0), 0.0, 1.0, 0.0),
                'breakout_probability': safe_numeric(signal.get('breakout_probability', 0.0), 0.0, 1.0, 0.0),
                'risk_reward_ratio': safe_numeric(signal.get('risk_reward_ratio', 1.67), 0.1, 10.0, 1.67),
                'confidence_level': safe_numeric(signal.get('confidence_level', signal.get('confidence', 0.0)), 0.0, 1.0, 0.0),
                'signal_strength': safe_numeric(signal.get('signal_strength', 0.0), 0.0, 1.0, 0.0),
                'market_sentiment': safe_numeric(signal.get('market_sentiment', 0.0), 0.0, 1.0, 0.0),
                'take_profit': safe_numeric(take_profit, 0.0, 1000000.0, 0.0),
                'stop_loss': safe_numeric(stop_loss, 0.0, 1000000.0, 0.0),
                'support_level': safe_numeric(support_level, 0.0, 1000000.0, 0.0),
                'resistance_level': safe_numeric(resistance_level, 0.0, 1000000.0, 0.0),
                'target_time_hours': safe_numeric(target_time, 1.0, 168.0, 24.0),
                'predicted_breakout_threshold': safe_numeric(breakout_threshold, 0.0, 1.0, 0.025),
                'predicted_breakout_time_hours': safe_numeric(breakout_time, 0.5, 168.0, 24.0),
                'order_book_imbalance': safe_numeric(signal.get('order_book_imbalance', 0.0), -1.0, 1.0, 0.0),
                'top_bid_walls': str(signal.get('top_bid_walls', '[]')),
                'top_ask_walls': str(signal.get('top_ask_walls', '[]')),
                'whale_direction_score': safe_numeric(signal.get('whale_direction_score', 0.0), -1.0, 1.0, 0.0),
            }

            self.logger.info(f"SQL sorgusu hazırlandı. Parametre sayısı: {len(params)}")
            self.logger.info(f"Parametreler: {params}")

            # Veritabanı bağlantısını aç ve sorguyu çalıştır
            with self.engine.connect() as conn:
                self.logger.info("Veritabanı bağlantısı açıldı")
                try:
                    result = conn.execute(text(query), params)
                except Exception as sql_exc:
                    self.logger.error(f"[DB KAYIT HATASI] SQL sorgusu başarısız! Hata: {sql_exc}\nParametreler: {params}\nSinyal: {signal}")
                    raise
                self.logger.info(f"Sorgu çalıştırıldı. Etkilenen satır sayısı: {result.rowcount}")
                conn.commit()
                self.logger.info("Değişiklikler kaydedildi")

            self.logger.info(f"Sinyal PostgreSQL veritabanına başarıyla kaydedildi. Symbol: {signal.get('symbol', 'Unknown')}, Fiyat: {entry_price}")

        except Exception as e:
            self.logger.error(f"[DB KAYIT HATASI] Sinyal DB kaydedilemedi: {e}\nSinyal: {signal}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")
            raise  # Hatanın üst katmana çıkmasını sağla

    def load_signals(self, start_date=None, end_date=None):
        try:
            query = "SELECT * FROM signals"
            params = {}
            if start_date and end_date:
                query += " WHERE timestamp::timestamp >= %(start_date)s AND timestamp::timestamp <= %(end_date)s"
                params = {'start_date': start_date, 'end_date': end_date}
            query += " ORDER BY timestamp DESC"
            df = pd.read_sql(query, self.engine, params=params)
            return df
        except Exception as e:
            self.logger.error(f"Sinyaller yüklenemedi: {e}")
            return pd.DataFrame()

    def get_latest_signals(self, limit=20, offset=0):
        try:
            query = f"""
                SELECT * FROM signals 
                ORDER BY timestamp DESC 
                LIMIT {limit} OFFSET {offset}
            """
            df = pd.read_sql(query, self.engine, coerce_float=True)
            
            # UTF-8 encoding sorununu çöz
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            
            return df
        except Exception as e:
            self.logger.error(f"Son sinyaller alınamadı: {e}")
            return pd.DataFrame()

    def get_total_signal_count(self):
        """Toplam sinyal sayısını al"""
        try:
            query = "SELECT COUNT(*) as count FROM signals"
            result = pd.read_sql(query, self.engine)
            return result['count'].iloc[0] if not result.empty else 0
        except Exception as e:
            self.logger.error(f"Toplam sinyal sayısı alınamadı: {e}")
            return 0

    def get_open_signal_count(self):
        """Açık sinyal sayısını al"""
        try:
            query = "SELECT COUNT(*) as count FROM signals WHERE result IS NULL OR result = 'None'"
            result = pd.read_sql(query, self.engine)
            return result['count'].iloc[0] if not result.empty else 0
        except Exception as e:
            self.logger.error(f"Açık sinyal sayısı alınamadı: {e}")
            return 0

    def get_closed_signal_count(self):
        """Kapalı sinyal sayısını al"""
        try:
            query = "SELECT COUNT(*) as count FROM signals WHERE result IS NOT NULL AND result != 'None'"
            result = pd.read_sql(query, self.engine)
            return result['count'].iloc[0] if not result.empty else 0
        except Exception as e:
            self.logger.error(f"Kapalı sinyal sayısı alınamadı: {e}")
            return 0

    def get_success_rate(self):
        """Başarı oranını al"""
        try:
            query = """
            SELECT 
                COUNT(CASE WHEN realized_gain > 0 THEN 1 END) as successful,
                COUNT(*) as total
            FROM signals 
            WHERE result IS NOT NULL AND result != 'None'
            """
            result = pd.read_sql(query, self.engine)
            if not result.empty and result['total'].iloc[0] > 0:
                success_rate = (result['successful'].iloc[0] / result['total'].iloc[0]) * 100
                return round(success_rate, 2)
            return 0.0
        except Exception as e:
            self.logger.error(f"Başarı oranı alınamadı: {e}")
            return 0.0

    def get_average_profit(self):
        """Ortalama kazanç hesapla"""
        try:
            query = """
                SELECT AVG(realized_gain) as avg_profit 
                FROM signals 
                WHERE result IS NOT NULL AND result != 'None' AND result != 'TIMEOUT'
            """
            result = pd.read_sql(query, self.engine)
            return result['avg_profit'].iloc[0] if not result.empty and not pd.isna(result['avg_profit'].iloc[0]) else 0.0
        except Exception as e:
            self.logger.error(f"Ortalama kazanç hesaplanamadı: {e}")
            return 0.0

    def get_signal_by_id(self, signal_id):
        """ID'ye göre sinyal getir"""
        try:
            query = f"SELECT * FROM signals WHERE id = {signal_id}"
            df = pd.read_sql(query, self.engine, coerce_float=True)
            
            if df.empty:
                return None
            
            # UTF-8 encoding sorununu çöz
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            
            return df.iloc[0]
        except Exception as e:
            self.logger.error(f"Sinyal ID {signal_id} ile alınamadı: {e}")
            return None

    def signal_exists(self, symbol, timeframe, direction, min_hours=None):
        """Aynı coin, timeframe ve direction için son min_hours saat içinde sinyal var mı?"""
        try:
            # Config'den varsayılan süreyi al
            if min_hours is None:
                from config import Config
                min_hours = Config.SIGNAL_EXPIRY_HOURS
            
            dt_limit = datetime.now() - pd.Timedelta(hours=min_hours)
            query = """
                SELECT COUNT(*) FROM signals 
                WHERE symbol = :symbol 
                AND timeframe = :timeframe 
                AND direction = :direction
                AND timestamp >= :dt_limit
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'dt_limit': str(dt_limit)
                })
                count = result.scalar()
                return count > 0
        except Exception as e:
            self.logger.error(f"Sinyal varlığı kontrol edilemedi: {e}")
            return False

    def update_signal_result(self, signal_id, result, realized_gain=None, duration=None, exit_price=None):
        """Sinyal sonucunu güncelle"""
        try:
            self.logger.info(f"[DEBUG] update_signal_result çağrıldı. signal_id={signal_id}, result={result}, realized_gain={realized_gain}, duration={duration}, exit_price={exit_price}")
            query = """
                UPDATE signals 
                SET result = :result, 
                    realized_gain = :realized_gain, 
                    duration = :duration,
                    exit_price = :exit_price
                WHERE id = :signal_id
            """
            with self.engine.connect() as conn:
                conn.execute(text(query), {
                    'result': result,
                    'realized_gain': realized_gain,
                    'duration': duration,
                    'exit_price': exit_price,
                    'signal_id': signal_id
                })
                conn.commit()
            self.logger.info(f"Sinyal {signal_id} sonucu güncellendi: {result}")
            # --- JSON dosyasını da güncelle ---
            self.update_signal_json_result(signal_id, result)
        except Exception as e:
            self.logger.error(f"Sinyal sonucu güncellenemedi: {e}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")

    def update_signal_json_result(self, signal_id, result):
        """Sinyalin JSON dosyasına result alanı ekle/güncelle"""
        try:
            # Sinyal dosyasını bul (id genellikle dosya adında veya içinde olabilir)
            pattern = os.path.join(self.signals_dir, f"signal_*_{signal_id}.json")
            files = glob.glob(pattern)
            if not files:
                # Alternatif: id dosya adında yoksa, tüm dosyaları tara
                for f in glob.glob(os.path.join(self.signals_dir, "signal_*.json")):
                    with open(f, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                        if str(data.get('id', '')) == str(signal_id):
                            files = [f]
                            break
            if not files:
                self.logger.warning(f"JSON dosyası bulunamadı: {signal_id}")
                return
            json_file = files[0]
            with open(json_file, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            data['result'] = result
            with open(json_file, 'w', encoding='utf-8') as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
            self.logger.info(f"JSON dosyasında result güncellendi: {json_file} -> {result}")
        except Exception as e:
            self.logger.error(f"JSON dosyasında result güncellenemedi: {e}")

    def get_performance_stats(self, days=30):
        """Performans istatistiklerini al"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN result = 'profit' THEN 1 END) as profitable_signals,
                    COUNT(CASE WHEN result = 'loss' THEN 1 END) as loss_signals,
                    AVG(CASE WHEN result = 'profit' THEN realized_gain END) as avg_profit,
                    AVG(CASE WHEN result = 'loss' THEN realized_gain END) as avg_loss,
                    AVG(realized_gain) as avg_total_return
                FROM signals 
                WHERE timestamp::timestamp >= NOW() - INTERVAL '%s days'
                AND result IS NOT NULL
            """ % days
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                else:
                    return {}
        except Exception as e:
            self.logger.error(f"Performans istatistikleri alınamadı: {e}")
            return {}

    def calculate_signal_score(self, signal_data):
        """Gelişmiş sinyal skorlaması (0-1 arası)"""
        try:
            symbol = signal_data.get('symbol', 'Unknown')
            weights = {
                'ai_score': 0.5,
                'ta_strength': 0.35,
                'whale_score': 0.15
            }
            total_score = 0
            total_weight = 0
            self.logger.info(f"   📊 {symbol} skor hesaplama:")
            for key, weight in weights.items():
                if key in signal_data and signal_data[key] is not None:
                    raw_score = signal_data[key]
                    normalized_score = self._normalize_score(raw_score)
                    weighted_score = normalized_score * weight
                    total_score += weighted_score
                    total_weight += weight
                    self.logger.info(f"      {key}: {raw_score:.4f} -> {normalized_score:.4f} * {weight} = {weighted_score:.4f}")
                else:
                    self.logger.error(f"      {key}: Eksik veya None, sinyal üretimi iptal edildi.")
                    return None
            if total_weight > 0:
                final_score = total_score / total_weight
                self.logger.info(f"      Toplam: {total_score:.4f} / {total_weight} = {final_score:.4f}")
            else:
                self.logger.error(f"      Toplam ağırlık 0, skor hesaplanamadı. Sinyal üretimi iptal edildi.")
                return None
            direction = signal_data.get('direction', 'NEUTRAL')
            original_score = final_score
            if direction == 'LONG':
                final_score = min(1.0, final_score * 1.1)
                self.logger.info(f"      LONG yönü: {original_score:.4f} * 1.1 = {final_score:.4f}")
            elif direction == 'SHORT':
                final_score = min(1.0, final_score * 1.1)
                self.logger.info(f"      SHORT yönü: {original_score:.4f} * 1.1 = {final_score:.4f}")
            else:
                final_score = final_score * 0.9
                self.logger.info(f"      NEUTRAL yönü: {original_score:.4f} * 0.9 = {final_score:.4f}")
            final_score = round(final_score, 4)
            self.logger.info(f"   🎯 {symbol} Final Score: {final_score:.4f}")
            return final_score
        except Exception as e:
            self.logger.error(f"Sinyal skorlama hatası: {e}")
            return 0.5

    def determine_signal_direction(self, signal_data):
        """Sinyal yönünü belirle (LONG/SHORT/NEUTRAL) - DÜZELTİLMİŞ"""
        try:
            # AI skorları - Ana kriter
            if 'ai_score' not in signal_data or signal_data['ai_score'] is None:
                self.logger.error("AI skoru eksik, sinyal yönü belirlenemedi. Sinyal üretimi iptal edildi.")
                return None
            ai_score = signal_data['ai_score']
            # Teknik analiz göstergeleri
            rsi = signal_data.get('rsi_14')
            macd = signal_data.get('macd')
            macd_signal = signal_data.get('macd_signal')
            bb_position = signal_data.get('bb_position')
            long_score = 0
            short_score = 0
            # AI skoru analizi - Ana kriter
            if ai_score > 0.65:
                long_score += 0.5
            elif ai_score > 0.55:
                long_score += 0.3
            elif ai_score < 0.35:
                short_score += 0.5
            elif ai_score < 0.45:
                short_score += 0.3
            # RSI analizi - Destekleyici
            if rsi is not None:
                if rsi < 30:
                    long_score += 0.2
                elif rsi < 40:
                    long_score += 0.1
                elif rsi > 70:
                    short_score += 0.2
                elif rsi > 60:
                    short_score += 0.1
            # MACD analizi - Destekleyici
            if macd is not None and macd_signal is not None:
                if macd > macd_signal and macd > 0:
                    long_score += 0.2
                elif macd < macd_signal and macd < 0:
                    short_score += 0.2
            # Bollinger Band analizi - Destekleyici
            if bb_position is not None:
                if bb_position < 0.2:
                    long_score += 0.2
                elif bb_position > 0.8:
                    short_score += 0.2
            if long_score > short_score:
                return 'LONG'
            elif short_score > long_score:
                return 'SHORT'
            else:
                return 'NEUTRAL'
        except Exception as e:
            self.logger.error(f"Sinyal yönü belirleme hatası: {e}")
            return None

    def filter_signals(self, signals, min_confidence=None, min_ai_score=None, min_ta_strength=None, min_volatility=None, min_success_rate=None, max_signals=5, recent_success_rate=0.5):
        """
        Fırsat skoru ile filtrele: AI skoru, confidence, TA strength, volatilite ve geçmiş başarı oranı birleştirilir.
        Sadece yüksek fırsat skoru olanlar geçer.
        """
        try:
            if not signals:
                self.logger.info("Filtreleme: Hiç sinyal yok")
                return []
            filtered_signals = []
            for i, signal in enumerate(signals):
                symbol = signal.get('symbol', 'Unknown')
                ai_score = signal.get('ai_score', 0)
                confidence = signal.get('confidence', 0.5)
                ta_strength = signal.get('ta_strength', 0.5)
                volatility = signal.get('volatility', 0.5)
                # Minimum AI ve TA kontrolü
                if min_ai_score is not None and ai_score < min_ai_score:
                    self.logger.info(f"{symbol}: AI skoru düşük ({ai_score:.3f} < {min_ai_score}) - Filtrelenmedi")
                    continue
                if min_ta_strength is not None and ta_strength < min_ta_strength:
                    self.logger.info(f"{symbol}: TA skoru düşük ({ta_strength:.3f} < {min_ta_strength}) - Filtrelenmedi")
                    continue
                # Fırsat skoru hesapla (daha seçici ağırlıklar)
                opportunity_score = (
                    0.5 * ai_score +
                    0.2 * confidence +
                    0.2 * ta_strength +
                    0.1 * volatility +
                    0.1 * recent_success_rate
                )
                min_opportunity_score = min_confidence if min_confidence is not None else 0.65
                if opportunity_score >= min_opportunity_score:
                    filtered_signals.append(signal)
                    self.logger.info(f"{symbol}: Fırsat skoru geçti ({opportunity_score:.3f} >= {min_opportunity_score})")
                else:
                    self.logger.info(f"{symbol}: Fırsat skoru düşük ({opportunity_score:.3f} < {min_opportunity_score})")
            filtered_signals.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
            final_signals = filtered_signals[:max_signals]
            self.logger.info(f"Final sonuç: {len(final_signals)} sinyal (max: {max_signals})")
            return final_signals
        except Exception as e:
            self.logger.error(f"Sinyal filtreleme hatası: {e}")
            return []

    def validate_signal(self, signal_data):
        """Sinyal geçerliliğini kontrol et - Düzeltilmiş"""
        try:
            required_fields = ['symbol', 'timeframe', 'direction', 'ai_score', 'ta_strength']
            
            # Gerekli alanları kontrol et
            for field in required_fields:
                if field not in signal_data or signal_data[field] is None:
                    return False, f"Eksik alan: {field}"
            
            # Skor değerlerini kontrol et
            score_fields = ['ai_score', 'ta_strength', 'whale_score', 'social_score', 'news_score']
            for field in score_fields:
                if field in signal_data and signal_data[field] is not None:
                    if not (0 <= signal_data[field] <= 1):
                        return False, f"Geçersiz skor: {field} = {signal_data[field]}"
            
            # Yön kontrolü - LONG/SHORT/NEUTRAL
            if signal_data['direction'] not in ['LONG', 'SHORT', 'NEUTRAL']:
                return False, f"Geçersiz yön: {signal_data['direction']}"
            
            # Timeframe kontrolü
            valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            if signal_data['timeframe'] not in valid_timeframes:
                return False, f"Geçersiz timeframe: {signal_data['timeframe']}"
            
            return True, "Sinyal geçerli"
            
        except Exception as e:
            self.logger.error(f"Sinyal doğrulama hatası: {e}")
            return False, f"Doğrulama hatası: {e}"

    def get_signal_summary(self, days=7):
        """Sinyal özeti raporu"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.load_signals(start_date, end_date)
            if df.empty:
                return {
                    'total_signals': 0,
                    'avg_score': 0,
                    'success_rate': 0,
                    'top_coins': [],
                    'best_timeframes': []
                }
            
            # Genel istatistikler
            total_signals = len(df)
            avg_score = df['ai_score'].mean() if 'ai_score' in df.columns else 0
            
            # Başarı oranı (tamamlanmış sinyaller için)
            completed_signals = df[df['result'].notna()]
            success_rate = 0
            if not completed_signals.empty:
                success_rate = (completed_signals['result'] == 'profit').mean() * 100
            
            # En iyi coinler
            top_coins = df.groupby('symbol')['ai_score'].mean().sort_values(ascending=False).head(5).to_dict()
            
            # En iyi timeframeler
            best_timeframes = df.groupby('timeframe')['ai_score'].mean().sort_values(ascending=False).to_dict()
            
            return {
                'total_signals': total_signals,
                'avg_score': round(avg_score, 3),
                'success_rate': round(success_rate, 2),
                'top_coins': top_coins,
                'best_timeframes': best_timeframes
            }
            
        except Exception as e:
            self.logger.error(f"Sinyal özeti hatası: {e}")
            return {
                'total_signals': 0,
                'avg_score': 0,
                'success_rate': 0,
                'top_coins': [],
                'best_timeframes': []
            }

    def _normalize_score(self, score):
        """Skoru 0-1 arasına normalize et"""
        try:
            if score is None:
                return 0.5
            elif isinstance(score, str):
                # String skorları için
                if score.upper() in ['LONG', 'BUY']:
                    return 0.8
                elif score.upper() in ['SHORT', 'SELL']:
                    return 0.2
                else:
                    return 0.5
            else:
                # Sayısal skorlar için
                return max(0, min(1, float(score)))
        except:
            return 0.5

    def check_signal_status(self, signal_data, current_price=None):
        """Sinyal durumunu kontrol et ve dinamik hedefler kullan"""
        try:
            # Dinamik hedefleri hesapla
            prediction = self.predict_breakout_probability(signal_data)
            profit_target = prediction['profit_target']
            loss_target = profit_target * 0.6  # Zarar hedefi kar hedefinin %60'ı
            
            entry_price = signal_data.get('entry_price')
            signal_type = signal_data.get('signal_type', 'BUY')
            
            if not entry_price or not current_price:
                return signal_data
            
            # Dinamik hedeflerle kontrol
            if signal_type == 'BUY':
                # Kar hedefi kontrolü
                if current_price >= entry_price * (1 + profit_target):
                    signal_data['result'] = 'PROFIT'
                    signal_data['realized_gain'] = profit_target * 100
                    signal_data['close_reason'] = f'Dinamik kar hedefi %{profit_target*100:.1f}'
                
                # Zarar hedefi kontrolü
                elif current_price <= entry_price * (1 - loss_target):
                    signal_data['result'] = 'LOSS'
                    signal_data['realized_gain'] = -loss_target * 100
                    signal_data['close_reason'] = f'Dinamik zarar hedefi %{loss_target*100:.1f}'
                
                # Ani yükseliş kontrolü
                elif current_price >= entry_price * (1 + prediction['breakout_threshold']):
                    signal_data['breakout_detected'] = True
                    signal_data['breakout_probability'] = prediction['probability']
                    signal_data['breakout_category'] = prediction['category']
            
            elif signal_type == 'SELL':
                # Kar hedefi kontrolü (fiyat düşüşü)
                if current_price <= entry_price * (1 - profit_target):
                    signal_data['result'] = 'PROFIT'
                    signal_data['realized_gain'] = profit_target * 100
                    signal_data['close_reason'] = f'Dinamik kar hedefi %{profit_target*100:.1f}'
                
                # Zarar hedefi kontrolü (fiyat yükselişi)
                elif current_price >= entry_price * (1 + loss_target):
                    signal_data['result'] = 'LOSS'
                    signal_data['realized_gain'] = -loss_target * 100
                    signal_data['close_reason'] = f'Dinamik zarar hedefi %{loss_target*100:.1f}'
                
                # Ani düşüş kontrolü
                elif current_price <= entry_price * (1 - prediction['breakout_threshold']):
                    signal_data['breakout_detected'] = True
                    signal_data['breakout_probability'] = prediction['probability']
                    signal_data['breakout_category'] = prediction['category']
            
            # Tahmin bilgilerini ekle
            signal_data['predicted_profit_target'] = profit_target
            signal_data['predicted_loss_target'] = loss_target
            signal_data['breakout_threshold'] = prediction['breakout_threshold']
            signal_data['breakout_probability'] = prediction['probability']
            signal_data['confidence_level'] = prediction['confidence']
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Sinyal durumu kontrol edilemedi: {e}")
            return signal_data

    def get_open_signals(self):
        """Açık sinyalleri getir"""
        try:
            self.logger.info("[DEBUG] get_open_signals fonksiyonu çağrıldı.")
            query = """
                SELECT * FROM signals 
                WHERE result IS NULL 
                ORDER BY timestamp::timestamp DESC
            """
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"[DEBUG] get_open_signals: {len(df)} açık sinyal bulundu.")
            return df
        except Exception as e:
            self.logger.error(f"Açık sinyaller alınamadı: {e}")
            import traceback
            self.logger.error(f"Hata detayı: {traceback.format_exc()}")
            return pd.DataFrame()

    def get_closed_signals(self, days=30):
        """Kapalı sinyalleri getir"""
        try:
            query = """
                SELECT * FROM signals 
                WHERE result IS NOT NULL 
                AND timestamp::timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp::timestamp DESC
            """ % days
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Kapalı sinyaller alınamadı: {e}")
            return pd.DataFrame()

    def update_expired_signals(self):
        """Süresi dolmuş sinyalleri otomatik olarak kapat"""
        try:
            # Config'den expiry süresini al
            from config import Config
            expiry_hours = Config.SIGNAL_EXPIRY_HOURS
            
            query = f"""
                UPDATE signals 
                SET result = 'TIMEOUT', 
                    realized_gain = 0.0,
                    duration = EXTRACT(EPOCH FROM (NOW() - timestamp::timestamp)) / 3600
                WHERE result IS NULL 
                AND timestamp::timestamp < NOW() - INTERVAL '{expiry_hours} hours'
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                conn.commit()
                updated_count = result.rowcount
                if updated_count > 0:
                    self.logger.info(f"{updated_count} adet süresi dolmuş sinyal kapatıldı ({expiry_hours} saat sonra)")
                return updated_count
        except Exception as e:
            self.logger.error(f"Süresi dolmuş sinyaller güncellenemedi: {e}")
            return 0

    def calculate_dynamic_profit_target(self, signal_data):
        """Dinamik kar hedefi hesapla - Yön bazlı"""
        try:
            # Temel faktörler
            direction = signal_data.get('direction', 'LONG')
            ai_score = signal_data.get('ai_score', 0.5)
            ta_strength = signal_data.get('ta_strength', 0.5)
            symbol = signal_data.get('symbol', '')
            timeframe = signal_data.get('timeframe', '1h')
            entry_price = signal_data.get('entry_price', 0)
            
            if entry_price <= 0:
                return 0.05  # Varsayılan %5
            
            # 1. AI Skoruna Göre Kar Hedefi
            ai_based_target = 0.03 + (ai_score * 0.12)  # %3-%15 arası
            
            # 2. Teknik Analiz Gücüne Göre
            ta_based_target = 0.02 + (ta_strength * 0.10)  # %2-%12 arası
            
            # 3. Coin Volatilitesine Göre
            volatile_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF']
            volatility_multiplier = 1.5 if any(coin in symbol for coin in volatile_coins) else 1.0
            
            # 4. Timeframe'e Göre
            timeframe_multipliers = {
                '1m': 0.5,   # Kısa vadeli = düşük hedef
                '5m': 0.7,
                '15m': 0.9,
                '1h': 1.0,   # Standart
                '4h': 1.3,
                '1d': 1.8    # Uzun vadeli = yüksek hedef
            }
            tf_multiplier = timeframe_multipliers.get(timeframe, 1.0)
            
            # 5. Market Trendine Göre
            trend_multiplier = 1.2 if ai_score > 0.7 else 0.8
            
            # Final hedef hesaplama
            base_target = (ai_based_target + ta_based_target) / 2
            final_target = base_target * volatility_multiplier * tf_multiplier * trend_multiplier
            
            # Sınırlar
            final_target = max(0.01, min(0.25, final_target))  # %1-%25 arası
            
            # Yön bazlı hedef fiyatı hesapla
            if direction == 'LONG':
                target_price = entry_price * (1 + final_target)
            elif direction == 'SHORT':
                target_price = entry_price * (1 - final_target)
            else:  # NEUTRAL
                target_price = entry_price
            
            return round(target_price, 6)  # Fiyat hassasiyeti için 6 decimal
            
        except Exception as e:
            self.logger.error(f"Dinamik kar hedefi hesaplanamadı: {e}")
            return 0.05  # Varsayılan %5

    def calculate_breakout_threshold(self, signal_data):
        """Ani yükseliş eşiği hesapla"""
        try:
            ai_score = signal_data.get('ai_score', 0.5)
            ta_strength = signal_data.get('ta_strength', 0.5)
            symbol = signal_data.get('symbol', '')
            timeframe = signal_data.get('timeframe', '1h')
            
            # 1. AI Skoruna Göre Eşik
            ai_based_threshold = 0.02 + (ai_score * 0.08)  # %2-%10 arası
            
            # 2. Teknik Analiz Gücüne Göre
            ta_based_threshold = 0.015 + (ta_strength * 0.06)  # %1.5-%7.5 arası
            
            # 3. Coin Tipine Göre
            meme_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF']
            defi_coins = ['UNI', 'AAVE', 'COMP', 'CRV', 'SUSHI']
            
            if any(coin in symbol for coin in meme_coins):
                coin_multiplier = 1.8  # Meme coinler daha volatil
            elif any(coin in symbol for coin in defi_coins):
                coin_multiplier = 1.3  # DeFi coinler orta volatil
            else:
                coin_multiplier = 1.0  # Normal coinler
            
            # 4. Timeframe'e Göre
            tf_thresholds = {
                '1m': 0.005,  # %0.5
                '5m': 0.01,   # %1
                '15m': 0.015, # %1.5
                '1h': 0.025,  # %2.5
                '4h': 0.04,   # %4
                '1d': 0.06    # %6
            }
            base_threshold = tf_thresholds.get(timeframe, 0.025)
            
            # Final eşik hesaplama
            final_threshold = base_threshold * coin_multiplier
            final_threshold = max(0.005, min(0.15, final_threshold))  # %0.5-%15 arası
            
            return round(final_threshold, 4)
            
        except Exception as e:
            self.logger.error(f"Yükseliş eşiği hesaplanamadı: {e}")
            return 0.025  # Varsayılan %2.5

    def predict_breakout_probability(self, signal_data):
        """Ani yükseliş olasılığını tahmin et"""
        try:
            required_keys = ['ai_score', 'ta_strength', 'whale_score', 'social_score', 'news_score']
            for key in required_keys:
                if key not in signal_data or signal_data[key] is None:
                    self.logger.error(f"{key} eksik, breakout olasılığı hesaplanamadı. Sinyal üretimi iptal edildi.")
                    return None
            weights = {
                'ai_score': 0.35,
                'ta_strength': 0.25,
                'whale_score': 0.20,
                'social_score': 0.12,
                'news_score': 0.08
            }
            total_probability = 0
            total_weight = 0
            for factor, weight in weights.items():
                score = signal_data.get(factor)
                if score is None:
                    self.logger.error(f"{factor} eksik, breakout olasılığı hesaplanamadı. Sinyal üretimi iptal edildi.")
                    return None
                score = max(0.0, min(1.0, score))
                total_probability += score * weight
                total_weight += weight
            if total_weight == 0:
                self.logger.error("Breakout olasılığı hesaplanamadı, toplam ağırlık 0. Sinyal üretimi iptal edildi.")
                return None
            final_probability = total_probability / total_weight
            final_probability = max(0.0, min(1.0, final_probability))
            # Olasılığı 0-1 arasına sınırla (güvenlik için)
            final_probability = max(0.0, min(1.0, final_probability))
            # Olasılık kategorileri
            if final_probability >= 0.8:
                category = "YÜKSEK"
                confidence = "Çok güçlü sinyal"
            elif final_probability >= 0.6:
                category = "ORTA"
                confidence = "Güçlü sinyal"
            elif final_probability >= 0.4:
                category = "DÜŞÜK"
                confidence = "Zayıf sinyal"
            else:
                category = "ÇOK DÜŞÜK"
                confidence = "Riski yüksek"
            
            return {
                'probability': round(final_probability, 4),  # 0-1 arası
                'category': category,
                'confidence': confidence,
                'breakout_threshold': self.calculate_breakout_threshold(signal_data),
                'profit_target': self.calculate_dynamic_profit_target(signal_data),
                'ai_score': signal_data['ai_score']  # AI skorunu da ekle
            }
            
        except Exception as e:
            self.logger.error(f"Yükseliş olasılığı hesaplanamadı: {e}")
            return {
                'probability': 0.5,  # 0-1 arası
                'category': 'ORTA',
                'confidence': 'Hesaplanamadı',
                'breakout_threshold': 0.025,
                'profit_target': 0.05,
                'ai_score': 0.5
            }

    def predict_breakout_time(self, signal_data):
        """Yükseliş süresini tahmin et"""
        try:
            ai_score = signal_data.get('ai_score', 0.5)
            ta_strength = signal_data.get('ta_strength', 0.5)
            symbol = signal_data.get('symbol', '')
            timeframe = signal_data.get('timeframe', '1h')
            
            # 1. AI Skoruna Göre Süre Tahmini
            ai_based_time = 2 + (ai_score * 46)  # 2-48 saat arası
            
            # 2. Teknik Analiz Gücüne Göre
            ta_based_time = 4 + (ta_strength * 44)  # 4-48 saat arası
            
            # 3. Coin Volatilitesine Göre
            volatile_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF']
            volatility_multiplier = 0.6 if any(coin in symbol for coin in volatile_coins) else 1.0
            
            # 4. Timeframe'e Göre
            timeframe_multipliers = {
                '1m': 0.3,   # Kısa vadeli = hızlı
                '5m': 0.5,
                '15m': 0.7,
                '1h': 1.0,   # Standart
                '4h': 1.5,
                '1d': 2.0    # Uzun vadeli = yavaş
            }
            tf_multiplier = timeframe_multipliers.get(timeframe, 1.0)
            
            # 5. Market Trendine Göre
            trend_multiplier = 0.8 if ai_score > 0.7 else 1.2
            
            # Final süre hesaplama
            base_time = (ai_based_time + ta_based_time) / 2
            final_time = base_time * volatility_multiplier * tf_multiplier * trend_multiplier
            
            # Sınırlar
            final_time = max(0.5, min(72, final_time))  # 0.5-72 saat arası
            
            return round(final_time, 2)
            
        except Exception as e:
            self.logger.error(f"Yükseliş süresi tahmin edilemedi: {e}")
            return 24.0  # Varsayılan 24 saat

    def calculate_risk_reward_ratio(self, signal_data):
        """Risk/Ödül oranını hesapla - Yön bazlı"""
        try:
            direction = signal_data.get('direction', 'LONG')
            entry_price = signal_data.get('entry_price', 0)
            current_price = signal_data.get('current_price', entry_price)
            
            if entry_price <= 0:
                return 1.67  # Varsayılan değer
            
            # ATR bazlı hedef hesaplama
            atr = signal_data.get('atr', 0.01)  # Varsayılan %1
            
            if direction == 'LONG':
                # LONG pozisyon için
                take_profit = entry_price + (atr * 2)  # 2 ATR yukarı
                stop_loss = entry_price - (atr * 1.5)  # 1.5 ATR aşağı
                
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
                
            elif direction == 'SHORT':
                # SHORT pozisyon için
                take_profit = entry_price - (atr * 2)  # 2 ATR aşağı
                stop_loss = entry_price + (atr * 1.5)  # 1.5 ATR yukarı
                
                potential_profit = entry_price - take_profit
                potential_loss = stop_loss - entry_price
                
            else:  # NEUTRAL
                return 1.0
            
            # Risk/Ödül oranı hesapla
            if potential_loss > 0:
                risk_reward = potential_profit / potential_loss
            else:
                risk_reward = 1.0
            
            return round(risk_reward, 2)
            
        except Exception as e:
            self.logger.error(f"Risk/Ödül oranı hesaplanamadı: {e}")
            return 1.67  # Varsayılan değer

    def calculate_volatility_score(self, signal_data):
        """Volatilite skorunu hesapla"""
        try:
            symbol = signal_data.get('symbol', '')
            timeframe = signal_data.get('timeframe', '1h')
            
            # Coin tipine göre volatilite
            meme_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF']
            defi_coins = ['UNI', 'AAVE', 'COMP', 'CRV', 'SUSHI']
            stable_coins = ['USDT', 'USDC', 'BUSD', 'DAI']
            
            if any(coin in symbol for coin in meme_coins):
                base_volatility = 0.9
            elif any(coin in symbol for coin in defi_coins):
                base_volatility = 0.7
            elif any(coin in symbol for coin in stable_coins):
                base_volatility = 0.1
            else:
                base_volatility = 0.5
            
            # Timeframe'e göre ayarlama
            tf_multipliers = {
                '1m': 1.5,   # Çok volatil
                '5m': 1.3,
                '15m': 1.1,
                '1h': 1.0,   # Standart
                '4h': 0.8,
                '1d': 0.6    # Daha stabil
            }
            tf_multiplier = tf_multipliers.get(timeframe, 1.0)
            
            final_volatility = base_volatility * tf_multiplier
            return round(min(1.0, max(0.0, final_volatility)), 4)
            
        except Exception as e:
            self.logger.error(f"Volatilite skoru hesaplanamadı: {e}")
            return 0.5

    def calculate_trend_strength(self, signal_data):
        """Trend gücünü hesapla"""
        try:
            ai_score = signal_data.get('ai_score', 0.5)
            ta_strength = signal_data.get('ta_strength', 0.5)
            
            # AI ve TA skorlarının ağırlıklı ortalaması
            trend_strength = (ai_score * 0.6) + (ta_strength * 0.4)
            
            return round(trend_strength, 4)
            
        except Exception as e:
            self.logger.error(f"Trend gücü hesaplanamadı: {e}")
            return 0.5

    def determine_market_regime(self, signal_data):
        """Market rejimini belirle"""
        try:
            ai_score = signal_data.get('ai_score', 0.5)
            ta_strength = signal_data.get('ta_strength', 0.5)
            
            # Market rejimi belirleme
            if ai_score >= 0.8 and ta_strength >= 0.7:
                return "GÜÇLÜ YÜKSELIŞ"
            elif ai_score >= 0.6 and ta_strength >= 0.5:
                return "YÜKSELIŞ"
            elif ai_score >= 0.4 and ta_strength >= 0.3:
                return "SIDEWAYS"
            elif ai_score >= 0.2 and ta_strength >= 0.2:
                return "DÜŞÜŞ"
            else:
                return "GÜÇLÜ DÜŞÜŞ"
                
        except Exception as e:
            self.logger.error(f"Market rejimi belirlenemedi: {e}")
            return "BİLİNMEYEN"

    def calculate_signal_quality_score(self, signal_data):
        """Sinyal kalite skorunu hesapla - SIKI KRİTERLER"""
        try:
            required_keys = ['ai_score', 'ta_strength', 'whale_score']
            for key in required_keys:
                if key not in signal_data or signal_data[key] is None:
                    self.logger.error(f"{key} eksik, kalite skoru hesaplanamadı. Sinyal üretimi iptal edildi.")
                    return 0.0
            if signal_data['ai_score'] < Config.MIN_AI_SCORE:
                return 0.0
            if signal_data['ta_strength'] < Config.MIN_TA_STRENGTH:
                return 0.0
            if signal_data['whale_score'] < Config.MIN_WHALE_SCORE:
                return 0.0
            weights = {
                'ai_score': 0.5,
                'ta_strength': 0.35,
                'whale_score': 0.15
            }
            quality_score = 0
            for factor, weight in weights.items():
                score = signal_data.get(factor)
                if score is None:
                    self.logger.error(f"{factor} eksik, kalite skoru hesaplanamadı. Sinyal üretimi iptal edildi.")
                    return 0.0
                quality_score += score * weight
            return min(quality_score, 1.0)
        except Exception as e:
            self.logger.error(f"Sinyal kalite skoru hatası: {e}")
            return 0.0

    def calculate_success_metrics(self, signal_data):
        """Başarı metriklerini hesapla"""
        try:
            metrics = {
                'breakout_probability': self.predict_breakout_probability(signal_data)['probability'],
                'predicted_breakout_time': self.predict_breakout_time(signal_data),
                'risk_reward_ratio': self.calculate_risk_reward_ratio(signal_data),
                'volatility_score': self.calculate_volatility_score(signal_data),
                'trend_strength': self.calculate_trend_strength(signal_data),
                'signal_quality': self.calculate_signal_quality_score(signal_data),
                'market_regime': self.determine_market_regime(signal_data),
                'predicted_profit_target': self.calculate_dynamic_profit_target(signal_data),
                'predicted_breakout_threshold': self.calculate_breakout_threshold(signal_data)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Başarı metrikleri hesaplanamadı: {e}")
            return {}

    def update_signal_with_advanced_metrics(self, signal_id, current_price=None, max_price=None, min_price=None):
        """Sinyali gelişmiş metriklerle güncelle"""
        try:
            # Sinyal verilerini al
            query = "SELECT * FROM signals WHERE id = :signal_id"
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'signal_id': signal_id})
                signal_data = result.fetchone()
                
            if not signal_data:
                return False
                
            # Sinyal verilerini dict'e çevir
            signal_dict = dict(signal_data)
            
            # Başarı metriklerini hesapla
            success_metrics = self.calculate_success_metrics(signal_dict)
            
            # Gerçekleşen değerleri hesapla
            entry_price = signal_dict.get('entry_price')
            if entry_price and current_price:
                actual_gain = (current_price - entry_price) / entry_price
                actual_loss = (min_price - entry_price) / entry_price if min_price else 0
                
                # Breakout başarısını kontrol et
                predicted_threshold = success_metrics.get('predicted_breakout_threshold', 0.025)
                breakout_achieved = actual_gain >= predicted_threshold
                
                # Risk/Ödül oranını hesapla
                actual_rr = abs(actual_gain / actual_loss) if actual_loss != 0 else 0
                
                # Güncelleme sorgusu
                update_query = """
                    UPDATE signals SET
                        predicted_breakout_threshold = :predicted_threshold,
                        actual_max_gain = :actual_gain,
                        actual_max_loss = :actual_loss,
                        breakout_achieved = :breakout_achieved,
                        risk_reward_ratio = :predicted_rr,
                        actual_risk_reward_ratio = :actual_rr,
                        volatility_score = :volatility_score,
                        trend_strength = :trend_strength,
                        market_regime = :market_regime,
                        signal_quality_score = :signal_quality,
                        success_metrics = :success_metrics,
                        volume_score = :volume_score,
                        momentum_score = :momentum_score,
                        pattern_score = :pattern_score,
                        order_book_imbalance = :order_book_imbalance,
                        top_bid_walls = :top_bid_walls,
                        top_ask_walls = :top_ask_walls,
                        whale_direction_score = :whale_direction_score
                    WHERE id = :signal_id
                """
                
                conn.execute(text(update_query), {
                    'predicted_threshold': predicted_threshold,
                    'actual_gain': actual_gain,
                    'actual_loss': actual_loss,
                    'breakout_achieved': breakout_achieved,
                    'predicted_rr': success_metrics.get('risk_reward_ratio', 1.67),
                    'actual_rr': actual_rr,
                    'volatility_score': success_metrics.get('volatility_score', 0.5),
                    'trend_strength': success_metrics.get('trend_strength', 0.5),
                    'market_regime': success_metrics.get('market_regime', 'BİLİNMEYEN'),
                    'signal_quality': success_metrics.get('signal_quality', 0.5),
                    'success_metrics': json.dumps(success_metrics),
                    'volume_score': signal_dict.get('volume_score', 0.0),
                    'momentum_score': signal_dict.get('momentum_score', 0.0),
                    'pattern_score': signal_dict.get('pattern_score', 0.0),
                    'order_book_imbalance': signal_dict.get('order_book_imbalance', 0.0),
                    'top_bid_walls': str(signal_dict.get('top_bid_walls', '')),
                    'top_ask_walls': str(signal_dict.get('top_ask_walls', '')),
                    'whale_direction_score': signal_dict.get('whale_direction_score', 0.0),
                    'signal_id': signal_id
                })
                conn.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Sinyal gelişmiş metriklerle güncellenemedi: {e}")
            return False

    def get_recent_signals(self, hours=24):
        """Son X saatteki sinyalleri getir"""
        try:
            query = f"""
                SELECT * FROM signals 
                WHERE timestamp::timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp::timestamp DESC
            """
            df = pd.read_sql(query, self.engine, coerce_float=True)
            return df
        except Exception as e:
            self.logger.error(f"Son {hours} saatteki sinyaller alınamadı: {e}")
            return pd.DataFrame()

    def get_signals(self, limit=20, offset=0):
        """Sinyalleri getir (pagination ile)"""
        try:
            query = f"""
                SELECT * FROM signals 
                ORDER BY timestamp DESC
                LIMIT {limit} OFFSET {offset}
            """
            df = pd.read_sql(query, self.engine, coerce_float=True)
            return df
        except Exception as e:
            self.logger.error(f"Sinyaller alınamadı: {e}")
            return pd.DataFrame()

    def advanced_signal_filtering(self, signal_data):
        """Gelişmiş signal filtering - daha kaliteli sinyaller"""
        try:
            # Quality score calculation
            quality_score = self.calculate_signal_quality(signal_data)
            
            # Market regime adjustment
            market_adjusted_score = self.adjust_for_market_regime(signal_data, quality_score)
            
            # Volatility adjustment
            volatility_adjusted_score = self.adjust_for_volatility(signal_data, market_adjusted_score)
            
            # Time-based filtering
            time_filtered = self.time_based_filtering(signal_data)
            
            # Volume-based filtering
            volume_filtered = self.volume_based_filtering(signal_data)
            
            # Final quality check
            final_score = self.final_quality_check(signal_data, volatility_adjusted_score)
            
            return final_score, quality_score
            
        except Exception as e:
            self.logger.error(f"Advanced signal filtering hatası: {e}")
            return 0.0, 0.0
    
    def calculate_signal_quality(self, signal_data):
        """Signal quality score calculation"""
        try:
            # AI model confidence
            ai_score = signal_data.get('ai_score', 0.0)
            
            # Technical analysis strength
            ta_strength = signal_data.get('ta_strength', 0.0)
            
            # Volume confirmation
            volume_score = signal_data.get('volume_score', 0.0)
            
            # Price action confirmation
            price_action_score = signal_data.get('price_action_score', 0.0)
            
            # Pattern recognition score
            pattern_score = signal_data.get('pattern_score', 0.0)
            
            # Momentum confirmation
            momentum_score = signal_data.get('momentum_score', 0.0)
            
            # Weighted quality score
            quality_score = (
                ai_score * 0.3 +
                ta_strength * 0.25 +
                volume_score * 0.15 +
                price_action_score * 0.15 +
                pattern_score * 0.1 +
                momentum_score * 0.05
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Signal quality calculation hatası: {e}")
            return 0.0
    
    def adjust_for_market_regime(self, signal_data, base_score):
        """Market regime'e göre score adjustment"""
        try:
            market_regime = signal_data.get('market_regime', 'unknown')
            signal_type = signal_data.get('signal_type', 'NEUTRAL')
            
            # Market regime multipliers
            regime_multipliers = {
                'trending': 1.1,  # Trending market'te sinyaller daha güvenilir
                'sideways': 0.9,  # Sideways market'te daha dikkatli
                'volatile': 0.7,  # Volatile market'te daha düşük güven
                'unknown': 1.0
            }
            
            multiplier = regime_multipliers.get(market_regime, 1.0)
            
            # Signal type specific adjustments
            if signal_type == 'BUY' and market_regime == 'trending':
                multiplier *= 1.05  # Uptrend'de buy sinyalleri daha güçlü
            elif signal_type == 'SELL' and market_regime == 'volatile':
                multiplier *= 1.1   # Volatile market'te sell sinyalleri daha güçlü
                
            return min(1.0, base_score * multiplier)
            
        except Exception as e:
            self.logger.error(f"Market regime adjustment hatası: {e}")
            return base_score
    
    def adjust_for_volatility(self, signal_data, base_score):
        """Volatility'e göre score adjustment"""
        try:
            volatility = signal_data.get('volatility', 'normal')
            atr = signal_data.get('atr', 0.0)
            avg_atr = signal_data.get('avg_atr', 0.0)
            
            if avg_atr > 0:
                volatility_ratio = atr / avg_atr
                
                # High volatility adjustment
                if volatility_ratio > 1.5:  # High volatility
                    base_score *= 0.8  # Reduce confidence
                elif volatility_ratio < 0.5:  # Low volatility
                    base_score *= 1.1  # Increase confidence
                    
            return min(1.0, max(0.0, base_score))
            
        except Exception as e:
            self.logger.error(f"Volatility adjustment hatası: {e}")
            return base_score
    
    def time_based_filtering(self, signal_data):
        """Time-based signal filtering"""
        try:
            # Market hours check
            current_hour = datetime.now().hour
            
            # Asian session (0-8), European session (8-16), US session (16-24)
            session_multipliers = {
                'asian': 0.9,    # Asian session'da daha düşük volatility
                'european': 1.0, # European session normal
                'us': 1.1        # US session'da daha yüksek volatility
            }
            
            if 0 <= current_hour < 8:
                session = 'asian'
            elif 8 <= current_hour < 16:
                session = 'european'
            else:
                session = 'us'
                
            return session_multipliers.get(session, 1.0)
            
        except Exception as e:
            self.logger.error(f"Time-based filtering hatası: {e}")
            return 1.0
    
    def volume_based_filtering(self, signal_data):
        """Volume-based signal filtering"""
        try:
            volume = signal_data.get('volume', 0.0)
            avg_volume = signal_data.get('avg_volume', 0.0)
            
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                
                # Volume confirmation
                if volume_ratio > 1.5:  # High volume confirmation
                    return 1.2
                elif volume_ratio < 0.5:  # Low volume - reduce confidence
                    return 0.8
                    
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Volume-based filtering hatası: {e}")
            return 1.0
    
    def final_quality_check(self, signal_data, adjusted_score):
        """Final quality check ve threshold"""
        try:
            # Minimum quality threshold
            min_quality = Config.MIN_SIGNAL_CONFIDENCE
            
            # Signal type specific thresholds
            signal_type = signal_data.get('signal_type', 'NEUTRAL')
            
            if signal_type == 'BUY':
                threshold = min_quality + 0.05  # Buy sinyalleri için daha yüksek threshold
            elif signal_type == 'SELL':
                threshold = min_quality + 0.03  # Sell sinyalleri için orta threshold
            else:  # NEUTRAL
                threshold = min_quality + 0.1   # Neutral sinyaller için en yüksek threshold
                
            # Final score check
            if adjusted_score >= threshold:
                return adjusted_score
            else:
                return 0.0  # Signal rejected
                
        except Exception as e:
            self.logger.error(f"Final quality check hatası: {e}")
            return 0.0

    def calculate_advanced_signal_quality(self, signal_data):
        """Gelişmiş signal quality scoring"""
        try:
            quality_score = 0.0
            quality_factors = {}
            
            # 1. AI Score (0-30 puan)
            ai_score = signal_data.get('ai_score', 0)
            quality_factors['ai_score'] = ai_score * 30
            quality_score += quality_factors['ai_score']
            
            # 2. Technical Analysis Strength (0-25 puan)
            ta_strength = signal_data.get('technical_analysis_strength', 0)
            quality_factors['ta_strength'] = ta_strength * 25
            quality_score += quality_factors['ta_strength']
            
            # 3. Volume Analysis (0-15 puan)
            volume_score = self.calculate_volume_score(signal_data)
            quality_factors['volume_score'] = volume_score * 15
            quality_score += quality_factors['volume_score']
            
            # 4. Market Regime Alignment (0-10 puan)
            market_alignment = self.calculate_market_alignment(signal_data)
            quality_factors['market_alignment'] = market_alignment * 10
            quality_score += quality_factors['market_alignment']
            
            # 5. Volatility Analysis (0-10 puan)
            volatility_score = self.calculate_volatility_score(signal_data)
            quality_factors['volatility_score'] = volatility_score * 10
            quality_score += quality_factors['volatility_score']
            
            # 6. Trend Strength (0-10 puan)
            trend_score = self.calculate_trend_strength(signal_data)
            quality_factors['trend_score'] = trend_score * 10
            quality_score += quality_factors['trend_score']
            
            return {
                'total_score': quality_score,
                'factors': quality_factors,
                'grade': self.get_quality_grade(quality_score)
            }
            
        except Exception as e:
            self.logger.error(f"Signal quality hesaplama hatası: {e}")
            return {'total_score': 0, 'factors': {}, 'grade': 'F'}

    def calculate_volume_score(self, signal_data):
        """Volume analizi skoru"""
        try:
            volume_ratio = signal_data.get('volume_ratio', 1.0)
            obv_trend = signal_data.get('obv_trend', 0)
            
            # Volume ratio scoring
            if volume_ratio > 2.0:
                vol_score = 1.0
            elif volume_ratio > 1.5:
                vol_score = 0.8
            elif volume_ratio > 1.2:
                vol_score = 0.6
            else:
                vol_score = 0.3
            
            # OBV trend scoring
            if obv_trend > 0:
                obv_score = 0.5
            else:
                obv_score = 0.0
            
            return (vol_score + obv_score) / 1.5
            
        except Exception as e:
            self.logger.error(f"Volume score hesaplama hatası: {e}")
            return 0.0

    def calculate_market_alignment(self, signal_data):
        """Market regime alignment skoru"""
        try:
            market_regime = signal_data.get('market_regime', 'unknown')
            signal_type = signal_data.get('signal_type', 'NEUTRAL')
            
            # Market regime alignment
            if market_regime == 'bullish' and signal_type == 'BUY':
                return 1.0
            elif market_regime == 'bearish' and signal_type == 'SELL':
                return 1.0
            elif market_regime == 'sideways':
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Market alignment hesaplama hatası: {e}")
            return 0.0

    def calculate_volatility_score(self, signal_data):
        """Volatility analizi skoru"""
        try:
            volatility = signal_data.get('volatility', 0)
            atr = signal_data.get('atr', 0)
            
            # Optimal volatility range
            if 0.02 <= volatility <= 0.08:
                return 1.0
            elif 0.01 <= volatility <= 0.12:
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Volatility score hesaplama hatası: {e}")
            return 0.0

    def calculate_trend_strength(self, signal_data):
        """Trend strength skoru"""
        try:
            adx = signal_data.get('adx', 0)
            rsi = signal_data.get('rsi_14', 50)
            
            # ADX trend strength
            if adx > 25:
                adx_score = 1.0
            elif adx > 20:
                adx_score = 0.7
            else:
                adx_score = 0.3
            
            # RSI trend confirmation
            if 30 <= rsi <= 70:
                rsi_score = 0.5
            else:
                rsi_score = 0.0
            
            return (adx_score + rsi_score) / 1.5
                
        except Exception as e:
            self.logger.error(f"Trend strength hesaplama hatası: {e}")
            return 0.0

    def get_quality_grade(self, score):
        """Quality grade belirleme"""
        if score >= 85:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 75:
            return 'A-'
        elif score >= 70:
            return 'B+'
        elif score >= 65:
            return 'B'
        elif score >= 60:
            return 'B-'
        elif score >= 55:
            return 'C+'
        elif score >= 50:
            return 'C'
        else:
            return 'F'

    def create_signal(self, symbol, direction, confidence, analysis_data):
        """Gelişmiş sinyal oluşturma - Breakout threshold dahil"""
        try:
            ai_score = analysis_data.get('ai_score', 0)
            ta_strength = analysis_data.get('ta_strength', analysis_data.get('signal_strength', 0))
            whale_score = analysis_data.get('whale_score', 0)
            breakout_prob = analysis_data.get('breakout_probability', 0)
            current_price = analysis_data.get('close', 0)
            confidence_level = analysis_data.get('confidence', 0)
            # Skorlar için hem eski hem yeni anahtarları kontrol et
            signal_strength = analysis_data.get('signal_strength', analysis_data.get('ta_strength', 0))
            volume_score = analysis_data.get('volume_score', 0)
            momentum_score = analysis_data.get('momentum_score', 0)
            pattern_score = analysis_data.get('pattern_score', 0)
            whale_direction_score = analysis_data.get('whale_direction_score', 0.0)
            order_book_imbalance = analysis_data.get('order_book_imbalance', 0.0)
            top_bid_walls = analysis_data.get('top_bid_walls', '[]')
            top_ask_walls = analysis_data.get('top_ask_walls', '[]')

            # Breakout threshold hesapla
            breakout_threshold = self.calculate_breakout_threshold(analysis_data)
            breakout_time = self.predict_breakout_time(analysis_data)

            # Hedef fiyatları hesapla
            take_profit, stop_loss, support_level, resistance_level, target_time = self.calculate_target_levels(
                analysis_data, current_price
            )

            # Success metrics hesapla
            success_metrics = self.calculate_success_metrics(analysis_data)

            def _fix_missing(val):
                if val in [None, '', [], {}, float('nan')] or (isinstance(val, float) and (np.isnan(val) or val == 0.0)):
                    return 'Eksik Veri'
                return val

            signal = {
                'symbol': symbol,
                'timeframe': '1h',
                'direction': direction,
                'confidence': _fix_missing(confidence),
                'ai_score': _fix_missing(ai_score),
                'ta_strength': _fix_missing(ta_strength),
                'signal_strength': _fix_missing(signal_strength),
                'whale_score': _fix_missing(whale_score),
                'social_score': 0.0,
                'news_score': 0.0,
                'whale_direction_score': whale_direction_score,
                'breakout_probability': _fix_missing(breakout_prob),
                'entry_price': current_price,
                'current_price': current_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'target_time_hours': target_time,
                'predicted_breakout_threshold': breakout_threshold,
                'predicted_breakout_time_hours': breakout_time,
                'timestamp': datetime.now(),
                'status': 'ACTIVE',
                'success_metrics': success_metrics,
                'confidence_level': _fix_missing(confidence_level),
                'volume_score': _fix_missing(volume_score),
                'pattern_score': _fix_missing(pattern_score),
                'momentum_score': _fix_missing(momentum_score),
                'order_book_imbalance': order_book_imbalance,
                'top_bid_walls': top_bid_walls,
                'top_ask_walls': top_ask_walls,
            }
            # Eksik skorlar için log
            for key in ['confidence','signal_strength','volume_score','momentum_score','pattern_score']:
                if signal[key] == 'Eksik Veri':
                    self.logger.warning(f"{symbol} için {key} eksik! Sinyal eksik veriyle üretildi.")

            return signal
        except Exception as e:
            self.logger.error(f"Sinyal olusturma hatasi {symbol}: {e}")
            return None

    def _check_market_conditions(self, symbol, analysis_data):
        """Market koşullarını kontrol et"""
        try:
            # Market cap kontrolü
            market_cap = analysis_data.get('market_cap', 0)
            if market_cap < 10000000:  # 10M altı coinler riskli
                return False
            
            # 24h hacim kontrolü
            volume_24h = analysis_data.get('volume_24h', 0)
            if volume_24h < 1000000:  # 1M altı hacim riskli
                return False
            
            # Fiyat değişimi kontrolü
            price_change_24h = analysis_data.get('price_change_24h', 0)
            if abs(price_change_24h) > 0.3:  # %30'dan fazla değişim riskli
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Market koşulları kontrol hatası: {e}")
            return True  # Hata durumunda geç

    def _check_volatility_conditions(self, analysis_data):
        """Volatilite koşullarını kontrol et - ÇOK SIKI"""
        try:
            volatility = analysis_data.get('volatility', 0)
            
            # Çok düşük volatilite (sıkıcı) - Daha sıkı
            if volatility < 0.05:  # 0.03'ten 0.05'e yükseltildi
                return False
            
            # Çok yüksek volatilite (riskli) - Daha sıkı
            if volatility > 0.10:  # 0.12'den 0.10'a düşürüldü
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Volatilite kontrol hatası: {e}")
            return True

    def _check_volume_conditions(self, analysis_data):
        """Hacim koşullarını kontrol et - ÇOK SIKI"""
        try:
            volume_ratio = analysis_data.get('volume_ratio', 1.0)
            
            # Hacim çok düşükse - Daha sıkı
            if volume_ratio < 1.2:  # 0.8'den 1.2'ye yükseltildi
                return False
            
            # Hacim çok yüksekse (pump & dump olabilir) - Daha sıkı
            if volume_ratio > 2.5:  # 3.0'dan 2.5'e düşürüldü
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hacim kontrol hatası: {e}")
            return True

    def _check_trend_conditions(self, analysis_data, direction):
        """Trend koşullarını kontrol et - ÇOK SIKI"""
        try:
            trend_strength = analysis_data.get('trend_strength', 0.5)
            
            # Trend gücü çok düşükse - Daha sıkı
            if trend_strength < 0.8:  # 0.7'den 0.8'e yükseltildi
                return False
            
            # Yön ile trend uyumu kontrolü
            if direction == 'LONG' and trend_strength < 0.85:
                return False
            elif direction == 'SHORT' and trend_strength < 0.85:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trend kontrol hatası: {e}")
            return True

    def _calculate_target_price(self, analysis_data, direction):
        """Hedef fiyat hesapla"""
        try:
            current_price = analysis_data.get('current_price', 0)
            atr = analysis_data.get('atr', 0)
            
            if direction == 'LONG':
                return current_price + (atr * 2)  # 2 ATR yukarı
            elif direction == 'SHORT':
                return current_price - (atr * 2)  # 2 ATR aşağı
            else:
                return current_price
                
        except Exception as e:
            self.logger.error(f"Hedef fiyat hesaplama hatası: {e}")
            return 0

    def _calculate_stop_loss(self, analysis_data, direction):
        """Stop loss hesapla"""
        try:
            current_price = analysis_data.get('current_price', 0)
            atr = analysis_data.get('atr', 0)
            
            if direction == 'LONG':
                return current_price - (atr * 1.5)  # 1.5 ATR aşağı
            elif direction == 'SHORT':
                return current_price + (atr * 1.5)  # 1.5 ATR yukarı
            else:
                return current_price
                
        except Exception as e:
            self.logger.error(f"Stop loss hesaplama hatası: {e}")
            return 0

    def _calculate_risk_reward_ratio(self, analysis_data, direction):
        """Risk/Ödül oranı hesapla"""
        try:
            current_price = analysis_data.get('current_price', 0)
            target_price = self._calculate_target_price(analysis_data, direction)
            stop_loss = self._calculate_stop_loss(analysis_data, direction)
            
            if direction == 'LONG':
                reward = target_price - current_price
                risk = current_price - stop_loss
            elif direction == 'SHORT':
                reward = current_price - target_price
                risk = stop_loss - current_price
            else:
                return 1.0
            
            if risk > 0:
                return reward / risk
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Risk/Ödül oranı hesaplama hatası: {e}")
            return 1.0

    def _generate_signal_id(self, signal):
        """Benzersiz sinyal ID'si oluştur"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            symbol_val = signal.get('symbol', '')
            safe_symbol = str(symbol_val).replace('/', '_') if symbol_val else ''
            return f"{safe_symbol}_{timestamp}"
        except Exception as e:
            self.logger.error(f"Sinyal ID oluşturma hatası: {e}")
            return f"signal_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _save_signal_to_db(self, signal):
        """Sinyali veritabanına kaydet"""
        try:
            self.save_signal_db(signal)
        except Exception as e:
            self.logger.error(f"Veritabanı kaydetme hatası: {e}")

    def _save_signal_to_file(self, signal):
        """Sinyali dosyaya kaydet"""
        try:
            self.save_signal_json(signal)
        except Exception as e:
            self.logger.error(f"Dosya kaydetme hatası: {e}")

    def evaluate_signal_quality(self, signal_data):
        """Sinyal kalitesini değerlendir ve skor ver"""
        try:
            score = 0
            max_score = 100
            
            # AI Model Skoru (0-30 puan)
            if 'ai_score' in signal_data:
                ai_score = signal_data['ai_score']
                if ai_score > 0.8:
                    score += 30
                elif ai_score > 0.6:
                    score += 20
                elif ai_score > 0.4:
                    score += 10
            
            # Teknik Analiz Skoru (0-25 puan)
            if 'technical_score' in signal_data:
                tech_score = signal_data['technical_score']
                if tech_score > 0.8:
                    score += 25
                elif tech_score > 0.6:
                    score += 15
                elif tech_score > 0.4:
                    score += 8
            
            # Pattern Recognition Skoru (0-20 puan)
            pattern_score = 0
            if 'doji' in signal_data and signal_data['doji']:
                pattern_score += 3
            if 'hammer' in signal_data and signal_data['hammer']:
                pattern_score += 5
            if 'engulfing_bullish' in signal_data and signal_data['engulfing_bullish']:
                pattern_score += 8
            if 'breakout_up' in signal_data and signal_data['breakout_up']:
                pattern_score += 10
            if 'near_support' in signal_data and signal_data['near_support']:
                pattern_score += 4
            
            score += min(pattern_score, 20)
            
            # Hacim Analizi (0-15 puan)
            if 'volume_ratio' in signal_data:
                vol_ratio = signal_data['volume_ratio']
                if vol_ratio > 2.0:
                    score += 15
                elif vol_ratio > 1.5:
                    score += 10
                elif vol_ratio > 1.2:
                    score += 5
            
            # Trend Gücü (0-10 puan)
            if 'trend_strength' in signal_data:
                trend_strength = signal_data['trend_strength']
                if trend_strength > 0.7:
                    score += 10
                elif trend_strength > 0.5:
                    score += 6
                elif trend_strength > 0.3:
                    score += 3
            
            # Kalite kategorisi belirle
            if score >= 80:
                quality = "EXCELLENT"
            elif score >= 60:
                quality = "GOOD"
            elif score >= 40:
                quality = "FAIR"
            else:
                quality = "POOR"
            
            return {
                'total_score': score,
                'max_score': max_score,
                'quality': quality,
                'pattern_score': pattern_score,
                'details': {
                    'ai_contribution': score * 0.3 if 'ai_score' in signal_data else 0,
                    'technical_contribution': score * 0.25 if 'technical_score' in signal_data else 0,
                    'pattern_contribution': pattern_score,
                    'volume_contribution': score * 0.15 if 'volume_ratio' in signal_data else 0,
                    'trend_contribution': score * 0.1 if 'trend_strength' in signal_data else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Sinyal kalitesi değerlendirme hatası: {e}")
            return {
                'total_score': 0,
                'max_score': 100,
                'quality': 'ERROR',
                'pattern_score': 0,
                'details': {}
            }
    
    def filter_signals_by_quality(self, signals, min_quality_score=70):
        """Kalite skoruna göre sinyalleri filtrele (daha sıkı)"""
        try:
            filtered_signals = []
            
            for signal in signals:
                quality_eval = self.evaluate_signal_quality(signal)
                signal['quality_evaluation'] = quality_eval
                
                if quality_eval['total_score'] >= min_quality_score:
                    filtered_signals.append(signal)
                    self.logger.info(f"Sinyal kabul edildi: {signal.get('symbol', 'Unknown')} - "
                                   f"Skor: {quality_eval['total_score']}/{quality_eval['max_score']} "
                                   f"({quality_eval['quality']})")
                else:
                    self.logger.info(f"Sinyal reddedildi: {signal.get('symbol', 'Unknown')} - "
                                   f"Skor: {quality_eval['total_score']}/{quality_eval['max_score']} "
                                   f"({quality_eval['quality']})")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Sinyal filtreleme hatası: {e}")
            return signals
    
    def check_market_conditions(self, symbol):
        """Piyasa koşullarını kontrol et"""
        try:
            # Basit piyasa koşulları kontrolü
            # Gerçek uygulamada daha karmaşık analiz yapılabilir
            
            market_conditions = {
                'volatility': 'NORMAL',  # LOW, NORMAL, HIGH
                'trend': 'NEUTRAL',      # BULLISH, BEARISH, NEUTRAL
                'volume': 'NORMAL',      # LOW, NORMAL, HIGH
                'risk_level': 'MEDIUM'   # LOW, MEDIUM, HIGH
            }
            
            # Bu fonksiyon gerçek piyasa verilerine göre güncellenebilir
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"Piyasa koşulları kontrol hatası: {e}")
            return {
                'volatility': 'NORMAL',
                'trend': 'NEUTRAL',
                'volume': 'NORMAL',
                'risk_level': 'MEDIUM'
            }
    
    def adjust_signal_parameters(self, signal_data, market_conditions):
        """Piyasa koşullarına göre sinyal parametrelerini ayarla"""
        try:
            adjusted_signal = signal_data.copy()
            
            # Volatiliteye göre ayarlama
            if market_conditions['volatility'] == 'HIGH':
                # Yüksek volatilitede daha konservatif
                if 'stop_loss' in adjusted_signal:
                    adjusted_signal['stop_loss'] *= 1.2  # Stop loss'u genişlet
                if 'take_profit' in adjusted_signal:
                    adjusted_signal['take_profit'] *= 1.3  # Take profit'i artır
            
            elif market_conditions['volatility'] == 'LOW':
                # Düşük volatilitede daha agresif
                if 'stop_loss' in adjusted_signal:
                    adjusted_signal['stop_loss'] *= 0.8
                if 'take_profit' in adjusted_signal:
                    adjusted_signal['take_profit'] *= 0.9
            
            # Trend'e göre ayarlama
            if market_conditions['trend'] == 'BULLISH':
                # Yükseliş trendinde daha agresif
                if 'confidence' in adjusted_signal:
                    adjusted_signal['confidence'] *= 1.1
            
            elif market_conditions['trend'] == 'BEARISH':
                # Düşüş trendinde daha konservatif
                if 'confidence' in adjusted_signal:
                    adjusted_signal['confidence'] *= 0.9
            
            # Risk seviyesine göre pozisyon büyüklüğü
            if market_conditions['risk_level'] == 'HIGH':
                adjusted_signal['position_size'] = 'SMALL'
            elif market_conditions['risk_level'] == 'LOW':
                adjusted_signal['position_size'] = 'LARGE'
            else:
                adjusted_signal['position_size'] = 'MEDIUM'
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"Sinyal parametre ayarlama hatası: {e}")
            return signal_data

    def calculate_target_levels(self, signal, entry_price):
        """İyileştirilmiş hedef seviyeler hesaplama - Destek/Direnç bazlı"""
        try:
            direction = signal.get('direction', 'LONG')
            ai_score = signal.get('ai_score', 0.5)
            ta_strength = signal.get('ta_strength', 0.5)
            symbol = signal.get('symbol', '')
            
            # ATR hesaplama - Eğer sinyalde yoksa entry_price'ın %2'si olarak hesapla
            atr = signal.get('atr', 0)
            if atr == 0:
                # Coin volatilitesine göre ATR hesapla
                volatile_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF', 'BONK', 'MEME']
                if any(coin in symbol.upper() for coin in volatile_coins):
                    atr = entry_price * 0.05  # %5 volatilite
                else:
                    atr = entry_price * 0.025  # %2.5 volatilite
            
            # Volatilite analizi
            volatility = signal.get('volatility', atr / entry_price)
            
            # Destek/Direnç seviyeleri
            support_levels = signal.get('support_levels', [])
            resistance_levels = signal.get('resistance_levels', [])
            
            # AI skoruna göre ATR çarpanını ayarla
            if ai_score > 0.8:
                atr_multiplier = 1.2  # Yüksek güven - daha sıkı
            elif ai_score < 0.3:
                atr_multiplier = 2.0  # Düşük güven - daha geniş
            else:
                atr_multiplier = 1.5  # Orta güven
            
            # TA gücüne göre ayarlama
            if ta_strength > 0.7:
                atr_multiplier *= 0.9  # Güçlü TA - daha sıkı
            elif ta_strength < 0.3:
                atr_multiplier *= 1.2  # Zayıf TA - daha geniş
            
            # Coin volatilitesine göre ayarlama
            volatile_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF', 'BONK', 'MEME']
            if any(coin in symbol.upper() for coin in volatile_coins):
                atr_multiplier *= 1.3  # Volatil coinler - daha geniş
            
            # Volatilite bazlı ayarlama
            if volatility > 0.05:  # %5'ten yüksek volatilite
                atr_multiplier *= 1.2
            elif volatility < 0.01:  # %1'den düşük volatilite
                atr_multiplier *= 0.8
            
            if direction == 'LONG':
                # LONG pozisyon için iyileştirilmiş hesaplama
                
                # Stop Loss - En iyi seçenek
                sl_candidates = []
                
                # 1. Destek bazlı stop loss
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < entry_price], default=0)
                    if nearest_support > 0:
                        sl_candidates.append(nearest_support * 0.995)  # %0.5 altında
                
                # 2. ATR bazlı stop loss
                sl_candidates.append(entry_price - (atr * atr_multiplier))
                
                # 3. Volatilite bazlı stop loss
                sl_candidates.append(entry_price * (1 - volatility * 0.5))
                
                # 4. Minimum stop loss (%1.5)
                sl_candidates.append(entry_price * 0.985)
                
                # En iyi stop loss seçimi (en yüksek)
                stop_loss = max(sl_candidates)
                
                # Take Profit - En iyi seçenek
                tp_candidates = []
                
                # 1. Direnç bazlı take profit
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > entry_price], default=float('inf'))
                    if nearest_resistance != float('inf'):
                        tp_candidates.append(nearest_resistance * 1.005)  # %0.5 üstünde
                
                # 2. ATR bazlı take profit (2:1 risk/reward)
                tp_candidates.append(entry_price + (atr * atr_multiplier * 2))
                
                # 3. Volatilite bazlı take profit
                tp_candidates.append(entry_price * (1 + volatility * 1.5))
                
                # 4. Minimum take profit (%2.5)
                tp_candidates.append(entry_price * 1.025)
                
                # En iyi take profit seçimi (en düşük)
                take_profit = min(tp_candidates)
                
                # Destek/Direnç seviyeleri
                support_level = stop_loss * 0.995  # Stop loss'un altında
                resistance_level = take_profit * 1.005  # Take profit'in üstünde
                
            elif direction == 'SHORT':
                # SHORT pozisyon için iyileştirilmiş hesaplama
                
                # Stop Loss - En iyi seçenek
                sl_candidates = []
                
                # 1. Direnç bazlı stop loss
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > entry_price], default=float('inf'))
                    if nearest_resistance != float('inf'):
                        sl_candidates.append(nearest_resistance * 1.005)  # %0.5 üstünde
                
                # 2. ATR bazlı stop loss
                sl_candidates.append(entry_price + (atr * atr_multiplier))
                
                # 3. Volatilite bazlı stop loss
                sl_candidates.append(entry_price * (1 + volatility * 0.5))
                
                # 4. Minimum stop loss (%1.5)
                sl_candidates.append(entry_price * 1.015)
                
                # En iyi stop loss seçimi (en düşük)
                stop_loss = min(sl_candidates)
                
                # Take Profit - En iyi seçenek
                tp_candidates = []
                
                # 1. Destek bazlı take profit
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < entry_price], default=0)
                    if nearest_support > 0:
                        tp_candidates.append(nearest_support * 0.995)  # %0.5 altında
                
                # 2. ATR bazlı take profit (2:1 risk/reward)
                tp_candidates.append(entry_price - (atr * atr_multiplier * 2))
                
                # 3. Volatilite bazlı take profit
                tp_candidates.append(entry_price * (1 - volatility * 1.5))
                
                # 4. Minimum take profit (%2.5)
                tp_candidates.append(entry_price * 0.975)
                
                # En iyi take profit seçimi (en yüksek)
                take_profit = max(tp_candidates)
                
                # Destek/Direnç seviyeleri
                support_level = take_profit * 0.995  # Take profit'in altında
                resistance_level = stop_loss * 1.005  # Stop loss'un üstünde
                
            else:  # NEUTRAL
                take_profit = entry_price * 1.02
                stop_loss = entry_price * 0.98
                support_level = entry_price * 0.99
                resistance_level = entry_price * 1.01
            
            # Risk/Ödül oranı hesapla
            if direction == 'LONG':
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            elif direction == 'SHORT':
                potential_profit = entry_price - take_profit
                potential_loss = stop_loss - entry_price
            else:
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            
            risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 1.0
            
            # Hedef süreyi hesapla
            target_time = 24.0  # Varsayılan 24 saat
            
            # AI skoruna göre süre ayarlaması
            if ai_score > 0.8:
                target_time *= 0.8  # Daha hızlı
            elif ai_score < 0.3:
                target_time *= 1.2  # Daha yavaş
            
            # Coin tipine göre süre ayarlaması
            if any(coin in symbol.upper() for coin in volatile_coins):
                target_time *= 0.7  # Volatil coinler daha hızlı
            
            return take_profit, stop_loss, support_level, resistance_level, target_time
            
        except Exception as e:
            self.logger.error(f"İyileştirilmiş hedef seviyeler hesaplama hatası: {e}")
            # Varsayılan değerler
            if direction == 'LONG':
                return entry_price * 1.05, entry_price * 0.95, entry_price * 0.95, entry_price * 1.05, 24.0
            elif direction == 'SHORT':
                return entry_price * 0.95, entry_price * 1.05, entry_price * 0.95, entry_price * 1.05, 24.0
            else:
                return entry_price * 1.02, entry_price * 0.98, entry_price * 0.99, entry_price * 1.01, 24.0

    def calculate_improved_target_levels(self, signal, entry_price):
        """İyileştirilmiş hedef seviyeler hesaplama - Destek/Direnç bazlı"""
        try:
            direction = signal.get('direction', 'LONG')
            ai_score = signal.get('ai_score', 0.5)
            ta_strength = signal.get('ta_strength', 0.5)
            symbol = signal.get('symbol', '')
            
            # ATR hesaplama
            atr = signal.get('atr', entry_price * 0.02)
            
            # Volatilite analizi
            volatility = signal.get('volatility', 0.02)
            
            # Destek/Direnç seviyeleri
            support_levels = signal.get('support_levels', [])
            resistance_levels = signal.get('resistance_levels', [])
            
            # AI skoruna göre ATR çarpanını ayarla
            if ai_score > 0.8:
                atr_multiplier = 1.2  # Yüksek güven - daha sıkı
            elif ai_score < 0.3:
                atr_multiplier = 2.0  # Düşük güven - daha geniş
            else:
                atr_multiplier = 1.5  # Orta güven
            
            # TA gücüne göre ayarlama
            if ta_strength > 0.7:
                atr_multiplier *= 0.9  # Güçlü TA - daha sıkı
            elif ta_strength < 0.3:
                atr_multiplier *= 1.2  # Zayıf TA - daha geniş
            
            # Coin volatilitesine göre ayarlama
            volatile_coins = ['DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', 'WIF']
            if any(coin in symbol.upper() for coin in volatile_coins):
                atr_multiplier *= 1.3  # Volatil coinler - daha geniş
            
            # Volatilite bazlı ayarlama
            if volatility > 0.05:  # %5'ten yüksek volatilite
                atr_multiplier *= 1.2
            elif volatility < 0.01:  # %1'den düşük volatilite
                atr_multiplier *= 0.8
            
            if direction == 'LONG':
                # LONG pozisyon için iyileştirilmiş hesaplama
                
                # Stop Loss - En iyi seçenek
                sl_candidates = []
                
                # 1. Destek bazlı stop loss
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < entry_price], default=0)
                    if nearest_support > 0:
                        sl_candidates.append(nearest_support * 0.995)  # %0.5 altında
                
                # 2. ATR bazlı stop loss
                sl_candidates.append(entry_price - (atr * atr_multiplier))
                
                # 3. Volatilite bazlı stop loss
                sl_candidates.append(entry_price * (1 - volatility * 0.5))
                
                # 4. Minimum stop loss (%1.5)
                sl_candidates.append(entry_price * 0.985)
                
                # En iyi stop loss seçimi (en yüksek)
                stop_loss = max(sl_candidates)
                
                # Take Profit - En iyi seçenek
                tp_candidates = []
                
                # 1. Direnç bazlı take profit
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > entry_price], default=float('inf'))
                    if nearest_resistance != float('inf'):
                        tp_candidates.append(nearest_resistance * 1.005)  # %0.5 üstünde
                
                # 2. ATR bazlı take profit (2:1 risk/reward)
                tp_candidates.append(entry_price + (atr * atr_multiplier * 2))
                
                # 3. Volatilite bazlı take profit
                tp_candidates.append(entry_price * (1 + volatility * 1.5))
                
                # 4. Minimum take profit (%2.5)
                tp_candidates.append(entry_price * 1.025)
                
                # En iyi take profit seçimi (en düşük)
                take_profit = min(tp_candidates)
                
                # Destek/Direnç seviyeleri
                support_level = stop_loss * 0.995  # Stop loss'un altında
                resistance_level = take_profit * 1.005  # Take profit'in üstünde
                
            elif direction == 'SHORT':
                # SHORT pozisyon için iyileştirilmiş hesaplama
                
                # Stop Loss - En iyi seçenek
                sl_candidates = []
                
                # 1. Direnç bazlı stop loss
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > entry_price], default=float('inf'))
                    if nearest_resistance != float('inf'):
                        sl_candidates.append(nearest_resistance * 1.005)  # %0.5 üstünde
                
                # 2. ATR bazlı stop loss
                sl_candidates.append(entry_price + (atr * atr_multiplier))
                
                # 3. Volatilite bazlı stop loss
                sl_candidates.append(entry_price * (1 + volatility * 0.5))
                
                # 4. Minimum stop loss (%1.5)
                sl_candidates.append(entry_price * 1.015)
                
                # En iyi stop loss seçimi (en düşük)
                stop_loss = min(sl_candidates)
                
                # Take Profit - En iyi seçenek
                tp_candidates = []
                
                # 1. Destek bazlı take profit
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < entry_price], default=0)
                    if nearest_support > 0:
                        tp_candidates.append(nearest_support * 0.995)  # %0.5 altında
                
                # 2. ATR bazlı take profit (2:1 risk/reward)
                tp_candidates.append(entry_price - (atr * atr_multiplier * 2))
                
                # 3. Volatilite bazlı take profit
                tp_candidates.append(entry_price * (1 - volatility * 1.5))
                
                # 4. Minimum take profit (%2.5)
                tp_candidates.append(entry_price * 0.975)
                
                # En iyi take profit seçimi (en yüksek)
                take_profit = max(tp_candidates)
                
                # Destek/Direnç seviyeleri
                support_level = take_profit * 0.995  # Take profit'in altında
                resistance_level = stop_loss * 1.005  # Stop loss'un üstünde
                
            else:  # NEUTRAL
                take_profit = entry_price * 1.02
                stop_loss = entry_price * 0.98
                support_level = entry_price * 0.99
                resistance_level = entry_price * 1.01
            
            # Risk/Ödül oranı hesapla
            if direction == 'LONG':
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            elif direction == 'SHORT':
                potential_profit = entry_price - take_profit
                potential_loss = stop_loss - entry_price
            else:
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            
            risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 1.0
            
            # Hedef süreyi hesapla
            target_time = 24.0  # Varsayılan 24 saat
            
            # AI skoruna göre süre ayarlaması
            if ai_score > 0.8:
                target_time *= 0.8  # Daha hızlı
            elif ai_score < 0.3:
                target_time *= 1.2  # Daha yavaş
            
            # Coin tipine göre süre ayarlaması
            if any(coin in symbol.upper() for coin in volatile_coins):
                target_time *= 0.7  # Volatil coinler daha hızlı
            
            return take_profit, stop_loss, support_level, resistance_level, target_time
            
        except Exception as e:
            self.logger.error(f"İyileştirilmiş hedef seviyeler hesaplama hatası: {e}")
            # Varsayılan değerler
            if direction == 'LONG':
                return entry_price * 1.05, entry_price * 0.95, entry_price * 0.95, entry_price * 1.05, 24.0
            elif direction == 'SHORT':
                return entry_price * 0.95, entry_price * 1.05, entry_price * 0.95, entry_price * 1.05, 24.0
            else:
                return entry_price * 1.02, entry_price * 0.98, entry_price * 0.99, entry_price * 1.01, 24.0

    def analyze_symbol_advanced(self, symbol, timeframe='1h'):
        """Gelişmiş sembol analizi - Çoklu zaman dilimi ve yeni AI model yapısı"""
        try:
            self.logger.info(f"[DEBUG] analyze_symbol_advanced başlıyor: symbol={symbol}, timeframe={timeframe}")
            # Veri topla - Çoklu zaman dilimi
            data_collector = DataCollector()
            df_1h = data_collector.get_historical_data(symbol, '1h', 1000)
            df_4h = data_collector.get_historical_data(symbol, '4h', 1000)
            df_1d = data_collector.get_historical_data(symbol, '1d', 1000)
            self.logger.info(f"[DEBUG] DataFrames: 1h={df_1h.shape if df_1h is not None else None}, 4h={df_4h.shape if df_4h is not None else None}, 1d={df_1d.shape if df_1d is not None else None}")
            if df_1h is None or df_1h.empty:
                self.logger.warning(f"[DEBUG] df_1h boş veya None")
                return None
            # Teknik analiz - Ana zaman dilimi
            ta = TechnicalAnalysis()
            df_1h = ta.calculate_all_indicators(df_1h)
            if df_4h is not None and not df_4h.empty:
                df_4h = ta.calculate_all_indicators(df_4h)
            if df_1d is not None and not df_1d.empty:
                df_1d = ta.calculate_all_indicators(df_1d)
            if df_1h.empty:
                self.logger.warning(f"[DEBUG] df_1h teknik analiz sonrası boş")
                return None
            # Çoklu zaman dilimi analizi
            multi_tf_analysis = ta.multi_timeframe_analysis(df_1h, df_4h, df_1d)
            self.logger.info(f"[DEBUG] multi_timeframe_analysis: {multi_tf_analysis}")
            # AI analizi - Yeni yapı
            ai_model = AIModel()
            ai_result = ai_model.predict_simple(df_1h)
            self.logger.info(f"[DEBUG] ai_result: {ai_result}")
            if ai_result is None:
                self.logger.warning(f"[DEBUG] ai_result None")
                return None
            ai_score = ai_result.get('prediction', 0.0)
            confidence = ai_result.get('confidence', 0.0)
            features_used = ai_result.get('features_used', 0)
            model_performance = ai_result.get('model_performance', {})
            # Teknik analiz gücü - Gelişmiş
            ta_strength = self.calculate_advanced_ta_strength(df_1h, multi_tf_analysis)
            self.logger.info(f"[DEBUG] ta_strength: {ta_strength}")
            # Whale analizi
            try:
                whale_data = detect_whale_trades(symbol)
                whale_score = whale_data.get('whale_score', 0.0)
                whale_direction_score = whale_data.get('whale_direction_score', 0.0)
                order_book_imbalance = whale_data.get('order_book_imbalance', 0.0)
                top_bid_walls = whale_data.get('top_bid_walls', [])
                top_ask_walls = whale_data.get('top_ask_walls', [])
                self.logger.info(f"[DEBUG] whale_data: {whale_data}")
            except Exception as e:
                self.logger.error(f"Whale analizi hatası: {e}")
                whale_score = 0.0
                whale_direction_score = 0.0
                order_book_imbalance = 0.0
                top_bid_walls = []
                top_ask_walls = []
            # Sosyal medya analizi - Pasif
            social_score = 0.0  # Pasif
            # Haber analizi - Pasif
            news_score = 0.0  # Pasif
            # Breakout analizi - Gelişmiş
            breakout_analysis = self.analyze_breakout_advanced(df_1h, multi_tf_analysis)
            self.logger.info(f"[DEBUG] breakout_analysis: {breakout_analysis}")
            # Kalite skoru - Gelişmiş
            quality_score = self.calculate_advanced_quality_score(
                ai_score, ta_strength, whale_score, social_score, news_score,
                breakout_analysis, multi_tf_analysis
            )
            self.logger.info(f"[DEBUG] quality_score: {quality_score}")
            # Sinyal yönü
            signal_direction = self.determine_signal_direction_advanced(
                ai_score, ta_strength, multi_tf_analysis, breakout_analysis
            )
            self.logger.info(f"[DEBUG] signal_direction: {signal_direction}")
            # Ek skorlar (1h ana timeframe'den)
            volume_score = multi_tf_analysis.get('1h', {}).get('volume_score', 0.0)
            momentum_score = multi_tf_analysis.get('1h', {}).get('momentum', 0.0)
            pattern_score = multi_tf_analysis.get('1h', {}).get('pattern_score', 0.0)
            self.logger.info(f"[DEBUG] volume_score: {volume_score}, momentum_score: {momentum_score}, pattern_score: {pattern_score}")
            # Sinyal gücü ve güven seviyesi teknik analizden veya AI'dan alınabilir
            signal_strength = ta.calculate_signal_strength(df_1h) if hasattr(ta, 'calculate_signal_strength') else 0.0
            confidence_level = ai_result.get('confidence', 0.0)
            self.logger.info(f"[DEBUG] signal_strength: {signal_strength}, confidence_level: {confidence_level}")
            self.logger.info(f"[DEBUG][SM] multi_timeframe_analysis 1h: {multi_tf_analysis.get('1h', {})}")
            self.logger.info(f"[DEBUG][SM] df_1h shape: {df_1h.shape}")
            self.logger.info(f"[DEBUG][SM] volume_score: {volume_score}")
            self.logger.info(f"[DEBUG][SM] momentum_score: {momentum_score}")
            self.logger.info(f"[DEBUG][SM] pattern_score: {pattern_score}")
            self.logger.info(f"[DEBUG][SM] signal_strength: {signal_strength}")
            self.logger.info(f"[DEBUG][SM] confidence_level: {confidence_level}")
            return {
                'symbol': symbol,
                'ai_score': ai_score,
                'confidence': confidence,
                'ta_strength': ta_strength,
                'whale_score': whale_score,
                'whale_direction_score': whale_direction_score,
                'order_book_imbalance': order_book_imbalance,
                'top_bid_walls': top_bid_walls,
                'top_ask_walls': top_ask_walls,
                'social_score': social_score,
                'news_score': news_score,
                'breakout_probability': breakout_analysis.get('probability', 0.0),
                'breakout_threshold': breakout_analysis.get('threshold', 0.05),
                'quality_score': quality_score,
                'direction': signal_direction,
                'multi_timeframe': multi_tf_analysis,
                'support_levels': multi_tf_analysis.get('1h', {}).get('support', []),
                'resistance_levels': multi_tf_analysis.get('1h', {}).get('resistance', []),
                'trend_alignment': multi_tf_analysis.get('trend_alignment', 0.0),
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'pattern_score': pattern_score,
                'signal_strength': signal_strength,
                'confidence_level': confidence_level,
                'features_used': features_used,
                'model_performance': model_performance
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Gelişmiş sembol analizi hatası ({symbol}): {e}")
            return None
    
    def calculate_advanced_ta_strength(self, df, multi_tf_analysis):
        """Gelişmiş teknik analiz gücü hesaplama"""
        try:
            if df.empty:
                return 0.0
            
            # Temel TA gücü
            base_ta_strength = self.calculate_ta_strength(df)
            
            # Çoklu zaman dilimi uyumu
            trend_alignment = multi_tf_analysis.get('trend_alignment', 0.0)
            
            # Momentum skoru
            momentum_score = multi_tf_analysis.get('1h', {}).get('momentum', 0.5)
            
            # Trend gücü
            trend_strength = multi_tf_analysis.get('1h', {}).get('strength', 0.5)
            
            # Ağırlıklı ortalama
            advanced_ta_strength = (
                base_ta_strength * 0.3 +
                trend_alignment * 0.3 +
                momentum_score * 0.2 +
                trend_strength * 0.2
            )
            
            return min(advanced_ta_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Gelişmiş TA gücü hesaplama hatası: {e}")
            return 0.5
    
    def analyze_breakout_advanced(self, df, multi_tf_analysis):
        """Gelişmiş breakout analizi"""
        try:
            if df.empty:
                return {'probability': 0.5, 'threshold': 0.05}
            
            current_price = df['close'].iloc[-1]
            
            # Destek ve direnç seviyeleri
            support_levels = multi_tf_analysis.get('1h', {}).get('support', [])
            resistance_levels = multi_tf_analysis.get('1h', {}).get('resistance', [])
            
            # En yakın seviyeleri bul
            nearest_support = max([s for s in support_levels if s < current_price], default=0)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
            
            # Breakout olasılığı hesapla
            breakout_probability = 0.5
            
            # Fiyat seviyelerine yakınlık
            if nearest_resistance != float('inf'):
                distance_to_resistance = (nearest_resistance - current_price) / current_price
                if distance_to_resistance < 0.02:  # %2'den yakın
                    breakout_probability += 0.2
                elif distance_to_resistance < 0.05:  # %5'ten yakın
                    breakout_probability += 0.1
            
            if nearest_support > 0:
                distance_to_support = (current_price - nearest_support) / current_price
                if distance_to_support < 0.02:  # %2'den yakın
                    breakout_probability += 0.2
                elif distance_to_support < 0.05:  # %5'ten yakın
                    breakout_probability += 0.1
            
            # Trend gücü etkisi
            trend_strength = multi_tf_analysis.get('1h', {}).get('strength', 0.5)
            breakout_probability += trend_strength * 0.2
            
            # Momentum etkisi
            momentum_score = multi_tf_analysis.get('1h', {}).get('momentum', 0.5)
            breakout_probability += momentum_score * 0.1
            
            # Volatilite etkisi
            if 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_percent = atr / current_price
                if atr_percent > 0.05:  # Yüksek volatilite
                    breakout_probability += 0.1
            
            # Breakout threshold
            threshold = 0.03  # %3 varsayılan
            if 'bb_width' in df.columns:
                bb_width = df['bb_width'].iloc[-1]
                threshold = min(bb_width * 0.5, 0.05)  # Bollinger genişliğinin yarısı
            
            return {
                'probability': min(breakout_probability, 1.0),
                'threshold': threshold,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
        except Exception as e:
            self.logger.error(f"Gelişmiş breakout analizi hatası: {e}")
            return {'probability': 0.5, 'threshold': 0.05}
    
    def calculate_advanced_quality_score(self, ai_score, ta_strength, whale_score, 
                                       social_score, news_score, breakout_analysis, multi_tf_analysis):
        """Gelişmiş kalite skoru hesaplama"""
        try:
            # Temel kalite skoru (sadece AI, TA, Whale)
            base_quality = self.calculate_signal_quality_score({
                'ai_score': ai_score,
                'ta_strength': ta_strength,
                'whale_score': whale_score
            })
            # Breakout bonusu
            breakout_bonus = 0.0
            if breakout_analysis.get('probability', 0) > 0.7:
                breakout_bonus = 0.1
            elif breakout_analysis.get('probability', 0) > 0.6:
                breakout_bonus = 0.05
            # Trend uyumu bonusu
            trend_alignment = multi_tf_analysis.get('trend_alignment', 0.0)
            trend_bonus = trend_alignment * 0.1
            # Momentum bonusu
            momentum_score = multi_tf_analysis.get('1h', {}).get('momentum', 0.5)
            momentum_bonus = (momentum_score - 0.5) * 0.1
            # Toplam kalite skoru
            advanced_quality = base_quality + breakout_bonus + trend_bonus + momentum_bonus
            return min(advanced_quality, 1.0)
        except Exception as e:
            self.logger.error(f"Gelişmiş kalite skoru hesaplama hatası: {e}")
            return 0.5
    
    def determine_signal_direction_advanced(self, ai_score, ta_strength, multi_tf_analysis, breakout_analysis):
        """Gelişmiş sinyal yönü belirleme"""
        try:
            # AI skoru bazlı yön
            if ai_score > 0.7:
                ai_direction = 'BUY'
            elif ai_score < 0.3:
                ai_direction = 'SELL'
            else:
                ai_direction = 'NEUTRAL'
            
            # Trend yönü
            trend_direction = multi_tf_analysis.get('1h', {}).get('trend', 'neutral')
            if 'up' in trend_direction:
                trend_signal = 'BUY'
            elif 'down' in trend_direction:
                trend_signal = 'SELL'
            else:
                trend_signal = 'NEUTRAL'
            
            # Breakout yönü
            breakout_direction = 'NEUTRAL'
            if breakout_analysis.get('probability', 0) > 0.6:
                current_price = 1.0  # Normalize edilmiş
                nearest_resistance = breakout_analysis.get('nearest_resistance', 1.0)
                nearest_support = breakout_analysis.get('nearest_support', 0.0)
                
                resistance_distance = abs(nearest_resistance - current_price)
                support_distance = abs(current_price - nearest_support)
                
                if resistance_distance < support_distance:
                    breakout_direction = 'BUY'
                else:
                    breakout_direction = 'SELL'
            
            # Ağırlıklı karar
            buy_votes = 0
            sell_votes = 0
            
            if ai_direction == 'BUY':
                buy_votes += 2
            elif ai_direction == 'SELL':
                sell_votes += 2
            
            if trend_signal == 'BUY':
                buy_votes += 1
            elif trend_signal == 'SELL':
                sell_votes += 1
            
            if breakout_direction == 'BUY':
                buy_votes += 1
            elif breakout_direction == 'SELL':
                sell_votes += 1
            
            # Final karar
            if buy_votes > sell_votes:
                return 'BUY'
            elif sell_votes > buy_votes:
                return 'SELL'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"Gelişmiş sinyal yönü belirleme hatası: {e}")
            return 'NEUTRAL'

    def get_recent_signals_by_symbol(self, symbol, hours=2):  # 24'ten 2'ye düşürüldü
        """Belirli bir sembol için son X saatteki sinyalleri getir"""
        try:
            query = f"""
                SELECT * FROM signals 
                WHERE symbol = :symbol
                AND timestamp::timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp::timestamp DESC
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'symbol': symbol})
                signals = [dict(row._mapping) for row in result.fetchall()]
                
            self.logger.info(f"{symbol} için son {hours} saatte {len(signals)} sinyal bulundu")
            return signals
            
        except Exception as e:
            self.logger.error(f"{symbol} için son sinyaller alınamadı: {e}")
            return []

    def clear_all_signals(self):
        """Veritabanındaki tüm sinyalleri sil"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM signals"))
                conn.commit()
            self.logger.info("Tüm sinyaller veritabanından silindi.")
        except Exception as e:
            self.logger.error(f"Sinyaller silinirken hata: {e}")