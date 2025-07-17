import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import Config
from modules.signal_manager import SignalManager
from sqlalchemy import create_engine, text

class PerformanceAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.signal_manager = SignalManager()
        self.db_url = Config.DATABASE_URL
        self.engine = create_engine(self.db_url)

    def analyze_performance(self, start_date=None, end_date=None):
        """Genel performans analizi"""
        try:
            df = self.signal_manager.load_signals(start_date, end_date)
            if df.empty:
                return {
                    'total_signals': 0,
                    'success_rate': 0,
                    'avg_gain': 0,
                    'avg_duration': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'best_coin': None,
                    'worst_coin': None,
                    'signals': []
                }
            
            # Sonuçları analiz et
            total = len(df)
            completed_signals = df[df['result'].notna()]
            
            if completed_signals.empty:
                return {
                    'total_signals': total,
                    'success_rate': 0,
                    'avg_gain': 0,
                    'avg_duration': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'best_coin': None,
                    'worst_coin': None,
                    'signals': df.to_dict(orient='records')
                }
            
            # Başarı oranı
            profitable_signals = completed_signals[completed_signals['result'] == 'profit']
            success_rate = (len(profitable_signals) / len(completed_signals)) * 100
            
            # Ortalama kazanç/kayıp
            avg_gain = completed_signals['realized_gain'].mean()
            avg_duration = completed_signals['duration'].mean()
            
            # Toplam kazanç/kayıp
            total_profit = profitable_signals['realized_gain'].sum() if not profitable_signals.empty else 0
            loss_signals = completed_signals[completed_signals['result'] == 'loss']
            total_loss = loss_signals['realized_gain'].sum() if not loss_signals.empty else 0
            
            # En iyi/kötü coin
            coin_performance = completed_signals.groupby('symbol')['realized_gain'].mean()
            best_coin = coin_performance.idxmax() if not coin_performance.empty else None
            worst_coin = coin_performance.idxmin() if not coin_performance.empty else None
            
            # Performans özeti
            summary = {
                'total_signals': total,
                'profitable_signals': profitable_signals,
                'loss_signals': loss_signals,
                'success_rate': success_rate,
                'avg_profit': avg_gain,
                'avg_loss': avg_gain,
                'avg_total_return': avg_gain,
                'total_pnl': total_profit,
                'last_updated': str(datetime.now())
            }
            
            return summary
        except Exception as e:
            self.logger.error(f"Performans analizi hatası: {e}")
            return {
                'error': str(e),
                'total_signals': 0,
                'success_rate': 0,
                'avg_gain': 0,
                'avg_duration': 0,
                'signals': []
            }

    def coin_performance(self, coin, start_date=None, end_date=None):
        """Belirli bir coin için performans analizi"""
        try:
            df = self.signal_manager.load_signals(start_date, end_date)
            if df.empty:
                return None
            
            df = df[df['symbol'] == coin]
            if df.empty:
                return None
            
            completed_signals = df[df['result'].notna()]
            if completed_signals.empty:
                return {
                    'coin': coin,
                    'total_signals': len(df),
                    'completed_signals': 0,
                    'success_rate': 0,
                    'avg_gain': 0,
                    'avg_duration': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'net_profit': 0
                }
            
            total = len(df)
            profitable_signals = completed_signals[completed_signals['result'] == 'profit']
            success_rate = (len(profitable_signals) / len(completed_signals)) * 100
            
            avg_gain = completed_signals['realized_gain'].mean()
            avg_duration = completed_signals['duration'].mean()
            
            total_profit = profitable_signals['realized_gain'].sum() if not profitable_signals.empty else 0
            loss_signals = completed_signals[completed_signals['result'] == 'loss']
            total_loss = loss_signals['realized_gain'].sum() if not loss_signals.empty else 0
            
            return {
                'coin': coin,
                'total_signals': total,
                'completed_signals': len(completed_signals),
                'success_rate': float(round(success_rate, 2)),
                'avg_gain': float(round(avg_gain, 2)) if pd.notna(avg_gain) else 0.0,
                'avg_duration': float(round(avg_duration, 2)) if pd.notna(avg_duration) else 0.0,
                'total_profit': float(round(total_profit, 2)),
                'total_loss': float(round(total_loss, 2)),
                'net_profit': float(round(total_profit + total_loss, 2)),
                'best_coin': best_coin,
                'worst_coin': worst_coin,
                'signals': df.to_dict(orient='records')
            }
        except Exception as e:
            self.logger.error(f"Coin performans analizi hatası: {e}")
            return None

    def get_daily_performance(self, days=30):
        """Günlük performans analizi"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.signal_manager.load_signals(start_date, end_date)
            if df.empty:
                return []
            
            # Günlük gruplama - güvenli şekilde
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Boş değerleri kontrol et
            if 'result' not in df.columns or 'realized_gain' not in df.columns:
                return []
            
            # NaN değerleri temizle
            df_clean = df.dropna(subset=['result', 'realized_gain'])
            
            if df_clean.empty:
                return []
            
            daily_stats = df_clean.groupby('date').agg({
                'result': lambda x: (x == 'profit').sum() / len(x) * 100 if len(x) > 0 else 0,
                'realized_gain': 'sum',
                'symbol': 'count'
            }).reset_index()
            
            daily_stats.columns = ['date', 'success_rate', 'total_gain', 'signal_count']
            daily_stats['success_rate'] = daily_stats['success_rate'].apply(lambda x: float(round(x, 2)))
            daily_stats['total_gain'] = daily_stats['total_gain'].apply(lambda x: float(round(x, 2)))
            
            return daily_stats.to_dict(orient='records')
        except Exception as e:
            self.logger.error(f"Günlük performans analizi hatası: {e}")
            return []

    def get_top_coins(self, limit=10):
        """En iyi performans gösteren coinler"""
        try:
            df = self.signal_manager.load_signals()
            if df.empty:
                return []
            
            completed_signals = df[df['result'].notna()]
            if completed_signals.empty:
                return []
            
            coin_stats = completed_signals.groupby('symbol').agg({
                'realized_gain': ['mean', 'sum', 'count'],
                'result': lambda x: (x == 'profit').sum() / len(x) * 100
            })
            
            coin_stats.columns = ['avg_gain', 'total_gain', 'signal_count', 'success_rate']
            coin_stats = coin_stats.sort_values('avg_gain', ascending=False).head(limit)
            
            # Round values safely
            coin_stats['avg_gain'] = coin_stats['avg_gain'].apply(lambda x: float(round(x, 2)) if pd.notna(x) else 0.0)
            coin_stats['total_gain'] = coin_stats['total_gain'].apply(lambda x: float(round(x, 2)) if pd.notna(x) else 0.0)
            coin_stats['success_rate'] = coin_stats['success_rate'].apply(lambda x: float(round(x, 2)) if pd.notna(x) else 0.0)
            
            return coin_stats.reset_index().to_dict(orient='records')
        except Exception as e:
            self.logger.error(f"Top coins analizi hatası: {e}")
            return []

    def get_performance_summary(self):
        """Özet performans raporu"""
        try:
            # Son 30 günlük performans
            recent_performance = self.analyze_performance(
                start_date=datetime.now() - timedelta(days=30)
            )
            
            # Günlük performans
            daily_performance = self.get_daily_performance(days=7)
            
            # En iyi coinler
            top_coins = self.get_top_coins(limit=5)
            
            return {
                'recent_performance': recent_performance,
                'daily_performance': daily_performance,
                'top_coins': top_coins,
                'last_updated': str(datetime.now())
            }
        except Exception as e:
            self.logger.error(f"Performans özeti hatası: {e}")
            return {'error': str(e)}

    def get_signals_last_24h(self):
        """Son 24 saatteki sinyalleri al"""
        try:
            query = """
                SELECT * FROM signals
                WHERE timestamp::timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY timestamp::timestamp DESC
            """
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Son 24 saatteki sinyaller alınamadı: {e}")
            return pd.DataFrame()

    def real_time_monitoring(self):
        """Real-time performance monitoring"""
        try:
            # Son 24 saatteki performans
            recent_performance = self.get_recent_performance()
            
            # System health check
            system_health = self.check_system_health()
            
            # Auto-tuning recommendations
            tuning_recommendations = self.generate_tuning_recommendations(recent_performance)
            
            # Performance alerts
            alerts = self.check_performance_alerts(recent_performance)
            
            return {
                'recent_performance': recent_performance,
                'system_health': system_health,
                'tuning_recommendations': tuning_recommendations,
                'alerts': alerts,
                'timestamp': str(datetime.now())
            }
            
        except Exception as e:
            self.logger.error(f"Real-time monitoring hatası: {e}")
            return {}
    
    def get_recent_performance(self):
        """Son 24 saatteki performans metrikleri"""
        try:
            # Son 24 saatteki sinyaller
            recent_signals = self.get_signals_last_24h()
            
            if recent_signals.empty:
                return {
                    'total_signals': 0,
                    'success_rate': 0.0,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Success rate
            successful_signals = recent_signals[recent_signals['result'] == 'SUCCESS']
            success_rate = len(successful_signals) / len(recent_signals) if len(recent_signals) > 0 else 0.0
            
            # Profit/Loss metrics
            profits = recent_signals[recent_signals['realized_gain'] > 0]['realized_gain']
            losses = recent_signals[recent_signals['realized_gain'] < 0]['realized_gain']
            
            avg_profit = profits.mean() if len(profits) > 0 else 0.0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
            profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0.0
            
            # Sharpe ratio (simplified)
            returns = recent_signals['realized_gain'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'total_signals': len(recent_signals),
                'success_rate': success_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Recent performance calculation hatası: {e}")
            return {}
    
    def check_system_health(self):
        """Sistem sağlığını kontrol et ve öneriler sun"""
        try:
            # Son 24 saat performansını al
            from modules.performance import PerformanceTracker
            tracker = PerformanceTracker()
            performance = tracker.calculate_signal_performance(24)
            
            # Sistem durumu analizi
            health_status = {
                'overall_status': 'HEALTHY',
                'issues': [],
                'recommendations': [],
                'performance_metrics': performance
            }
            
            # Kritik sorunları kontrol et
            if performance.get('total_signals', 0) == 0:
                health_status['overall_status'] = 'CRITICAL'
                health_status['issues'].append('Hiç sinyal üretilmiyor')
                health_status['recommendations'].append('Ana sistem scriptini kontrol edin')
            
            # AI skor sorunları
            avg_ai_score = performance.get('avg_ai_score', 0)
            if avg_ai_score < 0.3:
                health_status['overall_status'] = 'WARNING'
                health_status['issues'].append(f'AI skor çok düşük: {avg_ai_score:.3f}')
                health_status['recommendations'].append('AI modelini yeniden eğitin')
            
            # Başarı oranı sorunları
            success_rate = performance.get('success_rate', 0)
            if success_rate < 30 and performance.get('closed_signals', 0) > 10:
                health_status['overall_status'] = 'WARNING'
                health_status['issues'].append(f'Başarı oranı çok düşük: %{success_rate:.1f}')
                health_status['recommendations'].append('Sinyal kriterlerini sıkılaştırın')
            
            # Sinyal kalitesi sorunları
            avg_ta_strength = performance.get('avg_ta_strength', 0)
            if avg_ta_strength < 0.4:
                health_status['issues'].append(f'Teknik analiz gücü düşük: {avg_ta_strength:.3f}')
                health_status['recommendations'].append('Teknik analiz parametrelerini gözden geçirin')
            
            # Whale aktivitesi sorunları
            avg_whale_score = performance.get('avg_whale_score', 0)
            if avg_whale_score < 0.2:
                health_status['issues'].append(f'Whale aktivitesi düşük: {avg_whale_score:.3f}')
                health_status['recommendations'].append('Whale tracking parametrelerini ayarlayın')
            
            # Sistem önerileri
            if health_status['overall_status'] == 'HEALTHY':
                health_status['recommendations'].append('Sistem sağlıklı çalışıyor')
                health_status['recommendations'].append('Performansı izlemeye devam edin')
            
            return health_status
        except Exception as e:
            return {
                'overall_status': 'ERROR',
                'issues': [f'Sistem sağlığı kontrolü hatası: {str(e)}'],
                'recommendations': ['Sistem loglarını kontrol edin'],
                'performance_metrics': {}
            }

    def generate_tuning_recommendations(self, performance):
        """Performance'a göre tuning önerileri"""
        try:
            recommendations = []
            
            # Success rate based recommendations
            success_rate = performance.get('success_rate', 0.0)
            if success_rate < 0.4:
                recommendations.append({
                    'type': 'threshold_increase',
                    'message': 'Success rate düşük. MIN_SIGNAL_CONFIDENCE artırılmalı.',
                    'current_value': Config.MIN_SIGNAL_CONFIDENCE,
                    'recommended_value': min(0.8, Config.MIN_SIGNAL_CONFIDENCE + 0.1)
                })
            elif success_rate > 0.7:
                recommendations.append({
                    'type': 'threshold_decrease',
                    'message': 'Success rate yüksek. Daha fazla sinyal üretilebilir.',
                    'current_value': Config.MIN_SIGNAL_CONFIDENCE,
                    'recommended_value': max(0.5, Config.MIN_SIGNAL_CONFIDENCE - 0.05)
                })
            
            # Profit factor based recommendations
            profit_factor = performance.get('profit_factor', 0.0)
            if profit_factor < 1.2:
                recommendations.append({
                    'type': 'risk_management',
                    'message': 'Profit factor düşük. Risk yönetimi iyileştirilmeli.',
                    'action': 'Stop-loss ve take-profit seviyeleri gözden geçirilmeli'
                })
            
            # Drawdown based recommendations
            max_drawdown = performance.get('max_drawdown', 0.0)
            if abs(max_drawdown) > 0.1:  # 10% drawdown
                recommendations.append({
                    'type': 'risk_reduction',
                    'message': 'Maximum drawdown yüksek. Risk azaltılmalı.',
                    'action': 'Position sizing ve stop-loss seviyeleri sıkılaştırılmalı'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Tuning recommendations hatası: {e}")
            return []
    
    def check_performance_alerts(self, performance):
        """Performance alert'leri kontrol et"""
        try:
            alerts = []
            
            # Success rate alert
            success_rate = performance.get('success_rate', 0.0)
            if success_rate < 0.3:
                alerts.append({
                    'level': 'critical',
                    'message': f'Success rate kritik seviyede: {success_rate:.2%}',
                    'action': 'Acil müdahale gerekli'
                })
            elif success_rate < 0.5:
                alerts.append({
                    'level': 'warning',
                    'message': f'Success rate düşük: {success_rate:.2%}',
                    'action': 'Sistem parametreleri gözden geçirilmeli'
                })
            
            # Profit factor alert
            profit_factor = performance.get('profit_factor', 0.0)
            if profit_factor < 1.0:
                alerts.append({
                    'level': 'warning',
                    'message': f'Profit factor 1.0 altında: {profit_factor:.2f}',
                    'action': 'Risk yönetimi iyileştirilmeli'
                })
            
            # Drawdown alert
            max_drawdown = performance.get('max_drawdown', 0.0)
            if abs(max_drawdown) > 0.15:  # 15% drawdown
                alerts.append({
                    'level': 'critical',
                    'message': f'Maximum drawdown kritik: {max_drawdown:.2%}',
                    'action': 'Trading durdurulmalı ve sistem gözden geçirilmeli'
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Performance alerts hatası: {e}")
            return []
    
    def auto_tune_parameters(self, recommendations):
        """Otomatik parameter tuning"""
        try:
            tuned_params = {}
            
            for rec in recommendations:
                if rec['type'] == 'threshold_increase':
                    new_value = rec['recommended_value']
                    tuned_params['MIN_SIGNAL_CONFIDENCE'] = new_value
                    self.logger.info(f"MIN_SIGNAL_CONFIDENCE artırıldı: {new_value}")
                    
                elif rec['type'] == 'threshold_decrease':
                    new_value = rec['recommended_value']
                    tuned_params['MIN_SIGNAL_CONFIDENCE'] = new_value
                    self.logger.info(f"MIN_SIGNAL_CONFIDENCE azaltıldı: {new_value}")
            
            return tuned_params
            
        except Exception as e:
            self.logger.error(f"Auto-tuning hatası: {e}")
            return {}

    def auto_tune_system(self):
        """Sistem otomatik ayarlama"""
        try:
            self.logger.info("Sistem otomatik ayarlama başlatılıyor...")
            
            # Son performans analizi
            recent_performance = self.get_recent_performance()
            
            # Başarı oranı düşükse ayarlamalar yap
            if recent_performance['success_rate'] < 0.6:
                self.adjust_signal_thresholds()
                self.adjust_ai_parameters()
                self.adjust_technical_parameters()
                
            # Volatilite analizi
            if recent_performance['avg_volatility'] > 0.15:
                self.adjust_risk_parameters()
                
            # Sinyal sayısı çok fazlaysa filtreleme artır
            if recent_performance['signals_per_hour'] > 50:
                self.increase_signal_filtering()
                
            self.logger.info("Sistem otomatik ayarlama tamamlandı")
            
        except Exception as e:
            self.logger.error(f"Auto-tuning hatası: {e}")

    def adjust_signal_thresholds(self):
        """Sinyal eşiklerini ayarla"""
        try:
            # Başarı oranına göre eşikleri ayarla
            success_rate = self.get_recent_performance()['success_rate']
            
            if success_rate < 0.5:
                # Daha sıkı filtreleme
                Config.MIN_AI_SCORE = min(0.8, Config.MIN_AI_SCORE + 0.05)
                Config.MIN_TA_STRENGTH = min(0.8, Config.MIN_TA_STRENGTH + 0.05)
                Config.MIN_WHALE_SCORE = min(0.5, Config.MIN_WHALE_SCORE + 0.05)
            elif success_rate > 0.7:
                # Daha gevşek filtreleme
                Config.MIN_AI_SCORE = max(0.4, Config.MIN_AI_SCORE - 0.02)
                Config.MIN_TA_STRENGTH = max(0.4, Config.MIN_TA_STRENGTH - 0.02)
                Config.MIN_WHALE_SCORE = max(0.2, Config.MIN_WHALE_SCORE - 0.02)
                
            self.logger.info(f"Sinyal eşikleri ayarlandı: AI={Config.MIN_AI_SCORE}, TA={Config.MIN_TA_STRENGTH}")
            
        except Exception as e:
            self.logger.error(f"Signal threshold ayarlama hatası: {e}")

    def adjust_ai_parameters(self):
        """AI model parametrelerini ayarla"""
        try:
            # Model performansına göre parametreleri ayarla
            model_performance = self.get_model_performance()
            
            if model_performance['accuracy'] < 0.6:
                # Daha fazla eğitim verisi topla
                Config.MIN_TRAINING_DATA = min(10000, Config.MIN_TRAINING_DATA + 1000)
                # Daha sık yeniden eğitim
                Config.MODEL_RETRAIN_INTERVAL = max(6, Config.MODEL_RETRAIN_INTERVAL - 6)
                
            self.logger.info("AI parametreleri ayarlandı")
            
        except Exception as e:
            self.logger.error(f"AI parametre ayarlama hatası: {e}")

    def adjust_technical_parameters(self):
        """Teknik analiz parametrelerini ayarla"""
        try:
            # Market koşullarına göre parametreleri ayarla
            market_conditions = self.get_market_conditions()
            
            if market_conditions['volatility'] > 0.2:
                # Yüksek volatilite için daha kısa periyotlar
                Config.RSI_PERIOD = max(7, Config.RSI_PERIOD - 2)
                Config.MACD_FAST = max(8, Config.MACD_FAST - 1)
                Config.MACD_SLOW = max(18, Config.MACD_SLOW - 2)
            else:
                # Düşük volatilite için daha uzun periyotlar
                Config.RSI_PERIOD = min(21, Config.RSI_PERIOD + 2)
                Config.MACD_FAST = min(12, Config.MACD_FAST + 1)
                Config.MACD_SLOW = min(26, Config.MACD_SLOW + 2)
                
            self.logger.info("Teknik analiz parametreleri ayarlandı")
            
        except Exception as e:
            self.logger.error(f"Technical parametre ayarlama hatası: {e}")

    def adjust_risk_parameters(self):
        """Risk parametrelerini ayarla"""
        try:
            # Volatiliteye göre risk ayarlaması
            volatility = self.get_market_conditions()['volatility']
            
            if volatility > 0.15:
                # Yüksek volatilite - daha düşük risk
                Config.PROFIT_TARGET = max(0.03, Config.PROFIT_TARGET - 0.01)
                Config.STOP_LOSS = max(0.02, Config.STOP_LOSS - 0.005)
            else:
                # Düşük volatilite - daha yüksek risk
                Config.PROFIT_TARGET = min(0.08, Config.PROFIT_TARGET + 0.01)
                Config.STOP_LOSS = min(0.04, Config.STOP_LOSS + 0.005)
                
            self.logger.info("Risk parametreleri ayarlandı")
            
        except Exception as e:
            self.logger.error(f"Risk parametre ayarlama hatası: {e}")

    def increase_signal_filtering(self):
        """Sinyal filtrelemeyi artır"""
        try:
            # Sinyal sayısını azaltmak için filtreleme artır
            Config.MAX_SIGNALS_PER_BATCH = max(5, Config.MAX_SIGNALS_PER_BATCH - 2)
            Config.MIN_AI_SCORE = min(0.9, Config.MIN_AI_SCORE + 0.05)
            Config.MIN_TA_STRENGTH = min(0.9, Config.MIN_TA_STRENGTH + 0.05)
            
            self.logger.info("Sinyal filtreleme artırıldı")
            
        except Exception as e:
            self.logger.error(f"Signal filtering artırma hatası: {e}")

    def get_model_performance(self):
        """Model performansını al"""
        try:
            # Son model performansını hesapla
            recent_signals = self.get_recent_signals(24)  # Son 24 saat
            
            if not recent_signals:
                return {'accuracy': 0, 'precision': 0, 'recall': 0}
            
            correct_predictions = 0
            total_predictions = 0
            
            for signal in recent_signals:
                if signal['result'] in ['PROFIT', 'LOSS']:
                    total_predictions += 1
                    if signal['result'] == 'PROFIT':
                        correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            return {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions
            }
            
        except Exception as e:
            self.logger.error(f"Model performans hesaplama hatası: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0}

    def get_market_conditions(self):
        """Market koşullarını al"""
        try:
            # Market koşullarını analiz et
            btc_data = self.get_btc_data()
            
            if btc_data is None or len(btc_data) < 100:
                return {'volatility': 0.1, 'trend': 'sideways', 'volume': 'normal'}
            
            # Volatilite hesapla
            returns = btc_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Trend hesapla
            sma_20 = btc_data['close'].rolling(20).mean().iloc[-1]
            current_price = btc_data['close'].iloc[-1]
            
            if current_price > sma_20 * 1.02:
                trend = 'bullish'
            elif current_price < sma_20 * 0.98:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            # Volume analizi
            avg_volume = btc_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = btc_data['volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                volume = 'high'
            elif current_volume < avg_volume * 0.5:
                volume = 'low'
            else:
                volume = 'normal'
            
            return {
                'volatility': volatility,
                'trend': trend,
                'volume': volume
            }
            
        except Exception as e:
            self.logger.error(f"Market conditions hesaplama hatası: {e}")
            return {'volatility': 0.1, 'trend': 'sideways', 'volume': 'normal'}

    def system_health_check(self):
        """Sistem sağlığı kontrolü ve performans analizi"""
        try:
            health_status = {
                'overall_status': 'HEALTHY',
                'checks': {},
                'recommendations': [],
                'timestamp': str(datetime.now())
            }
            
            # 1. Veritabanı sağlığı kontrolü
            db_health = self._check_database_health()
            health_status['checks']['database'] = db_health
            
            # 2. Model sağlığı kontrolü
            model_health = self._check_model_health()
            health_status['checks']['models'] = model_health
            
            # 3. Sinyal kalitesi kontrolü
            signal_health = self._check_signal_quality()
            health_status['checks']['signals'] = signal_health
            
            # 4. Sistem performansı kontrolü
            performance_health = self._check_system_performance()
            health_status['checks']['performance'] = performance_health
            
            # 5. Genel durum değerlendirmesi
            failed_checks = sum(1 for check in health_status['checks'].values() 
                              if check.get('status') == 'FAILED')
            
            if failed_checks > 2:
                health_status['overall_status'] = 'CRITICAL'
            elif failed_checks > 0:
                health_status['overall_status'] = 'WARNING'
            
            # 6. Öneriler oluştur
            health_status['recommendations'] = self._generate_recommendations(health_status['checks'])
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Sistem sağlığı kontrolü hatası: {e}")
            return {
                'overall_status': 'ERROR',
                'error': str(e),
                'timestamp': str(datetime.now())
            }
    
    def _check_database_health(self):
        """Veritabanı sağlığı kontrolü"""
        try:
            # Sinyal sayısı kontrolü
            df = self.signal_manager.load_signals()
            total_signals = len(df)
            
            # Son 24 saatteki sinyaller
            recent_signals = df[pd.to_datetime(df['timestamp']) > datetime.now() - timedelta(hours=24)]
            recent_count = len(recent_signals)
            
            # Veritabanı boyutu kontrolü (yaklaşık)
            db_size_ok = total_signals < 10000  # 10k sinyal limiti
            
            # Son sinyal zamanı kontrolü
            if not df.empty:
                latest_time = pd.to_datetime(df['timestamp'].max())
                time_since_last = (datetime.now() - latest_time).total_seconds() / 3600
            else:
                time_since_last = 999
            
            recent_activity_ok = time_since_last < 6  # 6 saat içinde sinyal olmalı
            
            status = 'HEALTHY'
            if not db_size_ok or not recent_activity_ok:
                status = 'WARNING'
            
            return {
                'status': status,
                'total_signals': total_signals,
                'recent_signals_24h': recent_count,
                'time_since_last_signal': time_since_last,
                'db_size_ok': db_size_ok,
                'recent_activity_ok': recent_activity_ok
            }
            
        except Exception as e:
            self.logger.error(f"Veritabanı sağlığı kontrolü hatası: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def _check_model_health(self):
        """Model sağlığı kontrolü"""
        try:
            import os
            
            # Model dosyalarının varlığı kontrolü
            model_files = ['models/lstm_model.h5', 'models/feature_cols.pkl']
            missing_models = []
            
            for model_file in model_files:
                if not os.path.exists(model_file):
                    missing_models.append(model_file)
            
            # Model performans kontrolü (son 100 sinyal)
            df = self.signal_manager.load_signals()
            if not df.empty:
                recent_signals = df.tail(100)
                completed_signals = recent_signals[recent_signals['result'].notna()]
                
                if not completed_signals.empty:
                    accuracy = (completed_signals['result'] == 'profit').mean() * 100
                    accuracy_ok = accuracy > 50  # %50'den fazla doğruluk
                else:
                    accuracy = 0
                    accuracy_ok = True  # Henüz tamamlanmış sinyal yoksa OK
            else:
                accuracy = 0
                accuracy_ok = True
            
            status = 'HEALTHY'
            if missing_models or not accuracy_ok:
                status = 'WARNING'
            
            return {
                'status': status,
                'missing_models': missing_models,
                'recent_accuracy': accuracy,
                'accuracy_ok': accuracy_ok,
                'models_loaded': len(model_files) - len(missing_models)
            }
            
        except Exception as e:
            self.logger.error(f"Model sağlığı kontrolü hatası: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def _check_signal_quality(self):
        """Sinyal kalitesi kontrolü"""
        try:
            # Son 50 sinyalin kalitesi
            df = self.signal_manager.load_signals()
            if df.empty:
                return {
                    'status': 'HEALTHY',
                    'avg_quality_score': 0,
                    'high_quality_signals': 0,
                    'total_signals': 0
                }
            
            recent_signals = df.tail(50)
            
            # Kalite skorları (basit hesaplama)
            quality_scores = []
            high_quality_count = 0
            
            for _, signal in recent_signals.iterrows():
                # Basit kalite skoru hesaplama
                score = 50  # Başlangıç skoru
                
                # Confidence'a göre puan
                if 'confidence' in signal and pd.notna(signal['confidence']):
                    score += signal['confidence'] * 20
                
                # AI score'a göre puan
                if 'ai_score' in signal and pd.notna(signal['ai_score']):
                    score += signal['ai_score'] * 20
                
                quality_scores.append(score)
                if score >= 70:  # Yüksek kalite eşiği
                    high_quality_count += 1
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            quality_ok = avg_quality > 50  # Ortalama 50'den fazla
            
            status = 'HEALTHY'
            if not quality_ok:
                status = 'WARNING'
            
            return {
                'status': status,
                'avg_quality_score': avg_quality,
                'high_quality_signals': high_quality_count,
                'total_signals': len(recent_signals),
                'quality_ok': quality_ok
            }
            
        except Exception as e:
            self.logger.error(f"Sinyal kalitesi kontrolü hatası: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def _check_system_performance(self):
        """Sistem performansı kontrolü"""
        try:
            # Basit performans kontrolü (psutil olmadan)
            # Gerçek uygulamada psutil kullanılabilir
            
            # Log dosyası boyutu kontrolü
            import os
            log_size = 0
            if os.path.exists('logs'):
                for filename in os.listdir('logs'):
                    if filename.endswith('.log'):
                        filepath = os.path.join('logs', filename)
                        log_size += os.path.getsize(filepath)
            
            # Log boyutu kontrolü (100MB limit)
            log_size_ok = log_size < 100 * 1024 * 1024
            
            # Sinyal dosyaları kontrolü
            signal_files = 0
            if os.path.exists('signals'):
                signal_files = len([f for f in os.listdir('signals') if f.endswith('.json')])
            
            signal_files_ok = signal_files < 1000  # 1000 sinyal dosyası limiti
            
            status = 'HEALTHY'
            if not log_size_ok or not signal_files_ok:
                status = 'WARNING'
            
            return {
                'status': status,
                'log_size_mb': log_size / (1024 * 1024),
                'signal_files': signal_files,
                'log_size_ok': log_size_ok,
                'signal_files_ok': signal_files_ok
            }
            
        except Exception as e:
            self.logger.error(f"Sistem performansı kontrolü hatası: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def _generate_recommendations(self, checks):
        """Sağlık kontrolü sonuçlarına göre öneriler oluştur"""
        recommendations = []
        
        # Veritabanı önerileri
        if 'database' in checks:
            db_check = checks['database']
            if not db_check.get('recent_activity_ok', True):
                recommendations.append("Son 6 saatte sinyal üretilmemiş. Sistem kontrolü gerekli.")
            if not db_check.get('db_size_ok', True):
                recommendations.append("Veritabanı boyutu büyük. Eski sinyaller temizlenebilir.")
        
        # Model önerileri
        if 'models' in checks:
            model_check = checks['models']
            if model_check.get('missing_models'):
                recommendations.append("Bazı model dosyaları eksik. Model yeniden eğitimi gerekli.")
            if not model_check.get('accuracy_ok', True):
                recommendations.append("Model doğruluğu düşük. Model parametreleri optimize edilebilir.")
        
        # Sinyal kalitesi önerileri
        if 'signals' in checks:
            signal_check = checks['signals']
            if not signal_check.get('quality_ok', True):
                recommendations.append("Sinyal kalitesi düşük. Filtreleme kriterleri sıkılaştırılabilir.")
        
        # Performans önerileri
        if 'performance' in checks:
            perf_check = checks['performance']
            if not perf_check.get('log_size_ok', True):
                recommendations.append("Log dosyaları büyük. Eski loglar temizlenebilir.")
            if not perf_check.get('signal_files_ok', True):
                recommendations.append("Çok fazla sinyal dosyası var. Eski dosyalar arşivlenebilir.")
        
        return recommendations
    
    def auto_optimize_system(self):
        """Sistem otomatik optimizasyonu"""
        try:
            optimizations = []
            
            # 1. Eski log dosyalarını temizle
            log_cleanup = self._cleanup_old_logs()
            if log_cleanup:
                optimizations.append(log_cleanup)
            
            # 2. Eski sinyal dosyalarını arşivle
            signal_cleanup = self._archive_old_signals()
            if signal_cleanup:
                optimizations.append(signal_cleanup)
            
            return {
                'success': True,
                'optimizations_applied': optimizations,
                'timestamp': str(datetime.now())
            }
            
        except Exception as e:
            self.logger.error(f"Sistem optimizasyonu hatası: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': str(datetime.now())
            }
    
    def _cleanup_old_logs(self):
        """Eski log dosyalarını temizle"""
        try:
            import os
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                return None
            
            # 7 günden eski log dosyalarını sil
            cutoff_date = datetime.now() - timedelta(days=7)
            deleted_files = []
            
            for filename in os.listdir(log_dir):
                if filename.endswith('.log'):
                    filepath = os.path.join(log_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        deleted_files.append(filename)
            
            if deleted_files:
                return f"Eski log dosyaları temizlendi: {len(deleted_files)} dosya"
            return None
            
        except Exception as e:
            self.logger.error(f"Log temizleme hatası: {e}")
            return None
    
    def _archive_old_signals(self):
        """Eski sinyal dosyalarını arşivle"""
        try:
            import os
            import shutil
            
            signal_dir = 'signals'
            if not os.path.exists(signal_dir):
                return None
            
            # 30 günden eski sinyal dosyalarını arşivle
            cutoff_date = datetime.now() - timedelta(days=30)
            archived_count = 0
            
            # Arşiv klasörü oluştur
            archive_dir = os.path.join(signal_dir, 'archive')
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            
            for filename in os.listdir(signal_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(signal_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_date:
                        archive_path = os.path.join(archive_dir, filename)
                        shutil.move(filepath, archive_path)
                        archived_count += 1
            
            if archived_count > 0:
                return f"Eski sinyal dosyaları arşivlendi: {archived_count} dosya"
            return None
            
        except Exception as e:
            self.logger.error(f"Sinyal arşivleme hatası: {e}")
            return None

class PerformanceTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_url = Config.DATABASE_URL
        self.engine = create_engine(self.db_url)
        
    def calculate_signal_performance(self, hours=24):
        """Son X saatteki sinyal performansını hesapla"""
        try:
            query = f"""
                SELECT * FROM signals
                WHERE timestamp::timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp::timestamp DESC
            """
            df = pd.read_sql(query, self.engine)
            if df.empty:
                return {
                    'total_signals': 0,
                    'open_signals': 0,
                    'closed_signals': 0,
                    'success_rate': 0.0,
                    'avg_ai_score': 0.0,
                    'avg_ta_strength': 0.0
                }
            
            # Açık ve kapalı sinyalleri ayır
            open_signals = df[df['result'].isna() | (df['result'] == 'None')]
            closed_signals = df[~df['result'].isna() & (df['result'] != 'None')]
            
            # Başarı oranı hesapla
            success_rate = 0.0
            if len(closed_signals) > 0:
                successful = closed_signals[closed_signals['result'] == 'SUCCESS']
                success_rate = (len(successful) / len(closed_signals)) * 100
            
            # Ortalama skorlar
            avg_ai_score = df['ai_score'].mean() if 'ai_score' in df.columns else 0.0
            avg_ta_strength = df['ta_strength'].mean() if 'ta_strength' in df.columns else 0.0
            avg_whale_score = df['whale_score'].mean() if 'whale_score' in df.columns else 0.0
            avg_profit = df['realized_gain'].mean() if 'realized_gain' in df.columns else 0.0
            return {
                'total_signals': len(df),
                'open_signals': len(open_signals),
                'closed_signals': len(closed_signals),
                'success_rate': success_rate,
                'avg_ai_score': avg_ai_score,
                'avg_ta_strength': avg_ta_strength,
                'avg_whale_score': avg_whale_score,
                'avg_profit': avg_profit
            }
        except Exception as e:
            self.logger.error(f"Son {hours} saatteki sinyaller alınamadı: {e}")
            return {
                'total_signals': 0,
                'open_signals': 0,
                'closed_signals': 0,
                'success_rate': 0.0,
                'avg_ai_score': 0.0,
                'avg_ta_strength': 0.0
            }
    
    def get_optimal_thresholds(self):
        """Optimal eşikleri hesapla"""
        try:
            # Son 7 günlük veriyi al
            performance = self.calculate_signal_performance(168)  # 7 gün
            
            if performance.get('total_signals', 0) < 50:
                return {
                    'min_ai_score': 0.45,
                    'min_ta_strength': 0.55,
                    'min_whale_score': 0.25,
                    'min_breakout_probability': 0.35,
                    'recommendation': 'Yeterli veri yok - varsayılan değerler kullanılıyor'
                }
            
            # Başarılı sinyallerin ortalamalarını hesapla
            coin_performance = performance.get('coin_performance', {})
            successful_coins = []
            
            for coin, data in coin_performance.items():
                if data.get('success_rate', 0) > 50:  # %50'den fazla başarı
                    successful_coins.append(data)
            
            if not successful_coins:
                return {
                    'min_ai_score': 0.50,
                    'min_ta_strength': 0.60,
                    'min_whale_score': 0.30,
                    'min_breakout_probability': 0.40,
                    'recommendation': 'Başarılı coin bulunamadı - eşikler yükseltildi'
                }
            
            # Optimal eşikleri hesapla
            avg_ai_score = sum(coin['avg_ai_score'] for coin in successful_coins) / len(successful_coins)
            optimal_ai_score = max(0.45, avg_ai_score * 0.8)  # %80'i
            
            return {
                'min_ai_score': round(optimal_ai_score, 3),
                'min_ta_strength': 0.55,
                'min_whale_score': 0.25,
                'min_breakout_probability': 0.35,
                'recommendation': f'Başarılı {len(successful_coins)} coin analiz edildi'
            }
            
        except Exception as e:
            return {
                'min_ai_score': 0.45,
                'min_ta_strength': 0.55,
                'min_whale_score': 0.25,
                'min_breakout_probability': 0.35,
                'recommendation': f'Hata: {str(e)}'
            }
    
    def get_signal_quality_metrics(self):
        """Sinyal kalite metriklerini hesapla"""
        try:
            # Son 24 saat performansını al
            performance = self.calculate_signal_performance(24)
            
            top_coins = performance.get('top_coins', {})
            coin_performance = performance.get('coin_performance', {})
            
            # Kalite metrikleri
            quality_metrics = {
                'total_signals': performance.get('total_signals', 0),
                'avg_ai_score': performance.get('avg_ai_score', 0),
                'avg_ta_strength': performance.get('avg_ta_strength', 0),
                'avg_whale_score': performance.get('avg_whale_score', 0),
                'avg_social_score': performance.get('avg_social_score', 0),
                'avg_news_score': performance.get('avg_news_score', 0),
                'top_performing_coins': [],
                'quality_assessment': 'UNKNOWN'
            }
            
            # En iyi performans gösteren coinler
            for coin, signal_count in list(top_coins.items())[:5]:
                coin_data = coin_performance.get(coin, {})
                quality_metrics['top_performing_coins'].append({
                    'coin': coin,
                    'signals': signal_count,
                    'success_rate': coin_data.get('success_rate', 0),
                    'avg_ai_score': coin_data.get('avg_ai_score', 0)
                })
            
            # Kalite değerlendirmesi
            avg_ai = quality_metrics['avg_ai_score']
            avg_ta = quality_metrics['avg_ta_strength']
            avg_whale = quality_metrics['avg_whale_score']
            
            if avg_ai > 0.6 and avg_ta > 0.7 and avg_whale > 0.4:
                quality_metrics['quality_assessment'] = 'HIGH'
            elif avg_ai > 0.4 and avg_ta > 0.5 and avg_whale > 0.2:
                quality_metrics['quality_assessment'] = 'MEDIUM'
            else:
                quality_metrics['quality_assessment'] = 'LOW'
            
            return quality_metrics
            
        except Exception as e:
            return {
                'total_signals': 0,
                'avg_ai_score': 0,
                'avg_ta_strength': 0,
                'avg_whale_score': 0,
                'avg_social_score': 0,
                'avg_news_score': 0,
                'top_performing_coins': [],
                'quality_assessment': 'ERROR',
                'error': str(e)
            }
    
    def get_system_health_report(self):
        """Sistem sağlık raporu"""
        try:
            # Son 24 saatlik performans
            performance_24h = self.calculate_signal_performance(hours=24)
            
            # Son 7 günlük performans
            performance_7d = self.calculate_signal_performance(hours=168)
            
            # Optimal eşikler
            optimal_thresholds = self.get_optimal_thresholds()
            
            # Kalite metrikleri
            quality_metrics = self.get_signal_quality_metrics()
            
            health_report = {
                'timestamp': str(datetime.now()),
                'performance_24h': performance_24h,
                'performance_7d': performance_7d,
                'optimal_thresholds': optimal_thresholds,
                'top_performing_coins': quality_metrics['top_performing_coins'][:5],
                'system_status': self._get_system_status(performance_24h),
                'recommendations': optimal_thresholds.get('recommendation', [])
            }
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"Sistem sağlık raporu hatası: {e}")
            return {}
    
    def _get_system_status(self, performance):
        """Sistem durumunu belirle"""
        if not performance:
            return "UNKNOWN"
        
        success_rate = performance.get('success_rate', 0)
        total_signals = performance.get('total_signals', 0)
        
        if total_signals < 10:
            return "INSUFFICIENT_DATA"
        elif success_rate >= 70:
            return "EXCELLENT"
        elif success_rate >= 50:
            return "GOOD"
        elif success_rate >= 30:
            return "FAIR"
        else:
            return "POOR"

    def track_real_time_performance(self, signal_id, symbol, direction, confidence, entry_price, timestamp):
        """Gerçek zamanlı sinyal performansını takip et"""
        try:
            # Sinyal kaydını oluştur
            signal_record = {
                'signal_id': signal_id,
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'entry_price': entry_price,
                'entry_time': timestamp,
                'status': 'OPEN',
                'current_price': entry_price,
                'current_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'exit_price': None,
                'exit_time': None,
                'final_pnl': None,
                'duration_hours': None,
                'success': None
            }
            
            # Veritabanına kaydet
            self.db.insert_signal_performance(signal_record)
            self.logger.info(f"Performans takibi başlatıldı: {symbol} {direction} @ {entry_price}")
            
            return signal_id
            
        except Exception as e:
            self.logger.error(f"Performans takibi başlatma hatası: {e}")
            return None
    
    def update_signal_performance(self, signal_id, current_price, current_time):
        """Sinyal performansını güncelle"""
        try:
            # Mevcut sinyal bilgilerini al
            signal = self.db.get_signal_performance(signal_id)
            if not signal:
                return False
            
            # PnL hesapla
            if signal['direction'] == 'BUY':
                pnl = (current_price - signal['entry_price']) / signal['entry_price']
            else:  # SELL
                pnl = (signal['entry_price'] - current_price) / signal['entry_price']
            
            # Maksimum kar/zarar güncelle
            max_profit = max(signal['max_profit'], pnl)
            max_loss = min(signal['max_loss'], pnl)
            
            # Güncelleme verileri
            update_data = {
                'current_price': current_price,
                'current_pnl': pnl,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'last_update': current_time
            }
            
            # Veritabanını güncelle
            self.db.update_signal_performance(signal_id, update_data)
            
            # Otomatik çıkış kontrolü
            self._check_exit_conditions(signal_id, pnl, signal['confidence'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performans güncelleme hatası: {e}")
            return False
    
    def _check_exit_conditions(self, signal_id, current_pnl, confidence):
        """Çıkış koşullarını kontrol et"""
        try:
            # Dinamik stop-loss ve take-profit
            if confidence > 0.8:  # Yüksek güven
                stop_loss = -0.05  # %5 stop-loss
                take_profit = 0.15  # %15 take-profit
            elif confidence > 0.6:  # Orta güven
                stop_loss = -0.03  # %3 stop-loss
                take_profit = 0.10  # %10 take-profit
            else:  # Düşük güven
                stop_loss = -0.02  # %2 stop-loss
                take_profit = 0.08  # %8 take-profit
            
            # Çıkış kontrolü
            if current_pnl <= stop_loss or current_pnl >= take_profit:
                self._close_signal(signal_id, current_pnl)
                
        except Exception as e:
            self.logger.error(f"Çıkış koşulu kontrolü hatası: {e}")
    
    def _close_signal(self, signal_id, final_pnl):
        """Sinyali kapat"""
        try:
            current_time = datetime.now()
            
            # Sinyal bilgilerini al
            signal = self.db.get_signal_performance(signal_id)
            if not signal:
                return False
            
            # Süre hesapla
            entry_time = pd.to_datetime(signal['entry_time'])
            duration_hours = (current_time - entry_time).total_seconds() / 3600
            
            # Başarı durumu
            success = final_pnl > 0
            
            # Güncelleme verileri
            update_data = {
                'status': 'CLOSED',
                'exit_price': signal['current_price'],
                'exit_time': current_time,
                'final_pnl': final_pnl,
                'duration_hours': duration_hours,
                'success': success
            }
            
            # Veritabanını güncelle
            self.db.update_signal_performance(signal_id, update_data)
            
            # Model performansını güncelle
            self._update_model_performance(signal, final_pnl, success)
            
            self.logger.info(f"Sinyal kapatıldı: {signal['symbol']} PnL: {final_pnl:.3f} Başarı: {success}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sinyal kapatma hatası: {e}")
            return False
    
    def _update_model_performance(self, signal, final_pnl, success):
        """Model performansını güncelle"""
        try:
            # Model performans metriklerini güncelle
            self.model_performance['total_signals'] += 1
            self.model_performance['successful_signals'] += 1 if success else 0
            self.model_performance['total_pnl'] += final_pnl
            
            # Başarı oranı
            self.model_performance['success_rate'] = (
                self.model_performance['successful_signals'] / 
                self.model_performance['total_signals']
            )
            
            # Ortalama PnL
            self.model_performance['avg_pnl'] = (
                self.model_performance['total_pnl'] / 
                self.model_performance['total_signals']
            )
            
            # Yön bazlı performans
            direction = signal['direction']
            if direction not in self.model_performance['direction_performance']:
                self.model_performance['direction_performance'][direction] = {
                    'total': 0, 'successful': 0, 'total_pnl': 0
                }
            
            self.model_performance['direction_performance'][direction]['total'] += 1
            self.model_performance['direction_performance'][direction]['successful'] += 1 if success else 0
            self.model_performance['direction_performance'][direction]['total_pnl'] += final_pnl
            
            # Güven seviyesi bazlı performans
            confidence_level = self._get_confidence_level(signal['confidence'])
            if confidence_level not in self.model_performance['confidence_performance']:
                self.model_performance['confidence_performance'][confidence_level] = {
                    'total': 0, 'successful': 0, 'total_pnl': 0
                }
            
            self.model_performance['confidence_performance'][confidence_level]['total'] += 1
            self.model_performance['confidence_performance'][confidence_level]['successful'] += 1 if success else 0
            self.model_performance['confidence_performance'][confidence_level]['total_pnl'] += final_pnl
            
            # Performansı kaydet
            self._save_model_performance()
            
        except Exception as e:
            self.logger.error(f"Model performans güncelleme hatası: {e}")
    
    def _get_confidence_level(self, confidence):
        """Güven seviyesini kategorize et"""
        if confidence >= 0.8:
            return 'HIGH'
        elif confidence >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_performance_analytics(self):
        """Detaylı performans analizi"""
        try:
            analytics = {
                'overall_performance': self.model_performance,
                'recent_signals': self.db.get_recent_signals(limit=50),
                'top_performing_symbols': self._get_top_performing_symbols(),
                'worst_performing_symbols': self._get_worst_performing_symbols(),
                'hourly_performance': self._get_hourly_performance(),
                'confidence_analysis': self._get_confidence_analysis()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Performans analizi hatası: {e}")
            return {}
    
    def _get_top_performing_symbols(self, limit=10):
        """En iyi performans gösteren semboller"""
        try:
            signals = self.db.get_all_signals()
            symbol_performance = {}
            
            for signal in signals:
                symbol = signal['symbol']
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {'total': 0, 'successful': 0, 'total_pnl': 0}
                
                symbol_performance[symbol]['total'] += 1
                if signal['success']:
                    symbol_performance[symbol]['successful'] += 1
                symbol_performance[symbol]['total_pnl'] += signal.get('final_pnl', 0)
            
            # Başarı oranına göre sırala
            sorted_symbols = sorted(
                symbol_performance.items(),
                key=lambda x: x[1]['successful'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                reverse=True
            )
            
            return sorted_symbols[:limit]
            
        except Exception as e:
            self.logger.error(f"Top performing symbols hatası: {e}")
            return []
    
    def _get_worst_performing_symbols(self, limit=10):
        """En kötü performans gösteren semboller"""
        try:
            signals = self.db.get_all_signals()
            symbol_performance = {}
            
            for signal in signals:
                symbol = signal['symbol']
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {'total': 0, 'successful': 0, 'total_pnl': 0}
                
                symbol_performance[symbol]['total'] += 1
                if signal['success']:
                    symbol_performance[symbol]['successful'] += 1
                symbol_performance[symbol]['total_pnl'] += signal.get('final_pnl', 0)
            
            # Başarı oranına göre sırala (en kötüden)
            sorted_symbols = sorted(
                symbol_performance.items(),
                key=lambda x: x[1]['successful'] / x[1]['total'] if x[1]['total'] > 0 else 0
            )
            
            return sorted_symbols[:limit]
            
        except Exception as e:
            self.logger.error(f"Worst performing symbols hatası: {e}")
            return []
    
    def _get_hourly_performance(self):
        """Saatlik performans analizi"""
        try:
            signals = self.db.get_all_signals()
            hourly_performance = {i: {'total': 0, 'successful': 0, 'total_pnl': 0} for i in range(24)}
            
            for signal in signals:
                if signal['entry_time']:
                    hour = pd.to_datetime(signal['entry_time']).hour
                    hourly_performance[hour]['total'] += 1
                    if signal['success']:
                        hourly_performance[hour]['successful'] += 1
                    hourly_performance[hour]['total_pnl'] += signal.get('final_pnl', 0)
            
            return hourly_performance
            
        except Exception as e:
            self.logger.error(f"Hourly performance hatası: {e}")
            return {}
    
    def _get_confidence_analysis(self):
        """Güven seviyesi analizi"""
        try:
            return self.model_performance.get('confidence_performance', {})
        except Exception as e:
            self.logger.error(f"Confidence analysis hatası: {e}")
            return {}

    def _save_model_performance(self):
        """Model performansını veritabanına kaydet"""
        try:
            # Model performansını veritabanına kaydet
            query = """
                INSERT INTO model_performance (total_signals, successful_signals, total_pnl, success_rate, avg_pnl)
                VALUES (:total_signals, :successful_signals, :total_pnl, :success_rate, :avg_pnl)
            """
            self.engine.execute(
                query,
                total_signals=self.model_performance['total_signals'],
                successful_signals=self.model_performance['successful_signals'],
                total_pnl=self.model_performance['total_pnl'],
                success_rate=self.model_performance['success_rate'],
                avg_pnl=self.model_performance['avg_pnl']
            )
            
        except Exception as e:
            self.logger.error(f"Model performans kaydetme hatası: {e}") 