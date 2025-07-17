#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dinamik Sıkılık Sistemi
Piyasa durumuna göre otomatik sıkılık ayarlama
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import json
import os

logger = logging.getLogger(__name__)

class DynamicStrictness:
    def __init__(self):
        self.current_strictness = 0.5  # Başlangıç değeri optimize edildi (0.5)
        self.strictness_history = []
        self.market_conditions = {}
        self.last_update = None
        self.update_interval = timedelta(hours=1)  # 1 saatte bir güncelle
        
        # Performans takibi için yeni parametreler
        self.performance_history = []
        self.signal_success_rate = 0.5
        self.last_signal_count = 0
        self.auto_adjustment_enabled = True
        self.min_success_rate = 0.6  # %60 başarı oranı hedefi
        self.max_success_rate = 0.8  # %80 başarı oranı hedefi
        
        self.min_strictness = 0.45  # Optimize edilmiş minimum değer
        self.max_strictness = 0.60  # Optimize edilmiş maksimum değer
        
    def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Piyasa koşullarını analiz et"""
        try:
            conditions = {
                'volatility': 0.0,
                'trend_strength': 0.0,
                'volume_activity': 0.0,
                'market_sentiment': 0.0,
                'ai_confidence': 0.0,
                'overall_score': 0.0
            }
            
            # Volatilite analizi
            if 'price_data' in market_data:
                prices = market_data['price_data']
                if len(prices) > 20:
                    returns = np.diff(np.log(prices))
                    conditions['volatility'] = np.std(returns) * np.sqrt(24)  # Günlük volatilite
            
            # Trend gücü analizi
            if 'technical_indicators' in market_data:
                ta_data = market_data['technical_indicators']
                if 'rsi' in ta_data and 'macd' in ta_data:
                    rsi = ta_data['rsi']
                    macd = ta_data['macd']
                    
                    # RSI trend analizi
                    rsi_trend = 0
                    if rsi > 70:
                        rsi_trend = -0.5  # Aşırı alım
                    elif rsi < 30:
                        rsi_trend = 0.5   # Aşırı satım
                    else:
                        rsi_trend = 0
                    
                    # MACD trend analizi
                    macd_trend = 1 if macd > 0 else -1
                    
                    conditions['trend_strength'] = (rsi_trend + macd_trend) / 2
            
            # Volume aktivitesi
            if 'volume_data' in market_data:
                volume = market_data['volume_data']
                if len(volume) > 10:
                    avg_volume = np.mean(volume[-10:])
                    current_volume = volume[-1]
                    conditions['volume_activity'] = (current_volume - avg_volume) / avg_volume
            
            # Market sentiment (haber, sosyal medya vb.)
            if 'sentiment_data' in market_data:
                sentiment = market_data['sentiment_data']
                conditions['market_sentiment'] = sentiment.get('overall_sentiment', 0)
            
            # AI güven skoru
            if 'ai_predictions' in market_data:
                ai_data = market_data['ai_predictions']
                conditions['ai_confidence'] = ai_data.get('confidence', 0.5)
            
            # Genel skor hesaplama
            conditions['overall_score'] = (
                conditions['volatility'] * 0.2 +
                conditions['trend_strength'] * 0.3 +
                conditions['volume_activity'] * 0.2 +
                conditions['market_sentiment'] * 0.15 +
                conditions['ai_confidence'] * 0.15
            )
            
            return conditions
            
        except Exception as e:
            logger.error(f"Market analizi hatası: {e}")
            return {'overall_score': 0.5}
    
    def calculate_dynamic_strictness(self, market_conditions: Dict) -> Tuple[float, str]:
        """Dinamik sıkılık hesaplama (optimize edilmiş aralık)"""
        try:
            overall_score = market_conditions.get('overall_score', 0.5)
            # Sıkılık hesaplama algoritması (daha dengeli)
            if overall_score > 0.7:
                strictness = self.min_strictness  # 0.45
                reason = "Güçlü yükseliş - Minimum sıkılık"
            elif overall_score > 0.5:
                strictness = 0.50
                reason = "Orta yükseliş - Orta sıkılık"
            elif overall_score > 0.3:
                strictness = 0.55
                reason = "Sideways - Orta-Yüksek sıkılık"
            elif overall_score > 0.1:
                strictness = 0.58
                reason = "Orta düşüş - Yüksek sıkılık"
            else:
                strictness = self.max_strictness  # 0.60
                reason = "Güçlü düşüş - Maksimum sıkılık"
            # Smoothing - ani değişimleri yumuşat
            if self.current_strictness:
                strictness = 0.7 * strictness + 0.3 * self.current_strictness
            # Sıkılık değerini optimize aralıkta sınırla
            strictness = max(self.min_strictness, min(strictness, self.max_strictness))
            return strictness, reason
        except Exception as e:
            logger.error(f"Sıkılık hesaplama hatası: {e}")
            return 0.55, "HATA - Varsayılan sıkılık kullanılıyor"
    
    def update_strictness(self, market_data: Dict) -> Dict:
        """Sıkılığı güncelle"""
        try:
            current_time = datetime.now()
            
            # Güncelleme kontrolü
            if (self.last_update and 
                current_time - self.last_update < self.update_interval):
                return self.get_current_status()
            
            # Market analizi
            market_conditions = self.analyze_market_conditions(market_data)
            
            # Yeni sıkılık hesaplama
            new_strictness, reason = self.calculate_dynamic_strictness(market_conditions)
            
            # Değişiklik kontrolü (daha hassas)
            change_threshold = 0.01
            if abs(new_strictness - self.current_strictness) > change_threshold:
                old_strictness = self.current_strictness
                self.current_strictness = new_strictness
                
                # Geçmiş kaydet
                self.strictness_history.append({
                    'timestamp': current_time.isoformat(),
                    'old_strictness': old_strictness,
                    'new_strictness': new_strictness,
                    'reason': reason,
                    'market_conditions': market_conditions
                })
                
                # Geçmişi sınırla (son 100 kayıt)
                if len(self.strictness_history) > 100:
                    self.strictness_history = self.strictness_history[-100:]
                
                logger.info(f"🔧 SIKILIK GÜNCELLENDİ: {old_strictness:.2f} → {new_strictness:.2f} - {reason}")
            
            self.market_conditions = market_conditions
            self.last_update = current_time
            
            return self.get_current_status()
            
        except Exception as e:
            logger.error(f"Sıkılık güncelleme hatası: {e}")
            return self.get_current_status()
    
    def update_performance_metrics(self, signal_results: Dict):
        """Sinyal performans metriklerini güncelle"""
        try:
            current_time = datetime.now()
            
            # Performans verilerini kaydet
            performance_data = {
                'timestamp': current_time.isoformat(),
                'signal_count': signal_results.get('total_signals', 0),
                'successful_signals': signal_results.get('successful_signals', 0),
                'failed_signals': signal_results.get('failed_signals', 0),
                'current_strictness': self.current_strictness,
                'market_conditions': self.market_conditions.copy()
            }
            
            self.performance_history.append(performance_data)
            
            # Son 24 saatlik performansı hesapla
            day_ago = current_time - timedelta(hours=24)
            recent_performance = [
                p for p in self.performance_history 
                if datetime.fromisoformat(p['timestamp']) > day_ago
            ]
            
            if recent_performance:
                total_signals = sum(p['signal_count'] for p in recent_performance)
                successful_signals = sum(p['successful_signals'] for p in recent_performance)
                
                if total_signals > 0:
                    self.signal_success_rate = successful_signals / total_signals
                else:
                    self.signal_success_rate = 0.5
            
            # Otomatik ayarlama yap
            if self.auto_adjustment_enabled:
                self.auto_adjust_strictness()
                
        except Exception as e:
            logger.error(f"Performans metrik güncelleme hatası: {e}")
    
    def auto_adjust_strictness(self):
        """Performansa göre otomatik sıkılık ayarlama (daha agresif)"""
        try:
            if self.signal_success_rate < self.min_success_rate:
                adjustment = 0.08
                new_strictness = min(0.85, self.current_strictness + adjustment)
                reason = f"Düşük başarı oranı ({self.signal_success_rate:.1%}) - Sıkılık artırıldı"
            elif self.signal_success_rate > self.max_success_rate:
                adjustment = 0.06
                new_strictness = max(0.7, self.current_strictness - adjustment)
                reason = f"Yüksek başarı oranı ({self.signal_success_rate:.1%}) - Sıkılık azaltıldı"
            else:
                adjustment = 0.02
                if self.current_strictness > 0.77:
                    new_strictness = self.current_strictness - adjustment
                    reason = f"Optimal performans - Hafif gevşetme"
                else:
                    new_strictness = self.current_strictness + adjustment
                    reason = f"Optimal performans - Hafif sıkılaştırma"
            if abs(new_strictness - self.current_strictness) > 0.01:
                old_strictness = self.current_strictness
                self.current_strictness = new_strictness
                logger.info(f"Otomatik ayarlama: {old_strictness:.3f} → {new_strictness:.3f} - {reason}")
                self.strictness_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'old_strictness': old_strictness,
                    'new_strictness': new_strictness,
                    'reason': reason,
                    'success_rate': self.signal_success_rate,
                    'adjustment_type': 'auto_performance'
                })
        except Exception as e:
            logger.error(f"Otomatik ayarlama hatası: {e}")
    
    def get_performance_summary(self):
        """Performans özeti döndür"""
        try:
            if not self.performance_history:
                return {
                    'success_rate': 0.5,
                    'total_signals': 0,
                    'auto_adjustment_status': 'No data'
                }
            
            recent_performance = self.performance_history[-10:]  # Son 10 kayıt
            
            total_signals = sum(p['signal_count'] for p in recent_performance)
            successful_signals = sum(p['successful_signals'] for p in recent_performance)
            
            success_rate = successful_signals / total_signals if total_signals > 0 else 0.5
            
            return {
                'success_rate': success_rate,
                'total_signals': total_signals,
                'auto_adjustment_status': 'Active' if self.auto_adjustment_enabled else 'Disabled',
                'last_adjustment': self.strictness_history[-1] if self.strictness_history else None
            }
            
        except Exception as e:
            logger.error(f"Performans özeti hatası: {e}")
            return {'error': str(e)}
    
    def get_current_status(self) -> Dict:
        """Mevcut durumu döndür"""
        performance_summary = self.get_performance_summary()
        
        return {
            'current_strictness': self.current_strictness,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'market_conditions': self.market_conditions,
            'strictness_level': self.get_strictness_level(),
            'recommendation': self.get_recommendation(),
            'history': self.strictness_history[-10:] if self.strictness_history else [],
            'performance': performance_summary,
            'auto_adjustment': {
                'enabled': self.auto_adjustment_enabled,
                'success_rate': self.signal_success_rate,
                'min_target': self.min_success_rate,
                'max_target': self.max_success_rate
            }
        }
    
    def get_strictness_level(self) -> str:
        """Sıkılık seviyesi metni (standart aralıklarla)"""
        if self.current_strictness <= 0.60:
            return "DÜŞÜK SIKILIK"
        elif self.current_strictness <= 0.70:
            return "ORTA SIKILIK"
        elif self.current_strictness <= 0.80:
            return "YÜKSEK SIKILIK"
        else:
            return "MAKSİMUM SIKILIK"
    
    def get_recommendation(self) -> str:
        """Öneri metni (standart aralıklarla)"""
        if self.current_strictness <= 0.60:
            return "Daha fazla sinyal - Düşük sıkılık, fırsat odaklı"
        elif self.current_strictness <= 0.70:
            return "Dengeli sinyal kalitesi - Orta sıkılık, risk/ödül dengesi"
        elif self.current_strictness <= 0.80:
            return "Yüksek kalite sinyaller - Yüksek sıkılık, güvenli seçim"
        else:
            return "Sadece en güçlü sinyaller - Maksimum sıkılık, ultra güvenlik"
    
    def save_status(self, filepath: str = "dynamic_strictness_status.json"):
        """Durumu dosyaya kaydet"""
        try:
            status = self.get_current_status()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
            logger.info(f"📁 Dinamik sıkılık durumu kaydedildi: {filepath}")
        except Exception as e:
            logger.error(f"Durum kaydetme hatası: {e}")
    
    def load_status(self, filepath: str = "dynamic_strictness_status.json"):
        """Durumu dosyadan yükle"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                
                self.current_strictness = status.get('current_strictness', 0.55)
                self.market_conditions = status.get('market_conditions', {})
                self.strictness_history = status.get('history', [])
                
                if status.get('last_update'):
                    self.last_update = datetime.fromisoformat(status['last_update'])
                
                logger.info(f"📂 Dinamik sıkılık durumu yüklendi: {self.current_strictness:.2f}")
        except Exception as e:
            logger.error(f"Durum yükleme hatası: {e}")

# Global instance
dynamic_strictness = DynamicStrictness() 